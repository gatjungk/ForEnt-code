"""Entrapment labels and their YAML file (the format of annotations/<annotator>.yaml in the release)."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import yaml

TYPES = ('collision', 'dragging', 'entanglement', 'suspension')
KINDS = ('controller', 'physical')


@dataclass
class Intervention:
    kind: str
    start: float
    end: float


@dataclass
class Entrapment:
    start: float
    end: float
    types: list = field(default_factory=list)
    fall: bool = False
    interventions: list = field(default_factory=list)


def load(path):
    """Entrapments from a label file; empty if it does not exist."""
    if not Path(path).exists():
        return []
    out = []
    for e in yaml.safe_load(open(path)).get('entrapments') or []:
        out.append(Entrapment(e['start']['rel_s'], e['end']['rel_s'], list(e['types']), e['fall'] == 'yes',
                              [Intervention(i['kind'], i['start']['rel_s'], i['end']['rel_s'])
                               for i in e.get('interventions') or []]))
    return out


def save(path, entrapments, annotator, seq, start):
    """Write the label file; times are stored as rel_s and as recording time (time_s = start + rel_s)."""
    def when(rel):
        return {'time_s': round(float(start + rel), 6), 'rel_s': round(float(rel), 3)}

    rows = []
    for k, e in enumerate(sorted(entrapments, key=lambda e: e.start)):
        rows.append({
            'id': f'E{k + 1:03d}',
            'start': when(e.start),
            'end': when(e.end),
            'duration_s': round(float(e.end - e.start), 3),
            'types': [t for t in TYPES if t in e.types],
            'fall': 'yes' if e.fall else 'no',
            'severity': 2 if any(i.kind == 'physical' for i in e.interventions) else 1,
            'interventions': [{'kind': i.kind, 'name': f'{i.kind} intervention', 'start': when(i.start),
                               'end': when(i.end), 'duration_s': round(float(i.end - i.start), 3)}
                              for i in sorted(e.interventions, key=lambda i: i.start)],
        })
    doc = {'created': datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'), 'labeller': annotator,
           'sequence': seq, 'entrapments': rows}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(doc, sort_keys=False))


TOL = 1.0   # s; start and end times closer than this count as the same


def iou(a, b):
    return max(0.0, min(a.end, b.end) - max(a.start, b.start)) / (max(a.end, b.end) - min(a.start, b.start))


def disagreements(a, b, name_a, name_b):
    """(time, text) for each difference between two annotators: entrapments one of them missed, and for matched
    entrapments (paired by temporal overlap) their start, end, types, fall and interventions."""
    pairs = sorted(((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, matched = set(), set(), []
    for v, i, j in pairs:
        if v > 0 and i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched.append((a[i], b[j]))
    out = [(e.start, f'{e.start:6.1f} s  only {name}: {e.start:.1f} - {e.end:.1f} s')
           for E, used, name in ((a, used_a, name_a), (b, used_b, name_b))
           for k, e in enumerate(E) if k not in used]
    for x, y in matched:
        t = min(x.start, y.start)
        if abs(x.start - y.start) > TOL:
            out.append((t, f'{t:6.1f} s  start: {name_a} {x.start:.1f}, {name_b} {y.start:.1f}'))
        if abs(x.end - y.end) > TOL:
            out.append((t, f'{t:6.1f} s  end: {name_a} {x.end:.1f}, {name_b} {y.end:.1f}'))
        if set(x.types) != set(y.types):
            out.append((t, f'{t:6.1f} s  types: {name_a} {", ".join(x.types) or "none"}; '
                           f'{name_b} {", ".join(y.types) or "none"}'))
        if x.fall != y.fall:
            out.append((t, f'{t:6.1f} s  fall: only {name_a if x.fall else name_b}'))
        for kind in KINDS:
            p = [i for i in x.interventions if i.kind == kind]
            q = [i for i in y.interventions if i.kind == kind]
            if bool(p) != bool(q):
                out.append((t, f'{t:6.1f} s  {kind} intervention: only {name_a if p else name_b}'))
            elif p and (abs(p[0].start - q[0].start) > TOL or abs(p[-1].end - q[-1].end) > TOL):
                out.append((t, f'{t:6.1f} s  {kind}: {name_a} {p[0].start:.1f} - {p[-1].end:.1f}, '
                               f'{name_b} {q[0].start:.1f} - {q[-1].end:.1f}'))
    return sorted(out)
