"""Agreement between the two annotators (annotations/A1.yaml and A2.yaml) over all walks.

Time: Cohen's kappa and temporal IoU of the labelled time on a 0.1 s grid over each whole recording (entrapment,
controller and physical intervention), all walks pooled. Events: matched one-to-one by temporal IoU (greedy, highest
first); F1 at IoU > 0 and > 0.5, median IoU of the matched events, and Cohen's kappa of each entrapment type over the
matched events.

usage: python agreement.py <dataset folder> [--json <out.json>]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from rosbags.highlevel import AnyReader

DT = 0.1
TYPES = ('collision', 'dragging', 'entanglement', 'suspension')


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float('nan')


def span(e):
    return e['start']['rel_s'], e['end']['rel_s']


def interventions(e, kind):
    return [(i['start']['rel_s'], i['end']['rel_s']) for i in e.get('interventions') or [] if i['kind'] == kind]


def iou(x, y):
    return max(0.0, min(x[1], y[1]) - max(x[0], y[0])) / (max(x[1], y[1]) - min(x[0], y[0]))


def match(A, B):
    pairs = sorted(((iou(span(a), span(b)), i, j) for i, a in enumerate(A) for j, b in enumerate(B)), reverse=True)
    used_a, used_b, out = set(), set(), []
    for v, i, j in pairs:
        if v > 0 and i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            out.append((i, j, v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--json')
    a = ap.parse_args()

    spans = {'entrapped': lambda e: [span(e)],
             'controller': lambda e: interventions(e, 'controller'),
             'physical': lambda e: interventions(e, 'physical')}
    grid = {k: ([], []) for k in spans}
    M, n1, n2 = [], 0, 0
    for d in sorted(Path(a.root).glob('site_*/site_*/annotations')):
        A = yaml.safe_load(open(d / 'A1.yaml'))['entrapments'] or []
        B = yaml.safe_load(open(d / 'A2.yaml'))['entrapments'] or []
        with AnyReader([d.parent / f'{d.parent.name}.mcap']) as r:
            t = np.arange(0, (r.end_time - r.start_time) * 1e-9, DT)
        for k, f in spans.items():
            for E, out in ((A, grid[k][0]), (B, grid[k][1])):
                m = np.zeros(len(t), bool)
                for e in E:
                    for s, end in f(e):
                        m[(t >= s) & (t < end)] = True
                out.append(m)
        n1, n2 = n1 + len(A), n2 + len(B)
        M += [(A[i], B[j], v) for i, j, v in match(A, B)]

    out = {'events': {'A1': n1, 'A2': n2}, 'time': {}, 'matched_events': {}}
    for k, (x, y) in grid.items():
        x, y = np.concatenate(x), np.concatenate(y)
        out['time'][k] = dict(kappa=round(kappa(x, y), 3), iou=round(float((x & y).sum() / max(1, (x | y).sum())), 3))
    me = out['matched_events']
    for th in (0.0, 0.5):
        k = sum(1 for *_, v in M if v > th)
        me[f'f1_iou>{th}'] = round(2 * k / (n1 + n2), 3)
    me['iou_median'] = round(float(np.median([v for *_, v in M])), 3)
    me['type_kappa'] = {ty: round(kappa([ty in x['types'] for x, _, _ in M], [ty in y['types'] for _, y, _ in M]), 3)
                        for ty in TYPES}

    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
