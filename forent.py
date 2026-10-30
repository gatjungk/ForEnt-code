"""Shared helpers: walk folders, reading the bags, and the /lowstate clock."""
from pathlib import Path
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def walk_dir(root, seq):
    return Path(root) / f"site_{seq.split('_')[1]}" / seq


def messages(root, seq, topics):
    """Yield (topic, message, receive time ns) in bag order."""
    with AnyReader([walk_dir(root, seq) / f'{seq}.mcap'], default_typestore=get_typestore(Stores.ROS2_FOXY)) as r:
        for con, t, raw in r.messages(connections=[c for c in r.connections if c.topic in topics]):
            yield con.topic, r.deserialize(raw, con.msgtype), t


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def lowstate_offset(go2, mid):
    """c in time = tick / 1000 + c: the shift that best correlates the Go2 and MID-360 gyros (3 axes, coarse to fine).

    go2: rows (tick_s, gx, gy, gz, receive_s); mid: rows (stamp_s, gx, gy, gz).
    """
    grid = np.arange(mid[0, 0] + 5, mid[-1, 0] - 5, 0.005)
    M = np.stack([np.interp(grid, mid[:, 0], mid[:, k]) for k in (1, 2, 3)], 1)

    def score(c):
        G = np.stack([np.interp(grid, go2[:, 0] + c, go2[:, k]) for k in (1, 2, 3)], 1)
        return np.mean([np.corrcoef(M[:, k], G[:, k])[0, 1] for k in range(3)])

    best = np.percentile(go2[:, 4] - go2[:, 0], 1)
    for half, step in ((10.0, 0.05), (0.1, 0.002), (0.004, 0.0002)):
        cand = np.arange(best - half, best + half, step)
        best = cand[int(np.argmax([score(c) for c in cand]))]
    return float(best)


SEQUENCE_KEYS = ('sequence', 'site', 'direction', 'route', 'start_utc', 'duration_s', 'path_m',
                 'recording', 'trajectory')


def update_sequence(root, seq, **fields):
    """Set fields in the walk's sequence.yaml, keeping a fixed key order."""
    path = walk_dir(root, seq) / 'sequence.yaml'
    d = yaml.safe_load(path.read_text()) if path.exists() else {}
    d.update(sequence=seq, site=int(seq.split('_')[1]), **fields)
    path.write_text(yaml.safe_dump({k: d[k] for k in SEQUENCE_KEYS if k in d}, sort_keys=False))


def transform(root, parent, child):
    """4x4 transform from common/calibration/extrinsics.yaml: p_parent = T p_child."""
    e = yaml.safe_load(open(Path(root) / 'common/calibration/extrinsics.yaml'))
    for x in e['transforms'] + e.get('derived', []):
        if (x['parent'], x['child']) == (parent, child):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = x['R'], x['t']
            return T
    raise KeyError((parent, child))
