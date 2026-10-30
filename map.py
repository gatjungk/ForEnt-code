"""Write map.pcd for one walk: the lidar scans placed with the released trajectory, moving objects removed with
DUFOMap, then reduced to 5 cm voxels.

Each point is placed at its own time (5 ms slices of the scan, pose interpolated), only where the trajectory is
trusted. Scans go to DUFOMap one PCD per scan; each kept voxel becomes one point at the mean of its points, with the
mean reflectivity and the number of scans that hit it.

usage: python map.py <dataset folder> <sequence> --dufomap <dufomap_run> --config <config.toml> [--work <dir>]
"""
import argparse
import csv
import struct
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation, Slerp
from forent import walk_dir, transform

MIN_RANGE, MAX_RANGE, VOXEL, SLICE_NS, MAX_POSE_GAP = 0.7, 40.0, 0.05, 5_000_000, 0.25
SCAN_MID_NS = 50_000_000
BAD_FLAGS = ('unverified', 'odometry_replaced', 'fall')
POINT = np.dtype([('offset_time', '<u4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                  ('refl', 'u1'), ('tag', 'u1'), ('line', 'u1'), ('pad', 'u1')])
SHIFT = 2 ** 20


def scan(raw):
    """Raw livox_ros_driver2/CustomMsg (CDR) -> (timebase ns, points)."""
    o = 4 + 8  # encapsulation header, header.stamp
    o += 4 + struct.unpack_from('<I', raw, o)[0]  # header.frame_id
    o = (o - 4 + 7) // 8 * 8 + 4
    tb = struct.unpack_from('<Q', raw, o)[0]
    o += 8 + 4 + 1 + 3  # timebase, point_num, lidar_id, rsvd
    o = (o - 4 + 3) // 4 * 4 + 4
    n = struct.unpack_from('<I', raw, o)[0]
    return tb, np.frombuffer(raw[o + 4:] + b'\0', POINT, count=n)


def trajectory(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r['t']) for r in rows])
    p = np.array([[float(r[k]) for k in 'xyz'] for r in rows])
    q = np.array([[float(r[k]) for k in ('qx', 'qy', 'qz', 'qw')] for r in rows])
    bad = np.array([any(b in r['flag'].split(';') for b in BAD_FLAGS) for r in rows])
    ok = (np.diff(t) <= MAX_POSE_GAP) & ~bad[:-1] & ~bad[1:]
    return t, p, Slerp(t, Rotation.from_quat(q)), ok


def write_scan(path, xyz, sid, refl, origin, q):
    """One scan as a binary PCD in the world frame; rgb carries scan index and reflectivity through DUFOMap
    (DUFOMap does not preserve the top byte, so it stays unused)."""
    rec = np.zeros(len(xyz), np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')]))
    rec['x'], rec['y'], rec['z'] = xyz.T.astype(np.float32)
    rec['rgb'] = (np.uint32(sid) << 8) | refl.astype(np.uint32)
    qx, qy, qz, qw = q
    n = len(rec)
    head = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n'
            f'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\n'
            f'VIEWPOINT {origin[0]} {origin[1]} {origin[2]} {qw} {qx} {qy} {qz}\nPOINTS {n}\nDATA binary\n')
    with open(path, 'wb') as f:
        f.write(head.encode())
        f.write(rec.tobytes())


def read_pcd(path):
    with open(path, 'rb') as f:
        head = {}
        while True:
            k = f.readline().decode().strip().split(' ', 1)
            head[k[0]] = k[-1]
            if k[0] == 'DATA':
                break
        types = {'F': '<f', 'U': '<u', 'I': '<i'}
        fields = zip(head['FIELDS'].split(), head['SIZE'].split(), head['TYPE'].split())
        dt = np.dtype([(n, types[t] + s) for n, s, t in fields])
        a = np.frombuffer(f.read(), dt, count=int(head['WIDTH']))
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64), a['rgb'].view('<u4')


def voxels(xyz, refl, sid):
    """Per 5 cm voxel: mean position, mean reflectivity and the number of distinct scans that hit it."""
    v = np.floor(xyz / VOXEL).astype(np.int64) + SHIFT
    u, inv = np.unique((v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2], return_inverse=True)
    n = np.bincount(inv, minlength=len(u))
    mean = np.stack([np.bincount(inv, xyz[:, k], minlength=len(u)) for k in range(3)], 1) / n[:, None]
    pair = np.unique(inv.astype(np.int64) * (1 << 24) + sid)
    scans = np.bincount((pair >> 24).astype(np.int64), minlength=len(u))
    return mean, np.bincount(inv, refl, minlength=len(u)) / n, scans


def write_map(path, xyz, refl, scans):
    rec = np.zeros(len(xyz), np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('reflectivity', '<f4'),
                                       ('scans', '<u2')]))
    rec['x'], rec['y'], rec['z'] = xyz.T
    rec['reflectivity'], rec['scans'] = refl, scans
    n = len(rec)
    head = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z reflectivity scans\nSIZE 4 4 4 4 2\n'
            f'TYPE F F F F U\nCOUNT 1 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n')
    with open(path, 'wb') as f:
        f.write(head.encode())
        f.write(rec.tobytes())


def export_scans(root, seq, out):
    """Deskewed scans in the world frame -> out/NNNNNN.pcd."""
    d = walk_dir(root, seq)
    t, p, slerp, ok = trajectory(d / 'trajectory.csv')
    L = transform(root, 'go2_base', 'mid360')
    R_bl, t_bl = L[:3, :3], L[:3, 3]
    n = 0
    with AnyReader([d / f'{seq}.mcap']) as r:
        for _, _, raw in r.messages(connections=[c for c in r.connections if c.topic == '/livox/lidar']):
            tb, a = scan(raw)
            rng = np.sqrt(a['x'].astype(np.float64) ** 2 + a['y'] ** 2 + a['z'] ** 2)
            a = a[(rng > MIN_RANGE) & (rng < MAX_RANGE)]
            if not len(a):
                continue
            tp = (tb + (a['offset_time'].astype(np.int64) // SLICE_NS) * SLICE_NS + SLICE_NS // 2) * 1e-9
            i = np.searchsorted(t, tp) - 1
            keep = (i >= 0) & (i < len(t) - 1)
            keep[keep] &= ok[i[keep]]
            tm = (tb + SCAN_MID_NS) * 1e-9
            im = np.searchsorted(t, tm) - 1
            if not keep.any() or im < 0 or im >= len(t) - 1 or not ok[im]:
                continue
            a, tp = a[keep], tp[keep]
            n += 1
            ut, inv = np.unique(tp, return_inverse=True)
            R = slerp(ut)
            P = np.stack([np.interp(ut, t, p[:, k]) for k in range(3)], 1)
            w = R[inv].apply(np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64) @ R_bl.T + t_bl) + P[inv]
            Rm = slerp([tm])[0]
            origin = np.array([np.interp(tm, t, p[:, k]) for k in range(3)]) + Rm.apply(t_bl)
            write_scan(out / f'{n:06d}.pcd', w, n, a['refl'], origin, (Rm * Rotation.from_matrix(R_bl)).as_quat())
    assert n < 65536


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('seq')
    ap.add_argument('--dufomap', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--work')
    a = ap.parse_args()

    with tempfile.TemporaryDirectory(dir=a.work) as work:
        work = Path(work)
        (work / 'pcd').mkdir()
        export_scans(a.root, a.seq, work / 'pcd')
        subprocess.run([a.dufomap, str(work), a.config], check=True, stdout=subprocess.DEVNULL)
        X, rgb = read_pcd(work / 'dufomap_output.pcd')

    xyz, refl, scans = voxels(X, (rgb & 0xFF).astype(np.float64), (rgb >> 8 & 0xFFFF).astype(np.int64))
    write_map(walk_dir(a.root, a.seq) / 'map.pcd', xyz.astype(np.float32), refl.astype(np.float32),
              np.minimum(scans, 65535).astype(np.uint16))


if __name__ == '__main__':
    main()
