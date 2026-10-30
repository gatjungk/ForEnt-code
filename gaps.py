"""Write gaps.csv and the recording block of sequence.yaml for one walk.

A gap is an interval between consecutive messages longer than 1.5 times the stream's period (its median interval;
2 ms for /lowstate, timed by tick). missing_est = round(length / period) - 1.

usage: python gaps.py <dataset folder> <sequence>
"""
import argparse
import csv
import numpy as np
import yaml
from forent import walk_dir, messages, update_sequence

TOPICS = ('/livox/lidar', '/livox/imu', '/camera/color/image_raw/compressed',
          '/camera/depth/image_rect_raw/compressedDepth', '/camera/imu', '/lowstate', '/utlidar/robot_odom')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('seq')
    a = ap.parse_args()
    d = walk_dir(a.root, a.seq)
    c = yaml.safe_load(open(d / 'time_sync.yaml'))['streams']['/lowstate']['c']

    times = {}
    for topic, m, _ in messages(a.root, a.seq, TOPICS):
        t = m.tick if topic == '/lowstate' else m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        times.setdefault(topic, []).append(t)

    rows, summary = [], {}
    for topic in sorted(times):
        x = np.sort(np.array(times[topic], np.int64))
        if topic == '/lowstate':
            period, unit, to_s = 2, 1e-3, lambda v: v / 1000 + c
        else:
            period, unit, to_s = None, 1e-9, lambda v: v * 1e-9
        dx = np.diff(x)
        period = period or float(np.median(dx))
        g = np.where(dx > 1.5 * period)[0]
        missing = np.maximum(np.round(dx[g] / period).astype(np.int64) - 1, 0)
        for i, k in zip(g, missing):
            rows.append(dict(topic=topic, start_s=f'{to_s(x[i]):.6f}', end_s=f'{to_s(x[i + 1]):.6f}',
                             length_s=f'{dx[i] * unit:.6f}', missing_est=int(k)))
        summary[topic] = dict(messages=len(x), period_s=round(period * unit, 6), gaps=len(g),
                              missing_est=int(missing.sum()),
                              missing_pct=round(100 * float(missing.sum()) / (len(x) + float(missing.sum())), 2),
                              gaps_over_1s=int((dx[g] * unit > 1).sum()),
                              longest_s=round(float(dx.max() * unit), 3) if len(dx) else 0.0)

    rows.sort(key=lambda r: (r['start_s'], r['topic']))
    with open(d / 'gaps.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['topic', 'start_s', 'end_s', 'length_s', 'missing_est'])
        w.writeheader()
        w.writerows(rows)
    update_sequence(a.root, a.seq, recording=summary)


if __name__ == '__main__':
    main()
