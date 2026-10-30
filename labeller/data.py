"""One walk from the dataset folder: robot signals on a common time grid, candidate entrapments and video timing.

Times are seconds after the first message in the bag (rel_s in the annotation files).
"""
import csv
import struct
from pathlib import Path
import av
import numpy as np
import yaml
from rosbags.highlevel import AnyReader

DT = 0.02
STALE = 1.0                  # /lowstate values older than this count as missing
GAIN, DELAY = 1.279, 0.28    # commanded speed per unit stick (m/s) and its delay (s)
WINDOW = 3.0                 # window for the distance the robot is behind its command (s)
BEHIND = 0.7                 # candidate when the robot is this far behind its command (m)
ROLL = 70.0                  # or rolled this far (deg)
MERGE, MIN_LEN = 1.0, 1.0    # join candidates closer than this, drop shorter ones (s)


def walk_dir(root, seq):
    return Path(root) / f"site_{seq.split('_')[1]}" / seq


def hold(src_t, src, t):
    """Last sample at or before each t; NaN where it is older than STALE."""
    i = np.searchsorted(src_t, t, side='right') - 1
    out = src[np.clip(i, 0, len(src) - 1)].astype(float)
    out[(i < 0) | (t - src_t[np.clip(i, 0, len(src_t) - 1)] > STALE)] = np.nan
    return out


def window_sum(x, n):
    """Integral of x (trapezoid rule) over t - n DT .. t + n DT; NaN if any sample in it is NaN."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    c = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 * DT)])
    bad = np.concatenate([[0], np.cumsum(~ok)])
    i = np.arange(len(x))
    lo, hi = np.clip(i - n, 0, len(x) - 1), np.clip(i + n, 0, len(x) - 1)
    out = c[hi] - c[lo]
    out[bad[hi + 1] - bad[lo] > 0] = np.nan
    return out


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


class Walk:
    def __init__(self, root, seq):
        self.seq = seq
        d = walk_dir(root, seq)
        sync = yaml.safe_load(open(d / 'time_sync.yaml'))['streams']
        self.read_lowstate(d / f'{seq}.mcap', sync['/lowstate']['c'])
        self.read_trajectory(d / 'trajectory.csv')
        self.find_candidates()
        self.video = Path(root) / 'videos' / f'{seq}.mp4'
        offset = sync['video'].get('offset_s')
        self.video_offset = offset if isinstance(offset, float) and self.video.exists() else None
        if self.video_offset is not None:
            with av.open(str(self.video)) as c:
                self.fps = float(c.streams.video[0].average_rate)

    def read_lowstate(self, bag, c):
        tick, stick, rpy, dq = [], [], [], []
        with AnyReader([bag]) as r:
            self.start, self.end = r.start_time * 1e-9, r.end_time * 1e-9
            con = [x for x in r.connections if x.topic == '/lowstate']
            for x, _, raw in r.messages(connections=con):
                m = r.deserialize(raw, x.msgtype)
                tick.append(m.tick)
                w = bytes(m.wireless_remote)
                stick.append([struct.unpack_from('<f', w, o)[0] for o in (4, 20, 8)])   # lx, ly, rx
                rpy.append(m.imu_state.rpy[:2])
                dq.append(np.abs([s.dq for s in m.motor_state[:12]]).sum())
        ts = np.array(tick) * 1e-3 + c - self.start
        self.t = np.arange(0, self.end - self.start, DT)
        self.stick = np.stack([hold(ts, np.array(stick)[:, k], self.t) for k in range(3)], 1)
        self.roll, self.pitch = (np.degrees(hold(ts, np.array(rpy)[:, k], self.t)) for k in range(2))
        cs = np.concatenate([[0.0], np.cumsum(dq)])
        a, b = np.searchsorted(ts, self.t - 0.05), np.searchsorted(ts, self.t + 0.05)
        with np.errstate(invalid='ignore'):
            self.legs = (cs[b] - cs[a]) / (b - a)
        self.legs[np.isnan(self.stick[:, 1])] = np.nan

    def read_trajectory(self, path):
        rows = list(csv.DictReader(open(path)))
        t = np.array([float(r['t']) for r in rows]) - self.start
        xy = np.array([[float(r['x']), float(r['y'])] for r in rows])
        yaw = np.unwrap(np.radians([float(r['yaw_deg']) for r in rows]))
        v = np.array([[float(r['vx_body'] or 'nan'), float(r['vy_body'] or 'nan')] for r in rows])

        def mean_velocity(i, j):
            """Body velocity from the positions at poses i and j (across lidar gaps)."""
            w = (xy[j] - xy[i]) / (t[j] - t[i])
            h = 0.5 * (yaw[i] + yaw[j])
            return np.array([np.cos(h) * w[0] + np.sin(h) * w[1], -np.sin(h) * w[0] + np.cos(h) * w[1]])

        for a, b in runs(np.isnan(v[:, 0])):
            if 0 < a and b < len(t) - 1:
                v[a:b + 1] = mean_velocity(a - 1, b + 1)
        self.velocity = np.stack([np.interp(self.t, t, v[:, k], left=np.nan, right=np.nan) for k in range(2)], 1)
        # where the measured speed is only an average over a lidar gap or the odometry was not confirmed
        bad = np.array([bool({'unverified', 'odometry_replaced', 'fall'} & set(r['flag'].split(';'))) for r in rows])
        self.unreliable = [(t[a] - 0.05, t[b] + 0.05) for a, b in runs(bad)]
        self.unreliable += [(t[i], t[i + 1]) for i in np.where(np.diff(t) > 0.5)[0]]
        for i in np.where(np.diff(t) > 0.5)[0]:
            self.velocity[(self.t > t[i]) & (self.t < t[i + 1])] = mean_velocity(i, i + 1)

    def find_candidates(self):
        """Spans where the robot falls behind its joystick command, or rolls over."""
        k = int(round(DELAY / DT))
        lx, ly = (np.concatenate([np.full(k, s[0]), s[:-k]]) for s in (self.stick[:, 0], self.stick[:, 1]))
        cmd = GAIN * np.stack([ly, -lx], 1)
        speed = np.hypot(cmd[:, 0], cmd[:, 1])
        with np.errstate(invalid='ignore'):
            active = speed >= 0.05
            along = np.where(active, (self.velocity * cmd).sum(1) / np.where(active, speed, 1), 0.0)
        self.cmd_speed = np.where(active, speed, 0.0)
        self.obs_speed = along
        self.cmd_speed[np.isnan(speed)] = self.obs_speed[np.isnan(speed)] = np.nan
        n = int(round(WINDOW / 2 / DT))
        self.behind = window_sum(self.cmd_speed, n) - window_sum(self.obs_speed, n)
        flag = (np.nan_to_num(self.behind, nan=-1) > BEHIND) | (np.abs(np.nan_to_num(self.roll)) >= ROLL)
        spans = []
        for a, b in runs(flag):
            if spans and self.t[a] - spans[-1][1] <= MERGE:
                spans[-1][1] = self.t[b]
            else:
                spans.append([self.t[a], self.t[b]])
        self.candidates = [(a, b) for a, b in spans if b - a >= MIN_LEN]

    def video_time(self, rel):
        return self.start + rel - self.video_offset
