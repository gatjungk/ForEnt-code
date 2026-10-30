"""Write time_sync.yaml for one walk: the /lowstate clock offset, /lowcmd timing and the timing checks.

usage: python time_sync.py <dataset folder> <sequence> [--video <video_sync output .yaml>]
"""
import argparse
import numpy as np
import yaml
from forent import walk_dir, messages, stamp, lowstate_offset

TOPICS = ('/lowstate', '/lowcmd', '/utlidar/robot_odom', '/livox/imu', '/camera/imu')
LOWCMD_TIME = ('time of the /lowstate recorded just before it in the bag; '
               'two commands after one state are spaced evenly up to the next state')


def peak(X, Y, lo, hi, rate=200.0, maxlag=0.1):
    """Lag (ms) to add to X's times to match Y in [lo, hi], with sub-sample refinement; None if not measurable."""
    g = np.arange(lo, hi, 1 / rate)
    y = np.interp(g, Y[:, 0], Y[:, 1])
    if y.std() < 0.15:
        return None
    y = (y - y.mean()) / y.std()
    lags = np.arange(-maxlag, maxlag, 1 / rate)
    cs = []
    for lag in lags:
        x = np.interp(g, X[:, 0] + lag, X[:, 1])
        cs.append(float(np.mean((x - x.mean()) / (x.std() + 1e-12) * y)))
    cs = np.array(cs)
    i = int(np.argmax(cs))
    if cs[i] < 0.6 or i in (0, len(cs) - 1):
        return None
    d = (cs[i - 1] - cs[i + 1]) / (2 * (cs[i - 1] - 2 * cs[i] + cs[i + 1]))
    return (lags[i] + d / rate) * 1000


def check(X, Y):
    """Lag per 30 s window, summarised as median, 95 % range and window count."""
    t0, t1 = max(X[0, 0], Y[0, 0]) + 5, min(X[-1, 0], Y[-1, 0]) - 5
    lags = [round(p, 3) for lo in np.arange(t0, t1 - 30, 30) if (p := peak(X, Y, lo, lo + 30)) is not None]
    if not lags:
        return 'not measured'
    r95 = [round(float(v), 2) for v in np.percentile(lags, [2.5, 97.5])]
    return f'{{median_ms: {round(float(np.median(lags)), 2)}, range95_ms: {r95}, windows: {len(lags)}}}'


def lowcmd_times(ts, order):
    """Each command takes the time of the preceding /lowstate; several commands after one state are spaced evenly."""
    order = np.array(order)
    prev = (np.cumsum(order == 0) - 1)[order == 1]
    tc, spaced = np.full(len(prev), np.nan), np.zeros(len(prev), bool)
    for p, s0, n in zip(*np.unique(prev, return_index=True, return_counts=True)):
        if p < 0:
            continue
        tn = ts[p + 1] if p + 1 < len(ts) else ts[p] + 0.002 * n
        tc[s0:s0 + n] = ts[p] + np.arange(n) * (tn - ts[p]) / n
        spaced[s0:s0 + n] = n > 1
    return tc, spaced


def response_lag(ts, q, tc, cmd):
    """Median over joints of the lag (ms) of the measured joint angle behind its commanded target."""
    act = (cmd[:, 12] > 5) & np.isfinite(tc)
    g = np.arange(ts[0] + 5, ts[-1] - 5, 0.001)
    n = len(g) - 120
    lags = []
    for j in range(12):
        a = np.interp(g, tc[act], cmd[act, j])
        b = np.interp(g, ts, q[:, j])
        a, b = a - a.mean(), b - b.mean()
        lags.append(int(np.argmax([np.dot(a[20:20 + n], b[20 + k:20 + k + n]) for k in range(-20, 101)])) - 20)
    return int(np.median(lags))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('seq')
    ap.add_argument('--video')
    a = ap.parse_args()

    go2, q, cmd, odom, mid, cam, order = [], [], [], [], [], [], []
    for topic, m, t in messages(a.root, a.seq, TOPICS):
        if topic == '/lowstate':
            go2.append((m.tick * 1e-3, *m.imu_state.gyroscope, t / 1e9))
            q.append([s.q for s in m.motor_state[:12]])
            order.append(0)
        elif topic == '/lowcmd':
            cmd.append([c.q for c in m.motor_cmd[:12]] + [float(np.median([c.kp for c in m.motor_cmd[:12]]))])
            order.append(1)
        elif topic == '/utlidar/robot_odom':
            odom.append((stamp(m), m.twist.twist.angular.z))
        else:
            w = m.angular_velocity
            (mid if topic == '/livox/imu' else cam).append((stamp(m), w.x, w.y, w.z))
    go2, q, cmd = np.array(go2), np.array(q), np.array(cmd)
    odom, mid, cam = np.array(sorted(odom)), np.array(sorted(mid)), np.array(sorted(cam))

    c = lowstate_offset(go2, mid)
    ts = go2[:, 0] + c
    tc, spaced = lowcmd_times(ts, order)

    def mag(A):
        return np.c_[A[:, 0], np.linalg.norm(A[:, 1:4], axis=1)]

    G, L, C = np.c_[ts, np.linalg.norm(go2[:, 1:4], axis=1)], mag(mid), mag(cam)
    video = yaml.safe_load(open(a.video)) if a.video else {}

    def v(k):
        return video.get(k, 'not available')

    text = f"""sequence: {a.seq}
streams:
  /livox/lidar: {{time: header stamp}}
  /livox/imu: {{time: header stamp}}
  /camera/color/image_raw/compressed: {{time: header stamp}}
  /camera/depth/image_rect_raw/compressedDepth: {{time: header stamp}}
  /camera/imu: {{time: header stamp}}
  /lowstate:
    time: tick / 1000 + c
    c: {round(c, 6)!r}
  /lowcmd:
    time: {LOWCMD_TIME}
    messages: {len(tc)}
    spaced: {int(spaced.sum())}
  /wirelesscontroller: {{time: none, use: wireless_remote in /lowstate}}
  /utlidar/robot_odom:
    time: header stamp on the Go2's own clock, not corrected
    lag_vs_mid360: {check(odom, mid[:, [0, 3]])}
  video:
    file: videos/{a.seq}.mp4
    time: frame time + offset_s
    offset_s: {v('offset_s')!r}
    offset_ns: {v('offset_ns')}
    frame_rate: {v('frame_rate')}
    measured_on: {v('measured_on')}
    match_r: {v('match_r')}
    sit_down_check_ms: {v('sit_down_check_ms')}
checks:
  go2_vs_mid360: {check(G, L)}
  d435i_vs_mid360: {check(C, L)}
  d435i_vs_go2: {check(C, G)}
  joint_response_lag_ms: {response_lag(ts, q, tc, cmd)}
"""
    (walk_dir(a.root, a.seq) / 'time_sync.yaml').write_text(text)


if __name__ == '__main__':
    main()
