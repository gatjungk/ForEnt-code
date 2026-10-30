"""Time offset of a walk's third-person video (bag time = video time + offset_s), for time_sync.py --video.

The robot's body height from its leg joint angles (/lowstate) is matched to its height in the video at the stand-up
that starts the walk, or at the sit-down that ends it. OWLv2 finds the robot, SAM 2.1 segments it, and the offset is the
shift with the highest correlation between the two height signals. The videos start within a few seconds of the bag,
so the search is centred on the bag start.

usage: python video_sync.py <dataset folder> <sequence> --sam <sam2.1_hiera_small.pt> --out <video.yaml>
"""
import argparse
import tempfile
from pathlib import Path
import av
import cv2
import numpy as np
import torch
import yaml
from scipy.ndimage import median_filter
from forent import messages, stamp, lowstate_offset

LEG = 0.213
WIN, SEARCH, SCALE = 16.0, 12.0, 0.5
R_MIN, STEP_MIN, STEP_MAX = 0.97, 10.0, 150.0
OWL = 'google/owlv2-base-patch16-ensemble'
SAM_CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'
GAP_PX = 15
GAP = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GAP_PX, GAP_PX))


def robot_height(root, seq):
    """Body height (m) at 100 Hz on the bag clock, and the bag start time."""
    go2, mid, q = [], [], []
    for topic, m, t in messages(root, seq, ('/lowstate', '/livox/imu')):
        if topic == '/lowstate':
            go2.append((m.tick * 1e-3, *m.imu_state.gyroscope, t / 1e9))
            q.append([s.q for s in m.motor_state[:12]])
        else:
            w = m.angular_velocity
            mid.append((stamp(m), w.x, w.y, w.z))
    go2, mid = np.array(go2), np.array(sorted(mid))
    t = go2[::5, 0] + lowstate_offset(go2, mid)
    q = np.array(q)[::5].reshape(-1, 4, 3)
    h = np.mean(LEG * np.cos(q[:, :, 1]) + LEG * np.cos(q[:, :, 1] + q[:, :, 2]), 1)
    return t, h, go2[0, 4]


def crossings(t, x, level):
    i = np.where(np.diff(np.sign(x - level)) != 0)[0]
    return [(t[j] + (level - x[j]) * (t[j + 1] - t[j]) / (x[j + 1] - x[j]), 1 if x[j + 1] > x[j] else -1) for j in i]


def transition(t, x, direction, near):
    """Start and end of the transition nearest `near` (smoothed slope over 10 % of peak) and the levels either side."""
    k = max(1, int(round(0.15 / np.median(np.diff(t)))))
    sl = np.gradient(np.convolve(x, np.ones(k) / k, 'same'), t) * direction
    j = int(np.argmax(np.where(np.abs(t - near) < 2, sl, -np.inf)))
    a = b = j
    while a > 0 and sl[a - 1] > 0.1 * sl[j]:
        a -= 1
    while b < len(t) - 1 and sl[b + 1] > 0.1 * sl[j]:
        b += 1
    pre = np.median(x[(t > t[a] - 0.8) & (t < t[a] - 0.1)])
    post = np.median(x[(t > t[b] + 0.1) & (t < t[b] + 0.8)])
    return t[a], t[b], pre, post


def robot_events(t, h):
    """Stand-up (first 90 s) and sit-down (last 90 s): 50 % time, transition and height scaled 0 lying .. 1 standing."""
    cr = crossings(t, h, 0.2)
    ev = {}
    up = [c for c, d in cr if d > 0 and c - t[0] < 90]
    if up:
        ta, tb, pre, post = transition(t, h, 1, up[0])
        n = (h - pre) / (post - pre)
        c = [x for x, d in crossings(t, n, 0.5) if d > 0 and ta - 0.2 <= x <= tb + 0.2]
        if c:
            ev['stand_up'] = dict(t50=c[0], n=n, win=(ta, tb))
    down = [c for c, d in cr if d < 0 and t[-1] - c < 90]
    if down:
        ta, tb, pre, post = transition(t, h, -1, down[-1])
        if pre > 0.28 and post < 0.12 and 1.5 < tb - ta < 2.5:
            n = (h - post) / (pre - post)
            c = [x for x, d in crossings(t, n, 0.5) if d < 0 and ta - 0.2 <= x <= tb + 0.2]
            if c:
                ev['sit_down'] = dict(t50=c[-1], n=n, win=(ta, tb))
    return ev


def decode(path, t0, t1):
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        v.thread_type = 'AUTO'
        c.seek(int(max(0, t0 - 2) / v.time_base), stream=v, backward=True, any_frame=False)
        T, F = [], []
        for fr in c.decode(v):
            if fr.time < t0:
                continue
            if fr.time > t1:
                break
            T.append(fr.time)
            F.append(cv2.resize(fr.to_ndarray(format='rgb24'), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA))
    return np.array(T), F


def find_robot(owl, img):
    proc, model = owl
    inp = proc(text=[['a robot dog', 'a quadruped robot', 'a robot']], images=img, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(**inp)
    size = torch.tensor([[max(img.shape[:2])] * 2], device=model.device)
    r = proc.post_process_grounded_object_detection(out, threshold=0.05, target_sizes=size)[0]
    if len(r['scores']) == 0:
        return None, 0.0
    i = int(r['scores'].argmax())
    return r['boxes'][i].cpu().numpy(), float(r['scores'][i])


def mask_heights(sam, frames, start, box):
    """Height in pixels of the robot's mask in every frame, tracked both ways from `start`.

    Parts of the mask within GAP_PX of each other count as one; only the largest such part is kept.
    """
    H = np.full(len(frames), np.nan)
    with tempfile.TemporaryDirectory() as d:
        for i, im in enumerate(frames):
            cv2.imwrite(f'{d}/{i:05d}.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            st = sam.init_state(video_path=d, offload_video_to_cpu=True, offload_state_to_cpu=True)
            sam.add_new_points_or_box(st, frame_idx=start, obj_id=1, box=box)
            for reverse in (False, True):
                for fi, _, logits in sam.propagate_in_video(st, start_frame_idx=start, reverse=reverse):
                    m = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
                    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, GAP), connectivity=8)
                    if n > 1:
                        main = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                        rows = np.where((main & (m > 0)).any(1))[0]
                        H[fi] = rows[-1] - rows[0] + 1
            sam.reset_state(st)
    return H


def best_shift(tr, nr, tv, hv, a, b):
    """Shift s with the highest correlation of robot(t) and video(t - s) over [a, b], refined by a parabola."""
    g = np.arange(a, b, 0.01)
    x = np.interp(g, tr, nr)
    ss = np.arange(-SEARCH, SEARCH, 0.005)
    r = np.full(len(ss), np.nan)
    for i, s in enumerate(ss):
        if g[0] - s >= tv[0] and g[-1] - s <= tv[-1]:
            r[i] = np.corrcoef(x, np.interp(g - s, tv, hv))[0, 1]
    k = int(np.nanargmax(r))
    s = ss[k]
    if 0 < k < len(r) - 1 and np.all(np.isfinite(r[k - 1:k + 2])):
        y0, y1, y2 = r[k - 1:k + 2]
        if y0 - 2 * y1 + y2:
            s += 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * 0.005
    return float(s), float(r[k])


def match(video, owl, sam, t, e, name, guess, duration):
    """Offset and correlation for one event, and the robot's height change in the image (%)."""
    tv = e['t50'] - guess
    if not WIN * 0.3 < tv < duration - WIN * 0.3:
        return None
    T, F = decode(video, max(0, tv - WIN), min(duration, tv + WIN))
    dets = [(find_robot(owl, F[int(i)]), int(i)) for i in np.linspace(0.1, 0.9, 5) * (len(F) - 1)]
    (box, _), idx = max(dets, key=lambda x: x[0][1])
    if box is None:
        return None
    H = mask_heights(sam, F, idx, box)
    ok = ~np.isnan(H)
    Tv, hv = T[ok], median_filter(H[ok], size=5, mode='nearest')
    # 1.5 s of the lying robot, the transition and 0.3 s standing (the robot soon walks off or is hidden)
    ta, tb = e['win']
    a, b = (ta - 1.5, tb + 0.3) if name == 'stand_up' else (ta - 0.3, tb + 1.5)
    s, r = best_shift(t, e['n'], Tv + guess, hv, a, b)
    off = guess + s
    ta, tb = ta - off, tb - off
    if name == 'stand_up':
        lying, standing = (Tv > ta - 1.5) & (Tv < ta - 0.1), (Tv > tb + 0.1) & (Tv < tb + 0.7)
    else:
        lying, standing = (Tv > tb + 0.1) & (Tv < tb + 1.5), (Tv > ta - 0.7) & (Tv < ta - 0.1)
    step = 100 * (np.median(hv[standing]) - np.median(hv[lying])) / np.median(hv[lying])
    return float(off), r, float(step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('seq')
    ap.add_argument('--sam', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    from sam2.build_sam import build_sam2_video_predictor

    video = Path(a.root) / 'videos' / f'{a.seq}.mp4'
    with av.open(str(video)) as c:
        duration = c.duration / 1e6
    t, h, bag_start = robot_height(a.root, a.seq)
    owl = Owlv2Processor.from_pretrained(OWL), Owlv2ForObjectDetection.from_pretrained(OWL).to('cuda').eval()
    sam = build_sam2_video_predictor(SAM_CONFIG, a.sam, device='cuda')

    found = {}
    for name, e in robot_events(t, h).items():
        m = match(video, owl, sam, t, e, name, bag_start, duration)
        if m is None:
            print(f'{name}: not measured')
            continue
        print(f'{name}: offset {m[0]:.4f} s, r {m[1]:.3f}, height change {m[2]:.0f} %')
        if m[1] >= R_MIN and STEP_MIN <= m[2] <= STEP_MAX:
            found[name] = m
    out = {}
    if found:
        name = 'stand_up' if 'stand_up' in found else 'sit_down'
        off, r = found[name][:2]
        out = dict(offset_s=round(off, 4), offset_ns=int(round(off * 1e6)) * 1000, frame_rate=23.976,
                   measured_on=name, match_r=round(r, 3))
        if len(found) == 2:
            out['sit_down_check_ms'] = int(round((found['sit_down'][0] - found['stand_up'][0]) * 1e3))
    Path(a.out).write_text(yaml.safe_dump(out, sort_keys=False))


if __name__ == '__main__':
    main()
