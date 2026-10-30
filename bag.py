"""Convert one walk's recorded ROS 2 Foxy bag (SQLite) to the released MCAP, then check it against the recording.

Colour and depth images are stored as PNG (sensor_msgs/CompressedImage in the image_transport 'compressed' and
'compressedDepth' formats); every other message is copied byte for byte. /pointcloud and /sportmodestate are not
released. Every message keeps its bag timestamp, and the MCAP carries the definitions of every message type
(from common/tools/msgs in the dataset).

The check reads both bags again: every released topic must have the same timestamps and the same bytes, and every
image the same header and pixels.

usage: python bag.py <recorded bag folder> <dataset folder> <sequence>
"""
import argparse
import hashlib
import shutil
import struct
from multiprocessing import Pool
from pathlib import Path
import cv2
import numpy as np
from rosbags.rosbag2 import Reader, Writer
from rosbags.rosbag2.enums import CompressionFormat, CompressionMode, StoragePlugin
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from forent import walk_dir

COLOR, DEPTH = '/camera/color/image_raw', '/camera/depth/image_rect_raw'
NEW = {COLOR: COLOR + '/compressed', DEPTH: DEPTH + '/compressedDepth'}
IMAGE, COMPRESSED = 'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage'
DROP = ('/pointcloud', '/sportmodestate')
DEPTH_HEADER = struct.pack('<iff', 1, 0.0, 0.0)   # compressedDepth header: PNG, no depth quantisation
STORE = None


def typestore(msgs):
    ts = get_typestore(Stores.ROS2_FOXY)
    for pkg in ('unitree_go', 'livox_ros_driver2'):
        for f in sorted((Path(msgs) / pkg).glob('*.msg')):
            ts.register(get_types_from_msg(f.read_text(), f'{pkg}/msg/{f.stem}'))
    return ts


def init(msgs):
    global STORE
    STORE = typestore(msgs)


def encode(item):
    topic, t, raw = item
    if topic not in NEW:
        return topic, t, raw
    m = STORE.deserialize_cdr(raw, IMAGE)
    if topic == COLOR:
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)[:, :, ::-1]
        data = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes()
        fmt = 'rgb8; png compressed bgr8'
    else:
        img = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width)
        data = DEPTH_HEADER + cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes()
        fmt = '16UC1; compressedDepth png'
    out = STORE.types[COMPRESSED](header=m.header, format=fmt, data=np.frombuffer(data, np.uint8))
    return NEW[topic], t, bytes(STORE.serialize_cdr(out, COMPRESSED))


def image_key(header, pixels):
    h = header.stamp
    stamp = repr((h.sec, h.nanosec, header.frame_id)).encode()
    return hashlib.sha1(np.ascontiguousarray(pixels).tobytes() + stamp).hexdigest()


def source_keys(bag):
    keys = {}
    with Reader(bag) as r:
        for c, t, raw in r.messages(connections=[c for c in r.connections if c.topic not in DROP]):
            if c.topic in NEW:
                m = STORE.deserialize_cdr(raw, IMAGE)
                k = image_key(m.header, np.frombuffer(m.data, np.uint8 if c.topic == COLOR else np.uint16))
            else:
                k = hashlib.sha1(raw).hexdigest()
            keys.setdefault(NEW.get(c.topic, c.topic), []).append((t, k))
    return keys


def release_keys(path):
    keys = {}
    with Reader(path) as r:
        for c, t, raw in r.messages():
            if c.topic in NEW.values():
                m = STORE.deserialize_cdr(raw, COMPRESSED)
                buf = m.data.tobytes()
                if c.topic == NEW[COLOR]:
                    px = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
                else:
                    px = cv2.imdecode(np.frombuffer(buf[12:], np.uint8), cv2.IMREAD_UNCHANGED)
                k = image_key(m.header, px)
            else:
                k = hashlib.sha1(raw).hexdigest()
            keys.setdefault(c.topic, []).append((t, k))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('root')
    ap.add_argument('seq')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    msgs = Path(a.root) / 'common/tools/msgs'
    init(msgs)
    d = walk_dir(a.root, a.seq)
    tmp = d / 'mcap_tmp'
    d.mkdir(parents=True, exist_ok=True)

    with Reader(a.bag) as r:
        keep = [c for c in r.connections if c.topic not in DROP]
        w = Writer(tmp, version=9, storage_plugin=StoragePlugin.MCAP)
        w.set_compression(CompressionMode.STORAGE, CompressionFormat.ZSTD)
        with w:
            con = {}
            for c in keep:
                topic, msgtype = (NEW[c.topic], COMPRESSED) if c.topic in NEW else (c.topic, c.msgtype)
                con[topic] = w.add_connection(topic, msgtype, typestore=STORE,
                                              offered_qos_profiles=c.ext.offered_qos_profiles)
            items = ((c.topic, t, raw) for c, t, raw in r.messages(connections=keep))
            with Pool(a.workers, initializer=init, initargs=(msgs,)) as pool:
                for topic, t, raw in pool.imap(encode, items, chunksize=32):
                    w.write(con[topic], t, raw)
    out = d / f'{a.seq}.mcap'
    (tmp / 'mcap_tmp.mcap').replace(out)
    shutil.rmtree(tmp)

    src, rel = source_keys(a.bag), release_keys(out)
    ok = sorted(src) == sorted(rel)
    for topic in sorted(set(src) | set(rel)):
        same = sorted(src.get(topic, [])) == sorted(rel.get(topic, []))
        ok &= same
        print(f'{topic}: {len(src.get(topic, []))} -> {len(rel.get(topic, []))} messages, '
              f'{"identical" if same else "DIFFERENT"}')
    print(f'{a.seq}: {"all messages identical" if ok else "CHECK FAILED"}')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
