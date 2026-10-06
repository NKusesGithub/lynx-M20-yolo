#!/usr/bin/env python3
"""
Live YOLO26 on the robot camera's RTSP stream, shown as three tiles in a row:
tracking, instance segmentation with the distance to each box, and semantic
segmentation, plus per-model inference times.

Box distance comes from geometry, not a depth model: the box's known height
(main estimate) cross-checked against the floor contact point. Settings are
under box_distance in the config; see BOX_DISTANCE.md.

Settings are in live_config.yaml (pick another file with --config).
Keys: q / Esc quit, + / - change inference fps, s save a snapshot.
"""

import argparse
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

from box_distance import BoxDistance

FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_HEIGHT = 72

# Standard Cityscapes colours (RGB), matching the 19 classes of yolo26*-sem.pt.
CITYSCAPES_RGB = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32),
]


def to_numpy(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    def resolve(p):
        if p is None or "://" in str(p):
            return p
        p = Path(p).expanduser()
        return str(p if p.is_absolute() else path.parent / p)

    cfg["source"] = resolve(cfg["source"])
    for key in ("segment", "semantic", "depth"):
        cfg["models"][key] = resolve(cfg["models"].get(key))
    return cfg


class LatestFrameGrabber:
    """Reads the stream in a background thread and keeps only the newest
    frame, so inference never works on a backlog of stale frames."""

    def __init__(self, source, transport, timeout_ms):
        self.source = source
        self.is_stream = "://" in str(source)
        if self.is_stream and transport:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay"
            )
        self.timeout_ms = timeout_ms
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.last_frame_time = 0.0
        self.stream_fps = 0.0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms,
        ])
        if self.is_stream:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        while not self.stop_event.is_set():
            cap = self._open()
            if not cap.isOpened():
                print(f"[video] cannot open {self.source}, retrying")
                time.sleep(1)
                continue
            print(f"[video] connected: {self.source}")
            file_period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
            stamps = deque(maxlen=30)
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    # A stream reconnects; a file reopens, which loops it.
                    if self.is_stream:
                        print("[video] stream lost, reconnecting")
                    break
                now = time.perf_counter()
                stamps.append(now)
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                    self.last_frame_time = now
                    if len(stamps) > 1:
                        self.stream_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                if not self.is_stream:
                    time.sleep(file_period)
            cap.release()

    def latest(self):
        with self.lock:
            return self.frame_id, self.frame

    def stop(self):
        # Exiting while FFmpeg is still inside cap.read() aborts the process
        # ("exception not rethrown"), so wait for the reader to release it.
        self.stop_event.set()
        self.thread.join(timeout=self.timeout_ms / 1000 + 1)


def put_text(img, text, org, color=(255, 255, 255), scale=0.5, bg=(0, 0, 0)):
    (w, h), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), max(int(org[1]), h + 4)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + base), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), FONT, scale, color, 1, cv2.LINE_AA)


def placeholder(size, title, message):
    tile = np.zeros((size[1], size[0], 3), np.uint8)
    put_text(tile, title, (0, 18), bg=(60, 60, 60))
    put_text(tile, message, (20, size[1] // 2), color=(160, 160, 160))
    return tile


def semantic_lut(names):
    lut = np.zeros((256, 3), np.uint8)
    for i in names:
        if len(names) == len(CITYSCAPES_RGB):
            r, g, b = CITYSCAPES_RGB[i]
            lut[i] = (b, g, r)
        else:
            lut[i] = colors(i, True)
    return lut


def extract_detections(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = to_numpy(boxes.xyxy)
    cls = to_numpy(boxes.cls).astype(int)
    conf = to_numpy(boxes.conf)
    ids = to_numpy(boxes.id).astype(int) if boxes.id is not None else [None] * len(xyxy)
    polys = result.masks.xy if result.masks is not None else [None] * len(xyxy)
    return [
        {"box": xyxy[i], "name": result.names[cls[i]], "conf": float(conf[i]),
         "id": None if ids[i] is None else int(ids[i]), "poly": polys[i]}
        for i in range(len(xyxy))
    ]


def object_mask(det, scale, size):
    # Instance polygons (masks.xy) are already in original-frame pixels,
    # so scaling them avoids the letterbox offset of the raw mask tensor.
    mask = np.zeros((size[1], size[0]), np.uint8)
    if det["poly"] is not None and len(det["poly"]) >= 3:
        cv2.fillPoly(mask, [(det["poly"] * scale).astype(np.int32)], 1)
    else:
        x1, y1, x2, y2 = (det["box"] * scale).astype(int)
        mask[max(y1, 0):y2, max(x1, 0):x2] = 1
    return mask.astype(bool)


def det_color(det):
    return colors(det["id"] if det["id"] is not None else 0, True)


def render_tracking(small, dets, scale):
    tile = small.copy()
    for d in dets:
        x1, y1, x2, y2 = (d["box"] * scale).astype(int)
        c = det_color(d)
        cv2.rectangle(tile, (x1, y1), (x2, y2), c, 2)
        track = f"ID{d['id']}" if d["id"] is not None else "ID--"
        label = f"{track} {d['name']} {d['conf']:.2f}"
        if d.get("range") is not None:
            label += f" ~{d['range']:.1f}m"
        put_text(tile, label, (x1, y1 - 2), color=(0, 0, 0), scale=0.45, bg=c)
    put_text(tile, f"Tracking + range  ({len(dets)} objects)", (0, 18), bg=(60, 60, 60))
    return tile


def render_instances(small, dets, masks):
    tile = (small * 0.35).astype(np.uint8)
    for d, m in zip(dets, masks):
        c = np.array(det_color(d), np.uint8)
        tile[m] = (0.4 * tile[m] + 0.6 * c).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs):
            put_text(tile, f"ID{d['id'] if d['id'] is not None else '--'}",
                     (xs.mean() - 12, ys.mean()), color=(0, 0, 0), scale=0.4, bg=tuple(int(v) for v in c))
    put_text(tile, "Instance segmentation (each object separated)", (0, 18), bg=(60, 60, 60))
    return tile


def render_distance_instances(small, dets, masks, scale):
    tile = (small * 0.35).astype(np.uint8)
    for d, m in zip(dets, masks):
        c = det_color(d)
        tile[m] = (0.4 * tile[m] + 0.6 * np.array(c, np.uint8)).astype(np.uint8)
    for d in dets:
        c, fg = det_color(d), (0, 0, 0)
        x1, y1 = (d["box"][:2] * scale).astype(int)
        label = f"ID{d['id']}" if d["id"] is not None else "ID--"
        dist = d.get("dist")
        if dist is not None:
            # The column where top and bottom were measured.
            u, v_top, v_bottom = (np.array(dist["points"]) * scale).astype(int)
            cv2.line(tile, (u, v_top), (u, v_bottom), (255, 255, 255), 1)
            if dist["height"] is None:
                label += " ?"
            else:
                label += f" {dist['height']:.2f}m"
                if dist["floor"] is not None:
                    label += f"  floor {dist['floor']:.2f}m"
                if dist["uncertain"]:
                    label += " ?"
                    c, fg = (0, 0, 255), (255, 255, 255)
        put_text(tile, label, (x1, y1 - 2), color=fg, scale=0.45, bg=c)
    put_text(tile, "Instances + distance (box height)", (0, 18), bg=(60, 60, 60))
    put_text(tile, "red label / ? = height and floor estimates disagree",
             (0, tile.shape[0] - 6), scale=0.4)
    return tile


def render_semantic(small, sem_small, names, lut):
    tile = cv2.addWeighted(small, 0.4, lut[sem_small], 0.6, 0)
    counts = np.bincount(sem_small.ravel(), minlength=256)
    total = counts.sum()
    y = 44
    for cls in np.argsort(counts)[::-1][:6]:
        if counts[cls] == 0 or cls not in names:
            break
        x = tile.shape[1] - 150
        cv2.rectangle(tile, (x, y - 12), (x + 14, y + 2), tuple(int(v) for v in lut[cls]), -1)
        put_text(tile, f"{names[cls]} {100 * counts[cls] / total:.0f}%", (x + 18, y), scale=0.42)
        y += 20
    put_text(tile, "Semantic segmentation (scene classes)", (0, 18), bg=(60, 60, 60))
    return tile


class LivePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        m = cfg["models"]
        self.seg = YOLO(m["segment"])
        self.sem = YOLO(m["semantic"]) if m.get("semantic") else None
        self.sem_lut = None
        self.sem_names = None
        bd = cfg.get("box_distance") or {}
        self.box_cfg = bd if bd.get("enabled") else None
        self.box_distance = None  # built on the first frame, once its size is known
        if self.box_cfg and not set(bd["box_class_names"]) & set(self.seg.names.values()):
            print(f"[distance] WARNING: none of {bd['box_class_names']} are classes of "
                  f"{m['segment']} ({list(self.seg.names.values())}); no distances will be shown")
        self.first_seen = {}
        self.new_ids = deque()

    def _timed(self, fn):
        t = time.perf_counter()
        out = fn()
        return out, (time.perf_counter() - t) * 1000

    def process(self, frame, display_width):
        cfg = self.cfg
        h, w = frame.shape[:2]
        tw = display_width // 3
        th = int(round(tw * h / w))
        scale = tw / w
        small = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        timings = {}

        seg_res, timings["track+seg"] = self._timed(lambda: self.seg.track(
            frame, persist=True, tracker=cfg["tracker"], conf=cfg["conf"],
            imgsz=cfg["imgsz"], classes=cfg.get("classes"), device=cfg["device"],
            verbose=False)[0])
        dets = extract_detections(seg_res)
        masks = [object_mask(d, scale, (tw, th)) for d in dets]

        if self.box_cfg:
            if self.box_distance is None:
                self.box_distance = BoxDistance(self.box_cfg, (w, h))
            self.box_distance.update(dets, h)
            for d in dets:
                if "dist" in d:
                    d["range"] = d["dist"]["height"]

        sem_small = None
        if self.sem is not None:
            sres, timings["semantic"] = self._timed(lambda: self.sem.predict(
                frame, device=cfg["device"], verbose=False)[0])
            sem_small = cv2.resize(to_numpy(sres.semantic_mask.data).astype(np.uint8),
                                   (tw, th), interpolation=cv2.INTER_NEAREST)
            if self.sem_lut is None:
                self.sem_lut = semantic_lut(sres.names)
                self.sem_names = sres.names

        now = time.perf_counter()
        for d in dets:
            if d["id"] is not None and d["id"] not in self.first_seen:
                self.first_seen[d["id"]] = now
                self.new_ids.append(now)
        while self.new_ids and now - self.new_ids[0] > 60:
            self.new_ids.popleft()

        tiles = [
            render_tracking(small, dets, scale),
            render_distance_instances(small, dets, masks, scale)
            if self.box_cfg else render_instances(small, dets, masks),
            render_semantic(small, sem_small, self.sem_names, self.sem_lut)
            if sem_small is not None else placeholder((tw, th), "Semantic segmentation", "disabled in config"),
        ]
        grid = np.hstack(tiles)
        stats = {"timings": timings, "active": len(dets), "new_ids_60s": len(self.new_ids)}
        return grid, stats


def status_bar(width, lines):
    bar = np.zeros((STATUS_HEIGHT, width, 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (8, 20 + i * 22), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return bar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).with_name("live_config.yaml")))
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    fps = float(cfg["inference_fps"])
    width = int(cfg["display_width"])
    snap_dir = Path(__file__).parent / "results" / "live"

    print("Loading models...")
    pipeline = LivePipeline(cfg)
    grabber = LatestFrameGrabber(cfg["source"], cfg.get("rtsp_transport"), int(cfg["open_timeout_ms"]))

    window = "YOLO26 live"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    grid, stats = None, None
    last_frame_id = 0
    next_time = 0.0
    infer_stamps = deque(maxlen=20)

    try:
        while True:
            frame_id, frame = grabber.latest()
            now = time.perf_counter()
            if frame is not None and frame_id != last_frame_id and now >= next_time:
                next_time = now + 1.0 / fps
                last_frame_id = frame_id
                grid, stats = pipeline.process(frame, width)
                infer_stamps.append(time.perf_counter())

            video_ok = frame is not None and now - grabber.last_frame_time < 2
            actual = ((len(infer_stamps) - 1) / (infer_stamps[-1] - infer_stamps[0])
                      if len(infer_stamps) > 1 else 0.0)
            lines = [
                (f"Video: {'OK' if video_ok else 'NO VIDEO - waiting'}  {grabber.stream_fps:.1f} fps   "
                 f"Inference: {actual:.1f} fps (target {fps:g})   imgsz {cfg['imgsz']}"),
                ("Inference ms: " + "   ".join(f"{k} {v:.1f}" for k, v in stats["timings"].items())
                 + f"   total {sum(stats['timings'].values()):.1f}") if stats else "Inference ms: -",
                (f"Tracks: {stats['active']} in view, {stats['new_ids_60s']} new IDs in last 60s   "
                 if stats else "Tracks: -   ") + "|   keys: q quit   +/- fps   s snapshot",
            ]
            view = grid if grid is not None else placeholder(
                (width, width * 9 // 16), "YOLO26 live", f"Waiting for video: {cfg['source']}")
            canvas = np.vstack([view, status_bar(view.shape[1], lines)])
            cv2.imshow(window, canvas)

            # Sleep until the next inference is due, capped so the window
            # still redraws and reacts to keys ~30 times a second.
            until_next = (next_time - time.perf_counter()) * 1000
            wait_ms = int(min(max(until_next, 1), 30)) if video_ok else 30
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("+"), ord("=")):
                fps = min(fps + (0.5 if fps < 2 else 1), 30)
                print(f"inference_fps = {fps:g}")
            elif key == ord("-"):
                fps = max(fps - (0.5 if fps <= 2 else 1), 0.5)
                print(f"inference_fps = {fps:g}")
            elif key == ord("s"):
                snap_dir.mkdir(parents=True, exist_ok=True)
                path = snap_dir / time.strftime("snap_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(str(path), canvas)
                print(f"saved {path}")
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
