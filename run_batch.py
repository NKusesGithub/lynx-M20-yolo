#!/usr/bin/env python3
"""
Run YOLO detection, instance segmentation, semantic segmentation, and monocular
depth on a video sampled at a fixed fps, and save per-frame results for review.

--family picks the detection and instance-segmentation models: yolo26 (the
default) or yolo11. YOLO11 has no semantic-segmentation or depth models, so
those always come from YOLO26. That keeps depth constant, so yolo11 and yolo26
runs are a fair A/B of the detector alone.

Read this before trusting the labels/numbers:

- Detection and instance-segmentation weights are COCO-trained (80 classes), so
  an object gets labeled as the nearest COCO class (an animal as elephant, cow,
  zebra, ...). Treat the label as "an object was detected here", not a species ID.
- Nothing is filtered by class: people, cars and bags are counted alongside
  animals. The console prints a per-class breakdown of what was found.
- The semantic-segmentation weights are trained on driving-scene classes (road,
  sidewalk, vegetation, terrain, sky, car, person, ...). They give only rough
  scene context, not individual objects.
- Depth values are monocular relative-depth estimates, not calibrated meters.
  Use them to rank which objects are nearer/farther, not as absolute distances.
- Instance segmentation (with per-pixel masks) is what separates individual
  objects in a crowd, so it is used to count/label objects and to sample depth.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

PATH_KEYS = ("source", "out_dir", "detect_weights", "seg_weights", "sem_weights", "depth_weights")


def to_numpy(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def sample_frame_indices(total_frames, native_fps, target_fps):
    step = max(1, round(native_fps / target_fps))
    return set(range(0, total_frames, step))


def media_files(source):
    """The videos and photos to process: the file itself, or every video and photo in a folder."""
    source = Path(source)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
        if not files:
            raise SystemExit(f"No videos or photos in {source}")
        return files
    if not source.exists():
        raise SystemExit(f"Not found: {source}")
    return [source]


def read_frames(path, target_fps):
    """Yield (frame_idx, timestamp, frame) for a photo (one frame) or the sampled frames of a video.
    Also returns nothing for an unreadable file, after a message."""
    if path.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Skipped {path}: not a readable photo")
            return
        yield 0, 0.0, frame
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Skipped {path}: not a readable video")
        return
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = sample_frame_indices(total_frames, native_fps, target_fps)
    step = max(1, round(native_fps / target_fps))
    print(f"Video: {path} ({native_fps:.1f} fps, {total_frames} frames); "
          f"sampling every {step} frames -> {len(sample_indices)} frames at ~{target_fps} fps")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in sample_indices:
                yield frame_idx, frame_idx / native_fps, frame
            frame_idx += 1
    finally:
        cap.release()


def video_fps(path, target_fps):
    """Frame rate for the output video, so it lasts as long as the input video."""
    cap = cv2.VideoCapture(str(path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return native_fps / max(1, round(native_fps / target_fps))


def label_image(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def build_panel(detect_img, seg_img, sem_img, depth_img, family, scale=0.5):
    imgs = [detect_img, seg_img, sem_img, depth_img]
    tag = family.upper()
    labels = [
        f"Detection ({tag})", f"Instance Segmentation ({tag})",
        "Semantic Segmentation (YOLO26)", "Monocular Depth (YOLO26)",
    ]
    h, w = detect_img.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    tiles = [label_image(cv2.resize(im, new_size), lab) for im, lab in zip(imgs, labels)]
    top = np.hstack(tiles[0:2])
    bottom = np.hstack(tiles[2:4])
    return np.vstack([top, bottom])


def off_tile(frame):
    """Stand-in panel view for a model that is switched off in the settings."""
    tile = np.zeros_like(frame)
    cv2.putText(tile, "off (null in settings)", (20, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (160, 160, 160), 2, cv2.LINE_AA)
    return tile


def annotate_instances(frame, seg_result, depth_map, writer, source_name, frame_idx, timestamp, family):
    annotated = frame.copy()
    n_instances = 0
    class_counts = Counter()

    if seg_result.boxes is not None and len(seg_result.boxes):
        boxes_xyxy = to_numpy(seg_result.boxes.xyxy)
        boxes_cls = to_numpy(seg_result.boxes.cls)
        boxes_conf = to_numpy(seg_result.boxes.conf)
        masks = to_numpy(seg_result.masks.data) if seg_result.masks is not None else None

        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            class_name = seg_result.names[int(boxes_cls[i])]
            conf = float(boxes_conf[i])

            depth_est = float("nan")
            mask_area = 0
            if masks is not None:
                mask_resized = cv2.resize(
                    masks[i], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                mask_bool = mask_resized > 0
                mask_area = int(mask_bool.sum())
                if mask_area > 0 and depth_map is not None:
                    depth_est = float(np.median(depth_map[mask_bool]))
                    overlay = annotated.copy()
                    overlay[mask_bool] = (0, 255, 0)
                    annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

            color = (0, 200, 0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"#{i} {class_name} {conf:.2f}" + (f" range~{depth_est:.1f}" if depth_map is not None else "")
            cv2.putText(
                annotated, label, (int(x1), max(0, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

            writer.writerow([
                source_name, frame_idx, f"{timestamp:.2f}", i, class_name, f"{conf:.3f}",
                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                mask_area, f"{depth_est:.3f}" if depth_map is not None else "",
            ])
            n_instances += 1
            class_counts[class_name] += 1

    cv2.putText(
        annotated, f"{family.upper()}  {source_name}  frame {frame_idx}  t={timestamp:.1f}s  objects={n_instances}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return annotated, n_instances, class_counts


def load_config(path, parser):
    """Read the settings file as parser defaults; the command line still wins over it."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "video" in cfg:  # the old name of source
        cfg.setdefault("source", cfg.pop("video"))
    known = {a.dest for a in parser._actions}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise SystemExit(f"Unknown setting(s) in {path}: {', '.join(unknown)}")
    for key in PATH_KEYS:
        if cfg.get(key) is not None:
            p = Path(cfg[key]).expanduser()
            cfg[key] = str(p if p.is_absolute() else path.parent / p)
    return cfg


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(Path(__file__).with_name("batch_config.yaml")),
                               help="Settings file; command-line options override it")
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     parents=[config_parser])
    parser.add_argument("--family", choices=["yolo26", "yolo11"], default="yolo26",
                        help="Model family for detection and instance segmentation")
    parser.add_argument("--source", "--video", dest="source", default="video/wilderbeast.mp4",
                        help="A video, a photo, or a folder of videos and photos")
    parser.add_argument("--out-dir", help="Default: results/<family>")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling rate to run inference at")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection/segmentation confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution; the single biggest lever for dense/distant herds")
    parser.add_argument("--max-det", type=int, default=300, help="Cap on detections per frame; raise it for large herds")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=False, help="Delete existing frames in --out-dir first, so output matches this run only")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False,
                        help="Also write annotated.mp4 and panels.mp4 at the --fps rate; use --fps 30 for smooth video")
    parser.add_argument("--detect-weights", default="auto",
                        help="Only for the Detection panel. auto = weights/<family>x.pt; settings file null = off")
    parser.add_argument("--seg-weights", help="Default: weights/<family>x-seg.pt")
    # YOLO11 has no sem/depth models, so these stay on YOLO26 for both families.
    parser.add_argument("--sem-weights", default="weights/yolo26x-sem.pt", help="Settings file null = off")
    parser.add_argument("--depth-weights", default="weights/yolo26x-depth.pt", help="Settings file null = off")
    parser.set_defaults(**load_config(Path(config_args.config).resolve(), parser))
    args = parser.parse_args()
    if args.detect_weights == "auto":
        args.detect_weights = f"weights/{args.family}x.pt"
    args.seg_weights = args.seg_weights or f"weights/{args.family}x-seg.pt"

    out_dir = Path(args.out_dir or f"results/{args.family}")
    annotated_dir = out_dir / "annotated"
    panel_dir = out_dir / "panels"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Frames are only overwritten when frame numbers collide, so changing --fps
    # between runs otherwise leaves orphans that the CSV (always rewritten) omits.
    stale_annotated = sorted(annotated_dir.glob("*.jpg"))
    stale_panels = sorted(panel_dir.glob("*.jpg"))
    stale = stale_annotated + stale_panels
    if stale and args.clean:
        for f in stale:
            f.unlink()
        print(
            f"--clean: removed {len(stale_annotated)} from {annotated_dir}/ "
            f"and {len(stale_panels)} from {panel_dir}/"
        )
    elif stale:
        print(
            f"WARNING: {len(stale)} image(s) already in {out_dir}/ from a previous run.\n"
            f"         Frames this run does not regenerate will be left behind and will\n"
            f"         NOT match detections.csv. Pass --clean to remove them first."
        )

    files = media_files(args.source)
    print(f"Source: {args.source} ({len(files)} file(s))")

    detect_model = YOLO(args.detect_weights) if args.detect_weights else None
    seg_model = YOLO(args.seg_weights)
    # null in the settings switches the semantic or depth model off
    sem_model = YOLO(args.sem_weights) if args.sem_weights else None
    depth_model = YOLO(args.depth_weights) if args.depth_weights else None
    det_kwargs = dict(conf=args.conf, imgsz=args.imgsz, max_det=args.max_det, verbose=False)

    csv_path = out_dir / "detections.csv"
    videos_written = []
    saved = 0
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "source", "frame_idx", "timestamp_s", "instance_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "mask_area_px", "depth_range_est",
        ])

        for path in files:
            is_photo = path.suffix.lower() in IMAGE_EXTS
            # One video per input video; it plays at the sampling rate, so it lasts as long as the input
            video_writers = {}

            def write_video(kind, img):
                if kind not in video_writers:
                    out = out_dir / f"{path.stem}_{kind}.mp4"
                    video_writers[kind] = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                                          video_fps(path, args.fps), (img.shape[1], img.shape[0]))
                    videos_written.append(out)
                video_writers[kind].write(img)

            for frame_idx, timestamp, frame in read_frames(path, args.fps):
                name = path.stem if is_photo else f"{path.stem}_{frame_idx:05d}"
                detect_result = detect_model.predict(frame, **det_kwargs)[0] if detect_model else None
                seg_result = seg_model.predict(frame, **det_kwargs)[0]
                sem_result = sem_model.predict(frame, verbose=False)[0] if sem_model else None
                depth_result = depth_model.predict(frame, verbose=False)[0] if depth_model else None
                depth_map = to_numpy(depth_result.depth.data) if depth_result else None

                annotated, n_instances, class_counts = annotate_instances(
                    frame, seg_result, depth_map, writer, path.name, frame_idx, timestamp, args.family
                )
                cv2.imwrite(str(annotated_dir / f"{name}.jpg"), annotated)

                panel = build_panel(detect_result.plot() if detect_result else off_tile(frame), seg_result.plot(),
                                    sem_result.plot() if sem_result else off_tile(frame),
                                    depth_result.plot() if depth_result else off_tile(frame), args.family)
                cv2.imwrite(str(panel_dir / f"{name}_panel.jpg"), panel)
                if args.save_video and not is_photo:
                    write_video("annotated", annotated)
                    write_video("panels", panel)

                saved += 1
                breakdown = ", ".join(f"{n} {count}" for n, count in class_counts.most_common())
                print(f"[{saved}] {name} t={timestamp:.1f}s -> {n_instances} instances"
                      + (f" ({breakdown})" if breakdown else ""))

            for video_writer in video_writers.values():
                video_writer.release()

    print(f"\nDone. {saved} frames processed from {len(files)} file(s).")
    print(f"Annotated frames (bbox + range):  {annotated_dir}/ ({len(list(annotated_dir.glob('*.jpg')))} files)")
    print(f"Detect/seg/sem/depth panels:      {panel_dir}/ ({len(list(panel_dir.glob('*.jpg')))} files)")
    print(f"Per-instance CSV log:             {csv_path}")
    for out in videos_written:
        print(f"Video:                            {out}")


if __name__ == "__main__":
    main()
