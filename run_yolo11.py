#!/usr/bin/env python3
"""
Run YOLO11 detection and instance segmentation on a video sampled at a fixed
fps, and save per-frame results for review. Standalone twin of run_yolo26.py,
with the same flags and the same output layout.

Read this before trusting the labels/numbers:

- YOLO11 has no semantic-segmentation or monocular-depth models. Those tasks
  arrived with YOLO26, and `yolo11x-sem.pt` / `yolo11x-depth.pt` do not exist.
  So the semantic and depth panels, and the depth used for the range estimate,
  come from the YOLO26 checkpoints. Only detection and instance segmentation
  are YOLO11 here - which is what makes this a fair A/B against run_yolo26.py:
  depth is held constant and only the detector changes.
- Detection and instance-segmentation weights are COCO-trained (80 classes), so
  an animal gets labeled as the nearest COCO class (elephant, cow, zebra, ...).
  Treat the label as "an object was detected here", not a species ID.
- Nothing is filtered by class: people, cars and bags are counted alongside
  animals. The console prints a per-class breakdown of what was found.
- Depth values are monocular relative-depth estimates, not calibrated meters.
  Use them to rank which objects are nearer/farther, not as absolute distances.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def to_numpy(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def sample_frame_indices(total_frames, native_fps, target_fps):
    step = max(1, round(native_fps / target_fps))
    return set(range(0, total_frames, step))


def label_image(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def build_panel(detect_img, seg_img, sem_img, depth_img, scale=0.5):
    imgs = [detect_img, seg_img, sem_img, depth_img]
    labels = [
        "Detection (YOLO11)", "Instance Segmentation (YOLO11)",
        "Semantic Segmentation (YOLO26)", "Monocular Depth (YOLO26)",
    ]
    h, w = detect_img.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    tiles = [label_image(cv2.resize(im, new_size), lab) for im, lab in zip(imgs, labels)]
    top = np.hstack(tiles[0:2])
    bottom = np.hstack(tiles[2:4])
    return np.vstack([top, bottom])


def annotate_instances(frame, seg_result, depth_map, writer, frame_idx, timestamp):
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
                if mask_area > 0:
                    depth_est = float(np.median(depth_map[mask_bool]))
                    overlay = annotated.copy()
                    overlay[mask_bool] = (0, 255, 0)
                    annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

            color = (0, 200, 0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"#{i} {class_name} {conf:.2f} range~{depth_est:.1f}"
            cv2.putText(
                annotated, label, (int(x1), max(0, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

            writer.writerow([
                frame_idx, f"{timestamp:.2f}", i, class_name, f"{conf:.3f}",
                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                mask_area, f"{depth_est:.3f}",
            ])
            n_instances += 1
            class_counts[class_name] += 1

    cv2.putText(
        annotated, f"YOLO11  frame {frame_idx}  t={timestamp:.1f}s  objects={n_instances}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return annotated, n_instances, class_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="video/wilderbeast.mp4")
    parser.add_argument("--out-dir", default="results/yolo11")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling rate to run inference at")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection/segmentation confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution; the single biggest lever for dense/distant objects")
    parser.add_argument("--max-det", type=int, default=300, help="Cap on detections per frame; raise it for crowded scenes")
    parser.add_argument("--clean", action="store_true", help="Delete existing frames in --out-dir first, so output matches this run only")
    parser.add_argument("--detect-weights", default="weights/yolo11x.pt")
    parser.add_argument("--seg-weights", default="weights/yolo11x-seg.pt")
    # YOLO11 has no sem/depth models - these stay on YOLO26.
    parser.add_argument("--sem-weights", default="weights/yolo26x-sem.pt")
    parser.add_argument("--depth-weights", default="weights/yolo26x-depth.pt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
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

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = sample_frame_indices(total_frames, native_fps, args.fps)

    print(f"Video: {args.video} ({native_fps:.1f} fps, {total_frames} frames)")
    print(f"Sampling every {round(native_fps / args.fps)} frames -> {len(sample_indices)} frames at ~{args.fps} fps")

    detect_model = YOLO(args.detect_weights)
    seg_model = YOLO(args.seg_weights)
    sem_model = YOLO(args.sem_weights)
    depth_model = YOLO(args.depth_weights)

    csv_path = out_dir / "detections.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame_idx", "timestamp_s", "instance_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2", "mask_area_px", "depth_range_est",
        ])

        frame_idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in sample_indices:
                timestamp = frame_idx / native_fps

                det_kwargs = dict(
                    conf=args.conf, imgsz=args.imgsz, max_det=args.max_det, verbose=False
                )
                detect_result = detect_model.predict(frame, **det_kwargs)[0]
                seg_result = seg_model.predict(frame, **det_kwargs)[0]
                sem_result = sem_model.predict(frame, verbose=False)[0]
                depth_result = depth_model.predict(frame, verbose=False)[0]
                depth_map = to_numpy(depth_result.depth.data)

                annotated, n_instances, class_counts = annotate_instances(
                    frame, seg_result, depth_map, writer, frame_idx, timestamp
                )
                cv2.imwrite(str(annotated_dir / f"frame_{frame_idx:05d}.jpg"), annotated)

                panel = build_panel(detect_result.plot(), seg_result.plot(), sem_result.plot(), depth_result.plot())
                cv2.imwrite(str(panel_dir / f"frame_{frame_idx:05d}_panel.jpg"), panel)

                saved += 1
                breakdown = ", ".join(f"{name} {count}" for name, count in class_counts.most_common())
                print(
                    f"[{saved}/{len(sample_indices)}] frame {frame_idx} t={timestamp:.1f}s"
                    f" -> {n_instances} instances"
                    + (f" ({breakdown})" if breakdown else "")
                )

            frame_idx += 1

    cap.release()
    print(f"\nDone. {saved} sampled frames processed.")
    print(f"Annotated frames (bbox + range):  {annotated_dir}/ ({len(list(annotated_dir.glob('*.jpg')))} files)")
    print(f"Detect/seg/sem/depth panels:      {panel_dir}/ ({len(list(panel_dir.glob('*.jpg')))} files)")
    print(f"Per-instance CSV log:             {csv_path}")


if __name__ == "__main__":
    main()
