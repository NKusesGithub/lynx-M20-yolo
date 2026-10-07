"""Make tool labels from a video automatically.

Saves every Nth frame of the video to the dataset, finds the tools with YOLOE
text prompts and makes their masks with SAM. Deletes the old labels first.

Run from the droneYolo2026 folder:
    python3 instance_training/autolabeller.py --video samples/IMG_6678.mp4
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import SAM, YOLOE

ROOT = Path(__file__).resolve().parent.parent  # the droneYolo2026 folder

# class name -> text prompt; descriptive prompts detect noticeably better than bare names
CLASSES = {
    "screwdriver": "screwdriver",
    "drill": "red cordless drill",
    "pliers": "red-handled pliers",
    "case": "grey metal case",
    "scissors": "orange-handled scissors",
}
SINGLE_INSTANCE = {"screwdriver", "drill", "pliers", "scissors"}  # only one of each per frame in the first video


def extract_frames(video, images_dir, stride):
    """Save every `stride`th frame; annotating a video directly would write every frame's labels to one file."""
    cap = cv2.VideoCapture(str(video))
    frame_idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            cv2.imwrite(str(images_dir / f"{video.stem}_{frame_idx:05d}.jpg"), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    print(f"Extracted {saved} of {frame_idx} frames to {images_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=ROOT / "samples/IMG_6678.mp4")
    parser.add_argument("--dataset", type=Path, default=ROOT / "instance_training/dataset")
    parser.add_argument("--stride", type=int, default=10, help="keep every Nth frame; neighbours are near-duplicates")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLOE scores are low even when right")
    parser.add_argument("--device", default="0")
    parser.add_argument("--frames-only", action="store_true",
                        help="only save the frames and keep the labels; the frames are not in Git")
    args = parser.parse_args()

    images_dir = args.dataset / "images"
    labels_dir = args.dataset / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    saved = extract_frames(args.video, images_dir, args.stride)
    if args.frames_only:
        return

    # YOLOE is open-vocabulary, so it finds tools COCO-trained models don't know; SAM refines the masks
    det_model = YOLOE(str(ROOT / "weights/yoloe-26l-seg.pt"))
    det_model.set_classes(list(CLASSES.values()))
    sam_model = SAM(str(ROOT / "weights/sam_b.pt"))

    for old in labels_dir.glob("*.txt"):
        old.unlink()

    labelled = 0
    single_ids = {i for i, name in enumerate(CLASSES) if name in SINGLE_INSTANCE}
    results = det_model(str(images_dir), stream=True, device=args.device, conf=args.conf, agnostic_nms=True,
                        verbose=False)
    for result in results:
        # boxes are sorted by confidence, so the first box of a single-instance class is its best one
        keep, seen = [], set()
        for i, class_id in enumerate(result.boxes.cls.int().tolist()):
            if class_id in single_ids:
                if class_id in seen:
                    continue
                seen.add(class_id)
            keep.append(i)
        if not keep:
            continue
        boxes = result.boxes[keep]
        class_ids = boxes.cls.int().tolist()
        sam_results = sam_model(result.orig_img, bboxes=boxes.xyxy, verbose=False, save=False, device=args.device)
        with open(labels_dir / f"{Path(result.path).stem}.txt", "w", encoding="utf-8") as f:
            for class_id, segment in zip(class_ids, sam_results[0].masks.xyn):
                if len(segment) >= 3:  # fewer than 3 points is not a polygon
                    f.write(f"{class_id} " + " ".join(map(str, segment.reshape(-1).tolist())) + "\n")
        labelled += 1
    print(f"Labelled {labelled} of {saved} frames in {labels_dir}")

    # Only create data.yaml once: label_unknown.py adds the "object" class and the train/val split lives there too
    data_yaml = args.dataset / "data.yaml"
    if not data_yaml.exists():
        names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
        data_yaml.write_text(f"path: .\ntrain: images\nval: images\nnames:\n{names}\n")


if __name__ == "__main__":
    main()
