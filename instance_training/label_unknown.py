"""Add a catch-all "object" class for items that aren't one of the known classes.

Runs a prompt-free (class-agnostic) detector over the dataset images, refines each box with SAM, and appends any
mask that doesn't overlap an existing label to that frame's label file. Existing labels are left untouched, so this
is safe to run on hand-corrected labels; re-running replaces the previous "object" labels.

Run from the droneYolo2026 folder:
    python3 instance_training/label_unknown.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLOE

ROOT = Path(__file__).resolve().parent.parent  # the droneYolo2026 folder

OBJECT_CLASS = 5
MIN_AREA, MAX_AREA = 0.01, 0.5  # box area as a fraction of the frame; drops screws/edges and whole-scene boxes
MAX_KNOWN_OVERLAP = 0.4  # drop masks whose pixels mostly lie on an already-labelled object
MAX_DUPLICATE_IOU = 0.5  # drop masks that repeat an "object" already added in this frame
EDGE_MARGIN = 3  # px; masks touching the frame edge are background (cardboard, floor) or cut-off items


def largest_contour(mask):
    """Outer outline of the biggest blob; joining every blob into one polygon gives zig-zag labels."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2) if contours else None


def polygon_mask(points, h, w):
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [(points * [w, h]).astype(np.int32)], 1)
    return mask.astype(bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=ROOT / "instance_training/dataset")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    images_dir = args.dataset / "images"
    labels_dir = args.dataset / "labels"

    det_model = YOLOE(str(ROOT / "weights/yoloe-26l-seg-pf.pt"))
    sam_model = SAM(str(ROOT / "weights/sam_b.pt"))

    added_total = 0
    for result in det_model(str(images_dir), stream=True, device=args.device, conf=args.conf, verbose=False):
        h, w = result.orig_shape
        label_file = labels_dir / f"{Path(result.path).stem}.txt"
        lines = label_file.read_text().splitlines() if label_file.exists() else []
        known = [l for l in lines if l and int(l.split()[0]) != OBJECT_CLASS]

        known_mask = np.zeros((h, w), bool)
        for line in known:
            known_mask |= polygon_mask(np.array(line.split()[1:], float).reshape(-1, 2), h, w)

        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (w * h)
        boxes = boxes[(areas > MIN_AREA) & (areas < MAX_AREA)]

        added, added_masks = [], []
        if len(boxes):
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=args.device)
            for raw in sam_results[0].masks.data.cpu().numpy():
                raw = cv2.resize(raw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                contour = largest_contour(raw)
                if contour is None or len(contour) < 3:
                    continue
                x0, y0 = contour.min(0)
                x1, y1 = contour.max(0)
                if x0 < EDGE_MARGIN or y0 < EDGE_MARGIN or x1 >= w - EDGE_MARGIN or y1 >= h - EDGE_MARGIN:
                    continue
                segment = contour / [w, h]
                mask = polygon_mask(segment, h, w)
                if not mask.any() or (mask & known_mask).sum() / mask.sum() > MAX_KNOWN_OVERLAP:
                    continue
                if any((mask & m).sum() / (mask | m).sum() > MAX_DUPLICATE_IOU for m in added_masks):
                    continue
                added_masks.append(mask)
                added.append(f"{OBJECT_CLASS} " + " ".join(map(str, segment.reshape(-1).tolist())))

        label_file.write_text("".join(l + "\n" for l in known + added))
        added_total += len(added)
    print(f"Added {added_total} 'object' labels")

    data_yaml = args.dataset / "data.yaml"
    if f"  {OBJECT_CLASS}: object" not in data_yaml.read_text():
        with open(data_yaml, "a") as f:
            f.write(f"  {OBJECT_CLASS}: object\n")


if __name__ == "__main__":
    main()
