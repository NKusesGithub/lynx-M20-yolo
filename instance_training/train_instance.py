"""Fine-tune the YOLO26 segmentation model on the auto-labelled tool dataset.

Run from the droneYolo2026 folder:
    python3 instance_training/train_instance.py
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # the droneYolo2026 folder


def resolve_data(data):
    """Return a copy of the dataset YAML with `path` made absolute.

    Ultralytics reads a relative `path` as relative to its own datasets folder,
    not to the YAML file, so it would look in the wrong place.
    """
    data = Path(data).resolve()
    cfg = yaml.safe_load(data.read_text())
    root = Path(cfg["path"])
    if root.is_absolute():
        return str(data)
    cfg["path"] = str((data.parent / root).resolve())
    if not Path(cfg["path"], cfg["train"]).exists():
        raise SystemExit(f"No training images at {Path(cfg['path'], cfg['train'])}. Run autolabeller.py first.")
    resolved = data.with_name(data.stem + ".resolved.yaml")
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(resolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="instance_training/dataset/data.yaml")
    parser.add_argument("--model", default="weights/yolo26n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="tools_seg")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=resolve_data(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        # Absolute: Ultralytics puts a relative project under its own runs/ folder.
        project=str(ROOT / "results/instance_training"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
