#!/usr/bin/env python3
"""Score how sharp the recorded colour images are, and optionally delete blurry pairs.

Sharpness is the variance of the Laplacian of the grey image: high = sharp
edges, low = blurry. A plain wall also scores low, so look at the lowest images
before you pick a threshold.

    python3 depth_training/check_sharpness.py                 # list the 15 blurriest
    python3 depth_training/check_sharpness.py --below 60      # list all under 60
    python3 depth_training/check_sharpness.py --below 60 --delete
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DATASET = Path(__file__).resolve().parent / 'dataset'


def sharpness(bgr):
    """Variance of the Laplacian. record_rgbd.py in the m20-orbbec repo has a copy; keep both the same."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--split', default='all', help='train, val or all')
    parser.add_argument('--below', type=float, help='list every image with sharpness under this')
    parser.add_argument('--delete', action='store_true',
                        help='delete the listed images and their depth maps (needs --below)')
    parser.add_argument('--show', type=int, default=15, help='how many to list without --below')
    args = parser.parse_args()

    if args.delete and args.below is None:
        raise SystemExit('--delete needs --below, so that you choose the threshold.')

    splits = ['train', 'val'] if args.split == 'all' else [args.split]
    scores = []
    for split in splits:
        for img_path in sorted((args.dataset / 'images' / split).glob('*.jpg')):
            img = cv2.imread(str(img_path))
            if img is not None:
                scores.append((sharpness(img), split, img_path))
    if not scores:
        raise SystemExit(f'No images found in {args.dataset}/images/{{{",".join(splits)}}}')

    scores.sort()
    values = np.array([s for s, _, _ in scores])
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    print(f'{len(scores)} images. Sharpness: lowest {values[0]:.0f}, '
          f'10% {p10:.0f}, median {p50:.0f}, 90% {p90:.0f}, highest {values[-1]:.0f}')

    listed = [s for s in scores if s[0] < args.below] if args.below is not None else scores[:args.show]
    print(f'\n{"sharpness":>9}  image')
    for score, split, path in listed:
        print(f'{score:9.0f}  {path.relative_to(args.dataset)}')

    if args.below is not None:
        print(f'\n{len(listed)} of {len(scores)} images are under {args.below:g}.')
    if args.delete:
        for _, split, img_path in listed:
            img_path.unlink()
            (args.dataset / 'depth' / split / (img_path.stem + '.png')).unlink(missing_ok=True)
        print(f'Deleted {len(listed)} pairs.')


if __name__ == '__main__':
    main()
