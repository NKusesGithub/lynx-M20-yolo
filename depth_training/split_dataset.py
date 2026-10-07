#!/usr/bin/env python3
"""Split the recorded pairs into train and val automatically.

Pairs half a second apart are almost the same image. If one goes to train and
the next to val, the val score is too good. So the split moves whole blocks of
consecutive pairs (default 20 pairs, about 10 s of recording), never single
pairs.

A block always goes to the same side, also after you record more and run the
script again, because the choice comes only from the block name. With few
recordings the val fraction is not exact; the script prints the real value.

    python3 depth_training/split_dataset.py --dry-run   # show the plan only
    python3 depth_training/split_dataset.py             # move the files
    python3 depth_training/split_dataset.py --val 0.2   # 20% val
"""
import argparse
import zlib
from collections import defaultdict
from pathlib import Path

DEFAULT_DATASET = Path(__file__).resolve().parent / 'dataset'
SPLITS = ('train', 'val')


def block_id(stem, block):
    """'20261006_070643_00045' with block 20 -> '20261006_070643/2'."""
    session, index = stem.rsplit('_', 1)
    return f'{session}/{int(index) // block}'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--val', type=float, default=0.15, help='fraction of pairs for val')
    parser.add_argument('--block', type=int, default=20, help='consecutive pairs that stay together')
    parser.add_argument('--dry-run', action='store_true', help='show the plan, move nothing')
    args = parser.parse_args()

    # Collect every pair from both splits, so that a run again gives the same result.
    blocks = defaultdict(list)  # block id -> [(split now, stem)]
    missing = 0
    for split in SPLITS:
        for img in sorted((args.dataset / 'images' / split).glob('*.jpg')):
            if not (args.dataset / 'depth' / split / f'{img.stem}.png').exists():
                missing += 1
                continue
            blocks[block_id(img.stem, args.block)].append((split, img.stem))
    total = sum(len(pairs) for pairs in blocks.values())
    if not total:
        raise SystemExit(f'No pairs found in {args.dataset}.')

    # Each block is decided from its own name only, so new recordings never
    # move old blocks. With few blocks the val fraction is only approximate.
    val_blocks = {b for b in blocks if zlib.crc32(b.encode()) % 1000 < args.val * 1000}
    n_val = sum(len(blocks[b]) for b in val_blocks)

    moves = []
    for b, pairs in blocks.items():
        dest = 'val' if b in val_blocks else 'train'
        moves += [(src, dest, stem) for src, stem in pairs if src != dest]

    sessions = {b.split('/')[0] for b in blocks}
    print(f'{total} pairs from {len(sessions)} recordings, in {len(blocks)} blocks of up to {args.block}.')
    print(f'Plan: train {total - n_val}, val {n_val} ({100 * n_val / total:.0f}%).')
    print(f'Pairs to move: {sum(d == "val" for _, d, _ in moves)} to val, '
          f'{sum(d == "train" for _, d, _ in moves)} to train.')
    if missing:
        print(f'Skipped {missing} images that have no depth map.')
    if args.dry_run or not moves:
        print('Nothing moved.' if args.dry_run else 'Already split.')
        return

    for src, dest, stem in moves:
        for kind, ext in (('images', 'jpg'), ('depth', 'png')):
            to = args.dataset / kind / dest
            to.mkdir(parents=True, exist_ok=True)
            (args.dataset / kind / src / f'{stem}.{ext}').rename(to / f'{stem}.{ext}')
    print('Done.')


if __name__ == '__main__':
    main()
