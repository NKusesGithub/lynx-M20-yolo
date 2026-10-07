"""Make a chart of the detections in each frame from the CSV file of a batch run.

Reads results/<run>/detections.csv (written by run_batch.py) and saves
results/<run>/detections_per_frame.png: a stacked bar for each processed frame.
The height is the number of objects in the frame; the colours are the classes.

Run from the droneYolo2026 folder:
    python3 plot_detections.py --run yolo26
"""
import argparse
import csv
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write a file, no window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# Fixed categorical order (validated for colour-blind separation); classes past
# the last slot are summed into "Other" instead of getting a generated colour.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
OTHER_COLOUR = "#8a8a86"
INK, INK_MUTED, GRID, SURFACE = "#1f1f1e", "#6b6b67", "#e6e6e3", "#fcfcfb"


def load(csv_path):
    """Return {source: ({frame_idx: timestamp_s}, {frame_idx: Counter(class_name)})}, in file order.
    CSV files from before the source column count as one source."""
    sources = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            times, counts = sources.setdefault(row.get("source", ""), ({}, defaultdict(Counter)))
            frame = int(row["frame_idx"])
            times[frame] = float(row["timestamp_s"])
            counts[frame][row["class_name"]] += 1
    return sources


def all_frames(times):
    """The CSV has no rows for frames with no detections. Fill them back in from
    the frame step (the run processes every Nth frame) so they plot as zero."""
    frames = sorted(times)
    if len(frames) < 2:
        return frames, times
    step = reduce(gcd, (b - a for a, b in zip(frames, frames[1:])))
    seconds_per_frame = (times[frames[-1]] - times[frames[0]]) / (frames[-1] - frames[0])
    filled = list(range(frames[0], frames[-1] + 1, step))
    return filled, {f: times.get(f, times[frames[0]] + (f - frames[0]) * seconds_per_frame) for f in filled}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="yolo26", help="the results folder of the batch run, in results/")
    parser.add_argument("--csv", type=Path, help="a CSV file in place of results/<run>/detections.csv")
    parser.add_argument("--out", type=Path, help="the chart file (default: next to the CSV file)")
    args = parser.parse_args()

    csv_path = args.csv or ROOT / "results" / args.run / "detections.csv"
    if not csv_path.exists():
        raise SystemExit(f"No {csv_path}. Run run_batch.py first.")
    out = args.out or csv_path.with_name("detections_per_frame.png")

    sources = load(csv_path)
    if not sources:
        raise SystemExit(f"{csv_path} has no detections.")
    # One bar per processed frame. With several videos or photos, their frames go one after the other.
    frames, x, counts, starts = [], [], defaultdict(Counter), []
    for source, (src_times, src_counts) in sources.items():
        src_frames, src_times = all_frames(src_times)
        starts.append((len(frames), source))
        for f in src_frames:
            key = (source, f)
            frames.append(key)
            counts[key] = src_counts.get(f, Counter())
            x.append(src_times[f])
    if len(sources) > 1:
        x = list(range(len(frames)))

    totals = Counter()
    for c in counts.values():
        totals.update(c)
    ranked = [name for name, _ in totals.most_common()]
    shown = ranked if len(ranked) <= len(SERIES_COLOURS) else ranked[:len(SERIES_COLOURS) - 1]
    other = ranked[len(shown):]

    series = [(name, colour, [counts[f][name] for f in frames]) for name, colour in zip(shown, SERIES_COLOURS)]
    if other:
        series.append((f"Other ({len(other)} classes)", OTHER_COLOUR,
                       [sum(counts[f][n] for n in other) for f in frames]))

    width = 0.8 * (x[1] - x[0]) if len(x) > 1 else 0.8
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bottom = [0] * len(frames)
    for name, colour, y in series:
        # Surface-coloured edges keep a small gap between stacked segments
        ax.bar(x, y, width=width, bottom=bottom, color=colour, edgecolor=SURFACE, linewidth=0.6, label=name)
        bottom = [b + v for b, v in zip(bottom, y)]

    # With several sources, leave room above the plot for their names
    ax.set_title(f"Detections per frame: {csv_path.parent.name}", loc="left", color=INK, fontsize=13,
                 pad=60 if len(sources) > 1 else 12)
    if len(sources) > 1:
        ax.set_xlabel("Frames, in the order of the run", color=INK_MUTED)
        ax.set_xticks([])
        for start, source in starts:
            ax.axvline(start - 0.5, color=INK_MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
            ax.annotate(source, (start - 0.4, 1), xycoords=("data", "axes fraction"), va="bottom", fontsize=8,
                        color=INK_MUTED, rotation=30)
    else:
        ax.set_xlabel("Time in video (s)", color=INK_MUTED)
    ax.set_ylabel("Objects in frame", color=INK_MUTED)
    ax.set_ylim(bottom=0)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=min(len(series), 4), frameon=False,
              labelcolor=INK)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"{len(frames)} frames from {len(sources)} source(s), {sum(totals.values())} detections, {len(ranked)} classes -> {out}")


if __name__ == "__main__":
    main()
