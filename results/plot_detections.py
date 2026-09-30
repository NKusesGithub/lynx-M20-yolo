#!/usr/bin/env python3
"""
Plot detections.csv as a stacked vertical bar chart.

One bar per sampled frame; each bar is split into segments, one per detection
class, so bar height is the total number of detections in that frame and each
segment is the count for that class (all `car` rows in a frame add into one
`car` segment).

Pick which run to plot with --run (a folder name under results/). The chart is
written next to that run's CSV, so plotting one run never overwrites another's.

Usage:
    python3 plot_detections.py                 # --run yolo26 (default)
    python3 plot_detections.py --run yolo11    # plots results/yolo11/
    python3 plot_detections.py --csv some/other.csv   # explicit path override
    python3 plot_detections.py                 # x axis = elapsed video time (m:ss)
    python3 plot_detections.py --x frame       # x axis = raw frame index
    python3 plot_detections.py --start-time "2026-09-20 08:30:00"   # x axis = clock time
    python3 plot_detections.py --csv other.csv --out chart.png
"""

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Categorical hues in fixed slot order - assigned to classes by total count,
# never cycled. A 9th class folds into "Other" rather than reusing a hue.
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
OTHER_COLOR = "#8a8a85"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

MAX_SERIES = len(SERIES_COLORS)


def load_counts(csv_path):
    """-> ({frame_idx: {class: count}}, Counter of class totals, {frame_idx: timestamp_s})"""
    per_frame = defaultdict(Counter)
    totals = Counter()
    timestamps = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame_idx"])
            name = row["class_name"]
            per_frame[frame][name] += 1
            totals[name] += 1
            timestamps[frame] = float(row["timestamp_s"])
    return per_frame, totals, timestamps


def format_elapsed(seconds):
    """Seconds into the video -> m:ss, or h:mm:ss once the clip passes an hour."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def build_xtick_labels(frames, timestamps, mode, start_time):
    if mode == "frame":
        return [str(f) for f in frames]
    if start_time is not None:
        return [(start_time + timedelta(seconds=timestamps[f])).strftime("%H:%M:%S") for f in frames]
    return [format_elapsed(timestamps[f]) for f in frames]


def resolve_series(totals):
    """Biggest classes keep their own hue; the tail collapses into 'Other'."""
    ranked = [name for name, _ in totals.most_common()]
    if len(ranked) <= MAX_SERIES:
        return ranked, dict(zip(ranked, SERIES_COLORS)), set()
    kept = ranked[:MAX_SERIES - 1]
    folded = set(ranked[MAX_SERIES - 1:])
    colors = dict(zip(kept, SERIES_COLORS))
    colors["Other"] = OTHER_COLOR
    return kept + ["Other"], colors, folded


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run", default="yolo26",
        help="Which run folder under results/ to plot, e.g. yolo26 or yolo11 (default: yolo26)",
    )
    parser.add_argument("--csv", help="Explicit CSV path; overrides --run")
    parser.add_argument("--out", help="Explicit output PNG path; defaults to <run folder>/detections_per_frame.png")
    parser.add_argument("--max-xticks", type=int, default=16, help="Thin x labels to at most this many")
    parser.add_argument(
        "--x", choices=["time", "frame"], default="time",
        help="Label the x axis with elapsed video time (default) or raw frame index",
    )
    parser.add_argument(
        "--start-time",
        help="Wall-clock time the recording started, e.g. '2026-09-20 08:30:00'. "
             "With this, x labels become real clock times instead of elapsed time.",
    )
    args = parser.parse_args()

    start_time = None
    if args.start_time:
        if args.x == "frame":
            raise SystemExit("--start-time has no effect with --x frame; drop one of them.")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%H:%M:%S"):
            try:
                start_time = datetime.strptime(args.start_time, fmt)
                break
            except ValueError:
                continue
        if start_time is None:
            raise SystemExit(f"Could not parse --start-time {args.start_time!r} (try '2026-09-20 08:30:00')")

    run_dir = here / args.run
    csv_path = Path(args.csv) if args.csv else run_dir / "detections.csv"
    # Chart lands beside the data it came from, so plotting one run never
    # overwrites another run's chart.
    out_path = Path(args.out) if args.out else csv_path.parent / "detections_per_frame.png"
    if not csv_path.exists():
        available = sorted(p.parent.name for p in here.glob("*/detections.csv"))
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            + (f"Available runs: {', '.join(available)}" if available else "No run folders found.")
        )

    per_frame, totals, timestamps = load_counts(csv_path)
    if not per_frame:
        raise SystemExit(f"No detection rows in {csv_path} - nothing to plot.")

    frames = sorted(per_frame)
    series, colors, folded = resolve_series(totals)

    # Grow with the frame count, but cap it - a few hundred frames would otherwise
    # produce a canvas too wide to read as one chart.
    fig_width = min(22, max(8, len(frames) * 0.34))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = range(len(frames))
    bottoms = [0] * len(frames)
    for name in series:
        if name == "Other":
            heights = [sum(per_frame[f][c] for c in folded) for f in frames]
        else:
            heights = [per_frame[f][name] for f in frames]
        ax.bar(
            x, heights, bottom=bottoms, width=0.78,
            color=colors[name], label=name,
            edgecolor=SURFACE, linewidth=1.2,  # 2px surface gap between segments
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    if args.x == "frame":
        xlabel = "frame index"
    elif start_time is not None:
        xlabel = "time of day (hh:mm:ss)"
    else:
        xlabel = "video time elapsed (m:ss)"
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=11, labelpad=10)
    ax.set_ylabel("number of detections", color=TEXT_SECONDARY, fontsize=11, labelpad=10)
    ax.set_title(
        f"{csv_path.parent.name} - detections per frame by class  "
        f"({sum(totals.values())} detections across {len(frames)} frames)",
        color=TEXT_PRIMARY, fontsize=13, pad=16, loc="left",
    )

    labels = build_xtick_labels(frames, timestamps, args.x, start_time)
    step = max(1, len(frames) // args.max_xticks)
    shown = list(x)[::step]
    ax.set_xticks(shown)
    ax.set_xticklabels([labels[i] for i in shown], rotation=45, ha="right")

    ax.yaxis.grid(True, color=GRID, linewidth=0.8, solid_capstyle="butt")
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.margins(x=0.01)

    if len(series) >= 2:
        handles = [Patch(facecolor=colors[n], label=n) for n in series]
        ax.legend(
            handles=handles, frameon=False, ncol=min(len(series), 8),
            loc="upper left", bbox_to_anchor=(0, -0.18),
            labelcolor=TEXT_SECONDARY, fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Wrote {out_path}")
    print(f"{len(frames)} frames, {sum(totals.values())} detections")
    for name, count in totals.most_common():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
