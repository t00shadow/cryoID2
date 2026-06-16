"""
Scatter plotting helper for benchmark Q-score CSVs.

Reads a Q-score CSV produced by ``benchmark.py`` or ``benchmark_gt_only.py``
and writes a Q-score-identified vs Q-score-GT scatter plot, color-coded by
map-resolution bin.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RESOLUTION_COLORS = {
    "<= 4A": "#f28e2b",
    "4-6A": "#9e9e9e",
    "> 6A": "#1f4e8c",
    "unknown": "#555555",
}


def resolution_bin(resolution: float) -> str:
    """Bucket a numeric resolution into the three legend bins used by the plot."""
    if resolution <= 4.0:
        return "<= 4A"
    if resolution <= 6.0:
        return "4-6A"
    return "> 6A"


def load_points(csv_path: Path) -> tuple[list[float], list[float], list[str]]:
    """Parse a Q-score CSV into ``(xs, ys, resolution_bins)`` ready for plotting.

    Silently skips rows missing the required Q-score values, and labels rows
    with missing/invalid resolution as ``"unknown"``. Raises if no row is
    plottable.
    """
    xs: list[float] = []
    ys: list[float] = []
    bins: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                x_raw = row.get("qscore_identified")
                if x_raw in (None, ""):
                    x_raw = row.get("qscore_new_method")
                x = float(x_raw)
                y = float(row["qscore_ground_truth"])
            except (KeyError, TypeError, ValueError):
                continue
            xs.append(x)
            ys.append(y)
            try:
                bins.append(resolution_bin(float(row["resolution"])))
            except (KeyError, TypeError, ValueError):
                bins.append("unknown")
    if not xs:
        raise RuntimeError(f"No plottable Q-score rows found in {csv_path}")
    return xs, ys, bins


def plot_qscores(
    csv_path: Path,
    png_path: Path,
    *,
    title: str = "Q-score: Ground Truth vs Identified",
) -> None:
    """Render the GT-vs-identified Q-score scatter plot from ``csv_path`` to ``png_path``."""
    xs, ys, bins = load_points(csv_path)
    colors = [RESOLUTION_COLORS[name] for name in bins]
    min_val, max_val = 0.0, 0.8
    ticks = [i / 10 for i in range(9)]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xs, ys, s=80, alpha=0.82, c=colors, edgecolors="white", linewidths=0.7)
    ax.plot([min_val, max_val], [min_val, max_val], linestyle=":", color="gray", linewidth=1)
    ax.set_xlabel("Q-score Identified", fontsize=17)
    ax.set_ylabel("Q-Score Ground Truth", fontsize=17)
    ax.set_title(title, fontsize=20)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])
    ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    used_bins = [name for name in RESOLUTION_COLORS if name in set(bins)]
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=10,
            label=label,
        )
        for label in used_bins
        for color in [RESOLUTION_COLORS[label]]
    ]
    ax.legend(handles=legend_handles, title="Map Resolution", loc="lower right", fontsize=12, title_fontsize=13)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``--csv`` in, ``--out`` PNG out, optional ``--title``."""
    parser = argparse.ArgumentParser(description="Plot benchmark Q-score results.")
    parser.add_argument("--csv", required=True, help="benchmark_qscores.csv path.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument(
        "--title",
        default="Q-score: Ground Truth vs Identified",
        help="Plot title.",
    )
    args = parser.parse_args(argv)
    plot_qscores(Path(args.csv), Path(args.out), title=args.title)
    print(f"Saved {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
