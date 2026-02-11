# =============================
# File: plot_qscore_scatter.py
# =============================
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INPUT_CSV = Path("Alphafold_fetch/qscore_groundtruth_run_2_8_2/qscore_groundtruth_results.csv")
OUTPUT_PNG = INPUT_CSV.with_suffix(".png")
SHOW_PLOT = True


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = ["gene_id", "qscore_ground_truth", "qscore_identified", "resolution"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"Missing column: {col}")

    x = pd.to_numeric(df["qscore_identified"], errors="coerce")
    y = pd.to_numeric(df["qscore_ground_truth"], errors="coerce")
    resolution = pd.to_numeric(df["resolution"], errors="coerce")
    labels = df["gene_id"].astype(str)

    mask = x.notna() & y.notna() & resolution.notna()
    x = x[mask]
    y = y[mask]
    resolution = resolution[mask]
    labels = labels[mask]

    if x.empty:
        raise SystemExit("No valid qscore_ground_truth/qscore_identified pairs found.")

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = []
    for r in resolution:
        if r <= 4.0:
            colors.append("tab:blue")
        elif r <= 6.0:
            colors.append("tab:orange")
        else:
            colors.append("tab:red")

    ax.scatter(x, y, s=30, alpha=0.85, c=colors)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle=":", color="gray", linewidth=1)

    for xi, yi, label in zip(x, y, labels):
        ax.text(xi, yi, label, fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("qscore_identified")
    ax.set_ylabel("qscore_ground_truth")
    ax.set_title("Q-score: Identified vs Ground Truth")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="<= 4A"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=8, label="4-6A"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=8, label="> 6A"),
    ]
    ax.legend(handles=legend_handles, title="GT resolution", loc="lower right")

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved plot: {OUTPUT_PNG}")
    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
