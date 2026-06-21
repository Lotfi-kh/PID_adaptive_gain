#!/usr/bin/env python3
"""Bar chart: peak-rate improvement by disturbance magnitude, c860k vs noiselat.

In-distribution aggressive eval (48 conditions x 20 ep). Reuses the axis-aware
recompute so both controllers are scored with the identical metric.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recompute_aggressive_summary import cond_metrics, winner
import glob

MAGS = ["low", "medium", "high", "extreme"]
RUNS = {
    "c860k (aggressive specialist)": "test_results/aggressive_eval",
    "noiselat (robust generalist)": "test_results/aggressive_eval_noiselat",
}


def by_mag(eval_dir):
    rows = [cond_metrics(f) for f in sorted(glob.glob(os.path.join(eval_dir, "*", "*.npz")))]
    return {m: np.mean([r["peak"] for r in rows if f"_{m}_" in r["name"]]) for m in MAGS}


def main():
    data = {lbl: by_mag(d) for lbl, d in RUNS.items()}
    x = np.arange(len(MAGS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = ["#1f77b4", "#2ca02c"]
    for i, (lbl, vals) in enumerate(data.items()):
        ys = [vals[m] for m in MAGS]
        bars = ax.bar(x + (i - 0.5) * w, ys, w, label=lbl, color=colors[i], alpha=0.88,
                      edgecolor="k", linewidth=0.4)
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:+.0f}%",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({n} N·m)" for m, n in
                        zip(MAGS, ["0.043", "0.17", "0.30", "0.40"])])
    ax.set_ylabel("peak rate-rejection improvement vs static PID (%)")
    ax.set_title("In-distribution aggressive eval (PyBullet F450, 960 rollouts each)\n"
                 "adaptive vs static baseline — both 48/48 wins, 0 crashes")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = "test_results/aggressive_eval_noiselat/c860k_vs_noiselat_by_magnitude.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
