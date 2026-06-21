#!/usr/bin/env python3
"""noiselat: peak rate-rejection improvement by magnitude, nominal vs off-nominal+noise.

Both are the SAME 48-condition grid (4 noise x 4 mag x 3 axes, 20 ep) scored with
the identical axis-aware aggregator. Nominal = clean F450 (c860k's distribution);
off-nominal = randomized mass/inertia + gyro noise 0.005 + 1-step latency
(noiselat's own training distribution).
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recompute_aggressive_summary import cond_metrics

MAGS = ["low", "medium", "high", "extreme"]
NLAB = ["0.043", "0.17", "0.30", "0.40"]
RUNS = {
    "Nominal (clean F450)": "test_results/aggressive_eval_noiselat",
    "Off-nominal + noise": "test_results/offnominal_eval_noiselat",
}


def by_mag(eval_dir, metric):
    rows = [cond_metrics(f) for f in sorted(glob.glob(os.path.join(eval_dir, "*", "*.npz")))]
    return {m: np.mean([r[metric] for r in rows if f"_{m}_" in r["name"]]) for m in MAGS}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(MAGS))
    w = 0.38
    colors = ["#1f77b4", "#d62728"]
    for ax, metric, title in (
        (axes[0], "peak", "Peak rate-rejection improvement"),
        (axes[1], "rms", "RMS rate improvement"),
    ):
        for i, (lbl, d) in enumerate(RUNS.items()):
            vals = by_mag(d, metric)
            ys = [vals[m] for m in MAGS]
            bars = ax.bar(x + (i - 0.5) * w, ys, w, label=lbl, color=colors[i],
                          alpha=0.88, edgecolor="k", linewidth=0.4)
            for b, v in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, v + (0.4 if v >= 0 else -0.4),
                        f"{v:+.0f}%", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n({n} N·m)" for m, n in zip(MAGS, NLAB)])
        ax.set_ylabel("adaptive improvement vs static PID (%)")
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("noiselat — same 48-condition grid, two regimes (960 rollouts each, 0 crashes)\n"
                 "off-nominal margin compresses (noisier ruler); only LOW disturbance loses RMS "
                 "(high gains amplify noise)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = "test_results/offnominal_eval_noiselat/nominal_vs_offnominal_by_magnitude.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
