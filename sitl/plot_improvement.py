#!/usr/bin/env python3
"""Diverging %-improvement chart: adaptive vs baseline, per metric, paired by seed.

Positive bar = adaptive better (perf metric lower, or cost metric lower).
Stars = paired Wilcoxon significance. Reads a metrics_master CSV directly.

    python sitl/plot_improvement.py                 # kick batch
    python sitl/plot_improvement.py --wind          # wind batch
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

GZ = os.path.join(os.path.dirname(__file__), "..", "test_results", "gazebo")

# metric -> (column, lower_is_better)  ; grouped for the figure
PERF = [
    ("peak|roll| deg", "peak |roll|"),
    ("peak|pitch| deg", "peak |pitch|"),
    ("RMS roll deg", "RMS roll"),
    ("RMS pitch deg", "RMS pitch"),
    ("RMS rollrate", "RMS rollrate"),
    ("RMS rate_err", "RMS rate-track err"),
    ("mean recovery s", "mean recovery"),
    ("max recovery s", "max recovery"),
    ("alt_dev m", "altitude dev"),
    ("peak|vz|", "peak |vz|"),
]
COST = [
    ("peak|tau_roll|", "peak |torque_roll|"),
    ("peak|tau_pitch|", "peak |torque_pitch|"),
    ("effort_roll Nms", "effort roll"),
    ("effort_pitch Nms", "effort pitch"),
]


def load_pairs(master):
    rows = list(csv.DictReader(open(master)))
    rows = [r for r in rows if r.get("tag") == "pair"]
    pairs = {}
    for r in rows:
        pairs.setdefault(r["seed"], {})[r["controller"]] = r
    return {s: d for s, d in pairs.items() if "baseline" in d and "adaptive" in d}


def stat(pairs, col):
    seeds = sorted(pairs, key=int)
    b = np.array([float(pairs[s]["baseline"][col]) for s in seeds])
    a = np.array([float(pairs[s]["adaptive"][col]) for s in seeds])
    bm, am = b.mean(), a.mean()
    imp = (bm - am) / bm * 100 if bm else np.nan          # +%=adaptive lower=better
    win = float(np.mean(a < b))
    try:
        p = wilcoxon(b, a).pvalue
    except ValueError:
        p = 1.0
    return imp, p, win, len(seeds)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind", action="store_true")
    args = ap.parse_args()
    master = os.path.join(GZ, "metrics_master_wind.csv" if args.wind
                          else "metrics_master.csv")
    pairs = load_pairs(master)
    title = ("Continuous wind" if args.wind else "Torque kicks") + \
        f" — adaptive vs baseline (N={len(pairs)} paired seeds, SITL/Gazebo F450)"

    labels, imps, ps, wins, cols = [], [], [], [], []
    for group, color in ((PERF, "perf"), (COST, "cost")):
        for col, lab in group:
            if not all(col in pairs[s]["baseline"] for s in pairs):
                continue
            imp, p, win, n = stat(pairs, col)
            if np.isnan(imp):
                continue
            labels.append(lab + ("  (cost)" if color == "cost" else ""))
            imps.append(imp); ps.append(p); wins.append(win); cols.append(color)

    order = list(range(len(labels)))[::-1]          # first metric on top
    labels = [labels[i] for i in order]
    imps = [imps[i] for i in order]; ps = [ps[i] for i in order]
    wins = [wins[i] for i in order]

    y = np.arange(len(labels))
    bar_c = ["#2ca02c" if v >= 0 else "#d62728" for v in imps]   # green better / red worse
    fig, ax = plt.subplots(figsize=(9, 0.52 * len(labels) + 1.6))
    ax.barh(y, imps, color=bar_c, alpha=0.85, edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("← adaptive worse      improvement vs baseline (%)      adaptive better →",
                  fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    xpad = (max(imps) - min(imps)) * 0.04 + 0.5
    for yi, (v, p, w) in enumerate(zip(imps, ps, wins)):
        txt = f"{v:+.1f}%  {stars(p)}".rstrip()
        ax.text(v + (xpad if v >= 0 else -xpad), yi, txt,
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.margins(x=0.18)
    fig.text(0.99, 0.01, "* p<.05  ** p<.01  *** p<.001  (paired Wilcoxon)",
             ha="right", fontsize=7, color="0.4")
    fig.tight_layout()
    out = os.path.join(GZ, "improvement_wind.png" if args.wind
                       else "improvement_kick.png")
    fig.savefig(out, dpi=150)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
