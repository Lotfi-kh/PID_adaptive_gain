#!/usr/bin/env python3
"""
paired_plot.py — overlay baseline vs adaptive RESPONSE from two flown runs.

Unlike ab_plot.py (shadow / input-side: gains + commanded torque), this plots
what the drone ACTUALLY did in two separate flights — roll, pitch, and their
rates — so you can read stability (overshoot, settling) and reaction speed
(time-to-peak, recovery) directly off the response.

Both CSVs come from ulog_to_eval_csv.py and carry a dist_active flag. The two
flights have different ulog clocks, so each is shifted to put its FIRST
disturbance onset at t=0; after that the shared event sequence lines up.

Usage:
    python sitl/paired_plot.py                       # defaults below
    python sitl/paired_plot.py base.csv rl.csv -o out.png
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results"))


def load(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def onset(d):
    """First time dist_active turns on — used to align the two flights."""
    on = np.where(d["dist_active"] > 0)[0]
    return d["timestamp"][on[0]] if len(on) else d["timestamp"][0]


def windows(d, t_shift):
    """Contiguous dist_active runs, returned on the shifted (aligned) clock."""
    idx = np.where(d["dist_active"] > 0)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    out = []
    for g in np.split(idx, splits + 1):
        out.append((d["timestamp"][g[0]] - t_shift, d["timestamp"][g[-1]] - t_shift))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("baseline", nargs="?",
                    default=os.path.join(RESULTS_DIR, "baseline_disturb.csv"))
    ap.add_argument("adaptive", nargs="?",
                    default=os.path.join(RESULTS_DIR, "rl_disturb.csv"))
    ap.add_argument("-o", "--out",
                    default=os.path.join(RESULTS_DIR, "gazebo", "paired_compare.png"))
    args = ap.parse_args()

    b = load(args.baseline)
    a = load(args.adaptive)
    tb = b["timestamp"] - onset(b)      # aligned: 0 = first kick
    ta = a["timestamp"] - onset(a)

    # plot a little before the first kick through the settled tail
    LO, HI = -3.0, 80.0

    # roll/pitch are radians in the CSV -> degrees; rates rad/s -> deg/s
    series = [
        ("roll (deg)",        "roll",       np.degrees),
        ("pitch (deg)",       "pitch",      np.degrees),
        ("roll rate (deg/s)",  "roll_rate",  np.degrees),
        ("pitch rate (deg/s)", "pitch_rate", np.degrees),
    ]

    fig, axes = plt.subplots(len(series), 1, figsize=(11, 2.4 * len(series)),
                             sharex=True)
    win = windows(b, onset(b))          # shade the baseline's disturbance windows
    for ax, (ylabel, col, conv) in zip(axes, series):
        for i, (w0, w1) in enumerate(win):
            ax.axvspan(w0, w1, color="orange", alpha=0.15,
                       label="disturbance" if i == 0 else None)
        ax.plot(tb, conv(b[col]), color="gray", ls="--", lw=1.3, label="baseline")
        ax.plot(ta, conv(a[col]), color="C0", ls="-", lw=1.3, label="adaptive (RL)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(LO, HI)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time since first disturbance onset (s)")
    axes[0].set_title(
        "Baseline static-gain PID vs adaptive (RL) — flown response, SITL/Gazebo F450\n"
        "two separate flights, same disturbance sequence, aligned on first kick",
        fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[paired-plot] wrote {args.out}")


if __name__ == "__main__":
    main()
