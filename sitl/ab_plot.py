#!/usr/bin/env python3
"""
ab_plot.py — plot the adaptive-vs-baseline comparison from ab_extract.py output.

Five stacked panels (Kp, Ki, Kd, roll torque, pitch torque). Adaptive is the
solid line, baseline/shadow is the dashed line. Disturbance windows are shaded.

Usage:
    python sitl/ab_plot.py                         # test_results/ab_test.csv -> ab_compare.png
    python sitl/ab_plot.py in.csv -o fig.png --events ev.csv
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRE_MARGIN = 3.0   # must match ab_extract.py so the shading lines up
RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results"))


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return cols


def shade_windows(ax, events_path, ymin_label=False):
    if not os.path.exists(events_path):
        return
    ev = list(csv.DictReader(open(events_path)))
    if not ev:
        return
    min_apply = min(float(r["t_apply_s"]) for r in ev)
    for i, r in enumerate(ev):
        a = float(r["t_apply_s"]) - min_apply + PRE_MARGIN
        c = float(r["t_clear_s"]) - min_apply + PRE_MARGIN
        ax.axvspan(a, c, color="orange", alpha=0.15,
                   label="disturbance" if i == 0 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("csv", nargs="?", default=os.path.join(RESULTS_DIR, "ab_test.csv"))
    ap.add_argument("--events", default=os.path.join(RESULTS_DIR, "ab_events.csv"))
    ap.add_argument("-o", "--out", default=os.path.join(RESULTS_DIR, "ab_compare.png"))
    args = ap.parse_args()

    d = load_csv(args.csv)
    t = d["t"]

    # each panel: (ylabel, [(column, color, linestyle, label), ...])
    panels = [
        ("attitude (deg)", [("roll_deg", "C3", "-", "roll (adaptive flight)"),
                            ("pitch_deg", "C2", "-", "pitch (adaptive flight)")]),
        ("Kp", [("base_kp", "gray", "--", "baseline"), ("active_kp", "C0", "-", "adaptive")]),
        ("Ki", [("base_ki", "gray", "--", "baseline"), ("active_ki", "C0", "-", "adaptive")]),
        ("Kd", [("base_kd", "gray", "--", "baseline"), ("active_kd", "C0", "-", "adaptive")]),
        ("roll torque", [("shadow_torque_roll", "gray", "--", "baseline (shadow)"),
                         ("active_torque_roll", "C0", "-", "adaptive")]),
        ("pitch torque", [("shadow_torque_pitch", "gray", "--", "baseline (shadow)"),
                          ("active_torque_pitch", "C0", "-", "adaptive")]),
    ]
    # drop panels whose columns are not in the CSV
    panels = [(yl, lines) for yl, lines in panels
              if all(c in d for c, *_ in lines)]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.3 * len(panels)), sharex=True)
    for ax, (ylabel, lines) in zip(axes, panels):
        shade_windows(ax, args.events)
        for col, color, ls, label in lines:
            ax.plot(t, d[col], color=color, ls=ls, lw=1.3, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time since window start (s)")
    axes[0].set_title("Adaptive (RL) vs baseline static-gain PID — SITL/Gazebo, F450\n"
                      "(attitude is the adaptive flight only; baseline is a shadow, it did not fly)",
                      fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[ab-plot] wrote {args.out}")


if __name__ == "__main__":
    main()
