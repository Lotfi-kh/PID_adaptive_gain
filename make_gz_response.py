#!/usr/bin/env python3
"""Gazebo response overlay: baseline vs adaptive, roll angle + roll rate during the
strong roll kick (0.40 N.m, t=44.265-47.265 s), seed 1. Aligns each flight to its own
disturbance onset so the two traces overlay cleanly."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "test_results/gazebo/_archive_kick_noiselat"
T_NOM, KICK_DUR = 44.265, 3.0

def load(tag):
    t, roll, rr = [], [], []
    for r in csv.DictReader(open(f"{D}/pair_seed1_{tag}_wide.csv")):
        try:
            t.append(float(r["t"])); roll.append(float(r["roll"])); rr.append(float(r["roll_rate"]))
        except (ValueError, KeyError):
            pass
    return np.array(t), np.rad2deg(np.array(roll)), np.array(rr)

def onset(t, rr):
    m = (t > T_NOM - 0.8) & (t < T_NOM + 1.0)
    idx = np.where(m & (np.abs(rr) > 0.12))[0]
    return t[idx[0]] if len(idx) else T_NOM

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)
for tag, color, lbl, ls in [("baseline", "#888888", "static-gain PID", "--"),
                            ("adaptive", "#1f77b4", "adaptive (RL)", "-")]:
    t, roll, rr = load(tag)
    t0 = onset(t, rr)
    tt = t - t0
    w = (tt > -0.8) & (tt < 5.5)
    print(f"{tag:9s} onset={t0:.3f}s  peak|roll|={np.abs(roll[w]).max():.2f}deg  peak|rate|={np.abs(rr[w]).max():.3f}")
    ax1.plot(tt[w], roll[w], color=color, ls=ls, lw=1.9, label=lbl)
    ax2.plot(tt[w], rr[w],   color=color, ls=ls, lw=1.9, label=lbl)

for ax in (ax1, ax2):
    ax.axvspan(0, KICK_DUR, color="0.9", alpha=0.7, zorder=0, label="_")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.grid(True, ls=":", alpha=0.5)
ax1.text(0.05, ax1.get_ylim()[1]*0.92, " 0.40 N$\\cdot$m roll kick", fontsize=8, color="0.4", va="top")
ax1.set_ylabel("roll angle (deg)")
ax2.set_ylabel("roll rate (rad/s)")
ax2.set_xlabel("time since kick onset (s)")
ax1.legend(loc="upper right", fontsize=9)
ax1.set_title("Gazebo SITL: response to a sustained 0.40 N$\\cdot$m roll kick (representative pair)")
fig.tight_layout()
fig.savefig("test_results/figures/gz_kick_response.png", dpi=200, bbox_inches="tight")
print("[OK] wrote test_results/figures/gz_kick_response.png")
