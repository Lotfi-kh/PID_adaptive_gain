#!/usr/bin/env python3
"""
ab_compare.py — Baseline-PID vs RL-PID disturbance-rejection A/B comparison.

Inputs: two CSVs from ulog_to_eval_csv.py (schema:
timestamp,roll,pitch,roll_rate,pitch_rate,kp,ki,kd,dist_active).

Alignment: each run is shifted so its FIRST dist_active rising edge = t=0
(robust event-based alignment; the disturbance sequence is identical both
runs, so this overlays the events without any clock mapping).

Outputs:
  - printed summary table (baseline vs RL, improvement %)
  - <out>.png : roll / pitch / roll_rate / pitch_rate / gains vs time,
                disturbance windows shaded
  - <out>_metrics.csv : the metric table

Usage:
    python sitl/ab_compare.py baseline.csv rl.csv -o ab_result
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SETTLE_RAD = 0.05      # |roll|,|pitch| within this ⇒ "recovered" (~2.9°)
TAIL_S = 8.0           # seconds of window kept after the last disturbance clear


def load(path):
    rows = list(csv.DictReader(open(path)))
    a = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    t = a["timestamp"]
    d = a["dist_active"].astype(int)
    # anchor: first rising edge of dist_active (fallback: t[0])
    re = np.where((d[1:] == 1) & (d[:-1] == 0))[0]
    t0 = t[re[0] + 1] if len(re) else t[0]
    a["timestamp"] = t - t0
    return a


def windows(a):
    """[(t_apply, t_clear)] in aligned time from dist_active edges."""
    d = a["dist_active"].astype(int)
    t = a["timestamp"]
    out, on = [], None
    for i in range(len(d)):
        if d[i] and on is None:
            on = t[i]
        elif not d[i] and on is not None:
            out.append((on, t[i]))
            on = None
    if on is not None:
        out.append((on, t[-1]))
    return out


def metrics(a, win):
    t = a["timestamp"]
    end = (max(c for _, c in win) + TAIL_S) if win else t[-1]
    m = (t >= 0) & (t <= end)
    roll, pitch = a["roll"][m], a["pitch"][m]
    rr, pr = a["roll_rate"][m], a["pitch_rate"][m]
    tm = t[m]
    deg = 180.0 / np.pi

    # per-event recovery: from each clear, time until |roll|&|pitch|<SETTLE_RAD
    rec = []
    for _, tc in win:
        seg = (t >= tc)
        if not seg.any():
            continue
        ts, rs, ps = t[seg], np.abs(a["roll"][seg]), np.abs(a["pitch"][seg])
        ok = np.where((rs < SETTLE_RAD) & (ps < SETTLE_RAD))[0]
        rec.append(float(ts[ok[0]] - tc) if len(ok) else np.nan)

    def tv(x):
        return float(np.sum(np.abs(np.diff(x))))

    return {
        "peak|roll| deg":   float(np.abs(roll).max() * deg),
        "peak|pitch| deg":  float(np.abs(pitch).max() * deg),
        "peak|rollrate|":   float(np.abs(rr).max()),
        "peak|pitchrate|":  float(np.abs(pr).max()),
        "RMS roll deg":     float(np.sqrt(np.mean(roll**2)) * deg),
        "RMS pitch deg":    float(np.sqrt(np.mean(pitch**2)) * deg),
        "RMS rollrate":     float(np.sqrt(np.mean(rr**2))),
        "RMS pitchrate":    float(np.sqrt(np.mean(pr**2))),
        "mean recovery s":  float(np.nanmean(rec)) if rec else float("nan"),
        "max recovery s":   float(np.nanmax(rec)) if rec else float("nan"),
        "kp range":         float(a["kp"][m].max() - a["kp"][m].min()),
        "ki min":           float(a["ki"][m].min()),
        "kp total-var":     tv(a["kp"][m]),
        "crash":            1.0 if (np.abs(roll).max() > np.radians(60) or
                                    np.abs(pitch).max() > np.radians(60)) else 0.0,
    }, (tm, roll, pitch, rr, pr)


def plot(B, R, winB, winR, out, title=None, mB=None, mR=None):
    fig = plt.figure(figsize=(13, 17))
    gs  = fig.add_gridspec(7, 1, hspace=0.45)
    ax  = [fig.add_subplot(gs[i]) for i in range(5)]
    ax_desc = fig.add_subplot(gs[5:])
    ax_desc.axis("off")

    series = [("roll", "roll [rad]"), ("pitch", "pitch [rad]"),
              ("roll_rate", "roll_rate [rad/s]"),
              ("pitch_rate", "pitch_rate [rad/s]")]
    for i, (key, lab) in enumerate(series):
        ax[i].plot(B["timestamp"], B[key], lw=0.9, label="baseline", color="#1565C0")
        ax[i].plot(R["timestamp"], R[key], lw=0.9, label="RL",       color="#C62828")
        ax[i].set_ylabel(lab)
        ax[i].grid(alpha=0.3)
        ax[i].legend(loc="upper right", fontsize=8)
    for g, c in (("kp", "#C62828"), ("ki", "#1565C0"), ("kd", "#2e7d32")):
        ax[4].plot(R["timestamp"], R[g], lw=1.0, color=c, label=f"RL {g}")
        ax[4].plot(B["timestamp"], B[g], lw=0.8, ls=":", color=c,
                   label=f"base {g}")
    ax[4].set_ylabel("gains")
    ax[4].set_xlabel("aligned time [s]")
    ax[4].grid(alpha=0.3)
    ax[4].legend(loc="upper right", fontsize=7, ncol=3)
    for a_, c_ in ((winB, "#bbb"), (winR, "#9e9")):
        for (ta, tc) in a_:
            for axx in ax:
                axx.axvspan(ta, tc, color=c_, alpha=0.25, lw=0)

    main_title = title if title else "Baseline PID vs RL PID — disturbance rejection"
    ax[0].set_title(main_title + "\n(blue=baseline  red=RL  shaded=disturbance active)",
                    fontsize=11)

    # Description box
    deg = 180.0 / 3.14159265
    if mB and mR:
        lines = [
            "RESULT SUMMARY",
            f"  Peak |roll|:   baseline {mB['peak|roll| deg']:.2f}°  →  RL {mR['peak|roll| deg']:.2f}°"
            f"   ({(mB['peak|roll| deg']-mR['peak|roll| deg'])/mB['peak|roll| deg']*100:+.1f}%)",
            f"  Peak |pitch|:  baseline {mB['peak|pitch| deg']:.2f}°  →  RL {mR['peak|pitch| deg']:.2f}°"
            f"   ({(mB['peak|pitch| deg']-mR['peak|pitch| deg'])/mB['peak|pitch| deg']*100:+.1f}%)",
            f"  RMS roll:      baseline {mB['RMS roll deg']:.3f}°  →  RL {mR['RMS roll deg']:.3f}°",
            f"  RMS pitch:     baseline {mB['RMS pitch deg']:.3f}°  →  RL {mR['RMS pitch deg']:.3f}°",
            f"  Peak rollrate: baseline {mB['peak|rollrate|']:.4f} rad/s  →  RL {mR['peak|rollrate|']:.4f} rad/s",
            f"  Crash:         baseline={'YES' if mB['crash'] else 'NO'}   RL={'YES' if mR['crash'] else 'NO'}",
            "",
            "HOW TO READ:",
            "  Rows 1-2: attitude error (lower = more stable).  "
            "Row 3-4: angular rate (lower = smoother).",
            "  Row 5: RL gain trajectory — Kp adapts upward during disturbance, "
            "resets to 0.171 when calm.",
            "  Shaded regions = disturbance active.  "
            "Positive improvement % = RL is better.",
        ]
        ax_desc.text(0.01, 0.98, "\n".join(lines), transform=ax_desc.transAxes,
                     fontsize=8.5, verticalalignment="top", fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                               edgecolor="#aaa", alpha=0.9))

    fig.savefig(out + ".png", dpi=110, bbox_inches="tight")
    print(f"[ab] plot → {out}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("baseline_csv")
    ap.add_argument("rl_csv")
    ap.add_argument("-o", "--out", default="ab_result")
    ap.add_argument("--title", default=None,
                    help="Plot title / description (e.g. 'Run 1 — windy world 60s hover')")
    args = ap.parse_args()

    B, R = load(args.baseline_csv), load(args.rl_csv)
    wB, wR = windows(B), windows(R)
    mB, _ = metrics(B, wB)
    mR, _ = metrics(R, wR)

    print(f"\n  {'metric':<18}{'baseline':>12}{'RL':>12}{'improvement':>14}")
    print("  " + "-" * 54)
    rows = [["metric", "baseline", "RL", "improvement_%"]]
    for k in mB:
        b, r = mB[k], mR[k]
        if k == "crash":
            v = "same" if b == r else ("RL CRASH" if r else "RL ok")
            print(f"  {k:<18}{b:>12.0f}{r:>12.0f}{v:>14}")
            rows.append([k, b, r, v])
            continue
        imp = (b - r) / b * 100 if b not in (0.0,) else 0.0   # lower = better
        tag = "" if imp >= 0 else " worse"
        print(f"  {k:<18}{b:>12.4f}{r:>12.4f}{imp:>+12.1f}%{tag}")
        rows.append([k, f"{b:.6f}", f"{r:.6f}", f"{imp:+.1f}"])
    print("\n  (+improvement% = RL better; 'kp range'/'total-var' just describe "
          "RL adaptation, not better/worse)")

    with open(args.out + "_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  [ab] metrics → {args.out}_metrics.csv")
    plot(B, R, wB, wR, args.out, title=args.title, mB=mB, mR=mR)


if __name__ == "__main__":
    main()
