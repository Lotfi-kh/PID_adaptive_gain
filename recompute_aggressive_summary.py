#!/usr/bin/env python3
"""Axis-aware recompute of the aggressive-eval summary from saved NPZ files.

Fixes the roll-only bug in run_aggressive_eval*.summarise(): for pitch-axis
disturbances the baseline roll rate is ~0, so peak% divides by near-zero and
produces nonsense (-1500%..-5600%). Here every metric is measured on the axis
that was actually disturbed (roll/pitch), and 'both' uses the L2 combination.

    python recompute_aggressive_summary.py <eval_dir> <label>
"""
import glob
import os
import sys

import numpy as np

DIST_STEP = 150          # disturbance onset (matches run_aggressive_eval)
SETTLE = 1e-2            # |rate| (rad/s) considered "recovered"
CTRL_DT = 1.0 / 500.0    # 500 Hz


def axis_signals(d, who, dist_axis):
    """Return (angle[ep,t], rate[ep,t]) on the disturbed axis. 'both' -> L2."""
    roll_a, pitch_a = d[f"{who}_rolls"], d[f"{who}_pitch_angles"]
    roll_r, pitch_r = d[f"{who}_roll_rates"], d[f"{who}_pitch_rates"]
    if dist_axis == "roll":
        return np.abs(roll_a), np.abs(roll_r)
    if dist_axis == "pitch":
        return np.abs(pitch_a), np.abs(pitch_r)
    return (np.hypot(roll_a, pitch_a), np.hypot(roll_r, pitch_r))   # both


def recovery_time(rate, dist_step=DIST_STEP):
    """Mean steps-after-onset until |rate| stays below SETTLE, per episode."""
    out = []
    for ep in rate:
        post = ep[dist_step:]
        below = post < SETTLE
        if below.all():
            out.append(0.0)
            continue
        # last index that is still above threshold
        last_bad = np.where(~below)[0]
        out.append((last_bad[-1] + 1) * 1.0)
    return np.mean(out)


def cond_metrics(npz):
    d = np.load(npz, allow_pickle=True)
    dist_axis = str(d["dist_axis"])
    _, bl_rate = axis_signals(d, "bl", dist_axis)
    _, rl_rate = axis_signals(d, "rl", dist_axis)

    # peak rate (post-onset) averaged over episodes
    bl_peak = bl_rate[:, DIST_STEP:].max(axis=1).mean()
    rl_peak = rl_rate[:, DIST_STEP:].max(axis=1).mean()
    bl_rms = np.sqrt((bl_rate[:, DIST_STEP:] ** 2).mean())
    rl_rms = np.sqrt((rl_rate[:, DIST_STEP:] ** 2).mean())
    bl_rec = recovery_time(bl_rate)
    rl_rec = recovery_time(rl_rate)

    peak_imp = (bl_peak - rl_peak) / bl_peak * 100 if bl_peak else 0.0
    rms_imp = (bl_rms - rl_rms) / bl_rms * 100 if bl_rms else 0.0
    rec_imp = (bl_rec - rl_rec) / bl_rec * 100 if bl_rec else 0.0

    return dict(
        name=os.path.basename(os.path.dirname(npz)),
        bl_crash=int(d["bl_crashes"].sum()), rl_crash=int(d["rl_crashes"].sum()),
        peak=peak_imp, rms=rms_imp, rec=rec_imp,
        bl_rec=bl_rec * CTRL_DT, rl_rec=rl_rec * CTRL_DT,
    )


def winner(m):
    # RL wins if no extra crashes, peak >= +10%, RMS not worse by >5%
    if m["rl_crash"] > m["bl_crash"]:
        return "BL"
    if m["peak"] >= 10.0 and m["rms"] >= -5.0:
        return "RL"
    if m["peak"] <= -10.0:
        return "BL"
    return "MIXED"


def main():
    eval_dir, label = sys.argv[1], sys.argv[2]
    npzs = sorted(glob.glob(os.path.join(eval_dir, "*", "*.npz")))
    rows = [cond_metrics(f) for f in npzs]
    for r in rows:
        r["winner"] = winner(r)

    rl = sum(r["winner"] == "RL" for r in rows)
    bl = sum(r["winner"] == "BL" for r in rows)
    mixed = sum(r["winner"] == "MIXED" for r in rows)
    n_eps = 20
    tot = len(rows) * n_eps
    bl_cr = sum(r["bl_crash"] for r in rows)
    rl_cr = sum(r["rl_crash"] for r in rows)
    mp = np.mean([r["peak"] for r in rows])
    mr = np.mean([r["rms"] for r in rows])
    mrec = np.mean([r["rec"] for r in rows])

    lines = []
    lines.append(f"AGGRESSIVE EVAL SUMMARY (axis-aware, corrected) — {label}")
    lines.append(f"  [{len(rows)} conditions x {n_eps} episodes = {len(rows)*n_eps} rollouts/controller]")
    lines.append("")
    lines.append(f"RL wins: {rl}/{len(rows)}   Baseline wins: {bl}   Mixed: {mixed}")
    lines.append(f"Crashes: {rl_cr}/{tot} RL, {bl_cr}/{tot} Baseline")
    lines.append(f"Mean peak rate improvement: {mp:+.1f}%")
    lines.append(f"Mean RMS rate improvement:  {mr:+.1f}%")
    lines.append(f"Mean recovery improvement:  {mrec:+.1f}%")
    lines.append("")
    hdr = f"{'Condition':<26}{'Crash':>12}{'RMS Δ%':>9}{'Peak Δ%':>10}{'Rec Δ%':>9}{'BL rec':>8}{'RL rec':>8}  Winner"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        crash = f"{r['bl_crash']}/{n_eps} vs {r['rl_crash']}/{n_eps}"
        lines.append(
            f"{r['name']:<26}{crash:>12}"
            f"{r['rms']:>8.1f}%{r['peak']:>9.1f}%{r['rec']:>8.1f}%"
            f"{r['bl_rec']:>8.2f}{r['rl_rec']:>8.2f}  {r['winner']:>5}")

    # by-magnitude breakdown
    lines.append("")
    lines.append("By magnitude:")
    for mag in ("low", "medium", "high", "extreme"):
        sub = [r for r in rows if f"_{mag}_" in r["name"]]
        w = sum(x["winner"] == "RL" for x in sub)
        lines.append(f"  {mag:<8} RL wins {w}/{len(sub)}   peak {np.mean([x['peak'] for x in sub]):+.1f}%")
    lines.append("")
    lines.append("By axis:")
    for ax in ("roll", "pitch", "both"):
        sub = [r for r in rows if r["name"].endswith(ax)]
        w = sum(x["winner"] == "RL" for x in sub)
        lines.append(f"  {ax:<6} RL wins {w}/{len(sub)}   peak {np.mean([x['peak'] for x in sub]):+.1f}%   rec {np.mean([x['rec'] for x in sub]):+.1f}%")

    out = os.path.join(eval_dir, "aggressive_eval_corrected.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
