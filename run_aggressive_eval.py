#!/usr/bin/env python3
"""
run_aggressive_eval.py — Aggressive multi-condition disturbance evaluation
=========================================================================
Tests the c860k model across 48 conditions:
    init_noise     ∈ {0.05, 0.10, 0.15, 0.20}     (4 noise levels)
    dist_magnitude ∈ {low, medium, high, extreme}  (0.043, 0.17, 0.30, 0.40 N·m)
    dist_axis      ∈ {roll, pitch, both}            (3 axes)

20 episodes per condition, 10 steps disturbance duration.
All results saved under test_results/aggressive_eval/
"""

import argparse
import os
import subprocess
import sys
import time
import numpy as np

MODEL = "/home/lotfikh/rl_pid_tuner/results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip"
OUT_ROOT = "/home/lotfikh/rl_pid_tuner/test_results/aggressive_eval"
SCRIPT = "/home/lotfikh/rl_pid_tuner/eval_disturbance.py"
PYTHON = sys.executable

INIT_NOISES = [0.05, 0.10, 0.15, 0.20]
DIST_MAGNITUDES = [
    ("low",     4.3e-2),   # baseline SITL mild kick equivalent
    ("medium",  1.7e-1),   # SITL mild-medium equivalent
    ("high",    3.0e-1),   # SITL strong kick equivalent
    ("extreme", 4.0e-1),   # SITL maximum in-envelope limit
]
DIST_AXES = ["roll", "pitch", "both"]
EPISODES = 20
DIST_STEP = 150
DIST_DURATION = 10   # longer than default 5 — more sustained pressure


def run_condition(noise, mag_label, mag_val, dist_axis):
    tag = f"noise{int(noise*100):02d}_{mag_label}_{dist_axis}"
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "disturbance_eval_results.npz")

    cmd = [
        PYTHON, SCRIPT,
        "--model",          MODEL,
        "--axis",           "roll+pitch",
        "--dist-axis",      dist_axis,
        "--episodes",       str(EPISODES),
        "--init-noise",     str(noise),
        "--dist-step",      str(DIST_STEP),
        "--dist-magnitude", str(mag_val),
        "--dist-duration",  str(DIST_DURATION),
        "--out-dir",        out_dir,
        "--seed",           "42",
        "--no-plots",
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"  ERROR: {tag}\n{r.stderr[-400:]}")
        return None, tag
    print(f"  OK  {tag:<40s}  {elapsed:4.0f}s")
    if not os.path.exists(npz):
        return None, tag
    return np.load(npz, allow_pickle=True), tag


def rms(arr_pad, ep_lens):
    vals = []
    for i in range(arr_pad.shape[0]):
        ep = arr_pad[i, :int(ep_lens[i])]
        ep = ep[~np.isnan(ep)]
        vals.extend(ep.tolist())
    sq = np.array(vals) ** 2
    return float(np.sqrt(sq.mean())) if sq.size else float("nan")


def peak_post(arr_pad, ep_lens, dist_start):
    peaks = []
    for i in range(arr_pad.shape[0]):
        end = int(ep_lens[i])
        if end > dist_start:
            w = arr_pad[i, dist_start:end]
            w = w[~np.isnan(w)]
            if w.size:
                peaks.append(float(np.max(np.abs(w))))
    return float(np.mean(peaks)) if peaks else float("nan")


def recovery(arr_pad, ep_lens, dist_end, thr=0.10, window=10):
    out = []
    for i in range(arr_pad.shape[0]):
        ep_len = int(ep_lens[i])
        seg = arr_pad[i, min(dist_end, ep_len):ep_len]
        consec, rec = 0, float("nan")
        for j, v in enumerate(seg):
            if not np.isnan(v) and abs(v) < thr:
                consec += 1
                if consec >= window:
                    rec = float(j - window + 1)
                    break
            else:
                consec = 0
        out.append(rec)
    arr = np.array(out, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan"), int(np.sum(np.isnan(arr)))


def summarise(data):
    dist_end = int(data["dist_step"]) + int(data["dist_duration"])
    dist_start = int(data["dist_step"])
    bl_rr, rl_rr = data["bl_roll_rates"], data["rl_roll_rates"]
    bl_lens, rl_lens = data["bl_ep_lens"], data["rl_ep_lens"]
    n = bl_rr.shape[0]
    bl_crashes = int(np.sum(data["bl_crashes"].astype(float)))
    rl_crashes = int(np.sum(data["rl_crashes"].astype(float)))

    bl_rms  = rms(bl_rr, bl_lens)
    rl_rms  = rms(rl_rr, rl_lens)
    bl_peak = peak_post(bl_rr, bl_lens, dist_start)
    rl_peak = peak_post(rl_rr, rl_lens, dist_start)
    bl_rec, bl_nan_rec = recovery(bl_rr, bl_lens, dist_end)
    rl_rec, rl_nan_rec = recovery(rl_rr, rl_lens, dist_end)

    peak_pct = (bl_peak - rl_peak) / (abs(bl_peak) + 1e-12) * 100.0
    rms_pct  = (bl_rms  - rl_rms)  / (abs(bl_rms)  + 1e-12) * 100.0
    rec_pct  = (bl_rec  - rl_rec)  / (abs(bl_rec)  + 1e-12) * 100.0 if not (np.isnan(bl_rec) or np.isnan(rl_rec)) else float("nan")

    winner = "RL" if (rl_crashes <= bl_crashes and peak_pct >= 10.0
                      and (rl_rms - bl_rms) / (abs(bl_rms) + 1e-12) * 100 <= 5.0) else \
             "BASE" if (bl_crashes < rl_crashes or
                        ((rl_rms - bl_rms) / (abs(bl_rms) + 1e-12) * 100 > 10.0 and bl_peak <= rl_peak)) else \
             "MIXED"

    return {
        "n": n, "bl_crash": bl_crashes, "rl_crash": rl_crashes,
        "bl_rms": bl_rms, "rl_rms": rl_rms, "rms_pct": rms_pct,
        "bl_peak": bl_peak, "rl_peak": rl_peak, "peak_pct": peak_pct,
        "bl_rec": bl_rec, "rl_rec": rl_rec, "rec_pct": rec_pct,
        "bl_nan_rec": bl_nan_rec, "rl_nan_rec": rl_nan_rec,
        "winner": winner,
    }


def fmt(v, prec=4):
    return f"{v:.{prec}f}" if not np.isnan(v) else "  nan  "


def fmt_pct(v):
    if np.isnan(v): return "   nan  "
    return f"{v:+6.1f}%"


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    total = len(INIT_NOISES) * len(DIST_MAGNITUDES) * len(DIST_AXES)
    print(f"\n{'='*80}")
    print(f"AGGRESSIVE EVALUATION — c860k model   [{total} conditions × {EPISODES} episodes]")
    print(f"Model: {MODEL}")
    print(f"{'='*80}\n")

    t_total = time.time()
    results = {}
    done, failed = 0, 0

    for noise in INIT_NOISES:
        for mag_label, mag_val in DIST_MAGNITUDES:
            for dist_axis in DIST_AXES:
                data, tag = run_condition(noise, mag_label, mag_val, dist_axis)
                done += 1
                if data is None:
                    failed += 1
                    results[tag] = None
                else:
                    results[tag] = summarise(data)
                    results[tag]["noise"] = noise
                    results[tag]["mag_label"] = mag_label
                    results[tag]["mag_val"] = mag_val
                    results[tag]["dist_axis"] = dist_axis

    elapsed_total = time.time() - t_total
    print(f"\n{'='*80}")
    print(f"All {total} conditions done in {elapsed_total/60:.1f} min  ({failed} failed)")
    print(f"{'='*80}\n")

    # ── Full table ────────────────────────────────────────────────────────────
    sep = "-" * 108
    header = (f"{'Condition':<38}  {'Crash':^10}  {'RMS Δ%':>8}  "
              f"{'Peak Δ%':>8}  {'Rec Δ%':>8}  "
              f"{'BL rec':>7}  {'RL rec':>7}  {'Winner':^7}")
    print(header)
    print(sep)

    rl_wins = 0; base_wins = 0; mixed = 0
    all_peak_pcts = []; all_rms_pcts = []; all_rec_pcts = []

    for tag, r in results.items():
        if r is None:
            print(f"  {tag:<36}  ERROR")
            continue
        crash_str = f"{r['bl_crash']}/{r['n']} vs {r['rl_crash']}/{r['n']}"
        rec_bl = f"{r['bl_rec']:.1f}" if not np.isnan(r['bl_rec']) else "  nan"
        rec_rl = f"{r['rl_rec']:.1f}" if not np.isnan(r['rl_rec']) else "  nan"
        print(f"  {tag:<36}  {crash_str:^10}  {fmt_pct(r['rms_pct']):>8}  "
              f"{fmt_pct(r['peak_pct']):>8}  {fmt_pct(r['rec_pct']):>8}  "
              f"{rec_bl:>7}  {rec_rl:>7}  {r['winner']:^7}")
        if r['winner'] == 'RL':   rl_wins += 1
        elif r['winner'] == 'BASE': base_wins += 1
        else: mixed += 1
        all_peak_pcts.append(r['peak_pct'])
        all_rms_pcts.append(r['rms_pct'])
        if not np.isnan(r['rec_pct']): all_rec_pcts.append(r['rec_pct'])

    print(sep)
    n_valid = rl_wins + base_wins + mixed
    print(f"\nSUMMARY  ({n_valid} valid conditions)")
    print(f"  RL wins : {rl_wins}   Baseline wins: {base_wins}   Mixed: {mixed}")
    if all_peak_pcts:
        print(f"  Mean peak improvement : {np.nanmean(all_peak_pcts):+.1f}%  "
              f"(positive = RL better)")
    if all_rms_pcts:
        print(f"  Mean RMS  improvement : {np.nanmean(all_rms_pcts):+.1f}%")
    if all_rec_pcts:
        print(f"  Mean recovery improve : {np.nanmean(all_rec_pcts):+.1f}%")

    # ── Per-axis sub-summary ──────────────────────────────────────────────────
    print(f"\n--- Per disturbance axis ---")
    for axis in DIST_AXES:
        sub = [r for tag, r in results.items()
               if r and r.get("dist_axis") == axis]
        if not sub: continue
        rls = sum(1 for r in sub if r['winner'] == 'RL')
        peak_m = np.nanmean([r['peak_pct'] for r in sub])
        rec_m  = np.nanmean([r['rec_pct']  for r in sub if not np.isnan(r['rec_pct'])])
        print(f"  {axis:6s}: RL wins {rls}/{len(sub)}  "
              f"peak Δ {peak_m:+.1f}%  rec Δ {rec_m:+.1f}%")

    # ── Per-magnitude sub-summary ─────────────────────────────────────────────
    print(f"\n--- Per disturbance magnitude ---")
    for mag_label, mag_val in DIST_MAGNITUDES:
        sub = [r for tag, r in results.items()
               if r and r.get("mag_label") == mag_label]
        if not sub: continue
        rls = sum(1 for r in sub if r['winner'] == 'RL')
        peak_m = np.nanmean([r['peak_pct'] for r in sub])
        crash_total_rl = sum(r['rl_crash'] for r in sub)
        crash_total_bl = sum(r['bl_crash'] for r in sub)
        print(f"  {mag_label:8s} ({mag_val:.2f} N·m): RL wins {rls}/{len(sub)}  "
              f"peak Δ {peak_m:+.1f}%  crashes BL={crash_total_bl} RL={crash_total_rl}")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = os.path.join(OUT_ROOT, "aggressive_eval_summary.txt")
    import io, contextlib
    report_lines = []
    report_lines.append(f"AGGRESSIVE EVAL SUMMARY — c860k  [{total} conditions × {EPISODES} episodes]")
    report_lines.append(f"Model: {MODEL}")
    report_lines.append(f"Noise levels: {INIT_NOISES}")
    report_lines.append(f"Dist magnitudes: {[m for m,_ in DIST_MAGNITUDES]} N·m = {[v for _,v in DIST_MAGNITUDES]}")
    report_lines.append(f"RL wins: {rl_wins}  Baseline: {base_wins}  Mixed: {mixed}")
    if all_peak_pcts:
        report_lines.append(f"Mean peak improvement: {np.nanmean(all_peak_pcts):+.1f}%")
    if all_rms_pcts:
        report_lines.append(f"Mean RMS improvement: {np.nanmean(all_rms_pcts):+.1f}%")
    if all_rec_pcts:
        report_lines.append(f"Mean recovery improvement: {np.nanmean(all_rec_pcts):+.1f}%")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n\n")
        f.write(header + "\n" + sep + "\n")
        for tag, r in results.items():
            if r is None:
                f.write(f"  {tag:<36}  ERROR\n")
                continue
            crash_str = f"{r['bl_crash']}/{r['n']} vs {r['rl_crash']}/{r['n']}"
            rec_bl = f"{r['bl_rec']:.1f}" if not np.isnan(r['bl_rec']) else "  nan"
            rec_rl = f"{r['rl_rec']:.1f}" if not np.isnan(r['rl_rec']) else "  nan"
            f.write(f"  {tag:<36}  {crash_str:^10}  {fmt_pct(r['rms_pct']):>8}  "
                    f"{fmt_pct(r['peak_pct']):>8}  {fmt_pct(r['rec_pct']):>8}  "
                    f"{rec_bl:>7}  {rec_rl:>7}  {r['winner']:^7}\n")

    print(f"\n[EVAL] Report → {report_path}")
    print(f"[EVAL] Per-condition NPZ data → {OUT_ROOT}/")


if __name__ == "__main__":
    main()
