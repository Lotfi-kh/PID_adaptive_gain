#!/usr/bin/env python3
"""Analyze an observer CSV — saturation rates, obs ranges, training-distribution checks."""
import sys
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else None
if CSV is None:
    sys.exit("usage: analyze_observer.py <csv>")

df = pd.read_csv(CSV)
N = len(df)
print(f"Rows: {N}\n")

# ── Detect schema version (new vs legacy) ───────────────────────────────────
HAS_K = "k_roll_raw" in df.columns
HAS_EFF = "kp_eff_roll" in df.columns
HAS_PERCHAN = "obs06_in_range" in df.columns
HAS_RATES_SANE = "flag_rates_sane" in df.columns

# 1) Saturation
print("=" * 70)
print("1) Action saturation")
print("=" * 70)
for i in range(3):
    col = f"action_{i}"
    a = df[col].to_numpy()
    pct_neg1 = 100.0 * np.mean(a <= -1.0 + 1e-6)
    pct_pos1 = 100.0 * np.mean(a >=  1.0 - 1e-6)
    pct_intr = 100.0 - pct_neg1 - pct_pos1
    print(f"  {col}:  =-1: {pct_neg1:6.2f}%   =+1: {pct_pos1:6.2f}%   interior: {pct_intr:6.2f}%")
print()

# 2) obs[0..12] stats
print("=" * 70)
print("2) obs min / mean / max")
print("=" * 70)
names = ["roll", "pitch", "rollspd", "pitchspd",
         "roll_rate_err", "pitch_rate_err",
         "Kp_roll_n", "Ki_roll_n", "Kd_roll_n",
         "Kp_pitch_n", "Ki_pitch_n", "Kd_pitch_n",
         "step_prog"]
for i in range(13):
    col = f"obs_{i:02d}"
    a = df[col].to_numpy()
    print(f"  obs_{i:02d} {names[i]:<16s}  min={a.min():+10.4f}  mean={a.mean():+10.4f}  max={a.max():+10.4f}")
print()

# 3) Normalized gain channels obs[6:12]
print("=" * 70)
print("3) Normalized gain channels — should be in [0, 1] for in-distribution")
print("=" * 70)
for i in range(6, 12):
    col = f"obs_{i:02d}"
    a = df[col].to_numpy()
    out_of_range = 100.0 * np.mean((a < 0.0) | (a > 1.0))
    flag = "  <-- OUT OF TRAINING RANGE" if out_of_range > 0 else ""
    print(f"  obs_{i:02d} {names[i]:<14s}  min={a.min():+8.4f}  max={a.max():+8.4f}  "
          f"oor={out_of_range:5.1f}%{flag}")
print()

# 4) Compare obs ranges vs training-time test-vector envelopes
print("=" * 70)
print("4) Comparison vs the 10 standalone test vectors")
print("=" * 70)
tv_envelopes = {
    0: (-0.40, 0.50),
    1: (-0.40, 0.15),
    2: (-4.00, 5.00),
    3: (-4.00, 5.00),
    4: (-6.50, 5.20),
    5: (-1.25, 5.20),
    6: (0.00290698, 0.988372),
    7: (0.00581395, 0.976744),
    8: (0.00581395, 0.976744),
    9: (0.00290698, 0.988372),
    10: (0.00581395, 0.976744),
    11: (0.00581395, 0.976744),
    12: (0.0, 1.0),
}
for i in range(13):
    col = f"obs_{i:02d}"
    a = df[col].to_numpy()
    lo, hi = tv_envelopes[i]
    pct_under = 100.0 * np.mean(a < lo)
    pct_over  = 100.0 * np.mean(a > hi)
    flag = ""
    if pct_under + pct_over > 0:
        flag = f"  TV-envelope [{lo:.4g}, {hi:.4g}]; over={pct_over:.1f}% under={pct_under:.1f}%"
    print(f"  obs_{i:02d} {names[i]:<16s}  observed=[{a.min():+8.4f},{a.max():+8.4f}]{flag}")
print()

# 5) Sanity flags
print("=" * 70)
print("5) Sanity flag counts")
print("=" * 70)
agg_flags = ["flag_rates_sane", "flag_obs_finite", "flag_action_finite",
             "flag_action_in_range", "flag_gains_match", "flag_gains_in_bounds"]
for flag in agg_flags:
    n_ones = int((df[flag] == 1).sum())
    print(f"  {flag:25s}  ones: {n_ones:4d} / {N}   ({100.0*n_ones/N:.1f}%)")
print()

# 5b) Per-channel in-range flags (new schema only)
if HAS_PERCHAN:
    print("=" * 70)
    print("5b) Per-channel obs[6:12] in-range flags (1 = in [0,1])")
    print("=" * 70)
    chan_names = [
        "Kp_roll_n  obs[06]", "Ki_roll_n  obs[07]", "Kd_roll_n  obs[08]",
        "Kp_pitch_n obs[09]", "Ki_pitch_n obs[10]", "Kd_pitch_n obs[11]",
    ]
    for i in range(6, 12):
        col = f"obs{i:02d}_in_range"
        n_ones = int((df[col] == 1).sum())
        status = "OK" if n_ones == N else "OOD"
        print(f"  {col:18s}  {chan_names[i-6]}  ones: {n_ones:4d} / {N}   ({100.0*n_ones/N:.1f}%)  [{status}]")
    print()

# 6) Raw PX4 gains seen
print("=" * 70)
print("6) PX4 raw gains (constant throughout run)")
print("=" * 70)
raw_cols = ["kp_roll_raw", "ki_roll_raw", "kd_roll_raw",
            "kp_pitch_raw", "ki_pitch_raw", "kd_pitch_raw"]
if HAS_K:
    raw_cols = ["kp_roll_raw", "ki_roll_raw", "kd_roll_raw", "k_roll_raw",
                "kp_pitch_raw", "ki_pitch_raw", "kd_pitch_raw", "k_pitch_raw"]
for col in raw_cols:
    a = df[col].to_numpy()
    print(f"  {col:18s}  unique={np.unique(a)}")
print()

# 7) Effective gains (new schema only)
if HAS_EFF:
    print("=" * 70)
    print("7) Effective gains K*P/I/D (constant throughout run)")
    print("=" * 70)
    for col in ["kp_eff_roll", "ki_eff_roll", "kd_eff_roll",
                "kp_eff_pitch", "ki_eff_pitch", "kd_eff_pitch"]:
        a = df[col].to_numpy()
        print(f"  {col:18s}  unique={np.unique(a)}")
    print()
