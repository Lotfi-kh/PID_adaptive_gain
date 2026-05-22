"""
run_disturbance_grid.py, Disturbance-rejection evaluation grid

Runs eval_disturbance.py for all combinations of:
    init_noise     in {0.05, 0.10, 0.15}
    dist_magnitude in {4.3e-2, 1.7e-1}   (low / medium, scaled for F450 Ixx=0.012 kg.m²)

Each condition: 10 deterministic episodes, dist_step=150, dist_duration=5.
Results saved per-condition under --out-dir.
A summary table + conclusion is printed and saved to <out_dir>/grid_summary.txt.

Usage:
    # run from the repo root
    python run_disturbance_grid.py --model runs/<ts>/best_model/best_model.zip

    # Re-generate summary from existing NPZ files without re-running evaluation:
    python run_disturbance_grid.py --model runs/<ts>/best_model/best_model.zip --reuse-existing

Options:
    --model           Path to trained .zip model (required)
    --episodes        Episodes per condition (default 10)
    --dist-step       Step at which disturbance fires (default 150)
    --dist-duration   Disturbance duration in steps (default 5)
    --out-dir         Root directory for all grid output
    --no-plots        Skip matplotlib figures
    --seed            RNG seed (default 42)
    --reuse-existing  Skip eval if disturbance_eval_results.npz already exists
"""

import argparse
import os
import subprocess
import sys
import time
import numpy as np

INIT_NOISES     = [0.05, 0.10, 0.15]
DIST_MAGNITUDES = [
    ("low",    4.3e-2),   # 5e-5 x 857 (F450/CF2X Ixx ratio) -> same α as CF2X low
    ("medium", 1.7e-1),   # 2e-4 x 857
]

parser = argparse.ArgumentParser()
parser.add_argument("--model",          required=True)
parser.add_argument("--axis",           choices=["roll", "pitch", "roll+pitch"], default="roll",
                    help="Which axis to tune: 'roll', 'pitch', or 'roll+pitch' (joint). Default: roll")
parser.add_argument("--dist-axis",      choices=["roll", "pitch", "both"], default=None,
                    help="Disturbance axis passed to eval_disturbance.py. "
                         "Default: matches --axis for single-axis, 'both' for roll+pitch.")
parser.add_argument("--episodes",       type=int,  default=10)
parser.add_argument("--dist-step",      type=int,  default=150)
parser.add_argument("--dist-duration",  type=int,  default=5)
parser.add_argument("--out-dir",        default=None)
parser.add_argument("--no-plots",       action="store_true")
parser.add_argument("--seed",           type=int,  default=42)
parser.add_argument("--reuse-existing", action="store_true",
                    help="Skip running eval_disturbance.py if NPZ already exists")
args = parser.parse_args()

AXIS      = args.axis
DIST_AXIS = args.dist_axis if args.dist_axis is not None else ("both" if AXIS == "roll+pitch" else AXIS)

MODEL_PATH = os.path.abspath(args.model)
ROOT_DIR   = args.out_dir or os.path.join(os.path.dirname(MODEL_PATH), "disturbance_grid")
os.makedirs(ROOT_DIR, exist_ok=True)

SCRIPT = os.path.join(os.path.dirname(__file__), "eval_disturbance.py")
PYTHON = sys.executable


def condition_tag(noise, mag_label):
    return f"noise{noise:.2f}_{mag_label}"


def run_condition(noise, mag_label, mag_val):
    tag     = condition_tag(noise, mag_label)
    out_dir = os.path.join(ROOT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "disturbance_eval_results.npz")

    if args.reuse_existing and os.path.exists(npz_path):
        print(f"\n[GRID] {tag}  (reusing existing NPZ)")
    else:
        cmd = [
            PYTHON, SCRIPT,
            "--model",          MODEL_PATH,
            "--axis",           AXIS,
            "--dist-axis",      DIST_AXIS,
            "--episodes",       str(args.episodes),
            "--init-noise",     str(noise),
            "--dist-step",      str(args.dist_step),
            "--dist-magnitude", str(mag_val),
            "--dist-duration",  str(args.dist_duration),
            "--out-dir",        out_dir,
            "--seed",           str(args.seed),
        ]
        if args.no_plots:
            cmd.append("--no-plots")

        print(f"\n[GRID] {tag}  noise={noise}  mag={mag_val:.1e} ({mag_label})")
        print(f"[GRID] Output -> {out_dir}")
        t0 = time.time()
        result = subprocess.run(cmd, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"[GRID] ERROR in condition {tag} (exit {result.returncode})")
            return None
        print(f"[GRID] Done in {elapsed:.0f}s")

    if not os.path.exists(npz_path):
        print(f"[GRID] WARNING: no npz found at {npz_path}")
        return None

    return np.load(npz_path, allow_pickle=True)


RECOVERY_THRESHOLD = 0.10   # rad/s
RECOVERY_WINDOW    = 10     # consecutive steps


def _recovery_steps_per_episode(rr_pad, ep_lens, dist_end):
    """Return steps from dist_end until recovery (or NaN) for each episode."""
    results = []
    for i in range(rr_pad.shape[0]):
        ep_len = int(ep_lens[i])
        start  = min(dist_end, ep_len)
        rr     = rr_pad[i, start:ep_len]
        rec    = float("nan")
        consec = 0
        for j, v in enumerate(rr):
            if not np.isnan(v) and abs(v) < RECOVERY_THRESHOLD:
                consec += 1
                if consec >= RECOVERY_WINDOW:
                    rec = float(j - RECOVERY_WINDOW + 1)
                    break
            else:
                consec = 0
        results.append(rec)
    return results


def _rms(rr_pad, ep_lens):
    vals = []
    for i in range(rr_pad.shape[0]):
        ep = rr_pad[i, :int(ep_lens[i])]
        ep = ep[~np.isnan(ep)]
        vals.extend(ep.tolist())
    sq = np.array(vals) ** 2
    return float(np.sqrt(sq.mean())) if sq.size else float("nan")


def _peak_post_dist(rr_pad, ep_lens, dist_start):
    """Mean over episodes of max |roll_rate| from dist_start to episode end."""
    peaks = []
    for i in range(rr_pad.shape[0]):
        ep_end = int(ep_lens[i])
        if ep_end > dist_start:
            window = rr_pad[i, dist_start:ep_end]
            window = window[~np.isnan(window)]
            if window.size > 0:
                peaks.append(float(np.max(np.abs(window))))
    return float(np.mean(peaks)) if peaks else float("nan")


def _max_roll_mean(rolls_pad, ep_lens):
    """Mean over episodes of max |roll angle| (rad)."""
    maxes = []
    for i in range(rolls_pad.shape[0]):
        ep_end = int(ep_lens[i])
        ep = rolls_pad[i, :ep_end]
        ep = ep[~np.isnan(ep)]
        if ep.size > 0:
            maxes.append(float(np.max(np.abs(ep))))
    return float(np.mean(maxes)) if maxes else float("nan")


def _mean_rew(rew_pad, ep_lens):
    vals = []
    for i in range(rew_pad.shape[0]):
        ep = rew_pad[i, :int(ep_lens[i])]
        ep = ep[~np.isnan(ep)]
        vals.extend(ep.tolist())
    return float(np.mean(vals)) if vals else float("nan")


def _pct_improvement(bl, rl, lower_better=True):
    """Positive = RL better."""
    d = (bl - rl) if lower_better else (rl - bl)
    return 100.0 * d / (abs(bl) + 1e-12)


def _nanmean(lst):
    arr = np.array(lst, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


def summarise(data, label):
    dist_step     = int(data["dist_step"])
    dist_duration = int(data["dist_duration"])
    dist_end      = dist_step + dist_duration

    bl_rr    = data["bl_roll_rates"]
    rl_rr    = data["rl_roll_rates"]
    bl_rolls = data["bl_rolls"]
    rl_rolls = data["rl_rolls"]
    bl_rew   = data["bl_rewards"]
    rl_rew   = data["rl_rewards"]
    bl_lens  = data["bl_ep_lens"]
    rl_lens  = data["rl_ep_lens"]

    n_ep          = bl_rr.shape[0]
    bl_crash_cnt  = int(np.sum(data["bl_crashes"].astype(float)))
    rl_crash_cnt  = int(np.sum(data["rl_crashes"].astype(float)))
    bl_rms_rr     = _rms(bl_rr, bl_lens)
    rl_rms_rr     = _rms(rl_rr, rl_lens)
    bl_peak_rr    = _peak_post_dist(bl_rr, bl_lens, dist_step)
    rl_peak_rr    = _peak_post_dist(rl_rr, rl_lens, dist_step)
    bl_max_roll   = _max_roll_mean(bl_rolls, bl_lens)
    rl_max_roll   = _max_roll_mean(rl_rolls, rl_lens)
    bl_mean_rew   = _mean_rew(bl_rew, bl_lens)
    rl_mean_rew   = _mean_rew(rl_rew, rl_lens)
    bl_rec_list   = _recovery_steps_per_episode(bl_rr, bl_lens, dist_end)
    rl_rec_list   = _recovery_steps_per_episode(rl_rr, rl_lens, dist_end)

    return {
        "label":        label,
        "n_ep":         n_ep,
        "bl_crashes":   bl_crash_cnt,
        "rl_crashes":   rl_crash_cnt,
        "bl_rms_rr":    bl_rms_rr,
        "rl_rms_rr":    rl_rms_rr,
        "rms_pct":      _pct_improvement(bl_rms_rr,   rl_rms_rr),
        "bl_peak_rr":   bl_peak_rr,
        "rl_peak_rr":   rl_peak_rr,
        "peak_pct":     _pct_improvement(bl_peak_rr,  rl_peak_rr),
        "bl_max_roll":  bl_max_roll,
        "rl_max_roll":  rl_max_roll,
        "roll_pct":     _pct_improvement(bl_max_roll, rl_max_roll),
        "bl_mean_rew":  bl_mean_rew,
        "rl_mean_rew":  rl_mean_rew,
        "rew_pct":      _pct_improvement(bl_mean_rew, rl_mean_rew, lower_better=False),
        "bl_recovery":  _nanmean(bl_rec_list),
        "rl_recovery":  _nanmean(rl_rec_list),
    }


def determine_winner(r):
    """
    RL wins if ALL of:
      a) rl_crashes <= bl_crashes
      b) rl peak |roll_rate| lower by >= 10%
      c) rl RMS roll_rate not worse by > 5%

    Baseline wins if EITHER of:
      a) bl_crashes < rl_crashes
      b) baseline RMS better by > 10% AND baseline peak not worse (bl_peak <= rl_peak)

    Otherwise: MIXED
    """
    bl_crash = r["bl_crashes"]
    rl_crash = r["rl_crashes"]
    bl_rms   = r["bl_rms_rr"]
    rl_rms   = r["rl_rms_rr"]
    bl_peak  = r["bl_peak_rr"]
    rl_peak  = r["rl_peak_rr"]

    # RL conditions
    rl_crash_ok  = rl_crash <= bl_crash
    peak_impr_pct = (bl_peak - rl_peak) / (abs(bl_peak) + 1e-12) * 100.0
    rl_peak_ok   = peak_impr_pct >= 10.0
    rms_degraded = (rl_rms - bl_rms) / (abs(bl_rms) + 1e-12) * 100.0
    rms_ok       = rms_degraded <= 5.0

    if rl_crash_ok and rl_peak_ok and rms_ok:
        return "RL"

    # Baseline conditions
    bl_crash_better  = bl_crash < rl_crash
    rms_bl_impr_pct  = (rl_rms - bl_rms) / (abs(bl_rms) + 1e-12) * 100.0
    bl_rms_better_10 = rms_bl_impr_pct > 10.0
    bl_peak_ok       = bl_peak <= rl_peak   # baseline peak not worse

    if bl_crash_better or (bl_rms_better_10 and bl_peak_ok):
        return "BASELINE"

    return "MIXED"


print(f"[GRID] Model      : {MODEL_PATH}")
print(f"[GRID] Output root: {ROOT_DIR}")
print(f"[GRID] Conditions : {len(INIT_NOISES)} noise x {len(DIST_MAGNITUDES)} dist "
      f"= {len(INIT_NOISES)*len(DIST_MAGNITUDES)} runs x {args.episodes} episodes")

rows = []
for noise in INIT_NOISES:
    for mag_label, mag_val in DIST_MAGNITUDES:
        tag  = condition_tag(noise, mag_label)
        data = run_condition(noise, mag_label, mag_val)
        if data is not None:
            r = summarise(data, tag)
            r["winner"] = determine_winner(r)
            rows.append(r)
        else:
            rows.append({"label": tag, "error": True})


def fmt_pct(v):
    if np.isnan(v):
        return "   nan  "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:5.1f}%"


def fmt_rec(v):
    return f"{v:5.1f}" if not np.isnan(v) else "  nan"


SEP  = "=" * 118
SEP2 = "-" * 118

lines = [
    SEP,
    f"DISTURBANCE-REJECTION GRID SUMMARY  [axis={AXIS}]",
    f"Model     : {MODEL_PATH}",
    f"Conditions: init_noise in {INIT_NOISES}  |  "
    f"dist_magnitudes in {[m for _, m in DIST_MAGNITUDES]}",
    f"Per-cond  : {args.episodes} episodes, dist_step={args.dist_step}, "
    f"dist_duration={args.dist_duration}",
    SEP,
]

_rms_hdr  = f"RMS {AXIS}_rate (rad/s)"
_peak_hdr = f"Peak |{AXIS}_rate| (rad/s)"
_max_hdr  = f"Max{AXIS.capitalize()}"


lines += [
    "",
    f"{'Condition':<22}  {'Crashes':^9}  "
    f"{_rms_hdr:^26}  "
    f"{_peak_hdr:^26}  "
    f"{_max_hdr:^8}  {'Rew/step':^8}  "
    f"{'Recovery (steps)':^16}  {'Winner':^8}",

    f"{'':22}  {'BL':>4} {'RL':>4}  "
    f"{'BL':>7} {'RL':>7} {'Δ%':>8}  "
    f"{'BL':>7} {'RL':>7} {'Δ%':>8}  "
    f"{'Δ%':>8}  {'Δ%':>8}  "
    f"{'BL':>7} {'RL':>7}  {'':8}",
    SEP2,
]

for r in rows:
    if r.get("error"):
        lines.append(f"{r['label']:<22}  ERROR")
        continue

    crash_str = f"{r['bl_crashes']:>4}/{r['n_ep']}  {r['rl_crashes']:>4}/{r['n_ep']}"
    lines.append(
        f"{r['label']:<22}  {crash_str}  "
        f"{r['bl_rms_rr']:>7.4f} {r['rl_rms_rr']:>7.4f} {fmt_pct(r['rms_pct']):>8}  "
        f"{r['bl_peak_rr']:>7.4f} {r['rl_peak_rr']:>7.4f} {fmt_pct(r['peak_pct']):>8}  "
        f"{fmt_pct(r['roll_pct']):>8}  {fmt_pct(r['rew_pct']):>8}  "
        f"{fmt_rec(r['bl_recovery']):>7} {fmt_rec(r['rl_recovery']):>7}  "
        f"{r['winner']:^8}"
    )

lines += [
    SEP2,
    "  Δ% = (RL − Baseline) improvement from Baseline's perspective:",
    "  positive Δ% = RL better  |  negative Δ% = Baseline better",
    "  Peak rr: max |roll_rate| from dist_step to episode end (captures full response)",
    "  Recovery: steps after disturbance until |rr| < 0.1 rad/s for 10 consecutive steps",
    "",
    "  Winner criteria --",
    "    RL      : crashes(RL) <= crashes(BL)  AND  peak improvement >= 10%  AND  RMS not worse by > 5%",
    "    BASELINE: crashes(BL) < crashes(RL)  OR   RMS(BL) better by > 10% AND peak(BL) not worse",
    "    MIXED   : neither of the above",
]


rl_wins   = [r["label"] for r in rows if not r.get("error") and r["winner"] == "RL"]
bl_wins   = [r["label"] for r in rows if not r.get("error") and r["winner"] == "BASELINE"]
mixed     = [r["label"] for r in rows if not r.get("error") and r["winner"] == "MIXED"]

valid = [r for r in rows if not r.get("error")]
peak_improvements = [r["peak_pct"] for r in valid if not np.isnan(r["peak_pct"])]
rms_improvements  = [r["rms_pct"]  for r in valid if not np.isnan(r["rms_pct"])]
roll_improvements = [r["roll_pct"] for r in valid if not np.isnan(r["roll_pct"])]

lines += ["", SEP, f"OVERALL CONCLUSION  [axis={AXIS}]", SEP]

def bullet_list(label, items):
    if items:
        return f"  {label} ({len(items)}): {', '.join(items)}"
    return f"  {label} (0): --"

lines.append(bullet_list("RL wins    ", rl_wins))
lines.append(bullet_list("BASELINE   ", bl_wins))
lines.append(bullet_list("MIXED      ", mixed))
lines.append("")

# Peak rejection verdict
if peak_improvements:
    mean_peak_impr = float(np.mean(peak_improvements))
    n_positive     = sum(1 for v in peak_improvements if v > 0)
    lines.append(
        f"  Peak |{AXIS}_rate| improvement: RL {'reduces' if mean_peak_impr > 0 else 'increases'} "
        f"peak disturbance response by {abs(mean_peak_impr):.1f}% on average "
        f"({n_positive}/{len(peak_improvements)} conditions positive)."
    )
else:
    lines.append(f"  Peak |{AXIS}_rate| improvement: insufficient data.")

# RMS verdict
if rms_improvements:
    mean_rms_impr = float(np.mean(rms_improvements))
    lines.append(
        f"  RMS {AXIS}_rate improvement   : RL {'better' if mean_rms_impr > 0 else 'worse'} "
        f"by {abs(mean_rms_impr):.1f}% on average across all conditions."
    )

# Max angle verdict
if roll_improvements:
    mean_roll_impr = float(np.mean(roll_improvements))
    lines.append(
        f"  Max |{AXIS}| improvement      : RL {'better' if mean_roll_impr > 0 else 'worse'} "
        f"by {abs(mean_roll_impr):.1f}% on average across all conditions."
    )

lines.append("")

# Actionable recommendation
if len(rl_wins) > len(bl_wins) + len(mixed):
    lines.append("  VERDICT: RL clearly wins. Disturbance-randomized training is working.")
elif len(bl_wins) == 0 and len(rl_wins) > 0:
    lines.append(
        "  VERDICT: RL wins in some conditions with no baseline regressions. "
        "Consider extending training (more steps or wider noise range) to cover MIXED conditions."
    )
elif len(bl_wins) > 0:
    lines.append(
        "  VERDICT: Baseline wins in some conditions, likely at high noise where RL RMS degrades. "
        "Consider widening init_noise_range or extending training timesteps."
    )
else:
    lines.append(
        "  VERDICT: All conditions MIXED. RL peak rejection is partially working but "
        "not yet consistent enough. Consider more training steps or higher dist_mag_max."
    )

lines.append(SEP)

report = "\n".join(lines)
print("\n" + report)


summary_path = os.path.join(ROOT_DIR, "grid_summary.txt")
with open(summary_path, "w") as f:
    f.write(report + "\n")

print(f"\n[GRID] Grid summary saved -> {summary_path}")
print(f"[GRID] Per-condition NPZs and plots in subdirs of: {ROOT_DIR}")
