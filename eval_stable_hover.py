"""
eval_stable_hover.py — Decisive offline validation for the joint PID-tuner policy
=================================================================================
Three criteria, all computed offline from a trained TD3 .zip (no SITL needed):

  1. stable-hover mean|action|  — feed real captured SITL stable-hover obs
     (results/observer_*.csv) through the deterministic policy. Target < 0.15.
     A good policy outputs ~0 (no gain change) when the drone is already stable.

  2. step_prog sensitivity      — REMOVED. step_prog is no longer an observation
     (12-D obs). Invariance is now guaranteed by construction — there is no
     input to sweep. Only the 12 real-state columns (obs_00..obs_11) are read
     from the captured CSV; the legacy obs_12 column is ignored.

  3. (disturbance grid is run separately via run_disturbance_grid.py)

Usage:
    # sweep every checkpoint in a run + the final model
    python eval_stable_hover.py --run runs/2026-05-17_16-29-20

    # single model
    python eval_stable_hover.py --model runs/<ts>/td3_pid_final.zip
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from stable_baselines3 import TD3

OBS_COLS = [f"obs_{i:02d}" for i in range(12)]   # step_prog (old obs_12) removed
DEFAULT_CSV = "results/observer_2026-05-16_19-24-48.csv"


def load_obs(csv_path):
    df = pd.read_csv(csv_path)
    return df[OBS_COLS].to_numpy(dtype=np.float32)


def mean_abs_action(model, obs):
    act, _ = model.predict(obs, deterministic=True)
    return float(np.mean(np.abs(act))), np.abs(act).mean(axis=0)


def ckpt_step(path):
    m = re.search(r"_(\d+)_steps\.zip$", path)
    return int(m.group(1)) if m else (10**9 if "final" in path else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="runs/<ts> dir — sweep all checkpoints + final")
    ap.add_argument("--model", default=None, help="single .zip to evaluate")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="captured stable-hover obs CSV")
    ap.add_argument("--target", type=float, default=0.15)
    args = ap.parse_args()

    obs = load_obs(args.csv)
    print(f"[EVAL] Stable-hover obs: {obs.shape[0]} samples from {args.csv}")
    print(f"[EVAL] Target mean|action| < {args.target}\n")

    if args.model:
        models = [args.model]
    else:
        ck = sorted(glob.glob(os.path.join(args.run, "checkpoints", "*_steps.zip")),
                    key=ckpt_step)
        fin = os.path.join(args.run, "td3_pid_final.zip")
        models = ck + ([fin] if os.path.isfile(fin) else [])

    print(f"{'model':<34}{'step':>9}  {'mean|a|':>8}  {'|a0|':>6} {'|a1|':>6} {'|a2|':>6}  verdict")
    print("-" * 86)
    results = []
    for mp in models:
        model = TD3.load(mp, device="cpu")
        ma, per = mean_abs_action(model, obs)
        step = ckpt_step(mp)
        tag = "final" if step == 10**9 else f"{step}"
        verdict = "PASS" if ma < args.target else ("near" if ma < 0.30 else "fail")
        results.append((tag, step, ma, per, mp))
        print(f"{os.path.basename(mp):<34}{tag:>9}  {ma:>8.4f}  "
              f"{per[0]:>6.3f} {per[1]:>6.3f} {per[2]:>6.3f}  {verdict}")

    results.sort(key=lambda r: r[2])
    best = results[0]
    print("\n" + "=" * 86)
    print(f"BEST by stable-hover mean|action|:  {os.path.basename(best[4])}  "
          f"(step={best[0]})  mean|a|={best[2]:.4f}")
    print("=" * 86)
    print("[EVAL] step_prog sensitivity: N/A — removed from observation (12-D). "
          "Invariance guaranteed by construction.")


if __name__ == "__main__":
    main()
