"""
Test 4 — Compare baseline vs TD3
==================================
Purpose:
    Load the .npz files saved by baseline_fixed.py and the Monitor CSV from
    short_td3.py and print a side-by-side comparison of the key metrics.

Usage:
    cd ~/rl_pid_tuner && python tests/compare_logs.py

    Optionally run against a trained TD3 policy for a live evaluation:
        python tests/compare_logs.py --eval-model tests/results/short_td3/td3_short_final
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "results")

parser = argparse.ArgumentParser()
parser.add_argument("--eval-model", type=str, default=None,
                    help="Path to a saved TD3 .zip to run live evaluation")
parser.add_argument("--eval-episodes", type=int, default=3)
args = parser.parse_args()


# ── Helper ────────────────────────────────────────────────────────────────────

def print_block(label, rr, rew, ep_lengths, crashes):
    print(f"\n  {label}")
    print(f"    Episodes              : {len(ep_lengths)}")
    print(f"    Crashes               : {int(crashes.sum())}/{len(ep_lengths)}")
    print(f"    Mean episode length   : {ep_lengths.mean():.0f} steps  "
          f"(max={ep_lengths.max()}, min={ep_lengths.min()})")
    print(f"    roll_rate  |mean|     : {np.abs(rr).mean():.5f} rad/s")
    print(f"    roll_rate  std        : {rr.std():.5f} rad/s")
    print(f"    roll_rate² mean       : {(rr**2).mean():.6f}   ← lower is better")
    print(f"    reward     mean/step  : {rew.mean():+.4f}")
    print(f"    reward     std/step   : {rew.std():.4f}")


def improvement(baseline_val, td3_val, label, lower_is_better=True):
    delta = baseline_val - td3_val if lower_is_better else td3_val - baseline_val
    pct   = 100 * delta / (abs(baseline_val) + 1e-9)
    arrow = "↓" if lower_is_better else "↑"
    sign  = "+" if delta >= 0 else ""
    tag   = "BETTER" if delta >= 0 else "WORSE "
    print(f"  {tag}  {label:30s}  baseline={baseline_val:+.5f}  "
          f"td3={td3_val:+.5f}  {arrow}{sign}{pct:.1f}%")


# ── Load baseline ─────────────────────────────────────────────────────────────

baseline_path = os.path.join(RESULTS, "baseline_pybullet.npz")
if not os.path.exists(baseline_path):
    print(f"[ERROR] Baseline results not found: {baseline_path}")
    print("        Run tests/baseline_fixed.py first.")
    sys.exit(1)

b = np.load(baseline_path)
b_rr        = b["roll_rates"]
b_rew       = b["rewards"]
b_lengths   = b["ep_lengths"]
b_crashes   = b["ep_crashes"]

# ── Load TD3 Monitor CSV (training curve) ─────────────────────────────────────

td3_csv = os.path.join(RESULTS, "short_td3", "monitor.csv")
td3_eval_rr  = None
td3_eval_rew = None
td3_ep_len   = None
td3_crashes  = None

if os.path.exists(td3_csv):
    # SB3 Monitor CSV: r, l, t  (reward, length, wall-time)
    rows = []
    with open(td3_csv) as f:
        for i, line in enumerate(f):
            if i < 2:        # skip header lines
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), int(parts[1])))
                except ValueError:
                    pass
    if rows:
        ep_rewards, ep_lengths_td3 = zip(*rows)
        ep_rewards    = np.array(ep_rewards)
        ep_lengths_td3 = np.array(ep_lengths_td3)
        print(f"\n[TD3 training curve — {len(ep_rewards)} episodes from monitor.csv]")
        first_q = ep_rewards[: max(1, len(ep_rewards)//4)]
        last_q  = ep_rewards[-max(1, len(ep_rewards)//4):]
        print(f"  First quarter mean reward/ep : {first_q.mean():.2f}")
        print(f"  Last  quarter mean reward/ep : {last_q.mean():.2f}")
        first_l = ep_lengths_td3[: max(1, len(ep_lengths_td3)//4)]
        last_l  = ep_lengths_td3[-max(1, len(ep_lengths_td3)//4):]
        print(f"  First quarter mean ep length : {first_l.mean():.0f} steps")
        print(f"  Last  quarter mean ep length : {last_l.mean():.0f} steps")
        improving = last_q.mean() > first_q.mean()
        print(f"  Reward trend: {'IMPROVING ✓' if improving else 'NOT improving yet'}")
else:
    print(f"[INFO] TD3 monitor.csv not found at {td3_csv}")
    print("       Run tests/short_td3.py first.")

# ── Live evaluation of a saved TD3 model ─────────────────────────────────────

if args.eval_model:
    print(f"\n[LIVE EVAL] Loading model: {args.eval_model}")
    from stable_baselines3 import TD3 as _TD3
    from envs import PyBulletPIDTunerEnv

    eval_env = PyBulletPIDTunerEnv(max_steps=500, init_noise=0.0)
    model    = _TD3.load(args.eval_model, env=eval_env)

    eval_rr, eval_rew, eval_lengths, eval_crashes_list = [], [], [], []

    for ep in range(args.eval_episodes):
        print(f"  eval episode {ep+1}/{args.eval_episodes} …")
        obs, _ = eval_env.reset()
        ep_rr, ep_rew = [], []
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_rr.append(info["roll_rate"])
            ep_rew.append(reward)
            if terminated or truncated:
                eval_crashes_list.append(info["crashed"])
                eval_lengths.append(info["step"])
                break
        eval_rr.extend(ep_rr)
        eval_rew.extend(ep_rew)

    eval_env.close()

    td3_eval_rr  = np.array(eval_rr)
    td3_eval_rew = np.array(eval_rew)
    td3_ep_len   = np.array(eval_lengths)
    td3_crashes  = np.array(eval_crashes_list)

# ── Print comparison ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("COMPARISON: BASELINE vs TD3")
print("="*60)

print_block("Baseline (fixed default gains)", b_rr, b_rew, b_lengths, b_crashes)

if td3_eval_rr is not None:
    print_block(f"TD3 (deterministic, {args.eval_episodes} eps)",
                td3_eval_rr, td3_eval_rew, td3_ep_len, td3_crashes)

    print("\n  Delta (positive = TD3 better):")
    improvement(np.abs(b_rr).mean(),  np.abs(td3_eval_rr).mean(),  "|roll_rate| mean")
    improvement((b_rr**2).mean(), (td3_eval_rr**2).mean(),         "roll_rate² mean")
    improvement(b_rr.std(),       td3_eval_rr.std(),               "roll_rate std")
    improvement(b_rew.mean(), td3_eval_rew.mean(),
                "reward/step", lower_is_better=False)
    improvement(b_lengths.mean(), td3_ep_len.mean(),
                "mean ep length", lower_is_better=False)
else:
    print("\n  (Run with --eval-model <path> for a live TD3 vs baseline comparison)")

print()
