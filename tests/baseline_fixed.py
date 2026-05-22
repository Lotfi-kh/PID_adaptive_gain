"""
Test 2 — Fixed-gain baseline (PyBullet)
=========================================
Purpose:
    Measure drone stability with the default roll-rate gains and
    zero agent action (gains never change). This is the reference performance
    that TD3 must beat.

Metrics recorded per step:
    - roll_rate (rad/s)
    - roll (rad)
    - pitch (rad)
    - reward (from env reward function, so comparable to TD3 training signal)

Summary saved to: tests/results/baseline_pybullet.npz

Usage:
    cd ~/rl_pid_tuner && python tests/baseline_fixed.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from envs import PyBulletPIDTunerEnv

EPISODES  = 3
MAX_STEPS = 500   # full episode
RESULTS   = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

ZERO_ACTION = np.zeros(3, dtype=np.float32)   # no gain change


def run():
    env = PyBulletPIDTunerEnv(max_steps=MAX_STEPS, init_noise=0.0)

    all_roll_rates = []
    all_rolls      = []
    all_pitches    = []
    all_rewards    = []
    ep_lengths     = []
    ep_crashes     = []

    for ep in range(EPISODES):
        print(f"\nEpisode {ep+1}/{EPISODES}")
        obs, _ = env.reset()

        ep_roll_rates, ep_rolls, ep_pitches, ep_rewards = [], [], [], []

        for t in range(MAX_STEPS):
            obs, reward, terminated, truncated, info = env.step(ZERO_ACTION)

            ep_roll_rates.append(info["roll_rate"])
            ep_rolls.append(np.deg2rad(info["roll_deg"]))
            ep_pitches.append(np.deg2rad(info["pitch_deg"]))
            ep_rewards.append(reward)

            if (t + 1) % 50 == 0:
                print(f"  step={t+1:3d}  roll_rate={info['roll_rate']:+.4f}  "
                      f"roll={info['roll_deg']:+.2f}°  "
                      f"reward={reward:+.4f}  alt={info['alt_m']:.2f}m")

            if terminated or truncated:
                reason = "CRASH" if info["crashed"] else "SURVIVED"
                print(f"  → {reason} at step {t+1}")
                ep_crashes.append(info["crashed"])
                ep_lengths.append(t + 1)
                break
        else:
            ep_crashes.append(False)
            ep_lengths.append(MAX_STEPS)

        all_roll_rates.extend(ep_roll_rates)
        all_rolls.extend(ep_rolls)
        all_pitches.extend(ep_pitches)
        all_rewards.extend(ep_rewards)

    env.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    rr  = np.array(all_roll_rates)
    ro  = np.array(all_rolls)
    pi  = np.array(all_pitches)
    rew = np.array(all_rewards)

    print("\n" + "="*55)
    print("BASELINE (fixed default gains, PyBullet) — SUMMARY")
    print("="*55)
    print(f"Episodes run      : {EPISODES}")
    print(f"Crashes           : {sum(ep_crashes)}/{EPISODES}")
    print(f"Mean ep length    : {np.mean(ep_lengths):.0f} steps")
    print(f"roll_rate  mean   : {rr.mean():+.5f} rad/s")
    print(f"roll_rate  std    : {rr.std():.5f} rad/s")
    print(f"roll_rate  |mean| : {np.abs(rr).mean():.5f} rad/s")
    print(f"roll_rate² mean   : {(rr**2).mean():.6f}")
    print(f"roll       std    : {ro.std():.5f} rad")
    print(f"pitch      std    : {pi.std():.5f} rad")
    print(f"reward     mean   : {rew.mean():+.5f}")
    print(f"reward     std    : {rew.std():.5f}")

    out = os.path.join(RESULTS, "baseline_pybullet.npz")
    np.savez(out,
             roll_rates=rr, rolls=ro, pitches=pi, rewards=rew,
             ep_lengths=np.array(ep_lengths),
             ep_crashes=np.array(ep_crashes))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    run()
