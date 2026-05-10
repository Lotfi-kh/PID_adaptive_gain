"""
Evaluate a trained TD3 model and print the tuned PID gains.

Usage:
    ~/miniconda3/bin/python evaluate.py --model runs/<timestamp>/best_model/best_model.zip
"""

import argparse
import numpy as np
from stable_baselines3 import TD3
from envs import PX4GainTunerEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .zip model file")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env   = PX4GainTunerEnv(max_steps=500, init_noise=0.0)
    model = TD3.load(args.model, env=env, device="cpu")

    all_rewards = []
    all_gains   = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        all_rewards.append(ep_reward)
        all_gains.append(dict(info["gains"]))
        print(f"\n[EP {ep+1}]  reward={ep_reward:.1f}  "
              f"crashed={info['crashed']}")
        for k, v in info["gains"].items():
            print(f"    {k:20s} = {v:.5f}")

    print("\n── Average across episodes ──────────────────────────────")
    print(f"Mean reward: {np.mean(all_rewards):.1f}")
    print("\nMean tuned gains:")
    for key in PX4GainTunerEnv.DEFAULT_GAINS:
        vals = [g[key] for g in all_gains]
        print(f"  {key:20s} = {np.mean(vals):.5f}  "
              f"(default {PX4GainTunerEnv.DEFAULT_GAINS[key]:.5f})")

    env.close()


if __name__ == "__main__":
    main()
