"""
Evaluate a trained TD3 model and print the tuned PID gains.

Usage:
    ~/miniconda3/bin/python evaluate.py --model runs/<timestamp>/best_model/best_model.zip
"""

import argparse
import numpy as np
from stable_baselines3 import TD3
from envs import PyBulletPIDTunerEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .zip model file")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env   = PyBulletPIDTunerEnv(max_steps=500, init_noise=0.0)
    model = TD3.load(args.model, env=env, device="cpu")

    all_rewards = []
    all_kp, all_ki, all_kd = [], [], []

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
        all_kp.append(info["Kp"])
        all_ki.append(info["Ki"])
        all_kd.append(info["Kd"])
        print(f"\n[EP {ep+1}]  reward={ep_reward:.1f}  crashed={info['crashed']}")
        print(f"    Kp_roll = {info['Kp']:.6f}  (default {PyBulletPIDTunerEnv.KP_DEFAULT:.6f})")
        print(f"    Ki_roll = {info['Ki']:.6f}  (default {PyBulletPIDTunerEnv.KI_DEFAULT:.6f})")
        print(f"    Kd_roll = {info['Kd']:.6f}  (default {PyBulletPIDTunerEnv.KD_DEFAULT:.6f})")

    print("\n── Average across episodes ──────────────────────────────")
    print(f"Mean reward : {np.mean(all_rewards):.1f}")
    print(f"Mean Kp_roll: {np.mean(all_kp):.6f}  (default {PyBulletPIDTunerEnv.KP_DEFAULT:.6f})")
    print(f"Mean Ki_roll: {np.mean(all_ki):.6f}  (default {PyBulletPIDTunerEnv.KI_DEFAULT:.6f})")
    print(f"Mean Kd_roll: {np.mean(all_kd):.6f}  (default {PyBulletPIDTunerEnv.KD_DEFAULT:.6f})")

    env.close()


if __name__ == "__main__":
    main()
