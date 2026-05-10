"""
Test 3 — Short TD3 training run (PyBullet backend)
====================================================
Purpose:
    Run TD3 for a small number of timesteps (default 10 000) to verify the
    training loop is healthy:
    - Learning starts after learning_starts steps
    - ep_len_mean grows (longer episodes = agent learning to not crash)
    - ep_rew_mean improves over time
    - Model and replay buffer are saved to tests/results/short_td3/

Usage:
    cd ~/rl_pid_tuner && python tests/short_td3.py [--steps N]
"""

import sys, os, argparse, signal, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from envs import PyBulletPIDTunerEnv

RESULTS = os.path.join(os.path.dirname(__file__), "results", "short_td3")
os.makedirs(RESULTS, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10_000)
args = parser.parse_args()

TOTAL_STEPS = args.steps


def make_env():
    env = PyBulletPIDTunerEnv(
        max_steps       = 500,
        target_alt      = 1.0,
        reward_w1       = 1.0,
        reward_w2       = 2.0,
        reward_w3       = 0.1,
        reward_w4       = 0.5,
        crash_penalty   = 50.0,
        stability_bonus = 200.0,
        init_noise      = 0.05,
    )
    return Monitor(env, RESULTS)


def main():
    print(f"[SHORT_TD3] Training for {TOTAL_STEPS} steps → {RESULTS}")

    env   = make_env()
    model = None

    def _shutdown(sig, frame):
        print("\n[SHORT_TD3] Interrupted — saving …")
        if model:
            model.save(os.path.join(RESULTS, "td3_short_interrupted"))
        env.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)

    n_actions    = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    model = TD3(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = 1e-3,
        buffer_size     = 20_000,
        batch_size      = 128,
        gamma           = 0.97,
        tau             = 0.005,
        policy_delay    = 2,
        action_noise    = action_noise,
        learning_starts = 500,
        train_freq      = (1, "step"),
        gradient_steps  = 1,
        policy_kwargs   = {"net_arch": [256, 256]},
        verbose         = 1,
        tensorboard_log = RESULTS,
        device          = "cpu",
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
    elapsed = time.time() - t0

    model.save(os.path.join(RESULTS, "td3_short_final"))
    print(f"\n[SHORT_TD3] Done in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"[SHORT_TD3] Model saved → {RESULTS}/td3_short_final.zip")
    env.close()


if __name__ == "__main__":
    main()
