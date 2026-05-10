"""
Training script — TD3 on PyBulletPIDTunerEnv (default) or PX4GainTunerEnv
============================================================================
Usage:
    cd ~/rl_pid_tuner && python train.py [--env pybullet|px4] [--steps N]

Checkpoints and logs are saved to ./runs/<timestamp>/
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from envs import PX4GainTunerEnv, PyBulletPIDTunerEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env",   choices=["pybullet", "px4"], default="pybullet")
parser.add_argument("--steps", type=int, default=None,
                    help="Override total_timesteps")
args = parser.parse_args()

# ── Run directory ──────────────────────────────────────────────────────────────
RUN_DIR = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(RUN_DIR, exist_ok=True)
LOG_DIR  = os.path.join(RUN_DIR, "logs")
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
EVAL_DIR = os.path.join(RUN_DIR, "eval")
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
HP = dict(
    total_timesteps   = 1_000_000,
    learning_rate     = 1e-3,
    buffer_size       = 500_000,
    batch_size        = 128,
    gamma             = 0.97,
    tau               = 0.005,
    policy_delay      = 2,
    action_noise_std  = 0.1,
    net_arch          = [256, 256],
    learning_starts   = 1_000,
    train_freq        = (1, "step"),
    gradient_steps    = 1,
)
if args.steps:
    HP["total_timesteps"] = args.steps


def make_env(eval_mode: bool = False):
    if args.env == "pybullet":
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
    else:
        env = PX4GainTunerEnv(
            step_duration   = 0.1,
            max_steps       = 500,
            takeoff_alt     = 5.0,
            reward_w1       = 1.0,
            reward_w2       = 2.0,
            reward_w3       = 0.1,
            reward_w4       = 0.5,
            crash_penalty   = 50.0,
            stability_bonus = 200.0,
            init_noise      = 0.05,
        )
    return Monitor(env, LOG_DIR)


def main():
    print(f"[TRAIN] Run directory: {RUN_DIR}")

    model = None
    env   = None

    # Graceful shutdown on Ctrl+C (prevents pymavlink segfault)
    def _shutdown(sig, frame):
        print("\n[TRAIN] Interrupted — saving and closing …")
        try:
            if model is not None:
                model.save(os.path.join(RUN_DIR, "td3_pid_interrupted"))
            if env is not None:
                env.close()
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)

    # ── Environment ────────────────────────────────────────────────────────────
    env      = make_env()
    eval_env = make_env(eval_mode=True)

    # ── Action noise (TD3 exploration) ─────────────────────────────────────────
    n_actions    = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean  = np.zeros(n_actions),
        sigma = HP["action_noise_std"] * np.ones(n_actions),
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = TD3(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = HP["learning_rate"],
        buffer_size   = HP["buffer_size"],
        batch_size    = HP["batch_size"],
        gamma         = HP["gamma"],
        tau           = HP["tau"],
        policy_delay  = HP["policy_delay"],
        action_noise  = action_noise,
        learning_starts = HP["learning_starts"],
        train_freq    = HP["train_freq"],
        gradient_steps = HP["gradient_steps"],
        policy_kwargs = {"net_arch": HP["net_arch"]},
        verbose       = 1,
        tensorboard_log = LOG_DIR,
        device        = "cpu",  # TD3 with MLP is faster on CPU
    )

    # ── Callbacks ──────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq      = 10_000,
        save_path      = CKPT_DIR,
        name_prefix    = "td3_pid",
        save_replay_buffer = True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(RUN_DIR, "best_model"),
        log_path             = EVAL_DIR,
        eval_freq            = 20_000,
        n_eval_episodes      = 3,
        deterministic        = True,
        render               = False,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("[TRAIN] Starting training …")
    t0 = time.time()

    model.learn(
        total_timesteps = HP["total_timesteps"],
        callback        = [checkpoint_cb, eval_cb],
        progress_bar    = True,
    )

    elapsed = time.time() - t0
    print(f"[TRAIN] Done in {elapsed/3600:.1f} h")

    final_path = os.path.join(RUN_DIR, "td3_pid_final")
    model.save(final_path)
    print(f"[TRAIN] Final model saved → {final_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
