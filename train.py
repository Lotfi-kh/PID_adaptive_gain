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
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from envs import PX4GainTunerEnv, PyBulletPIDTunerEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env",   choices=["pybullet", "px4"], default="pybullet")
parser.add_argument("--axis",  choices=["roll", "pitch", "roll+pitch"], default="roll",
                    help="Which axis to tune: 'roll', 'pitch', or 'roll+pitch' (joint). "
                         "Default: roll")
parser.add_argument("--steps", type=int, default=None,
                    help="Override total_timesteps")
# ── Disturbance randomization (training only) ──────────────────────────────────
parser.add_argument("--randomize-disturbance", action="store_true",
                    help="Sample disturbance params at each episode reset")
parser.add_argument("--init-noise-min",    type=float, default=0.03)
parser.add_argument("--init-noise-max",    type=float, default=0.15)
parser.add_argument("--dist-step-min",     type=int,   default=80)
parser.add_argument("--dist-step-max",     type=int,   default=250)
parser.add_argument("--dist-mag-min",      type=float, default=0.0)
parser.add_argument("--dist-mag-max",      type=float, default=0.25)  # F450-scaled: 3e-4 × 857
parser.add_argument("--dist-duration-min", type=int,   default=3)
parser.add_argument("--dist-duration-max", type=int,   default=10)
# ── Training-distribution coverage (SITL OOD fix) ─────────────────────────────
parser.add_argument("--randomize-initial-gains", action="store_true",
                    help="Sample initial Kp/Ki/Kd uniformly across bounds each "
                         "episode (covers the full gain-observation space)")
parser.add_argument("--hold-episode-prob", type=float, default=0.0,
                    help="Fraction of episodes that are 'hold': level start, no "
                         "disturbance, stable random gains. Teaches the policy "
                         "to output ~0 when already stable. Try 0.5.")
# ── Sustained-disturbance episodes (Ki-destruction fix) ───────────────────────
parser.add_argument("--sustained-episode-prob", type=float, default=0.0,
                    help="Fraction of episodes that are 'sustained': default "
                         "gains + moderate CONSTANT torque for the whole "
                         "episode. Makes the 'kill Ki' transient exploit "
                         "expensive. Try 0.25.")
parser.add_argument("--sustained-dist-mag-min", type=float, default=0.10)
parser.add_argument("--sustained-dist-mag-max", type=float, default=0.15)
# ── Action-noise decay (settle late-stage exploration) ────────────────────────
parser.add_argument("--action-noise-decay", action="store_true",
                    help="Linearly decay TD3 action-noise sigma over this "
                         "invocation's steps (start=action_noise_std).")
parser.add_argument("--action-noise-end", type=float, default=0.02,
                    help="Final action-noise sigma when --action-noise-decay "
                         "is set (default 0.02).")
# ── Resume from checkpoint ─────────────────────────────────────────────────────
parser.add_argument("--resume", default=None, metavar="MODEL_ZIP",
                    help="Path to a saved TD3 .zip to resume from. "
                         "Actor/critic weights are restored; replay buffer starts cold. "
                         "--steps is the TOTAL target (e.g. 1000000 to reach 1M "
                         "when resuming a 500k run).")
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
    net_arch          = {"pi": [64, 64], "qf": [256, 256]},
    learning_starts   = 1_000,
    train_freq        = (1, "step"),
    gradient_steps    = 1,
)
if args.steps:
    HP["total_timesteps"] = args.steps


class ActionNoiseDecayCallback(BaseCallback):
    """Linearly decay NormalActionNoise sigma over this invocation's steps.

    Decay is measured from num_timesteps at training start, so it behaves
    correctly with --resume (the additional steps get the full schedule).
    """

    def __init__(self, sigma_start, sigma_end, decay_steps, verbose=0):
        super().__init__(verbose)
        self.sigma_start = float(sigma_start)
        self.sigma_end   = float(sigma_end)
        self.decay_steps = max(1, int(decay_steps))
        self._start_ts   = 0

    def _on_training_start(self) -> None:
        self._start_ts = self.num_timesteps

    def _on_step(self) -> bool:
        prog  = (self.num_timesteps - self._start_ts) / self.decay_steps
        prog  = min(1.0, max(0.0, prog))
        sigma = self.sigma_start + prog * (self.sigma_end - self.sigma_start)
        an = self.model.action_noise
        if an is not None and hasattr(an, "_sigma"):
            an._sigma = sigma * np.ones_like(an._sigma)
        return True


def make_env(eval_mode: bool = False):
    if args.env == "pybullet":
        if args.axis == "roll+pitch":
            tune_axes = ["roll", "pitch"]
            # Training: randomize over roll-only / pitch-only / both each episode.
            # Eval env (deterministic): always disturb both axes.
            dist_axis = "random" if (args.randomize_disturbance and not eval_mode) else "both"
        else:
            tune_axes = [args.axis]
            dist_axis = args.axis
        env = PyBulletPIDTunerEnv(
            tune_axes        = tune_axes,
            disturbance_axis = dist_axis,
            max_steps       = 500,
            target_alt      = 1.0,
            reward_w1       = 1.0,
            reward_w2       = 2.0,
            reward_w3       = 0.1,
            reward_w4       = 0.001,
            crash_penalty   = 50.0,
            stability_bonus = 20.0,
            init_noise      = 0.05,
            # disturbance randomization — enabled for training, disabled for eval
            randomize_disturbance       = args.randomize_disturbance and not eval_mode,
            init_noise_range            = (args.init_noise_min,    args.init_noise_max),
            disturbance_step_range      = (args.dist_step_min,     args.dist_step_max),
            disturbance_magnitude_range = (args.dist_mag_min,      args.dist_mag_max),
            disturbance_duration_range  = (args.dist_duration_min, args.dist_duration_max),
            # Eval env stays on the frozen-benchmark protocol (default gains,
            # no hold episodes) so results remain comparable to Phase 2.
            randomize_initial_gains     = args.randomize_initial_gains and not eval_mode,
            hold_episode_prob           = (0.0 if eval_mode else args.hold_episode_prob),
            sustained_episode_prob      = (0.0 if eval_mode else args.sustained_episode_prob),
            sustained_dist_mag_range    = (args.sustained_dist_mag_min,
                                           args.sustained_dist_mag_max),
        )
    else:
        env = PX4GainTunerEnv(
            step_duration   = 0.1,
            max_steps       = 500,
            takeoff_alt     = 5.0,
            reward_w1       = 1.0,
            reward_w2       = 2.0,
            reward_w3       = 0.1,
            reward_w4       = 0.001,
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
    if args.resume:
        print(f"[TRAIN] Resuming from: {args.resume}")
        print("[TRAIN] Replay buffer: starts cold (not saved in checkpoint).")
        model = TD3.load(
            args.resume,
            env    = env,
            device = "cpu",
        )
        # Restore settings not persisted in the zip file
        model.action_noise      = action_noise
        model.tensorboard_log   = LOG_DIR
        model.verbose           = 1
    else:
        model = TD3(
            policy          = "MlpPolicy",
            env             = env,
            learning_rate   = HP["learning_rate"],
            buffer_size     = HP["buffer_size"],
            batch_size      = HP["batch_size"],
            gamma           = HP["gamma"],
            tau             = HP["tau"],
            policy_delay    = HP["policy_delay"],
            action_noise    = action_noise,
            learning_starts = HP["learning_starts"],
            train_freq      = HP["train_freq"],
            gradient_steps  = HP["gradient_steps"],
            policy_kwargs   = {"net_arch": HP["net_arch"]},
            verbose         = 1,
            tensorboard_log = LOG_DIR,
            device          = "cpu",  # TD3 with MLP is faster on CPU
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq      = 10_000,
        save_path      = CKPT_DIR,
        name_prefix    = "td3_pid",
        save_replay_buffer = False,
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
    # When resuming, SB3 internally does: effective_target += num_timesteps.
    # So pass (total_target - current_steps) as the steps argument so that
    # SB3's addition produces the intended total. E.g. target=1M, current=500k
    # → pass 500k → SB3 computes 500k + 500k = 1M → trains exactly 500k more.
    if args.resume:
        remaining = HP["total_timesteps"] - model.num_timesteps
        if remaining <= 0:
            print(f"[TRAIN] Already at {model.num_timesteps} steps — nothing to do.")
            env.close(); eval_env.close(); sys.exit(0)
        learn_steps = remaining
        print(f"[TRAIN] Resuming: {model.num_timesteps} → {HP['total_timesteps']} "
              f"({remaining} additional steps)")
    else:
        learn_steps = HP["total_timesteps"]

    callbacks = [checkpoint_cb, eval_cb]
    if args.action_noise_decay:
        decay_cb = ActionNoiseDecayCallback(
            sigma_start = HP["action_noise_std"],
            sigma_end   = args.action_noise_end,
            decay_steps = learn_steps,
        )
        callbacks.append(decay_cb)
        print(f"[TRAIN] Action-noise decay: {HP['action_noise_std']} → "
              f"{args.action_noise_end} over {learn_steps} steps")

    print("[TRAIN] Starting training …")
    t0 = time.time()

    model.learn(
        total_timesteps     = learn_steps,
        callback            = callbacks,
        progress_bar        = True,
        reset_num_timesteps = not bool(args.resume),
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
