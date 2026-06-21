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
from stable_baselines3.common.utils import set_random_seed
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
# ── Wind episodes (continuous-wind OOD fix) ───────────────────────────────────
parser.add_argument("--wind-episode-prob", type=float, default=0.0,
                    help="Fraction of episodes that are 'wind': default gains + an "
                         "Ornstein-Uhlenbeck GUSTING torque (sustained mean + "
                         "gusts) for the whole episode. Adds the time-varying "
                         "disturbance the Gazebo wind test exposed as missing. "
                         "Try 0.25 (mixed with hold/sustained/recovery).")
parser.add_argument("--wind-mean-torque-min", type=float, default=0.06,
                    help="Min magnitude of the wind episode's sustained mean torque (N·m).")
parser.add_argument("--wind-mean-torque-max", type=float, default=0.12,
                    help="Max magnitude of the wind episode's sustained mean torque (N·m).")
parser.add_argument("--wind-gust-sigma", type=float, default=0.05,
                    help="OU gust intensity for wind episodes (N·m).")
parser.add_argument("--wind-max-torque", type=float, default=0.20,
                    help="Cap on wind episode |torque| (N·m), keeps it recoverable.")
# ── Dynamics randomization (off-nominal mass/inertia → where adaptive wins) ────
parser.add_argument("--randomize-dynamics", action="store_true",
                    help="Perturb the body mass+inertia each episode (PyBullet "
                         "changeDynamics, NOT a URDF edit). Static gains tuned for "
                         "nominal become mismatched off-nominal — the regime where "
                         "an adaptive controller can beat fixed gains.")
parser.add_argument("--mass-frac-min", type=float, default=0.8)
parser.add_argument("--mass-frac-max", type=float, default=1.2)
parser.add_argument("--inertia-frac-min", type=float, default=0.7)
parser.add_argument("--inertia-frac-max", type=float, default=1.3)
# ── Effort penalty (drives 'less torque'; 0.0 keeps the frozen reward) ─────────
parser.add_argument("--reward-w5", type=float, default=0.0,
                    help="Weight on the ∫τ² effort penalty. 0.0 = frozen reward "
                         "(reproduces the benchmark). With the fixed normalization, "
                         "w5≈0.2 makes effort ~30%% of the other terms.")
# ── Hover-quietness penalty (targets noiselat twitchiness; 0.0 keeps frozen reward)
parser.add_argument("--reward-w6", type=float, default=0.0,
                    help="Weight on the gated hover-quietness penalty (action² applied "
                         "ONLY when the true state is calm and no disturbance is active). "
                         "0.0 = frozen reward. Use to make a noise-trained policy "
                         "effectively inactive in hover without hurting disturbance response.")
# ── Sensor noise + control latency (realism: makes high gains COSTLY) ──────────
parser.add_argument("--sensor-noise-att", type=float, default=0.0,
                    help="σ of Gaussian attitude-sensor noise (rad). e.g. 0.005 ≈ 0.3°.")
parser.add_argument("--sensor-noise-rate", type=float, default=0.0,
                    help="σ of Gaussian rate-gyro noise (rad/s). e.g. 0.02.")
parser.add_argument("--control-latency-steps", type=int, default=0,
                    help="Actuation delay in control steps (1 step ≈ 21 ms at 48 Hz). "
                         "e.g. 1. Phase lag punishes over-high gains.")
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

# ── TD3 optimisation knobs (defaults = legacy HP, so old runs reproduce) ───────
# Exposed for stability-focused retrains: under noisy observations TD3 can diverge
# (eval reward oscillates), and the main levers are a lower / decaying learning
# rate, a larger batch, and a fixed seed for best-of-N selection. None of these
# touch the reward, the actor architecture, or the plant.
parser.add_argument("--lr", type=float, default=None,
                    help="Initial learning rate (default: 1e-3).")
parser.add_argument("--lr-final", type=float, default=None,
                    help="If set, LINEARLY decay LR from --lr to this over the "
                         "(remaining) steps of this invocation. Resume-aware.")
parser.add_argument("--batch-size", type=int, default=None,
                    help="TD3 batch size (default: 128). Larger = smoother "
                         "updates under noisy observations.")
parser.add_argument("--gradient-steps", type=int, default=None,
                    help="Gradient steps per env step (default: 1).")
parser.add_argument("--buffer-size", type=int, default=None,
                    help="Replay buffer capacity (default: 500000).")
parser.add_argument("--seed", type=int, default=None,
                    help="Global + model seed for reproducible best-of-N runs.")
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
if args.lr is not None:
    HP["learning_rate"] = args.lr
if args.batch_size is not None:
    HP["batch_size"] = args.batch_size
if args.gradient_steps is not None:
    HP["gradient_steps"] = args.gradient_steps
if args.buffer_size is not None:
    HP["buffer_size"] = args.buffer_size


def linear_lr_schedule(lr_start, lr_end):
    """SB3 schedule: progress_remaining 1→0, so LR goes lr_start→lr_end."""
    lr_start, lr_end = float(lr_start), float(lr_end)
    return lambda progress_remaining: lr_end + progress_remaining * (lr_start - lr_end)


# Resolve the learning-rate object once: a float (constant) or a linear schedule.
LR = HP["learning_rate"]
if args.lr_final is not None:
    LR = linear_lr_schedule(HP["learning_rate"], args.lr_final)


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
            wind_episode_prob           = (0.0 if eval_mode else args.wind_episode_prob),
            wind_mean_torque_range      = (args.wind_mean_torque_min,
                                           args.wind_mean_torque_max),
            wind_gust_sigma             = args.wind_gust_sigma,
            wind_max_torque             = args.wind_max_torque,
            reward_w5                   = (0.0 if eval_mode else args.reward_w5),
            reward_w6                   = (0.0 if eval_mode else args.reward_w6),
            randomize_dynamics          = args.randomize_dynamics and not eval_mode,
            mass_frac_range             = (args.mass_frac_min,    args.mass_frac_max),
            inertia_frac_range          = (args.inertia_frac_min, args.inertia_frac_max),
            # Noise + latency are real-system properties → applied in eval too.
            sensor_noise_att            = args.sensor_noise_att,
            sensor_noise_rate           = args.sensor_noise_rate,
            control_latency_steps       = args.control_latency_steps,
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

    # ── Seeding (reproducible best-of-N) ────────────────────────────────────────
    if args.seed is not None:
        set_random_seed(args.seed)
        print(f"[TRAIN] Global seed: {args.seed}")

    # ── Environment ────────────────────────────────────────────────────────────
    env      = make_env()
    eval_env = make_env(eval_mode=True)
    if args.seed is not None:
        env.reset(seed=args.seed)
        eval_env.reset(seed=args.seed + 10_000)

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
        # Override optimisation knobs on the resumed model (the zip restores the
        # OLD lr/batch/grad-steps/buffer; re-apply the requested ones so a
        # stability retrain actually takes effect).
        if args.seed is not None:
            model.set_random_seed(args.seed)
        if args.batch_size is not None:
            model.batch_size = HP["batch_size"]
        if args.gradient_steps is not None:
            model.gradient_steps = HP["gradient_steps"]
        if args.lr is not None or args.lr_final is not None:
            model.learning_rate = LR
            model._setup_lr_schedule()   # rebuilds model.lr_schedule from learning_rate
            print(f"[TRAIN] LR override applied to resumed model"
                  + (f" (decay → {args.lr_final})" if args.lr_final is not None else ""))
    else:
        model = TD3(
            policy          = "MlpPolicy",
            env             = env,
            learning_rate   = LR,
            seed            = args.seed,
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
