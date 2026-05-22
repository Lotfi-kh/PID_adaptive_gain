"""
eval_disturbance.py, Disturbance-rejection evaluation

Evaluates fixed-gain baseline PID, trained RL-adaptive PID, or both, under:
  - configurable initial noise (init_noise)
  - optional mid-episode torque impulse on a chosen axis

Axes (--axis)
-------------
  roll        (default) tune & report the roll-rate PID         (3-D action)
  pitch                 tune & report the pitch-rate PID        (3-D action)
  roll+pitch            joint: tune & report both rate PIDs     (3-D shared action)

Disturbance axis (--dist-axis)
------------------------------
  roll | pitch | both
  Default: matches --axis for single-axis; 'both' for roll+pitch.

Disturbance magnitudes (calibrated for F450, Ixx = Iyy = 0.012 kg.m²):
  low    4.3e-2 N.m  ;  medium 1.7e-1 N.m

Examples
--------
  # joint model, roll-only disturbance, baseline vs RL
  python eval_disturbance.py --model runs/<ts>/td3_pid_final.zip \\
      --axis roll+pitch --dist-axis roll \\
      --init-noise 0.10 --dist-step 150 --dist-magnitude 0.17 --dist-duration 5 \\
      --episodes 10 --out-dir results/joint_100k/dist_roll

  # baseline only (no model needed)
  python eval_disturbance.py --baseline-only --axis roll+pitch --dist-axis both \\
      --init-noise 0.10 --dist-step 150 --dist-magnitude 0.17 --dist-duration 5 \\
      --episodes 10 --out-dir results/joint_100k/baseline_both
"""

import argparse
import os
import numpy as np
from stable_baselines3 import TD3
from envs import PyBulletPIDTunerEnv


parser = argparse.ArgumentParser(
    description="Disturbance-rejection evaluation: baseline, RL-adaptive, or both.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
mode_group = parser.add_mutually_exclusive_group()
mode_group.add_argument("--baseline-only", action="store_true",
                        help="Run fixed-gain baseline only (--model not required)")
mode_group.add_argument("--rl-only", action="store_true",
                        help="Run RL-adaptive evaluation only (--model required)")

parser.add_argument("--axis", choices=["roll", "pitch", "roll+pitch"], default="roll",
                    help="Axis to tune & report (default: roll)")
parser.add_argument("--dist-axis", choices=["roll", "pitch", "both"], default=None,
                    help="Disturbance axis. Default: matches --axis for single-axis, "
                         "'both' for roll+pitch.")
parser.add_argument("--model", default=None,
                    help="Path to trained TD3 .zip. Required unless --baseline-only.")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--init-noise", type=float, default=0.05,
                    help="Initial orientation/rate noise (rad). 0.05 / 0.10 / 0.15")
parser.add_argument("--dist-step", type=int, default=150,
                    help="Control step at which to begin the disturbance impulse")
parser.add_argument("--dist-magnitude", type=float, default=0.0,
                    help="Axis-torque impulse magnitude (N.m). 0 = disabled. "
                         "F450: low=4.3e-2, medium=1.7e-1")
parser.add_argument("--dist-duration", type=int, default=5,
                    help="Number of control steps to sustain the impulse")
parser.add_argument("--out-dir", default=None,
                    help="Directory for results. Auto-generated if not given.")
parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

RUN_BASELINE = not args.rl_only
RUN_RL       = not args.baseline_only
if RUN_RL and args.model is None:
    parser.error("--model is required unless --baseline-only is used")

JOINT     = (args.axis == "roll+pitch")
TUNE_AXES = ["roll", "pitch"] if JOINT else [args.axis]
DIST_AXIS = args.dist_axis if args.dist_axis is not None else ("both" if JOINT else args.axis)
N_ACT     = 3   # always 3: joint mode uses shared [ΔKp,ΔKi,ΔKd]
ZERO_ACTION = np.zeros(N_ACT, dtype=np.float32)

DIST_ACTIVE = args.dist_magnitude > 0.0
CTRL_FREQ   = 48
DT          = 1.0 / CTRL_FREQ
RECOVERY_THRESHOLD = 0.10   # rad/s
RECOVERY_WINDOW    = 10     # consecutive steps

if args.out_dir is None:
    axtag = "roll+pitch" if JOINT else args.axis
    tag = (f"{axtag}_dist-{DIST_AXIS}_noise{int(args.init_noise*100):02d}_mag{args.dist_magnitude:.0e}"
           if DIST_ACTIVE else f"{axtag}_noise{int(args.init_noise*100):02d}_nodist")
    args.out_dir = os.path.join("results", "dist_eval", tag)
os.makedirs(args.out_dir, exist_ok=True)

ENV_KWARGS = dict(
    tune_axes             = TUNE_AXES,
    disturbance_axis      = DIST_AXIS,
    max_steps             = 500,
    target_alt            = 1.0,
    init_noise            = args.init_noise,
    reward_w1=1.0, reward_w2=2.0, reward_w3=0.1, reward_w4=0.001,
    crash_penalty=50.0, stability_bonus=200.0,
    disturbance_step      = args.dist_step if DIST_ACTIVE else None,
    disturbance_magnitude = args.dist_magnitude,
    disturbance_duration  = args.dist_duration,
)


def _recovery_steps(rate_series, dist_end):
    """Steps from dist_end until |rate| < threshold for RECOVERY_WINDOW consecutive
    steps. Returns None if not recovered within the trajectory."""
    if dist_end is None or dist_end >= len(rate_series):
        return None
    post = np.abs(np.asarray(rate_series[dist_end:]))
    for i in range(len(post) - RECOVERY_WINDOW + 1):
        if (post[i:i + RECOVERY_WINDOW] < RECOVERY_THRESHOLD).all():
            return i
    return None


def rollout(env, model_or_none, n_episodes, label):
    """Deterministic rollout. Tracks both roll and pitch trajectories regardless
    of which axis is tuned, so the comparison report is symmetric."""
    dist_start = args.dist_step if DIST_ACTIVE else None
    dist_end   = (args.dist_step + args.dist_duration) if DIST_ACTIVE else None

    T_roll, T_pitch = [], []   # angles (rad)
    T_rr, T_pr      = [], []   # body rates (rad/s)
    T_rew, T_act    = [], []
    T_kp_r, T_ki_r, T_kd_r = [], [], []   # roll gains
    T_kp_p, T_ki_p, T_kd_p = [], [], []   # pitch gains

    ep_crashes, ep_lens, rec_list = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        roll_a, pitch_a, rr, pr, rew, act = [], [], [], [], [], []
        kp_r, ki_r, kd_r, kp_p, ki_p, kd_p = [], [], [], [], [], []

        while not done:
            action = (ZERO_ACTION if model_or_none is None
                      else model_or_none.predict(obs, deterministic=True)[0])
            obs, reward, terminated, truncated, info = env.step(action)

            roll_a.append(float(np.deg2rad(info["roll_deg"])))
            pitch_a.append(float(np.deg2rad(info["pitch_deg"])))
            rr.append(float(info["roll_rate"]))
            pr.append(float(info["pitch_rate"]))
            rew.append(float(reward))
            act.append(float(np.mean(np.abs(action))))

            if JOINT:
                kp_r.append(info["Kp_roll"]);  ki_r.append(info["Ki_roll"]);  kd_r.append(info["Kd_roll"])
                kp_p.append(info["Kp_pitch"]); ki_p.append(info["Ki_pitch"]); kd_p.append(info["Kd_pitch"])
            elif args.axis == "roll":
                kp_r.append(info["Kp"]); ki_r.append(info["Ki"]); kd_r.append(info["Kd"])
                kp_p.append(np.nan);     ki_p.append(np.nan);     kd_p.append(np.nan)
            else:  # pitch single-axis
                kp_p.append(info["Kp"]); ki_p.append(info["Ki"]); kd_p.append(info["Kd"])
                kp_r.append(np.nan);     ki_r.append(np.nan);     kd_r.append(np.nan)

            done = terminated or truncated

        ep_crashes.append(bool(info["crashed"]))
        ep_lens.append(int(info["step"]))

        if DIST_ACTIVE:
            if DIST_AXIS == "roll":
                rec_list.append(_recovery_steps(rr, dist_end))
            elif DIST_AXIS == "pitch":
                rec_list.append(_recovery_steps(pr, dist_end))
            else:  # both, recover when the larger of the two rates is below threshold
                combined = [max(abs(a), abs(b)) for a, b in zip(rr, pr)]
                rec_list.append(_recovery_steps(combined, dist_end))

        T_roll.append(roll_a); T_pitch.append(pitch_a); T_rr.append(rr); T_pr.append(pr)
        T_rew.append(rew); T_act.append(act)
        T_kp_r.append(kp_r); T_ki_r.append(ki_r); T_kd_r.append(kd_r)
        T_kp_p.append(kp_p); T_ki_p.append(ki_p); T_kd_p.append(kd_p)

    def _rms(series_list):
        v = []
        for s in series_list: v.extend([x * x for x in s])
        a = np.asarray(v)
        return float(np.sqrt(a.mean())) if a.size else float("nan")

    def _peak(series_list):
        peaks = []
        for s in series_list:
            window = s[dist_start:] if dist_start is not None else s
            if len(window): peaks.append(float(np.max(np.abs(window))))
        return float(np.mean(peaks)) if peaks else float("nan")

    def _max_angle(series_list):
        m = [float(np.max(np.abs(s))) for s in series_list if len(s)]
        return float(np.mean(m)) if m else float("nan")

    def _last_mean(series_list):
        vals = [s[-1] for s in series_list if len(s) and not np.isnan(s[-1])]
        return float(np.mean(vals)) if vals else float("nan")

    def _all_mean(series_list):
        v = []
        for s in series_list: v.extend(s)
        return float(np.mean(v)) if v else float("nan")

    valid_rec = [r for r in rec_list if r is not None]

    return dict(
        label=label, n_episodes=n_episodes,
        crashes=int(sum(ep_crashes)), crash_rate=float(sum(ep_crashes)) / n_episodes,
        mean_ep_len=float(np.mean(ep_lens)),
        rms_roll_rate=_rms(T_rr),   rms_pitch_rate=_rms(T_pr),
        peak_roll_rate=_peak(T_rr), peak_pitch_rate=_peak(T_pr),
        max_abs_roll=_max_angle(T_roll), max_abs_pitch=_max_angle(T_pitch),
        mean_rew=_all_mean(T_rew), mean_act=_all_mean(T_act),
        kp_roll=_last_mean(T_kp_r),  ki_roll=_last_mean(T_ki_r),  kd_roll=_last_mean(T_kd_r),
        kp_pitch=_last_mean(T_kp_p), ki_pitch=_last_mean(T_ki_p), kd_pitch=_last_mean(T_kd_p),
        recovery_mean=float(np.mean(valid_rec)) if valid_rec else None,
        recovery_rate=float(len(valid_rec)) / n_episodes if DIST_ACTIVE else None,
        T_roll=T_roll, T_pitch=T_pitch, T_rr=T_rr, T_pr=T_pr,
        T_rew=T_rew, T_act=T_act, T_kp_r=T_kp_r, T_kp_p=T_kp_p,
        ep_lens=ep_lens, ep_crashes=ep_crashes,
    )


mode_label = ("Baseline + RL" if (RUN_BASELINE and RUN_RL)
              else "Baseline only" if RUN_BASELINE else "RL only")
print(f"\n[DIST_EVAL] Mode: {mode_label}  |  tune={'roll+pitch' if JOINT else args.axis}  |  dist-axis={DIST_AXIS}")
print(f"[DIST_EVAL] init_noise={args.init_noise}  dist_magnitude={args.dist_magnitude}  "
      f"dist_step={args.dist_step}  dist_duration={args.dist_duration}")
print(f"[DIST_EVAL] Results -> {args.out_dir}\n")

model = None
if RUN_RL:
    print(f"[DIST_EVAL] Loading model: {args.model}")
    model = TD3.load(args.model, device="cpu")
    if tuple(model.action_space.shape) != (N_ACT,):
        raise SystemExit(f"[DIST_EVAL] Model action dim {model.action_space.shape} != expected "
                         f"({N_ACT},). Wrong --axis for this model?")
    env_obs_dim = 12 if JOINT else 9
    if tuple(model.observation_space.shape) != (env_obs_dim,):
        hint = "roll+pitch" if model.observation_space.shape == (12,) else "roll or pitch"
        raise SystemExit(f"[DIST_EVAL] Model obs dim {model.observation_space.shape} != env obs dim "
                         f"({env_obs_dim},). Use --axis {hint} to match this model.")

env = PyBulletPIDTunerEnv(**ENV_KWARGS)
baseline = rollout(env, None,  args.episodes, "Baseline (fixed gains)") if RUN_BASELINE else None
rl       = rollout(env, model, args.episodes, "RL-adaptive (TD3)")      if RUN_RL       else None
env.close()


def fmt_gains(r, axis):
    if axis == "roll":
        return f"{r['kp_roll']:.6f} / {r['ki_roll']:.7f} / {r['kd_roll']:.8f}"
    return f"{r['kp_pitch']:.6f} / {r['ki_pitch']:.7f} / {r['kd_pitch']:.8f}"

ROWS = [
    ("Crashes",                  lambda r: f"{r['crashes']}/{r['n_episodes']} ({r['crash_rate']*100:.0f}%)"),
    ("Mean ep length (steps)",   lambda r: f"{r['mean_ep_len']:.1f}"),
    ("RMS roll_rate (rad/s)",    lambda r: f"{r['rms_roll_rate']:.4f}"),
    ("RMS pitch_rate (rad/s)",   lambda r: f"{r['rms_pitch_rate']:.4f}"),
    ("Peak |roll_rate| (rad/s)", lambda r: f"{r['peak_roll_rate']:.4f}"),
    ("Peak |pitch_rate|(rad/s)", lambda r: f"{r['peak_pitch_rate']:.4f}"),
    ("Mean max |roll| (rad)",    lambda r: f"{r['max_abs_roll']:.4f}"),
    ("Mean max |pitch| (rad)",   lambda r: f"{r['max_abs_pitch']:.4f}"),
    ("Mean reward/step",         lambda r: f"{r['mean_rew']:+.4f}"),
    ("Mean |action|",            lambda r: f"{r['mean_act']:.4f}"),
    ("Final roll  Kp/Ki/Kd",     lambda r: fmt_gains(r, "roll")),
    ("Final pitch Kp/Ki/Kd",     lambda r: fmt_gains(r, "pitch")),
]
if DIST_ACTIVE:
    ROWS.append(("Recovery time (steps)", lambda r: (
        f"{r['recovery_mean']:.1f} ({r['recovery_mean']/CTRL_FREQ*1000:.0f} ms) "
        f"[{r['recovery_rate']*100:.0f}% rec.]" if r['recovery_mean'] is not None else "did not recover")))

title_ax = "ROLL+PITCH (joint)" if JOINT else args.axis.upper()
lines = ["=" * 80]
if RUN_BASELINE and RUN_RL:
    lines += [f"DISTURBANCE-REJECTION EVALUATION, Baseline vs RL-Adaptive PID  [{title_ax}]",
              f"Model      : {args.model}"]
elif RUN_BASELINE:
    lines += [f"DISTURBANCE-REJECTION EVALUATION, Baseline (fixed gains)  [{title_ax}]",
              "Model      : N/A, fixed default gains"]
else:
    lines += [f"DISTURBANCE-REJECTION EVALUATION, RL-Adaptive PID  [{title_ax}]",
              f"Model      : {args.model}"]
lines.append(f"init_noise : {args.init_noise} rad")
if DIST_ACTIVE:
    lines.append(f"Disturbance: {args.dist_magnitude:.1e} N.m at step {args.dist_step} for "
                 f"{args.dist_duration} steps ({args.dist_duration/CTRL_FREQ*1000:.0f} ms)  "
                 f"[dist-axis = {DIST_AXIS}]")
else:
    lines.append("Disturbance: NONE")
lines.append(f"Episodes   : {args.episodes}  |  Seed: {args.seed}")
lines.append("=" * 80)

if RUN_BASELINE and RUN_RL:
    lines.append(f"\n{'Metric':<28} {'Baseline':>22}  {'RL-adaptive':>22}")
    lines.append("-" * 80)
    for label, fmt in ROWS:
        lines.append(f"  {label:<26} {fmt(baseline):>22}  {fmt(rl):>22}")

    lines.append("\n── Delta (positive = RL better) ────────────────────────────────────────────")
    def drow(label, b, r, lower_better=True):
        d   = (b - r) if lower_better else (r - b)
        pct = 100 * d / (abs(b) + 1e-12)
        tag = "BETTER" if d >= 0 else "WORSE "
        return f"  {tag}  {label:<24}  baseline={b:+.5f}  rl={r:+.5f}  {'+' if d>=0 else ''}{pct:.1f}%"
    lines.append(drow("crash rate",        baseline['crash_rate'],      rl['crash_rate']))
    lines.append(drow("RMS roll_rate",     baseline['rms_roll_rate'],   rl['rms_roll_rate']))
    lines.append(drow("RMS pitch_rate",    baseline['rms_pitch_rate'],  rl['rms_pitch_rate']))
    lines.append(drow("peak |roll_rate|",  baseline['peak_roll_rate'],  rl['peak_roll_rate']))
    lines.append(drow("peak |pitch_rate|", baseline['peak_pitch_rate'], rl['peak_pitch_rate']))
    lines.append(drow("mean max |roll|",   baseline['max_abs_roll'],    rl['max_abs_roll']))
    lines.append(drow("mean max |pitch|",  baseline['max_abs_pitch'],   rl['max_abs_pitch']))
    lines.append(drow("mean reward/step",  baseline['mean_rew'],        rl['mean_rew'], lower_better=False))
    if DIST_ACTIVE and baseline['recovery_mean'] is not None and rl['recovery_mean'] is not None:
        lines.append(drow("recovery (steps)", baseline['recovery_mean'], rl['recovery_mean']))
else:
    result = baseline if RUN_BASELINE else rl
    lines.append(f"\n  {'Metric':<28} {'Value':>22}")
    lines.append("  " + "-" * 52)
    for label, fmt in ROWS:
        lines.append(f"  {label:<28} {fmt(result):>22}")
lines.append("=" * 80)

report = "\n".join(lines)
print("\n" + report)


# ── Save NPZ + TXT ─────────────────────────────────────────────────────────────

def pad(traj_list):
    if not traj_list: return np.zeros((0, 0))
    n = max(len(t) for t in traj_list)
    out = np.full((len(traj_list), n), np.nan)
    for i, t in enumerate(traj_list): out[i, :len(t)] = t
    return out

npz = dict(
    axis           = np.array("roll+pitch" if JOINT else args.axis),
    dist_axis      = np.array(DIST_AXIS),
    dist_step      = np.array(args.dist_step),
    dist_duration  = np.array(args.dist_duration),
    dist_magnitude = np.array(args.dist_magnitude),
    init_noise     = np.array(args.init_noise),
    ctrl_freq      = np.array(CTRL_FREQ),
)
if baseline is not None:
    npz.update(
        bl_rolls=pad(baseline['T_roll']),       bl_pitch_angles=pad(baseline['T_pitch']),
        bl_roll_rates=pad(baseline['T_rr']),    bl_pitch_rates=pad(baseline['T_pr']),
        bl_rewards=pad(baseline['T_rew']),      bl_actions=pad(baseline['T_act']),
        bl_kps=pad(baseline['T_kp_r']),         bl_kps_pitch=pad(baseline['T_kp_p']),
        bl_ep_lens=np.array(baseline['ep_lens']), bl_crashes=np.array(baseline['ep_crashes']),
    )
if rl is not None:
    npz.update(
        rl_rolls=pad(rl['T_roll']),       rl_pitch_angles=pad(rl['T_pitch']),
        rl_roll_rates=pad(rl['T_rr']),    rl_pitch_rates=pad(rl['T_pr']),
        rl_rewards=pad(rl['T_rew']),      rl_actions=pad(rl['T_act']),
        rl_kps=pad(rl['T_kp_r']),         rl_kps_pitch=pad(rl['T_kp_p']),
        rl_ep_lens=np.array(rl['ep_lens']), rl_crashes=np.array(rl['ep_crashes']),
    )
npz_path = os.path.join(args.out_dir, "disturbance_eval_results.npz")
np.savez(npz_path, **npz)
txt_path = os.path.join(args.out_dir, "disturbance_comparison_summary.txt")
with open(txt_path, "w") as f:
    f.write(report + "\n")
print(f"\n[DIST_EVAL] NPZ saved  -> {npz_path}")
print(f"[DIST_EVAL] TXT saved  -> {txt_path}")


if args.no_plots:
    print("[DIST_EVAL] Plots skipped (--no-plots).")
else:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        C_BL, C_RL = "#e74c3c", "#3498db"

        def band(ax, trajs, color, label):
            arr = pad(trajs)
            if arr.size == 0: return
            mean = np.nanmean(arr, 0); std = np.nanstd(arr, 0)
            t = np.arange(arr.shape[1]) / CTRL_FREQ
            ax.plot(t, mean, color=color, lw=2, label=label)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

        def shade(ax):
            if DIST_ACTIVE:
                ax.axvspan(args.dist_step / CTRL_FREQ,
                           (args.dist_step + args.dist_duration) / CTRL_FREQ,
                           color="orange", alpha=0.25)

        srcs = [(s, c, l) for s, c, l in [(baseline, C_BL, "Baseline"), (rl, C_RL, "RL")] if s is not None]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"Disturbance-Rejection | tune={'roll+pitch' if JOINT else args.axis} | "
                     f"dist-axis={DIST_AXIS} | noise={args.init_noise} | mag={args.dist_magnitude:.1e}",
                     fontsize=11)
        for s, c, l in srcs:
            band(axes[0, 0], s['T_roll'], c, l); band(axes[0, 1], s['T_pitch'], c, l)
            band(axes[1, 0], s['T_rr'],   c, l); band(axes[1, 1], s['T_pr'],    c, l)
        for ax in axes.flat:
            shade(ax); ax.grid(alpha=0.3); ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel("Time (s)"); ax.legend(fontsize=9)
        axes[0, 0].set_title("Roll angle (rad)");  axes[0, 1].set_title("Pitch angle (rad)")
        axes[1, 0].set_title("Roll rate (rad/s)"); axes[1, 1].set_title("Pitch rate (rad/s)")
        plt.tight_layout()
        pp = os.path.join(args.out_dir, "disturbance_trajectories.png")
        plt.savefig(pp, dpi=130, bbox_inches="tight"); plt.close()
        print(f"[DIST_EVAL] Plot saved -> {pp}")
    except Exception as exc:
        print(f"[DIST_EVAL][WARN] Plotting failed: {exc}")

print("\n[DIST_EVAL] Done.")
