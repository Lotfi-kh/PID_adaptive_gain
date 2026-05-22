"""
eval_compare.py, Post-training evaluation and baseline comparison

Loads a trained TD3 model and runs deterministic evaluation against
a fixed-gain baseline under identical conditions (same env, same noise seed).

Usage:
    python eval_compare.py --model runs/<ts>/best_model/best_model.zip [--episodes 10]
"""

import argparse, os
import numpy as np
from stable_baselines3 import TD3
from envs import PyBulletPIDTunerEnv

ZERO_ACTION = np.zeros(3, dtype=np.float32)

parser = argparse.ArgumentParser()
parser.add_argument("--model",    required=True)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--out-dir",  default=None,
                    help="Directory to save eval_results.npz and comparison_summary.txt")
args = parser.parse_args()

OUT_DIR = args.out_dir or os.path.dirname(args.model)
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
ENV_KWARGS = dict(
    max_steps       = 500,
    target_alt      = 1.0,
    init_noise      = 0.05,
    reward_w1       = 1.0,
    reward_w2       = 2.0,
    reward_w3       = 0.1,
    reward_w4       = 0.001,
    crash_penalty   = 50.0,
    stability_bonus = 200.0,
)


def run_episodes(env, model_or_none, n_episodes, label):
    """Roll out n_episodes. model_or_none=None -> fixed zero action (baseline)."""
    rr2_buf, rew_buf, act_buf = [], [], []
    ep_lens, ep_crashes = [], []
    kp_list, ki_list, kd_list = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        ep_rr2, ep_rew, ep_act = [], [], []

        while not done:
            if model_or_none is None:
                action = ZERO_ACTION
            else:
                action, _ = model_or_none.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_rr2.append(info["roll_rate"] ** 2)
            ep_rew.append(reward)
            ep_act.append(float(np.mean(np.abs(action))))
            done = terminated or truncated

        ep_crashes.append(bool(info["crashed"]))
        ep_lens.append(int(info["step"]))
        rr2_buf.extend(ep_rr2)
        rew_buf.extend(ep_rew)
        act_buf.extend(ep_act)
        kp_list.append(info["Kp"])
        ki_list.append(info["Ki"])
        kd_list.append(info["Kd"])

    rr2 = np.array(rr2_buf)
    rew = np.array(rew_buf)
    act = np.array(act_buf)

    return dict(
        label        = label,
        n_episodes   = n_episodes,
        crashes      = int(sum(ep_crashes)),
        crash_rate   = float(sum(ep_crashes)) / n_episodes,
        mean_ep_len  = float(np.mean(ep_lens)),
        mean_rr2     = float(rr2.mean()),
        rms_rr       = float(np.sqrt(rr2.mean())),
        mean_rew     = float(rew.mean()),
        mean_act     = float(act.mean()),
        kp           = float(np.mean(kp_list)),
        ki           = float(np.mean(ki_list)),
        kd           = float(np.mean(kd_list)),
        ep_lens      = np.array(ep_lens),
        ep_crashes   = np.array(ep_crashes),
        rr2_all      = rr2,
        rew_all      = rew,
        act_all      = act,
    )


print(f"[EVAL] Loading model: {args.model}")
model = TD3.load(args.model, device="cpu")

env = PyBulletPIDTunerEnv(**ENV_KWARGS)

print(f"[EVAL] Baseline (fixed default gains, zero action), {args.episodes} episodes …")
baseline = run_episodes(env, None, args.episodes, "Baseline (fixed gains)")

print(f"[EVAL] RL-adaptive (trained TD3), {args.episodes} episodes …")
rl = run_episodes(env, model, args.episodes, "RL-adaptive (TD3)")

env.close()


lines = []
lines.append("=" * 65)
lines.append("PHASE 1 EVALUATION, Baseline vs RL-Adaptive PID")
lines.append(f"Model : {args.model}")
lines.append(f"Env   : init_noise={ENV_KWARGS['init_noise']}  "
             f"w1={ENV_KWARGS['reward_w1']}  w2={ENV_KWARGS['reward_w2']}  "
             f"w3={ENV_KWARGS['reward_w3']}  w4={ENV_KWARGS['reward_w4']}")
lines.append(f"Episodes per condition: {args.episodes}")
lines.append("=" * 65)

for r in [baseline, rl]:
    lines.append(f"\n── {r['label']} ──")
    lines.append(f"  Crashes           : {r['crashes']}/{r['n_episodes']}"
                 f"  ({r['crash_rate']*100:.0f}%)")
    lines.append(f"  Mean ep length    : {r['mean_ep_len']:.1f} steps / {ENV_KWARGS['max_steps']}")
    lines.append(f"  Mean roll_rate²   : {r['mean_rr2']:.6f}  (RMS = {r['rms_rr']:.4f} rad/s)")
    lines.append(f"  Mean reward/step  : {r['mean_rew']:+.4f}")
    lines.append(f"  Mean |action|     : {r['mean_act']:.4f}")
    lines.append(f"  Final Kp / Ki / Kd: {r['kp']:.6f} / {r['ki']:.7f} / {r['kd']:.8f}")
    lines.append(f"  (Defaults:          "
                 f"{PyBulletPIDTunerEnv.KP_DEFAULT:.6f} / "
                 f"{PyBulletPIDTunerEnv.KI_DEFAULT:.7f} / "
                 f"{PyBulletPIDTunerEnv.KD_DEFAULT:.8f})")

lines.append("\n── Delta (RL vs Baseline) ──")


def delta_row(label, b_val, rl_val, lower_better=True):
    d   = b_val - rl_val if lower_better else rl_val - b_val
    pct = 100 * d / (abs(b_val) + 1e-12)
    tag = "BETTER" if d >= 0 else "WORSE "
    arrow = "↓" if lower_better else "↑"
    return (f"  {tag}  {label:28s}  "
            f"baseline={b_val:+.5f}  rl={rl_val:+.5f}  "
            f"{arrow}{'+' if d>=0 else ''}{pct:.1f}%")


lines.append(delta_row("crash rate",        baseline["crash_rate"],  rl["crash_rate"]))
lines.append(delta_row("mean ep length",    baseline["mean_ep_len"], rl["mean_ep_len"],   lower_better=False))
lines.append(delta_row("mean roll_rate²",   baseline["mean_rr2"],    rl["mean_rr2"]))
lines.append(delta_row("RMS roll_rate",     baseline["rms_rr"],      rl["rms_rr"]))
lines.append(delta_row("mean reward/step",  baseline["mean_rew"],    rl["mean_rew"],      lower_better=False))

lines.append("\n" + "=" * 65)

report = "\n".join(lines)
print("\n" + report)


npz_path = os.path.join(OUT_DIR, "eval_results.npz")
np.savez(npz_path,
         baseline_rr2     = baseline["rr2_all"],
         baseline_rew     = baseline["rew_all"],
         baseline_act     = baseline["act_all"],
         baseline_lengths = baseline["ep_lens"],
         baseline_crashes = baseline["ep_crashes"],
         rl_rr2           = rl["rr2_all"],
         rl_rew           = rl["rew_all"],
         rl_act           = rl["act_all"],
         rl_lengths       = rl["ep_lens"],
         rl_crashes       = rl["ep_crashes"])

txt_path = os.path.join(OUT_DIR, "comparison_summary.txt")
with open(txt_path, "w") as f:
    f.write(report + "\n")

print(f"\n[EVAL] Results saved -> {npz_path}")
print(f"[EVAL] Summary saved -> {txt_path}")
