"""
make_pybullet_signals.py — time-series 'signal' figures from PyBullet rollouts:
the quiet model (noiselat-quiet) vs the fixed-gain baseline under the same roll
disturbance, in the nominal (clean) environment used for the 48-condition grid.

Outputs to ~/thesis/images/:
  pybullet_signal_roll.png   — roll angle + roll rate, baseline vs adaptive
  pybullet_signal_gains.png  — live Kp/Ki/Kd of the adaptive controller
Frozen model, evaluation only (no training). Run from rl_pid_tuner/.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from envs import PyBulletPIDTunerEnv

MODEL = "results/frozen_joint_12d_noiselat_quiet/td3_pid_noiselat_quiet_best.zip"
IMG   = os.path.expanduser("~/thesis/images")
CTRL  = 48
DIST_STEP, DIST_MAG, DIST_DUR = 80, 0.30, 24      # 0.30 N·m roll kick, ~0.5 s

def make_env():
    return PyBulletPIDTunerEnv(
        tune_axes=["roll", "pitch"], disturbance_axis="roll",
        max_steps=320, target_alt=1.0, init_noise=0.0,
        reward_w1=1.0, reward_w2=2.0, reward_w3=0.1, reward_w4=0.001,
        crash_penalty=50.0, stability_bonus=200.0,
        disturbance_step=DIST_STEP, disturbance_magnitude=DIST_MAG,
        disturbance_duration=DIST_DUR)

def rollout(model):
    """Return dict of time-series. model=None → baseline (zero action = default gains)."""
    env = make_env()
    obs, info = env.reset(seed=7)
    rec = {k: [] for k in ("t","roll","rr","kp","ki","kd","dist")}
    step = 0; done = False
    while not done:
        if model is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action = model.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(action)
        rec["t"].append(step / CTRL)
        rec["roll"].append(info["roll_deg"]);  rec["rr"].append(info["roll_rate"])
        rec["kp"].append(info["Kp_roll"]); rec["ki"].append(info["Ki_roll"]); rec["kd"].append(info["Kd_roll"])
        rec["dist"].append(info["disturbance_active"])
        step += 1; done = term or trunc
    env.close()
    return {k: np.asarray(v) for k, v in rec.items()}

model = TD3.load(MODEL, device="cpu")
A = rollout(model)
B = rollout(None)
d0 = A["t"][np.argmax(A["dist"])] if A["dist"].any() else 0.0   # disturbance onset
d1 = A["t"][len(A["dist"]) - 1 - np.argmax(A["dist"][::-1])] if A["dist"].any() else 0.0

# ── Figure 1: roll angle + roll rate, baseline vs adaptive ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
for R, c, lbl, ls in [(B, "#888888", "baseline (fixed gains)", "--"),
                      (A, "#1f77b4", "adaptive (quiet model)", "-")]:
    ax1.plot(R["t"], R["roll"], color=c, ls=ls, lw=1.9, label=lbl)
    ax2.plot(R["t"], R["rr"],   color=c, ls=ls, lw=1.9, label=lbl)
for ax in (ax1, ax2):
    ax.axvspan(d0, d1, color="0.9", zorder=0, label="_")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4); ax.grid(True, ls=":", alpha=0.5)
ax1.set_ylabel("roll angle (deg)"); ax2.set_ylabel("roll rate (rad/s)")
ax2.set_xlabel("time (s)"); ax1.legend(loc="upper right", fontsize=9)
ax1.set_title("PyBullet: roll response to a 0.30 N$\\cdot$m kick — quiet model vs baseline")
fig.tight_layout(); fig.savefig(f"{IMG}/pybullet_signal_roll.png", dpi=160); plt.close(fig)

# ── Figure 2: live gains of the adaptive controller ──
fig, ax = plt.subplots(figsize=(8.5, 3.6))
for key, c, lbl in [("kp", "#1f77b4", "$K_p$"), ("ki", "#2ca02c", "$K_i$"), ("kd", "#ff7f0e", "$K_d$")]:
    ax.plot(A["t"], A[key], color=c, lw=1.8, label=lbl)
ax.axvspan(d0, d1, color="0.9", zorder=0, label="disturbance")
ax.set_xlabel("time (s)"); ax.set_ylabel("gain value"); ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=9, ncol=4)
ax.set_title("PyBullet: the quiet model's live rate-PID gains during the kick")
fig.tight_layout(); fig.savefig(f"{IMG}/pybullet_signal_gains.png", dpi=160); plt.close(fig)

print("peak|roll|  baseline=%.2f  adaptive=%.2f deg" % (np.abs(B["roll"]).max(), np.abs(A["roll"]).max()))
print("peak|rate|  baseline=%.3f  adaptive=%.3f rad/s" % (np.abs(B["rr"]).max(), np.abs(A["rr"]).max()))
print("Kp range %.3f–%.3f  Ki range %.4f–%.4f" % (A["kp"].min(), A["kp"].max(), A["ki"].min(), A["ki"].max()))
print("[OK]", f"{IMG}/pybullet_signal_roll.png", "+ pybullet_signal_gains.png")
