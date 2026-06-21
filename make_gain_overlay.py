#!/usr/bin/env python3
"""Three-model gain-overlay figure: Kp and Ki under an identical sustained roll
torque (0.17 N·m, applied step 100 for 300 steps), noise-free and deterministic.
Same rollout as eval_gain_trajectory_compare.py, but captures the trajectories and
plots them overlaid.  Run:  PYTHONPATH=. python make_gain_overlay.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from envs import PyBulletPIDTunerEnv

DIST_STEP, DIST_DUR, DT = 100, 300, 0.02   # DT≈policy step → time axis in seconds

CANDS = [
    ("clean (c860k)",          "results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip",            "#1f77b4"),
    ("hardened (noiselat)",    "results/frozen_joint_12d_noiselat/td3_pid_noiselat_best.zip",        "#ff7f0e"),
    ("quiet (noiselat-quiet)", "results/frozen_joint_12d_noiselat_quiet/td3_pid_noiselat_quiet_best.zip", "#2ca060"),
]

def rollout(model):
    env = PyBulletPIDTunerEnv(
        tune_axes=["roll", "pitch"], disturbance_axis="roll",
        max_steps=500, target_alt=1.0, init_noise=0.05,
        reward_w1=1.0, reward_w2=2.0, reward_w3=0.1, reward_w4=0.001,
        crash_penalty=50.0, stability_bonus=200.0,
        disturbance_step=DIST_STEP, disturbance_magnitude=0.17, disturbance_duration=DIST_DUR)
    obs, _ = env.reset(seed=42); kp, ki = [], []; done = False
    while not done:
        a = model.predict(obs, deterministic=True)[0]
        obs, _, t, tr, info = env.step(a)
        kp.append(info["Kp_roll"]); ki.append(info["Ki_roll"]); done = t or tr
    env.close(); return np.asarray(kp), np.asarray(ki)

KP0, KI0 = PyBulletPIDTunerEnv.KP_DEFAULT, PyBulletPIDTunerEnv.KI_DEFAULT
fig, (axp, axi) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
for name, path, c in CANDS:
    kp, ki = rollout(TD3.load(path, device="cpu"))
    t = np.arange(len(kp)) * DT
    axp.plot(t, kp, color=c, lw=1.9, label=name)
    axi.plot(t, ki, color=c, lw=1.9, label=name)
    print(f"{name:24s} Kp peak {kp.max():.3f}  Ki min-in-torque {ki[DIST_STEP:DIST_STEP+DIST_DUR].min():.4f}")

t0, t1 = DIST_STEP*DT, (DIST_STEP+DIST_DUR)*DT
for ax in (axp, axi):
    ax.axvspan(t0, t1, color="0.85", alpha=0.6, zorder=0)
    ax.grid(True, ls=":", alpha=0.5)
axp.axhline(KP0, color="k", ls="--", lw=0.8, alpha=0.6)
axi.axhline(KI0, color="k", ls="--", lw=0.8, alpha=0.6)
axp.text(t0+0.1, axp.get_ylim()[1], "  sustained torque applied", fontsize=8,
         color="0.35", va="top")
axp.text(0.05, KP0, " default", fontsize=7.5, color="0.4", va="bottom")
axp.set_ylabel(r"$K_p$ (roll)")
axi.set_ylabel(r"$K_i$ (roll)")
axi.set_xlabel("time (s)")
axp.legend(loc="upper right", fontsize=8.5, ncol=1, framealpha=0.95)
axp.set_title(r"Gain response to an identical sustained $0.17\,$N$\cdot$m roll torque")
fig.tight_layout()
fig.savefig("test_results/figures/gain_overlay.png", dpi=200, bbox_inches="tight")
print("[OK] wrote test_results/figures/gain_overlay.png")
