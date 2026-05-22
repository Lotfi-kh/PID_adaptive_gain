"""
make_thesis_figs.py, generate the two Section 4.2 figures for the thesis.

  ~/thesis/images/hover_action.png    , action magnitude of the selected model
                                         over 300 captured SITL hover samples
  ~/thesis/images/gain_trajectory.png , Kp/Ki/Kd of the selected model over one
                                         sustained-disturbance episode

Evaluation rollouts of the FROZEN model only, no training. Run from rl_pid_tuner/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from envs import PyBulletPIDTunerEnv

MODEL     = "results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip"
HOVER_CSV = "results/observer_2026-05-16_19-24-48.csv"
IMG_DIR   = os.path.expanduser("~/thesis/images")
OBS_COLS  = [f"obs_{i:02d}" for i in range(12)]

CTRL_FREQ           = 48
DIST_STEP, DIST_DUR = 100, 300          # disturbance window = steps [100, 400)


def fig_hover(model):
    df  = pd.read_csv(HOVER_CSV)
    obs = df[OBS_COLS].to_numpy(dtype=np.float32)
    act, _  = model.predict(obs, deterministic=True)        # (N, 3)
    mag     = np.abs(act).mean(axis=1)                      # per-sample mean |action|
    overall = float(np.abs(act).mean())
    n       = len(mag)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(np.arange(n), mag, color="tab:blue", lw=1.2)
    ax.axhline(overall, color="tab:red", ls="--", lw=1.3,
               label=f"mean = {overall:.3f}")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, n - 1)
    ax.set_xlabel("Hover sample")
    ax.set_ylabel("Mean |action|")
    ax.text(n * 0.5, 0.95, "tanh saturation limit = 1.0",
            ha="center", va="top", fontsize=9, color="0.4")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "hover_action.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[hover]  N={n}  mean|action| = {overall:.4f}  ->  {out}")


def fig_gain_traj(model):
    env = PyBulletPIDTunerEnv(
        tune_axes=["roll", "pitch"], disturbance_axis="roll",
        max_steps=500, target_alt=1.0, init_noise=0.05,
        reward_w1=1.0, reward_w2=2.0, reward_w3=0.1, reward_w4=0.001,
        crash_penalty=50.0, stability_bonus=200.0,
        disturbance_step=DIST_STEP, disturbance_magnitude=0.17,
        disturbance_duration=DIST_DUR,
    )
    obs, info  = env.reset(seed=42)
    kp, ki, kd = [], [], []
    done = False
    while not done:
        action = model.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(action)
        kp.append(float(info["Kp_roll"]))
        ki.append(float(info["Ki_roll"]))
        kd.append(float(info["Kd_roll"]))
        done = term or trunc
    env.close()

    t      = np.arange(len(kp)) / CTRL_FREQ
    d0, d1 = DIST_STEP / CTRL_FREQ, (DIST_STEP + DIST_DUR) / CTRL_FREQ
    series = [("$K_p$", kp, PyBulletPIDTunerEnv.KP_DEFAULT, "tab:blue"),
              ("$K_i$", ki, PyBulletPIDTunerEnv.KI_DEFAULT, "tab:green"),
              ("$K_d$", kd, PyBulletPIDTunerEnv.KD_DEFAULT, "tab:orange")]

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
    for ax, (name, ser, dflt, col) in zip(axes, series):
        ax.axvspan(d0, d1, color="0.86", label="disturbance applied")
        ax.plot(t, ser, color=col, lw=1.6)
        ax.axhline(dflt, color="0.4", ls="--", lw=1.0, label="default value")
        ax.set_ylabel(name)
        ax.grid(True, ls=":", alpha=0.5)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "gain_trajectory.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[gains]  ->  {out}")
    for name, ser, dflt in [("Kp", kp, PyBulletPIDTunerEnv.KP_DEFAULT),
                            ("Ki", ki, PyBulletPIDTunerEnv.KI_DEFAULT),
                            ("Kd", kd, PyBulletPIDTunerEnv.KD_DEFAULT)]:
        a = np.asarray(ser)
        peak  = a.max()
        ssm   = a[300:400].mean()      # steady-state window of the sustained torque
        final = a[-1]
        print(f"   {name}: default={dflt:.5f}  peak={peak:.5f}  "
              f"steady-state={ssm:.5f}  final={final:.5f}")


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    model = TD3.load(MODEL, device="cpu")
    fig_hover(model)
    fig_gain_traj(model)


if __name__ == "__main__":
    main()
