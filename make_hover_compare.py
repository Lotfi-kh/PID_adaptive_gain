"""
make_hover_compare.py — hover-quietness comparison figure for the thesis.

Overlays the per-sample gain-adjustment magnitude of the hardened model
(noiselat) and the quiet model (noiselat-quiet) over the SAME recorded hover
states, to show directly how much calmer the quiet model is in undisturbed hover.

Writes ~/thesis/images/hover_action_compare.png
Evaluation of the FROZEN models only — no training. Run from rl_pid_tuner/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3

HOVER_CSV = "results/observer_2026-05-16_19-24-48.csv"
IMG_DIR   = os.path.expanduser("~/thesis/images")
OBS_COLS  = [f"obs_{i:02d}" for i in range(12)]

MODELS = [
    ("hardened model (noiselat)",      "results/frozen_joint_12d_noiselat/td3_pid_noiselat_best.zip",            "tab:orange"),
    ("quiet model (noiselat-quiet)",   "results/frozen_joint_12d_noiselat_quiet/td3_pid_noiselat_quiet_best.zip", "tab:blue"),
]


def main():
    df  = pd.read_csv(HOVER_CSV)
    obs = df[OBS_COLS].to_numpy(dtype=np.float32)
    n   = len(obs)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for label, path, color in MODELS:
        model   = TD3.load(path, device="cpu")
        act, _  = model.predict(obs, deterministic=True)     # (N, 3)
        mag     = np.abs(act).mean(axis=1)                   # per-sample mean |action|
        overall = float(np.abs(act).mean())
        ax.plot(np.arange(n), mag, color=color, lw=1.2, alpha=0.9,
                label=f"{label}   (mean = {overall:.3f})")
        ax.axhline(overall, color=color, ls="--", lw=1.3)
        print(f"{label:34s} mean|action| = {overall:.4f}")

    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, n - 1)
    ax.set_xlabel("Hover sample")
    ax.set_ylabel("Mean gain adjustment  |action|")
    ax.text(n * 0.015, 0.965, "tanh saturation limit = 1.0",
            ha="left", va="top", fontsize=9, color="0.4")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "hover_action_compare.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
