"""
eval_hover_quietness.py — hover "do-nothing" check.

Loads each candidate model and runs it on 300 recorded SITL hover observations
(no disturbance). Prints mean|action| — how much the policy fidgets the gains
when the drone is already stable. Lower = quieter (c860k ~0.04, noiselat ~0.72,
noiselat-quiet ~0.17). Run:  PYTHONPATH=. python eval_hover_quietness.py
"""
import os, numpy as np, pandas as pd
from stable_baselines3 import TD3

HOVER_CSV = "results/observer_2026-05-16_19-24-48.csv"
OBS = [f"obs_{i:02d}" for i in range(12)]

CANDS = [
    ("c860k",          "results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip"),
    ("noiselat",       "results/frozen_joint_12d_noiselat/td3_pid_noiselat_best.zip"),
    ("noiselat-quiet", "results/frozen_joint_12d_noiselat_quiet/td3_pid_noiselat_quiet_best.zip"),
]

def main():
    df = pd.read_csv(HOVER_CSV)
    obs = df[OBS].to_numpy(dtype=np.float32)
    print(f"{'model':>16}  mean|action|  (N={len(obs)} hover samples)")
    for name, p in CANDS:
        if not os.path.exists(p):
            print(f"{name:>16}  MISSING ({p})"); continue
        m = TD3.load(p, device="cpu")
        a, _ = m.predict(obs, deterministic=True)
        print(f"{name:>16}  {float(np.abs(a).mean()):.4f}")

if __name__ == "__main__":
    main()
