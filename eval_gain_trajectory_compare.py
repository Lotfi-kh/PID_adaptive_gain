"""
eval_gain_trajectory_compare.py — gain behaviour under a sustained torque.

Runs one deterministic, noise-free episode (sustained 0.17 N·m roll torque applied
at step 100 for 300 steps) for each candidate and prints how it drives the roll
gains: Kp peak/steady/final, Ki peak + min-during-torque (the Ki-collapse check),
Kd. Same episode config as make_thesis_figs_noiselat.py's gain figure.
Run:  PYTHONPATH=. python eval_gain_trajectory_compare.py
"""
import numpy as np
from stable_baselines3 import TD3
from envs import PyBulletPIDTunerEnv

DIST_STEP, DIST_DUR, CTRL = 100, 300, 48

CANDS = [
    ("noiselat-quiet",     "results/frozen_joint_12d_noiselat_quiet/td3_pid_noiselat_quiet_best.zip"),
    ("noiselat(deployed)", "results/frozen_joint_12d_noiselat/td3_pid_noiselat_best.zip"),
    ("c860k",              "results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip"),
]

def rollout(model):
    env = PyBulletPIDTunerEnv(
        tune_axes=["roll", "pitch"], disturbance_axis="roll",
        max_steps=500, target_alt=1.0, init_noise=0.05,
        reward_w1=1.0, reward_w2=2.0, reward_w3=0.1, reward_w4=0.001,
        crash_penalty=50.0, stability_bonus=200.0,
        disturbance_step=DIST_STEP, disturbance_magnitude=0.17, disturbance_duration=DIST_DUR)
    obs, _ = env.reset(seed=42); kp, ki, kd = [], [], []; done = False
    while not done:
        a = model.predict(obs, deterministic=True)[0]
        obs, _, t, tr, info = env.step(a)
        kp.append(info["Kp_roll"]); ki.append(info["Ki_roll"]); kd.append(info["Kd_roll"])
        done = t or tr
    env.close(); return map(np.asarray, (kp, ki, kd))

def main():
    KP0, KI0, KD0 = (PyBulletPIDTunerEnv.KP_DEFAULT, PyBulletPIDTunerEnv.KI_DEFAULT,
                     PyBulletPIDTunerEnv.KD_DEFAULT)
    print(f"defaults: Kp={KP0:.4f} Ki={KI0:.5f} Kd={KD0:.5f}\n")
    for name, p in CANDS:
        kp, ki, kd = rollout(TD3.load(p, device="cpu"))
        print(f"=== {name} ===")
        for gn, a in [("Kp", kp), ("Ki", ki), ("Kd", kd)]:
            print(f"  {gn}: peak={a.max():.5f}  steady(torque)={a[300:400].mean():.5f}  final={a[-1]:.5f}")
        ki_min = ki[DIST_STEP:DIST_STEP+DIST_DUR].min()
        print(f"  Ki min during torque = {ki_min:.5f}  (default {KI0:.5f}) "
              f"{'COLLAPSE' if ki_min < KI0*0.5 else 'alive'}\n")

if __name__ == "__main__":
    main()
