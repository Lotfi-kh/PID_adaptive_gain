"""
eval_reaction_time.py — true reaction/recovery TIMING metric.

Runs one deterministic, noise-free episode (sustained 0.17 N·m roll torque at
step 100 for 300 steps) for the static baseline + each model, and measures, from
the true roll-rate trace:
  peak       — peak |roll-rate| in the 1.0 s onset transient (rad/s)
  t_peak     — onset -> peak (s)
  IAE_on     — integral of |roll-rate| over the onset transient (rad)  [reaction]
  arrest     — onset -> rate stays below settle band for rest of sustained window
  IAE_rec    — integral of |roll-rate| over 3 s after torque removal (rad)
  recov      — torque-off -> rate back below settle band & stays (s)   [recovery]
Lower IAE / time = faster, cleaner rejection.
Run:  PYTHONPATH=. python eval_reaction_time.py
"""
import numpy as np
from stable_baselines3 import TD3
from envs import PyBulletPIDTunerEnv

DIST_STEP, DIST_DUR, CTRL = 100, 300, 48
OFF = DIST_STEP + DIST_DUR; SETTLE = 0.03; dt = 1.0 / CTRL
ONSET_W = 48     # 1.0 s onset transient window
REC_W = 144      # 3.0 s post-removal window

CTRLS = [
    ("static baseline", None),
    ("c860k",           "results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip"),
    ("noiselat",        "results/frozen_joint_12d_noiselat/td3_pid_noiselat_best.zip"),
    ("noiselat-quiet",  "results/frozen_joint_12d_noiselat_quiet/td3_pid_noiselat_quiet_best.zip"),
]

def episode(predict):
    env = PyBulletPIDTunerEnv(
        tune_axes=["roll", "pitch"], disturbance_axis="roll",
        max_steps=500, target_alt=1.0, init_noise=0.05,
        reward_w1=1.0, reward_w2=2.0, reward_w3=0.1, reward_w4=0.001,
        crash_penalty=50.0, stability_bonus=200.0,
        disturbance_step=DIST_STEP, disturbance_magnitude=0.17, disturbance_duration=DIST_DUR)
    obs, _ = env.reset(seed=42); rr = []; done = False
    while not done:
        obs, _, t, tr, info = env.step(predict(obs)); rr.append(info["roll_rate"]); done = t or tr
    env.close(); return np.asarray(rr)

def metrics(rr):
    a = np.abs(rr)
    on = a[DIST_STEP:DIST_STEP+ONSET_W]
    peak = float(on.max()); t_peak = int(on.argmax()) * dt; iae_on = float(on.sum() * dt)
    sus = a[DIST_STEP:OFF]; arrest = np.nan
    for k in range(len(sus)):
        if np.all(sus[k:] < SETTLE): arrest = k * dt; break
    post = a[OFF:OFF+REC_W]; iae_rec = float(post.sum() * dt); rec = np.nan
    for k in range(len(post)):
        if np.all(post[k:] < SETTLE): rec = k * dt; break
    return peak, t_peak, iae_on, arrest, iae_rec, rec

def main():
    print(f"onset transient=1.0s | settle band={SETTLE} rad/s | IAE=int|roll_rate| (rad)\n")
    print(f"{'controller':>16} | {'peak':>6} {'t_peak':>6} {'IAE_on':>7} {'arrest':>7} | {'IAE_rec':>7} {'recov':>6}")
    print("-" * 74)
    for name, src in CTRLS:
        if src is None:
            pred = lambda obs: np.zeros(3, dtype=np.float32)
        else:
            m = TD3.load(src, device="cpu"); pred = lambda obs, m=m: m.predict(obs, deterministic=True)[0]
        pk, tp, io, ar, ir, rc = metrics(episode(pred))
        g = lambda x: f"{x:.3f}" if not np.isnan(x) else " n/a"
        print(f"{name:>16} | {pk:6.3f} {tp:6.3f} {io:7.3f} {g(ar):>7} | {ir:7.3f} {g(rc):>6}")

if __name__ == "__main__":
    main()
