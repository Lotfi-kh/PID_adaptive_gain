#!/usr/bin/env python3
"""
eval_offnominal_ab.py — paired A/B of an adaptive RL model vs the STATIC baseline
gains, under OFF-NOMINAL dynamics (randomized mass/inertia), in PyBullet.

This is the regime where adaptation can actually beat fixed gains: the static
gains are tuned for the nominal airframe, so on a heavier / different-inertia
drone they are mismatched, while the RL policy (trained across the variation)
can re-tune. Each seed draws ONE dynamics perturbation and the SAME transient
disturbance, then runs both controllers on it (paired design):

  baseline  = action 0 every step  → gains frozen at the training defaults (the
              static PID the firmware ships).
  adaptive  = model.predict(obs)   → the RL policy adjusts the gains live.

Reports the three things we care about, baseline vs adaptive, with the paired
delta: recovery time, control effort (∫τ²), and stability (RMS attitude).

Usage:
    python eval_offnominal_ab.py MODEL.zip [--seeds 40] [--nominal]
    python eval_offnominal_ab.py runs/<run>/td3_pid_final.zip --seeds 60
    # --nominal runs the same A/B WITHOUT dynamics randomization (sanity: expect a tie)
"""
import argparse
import numpy as np
from stable_baselines3 import TD3
from envs.pybullet_pid_tuner_env import PyBulletPIDTunerEnv

SETTLE_DEG = 3.0     # |attitude| below this counts as "recovered"


def run_episode(env, model, seed, max_steps):
    """One episode; returns metrics dict. model=None → static baseline (zero action)."""
    obs, info = env.reset(seed=seed)
    roll = []; pitch = []; eff = []; dist_on = []
    crashed = False
    for _ in range(max_steps):
        if model is None:
            a = np.zeros(3, dtype=np.float32)
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        roll.append(info["roll_deg"]); pitch.append(info["pitch_deg"])
        eff.append(info["effort_tau_sq"]); dist_on.append(info["disturbance_active"])
        if term:
            crashed = True; break
        if trunc:
            break
    roll = np.array(roll); pitch = np.array(pitch)
    att = np.maximum(np.abs(roll), np.abs(pitch))
    dt = env.CTRL_TIMESTEP

    # Recovery time: from the last disturbance-active step to the first step the
    # attitude stays below SETTLE_DEG to the end. If never settles → full window.
    dist_on = np.array(dist_on, dtype=bool)
    onset = np.argmax(dist_on) if dist_on.any() else 0
    last_on = (len(dist_on) - 1 - np.argmax(dist_on[::-1])) if dist_on.any() else 0
    rec_steps = len(att) - last_on
    for k in range(last_on, len(att)):
        if np.all(att[k:] < SETTLE_DEG):
            rec_steps = k - last_on; break
    return {
        "peak_att":  float(att[onset:].max()) if len(att) > onset else float(att.max()),
        "rms_att":   float(np.sqrt(np.mean(att**2))),
        "recov_s":   float(rec_steps * dt),
        "effort":    float(np.sum(eff) * dt),     # ∫τ² over the episode
        "crash":     int(crashed),
        "mass_frac": float(info["mass_frac"]),
        "inertia_frac": float(info["inertia_frac"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--nominal", action="store_true",
                    help="Disable dynamics randomization (sanity check; expect a tie).")
    # Realistic-system properties: a policy trained WITH noise+latency must be
    # judged WITH them (high gains amplify noise, so a clean-env test is unfair to
    # a conservative policy and rewards the high-gain exploit). Match training.
    ap.add_argument("--sensor-noise-att", type=float, default=0.0)
    ap.add_argument("--sensor-noise-rate", type=float, default=0.0)
    ap.add_argument("--control-latency-steps", type=int, default=0)
    args = ap.parse_args()

    model = TD3.load(args.model, device="cpu")
    # Strong transient kicks (recovery episodes) under off-nominal dynamics — the
    # kick must be big enough to cause a real excursion so recovery time is
    # meaningful (the env default range can draw near-zero kicks).
    env = PyBulletPIDTunerEnv(
        tune_axes=["roll", "pitch"], disturbance_axis="random",
        max_steps=args.max_steps, randomize_disturbance=True,
        disturbance_magnitude_range=(0.18, 0.30),
        disturbance_duration_range=(8, 16),
        disturbance_step_range=(60, 90),
        randomize_dynamics=not args.nominal,
        sensor_noise_att=args.sensor_noise_att,
        sensor_noise_rate=args.sensor_noise_rate,
        control_latency_steps=args.control_latency_steps,
    )

    keys = ["peak_att", "rms_att", "recov_s", "effort"]
    B = {k: [] for k in keys}; A = {k: [] for k in keys}
    bc = ac = 0
    for sd in range(1000, 1000 + args.seeds):
        rb = run_episode(env, None,  sd, args.max_steps)    # baseline (static gains)
        ra = run_episode(env, model, sd, args.max_steps)    # adaptive (same seed/dynamics/kick)
        for k in keys:
            B[k].append(rb[k]); A[k].append(ra[k])
        bc += rb["crash"]; ac += ra["crash"]
    env.close()

    from scipy.stats import wilcoxon
    regime = "NOMINAL (sanity)" if args.nominal else "OFF-NOMINAL (±mass/inertia)"
    print(f"\n[A/B] {args.seeds} paired episodes — {regime}")
    print(f"[A/B] model: {args.model}")
    print(f"{'metric':>12} | {'baseline':>9} | {'adaptive':>9} | {'impr%':>7} | {'p':>7} | {'win':>5}")
    print("-" * 64)
    label = {"peak_att": "peak att °", "rms_att": "RMS att °",
             "recov_s": "recovery s", "effort": "effort 1e3·∫τ²"}
    scale = {"peak_att": 1.0, "rms_att": 1.0, "recov_s": 1.0, "effort": 1e3}
    for k in keys:
        b = np.array(B[k]); a = np.array(A[k])
        impr = 100.0 * (b.mean() - a.mean()) / (abs(b.mean()) + 1e-12)  # lower=better
        win = float(np.mean(a < b))
        try:
            p = wilcoxon(b, a).pvalue if np.any(b != a) else 1.0
        except ValueError:
            p = 1.0
        s = scale[k]
        print(f"{label[k]:>12} | {b.mean()*s:>9.3f} | {a.mean()*s:>9.3f} | "
              f"{impr:>+6.1f} | {p:>7.4f} | {win:>5.2f}")
    print(f"\n crashes:  baseline {bc}/{args.seeds}   adaptive {ac}/{args.seeds}")
    print(" (impr% > 0 and win > 0.5 = adaptive better; lower is better on all 4)")


if __name__ == "__main__":
    main()
