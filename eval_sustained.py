"""
eval_sustained.py, Sustained constant-torque disturbance evaluation

The disturbance grid (run_disturbance_grid.py) only applies a 5-step transient
kick. A transient is rejected by Kp/Kd alone, it tells you NOTHING about
integral action. This script applies a CONSTANT torque held for hundreds of
steps, which is the only test that exposes a collapsed Ki:

  control theory, under a constant disturbance, a P/D controller settles at a
  non-zero steady-state error about  disturbance / Kp. Only the I-term drives that
  error to zero. So if a policy has driven Ki->0, it MUST hold a standing tilt
  under sustained torque, while the fixed-gain baseline (default Ki) nulls it.

Episode layout (500 steps @ 48 Hz):
    [0,100)    settle, no torque
    [100,400)  SUSTAINED constant torque   (300 steps about  6.25 s)
    [400,500)  torque removed -> recovery measured here
  steady-state window = [300,400): last 2 s of the sustained phase, long after
  the initial transient, this is where a standing tilt is read off.

Conditions: low/med x roll/pitch, plus combined-medium roll+pitch.

Subjects (default): fixed-gain baseline + the 3 models under question.

Usage:
    python eval_sustained.py                 # default 3-model comparison
    python eval_sustained.py --episodes 5
    python eval_sustained.py --models frozen=PATH.zip mine=PATH2.zip
"""
import argparse
import os

import numpy as np
from stable_baselines3 import TD3

from envs import PyBulletPIDTunerEnv

CTRL_FREQ   = 48
MAX_STEPS   = 500
DIST_STEP   = 100               # torque on at step 100
DIST_DUR    = 300               # … held for 300 steps -> off at step 400
SS_LO, SS_HI = 300, 400         # steady-state window (last 2 s of sustained)
REC_THRESH  = 0.10              # rad/s
REC_WINDOW  = 10                # consecutive steps below threshold

KP_DEF = PyBulletPIDTunerEnv.KP_DEFAULT
KI_DEF = PyBulletPIDTunerEnv.KI_DEFAULT
KD_DEF = PyBulletPIDTunerEnv.KD_DEFAULT
KI_COLLAPSE = 0.10 * KI_DEF     # Ki below this ⇒ "collapsed"

# (label, dist_axis, magnitude N.m)
CONDITIONS = [
    ("low_roll",      "roll",  0.043),
    ("low_pitch",     "pitch", 0.043),
    ("med_roll",      "roll",  0.170),
    ("med_pitch",     "pitch", 0.170),
    ("combined_med",  "both",  0.170),
]

DEFAULT_MODELS = [
    ("frozen_1p05M", "results/frozen_joint_1p05M_shared3D/td3_pid_interrupted.zip"),
    ("sb20_810k",    "runs/2026-05-17_20-08-11/checkpoints/td3_pid_810000_steps.zip"),
    ("cur_810k",     "runs/2026-05-17_22-53-52/checkpoints/td3_pid_810000_steps.zip"),
]


def recovery_steps(rate_series, dist_end):
    """Steps after dist_end until |rate| < REC_THRESH for REC_WINDOW consecutive
    steps. None = never recovered within the trajectory."""
    if dist_end >= len(rate_series):
        return None
    post = np.abs(np.asarray(rate_series[dist_end:]))
    for i in range(len(post) - REC_WINDOW + 1):
        if (post[i:i + REC_WINDOW] < REC_THRESH).all():
            return i
    return None


def rollout(env, model, dist_axis, n_episodes, seed):
    """Deterministic rollout under a sustained constant torque. Returns aggregate
    metrics averaged over episodes."""
    zero = np.zeros(3, dtype=np.float32)
    dist_end = DIST_STEP + DIST_DUR     # step index where torque is removed

    ss_roll, ss_pitch = [], []
    rms_rr, rms_pr     = [], []
    max_roll, max_pit  = [], []
    recs, crashes      = [], []
    fkp = fki = fkd    = np.nan

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        roll_a, pitch_a, rr, pr = [], [], [], []
        done = False
        while not done:
            action = zero if model is None else model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(action)
            roll_a.append(np.deg2rad(info["roll_deg"]))
            pitch_a.append(np.deg2rad(info["pitch_deg"]))
            rr.append(float(info["roll_rate"]))
            pr.append(float(info["pitch_rate"]))
            done = term or trunc

        crashes.append(bool(info["crashed"]))
        fkp, fki, fkd = info["Kp_roll"], info["Ki_roll"], info["Kd_roll"]

        ra = np.abs(np.asarray(roll_a))
        pa = np.abs(np.asarray(pitch_a))
        # steady-state window may be truncated if the episode crashed early
        hi = min(SS_HI, len(ra))
        if hi > SS_LO:
            ss_roll.append(float(ra[SS_LO:hi].mean()))
            ss_pitch.append(float(pa[SS_LO:hi].mean()))
        else:                       # crashed before steady state -> worst case
            ss_roll.append(float(ra.max()) if ra.size else np.nan)
            ss_pitch.append(float(pa.max()) if pa.size else np.nan)

        rms_rr.append(float(np.sqrt(np.mean(np.square(rr)))))
        rms_pr.append(float(np.sqrt(np.mean(np.square(pr)))))
        max_roll.append(float(ra.max()) if ra.size else np.nan)
        max_pit.append(float(pa.max()) if pa.size else np.nan)

        if dist_axis == "roll":
            recs.append(recovery_steps(rr, dist_end))
        elif dist_axis == "pitch":
            recs.append(recovery_steps(pr, dist_end))
        else:
            comb = [max(abs(a), abs(b)) for a, b in zip(rr, pr)]
            recs.append(recovery_steps(comb, dist_end))

    valid_rec = [r for r in recs if r is not None]
    return dict(
        ss_roll_deg  = np.rad2deg(np.nanmean(ss_roll)),
        ss_pitch_deg = np.rad2deg(np.nanmean(ss_pitch)),
        rms_roll     = float(np.nanmean(rms_rr)),
        rms_pitch    = float(np.nanmean(rms_pr)),
        max_roll_deg = np.rad2deg(np.nanmean(max_roll)),
        max_pit_deg  = np.rad2deg(np.nanmean(max_pit)),
        recovery     = float(np.mean(valid_rec)) if valid_rec else None,
        rec_rate     = len(valid_rec) / n_episodes,
        crashes      = int(sum(crashes)),
        n            = n_episodes,
        kp=float(fkp), ki=float(fki), kd=float(fkd),
        ki_collapsed = bool(fki < KI_COLLAPSE),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="LABEL=PATH entries. Default: frozen + sb20_810k + cur_810k")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/sustained_eval")
    args = ap.parse_args()

    if args.models:
        models = []
        for m in args.models:
            if "=" not in m:
                ap.error(f"--models entry must be LABEL=PATH, got: {m}")
            lbl, pth = m.split("=", 1)
            models.append((lbl, pth))
    else:
        models = DEFAULT_MODELS

    for lbl, pth in models:
        if not os.path.isfile(pth):
            raise SystemExit(f"[SUSTAINED] model not found: {lbl} -> {pth}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[SUSTAINED] Constant-torque rejection, does Ki=0 actually hurt?")
    print(f"[SUSTAINED] Episode: settle[0,{DIST_STEP}) | TORQUE[{DIST_STEP},{DIST_STEP+DIST_DUR}) "
          f"| recover[{DIST_STEP+DIST_DUR},{MAX_STEPS})")
    print(f"[SUSTAINED] Steady-state window = [{SS_LO},{SS_HI}) | episodes={args.episodes}")
    print(f"[SUSTAINED] Defaults: Kp={KP_DEF:.4f} Ki={KI_DEF:.5f} Kd={KD_DEF:.6f} "
          f"| Ki<{KI_COLLAPSE:.5f} ⇒ collapsed\n")

    # subjects: fixed-gain baseline first, then each model
    subjects = [("baseline", None)] + [
        (lbl, TD3.load(pth, device="cpu")) for lbl, pth in models
    ]

    report = []
    def emit(s=""):
        print(s); report.append(s)

    for cond, dist_axis, mag in CONDITIONS:
        env = PyBulletPIDTunerEnv(
            tune_axes        = ["roll", "pitch"],
            disturbance_axis = dist_axis,
            max_steps        = MAX_STEPS,
            target_alt       = 1.0,
            init_noise       = 0.05,
            reward_w1=1.0, reward_w2=2.0, reward_w3=0.1, reward_w4=0.001,
            crash_penalty=50.0, stability_bonus=200.0,
            disturbance_step      = DIST_STEP,
            disturbance_magnitude = mag,
            disturbance_duration  = DIST_DUR,
        )

        emit("=" * 108)
        emit(f"CONDITION: {cond}   (axis={dist_axis}, constant torque={mag:.3f} N.m, "
             f"held {DIST_DUR} steps about  {DIST_DUR/CTRL_FREQ:.2f} s)")
        emit("=" * 108)
        emit(f"{'subject':<14}{'final Kp':>9}{'final Ki':>10}{'final Kd':>10}"
             f"{'Ki=0?':>7}{'ssRoll°':>9}{'ssPitch°':>9}"
             f"{'RMSrr':>8}{'RMSpr':>8}{'maxR°':>7}{'maxP°':>7}"
             f"{'recov':>7}{'crash':>6}")
        emit("-" * 108)

        for name, mdl in subjects:
            r = rollout(env, mdl, dist_axis, args.episodes, args.seed)
            rec = "n/a" if r["recovery"] is None else f"{r['recovery']:.0f}"
            emit(f"{name:<14}{r['kp']:>9.4f}{r['ki']:>10.5f}{r['kd']:>10.6f}"
                 f"{('YES' if r['ki_collapsed'] else 'no'):>7}"
                 f"{r['ss_roll_deg']:>9.2f}{r['ss_pitch_deg']:>9.2f}"
                 f"{r['rms_roll']:>8.4f}{r['rms_pitch']:>8.4f}"
                 f"{r['max_roll_deg']:>7.1f}{r['max_pit_deg']:>7.1f}"
                 f"{rec:>7}{r['crashes']:>4}/{r['n']}")
        env.close()
        emit("")


    emit("=" * 108)
    emit("READING THIS TABLE")
    emit("=" * 108)
    emit("• ssRoll°/ssPitch° = standing tilt during the LAST 2 s of constant torque.")
    emit("  Baseline keeps default Ki and should null it to a small residual.")
    emit("  A model with Ki=0 (YES) must hold a LARGER standing tilt on the")
    emit("  disturbed axis, that gap is the cost of a collapsed Ki.")
    emit("• If a Ki=0 model's ssError about  baseline's, then Ki=0 does NOT hurt in")
    emit("  practice (Kp is high enough to mask it). If it is much larger, Ki=0")
    emit("  is a real sustained-rejection defect and must be fixed before deploy.")
    emit("• recov = steps to settle after torque removal; crash = episodes lost.")
    emit("=" * 108)

    out = os.path.join(args.out_dir, "sustained_summary.txt")
    with open(out, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n[SUSTAINED] Summary saved -> {out}")


if __name__ == "__main__":
    main()
