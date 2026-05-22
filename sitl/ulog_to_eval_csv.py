#!/usr/bin/env python3
"""
ulog_to_eval_csv.py — Extract a standardized A/B comparison CSV from a PX4 .ulg.

One CSV per run (baseline OR rl). Columns:
    timestamp, roll, pitch, roll_rate, pitch_rate, kp, ki, kd, dist_active

Why a ulog extractor and not the MAVLink observer:
    rl_gain_tuner sets gains *inside* mc_rate_control (no PARAM_SET), so the RL
    gain trajectory exists ONLY in the ulog (topic rl_rate_gains, logged via
    rootfs/etc/logging/logger_topics.txt). A MAVLink observer cannot see it.
    For a BASELINE run (rl_gain_tuner stopped) rl_rate_gains is absent/invalid,
    so kp/ki/kd are filled with the constant training-default gains — which IS
    the baseline applied gain. Same schema both runs → directly comparable.

Usage:
    python sitl/ulog_to_eval_csv.py RUN.ulg -o baseline.csv
    python sitl/ulog_to_eval_csv.py RUN.ulg -o rl.csv --events events.csv

--events fills dist_active by anchoring the event log's first APPLY to the
first disturbance onset detected in the signal (no clock-mapping needed).
"""
import argparse
import csv
import sys

import numpy as np
from pyulog import ULog

# Training defaults (== airframe MC_*RATE_* == baseline applied gain).
KP_DEF, KI_DEF, KD_DEF = 0.171, 0.0086, 0.00171
ONSET_RATE_THR = 0.20   # rad/s on |rollrate|+|pitchrate| → first-disturbance anchor


def _quat_to_rp(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("ulog")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--events", default=None,
                    help="sitl_disturb.py --event-log CSV (fills dist_active).")
    ap.add_argument("--always-active", type=float, default=None,
                    metavar="SETTLE_S",
                    help="Mark dist_active=1 from SETTLE_S seconds onward "
                         "(no event log needed — use for constant-wind runs). "
                         "E.g. --always-active 10 skips first 10 s of takeoff.")
    args = ap.parse_args()

    u = ULog(args.ulog)
    D = {d.name: d for d in u.data_list}
    if "vehicle_attitude" not in D:
        sys.exit("[ulog2csv] vehicle_attitude not in log — nothing to extract.")

    qd = D["vehicle_attitude"]
    tq = np.array(qd.data["timestamp"]) / 1e6
    t0 = tq[0]
    t = tq - t0                                  # seconds since log start
    q = np.stack([np.array(qd.data[f"q[{i}]"]) for i in range(4)], 1)
    roll, pitch = _quat_to_rp(q)

    if "vehicle_angular_velocity" in D:
        av = D["vehicle_angular_velocity"]
        ta = np.array(av.data["timestamp"]) / 1e6 - t0
        rr = np.interp(t, ta, np.array(av.data["xyz[0]"]))
        pr = np.interp(t, ta, np.array(av.data["xyz[1]"]))
    else:
        rr = np.gradient(roll, t)
        pr = np.gradient(pitch, t)

    # Gains: rl_rate_gains if valid, else constant training defaults (baseline).
    kp = np.full_like(t, KP_DEF)
    ki = np.full_like(t, KI_DEF)
    kd = np.full_like(t, KD_DEF)
    if "rl_rate_gains" in D:
        g = D["rl_rate_gains"]
        tg = np.array(g.data["timestamp"]) / 1e6 - t0
        valid = np.array(g.data["valid"]).astype(int)
        if valid.any():
            # forward-fill the latest sample onto attitude timestamps
            idx = np.searchsorted(tg, t, side="right") - 1
            idx = np.clip(idx, 0, len(tg) - 1)
            gv = valid[idx] == 1
            kp = np.where(gv, np.array(g.data["kp"])[idx], KP_DEF)
            ki = np.where(gv, np.array(g.data["ki"])[idx], KI_DEF)
            kd = np.where(gv, np.array(g.data["kd"])[idx], KD_DEF)
            print(f"[ulog2csv] rl_rate_gains: {int(valid.sum())}/{len(valid)} "
                  f"valid samples → RL run")
        else:
            print("[ulog2csv] rl_rate_gains present but never valid → "
                  "baseline run (constant default gains)")
    else:
        print("[ulog2csv] no rl_rate_gains topic → baseline run "
              "(constant default gains)")

    # dist_active: event log OR always-active window.
    dist = np.zeros_like(t, dtype=int)
    if args.always_active is not None:
        dist[t >= args.always_active] = 1
        print(f"[ulog2csv] --always-active: dist_active=1 from t={args.always_active:.1f}s "
              f"({int((t >= args.always_active).sum())} samples)")
    elif args.events:
        ev = []
        with open(args.events) as f:
            for row in csv.DictReader(f):
                ev.append((row["event"], float(row["t_apply_s"]),
                           float(row["t_clear_s"]), float(row["torque_x"]),
                           float(row["torque_y"])))
        if ev:
            comb = np.abs(rr) + np.abs(pr)
            over = np.where(comb > ONSET_RATE_THR)[0]
            if len(over):
                t_onset = t[over[0]]
                ev_first_apply = min(e[1] for e in ev)
                shift = t_onset - ev_first_apply
                for _, ta_, tc_, _, _ in ev:
                    a, c = ta_ + shift, tc_ + shift
                    dist[(t >= a) & (t <= c)] = 1
                print(f"[ulog2csv] first disturbance onset @ {t_onset:.1f}s; "
                      f"event log shifted by {shift:+.1f}s; "
                      f"{len(ev)} windows marked")
            else:
                print("[ulog2csv] WARN: no onset detected "
                      f"(|r|+|p| never > {ONSET_RATE_THR}); dist_active=0")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "roll", "pitch", "roll_rate", "pitch_rate",
                    "kp", "ki", "kd", "dist_active"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}", f"{roll[i]:.6f}", f"{pitch[i]:.6f}",
                        f"{rr[i]:.6f}", f"{pr[i]:.6f}", f"{kp[i]:.7f}",
                        f"{ki[i]:.7f}", f"{kd[i]:.8f}", dist[i]])
    print(f"[ulog2csv] wrote {len(t)} rows → {args.out}")


if __name__ == "__main__":
    main()
