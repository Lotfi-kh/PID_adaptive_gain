#!/usr/bin/env python3
"""
extract_metrics.py — rich per-run extractor + metrics for one SITL flight.

Pulls everything the A/B claim needs from a single ulog (all topics are already
logged by logged_topics.cpp — no firmware change needed):

  attitude / rate response   vehicle_attitude, vehicle_angular_velocity
  TRUE rate-tracking error   vehicle_rates_setpoint  (the loop the NN controls)
  control effort + saturation vehicle_torque_setpoint, actuator_motors
  altitude trade-off         vehicle_local_position
  adapted gains              rl_rate_gains (or constant defaults for baseline)

Outputs:
  - a wide per-sample CSV over the disturbance+recovery window (-o)
  - one metrics row, printed and optionally appended to a master CSV (--append)

The disturbance window is found by anchoring the event log's first APPLY to the
first rate onset in the signal (same trick as ab_extract.py / ulog_to_eval_csv),
then everything is aligned so the first kick sits at t=0.

Usage:
    python sitl/extract_metrics.py RUN.ulg --events ev.csv -o wide.csv
    python sitl/extract_metrics.py RUN.ulg --events ev.csv \\
        --append master.csv --controller adaptive --seed 7 --tag pair
"""
import argparse
import csv
import os
import sys

import numpy as np
from pyulog import ULog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_compare import metrics as ab_metrics, windows as ab_windows, SETTLE_RAD  # noqa: E402
from ulog_to_eval_csv import _quat_to_rp                             # noqa: E402

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # numpy 2.x renamed it

KP_DEF, KI_DEF, KD_DEF = 0.171, 0.0086, 0.00171
ONSET_RATE_THR = 0.20    # rad/s on |rr|+|pr| → first-disturbance anchor
TAIL_S = 8.0             # window kept after the last clear (matches ab_compare)
SAT_HI, SAT_LO = 0.98, 0.02   # actuator_motors normalized [0,1] saturation bands


def _interp(D, topic, field, t, t0):
    """Interp a ulog field onto master clock t (seconds, offset by t0)."""
    d = D[topic]
    td = np.array(d.data["timestamp"]) / 1e6 - t0
    return np.interp(t, td, np.array(d.data[field]))


def load_run(ulog_path, events_path=None, always_active=None):
    """Build one aligned dict of every signal on a common clock.

    Returns (a, extra) where `a` is the eval-schema dict ab_compare expects
    (timestamp/roll/pitch/roll_rate/pitch_rate/kp/ki/kd/dist_active) and `extra`
    holds the richer signals on the same clock.
    """
    u = ULog(ulog_path)
    D = {d.name: d for d in u.data_list}
    if "vehicle_attitude" not in D:
        sys.exit("[metrics] vehicle_attitude not in log — nothing to extract.")

    qd = D["vehicle_attitude"]
    tq = np.array(qd.data["timestamp"]) / 1e6
    t0 = tq[0]
    t = tq - t0
    q = np.stack([np.array(qd.data[f"q[{i}]"]) for i in range(4)], 1)
    roll, pitch = _quat_to_rp(q)

    if "vehicle_angular_velocity" in D:
        rr = _interp(D, "vehicle_angular_velocity", "xyz[0]", t, t0)
        pr = _interp(D, "vehicle_angular_velocity", "xyz[1]", t, t0)
    else:
        rr, pr = np.gradient(roll, t), np.gradient(pitch, t)

    # adapted gains, else constant defaults (baseline)
    kp = np.full_like(t, KP_DEF); ki = np.full_like(t, KI_DEF); kd = np.full_like(t, KD_DEF)
    if "rl_rate_gains" in D:
        g = D["rl_rate_gains"]
        tg = np.array(g.data["timestamp"]) / 1e6 - t0
        valid = np.array(g.data["valid"]).astype(int)
        if valid.any():
            idx = np.clip(np.searchsorted(tg, t, side="right") - 1, 0, len(tg) - 1)
            gv = valid[idx] == 1
            kp = np.where(gv, np.array(g.data["kp"])[idx], KP_DEF)
            ki = np.where(gv, np.array(g.data["ki"])[idx], KI_DEF)
            kd = np.where(gv, np.array(g.data["kd"])[idx], KD_DEF)

    # richer signals (graceful if a topic is missing)
    extra = {}
    if "vehicle_rates_setpoint" in D:
        extra["rate_sp_roll"] = _interp(D, "vehicle_rates_setpoint", "roll", t, t0)
        extra["rate_sp_pitch"] = _interp(D, "vehicle_rates_setpoint", "pitch", t, t0)
    if "vehicle_torque_setpoint" in D:
        extra["tau_roll"] = _interp(D, "vehicle_torque_setpoint", "xyz[0]", t, t0)
        extra["tau_pitch"] = _interp(D, "vehicle_torque_setpoint", "xyz[1]", t, t0)
        extra["tau_yaw"] = _interp(D, "vehicle_torque_setpoint", "xyz[2]", t, t0)
    if "actuator_motors" in D:
        for i in range(4):
            extra[f"motor{i}"] = _interp(D, "actuator_motors", f"control[{i}]", t, t0)
    if "vehicle_local_position" in D:
        extra["z"] = _interp(D, "vehicle_local_position", "z", t, t0)
        extra["vz"] = _interp(D, "vehicle_local_position", "vz", t, t0)

    # dist_active: event log (onset-anchored) OR a fixed settle window
    dist = np.zeros_like(t, dtype=int)
    if always_active is not None:
        dist[t >= always_active] = 1
    elif events_path and os.path.exists(events_path):
        ev = list(csv.DictReader(open(events_path)))
        comb = np.abs(rr) + np.abs(pr)
        over = np.where(comb > ONSET_RATE_THR)[0]
        if ev and len(over):
            shift = t[over[0]] - min(float(r["t_apply_s"]) for r in ev)
            for r in ev:
                a_, c_ = float(r["t_apply_s"]) + shift, float(r["t_clear_s"]) + shift
                dist[(t >= a_) & (t <= c_)] = 1

    a = {"timestamp": t, "roll": roll, "pitch": pitch,
         "roll_rate": rr, "pitch_rate": pr,
         "kp": kp, "ki": ki, "kd": kd, "dist_active": dist}

    # align so the first disturbance rising edge = t=0 (mirror ab_compare.load)
    d = dist.astype(int)
    re = np.where((d[1:] == 1) & (d[:-1] == 0))[0]
    t_edge = t[re[0] + 1] if len(re) else t[0]
    a["timestamp"] = t - t_edge
    extra["_t"] = a["timestamp"]
    return a, extra


def rich_metrics(a, extra):
    """ab_compare metrics + tracking error, effort, saturation, altitude."""
    win = ab_windows(a)
    base, _ = ab_metrics(a, win)

    t = a["timestamp"]
    end = (max(c for _, c in win) + TAIL_S) if win else t[-1]
    m = (t >= 0) & (t <= end)
    tm = t[m]

    def iae(x):      # integral of |x| over the window, in signal·seconds
        return float(_trapz(np.abs(x[m]), tm))

    out = dict(base)

    # TRUE rate-tracking error (setpoint − actual) — the loop the NN controls
    if "rate_sp_roll" in extra:
        er = extra["rate_sp_roll"] - a["roll_rate"]
        ep = extra["rate_sp_pitch"] - a["pitch_rate"]
        out["IAE rate_roll"] = iae(er)
        out["IAE rate_pitch"] = iae(ep)
        out["RMS rate_err"] = float(np.sqrt(np.mean(er[m] ** 2 + ep[m] ** 2)))

    # control effort + peak torque
    if "tau_roll" in extra:
        out["effort_roll Nms"] = iae(extra["tau_roll"])
        out["effort_pitch Nms"] = iae(extra["tau_pitch"])
        out["peak|tau_roll|"] = float(np.abs(extra["tau_roll"][m]).max())
        out["peak|tau_pitch|"] = float(np.abs(extra["tau_pitch"][m]).max())

    # motor saturation — guards against a "free lunch" that's really clipping
    if "motor0" in extra:
        mot = np.stack([extra[f"motor{i}"][m] for i in range(4)], 1)
        sat = (mot >= SAT_HI) | (mot <= SAT_LO)
        out["motor_sat %"] = float(100.0 * np.any(sat, axis=1).mean())
        out["peak motor"] = float(mot.max())

    # altitude trade-off: did it hold height while recovering attitude?
    if "z" in extra:
        z = extra["z"]
        pre = z[(t >= -3.0) & (t < 0.0)]
        z_ref = float(np.median(pre)) if len(pre) else float(z[m][0])
        out["alt_dev m"] = float(np.abs(z[m] - z_ref).max())
        out["peak|vz|"] = float(np.abs(extra["vz"][m]).max())

    return out, m


def per_event_metrics(a, extra, events_path):
    """One row per disturbance event — magnitude, peak, recovery, rate-IAE.

    Pairs the aligned dist_active windows with the event-log magnitudes (both in
    apply-time order), so a single graduated --safe flight yields a magnitude
    sweep (0.15 and 0.40 single-axis, 0.15+0.15 and 0.20+0.20 combined) for free.
    """
    if not (events_path and os.path.exists(events_path)):
        return []
    ev = sorted(csv.DictReader(open(events_path)),
                key=lambda r: float(r["t_apply_s"]))
    win = ab_windows(a)                       # aligned (t_apply, t_clear) per event
    if not ev or not win:
        return []

    t, roll, pitch = a["timestamp"], a["roll"], a["pitch"]
    deg = 180.0 / np.pi
    have_rate = "rate_sp_roll" in extra
    rows = []
    n = min(len(ev), len(win))
    for i in range(n):
        ta, tc = win[i]
        t_next = win[i + 1][0] if i + 1 < len(win) else t[-1]   # don't bleed into next kick
        tx, ty = float(ev[i]["torque_x"]), float(ev[i]["torque_y"])
        seg = (t >= ta) & (t < t_next)
        if not seg.any():
            continue
        peak_roll = float(np.abs(roll[seg]).max() * deg)
        peak_pitch = float(np.abs(pitch[seg]).max() * deg)

        rseg = (t >= tc) & (t < t_next)        # recovery search, from this kick's clear
        ts, rs, ps = t[rseg], np.abs(roll[rseg]), np.abs(pitch[rseg])
        ok = np.where((rs < SETTLE_RAD) & (ps < SETTLE_RAD))[0]
        rec = float(ts[ok[0]] - tc) if len(ok) else float("nan")

        if have_rate:
            er = extra["rate_sp_roll"] - a["roll_rate"]
            ep = extra["rate_sp_pitch"] - a["pitch_rate"]
            iae = float(_trapz(np.abs(er[seg]) + np.abs(ep[seg]), t[seg]))
        else:
            iae = float("nan")

        rows.append({
            "event": ev[i]["event"], "torque_x": tx, "torque_y": ty,
            "mag": max(abs(tx), abs(ty)),
            "peak_roll_deg": peak_roll, "peak_pitch_deg": peak_pitch,
            "recovery_s": rec, "IAE_rate": iae,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("ulog")
    ap.add_argument("--events", default=None,
                    help="sitl_disturb.py --event-log CSV (defines the window)")
    ap.add_argument("--always-active", type=float, default=None,
                    help="Mark dist_active=1 from N s onward (no event log).")
    ap.add_argument("-o", "--out", default=None,
                    help="Wide per-sample CSV over the window (optional).")
    ap.add_argument("--append", default=None,
                    help="Append the one metrics row to this master CSV.")
    ap.add_argument("--append-events", default=None,
                    help="Per-event rows CSV (default: <append>_events.csv).")
    ap.add_argument("--no-events", action="store_true",
                    help="Skip writing the per-event (sweep) rows.")
    # provenance written alongside the metrics row
    ap.add_argument("--controller", default="")
    ap.add_argument("--seed", default="")
    ap.add_argument("--scale", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    a, extra = load_run(args.ulog, args.events, args.always_active)
    met, m = rich_metrics(a, extra)

    print(f"[metrics] {os.path.basename(args.ulog)}  ctrl={args.controller or '?'} "
          f"seed={args.seed or '?'} tag={args.tag or '?'}")
    for k, v in met.items():
        print(f"    {k:<18} {v:.5g}")

    # wide CSV (window only, restart t at 0)
    if args.out:
        t = a["timestamp"]
        keep = m
        cols = {"t": t[keep] - t[keep][0]}
        for k in ("roll", "pitch", "roll_rate", "pitch_rate", "kp", "ki", "kd",
                  "dist_active"):
            cols[k] = a[k][keep]
        for k, v in extra.items():
            if k != "_t":
                cols[k] = v[keep]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(cols))
            for i in range(int(keep.sum())):
                w.writerow([f"{cols[k][i]:.6g}" for k in cols])
        print(f"[metrics] wide CSV → {args.out}")

    # append one provenance+metrics row to the master CSV
    if args.append:
        meta = {"tag": args.tag, "seed": args.seed, "scale": args.scale,
                "controller": args.controller, "ulog": os.path.basename(args.ulog)}
        header = list(meta) + list(met)
        new = not os.path.exists(args.append) or os.path.getsize(args.append) == 0
        os.makedirs(os.path.dirname(args.append) or ".", exist_ok=True)
        with open(args.append, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow([meta[k] for k in meta] + [f"{met[k]:.6g}" for k in met])
        print(f"[metrics] row appended → {args.append}")

        # per-event rows → sibling CSV (the in-flight magnitude sweep)
        if not args.no_events:
            ev_rows = per_event_metrics(a, extra, args.events)
            if ev_rows:
                ev_path = (args.append_events or
                           os.path.splitext(args.append)[0] + "_events.csv")
                ev_meta = ["tag", "seed", "scale", "controller", "ulog"]
                ev_cols = list(ev_rows[0])
                ev_new = not os.path.exists(ev_path) or os.path.getsize(ev_path) == 0
                with open(ev_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if ev_new:
                        w.writerow(ev_meta + ev_cols)
                    for r in ev_rows:
                        w.writerow([meta[k] for k in ev_meta] +
                                   [f"{r[c]:.6g}" if isinstance(r[c], float) else r[c]
                                    for c in ev_cols])
                print(f"[metrics] {len(ev_rows)} event rows → {ev_path}")


if __name__ == "__main__":
    main()
