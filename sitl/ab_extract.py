#!/usr/bin/env python3
"""
ab_extract.py — pull the disturbance-test window out of a SITL ulog.

Reads rl_gain_compare (active vs baseline gains, active vs shadow torque) and
writes a CSV covering ONLY the disturbance window, not the whole flight. Finds
the newest ulog by itself so you don't have to go looking for the file.

Usage:
    python sitl/ab_extract.py                       # newest ulog, test_results/ab_events.csv
    python sitl/ab_extract.py RUN.ulg --events ev.csv -o out.csv

The event log times are seconds since sitl_disturb.py started, so they don't
share a clock with the ulog. We anchor them by matching the first APPLY to the
first rate onset in the signal (same trick as ulog_to_eval_csv.py).
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
from pyulog import ULog

LOG_GLOB = os.path.expanduser(
    "~/PX4-Autopilot/build/px4_sitl_default/rootfs/log/*/*.ulg")
RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results"))
ONSET_RATE_THR = 0.20   # rad/s on |rollrate|+|pitchrate|, marks first kick
PRE_MARGIN  = 3.0       # seconds kept before the first kick
TAIL_MARGIN = 16.0      # seconds kept after the last clear (covers the calm tail)

CMP_FIELDS = ["active_kp", "active_ki", "active_kd",
              "base_kp", "base_ki", "base_kd",
              "active_torque_roll", "active_torque_pitch",
              "shadow_torque_roll", "shadow_torque_pitch", "rl_active"]


def newest_ulog():
    files = glob.glob(LOG_GLOB)
    if not files:
        sys.exit(f"[ab] no ulog found under {LOG_GLOB}")
    f = max(files, key=os.path.getmtime)
    print(f"[ab] newest ulog: {f}")
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("ulog", nargs="?", default=None,
                    help="ulog path (default: newest under the SITL log dir)")
    ap.add_argument("--events", default=os.path.join(RESULTS_DIR, "ab_events.csv"),
                    help=f"sitl_disturb.py --event-log CSV "
                         f"(default {os.path.join(RESULTS_DIR, 'ab_events.csv')})")
    ap.add_argument("-o", "--out", default=os.path.join(RESULTS_DIR, "ab_test.csv"))
    args = ap.parse_args()

    ulog_path = args.ulog or newest_ulog()
    u = ULog(ulog_path)
    D = {d.name: d for d in u.data_list}

    if "rl_gain_compare" not in D:
        sys.exit("[ab] rl_gain_compare not in log — fly with the updated firmware "
                 "and confirm the logger registers the topic.")

    cmp = D["rl_gain_compare"]
    tc_raw = np.array(cmp.data["timestamp"]) / 1e6
    t0 = tc_raw[0]
    tc = tc_raw - t0

    # attitude of the actual (adaptive) flight, on the compare clock.
    # Shadow run, so there is no baseline attitude — the baseline never flew.
    roll_deg = pitch_deg = None
    if "vehicle_attitude" in D:
        att = D["vehicle_attitude"]
        tqa = np.array(att.data["timestamp"]) / 1e6 - t0
        q = np.stack([np.array(att.data[f"q[{i}]"]) for i in range(4)], 1)
        w_, x_, y_, z_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        roll = np.degrees(np.arctan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_ * x_ + y_ * y_)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w_ * y_ - z_ * x_), -1.0, 1.0)))
        roll_deg = np.interp(tc, tqa, roll)
        pitch_deg = np.interp(tc, tqa, pitch)

    # rate onset, for anchoring the event log to the ulog clock
    if "vehicle_angular_velocity" in D:
        av = D["vehicle_angular_velocity"]
        ta = np.array(av.data["timestamp"]) / 1e6 - t0
        rr = np.array(av.data["xyz[0]"])
        pr = np.array(av.data["xyz[1]"])
        comb = np.abs(rr) + np.abs(pr)
        over = np.where(comb > ONSET_RATE_THR)[0]
        t_onset = ta[over[0]] if len(over) else None
    else:
        t_onset = None

    # window from the event log, shifted onto the ulog clock
    win_lo, win_hi = tc[0], tc[-1]
    if os.path.exists(args.events) and t_onset is not None:
        ev = list(csv.DictReader(open(args.events)))
        if ev:
            applies = [float(r["t_apply_s"]) for r in ev]
            clears  = [float(r["t_clear_s"]) for r in ev]
            shift = t_onset - min(applies)
            win_lo = min(applies) + shift - PRE_MARGIN
            win_hi = max(clears)  + shift + TAIL_MARGIN
            print(f"[ab] onset @ {t_onset:.1f}s, event log shifted {shift:+.1f}s")
    else:
        print("[ab] no event log or no onset — keeping the whole log")

    keep = (tc >= win_lo) & (tc <= win_hi)
    n = int(keep.sum())
    if n == 0:
        sys.exit("[ab] window is empty — check the event log matches this flight")

    cols = {f: np.array(cmp.data[f])[keep] for f in CMP_FIELDS}
    out_fields = list(CMP_FIELDS)
    if roll_deg is not None:
        cols["roll_deg"] = roll_deg[keep]
        cols["pitch_deg"] = pitch_deg[keep]
        out_fields += ["roll_deg", "pitch_deg"]
    t_out = tc[keep] - tc[keep][0]   # restart at 0 for the window

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + out_fields)
        for i in range(n):
            w.writerow([f"{t_out[i]:.4f}"] + [f"{cols[c][i]:.7g}" for c in out_fields])

    # summary
    rl_on = cols["rl_active"].astype(bool)
    dkp = np.abs(cols["active_kp"] - cols["base_kp"])
    dtau = np.abs(cols["active_torque_roll"] - cols["shadow_torque_roll"])
    print(f"[ab] window {win_hi - win_lo:.1f}s, {n} samples → {args.out}")
    print(f"[ab] rl_active: {int(rl_on.sum())}/{n} samples")
    print(f"[ab] max |active_kp - base_kp| = {dkp.max():.4f} "
          f"(baseline kp {cols['base_kp'][0]:.4f})")
    print(f"[ab] max |active - shadow| roll torque = {dtau.max():.4f}")
    if dkp.max() < 1e-4:
        print("[ab] NOTE: gains never diverged — was the drone disturbed while armed?")


if __name__ == "__main__":
    main()
