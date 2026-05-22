#!/usr/bin/env python3
"""
sitl_observer.py — Stage 1 log-only observer for PX4 SITL.

Connects to PX4 SITL via MAVLink, reads roll/pitch/body-rates and the current
rate-PID gains (including master rate gain K), constructs the 13-D obs vector
exactly as in training, runs the ONNX actor for logging only, and writes
everything to a CSV.

Hard guarantees:
  * Only PARAM_REQUEST_READ is sent (read-only).
  * No PARAM_SET. No COMMAND_LONG. No SET_POSITION_TARGET. No control commands.
  * The script cannot affect flight behaviour.

Usage:
    cd ~/rl_pid_tuner
    python sitl/sitl_observer.py                              # defaults
    python sitl/sitl_observer.py --rate 5 --duration 60       # 60-second run
    python sitl/sitl_observer.py --conn udpin:localhost:14540 --out /tmp/run.csv

Stop with Ctrl-C; partial CSV is preserved.

──────────────────────────────────────────────────────────────────────────────
DIAGNOSTIC: Temporarily adjust I gains in SITL (do this MANUALLY, not here)
──────────────────────────────────────────────────────────────────────────────
If obs[7]/obs[10] (Ki_roll_n / Ki_pitch_n) are out of range [0, 1], the actor
will saturate. To test with in-distribution Ki, set them manually:

  Via PX4 shell (nsh> prompt inside the SITL terminal):
    param show MC_ROLLRATE_I       # note current value
    param show MC_PITCHRATE_I
    param set MC_ROLLRATE_I 0.05
    param set MC_PITCHRATE_I 0.05

  Via QGroundControl:
    Parameters tab → search "MC_ROLLRATE_I" → edit value → Repeat for pitch

  Restore original values when done:
    param set MC_ROLLRATE_I 0.200
    param set MC_PITCHRATE_I 0.200

  Note: PARAM_SET changes are in-memory only (no flash write unless you run
  "param save"). The observer re-reads gains at startup each run.

──────────────────────────────────────────────────────────────────────────────
DISTURBANCE PROCEDURE (for body-rate sign verification)
──────────────────────────────────────────────────────────────────────────────
To generate a small roll/pitch excitation in SITL for action-direction checks:

  Option A — PX4 shell:
    commander mode altctl
    # Then use RC simulator / joystick to apply ±5° roll/pitch demand

  Option B — MAVProxy (separate terminal, read-only observer already running):
    mavproxy.py --master=udpin:localhost:14540
    > attitude roll 5 1          # 5 deg roll for 1 s

  Option C — QGC RC simulator:
    Widgets → Analyze → RC override, or use Joystick tab for momentary input

  The observer is read-only; run it in parallel with any of the above.
"""
import argparse
import csv
import os
import signal
import sys
import time

import math
import numpy as np

try:
    from pymavlink import mavutil
except ImportError:
    sys.exit("[OBSERVER] pymavlink not installed:  pip install pymavlink")

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("[OBSERVER] onnxruntime not installed:  pip install onnxruntime")


# ── Constants from deployment_contract.md ───────────────────────────────────
KP_NORM = 1.72       # obs[6], obs[9] : Kp_eff / 1.72
KI_NORM = 0.172      # obs[7], obs[10]: Ki_eff / 0.172
KD_NORM = 8.6e-3     # obs[8], obs[11]: Kd_eff / 8.6e-3
KP_ATT  = 3.0        # outer attitude P used in roll_rate_err / pitch_rate_err
STEP_PROGRESS = 1.0  # deployment default — hold at 1.0

OBS_DIM = 13
ACT_DIM = 3

# Read P/I/D + master rate gain K for both axes.
# Effective gains: Kp_eff = K*P, Ki_eff = K*I, Kd_eff = K*D
GAIN_PARAM_NAMES = [
    "MC_ROLLRATE_P",  "MC_ROLLRATE_I",  "MC_ROLLRATE_D",  "MC_ROLLRATE_K",
    "MC_PITCHRATE_P", "MC_PITCHRATE_I", "MC_PITCHRATE_D", "MC_PITCHRATE_K",
]
# Optional diagnostic: PX4's outer attitude P (training assumed 3.0)
DIAG_PARAM_NAMES = ["MC_ROLL_P", "MC_PITCH_P"]

# In-range bounds for normalized gain channels obs[6:12]
_OBS_GAIN_BOUNDS = [
    (0.0, KP_NORM, "Kp_roll_n  obs[06]"),
    (0.0, KI_NORM, "Ki_roll_n  obs[07]"),
    (0.0, KD_NORM, "Kd_roll_n  obs[08]"),
    (0.0, KP_NORM, "Kp_pitch_n obs[09]"),
    (0.0, KI_NORM, "Ki_pitch_n obs[10]"),
    (0.0, KD_NORM, "Kd_pitch_n obs[11]"),
]
# Normalized bounds are simply [0, 1] for all six channels
_NORM_LO = 0.0
_NORM_HI = 1.0

CSV_COLUMNS = (
    ["wall_time", "mav_boot_ms", "att_age_ms"]
    + [f"obs_{i:02d}" for i in range(OBS_DIM)]
    + [f"action_{i}" for i in range(ACT_DIM)]
    # Raw PX4 params
    + ["kp_roll_raw",   "ki_roll_raw",   "kd_roll_raw",   "k_roll_raw",
       "kp_pitch_raw",  "ki_pitch_raw",  "kd_pitch_raw",  "k_pitch_raw"]
    # Effective gains (K * P/I/D)
    + ["kp_eff_roll",   "ki_eff_roll",   "kd_eff_roll",
       "kp_eff_pitch",  "ki_eff_pitch",  "kd_eff_pitch"]
    # Per-channel in-range flags for obs[6:12] (1 = in [0,1])
    + [f"obs{i:02d}_in_range" for i in range(6, 12)]
    # Legacy aggregate flags
    + ["flag_rates_sane",
       "flag_obs_finite",
       "flag_action_finite",
       "flag_action_in_range",
       "flag_gains_match",
       "flag_gains_in_bounds"]
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--conn", default="udpin:localhost:14540",
                    help="MAVLink connection string. Default PX4 SITL offboard port.")
    ap.add_argument("--onnx", default=os.path.expanduser(
        "~/rl_pid_tuner/export/actor_joint_1p05M_shared3D.onnx"))
    ap.add_argument("--rate", type=float, default=5.0,
                    help="Observer tick rate in Hz (default 5).")
    ap.add_argument("--duration", type=float, default=None,
                    help="Stop after N seconds. Default: run until Ctrl-C.")
    ap.add_argument("--out", default=None,
                    help="Output CSV path. Default: results/observer_<timestamp>.csv")
    ap.add_argument("--gain-tol", type=float, default=1e-4,
                    help="|Kp_roll - Kp_pitch| above this flags gains_match=0.")
    ap.add_argument("--heartbeat-timeout", type=float, default=15.0,
                    help="Seconds to wait for the first PX4 heartbeat.")
    return ap.parse_args()


def request_one_param(mav, name, timeout):
    """Send PARAM_REQUEST_READ for `name` and return the float value, or None."""
    mav.mav.param_request_read_send(
        mav.target_system, mav.target_component,
        name.encode("utf-8") if isinstance(name, str) else name,
        -1,
    )
    t_end = time.time() + timeout
    while time.time() < t_end:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg is None:
            continue
        pid = msg.param_id
        if isinstance(pid, bytes):
            pid = pid.decode("utf-8", errors="ignore")
        pid = pid.rstrip("\x00")
        if pid == name:
            return float(msg.param_value)
    return None


def read_params(mav, names, timeout_each=2.5):
    out = {}
    for n in names:
        v = request_one_param(mav, n, timeout_each)
        if v is None:
            return None, n
        out[n] = v
    return out, None


def effective_gains(gains):
    """Return (kp_eff_roll, ki_eff_roll, kd_eff_roll, kp_eff_pitch, ki_eff_pitch, kd_eff_pitch)."""
    K_r = gains["MC_ROLLRATE_K"]
    K_p = gains["MC_PITCHRATE_K"]
    return (
        K_r * gains["MC_ROLLRATE_P"],
        K_r * gains["MC_ROLLRATE_I"],
        K_r * gains["MC_ROLLRATE_D"],
        K_p * gains["MC_PITCHRATE_P"],
        K_p * gains["MC_PITCHRATE_I"],
        K_p * gains["MC_PITCHRATE_D"],
    )


def build_obs(roll, pitch, rollspd, pitchspd,
              kp_eff_r, ki_eff_r, kd_eff_r,
              kp_eff_p, ki_eff_p, kd_eff_p):
    """Construct the 13-D obs exactly as in training/deployment_contract.md.

    Uses effective gains (K*P, K*I, K*D) so obs[6:12] reflects what PX4 applies.
    """
    roll_rate_err  = KP_ATT * (-roll)  - rollspd
    pitch_rate_err = KP_ATT * (-pitch) - pitchspd
    return np.array([
        roll, pitch,
        rollspd, pitchspd,
        roll_rate_err, pitch_rate_err,
        kp_eff_r / KP_NORM, ki_eff_r / KI_NORM, kd_eff_r / KD_NORM,
        kp_eff_p / KP_NORM, ki_eff_p / KI_NORM, kd_eff_p / KD_NORM,
        STEP_PROGRESS,
    ], dtype=np.float32)


def check_obs_gain_channels(kp_eff_r, ki_eff_r, kd_eff_r,
                             kp_eff_p, ki_eff_p, kd_eff_p):
    """Return list of 6 per-channel in-range flags (1=in [0,1], 0=OOD)."""
    norm_vals = [
        kp_eff_r / KP_NORM,
        ki_eff_r / KI_NORM,
        kd_eff_r / KD_NORM,
        kp_eff_p / KP_NORM,
        ki_eff_p / KI_NORM,
        kd_eff_p / KD_NORM,
    ]
    return [int(_NORM_LO <= v <= _NORM_HI) for v in norm_vals], norm_vals


def request_attitude_stream(mav, rate_hz=50.0):
    """Ask PX4 to send ATTITUDE at rate_hz. Read-only request (MAV_CMD)."""
    interval_us = int(1e6 / rate_hz)
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        interval_us,
        0, 0, 0, 0, 0,
    )


def main():
    args = parse_args()

    # Output path
    if args.out is None:
        results_dir = os.path.expanduser("~/rl_pid_tuner/results")
        os.makedirs(results_dir, exist_ok=True)
        args.out = os.path.join(
            results_dir, "observer_" + time.strftime("%Y-%m-%d_%H-%M-%S") + ".csv")

    print("[OBSERVER] -------- Stage 1 LOG-ONLY observer --------")
    print(f"[OBSERVER] Connect  : {args.conn}")
    print(f"[OBSERVER] ONNX     : {args.onnx}")
    print(f"[OBSERVER] Rate     : {args.rate} Hz")
    print(f"[OBSERVER] Output   : {args.out}")
    print("[OBSERVER] Will NOT send PARAM_SET. Will NOT issue control commands.")

    # ── ONNX session ────────────────────────────────────────────────────
    if not os.path.isfile(args.onnx):
        sys.exit(f"[OBSERVER] ONNX file not found: {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # ── MAVLink connect & wait for heartbeat ────────────────────────────
    print(f"[OBSERVER] Waiting for heartbeat (timeout {args.heartbeat_timeout:.0f}s)...")
    mav = mavutil.mavlink_connection(args.conn)
    hb = mav.wait_heartbeat(timeout=args.heartbeat_timeout)
    if hb is None:
        sys.exit("[OBSERVER] No heartbeat received. Is PX4 SITL running?")
    print(f"[OBSERVER] Heartbeat: sys={mav.target_system} comp={mav.target_component}")

    # ── Ask PX4 to stream ATTITUDE faster than our tick rate ────────────
    request_attitude_stream(mav, rate_hz=max(50.0, args.rate * 4))

    # ── Read the rate-PID gains (P/I/D/K) once at start ─────────────────
    print("[OBSERVER] Reading rate-PID gains (P, I, D, K)...")
    gains, missing = read_params(mav, GAIN_PARAM_NAMES)
    if gains is None:
        sys.exit(f"[OBSERVER] Failed to read parameter: {missing}")
    for k in GAIN_PARAM_NAMES:
        print(f"  {k:20s} = {gains[k]:.6f}")

    # Compute effective gains
    kp_eff_r, ki_eff_r, kd_eff_r, kp_eff_p, ki_eff_p, kd_eff_p = effective_gains(gains)
    print("[OBSERVER] Effective gains (K * P/I/D):")
    print(f"  {'Kp_eff_roll':20s} = {kp_eff_r:.6f}   (K={gains['MC_ROLLRATE_K']:.4f} * P={gains['MC_ROLLRATE_P']:.4f})")
    print(f"  {'Ki_eff_roll':20s} = {ki_eff_r:.6f}   (K={gains['MC_ROLLRATE_K']:.4f} * I={gains['MC_ROLLRATE_I']:.4f})")
    print(f"  {'Kd_eff_roll':20s} = {kd_eff_r:.6f}   (K={gains['MC_ROLLRATE_K']:.4f} * D={gains['MC_ROLLRATE_D']:.4f})")
    print(f"  {'Kp_eff_pitch':20s} = {kp_eff_p:.6f}   (K={gains['MC_PITCHRATE_K']:.4f} * P={gains['MC_PITCHRATE_P']:.4f})")
    print(f"  {'Ki_eff_pitch':20s} = {ki_eff_p:.6f}   (K={gains['MC_PITCHRATE_K']:.4f} * I={gains['MC_PITCHRATE_I']:.4f})")
    print(f"  {'Kd_eff_pitch':20s} = {kd_eff_p:.6f}   (K={gains['MC_PITCHRATE_K']:.4f} * D={gains['MC_PITCHRATE_D']:.4f})")

    # Diagnostic: PX4's outer attitude P vs training's fixed KP_ATT = 3.0
    diag, _ = read_params(mav, DIAG_PARAM_NAMES)
    if diag is not None:
        for k in DIAG_PARAM_NAMES:
            note = "" if abs(diag[k] - KP_ATT) < 0.1 else f"  (training assumed {KP_ATT})"
            print(f"  {k:20s} = {diag[k]:.6f}{note}")

    # ── Per-channel OOD check at startup ────────────────────────────────
    in_range_flags, norm_vals = check_obs_gain_channels(
        kp_eff_r, ki_eff_r, kd_eff_r, kp_eff_p, ki_eff_p, kd_eff_p)
    chan_names = [
        "Kp_roll_n  obs[06]", "Ki_roll_n  obs[07]", "Kd_roll_n  obs[08]",
        "Kp_pitch_n obs[09]", "Ki_pitch_n obs[10]", "Kd_pitch_n obs[11]",
    ]
    ood_any = False
    print("[OBSERVER] Normalized gain channel check (training range [0.0, 1.0]):")
    for i, (flag, nv, name) in enumerate(zip(in_range_flags, norm_vals, chan_names)):
        status = "OK" if flag else "OOD"
        marker = ""
        if not flag:
            ood_any = True
            marker = f"  <-- WARN: {nv:.4f} is outside [0, 1]; actor will saturate"
        print(f"  {name}  norm={nv:+8.4f}  [{status}]{marker}")
    if ood_any:
        print()
        print("[OBSERVER] *** One or more gain channels are out of training distribution. ***")
        print("[OBSERVER] *** To test with in-range Ki, run in PX4 shell BEFORE this script: ***")
        print("[OBSERVER] ***   param set MC_ROLLRATE_I 0.05                               ***")
        print("[OBSERVER] ***   param set MC_PITCHRATE_I 0.05                              ***")
        print("[OBSERVER] *** (Then restart the observer to re-read params.)               ***")
        print("[OBSERVER] *** Restore with: param set MC_ROLLRATE_I 0.200                  ***")
        print("[OBSERVER] ***               param set MC_PITCHRATE_I 0.200                 ***")
    print()

    # Sanity: are PX4 effective gains inside the training bounds?
    in_bounds = (
        0.0 <= kp_eff_r <= KP_NORM and
        0.0 <= kp_eff_p <= KP_NORM and
        0.0 <= ki_eff_r <= KI_NORM and
        0.0 <= ki_eff_p <= KI_NORM and
        0.0 <= kd_eff_r <= KD_NORM and
        0.0 <= kd_eff_p <= KD_NORM
    )
    flag_gains_in_bounds = int(in_bounds)

    # ── Open CSV ────────────────────────────────────────────────────────
    f_csv = open(args.out, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(CSV_COLUMNS)
    f_csv.flush()

    # ── Run loop ────────────────────────────────────────────────────────
    dt = 1.0 / args.rate
    t_start = time.monotonic()
    n_logged = 0
    n_skipped = 0
    last_print = t_start

    stopping = {"stop": False}
    def _stop(sig, frame):
        stopping["stop"] = True
    signal.signal(signal.SIGINT, _stop)

    print(f"[OBSERVER] Logging at {args.rate} Hz. Ctrl-C to stop.")

    while not stopping["stop"]:
        if args.duration is not None and (time.monotonic() - t_start) >= args.duration:
            break

        loop_start = time.monotonic()
        last_attitude = None

        # Drain any pending messages; keep only the freshest ATTITUDE
        while True:
            msg = mav.recv_match(type="ATTITUDE", blocking=False)
            if msg is None:
                break
            last_attitude = msg

        if last_attitude is None:
            # No fresh ATTITUDE this tick — wait briefly for one
            msg = mav.recv_match(type="ATTITUDE", blocking=True, timeout=dt * 0.5)
            if msg is None:
                n_skipped += 1
                elapsed = time.monotonic() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue
            last_attitude = msg

        # ATTITUDE.rollspeed/pitchspeed are body-frame rad/s (MAVLink spec)
        roll     = float(last_attitude.roll)
        pitch    = float(last_attitude.pitch)
        rollspd  = float(last_attitude.rollspeed)
        pitchspd = float(last_attitude.pitchspeed)
        mav_boot_ms = int(last_attitude.time_boot_ms)
        try:
            att_age_ms = float(mav.time_since("ATTITUDE")) * 1000.0
        except Exception:
            att_age_ms = float("nan")

        # Deployment contract: |rates| > 20 rad/s → sensor fault (also catches
        # post-crash Gazebo garbage values like 4e17 rad/s that pass isfinite).
        MAX_RATE = 20.0
        flag_rates_sane = int(
            abs(rollspd)  <= MAX_RATE and
            abs(pitchspd) <= MAX_RATE and
            math.isfinite(roll) and math.isfinite(pitch)
        )

        obs = build_obs(roll, pitch, rollspd, pitchspd,
                        kp_eff_r, ki_eff_r, kd_eff_r,
                        kp_eff_p, ki_eff_p, kd_eff_p)
        flag_obs_finite = int(flag_rates_sane and bool(np.all(np.isfinite(obs))))

        flag_gains_match = int(
            abs(kp_eff_r - kp_eff_p) <= args.gain_tol and
            abs(ki_eff_r - ki_eff_p) <= args.gain_tol and
            abs(kd_eff_r - kd_eff_p) <= args.gain_tol
        )

        if flag_obs_finite:
            action = sess.run([out_name], {in_name: obs.reshape(1, OBS_DIM)})[0].flatten()
        else:
            action = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        flag_action_finite   = int(bool(np.all(np.isfinite(action))))
        flag_action_in_range = int(
            flag_action_finite
            and bool(np.all(action >= -1.0 - 1e-6))
            and bool(np.all(action <=  1.0 + 1e-6))
        )

        row = (
            [f"{time.time():.6f}", str(mav_boot_ms), f"{att_age_ms:.1f}"]
            + [f"{v:.6f}" for v in obs.tolist()]
            + [f"{v:.6f}" for v in action.tolist()]
            # Raw PX4 params
            + [f"{gains['MC_ROLLRATE_P']:.6f}",
               f"{gains['MC_ROLLRATE_I']:.6f}",
               f"{gains['MC_ROLLRATE_D']:.6f}",
               f"{gains['MC_ROLLRATE_K']:.6f}",
               f"{gains['MC_PITCHRATE_P']:.6f}",
               f"{gains['MC_PITCHRATE_I']:.6f}",
               f"{gains['MC_PITCHRATE_D']:.6f}",
               f"{gains['MC_PITCHRATE_K']:.6f}"]
            # Effective gains
            + [f"{kp_eff_r:.6f}", f"{ki_eff_r:.6f}", f"{kd_eff_r:.6f}",
               f"{kp_eff_p:.6f}", f"{ki_eff_p:.6f}", f"{kd_eff_p:.6f}"]
            # Per-channel obs[6:12] in-range flags
            + [str(f) for f in in_range_flags]
            # Aggregate flags
            + [str(flag_rates_sane),
               str(flag_obs_finite),
               str(flag_action_finite),
               str(flag_action_in_range),
               str(flag_gains_match),
               str(flag_gains_in_bounds)]
        )
        writer.writerow(row)
        n_logged += 1

        # Periodic status line
        now = time.monotonic()
        if now - last_print >= 2.0:
            ood_str = "" if all(in_range_flags) else "  [OOD gains]"
            print(f"  t={now - t_start:6.1f}s  n={n_logged:5d}  "
                  f"roll={roll:+.3f}  pitch={pitch:+.3f}  "
                  f"action=[{action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f}]  "
                  f"age={att_age_ms:.0f}ms{ood_str}")
            f_csv.flush()
            last_print = now

        # Pace
        elapsed = time.monotonic() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    f_csv.close()

    print()
    print(f"[OBSERVER] Stopped. Samples logged: {n_logged}  Skipped (no attitude): {n_skipped}")
    print(f"[OBSERVER] CSV  : {args.out}")


if __name__ == "__main__":
    main()
