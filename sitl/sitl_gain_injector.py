#!/usr/bin/env python3
"""
sitl_gain_injector.py — Stage 2 SITL gain injection (PARAM_SET via MAVLink).

⚠️  SITL PROTOTYPING ONLY.
    PARAM_SET writes gains to PX4's parameter RAM. This is a temporary shortcut
    for SITL validation only. Final integration target: a dedicated uORB topic
    consumed by mc_rate_control — no MAVLink, no parameter system, no flash writes.

What this adds over Stage 1 (sitl_observer.py):
  • Tracks running Kp/Ki/Kd state
  • Applies DELTA_SCALE to ONNX actions → gain deltas
  • Clips gains to training bounds
  • Checks safety conditions before every update
  • Sends PARAM_SET for MC_ROLLRATE_P/I/D and MC_PITCHRATE_P/I/D

Safety rules (from deployment_contract.md) — gains are FROZEN when:
  • |roll| > 60° or |pitch| > 60°
  • |roll_rate| > 20 rad/s or |pitch_rate| > 20 rad/s (also catches crash garbage)
  • obs not finite

Usage:
    # Dry run — compute everything, print what PARAM_SET would be sent, don't send:
    python sitl/sitl_gain_injector.py --dry-run

    # Live gain injection at 5 Hz:
    python sitl/sitl_gain_injector.py

    # Slower/safer rate to start:
    python sitl/sitl_gain_injector.py --rate 2

Run sitl_disturb.py in a second terminal to apply physics disturbances.
Stop with Ctrl-C — gains are reset to training defaults on exit.
"""
import argparse
import csv
import math
import os
import signal
import sys
import time

import numpy as np

try:
    from pymavlink import mavutil
except ImportError:
    sys.exit("[INJECTOR] pymavlink not installed:  pip install pymavlink")

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("[INJECTOR] onnxruntime not installed:  pip install onnxruntime")


# ── Deployment contract constants ────────────────────────────────────────────
KP_NORM = 1.72
KI_NORM = 0.172
KD_NORM = 8.6e-3
KP_ATT  = 3.0
# step_prog REMOVED from the observation entirely (12-D model). The faked
# deployment constant that used to live here is gone — its deployment-only
# mismatch was the actual blocker and is now eliminated by construction.

KP_INIT = 0.171
KI_INIT = 0.0086
KD_INIT = 0.00171

# ── Deployment safety clamp ──────────────────────────────────────────────────
# Hard floor+ceiling on every gain, as a multiple of the training defaults.
# Backstop only: it does NOT make an unrecoverable disturbance recoverable —
# it bounds gain STATE so the NN can never (a) collapse Ki→0 (kills integral
# action) or (b) run Kp away (the 0.47 / Ki=0 divergence seen when the NN is
# given full authority through an unrecoverable sharp transient).
# c860k's validated operating range (Kp ≤0.23, Ki ≈0.008) sits well inside
# this band, so the clamp never interferes with normal adaptation.
CLAMP_LO_MULT = 0.5    # floor = 0.5× default  (Ki floor ⇒ integral never dies)
CLAMP_HI_MULT = 2.5    # ceiling = 2.5× default
KP_MIN, KP_MAX = CLAMP_LO_MULT * KP_INIT, CLAMP_HI_MULT * KP_INIT   # [0.0855, 0.4275]
KI_MIN, KI_MAX = CLAMP_LO_MULT * KI_INIT, CLAMP_HI_MULT * KI_INIT   # [0.00430, 0.02150]
KD_MIN, KD_MAX = CLAMP_LO_MULT * KD_INIT, CLAMP_HI_MULT * KD_INIT   # [0.000855, 0.004275]

# DELTA_SCALE from training (sized for 48 Hz).
# At 5 Hz the per-second change is 48/5 = 9.6× slower — intentionally conservative.
DELTA_SCALE = np.array([3.4e-2, 3.4e-3, 1.7e-4])

OBS_DIM = 12   # step_prog removed from the observation (12-D model: c860k)
ACT_DIM = 3

# Safety thresholds (deployment_contract.md)
MAX_ATT_RAD = math.radians(60.0)   # 60°
MAX_RATE_RAD = 20.0                 # rad/s

# Stability deadband — gains are HELD when ALL four conditions hold simultaneously.
# Derived from real SITL stable-hover data (300 samples, observer_2026-05-16_19-24-48.csv):
#   att p99 < 0.006 rad, rate p99 < 0.013 rad/s  →  thresholds sit ~16-19× above the noise floor.
ATT_STABLE_THR  = 0.10   # rad  (~5.7°)
RATE_STABLE_THR = 0.10   # rad/s

GAIN_PARAM_NAMES = [
    "MC_ROLLRATE_P", "MC_ROLLRATE_I", "MC_ROLLRATE_D", "MC_ROLLRATE_K",
    "MC_PITCHRATE_P", "MC_PITCHRATE_I", "MC_PITCHRATE_D", "MC_PITCHRATE_K",
]
DIAG_PARAM_NAMES = ["MC_ROLL_P", "MC_PITCH_P"]

# Params that receive gain updates (shared Kp/Ki/Kd for roll and pitch)
WRITE_PARAMS = [
    "MC_ROLLRATE_P", "MC_ROLLRATE_I", "MC_ROLLRATE_D",
    "MC_PITCHRATE_P", "MC_PITCHRATE_I", "MC_PITCHRATE_D",
]

CSV_COLUMNS = (
    ["wall_time", "mav_boot_ms", "att_age_ms"]
    + [f"obs_{i:02d}" for i in range(OBS_DIM)]
    + [f"action_{i}" for i in range(ACT_DIM)]
    + ["kp", "ki", "kd"]                          # running gain state
    + ["dkp", "dki", "dkd"]                       # delta applied this tick
    + ["freeze_reason"]                            # "" or why gains were frozen
    + ["flag_rates_sane", "flag_obs_finite",
       "flag_action_finite", "flag_action_in_range",
       "flag_frozen", "flag_stable"]
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--conn", default="udpin:localhost:14540")
    ap.add_argument("--onnx", default=os.path.expanduser(
        "~/rl_pid_tuner/results/frozen_joint_12d_c860k/actor_joint_12d_c860k.onnx"))
    ap.add_argument("--rate", type=float, default=5.0,
                    help="Update rate in Hz (default 5). Use 2 for first test.")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute everything but do not send PARAM_SET.")
    ap.add_argument("--deadband-att",  type=float, default=ATT_STABLE_THR,
                    help="Attitude threshold (rad) for stable-hold gate. Default 0.10 (~5.7°).")
    ap.add_argument("--deadband-rate", type=float, default=RATE_STABLE_THR,
                    help="Rate threshold (rad/s) for stable-hold gate. Default 0.10.")
    ap.add_argument("--max-ok-ticks", type=int, default=3,
                    help="Max consecutive ticks the NN may update gains outside the stable band. "
                         "If exceeded, gains freeze until the drone re-enters the stable band. "
                         "At 2 Hz the default (3) = 1.5 s of NN action per disturbance event.")
    ap.add_argument("--heartbeat-timeout", type=float, default=15.0)
    return ap.parse_args()


# ── MAVLink helpers ──────────────────────────────────────────────────────────

def request_one_param(mav, name, timeout=2.5):
    mav.mav.param_request_read_send(
        mav.target_system, mav.target_component,
        name.encode("utf-8") if isinstance(name, str) else name, -1)
    t_end = time.time() + timeout
    while time.time() < t_end:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg is None:
            continue
        pid = msg.param_id
        if isinstance(pid, bytes):
            pid = pid.decode("utf-8", errors="ignore")
        if pid.rstrip("\x00") == name:
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


def send_param_set(mav, name, value, retries=2, ack_timeout=0.3):
    """Send PARAM_SET and wait briefly for PARAM_VALUE ack. Returns True if acked."""
    for _ in range(retries):
        mav.mav.param_set_send(
            mav.target_system, mav.target_component,
            name.encode("utf-8"),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        t_end = time.time() + ack_timeout
        while time.time() < t_end:
            msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.1)
            if msg is None:
                continue
            pid = msg.param_id
            if isinstance(pid, bytes):
                pid = pid.decode("utf-8", errors="ignore")
            if pid.rstrip("\x00") == name:
                return True
    return False


def request_attitude_stream(mav, rate_hz=50.0):
    interval_us = int(1e6 / rate_hz)
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        interval_us, 0, 0, 0, 0, 0)


# ── Obs / physics helpers ────────────────────────────────────────────────────

def build_obs(roll, pitch, rollspd, pitchspd, kp, ki, kd):
    # 12-D layout — matches envs/pybullet_pid_tuner_env.py joint obs
    # (step_prog removed). Shared roll/pitch gains (joint shared-3D policy).
    roll_rate_err  = KP_ATT * (-roll)  - rollspd
    pitch_rate_err = KP_ATT * (-pitch) - pitchspd
    return np.array([
        roll, pitch, rollspd, pitchspd,
        roll_rate_err, pitch_rate_err,
        kp / KP_NORM, ki / KI_NORM, kd / KD_NORM,
        kp / KP_NORM, ki / KI_NORM, kd / KD_NORM,
    ], dtype=np.float32)


def safety_check(roll, pitch, rollspd, pitchspd):
    """Return (is_safe, reason_string)."""
    if not (math.isfinite(roll) and math.isfinite(pitch)):
        return False, "attitude_nan"
    if abs(roll) > MAX_ATT_RAD:
        return False, f"roll_limit({math.degrees(roll):.1f}deg)"
    if abs(pitch) > MAX_ATT_RAD:
        return False, f"pitch_limit({math.degrees(pitch):.1f}deg)"
    if abs(rollspd) > MAX_RATE_RAD:
        return False, f"rollspd_limit({rollspd:.1f}rad/s)"
    if abs(pitchspd) > MAX_RATE_RAD:
        return False, f"pitchspd_limit({pitchspd:.1f}rad/s)"
    return True, ""


def inside_stable_band(roll, pitch, rollspd, pitchspd, att_thr, rate_thr):
    """Return True when drone is inside the stable-hover region — hold gains, skip NN update."""
    return (abs(roll)     < att_thr  and
            abs(pitch)    < att_thr  and
            abs(rollspd)  < rate_thr and
            abs(pitchspd) < rate_thr)


def apply_delta(kp, ki, kd, action):
    """Apply DELTA_SCALE, clip to bounds. Returns (new_kp, new_ki, new_kd, dkp, dki, dkd)."""
    delta = action * DELTA_SCALE
    new_kp = float(np.clip(kp + delta[0], KP_MIN, KP_MAX))
    new_ki = float(np.clip(ki + delta[1], KI_MIN, KI_MAX))
    new_kd = float(np.clip(kd + delta[2], KD_MIN, KD_MAX))
    return new_kp, new_ki, new_kd, new_kp - kp, new_ki - ki, new_kd - kd


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.out is None:
        results_dir = os.path.expanduser("~/rl_pid_tuner/results")
        os.makedirs(results_dir, exist_ok=True)
        args.out = os.path.join(
            results_dir, "injector_" + time.strftime("%Y-%m-%d_%H-%M-%S") + ".csv")

    mode = "DRY-RUN (no PARAM_SET)" if args.dry_run else "LIVE (PARAM_SET active)"
    print(f"[INJECTOR] -------- Stage 2 gain injector — {mode} --------")
    print(f"[INJECTOR] Connect : {args.conn}")
    print(f"[INJECTOR] ONNX    : {args.onnx}")
    print(f"[INJECTOR] Rate    : {args.rate} Hz")
    print(f"[INJECTOR] Output  : {args.out}")
    if args.dry_run:
        print("[INJECTOR] --dry-run: PARAM_SET will NOT be sent.")
    else:
        print("[INJECTOR] ⚠️  SITL ONLY — will send PARAM_SET to update rate gains.")
    print()

    # ── ONNX ────────────────────────────────────────────────────────────────
    if not os.path.isfile(args.onnx):
        sys.exit(f"[INJECTOR] ONNX not found: {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # ── MAVLink ─────────────────────────────────────────────────────────────
    print(f"[INJECTOR] Waiting for heartbeat (timeout {args.heartbeat_timeout:.0f}s)...")
    mav = mavutil.mavlink_connection(args.conn)
    hb = mav.wait_heartbeat(timeout=args.heartbeat_timeout)
    if hb is None:
        sys.exit("[INJECTOR] No heartbeat. Is PX4 SITL running?")
    print(f"[INJECTOR] Heartbeat: sys={mav.target_system} comp={mav.target_component}")

    request_attitude_stream(mav, rate_hz=max(50.0, args.rate * 4))

    # ── Read initial gains ───────────────────────────────────────────────────
    print("[INJECTOR] Reading initial rate-PID gains...")
    gains, missing = read_params(mav, GAIN_PARAM_NAMES)
    if gains is None:
        sys.exit(f"[INJECTOR] Failed to read: {missing}")

    # Effective gains
    K_r = gains["MC_ROLLRATE_K"]
    K_p = gains["MC_PITCHRATE_K"]
    kp = K_r * gains["MC_ROLLRATE_P"]
    ki = K_r * gains["MC_ROLLRATE_I"]
    kd = K_r * gains["MC_ROLLRATE_D"]

    print(f"  Kp_eff = {kp:.6f}   Ki_eff = {ki:.6f}   Kd_eff = {kd:.6f}")
    print(f"  K_roll={K_r:.4f}  K_pitch={K_p:.4f}")

    # Warn if K != 1 — PARAM_SET targets P/I/D not the effective gain
    if abs(K_r - 1.0) > 0.01 or abs(K_p - 1.0) > 0.01:
        print("[INJECTOR] WARN: MC_ROLLRATE_K or MC_PITCHRATE_K != 1.0")
        print("           PARAM_SET will update P/I/D; effective gain = K * new_P/I/D")
        print("           Ensure K=1.0 for gain updates to match training exactly.")

    diag, _ = read_params(mav, DIAG_PARAM_NAMES)
    if diag:
        for k in DIAG_PARAM_NAMES:
            note = "" if abs(diag[k] - KP_ATT) < 0.1 else f"  ← training assumed {KP_ATT}"
            print(f"  {k} = {diag[k]:.4f}{note}")

    # ── Force gains to training defaults at startup ───────────────────────────
    # PX4 does not reliably retain PARAM_SET resets across sessions, and stale
    # crash-drifted gains contaminate every subsequent test. The startup reset
    # is a SAFETY PRECONDITION, not an NN action — it is applied even in
    # --dry-run (dry-run only suppresses the NN's per-tick gain updates).
    if not (abs(kp - KP_INIT) < 1e-5 and abs(ki - KI_INIT) < 1e-5 and abs(kd - KD_INIT) < 1e-5):
        print(f"[INJECTOR] Gains differ from training defaults — resetting to defaults before run.")
        send_param_set(mav, "MC_ROLLRATE_P",  KP_INIT)
        send_param_set(mav, "MC_ROLLRATE_I",  KI_INIT)
        send_param_set(mav, "MC_ROLLRATE_D",  KD_INIT)
        send_param_set(mav, "MC_PITCHRATE_P", KP_INIT)
        send_param_set(mav, "MC_PITCHRATE_I", KI_INIT)
        send_param_set(mav, "MC_PITCHRATE_D", KD_INIT)
    kp, ki, kd = KP_INIT, KI_INIT, KD_INIT
    print(f"[INJECTOR] Starting gain state: Kp={kp:.6f}  Ki={ki:.6f}  Kd={kd:.8f}  "
          f"(reset applied to PX4{' even in dry-run' if args.dry_run else ''})")

    print(f"[INJECTOR] DELTA_SCALE: {DELTA_SCALE}")
    att_thr  = args.deadband_att
    rate_thr = args.deadband_rate
    print(f"[INJECTOR] Safety : att<60°, |rate|<20 rad/s → freeze gains")
    print(f"[INJECTOR] Deadband: att<{math.degrees(att_thr):.1f}°  rate<{rate_thr:.3f} rad/s → hold gains (no NN update)")
    max_ok_ticks = args.max_ok_ticks
    print(f"[INJECTOR] OK limit: freeze after {max_ok_ticks} consecutive ticks outside band "
          f"({max_ok_ticks / args.rate:.1f} s at {args.rate} Hz)")
    print()

    # ── CSV ─────────────────────────────────────────────────────────────────
    f_csv = open(args.out, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(CSV_COLUMNS)
    f_csv.flush()

    # ── Run loop ─────────────────────────────────────────────────────────────
    dt = 1.0 / args.rate
    t_start = time.monotonic()
    n_logged       = 0
    n_frozen       = 0
    n_held_stable  = 0
    consecutive_ok = 0      # ticks spent outside stable band since last re-entry
    kp_snap = ki_snap = kd_snap = None   # gain snapshot taken when disturbance is first detected
    last_print = t_start

    stopping = {"stop": False}
    def _stop(sig, frame):
        stopping["stop"] = True
    signal.signal(signal.SIGINT, _stop)

    print(f"[INJECTOR] Updating gains at {args.rate} Hz. Ctrl-C to stop and reset.")
    print(f"  {'t':>7s}  {'Kp':>7s}  {'Ki':>8s}  {'Kd':>8s}  {'action':>22s}  status")

    while not stopping["stop"]:
        if args.duration is not None and (time.monotonic() - t_start) >= args.duration:
            break

        loop_start = time.monotonic()

        # Drain ATTITUDE messages, keep freshest
        last_attitude = None
        while True:
            msg = mav.recv_match(type="ATTITUDE", blocking=False)
            if msg is None:
                break
            last_attitude = msg

        if last_attitude is None:
            msg = mav.recv_match(type="ATTITUDE", blocking=True, timeout=dt * 0.5)
            if msg is None:
                elapsed = time.monotonic() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue
            last_attitude = msg

        roll     = float(last_attitude.roll)
        pitch    = float(last_attitude.pitch)
        rollspd  = float(last_attitude.rollspeed)
        pitchspd = float(last_attitude.pitchspeed)
        mav_boot_ms = int(last_attitude.time_boot_ms)
        try:
            att_age_ms = float(mav.time_since("ATTITUDE")) * 1000.0
        except Exception:
            att_age_ms = float("nan")

        # ── Safety check ────────────────────────────────────────────────────
        safe, freeze_reason = safety_check(roll, pitch, rollspd, pitchspd)
        flag_rates_sane = int(safe)
        flag_stable = int(inside_stable_band(roll, pitch, rollspd, pitchspd, att_thr, rate_thr))

        # ── Build obs ────────────────────────────────────────────────────────
        obs = build_obs(roll, pitch, rollspd, pitchspd, kp, ki, kd)
        flag_obs_finite = int(safe and bool(np.all(np.isfinite(obs))))

        # ── ONNX inference ───────────────────────────────────────────────────
        if flag_obs_finite:
            action = sess.run([out_name], {in_name: obs.reshape(1, OBS_DIM)})[0].flatten()
        else:
            action = np.zeros(ACT_DIM, dtype=np.float32)
            if not freeze_reason:
                freeze_reason = "obs_not_finite"

        flag_action_finite   = int(bool(np.all(np.isfinite(action))))
        flag_action_in_range = int(
            flag_action_finite and
            bool(np.all(action >= -1.0 - 1e-6)) and
            bool(np.all(action <=  1.0 + 1e-6))
        )

        # ── Gain update ──────────────────────────────────────────────────────
        # Track consecutive ticks outside the stable band; reset counter on re-entry.
        if flag_stable:
            consecutive_ok = 0
            kp_snap = ki_snap = kd_snap = None   # clear snapshot on re-entry
        else:
            if consecutive_ok == 0:
                # First tick outside the band — snapshot current gains.
                kp_snap, ki_snap, kd_snap = kp, ki, kd
            consecutive_ok += 1
        flag_ok_limit = int(consecutive_ok > max_ok_ticks)

        flag_frozen = int(not safe or not flag_obs_finite or bool(flag_stable) or bool(flag_ok_limit))
        dkp = dki = dkd = 0.0

        if not flag_frozen and flag_action_in_range:
            new_kp, new_ki, new_kd, dkp, dki, dkd = apply_delta(kp, ki, kd, action)

            if not args.dry_run:
                send_param_set(mav, "MC_ROLLRATE_P",  new_kp)
                send_param_set(mav, "MC_ROLLRATE_I",  new_ki)
                send_param_set(mav, "MC_ROLLRATE_D",  new_kd)
                send_param_set(mav, "MC_PITCHRATE_P", new_kp)
                send_param_set(mav, "MC_PITCHRATE_I", new_ki)
                send_param_set(mav, "MC_PITCHRATE_D", new_kd)

            kp, ki, kd = new_kp, new_ki, new_kd
        else:
            if flag_stable and not freeze_reason:
                freeze_reason = "stable_hold"
                n_held_stable += 1
            elif flag_ok_limit and not freeze_reason:
                freeze_reason = f"ok_limit({consecutive_ok})"
                # First tick we hit the limit — restore the pre-disturbance snapshot so the
                # drone recovers with known-good gains rather than whatever the NN drifted to.
                if kp_snap is not None and not args.dry_run:
                    send_param_set(mav, "MC_ROLLRATE_P",  kp_snap)
                    send_param_set(mav, "MC_ROLLRATE_I",  ki_snap)
                    send_param_set(mav, "MC_ROLLRATE_D",  kd_snap)
                    send_param_set(mav, "MC_PITCHRATE_P", kp_snap)
                    send_param_set(mav, "MC_PITCHRATE_I", ki_snap)
                    send_param_set(mav, "MC_PITCHRATE_D", kd_snap)
                if kp_snap is not None:
                    kp, ki, kd = kp_snap, ki_snap, kd_snap
            n_frozen += 1

        n_logged += 1

        # ── CSV row ─────────────────────────────────────────────────────────
        row = (
            [f"{time.time():.6f}", str(mav_boot_ms), f"{att_age_ms:.1f}"]
            + [f"{v:.6f}" for v in obs.tolist()]
            + [f"{v:.6f}" for v in action.tolist()]
            + [f"{kp:.6f}", f"{ki:.6f}", f"{kd:.6f}"]
            + [f"{dkp:.6f}", f"{dki:.6f}", f"{dkd:.6f}"]
            + [freeze_reason]
            + [str(flag_rates_sane), str(flag_obs_finite),
               str(flag_action_finite), str(flag_action_in_range),
               str(flag_frozen), str(flag_stable)]
        )
        writer.writerow(row)

        # ── Periodic status ─────────────────────────────────────────────────
        now = time.monotonic()
        if now - last_print >= 2.0:
            if flag_stable:
                status = "HOLD_STABLE"
            elif flag_ok_limit:
                status = f"HOLD_LIMIT({consecutive_ok}ticks)"
            elif flag_frozen:
                status = f"FROZEN({freeze_reason})"
            else:
                status = f"OK({consecutive_ok}/{max_ok_ticks})"
            dry = "[DRY]" if args.dry_run else ""
            print(f"  t={now - t_start:6.1f}s  "
                  f"Kp={kp:.4f}  Ki={ki:.5f}  Kd={kd:.5f}  "
                  f"act=[{action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f}]  "
                  f"{status} {dry}", flush=True)
            f_csv.flush()
            last_print = now

        elapsed = time.monotonic() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # ── Shutdown: reset gains to training defaults ───────────────────────────
    f_csv.close()
    print()
    print(f"[INJECTOR] Stopped. Logged: {n_logged}  Frozen ticks: {n_frozen}  (stable hold: {n_held_stable}  safety freeze: {n_frozen - n_held_stable})")
    print(f"[INJECTOR] Final gains: Kp={kp:.4f}  Ki={ki:.5f}  Kd={kd:.5f}")
    print(f"[INJECTOR] CSV: {args.out}")

    # Exit reset is also a safety/hygiene action (prevents stale gains from
    # contaminating the next run) — applied even in --dry-run.
    print("[INJECTOR] Resetting gains to training defaults...")
    send_param_set(mav, "MC_ROLLRATE_P",  KP_INIT)
    send_param_set(mav, "MC_ROLLRATE_I",  KI_INIT)
    send_param_set(mav, "MC_ROLLRATE_D",  KD_INIT)
    send_param_set(mav, "MC_PITCHRATE_P", KP_INIT)
    send_param_set(mav, "MC_PITCHRATE_I", KI_INIT)
    send_param_set(mav, "MC_PITCHRATE_D", KD_INIT)
    print(f"[INJECTOR] Reset to Kp={KP_INIT}  Ki={KI_INIT}  Kd={KD_INIT}")


if __name__ == "__main__":
    main()
