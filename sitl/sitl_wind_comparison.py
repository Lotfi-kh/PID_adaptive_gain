#!/usr/bin/env python3
"""
sitl_wind_comparison.py — Autonomous Gazebo physics-wind comparison: Baseline vs RL.

Runs two complete SITL sessions automatically:
  Run 1 (Baseline): rl_gain_tuner silenced → mc_rate_control uses fixed PX4 params.
  Run 2 (RL):       rl_gain_tuner active   → in-firmware c860k actor adapts gains.

Wind disturbance: Ornstein-Uhlenbeck aerodynamic forces via Gazebo Wind system
  (sitl_wind_gz.py).  Same --seed used both runs → identical wind profiles.

Output:
  test_results/baseline_wind_gz.ulg / .csv
  test_results/rl_wind_gz.ulg       / .csv
  test_results/result_wind_gz.png + _metrics.csv

Usage (let it run fully autonomously — takes ~8 minutes total):
    cd ~/rl_pid_tuner
    python sitl/sitl_wind_comparison.py

    # Custom wind:
    python sitl/sitl_wind_comparison.py --vx 6 --vy 3 --sigma 2.5 --wind-duration 120
"""

import argparse
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from pymavlink import mavutil

# ── Paths ─────────────────────────────────────────────────────────────────────
PX4_DIR      = os.path.expanduser("~/PX4-Autopilot")
RC_MC_APPS   = os.path.join(PX4_DIR,
    "ROMFS/px4fmu_common/init.d/rc.mc_apps")
LOG_ROOT     = os.path.join(PX4_DIR,
    "build/px4_sitl_default/rootfs/log")
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
OUT_DIR      = os.path.join(REPO_ROOT, "test_results")

WIND_SCRIPT    = os.path.join(SCRIPT_DIR, "sitl_wind_gz.py")
EVAL_CSV       = os.path.join(SCRIPT_DIR, "ulog_to_eval_csv.py")
AB_COMPARE     = os.path.join(SCRIPT_DIR, "ab_compare.py")
PYTHON         = sys.executable

# ── PX4 custom-mode encoding (from px4_custom_mode.h) ────────────────────────
_PX4_MODE_AUTO_TAKEOFF = (4 << 16) | (2 << 24)
_PX4_MODE_AUTO_LOITER  = (4 << 16) | (3 << 24)
_PX4_MODE_AUTO_LAND    = (4 << 16) | (5 << 24)

# ── MAVLink helpers ───────────────────────────────────────────────────────────

def mav_connect(conn="udpin:localhost:14540", timeout=120):
    """Connect to PX4 SITL and wait for heartbeat."""
    print(f"[MAV] Connecting {conn} …", flush=True)
    mav = mavutil.mavlink_connection(conn)
    hb = mav.wait_heartbeat(timeout=timeout)
    if hb is None:
        raise RuntimeError("[MAV] No heartbeat — SITL not running?")
    mav.target_system    = mav.target_system
    mav.target_component = mav.target_component
    print(f"[MAV] Heartbeat from sys={mav.target_system} comp={mav.target_component}")
    return mav


def _cmd_long(mav, cmd, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        cmd, 0,
        float(p1), float(p2), float(p3), float(p4),
        float(p5), float(p6), float(p7))


def _set_mode(mav, custom_mode):
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        custom_mode)


def _set_param(mav, name, value):
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        name.encode(), float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.05)


def _wait_armed(mav, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        msg = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if msg and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            return True
    return False


def _wait_altitude(mav, target_m, timeout=45.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2.0)
        if msg:
            alt = msg.relative_alt / 1000.0
            if alt >= target_m * 0.85:
                return True
    return False


def _wait_system_ready(mav, timeout=30.0):
    """Wait until the drone is not in a critical prearm state."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        msg = mav.recv_match(type="EXTENDED_SYS_STATE", blocking=False)
        hb  = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if hb and hb.system_status in (
                mavutil.mavlink.MAV_STATE_STANDBY,
                mavutil.mavlink.MAV_STATE_ACTIVE):
            return True
    return False


def arm_and_takeoff(mav, altitude=10.0):
    """Arm the drone and climb to altitude, then enter LOITER."""
    print(f"[FLY] Disabling prearm timer …")
    _set_param(mav, "COM_DISARM_PRFLT", 0.0)
    time.sleep(1.0)

    _wait_system_ready(mav, timeout=40.0)

    print("[FLY] Arming …")
    armed = False
    for attempt in range(6):
        _cmd_long(mav, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                  p1=1, p2=21196)
        if _wait_armed(mav, timeout=6.0):
            armed = True
            break
        print(f"[FLY] Arm attempt {attempt+1}/6 denied — waiting 5 s …")
        time.sleep(5.0)
    if not armed:
        raise RuntimeError("[FLY] ARM failed after 6 attempts")
    print(f"[FLY] Armed. Switching to AUTO.TAKEOFF → {altitude} m …")

    _set_mode(mav, _PX4_MODE_AUTO_TAKEOFF)
    if not _wait_altitude(mav, altitude, timeout=45.0):
        print("[FLY] WARNING: altitude not reached in 45 s — proceeding anyway")
    print("[FLY] Altitude reached. Switching to LOITER.")
    _set_mode(mav, _PX4_MODE_AUTO_LOITER)
    time.sleep(3.0)


def land(mav):
    print("[FLY] Landing …")
    _set_mode(mav, _PX4_MODE_AUTO_LAND)
    time.sleep(15.0)


# ── ULog helpers ──────────────────────────────────────────────────────────────

def find_active_ulg(timeout=120, poll=1.5):
    """Return path of the actively-growing .ulg file."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        ulgs = glob.glob(os.path.join(LOG_ROOT, "**", "*.ulg"), recursive=True)
        if ulgs:
            newest = max(ulgs, key=os.path.getmtime)
            s0 = os.path.getsize(newest)
            time.sleep(1.5)
            s1 = os.path.getsize(newest)
            if s1 > s0:
                print(f"[LOG] Active: {os.path.basename(newest)} ({s1/1024:.0f} KB, growing)")
                return newest
        time.sleep(poll)
    return None


def wait_log_settle(path, secs=5.0):
    """Wait until the log stops growing (SITL landed / disarmed)."""
    for _ in range(10):
        s0 = os.path.getsize(path)
        time.sleep(secs / 10.0)
        if os.path.getsize(path) == s0:
            return
    # Log still growing — that's fine, we copy anyway


# ── rl_gain_tuner toggle ──────────────────────────────────────────────────────

_RL_LINE_PATTERN = re.compile(r'^(rl_gain_tuner start)', re.MULTILINE)


def disable_rl_in_rc():
    """Comment out 'rl_gain_tuner start' so baseline SITL has fixed gains."""
    text = open(RC_MC_APPS).read()
    if "#BASELINE_DISABLED# rl_gain_tuner start" in text:
        return  # already disabled
    text_new = _RL_LINE_PATTERN.sub(r'#BASELINE_DISABLED# \1', text)
    open(RC_MC_APPS, "w").write(text_new)
    print("[SETUP] rl_gain_tuner disabled in rc.mc_apps (baseline run)")


def enable_rl_in_rc():
    """Restore 'rl_gain_tuner start' for the RL run."""
    text = open(RC_MC_APPS).read()
    text_new = text.replace("#BASELINE_DISABLED# rl_gain_tuner start",
                             "rl_gain_tuner start")
    open(RC_MC_APPS, "w").write(text_new)
    print("[SETUP] rl_gain_tuner re-enabled in rc.mc_apps (RL run)")


# ── SITL lifecycle ────────────────────────────────────────────────────────────

def start_sitl(world="windy"):
    env = os.environ.copy()
    env["PX4_GZ_WORLD"] = world
    env["DISPLAY"]      = os.environ.get("DISPLAY", ":1")
    proc = subprocess.Popen(
        ["make", "px4_sitl_default", "gz_f450_rl"],
        cwd=PX4_DIR, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    print(f"[SITL] Started (PID={proc.pid})  PX4_GZ_WORLD={world}")
    return proc


def stop_sitl(proc):
    if proc is None or proc.poll() is not None:
        return
    print(f"[SITL] Stopping (PID={proc.pid}) …")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    # Kill any leftover gz/px4 processes
    for name in ("gzserver", "gz", "px4"):
        subprocess.run(["pkill", "-f", name],
                       capture_output=True)
    time.sleep(3.0)
    print("[SITL] Stopped.")


# ── Single run orchestration ──────────────────────────────────────────────────

def run_once(label, out_ulg_path, wind_args, settle_s=20.0):
    """
    Start SITL, arm, hover, apply wind, record, land, copy ulg.
    Returns path to copied ulg.
    """
    sitl_proc = None
    wind_proc  = None

    try:
        # 1. Start SITL
        sitl_proc = start_sitl(world="windy")
        print(f"[{label}] Waiting for SITL to initialize (60 s) …")
        time.sleep(60.0)

        # 2. Connect MAVLink
        mav = mav_connect("udpin:localhost:14540", timeout=90)

        # 3. Arm + takeoff
        arm_and_takeoff(mav, altitude=10.0)

        # 4. Find active log
        ulg_src = find_active_ulg(timeout=120)
        if ulg_src is None:
            raise RuntimeError(f"[{label}] No active .ulg found")

        # 5. Settle hover before applying wind
        print(f"[{label}] Hovering {settle_s:.0f} s before wind …", flush=True)
        time.sleep(settle_s)

        # 6. Apply Gazebo aerodynamic wind
        print(f"[{label}] Starting physics wind …", flush=True)
        wind_proc = subprocess.Popen(
            [PYTHON, WIND_SCRIPT] + wind_args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)

        # 7. Wait for wind to finish (duration is in wind_args)
        wind_duration = float(wind_args[wind_args.index("--duration") + 1])
        t_wind_start = time.monotonic()
        while wind_proc.poll() is None:
            line = wind_proc.stdout.readline()
            if line:
                print(f"  [WIND] {line.rstrip()}", flush=True)
            if time.monotonic() - t_wind_start > wind_duration + 20:
                wind_proc.terminate()
                break

        print(f"[{label}] Wind done. Landing …", flush=True)
        land(mav)
        wait_log_settle(ulg_src, secs=5.0)

        # 8. Copy ulg
        os.makedirs(OUT_DIR, exist_ok=True)
        shutil.copy2(ulg_src, out_ulg_path)
        sz = os.path.getsize(out_ulg_path) / 1024
        print(f"[{label}] ✓ Saved {os.path.basename(out_ulg_path)} ({sz:.0f} KB)")
        return out_ulg_path

    finally:
        if wind_proc and wind_proc.poll() is None:
            wind_proc.terminate()
        stop_sitl(sitl_proc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--vx",            type=float, default=5.0)
    ap.add_argument("--vy",            type=float, default=2.0)
    ap.add_argument("--sigma",         type=float, default=2.0)
    ap.add_argument("--theta",         type=float, default=0.10)
    ap.add_argument("--max-gust",      type=float, default=None)
    ap.add_argument("--wind-duration", type=float, default=90.0,
                    help="Wind-on duration in seconds (default 90)")
    ap.add_argument("--seed",          type=int,   default=42,
                    help="RNG seed — same seed = identical wind both runs")
    ap.add_argument("--settle",        type=float, default=20.0,
                    help="Hover seconds before wind starts (default 20)")
    ap.add_argument("--title",         default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    event_log = os.path.join(OUT_DIR, "wind_gz_events.csv")
    baseline_ulg = os.path.join(OUT_DIR, "baseline_wind_gz.ulg")
    rl_ulg       = os.path.join(OUT_DIR, "rl_wind_gz.ulg")
    baseline_csv = os.path.join(OUT_DIR, "baseline_wind_gz.csv")
    rl_csv       = os.path.join(OUT_DIR, "rl_wind_gz.csv")

    wind_args = [
        "--vx",       str(args.vx),
        "--vy",       str(args.vy),
        "--sigma",    str(args.sigma),
        "--theta",    str(args.theta),
        "--duration", str(args.wind_duration),
        "--seed",     str(args.seed),
        "--event-log", event_log,
    ]
    if args.max_gust is not None:
        wind_args += ["--max-gust", str(args.max_gust)]

    print("=" * 70)
    print(f"AUTONOMOUS GAZEBO WIND COMPARISON")
    print(f"  Wind: vx={args.vx} vy={args.vy} σ={args.sigma} θ={args.theta}"
          f"  duration={args.wind_duration}s  seed={args.seed}")
    print(f"  Settle: {args.settle}s  Output: {OUT_DIR}")
    print("=" * 70)

    # ── Run 1: Baseline ───────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("RUN 1/2 — BASELINE  (rl_gain_tuner disabled → fixed PX4 gains)")
    print("─" * 70)
    disable_rl_in_rc()
    try:
        run_once("BASELINE", baseline_ulg, wind_args, settle_s=args.settle)
    except Exception as e:
        print(f"[ERROR] Baseline run failed: {e}")
        enable_rl_in_rc()
        sys.exit(1)
    finally:
        enable_rl_in_rc()   # always restore before RL run

    # Wait between sessions for system to settle
    print("\n[PAUSE] 10 s between sessions …")
    time.sleep(10.0)

    # ── Run 2: RL ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("RUN 2/2 — RL  (rl_gain_tuner active → in-firmware c860k actor)")
    print("─" * 70)
    try:
        run_once("RL", rl_ulg, wind_args, settle_s=args.settle)
    except Exception as e:
        print(f"[ERROR] RL run failed: {e}")
        sys.exit(1)

    # ── Convert ULogs to CSV ──────────────────────────────────────────────────
    print("\n[POST] Converting ulogs …")
    settle_total = args.settle  # wind starts at settle_s; mark everything after as active
    for ulg, csv_out in ((baseline_ulg, baseline_csv), (rl_ulg, rl_csv)):
        cmd = [PYTHON, EVAL_CSV, ulg, "-o", csv_out,
               "--always-active", str(int(settle_total))]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [WARN] ulog_to_eval_csv failed for {ulg}: {r.stderr[-200:]}")
        else:
            print(f"  OK  {os.path.basename(csv_out)}")

    # ── A/B Compare ───────────────────────────────────────────────────────────
    print("\n[POST] Running A/B comparison …")
    title = (args.title or
             f"Baseline PID vs in-firmware RL — Gazebo wind physics "
             f"(vx={args.vx} vy={args.vy} m/s, σ={args.sigma}, {args.wind_duration}s)")
    result_stem = os.path.join(OUT_DIR, "result_wind_gz")
    cmd = [PYTHON, AB_COMPARE, baseline_csv, rl_csv,
           "-o", result_stem, "--title", title]
    r = subprocess.run(cmd, text=True)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Plot    → {result_stem}.png")
    print(f"  Metrics → {result_stem}_metrics.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
