#!/usr/bin/env python3
"""
run_pairs.py — Automated SITL paired A/B flights (pairs 2..12).

Keeps SITL running across all flights (no restart between flights).
Toggles rl_gain_tuner via MAVLink shell between baseline and adaptive halves.

Usage:
    cd ~/rl_pid_tuner
    python sitl/run_pairs.py              # pairs 2-12
    python sitl/run_pairs.py --start 5   # resume from pair 5
"""

import argparse
import glob
import os
import signal
import subprocess
import sys
import time

from pymavlink import mavutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sitl_wind_comparison import (          # noqa: E402
    mav_connect, arm_and_takeoff, land,
    find_active_ulg, wait_log_settle,
    stop_sitl, PX4_DIR, LOG_ROOT,
)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
PYTHON      = sys.executable
SYS_PYTHON  = "/usr/bin/python3"   # has gz-transport bindings for the wind injector
DISTURB     = os.path.join(SCRIPT_DIR, "sitl_disturb.py")
WIND        = os.path.join(SCRIPT_DIR, "sitl_wind_force.py")
EXTRACT     = os.path.join(SCRIPT_DIR, "extract_metrics.py")
AGGREGATE   = os.path.join(SCRIPT_DIR, "aggregate_stats.py")
EVENTS_DIR  = os.path.join(REPO_ROOT, "test_results", "gazebo")
MASTER_CSV  = os.path.join(EVENTS_DIR, "metrics_master.csv")
WIND_MASTER = os.path.join(EVENTS_DIR, "metrics_master_wind.csv")
_sitl_proc = None


def _set_param(mav, name, value):
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        name.encode(), float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.15)


def _wait_disarmed(mav, timeout=30.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        msg = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1.5)
        if msg and not (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            return True
    return False


def pxh_cmd(mav, cmd, wait=3.0):
    """Send one command to the PX4 NuttX shell via SERIAL_CONTROL (same MAVLink conn)."""
    # SERIAL_CONTROL_DEV_SHELL = 10; flags = EXCLUSIVE | RESPOND
    SHELL_PORT = 10
    FLAG_EXCLUSIVE = mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE
    FLAG_RESPOND   = mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND
    line = (cmd + "\n").encode()
    while len(line) > 0:
        chunk = line[:70]
        buf = list(chunk) + [0] * (70 - len(chunk))
        mav.mav.serial_control_send(
            SHELL_PORT, FLAG_EXCLUSIVE | FLAG_RESPOND,
            0, 0, len(chunk), buf)
        line = line[70:]
    time.sleep(wait)   # give PX4 time to process the command


def clean_slate():
    """Kill any leftover PX4/Gazebo instances before launching a fresh one.
    Interrupted runs leave a px4+gz alive (setsid group survives a -9 of the
    parent), and multiple instances collide on UDP 14540/14580 and the gz
    transport — which makes MAVLink connect hang and lockstep desync. These
    patterns never match this orchestrator's own command line, so it is safe."""
    import shutil
    for pat in ("px4_sitl_default/bin/px4", "gz sim", "ruby.*gz", "gzserver"):
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    time.sleep(3)
    # Verify the offboard ports are free; warn (don't crash) if not.
    if shutil.which("ss"):
        out = subprocess.run(["ss", "-ulnp"], capture_output=True, text=True).stdout
        if ":14540" in out or ":14580" in out:
            print("[SITL] WARNING: MAVLink ports still busy after cleanup — "
                  "a stale instance may remain.")
        else:
            print("[SITL] Clean slate: no leftover PX4/Gazebo, ports free.")


def start_sitl(world="default"):
    global _sitl_proc
    env = os.environ.copy()
    env["HEADLESS"] = "1"                  # suppress the gz GUI client (no window)
    env.setdefault("DISPLAY", ":1")       # X server for offscreen render context
    env["GZ_HEADLESS_RENDERING"] = "1"    # render offscreen, not to a window
    if world != "default":
        env["PX4_GZ_WORLD"] = world
    _sitl_proc = subprocess.Popen(
        ["make", "px4_sitl_default", "gz_f450_rl"],
        cwd=PX4_DIR, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    print(f"[SITL] Started (world={world}, display :1) PID={_sitl_proc.pid}")
    return _sitl_proc


_already_running_flag = False

def _cleanup(sig=None, frame=None):
    global _sitl_proc
    print("\n[CLEANUP] SIGINT — interrupted.")
    if _sitl_proc and not _already_running_flag:
        stop_sitl(_sitl_proc)
    sys.exit(0)


def run_flight(seed, controller, mav, wind=False, wind_duration=90.0):
    """Arm → take off → disturb → land → wait → extract.  Returns ulg path."""
    prefix = "wind" if wind else "pair"
    events_csv = os.path.join(EVENTS_DIR, f"{prefix}_seed{seed}_events.csv")
    print(f"\n[{prefix.upper()} {seed}] ── {controller.upper()} ──────────────────────")

    # Toggle controller
    if controller == "baseline":
        print("[CTRL] Sending rl_gain_tuner stop …")
        pxh_cmd(mav, "rl_gain_tuner stop", wait=3.0)
    else:
        print("[CTRL] Sending rl_gain_tuner start …")
        pxh_cmd(mav, "rl_gain_tuner start", wait=3.0)

    # Record time just before arming so we can find the log that opened for THIS flight
    arm_time = time.time()

    # Arm and climb
    arm_and_takeoff(mav, altitude=10.0)

    # Find the log that opened AFTER we armed (not a log left over from a prior flight)
    def _find_ulg_after(t_min, timeout=60, poll=1.5):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            ulgs = glob.glob(os.path.join(LOG_ROOT, "**", "*.ulg"), recursive=True)
            fresh = [u for u in ulgs if os.path.getmtime(u) >= t_min]
            if fresh:
                newest = max(fresh, key=os.path.getmtime)
                s0 = os.path.getsize(newest)
                time.sleep(poll)
                if os.path.getsize(newest) > s0:
                    print(f"[LOG] Active (post-arm): {os.path.basename(newest)}")
                    return newest
            time.sleep(poll)
        return None

    ulg = _find_ulg_after(arm_time - 5, timeout=60)   # -5 s margin for clock skew
    if ulg is None:
        raise RuntimeError(f"[PAIR {seed}] Could not find active ulg")
    print(f"[LOG] logging to {os.path.basename(ulg)}")

    # Disturbance
    if wind:
        print(f"[WIND] seed={seed} duration={wind_duration:.0f}s …")
        subprocess.run(
            [SYS_PYTHON, WIND, "--duration", str(wind_duration), "--seed", str(seed),
             "--event-log", events_csv],
            check=True,
        )
    else:
        print(f"[DISTURB] seed={seed} …")
        subprocess.run(
            [PYTHON, DISTURB, "--safe", "--seed", str(seed),
             "--event-log", events_csv],
            check=True,
        )

    # Land + wait for disarm
    land(mav)
    _wait_disarmed(mav, timeout=30)
    wait_log_settle(ulg, secs=4.0)
    print(f"[LOG] settled: {os.path.basename(ulg)}")

    # Extract
    master = WIND_MASTER if wind else MASTER_CSV
    out_wide = os.path.join(EVENTS_DIR,
                            f"{prefix}_seed{seed}_{controller}_wide.csv")
    extract_cmd = [PYTHON, EXTRACT, ulg,
                   "--events", events_csv,
                   "--append", master,
                   "--controller", controller,
                   "--seed", str(seed),
                   "--tag", "pair",
                   "-o", out_wide]
    if wind:
        extract_cmd.append("--no-events")   # wind is a single sustained window, no sweep
    subprocess.run(extract_cmd, check=True)
    return ulg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--start", type=int, default=2,
                    help="First pair seed to run (default=2)")
    ap.add_argument("--end", type=int, default=12,
                    help="Last pair seed to run (default=12)")
    ap.add_argument("--already-running", action="store_true",
                    help="Skip SITL launch/stop — connect to an existing SITL instance.")
    ap.add_argument("--wind", action="store_true",
                    help="Wind disturbance (OU aerodynamic forces) instead of torque kicks. "
                         "Requires SITL launched in the 'windy' world.")
    ap.add_argument("--wind-duration", type=float, default=90.0,
                    help="Wind exposure seconds per flight (default 90).")
    args = ap.parse_args()

    global _already_running_flag
    _already_running_flag = args.already_running
    signal.signal(signal.SIGINT, _cleanup)
    os.makedirs(EVENTS_DIR, exist_ok=True)

    kind = "wind pairs" if args.wind else "pairs"
    master = WIND_MASTER if args.wind else MASTER_CSV
    print(f"[BATCH] Running {kind} {args.start}..{args.end}")

    if not args.already_running:
        world = "windy" if args.wind else "default"
        clean_slate()                       # kill leftovers → no port collisions
        print(f"[SITL] Starting (world={world}) …")
        start_sitl(world=world)
        warmup = 130 if args.wind else 95   # headless render init is slow (~2 min to MAVLink)
        print(f"[SITL] Waiting {warmup} s for PX4 + Gazebo …")
        time.sleep(warmup)
    else:
        print("[SITL] Using already-running SITL (skipping launch)")

    mav = mav_connect(timeout=240)
    print("[MAV] Setting SDLOG_MODE=1, MIS_TAKEOFF_ALT=10, COM_DISARM_PRFLT=0 …")
    _set_param(mav, "SDLOG_MODE",      1.0)
    _set_param(mav, "MIS_TAKEOFF_ALT", 10.0)
    _set_param(mav, "COM_DISARM_PRFLT", 0.0)
    time.sleep(2.0)

    failed = []
    for seed in range(args.start, args.end + 1):
        for controller in ("baseline", "adaptive"):
            try:
                run_flight(seed, controller, mav,
                           wind=args.wind, wind_duration=args.wind_duration)
                print(f"[PAIR {seed}] {controller} OK")
                time.sleep(5)   # brief pause between flights
            except Exception as e:
                print(f"[PAIR {seed}] {controller} FAILED: {e}")
                failed.append((seed, controller))
                # try to re-disarm / stabilise before next attempt
                try:
                    land(mav)
                    _wait_disarmed(mav, timeout=20)
                except Exception:
                    pass
                time.sleep(8)

    if not args.already_running:
        print("\n[BATCH] All flights done. Stopping SITL …")
        stop_sitl(_sitl_proc)
    else:
        print("\n[BATCH] All flights done. (SITL left running — stop it from your terminal)")

    print("[BATCH] Running aggregate_stats …")
    subprocess.run([PYTHON, AGGREGATE, master])

    if failed:
        print(f"\n[BATCH] WARNING — {len(failed)} flight(s) failed: {failed}")
        print("  Re-run those manually using the pair N commands.")
    else:
        print("\n[BATCH] All pairs completed successfully.")


if __name__ == "__main__":
    main()
