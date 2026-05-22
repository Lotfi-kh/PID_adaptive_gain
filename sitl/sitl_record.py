#!/usr/bin/env python3
"""
sitl_record.py — Latch onto the active PX4 SITL ulog and copy it after DURATION seconds.

Finds the most recently modified .ulg in the SITL log directory (the one currently
being written), records for DURATION seconds, then copies it to test_results/.

Run this in a separate terminal at any point — before OR after SITL starts.

Usage:
    python sitl/sitl_record.py                   # 60 s → test_results/
    python sitl/sitl_record.py --duration 90
    python sitl/sitl_record.py --name baseline_windy
    python sitl/sitl_record.py --name rl_windy --duration 60
"""
import argparse
import glob
import os
import shutil
import time
from datetime import datetime

LOG_ROOT = os.path.expanduser(
    "~/PX4-Autopilot/build/px4_sitl_default/rootfs/log")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")


def find_active_ulg(timeout=120, poll=1.0):
    """Return the path of the .ulg currently being written (growing file)."""
    print(f"[REC] Scanning {LOG_ROOT} for active log ...", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        ulgs = glob.glob(os.path.join(LOG_ROOT, "**", "*.ulg"), recursive=True)
        if ulgs:
            # Pick newest by modification time
            newest = max(ulgs, key=os.path.getmtime)
            size0  = os.path.getsize(newest)
            time.sleep(1.5)
            size1  = os.path.getsize(newest)
            if size1 > size0:
                print(f"[REC] Active log: {newest}  ({size1/1024:.0f} KB, growing)",
                      flush=True)
                return newest
            else:
                # File exists but not growing — SITL may be paused or not yet flying
                age = time.monotonic() - os.path.getmtime(newest)
                print(f"[REC] Found {os.path.basename(newest)} "
                      f"but not growing (age {age:.0f}s). "
                      f"Is SITL running? Retrying...", flush=True)
        else:
            print("[REC] No .ulg found yet. Start SITL + arm the drone...", flush=True)
        time.sleep(poll)
    return None


def record(path, duration, poll=2.0):
    print(f"[REC] Recording {duration:.0f} s from {os.path.basename(path)} ...",
          flush=True)
    t0 = time.monotonic()
    last_print = -10.0
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= duration:
            break
        if elapsed - last_print >= 10.0:
            size_kb   = os.path.getsize(path) / 1024
            remaining = duration - elapsed
            print(f"[REC]   {elapsed:4.0f} / {duration:.0f} s   "
                  f"size={size_kb:.0f} KB   remaining={remaining:.0f}s", flush=True)
            last_print = elapsed
        time.sleep(poll)
    print(f"[REC] Done. Final size: {os.path.getsize(path)/1024:.0f} KB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--duration", type=float, default=60.0,
                    help="Seconds to record. Default 60.")
    ap.add_argument("--name", default=None,
                    help="Output filename stem (e.g. 'baseline_windy'). "
                         "Default: auto timestamp.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir or OUT_DIR))
    os.makedirs(out_dir, exist_ok=True)

    ulg = find_active_ulg(timeout=180)
    if ulg is None:
        print("[REC] TIMEOUT: no growing log found in 3 minutes.")
        return 1

    record(ulg, args.duration)

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.name if args.name else ts
    dest = os.path.join(out_dir, f"{stem}.ulg")
    shutil.copy2(ulg, dest)

    print(f"\n[REC] ✓ Saved  →  {dest}")
    print(f"\n  Diagnostics:")
    print(f"    python sitl/diag_run.py {dest}")
    print(f"\n  Extract CSV:")
    print(f"    python sitl/ulog_to_eval_csv.py {dest} "
          f"-o {out_dir}/{stem}.csv --always-active 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
