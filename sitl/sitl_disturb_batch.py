#!/usr/bin/env python3
"""
sitl_disturb_batch.py — one command for the whole Gazebo A/B validation batch.

For each seed it flies a MATCHED PAIR in PX4 SITL + Gazebo (headless):
  baseline  (rl_gain_tuner disabled → fixed PX4 gains)
  adaptive  (rl_gain_tuner active   → in-firmware c860k actor)
both hit by the SAME seeded in-envelope disturbance sequence (sitl_disturb.py
--safe --seed). Repeating over seeds turns N=1 into a statistical sample.

Then (bundled) it runs a disturbance-MAGNITUDE SWEEP: one matched pair per level,
a single roll kick of that magnitude, to map where adaptive helps and where the
recoverable envelope ends.

Every flight is reduced to one metrics row by extract_metrics.py (rate-tracking
error, control effort, motor saturation, altitude trade-off, peak/RMS/recovery).
aggregate_stats.py then produces the mean±CI / paired-test / win-rate summary.

Reuses the proven lifecycle from sitl_wind_comparison.py (launch / arm / takeoff /
land / rl toggle) — only the disturbance source and the headless flag differ.

Usage:
    cd ~/rl_pid_tuner
    python sitl/sitl_disturb_batch.py                 # 10 pairs + sweep (~2-3 hr)
    python sitl/sitl_disturb_batch.py --runs 3        # shorter batch
    python sitl/sitl_disturb_batch.py --runs 1 --no-sweep   # pipeline smoke
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sitl_wind_comparison as W   # noqa: E402  (lifecycle/flight/toggle helpers)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
OUT_DIR    = os.path.join(REPO_ROOT, "test_results", "gazebo")
PX4_DIR    = W.PX4_DIR
PYTHON     = sys.executable

DISTURB    = os.path.join(SCRIPT_DIR, "sitl_disturb.py")
EXTRACT    = os.path.join(SCRIPT_DIR, "extract_metrics.py")
AGGREGATE  = os.path.join(SCRIPT_DIR, "aggregate_stats.py")
MASTER_CSV = os.path.join(OUT_DIR, "metrics_master.csv")


def start_sitl_headless(world="default"):
    """Launch PX4 SITL + Gazebo with no GUI window.

    Use the X server on :1 for OFFSCREEN rendering (GZ_HEADLESS_RENDERING=1) —
    the same env the working run_pairs.py uses. HEADLESS=1 + no DISPLAY fails
    here because gz still needs a render context."""
    env = os.environ.copy()
    env["HEADLESS"] = "1"                   # suppress the gz GUI client (no window)
    env.setdefault("DISPLAY", ":1")        # X server for offscreen render context
    env["GZ_HEADLESS_RENDERING"] = "1"     # render offscreen, not to a window
    if world != "default":
        env["PX4_GZ_WORLD"] = world
    proc = subprocess.Popen(
        ["make", "px4_sitl_default", "gz_f450_rl"],
        cwd=PX4_DIR, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    print(f"[SITL] Started headless (PID={proc.pid})  world={world}")
    return proc


def run_flight(controller, disturb_args, event_log, out_ulg,
               init_s, settle_s, alt):
    """One full flight: launch → fly → disturb → land → copy ulg. Returns path."""
    sitl = None
    try:
        sitl = start_sitl_headless()
        print(f"[{controller}] SITL init ({init_s:.0f}s) …")
        time.sleep(init_s)

        # 150s (not 90s): PX4 takes >150s total to heartbeat on this machine
        # when the gz GUI client shares the X server — matches run_pairs.py.
        mav = W.mav_connect("udpin:localhost:14540", timeout=150)
        W.arm_and_takeoff(mav, altitude=alt)

        ulg_src = W.find_active_ulg(timeout=120)
        if ulg_src is None:
            raise RuntimeError(f"[{controller}] no active .ulg")

        print(f"[{controller}] hover {settle_s:.0f}s before disturbance …")
        time.sleep(settle_s)

        # sitl_disturb.py blocks through its own sequence + calm tail
        print(f"[{controller}] disturbance: {' '.join(disturb_args)}")
        subprocess.run([PYTHON, DISTURB, "--event-log", event_log] + disturb_args,
                       check=False)

        print(f"[{controller}] landing …")
        W.land(mav)
        W.wait_log_settle(ulg_src, secs=5.0)

        os.makedirs(OUT_DIR, exist_ok=True)
        shutil.copy2(ulg_src, out_ulg)
        print(f"[{controller}] ✓ {os.path.basename(out_ulg)} "
              f"({os.path.getsize(out_ulg)/1024:.0f} KB)")
        return out_ulg
    finally:
        W.stop_sitl(sitl)


def run_pair(tag, seed, disturb_args, scale_label,
             init_s, settle_s, alt):
    """Baseline then adaptive flight with identical disturbance; extract metrics."""
    stem = f"{tag}_seed{seed}"
    ev   = os.path.join(OUT_DIR, f"{stem}_events.csv")

    for controller, toggle in (("baseline", W.disable_rl_in_rc),
                               ("adaptive", W.enable_rl_in_rc)):
        print("\n" + "─" * 70)
        print(f"{tag.upper()} seed={seed} — {controller.upper()}")
        print("─" * 70)
        toggle()
        out_ulg = os.path.join(OUT_DIR, f"{stem}_{controller}.ulg")
        try:
            run_flight(controller, disturb_args, ev, out_ulg,
                       init_s, settle_s, alt)
        except Exception as e:                       # noqa: BLE001
            print(f"[ERROR] {controller} seed={seed} failed: {e} — skipping pair")
            return False
        finally:
            W.enable_rl_in_rc()                      # never leave RL disabled

        subprocess.run(
            [PYTHON, EXTRACT, out_ulg, "--events", ev, "--append", MASTER_CSV,
             "--controller", controller, "--seed", str(seed),
             "--scale", scale_label, "--tag", tag,
             "-o", os.path.join(OUT_DIR, f"{stem}_{controller}_wide.csv")],
            check=False)
        time.sleep(5.0)   # let processes settle between flights
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--runs", type=int, default=10,
                    help="Number of matched pairs in the seeded batch (default 10)")
    ap.add_argument("--seed0", type=int, default=1000,
                    help="First seed; subsequent pairs use seed0+1, +2, … ")
    ap.add_argument("--sweep", default="0.10,0.20,0.30,0.40,0.50",
                    help="Comma list of single-roll-kick magnitudes (N·m).")
    ap.add_argument("--no-sweep", action="store_true",
                    help="Skip the magnitude sweep (paired batch only).")
    ap.add_argument("--init", type=float, default=60.0,
                    help="Seconds to wait for SITL to boot (default 60)")
    ap.add_argument("--settle", type=float, default=10.0,
                    help="Hover seconds before disturbing (default 10)")
    ap.add_argument("--alt", type=float, default=10.0)
    ap.add_argument("--fresh", action="store_true",
                    help="Start a new master metrics CSV (archive any existing).")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.fresh and os.path.exists(MASTER_CSV):
        bak = MASTER_CSV + time.strftime(".%Y%m%d_%H%M%S.bak")
        shutil.move(MASTER_CSV, bak)
        print(f"[batch] archived old master → {os.path.basename(bak)}")

    print("=" * 70)
    print("GAZEBO A/B VALIDATION BATCH (headless)")
    print(f"  pairs={args.runs}  seeds={args.seed0}..{args.seed0+args.runs-1}  "
          f"sweep={'off' if args.no_sweep else args.sweep}")
    print(f"  out={OUT_DIR}")
    print("=" * 70)
    t_start = time.monotonic()
    ok = fail = 0

    # ── Seeded paired batch ───────────────────────────────────────────────────
    for k in range(args.runs):
        seed = args.seed0 + k
        print(f"\n########## PAIR {k+1}/{args.runs}  (seed {seed}) ##########")
        good = run_pair("pair", seed, ["--safe", "--seed", str(seed)], "1.0",
                        args.init, args.settle, args.alt)
        ok += good; fail += (not good)

    # ── Bundled magnitude sweep ───────────────────────────────────────────────
    if not args.no_sweep:
        levels = [s.strip() for s in args.sweep.split(",") if s.strip()]
        for j, mag in enumerate(levels):
            print(f"\n########## SWEEP {j+1}/{len(levels)}  (roll {mag} N·m) ##########")
            seed = 9000 + j   # stable, distinct from batch seeds
            good = run_pair("sweep", seed,
                            ["--roll", mag, "--t0", "12", "--duration", "3"],
                            mag, args.init, args.settle, args.alt)
            ok += good; fail += (not good)

    mins = (time.monotonic() - t_start) / 60.0
    print("\n" + "=" * 70)
    print(f"BATCH DONE — {ok} pairs ok, {fail} failed, {mins:.0f} min")
    print(f"  master metrics → {MASTER_CSV}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if os.path.exists(MASTER_CSV):
        print("\n[batch] aggregating …")
        subprocess.run([PYTHON, AGGREGATE, MASTER_CSV], check=False)
    print("=" * 70)


def _sigint(*_):
    print("\n[batch] interrupted — restoring rc.mc_apps and killing SITL …")
    try:
        W.enable_rl_in_rc()
    finally:
        for name in ("gzserver", "gz", "px4"):
            subprocess.run(["pkill", "-f", name], capture_output=True)
    sys.exit(130)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sigint)
    main()
