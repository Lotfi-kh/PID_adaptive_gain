#!/usr/bin/env python3
"""
sitl_disturb.py — Apply timed torque disturbances to the Gazebo Harmonic vehicle.

Publishes to /world/default/wrench/persistent via the gz CLI.
Run this in a SECOND terminal while sitl_observer.py is logging.

The script applies a predefined disturbance sequence and prints a timeline so
you can match timestamps in the observer CSV to know when each kick happened.

Usage:
    # Default sequence (roll kick, pitch kick, combined, strong roll):
    python sitl/sitl_disturb.py

    # Custom single roll kick:
    python sitl/sitl_disturb.py --roll 0.4 --t0 10 --duration 3

    # Custom pitch kick:
    python sitl/sitl_disturb.py --pitch 0.4 --t0 10 --duration 3

    # List mode: print the sequence that will run without executing:
    python sitl/sitl_disturb.py --dry-run

Model name:
    PX4 Gazebo Harmonic spawns the model as '<PX4_SIM_MODEL>_0', i.e. 'f450_rl_0'.
    If your SITL uses a different name, override with --model.
    To check: run `gz model --list` while SITL is running.

Physics note:
    Torques are applied in the WORLD frame by Gazebo's wrench system.
    Units: N·m.  Values of 0.1–0.5 N·m are sufficient for the F450-mass model
    (Ixx=0.012) to produce roll/pitch excitations in the ±5–20°/s range.
"""
import argparse
import csv
import os
import subprocess
import sys
import time


DEFAULT_MODEL = "f450_rl_0"  # PX4 spawns as <PX4_SIM_MODEL>_<px4_instance>


def detect_world():
    """Read the active Gazebo world name from the running gz instance."""
    try:
        r = subprocess.run(["gz", "topic", "-l"],
                           capture_output=True, text=True, timeout=5.0)
        for line in r.stdout.splitlines():
            parts = line.strip().split("/")
            # pattern: /world/<name>/wrench/persistent
            if (len(parts) == 5 and parts[1] == "world"
                    and parts[3] == "wrench" and parts[4] == "persistent"):
                return parts[2]
    except Exception:
        pass
    return os.environ.get("PX4_GZ_WORLD", "default")


WORLD = detect_world()


# ── Predefined disturbance sequence ─────────────────────────────────────────
# Each entry: (label, t_start_s, duration_s, torque_x_Nm, torque_y_Nm)
# torque_x = roll axis,  torque_y = pitch axis  (world frame)
DEFAULT_SEQUENCE = [
    ("mild roll kick",      10.0,  2.0,  0.15, 0.00),
    ("mild pitch kick",     20.0,  2.0,  0.00, 0.15),
    ("combined kick",       32.0,  2.0,  0.15, 0.15),
    ("strong roll kick",    45.0,  3.0,  0.40, 0.00),
    ("strong pitch kick",   55.0,  3.0,  0.00, 0.40),
    ("strong combined",     68.0,  3.0,  0.20, 0.20),
    ("brief sharp roll",    80.0,  1.0,  0.30, 0.00),
]

# In-envelope sequence: every event is within the SITL-validated recoverable
# envelope (<=0.40 single-axis / 0.20+0.20 combined). NO 0.30 sharp-roll
# (documented unrecoverable, out of RL scope). Ends with a long calm tail so
# post-disturbance recovery is observed before the run stops.
SAFE_SEQUENCE = [
    ("mild roll kick",      10.0,  2.0,  0.15, 0.00),
    ("mild pitch kick",     20.0,  2.0,  0.00, 0.15),
    ("combined kick",       32.0,  2.0,  0.15, 0.15),
    ("strong roll kick",    45.0,  3.0,  0.40, 0.00),
    ("strong pitch kick",   58.0,  3.0,  0.00, 0.40),
    ("combined sustained",  72.0,  3.0,  0.20, 0.20),
    # then 15 s of NO disturbance to watch both controllers settle.
]


def gz_pub_wrench(tx, ty, tz, model_name, world=WORLD, topic_suffix="persistent"):
    """Publish an EntityWrench to /world/<world>/wrench/<topic_suffix>."""
    topic = f"/world/{world}/wrench/{topic_suffix}"
    msg = (
        f'entity: {{name: "{model_name}::base_link", type: LINK}}, '
        f'wrench: {{torque: {{x: {tx:.6f}, y: {ty:.6f}, z: {tz:.6f}}}}}'
    )
    cmd = ["gz", "topic", "-t", topic, "-m", "gz.msgs.EntityWrench",
           "-p", msg, "--num", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
    if result.returncode != 0:
        print(f"  [WARN] gz topic failed: {result.stderr.strip()}", flush=True)
        return False
    return True


def gz_clear_wrench(model_name, world=WORLD):
    """Clear all persistent wrenches on base_link."""
    topic = f"/world/{world}/wrench/clear"
    msg = f'name: "{model_name}::base_link", type: LINK'
    cmd = ["gz", "topic", "-t", topic, "-m", "gz.msgs.Entity",
           "-p", msg, "--num", "1"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Gazebo model name (default: {DEFAULT_MODEL}). "
                         "Run 'gz model --list' while SITL is running to verify.")
    ap.add_argument("--world", default=WORLD)
    ap.add_argument("--roll",     type=float, default=None,
                    help="Single-kick: roll torque in N·m")
    ap.add_argument("--pitch",    type=float, default=None,
                    help="Single-kick: pitch torque in N·m")
    ap.add_argument("--t0",       type=float, default=10.0,
                    help="Single-kick: seconds before applying (default 10)")
    ap.add_argument("--duration", type=float, default=2.0,
                    help="Single-kick: duration in seconds (default 2)")
    ap.add_argument("--safe", action="store_true",
                    help="Use the in-envelope sequence (<=0.40 single / "
                         "0.20+0.20 combined, no 0.30 sharp roll, calm tail). "
                         "Use this for a fair baseline-vs-RL A/B.")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print the sequence without executing")
    ap.add_argument("--event-log", default=None,
                    help="Write a CSV of disturbance events for A/B alignment: "
                         "event,t_apply_s,t_clear_s,torque_x,torque_y "
                         "(t_*_s = seconds since this script started).")
    return ap.parse_args()


def write_event_log(path, events):
    """events: list of dicts with event,t_apply_s,t_clear_s,torque_x,torque_y."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "t_apply_s", "t_clear_s", "torque_x", "torque_y"])
        for e in events:
            w.writerow([e["event"], f"{e['t_apply_s']:.3f}",
                        f"{e['t_clear_s']:.3f}",
                        f"{e['torque_x']:.6f}", f"{e['torque_y']:.6f}"])
    print(f"[DISTURB] Event log written: {path}", flush=True)


def run_sequence(sequence, model_name, world, dry_run=False, event_log=None):
    t_start = time.monotonic()
    ev_rows = {}   # label -> event dict (filled as APPLY/CLEAR happen)

    print(f"[DISTURB] Model : {model_name}")
    print(f"[DISTURB] World : {world}")
    print(f"[DISTURB] Disturbance sequence ({len(sequence)} events):")
    for label, t0, dur, tx, ty in sequence:
        print(f"  t={t0:5.1f}s  dur={dur:.1f}s  "
              f"torque_x={tx:+.3f} Nm  torque_y={ty:+.3f} Nm  ({label})")
    print()

    if dry_run:
        print("[DISTURB] --dry-run: not executing.")
        return

    # Make sure wrenches are clear before starting
    gz_clear_wrench(model_name, world)

    print("[DISTURB] Running. Match these timestamps to the observer CSV.")
    print(f"  {'Event':<25s}  {'t_apply':>8s}  {'t_clear':>8s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}")

    active = []   # (t_clear, label)

    # Walk through time, firing and clearing as needed
    all_events = []
    for label, t0, dur, tx, ty in sequence:
        all_events.append(("start", t0,       label, tx, ty))
        all_events.append(("stop",  t0 + dur, label, 0.0, 0.0))
    all_events.sort(key=lambda e: e[1])

    last_t = 0.0
    for ev in all_events:
        kind, t_ev, label, tx, ty = ev
        # Sleep until this event
        now = time.monotonic() - t_start
        wait = t_ev - now
        if wait > 0:
            time.sleep(wait)

        wall = time.monotonic() - t_start
        if kind == "start":
            ok = gz_pub_wrench(tx, ty, 0.0, model_name, world)
            status = "OK" if ok else "FAIL"
            print(f"  {label:<25s}  t={wall:6.1f}s  APPLY [{status}]  "
                  f"tx={tx:+.3f} ty={ty:+.3f}", flush=True)
            ev_rows[label] = {"event": label, "t_apply_s": wall,
                              "t_clear_s": float("nan"),
                              "torque_x": tx, "torque_y": ty}
        else:
            gz_clear_wrench(model_name, world)
            print(f"  {label:<25s}  t={wall:6.1f}s  CLEAR", flush=True)
            if label in ev_rows:
                ev_rows[label]["t_clear_s"] = wall

    # Guaranteed multi-clear + calm tail: kills any residual persistent wrench
    # and lets both controllers be observed settling before the run stops.
    for _ in range(3):
        gz_clear_wrench(model_name, world)
        time.sleep(0.3)
    print()
    print("[DISTURB] All wrenches cleared (x3). Calm tail 15 s — "
          "keep SITL + logging running so recovery is captured...", flush=True)
    time.sleep(15.0)
    gz_clear_wrench(model_name, world)
    print("[DISTURB] Sequence complete. Safe to land + stop SITL now.")

    if event_log:
        write_event_log(event_log,
                         sorted(ev_rows.values(), key=lambda e: e["t_apply_s"]))


def main():
    args = parse_args()

    if args.roll is not None or args.pitch is not None:
        # Single custom kick
        tx = args.roll  if args.roll  is not None else 0.0
        ty = args.pitch if args.pitch is not None else 0.0
        sequence = [("custom kick", args.t0, args.duration, tx, ty)]
    elif args.safe:
        sequence = SAFE_SEQUENCE
    else:
        sequence = DEFAULT_SEQUENCE

    try:
        run_sequence(sequence, args.model, args.world,
                     dry_run=args.dry_run, event_log=args.event_log)
    except KeyboardInterrupt:
        print("\n[DISTURB] Interrupted — clearing wrenches.")
        gz_clear_wrench(args.model, args.world)


if __name__ == "__main__":
    main()
