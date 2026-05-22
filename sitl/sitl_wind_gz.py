#!/usr/bin/env python3
"""
sitl_wind_gz.py — Dynamic stochastic wind via Gazebo's built-in Wind system.

Publishes varying wind velocity to /world/<world>/wind using gz.msgs.Wind.
Gazebo applies aerodynamic drag forces to the drone body (physically realistic —
the drone has to lean and thrust to fight the wind, not just absorb a torque kick).

Requires: PX4_GZ_WORLD=windy (or any world with the Wind plugin loaded).
The F450 RL model must have <enable_wind>true</enable_wind> on base_link (already done).

Wind model: two independent Ornstein-Uhlenbeck processes on Vx and Vy,
mean-reverting to a configurable base wind vector.

  θ (theta)   — reversion rate toward mean (0.05=slow gusts, 0.3=choppy)
  σ (sigma)   — noise intensity in m/s
  --vx/--vy   — mean wind velocity components (m/s); windy.sdf default = 5/2

The Z component is always 0 (no vertical wind — the drone is a quadrotor, not a glider).

Usage:
    # Dynamic version of the default windy world (5 m/s X, 2 m/s Y with gusts):
    python sitl/sitl_wind_gz.py

    # Calm with occasional gusts:
    python sitl/sitl_wind_gz.py --vx 2 --vy 0 --sigma 2.0 --theta 0.08

    # Strong gusty crosswind:
    python sitl/sitl_wind_gz.py --vx 6 --vy 3 --sigma 3.0 --theta 0.15 --max-gust 12

    # Reproduce a specific run:
    python sitl/sitl_wind_gz.py --seed 42

    # Preview profile without applying:
    python sitl/sitl_wind_gz.py --dry-run

    # Match duration to a disturbance run and log event window:
    python sitl/sitl_wind_gz.py --duration 90 --event-log /tmp/wind_events.csv
"""
import argparse
import csv
import math
import os
import random
import subprocess
import time

RATE_HZ = 10.0


def detect_world(fallback="windy"):
    try:
        r = subprocess.run(["gz", "topic", "-l"],
                           capture_output=True, text=True, timeout=5.0)
        for line in r.stdout.splitlines():
            parts = line.strip().split("/")
            if (len(parts) >= 4 and parts[1] == "world" and parts[3] == "wind"
                    and len(parts) == 4):
                return parts[2]
    except Exception:
        pass
    return os.environ.get("PX4_GZ_WORLD", fallback)


WORLD_DEFAULT = detect_world()


class OUProcess:
    """Ornstein-Uhlenbeck process around a fixed mean."""
    def __init__(self, mean=0.0, theta=0.10, sigma=2.0):
        self.mean  = mean
        self.theta = theta
        self.sigma = sigma
        self.x     = mean

    def step(self, dt):
        dw    = random.gauss(0.0, 1.0)
        self.x += self.theta * (self.mean - self.x) * dt + self.sigma * math.sqrt(dt) * dw
        return self.x


def gz_set_wind(vx, vy, world):
    """Publish gz.msgs.Wind to /world/<world>/wind."""
    topic = f"/world/{world}/wind"
    msg   = (f"linear_velocity: {{x: {vx:.4f}, y: {vy:.4f}, z: 0.0}}, "
             f"enable_wind: true")
    cmd   = ["gz", "topic", "-t", topic, "-m", "gz.msgs.Wind",
             "-p", msg, "--num", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
    return r.returncode == 0


def gz_disable_wind(world):
    topic = f"/world/{world}/wind"
    msg   = "linear_velocity: {x: 0.0, y: 0.0, z: 0.0}, enable_wind: false"
    cmd   = ["gz", "topic", "-t", topic, "-m", "gz.msgs.Wind",
             "-p", msg, "--num", "1"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--world",    default=WORLD_DEFAULT,
                    help=f"Gazebo world name (default: {WORLD_DEFAULT}). "
                         "Must have Wind plugin loaded.")
    ap.add_argument("--vx",       type=float, default=5.0,
                    help="Mean wind X velocity m/s (default 5.0).")
    ap.add_argument("--vy",       type=float, default=2.0,
                    help="Mean wind Y velocity m/s (default 2.0).")
    ap.add_argument("--sigma",    type=float, default=2.0,
                    help="OU noise intensity m/s (default 2.0).")
    ap.add_argument("--theta",    type=float, default=0.10,
                    help="OU reversion rate (0.05=slow/persistent, 0.3=choppy). "
                         "Default 0.10.")
    ap.add_argument("--max-gust", type=float, default=None,
                    help="Hard clamp on wind speed magnitude m/s (default: unclamped).")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="Total wind duration s. Default 60.")
    ap.add_argument("--seed",     type=int,   default=None,
                    help="Random seed for reproducible profiles.")
    ap.add_argument("--event-log", default=None,
                    help="Write a one-row events CSV (wind active window) for "
                         "ulog_to_eval_csv.py / ab_compare.py.")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    ou_x = OUProcess(mean=args.vx, theta=args.theta, sigma=args.sigma)
    ou_y = OUProcess(mean=args.vy, theta=args.theta, sigma=args.sigma)

    dt   = 1.0 / RATE_HZ
    ramp = min(3.0, args.duration / 4.0)
    max_g = args.max_gust

    print(f"[WINDGZ] world={args.world}  mean=({args.vx:.1f},{args.vy:.1f}) m/s  "
          f"σ={args.sigma}  θ={args.theta}  "
          f"seed={'random' if args.seed is None else args.seed}")
    print(f"[WINDGZ] duration={args.duration:.0f}s  ramp={ramp:.1f}s  "
          f"max_gust={'none' if max_g is None else f'{max_g:.1f} m/s'}")

    if args.dry_run:
        print("\n  t(s)    Vx(m/s)   Vy(m/s)   |V|    env")
        t = 0.0
        while t <= args.duration:
            env = 1.0
            if t < ramp:
                env = t / ramp
            elif t > args.duration - ramp:
                env = max(0.0, (args.duration - t) / ramp)
            vx = ou_x.step(dt if t > 0 else 0.0) * env
            vy = ou_y.step(dt if t > 0 else 0.0) * env
            spd = math.hypot(vx, vy)
            print(f"  {t:5.1f}   {vx:+7.3f}   {vy:+7.3f}   {spd:5.2f}   {env:.2f}")
            t += 5.0
        print("[WINDGZ] --dry-run: not applied.")
        return

    t0 = time.monotonic()
    last_print = -5.0
    try:
        while True:
            t = time.monotonic() - t0
            if t >= args.duration:
                break
            env = 1.0
            if t < ramp:
                env = t / ramp
            elif t > args.duration - ramp:
                env = max(0.0, (args.duration - t) / ramp)

            vx = ou_x.step(dt) * env
            vy = ou_y.step(dt) * env

            if max_g is not None:
                spd = math.hypot(vx, vy)
                if spd > max_g:
                    scale = max_g / spd
                    vx *= scale
                    vy *= scale

            gz_set_wind(vx, vy, args.world)

            if t - last_print >= 5.0:
                spd = math.hypot(vx, vy)
                print(f"  t={t:5.1f}s  Vx={vx:+6.2f}  Vy={vy:+6.2f}  "
                      f"|V|={spd:5.2f} m/s  env={env:.2f}", flush=True)
                last_print = t
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[WINDGZ] interrupted — disabling wind.")
    finally:
        for _ in range(3):
            gz_disable_wind(args.world)
            time.sleep(0.3)
        print("[WINDGZ] wind disabled (x3). Watch ~10 s recovery, then land.")
        time.sleep(10.0)
        gz_disable_wind(args.world)

    if args.event_log:
        with open(args.event_log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event", "t_apply_s", "t_clear_s",
                        "torque_x", "torque_y"])
            w.writerow(["gz_wind", f"{ramp:.3f}",
                        f"{args.duration - ramp:.3f}", "0.0", "0.0"])
        print(f"[WINDGZ] event log → {args.event_log}")


if __name__ == "__main__":
    main()
