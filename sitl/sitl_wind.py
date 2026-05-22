#!/usr/bin/env python3
"""
sitl_wind.py — Stochastic turbulent wind via Ornstein-Uhlenbeck process.

Two independent OU processes drive tx and ty continuously at 10 Hz.
OU gives mean-reverting colored noise: realistic sustained gusts that
wander, reverse, and occasionally spike — nothing like deterministic sines.

Physics: dx = θ(0 - x)dt + σ√dt·N(0,1), clamped to [-1,1], scaled by amp·envelope.

  θ (theta)  — reversion rate: how quickly gusts die down and reverse.
               0.1 = very sluggish/persistent, 0.5 = choppy/turbulent (default 0.25)
  σ (sigma)  — noise intensity: how violently the gust wanders.
               0.6 produces realistic spread over 60 s (default 0.6)

Envelope: 3 s ramp-in, 3 s ramp-out, guaranteed triple wrench-clear.

Unlike sitl_disturb.py (discrete impulse kicks), this is SUSTAINED and NON-PERIODIC.
Use it to test sustained RL adaptation in isolation, or run together with
sitl_disturb.py to stack discrete kicks on top of a background wind.

Usage:
    python sitl/sitl_wind.py                        # 60 s, 0.12 N·m peak
    python sitl/sitl_wind.py --max 0.18 --duration 90
    python sitl/sitl_wind.py --theta 0.1 --sigma 0.8   # slow heavy gusts
    python sitl/sitl_wind.py --theta 0.5 --sigma 0.5   # choppy turbulence
    python sitl/sitl_wind.py --dry-run              # print profile, don't apply
    python sitl/sitl_wind.py --event-log /tmp/wind_events.csv
"""
import argparse
import csv
import math
import os
import random
import subprocess
import time

DEFAULT_MODEL = "f450_rl_0"
ENVELOPE_CAP = 0.20   # hard ceiling — never exceed validated single-axis envelope


def detect_world():
    try:
        r = subprocess.run(["gz", "topic", "-l"],
                           capture_output=True, text=True, timeout=5.0)
        for line in r.stdout.splitlines():
            parts = line.strip().split("/")
            if (len(parts) == 5 and parts[1] == "world"
                    and parts[3] == "wrench" and parts[4] == "persistent"):
                return parts[2]
    except Exception:
        pass
    return os.environ.get("PX4_GZ_WORLD", "default")


WORLD = detect_world()
RATE_HZ = 10.0        # wrench re-publish rate


class OUProcess:
    """Ornstein-Uhlenbeck mean-reverting stochastic process, clamped to [-1, 1]."""
    def __init__(self, theta=0.25, sigma=0.6, seed=None):
        self.theta = theta
        self.sigma = sigma
        self.x = 0.0
        if seed is not None:
            random.seed(seed)

    def step(self, dt):
        dw = random.gauss(0.0, 1.0)
        self.x += -self.theta * self.x * dt + self.sigma * math.sqrt(dt) * dw
        self.x = max(-1.0, min(1.0, self.x))
        return self.x


def gz_pub_wrench(tx, ty, model_name, world=WORLD):
    topic = f"/world/{world}/wrench/persistent"
    msg = (f'entity: {{name: "{model_name}::base_link", type: LINK}}, '
           f'wrench: {{torque: {{x: {tx:.6f}, y: {ty:.6f}, z: 0.0}}}}')
    cmd = ["gz", "topic", "-t", topic, "-m", "gz.msgs.EntityWrench",
           "-p", msg, "--num", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
    return r.returncode == 0


def gz_clear_wrench(model_name, world=WORLD):
    cmd = ["gz", "topic", "-t", f"/world/{world}/wrench/clear",
           "-m", "gz.msgs.Entity",
           "-p", f'name: "{model_name}::base_link", type: LINK',
           "--num", "1"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model",    default=DEFAULT_MODEL)
    ap.add_argument("--world",    default=WORLD)
    ap.add_argument("--max",      type=float, default=0.12,
                    help="Peak torque N·m (capped at 0.20). Default 0.12.")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="Total wind duration s (incl. ramp in/out). Default 60.")
    ap.add_argument("--theta",    type=float, default=0.25,
                    help="OU reversion rate (0.1=slow/persistent, 0.5=choppy). "
                         "Default 0.25.")
    ap.add_argument("--sigma",    type=float, default=0.6,
                    help="OU noise intensity (higher = more violent). Default 0.6.")
    ap.add_argument("--seed",     type=int, default=None,
                    help="Random seed for reproducible profiles.")
    ap.add_argument("--event-log", default=None,
                    help="Write a one-row events CSV (wind active window) for "
                         "ulog_to_eval_csv.py / ab_compare.py.")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Simulate and print profile; don't apply.")
    args = ap.parse_args()

    amp  = min(args.max, ENVELOPE_CAP)
    ramp = min(3.0, args.duration / 4.0)
    dt   = 1.0 / RATE_HZ

    if args.max > ENVELOPE_CAP:
        print(f"[WIND] --max {args.max} clamped to envelope cap {ENVELOPE_CAP}")
    print(f"[WIND] model={args.model}  peak={amp:.3f} N·m  "
          f"duration={args.duration:.0f}s  ramp={ramp:.1f}s  "
          f"rate={RATE_HZ:.0f}Hz")
    print(f"[WIND] OU: θ={args.theta}  σ={args.sigma}  "
          f"seed={'random' if args.seed is None else args.seed}")

    ou_x = OUProcess(theta=args.theta, sigma=args.sigma, seed=args.seed)
    ou_y = OUProcess(theta=args.theta, sigma=args.sigma,
                     seed=None if args.seed is None else args.seed + 1)

    if args.dry_run:
        print("\n  t(s)    tx(N·m)   ty(N·m)   env")
        t = 0.0
        while t <= args.duration:
            env = 1.0
            if t < ramp:
                env = t / ramp
            elif t > args.duration - ramp:
                env = max(0.0, (args.duration - t) / ramp)
            raw_x = ou_x.step(dt if t > 0 else 0.0)
            raw_y = ou_y.step(dt if t > 0 else 0.0)
            tx = raw_x * amp * env
            ty = raw_y * amp * env
            print(f"  {t:5.1f}   {tx:+.4f}    {ty:+.4f}    {env:.2f}")
            t += 5.0
        print("[WIND] --dry-run: not applied.")
        return

    gz_clear_wrench(args.model, args.world)
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
            raw_x = ou_x.step(dt)
            raw_y = ou_y.step(dt)
            tx = raw_x * amp * env
            ty = raw_y * amp * env
            gz_pub_wrench(tx, ty, args.model, args.world)
            if t - last_print >= 5.0:
                print(f"  t={t:5.1f}s  tx={tx:+.4f}  ty={ty:+.4f}  "
                      f"env={env:.2f}  (ou_x={raw_x:+.3f} ou_y={raw_y:+.3f})",
                      flush=True)
                last_print = t
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[WIND] interrupted — clearing.")
    finally:
        for _ in range(3):
            gz_clear_wrench(args.model, args.world)
            time.sleep(0.3)
        print("[WIND] wind off, wrenches cleared (x3). "
              "Watch ~10 s of recovery, then land + stop SITL.")
        time.sleep(10.0)
        gz_clear_wrench(args.model, args.world)

    if args.event_log:
        with open(args.event_log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event", "t_apply_s", "t_clear_s",
                        "torque_x", "torque_y"])
            w.writerow(["ou_wind", f"{ramp:.3f}",
                        f"{args.duration - ramp:.3f}",
                        f"{amp:.6f}", f"{amp:.6f}"])
        print(f"[WIND] event log → {args.event_log}")


if __name__ == "__main__":
    main()
