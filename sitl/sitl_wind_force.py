#!/usr/bin/env python3
"""
sitl_wind_force.py — Wind as a sustained, gusting lateral FORCE via Gazebo's
ApplyLinkWrench system (the only disturbance system actually loaded in PX4's gz
server config — the same one the torque-kick injector uses).

Physically this is a wind load on a station-keeping drone: a continuous horizontal
push (world frame) the controller must lean into and hold against, with
Ornstein-Uhlenbeck gusts superimposed.  For a hovering vehicle body velocity stays
~0, so a force is a faithful steady-wind model.

IMPORTANT: this must run under the SYSTEM python (/usr/bin/python3) which has the
gz-transport13 / gz-msgs10 bindings.  A single long-lived Node publishes at 10 Hz
so messages actually deliver — the `gz topic --num 1` CLI exits before the pub/sub
handshake completes and silently drops the wrench.

The F450 weighs ~14.7 N; ~2 N mean force ≈ a steady ~8° lean.  Keep it modest —
15 N flips the aircraft.

Usage (always with system python):
    /usr/bin/python3 sitl/sitl_wind_force.py --duration 90 --seed 1 --event-log ev.csv
    /usr/bin/python3 sitl/sitl_wind_force.py --fx 2 --fy 0.8 --sigma 0.6 --dry-run
"""
import argparse
import csv
import math
import os
import random
import time

RATE_HZ = 10.0
DEFAULT_MODEL = "f450_rl_0"


def detect_world(fallback="windy"):
    return os.environ.get("PX4_GZ_WORLD", fallback)


WORLD_DEFAULT = detect_world()


class OUProcess:
    """Ornstein-Uhlenbeck process around a fixed mean."""
    def __init__(self, mean=0.0, theta=0.10, sigma=0.6):
        self.mean, self.theta, self.sigma = mean, theta, sigma
        self.x = mean

    def step(self, dt):
        dw = random.gauss(0.0, 1.0)
        self.x += self.theta * (self.mean - self.x) * dt + self.sigma * math.sqrt(dt) * dw
        return self.x


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model",    default=DEFAULT_MODEL)
    ap.add_argument("--world",    default=WORLD_DEFAULT)
    ap.add_argument("--fx",       type=float, default=1.5,
                    help="Mean wind force, world-X (N).  ~1.5 N ≈ 6° lean.")
    ap.add_argument("--fy",       type=float, default=0.6,
                    help="Mean wind force, world-Y (N).")
    ap.add_argument("--sigma",    type=float, default=0.5, help="Gust intensity (N).")
    ap.add_argument("--theta",    type=float, default=0.10, help="OU reversion rate.")
    ap.add_argument("--max-gust", type=float, default=3.0,
                    help="Cap on |force| (N) so it stays recoverable.")
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--seed",     type=int,   default=None)
    ap.add_argument("--event-log", default=None)
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    ou_x = OUProcess(mean=args.fx, theta=args.theta, sigma=args.sigma)
    ou_y = OUProcess(mean=args.fy, theta=args.theta, sigma=args.sigma)
    dt   = 1.0 / RATE_HZ
    ramp = min(3.0, args.duration / 4.0)

    print(f"[WINDF] world={args.world}  model={args.model}  "
          f"mean=({args.fx:.1f},{args.fy:.1f}) N  σ={args.sigma}  θ={args.theta}  "
          f"max_gust={args.max_gust} N  seed={'random' if args.seed is None else args.seed}")
    print(f"[WINDF] duration={args.duration:.0f}s  ramp={ramp:.1f}s")

    def envelope(t):
        if t < ramp:
            return t / ramp
        if t > args.duration - ramp:
            return max(0.0, (args.duration - t) / ramp)
        return 1.0

    if args.dry_run:
        print("\n  t(s)    Fx(N)    Fy(N)    |F|   env")
        t = 0.0
        while t <= args.duration:
            fx = ou_x.step(dt if t > 0 else 0.0) * envelope(t)
            fy = ou_y.step(dt if t > 0 else 0.0) * envelope(t)
            print(f"  {t:5.1f}  {fx:+7.3f}  {fy:+7.3f}  {math.hypot(fx,fy):5.2f}  {envelope(t):.2f}")
            t += 5.0
        print("[WINDF] --dry-run: not applied.")
        return

    # --- live: one long-lived gz-transport publisher at 10 Hz ---
    from gz.transport13 import Node
    from gz.msgs10.entity_wrench_pb2 import EntityWrench
    from gz.msgs10.entity_pb2 import Entity

    node = Node()
    topic = f"/world/{args.world}/wrench/persistent"
    clear_topic = f"/world/{args.world}/wrench/clear"
    pub = node.advertise(topic, EntityWrench)
    clear_pub = node.advertise(clear_topic, Entity)
    time.sleep(0.5)   # let discovery/handshake settle so the first msgs deliver

    def make_msg(fx, fy):
        m = EntityWrench()
        m.entity.name = f"{args.model}::base_link"
        m.entity.type = Entity.LINK
        m.wrench.force.x = fx
        m.wrench.force.y = fy
        m.wrench.force.z = 0.0
        return m

    def clear():
        e = Entity()
        e.name = f"{args.model}::base_link"
        e.type = Entity.LINK
        clear_pub.publish(e)

    t0 = time.monotonic()
    last_print = -5.0
    try:
        while True:
            t = time.monotonic() - t0
            if t >= args.duration:
                break
            env = envelope(t)
            fx = ou_x.step(dt) * env
            fy = ou_y.step(dt) * env
            mag = math.hypot(fx, fy)
            if mag > args.max_gust:
                s = args.max_gust / mag
                fx, fy = fx * s, fy * s
            clear()                            # remove previous persistent wrench …
            time.sleep(0.005)                  # … let the clear land first
            pub.publish(make_msg(fx, fy))      # … then set the new one (no accumulation)
            if t - last_print >= 5.0:
                print(f"  t={t:5.1f}s  Fx={fx:+6.2f}  Fy={fy:+6.2f}  "
                      f"|F|={math.hypot(fx,fy):5.2f} N  env={env:.2f}", flush=True)
                last_print = t
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[WINDF] interrupted — clearing force.")
    finally:
        for _ in range(5):
            clear()
            time.sleep(0.2)
        print("[WINDF] force cleared. ~10 s recovery, then land.")
        time.sleep(10.0)
        clear()

    if args.event_log:
        with open(args.event_log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event", "t_apply_s", "t_clear_s", "torque_x", "torque_y"])
            w.writerow(["gz_wind_force", f"{ramp:.3f}",
                        f"{args.duration - ramp:.3f}", "0.0", "0.0"])
        print(f"[WINDF] event log → {args.event_log}")


if __name__ == "__main__":
    main()
