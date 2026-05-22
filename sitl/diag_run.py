#!/usr/bin/env python3
"""diag_run.py RUN.ulg — did the drone actually arm, fly, get disturbed?

Sanity-gate before A/B comparison. Prints PASS/FAIL on:
  armed, climbed (>1 m), disturbed (attitude excursion), RL active.
"""
import sys
import numpy as np
from pyulog import ULog

u = ULog(sys.argv[1])
D = {d.name: d for d in u.data_list}
print("logged topics:", sorted(D))


def arr(topic, field):
    return np.array(D[topic].data[field]) if topic in D and field in D[topic].data else None


# armed
arm = arr("vehicle_status", "arming_state")
if arm is not None:
    print(f"armed: {'YES' if (arm == 2).any() else 'NO — never ARMED'} "
          f"(states seen: {sorted(set(arm.tolist()))})")
else:
    print("armed: UNKNOWN (vehicle_status not logged)")

# altitude
z = arr("vehicle_local_position", "z")
if z is not None:
    print(f"max climb: {(-z).max():.2f} m  "
          f"{'PASS' if (-z).max() > 1.0 else 'FAIL — never left ground'}")
else:
    print("altitude: UNKNOWN (vehicle_local_position not logged)")

# attitude excursion (was it actually disturbed?)
qd = D.get("vehicle_attitude")
if qd:
    q = np.stack([np.array(qd.data[f"q[{i}]"]) for i in range(4)], 1)
    w, x, y, zz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.degrees(np.arctan2(2*(w*x+y*zz), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-zz*x), -1, 1)))
    pk = max(np.abs(roll).max(), np.abs(pitch).max())
    print(f"peak attitude: |roll|={np.abs(roll).max():.2f}° "
          f"|pitch|={np.abs(pitch).max():.2f}°")
    if pk < 1.0:
        print("  → FAIL: never disturbed (motionless — not a valid run)")
    elif pk > 60.0:
        print("  → CRASH: exceeded 60° (diverged)")
    else:
        print("  → disturbed and bounded")

# RL gains
g = D.get("rl_rate_gains")
if g:
    v = np.array(g.data["valid"]).astype(int)
    kp = np.array(g.data["kp"])
    if v.any():
        m = v == 1
        print(f"rl_rate_gains: {int(v.sum())}/{len(v)} valid  "
              f"kp[{kp[m].min():.4f},{kp[m].max():.4f}]  → RL ACTIVE")
    else:
        print(f"rl_rate_gains: 0/{len(v)} valid  → RL NOT active (baseline)")
else:
    print("rl_rate_gains: absent → baseline run")
