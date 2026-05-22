# Deployment Contract — Joint Roll+Pitch Actor (1.05M Steps)

**ONNX artifact**: `export/actor_joint_1p05M_shared3D.onnx`  
**Source model**: `results/frozen_joint_1p05M_shared3D/td3_pid_interrupted.zip`  
**Architecture**: Linear(13,64)→ReLU→Linear(64,64)→ReLU→Linear(64,3)→Tanh  
**Parameters**: 5,251 (FP32 ≈ 21 KB; INT8 ≈ 5–6 KB after PTQ)

---

## Input — obs[13], float32, shape [1, 13]

| Index | Name | Unit | How to compute |
|---|---|---|---|
| 0 | `roll` | rad | Euler roll from attitude quaternion |
| 1 | `pitch` | rad | Euler pitch from attitude quaternion |
| 2 | `roll_rate` | rad/s | **Body-frame** roll rate: `R.T @ ω_world` |
| 3 | `pitch_rate` | rad/s | **Body-frame** pitch rate: `R.T @ ω_world` |
| 4 | `roll_rate_err` | rad/s | `KP_ATT × (−roll) − roll_rate` |
| 5 | `pitch_rate_err` | rad/s | `KP_ATT × (−pitch) − pitch_rate` |
| 6 | `kp_roll_norm` | — | `Kp_roll / 1.72` |
| 7 | `ki_roll_norm` | — | `Ki_roll / 0.172` |
| 8 | `kd_roll_norm` | — | `Kd_roll / 8.6e-3` |
| 9 | `kp_pitch_norm` | — | `Kp_pitch / 1.72` (always == obs[6]) |
| 10 | `ki_pitch_norm` | — | `Ki_pitch / 0.172` (always == obs[7]) |
| 11 | `kd_pitch_norm` | — | `Kd_pitch / 8.6e-3` (always == obs[8]) |
| 12 | `step_prog` | — | See §step_prog below |

**KP_ATT = 3.0** (fixed outer attitude-P, not tuned by RL)

Obs[6:12] must reflect the *current* running gains on the device — the network uses
them to decide how large a delta to apply. Feeding stale or wrong gains here is a
silent failure mode.

---

## Output — action[3], float32, shape [1, 3], range [−1, 1]

| Index | Name | Meaning |
|---|---|---|
| 0 | `a[0]` | Normalised ΔKp command in [−1, 1] |
| 1 | `a[1]` | Normalised ΔKi command in [−1, 1] |
| 2 | `a[2]` | Normalised ΔKd command in [−1, 1] |

Output is already saturated by Tanh. Do not re-clip on the device.

---

## Gain update formula (every control tick, 48 Hz)

```
DELTA_SCALE = [3.4e-2,  3.4e-3,  1.7e-4]

dKp = a[0] * 3.4e-2
dKi = a[1] * 3.4e-3
dKd = a[2] * 1.7e-4

Kp_new = clip(Kp + dKp,  0.0,   1.72)
Ki_new = clip(Ki + dKi,  0.0,   0.172)
Kd_new = clip(Kd + dKd,  0.0,   8.6e-3)

// Applied identically to BOTH roll and pitch (shared action):
Kp_roll = Kp_pitch = Kp_new
Ki_roll = Ki_pitch = Ki_new
Kd_roll = Kd_pitch = Kd_new
```

---

## Gain bounds (hard clip, onboard)

| Gain | Min | Max |
|---|---|---|
| Kp | 0.0 | 1.72 |
| Ki | 0.0 | 0.172 |
| Kd | 0.0 | 8.6e-3 |

---

## Default (initial) gains

```
Kp_init = 0.171
Ki_init = 8.6e-3
Kd_init = 1.71e-3
```

Reset to these at the start of every flight / arm event.

---

## step_prog (obs[12])

During training: `step_prog = step_count / 500`.

**In deployment: hold at 1.0.**

Rationale: the policy learned that `step_prog = 1.0` means sustained steady flight
(the stability bonus is given at step 500). Fixing it to 1.0 keeps the network in
its steady-state operating mode from the first inference call. Starting at 0.0 would
mimic episode-start behaviour (more exploratory/aggressive).

If your firmware naturally increments a time counter, you may ramp from 0→1 over
10.4 s (500 steps × 1/48 Hz) — both approaches are valid. 1.0-fixed is simpler.

---

## Body-frame rate requirement

Indices 2–3 **must** be body-frame angular rates, not world-frame.

```
R       = rotation_matrix_from_quaternion(q)   // 3×3
omega_b = R.T @ omega_world                    // body-frame
roll_rate  = omega_b[0]
pitch_rate = omega_b[1]
```

Feeding world-frame rates here is a silent failure: the network receives
plausible numbers but the wrong physical meaning, causing incorrect gain updates.

---

## Rate PID implementation (must match training exactly)

The actor outputs gain deltas. The PID that consumes them must match the env:

```
// 1-pole IIR derivative filter (fc ≈ 30 Hz at 48 Hz ctrl)
ALPHA = 0.797
d_raw      = -(rate - rate_prev) / dt
d_filtered = ALPHA * d_filtered + (1.0 - ALPHA) * d_raw

// Torque-space anti-windup
integral  += err * dt
tau_I      = Ki * integral
tau_I      = clip(tau_I, -0.30*MAX_XY_TORQUE, +0.30*MAX_XY_TORQUE)
integral   = tau_I / (Ki + 1e-12)   // back-calculate

// PID torque
tau = Kp * err + tau_I + Kd * d_filtered
tau = clip(tau, -MAX_XY_TORQUE, +MAX_XY_TORQUE)
```

`MAX_XY_TORQUE`: read once from the F450 motor/geometry model in PyBullet and
hardcode in firmware. Do not approximate.

---

## Safety fallback rules

Apply these checks every tick **before** writing new gains to the PID:

| Condition | Threshold | Action |
|---|---|---|
| Attitude limit | \|roll\| > 60° or \|pitch\| > 60° | Freeze gains at Kp_init/Ki_init/Kd_init |
| Altitude limit | z < 0.15 m or z > 2.5 m | Freeze gains |
| Gain delta rate | Any \|dGain\| > 2 × DELTA_SCALE[i] per tick | Clamp delta, log fault |
| Obs range fault | \|rates\| > 20 rad/s or norm(obs[6:12]) > 1.05 | Freeze gains, log fault |
| Sensor watchdog | No new obs within 2× control period (> 42 ms) | Freeze gains at last good |

---

## STM32Cube.AI validation workflow

1. Run `stedgeai validate` against the ONNX file with the test vectors in
   `export/actor_joint_1p05M_test_vectors.json` (or the CSV equivalent).
2. Expected FP32 tolerance: max |Δaction| < **1e-5** (pure floating-point rounding).
3. Expected INT8 PTQ tolerance: max |Δaction| < **0.05** per element (≈ 1–3% of
   the [−1,1] range). After gain scaling this is:
   - ΔKp error < 1.7e-3  (0.1% of the 1.72 Kp range)
   - ΔKi error < 1.7e-4
   - ΔKd error < 8.5e-6
4. If any test vector exceeds these tolerances, do not proceed to hardware.
5. Calibration dataset for PTQ: collect ≥ 500 representative obs vectors from
   PyBullet rollouts (or Gazebo SITL) covering all 6 grid conditions.

---

## stedgeai CLI quick reference

```bash
# Generate deterministic .npy test vectors (run once)
cd ~/rl_pid_tuner
python export/gen_test_vectors.py   # writes JSON + CSV
python - <<'EOF'
import numpy as np, json
with open("export/actor_joint_1p05M_test_vectors.json") as f:
    vecs = json.load(f)["vectors"]
obs = np.array([v["obs"] for v in vecs], dtype=np.float32).reshape(-1, 1, 1, 13)
act = np.array([v["action"] for v in vecs], dtype=np.float32).reshape(-1, 1, 1, 3)
np.save("export/testvec_obs.npy", obs)
np.save("export/testvec_action.npy", act)
EOF

# Validate ONNX (FP32, no quantization) against deterministic test vectors
~/x_cube/Utilities/linux/stedgeai validate \
    --model export/actor_joint_1p05M_shared3D.onnx \
    --target stm32 \
    --valinput  export/testvec_obs.npy \
    --valoutput export/testvec_action.npy

# Generate C code
~/x_cube/Utilities/linux/stedgeai generate \
    --model export/actor_joint_1p05M_shared3D.onnx \
    --target stm32 \
    --output export/stm32_c_code/
```

---

## What is NOT in the ONNX graph

The following must be reproduced exactly in firmware — they are outside the network:

1. Obs construction (indices 0–12 above, including body-rate transform)
2. `DELTA_SCALE` multiply
3. Running-gain integrator (Kp/Ki/Kd state variables)
4. Gain clipping to bounds
5. Rate PID (D-filter, anti-windup, torque clipping)
6. Safety fallback rules

These are the deployment correctness surface. Test each one independently against
the Python env before closing the loop on hardware.
