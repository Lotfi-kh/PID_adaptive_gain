# Deployment contract, joint roll+pitch actor (c860k, 12-D)

This file says exactly what the firmware has to do around the ONNX network.
The network alone is not enough, the obs construction, the gain integrator,
the bounds and the safety rules all live outside the ONNX graph and must
match the training environment.

- ONNX file: `export/actor_joint_12d_c860k.onnx`
- Source model: `results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip`
- Architecture: Linear(12,64) -> ReLU -> Linear(64,64) -> ReLU -> Linear(64,3) -> Tanh
- Parameters: 5187 (FP32 about 21 KB, INT8 about 5 to 6 KB after PTQ)

## Input, obs[12], float32, shape [1, 12]

| Index | Name | Unit | How to compute |
|---|---|---|---|
| 0  | `roll`           | rad   | Euler roll from the attitude quaternion |
| 1  | `pitch`          | rad   | Euler pitch from the attitude quaternion |
| 2  | `roll_rate`      | rad/s | Body-frame roll rate, `R.T @ omega_world` |
| 3  | `pitch_rate`     | rad/s | Body-frame pitch rate, `R.T @ omega_world` |
| 4  | `roll_rate_err`  | rad/s | `KP_ATT * (-roll) - roll_rate` |
| 5  | `pitch_rate_err` | rad/s | `KP_ATT * (-pitch) - pitch_rate` |
| 6  | `kp_roll_norm`   |       | `Kp_roll  / 1.72` |
| 7  | `ki_roll_norm`   |       | `Ki_roll  / 0.172` |
| 8  | `kd_roll_norm`   |       | `Kd_roll  / 8.6e-3` |
| 9  | `kp_pitch_norm`  |       | `Kp_pitch / 1.72`  (always equal to obs[6]) |
| 10 | `ki_pitch_norm`  |       | `Ki_pitch / 0.172` (always equal to obs[7]) |
| 11 | `kd_pitch_norm`  |       | `Kd_pitch / 8.6e-3`(always equal to obs[8]) |

`KP_ATT = 3.0` is the fixed outer attitude P gain, it is not tuned by RL.

Important: obs[6:12] must be the current running gains on the device. The
network uses them to decide how big the next delta is. Feeding stale or
wrong gains here is a silent failure mode.

## Output, action[3], float32, shape [1, 3], range [-1, 1]

| Index | Name | Meaning |
|---|---|---|
| 0 | `a[0]` | Normalised dKp command, [-1, 1] |
| 1 | `a[1]` | Normalised dKi command, [-1, 1] |
| 2 | `a[2]` | Normalised dKd command, [-1, 1] |

The output is already saturated by the final Tanh. Do not clip again on
the device.

## Gain update at every control tick (48 Hz)

    DELTA_SCALE = [3.4e-2,  3.4e-3,  1.7e-4]

    dKp = a[0] * 3.4e-2
    dKi = a[1] * 3.4e-3
    dKd = a[2] * 1.7e-4

    Kp_new = clip(Kp + dKp,  0.0,  1.72)
    Ki_new = clip(Ki + dKi,  0.0,  0.172)
    Kd_new = clip(Kd + dKd,  0.0,  8.6e-3)

    // The same gains are applied to both axes (shared action).
    Kp_roll = Kp_pitch = Kp_new
    Ki_roll = Ki_pitch = Ki_new
    Kd_roll = Kd_pitch = Kd_new

## Gain bounds (hard clip onboard)

| Gain | Min | Max |
|---|---|---|
| Kp | 0.0 | 1.72   |
| Ki | 0.0 | 0.172  |
| Kd | 0.0 | 8.6e-3 |

## Initial gains

    Kp_init = 0.171
    Ki_init = 8.6e-3
    Kd_init = 1.71e-3

Reset to these values at the start of every flight or arm event.

## Body-frame rate requirement

Indices 2 and 3 must be body-frame angular rates, not world-frame. World
frame rates have plausible magnitude but wrong physical meaning and will
cause wrong gain updates.

    R          = rotation_matrix_from_quaternion(q)   // 3x3
    omega_b    = R.T @ omega_world                    // body frame
    roll_rate  = omega_b[0]
    pitch_rate = omega_b[1]

## Rate PID (must match the training env exactly)

The actor outputs gain deltas. The PID that uses them has to be the same
as the one in the training env, otherwise the policy is solving a
different problem.

    // Derivative filter, 1-pole IIR, fc about 30 Hz at 48 Hz ctrl
    ALPHA = 0.797
    d_raw      = -(rate - rate_prev) / dt
    d_filtered = ALPHA * d_filtered + (1.0 - ALPHA) * d_raw

    // Torque-space anti-windup
    integral  += err * dt
    tau_I      = Ki * integral
    tau_I      = clip(tau_I, -0.30 * MAX_XY_TORQUE, +0.30 * MAX_XY_TORQUE)
    integral   = tau_I / (Ki + 1e-12)   // back-compute the I state

    // PID torque, then explicit clip
    tau = Kp * err + tau_I + Kd * d_filtered
    tau = clip(tau, -MAX_XY_TORQUE, +MAX_XY_TORQUE)

`MAX_XY_TORQUE` is read once from the F450 motor and geometry model in
the PyBullet env. Do not approximate it on the device, use the same value.

## Safety fallback rules

Run these checks every tick before writing the new gains to the PID:

| Condition | Threshold | Action |
|---|---|---|
| Attitude limit  | `|roll| > 60deg` or `|pitch| > 60deg`         | Freeze gains at Kp_init / Ki_init / Kd_init |
| Altitude limit  | `z < 0.15 m` or `z > 2.5 m`                   | Freeze gains |
| Gain delta rate | Any `|dGain| > 2 * DELTA_SCALE[i]` per tick   | Clamp the delta, log a fault |
| Obs range fault | `|rates| > 20 rad/s` or `norm(obs[6:12]) > 1.05` | Freeze gains, log a fault |
| Sensor watchdog | No new observation within twice the control period (> 42 ms) | Freeze gains at the last good value |

## STM32Cube.AI validation steps

1. Run `stedgeai validate` against the ONNX file with the test vectors in
   `export/actor_joint_12d_c860k_test_vectors.json` (the CSV form is
   equivalent).
2. FP32 tolerance: max `|d_action|` below 1e-5 (pure floating-point
   rounding).
3. INT8 PTQ tolerance: max `|d_action|` below 0.05 per element (about 1
   to 3 percent of the [-1, 1] range). After gain scaling that means
   dKp error below 1.7e-3 (0.1 percent of the 1.72 Kp range),
   dKi error below 1.7e-4, and dKd error below 8.5e-6.
4. If any test vector goes over these tolerances, stop. Do not push to
   hardware.
5. For PTQ calibration use at least 500 representative obs vectors from
   PyBullet rollouts or Gazebo SITL, covering the grid conditions.

## stedgeai quick reference

    # Generate the test vectors once.
    python export/gen_test_vectors_12d_c860k.py

    # Validate the ONNX in FP32 against the test vectors.
    stedgeai validate \
        --model export/actor_joint_12d_c860k.onnx \
        --target stm32 \
        --valinput  export/testvec_obs.npy \
        --valoutput export/testvec_action.npy

    # Generate the C code.
    stedgeai generate \
        --model export/actor_joint_12d_c860k.onnx \
        --target stm32 \
        --output export/stm32_c_code_12d_c860k/

## What is NOT inside the ONNX graph

These pieces have to live in the firmware. They are the correctness
surface of the deployment, test each one against the Python env before
closing the loop on hardware.

1. The observation builder (the 12 fields above, including the body-rate
   transform).
2. The `DELTA_SCALE` multiply.
3. The running-gain integrator (Kp, Ki, Kd state).
4. The hard clip to the gain bounds.
5. The rate PID (derivative filter, anti-windup, torque clip).
6. The safety fallback rules.
