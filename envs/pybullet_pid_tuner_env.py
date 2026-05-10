"""
PyBullet PID Tuner Environment
================================
RL environment for adaptive roll-rate PID gain tuning using CF2X (Crazyflie 2.0 X)
in PyBullet via gym-pybullet-drones.

Phase 1 scope: roll-rate inner loop only.
  - Action  (3-dim): [ΔKp_roll, ΔKi_roll, ΔKd_roll]  (normalized, in [-1, 1])
  - Obs    (10-dim): [roll, pitch, roll_rate, pitch_rate,
                      roll_rate_err, pitch_rate_err,
                      Kp_n, Ki_n, Kd_n, step_progress]
  - Reward: same 4-term formula as PX4 environment

Controller chain (matches PX4 MC_ROLLRATE structure):
  Altitude hold  → total thrust
  Outer attitude → roll_rate_setpoint = Kp_att * (0 - roll)
  Inner rate PID → roll torque  (RL-tunable gains)
  Fixed pitch/yaw PIDs  (not tuned by RL)

Motor layout (CF2X, from URDF inertial origins):
  Motor 0: ( 0.028,  -0.028, 0)  front-right  CCW
  Motor 1: (-0.028,  -0.028, 0)  back-left    CCW  (wait — see _physics z-torque sign)
  Motor 2: (-0.028,   0.028, 0)  back-right   CW
  Motor 3: ( 0.028,   0.028, 0)  front-left   CW

Physical z_torque in BaseAviary._physics:
  tau_yaw = -km*rpm0² + km*rpm1² - km*rpm2² + km*rpm3²
  → motors 0,2: CCW;  motors 1,3: CW

Allocation matrix derivation (arm d = 0.028 m, body frame X=forward Y=right):
  τ_roll  = kf*d * (-F0 - F1 + F2 + F3)   [y-axis moment]
  τ_pitch = kf*d * ( F0 - F1 - F2 + F3)   [x-axis moment — wait, sign depends on layout]
  The inverse is computed numerically via np.linalg.inv(A).

The deployment path: PyTorch actor → ONNX → STM32Cube.AI → PX4 module (STM32H7).
"""

import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class PyBulletPIDTunerEnv(BaseAviary):
    """Gymnasium env: RL tunes CF2X roll-rate PID gains in PyBullet."""

    # ── Gain bounds (physical torque units: N·m / ...) ──────────────────────
    # CF2X Ixx = 1.4e-5 kg·m², ctrl_freq=48 Hz → dt=0.0208 s
    # Stability requires τ = Ixx/Kp >> dt, i.e. Kp << 6.7e-4 N·m/(rad/s).
    # Defaults give τ_inner ≈ 70 ms (safe); bounds allow RL to explore harder.
    KP_BOUNDS = (0.0,   2.0e-3)
    KI_BOUNDS = (0.0,   2.0e-4)
    KD_BOUNDS = (0.0,   1.0e-5)

    KP_DEFAULT = 2.0e-4   # τ_inner = 70 ms >> dt = 21 ms
    KI_DEFAULT = 1.0e-5
    KD_DEFAULT = 2.0e-6

    # max delta per step (≈2% of each bound's full range)
    DELTA_SCALE = np.array([4.0e-5, 4.0e-6, 2.0e-7])

    # ── Fixed outer attitude-P gain (rad/s per rad) ────────────────────────
    KP_ATT = 3.0

    # ── Fixed pitch/yaw rate PID ───────────────────────────────────────────
    KP_PITCH_RATE = 2.0e-4
    KI_PITCH_RATE = 1.0e-5
    KD_PITCH_RATE = 2.0e-6

    KP_YAW_RATE   = 5.0e-5   # proportional only

    # ── Altitude hold: total thrust = hover_thrust + PD(z_err) ──────────
    KP_ALT = 0.06   # N per m
    KD_ALT = 0.04   # N·s per m

    # ── Crash / done thresholds ────────────────────────────────────────────
    MAX_ROLL_RAD  = np.deg2rad(60)
    MAX_PITCH_RAD = np.deg2rad(60)
    MIN_ALT       = 0.15   # m
    MAX_ALT       = 2.5    # m

    def __init__(self,
                 max_steps: int = 500,
                 target_alt: float = 1.0,
                 init_noise: float = 0.05,
                 reward_w1: float = 1.0,
                 reward_w2: float = 2.0,
                 reward_w3: float = 0.1,
                 reward_w4: float = 0.5,
                 crash_penalty: float = 50.0,
                 stability_bonus: float = 200.0,
                 gui: bool = False):

        self.max_steps       = max_steps
        self.target_alt      = target_alt
        self.init_noise      = init_noise
        self.reward_w1       = reward_w1
        self.reward_w2       = reward_w2
        self.reward_w3       = reward_w3
        self.reward_w4       = reward_w4
        self.crash_penalty   = crash_penalty
        self.stability_bonus = stability_bonus

        # instance state — must exist before super().__init__ calls _actionSpace etc.
        self._step_count     = 0
        self._roll_Kp        = self.KP_DEFAULT
        self._roll_Ki        = self.KI_DEFAULT
        self._roll_Kd        = self.KD_DEFAULT
        self._roll_integral  = 0.0
        self._pitch_integral = 0.0
        self._prev_roll_rate = 0.0
        self._prev_pitch_rate = 0.0
        self._prev_reward    = 0.0

        super().__init__(
            drone_model    = DroneModel.CF2X,
            num_drones     = 1,
            physics        = Physics.PYB,
            pyb_freq       = 240,
            ctrl_freq      = 48,
            gui            = gui,
            user_debug_gui = False,
        )

        # Build inverse allocation matrix once (using loaded URDF params).
        self._build_inv_alloc()

    # ── BaseAviary abstract method implementations ─────────────────────────

    def _actionSpace(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _observationSpace(self):
        obs_low  = np.array([-np.pi, -np.pi, -20., -20., -20., -20., 0., 0., 0., 0.], dtype=np.float32)
        obs_high = np.array([ np.pi,  np.pi,  20.,  20.,  20.,  20., 1., 1., 1., 1.], dtype=np.float32)
        return spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def _computeObs(self):
        roll, pitch, _  = self.rpy[0]
        ang_w           = self.ang_v[0]
        body_rates      = self._world_to_body_rates(ang_w)
        roll_rate, pitch_rate, _ = body_rates

        roll_rate_err  = -roll_rate    # setpoint = 0
        pitch_rate_err = -pitch_rate

        step_prog = float(self._step_count) / float(self.max_steps)
        kp_n, ki_n, kd_n = self._normalize_gains()

        return np.array([roll, pitch,
                         roll_rate, pitch_rate,
                         roll_rate_err, pitch_rate_err,
                         kp_n, ki_n, kd_n,
                         step_prog], dtype=np.float32)

    def _preprocessAction(self, action):
        """Convert RL action → per-motor RPMs.

        The action updates the roll-rate PID gains, then the full controller
        pipeline runs: altitude PD + attitude P + rate PID → torques → RPMs.
        """
        action = np.clip(action, -1.0, 1.0)
        delta  = action * self.DELTA_SCALE

        # Store previous gains for reward gain-change penalty
        kp_prev, ki_prev, kd_prev = self._roll_Kp, self._roll_Ki, self._roll_Kd
        self._prev_gains = np.array([kp_prev, ki_prev, kd_prev])

        self._roll_Kp = float(np.clip(self._roll_Kp + delta[0], *self.KP_BOUNDS))
        self._roll_Ki = float(np.clip(self._roll_Ki + delta[1], *self.KI_BOUNDS))
        self._roll_Kd = float(np.clip(self._roll_Kd + delta[2], *self.KD_BOUNDS))

        # Current state
        roll, pitch, _   = self.rpy[0]
        pos_z            = self.pos[0, 2]
        vel_z            = self.vel[0, 2]
        body_rates       = self._world_to_body_rates(self.ang_v[0])
        roll_rate, pitch_rate, yaw_rate = body_rates

        dt = self.CTRL_TIMESTEP  # 1/48 s

        # ── Altitude hold ───────────────────────────────────────────────────
        hover_thrust = self.GRAVITY   # m*g in Newtons (BaseAviary stores m*g as GRAVITY)
        z_err        = self.target_alt - pos_z
        thrust       = hover_thrust + self.KP_ALT * z_err + self.KD_ALT * (-vel_z)
        thrust       = float(np.clip(thrust, 0.0, 2.0 * hover_thrust))

        # ── Outer attitude loop (fixed) → rate setpoints ────────────────────
        roll_rate_sp  = self.KP_ATT * (0.0 - roll)
        pitch_rate_sp = self.KP_ATT * (0.0 - pitch)

        # ── Roll rate PID (RL-tunable) ──────────────────────────────────────
        roll_err             = roll_rate_sp - roll_rate
        self._roll_integral += roll_err * dt
        self._roll_integral  = float(np.clip(self._roll_integral, -0.5, 0.5))
        d_roll               = -(roll_rate - self._prev_roll_rate) / dt   # derivative on measurement
        tau_roll = (self._roll_Kp * roll_err
                    + self._roll_Ki * self._roll_integral
                    + self._roll_Kd * d_roll)

        # ── Pitch rate PID (fixed gains) ────────────────────────────────────
        pitch_err             = pitch_rate_sp - pitch_rate
        self._pitch_integral += pitch_err * dt
        self._pitch_integral  = float(np.clip(self._pitch_integral, -0.5, 0.5))
        d_pitch               = -(pitch_rate - self._prev_pitch_rate) / dt
        tau_pitch = (self.KP_PITCH_RATE * pitch_err
                     + self.KI_PITCH_RATE * self._pitch_integral
                     + self.KD_PITCH_RATE * d_pitch)

        # ── Yaw rate P (fixed, zero setpoint) ──────────────────────────────
        tau_yaw = self.KP_YAW_RATE * (0.0 - yaw_rate)

        # ── Save state for derivative / reward ─────────────────────────────
        self._prev_roll_rate  = roll_rate
        self._prev_pitch_rate = pitch_rate
        self._step_count     += 1

        # ── Torques → RPMs via inverse allocation ───────────────────────────
        rpms = self._torques_to_rpms(thrust, tau_roll, tau_pitch, tau_yaw)
        return rpms.reshape(1, 4)   # BaseAviary expects (NUM_DRONES, 4)

    def _computeReward(self):
        roll, pitch, _ = self.rpy[0]
        body_rates      = self._world_to_body_rates(self.ang_v[0])
        roll_rate, _, _ = body_rates

        w1, w2, w3, w4 = self.reward_w1, self.reward_w2, self.reward_w3, self.reward_w4

        att_err     = roll**2 + pitch**2
        rate_err    = roll_rate**2
        kp_prev, ki_prev, kd_prev = self._prev_gains
        gain_change = ((self._roll_Kp - kp_prev)**2
                       + (self._roll_Ki - ki_prev)**2
                       + (self._roll_Kd - kd_prev)**2)
        dt          = self.CTRL_TIMESTEP
        d_roll_rate = (roll_rate - self._prev_roll_rate) / dt
        oscillation = d_roll_rate**2

        return float(-(w1*att_err + w2*rate_err + w3*gain_change + w4*oscillation))

    def _computeTerminated(self):
        roll, pitch, _ = self.rpy[0]
        z              = self.pos[0, 2]
        crashed = (abs(roll)  > self.MAX_ROLL_RAD
                   or abs(pitch) > self.MAX_PITCH_RAD
                   or z < self.MIN_ALT
                   or z > self.MAX_ALT)
        self._crashed = bool(crashed)
        if crashed:
            self._prev_reward = -self.crash_penalty
        return crashed

    def _computeTruncated(self):
        done = self._step_count >= self.max_steps
        if done and not self._crashed:
            self._prev_reward = self.stability_bonus
        return done

    def _computeInfo(self):
        roll, pitch, _ = self.rpy[0]
        body_rates      = self._world_to_body_rates(self.ang_v[0])
        roll_rate, _, _ = body_rates
        return {
            "step"          : self._step_count,
            "roll_deg"      : float(np.rad2deg(roll)),
            "pitch_deg"     : float(np.rad2deg(pitch)),
            "roll_rate"     : float(roll_rate),
            "alt_m"         : float(self.pos[0, 2]),
            "crashed"       : self._crashed,
            "Kp"            : self._roll_Kp,
            "Ki"            : self._roll_Ki,
            "Kd"            : self._roll_Kd,
        }

    # ── Reset override to reinitialize PID state ──────────────────────────

    def reset(self, seed=None, options=None):
        self._step_count      = 0
        self._roll_Kp         = self.KP_DEFAULT
        self._roll_Ki         = self.KI_DEFAULT
        self._roll_Kd         = self.KD_DEFAULT
        self._roll_integral   = 0.0
        self._pitch_integral  = 0.0
        self._prev_roll_rate  = 0.0
        self._prev_pitch_rate = 0.0
        self._crashed         = False
        self._prev_gains      = np.array([self.KP_DEFAULT, self.KI_DEFAULT, self.KD_DEFAULT])

        if seed is not None:
            np.random.seed(seed)

        obs, info = super().reset(seed=seed, options=options)

        # Always start at target altitude; apply orientation + rate noise when requested.
        noise_rpy   = np.random.uniform(-self.init_noise, self.init_noise, 3) if self.init_noise > 0.0 else np.zeros(3)
        noise_rates = np.random.uniform(-self.init_noise * 5, self.init_noise * 5, 3) if self.init_noise > 0.0 else np.zeros(3)

        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            [0.0, 0.0, self.target_alt],
            p.getQuaternionFromEuler(noise_rpy.tolist()),
            physicsClientId=self.CLIENT
        )
        p.resetBaseVelocity(
            self.DRONE_IDS[0],
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=noise_rates.tolist(),
            physicsClientId=self.CLIENT
        )
        self._updateAndStoreKinematicInformation()

        # Seed derivative terms with actual post-reset rates to avoid a
        # spurious derivative spike on the first control step.
        body_rates = self._world_to_body_rates(self.ang_v[0])
        self._prev_roll_rate  = float(body_rates[0])
        self._prev_pitch_rate = float(body_rates[1])

        return self._computeObs(), self._computeInfo()

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_inv_alloc(self):
        """Pre-compute the inverse allocation matrix from physical params."""
        d  = 0.028          # motor arm (meters, from URDF inertial xyz)
        kf = self.KF        # thrust coeff (loaded by BaseAviary from URDF)
        km = self.KM        # torque coeff

        # Allocation: [T, τ_roll, τ_pitch, τ_yaw] = A @ [F0, F1, F2, F3]
        # Motor layout (x_i, y_i):
        #   M0: ( d, -d)  front-right  CCW
        #   M1: (-d, -d)  back-left    CCW  (BaseAviary z_torque: -km*rpm0² + km*rpm1² ...)
        # Wait: BaseAviary _physics: z_torque = -torques[0]+torques[1]-torques[2]+torques[3]
        # CCW motors: 0, 2  CW motors: 1, 3
        #   M0: ( 0.028, -0.028)  CCW  front-right
        #   M1: (-0.028, -0.028)  CW   back-left  (WAIT — check _physics sign: +km*rpm1²)
        #   M2: (-0.028,  0.028)  CCW  back-right (-km*rpm2²)
        #   M3: ( 0.028,  0.028)  CW   front-left (+km*rpm3²)
        #
        # τ_roll  = Σ y_i * kf * rpm_i²   (y positive = right wing down = positive roll)
        # τ_pitch = Σ (-x_i) * kf * rpm_i² (x pos forward; -x because nose-down = positive pitch)
        # τ_yaw   = sign_i * km * rpm_i²   (sign: CW=+1, CCW=-1)
        #
        # Motor positions: M0=(+d,-d), M1=(-d,-d), M2=(-d,+d), M3=(+d,+d)
        # y: M0=-d, M1=-d, M2=+d, M3=+d  → roll col: [-d,-d,+d,+d] (× kf)
        # x: M0=+d, M1=-d, M2=-d, M3=+d  → -x: [-d,+d,+d,-d]      (× kf)
        # yaw: M0=CCW(-1), M1=CW(+1), M2=CCW(-1), M3=CW(+1)         (× km)

        A = np.array([
            [1,          1,          1,          1         ],  # T = F0+F1+F2+F3
            [-d,        -d,         +d,         +d        ],  # tau_roll  = sum(y_i * F_i)
            [-d,        +d,         +d,         -d        ],  # tau_pitch = sum(-x_i * F_i)
            [-km/kf,    +km/kf,    -km/kf,     +km/kf    ],  # tau_yaw   (CCW=-1, CW=+1)
        ])

        self._inv_A = np.linalg.inv(A)

    def _torques_to_rpms(self, thrust, tau_roll, tau_pitch, tau_yaw):
        """Convert [T, τ_roll, τ_pitch, τ_yaw] → per-motor RPMs."""
        cmd = np.array([thrust, tau_roll, tau_pitch, tau_yaw])
        F   = self._inv_A @ cmd                        # per-motor forces (N)
        F   = np.maximum(F, 0.0)
        rpms = np.sqrt(F / self.KF)
        return np.clip(rpms, 0.0, self.MAX_RPM)

    def _world_to_body_rates(self, world_ang_v):
        """Rotate world-frame angular velocity to body frame."""
        R = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape(3, 3)
        return R.T @ world_ang_v

    def _normalize_gains(self):
        kp_n = (self._roll_Kp - self.KP_BOUNDS[0]) / (self.KP_BOUNDS[1] - self.KP_BOUNDS[0])
        ki_n = (self._roll_Ki - self.KI_BOUNDS[0]) / (self.KI_BOUNDS[1] - self.KI_BOUNDS[0])
        kd_n = (self._roll_Kd - self.KD_BOUNDS[0]) / (self.KD_BOUNDS[1] - self.KD_BOUNDS[0])
        return float(kp_n), float(ki_n), float(kd_n)
