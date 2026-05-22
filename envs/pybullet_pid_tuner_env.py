"""
PyBullet environment for the adaptive PID gain tuning task on an F450
quadrotor. The RL agent does not produce motor commands. It only moves
the gains of the inner rate PID, while the PID itself stays the same.

The environment is built on top of gym-pybullet-drones (BaseAviary).

Modes (tune_axes parameter):

  ["roll"]           action [dKp, dKi, dKd] applied to roll only.
                     Observation contains the roll gains. (10-D obs, 3-D act)

  ["pitch"]          same idea but for pitch. (10-D obs, 3-D act)

  ["roll","pitch"]   the shared mode used in this project. One single
                     [dKp, dKi, dKd] action is applied to both axes at
                     the same time. The observation contains the gains
                     of both axes (they are always equal because the
                     action is shared). (12-D obs, 3-D act)

The disturbance_axis option selects where the external torque is applied
during one episode: "roll", "pitch", "both", or "random" (drawn each reset).

Controller structure (mirrors the PX4 MC rate loop):

  altitude hold       -> total thrust
  outer attitude P    -> rate setpoints (roll_rate_sp, pitch_rate_sp)
  inner rate PID      -> roll torque   (tuned by RL when "roll" is in tune_axes)
  inner rate PID      -> pitch torque  (tuned by RL when "pitch" is in tune_axes)
  yaw rate P          -> yaw torque    (fixed, not tuned)

Both rate PIDs are the same quality:
  - derivative measured on rate, low-pass filtered (1-pole IIR, fc about
    30 Hz at the 48 Hz control rate)
  - torque-space anti-windup (the integral term is clipped to 30% of the
    max XY torque, then back-computed)
  - explicit clipping on the final torque

F450 X-configuration motor layout (arm = 0.225 m, d = arm / sqrt(2) = 0.159 m):

  motor 0: ( 0.159, -0.159, 0)  front-right  CCW
  motor 1: (-0.159, -0.159, 0)  back-left    CW
  motor 2: (-0.159,  0.159, 0)  back-right   CCW
  motor 3: ( 0.159,  0.159, 0)  front-left   CW
"""

import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

_ROLL  = 0
_PITCH = 1


class PyBulletPIDTunerEnv(BaseAviary):
    """Gym environment where RL tunes the rate-PID gains of an F450 in PyBullet."""

    # Gain bounds for the F450. They come from scaling the CF2X bounds by
    # the inertia ratio (Ixx of the F450 is about 857 times the CF2X).
    KP_BOUNDS = (0.0,  1.72)
    KI_BOUNDS = (0.0,  0.172)
    KD_BOUNDS = (0.0,  8.6e-3)

    # Defaults used as the starting point of every episode (except hold).
    KP_DEFAULT = 0.171   # inner time constant Ixx/Kp is around 70 ms
    KI_DEFAULT = 8.6e-3
    KD_DEFAULT = 1.71e-3

    # A full-scale action of 1.0 moves a gain by about 2% of its range
    # in one control step. Same fraction for the three gains.
    DELTA_SCALE = np.array([3.4e-2, 3.4e-3, 1.7e-4])

    # "Hold" episode. Near-level start with a small constant torque kept
    # for the whole episode. To reject a constant torque without standing
    # tilt the integral term is needed, so a policy that drops Ki keeps
    # a residual attitude error and pays the w1 penalty all the way. The
    # idea is to teach "do not touch the gains when the drone is stable"
    # without changing any reward weight.
    HOLD_INIT_NOISE      = 0.03   # rad, small initial tilt
    HOLD_DIST_MAGNITUDE  = 0.03   # N.m, small constant torque on both axes

    # "Sustained" episode. Starts at the default gains and applies a
    # moderate constant torque during the whole episode. Same reason as
    # above: dropping Ki makes the policy pay every step.
    SUSTAINED_INIT_NOISE     = 0.03
    SUSTAINED_DIST_MAG_RANGE = (0.10, 0.15)

    # Outer attitude P gain (fixed, not tuned by RL).
    KP_ATT = 3.0

    # Yaw is just a fixed proportional gain. Yaw is not in scope.
    KP_YAW_RATE = 4.38e-2

    # Altitude hold (PD on z).
    KP_ALT = 2.67
    KD_ALT = 1.78

    # Derivative filter alpha. 1-pole IIR, fc about 30 Hz at 48 Hz ctrl.
    _D_FILTER_ALPHA = 0.797

    # Crash thresholds.
    MAX_ROLL_RAD  = np.deg2rad(60)
    MAX_PITCH_RAD = np.deg2rad(60)
    MIN_ALT       = 0.15
    MAX_ALT       = 2.5

    def __init__(self,
                 tune_axes: list = None,
                 disturbance_axis: str = "roll",
                 max_steps: int = 500,
                 target_alt: float = 1.0,
                 init_noise: float = 0.05,
                 reward_w1: float = 1.0,
                 reward_w2: float = 2.0,
                 reward_w3: float = 0.1,
                 reward_w4: float = 0.001,
                 crash_penalty: float = 50.0,
                 stability_bonus: float = 200.0,
                 disturbance_step: int = None,
                 disturbance_magnitude: float = 0.0,
                 disturbance_duration: int = 5,
                 randomize_disturbance: bool = False,
                 init_noise_range: tuple = (0.03, 0.15),
                 disturbance_step_range: tuple = (80, 250),
                 disturbance_magnitude_range: tuple = (0.0, 0.25),
                 disturbance_duration_range: tuple = (3, 10),
                 randomize_initial_gains: bool = False,
                 hold_episode_prob: float = 0.0,
                 init_gain_frac_range: tuple = (0.0, 1.0),
                 hold_gain_mult_range: tuple = (0.5, 3.0),
                 sustained_episode_prob: float = 0.0,
                 sustained_dist_mag_range: tuple = SUSTAINED_DIST_MAG_RANGE,
                 gui: bool = False):

        if tune_axes is None:
            tune_axes = ["roll"]

        for ax in tune_axes:
            if ax not in ("roll", "pitch"):
                raise ValueError(f"Unknown axis '{ax}'. Use 'roll' and/or 'pitch'.")
        if len(tune_axes) > 2 or len(set(tune_axes)) != len(tune_axes):
            raise ValueError("tune_axes must be ['roll'], ['pitch'], or "
                             f"['roll','pitch']. Got {tune_axes}.")
        if disturbance_axis not in ("roll", "pitch", "both", "random"):
            raise ValueError(f"Unknown disturbance_axis '{disturbance_axis}'. "
                             "Use 'roll', 'pitch', 'both', or 'random'.")

        self._joint     = len(tune_axes) > 1
        # In joint mode we always keep roll first, then pitch.
        self._tune_axes = ["roll", "pitch"] if self._joint else list(tune_axes)
        self._tuned_idx = None if self._joint else (_ROLL if tune_axes[0] == "roll" else _PITCH)
        self._dist_axis = disturbance_axis
        # The actual axis used for this episode. Different from _dist_axis
        # only when "random" is picked (then it is resolved at reset).
        self._active_dist_axis = "both" if disturbance_axis == "random" else disturbance_axis
        # The action is always 3-D. In joint mode the same vector is
        # applied to both roll and pitch.
        self._n_act = 3

        self.max_steps       = max_steps
        self.target_alt      = target_alt
        self.init_noise      = init_noise
        self.reward_w1       = reward_w1
        self.reward_w2       = reward_w2
        self.reward_w3       = reward_w3
        self.reward_w4       = reward_w4
        self.crash_penalty   = crash_penalty
        self.stability_bonus = stability_bonus
        self._dist_step      = disturbance_step
        self._dist_magnitude = disturbance_magnitude
        self._dist_duration  = disturbance_duration
        self._randomize_disturbance        = randomize_disturbance
        self._init_noise_range             = init_noise_range
        self._disturbance_step_range       = disturbance_step_range
        self._disturbance_magnitude_range  = disturbance_magnitude_range
        self._disturbance_duration_range   = disturbance_duration_range

        # Mix of recovery and hold episodes with randomized starting gains.
        # When the flags below are off the env reproduces the simple
        # "fixed defaults + transient kick" behaviour.
        self._randomize_initial_gains      = randomize_initial_gains
        self._hold_episode_prob            = float(hold_episode_prob)
        self._init_gain_frac_range         = init_gain_frac_range
        self._hold_gain_mult_range         = hold_gain_mult_range
        self._is_hold_episode              = False
        self._sustained_episode_prob       = float(sustained_episode_prob)
        self._sustained_dist_mag_range     = sustained_dist_mag_range
        self._is_sustained_episode         = False

        # Per-axis PID state. Indexed by _ROLL / _PITCH. Must be set
        # before super().__init__ because it calls _actionSpace and
        # _observationSpace.
        self._gains      = np.array([
            [self.KP_DEFAULT, self.KI_DEFAULT, self.KD_DEFAULT],  # roll
            [self.KP_DEFAULT, self.KI_DEFAULT, self.KD_DEFAULT],  # pitch
        ], dtype=np.float64)
        self._integral   = np.zeros(2, dtype=np.float64)
        self._prev_rate  = np.zeros(2, dtype=np.float64)
        self._d_filtered = np.zeros(2, dtype=np.float64)

        self._step_count  = 0
        self._crashed     = False
        self._prev_action = np.zeros(self._n_act, dtype=np.float32)

        super().__init__(
            drone_model    = DroneModel.F450,
            num_drones     = 1,
            physics        = Physics.PYB,
            pyb_freq       = 240,
            ctrl_freq      = 48,
            gui            = gui,
            user_debug_gui = False,
        )

        self._build_inv_alloc()


    def _actionSpace(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._n_act,), dtype=np.float32)

    def _observationSpace(self):
        if self._joint:
            obs_low  = np.array([-np.pi, -np.pi, -20., -20., -20., -20.,
                                 0., 0., 0., 0., 0., 0.], dtype=np.float32)
            obs_high = np.array([ np.pi,  np.pi,  20.,  20.,  20.,  20.,
                                 1., 1., 1., 1., 1., 1.], dtype=np.float32)
        else:
            obs_low  = np.array([-np.pi, -np.pi, -20., -20., -20., -20., 0., 0., 0.], dtype=np.float32)
            obs_high = np.array([ np.pi,  np.pi,  20.,  20.,  20.,  20., 1., 1., 1.], dtype=np.float32)
        return spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def _computeObs(self):
        roll, pitch, _  = self.rpy[0]
        ang_w           = self.ang_v[0]
        body_rates      = self._world_to_body_rates(ang_w)
        roll_rate, pitch_rate, _ = body_rates

        roll_rate_sp   = self.KP_ATT * (0.0 - roll)
        pitch_rate_sp  = self.KP_ATT * (0.0 - pitch)
        roll_rate_err  = roll_rate_sp - roll_rate
        pitch_rate_err = pitch_rate_sp - pitch_rate

        if self._joint:
            kp_r = self._gains[_ROLL][0]  / self.KP_BOUNDS[1]
            ki_r = self._gains[_ROLL][1]  / self.KI_BOUNDS[1]
            kd_r = self._gains[_ROLL][2]  / self.KD_BOUNDS[1]
            kp_p = self._gains[_PITCH][0] / self.KP_BOUNDS[1]
            ki_p = self._gains[_PITCH][1] / self.KI_BOUNDS[1]
            kd_p = self._gains[_PITCH][2] / self.KD_BOUNDS[1]
            return np.array([roll, pitch,
                             roll_rate, pitch_rate,
                             roll_rate_err, pitch_rate_err,
                             kp_r, ki_r, kd_r,
                             kp_p, ki_p, kd_p], dtype=np.float32)

        # Single-axis: the gain channels are the ones of the tuned axis.
        kp_n = self._gains[self._tuned_idx][0] / self.KP_BOUNDS[1]
        ki_n = self._gains[self._tuned_idx][1] / self.KI_BOUNDS[1]
        kd_n = self._gains[self._tuned_idx][2] / self.KD_BOUNDS[1]
        return np.array([roll, pitch,
                         roll_rate, pitch_rate,
                         roll_rate_err, pitch_rate_err,
                         kp_n, ki_n, kd_n], dtype=np.float32)

    def _preprocessAction(self, action):
        """Take the RL action and produce the four motor RPMs.

        The function updates the tuned gains, runs the two rate PIDs,
        adds altitude PD and the outer attitude P, then turns the
        torques into RPMs through the inverse allocation.
        """
        action = np.clip(action, -1.0, 1.0)
        self._prev_action = action.copy()

        gain_lo = np.array([self.KP_BOUNDS[0], self.KI_BOUNDS[0], self.KD_BOUNDS[0]])
        gain_hi = np.array([self.KP_BOUNDS[1], self.KI_BOUNDS[1], self.KD_BOUNDS[1]])
        if self._joint:
            delta = action * self.DELTA_SCALE   # the same delta goes to both axes
            self._gains[_ROLL]  = np.clip(self._gains[_ROLL]  + delta, gain_lo, gain_hi)
            self._gains[_PITCH] = np.clip(self._gains[_PITCH] + delta, gain_lo, gain_hi)
        else:
            self._gains[self._tuned_idx] = np.clip(
                self._gains[self._tuned_idx] + action * self.DELTA_SCALE,
                gain_lo, gain_hi,
            )

        # Current state of the drone.
        roll, pitch, _      = self.rpy[0]
        pos_z               = self.pos[0, 2]
        vel_z               = self.vel[0, 2]
        body_rates          = self._world_to_body_rates(self.ang_v[0])
        roll_rate, pitch_rate, yaw_rate = body_rates

        dt = self.CTRL_TIMESTEP

        # Altitude PD.
        hover_thrust = self.GRAVITY
        z_err        = self.target_alt - pos_z
        thrust       = hover_thrust + self.KP_ALT * z_err + self.KD_ALT * (-vel_z)
        thrust       = float(np.clip(thrust, 0.0, 2.0 * hover_thrust))

        # Outer attitude loop, produces the rate setpoints.
        roll_rate_sp  = self.KP_ATT * (0.0 - roll)
        pitch_rate_sp = self.KP_ATT * (0.0 - pitch)

        # Inner rate PIDs (the part that RL tunes).
        tau_roll  = self._run_rate_pid(_ROLL,  roll_rate_sp,  roll_rate,  *self._gains[_ROLL])
        tau_pitch = self._run_rate_pid(_PITCH, pitch_rate_sp, pitch_rate, *self._gains[_PITCH])

        # Yaw stays simple, just a P on the yaw rate setpoint of 0.
        tau_yaw = self.KP_YAW_RATE * (0.0 - yaw_rate)

        # Save the rates we just used so the reward sees the value from
        # before physics is stepped.
        self._prev_rate[_ROLL]  = roll_rate
        self._prev_rate[_PITCH] = pitch_rate
        self._step_count += 1

        # Apply the external torque if we are inside the disturbance window.
        if (self._dist_step is not None
                and self._dist_magnitude != 0.0
                and self._dist_step <= self._step_count
                        < self._dist_step + self._dist_duration):
            mag = self._dist_magnitude
            if self._active_dist_axis == "roll":
                torque_vec = [mag, 0.0, 0.0]
            elif self._active_dist_axis == "pitch":
                torque_vec = [0.0, mag, 0.0]
            else:   # "both"
                torque_vec = [mag, mag, 0.0]
            p.applyExternalTorque(
                self.DRONE_IDS[0], -1, torque_vec,
                flags=p.LINK_FRAME, physicsClientId=self.CLIENT,
            )

        rpms = self._torques_to_rpms(thrust, tau_roll, tau_pitch, tau_yaw)
        return rpms.reshape(1, 4)

    def _computeReward(self):
        roll, pitch, _ = self.rpy[0]
        body_rates      = self._world_to_body_rates(self.ang_v[0])
        roll_rate, pitch_rate, _ = body_rates
        z               = self.pos[0, 2]

        crashed = (abs(roll)  > self.MAX_ROLL_RAD
                   or abs(pitch) > self.MAX_PITCH_RAD
                   or z < self.MIN_ALT
                   or z > self.MAX_ALT)
        if crashed:
            return float(-self.crash_penalty)

        w1, w2, w3, w4 = self.reward_w1, self.reward_w2, self.reward_w3, self.reward_w4
        dt = self.CTRL_TIMESTEP

        att_err = roll**2 + pitch**2

        # The action is 3-D in both joint and single-axis mode, so this
        # term is computed the same way for both.
        gain_change = float(np.sum(self._prev_action ** 2))

        if self._joint:
            roll_rate_sp  = self.KP_ATT * (0.0 - roll)
            pitch_rate_sp = self.KP_ATT * (0.0 - pitch)
            rate_err    = (roll_rate_sp - roll_rate) ** 2 + (pitch_rate_sp - pitch_rate) ** 2
            oscillation = (((roll_rate  - self._prev_rate[_ROLL])  / dt) ** 2
                           + ((pitch_rate - self._prev_rate[_PITCH]) / dt) ** 2)
        else:
            if self._tuned_idx == _ROLL:
                rate      = roll_rate
                rate_sp   = self.KP_ATT * (0.0 - roll)
                prev_rate = self._prev_rate[_ROLL]
            else:
                rate      = pitch_rate
                rate_sp   = self.KP_ATT * (0.0 - pitch)
                prev_rate = self._prev_rate[_PITCH]
            rate_err    = (rate_sp - rate) ** 2
            oscillation = ((rate - prev_rate) / dt) ** 2

        step_reward = -(w1*att_err + w2*rate_err + w3*gain_change + w4*oscillation)

        if self._step_count >= self.max_steps:
            step_reward += self.stability_bonus

        return float(step_reward)

    def _computeTerminated(self):
        roll, pitch, _ = self.rpy[0]
        z              = self.pos[0, 2]
        crashed = (abs(roll)  > self.MAX_ROLL_RAD
                   or abs(pitch) > self.MAX_PITCH_RAD
                   or z < self.MIN_ALT
                   or z > self.MAX_ALT)
        self._crashed = bool(crashed)
        return crashed

    def _computeTruncated(self):
        return self._step_count >= self.max_steps

    def _computeInfo(self):
        roll, pitch, _ = self.rpy[0]
        body_rates      = self._world_to_body_rates(self.ang_v[0])
        roll_rate, pitch_rate, _ = body_rates
        dist_active = (
            self._dist_step is not None
            and self._dist_magnitude != 0.0
            and self._dist_step < self._step_count <= self._dist_step + self._dist_duration
        )
        info = {
            "step"               : self._step_count,
            "roll_deg"           : float(np.rad2deg(roll)),
            "pitch_deg"          : float(np.rad2deg(pitch)),
            "roll_rate"          : float(roll_rate),
            "pitch_rate"         : float(pitch_rate),
            "alt_m"              : float(self.pos[0, 2]),
            "crashed"            : self._crashed,
            "disturbance_active" : bool(dist_active),
            "disturbance_axis"   : self._active_dist_axis,
            "hold_episode"       : bool(self._is_hold_episode),
            "sustained_episode"  : bool(self._is_sustained_episode),
        }
        if self._joint:
            info["tune_axis"] = "roll+pitch"
            info["Kp_roll"]   = float(self._gains[_ROLL][0])
            info["Ki_roll"]   = float(self._gains[_ROLL][1])
            info["Kd_roll"]   = float(self._gains[_ROLL][2])
            info["Kp_pitch"]  = float(self._gains[_PITCH][0])
            info["Ki_pitch"]  = float(self._gains[_PITCH][1])
            info["Kd_pitch"]  = float(self._gains[_PITCH][2])
            # Old code expects Kp/Ki/Kd without a suffix. We expose the
            # roll values under those names so the existing scripts work.
            info["Kp"], info["Ki"], info["Kd"] = info["Kp_roll"], info["Ki_roll"], info["Kd_roll"]
        else:
            kp, ki, kd = self._gains[self._tuned_idx]
            info["tune_axis"] = self._tune_axes[0]
            info["Kp"], info["Ki"], info["Kd"] = float(kp), float(ki), float(kd)
        return info


    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed)

        # We pick the episode type with a single draw in [0,1):
        #   [0, hold_p)                    -> hold      (gentle, random gains)
        #   [hold_p, hold_p + sus_p)       -> sustained (default gains,
        #                                                moderate torque)
        #   [hold_p + sus_p, 1)            -> recovery  (transient kick)
        _r = float(rng.random())
        self._is_hold_episode = (
            self._hold_episode_prob > 0.0 and _r < self._hold_episode_prob
        )
        self._is_sustained_episode = (
            self._sustained_episode_prob > 0.0
            and not self._is_hold_episode
            and _r < self._hold_episode_prob + self._sustained_episode_prob
        )

        if self._randomize_disturbance:
            self.init_noise      = float(rng.uniform(*self._init_noise_range))
            self._dist_step      = int(rng.integers(*self._disturbance_step_range))
            self._dist_magnitude = float(rng.uniform(*self._disturbance_magnitude_range))
            self._dist_duration  = int(rng.integers(*self._disturbance_duration_range))
            self._active_dist_axis = (
                str(rng.choice(["roll", "pitch", "both"]))
                if self._dist_axis == "random" else self._dist_axis
            )
        else:
            self._active_dist_axis = "both" if self._dist_axis == "random" else self._dist_axis

        # Hold episode: near-level start with a small constant torque on
        # for the full episode. If the policy drops Ki it keeps a residual
        # tilt that w1 penalises every step.
        if self._is_hold_episode:
            self.init_noise        = self.HOLD_INIT_NOISE
            self._dist_magnitude   = self.HOLD_DIST_MAGNITUDE
            self._dist_step        = 1                  # torque on from step 1
            self._dist_duration    = self.max_steps     # until the end
            self._active_dist_axis = "both"             # both axes excited

        # Sustained: default gains, moderate constant torque on the whole
        # episode. Same idea as hold but with a stronger torque.
        if self._is_sustained_episode:
            self.init_noise        = self.SUSTAINED_INIT_NOISE
            self._dist_magnitude   = float(rng.uniform(*self._sustained_dist_mag_range))
            self._dist_step        = 1
            self._dist_duration    = self.max_steps
            self._active_dist_axis = "both"

        self._step_count  = 0

        if self._is_sustained_episode:
            # Always start at the training defaults. The point here is
            # "do not destroy a healthy Ki", not "recover from bad gains".
            self._gains = np.array([
                [self.KP_DEFAULT, self.KI_DEFAULT, self.KD_DEFAULT],
                [self.KP_DEFAULT, self.KI_DEFAULT, self.KD_DEFAULT],
            ], dtype=np.float64)
        elif self._randomize_initial_gains or self._is_hold_episode:
            mode = "hold" if self._is_hold_episode else "recovery"
            g0   = self._sample_initial_gains(rng, mode)
            self._gains = np.array([g0.copy(), g0.copy()], dtype=np.float64)
        else:
            self._gains = np.array([
                [self.KP_DEFAULT, self.KI_DEFAULT, self.KD_DEFAULT],
                [self.KP_DEFAULT, self.KI_DEFAULT, self.KD_DEFAULT],
            ], dtype=np.float64)
        self._integral    = np.zeros(2, dtype=np.float64)
        self._prev_rate   = np.zeros(2, dtype=np.float64)
        self._d_filtered  = np.zeros(2, dtype=np.float64)
        self._crashed     = False
        self._prev_action = np.zeros(self._n_act, dtype=np.float32)

        if seed is not None:
            np.random.seed(seed)

        obs, info = super().reset(seed=seed, options=options)

        noise_rpy   = (np.random.uniform(-self.init_noise, self.init_noise, 3)
                       if self.init_noise > 0.0 else np.zeros(3))
        noise_rates = (np.random.uniform(-self.init_noise * 5, self.init_noise * 5, 3)
                       if self.init_noise > 0.0 else np.zeros(3))

        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            [0.0, 0.0, self.target_alt],
            p.getQuaternionFromEuler(noise_rpy.tolist()),
            physicsClientId=self.CLIENT,
        )
        p.resetBaseVelocity(
            self.DRONE_IDS[0],
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=noise_rates.tolist(),
            physicsClientId=self.CLIENT,
        )
        self._updateAndStoreKinematicInformation()

        # Seed the derivative state with the real post-reset rates so the
        # first control step does not see a fake big derivative spike.
        body_rates = self._world_to_body_rates(self.ang_v[0])
        self._prev_rate[_ROLL]  = float(body_rates[0])
        self._prev_rate[_PITCH] = float(body_rates[1])

        return self._computeObs(), self._computeInfo()


    def _sample_initial_gains(self, rng, mode):
        """Pick a [Kp, Ki, Kd] start vector for one episode.

        mode="recovery": uniform over a fraction of the full bounds. This
            exposes the policy to gain values it would not normally reach
            by itself.
        mode="hold": a multiplier band around the training defaults. The
            band is small enough that a level start with these gains stays
            level, so the only useful action is around zero. This teaches
            the policy to leave the gains alone when nothing is wrong.
        """
        gain_lo = np.array([self.KP_BOUNDS[0], self.KI_BOUNDS[0], self.KD_BOUNDS[0]])
        gain_hi = np.array([self.KP_BOUNDS[1], self.KI_BOUNDS[1], self.KD_BOUNDS[1]])
        if mode == "hold":
            lo, hi = self._hold_gain_mult_range
            m = rng.uniform(lo, hi, 3)
            g = np.array([self.KP_DEFAULT, self.KI_DEFAULT, self.KD_DEFAULT]) * m
        else:  # "recovery"
            lo, hi = self._init_gain_frac_range
            f = rng.uniform(lo, hi, 3)
            g = gain_hi * f
        return np.clip(g, gain_lo, gain_hi)

    def _run_rate_pid(self, axis_idx, rate_sp, rate, kp, ki, kd):
        """Run one rate-PID axis for one timestep. Returns the clipped torque in N.m.

        The function updates self._integral[axis_idx] and self._d_filtered[axis_idx]
        in place. It does not update self._prev_rate. The caller does that
        after both axes are done, so the reward sees the rate from before
        physics is stepped.
        """
        dt  = self.CTRL_TIMESTEP
        err = rate_sp - rate

        # Anti-windup is done in torque space: we clip the integral term
        # to 30% of the max XY torque, then back-compute the integral so
        # the I-state stays consistent with what was actually applied.
        self._integral[axis_idx] += err * dt
        tau_I_max = 0.30 * self.MAX_XY_TORQUE
        tau_I     = ki * self._integral[axis_idx]
        tau_I     = float(np.clip(tau_I, -tau_I_max, tau_I_max))
        self._integral[axis_idx] = tau_I / (ki + 1e-12)

        # Derivative on measurement, with a 1-pole IIR low-pass filter.
        d_raw = -(rate - self._prev_rate[axis_idx]) / dt
        self._d_filtered[axis_idx] = (
            self._D_FILTER_ALPHA * self._d_filtered[axis_idx]
            + (1.0 - self._D_FILTER_ALPHA) * d_raw
        )

        tau = kp * err + tau_I + kd * self._d_filtered[axis_idx]
        return float(np.clip(tau, -self.MAX_XY_TORQUE, self.MAX_XY_TORQUE))

    def _build_inv_alloc(self):
        # Inverse mixer for the F450 in X-configuration.
        d  = self.L / np.sqrt(2)
        kf = self.KF
        km = self.KM
        A = np.array([
            [1,       1,       1,       1      ],
            [-d,     -d,      +d,      +d     ],
            [-d,     +d,      +d,      -d     ],
            [-km/kf, +km/kf, -km/kf, +km/kf ],
        ])
        self._inv_A = np.linalg.inv(A)

    def _torques_to_rpms(self, thrust, tau_roll, tau_pitch, tau_yaw):
        cmd  = np.array([thrust, tau_roll, tau_pitch, tau_yaw])
        F    = self._inv_A @ cmd
        F    = np.maximum(F, 0.0)
        rpms = np.sqrt(F / self.KF)
        return np.clip(rpms, 0.0, self.MAX_RPM)

    def _world_to_body_rates(self, world_ang_v):
        R = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape(3, 3)
        return R.T @ world_ang_v
