"""
PX4 Rate-PID Gain Tuner — Gymnasium Environment  (Phase 1: roll-rate only)
===========================================================================
The RL agent adapts roll-rate PID gains (MC_ROLLRATE_P/I/D) on a live PX4
SITL instance. Pitch gains are fixed at defaults throughout Phase 1.

Episode flow:
    reset()  → land → disarm → reset Gazebo pose → arm → takeoff → hover
    step()   → apply ΔKp/Ki/Kd (roll only) → wait → observe → reward

Observation (10-dim):
    [roll,            current roll angle (rad)
     pitch,           current pitch angle (rad)
     roll_rate,       roll body rate (rad/s)
     pitch_rate,      pitch body rate (rad/s)
     roll_rate_err,   roll rate error vs zero setpoint (rad/s)
     pitch_rate_err,  pitch rate error vs zero setpoint (rad/s)
     Kp_roll_n,       normalized roll P gain [0, 1]
     Ki_roll_n,       normalized roll I gain [0, 1]
     Kd_roll_n,       normalized roll D gain [0, 1]
     step_progress]   current_step / max_steps  [0, 1]

Action (3-dim, each in [-1, 1]):
    [ΔKp_roll, ΔKi_roll, ΔKd_roll]
    Scaled by DELTA_SCALE before being added to current roll gains.

Reward:
    r = - w1*(roll² + pitch²)               attitude penalty
        - w2*roll_rate_err²                  rate tracking penalty
        - w3*Σ(Δgain_i²)                    gain-change smoothness
        - w4*(Δroll_rate / dt)²             oscillation penalty
        - crash_penalty                      on crash
        + stability_bonus                    on full episode survival
"""

import subprocess
import time

import gymnasium as gym
import numpy as np
from pymavlink import mavutil


# ── PX4 custom-mode encoding ──────────────────────────────────────────────────
# main_mode at bits 16-23, sub_mode at bits 24-31 (px4_custom_mode.h)
_PX4_MODE_AUTO_TAKEOFF = (4 << 16) | (2 << 24)
_PX4_MODE_AUTO_LOITER  = (4 << 16) | (3 << 24)
_PX4_MODE_AUTO_LAND    = (4 << 16) | (5 << 24)


class PX4GainTunerEnv(gym.Env):
    metadata = {"render_modes": []}

    # ── Iris defaults (mc_rate_control_params.c) ──────────────────────────────
    DEFAULT_GAINS = {
        "MC_ROLLRATE_P":  0.15,
        "MC_ROLLRATE_I":  0.20,
        "MC_ROLLRATE_D":  0.003,
        "MC_PITCHRATE_P": 0.15,
        "MC_PITCHRATE_I": 0.20,
        "MC_PITCHRATE_D": 0.003,
    }

    # Phase 1: only roll gains are tunable; pitch stays fixed
    ROLL_KEYS  = ["MC_ROLLRATE_P", "MC_ROLLRATE_I", "MC_ROLLRATE_D"]
    PITCH_KEYS = ["MC_PITCHRATE_P", "MC_PITCHRATE_I", "MC_PITCHRATE_D"]

    # Exploration bounds for roll gains only
    # D-gain max 0.008 stays within PX4 v1.16 parameter validation limits
    GAIN_BOUNDS = {
        "MC_ROLLRATE_P": (0.02, 0.50),
        "MC_ROLLRATE_I": (0.02, 0.50),
        "MC_ROLLRATE_D": (0.0005, 0.008),
    }

    # Max delta per step: [P, I, D]
    DELTA_SCALE = [0.01, 0.01, 0.0005]

    def __init__(
        self,
        connection_string: str = "udp:127.0.0.1:14540",
        step_duration: float = 0.05,      # seconds between gain updates (PX4 rate loop = 250 Hz, 50 ms is plenty)
        max_steps: int = 500,             # steps per episode (50 s)
        takeoff_alt: float = 5.0,         # metres (matches MIS_TAKEOFF_ALT in airframe)
        reward_w1: float = 1.0,           # attitude error weight
        reward_w2: float = 2.0,           # roll-rate error weight
        reward_w3: float = 0.1,           # gain-change smoothness weight
        reward_w4: float = 0.5,           # oscillation weight
        crash_penalty: float = 50.0,
        stability_bonus: float = 200.0,
        init_noise: float = 0.05,         # fractional noise added to roll gains at reset
    ):
        super().__init__()

        self.connection_string = connection_string
        self.step_duration     = step_duration
        self.max_steps         = max_steps
        self.takeoff_alt       = takeoff_alt
        self.reward_w1         = reward_w1
        self.reward_w2         = reward_w2
        self.reward_w3         = reward_w3
        self.reward_w4         = reward_w4
        self.crash_penalty     = crash_penalty
        self.stability_bonus   = stability_bonus
        self.init_noise        = init_noise

        self.mav              = None
        self.step_count       = 0
        self._first_reset     = True
        self.prev_roll_rate   = 0.0       # for oscillation penalty computation

        # Telemetry health tracking
        self._last_attitude        = np.zeros(6, dtype=np.float32)
        self._telemetry_timeout    = False
        self._consecutive_timeouts = 0

        # Tracks last values sent via PARAM_SET so we skip redundant sends
        self._last_sent_gains: dict = {}

        # Roll gains start at defaults; pitch gains are always fixed
        self.current_gains = dict(self.DEFAULT_GAINS)

        # ── Spaces ────────────────────────────────────────────────────────────
        obs_low = np.array([
            -np.pi, -np.pi,   # roll, pitch
            -15.0,  -15.0,    # roll_rate, pitch_rate
            -15.0,  -15.0,    # roll_rate_err, pitch_rate_err
             0.0, 0.0, 0.0,   # normalized roll gains
             0.0,              # step_progress
        ], dtype=np.float32)

        obs_high = np.array([
             np.pi,  np.pi,
             15.0,   15.0,
             15.0,   15.0,
             1.0, 1.0, 1.0,
             1.0,
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    # ── MAVLink helpers ───────────────────────────────────────────────────────

    def _connect(self):
        if self.mav is not None:
            try:
                self.mav.close()
            except Exception:
                pass
        self.mav = mavutil.mavlink_connection(self.connection_string)
        print("[ENV] Waiting for heartbeat …")
        self.mav.wait_heartbeat(timeout=30)
        print(f"[ENV] Connected — system {self.mav.target_system}, "
              f"component {self.mav.target_component}")
        # Disable pre-flight auto-disarm so the timer never fires while we
        # set up the takeoff sequence (0 = disabled).
        self._set_param("COM_DISARM_PRFLT", 0.0)
        time.sleep(0.3)

    def _cmd_long(self, command, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0):
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            command, 0,
            float(p1), float(p2), float(p3),
            float(p4), float(p5), float(p6), float(p7),
        )

    def _set_mode(self, custom_mode: int):
        # SET_MODE uses a uint32 field — avoids float32 precision loss from COMMAND_LONG
        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            custom_mode,
        )
        time.sleep(0.3)

    def _disarm_force(self):
        # p2=21196 bypasses preflight checks
        self._cmd_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, p1=0, p2=21196)
        time.sleep(0.5)

    def _set_param(self, param_id: str, value, int32: bool = False):
        ptype = (mavutil.mavlink.MAV_PARAM_TYPE_INT32
                 if int32 else mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        self.mav.mav.param_set_send(
            self.mav.target_system,
            self.mav.target_component,
            param_id.encode("utf-8"),
            float(value),
            ptype,
        )

    # ── Flight helpers ────────────────────────────────────────────────────────

    def _wait_armed(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if msg and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                return True
        return False

    def _wait_system_ready(self, timeout: float = 60.0) -> bool:
        """Wait for HEARTBEAT system_status == STANDBY (repeating, never missed).
        After STANDBY is confirmed, an extra 3 s sleep lets the EKF finish
        converging before we arm — skipping this causes immediate failsafe on
        the next episode after a crash-reset cycle."""
        print("[ENV] Waiting for system STANDBY …")
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if msg and msg.system_status == mavutil.mavlink.MAV_STATE_STANDBY:
                print("[ENV] System ready (STANDBY) — waiting 3 s for EKF …")
                time.sleep(3.0)
                print("[ENV] Proceeding to arm.")
                return True
        print(f"[ENV] WARNING: STANDBY not seen within {timeout}s — proceeding anyway")
        return False

    def _takeoff(self):
        # 1. Wait for system readiness
        self._wait_system_ready()

        # 2. Force-arm with retry backoff.
        # After a Gazebo teleport the EKF health flags ("GPS Vertical Pos Drift",
        # "Attitude failure") can linger beyond the 10 s sleep. Each denied arm
        # waits 5 s for the flags to clear before retrying.
        armed = False
        for attempt in range(5):
            print(f"[ENV] Arm attempt {attempt + 1}/5 …")
            self._cmd_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, p1=1, p2=21196)
            if self._wait_armed(timeout=5.0):
                armed = True
                break
            print("[ENV] Arm denied — waiting 5 s for health checks to clear …")
            time.sleep(5.0)
        if not armed:
            print("[ENV] WARNING: arm failed after 5 attempts — skipping takeoff")
            return
        print("[ENV] Armed.")

        # 3. Switch to AUTO.TAKEOFF after arming — PX4 climbs to MIS_TAKEOFF_ALT (5 m).
        #    Done AFTER arm (not before) to avoid mode-race conditions.
        print(f"[ENV] Switching to AUTO.TAKEOFF (target {self.takeoff_alt} m) …")
        self._set_mode(_PX4_MODE_AUTO_TAKEOFF)

        # 4. Wait until the drone reaches takeoff altitude
        deadline = time.time() + 30.0
        while time.time() < deadline:
            msg = self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2.0)
            if msg:
                alt = msg.relative_alt / 1000.0
                if alt >= self.takeoff_alt * 0.85:
                    print(f"[ENV] At altitude {alt:.2f} m — switching to LOITER")
                    break

        # 5. Hold position during the episode
        self._set_mode(_PX4_MODE_AUTO_LOITER)
        time.sleep(2.0)

    def _land_and_reset(self):
        # In SITL there is no reason to wait for the drone to physically land.
        # Disarm immediately (motors cut) then teleport the Gazebo model to
        # the origin. This cuts reset time from ~30 s to ~8 s.
        self._disarm_force()
        time.sleep(0.5)
        try:
            subprocess.run(
                ["gz", "model", "--model-name", "iris",
                 "--pose", "0 0 0.15 0 0 0"],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        # 10 s for the EKF to re-converge after the Gazebo teleport.
        # The GPS position jumps instantly (5 m → 0.15 m), so PX4 reports
        # "GPS Vertical Pos Drift too high" until EKF absorbs the new fix.
        # 6 s was too short; 10 s covers even the slow-converge cases.
        time.sleep(10.0)

    # ── Gain helpers ──────────────────────────────────────────────────────────

    def _apply_gains(self):
        """Push all 6 gains (roll tunable + pitch fixed) to PX4. Used at reset only.
        Updates _last_sent_gains so the first step skips redundant re-sends."""
        for param, value in self.current_gains.items():
            self._set_param(param, value)
            time.sleep(0.005)
        for key in self.ROLL_KEYS:
            self._last_sent_gains[key] = self.current_gains[key]

    def _apply_roll_gains(self):
        """Push only roll gains that changed since the last send.
        Skipping unchanged gains reduces PARAM_SET traffic by ~60-80% during
        fine-tuning steps. 5 ms gap spaces out bursts without blocking the loop."""
        for key in self.ROLL_KEYS:
            if self._last_sent_gains.get(key) != self.current_gains[key]:
                self._set_param(key, self.current_gains[key])
                self._last_sent_gains[key] = self.current_gains[key]
                time.sleep(0.005)

    def _update_roll_gains(self, action: np.ndarray) -> dict:
        """Apply action deltas to roll gains. Returns dict of actual deltas applied."""
        deltas = {}
        for i, key in enumerate(self.ROLL_KEYS):
            lo, hi = self.GAIN_BOUNDS[key]
            old = self.current_gains[key]
            new = float(np.clip(old + float(action[i]) * self.DELTA_SCALE[i], lo, hi))
            self.current_gains[key] = new
            deltas[key] = new - old
        return deltas

    def _normalize_roll_gains(self) -> np.ndarray:
        out = []
        for key in self.ROLL_KEYS:
            lo, hi = self.GAIN_BOUNDS[key]
            out.append((self.current_gains[key] - lo) / (hi - lo))
        return np.array(out, dtype=np.float32)

    # ── Observation / reward ──────────────────────────────────────────────────

    def _read_attitude(self) -> np.ndarray:
        """Return [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate].

        On timeout, returns the last valid reading and sets
        self._telemetry_timeout = True so step() can propagate the flag.
        self._consecutive_timeouts tracks repeated misses; step() terminates
        the episode after 5 in a row to avoid silent stale-state training.
        """
        self._telemetry_timeout = False
        msg = self.mav.recv_match(type="ATTITUDE", blocking=True, timeout=0.5)
        if msg is not None:
            self._consecutive_timeouts = 0
            self._last_attitude = np.array(
                [msg.roll, msg.pitch, msg.yaw,
                 msg.rollspeed, msg.pitchspeed, msg.yawspeed],
                dtype=np.float32,
            )
        else:
            self._telemetry_timeout    = True
            self._consecutive_timeouts += 1
        return self._last_attitude.copy()

    def _read_altitude(self) -> float:
        msg = self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=0.5)
        return (msg.relative_alt / 1000.0) if msg else self.takeoff_alt

    def _is_crashed(self, roll: float, pitch: float, alt: float) -> bool:
        return (
            abs(roll)  > np.deg2rad(50) or
            abs(pitch) > np.deg2rad(50) or
            alt < (self.takeoff_alt * 0.4)
        )

    def _build_obs(self, att: np.ndarray) -> np.ndarray:
        roll, pitch, _, roll_rate, pitch_rate, _ = att
        # Setpoint is zero during hover → error = 0 - rate = -rate
        roll_rate_err  = -roll_rate
        pitch_rate_err = -pitch_rate
        step_progress  = float(self.step_count) / float(self.max_steps)
        gains_n        = self._normalize_roll_gains()
        return np.array([
            roll, pitch,
            roll_rate, pitch_rate,
            roll_rate_err, pitch_rate_err,
            gains_n[0], gains_n[1], gains_n[2],
            step_progress,
        ], dtype=np.float32)

    def _compute_reward(
        self,
        att: np.ndarray,
        crashed: bool,
        gain_deltas: dict,
    ) -> float:
        if crashed:
            return -self.crash_penalty

        roll, pitch, _, roll_rate, pitch_rate, _ = att

        # Attitude penalty
        att_err = roll**2 + pitch**2

        # Roll-rate tracking penalty (setpoint = 0)
        rate_err = roll_rate**2

        # Gain-change smoothness penalty
        gain_change = sum(d**2 for d in gain_deltas.values())

        # Oscillation penalty: penalise rapid changes in roll rate
        delta_roll_rate = (roll_rate - self.prev_roll_rate) / self.step_duration
        oscillation = delta_roll_rate**2

        return float(
            -(self.reward_w1 * att_err
              + self.reward_w2 * rate_err
              + self.reward_w3 * gain_change
              + self.reward_w4 * oscillation)
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        if self.mav is None:
            self._connect()

        # Perturb roll gains around defaults; pitch stays fixed at defaults
        for key in self.ROLL_KEYS:
            lo, hi = self.GAIN_BOUNDS[key]
            noise = rng.uniform(-self.init_noise, self.init_noise) * self.DEFAULT_GAINS[key]
            self.current_gains[key] = float(
                np.clip(self.DEFAULT_GAINS[key] + noise, lo, hi)
            )
        for key in self.PITCH_KEYS:
            self.current_gains[key] = self.DEFAULT_GAINS[key]

        if not self._first_reset:
            self._land_and_reset()
        self._first_reset = False

        self._apply_gains()          # all 6 gains at reset (roll new + pitch defaults)
        self._takeoff()

        self.step_count            = 0
        self.prev_roll_rate        = 0.0
        self._consecutive_timeouts = 0   # fresh episode — clear any previous misses
        att = self._read_attitude()
        self.prev_roll_rate = float(att[3])
        return self._build_obs(att), {}

    def step(self, action: np.ndarray):
        try:
            return self._step_impl(action)
        except Exception as e:
            print(f"[ENV] ERROR in step — terminating episode: {e}")
            obs = self._build_obs(self._last_attitude)
            info = {"step": self.step_count, "crashed": True,
                    "telemetry_timeout": False, "error": str(e),
                    "roll_gains": {k: self.current_gains[k] for k in self.ROLL_KEYS},
                    "roll_deg": 0.0, "pitch_deg": 0.0,
                    "roll_rate": 0.0, "alt_m": 0.0}
            return obs, -self.crash_penalty, True, False, info

    def _step_impl(self, action: np.ndarray):
        # 1. Apply roll gain deltas only; pitch gains are never touched here
        gain_deltas = self._update_roll_gains(action)
        self._apply_roll_gains()     # changed gains only, max 3 PARAM_SET

        # 2. Let PID respond
        time.sleep(self.step_duration)

        # 3. Read state  (_read_attitude sets self._telemetry_timeout)
        att = self._read_attitude()
        alt = self._read_altitude()
        crashed = self._is_crashed(att[0], att[1], alt)

        # 4. Reward
        reward = self._compute_reward(att, crashed, gain_deltas)

        # 5. Termination / truncation
        self.step_count += 1
        terminated = crashed
        # Terminate on persistent telemetry loss (5 consecutive misses) so the
        # agent never trains on stale state masquerading as perfect hover.
        if self._telemetry_timeout and self._consecutive_timeouts >= 5:
            terminated = True
        truncated = self.step_count >= self.max_steps

        if truncated and not crashed:
            reward += self.stability_bonus

        # 6. Update state for next step
        self.prev_roll_rate = float(att[3])

        # 7. Observation
        obs  = self._build_obs(att)
        info = {
            "step":              self.step_count,
            "roll_gains":        {k: self.current_gains[k] for k in self.ROLL_KEYS},
            "roll_deg":          float(np.rad2deg(att[0])),
            "pitch_deg":         float(np.rad2deg(att[1])),
            "roll_rate":         float(att[3]),
            "alt_m":             float(alt),
            "crashed":           crashed,
            "telemetry_timeout": self._telemetry_timeout,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.mav:
            try:
                self._disarm_force()
                self.mav.close()
            except Exception:
                pass
            self.mav = None
