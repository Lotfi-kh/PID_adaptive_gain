"""
Test 1 — Random-action smoke test
==================================
Purpose:
    Verify the environment infrastructure works end-to-end:
    - reset() connects, takes off, returns a valid 10-dim observation
    - step() returns correct shapes and types
    - Gain updates are applied (roll gains change, pitch stays fixed)
    - Telemetry timeout flag is present in info
    - Crash detection terminates the episode
    - _land_and_reset() runs cleanly on the second episode

Usage:
    Start PX4 SITL first:
        cd ~/PX4-Autopilot && make px4_sitl gazebo-classic_iris

    Then run:
        cd ~/rl_pid_tuner && python tests/smoke_random.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from envs import PX4GainTunerEnv

MAX_EPISODES = 2
MAX_STEPS    = 60     # ~6 s per episode — just enough to verify the loop

def check_obs(obs, step):
    assert obs.shape == (10,), f"step {step}: obs shape {obs.shape} != (10,)"
    assert not np.any(np.isnan(obs)), f"step {step}: NaN in obs"
    assert not np.any(np.isinf(obs)), f"step {step}: Inf in obs"
    # gains should be in [0, 1]
    assert 0.0 <= obs[6] <= 1.0, f"step {step}: Kp_n={obs[6]} out of [0,1]"
    assert 0.0 <= obs[7] <= 1.0, f"step {step}: Ki_n={obs[7]} out of [0,1]"
    assert 0.0 <= obs[8] <= 1.0, f"step {step}: Kd_n={obs[8]} out of [0,1]"
    # step_progress in [0, 1]
    assert 0.0 <= obs[9] <= 1.0, f"step {step}: step_progress={obs[9]} out of [0,1]"


def run():
    env = PX4GainTunerEnv(max_steps=MAX_STEPS, init_noise=0.0)
    rng = np.random.default_rng(42)

    for ep in range(MAX_EPISODES):
        print(f"\n{'='*50}")
        print(f"Episode {ep+1}/{MAX_EPISODES}")
        obs, info = env.reset(seed=ep)
        check_obs(obs, step=0)
        print(f"  reset OK — obs={np.round(obs, 3)}")

        # Snapshot pitch gains at reset — they must never change
        pitch_gains_at_reset = {k: env.current_gains[k] for k in env.PITCH_KEYS}

        timeouts = 0
        for t in range(MAX_STEPS):
            action = rng.uniform(-1, 1, size=(3,)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            check_obs(obs, step=t+1)
            assert isinstance(reward, float), f"reward is {type(reward)}"
            assert "telemetry_timeout" in info, "telemetry_timeout missing from info"
            assert "crashed" in info

            if info["telemetry_timeout"]:
                timeouts += 1
                print(f"  [WARN] step {t+1}: telemetry timeout "
                      f"(consecutive={env._consecutive_timeouts})")

            # Pitch gains must not have changed
            for k in env.PITCH_KEYS:
                assert env.current_gains[k] == pitch_gains_at_reset[k], \
                    f"Pitch gain {k} changed from {pitch_gains_at_reset[k]} " \
                    f"to {env.current_gains[k]}"

            if (t + 1) % 10 == 0 or terminated or truncated:
                print(f"  step={t+1:3d}  reward={reward:+.3f}  "
                      f"roll={np.rad2deg(obs[0]):+.1f}°  "
                      f"roll_rate={obs[2]:+.3f} rad/s  "
                      f"alt={info['alt_m']:.2f}m  "
                      f"Kp={env.current_gains['MC_ROLLRATE_P']:.4f}  "
                      f"crash={info['crashed']}")

            if terminated or truncated:
                reason = "CRASH" if info["crashed"] else "TRUNCATED"
                print(f"  Episode ended: {reason} at step {t+1}  "
                      f"timeouts={timeouts}")
                break

    env.close()
    print("\nSmoke test PASSED — all assertions OK")


if __name__ == "__main__":
    run()
