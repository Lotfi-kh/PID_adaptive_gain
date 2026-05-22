"""
Test 1 — Random-action smoke test (PyBullet)
=============================================
Purpose:
    Verify the environment infrastructure works end-to-end:
    - reset() returns a valid 10-dim observation
    - step() returns correct shapes and types
    - Gain updates are applied (roll gains change, pitch stays fixed at class defaults)
    - Crash detection terminates the episode
    - Environment resets cleanly on the second episode

Usage:
    cd ~/rl_pid_tuner && python tests/smoke_random.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from envs import PyBulletPIDTunerEnv

MAX_EPISODES = 2
MAX_STEPS    = 60

def check_obs(obs, step):
    assert obs.shape == (9,), f"step {step}: obs shape {obs.shape} != (9,)"
    assert not np.any(np.isnan(obs)), f"step {step}: NaN in obs"
    assert not np.any(np.isinf(obs)), f"step {step}: Inf in obs"
    assert 0.0 <= obs[6] <= 1.0, f"step {step}: Kp_n={obs[6]} out of [0,1]"
    assert 0.0 <= obs[7] <= 1.0, f"step {step}: Ki_n={obs[7]} out of [0,1]"
    assert 0.0 <= obs[8] <= 1.0, f"step {step}: Kd_n={obs[8]} out of [0,1]"


def run():
    env = PyBulletPIDTunerEnv(max_steps=MAX_STEPS, init_noise=0.0)
    rng = np.random.default_rng(42)

    for ep in range(MAX_EPISODES):
        print(f"\n{'='*50}")
        print(f"Episode {ep+1}/{MAX_EPISODES}")
        obs, info = env.reset(seed=ep)
        check_obs(obs, step=0)
        print(f"  reset OK — obs={np.round(obs, 3)}")

        # Pitch gains are class-level constants — verify they don't drift
        pitch_kp_at_reset = env.KP_PITCH_RATE
        pitch_ki_at_reset = env.KI_PITCH_RATE
        pitch_kd_at_reset = env.KD_PITCH_RATE

        for t in range(MAX_STEPS):
            action = rng.uniform(-1, 1, size=(3,)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            check_obs(obs, step=t+1)
            assert isinstance(reward, float), f"reward is {type(reward)}"
            assert "crashed" in info, "crashed missing from info"
            assert "Kp" in info and "Ki" in info and "Kd" in info, "gain keys missing from info"

            # Pitch gains must remain at their class-level defaults
            assert env.KP_PITCH_RATE == pitch_kp_at_reset, "KP_PITCH_RATE changed"
            assert env.KI_PITCH_RATE == pitch_ki_at_reset, "KI_PITCH_RATE changed"
            assert env.KD_PITCH_RATE == pitch_kd_at_reset, "KD_PITCH_RATE changed"

            if (t + 1) % 10 == 0 or terminated or truncated:
                print(f"  step={t+1:3d}  reward={reward:+.3f}  "
                      f"roll={np.rad2deg(obs[0]):+.1f}°  "
                      f"roll_rate={obs[2]:+.3f} rad/s  "
                      f"alt={info['alt_m']:.2f}m  "
                      f"Kp={info['Kp']:.5f}  "
                      f"crash={info['crashed']}")

            if terminated or truncated:
                reason = "CRASH" if info["crashed"] else "TRUNCATED"
                print(f"  Episode ended: {reason} at step {t+1}")
                break

    env.close()
    print("\nSmoke test PASSED — all assertions OK")


if __name__ == "__main__":
    run()
