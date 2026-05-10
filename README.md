# RL Adaptive PID Tuner — Phase 1 (PyBullet)

An RL agent (TD3) that **tunes the PID gains of a roll-rate inner loop online**, while a
conventional PID controller stays in charge of the actual motor commands.
The drone is a Crazyflie 2.0 (CF2X) simulated in PyBullet via
[gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones).

Long-term goal: export the trained actor (10 → 256 → 256 → 3 MLP) to ONNX →
STM32Cube.AI → PX4 module on a Pixhawk STM32H7. Phase 1 focuses only on
training in PyBullet.

---

## What is implemented

| Component | File | Status |
|---|---|---|
| PyBullet PID-tuner Gymnasium env | `envs/pybullet_pid_tuner_env.py` | **Working, tested** |
| TD3 short-run trainer            | `tests/short_td3.py`             | **Working** (10k-step run completes in ~52 s) |
| Full TD3 trainer with run dirs   | `train.py`                       | **Working** (`--env pybullet`) |
| Legacy PX4 SITL env (kept for later deployment work) | `envs/px4_gain_tuner_env.py` | Working but requires PX4 SITL + Gazebo |

Hooks already wired:
- 3-dim action: `[ΔKp, ΔKi, ΔKd]` for the **roll-rate** loop only
- 10-dim observation: `[roll, pitch, roll_rate, pitch_rate, roll_rate_err, pitch_rate_err, Kp_n, Ki_n, Kd_n, step_progress]`
- Reward: `-w1·att² - w2·rate² - w3·Δgain² - w4·oscillation²` + crash penalty / stability bonus
- Conservative gain bounds (Kp ∈ [0, 2e-3], Ki ∈ [0, 2e-4], Kd ∈ [0, 1e-5])
  with safe defaults that hover stably under noise without RL
- Roll-only Phase 1: pitch and yaw rate loops use **fixed** PID gains
- Altitude hold via fixed PD on z

---

## What still remains (NOT yet implemented)

Be aware before reading further: these scripts are referenced by my
earlier notes but are **not in this repo yet**. They still need to be ported
from the legacy PX4 versions.

| Missing | Workaround for now |
|---|---|
| **PyBullet baseline runner** (`tests/baseline_pb.py`) | Use `tests/short_td3.py` and look at `monitor.csv`; the legacy `tests/baseline_fixed.py` only runs against the PX4 env |
| **PyBullet baseline-vs-RL comparison** (`tests/compare_pb.py`) | Same — `tests/compare_logs.py` only loads PX4 baseline data |
| **Evaluation script for the PyBullet env** | `evaluate.py` exists but currently imports `PX4GainTunerEnv`. Needs a one-line swap to `PyBulletPIDTunerEnv` to be useful here |
| **Training-curve plot helper** | None — read `monitor.csv` manually or use TensorBoard via `tensorboard --logdir runs/` |
| **Multi-axis (pitch + yaw) tuning** | Out of Phase 1 scope by design — the env code is structured so adding more action dims is mechanical |

The legacy PX4 helper scripts (`tests/smoke_random.py`, `tests/baseline_fixed.py`,
`tests/compare_logs.py`, `evaluate.py`) are kept because they will be useful
once we move to the PX4 deployment phase, but **none of them works against
the PyBullet env without modification**.

---

## Setup

```bash
# Conda or venv with Python 3.10+
pip install -r requirements.txt
```

That installs PyBullet, gym-pybullet-drones, gymnasium, stable-baselines3,
PyTorch (CPU is recommended — see note below), and numpy/scipy.

**GPU note:** the actor is small (~70 k params). On a GTX 1660 Ti the GPU
update is ~0.9× CPU speed because PyBullet (CPU-bound) is the real
bottleneck. Stick with `device="cpu"` unless you switch to vectorized envs.

---

## How to run the PyBullet environment

Smoke-test (1 episode, random actions, no training):

```python
from envs import PyBulletPIDTunerEnv
import numpy as np

env = PyBulletPIDTunerEnv(max_steps=200, init_noise=0.05)
obs, info = env.reset(seed=0)
for _ in range(200):
    action = np.random.uniform(-1, 1, 3).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

The env runs at **~2,900 steps/sec** on CPU.

---

## How to launch training

Short sanity-check run (10k steps, ~1 minute):

```bash
python tests/short_td3.py --steps 10000
```

Outputs to `tests/results/short_td3/` (Monitor CSV, TensorBoard logs,
final model `td3_short_final.zip`).

Full training run (1M steps, ~6 minutes on CPU):

```bash
python train.py --env pybullet --steps 1000000
```

Outputs to `runs/<timestamp>/` (checkpoints every 10k steps, eval every 20k
steps, best model saved separately).

To monitor live:

```bash
tensorboard --logdir runs/
```

---

## How to run evaluation

There is **no PyBullet-native evaluation script in the repo yet**. The two
manual paths that work today:

1. **Inspect the Monitor CSV** at
   `tests/results/short_td3/monitor.csv` (or `runs/<timestamp>/logs/monitor.csv`)
   — columns are reward `r`, episode length `l`, wall-clock `t`.

2. **Roll out a saved model by hand**:

   ```python
   from stable_baselines3 import TD3
   from envs import PyBulletPIDTunerEnv
   import numpy as np

   env   = PyBulletPIDTunerEnv(max_steps=500, init_noise=0.05)
   model = TD3.load("runs/<timestamp>/best_model/best_model.zip", env=env)

   obs, _ = env.reset(seed=0)
   total_rew = 0.0
   for _ in range(500):
       action, _ = model.predict(obs, deterministic=True)
       obs, r, term, trunc, info = env.step(action)
       total_rew += r
       if term or trunc: break
   print(f"reward={total_rew:.1f}  Kp={info['Kp']:.4e}  Ki={info['Ki']:.4e}  Kd={info['Kd']:.4e}")
   env.close()
   ```

A proper `tests/baseline_pb.py` + `tests/compare_pb.py` pair is the next
thing to add — see the "What still remains" table above.

---

## Repository layout

```
envs/
  __init__.py                   exports both envs
  pybullet_pid_tuner_env.py     ← Phase 1 environment (use this)
  px4_gain_tuner_env.py         legacy PX4 SITL env
tests/
  short_td3.py                  10k-step TD3 sanity run on PyBullet env
  smoke_random.py               legacy PX4 smoke test
  baseline_fixed.py             legacy PX4 baseline (NOT ported to PyBullet)
  compare_logs.py               legacy PX4 baseline-vs-TD3 comparison (NOT ported)
train.py                        full TD3 trainer, --env pybullet|px4
evaluate.py                     legacy — uses PX4 env, NOT yet PyBullet
requirements.txt
```

---

## Locked Phase 1 design choices

These are intentional and shouldn't drift without a discussion:

- **One axis only**: roll-rate inner loop. Pitch/yaw rate loops use fixed
  defaults. RL action stays 3-dim until a measurable Phase 1 improvement
  is shown.
- **Tune gains, not motors**: the action is `[ΔKp, ΔKi, ΔKd]`. The actual
  torques and RPMs are produced by a conventional PID + inverse mixer.
- **Small actor**: 256×256 MLP (~70 k params), well under the
  STM32Cube.AI / Pixhawk H7 envelope.
- **PyBullet now, PX4 later**: PX4+Gazebo proved too slow and unreliable
  for the deadline (~25 s per env reset). PyBullet runs ~2,900 steps/sec.
  Deployment to PX4 happens after the policy is trained, via ONNX →
  STM32Cube.AI → custom PX4 module.
