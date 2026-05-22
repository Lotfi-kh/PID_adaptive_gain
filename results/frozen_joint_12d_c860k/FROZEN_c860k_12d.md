# FROZEN — Joint 12-D c860k (deployment candidate, 2026-05-18)

**This supersedes the old 13-D frozen 1.05M model
(`results/frozen_joint_1p05M_shared3D/`) as the deployment candidate.**
The old 13-D frozen artifacts are NOT deleted or modified.

## Model

| | |
|---|---|
| Frozen checkpoint | `results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip` |
| Source checkpoint | `runs/2026-05-18_13-27-05/checkpoints/td3_pid_860000_steps.zip` |
| SHA-256 | `c7d3a24b8aac2a5bb1fb56023d5af9571451962282f8d7d67999d073e40c94be` |
| Training step | 860000 (mid-run; run interrupted ~1,000,120) |
| Observation dim | **12** (step_prog removed entirely) |
| Action dim | 3 (shared ΔKp,ΔKi,ΔKd applied to roll AND pitch) |
| Actor (`policy.actor.mu`) | Linear(12,64)→ReLU→Linear(64,64)→ReLU→Linear(64,3)→Tanh |
| actor.mu parameters | 5187 |

### 12-D observation layout (step_prog GONE — invariance by construction)
`[roll, pitch, roll_rate, pitch_rate, roll_rate_err, pitch_rate_err,
  Kp_r/n, Ki_r/n, Kd_r/n, Kp_p/n, Ki_p/n, Kd_p/n]`
(normalized by KP/KI/KD_BOUNDS[1]; old obs[12]=step_prog deleted)

### Reproducibility / environment dependency
This `.zip` is only runnable through the **post-edit 12-D**
`envs/pybullet_pid_tuner_env.py` (`_observationSpace`/`_computeObs` with
step_prog removed; `_sp_rng`/`randomize_step_prog` deleted). A pre-edit 13-D
env will NOT load it. See `training_command.txt`.

## Validation results

### Stable-hover (eval_stable_hover.py, 300 real SITL hover samples)
- **mean|action| = 0.0407** — best in project history
  (sb20-810k 0.0473, cur-810k 0.1469, frozen-1.05M ~1.0)
- step_prog sensitivity: **N/A — eliminated by construction** (no such input)

### Transient disturbance grid (run_disturbance_grid.py, axis=roll+pitch)
- **RL wins 6/6** conditions, 0 crashes
- Peak |roll+pitch_rate| improvement: **−53.8%** avg (6/6 positive)
- RMS roll+pitch_rate improvement: **+8.5%** avg
- Max |roll+pitch| improvement: +1.5% avg
- Full table: `grid_summary.txt` (copied alongside)

### Sustained constant-torque eval (eval_sustained.py, 5 ep, 0.17/0.043 N·m, 6.25 s hold)
Standing tilt on the disturbed axis (deg) — c860k vs fixed baseline:

| condition | baseline | c860k | note |
|---|---|---|---|
| low_roll | 0.87 | 0.27 | ≫ baseline |
| low_pitch | 0.87 | 0.32 | ≫ baseline |
| med_roll | 3.42 | 0.97 | ≫ baseline |
| med_pitch | 3.42 | **3.32** | ⚠ ≈ baseline — see caveat |
| combined_med | 3.42/3.42 | 1.03/1.03 | ≫ baseline |

- **Crashes: 0/5 every condition**
- **Ki PRESERVED: 0.035–0.068** across conditions — never collapses.
  First strong checkpoint in the project that does NOT drive Ki→0.
- **Kp range: 0.37–0.86** (condition-dependent; peak 0.858 on med_roll).
  NOT a runaway — contrast rejected c750k (Kp→1.03, Ki=0) and sb20 (Kp→1.02).
- Kd: ~0.0007–0.0018.

### ⚠ Documented caveat (monitor in SITL — NOT a blocker)
Sustained **med_pitch** (0.17 N·m, 6.25 s): ss tilt **3.32°** vs baseline
3.42°, maxP **7.1°**, recovery **9 steps**. Roughly EQUAL to the fixed
baseline on this one axis — i.e. no improvement there, but **not a
regression** (≥ baseline), 0 crashes. Roll/pitch asymmetry under sustained
pitch torque; every other sustained condition is dramatically better than
baseline. Deadband + HOLD_LIMIT deployment guardrails backstop it. Watch the
pitch axis specifically during SITL validation.

## Selection-protocol verdict
PASS — stable-hover ✓ (0.0407, best) · step_prog ✓ (gone by construction) ·
transient grid ✓ (6/6) · sustained ✓ (≫ baseline on 4/5, ≈ baseline on
med_pitch, 0 crashes) · Kp not running away ✓ · no crashes ✓.
Rejected siblings: c700k (grid 2/6), c750k (Kp/Ki exploit).

## Deployable ONNX (exported + verified 2026-05-18)

| | |
|---|---|
| Frozen ONNX | `results/frozen_joint_12d_c860k/actor_joint_12d_c860k.onnx` |
| Frozen meta | `results/frozen_joint_12d_c860k/actor_joint_12d_c860k.meta.json` |
| Working copy | `export/actor_joint_12d_c860k.onnx` (+ `.meta.json`) |
| ONNX SHA-256 | `4802122f3d5dc30ba70fab41b6a8cf9898413286502e613d2f5a547ed431c9b6` |
| ONNX I/O | `obs[1,12]` → `action[1,3]`, opset 13, 20.9 KB |

- **Export: DONE** — `export/export_actor_onnx.py`, input dim auto-detected 12,
  5187 params, architecture Linear(12,64)→ReLU→Linear(64,64)→ReLU→Linear(64,3)→Tanh.
- **Verify: PASS** — `export/verify_actor_onnx.py`, ONNX vs PyTorch bit-exact:
  max|Δ| = 5.96e-08 (zeros) / 0.00 (ones, random, edge), tol 1e-5, all 4 PASS.
  (Synthetic vectors saturate ±1.0 — that is an out-of-distribution-input
  artifact, NOT behaviour; real-obs stable-hover validated separately at 0.0407.)
- The old 13-D `export/actor_joint_1p05M_shared3D.onnx` was NOT overwritten
  (explicit `--out`).

## Status / pending (NOT started — do not touch until approved)
- Deployment-time 12-D edits: `sitl/sitl_gain_injector.py` (`OBS_DIM` 13→12 +
  drop step_prog slot), `export/gen_test_vectors.py`, STM32 regen.
- SITL validation of c860k (watch the pitch axis — see med_pitch caveat above).
- X-CUBE-AI: NOT run.

## Code-hygiene fixes applied to `export/export_actor_onnx.py`
- Docstring + meta string 13-D → 12-D (exported `.meta.json` reports correct dim).
- `datetime.utcnow()` → `datetime.now(timezone.utc)` (deprecation-free; identical
  `...Z` output format preserved).
