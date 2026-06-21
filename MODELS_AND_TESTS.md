# Models & Tests — Project Reference

*Written as an onboarding doc: assume you are seeing this project for the first time.*
*Last updated: 2026-06-09.*

---

## 0. What this project is (30-second version)

A quadcopter (DJI F450 / PX4) normally flies with **fixed PID gains** for its
roll/pitch **rate** controller. This project trains a small neural network (RL
policy) that **nudges those gains in real time** — raising them when a disturbance
hits, relaxing them when things are calm. The claim we are trying to prove is:

> An adaptive (NN-tuned) PID beats a static (fixed-gain) PID at rejecting
> disturbances — *especially* when the drone is not perfect (noisy sensors, slightly
> wrong airframe).

The NN has a tiny, fixed shape: **12 inputs → 64 → 64 → 3 outputs** (Tanh). The 3
outputs are `[ΔKp, ΔKi, ΔKd]` — small changes added to the running gains each
control step (48 Hz). It is trained with **TD3** (stable-baselines3) in a **PyBullet**
simulation, then frozen and tested.

There are **two simulators** in play:
- **PyBullet** — fast, used for *training* and the big *statistical* test grids.
- **PX4 SITL + Gazebo** — the real firmware in a realistic sim, used for final
  *validation* (the NN actually runs inside the PX4 module `rl_gain_tuner`).

---

## 1. The three models

All three share the *same* network architecture. They differ ONLY in **what the
training environment punished**, which is the single knob that trades
**aggressiveness ↔ robustness**.

| | **c860k** | **noiselat** | **noiselat-quiet** |
|---|---|---|---|
| Role | aggressive specialist | robust generalist (deployed) | robust + quiet (new, best) |
| Frozen file | `results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip` | `results/frozen_joint_12d_noiselat/td3_pid_noiselat_best.zip` | `results/frozen_joint_12d_noiselat_quiet/td3_pid_noiselat_quiet_best.zip` |
| Trained in | **clean** sim | **hardened** sim | hardened sim + quietness reward |
| Resumed from | phase-1 roll model | c860k | noiselat |
| Best checkpoint | 860k steps | 1.42M steps | 1.76M steps |

> ⚠️ For **all three**, the *final* model of the run is NOT the one to use — every run
> overshot/diverged late. Use the frozen `best_model`/`860000` checkpoint named above.

### 1a. c860k — the aggressive specialist
- **Trained:** clean PyBullet (no sensor noise, no latency, fixed airframe).
  Two phases: roll-only, then joint roll+pitch with a 12-D observation (the old
  `step_prog` input was removed). Exact command:
  `results/frozen_joint_12d_c860k/training_command.txt`.
- **Good at:** raw nominal disturbance rejection — biggest peak-rate reduction
  (+55% in-distribution), **fastest reaction & recovery** (see §3.4), crisp quiet
  hover (mean|action| 0.041), a dramatic-but-bounded Kp (0.95 then relaxes to 0.86),
  and a live Ki.
- **Bad at:** off-nominal + sensor noise — it *loses to the static baseline* there.
- **Why:** a clean sim never punished high gains, so it learned to crank Kp. Under
  real sensor noise, high Kp *amplifies the noise* and the advantage flips to a
  liability.

### 1b. noiselat — the deployed robust generalist
- **Trained:** *hardened* PyBullet — sensor noise (σ=0.005 att & rate), 1-step
  control latency, randomized mass/inertia, and a mix of disturbance episodes
  (kick / hold / sustained / **wind ~20%**). Resumed from c860k. Name = **noi**se +
  **lat**ency. Command: `results/frozen_joint_12d_noiselat/training_command.txt`.
- **Good at:** **robustness** — 48/48 nominal wins, 36/48 off-nominal wins, **0
  crashes in 1920 rollouts**. Holds up where c860k collapses. This is the model that
  actually delivers the project's claim, so it is the **deployed** one.
- **Bad at:** (a) **twitchy hover** — mean|action| 0.724 (~18× c860k) because it
  reacts to gyro noise as if it were a disturbance; (b) **Ki collapses to 0.000**
  under sustained torque (a defect — leaves the steady-state tilt uncorrected).
- **Why:** noise/latency training killed the high-gain exploit, forcing moderate
  robust gains — but the same noise-reactivity makes it fidget in hover, and it
  found a degenerate "dump Ki" strategy.

### 1c. noiselat-quiet — robust + quiet + Ki fixed (NEW, 2026-06-09)
- **Trained:** resume from noiselat into the *same hardened env*, plus a new
  **gated hover-quietness reward** (`reward_w6=4.0`): the action is penalized
  *only* when the true state is calm AND no disturbance is active — i.e. exactly
  when the policy should be doing nothing. Recipe (env-overridable wrapper):
  `SEED=0 W6=4.0 LR=3e-4 LR_FINAL=5e-5 TOTAL_STEPS=2020000 ./train_noiselat_quiet.sh`
  Full provenance: `results/frozen_joint_12d_noiselat_quiet/PROVENANCE.txt`.
- **Good at:** **strictly dominates deployed noiselat** — same robustness (48/48,
  36/48, 0 crashes), **4.2× quieter hover (0.173)**, and **Ki kept alive (0.045,
  no collapse — defect fixed)**.
- **Bad at:** 0.173 is quieter than noiselat but still 4× c860k's 0.041 (not
  "crisply inactive"); reaction/recovery are **identical to noiselat** (quieting
  bought no timing gain); the run **overshot** (late checkpoints saturated to 0.98 —
  junk), so ONLY `best_model` (1.76M) is usable.
- **Why:** the gated penalty taught it to stop fidgeting in calm hover without
  touching its disturbance response (gated out), and as a side effect it abandoned
  the Ki→0 trick.

### Models that are NOT contenders (so you don't get confused)
- **frozen 1.05M** — earlier joint model, superseded by c860k.
- **seed1 stability-retrain** — low-LR retrain from c860k; looked stable on reward
  but **lost to baseline** on the grids (8/48). Results in
  `test_results/aggressive_eval_seed1/`, `test_results/offnominal_eval_seed1/`.
  Lesson: judge a model on the disturbance grid, not on training reward.
- **noiselat-quiet `final` (2.02M)** — the same good run's endpoint, saturated.
  Junk. Only `best_model` survives.

---

## 1.5 The training environment (shared by all three)

All models train in the SAME PyBullet env (`envs/pybullet_pid_tuner_env.py`,
`PyBulletPIDTunerEnv`). The models differ only in which **hardening flags** and
**reward weights** are switched on (next two sections). The fixed setup:

- **Sim / airframe:** PyBullet (gym-pybullet-drones `BaseAviary`), DJI F450 URDF —
  mass 1.2 kg, Ixx=Iyy=0.012, Izz=0.019.
- **Rate:** 48 Hz control; episode = 500 steps (~10.4 s).
- **Controller chain (mirrors PX4):** outer attitude P (`KP_ATT=3.0`) makes a rate
  setpoint → inner **rate PID** (this is what the NN tunes: Kp,Ki,Kd) → torque →
  motors. Yaw is fixed. Includes anti-windup + torque clipping.
- **Gains:** defaults Kp=0.171, Ki=0.0086, Kd=0.00171; hard bounds Kp∈[0,1.72],
  Ki∈[0,0.172], Kd∈[0,0.0086].
- **Action (3-D):** `[ΔKp, ΔKi, ΔKd]` each in [−1,1] (Tanh), scaled by
  `DELTA_SCALE=[0.034, 0.0034, 0.00017]` (= 2% of each gain's range *per step*),
  added to the running gains and clipped to the bounds. Shared across roll & pitch.
- **Observation (12-D):** `[roll, pitch, roll_rate, pitch_rate, roll_rate_err,
  pitch_rate_err, Kp_r, Ki_r, Kd_r, Kp_p, Ki_p, Kd_p]` (the 6 gains normalized to
  their upper bound). *(The old `step_prog` input was removed — that is what
  "12-D" refers to throughout this project.)*
- **Crash:** |roll|>60° or |pitch|>60° or altitude outside [0.15, 2.5] m.
- **Disturbance / episode types** (mixed per-episode by probability):
  | type | torque | when |
  |---|---|---|
  | default kick | up to 0.25 N·m, 3–10 steps | random step |
  | hold episode | 0.03 N·m constant, both axes | from step 1 |
  | sustained episode | 0.10–0.15 N·m constant | from step 1 |
  | wind episode | OU gusty: mean 0.06–0.12 + gust σ=0.05, cap 0.20 N·m | whole episode |
- **Hardening flags (noiselat family ONLY):** `--randomize-dynamics` (mass/inertia
  perturbed each episode), `--sensor-noise-att/rate 0.005` (the gyro/attitude the
  controller AND policy SEE is noisy), `--control-latency-steps 1` (commands act a
  beat late). c860k had ALL of these OFF.

> **The single most important design choice:** the controller and policy see *noisy*
> measurements, but the **reward is computed from the TRUE state**. So a policy
> cannot earn reward by reacting to noise — it must stabilize the *real* drone. This
> is exactly what later exposed noiselat's hover twitch (it fidgets at noise that
> doesn't help true-state reward) and what the quietness reward then penalized.

## 1.6 The reward function

Per step (from `_computeReward`), the reward is a weighted penalty (more negative =
worse), computed on TRUE state:

```
reward = −( w1·att_err + w2·rate_err + w3·gain_change + w4·oscillation
            + w5·effort + w6·quiet )
  + stability_bonus   (+200, added only on surviving to the last step)
  − crash_penalty     (50, on a crash)
```

| term | what it is | weight |
|---|---|---|
| `att_err` | roll² + pitch²  (stay level) | **w1 = 1.0** |
| `rate_err` | (rate_sp − rate)² roll+pitch  (track the rate setpoint) | **w2 = 2.0** |
| `gain_change` | Σ(action²)  (don't thrash the gains) | **w3 = 0.1** |
| `oscillation` | (Δrate/dt)²  (no jerky rates) | **w4 = 0.001** |
| `effort` | τ²/τ_ref²  (use less torque) | **w5 = 0.0 (off)** |
| `quiet` | `gain_change`, but ONLY when calm & no disturbance | **w6 = 0.0 default** |

- **w5 (effort)** is wired but left at 0 → identical to the original frozen reward.
- **w6 (quiet)** is the NEW term added for noiselat-quiet. It is **gated**: it equals
  `gain_change` *only* when `att_err < 5e-3 rad²` (drone is calm) AND no disturbance
  is active — i.e. exactly when the policy should be doing nothing. Disturbance
  response is gated OUT, so robustness is untouched. **w6 = 0 → byte-identical to the
  frozen reward; noiselat-quiet used w6 = 4.0.**
- c860k and noiselat both trained with **w1..w4 only** (w5=w6=0). The ONLY reward
  difference in the whole project is noiselat-quiet's w6=4.0.

## 1.7 How we got here: c860k → noiselat → noiselat-quiet (the problem chain)

Each model exists because the previous one had a concrete, measured flaw. The single
lever each time was *what the env punishes*.

**Start → c860k.** Phase 1 trained roll-only (+73.7% peak on the roll grid) — proof
the idea works on one axis. Phase 2 extended to joint roll+pitch and removed
`step_prog` (→ 12-D obs), in a **clean** sim. Gave c860k: aggressive, +55% nominal,
fast, crisp hover.
- **Problem:** the clean sim never punished high gains, so c860k learned to crank Kp
  to ~0.95. The moment you add sensor noise / a wrong airframe (SITL, off-nominal
  grid), high Kp *amplifies the noise* and c860k **loses to the static baseline**.
  It is a nominal-only specialist. → we need robustness.

**c860k → noiselat.** Retrained *from* c860k with all the **hardening flags** on
(noise + latency + randomized dynamics) plus the wind/hold/sustained episode mix.
Now high gains are costly (they amplify the noise the reward sees through the true
state), forcing moderate, robust gains. Gave noiselat: 48/48 nominal, 36/48
off-nominal, 0 crashes — so it became the **deployed** model.
- **Problem (two flaws):**
  1. **Twitchy hover** — mean|action| 0.724 (~18× c860k). Having learned to watch the
     gyro closely, it reacts to *noise* in calm hover as if it were a disturbance,
     constantly nudging gains for no reason.
  2. **Ki collapse** — under a sustained torque it dumps Ki→0.000 (the "spike Kp /
     dump Ki" exploit), a defect that leaves the steady-state tilt uncorrected.
  Both contradict the desired "quiet in hover, live integral" behaviour. → we need to
  quiet it *without* losing the robustness.

**noiselat → noiselat-quiet.** Added the **gated quietness reward** (w6, §1.6):
penalize gain changes *only* in calm, undisturbed flight; gate the term OFF whenever
a disturbance is active so disturbance rejection is untouched. Retrained from
noiselat.
- **Iteration:** first try (w6=1.0, gentle LR 1e-4→2e-5) barely moved hover
  (0.724→0.69) — too weak. Second try (**w6=4.0, LR 3e-4→5e-5, +600k steps**) worked:
  hover 0.173 (4.2× quieter), Ki alive at 0.045 (collapse fixed), robustness held
  (48/48, 36/48, 0 crashes). The run **overshot** (late checkpoints saturated to
  0.98), so the keeper is `best_model` @ 1.76M, not `final`.
- **Remaining caveats:** 0.173 is much quieter than noiselat but still 4× c860k's
  0.041 (not perfectly "inactive"); and adding w6 broke the project's "no reward-
  weight changes" rule — done deliberately, **as a sanctioned experiment**.

---

## 2. The headline numbers (all three)

| metric | c860k | noiselat | noiselat-quiet |
|---|---|---|---|
| Hover mean\|action\| (quiet = good) | **0.041** | 0.724 | 0.173 |
| Nominal grid (wins / peak Δ) | +55% in-dist | 48/48 / +28.0% | 48/48 / +26.8% |
| Off-nominal grid (wins / peak Δ) | **loses to baseline** | 36/48 / +13.5% | 36/48 / +13.1% |
| Off-nom RMS Δ | — | +17.0% | +10.6% |
| Reaction (IAE onset, lower=better) | **0.026** | 0.042 | 0.038 |
| Recovery time (lower=better) | **0.208 s** | 0.479 s | 0.479 s |
| Kp under torque | 0.95→0.86 | 0.33 (Ki dead) | 0.26 (Ki alive) |
| Ki under torque | alive | **collapses→0** | alive 0.045 |
| Crashes (per 1920 rollouts) | — | 0 | 0 |

**The one-axis story:** clean sim → fast but fragile (c860k); add noise/latency →
robust but twitchy (noiselat); add a gated quietness reward → robust AND quiet AND
Ki-fixed (noiselat-quiet), at no cost to (and no gain in) reaction speed.

---

## 3. The tests — what each one is, how it's run, where the data lands

There are **5 distinct tests**. Tests 1–4 run in **PyBullet** (fast, statistical).
Test 5 runs in **PX4 SITL + Gazebo** (slow, realistic, real firmware).

All PyBullet eval scripts run from `~/rl_pid_tuner/` and need the conda python:
`PYTHONPATH=. ~/miniconda3/bin/python <script>.py`.
Every script takes the model via `EVAL_MODEL=<path>` and the output dir via
`EVAL_OUT_ROOT=<path>` (defaults point at noiselat).

### 3.1 Nominal disturbance grid  ("aggressive" grid)
- **Question:** on a *perfect* drone, does the NN beat fixed gains across many
  disturbance sizes / axes / starting wobbles?
- **How:** a 48-condition sweep =
  `init_noise ∈ {0.05,0.10,0.15,0.20}` × `dist_magnitude ∈ {0.043,0.17,0.30,0.40} N·m`
  × `axis ∈ {roll,pitch,both}`, **20 episodes each = 960 rollouts** per controller.
  Each rollout applies a torque kick and records roll/pitch rate. The NN run is
  compared against a static-gain run in the SAME condition.
- **Run it:**
  `EVAL_MODEL=<model> EVAL_OUT_ROOT=test_results/aggressive_eval_<name> python run_aggressive_eval_noiselat.py`
- **Data saved to:** `test_results/aggressive_eval_<name>/`
  - one sub-folder per condition (e.g. `noise05_medium_roll/`) each containing
    `disturbance_eval_results.npz` (raw per-episode arrays) +
    `disturbance_comparison_summary.txt`.
  - the per-cell rollout is produced by `eval_disturbance.py` (the single-condition
    worker the grid calls in a loop).
- **Frozen results:** c860k `test_results/aggressive_eval/`, noiselat
  `test_results/aggressive_eval_noiselat/`, noiselat-quiet
  `test_results/aggressive_eval_quiet/`.

### 3.2 Off-nominal disturbance grid
- **Question:** on an *imperfect* drone (wrong airframe + noisy gyro + control lag),
  does the NN still beat fixed gains? **This is the project's real claim.**
- **How:** SAME 48-condition sweep as 3.1, but every rollout ALSO has
  `--randomize-dynamics` (perturbed mass/inertia) + `--sensor-noise-att/rate 0.005`
  + `--control-latency-steps 1`. Again 960 rollouts/controller.
- **Run it:**
  `EVAL_MODEL=<model> EVAL_OUT_ROOT=test_results/offnominal_eval_<name> python run_offnominal_eval_noiselat.py`
- **Data saved to:** `test_results/offnominal_eval_<name>/` (same structure as 3.1).
- **Frozen results:** noiselat `test_results/offnominal_eval_noiselat/`,
  noiselat-quiet `test_results/offnominal_eval_quiet/`.

> **IMPORTANT — read summaries with the corrected aggregator.** The grid scripts'
> built-in summary is **roll-axis-only and buggy**. Always re-score with the
> axis-aware tool:
> `python recompute_aggressive_summary.py test_results/<dir> "<label>"`
> It writes `aggressive_eval_corrected.txt` into the dir and prints wins/peak/RMS/
> recovery broken out by magnitude and axis. **Trust the `_corrected.txt` file, not
> the raw `_summary.txt`.**

### 3.3 Hover quietness check
- **Question:** when nothing is wrong, does the policy stay still (or fidget)?
- **How:** load the model, feed it **300 recorded SITL hover observations**
  (`results/observer_2026-05-16_19-24-48.csv`), print `mean|action|`. No disturbance.
- **Run it:** `PYTHONPATH=. python eval_hover_quietness.py` (prints all 3 models).
- **Thesis figure version:** `make_thesis_figs_noiselat.py` does the same on the
  current headline model and writes `~/thesis/images/hover_action.png` (it currently
  points at noiselat-quiet).

### 3.4 Gain-trajectory & reaction-time checks (single controlled episode)
- **Question (gains):** under a *sustained* torque, does the NN raise Kp sensibly
  and keep Ki alive (vs the collapse defect)?
- **Question (timing):** how *fast* does it react and recover?
- **How:** one deterministic, **noise-free** episode — sustained 0.17 N·m roll
  torque applied at step 100 for 300 steps (48 Hz) — same episode for every model.
  - `eval_gain_trajectory_compare.py` → prints Kp/Ki/Kd peak/steady/final +
    the Ki-min-during-torque collapse check.
  - `eval_reaction_time.py` → prints peak |rate|, time-to-peak, **IAE** (integral of
    |roll-rate| = how big an excursion for how long), and post-torque recovery time,
    for static baseline + all 3 models.
- **Run them:** `PYTHONPATH=. python eval_gain_trajectory_compare.py` /
  `PYTHONPATH=. python eval_reaction_time.py`.
- **Data saved to:** these print to stdout (no files). The thesis gain figure is
  `~/thesis/images/gain_trajectory.png` via `make_thesis_figs_noiselat.py`.
- **Note:** the timing test is *noise-free nominal*, so c860k looks best here — that
  is exactly the high-gain edge that *inverts* under noise (§3.2). Fast here ≠ better
  deployed.

### 3.5 PX4 SITL + Gazebo A/B  (final, realistic validation)
- **Question:** does any of this hold up with the *real PX4 firmware* in a realistic
  sim, where the NN runs inside the `rl_gain_tuner` module?
- **How:** paired runs — same disturbance, once with the RL module OFF (baseline)
  and once ON (adaptive) — flown autonomously (arm → takeoff → disturb → land),
  logged to PX4 ulog, then metrics extracted and compared with proper stats
  (paired Wilcoxon + Cohen's d + win-rate). One command:
  `python sitl/sitl_disturb_batch.py` (10 paired runs + a magnitude sweep, headless).
- **Key scripts:** `sitl/sitl_disturb_batch.py` (orchestrator),
  `sitl/sitl_disturb.py` (torque injector, `--seed`/`--scale`),
  `sitl/extract_metrics.py` (ulog → rich metrics), `sitl/aggregate_stats.py`
  (stats + plots), `sitl/sitl_wind_comparison.py` (wind variant + shared lifecycle
  helpers), `sitl/ab_compare.py` (metric math).
- **Data saved to:** `test_results/gazebo/`
  - `metrics_master.csv` (kick A/B, one row per run — columns: peak|roll|/|pitch|,
    peak rates, RMS, recovery, IAE rate, ∫|torque|, motor saturation %, alt
    deviation, crash flag, …), `metrics_master_wind.csv` (wind A/B),
    `metrics_master_events.csv`.
  - plots: `paired_compare.png`, `improvement_kick.png`, `improvement_wind.png`.
  - raw ulogs + older runs under `test_results/gazebo/_archive*/`.
- **What it showed (important reality check):**
  - **Wind A/B = TIE** — the adaptive model does NOT beat the baseline under pure
    wind (a wind-focused retrain only reached *parity*).
  - **Kick A/B = MIXED** — significant wins on RMS roll / rate-error and altitude
    hold, but a loss on peak |roll| and peak torque.
  - **0 crashes throughout.**
  - Takeaway: the big PyBullet win lives in the **off-nominal + noise** regime
    (§3.2), not in nominal Gazebo, because the Gazebo airframe is nominal and its
    static gains are already well-tuned.

---

## 4. Quick "where is it" index

| thing | path |
|---|---|
| Frozen models | `results/frozen_joint_12d_{c860k,noiselat,noiselat_quiet}/` |
| Each model's training command | `<frozen_dir>/training_command.txt` (quiet: `PROVENANCE.txt`) |
| Nominal grid results | `test_results/aggressive_eval{,_noiselat,_quiet,_seed1}/` |
| Off-nominal grid results | `test_results/offnominal_eval_{noiselat,quiet,seed1}/` |
| Corrected grid summaries | `<grid_dir>/aggressive_eval_corrected.txt` |
| Gazebo / SITL results | `test_results/gazebo/metrics_master*.csv` + `*.png` |
| Hover capture (for §3.3) | `results/observer_2026-05-16_19-24-48.csv` |
| Thesis figures | `~/thesis/images/{hover_action,gain_trajectory}.png` |
| Training entrypoint | `train.py` (+ `train_noiselat_quiet.sh` wrapper) |

### Eval scripts cheat-sheet (run from `~/rl_pid_tuner/`)
```bash
# hover quietness (all 3 models)
PYTHONPATH=. ~/miniconda3/bin/python eval_hover_quietness.py
# gain trajectory + Ki-collapse check
PYTHONPATH=. ~/miniconda3/bin/python eval_gain_trajectory_compare.py
# reaction / recovery timing
PYTHONPATH=. ~/miniconda3/bin/python eval_reaction_time.py
# nominal grid on a model  (then re-score)
EVAL_MODEL=results/frozen_joint_12d_noiselat_quiet/td3_pid_noiselat_quiet_best.zip \
EVAL_OUT_ROOT=test_results/aggressive_eval_quiet \
  ~/miniconda3/bin/python run_aggressive_eval_noiselat.py
python recompute_aggressive_summary.py test_results/aggressive_eval_quiet "noiselat-quiet NOMINAL"
# off-nominal grid on a model  (then re-score)
EVAL_MODEL=...quiet... EVAL_OUT_ROOT=test_results/offnominal_eval_quiet \
  ~/miniconda3/bin/python run_offnominal_eval_noiselat.py
python recompute_aggressive_summary.py test_results/offnominal_eval_quiet "noiselat-quiet OFFNOM"
# full Gazebo A/B (slow, ~hours)
python sitl/sitl_disturb_batch.py
```

---

## 5. Bottom line

- **Deployed today:** noiselat. **Best model we have:** noiselat-quiet (strictly
  dominates noiselat — same robustness, 4.2× quieter, Ki fixed).
- **For raw nominal speed only:** c860k wins, but it can't honestly be called the
  robust deployed model — it loses off-nominal.
- The real, defensible contribution is the **off-nominal + sensor-noise** regime,
  where adaptive genuinely beats static (§3.2). Pure-wind Gazebo is a tie (§3.5).
