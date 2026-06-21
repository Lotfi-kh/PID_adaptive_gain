#!/usr/bin/env bash
# =============================================================================
# Stability-focused retrain of the noise+latency+dynamics policy (noiselat v2).
#
# WHY: the original noiselat run (2026-06-06_14-20-35) DIVERGED at 1.5M — eval
# reward oscillated +18..-300 and the deployed model is a hand-picked mid-run
# peak. That is an OPTIMISATION-STABILITY problem under noisy observations, not
# an under-training one, so the fix is a gentler optimiser, NOT more steps:
#   - lower + linearly DECAYING learning rate  (1e-3 -> 3e-4, decaying to 5e-5)
#   - larger batch (128 -> 256)  ... smoother gradients under sensor noise
#   - a FIXED SEED so the run is reproducible and we can do best-of-N
#
# Constraints honoured: reward weights unchanged (reward_w5=0), actor arch
# unchanged ([64,64]), no URDF / motor / yaw changes. Env physics identical to
# the original noiselat run (randomize-dynamics + noise 0.005 + latency 1 + wind).
#
# Base = c860k (the clean pre-noise model) so the noise phase is redone cleanly
# from a known-good start, rather than continuing a run that already diverged.
#
# Run best-of-N: launch with SEED=0, then 1, then 2; keep the run whose
# best_model has the highest, most STABLE eval reward (check tensorboard).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SEED="${SEED:-0}"
BASE="results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip"   # clean base
TOTAL_STEPS=1500000                                              # c860k@860k -> +640k

echo "[launch] stability retrain  seed=$SEED  base=$BASE  target=$TOTAL_STEPS"

python train.py \
  --axis roll+pitch \
  --resume "$BASE" \
  --steps "$TOTAL_STEPS" \
  --seed "$SEED" \
  --lr 3e-4 \
  --lr-final 5e-5 \
  --batch-size 256 \
  --randomize-disturbance \
  --randomize-initial-gains \
  --hold-episode-prob 0.15 \
  --sustained-episode-prob 0.15 \
  --wind-episode-prob 0.20 \
  --randomize-dynamics \
  --sensor-noise-att 0.005 \
  --sensor-noise-rate 0.005 \
  --control-latency-steps 1 \
  --action-noise-decay

# After training, the usable checkpoint is the EvalCallback best_model:
#   runs/<timestamp>/best_model/best_model.zip
# Evaluate it against the current noiselat on BOTH grids before trusting it:
#   1) edit MODEL/OUT_ROOT in run_aggressive_eval_noiselat.py  -> nominal grid
#   2) edit MODEL/OUT_ROOT in run_offnominal_eval_noiselat.py  -> off-nominal grid
#   3) python recompute_aggressive_summary.py <out_dir> "noiselat-v2"
# Compare 48/48, peak% and (critically) whether the eval-reward curve is STABLE.
