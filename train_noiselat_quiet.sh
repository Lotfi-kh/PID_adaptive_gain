#!/usr/bin/env bash
# =============================================================================
# Hover-quietness fine-tune of noiselat ("noiselat-quiet").
#
# GOAL: keep noiselat's robustness (off-nominal + noise disturbance rejection)
# but fix its ONE flaw — it is TWITCHY in hover (mean|action| 0.72 vs c860k 0.04)
# because it was trained on noisy gyros and reacts to sensor noise as if it were a
# real disturbance. We add a GATED quietness penalty (reward_w6, new this session):
# action² is penalised ONLY when the TRUE state is calm (att_err < CALM_ATT_THRESH)
# AND no disturbance is active — i.e. exactly when the policy should be doing nothing.
# Disturbance response is gated OUT of the penalty, so it should be untouched.
#
# This is a short, GENTLE fine-tune from the noiselat checkpoint, NOT a fresh run:
#   - resume from noiselat best (1.42M steps) → +400k = 1.82M
#   - low, decaying LR (1e-4 → 2e-5) so robustness is preserved, hover is reshaped
#   - identical env recipe to the noiselat run (noise 0.005, latency 1, dynamics, wind)
#
# CONSTRAINT NOTE: this DELIBERATELY changes the reward (adds w6) — sanctioned by the
# user as an EXPERIMENT, overriding the usual "no reward weight changes" rule. Arch
# [64,64], URDF, motors, yaw all unchanged. w1..w5 unchanged.
#
# TUNING: W6 is env-overridable. Start 1.0. If hover is still twitchy after eval,
# raise it (1.5, 2.0); if disturbance robustness drops on the 48-grid, lower it.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SEED="${SEED:-0}"
W6="${W6:-1.0}"
LR="${LR:-1e-4}"            # iteration 1 used 1e-4 (too gentle — barely quieted)
LR_FINAL="${LR_FINAL:-2e-5}"
BASE="results/frozen_joint_12d_noiselat/td3_pid_noiselat_best.zip"   # robust start
TOTAL_STEPS="${TOTAL_STEPS:-1820000}"                                # 1.42M + 400k

echo "[launch] noiselat-quiet fine-tune  seed=$SEED  w6=$W6  base=$BASE  target=$TOTAL_STEPS"

python train.py \
  --axis roll+pitch \
  --resume "$BASE" \
  --steps "$TOTAL_STEPS" \
  --seed "$SEED" \
  --lr "$LR" \
  --lr-final "$LR_FINAL" \
  --batch-size 256 \
  --reward-w6 "$W6" \
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

# ── AFTER TRAINING — do NOT trust best_model alone ───────────────────────────
# The EvalCallback eval env runs w6=0 (frozen benchmark), so best_model is picked on
# disturbance rejection ONLY — it does not "see" hover quietness. So evaluate the
# final model AND late checkpoints on BOTH axes and pick the one that is quiet AND
# still robust:
#   1) hover quietness:  point make_thesis_figs_noiselat.py MODEL at the candidate,
#      check printed hover mean|action|  (want ≈ c860k's 0.04, not 0.72)
#   2) robustness kept:  EVAL_MODEL=<cand> EVAL_OUT_ROOT=test_results/aggressive_eval_quiet \
#                        python run_aggressive_eval_noiselat.py   (+ offnominal runner)
#      then: python recompute_aggressive_summary.py <out> "noiselat-quiet"
#      want wins/peak/RMS ≈ deployed noiselat (48/48, +28% / 36/48, +13.5%)
# Promote only a candidate that is BOTH quiet AND ~as robust as noiselat.
