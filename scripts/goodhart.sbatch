#!/bin/bash
#SBATCH --job-name=dv3_goodhart
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Evaluator-Goodhart sprint (prereg/PREREG_goodhart_sprint_20260714.md):
# score every policy checkpoint matching POLICY_GLOB inside one learned
# world-model evaluator, then collate pool metrics (rank correlation,
# top-k regret, inversions). Inference only; resumable (per-policy skip).
#
# Submit directly (single job):
#   sbatch --export=ALL,EVALUATOR=$RUNROOT/adapt_ax1q1s1_finger_seed1_ckpt500000,\
# POLICY_GLOB='adapt_*_finger_*_ckpt500000',INIT_REPLAY=$RUNROOT/pilot_goal_finger/replay,\
# OUT=$RUNROOT/goodhart_finger scripts/goodhart.sbatch
#
# Required env: EVALUATOR, POLICY_GLOB, INIT_REPLAY, OUT
# Optional: N_STARTS (20), BURN_IN (16), HORIZON (64)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${EVALUATOR:?}"; : "${POLICY_GLOB:?}"; : "${INIT_REPLAY:?}"; : "${OUT:?}"
N_STARTS="${N_STARTS:-20}"; BURN_IN="${BURN_IN:-16}"; HORIZON="${HORIZON:-64}"
mkdir -p "$OUT"

n=0
for pol in "$RUNROOT"/$POLICY_GLOB; do
  [ -d "$pol/ckpt" ] || continue
  [ -f "$pol/scores.jsonl" ] || continue
  pid="$(basename "$pol")"
  [ "$pol" = "$EVALUATOR" ] && { echo "SKIP evaluator itself: $pid"; continue; }
  out_json="$OUT/eval_${pid}.json"
  [ -f "$out_json" ] && { echo "SKIP scored: $pid"; continue; }
  python -m probing.wm_evaluator score \
    --evaluator_run "$EVALUATOR" --policy_run "$pol" \
    --init_replay "$INIT_REPLAY" --n_starts "$N_STARTS" \
    --burn_in "$BURN_IN" --horizon "$HORIZON" \
    --output "$out_json"
  n=$((n+1))
done
echo "[$(date)] scored $n new policies"

python -m probing.wm_evaluator collate \
  --evals "$OUT/eval_*.json" --runroot "$RUNROOT" \
  --real_window 10 --output "$OUT/pool_metrics.json"
echo "[$(date)] Goodhart sweep DONE -> $OUT/pool_metrics.json"
