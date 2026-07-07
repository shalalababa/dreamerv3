#!/bin/bash
#SBATCH --job-name=dv3_adapt
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Phase 5: frozen-readout adaptation from a retained pretraining snapshot.
# Loads ONLY the world model (enc/dyn/dec) via the frozen_readout regex and
# adapts fresh actor/critic/reward heads on the task reward.
#
# Required env:
#   RUN_ID  e.g. adapt_p2e_cup_seed1_ckpt100000
#   TASK    dmc task
#   SEED    integer
#   CKPT    absolute path to a snapshot dir (ckpt_snapshots/step0000....),
#           e.g. from probing.checkpoint_watcher --select / nearest.json
# Optional: STEPS (default 2.5e5; use 1.25e5 for early-AUC dose-response points)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUN_ID:?set RUN_ID}"; : "${TASK:?set TASK}"; : "${SEED:?set SEED}"; : "${CKPT:?set CKPT}"
STEPS="${STEPS:-2.5e5}"
LOGDIR="$RUNROOT/$RUN_ID"
mkdir -p "$LOGDIR"
[ -f "$CKPT/done" ] || { echo "ERROR: $CKPT has no 'done' marker"; exit 2; }
echo "[$(date)] adapt RUN_ID=$RUN_ID TASK=$TASK SEED=$SEED CKPT=$CKPT STEPS=$STEPS"
manifest_append "$RUN_ID" adapt "$TASK" frozen_readout "$SEED" "$STEPS" "$CKPT" "$LOGDIR" RUNNING

RC=0
python dreamerv3/main.py \
  --logdir "$LOGDIR" --configs dmc_proprio frozen_readout \
  --task "$TASK" --seed "$SEED" \
  --env.dmc.render "${RENDER:-False}" \
  --jax.precompile "${JAX_PRECOMPILE:-False}" \
  --jax.prealloc "${JAX_PREALLOC:-False}" \
  --jax.gpuflags "${JAX_GPUFLAGS:-False}" \
  --jax.autotune "${JAX_AUTOTUNE:-0}" \
  --jax.deterministic "${JAX_DETERMINISTIC:-False}" \
  --run.from_checkpoint "$CKPT" --run.steps "$STEPS" || RC=$?
STATUS=$([ "$RC" -eq 0 ] && echo DONE || echo FAILED)
echo "[$(date)] adapt $RUN_ID rc=$RC status=$STATUS"
if [ "$RC" -eq 0 ]; then
  touch "$LOGDIR/ADAPT_DONE"
fi
sed -i "s#^$RUN_ID,.*,RUNNING,#$RUN_ID,adapt,$TASK,frozen_readout,$SEED,$STEPS,$CKPT,$LOGDIR,$STATUS,#" "$MANIFEST" 2>/dev/null || true
exit "$RC"
