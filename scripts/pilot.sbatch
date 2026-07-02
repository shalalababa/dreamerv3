#!/bin/bash
#SBATCH --job-name=dv3_pilot
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Phase 3 (Gate 0): short pilot collection to populate the (coverage, occupancy)
# feasibility plot.  One buffer per policy family; the replay/ dir it writes is
# what probing/gate0.py consumes.
#
#   MODE = expl_p2e | expl_apt | expl_random   -> reward-free explorer
#   MODE = goal                                -> reward-ON run (task reward),
#          whose learned policy visits the high-occupancy corner. This is the
#          goal-reaching policy Gate 0 needs; without it that quadrant is empty.
#
# Required env: RUN_ID TASK MODE SEED   Optional: STEPS (default 1e5)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUN_ID:?set RUN_ID}"; : "${TASK:?set TASK}"; : "${MODE:?set MODE}"; : "${SEED:?set SEED}"
STEPS="${STEPS:-1e5}"
LOGDIR="$RUNROOT/$RUN_ID"
mkdir -p "$LOGDIR"

# 'goal' = reward-on (default expl.mode=task); explorers add the expl_* block.
if [ "$MODE" = "goal" ]; then CONFIGS="dmc_proprio"; else CONFIGS="dmc_proprio $MODE"; fi
echo "[$(date)] pilot RUN_ID=$RUN_ID TASK=$TASK MODE=$MODE SEED=$SEED STEPS=$STEPS"
manifest_append "$RUN_ID" pilot "$TASK" "$MODE" "$SEED" "$STEPS" replay_kept "$LOGDIR" RUNNING

# render=False: proprio pilots don't use the image; skip render overhead (and a
# local WSL2 EGL/CUDA crash). save_every small so the replay flushes to disk
# (gate0 reads replay/); short runs otherwise lose their in-memory replay.
RC=0
python dreamerv3/main.py \
  --logdir "$LOGDIR" --configs $CONFIGS \
  --task "$TASK" --seed "$SEED" --run.steps "$STEPS" \
  --env.dmc.render "${RENDER:-False}" --run.save_every "${SAVE_EVERY:-120}" || RC=$?
STATUS=$([ "$RC" -eq 0 ] && echo DONE || echo FAILED)
echo "[$(date)] pilot $RUN_ID rc=$RC status=$STATUS  (replay at $LOGDIR/replay)"
exit "$RC"
