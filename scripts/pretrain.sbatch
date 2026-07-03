#!/bin/bash
#SBATCH --job-name=dv3_pretrain
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Phase 4: reward-free pretraining WITH checkpoint retention.
# Runs the DreamerV3 trainer and, alongside it, the copy-out watcher that keeps
# every completed save (keyed by exact env step) so the Phase 5 dose-response is
# possible.  save_every is set small so milestones are well covered.
#
# Required env (pass via `sbatch --export`):
#   RUN_ID   run/dir name, e.g. pretrain_p2e_cup_seed1
#   TASK     dmc task, e.g. dmc_cup_catch
#   MODE     expl_p2e | expl_apt | expl_random
#   SEED     integer
# Optional: STEPS (default 5e5), SAVE_EVERY seconds (default 300)
# ---------------------------------------------------------------------------
set -euo pipefail
# Slurm spools the script, so source env.sh via REPO (set by submit_all.sh, or
# defaults to the directory sbatch was invoked from -- run it from the repo root).
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUN_ID:?set RUN_ID}"; : "${TASK:?set TASK}"; : "${MODE:?set MODE}"; : "${SEED:?set SEED}"
STEPS="${STEPS:-5e5}"; SAVE_EVERY="${SAVE_EVERY:-300}"
LOGDIR="$RUNROOT/$RUN_ID"
mkdir -p "$LOGDIR"
rm -f "$LOGDIR/TRAINING_DONE"
echo "[$(date)] pretrain RUN_ID=$RUN_ID TASK=$TASK MODE=$MODE SEED=$SEED STEPS=$STEPS -> $LOGDIR"
manifest_append "$RUN_ID" pretrain "$TASK" "$MODE" "$SEED" "$STEPS" retained "$LOGDIR" RUNNING

# Integer stop_step for the watcher (STEPS may be like 5e5).
STOP_STEP=$(python -c "print(int(float('$STEPS')))")

# Launch trainer. render=False: proprio runs never consume the image, so skip
# the per-step MuJoCo render (pure overhead here; also sidesteps a local WSL2
# EGL/CUDA crash). Set RENDER=True if you want eval videos.
python dreamerv3/main.py \
  --logdir "$LOGDIR" --configs dmc_proprio "$MODE" \
  --task "$TASK" --seed "$SEED" \
  --env.dmc.render "${RENDER:-False}" \
  --run.steps "$STEPS" --run.save_every "$SAVE_EVERY" &
TRAIN_PID=$!

# Launch retention watcher (stops on stop_file or when stop_step reached).
python -m probing.checkpoint_watcher \
  --run_logdir "$LOGDIR" --stop_step "$STOP_STEP" \
  --stop_file "$LOGDIR/TRAINING_DONE" --poll_seconds 60 &
WATCH_PID=$!

TRAIN_RC=0; wait "$TRAIN_PID" || TRAIN_RC=$?
touch "$LOGDIR/TRAINING_DONE"        # tell the watcher to do a final sweep
wait "$WATCH_PID" || true

# Record the milestone->nearest-snapshot mapping for Phase 5.
python -m probing.checkpoint_watcher --select --run_logdir "$LOGDIR" || true

STATUS=$([ "$TRAIN_RC" -eq 0 ] && echo DONE || echo FAILED)
echo "[$(date)] pretrain $RUN_ID finished rc=$TRAIN_RC status=$STATUS"
# Update the manifest status in place.
sed -i "s#^$RUN_ID,.*,RUNNING,#$RUN_ID,pretrain,$TASK,$MODE,$SEED,$STEPS,retained,$LOGDIR,$STATUS,#" "$MANIFEST" 2>/dev/null || true
exit "$TRAIN_RC"
