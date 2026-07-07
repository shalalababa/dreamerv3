#!/bin/bash
#SBATCH --job-name=dv3_axis1
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Phase 6 (Axis 1): one paired-seed cell member = offline WM retrain on a
# controlled buffer, then frozen-readout adaptation, in a single job.
#
# Stage 1  probing.offline_fit on the built side directory. UPDATES gradient
#          steps; dmc_proprio has train_ratio 1024 = exactly one 16x64 batch
#          per env step, so UPDATES=500000 equalizes the gradient-update count
#          with the 500K-step online pretrains (runbook Phase 6 item 2).
#          Idempotent: skipped when the WM ckpt already has a `done` marker.
# Stage 2  frozen-readout adapt from that ckpt (same config as adapt.sbatch).
#
# Required env:
#   RUN_ID  adapt run id, e.g. adapt_ax1q1s0_cup_seed1_ckpt500000
#   WM_RUN  offline-fit run id, e.g. ax1wm_cup_q1s0_seed1
#   TASK    dmc task
#   SEED    integer (the paired seed: same k across sides/quadrants)
#   REPLAY  built buffer side dir, e.g. $RUNROOT/axis1_cup/q1/side0
# Optional: UPDATES (default 500000), STEPS (default 1.25e5)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUN_ID:?set RUN_ID}"; : "${WM_RUN:?set WM_RUN}"; : "${TASK:?set TASK}"
: "${SEED:?set SEED}"; : "${REPLAY:?set REPLAY}"
UPDATES="${UPDATES:-500000}"
STEPS="${STEPS:-1.25e5}"
WMDIR="$RUNROOT/$WM_RUN"
LOGDIR="$RUNROOT/$RUN_ID"

[ -f "$(dirname "$REPLAY")/manifest.json" ] || {
  echo "ERROR: no manifest.json beside $REPLAY (run build_controlled_replay build first)"
  exit 2
}

latest_ckpt() {  # newest completed elements.Checkpoint dir under $1/ckpt
  find "$1/ckpt" -mindepth 2 -maxdepth 2 -name done -printf '%h\n' 2>/dev/null |
    sort | tail -1
}

# --- Stage 1: offline WM fit -----------------------------------------------
CKPT="$(latest_ckpt "$WMDIR" || true)"
if [ -n "$CKPT" ] && [ "${FORCE_WM:-0}" != "1" ]; then
  echo "[$(date)] offline WM already fit: $CKPT (skipping stage 1)"
else
  echo "[$(date)] offline_fit WM_RUN=$WM_RUN REPLAY=$REPLAY UPDATES=$UPDATES SEED=$SEED"
  manifest_append "$WM_RUN" offline_fit "$TASK" static "$SEED" "$UPDATES" "$REPLAY" "$WMDIR" RUNNING
  RC=0
  python -m probing.offline_fit \
    --static_replay "$REPLAY" --updates "$UPDATES" \
    --logdir "$WMDIR" --configs dmc_proprio \
    --task "$TASK" --seed "$SEED" \
    --env.dmc.render "${RENDER:-False}" || RC=$?
  STATUS=$([ "$RC" -eq 0 ] && echo DONE || echo FAILED)
  sed -i "s#^$WM_RUN,.*,RUNNING,#$WM_RUN,offline_fit,$TASK,static,$SEED,$UPDATES,$REPLAY,$WMDIR,$STATUS,#" "$MANIFEST" 2>/dev/null || true
  [ "$RC" -eq 0 ] || { echo "[$(date)] offline_fit FAILED rc=$RC"; exit "$RC"; }
  CKPT="$(latest_ckpt "$WMDIR")"
  [ -n "$CKPT" ] || { echo "ERROR: offline_fit succeeded but no ckpt with done marker under $WMDIR/ckpt"; exit 2; }
fi

# --- Stage 2: frozen-readout adapt (mirrors adapt.sbatch) -------------------
mkdir -p "$LOGDIR"
echo "[$(date)] adapt RUN_ID=$RUN_ID TASK=$TASK SEED=$SEED CKPT=$CKPT STEPS=$STEPS"
manifest_append "$RUN_ID" adapt "$TASK" frozen_readout "$SEED" "$STEPS" "$CKPT" "$LOGDIR" RUNNING

RC=0
python dreamerv3/main.py \
  --logdir "$LOGDIR" --configs dmc_proprio frozen_readout \
  --task "$TASK" --seed "$SEED" \
  --run.from_checkpoint "$CKPT" --run.steps "$STEPS" || RC=$?
STATUS=$([ "$RC" -eq 0 ] && echo DONE || echo FAILED)
echo "[$(date)] adapt $RUN_ID rc=$RC status=$STATUS"
if [ "$RC" -eq 0 ]; then
  touch "$LOGDIR/ADAPT_DONE"
fi
sed -i "s#^$RUN_ID,.*,RUNNING,#$RUN_ID,adapt,$TASK,frozen_readout,$SEED,$STEPS,$CKPT,$LOGDIR,$STATUS,#" "$MANIFEST" 2>/dev/null || true
exit "$RC"
