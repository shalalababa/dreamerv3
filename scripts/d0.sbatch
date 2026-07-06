#!/bin/bash
#SBATCH --job-name=dv3_d0
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Gate D0 (EVPI note App. A): one short-budget TASK-MODE training run with
# the measurement instruments on (K=5 valens critic heads + task-mode disag
# ensemble; config d0_probe), at one distractor dose.
#
# PRE-REG DISCIPLINE: App. A must be frozen before the first E2 run launches
# (deadline 1 Oct 2026). Submit via scripts/submit_d0.sh, which enforces the
# D0_FROZEN guard.
#
# Required env (pass via `sbatch --export`):
#   RUN_ID   e.g. d0_cup_d8x1_seed1
#   TASK     dmc task, e.g. dmc_cup_catch
#   DOSE     "" (E1 dose zero) | d0_dose1 | d0_dose2 | d0_dose3
#   SEED     integer (the E1/E2 pairing key: same seed index across cells)
# Optional: STEPS (default 1e5 per App. A.8)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUN_ID:?set RUN_ID}"; : "${TASK:?set TASK}"; : "${SEED:?set SEED}"
DOSE="${DOSE:-}"
STEPS="${STEPS:-1e5}"
LOGDIR="$RUNROOT/$RUN_ID"
mkdir -p "$LOGDIR"
rm -f "$LOGDIR/TRAINING_DONE"
echo "[$(date)] d0 RUN_ID=$RUN_ID TASK=$TASK DOSE=${DOSE:-e1} SEED=$SEED STEPS=$STEPS -> $LOGDIR"
manifest_append "$RUN_ID" d0 "$TASK" "task${DOSE:+_$DOSE}" "$SEED" "$STEPS" final "$LOGDIR" RUNNING

# Task-mode run at the Phase-4 (dmc_proprio/size1m) config + d0_probe
# instruments; the dose config appends the OU distractor wrapper. render
# False as in pretrain.sbatch (proprio runs never consume the image).
TRAIN_RC=0
python dreamerv3/main.py \
  --logdir "$LOGDIR" --configs dmc_proprio d0_probe $DOSE \
  --task "$TASK" --seed "$SEED" \
  --env.dmc.render False \
  --run.steps "$STEPS" || TRAIN_RC=$?

touch "$LOGDIR/TRAINING_DONE"
STATUS=$([ "$TRAIN_RC" -eq 0 ] && echo DONE || echo FAILED)
echo "[$(date)] d0 $RUN_ID finished rc=$TRAIN_RC status=$STATUS"
sed -i "s#^$RUN_ID,.*,RUNNING,#$RUN_ID,d0,$TASK,task${DOSE:+_$DOSE},$SEED,$STEPS,final,$LOGDIR,$STATUS,#" "$MANIFEST" 2>/dev/null || true
exit "$TRAIN_RC"
