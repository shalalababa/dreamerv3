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
export PATH="$CONDA_ENV/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export CONDA_PREFIX="$CONDA_ENV"
export CONDA_DEFAULT_ENV="$(basename "$CONDA_ENV")"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHON="$CONDA_ENV/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python || true)"
[ -n "$PYTHON" ] || { echo "ERROR: no python found for CONDA_ENV=$CONDA_ENV"; exit 2; }
cd "$REPO"

: "${RUN_ID:?set RUN_ID}"; : "${WM_RUN:?set WM_RUN}"; : "${TASK:?set TASK}"
: "${SEED:?set SEED}"; : "${REPLAY:?set REPLAY}"
UPDATES="${UPDATES:-500000}"
STEPS="${STEPS:-1.25e5}"
# Scaling pilot (PREREG_scaling_pilot): optional model-size config applied
# to BOTH stages -- the adapt stage must match the checkpoint's size or
# loading fails, which is the built-in guard against mixed-size cells.
# Empty AXIS1_SIZE keeps the historical default (size1m via dmc_proprio).
case "${AXIS1_SIZE:-}" in
  ''|size1m|size12m|size25m|size50m|size100m|size200m|size400m) ;;
  *) echo "ERROR: unknown AXIS1_SIZE=${AXIS1_SIZE} (size1m..size400m or empty)"; exit 2 ;;
esac
# Pixel study (Plan_PixelReplication): base config applied to BOTH stages
# (like AXIS1_SIZE, the adapt stage must match the checkpoint's spaces --
# a mismatch fails at load, which is the built-in guard).
case "${AXIS1_BASE_CONFIG:-}" in
  ''|dmc_proprio) ;;
  pixel_wm) RENDER=True ;;  # pixel obs require rendering (dmc.py asserts)
  *) echo "ERROR: unknown AXIS1_BASE_CONFIG=${AXIS1_BASE_CONFIG} (dmc_proprio|pixel_wm or empty)"; exit 2 ;;
esac
WMDIR="$RUNROOT/$WM_RUN"
LOGDIR="$RUNROOT/$RUN_ID"
mkdir -p "$LOGDIR"

stage() {
  printf '%s\n' "$1" > "$LOGDIR/AXIS1_STAGE"
}
stage "script_started"
echo "[$(date)] axis1 child starting RUN_ID=$RUN_ID WM_RUN=$WM_RUN job=${SLURM_JOB_ID:-NA} host=$(hostname)"
echo "[$(date)] env prepared: python=$PYTHON"
"$PYTHON" -u - <<'PY'
import sys
print("python_executable=" + sys.executable)
PY

[ -f "$(dirname "$REPLAY")/manifest.json" ] || {
  echo "ERROR: no manifest.json beside $REPLAY (run build_controlled_replay build first)"
  exit 2
}

latest_ckpt() {  # newest completed elements.Checkpoint dir under $1/ckpt
  find "$1/ckpt" -mindepth 2 -maxdepth 2 -name done -printf '%h\n' 2>/dev/null |
    sort | tail -1
}

ckpt_update() {  # parse elements.Checkpoint folder suffix: <timestamp>-000000500000
  local ckpt="${1:-}"
  [ -n "$ckpt" ] || { echo 0; return; }
  local base suffix
  base="$(basename "$ckpt")"
  suffix="${base##*-}"
  case "$suffix" in
    ''|*[!0-9]*) echo 0 ;;
    *) echo "$suffix" ;;
  esac
}

progress_update() {
  local progress="$1/OFFLINE_FIT_PROGRESS"
  [ -f "$progress" ] || { echo 0; return; }
  awk -F= '$1=="update"{v=$2} END{print v+0}' "$progress" 2>/dev/null || echo 0
}

offline_fit_complete() {
  local dir="$1" updates="$2" ckpt ckpt_step prog_step
  ckpt="$(latest_ckpt "$dir" || true)"
  [ -n "$ckpt" ] || return 1
  ckpt_step="$(ckpt_update "$ckpt")"
  prog_step="$(progress_update "$dir")"
  [ "$ckpt_step" -ge "$updates" ] || [ "$prog_step" -ge "$updates" ]
}

# --- Stage 1: offline WM fit -----------------------------------------------
CKPT="$(latest_ckpt "$WMDIR" || true)"
if [ "${FORCE_WM:-0}" != "1" ] && offline_fit_complete "$WMDIR" "$UPDATES"; then
  stage "offline_fit_done"
  echo "[$(date)] offline WM already fit: $CKPT (skipping stage 1)"
else
  stage "offline_fit_running"
  if [ -n "$CKPT" ] && [ "${FORCE_WM:-0}" != "1" ]; then
    echo "[$(date)] resuming partial offline WM from $CKPT update=$(ckpt_update "$CKPT") progress=$(progress_update "$WMDIR")/$UPDATES"
  fi
  echo "[$(date)] offline_fit WM_RUN=$WM_RUN REPLAY=$REPLAY UPDATES=$UPDATES SEED=$SEED"
  manifest_append "$WM_RUN" offline_fit "$TASK" static "$SEED" "$UPDATES" "$REPLAY" "$WMDIR" RUNNING
  RC=0
  # PREREG_axis1_corrective_20260711: expl mode must be explicit. 'apt' is
  # the reward-free fitting objective (matches the Phase-4 online pretrains'
  # replay losses; task reward enters no loss); 'task' reproduces the 10-Jul
  # reward-aware arm (discovered deviation, analysis/DEVIATIONS.md).
  # AXIS1_ARM: P3 gradient-path factorial arm code
  # (PREREG_p3_gradient_path_20260714.md). Flags are derived HERE from the
  # registered code so the exact objective travels as one token; the saved
  # config.yaml records the resulting reward_grad/repval_grad for the audit.
  : "${AXIS1_EXPL_MODE:?set AXIS1_EXPL_MODE=apt (reward-free, corrective protocol) or task (reward-aware arm)}"
  case "${AXIS1_ARM:-full}" in
    full) AXIS1_FIT_FLAGS="" ;;
    sgb)  AXIS1_FIT_FLAGS="--agent.reward_grad False --agent.repval_grad False" ;;
    rgo)  AXIS1_FIT_FLAGS="--agent.repval_grad False" ;;
    vgo)  AXIS1_FIT_FLAGS="--agent.reward_grad False" ;;
    # rde (PREREG_rde_pixel_20260802): reconstruction detached from the
    # trunk — the complement of the pe arm (full head gradients, no recon
    # bid). The saved config.yaml records recon_grad for the audit sweep.
    rde)  AXIS1_FIT_FLAGS="--agent.recon_grad False" ;;
    *) echo "ERROR: unknown AXIS1_ARM=${AXIS1_ARM} (full|sgb|rgo|vgo|rde)"; exit 2 ;;
  esac
  if [ "${AXIS1_ARM:-full}" != full ] && [ "$AXIS1_EXPL_MODE" != task ]; then
    echo "ERROR: factorial arm ${AXIS1_ARM} requires AXIS1_EXPL_MODE=task"; exit 2
  fi
  # Pretrained-encoder arm (PREREG_pe_pixel_20260730): AXIS1_INIT_WM names a
  # donor fit run; the fit loads ONLY '^enc/' from its latest ckpt and
  # freezes the encoder (agent.frozen_enc). Coupled by design — a frozen
  # RANDOM encoder control would pass the flags explicitly instead.
  if [ -n "${AXIS1_INIT_WM:-}" ]; then
    INIT_CKPT="$(latest_ckpt "$RUNROOT/$AXIS1_INIT_WM" || true)"
    [ -n "$INIT_CKPT" ] || { echo "ERROR: no done ckpt under $RUNROOT/$AXIS1_INIT_WM/ckpt"; exit 2; }
    AXIS1_FIT_FLAGS="${AXIS1_FIT_FLAGS:-} --run.from_checkpoint $INIT_CKPT --run.from_checkpoint_regex ${AXIS1_INIT_REGEX:-^enc/} --agent.frozen_enc True"
    echo "[$(date)] partial-init: enc from $INIT_CKPT (regex ${AXIS1_INIT_REGEX:-^enc/}), frozen_enc True"
  fi
  # s x w_r wave (PREREG_swave_theory_correction_20260808 constraint 1):
  # AXIS1_WR overrides the reward-head loss scale (pure fit-pressure
  # knob). Empty = default path byte-identical.
  if [ -n "${AXIS1_WR:-}" ]; then
    [ "$AXIS1_EXPL_MODE" = task ] || { echo "ERROR: AXIS1_WR requires AXIS1_EXPL_MODE=task"; exit 2; }
    AXIS1_FIT_FLAGS="${AXIS1_FIT_FLAGS:-} --agent.loss_scales.rew ${AXIS1_WR}"
    echo "[$(date)] s x w_r: agent.loss_scales.rew=${AXIS1_WR}"
  fi
  echo "[$(date)] fit flags: --agent.expl.mode $AXIS1_EXPL_MODE ${AXIS1_FIT_FLAGS:-} (arm=${AXIS1_ARM:-full} size=${AXIS1_SIZE:-default} base=${AXIS1_BASE_CONFIG:-dmc_proprio})"
  "$PYTHON" -u -m probing.offline_fit \
    --static_replay "$REPLAY" --updates "$UPDATES" \
    --save_every_updates "${AXIS1_SAVE_EVERY_UPDATES:-50000}" \
    --logdir "$WMDIR" --configs "${AXIS1_BASE_CONFIG:-dmc_proprio}" ${AXIS1_SIZE:+$AXIS1_SIZE} \
    --task "$TASK" --seed "$SEED" \
    --agent.expl.mode "$AXIS1_EXPL_MODE" ${AXIS1_FIT_FLAGS:-} \
    --env.dmc.render "${RENDER:-False}" || RC=$?
  STATUS=$([ "$RC" -eq 0 ] && echo DONE || echo FAILED)
  sed -i "s#^$WM_RUN,.*,RUNNING,#$WM_RUN,offline_fit,$TASK,static,$SEED,$UPDATES,$REPLAY,$WMDIR,$STATUS,#" "$MANIFEST" 2>/dev/null || true
  [ "$RC" -eq 0 ] || { stage "offline_fit_failed"; echo "[$(date)] offline_fit FAILED rc=$RC"; exit "$RC"; }
  CKPT="$(latest_ckpt "$WMDIR")"
  [ -n "$CKPT" ] || { echo "ERROR: offline_fit succeeded but no ckpt with done marker under $WMDIR/ckpt"; exit 2; }
  stage "offline_fit_done"
fi

# --- Stage 2: adapt (mirrors adapt.sbatch) ----------------------------------
# AXIS1_ADAPT_CONFIG selects the adapt config(s); default frozen_readout
# preserves every existing wave. Multi-token values are allowed (e.g.
# "orth_spin frozen_readout"); the manifest joins tokens with '+'.
ADAPT_CFG="${AXIS1_ADAPT_CONFIG:-frozen_readout}"
ADAPT_CFG_TAG="${ADAPT_CFG// /+}"
stage "adapt_running"
echo "[$(date)] adapt RUN_ID=$RUN_ID TASK=$TASK SEED=$SEED CKPT=$CKPT STEPS=$STEPS CFG=$ADAPT_CFG_TAG"
manifest_append "$RUN_ID" adapt "$TASK" "$ADAPT_CFG_TAG" "$SEED" "$STEPS" "$CKPT" "$LOGDIR" RUNNING

RC=0
"$PYTHON" -u dreamerv3/main.py \
  --logdir "$LOGDIR" --configs "${AXIS1_BASE_CONFIG:-dmc_proprio}" ${AXIS1_SIZE:+$AXIS1_SIZE} $ADAPT_CFG \
  --task "$TASK" --seed "$SEED" \
  --env.dmc.render "${RENDER:-False}" \
  --run.from_checkpoint "$CKPT" --run.steps "$STEPS" || RC=$?
STATUS=$([ "$RC" -eq 0 ] && echo DONE || echo FAILED)
echo "[$(date)] adapt $RUN_ID rc=$RC status=$STATUS"
if [ "$RC" -eq 0 ]; then
  touch "$LOGDIR/ADAPT_DONE"
  stage "adapt_done"
else
  stage "adapt_failed"
fi
sed -i "s#^$RUN_ID,.*,RUNNING,#$RUN_ID,adapt,$TASK,$ADAPT_CFG_TAG,$SEED,$STEPS,$CKPT,$LOGDIR,$STATUS,#" "$MANIFEST" 2>/dev/null || true
exit "$RC"
