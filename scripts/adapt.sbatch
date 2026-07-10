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
if [ ! -x "${CONDA_ENV:-}/bin/python" ]; then
  CONDA_ENV="$(python - <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.executable).resolve().parents[1])
PY
)"
  export CONDA_ENV
fi
export PATH="$CONDA_ENV/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export CONDA_PREFIX="$CONDA_ENV"
export CONDA_DEFAULT_ENV="$(basename "$CONDA_ENV")"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHON="$CONDA_ENV/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python || true)"
[ -n "$PYTHON" ] || { echo "ERROR: no python found for CONDA_ENV=$CONDA_ENV"; exit 2; }
cd "$REPO"

echo "[$(date)] python=$PYTHON"
"$PYTHON" - <<'PY'
import elements
import jax
from dm_control import suite
print("env ok: elements, jax", jax.__version__, "devices", jax.devices(), flush=True)
PY

: "${RUN_ID:?set RUN_ID}"; : "${TASK:?set TASK}"; : "${SEED:?set SEED}"; : "${CKPT:?set CKPT}"
STEPS="${STEPS:-2.5e5}"
LOGDIR="$RUNROOT/$RUN_ID"
mkdir -p "$LOGDIR"
[ -f "$CKPT/done" ] || { echo "ERROR: $CKPT has no 'done' marker"; exit 2; }
echo "[$(date)] adapt RUN_ID=$RUN_ID TASK=$TASK SEED=$SEED CKPT=$CKPT STEPS=$STEPS"
manifest_append "$RUN_ID" adapt "$TASK" frozen_readout "$SEED" "$STEPS" "$CKPT" "$LOGDIR" RUNNING

RC=0
"$PYTHON" -u dreamerv3/main.py \
  --logdir "$LOGDIR" --configs dmc_proprio frozen_readout \
  --task "$TASK" --seed "$SEED" \
  --env.dmc.render "${RENDER:-False}" \
  --run.from_checkpoint "$CKPT" --run.steps "$STEPS" || RC=$?
STATUS=$([ "$RC" -eq 0 ] && echo DONE || echo FAILED)
echo "[$(date)] adapt $RUN_ID rc=$RC status=$STATUS"
if [ "$RC" -eq 0 ]; then
  touch "$LOGDIR/ADAPT_DONE"
fi
sed -i "s#^$RUN_ID,.*,RUNNING,#$RUN_ID,adapt,$TASK,frozen_readout,$SEED,$STEPS,$CKPT,$LOGDIR,$STATUS,#" "$MANIFEST" 2>/dev/null || true
exit "$RC"
