#!/bin/bash
#SBATCH --job-name=serefine
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=%x_%A_%a.out
# ---------------------------------------------------------------------------
# Wave A: policy-seeded refinement — PREREG_planner_refine_20260822.
# One task = ONE Stage-1 checkpoint (seeds 10-17), four arms on one
# GPU, the se_refine_probe (registered constants + refine_std 0.2).
# No training. RUNS_ROOT = the Stage-1 runs parent dir.
# SMOKE=1 (rev A-M7): --smoke to a SEPARATE output dir (#32 m2) —
# one registered GPU smoke (array index 0, seed 10) runs BEFORE the
# full array: the warm-start path has only ever run on CPU.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUNS_ROOT="${RUNS_ROOT:?set RUNS_ROOT to the Stage-1 runs parent dir}"
SEED=$((10 + SLURM_ARRAY_TASK_ID))
RUN_DIR="$RUNS_ROOT/se_cheetah_seed${SEED}"
EXTRA=""
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA="--smoke --output $RUN_DIR/se_refine_smoke"
fi
echo "[$(date)] se_refine seed ${SEED} on $(hostname) $EXTRA"
python -m uncfield.se_refine_probe --run_logdir "$RUN_DIR" \
  --platform cuda --seed 0 $EXTRA
echo "[$(date)] DONE seed ${SEED}"
