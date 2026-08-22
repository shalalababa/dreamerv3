#!/bin/bash
#SBATCH --job-name=seplannerloc
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
# Wave C: localized-channel planner wave —
# PREREG_planner_localized_20260822. One task = ONE A1 checkpoint
# (hetero seeds 44-47, flat 48-51), ALL FIVE ARMS on ONE GPU, the
# FROZEN se_planner_probe UNCHANGED (registered constants). No
# training. RUNS_ROOT must point at the A1 run dirs (config.yaml +
# ckpt present; replay not needed). Checkout >= the A2 freeze.
# SMOKE=1 (rev C-M7): --smoke to a SEPARATE output dir (#32 m2) —
# a registered 1-task hetero smoke (array index 0, seed 44) runs
# BEFORE the full array: the hetero env path (mod quadruple) has
# never been exercised by this probe.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUNS_ROOT="${RUNS_ROOT:?set RUNS_ROOT to the A1 runs parent dir}"
IDS=(se_avoid_het_s44 se_avoid_het_s45 se_avoid_het_s46 se_avoid_het_s47 \
     se_avoid_flat_s48 se_avoid_flat_s49 se_avoid_flat_s50 se_avoid_flat_s51)
RUN_DIR="$RUNS_ROOT/${IDS[$SLURM_ARRAY_TASK_ID]}"
EXTRA=""
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA="--smoke --output $RUN_DIR/se_planner_smoke"
fi
echo "[$(date)] se_planner_loc ${IDS[$SLURM_ARRAY_TASK_ID]} on $(hostname) $EXTRA"
python -m uncfield.se_planner_probe --run_logdir "$RUN_DIR" \
  --platform cuda --seed 0 $EXTRA
echo "[$(date)] DONE ${IDS[$SLURM_ARRAY_TASK_ID]}"
