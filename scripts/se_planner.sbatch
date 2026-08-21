#!/bin/bash
#SBATCH --job-name=seplanner
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
# Planner-in-the-loop wave — PREREG_nfi_planner_20260821.md.
# One task = ONE Stage-1 checkpoint (seeds 10-17), ALL FIVE ARMS on
# ONE GPU (the panel = one device; 16-Aug rule). No training — the
# probe loads the frozen ckpt, reconstructs the env from the run's
# own config, and rolls 4x500 steps per arm with the CEM planner
# (N=256, K=32, I=3, H=12).
#
# SMOKE FIRST (registered): submit ONE task with SMOKE=1 before the
# wave — it runs the tiny end-to-end config on the GPU and prints
# per-arm timing; the wave submits only after the smoke gate.
#   SMOKE=1 sbatch --array=0 scripts/se_planner.sbatch
# WALLTIME: CPU smoke extrapolates to ~4h/ckpt worst case on CPU;
# GPU expected far under; 10h = wide margin. The probe prints per-arm
# elapsed (fit-counter analogue for a no-training instrument).
#
# RUNS_ROOT must point at the Stage-1 run dirs (se_cheetah_seed10..17
# with ckpt/ + config.yaml). Output: <run>/se_planner/{json,npz}.
# Collect per results-sync v2 before the ONE registered read.
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
# #32 m2: the smoke writes to its OWN output dir so it can never
# collide with the wave's registered <run>/se_planner outputs
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA="--smoke --output $RUN_DIR/se_planner_smoke"
fi
echo "[$(date)] se_planner seed ${SEED} on $(hostname) $EXTRA"
python -m uncfield.se_planner_probe --run_logdir "$RUN_DIR" \
  --platform cuda --seed 0 $EXTRA
echo "[$(date)] DONE seed ${SEED}"
