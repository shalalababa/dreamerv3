#!/bin/bash
# Condition C1 -- DreamerV3 trained from scratch, no pretraining (the lower
# bound). research_procedure.md Phase 2.
#
# Usage:  sbatch --job-name=c1-<task>-seed<seed> scripts/c1_scratch.sbatch <task> <seed>
#   <task>  one of: stand | walk | run
#   <seed>  integer RNG seed (this study uses 1 and 2)
#
# Note: C1/walk is already covered by the existing 500K walker_walk seed1/seed2
# runs, so only `stand` and `run` need new C1 jobs. Runs to 500K, which yields
# both the adaptation-budget segment and a full-task-performance reference.

#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --chdir=/home/rickybao/projects/dreamerv3
#SBATCH --output=/home/rickybao/projects/dreamerv3/logs/%x-%j.out
#SBATCH --error=/home/rickybao/projects/dreamerv3/logs/%x-%j.err

set -eo pipefail

TASK="${1:?Usage: sbatch scripts/c1_scratch.sbatch <stand|walk|run> <seed>}"
SEED="${2:?Usage: sbatch scripts/c1_scratch.sbatch <stand|walk|run> <seed>}"
case "${TASK}" in
  stand|walk|run) ;;
  *) echo "ERROR: <task> must be stand, walk or run (got '${TASK}')"; exit 1 ;;
esac

echo "=============================="
echo "Slurm job started | Job ID: ${SLURM_JOB_ID} | Node: $(hostname)"
echo "Start time: $(date)"
echo "C1 from-scratch | Task: dmc_walker_${TASK} | Seed: ${SEED}"
echo "=============================="

PROJECT_DIR="/home/rickybao/projects/dreamerv3"
ENV_PATH="/scratch/midway3/${USER}/conda_envs/dreamerv3"
PYTHON="${ENV_PATH}/bin/python"
RUN_ROOT="/scratch/midway3/${USER}/dreamerv3_runs"
LOGDIR="${RUN_ROOT}/c1_${TASK}_seed${SEED}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

echo "PYTHON=${PYTHON}"
echo "LOGDIR=${LOGDIR}"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python executable not found: ${PYTHON}"; exit 1
fi
mkdir -p "${RUN_ROOT}"
cd "${PROJECT_DIR}"
if [ ! -f "dreamerv3/main.py" ]; then
  echo "ERROR: dreamerv3/main.py not found in $(pwd)"; exit 1
fi

echo "GPU check:"
nvidia-smi

echo "Starting C1 from-scratch: task=dmc_walker_${TASK} seed=${SEED}"
"${PYTHON}" -u dreamerv3/main.py \
  --logdir "${LOGDIR}" \
  --configs dmc_proprio \
  --task "dmc_walker_${TASK}" \
  --seed "${SEED}" \
  --env.dmc.image False \
  --env.dmc.proprio True \
  --run.steps 500000 \
  --run.train_ratio 1024 \
  --jax.prealloc False

echo "C1 run finished | End time: $(date)"
echo "Logdir: ${LOGDIR}"
