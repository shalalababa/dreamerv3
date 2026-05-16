#!/bin/bash
# Stage 1 -- reward-free world-model pretraining (research_procedure.md Phase 1).
#
# Usage:  sbatch --job-name=pretrain-<cond>-seed<seed> scripts/pretrain.sbatch <cond> <seed>
#   <cond>  one of: random (C2) | p2e (C3) | apt (C4)
#   <seed>  integer RNG seed (this study uses 1 and 2)
#
# Produces a world-model checkpoint + replay buffer under
#   /scratch/midway3/$USER/dreamerv3_runs/pretrain_<cond>_seed<seed>/
# which scripts/adapt.sbatch consumes for frozen-readout adaptation.

#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --chdir=/home/rickybao/projects/dreamerv3
#SBATCH --output=/home/rickybao/projects/dreamerv3/logs/%x-%j.out
#SBATCH --error=/home/rickybao/projects/dreamerv3/logs/%x-%j.err

# Do NOT use `set -u` here.
# It broke /etc/bashrc before through an unset BASHRCSOURCED variable.
set -eo pipefail

COND="${1:?Usage: sbatch scripts/pretrain.sbatch <random|p2e|apt> <seed>}"
SEED="${2:?Usage: sbatch scripts/pretrain.sbatch <random|p2e|apt> <seed>}"
case "${COND}" in
  random|p2e|apt) ;;
  *) echo "ERROR: <cond> must be random, p2e or apt (got '${COND}')"; exit 1 ;;
esac

echo "=============================="
echo "Slurm job started"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Condition: ${COND}  |  Seed: ${SEED}"
echo "=============================="

# Project and environment paths.
PROJECT_DIR="/home/rickybao/projects/dreamerv3"
ENV_PATH="/scratch/midway3/${USER}/conda_envs/dreamerv3"
PYTHON="${ENV_PATH}/bin/python"
RUN_ROOT="/scratch/midway3/${USER}/dreamerv3_runs"
LOGDIR="${RUN_ROOT}/pretrain_${COND}_seed${SEED}"

# Headless MuJoCo/DMC rendering.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Make Python logs appear immediately in .out.
export PYTHONUNBUFFERED=1
# Avoid JAX reserving all GPU memory immediately.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

echo "PYTHON=${PYTHON}"
echo "LOGDIR=${LOGDIR}"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python executable not found: ${PYTHON}"
  exit 1
fi
mkdir -p "${RUN_ROOT}"
cd "${PROJECT_DIR}"
if [ ! -f "dreamerv3/main.py" ]; then
  echo "ERROR: dreamerv3/main.py not found in $(pwd)"; exit 1
fi

echo "GPU check:"
nvidia-smi
echo "JAX check:"
"${PYTHON}" - <<'PY'
import jax
print("JAX devices:", jax.devices(), "| backend:", jax.default_backend())
PY

echo "Starting reward-free pretraining: condition=${COND} seed=${SEED}"
# 500K reward-free steps, held fixed for all DMC proprio runs (procedure A).
"${PYTHON}" -u dreamerv3/main.py \
  --logdir "${LOGDIR}" \
  --configs dmc_proprio "expl_${COND}" \
  --task dmc_walker_walk \
  --seed "${SEED}" \
  --env.dmc.image False \
  --env.dmc.proprio True \
  --run.steps 500000 \
  --run.train_ratio 1024 \
  --jax.prealloc False

echo "Pretraining finished"
echo "End time: $(date)"
echo "Checkpoint + replay: ${LOGDIR}"
