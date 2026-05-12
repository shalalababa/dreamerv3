#!/bin/bash
#SBATCH --job-name=dmc-walker-100k
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/rickybao/projects/dreamerv3
#SBATCH --output=/home/rickybao/projects/dreamerv3/logs/%x-%j.out
#SBATCH --error=/home/rickybao/projects/dreamerv3/logs/%x-%j.err

# Do NOT use `set -u` here.
# It broke /etc/bashrc before through an unset BASHRCSOURCED variable.
set -eo pipefail

echo "=============================="
echo "Slurm job started"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "User: ${USER}"
echo "Working directory: $(pwd)"
echo "=============================="

# Project and environment paths.
PROJECT_DIR="/home/rickybao/projects/dreamerv3"
ENV_PATH="/scratch/midway3/${USER}/conda_envs/dreamerv3"
PYTHON="${ENV_PATH}/bin/python"
RUN_ROOT="/scratch/midway3/${USER}/dreamerv3_runs"
LOGDIR="${RUN_ROOT}/dmc_walker_walk_100k_seed0"

# Headless MuJoCo/DMC rendering.
# Required because DreamerV3's DMC env renders log/image even for dmc_proprio.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Make Python logs appear immediately in .out.
export PYTHONUNBUFFERED=1

# Avoid JAX reserving all GPU memory immediately.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Helpful for conda/shared libraries even without full conda activation.
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

echo "Checking paths..."
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "ENV_PATH=${ENV_PATH}"
echo "PYTHON=${PYTHON}"
echo "LOGDIR=${LOGDIR}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}"
  exit 1
fi

if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python executable not found: ${PYTHON}"
  echo "Available envs under /scratch/midway3/${USER}/conda_envs:"
  ls -lh "/scratch/midway3/${USER}/conda_envs" || true
  exit 1
fi

mkdir -p "${RUN_ROOT}"
cd "${PROJECT_DIR}"

echo "Checking repository..."
if [ ! -f "dreamerv3/main.py" ]; then
  echo "ERROR: dreamerv3/main.py not found. Current directory is:"
  pwd
  ls -lh
  exit 1
fi

echo "Python executable:"
which python || true
"${PYTHON}" --version

echo "GPU check:"
nvidia-smi

echo "JAX check:"
"${PYTHON}" - <<'PY'
import jax
print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())
PY

echo "DMC EGL render check:"
"${PYTHON}" - <<'PY'
import os
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM"))

from dm_control import suite
env = suite.load("walker", "walk")
img = env.physics.render(64, 64, camera_id=0)
print("DMC render OK:", img.shape, img.dtype)
PY

echo "Starting DreamerV3 DMC proprio smoke run..."

"${PYTHON}" -u dreamerv3/main.py \
  --logdir "${LOGDIR}" \
  --configs dmc_proprio \
  --task dmc_walker_walk \
  --seed 0 \
  --env.dmc.image False \
  --env.dmc.proprio True \
  --run.steps 100000 \
  --run.envs 8 \
  --run.eval_envs 1 \
  --run.train_ratio 128 \
  --jax.prealloc False

echo "DreamerV3 run finished"
echo "End time: $(date)"
echo "Logdir: ${LOGDIR}"
