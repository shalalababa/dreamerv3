#!/bin/bash
# Fast pre-flight check for the exploration pipeline -- a short reward-free run
# that confirms the condition launches, checkpoints, and produces sane metrics
# (non-zero ensemble disagreement / intrinsic reward) before committing a GPU
# to the full 16h / 500K job. Inspect logs/ and the run's metrics.jsonl after.
#
# Usage:  sbatch --job-name=smoke-<cond> scripts/pretrain_smoke.sbatch <cond>
#   <cond>  one of: random (C2) | p2e (C3) | apt (C4)

#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --chdir=/home/rickybao/projects/dreamerv3
#SBATCH --output=/home/rickybao/projects/dreamerv3/logs/%x-%j.out
#SBATCH --error=/home/rickybao/projects/dreamerv3/logs/%x-%j.err

set -eo pipefail

COND="${1:?Usage: sbatch scripts/pretrain_smoke.sbatch <random|p2e|apt>}"
case "${COND}" in random|p2e|apt) ;; *)
  echo "ERROR: <cond> must be random, p2e or apt (got '${COND}')"; exit 1 ;; esac

echo "Smoke pretraining | Cond: ${COND} | Job ID: ${SLURM_JOB_ID} | $(date)"

PROJECT_DIR="/home/rickybao/projects/dreamerv3"
ENV_PATH="/scratch/midway3/${USER}/conda_envs/dreamerv3"
PYTHON="${ENV_PATH}/bin/python"
RUN_ROOT="/scratch/midway3/${USER}/dreamerv3_runs"
LOGDIR="${RUN_ROOT}/smoke_pretrain_${COND}_${SLURM_JOB_ID}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python executable not found: ${PYTHON}"; exit 1
fi
mkdir -p "${RUN_ROOT}"
cd "${PROJECT_DIR}"

echo "GPU check:"
nvidia-smi

echo "Starting 20K-step smoke run: condition=${COND}"
"${PYTHON}" -u dreamerv3/main.py \
  --logdir "${LOGDIR}" \
  --configs dmc_proprio "expl_${COND}" \
  --task dmc_walker_walk \
  --seed 1 \
  --env.dmc.image False \
  --env.dmc.proprio True \
  --run.steps 20000 \
  --run.train_ratio 1024 \
  --jax.prealloc False

echo "Smoke run finished | $(date) | Logdir: ${LOGDIR}"
echo "Check ${LOGDIR}/metrics.jsonl for train/expl/intr_rew_raw,"
echo "train/expl/intr_rew, train/expl/disag_replay_rew, and train/loss/disag."
