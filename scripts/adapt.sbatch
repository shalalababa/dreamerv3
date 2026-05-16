#!/bin/bash
# Stage 2 -- frozen-readout adaptation (research_procedure.md Phase 2 / 3).
#
# Loads a pretrained world model (encoder + RSSM + decoder), freezes it, and
# trains only a fresh task head (actor + critic + reward predictor) on the
# downstream task reward.
#
# Usage:  sbatch --job-name=adapt-<cond>-<task>-seed<seed> \
#                scripts/adapt.sbatch <cond> <task> <seed>
#   <cond>  one of: random (C2) | p2e (C3) | apt (C4)
#   <task>  one of: stand | walk | run
#   <seed>  integer RNG seed -- must match the pretraining run's seed
#
# Requires scripts/pretrain.sbatch <cond> <seed> to have completed first.

#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/rickybao/projects/dreamerv3
#SBATCH --output=/home/rickybao/projects/dreamerv3/logs/%x-%j.out
#SBATCH --error=/home/rickybao/projects/dreamerv3/logs/%x-%j.err

set -eo pipefail

COND="${1:?Usage: sbatch scripts/adapt.sbatch <random|p2e|apt> <stand|walk|run> <seed>}"
TASK="${2:?Usage: sbatch scripts/adapt.sbatch <random|p2e|apt> <stand|walk|run> <seed>}"
SEED="${3:?Usage: sbatch scripts/adapt.sbatch <random|p2e|apt> <stand|walk|run> <seed>}"
case "${COND}" in random|p2e|apt) ;; *)
  echo "ERROR: <cond> must be random, p2e or apt (got '${COND}')"; exit 1 ;; esac
case "${TASK}" in stand|walk|run) ;; *)
  echo "ERROR: <task> must be stand, walk or run (got '${TASK}')"; exit 1 ;; esac

echo "=============================="
echo "Slurm job started | Job ID: ${SLURM_JOB_ID} | Node: $(hostname)"
echo "Start time: $(date)"
echo "Frozen-readout adaptation | Cond: ${COND} | Task: ${TASK} | Seed: ${SEED}"
echo "=============================="

PROJECT_DIR="/home/rickybao/projects/dreamerv3"
ENV_PATH="/scratch/midway3/${USER}/conda_envs/dreamerv3"
PYTHON="${ENV_PATH}/bin/python"
RUN_ROOT="/scratch/midway3/${USER}/dreamerv3_runs"
LOGDIR="${RUN_ROOT}/adapt_${COND}_${TASK}_seed${SEED}"

# Resolve the pretrained world-model checkpoint (the latest snapshot folder).
PRETRAIN_DIR="${RUN_ROOT}/pretrain_${COND}_seed${SEED}"
CKPT_DIR="${PRETRAIN_DIR}/ckpt"
if [ ! -f "${CKPT_DIR}/latest" ]; then
  echo "ERROR: no pretraining checkpoint at ${CKPT_DIR}"
  echo "       run scripts/pretrain.sbatch ${COND} ${SEED} first."
  exit 1
fi
FROM_CKPT="${CKPT_DIR}/$(cat "${CKPT_DIR}/latest")"
if [ ! -f "${FROM_CKPT}/done" ]; then
  echo "ERROR: pretraining snapshot incomplete: ${FROM_CKPT}"
  exit 1
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

echo "PYTHON=${PYTHON}"
echo "LOGDIR=${LOGDIR}"
echo "Pretrained world model: ${FROM_CKPT}"
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

# Adaptation budget A = 250K steps (half the 500K pretraining budget).
# The `frozen_readout` config freezes enc+RSSM+decoder and restricts the
# checkpoint load to `^(enc|dyn|dec)/`, so the task head trains from scratch.
echo "Starting frozen-readout adaptation"
"${PYTHON}" -u dreamerv3/main.py \
  --logdir "${LOGDIR}" \
  --configs dmc_proprio frozen_readout \
  --task "dmc_walker_${TASK}" \
  --seed "${SEED}" \
  --env.dmc.image False \
  --env.dmc.proprio True \
  --run.steps 250000 \
  --run.train_ratio 1024 \
  --run.from_checkpoint "${FROM_CKPT}" \
  --jax.prealloc False

echo "Adaptation finished | End time: $(date)"
echo "Logdir: ${LOGDIR}"
