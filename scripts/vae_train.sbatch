#!/bin/bash
#SBATCH --job-name=vae-walker-walk
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --chdir=/home/rickybao/projects/dreamerv3
#SBATCH --output=/home/rickybao/projects/dreamerv3/logs/%x-%A_%a.out
#SBATCH --error=/home/rickybao/projects/dreamerv3/logs/%x-%A_%a.err

# Train the static frame-level VAE on a DreamerV3 replay buffer.
# One array task per DreamerV3 seed (0,1,2); each VAE trains on the frames
# its matching RSSM world model saw.

set -eo pipefail
SEED=${SLURM_ARRAY_TASK_ID}

echo "=============================="
echo "VAE training | seed ${SEED} | job ${SLURM_JOB_ID} | $(date)"
echo "=============================="

PROJECT_DIR="/home/rickybao/projects/dreamerv3"
ENV_PATH="/scratch/midway3/${USER}/conda_envs/dreamerv3"
PYTHON="${ENV_PATH}/bin/python"
RUN_ROOT="/scratch/midway3/${USER}/dreamerv3_runs"
DV3_RUN="${RUN_ROOT}/dmc_proprio_walker_walk_seed${SEED}"
VAE_DIR="${RUN_ROOT}/vae_walker_walk_seed${SEED}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_DIR}"
[ -x "${PYTHON}" ] || { echo "ERROR: missing python ${PYTHON}"; exit 1; }
[ -d "${DV3_RUN}/replay" ] || { echo "ERROR: no replay at ${DV3_RUN}"; exit 1; }

"${PYTHON}" -u -m probing.train_vae \
  --replay_dirs "${DV3_RUN}" \
  --run_config  "${DV3_RUN}/config.yaml" \
  --logdir      "${VAE_DIR}" \
  --latent 128 \
  --beta 1.0 \
  --free_nats 0.0 \
  --steps 60000 \
  --batch_size 256 \
  --lr 3e-4 \
  --seed "${SEED}" \
  --dtype float32

# vae.ckpt stays on scratch (it is an input to probing.features); copy the
# small training curve and metadata into the repo so they are versioned.
RESULTS="${PROJECT_DIR}/results/probing_walker_walk_seed${SEED}"
mkdir -p "${RESULTS}"
cp "${VAE_DIR}/metrics.jsonl" "${RESULTS}/vae_metrics.jsonl"
cp "${VAE_DIR}/meta.json"     "${RESULTS}/vae_meta.json"

echo "VAE seed ${SEED} done | $(date)"
echo "  scratch checkpoint: ${VAE_DIR}"
echo "  repo training log:  ${RESULTS}"
