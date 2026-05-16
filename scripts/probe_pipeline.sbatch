#!/bin/bash
#SBATCH --job-name=probe-walker-walk
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

# End-to-end probing pipeline for one DreamerV3 seed:
#   1. collect held-out probe trajectories from the frozen policy,
#   2. extract features from the frozen RSSM and the frozen VAE,
#   3. run the probing battery.
# Requires the matching DreamerV3 checkpoint and VAE checkpoint to exist.

set -eo pipefail
SEED=${SLURM_ARRAY_TASK_ID}

echo "=============================="
echo "Probe pipeline | seed ${SEED} | job ${SLURM_JOB_ID} | $(date)"
echo "=============================="

PROJECT_DIR="/home/rickybao/projects/dreamerv3"
ENV_PATH="/scratch/midway3/${USER}/conda_envs/dreamerv3"
PYTHON="${ENV_PATH}/bin/python"
RUN_ROOT="/scratch/midway3/${USER}/dreamerv3_runs"
DV3_RUN="${RUN_ROOT}/dmc_proprio_walker_walk_seed${SEED}"
VAE_CKPT="${RUN_ROOT}/vae_walker_walk_seed${SEED}/vae.ckpt"
OUT="${RUN_ROOT}/probing_walker_walk_seed${SEED}"               # scratch: heavy .npz
RESULTS="${PROJECT_DIR}/results/probing_walker_walk_seed${SEED}"  # repo: small results

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_DIR}"
mkdir -p "${OUT}" "${RESULTS}"
[ -x "${PYTHON}" ]      || { echo "ERROR: missing python";   exit 1; }
[ -d "${DV3_RUN}/ckpt" ] || { echo "ERROR: no DreamerV3 ckpt"; exit 1; }
[ -f "${VAE_CKPT}" ]     || { echo "ERROR: no VAE ckpt; run vae_train first"; exit 1; }

echo "--- [1/3] collecting probe trajectories ---"
"${PYTHON}" -u -m probing.collect \
  --run_logdir "${DV3_RUN}" \
  --output     "${OUT}/traj.npz" \
  --episodes 40 --max_steps 1000 --mode eval --seed "${SEED}"

echo "--- [2/3] extracting features ---"
"${PYTHON}" -u -m probing.features \
  --traj       "${OUT}/traj.npz" \
  --run_logdir "${DV3_RUN}" \
  --vae_ckpt   "${VAE_CKPT}" \
  --output     "${OUT}/features.npz" \
  --horizons 1 5 20 --ep_batch 8 --seed "${SEED}"

echo "--- [3/3] running probing battery ---"
# Heavy traj.npz/features.npz stay on scratch (${OUT}); the small probe
# results are written into the repo so they are versioned with the code.
"${PYTHON}" -u -m probing.probe \
  --traj     "${OUT}/traj.npz" \
  --features "${OUT}/features.npz" \
  --output   "${RESULTS}" \
  --horizons 0 1 5 20 --mlp_steps 2000

echo "Probe pipeline seed ${SEED} done | $(date)"
echo "  scratch intermediates: ${OUT}"
echo "  repo results:          ${RESULTS}"
