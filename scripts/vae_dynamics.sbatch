#!/bin/bash
#SBATCH --job-name=vae-dynamics-walker
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --chdir=/home/rickybao/projects/dreamerv3
#SBATCH --output=/home/rickybao/projects/dreamerv3/logs/%x-%A_%a.out
#SBATCH --error=/home/rickybao/projects/dreamerv3/logs/%x-%A_%a.err

# Add the VAE-latent dynamics baseline to an already-extracted run and re-probe.
# Trains an action-conditioned forward model in the frozen VAE's latent space
# (probing.vae_dynamics), then re-runs the probing battery so RSSM imag{k} and
# VAE-dynamics vae_imag{k} are scored side by side. Requires features.npz +
# traj.npz from probe_pipeline.sbatch / probing.features.

set -eo pipefail
SEED=${SLURM_ARRAY_TASK_ID}

echo "=============================="
echo "VAE-dynamics baseline | seed ${SEED} | job ${SLURM_JOB_ID} | $(date)"
echo "=============================="

PROJECT_DIR="/home/rickybao/projects/dreamerv3"
ENV_PATH="/scratch/midway3/${USER}/conda_envs/dreamerv3"
PYTHON="${ENV_PATH}/bin/python"
RUN_ROOT="/scratch/midway3/${USER}/dreamerv3_runs"
OUT="${RUN_ROOT}/probing_walker_walk_seed${SEED}"               # scratch: heavy .npz
RESULTS="${PROJECT_DIR}/results/probing_walker_walk_seed${SEED}"  # repo: small results

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_DIR}"
mkdir -p "${RESULTS}"
[ -x "${PYTHON}" ]            || { echo "ERROR: missing python"; exit 1; }
[ -f "${OUT}/features.npz" ] || { echo "ERROR: no features.npz; run probe_pipeline first"; exit 1; }
[ -f "${OUT}/traj.npz" ]     || { echo "ERROR: no traj.npz; run probe_pipeline first"; exit 1; }

echo "--- training VAE-latent dynamics + open-loop rollout ---"
"${PYTHON}" -u -m probing.vae_dynamics \
  --features "${OUT}/features.npz" \
  --traj     "${OUT}/traj.npz" \
  --output   "${OUT}/features_vaedyn.npz" \
  --horizons 1 5 20 --test_frac 0.3 --steps 6000 --seed "${SEED}"

echo "--- re-probing with the baseline included ---"
"${PYTHON}" -u -m probing.probe \
  --traj     "${OUT}/traj.npz" \
  --features "${OUT}/features_vaedyn.npz" \
  --output   "${RESULTS}" \
  --horizons 0 1 5 20 --mlp_steps 2000

echo "VAE-dynamics seed ${SEED} done | $(date) | -> ${RESULTS}"
echo "After all seeds finish, rebuild figures: python result_images/build_summary_figures.py"
