#!/bin/bash
# Representation analysis -- research_procedure.md section G.
#
# One job that, for the three pretrained conditions x two seeds:
#   1. collects a common held-out trajectory set (uniform random policy, with
#      MuJoCo physics logged) -- shared by every condition so representations
#      are compared on identical underlying states;
#   2. extracts frozen world-model features for each condition's checkpoint;
#   3. runs the probe battery (state / future-state / derived-quantity /
#      task-reward R^2) on those features;
#   4. estimates pretraining-replay state coverage;
#   5. synthesises the section-H table + correlations.
#
# Requires the six Phase-1 pretraining runs to have finished. Step 5
# (synthesise) can also be re-run later, once adaptation runs complete, via
#   python -m probing.synthesize --run_root <RUN_ROOT>
#
# Usage:  sbatch --job-name=probe-g scripts/probe_g.sbatch

#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/rickybao/projects/dreamerv3
#SBATCH --output=/home/rickybao/projects/dreamerv3/logs/%x-%j.out
#SBATCH --error=/home/rickybao/projects/dreamerv3/logs/%x-%j.err

set -eo pipefail

echo "Section-G representation analysis | Job ID: ${SLURM_JOB_ID} | $(date)"

PROJECT_DIR="/home/rickybao/projects/dreamerv3"
ENV_PATH="/scratch/midway3/${USER}/conda_envs/dreamerv3"
PYTHON="${ENV_PATH}/bin/python"
RUN_ROOT="/scratch/midway3/${USER}/dreamerv3_runs"
HELDOUT_DIR="${RUN_ROOT}/probe_heldout"
HELDOUT="${HELDOUT_DIR}/heldout_walker.npz"
FEAT_DIR="${RUN_ROOT}/probe_features"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PATH="${ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python executable not found: ${PYTHON}"; exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p "${HELDOUT_DIR}" "${FEAT_DIR}"

CONDS=(random p2e apt)
SEEDS=(1 2)

# Reference run dir, used only for its config.yaml when collecting the
# (checkpoint-free) random held-out set.
REF_RUN="${RUN_ROOT}/pretrain_p2e_seed1"

echo "GPU check:"
nvidia-smi

# --- 1. common held-out set (collect once, reuse) -------------------------
if [ -f "${HELDOUT}" ]; then
  echo "[1] held-out set already present: ${HELDOUT}"
else
  echo "[1] collecting common held-out set (uniform random policy)"
  "${PYTHON}" -m probing.collect \
    --run_logdir "${REF_RUN}" \
    --output "${HELDOUT}" \
    --task dmc_walker_walk \
    --random \
    --episodes 40 \
    --seed 0
fi

# --- 2-3. per-condition features + probes ---------------------------------
for COND in "${CONDS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN="${RUN_ROOT}/pretrain_${COND}_seed${SEED}"
    if [ ! -d "${RUN}/ckpt" ]; then
      echo "[2] SKIP ${COND} seed${SEED}: no checkpoint at ${RUN}/ckpt"
      continue
    fi
    FEAT="${FEAT_DIR}/features_${COND}_seed${SEED}.npz"
    echo "[2] features: ${COND} seed${SEED}"
    "${PYTHON}" -m probing.features \
      --traj "${HELDOUT}" \
      --run_logdir "${RUN}" \
      --output "${FEAT}"
    echo "[3] probes: ${COND} seed${SEED}"
    "${PYTHON}" -m probing.probe \
      --traj "${HELDOUT}" \
      --features "${FEAT}" \
      --output "${RUN_ROOT}/probes_${COND}_seed${SEED}"
  done
done

# --- 4. pretraining-replay coverage ---------------------------------------
echo "[4] coverage over pretraining replay buffers"
COV_ARGS=()
for COND in "${CONDS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    REPLAY="${RUN_ROOT}/pretrain_${COND}_seed${SEED}/replay"
    [ -d "${REPLAY}" ] && COV_ARGS+=("${COND}_seed${SEED}=${REPLAY}")
  done
done
if [ "${#COV_ARGS[@]}" -gt 0 ]; then
  "${PYTHON}" -m probing.coverage \
    --replay "${COV_ARGS[@]}" \
    --output "${RUN_ROOT}/coverage" \
    --time_buckets 6
else
  echo "    no replay buffers found; skipping coverage"
fi

# --- 5. section-H synthesis (AUC cells fill in as adaptation runs finish) --
echo "[5] synthesis"
"${PYTHON}" -m probing.synthesize --run_root "${RUN_ROOT}"

echo "Section-G analysis finished | $(date)"
echo "Outputs under ${RUN_ROOT}: probes_*/, coverage/, g_synthesis/"
