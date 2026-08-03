#!/bin/bash
#SBATCH --job-name=uncf_p1
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=%x_%A_%a.out
# ---------------------------------------------------------------------------
# P1: PRIME aleatoric-penalized carried-EIG re-score (uncfield/p1_rescore.py;
# claim-freeze memo §6.2 spec). One array task per pilot2 member: exact
# carried replay recording per-sense EIG + both aleatoric estimates over all
# 626 cycles (validation gate: reproduces the saved `carried` column
# bitwise). The λ/σ₀² grid + gate variant are post-hoc re-scores done by the
# registered read script — nothing to configure here.
# PREREQS: local_results/uncfield/pilot2/{ensemble.pkl,episodes.pkl,
#   results_m*.npz} on this machine (same anchor as the sweep queue).
# GOTCHAS: clean shell; CPU-only jax forced.
# COLLECT: rsync local_results/uncfield/p1_rescore/ (jsons, ~tens of MB) +
#   manifest per results-sync v2. READ gated on instrument review + prereg.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

export JAX_PLATFORMS=cpu

python -m uncfield.p1_rescore --member "$SLURM_ARRAY_TASK_ID"
echo "[$(date)] DONE member $SLURM_ARRAY_TASK_ID"
