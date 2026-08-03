#!/bin/bash
#SBATCH --job-name=uncf_ratio
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-43
#SBATCH --output=%x_%A_%a.out
# ---------------------------------------------------------------------------
# Per-member self-consistency ratio re-search (uncfield/ratio_research.py) —
# sweep-review B2 repair: re-search each (job, member) with the job's exact
# (y_mode, seed), save the per-cycle table, trace the ratio on the top
# NEUTRAL BOTH-EXPLOIT cycle (fallback tiers recorded). 44 tasks = 11 job
# configs (pilot2 + ymode×3 + train×3 + dataseed×2 + hid×2) × 4 members,
# ~2-10 min each.
#
# PREREQS: the sweep ensembles must be on this machine —
#   local_results/uncfield/pilot2/ensemble.pkl (already there from the sweep
#   prereq) + local_results/uncfield/sweeps/*/ensemble.pkl (written by the
#   sweep jobs; still on scratch if you ran the queue here).
# GOTCHAS: submit from a clean shell (module purge). CPU-only jax forced.
# COLLECT: rsync local_results/uncfield/ratio_research/ back (jsons + npz,
#   ~few MB) + manifest per results-sync v2.
# NOTE: the READ of these outputs is gated on the instrument review of
#   ratio_research.py (standing policy).
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

export JAX_PLATFORMS=cpu

python -m uncfield.ratio_research --task "$SLURM_ARRAY_TASK_ID"
echo "[$(date)] DONE task $SLURM_ARRAY_TASK_ID"
