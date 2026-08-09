#!/bin/bash
#SBATCH --job-name=uncf_fullcov
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Paper-5 L2 representation-adequacy control (uncfield/fullcov.py) —
# descriptive, 9 Aug 2026. Trains the 4-member full-covariance-head
# ensemble on the pilot2 episodes (skipped if ensemble_full.pkl already
# present, e.g. rsynced from the local run), scores the 46 pure-s4
# cycles, then runs the FULL 626-cycle search under both entropy views
# (marginal = inherited thresholds; joint log-det = descriptive).
#
# PREREQ: local_results/uncfield/pilot2/{episodes.pkl,ensemble.pkl,
#         results_m*.npz} on this machine.
# GOTCHAS: clean shell (module purge); CPU-only jax forced below.
#
# COLLECT (from the local machine):
#   ssh midway3 'cd <REPO> && ./scripts/bundle_manifest.sh generate local_results/uncfield/fullcov'
#   rsync -av midway3:<REPO>/local_results/uncfield/fullcov/ local_results/uncfield/fullcov/
#   ./scripts/bundle_manifest.sh verify local_results/uncfield/fullcov
#   cp local_results/uncfield/fullcov/MANIFEST.sha256 manifests/uncfield_fullcov_20260809.sha256
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

export JAX_PLATFORMS=cpu

[ -f local_results/uncfield/pilot2/episodes.pkl ] || {
    echo "FATAL: pilot2 episodes.pkl missing"; exit 1; }

python -m uncfield.fullcov --selfcheck
echo "[$(date)] fullcov run+full"
python -m uncfield.fullcov --run --full
echo "[$(date)] DONE"
