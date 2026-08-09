#!/bin/bash
#SBATCH --job-name=uncf_raw
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-13
#SBATCH --output=%x_%A_%a.out
# ---------------------------------------------------------------------------
# Paper-5 raw-ΔH recovery pass (uncfield/raw_research.py) — descriptive,
# 9 Aug 2026. One array task per sweep job; re-runs search_exploits from
# each cell's bundled ensemble at the registered (y_mode, seed), writes
# per-cycle npz + raw-vs-adjusted census, and re-derives every member
# verdict from the model (verdict_match gate vs the stored summary —
# matched-device: sweeps were CPU, this is CPU).
#
# PREREQ: the sweep dirs must still hold ensemble.pkl on this machine
# (they were written in place by the original wave; verify with
#   ls local_results/uncfield/sweeps/*/ensemble.pkl
# — ymode tasks additionally need the pilot2 anchor ensemble).
# Jobs are idempotent per job-json; resubmission free.
# GOTCHAS: clean shell (module purge); CPU-only jax forced below.
#
# COLLECT (after all 14 tasks), from the local machine:
#   ssh midway3 'cd <REPO> && ./scripts/bundle_manifest.sh generate local_results/uncfield/raw_research'
#   rsync -av midway3:<REPO>/local_results/uncfield/raw_research/ local_results/uncfield/raw_research/
#   ./scripts/bundle_manifest.sh verify local_results/uncfield/raw_research
#   cp local_results/uncfield/raw_research/MANIFEST.sha256 manifests/uncfield_raw_20260809.sha256
# Then locally: the raw tables fold into artifacts/nfi_review_response_20260809/
# (dated addendum; descriptive — no registered decision rules).
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

export JAX_PLATFORMS=cpu

JOBS=(ymode_s0 ymode_s1 ymode_s2 train1k train10k train30k
      dataseed1 dataseed2 hid32 hid128 hid128t10k hid128t30k
      dataseed3 dataseed4)
JOB="${JOBS[$SLURM_ARRAY_TASK_ID]}"

[ -f local_results/uncfield/pilot2/ensemble.pkl ] || {
    echo "FATAL: pilot2 anchor ensemble missing"; exit 1; }

echo "[$(date)] raw_research job=$JOB (array $SLURM_ARRAY_TASK_ID)"
python -m uncfield.raw_research --job "$JOB"
echo "[$(date)] DONE $JOB"
