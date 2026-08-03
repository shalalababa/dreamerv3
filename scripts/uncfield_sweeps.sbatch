#!/bin/bash
#SBATCH --job-name=uncf_sweeps
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=%x_%A_%a.out
# ---------------------------------------------------------------------------
# Paper-5 NFI sensitivity sweep queue (uncfield/sweeps.py) as a CPU job
# array — one array task per sweep job. Moved off the local machine 2 Aug
# (DRAM pressure); jobs are idempotent (a job whose summary.json exists is
# skipped), so resubmission after any failure is free.
#
# PREREQS (once, from the local machine):
#   rsync -av local_results/uncfield/pilot2/{episodes.pkl,ensemble.pkl,config.json} \
#         midway3:<REPO>/local_results/uncfield/pilot2/
#   rsync -av local_results/uncfield/sweeps/ymode_s0/ \
#         midway3:<REPO>/local_results/uncfield/sweeps/ymode_s0/   # done locally; guard skips it
#   Optionally first: ln -sfn /scratch/midway3/$USER/uncfield_local <REPO>/local_results
#   (dataseed jobs write ~73M episodes each; keep them out of home quota)
#
# GOTCHAS: submit from a clean shell (module purge) — --export=ALL module
# leakage segfaults jax (memory, 24 Jul). Partition: caslake assumed; verify
# with `sinfo` on first submit. CPU-only jax is forced below.
#
# COLLECT (after all 8 tasks): from the local machine —
#   ssh midway3 'cd <REPO> && find local_results/uncfield/sweeps -name episodes.pkl -delete \
#     && ./scripts/bundle_manifest.sh generate local_results/uncfield/sweeps'
#   rsync -av midway3:<REPO>/local_results/uncfield/sweeps/ local_results/uncfield/sweeps/
#   ./scripts/bundle_manifest.sh verify local_results/uncfield/sweeps
#   cp local_results/uncfield/sweeps/MANIFEST.sha256 manifests/uncfield_sweeps_20260802.sha256
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

export JAX_PLATFORMS=cpu

# indices 8-9 = capacity×training interaction arm (submit with --array=8-9;
# indices 0-7 are complete and skip via the summary.json guard anyway)
JOBS=(ymode train1k train10k train30k dataseed1 dataseed2 hid32 hid128
      hid128t10k hid128t30k)
JOB="${JOBS[$SLURM_ARRAY_TASK_ID]}"

[ -f local_results/uncfield/pilot2/episodes.pkl ] || {
    echo "FATAL: pilot2 anchor missing — rsync the anchor bundle first (see header)"; exit 1; }

echo "[$(date)] uncfield sweep job=$JOB (array $SLURM_ARRAY_TASK_ID)"
python -m uncfield.sweeps --jobs "$JOB"
echo "[$(date)] DONE $JOB"
