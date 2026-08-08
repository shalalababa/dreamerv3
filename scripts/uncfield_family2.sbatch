#!/bin/bash
#SBATCH --job-name=uncf_fam2
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=%x_%A_%a.out
# ---------------------------------------------------------------------------
# Paper-5 NFI family-generality factorial (PREREG_nfi_family2_20260807).
# Submit ONLY after the prereg freeze-commit; the commit must carry the
# reviewed uncfield/{dcfield,dcwm,family2,family2_read}.py + planner/
# learnedwm/lgfield edits.
#
# Task 0 = lg_lstm (needs the pilot2 anchor bundle on the node, same
#          prereq as uncfield_sweeps.sbatch).
# Task 1 = dc_gru then dc_lstm SERIALLY in one task — both consume the
#          shared dc_episodes.pkl, collected once by whichever starts it;
#          serializing removes any concurrent-collect race by construction.
#
# STEP 0 (task-local, both tasks): the registered validity gate — the
# planted battery MUST pass at deploy scale ON THIS HOST before any cell
# trains. A failure HALTs the wave (instrument failure, not data).
#
# Jobs are idempotent (summary.json guard); resubmission is free.
# GOTCHAS: submit from a clean shell (module purge) — --export=ALL module
# leakage segfaults jax. CPU-only jax forced below. NO bitwise cross-device
# gates exist in this wave by design (P1 review lesson): stamps are git +
# thresholds only.
#
# COLLECT (after both tasks), from the local machine:
#   ssh midway3 'cd <REPO> && rm -f local_results/uncfield/family2/dc_episodes.pkl \
#     && ./scripts/bundle_manifest.sh generate local_results/uncfield/family2'
#   rsync -av midway3:<REPO>/local_results/uncfield/family2/ local_results/uncfield/family2/
#   ./scripts/bundle_manifest.sh verify local_results/uncfield/family2
#   cp local_results/uncfield/family2/MANIFEST.sha256 manifests/uncfield_family2_20260807.sha256
# (dc_episodes.pkl is regenerable from seed 0; its sha256 is recorded inside
#  both dc summaries, so deleting it pre-manifest loses no provenance.)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

export JAX_PLATFORMS=cpu

echo "[$(date)] family2 selfcheck gate (host $(hostname))"
python -m uncfield.family2 --selfcheck

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    [ -f local_results/uncfield/pilot2/episodes.pkl ] || {
        echo "FATAL: pilot2 anchor missing — rsync the anchor bundle first"; exit 1; }
    echo "[$(date)] family2 cell lg_lstm"
    python -m uncfield.family2 --cells lg_lstm
else
    echo "[$(date)] family2 cells dc_gru,dc_lstm (serial)"
    python -m uncfield.family2 --cells dc_gru,dc_lstm
fi
echo "[$(date)] DONE task $SLURM_ARRAY_TASK_ID"
