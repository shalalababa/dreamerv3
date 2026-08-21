#!/bin/bash
#SBATCH --job-name=minclip
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-43
#SBATCH --output=%x_%A_%a.out
# ---------------------------------------------------------------------------
# MIN-CLIP re-scoring array — PREREG_nfi_minclip_20260821.md.
# One task = one (job, member) of ratio_research.JOBCONFIGS x 4: full
# exploit search (asserted to reproduce the frozen ratio_research table)
# + licensed trace on EVERY cycle + the per-task json/npz.
#
# WALLTIME: measured 4-20 min/task end-to-end (review #29 finding 14,
# double-sourced); 2h = >6x margin per the 7-Aug realized-training rule.
# COMPLETION COUNTER (same rule): the collate step REFUSES on fewer
# than 44 task jsons, so silent truncation cannot enter the read; the
# grep line below is the human-visible counter.
#
# Collect per results-sync v2: local_results/uncfield/minclip/ ->
# bundle + manifest before the ONE collate execution.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
echo "[$(date)] minclip task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
python -m uncfield.minclip_rescore --task "${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] DONE task ${SLURM_ARRAY_TASK_ID}"
ls local_results/uncfield/minclip/*_m*.json 2>/dev/null | wc -l | \
  sed 's/^/completed task jsons so far: /'
