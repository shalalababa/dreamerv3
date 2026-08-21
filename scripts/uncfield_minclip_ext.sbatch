#!/bin/bash
#SBATCH --job-name=minclipext
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=%x_%A_%a.out
# ---------------------------------------------------------------------------
# MIN-CLIP EXTENSION bank — PREREG_nfi_minclip_ext_20260821.md.
# One task = one (job, member) of minclip_rescore.JOBCONFIGS_EXT x 4
# (dataseed3/4 + hid128t10k/t30k = 16 distinct models): full exploit
# search (asserted to reproduce the frozen sweeps/<job>/summary.json
# counts) + licensed trace on EVERY cycle + the per-task json/npz to
# local_results/uncfield/minclip_ext/.
#
# PARALLEL to the main 0-43 array (scripts/uncfield_minclip.sbatch) —
# the two banks never share outputs, and collate_ext REFUSES on fewer
# than 16 task jsons (completion counter per the 7-Aug rule).
# WALLTIME: same per-task shape as the main bank (4-20 min measured;
# hid128 members sit at the slow end) -> 2h = >6x margin.
#
# Collect per results-sync v2: local_results/uncfield/minclip_ext/ ->
# bundle + manifest before the ONE collate_ext execution.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
echo "[$(date)] minclip-ext task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
python -m uncfield.minclip_rescore --task_ext "${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] DONE ext task ${SLURM_ARRAY_TASK_ID}"
{ ls local_results/uncfield/minclip_ext/*_m*.json 2>/dev/null | wc -l | \
  sed 's/^/completed ext task jsons so far: /'; } || true
