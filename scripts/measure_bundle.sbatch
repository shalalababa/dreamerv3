#!/bin/bash
#SBATCH --job-name=dv3_measure_bundle
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=17:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Phase 5 bundled launcher: run several driver-measurement jobs sequentially
# inside one Slurm allocation.
#
# Required env:
#   RUN_ID   bundle id, e.g. measure_bundle_20260706_120000_001
#   RUNLIST  TSV with columns: pre_run task ref_replay critic
# Optional:
#   FORCE, MILESTONES, EPISODES
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUN_ID:?set RUN_ID}"; : "${RUNLIST:?set RUNLIST}"
[ -f "$RUNLIST" ] || { echo "ERROR: missing RUNLIST=$RUNLIST"; exit 2; }

BUNDLE_LOGDIR="$RUNROOT/_bundles/$RUN_ID"
mkdir -p "$BUNDLE_LOGDIR"
cp "$RUNLIST" "$BUNDLE_LOGDIR/runlist.tsv"

echo "[$(date)] measure bundle RUN_ID=$RUN_ID RUNLIST=$RUNLIST"
echo "[$(date)] bundle contents:"
sed 's/^/  /' "$RUNLIST"

while IFS=$'\t' read -r pre_run task ref_replay critic; do
  [ -n "${pre_run:-}" ] || continue
  if [ -z "${task:-}" ] || [ -z "${ref_replay:-}" ]; then
    echo "ERROR: malformed runlist row: ${pre_run:-} ${task:-} ${ref_replay:-} ${critic:-}"
    exit 2
  fi

  if [ "${FORCE:-0}" != "1" ] && [ -f "$RUNROOT/$pre_run/measure/MEASURE_DONE" ]; then
    echo "[$(date)] SKIP measured: $pre_run"
    continue
  fi

  echo "[$(date)] START child measure $pre_run"
  PRE_RUN="$pre_run" TASK="$task" REF_REPLAY="$ref_replay" CRITIC="$critic" \
    bash "$REPO/scripts/measure.sbatch"
  echo "[$(date)] DONE child measure $pre_run"
done < "$RUNLIST"

echo "[$(date)] measure bundle $RUN_ID finished"
