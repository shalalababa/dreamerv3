#!/bin/bash
#SBATCH --job-name=dv3_adapt_bundle
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=34:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Phase 5 bundled launcher: run several frozen-readout adaptation jobs
# sequentially inside one Slurm allocation. This is for clusters with strict
# submitted-job and concurrent-node caps.
#
# Required env:
#   RUN_ID   bundle id, e.g. adapt_bundle_001
#   RUNLIST  TSV with columns: run_id task seed ckpt steps est_minutes
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUN_ID:?set RUN_ID}"; : "${RUNLIST:?set RUNLIST}"
[ -f "$RUNLIST" ] || { echo "ERROR: missing RUNLIST=$RUNLIST"; exit 2; }

manifest_status() {  # manifest_status <RUN_ID>
  [ -f "$MANIFEST" ] || return 0
  awk -F, -v run_id="$1" '$1 == run_id {status=$9} END {print status}' "$MANIFEST"
}

adapt_done() {  # adapt_done <RUN_ID>
  local status
  status="$(manifest_status "$1")"
  [ "$status" = "DONE" ] && return 0
  [ -f "$RUNROOT/$1/ADAPT_DONE" ]
}

BUNDLE_LOGDIR="$RUNROOT/_bundles/$RUN_ID"
mkdir -p "$BUNDLE_LOGDIR"
cp "$RUNLIST" "$BUNDLE_LOGDIR/runlist.tsv"

echo "[$(date)] adapt bundle RUN_ID=$RUN_ID RUNLIST=$RUNLIST"
echo "[$(date)] bundle contents:"
sed 's/^/  /' "$RUNLIST"

while IFS=$'\t' read -r child_run_id task seed ckpt steps est_minutes; do
  [ -n "${child_run_id:-}" ] || continue
  if [ -z "${task:-}" ] || [ -z "${seed:-}" ] || [ -z "${ckpt:-}" ]; then
    echo "ERROR: malformed runlist row: ${child_run_id:-} ${task:-} ${seed:-} ${ckpt:-} ${steps:-} ${est_minutes:-}"
    exit 2
  fi

  if [ "${FORCE:-0}" != "1" ] && adapt_done "$child_run_id"; then
    echo "[$(date)] SKIP done: $child_run_id"
    continue
  fi

  echo "[$(date)] START child $child_run_id est_minutes=${est_minutes:-unknown}"
  RUN_ID="$child_run_id" TASK="$task" SEED="$seed" CKPT="$ckpt" \
    STEPS="${steps:-${STEPS:-1.25e5}}" \
    bash "$REPO/scripts/adapt.sbatch"
  echo "[$(date)] DONE child $child_run_id"
done < "$RUNLIST"

echo "[$(date)] adapt bundle $RUN_ID finished"
