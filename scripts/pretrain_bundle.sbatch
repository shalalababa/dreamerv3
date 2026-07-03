#!/bin/bash
#SBATCH --job-name=dv3_pretrain_bundle
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=32:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Phase 4 bundled launcher: run several reward-free pretraining jobs
# sequentially inside one Slurm allocation. This is for clusters with a strict
# submitted-job cap; it keeps one GPU busy while Slurm sees only one job.
#
# Required env:
#   RUN_ID   bundle id, e.g. pretrain_bundle_001
#   RUNLIST  TSV with columns: run_id task mode seed steps
# Optional:
#   SAVE_EVERY, FORCE
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

pretrain_done() {  # pretrain_done <RUN_ID>
  local status
  status="$(manifest_status "$1")"
  [ "$status" = "DONE" ] && return 0
  [ -n "$status" ] && return 1
  [ -f "$RUNROOT/$1/TRAINING_DONE" ] &&
    [ -f "$RUNROOT/$1/ckpt_snapshots/nearest.json" ]
}

BUNDLE_LOGDIR="$RUNROOT/_bundles/$RUN_ID"
mkdir -p "$BUNDLE_LOGDIR"
cp "$RUNLIST" "$BUNDLE_LOGDIR/runlist.tsv"

echo "[$(date)] bundle RUN_ID=$RUN_ID RUNLIST=$RUNLIST"
echo "[$(date)] bundle contents:"
sed 's/^/  /' "$RUNLIST"

while IFS=$'\t' read -r child_run_id task mode seed steps; do
  [ -n "${child_run_id:-}" ] || continue
  if [ -z "${task:-}" ] || [ -z "${mode:-}" ] || [ -z "${seed:-}" ]; then
    echo "ERROR: malformed runlist row: ${child_run_id:-} ${task:-} ${mode:-} ${seed:-} ${steps:-}"
    exit 2
  fi

  if [ "${FORCE:-0}" != "1" ] && pretrain_done "$child_run_id"; then
    echo "[$(date)] SKIP done: $child_run_id"
    continue
  fi

  echo "[$(date)] START child $child_run_id"
  RUN_ID="$child_run_id" TASK="$task" MODE="$mode" SEED="$seed" \
    STEPS="${steps:-${STEPS:-5e5}}" SAVE_EVERY="${SAVE_EVERY:-300}" \
    bash "$REPO/scripts/pretrain.sbatch"
  echo "[$(date)] DONE child $child_run_id"
done < "$RUNLIST"

echo "[$(date)] bundle $RUN_ID finished"
