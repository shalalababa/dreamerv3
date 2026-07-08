#!/bin/bash
#SBATCH --job-name=dv3_axis1_bundle
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=33:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Phase 6 Axis-1 bundled launcher: run several offline-WM + frozen-readout
# jobs sequentially inside one Slurm allocation.
#
# Required env:
#   RUN_ID   bundle id, e.g. axis1_bundle_20260707_120000_001
#   RUNLIST  TSV columns:
#            run_id wm_run task seed replay updates steps axis paired_seed_set
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

echo "[$(date)] axis1 bundle RUN_ID=$RUN_ID RUNLIST=$RUNLIST"
echo "[$(date)] bundle contents:"
sed 's/^/  /' "$RUNLIST"

while IFS=$'\t' read -r child_run_id wm_run task seed replay updates steps axis paired_seed_set; do
  [ -n "${child_run_id:-}" ] || continue
  if [ -z "${wm_run:-}" ] || [ -z "${task:-}" ] || [ -z "${seed:-}" ] || [ -z "${replay:-}" ]; then
    echo "ERROR: malformed runlist row: ${child_run_id:-} ${wm_run:-} ${task:-} ${seed:-} ${replay:-}"
    exit 2
  fi

  if [ "${FORCE:-0}" != "1" ] && adapt_done "$child_run_id"; then
    echo "[$(date)] SKIP done: $child_run_id"
    continue
  fi

  CHILD_DIR="$RUNROOT/$child_run_id"
  mkdir -p "$CHILD_DIR"
  {
    echo "bundle=$RUN_ID"
    echo "started_at=$(date -Is)"
    echo "wm_run=$wm_run"
    echo "runlist=$RUNLIST"
  } > "$CHILD_DIR/BUNDLE_CHILD_STARTED"
  echo "[$(date)] START child $child_run_id WM_RUN=$wm_run"
  set +e
  RUN_ID="$child_run_id" WM_RUN="$wm_run" TASK="$task" SEED="$seed" \
    REPLAY="$replay" UPDATES="${updates:-${AXIS1_UPDATES:-500000}}" \
    STEPS="${steps:-${STEPS:-1.25e5}}" AXIS="${axis:-axis1}" \
    PAIRED_SEED_SET="${paired_seed_set:-}" \
    FORCE_WM="${FORCE_WM:-0}" RENDER="${RENDER:-False}" \
    bash "$REPO/scripts/axis1.sbatch"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    date -Is > "$CHILD_DIR/BUNDLE_CHILD_DONE"
    echo "[$(date)] DONE child $child_run_id"
  else
    {
      echo "bundle=$RUN_ID"
      echo "failed_at=$(date -Is)"
      echo "rc=$rc"
    } > "$CHILD_DIR/BUNDLE_CHILD_FAILED"
    echo "[$(date)] FAILED child $child_run_id rc=$rc"
    exit "$rc"
  fi
done < "$RUNLIST"

echo "[$(date)] axis1 bundle $RUN_ID finished"
