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
: "${RUN_ID:?set RUN_ID}"; : "${RUNLIST:?set RUNLIST}"

echo "[$(date)] axis1 bundle script starting"
echo "[$(date)] job=${SLURM_JOB_ID:-NA} host=$(hostname) pwd=$PWD"
echo "[$(date)] REPO=$REPO RUNROOT=${RUNROOT:-unset} CONDA_ENV=${CONDA_ENV:-unset}"
echo "[$(date)] RUN_ID=$RUN_ID RUNLIST=$RUNLIST"

[ -f "$RUNLIST" ] || { echo "ERROR: missing RUNLIST=$RUNLIST"; exit 2; }

source "$REPO/scripts/env.sh"
export PATH="$CONDA_ENV/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
export CONDA_PREFIX="$CONDA_ENV"
export CONDA_DEFAULT_ENV="$(basename "$CONDA_ENV")"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHON="$CONDA_ENV/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python || true)"
[ -n "$PYTHON" ] || { echo "ERROR: no python found for CONDA_ENV=$CONDA_ENV"; exit 2; }

BUNDLE_LOGDIR="$RUNROOT/_bundles/$RUN_ID"
mkdir -p "$BUNDLE_LOGDIR"
cp "$RUNLIST" "$BUNDLE_LOGDIR/runlist.tsv"
{
  echo "bundle=$RUN_ID"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "host=$(hostname)"
  echo "started_at=$(date -Is)"
  echo "runlist=$RUNLIST"
} > "$BUNDLE_LOGDIR/BUNDLE_STARTED"

heartbeat() {
  while true; do
    date -Is > "$BUNDLE_LOGDIR/BUNDLE_HEARTBEAT"
    sleep 60
  done
}
heartbeat &
HEARTBEAT_PID=$!
cleanup() {
  rc=$?
  {
    echo "ended_at=$(date -Is)"
    echo "rc=$rc"
  } > "$BUNDLE_LOGDIR/BUNDLE_EXIT"
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT

echo "[$(date)] env prepared: python=$PYTHON"
"$PYTHON" -u - <<'PY'
import sys
print("python_executable=" + sys.executable)
PY
date -Is > "$BUNDLE_LOGDIR/BUNDLE_ENV_OK"
cd "$REPO"

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
    AXIS1_EXPL_MODE="${AXIS1_EXPL_MODE:?bundle env missing AXIS1_EXPL_MODE}" \
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
date -Is > "$BUNDLE_LOGDIR/BUNDLE_FINISHED"
