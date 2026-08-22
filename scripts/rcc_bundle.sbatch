#!/bin/bash
#SBATCH --job-name=dv3_bundle
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Generic RCC BUNDLE runner -- run N array cells inside ONE Slurm job.
#
# WHY: the gpu QOS caps SUBMITTED jobs at 12 per user, and the surplus is
# REJECTED, not queued (`QOSMaxSubmitJobPerUserLimit`). An 8-element array
# therefore burns 8 of 12 slots. When the per-cell cost is small relative to
# the queue wait -- se_planner_loc and se_refine measured ~20-24 s per cell in
# their smokes -- bundling is strictly better: 1 slot instead of 8, no queue
# wait per cell, and the same work.
#
# This is the house shape already used by ops/waves/routedrepair/bundle.sbatch
# (CELLS="rrp:1 raft:1" in one job); generalised so any array payload can use
# it without a bespoke wrapper.
#
# Env:
#   PAYLOAD  path to an array sbatch that indexes off $SLURM_ARRAY_TASK_ID
#   IDX      space-separated indices to run (default "0 1 2 3 4 5 6 7")
#   plus whatever the payload itself needs (e.g. RUNS_ROOT).
#
# The payload's own #SBATCH lines are comments under bash, so invoking it
# directly is safe -- that is explicitly NOT a reason to give a script its own
# Slurm job (house routing note).
#
# on_failure=continue by design: one bad cell must not strand the rest of the
# bundle. Every cell's rc is recorded and the job exits non-zero if any failed,
# so a partial bundle can never read as success.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
: "${PAYLOAD:?set PAYLOAD to the array sbatch to run}"
[ -f "$PAYLOAD" ] || { echo "ERROR: no payload at $PAYLOAD"; exit 2; }
IDX="${IDX:-0 1 2 3 4 5 6 7}"

echo "[bundle] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[bundle] payload=$PAYLOAD"
echo "[bundle] indices: $IDX"
echo

n_ok=0; n_fail=0; failed=""
for i in $IDX; do
  echo "===== [$(date)] BUNDLE CELL $i ====="
  SLURM_ARRAY_TASK_ID="$i" bash "$PAYLOAD"
  rc=$?
  if [ "$rc" -eq 0 ]; then echo "OK   cell $i"; n_ok=$((n_ok+1))
  else echo "FAIL cell $i (rc=$rc)"; n_fail=$((n_fail+1)); failed="$failed $i"; fi
done

echo
echo "[bundle] ok=$n_ok failed=$n_fail"
[ -n "$failed" ] && echo "FAILED CELLS:$failed"
[ "$n_fail" -eq 0 ] || exit 1
exit 0
