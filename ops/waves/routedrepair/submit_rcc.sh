#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Routed repair -- submit the remaining bundles under the gpu QOS's
# 12-submitted-job cap. Same drip contract as the other wave submitters:
# DV3_MAX_SUBMIT caps successful sbatch calls per pass, a bundle already
# queued (by job name) or already complete on disk is skipped, and
# "ALL CELLS ACCOUNTED FOR" prints only when nothing is left.
#
# Bundle plan (see bundle.sbatch for why): eight seed-paired D1 jobs carrying
# rrp+raft of the same seed, then two S&P jobs of four descriptive cells.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-/home/rickybao/projects/dreamerv3}"
source "$REPO/scripts/env.sh"
SB=/software/slurm-current-el8-x86_64/bin/sbatch
SQ=/software/slurm-current-el8-x86_64/bin/squeue
MAX_SUBMIT="${DV3_MAX_SUBMIT:-9999}"
QUEUED="$($SQ -h -u "$USER" -o "%j" 2>/dev/null)"
n_ok=0; n_skip=0; n_left=0

submit_bundle () {  # submit_bundle <job_name> <cells...>
  local jn="$1"; shift; local cells="$*"
  # complete when every cell in the bundle has its ADAPT_DONE
  local done_all=1 c arm k
  for c in $cells; do
    arm="${c%%:*}"; k="${c##*:}"
    [ -f "$RUNROOT/adapt_ax1${arm}q1s1_finger_seed${k}_ckpt500000/ADAPT_DONE" ] || done_all=0
  done
  if [ "$done_all" -eq 1 ]; then n_skip=$((n_skip+1)); return 0; fi
  case "$QUEUED" in *"$jn"*) n_skip=$((n_skip+1)); return 0 ;; esac
  if [ "$n_ok" -ge "$MAX_SUBMIT" ]; then n_left=$((n_left+1)); return 0; fi
  local out
  out="$($SB --job-name="$jn" \
      --export=ALL,REPO="$REPO",RUNROOT="$RUNROOT",CELLS="$cells" \
      "$REPO/ops/waves/routedrepair/bundle.sbatch" 2>&1 | grep -o "Submitted batch job [0-9]*")"
  if [ -n "$out" ]; then echo "  OK   $jn ($cells) -> $out"; n_ok=$((n_ok+1))
  else echo "  REJ  $jn ($cells)"; n_left=$((n_left+1)); fi
}

for k in 1 2 3 4 5 6 7 8; do submit_bundle "rrep_k${k}" "rrp:${k}" "raft:${k}"; done
submit_bundle "rrep_snpA" rsp:1 rsp:2 rsp:3 rsp:4
submit_bundle "rrep_snpB" rsp:5 rsp:6 rsp:7 rsp:8

echo "submitted=$n_ok  already-queued-or-done=$n_skip  still-pending=$n_left"
[ "$n_left" -eq 0 ] && echo "ALL CELLS ACCOUNTED FOR"
exit 0
