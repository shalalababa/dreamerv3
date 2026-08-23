#!/bin/bash
#SBATCH --job-name=rrp2_bundle
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Routed-repair PAIRED certification wave -- PREREG_rrp2_20260823
# (freeze-commit prereg + analysis/rrp2_read.py + builder rrp2 support
# BEFORE any run). Background priority; cheap/slow lanes acceptable.
#
# ONE SEED PER JOB, all three arms in sequence on the SAME GPU — the
# invocation-pairing gate (the point of this wave) is derived from these
# job logs: the reader REFUSES any seed whose three adapts do not share
# one log file. Do not split a seed across jobs, ever. CONSEQUENCE: a
# seed that crashes mid-job is UNSALVAGEABLE by resume (its completed
# arms' OK lines live in the first job's log) — the remedy is
# archive-then-delete that seed's adapt dirs (never-delete rule: archive
# first) and re-run the seed whole in one fresh job.
#
#   rp2r  repair'  re-adapt of the EXISTING rrp refit          (~5 h)
#   rp2a  rca'     adapt of the rgo source fit                 (~5 h)
#   rp2b  rcb'     rrp-recipe refit FROM the source fit + adapt (~12 h)
# ~22 h total against the 36 h wall.
#
# rca'-SOURCE PREFIX: resolved by the registered existence FILL --
# surviving ax1wm_finger_rgoq1s1_seed{1..8} (all 8 done-ckpt at step
# 500000) else fresh ax1wm_finger_rp2gq1s1_seed{1..8} (run those FIRST,
# path B). This script auto-detects with the SAME rule the builder uses;
# mixed sources across seeds are impossible (the check is all-8-or-none).
#
# Readiness = newest done-ckpt STEP == 500000 (never ckpt/latest);
# inherited AXIS1_*/FORCE_WM dials neutralized per cell (rrpctl B1/m6).
#
# Env: CELLS = space-separated seeds (e.g. "1 2"), JOB_WALL_S (129600)
# ---------------------------------------------------------------------------
set -uo pipefail
T0=$SECONDS
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
# m10: guard BEFORE env.sh, which would default RUNROOT silently.
: "${RUNROOT:?set RUNROOT}"
: "${CELLS:?set CELLS (seeds)}"
source "$REPO/scripts/env.sh"
cd "$REPO"
# m3: 1h margin under the 36h wall (a cell starting at left==est would
# otherwise be wall-killed at zero margin).
JOB_WALL_S="${JOB_WALL_S:-126000}"

echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

done_step() {
  local ck base suffix
  ck="$(find "$1/ckpt" -mindepth 2 -maxdepth 2 -name done -printf '%h\n' 2>/dev/null | sort | tail -1)"
  [ -n "$ck" ] || { echo 0; return; }
  base="$(basename "$ck")"; suffix="${base##*-}"
  case "$suffix" in ''|*[!0-9]*) echo 0 ;; *) echo "$suffix" ;; esac
}

# rca'-source prefix: all-8-or-none, same rule as the builder.
SRC_PREFIX=""
for pfx in ax1wm_finger_rgoq1s1 ax1wm_finger_rp2gq1s1; do
  ok=1
  for k in 1 2 3 4 5 6 7 8; do
    [ "$(done_step "$RUNROOT/${pfx}_seed${k}")" -eq 500000 ] || { ok=0; break; }
  done
  [ "$ok" -eq 1 ] && { SRC_PREFIX="$pfx"; break; }
done
[ -n "$SRC_PREFIX" ] || { echo "ERROR: neither rca-source prefix has 8/8 complete fits at 500000 (path B: run the rp2g fits first)"; exit 2; }
echo "[bundle] seeds: $CELLS  rca-source: $SRC_PREFIX"; echo

n_ok=0; n_skip=0; n_defer=0; n_fail=0; summary=""
run_cell() {  # arm seed wm_run est extra...
  local arm="$1" k="$2" WM_RUN="$3" est="$4"; shift 4
  local RUN_ID="adapt_ax1rp2${arm}q1s1_finger_seed${k}_ckpt500000"
  local LOGDIR="$RUNROOT/$RUN_ID"
  if [ -f "$LOGDIR/ADAPT_DONE" ]; then
    echo "SKIP   $RUN_ID (ADAPT_DONE)"; n_skip=$((n_skip+1))
    summary="$summary\n  SKIP    $RUN_ID"; return 0
  fi
  local left=$(( JOB_WALL_S - (SECONDS - T0) ))
  if [ "$left" -lt "$est" ]; then
    echo "DEFER  $RUN_ID -- ${left}s left < ${est}s; not started"
    n_defer=$((n_defer+1)); summary="$summary\n  DEFER   $RUN_ID"; return 0
  fi
  echo "[$(date)] CELL $RUN_ID  arm=$arm  (${left}s left)"
  ( unset FORCE_WM AXIS1_WR AXIS1_SIZE AXIS1_BASE_CONFIG \
        AXIS1_INIT_WM AXIS1_INIT_REGEX AXIS1_INIT_FROZEN_ENC \
        AXIS1_SAVE_EVERY_UPDATES AXIS1_ARM 2>/dev/null || true
    export RUN_ID="$RUN_ID" WM_RUN="$WM_RUN" TASK=dmc_finger_turn_hard SEED="$k" \
      REPLAY="$RUNROOT/axis1_finger/q1/side1" \
      UPDATES=500000 STEPS=1.25e5 AXIS1_EXPL_MODE=task \
      AXIS1_ADAPT_CONFIG=unfrozen_readout \
      REPO="$REPO" RUNROOT="$RUNROOT" CONDA_ENV="$CONDA_ENV" MANIFEST="$MANIFEST"
    for e in "$@"; do export "$e"; done
    bash "$REPO/scripts/axis1.sbatch" )
  local rc=$?
  local nep; nep=$(grep -c . "$LOGDIR/scores.jsonl" 2>/dev/null || echo 0)
  if [ "$rc" -eq 0 ] && [ -f "$LOGDIR/ADAPT_DONE" ] && [ "$nep" -ge 100 ]; then
    echo "OK     $RUN_ID (scores.jsonl $nep lines)"; n_ok=$((n_ok+1))
    summary="$summary\n  DONE    $RUN_ID  n_ep_lines=$nep"
  else
    echo "FAIL   $RUN_ID rc=$rc scores=$nep"; n_fail=$((n_fail+1))
    summary="$summary\n  FAIL    $RUN_ID rc=$rc scores=$nep"
  fi
}

for k in $CELLS; do
  case "$k" in [1-8]) ;; *) echo "FAIL  seed '$k' not in 1..8 (PREREG_rrp2_20260823)"; n_fail=$((n_fail+1)); continue ;; esac
  RRP="ax1wm_finger_rrpq1s1_seed${k}"
  SRC="${SRC_PREFIX}_seed${k}"
  for d in "$RRP" "$SRC"; do
    if [ "$(done_step "$RUNROOT/$d")" -ne 500000 ]; then
      echo "FAIL   seed $k: source fit incomplete ($d)"; n_fail=$((n_fail+1)); continue 2
    fi
  done
  # All three arms of the seed in THIS job, LONGEST FIRST (m4: under
  # the all-or-nothing pairing gate this maximizes P(all three finish)
  # -- the 12h cell is budget-checked against the fullest wall).
  run_cell b "$k" "ax1wm_finger_rp2bq1s1_seed${k}" 43200 \
      "AXIS1_INIT_WM=$SRC" "AXIS1_INIT_REGEX=^(enc|dyn|dec)/" \
      "AXIS1_INIT_FROZEN_ENC=False" "AXIS1_ARM=rgo"
  run_cell r "$k" "$RRP" 18000 "AXIS1_ARM=rgo"
  run_cell a "$k" "$SRC" 18000 "AXIS1_ARM=rgo"
done

echo
echo "[$(date)] rrp2 bundle: ok=$n_ok skipped=$n_skip deferred=$n_defer failed=$n_fail"
printf "%b\n" "$summary"
{ [ "$n_defer" -gt 0 ] || [ "$n_fail" -gt 0 ]; } && echo "NOTE: a PARTIAL seed is NOT resumable (the pairing gate refuses split job logs): archive-then-delete its adapt dirs, then re-run the seed whole in ONE fresh job. Resume-as-is ONLY if all three arms were deferred (no OK line written)."
[ "$n_fail" -eq 0 ] || exit 1
exit 0
