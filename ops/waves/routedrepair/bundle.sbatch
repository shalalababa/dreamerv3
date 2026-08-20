#!/bin/bash
#SBATCH --job-name=rrepair
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Routed repair -- PREREG_routedrepair_20260820 + amend1 (B13: 24 cells) +
# amend2 (seed5 donor completed by re-save).
#
# THREE ARMS, all on finger q1 side1, all adapting UNFROZEN:
#   rrp   repair       rgo re-fit from the sunk apt trunk, then adapt
#   raft  apt control  plain full fine-tune of the SAME donor (B13)
#   rsp   S&P          adapt from probing/shrink_perturb's output (descriptive)
# run_ids/modes transcribed from the frozen reader
# (analysis/routedrepair_read.py:47): ax1rrpq1s1 / ax1raftq1s1 / ax1rspq1s1.
#
# BUNDLE BY SEED, and that is the design rather than packing: D1 is the
# PAIRED per-donor difference [repair_i - aptft_i], so a job carries the rrp
# and raft cells of the SAME seed and the pair is computed from two runs on
# one GPU in one job -- stronger than --constraint=v100, which fixes only the
# model. (Same argument as the crowding bundles and the carrier's seed-whole
# blocks.) The rsp arm is descriptive-only, so it bundles by convenience.
#
# REPAIR-ARM WIRING, the part that is easy to get silently wrong:
#   AXIS1_INIT_REGEX='^(enc|dyn|dec)/'  loads the whole sunk TRUNK
#   AXIS1_INIT_FROZEN_ENC=False         and does NOT freeze it -- the claim
#                                       is reward gradients reaching the
#                                       representation, so a frozen encoder
#                                       would test the opposite thing
#   AXIS1_ARM=rgo                       the routing arm
# The regex deliberately omits ^rew/: the apt donors carry NO reward-head
# params at all (0 /rew/ entries -- the fq1 finding), so the head trains
# fresh. Fresh head + gradients routed into a loaded, unfrozen trunk IS the
# arm. Requiring ^rew/ here would fail the load.
#
# Truncation guard, as in the crowding bundles: a cell is never STARTED
# unless its arm's estimated cost fits the remaining walltime, so a cell
# either completes or never begins. Resumable; ADAPT_DONE cells are skipped.
#
# Env: CELLS = space-separated <arm>:<seed> tokens (e.g. "rrp:1 raft:1")
#      JOB_WALL_S (86400)
# ---------------------------------------------------------------------------
set -uo pipefail
T0=$SECONDS
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
cd "$REPO"
: "${RUNROOT:?set RUNROOT}"; : "${CELLS:?set CELLS}"
JOB_WALL_S="${JOB_WALL_S:-86400}"

echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
echo "[bundle] cells: $CELLS"; echo

n_ok=0; n_skip=0; n_defer=0; n_fail=0; summary=""
for cell in $CELLS; do
  arm="${cell%%:*}"; k="${cell##*:}"
  RUN_ID="adapt_ax1${arm}q1s1_finger_seed${k}_ckpt500000"
  LOGDIR="$RUNROOT/$RUN_ID"
  extra=()
  case "$arm" in
    rrp)  WM_RUN="ax1wm_finger_rrpq1s1_seed${k}"; est=43200
          extra=( "AXIS1_INIT_WM=ax1wm_finger_fq1s1_seed${k}"
                  "AXIS1_INIT_REGEX=^(enc|dyn|dec)/"
                  "AXIS1_INIT_FROZEN_ENC=False"
                  "AXIS1_ARM=rgo" ) ;;
    raft) WM_RUN="ax1wm_finger_fq1s1_seed${k}";   est=18000 ;;
    rsp)  WM_RUN="ax1wm_finger_rspq1s1_seed${k}"; est=18000 ;;
    *) echo "FAIL  unknown arm '$arm'"; n_fail=$((n_fail+1)); continue ;;
  esac
  if [ -f "$LOGDIR/ADAPT_DONE" ]; then
    echo "SKIP   $RUN_ID (ADAPT_DONE)"; n_skip=$((n_skip+1))
    summary="$summary\n  SKIP    $RUN_ID"; continue
  fi
  # raft/rsp adapt from an EXISTING run dir; rrp creates its own via the refit.
  if [ "$arm" != rrp ] && [ ! -f "$RUNROOT/$WM_RUN/ckpt/latest" ]; then
    echo "FAIL   $RUN_ID: source run dir missing ($WM_RUN)"; n_fail=$((n_fail+1))
    summary="$summary\n  FAIL    $RUN_ID (no source)"; continue
  fi
  left=$(( JOB_WALL_S - (SECONDS - T0) ))
  if [ "$left" -lt "$est" ]; then
    echo "DEFER  $RUN_ID -- ${left}s left < ${est}s estimated; not started"
    n_defer=$((n_defer+1)); summary="$summary\n  DEFER   $RUN_ID"; continue
  fi
  echo "[$(date)] CELL $RUN_ID  arm=$arm  (${left}s left)"
  ( export RUN_ID="$RUN_ID" WM_RUN="$WM_RUN" TASK=dmc_finger_turn_hard SEED="$k" \
      REPLAY="$RUNROOT/axis1_finger/q1/side1" \
      UPDATES=500000 STEPS=1.25e5 AXIS1_EXPL_MODE=task \
      AXIS1_ADAPT_CONFIG=unfrozen_readout \
      REPO="$REPO" RUNROOT="$RUNROOT" CONDA_ENV="$CONDA_ENV" MANIFEST="$MANIFEST"
    for e in "${extra[@]}"; do export "$e"; done
    bash "$REPO/scripts/axis1.sbatch" )
  rc=$?
  nep=$(grep -c . "$LOGDIR/scores.jsonl" 2>/dev/null || echo 0)
  if [ "$rc" -eq 0 ] && [ -f "$LOGDIR/ADAPT_DONE" ] && [ "$nep" -ge 100 ]; then
    echo "OK     $RUN_ID (scores.jsonl $nep lines)"; n_ok=$((n_ok+1))
    summary="$summary\n  DONE    $RUN_ID  n_ep_lines=$nep"
  else
    echo "FAIL   $RUN_ID rc=$rc ADAPT_DONE=$([ -f "$LOGDIR/ADAPT_DONE" ] && echo yes || echo no) scores=$nep"
    n_fail=$((n_fail+1)); summary="$summary\n  FAIL    $RUN_ID rc=$rc scores=$nep"
  fi
done

echo
echo "[$(date)] routed repair bundle: ok=$n_ok skipped=$n_skip deferred=$n_defer failed=$n_fail"
printf "%b\n" "$summary"
[ "$n_defer" -gt 0 ] && echo "NOTE: deferred cells were never started -- re-submit to finish them."
[ "$n_fail" -eq 0 ] || exit 1
exit 0
