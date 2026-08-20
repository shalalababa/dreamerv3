#!/bin/bash
#SBATCH --job-name=crowd_bundle
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Crowding behavioral leg -- PREREG_crowding_behavioral_20260820 (+amend1).
# FOUR cells per job, so the 16-cell wave fits the gpu QOS's 12-submitted-job
# cap in ONE pass instead of trickling through a drip-feed.
#
# BUNDLE MEMBERSHIP IS PAIR-PRESERVING, and that is the point rather than a
# packing convenience. The primary is the paired difference
# [AUC100k(w100) - AUC100k(w1)] at a fixed (side, seed), so each bundle
# carries BOTH w_r members of the two (side, seed) pairs it owns. Every
# paired difference is therefore computed from two runs on the SAME GPU in
# the same job -- strictly stronger than the --constraint=v100 pin alone,
# which only guarantees the same GPU MODEL. Same reasoning as the carrier
# wave's seed-whole blocks.
#
# TRUNCATION IS THE HAZARD BUNDLES CARRY (08-07 realized-training discovery:
# a bundle silently ran cells past its walltime boundary and recorded
# nothing). Two guards, both cheap:
#   * a cell is never STARTED unless MIN_CELL_S of walltime remain, so a cell
#     either completes or never begins -- there is no partial cell;
#   * every finished cell is checked against its realized counters, and the
#     summary prints DONE/DEFERRED/FAILED per cell.
# Resumable: a cell with ADAPT_DONE is skipped, so a re-submit continues.
#
# The donor fits are complete WMs, so axis1.sbatch's stage 1 idempotently
# skips and each cell is adapt-only (~2-3 h). 4 x 3 h = 12 h against a 24 h
# wall is 2x headroom.
#
# AXIS1_WR IS DELIBERATELY UNSET (w_r is a pretraining-time loss weight;
# the registration is explicit the adapt carries no w_r manipulation).
# AXIS1_BASE_CONFIG=dzs2 is required -- it is the sigma=2 substrate the
# donors were fit under, and a mismatch fails at checkpoint load.
#
# Env: SIDE (0|1), SEEDS (e.g. "1 2"), JOB_WALL_S (86400), MIN_CELL_S (18000)
# ---------------------------------------------------------------------------
set -uo pipefail
T0=$SECONDS
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
cd "$REPO"
: "${RUNROOT:?set RUNROOT}"
: "${SIDE:?set SIDE}"
: "${SEEDS:?set SEEDS}"
JOB_WALL_S="${JOB_WALL_S:-86400}"
MIN_CELL_S="${MIN_CELL_S:-18000}"

echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
echo "[bundle] side=$SIDE seeds='$SEEDS'  (both w_r arms per seed -> pairs stay in-job)"
echo

n_ok=0; n_skip=0; n_defer=0; n_fail=0; summary=""
for k in $SEEDS; do
  for w in 1 100; do
    RUN_ID="adapt_ax1uzruw${w}q1s${SIDE}_finger_seed${k}_ckpt500000"
    WM_RUN="ax1wm_finger_ruw${w}q1s${SIDE}_seed${k}"
    LOGDIR="$RUNROOT/$RUN_ID"
    if [ -f "$LOGDIR/ADAPT_DONE" ]; then
      echo "SKIP   $RUN_ID (ADAPT_DONE)"; n_skip=$((n_skip+1))
      summary="$summary\n  SKIP    $RUN_ID"; continue
    fi
    if [ ! -f "$RUNROOT/$WM_RUN/config.yaml" ]; then
      echo "FAIL   $RUN_ID: donor fit missing ($WM_RUN)"; n_fail=$((n_fail+1))
      summary="$summary\n  FAIL    $RUN_ID (no donor)"; continue
    fi
    left=$(( JOB_WALL_S - (SECONDS - T0) ))
    if [ "$left" -lt "$MIN_CELL_S" ]; then
      echo "DEFER  $RUN_ID -- ${left}s left < ${MIN_CELL_S}s needed; not started"
      n_defer=$((n_defer+1)); summary="$summary\n  DEFER   $RUN_ID"; continue
    fi
    echo "[$(date)] CELL $RUN_ID  (${left}s of walltime left)"
    ( RUN_ID="$RUN_ID" WM_RUN="$WM_RUN" TASK=dmc_finger_turn_hard SEED="$k" \
      REPLAY="$RUNROOT/axis1_finger/q1_nzs2/side${SIDE}" \
      UPDATES=500000 STEPS=1.25e5 AXIS1_EXPL_MODE=task \
      AXIS1_ADAPT_CONFIG=unfrozen_readout AXIS1_BASE_CONFIG=dzs2 \
      REPO="$REPO" RUNROOT="$RUNROOT" CONDA_ENV="$CONDA_ENV" MANIFEST="$MANIFEST" \
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
done

echo
echo "[$(date)] bundle side=$SIDE seeds='$SEEDS': ok=$n_ok skipped=$n_skip deferred=$n_defer failed=$n_fail"
printf "%b\n" "$summary"
[ "$n_defer" -gt 0 ] && echo "NOTE: deferred cells were never started -- re-submit this bundle to finish them."
[ "$n_fail" -eq 0 ] || exit 1
exit 0
