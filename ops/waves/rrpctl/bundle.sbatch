#!/bin/bash
#SBATCH --job-name=rrpctl_bundle
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Routed-repair CONTROL wave -- PREREG_rrpctl_20260822 (freeze-commit the
# prereg + analysis/rrpctl_read.py + the builder rrpctl support BEFORE any
# run). Mirrors ops/waves/routedrepair/bundle.sbatch.
#
# Arms (finger q1 side1, seeds 17-24; phase-1 = the EXISTING carrier fits
# ax1wm_finger_rgoq1s1_seed{k}):
#   rca   rgo-from-random-init control: adapt-only off the carrier fit
#         (~5 h/cell)
#   rcb   matched budget: rgo-refit (500k) INITIALIZED from the carrier
#         fit -- the EXACT rrp recipe with the init source swapped
#         (AXIS1_INIT_REGEX/FROZEN_ENC identical) -- then adapt
#         (~12 h/cell)
#
# EXISTENCE PRECONDITION (prereg Ops): all 8 carrier rgo fits must hold a
# done-ckpt at STEP 500000 BEFORE any submission -- axis1's fit-skip guard
# would otherwise silently REFIT the shared registered carrier fit (review
# B1: the check must mirror axis1's latest_ckpt/ckpt_update, never the
# ckpt/latest pointer file, which exists after ANY save incl. a 50k one).
# Checked per-cell below with that exact rule; run the 8-dir sweep first.
#
# Job-log lines feed the witness: build_rescue_bundle --wave rrpctl parses
# rrpctl*.out for the CELL/OK/DONE lines (adapt completion evidence), so
# keep these logs with the wave.
#
# TRUNCATION GUARD (08-07 realized-training discovery): a cell is never
# STARTED unless its estimated walltime remains -- complete or never begun.
# Resumable: ADAPT_DONE cells are skipped.
#
# Env: CELLS = space-separated <arm>:<seed> tokens (e.g. "rcb:17 rca:17"),
#      JOB_WALL_S (86400)
# ---------------------------------------------------------------------------
set -uo pipefail
T0=$SECONDS
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
cd "$REPO"
: "${RUNROOT:?set RUNROOT}"
: "${CELLS:?set CELLS}"
JOB_WALL_S="${JOB_WALL_S:-86400}"

echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
echo "[bundle] cells: $CELLS"; echo

# Review B1: readiness = the newest COMPLETED ckpt's step (axis1's own
# latest_ckpt/ckpt_update rule), not the ckpt/latest pointer.
carrier_done_step() {
  local ck base suffix
  ck="$(find "$1/ckpt" -mindepth 2 -maxdepth 2 -name done -printf '%h\n' 2>/dev/null | sort | tail -1)"
  [ -n "$ck" ] || { echo 0; return; }
  base="$(basename "$ck")"; suffix="${base##*-}"
  case "$suffix" in ''|*[!0-9]*) echo 0 ;; *) echo "$suffix" ;; esac
}

n_ok=0; n_skip=0; n_defer=0; n_fail=0; summary=""
for cell in $CELLS; do
  arm="${cell%%:*}"; k="${cell##*:}"
  case "$k" in 1[7-9]|2[0-4]) ;; *) echo "FAIL  seed '$k' not in 17..24 (PREREG_rrpctl_20260822)"; n_fail=$((n_fail+1)); continue ;; esac
  RUN_ID="adapt_ax1${arm}q1s1_finger_seed${k}_ckpt500000"
  LOGDIR="$RUNROOT/$RUN_ID"
  SRC="ax1wm_finger_rgoq1s1_seed${k}"
  extra=()
  case "$arm" in
    # rca is adapt-only, but AXIS1_ARM=rgo rides along as defense in
    # depth (review B1): if the completeness guard were ever wrong,
    # an accidental refit at least runs the registered arm, not full.
    rca) WM_RUN="$SRC"; est=18000
         extra=( "AXIS1_ARM=rgo" ) ;;
    rcb) WM_RUN="ax1wm_finger_rcbq1s1_seed${k}"; est=43200
         extra=( "AXIS1_INIT_WM=$SRC"
                 "AXIS1_INIT_REGEX=^(enc|dyn|dec)/"
                 "AXIS1_INIT_FROZEN_ENC=False"
                 "AXIS1_ARM=rgo" ) ;;
    *) echo "FAIL  unknown arm '$arm'"; n_fail=$((n_fail+1)); continue ;;
  esac
  if [ -f "$LOGDIR/ADAPT_DONE" ]; then
    echo "SKIP   $RUN_ID (ADAPT_DONE)"; n_skip=$((n_skip+1))
    summary="$summary\n  SKIP    $RUN_ID"; continue
  fi
  # Both arms need the carrier fit COMPLETE (rca adapts from it; rcb
  # initializes from it). Review B1: test the newest done-ckpt's step
  # == 500000 -- a present-but-short fit (stale copy / partial sync)
  # would otherwise fall through into axis1's refit path and clobber
  # the shared registered carrier fit.
  src_step="$(carrier_done_step "$RUNROOT/$SRC")"
  if [ "$src_step" -ne 500000 ]; then
    echo "FAIL   $RUN_ID: carrier rgo fit incomplete ($SRC done-ckpt step $src_step != 500000)"; n_fail=$((n_fail+1))
    summary="$summary\n  FAIL    $RUN_ID (carrier step $src_step)"; continue
  fi
  left=$(( JOB_WALL_S - (SECONDS - T0) ))
  if [ "$left" -lt "$est" ]; then
    echo "DEFER  $RUN_ID -- ${left}s left < ${est}s estimated; not started"
    n_defer=$((n_defer+1)); summary="$summary\n  DEFER   $RUN_ID"; continue
  fi
  echo "[$(date)] CELL $RUN_ID  arm=$arm  (${left}s left)"
  ( # Review m6: neutralize inherited axis1 dials (sbatch --export=ALL
    # is the default; FORCE_WM=1 or a stray AXIS1_* in the submitting
    # shell would change the experiment silently).
    unset FORCE_WM AXIS1_WR AXIS1_SIZE AXIS1_BASE_CONFIG \
          AXIS1_INIT_WM AXIS1_INIT_REGEX AXIS1_INIT_FROZEN_ENC \
          AXIS1_SAVE_EVERY_UPDATES AXIS1_ARM 2>/dev/null || true
    export RUN_ID="$RUN_ID" WM_RUN="$WM_RUN" TASK=dmc_finger_turn_hard SEED="$k" \
      REPLAY="$RUNROOT/axis1_finger/q1/side1" \
      UPDATES=500000 STEPS=1.25e5 AXIS1_EXPL_MODE=task \
      AXIS1_ADAPT_CONFIG=unfrozen_readout \
      REPO="$REPO" RUNROOT="$RUNROOT" CONDA_ENV="$CONDA_ENV" MANIFEST="$MANIFEST"
    for e in "${extra[@]+"${extra[@]}"}"; do export "$e"; done
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
echo "[$(date)] rrpctl bundle: ok=$n_ok skipped=$n_skip deferred=$n_defer failed=$n_fail"
printf "%b\n" "$summary"
[ "$n_defer" -gt 0 ] && echo "NOTE: deferred cells were never started -- re-submit this bundle to finish them."
[ "$n_fail" -eq 0 ] || exit 1
exit 0
