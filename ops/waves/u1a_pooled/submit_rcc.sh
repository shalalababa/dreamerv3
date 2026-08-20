#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# U1a pooled re-test -- PREREG_u1a_pooled_20260820 (+amend1). RUNS ON RCC.
#
# 32 fit+adapt jobs, seeds 9-16: the REBUILD branch (inventory 20 Aug found
# 0/16 of the seeds 9-16 fits surviving). axis1.sbatch does both stages in one
# job -- offline fit at UPDATES=500000, then the UNFROZEN adapt.
#
# run_ids and modes transcribed from the frozen reader
# (analysis/u1a_pooled_read.py:43-45): task -> ax1uzq1s{side},
# apt -> ax1uzfq1s{side}. NOTE these differ from the U1 look-1 wave's
# ax1uzt*/ax1uzf* strings; the reader is what consumes these rows, so the
# reader wins. Donor WM names follow the U1 registration verbatim
# (ax1wm_finger_q1s* task / ax1wm_finger_fq1s* apt).
#
# Walltime: the apt arm's offline fit is the expensive half (the reward-free
# objective computes intrinsic rewards every update; the house RCC figure is
# ~15 h vs ~5-7 h for task), so apt gets the 36 h QOS ceiling and task 24 h.
# Both are single jobs -- no bundle -- so a truncation cannot hide inside a
# multi-cell job the way the 08-07 capacity waves did.
#
# --constraint=v100: one GPU model across all 32 cells; the interaction is
# built from per-seed differences and pooled with look-1 rows that were
# themselves measured on RCC.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-/home/rickybao/projects/dreamerv3}"
source "$REPO/scripts/env.sh"
SB=/software/slurm-current-el8-x86_64/bin/sbatch
n=0
for arm in task apt; do
  case "$arm" in
    task) pfx=uz;  wmpfx="";  mode=task; wall=24:00:00 ;;
    apt)  pfx=uzf; wmpfx="f"; mode=apt;  wall=36:00:00 ;;
  esac
  for side in 0 1; do
    for k in 9 10 11 12 13 14 15 16; do
      RUN_ID="adapt_ax1${pfx}q1s${side}_finger_seed${k}_ckpt500000"
      WM_RUN="ax1wm_finger_${wmpfx}q1s${side}_seed${k}"
      if [ -d "$RUNROOT/$RUN_ID" ]; then echo "SKIP exists: $RUN_ID"; continue; fi
      $SB --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
        --gres="$SLURM_GRES" --time="$wall" --constraint=v100 \
        --job-name="$RUN_ID" \
        --export=ALL,REPO="$REPO",RUNROOT="$RUNROOT",CONDA_ENV="$CONDA_ENV",MANIFEST="$MANIFEST",RUN_ID="$RUN_ID",WM_RUN="$WM_RUN",TASK=dmc_finger_turn_hard,SEED="$k",REPLAY="$RUNROOT/axis1_finger/q1/side${side}",UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE="$mode",AXIS1_ADAPT_CONFIG=unfrozen_readout \
        "$REPO/scripts/axis1.sbatch" | sed "s/^/  $RUN_ID: /"
      n=$((n+1))
    done
  done
done
echo "u1a: submitted $n job(s)"
