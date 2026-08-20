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

# -- submit guards (drip-feed contract) -------------------------------------
# DV3_MAX_SUBMIT caps successful sbatch calls this pass (the gpu QOS allows
# only 12 submitted jobs per user; the surplus is REJECTED, not queued).
# A cell is skipped when its run dir exists OR a job of that name is already
# queued/running, so re-running this script is always safe.
MAX_SUBMIT="${DV3_MAX_SUBMIT:-9999}"
SQ=/software/slurm-current-el8-x86_64/bin/squeue
QUEUED="$($SQ -h -u "$USER" -o "%j" 2>/dev/null)"
n_ok=0; n_skip=0; n_rej=0; n_left=0

submit_cell () {  # submit_cell <run_id> <walltime> <export-list>
  local rid="$1" wall="$2" exp="$3"
  if [ -d "$RUNROOT/$rid" ]; then n_skip=$((n_skip+1)); return 0; fi
  case "$QUEUED" in *"$rid"*) n_skip=$((n_skip+1)); return 0 ;; esac
  if [ "$n_ok" -ge "$MAX_SUBMIT" ]; then n_left=$((n_left+1)); return 0; fi
  local out rc
  out="$($SB --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
      --gres="$SLURM_GRES" --time="$wall" --constraint=v100 \
      --job-name="$rid" --export="$exp" "$REPO/scripts/axis1.sbatch" 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ] && echo "$out" | grep -q "Submitted batch job"; then
    echo "  OK   $rid  ($(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk "{print \$4}"))"
    n_ok=$((n_ok+1))
  else
    echo "  REJ  $rid  ($(echo "$out" | grep -oE "QOS[A-Za-z]+|error: [^|]*" | head -1))"
    n_rej=$((n_rej+1)); n_left=$((n_left+1))
  fi
}

for arm in task apt; do
  case "$arm" in
    task) pfx=uz;  wmpfx="";  mode=task; wall=24:00:00 ;;
    apt)  pfx=uzf; wmpfx="f"; mode=apt;  wall=36:00:00 ;;
  esac
  for side in 0 1; do
    for k in 9 10 11 12 13 14 15 16; do
      RUN_ID="adapt_ax1${pfx}q1s${side}_finger_seed${k}_ckpt500000"
      WM_RUN="ax1wm_finger_${wmpfx}q1s${side}_seed${k}"
      submit_cell "$RUN_ID" "$wall" \
        "ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=$RUN_ID,WM_RUN=$WM_RUN,TASK=dmc_finger_turn_hard,SEED=$k,REPLAY=$RUNROOT/axis1_finger/q1/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=$mode,AXIS1_ADAPT_CONFIG=unfrozen_readout"
    done
  done
done

echo "submitted=$n_ok  already-present=$n_skip  rejected=$n_rej  still-pending=$n_left"
[ "$n_left" -eq 0 ] && echo "ALL CELLS ACCOUNTED FOR"
exit 0
