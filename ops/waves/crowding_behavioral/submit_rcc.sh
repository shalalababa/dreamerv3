#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Crowding behavioral leg -- PREREG_crowding_behavioral_20260820 (+amend1).
# RUNS ON RCC. 16 UNFROZEN adapts from the EXISTING rescue-wave fits; no new
# fits (axis1 stage 1 idempotently skips a complete WM).
#
# run_ids are transcribed from the frozen reader, not invented:
#   analysis/crowding_behavioral_read.py:45 ADAPTS =
#   adapt_ax1uzruw{w}q1s{side}_finger_seed{s}_ckpt500000, w in {1,100},
#   side in {0,1}, seed in 1..4.  A different id would collate under a mode
#   the reader does not know and the read would refuse.
#
# AXIS1_WR IS DELIBERATELY UNSET: w_r was a PRETRAINING loss weight; the
# registration is explicit that the adapt-time config carries no w_r
# manipulation. Setting it here would silently make the contrast something
# else. AXIS1_BASE_CONFIG=dzs2 is required (it is the sigma=2 substrate the
# donor fits were built under; a mismatch fails at checkpoint load).
#
# --constraint=v100 on ALL 16: the primary is a PAIRED w100-vs-w1 contrast,
# and probe/adapt numbers move across GPU models (probe-panels-single-gpu).
# One model for the whole cell set keeps device out of the pairing.
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

for w in 1 100; do
  for side in 0 1; do
    for k in 1 2 3 4; do
      RUN_ID="adapt_ax1uzruw${w}q1s${side}_finger_seed${k}_ckpt500000"
      WM_RUN="ax1wm_finger_ruw${w}q1s${side}_seed${k}"
      [ -f "$RUNROOT/$WM_RUN/config.yaml" ] || { echo "REFUSE: donor fit missing: $WM_RUN" >&2; exit 2; }
      submit_cell "$RUN_ID" 08:00:00 \
        "ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=$RUN_ID,WM_RUN=$WM_RUN,TASK=dmc_finger_turn_hard,SEED=$k,REPLAY=$RUNROOT/axis1_finger/q1_nzs2/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_ADAPT_CONFIG=unfrozen_readout,AXIS1_BASE_CONFIG=dzs2"
    done
  done
done

echo "submitted=$n_ok  already-present=$n_skip  rejected=$n_rej  still-pending=$n_left"
[ "$n_left" -eq 0 ] && echo "ALL CELLS ACCOUNTED FOR"
exit 0
