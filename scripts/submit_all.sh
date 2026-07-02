#!/bin/bash
# ---------------------------------------------------------------------------
# Expand the sweep grids and submit them. Sources env.sh so account/partition/
# paths come from one place and propagate to jobs via --export.
#
#   ./submit_all.sh pilots [dmc_cup_catch dmc_finger_turn_hard]  # Phase 3 Gate 0
#   ./submit_all.sh pretrain                                     # Phase 4 grid
#   ./submit_all.sh adapt <pretrain_run_id> <task> [STEPS]       # Phase 5 dose-response
#
# Dry run: set DRYRUN=1 to print sbatch commands without submitting.
# Duplicate guard: by default, skip a RUN_ID that is already in Slurm or whose
# logdir already exists under RUNROOT. Set FORCE=1 to submit anyway.
# RCC job cap guard: submit only until current Slurm jobs + this invocation
# reaches MAX_JOBS (default 12). Re-run the command after jobs finish.
# Tunables (env): STEPS, SEEDS, DECOUPLERS, CONTROL, MAX_JOBS.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
source "$REPO/scripts/env.sh"

slurm_user() {
  local user="${USER:-${LOGNAME:-}}"
  [ -n "$user" ] || return 1
  printf '%s\n' "$user"
}

active_job_count() {
  command -v squeue >/dev/null 2>&1 || { echo 0; return 0; }
  local user
  user="$(slurm_user)" || { echo 0; return 0; }
  squeue -h -u "$user" | wc -l | tr -d ' '
}

queued_job() {  # queued_job <RUN_ID>
  command -v squeue >/dev/null 2>&1 || return 1
  local user
  user="$(slurm_user)" || return 1
  squeue -h -u "$user" -o "%j" |
    awk -v name="$1" '$0 == name {found=1} END {exit found ? 0 : 1}'
}

existing_logdir() {  # existing_logdir <RUN_ID>
  local dir="$RUNROOT/$1"
  [ -e "$dir" ] && [ -n "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]
}

MAX_JOBS="${MAX_JOBS:-12}"
JOBS_AT_START="$(active_job_count)"
SUBMITTED_THIS_RUN=0

submit() {  # submit <script> <RUN_ID> KEY=VAL ...
  local script="$1"; local run_id="$2"; shift 2
  if [ "${FORCE:-0}" != "1" ]; then
    if queued_job "$run_id"; then
      echo "SKIP queued/running: $run_id"
      return 0
    fi
    if existing_logdir "$run_id"; then
      echo "SKIP existing logdir: $RUNROOT/$run_id  (set FORCE=1 to resubmit)"
      return 0
    fi
  fi
  local jobs_used=$((JOBS_AT_START + SUBMITTED_THIS_RUN))
  if [ "${IGNORE_JOB_CAP:-0}" != "1" ] && [ "$MAX_JOBS" -gt 0 ] &&
     [ "$jobs_used" -ge "$MAX_JOBS" ]; then
    echo "STOP job cap reached: $jobs_used/$MAX_JOBS Slurm jobs active/submitted."
    echo "Re-run this command after some jobs finish; existing/queued RUN_IDs will be skipped."
    exit 0
  fi
  local exports="ALL,REPO=$REPO,RUN_ID=$run_id"
  for kv in "$@"; do exports="$exports,$kv"; done
  local cmd=(sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION"
             --gres="$SLURM_GRES" --time="$SLURM_TIME"
             --job-name="$run_id" --export="$exports"
             "$REPO/scripts/$script")
  if [ "${DRYRUN:-0}" = "1" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
  SUBMITTED_THIS_RUN=$((SUBMITTED_THIS_RUN + 1))
}

short_of() { echo "$1" | sed -E 's/^dmc_//; s/_.*//'; }

read -ra DECOUPLERS <<< "${DECOUPLERS:-dmc_cup_catch dmc_finger_turn_hard}"
CONTROL="${CONTROL:-dmc_walker_walk}"

cmd="${1:-}"; shift || true
case "$cmd" in

  pilots)
    if [ "$#" -gt 0 ]; then domains=("$@"); else domains=("${DECOUPLERS[@]}"); fi
    for task in "${domains[@]}"; do
      short=$(short_of "$task")
      for mode in expl_p2e expl_apt expl_random goal; do
        m=${mode#expl_}
        submit pilot.sbatch "pilot_${m}_${short}_seed1" \
          "TASK=$task" "MODE=$mode" "SEED=1" "STEPS=${STEPS:-1e5}" "AXIS=d0"
      done
    done ;;

  pretrain)
    # v3 power policy (plan Sec. power): >=5 seeds/cell on the dose-response.
    read -ra SEEDS <<< "${SEEDS:-1 2 3 4 5}"
    for task in "$CONTROL" "${DECOUPLERS[@]}"; do
      short=$(short_of "$task")
      for mode in expl_p2e expl_apt expl_random; do
        m=${mode#expl_}
        for s in "${SEEDS[@]}"; do
          submit pretrain.sbatch "pretrain_${m}_${short}_seed${s}" \
            "TASK=$task" "MODE=$mode" "SEED=$s" "STEPS=${STEPS:-5e5}"
        done
      done
    done ;;

  adapt)
    pre_run="${1:?usage: adapt <pretrain_run_id> <task> [STEPS]}"
    task="${2:?task}"; steps="${3:-1.25e5}"
    nearest="$RUNROOT/$pre_run/ckpt_snapshots/nearest.json"
    [ -f "$nearest" ] || { echo "no $nearest (run: python -m probing.checkpoint_watcher --select --run_logdir $RUNROOT/$pre_run)"; exit 1; }
    short=$(short_of "$task")
    # De-duplicated (milestone, snapshot) pairs, one adapt job each.
    mapfile -t pairs < <(python -c "
import json,sys
rows=json.load(open('$nearest')); seen=set()
for r in rows:
    s=r.get('snapshot')
    if s and s not in seen:
        seen.add(s); print(r['milestone'], s)
")
    for line in "${pairs[@]}"; do
      ms=${line%% *}; snap=${line#* }
      submit adapt.sbatch "adapt_${short}_seed1_ckpt${ms}" \
        "TASK=$task" "SEED=1" "CKPT=$snap" "STEPS=$steps" "AXIS=dose"
    done ;;

  *)
    echo "usage: $0 {pilots|pretrain|adapt} ..."; exit 1 ;;
esac
