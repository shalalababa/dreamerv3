#!/bin/bash
# ---------------------------------------------------------------------------
# Expand the sweep grids and submit them. Sources env.sh so account/partition/
# paths come from one place and propagate to jobs via --export.
#
#   ./submit_all.sh pilots [dmc_cup_catch dmc_finger_turn_hard]  # Phase 3 Gate 0
#   ./submit_all.sh pretrain                                     # Phase 4 grid
#   ./submit_all.sh pretrain-bundles                             # Phase 4, 3 runs/job
#   ./submit_all.sh adapt <pretrain_run_id> <task> [STEPS]       # Phase 5 dose-response
#
# Dry run: set DRYRUN=1 to print sbatch commands without submitting.
# Duplicate guard: by default, skip a RUN_ID that is already in Slurm or whose
# logdir already exists under RUNROOT. Set FORCE=1 to submit anyway.
# RCC job cap guard: submit only until current Slurm jobs + this invocation
# reaches MAX_JOBS (default 12). Re-run the command after jobs finish.
# Tunables (env): STEPS, SEEDS, DECOUPLERS, CONTROL, MAX_JOBS,
# BUNDLE_SIZE, BUNDLE_TIME.
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

manifest_status() {  # manifest_status <RUN_ID>
  [ -f "$MANIFEST" ] || return 0
  awk -F, -v run_id="$1" '$1 == run_id {status=$9} END {print status}' "$MANIFEST"
}

pretrain_done() {  # pretrain_done <RUN_ID>
  local status
  status="$(manifest_status "$1")"
  [ "$status" = "DONE" ] && return 0
  [ -n "$status" ] && return 1
  [ -f "$RUNROOT/$1/TRAINING_DONE" ] &&
    [ -f "$RUNROOT/$1/ckpt_snapshots/nearest.json" ]
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

  pretrain-bundles|pretrain_bundles)
    # Sequentially pack pretrain runs into fewer Slurm submissions. With RCC's
    # 36h max walltime, 3x 8h runs leaves a useful safety margin.
    read -ra SEEDS <<< "${SEEDS:-1 2 3 4 5}"
    BUNDLE_SIZE="${BUNDLE_SIZE:-3}"
    BUNDLE_TIME="${BUNDLE_TIME:-32:00:00}"
    [ "$BUNDLE_SIZE" -gt 0 ] || { echo "BUNDLE_SIZE must be > 0"; exit 1; }

    pending=()
    for task in "$CONTROL" "${DECOUPLERS[@]}"; do
      short=$(short_of "$task")
      for mode in expl_p2e expl_apt expl_random; do
        m=${mode#expl_}
        for s in "${SEEDS[@]}"; do
          run_id="pretrain_${m}_${short}_seed${s}"
          if [ "${FORCE:-0}" != "1" ]; then
            if queued_job "$run_id"; then
              echo "SKIP queued/running: $run_id"
              continue
            fi
            if pretrain_done "$run_id"; then
              echo "SKIP done: $run_id"
              continue
            fi
          fi
          pending+=("$run_id"$'\t'"$task"$'\t'"$mode"$'\t'"$s"$'\t'"${STEPS:-5e5}")
        done
      done
    done

    if [ "${#pending[@]}" -eq 0 ]; then
      echo "No pretrain runs need submission."
      exit 0
    fi

    runlist_dir="$RUNROOT/_submit_runlists/pretrain_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$runlist_dir"
    bundle=1
    for ((i=0; i<${#pending[@]}; i+=BUNDLE_SIZE)); do
      runlist="$runlist_dir/bundle_$(printf '%03d' "$bundle").tsv"
      : > "$runlist"
      for ((j=i; j<i+BUNDLE_SIZE && j<${#pending[@]}; j++)); do
        printf '%s\n' "${pending[$j]}" >> "$runlist"
      done
      bundle_id="pretrain_bundle_$(printf '%03d' "$bundle")"
      SLURM_TIME="$BUNDLE_TIME" submit pretrain_bundle.sbatch "$bundle_id" \
        "RUNLIST=$runlist" "STEPS=${STEPS:-5e5}"
      bundle=$((bundle + 1))
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
    echo "usage: $0 {pilots|pretrain|pretrain-bundles|adapt} ..."; exit 1 ;;
esac
