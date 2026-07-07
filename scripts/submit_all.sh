#!/bin/bash
# ---------------------------------------------------------------------------
# Expand the sweep grids and submit them. Sources env.sh so account/partition/
# paths come from one place and propagate to jobs via --export.
#
#   ./submit_all.sh pilots [dmc_cup_catch dmc_finger_turn_hard]  # Phase 3 Gate 0
#   ./submit_all.sh pretrain                                     # Phase 4 grid
#   ./submit_all.sh pretrain-bundles                             # Phase 4, 3 runs/job
#   ./submit_all.sh adapt <pretrain_run_id> <task> [STEPS]       # Phase 5 dose-response
#   ./submit_all.sh adapt-completed [STEPS]                      # Phase 5 rolling 125K sweep
#   ./submit_all.sh adapt-bundles [STEPS]                        # Phase 5 bundled sweep
#   ./submit_all.sh measure                                      # Phase 5 driver measurement
#   ./submit_all.sh measure-bundles                              # Phase 5 bundled measurement
#   ./submit_all.sh axis1                                        # Phase 6 offline+adapt grid
#
# Dry run: set DRYRUN=1 to print sbatch commands without submitting.
# Duplicate guard: by default, skip a RUN_ID that is already in Slurm or whose
# logdir already exists under RUNROOT. Set FORCE=1 to submit anyway.
# RCC job cap guard: submit only until current Slurm jobs + this invocation
# reaches MAX_JOBS (default 12). Re-run the command after jobs finish.
# Tunables (env): STEPS, SEEDS, DECOUPLERS, CONTROL, MAX_JOBS,
# BUNDLE_SIZE, BUNDLE_TIME, ADAPT_PRETRAINS, ADAPT_MILESTONES,
# ADAPT_BUNDLE_TIME, ADAPT_BUNDLE_MINUTES, ADAPT_BUNDLE_BUFFER_MINUTES,
# ADAPT_EST_*_MINUTES, MEASURE_PRETRAINS, MEASURE_BUNDLE_SIZE,
# MEASURE_EST_MINUTES, MEASURE_BUNDLE_BUFFER_MINUTES, AXIS1_SEEDS,
# AXIS1_QUADS, AXIS1_DOMAINS, AXIS1_UPDATES, AXIS1_TIME.
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

adapt_done() {  # adapt_done <RUN_ID>
  local status
  status="$(manifest_status "$1")"
  [ "$status" = "DONE" ] && return 0
  [ -f "$RUNROOT/$1/ADAPT_DONE" ]
}

measure_done() {  # measure_done <pretrain_run_id>
  [ -f "$RUNROOT/$1/measure/MEASURE_DONE" ]
}

measure_reserved() {  # measure_reserved <pretrain_run_id>
  [ -f "$RUNROOT/$1/measure/SUBMITTED_BY_BUNDLE" ]
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

source_of_pretrain() {  # source_of_pretrain <pretrain_run_id>
  echo "$1" | sed -E 's/^pretrain_//'
}

seed_of_pretrain() {  # seed_of_pretrain <pretrain_run_id>
  local seed
  seed="$(echo "$1" | sed -nE 's/.*_seed([0-9]+)$/\1/p')"
  [ -n "$seed" ] || { echo "could not parse seed from $1" >&2; return 1; }
  printf '%s\n' "$seed"
}

task_of_pretrain() {  # task_of_pretrain <pretrain_run_id>
  local run_id="$1"
  local task=""
  if [ -f "$MANIFEST" ]; then
    task="$(awk -F, -v run_id="$run_id" '$1 == run_id {task=$3} END {print task}' "$MANIFEST")"
  fi
  if [ -n "$task" ]; then
    printf '%s\n' "$task"
    return 0
  fi
  case "$run_id" in
    *_walker_seed*) echo "$CONTROL" ;;
    *_cup_seed*) echo "dmc_cup_catch" ;;
    *_finger_seed*) echo "dmc_finger_turn_hard" ;;
    *) echo "could not infer task for $run_id" >&2; return 1 ;;
  esac
}

adapt_pairs_from_nearest() {  # adapt_pairs_from_nearest <nearest.json>
  python - "$1" "${ADAPT_MILESTONES:-100000 200000 300000 400000 500000}" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1]))
wanted = {int(x) for x in sys.argv[2].split()}
seen = set()
for row in rows:
    snap = row.get("snapshot")
    milestone = row.get("milestone")
    if not snap or milestone is None or snap in seen:
        continue
    if int(milestone) not in wanted:
        continue
    seen.add(snap)
    print(f"{int(milestone)}\t{snap}")
PY
}

adapt_est_minutes() {  # adapt_est_minutes <task> <steps>
  local task="$1"; local steps="$2"; local base
  case "$task" in
    *walker*) base="${ADAPT_EST_WALKER_MINUTES:-72}" ;;
    *cup*) base="${ADAPT_EST_CUP_MINUTES:-72}" ;;
    *finger*) base="${ADAPT_EST_FINGER_MINUTES:-72}" ;;
    *) base="${ADAPT_EST_MINUTES:-72}" ;;
  esac
  python - "$base" "$steps" <<'PY'
import math
import sys

base = float(sys.argv[1])
steps = float(sys.argv[2])
print(int(math.ceil(base * steps / 125000.0)))
PY
}

record_adapt_bundle_reservations() {  # record_adapt_bundle_reservations <runlist> <bundle_id>
  local runlist="$1"; local bundle_id="$2"; local stamp
  stamp="$(date -Is)"
  while IFS=$'\t' read -r child_run_id task seed ckpt steps est_minutes; do
    [ -n "${child_run_id:-}" ] || continue
    mkdir -p "$RUNROOT/$child_run_id"
    printf 'bundle=%s\nreserved_at=%s\nrunlist=%s\n' \
      "$bundle_id" "$stamp" "$runlist" > "$RUNROOT/$child_run_id/SUBMITTED_BY_BUNDLE"
  done < "$runlist"
}

adapt_bundle_walltime() {  # adapt_bundle_walltime <estimated_minutes>
  if [ -n "${ADAPT_BUNDLE_TIME:-}" ]; then
    printf '%s\n' "$ADAPT_BUNDLE_TIME"
    return 0
  fi
  local minutes="$1"
  local buffer="${ADAPT_BUNDLE_BUFFER_MINUTES:-240}"
  local min_minutes="${ADAPT_BUNDLE_MIN_TIME_MINUTES:-120}"
  local max_minutes="${ADAPT_BUNDLE_MAX_TIME_MINUTES:-2040}"
  minutes=$((minutes + buffer))
  [ "$minutes" -ge "$min_minutes" ] || minutes="$min_minutes"
  [ "$minutes" -le "$max_minutes" ] || minutes="$max_minutes"
  printf '%02d:%02d:00\n' $((minutes / 60)) $((minutes % 60))
}

submit_adapt_bundle() {  # submit_adapt_bundle <bundle_id> <runlist> <estimated_minutes>
  local bundle_id="$1"; local runlist="$2"
  local estimated_minutes="$3"
  local walltime
  walltime="$(adapt_bundle_walltime "$estimated_minutes")"
  local jobs_used=$((JOBS_AT_START + SUBMITTED_THIS_RUN))
  if [ "${IGNORE_JOB_CAP:-0}" != "1" ] && [ "$MAX_JOBS" -gt 0 ] &&
     [ "$jobs_used" -ge "$MAX_JOBS" ]; then
    echo "STOP job cap reached: $jobs_used/$MAX_JOBS Slurm jobs active/submitted."
    echo "Re-run this command after some jobs finish; existing/queued RUN_IDs will be skipped."
    exit 0
  fi

  local exports="ALL,REPO=$REPO,RUN_ID=$bundle_id,RUNLIST=$runlist,STEPS=${STEPS:-1.25e5}"
  local cmd=(sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION"
             --gres="$SLURM_GRES" --time="$walltime"
             --job-name="$bundle_id" --export="$exports"
             "$REPO/scripts/adapt_bundle.sbatch")
  if [ "${DRYRUN:-0}" = "1" ]; then
    printf '%q ' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
    record_adapt_bundle_reservations "$runlist" "$bundle_id"
  fi
  SUBMITTED_THIS_RUN=$((SUBMITTED_THIS_RUN + 1))
}

measure_bundle_walltime() {  # measure_bundle_walltime <num_tasks>
  if [ -n "${MEASURE_BUNDLE_TIME:-}" ]; then
    printf '%s\n' "$MEASURE_BUNDLE_TIME"
    return 0
  fi
  local tasks="$1"
  local est="${MEASURE_EST_MINUTES:-180}"
  local buffer="${MEASURE_BUNDLE_BUFFER_MINUTES:-120}"
  local minutes=$((tasks * est + buffer))
  printf '%02d:%02d:00\n' $((minutes / 60)) $((minutes % 60))
}

record_measure_bundle_reservations() {  # record_measure_bundle_reservations <runlist> <bundle_id>
  local runlist="$1"; local bundle_id="$2"; local stamp
  stamp="$(date -Is)"
  while IFS=$'\t' read -r pre_run task ref_replay critic; do
    [ -n "${pre_run:-}" ] || continue
    mkdir -p "$RUNROOT/$pre_run/measure"
    printf 'bundle=%s\nreserved_at=%s\nrunlist=%s\n' \
      "$bundle_id" "$stamp" "$runlist" > "$RUNROOT/$pre_run/measure/SUBMITTED_BY_BUNDLE"
  done < "$runlist"
}

submit_measure_bundle() {  # submit_measure_bundle <bundle_id> <runlist> <num_tasks>
  local bundle_id="$1"; local runlist="$2"; local num_tasks="$3"
  local walltime
  walltime="$(measure_bundle_walltime "$num_tasks")"
  local jobs_used=$((JOBS_AT_START + SUBMITTED_THIS_RUN))
  if [ "${IGNORE_JOB_CAP:-0}" != "1" ] && [ "$MAX_JOBS" -gt 0 ] &&
     [ "$jobs_used" -ge "$MAX_JOBS" ]; then
    echo "STOP job cap reached: $jobs_used/$MAX_JOBS Slurm jobs active/submitted."
    echo "Re-run this command after some jobs finish; existing/queued RUN_IDs will be skipped."
    exit 0
  fi

  local exports="ALL,REPO=$REPO,RUN_ID=$bundle_id,RUNLIST=$runlist"
  local cmd=(sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION"
             --gres="$SLURM_GRES" --time="$walltime"
             --job-name="$bundle_id" --export="$exports"
             "$REPO/scripts/measure_bundle.sbatch")
  if [ "${DRYRUN:-0}" = "1" ]; then
    printf '%q ' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
    record_measure_bundle_reservations "$runlist" "$bundle_id"
  fi
  SUBMITTED_THIS_RUN=$((SUBMITTED_THIS_RUN + 1))
}

submit_adapts_for_pretrain() {  # submit_adapts_for_pretrain <pretrain_run_id> <task> <steps>
  local pre_run="$1"; local task="$2"; local steps="$3"
  local nearest="$RUNROOT/$pre_run/ckpt_snapshots/nearest.json"
  [ -f "$nearest" ] || {
    echo "SKIP no nearest.json: $pre_run"
    echo "  run: python -m probing.checkpoint_watcher --select --run_logdir $RUNROOT/$pre_run"
    return 0
  }

  local source seed
  source="$(source_of_pretrain "$pre_run")"
  seed="$(seed_of_pretrain "$pre_run")"

  mapfile -t pairs < <(adapt_pairs_from_nearest "$nearest")
  if [ "${#pairs[@]}" -eq 0 ]; then
    echo "SKIP no selected snapshots: $pre_run"
    return 0
  fi

  for line in "${pairs[@]}"; do
    IFS=$'\t' read -r ms snap <<< "$line"
    submit adapt.sbatch "adapt_${source}_ckpt${ms}" \
      "TASK=$task" "SEED=$seed" "CKPT=$snap" "STEPS=$steps" "AXIS=dose"
  done
}

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
    submit_adapts_for_pretrain "$pre_run" "$task" "$steps" ;;

  adapt-completed|adapt_completed|adapt-125k|adapt_125k)
    steps="${1:-1.25e5}"
    if [ -n "${ADAPT_PRETRAINS:-}" ]; then
      read -ra pretrains <<< "$ADAPT_PRETRAINS"
    else
      mapfile -t pretrains < <(
        find "$RUNROOT" -maxdepth 1 -type d -name 'pretrain_*' -printf '%f\n' 2>/dev/null |
          sort
      )
    fi

    if [ "${#pretrains[@]}" -eq 0 ]; then
      echo "No pretrain runs found."
      exit 0
    fi

    for pre_run in "${pretrains[@]}"; do
      if pretrain_done "$pre_run"; then
        task="$(task_of_pretrain "$pre_run")"
        submit_adapts_for_pretrain "$pre_run" "$task" "$steps"
      else
        echo "SKIP not done: $pre_run"
      fi
    done ;;

  adapt-bundles|adapt_bundles|adapt-completed-bundles|adapt_completed_bundles)
    steps="${1:-1.25e5}"
    max_minutes="${ADAPT_BUNDLE_MINUTES:-1800}"
    [ "$max_minutes" -gt 0 ] || { echo "ADAPT_BUNDLE_MINUTES must be > 0"; exit 1; }

    if [ -n "${ADAPT_PRETRAINS:-}" ]; then
      read -ra pretrains <<< "$ADAPT_PRETRAINS"
    else
      mapfile -t pretrains < <(
        find "$RUNROOT" -maxdepth 1 -type d -name 'pretrain_*' -printf '%f\n' 2>/dev/null |
          sort
      )
    fi

    pending=()
    for pre_run in "${pretrains[@]}"; do
      if ! pretrain_done "$pre_run"; then
        echo "SKIP not done: $pre_run"
        continue
      fi
      task="$(task_of_pretrain "$pre_run")"
      source="$(source_of_pretrain "$pre_run")"
      seed="$(seed_of_pretrain "$pre_run")"
      nearest="$RUNROOT/$pre_run/ckpt_snapshots/nearest.json"
      mapfile -t pairs < <(adapt_pairs_from_nearest "$nearest")
      for line in "${pairs[@]}"; do
        IFS=$'\t' read -r ms snap <<< "$line"
        child_run_id="adapt_${source}_ckpt${ms}"
        if [ "${FORCE:-0}" != "1" ]; then
          if queued_job "$child_run_id"; then
            echo "SKIP queued/running: $child_run_id"
            continue
          fi
          if adapt_done "$child_run_id"; then
            echo "SKIP done: $child_run_id"
            continue
          fi
          if existing_logdir "$child_run_id"; then
            echo "SKIP existing/reserved logdir: $RUNROOT/$child_run_id  (set FORCE=1 to resubmit)"
            continue
          fi
        fi
        est="$(adapt_est_minutes "$task" "$steps")"
        pending+=("$child_run_id"$'\t'"$task"$'\t'"$seed"$'\t'"$snap"$'\t'"$steps"$'\t'"$est")
      done
    done

    if [ "${#pending[@]}" -eq 0 ]; then
      echo "No adapt runs need submission."
      exit 0
    fi

    stamp="$(date +%Y%m%d_%H%M%S)"
    runlist_dir="$RUNROOT/_submit_runlists/adapt_$stamp"
    bundle_prefix="adapt_bundle_$stamp"
    mkdir -p "$runlist_dir"
    bundle=1
    runlist=""
    bundle_minutes=0
    bundle_rows=0
    flush_bundle() {
      [ "$bundle_rows" -gt 0 ] || return 0
      bundle_id="${bundle_prefix}_$(printf '%03d' "$bundle")"
      echo "Bundle $bundle_id estimated minutes: $bundle_minutes"
      submit_adapt_bundle "$bundle_id" "$runlist" "$bundle_minutes"
      bundle=$((bundle + 1))
      runlist=""
      bundle_minutes=0
      bundle_rows=0
    }

    for row in "${pending[@]}"; do
      IFS=$'\t' read -r _ _ _ _ _ est <<< "$row"
      if [ "$bundle_rows" -gt 0 ] && [ $((bundle_minutes + est)) -gt "$max_minutes" ]; then
        flush_bundle
      fi
      if [ "$bundle_rows" -eq 0 ]; then
        runlist="$runlist_dir/bundle_$(printf '%03d' "$bundle").tsv"
        : > "$runlist"
      fi
      printf '%s\n' "$row" >> "$runlist"
      bundle_minutes=$((bundle_minutes + est))
      bundle_rows=$((bundle_rows + 1))
    done
    flush_bundle ;;

  measure|measure-completed|measure_completed)
    # Phase 5 driver measurement (PREREG_phase5a Sec. 3): one job per
    # completed pretrain run; skips runs whose measure/MEASURE_DONE exists.
    # Fit the per-domain critics first (cheap, login node):
    #   python -m probing.value_sensitive fit-critic ...
    if [ -n "${MEASURE_PRETRAINS:-}" ]; then
      read -ra pretrains <<< "$MEASURE_PRETRAINS"
    else
      mapfile -t pretrains < <(
        find "$RUNROOT" -maxdepth 1 -type d -name 'pretrain_*' -printf '%f\n' 2>/dev/null |
          sort
      )
    fi
    for pre_run in "${pretrains[@]}"; do
      if ! pretrain_done "$pre_run"; then
        echo "SKIP not done: $pre_run"
        continue
      fi
      if [ -f "$RUNROOT/$pre_run/measure/MEASURE_DONE" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "SKIP measured: $pre_run"
        continue
      fi
      task="$(task_of_pretrain "$pre_run")"
      short="$(short_of "$task")"
      if [ "$short" = "walker" ]; then
        ref="$RUNROOT/pretrain_random_walker_seed1/replay"
      else
        ref="$RUNROOT/pilot_goal_${short}_seed1/replay"
      fi
      critic="$RUNROOT/critics/critic_${short}_v1.npz"
      submit measure.sbatch "measure_$(source_of_pretrain "$pre_run")" \
        "PRE_RUN=$pre_run" "TASK=$task" "REF_REPLAY=$ref" "CRITIC=$critic" \
        "AXIS=dose"
    done ;;

  measure-bundles|measure_bundles|measure-completed-bundles|measure_completed_bundles)
    # Phase 5 driver measurement bundled for RCC caps. Defaults: five
    # pretrain runs per bundle, 3h per run, +2h buffer -> 17h full bundle.
    bundle_size="${MEASURE_BUNDLE_SIZE:-5}"
    [ "$bundle_size" -gt 0 ] || { echo "MEASURE_BUNDLE_SIZE must be > 0"; exit 1; }

    if [ -n "${MEASURE_PRETRAINS:-}" ]; then
      read -ra pretrains <<< "$MEASURE_PRETRAINS"
    else
      mapfile -t pretrains < <(
        find "$RUNROOT" -maxdepth 1 -type d -name 'pretrain_*' -printf '%f\n' 2>/dev/null |
          sort
      )
    fi

    pending=()
    for pre_run in "${pretrains[@]}"; do
      if ! pretrain_done "$pre_run"; then
        echo "SKIP not done: $pre_run"
        continue
      fi
      if [ "${FORCE:-0}" != "1" ]; then
        if queued_job "measure_$(source_of_pretrain "$pre_run")"; then
          echo "SKIP queued/running: measure_$(source_of_pretrain "$pre_run")"
          continue
        fi
        if measure_done "$pre_run"; then
          echo "SKIP measured: $pre_run"
          continue
        fi
        if measure_reserved "$pre_run"; then
          echo "SKIP reserved in bundle: $pre_run"
          continue
        fi
      fi
      task="$(task_of_pretrain "$pre_run")"
      short="$(short_of "$task")"
      if [ "$short" = "walker" ]; then
        ref="$RUNROOT/pretrain_random_walker_seed1/replay"
      else
        ref="$RUNROOT/pilot_goal_${short}_seed1/replay"
      fi
      critic="$RUNROOT/critics/critic_${short}_v1.npz"
      pending+=("$pre_run"$'\t'"$task"$'\t'"$ref"$'\t'"$critic")
    done

    if [ "${#pending[@]}" -eq 0 ]; then
      echo "No measure runs need submission."
      exit 0
    fi

    stamp="$(date +%Y%m%d_%H%M%S)"
    runlist_dir="$RUNROOT/_submit_runlists/measure_$stamp"
    bundle_prefix="measure_bundle_$stamp"
    mkdir -p "$runlist_dir"
    bundle=1
    for ((i=0; i<${#pending[@]}; i+=bundle_size)); do
      runlist="$runlist_dir/bundle_$(printf '%03d' "$bundle").tsv"
      : > "$runlist"
      rows=0
      for ((j=i; j<i+bundle_size && j<${#pending[@]}; j++)); do
        printf '%s\n' "${pending[$j]}" >> "$runlist"
        rows=$((rows + 1))
      done
      bundle_id="${bundle_prefix}_$(printf '%03d' "$bundle")"
      echo "Bundle $bundle_id tasks: $rows walltime: $(measure_bundle_walltime "$rows")"
      submit_measure_bundle "$bundle_id" "$runlist" "$rows"
      bundle=$((bundle + 1))
    done ;;

  axis1)
    # Phase 6 Axis-1 (runbook items 3-4): for every built controlled buffer
    # side, retrain a fresh WM offline (gradient count equalized) and adapt
    # frozen-readout, one job per (domain, quadrant, side, seed). Requires
    # the buffers built first (login node, CPU-only, minutes):
    #   python -m probing.build_controlled_replay build \
    #     --index $RUNROOT/axis1_<dom>/episodes.json \
    #     --pairs $RUNROOT/axis1_<dom>/pairs.json \
    #     --which q1 --output_root $RUNROOT/axis1_<dom>/q1   (and q2)
    # Seeds are paired across cells (same k everywhere); deepen Q1 later via
    # AXIS1_SEEDS="9 10" AXIS1_QUADS=q1 (plan v3 surprise table).
    read -ra seeds <<< "${AXIS1_SEEDS:-1 2 3 4 5 6 7 8}"
    read -ra quads <<< "${AXIS1_QUADS:-q1 q2}"
    read -ra doms <<< "${AXIS1_DOMAINS:-cup finger}"
    updates="${AXIS1_UPDATES:-500000}"
    steps="${STEPS:-1.25e5}"
    for dom in "${doms[@]}"; do
      case "$dom" in
        cup) task=dmc_cup_catch ;;
        finger) task=dmc_finger_turn_hard ;;
        *) echo "unknown axis1 domain: $dom"; exit 1 ;;
      esac
      for q in "${quads[@]}"; do
        root="$RUNROOT/axis1_${dom}/${q}"
        if [ ! -f "$root/manifest.json" ]; then
          echo "SKIP not built: $root (run build_controlled_replay build --which $q)"
          continue
        fi
        for side in 0 1; do
          for s in "${seeds[@]}"; do
            run_id="adapt_ax1${q}s${side}_${dom}_seed${s}_ckpt${updates}"
            SLURM_TIME="${AXIS1_TIME:-12:00:00}" submit axis1.sbatch "$run_id" \
              "WM_RUN=ax1wm_${dom}_${q}s${side}_seed${s}" \
              "TASK=$task" "SEED=$s" "REPLAY=$root/side${side}" \
              "UPDATES=$updates" "STEPS=$steps" \
              "AXIS=axis1" "PAIRED_SEED_SET=ax1_${dom}_${q}"
          done
        done
      done
    done ;;

  *)
    echo "usage: $0 {pilots|pretrain|pretrain-bundles|adapt|adapt-completed|adapt-bundles|measure|measure-bundles|axis1} ..."; exit 1 ;;
esac
