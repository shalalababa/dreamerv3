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
# Tunables (env): STEPS, SEEDS, DECOUPLERS, CONTROL.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
source "$REPO/scripts/env.sh"

submit() {  # submit <script> <RUN_ID> KEY=VAL ...
  local script="$1"; local run_id="$2"; shift 2
  local exports="ALL,REPO=$REPO,RUN_ID=$run_id"
  for kv in "$@"; do exports="$exports,$kv"; done
  local cmd=(sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION"
             --gres="$SLURM_GRES" --time="$SLURM_TIME"
             --job-name="$run_id" --export="$exports"
             "$REPO/scripts/$script")
  if [ "${DRYRUN:-0}" = "1" ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
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
    read -ra SEEDS <<< "${SEEDS:-1 2}"
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
