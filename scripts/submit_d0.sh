#!/bin/bash
# ---------------------------------------------------------------------------
# Gate D0 submission driver (EVPI note App. A; run from the repo root on
# the cluster). Three subcommands:
#
#   ./scripts/submit_d0.sh stage0
#       Calibration addendum (Latent-UQ Stage 0) on the Phase-4 pretrain
#       grid: cup + finger x {p2e, apt, random} x seeds 1-5 = 30 inference
#       jobs against the frozen v1 probe sets. NO freeze dependency --
#       submit any time. Walker is excluded (no pilots -> no probe set).
#
#   D0_FROZEN=1 ./scripts/submit_d0.sh runs
#       The 24 pre-registered D0 training runs (2 tasks x 4 dose cells x
#       3 seeds, 100K steps, ~20 SU each ~ 0.48K SU total) each with a
#       dependent measure job (sweep + calibration partials). REFUSES to
#       submit unless D0_FROZEN=1: App. A must be frozen (all [prov.]
#       params confirmed, dated entry in note Sec. 6) strictly before the
#       first E2 run launches. Deadline 1 Oct 2026.
#
#   ./scripts/submit_d0.sh measure [RUN_ID ...]
#       (Re)submit measure jobs only, for completed runs (default: all 24).
#
# Task choice [prov.]: cup_catch + finger_turn_hard. App. A.2 says two of
# {walker, cup, finger} chosen at freeze; walker has no pilot buffers so
# no held-out probe set for the calibration partials can be built --
# confirm cup+finger (or build walker pilots first) at the freeze.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
cd "$REPO"

CMD="${1:-}"; shift || true

declare -A DMC_TASK=( [cup]=dmc_cup_catch [finger]=dmc_finger_turn_hard )
# Dose cells: label -> named config ("" = E1 dose zero). Note App. A.2.
DOSE_LABELS=(e1 d8x1 d32x1 d32x3)
declare -A DOSE_CFG=( [e1]="" [d8x1]=d0_dose1 [d32x1]=d0_dose2 [d32x3]=d0_dose3 )
TASKS=(cup finger)
SEEDS=(1 2 3)

probeset_for() {  # probeset_for <task> <dose_label> <seed> -> leave-one-out set
  echo "$RUNROOT/probesets/d0_${1}_${2}_wo${3}_v1"
}

siblings_for() {  # siblings_for <task> <dose_label> <seed> -> labeled replays
  local out=""
  for s in "${SEEDS[@]}"; do
    [ "$s" = "$3" ] && continue
    out+="s${s}=$RUNROOT/d0_${1}_${2}_seed${s}/replay "
  done
  echo "$out"
}

submit_measure() {  # submit_measure <task> <dose_label> <seed> [dependency]
  local task=$1 dose=$2 seed=$3 dep=${4:-}
  local run_id="d0_${task}_${dose}_seed${seed}"
  sbatch ${dep:+--dependency=afterok:$dep} \
    --export=ALL,REPO="$REPO",RUN_ID="$run_id",TASK="${DMC_TASK[$task]}",PROBESET="$(probeset_for "$task" "$dose" "$seed")",SIBLINGS="$(siblings_for "$task" "$dose" "$seed")" \
    scripts/d0_measure.sbatch
}

case "$CMD" in
  stage0)
    for task in "${TASKS[@]}"; do
      for mode in p2e apt random; do
        for seed in 1 2 3 4 5; do
          sbatch --export=ALL,REPO="$REPO",PRE_RUN="pretrain_${mode}_${task}_seed${seed}",PROBESET="$RUNROOT/probesets/${task}_v1" \
            scripts/d0_uq_pretrain.sbatch
        done
      done
    done
    echo "Submitted 30 Stage 0 addendum jobs (cup+finger x 3 modes x 5 seeds)."
    ;;

  runs)
    if [ "${D0_FROZEN:-0}" != "1" ]; then
      echo "REFUSED: App. A is not marked frozen. Confirm every [prov.]"
      echo "parameter, add the dated freeze entry to the note Sec. 6, then"
      echo "re-run with D0_FROZEN=1. (Pre-reg: freeze strictly precedes the"
      echo "first E2 launch; deadline 1 Oct 2026.)"
      exit 1
    fi
    for task in "${TASKS[@]}"; do
      for dose in "${DOSE_LABELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          run_id="d0_${task}_${dose}_seed${seed}"
          jid=$(sbatch --parsable \
            --export=ALL,REPO="$REPO",RUN_ID="$run_id",TASK="${DMC_TASK[$task]}",DOSE="${DOSE_CFG[$dose]}",SEED="$seed",AXIS=d0,PAIRED_SEED_SET="d0_${task}_seed${seed}" \
            scripts/d0.sbatch)
          submit_measure "$task" "$dose" "$seed" "$jid"
          echo "submitted $run_id (train $jid + dependent measure)"
        done
      done
    done
    echo "Submitted 24 train + 24 measure jobs."
    ;;

  measure)
    if [ "$#" -gt 0 ]; then
      for run_id in "$@"; do
        # d0_<task>_<dose>_seed<N>
        IFS=_ read -r _ task dose seedtok <<< "$run_id"
        submit_measure "$task" "$dose" "${seedtok#seed}"
        echo "submitted measure for $run_id"
      done
    else
      for task in "${TASKS[@]}"; do
        for dose in "${DOSE_LABELS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            submit_measure "$task" "$dose" "$seed"
          done
        done
      done
      echo "Submitted 24 measure jobs."
    fi
    ;;

  *)
    echo "usage: $0 stage0 | runs | measure [RUN_ID ...]   (see header)"
    exit 1
    ;;
esac
