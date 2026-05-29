#!/bin/bash
# Submit project jobs in grouped batches.
#
#   ./scripts/submit_all.sh <phase>
#
# Groups:
#   smoke         optional 20K checks for p2e and apt
#   phase1-gate   pretrain p2e and apt, seed 1 only
#   phase1-rest   pretrain random x2, p2e seed2, apt seed2
#   c1            C1 from-scratch: stand + run, seeds 1 & 2
#   phase2-adapt  frozen-readout adaptation for C2 + C3 (12 runs)
#   phase3-adapt  frozen-readout adaptation for C4 (6 runs)
#   list          print the full job matrix without submitting
#
# Seeds: this study uses seeds {1, 2} for every condition, matching the two
# existing 500K walker_walk runs that serve as C1/walk (no new C1/walk job).

set -eo pipefail
cd "$(dirname "$0")/.."
SEEDS=(1 2)

submit () {  # submit <job-name> <script> <args...>
  local name="$1"; shift
  local script="$1"; shift
  echo "  sbatch --job-name=${name} ${script} $*"
  sbatch --job-name="${name}" "${script}" "$@"
}

phase_smoke () {
  echo "[smoke] optional 20K pre-flight runs (~45 min each):"
  submit "smoke-p2e" scripts/pretrain_smoke.sbatch p2e
  submit "smoke-apt" scripts/pretrain_smoke.sbatch apt
}

phase1_gate () {
  echo "[phase1-gate] pretrain p2e + apt seed 1."
  echo "  After these finish, check world-model losses and intrinsic-reward"
  echo "  metrics before submitting the remaining pretraining runs."
  submit "pretrain-p2e-seed1" scripts/pretrain.sbatch p2e 1
  submit "pretrain-apt-seed1" scripts/pretrain.sbatch apt 1
}

phase1_rest () {
  echo "[phase1-rest] remaining pretraining runs:"
  submit "pretrain-p2e-seed2" scripts/pretrain.sbatch p2e 2
  submit "pretrain-apt-seed2" scripts/pretrain.sbatch apt 2
  for s in "${SEEDS[@]}"; do
    submit "pretrain-random-seed${s}" scripts/pretrain.sbatch random "${s}"
  done
}

phase_c1 () {
  echo "[c1] C1 from-scratch -- stand and run only (walk is reused):"
  for task in stand run; do
    for s in "${SEEDS[@]}"; do
      submit "c1-${task}-seed${s}" scripts/c1_scratch.sbatch "${task}" "${s}"
    done
  done
}

phase2_adapt () {
  echo "[phase2-adapt] frozen-readout adaptation for C2 (random) + C3 (p2e):"
  echo "  Requires the matching pretraining runs to have finished."
  for cond in random p2e; do
    for task in stand walk run; do
      for s in "${SEEDS[@]}"; do
        submit "adapt-${cond}-${task}-seed${s}" \
          scripts/adapt.sbatch "${cond}" "${task}" "${s}"
      done
    done
  done
}

phase3_adapt () {
  echo "[phase3-adapt] frozen-readout adaptation for C4 (apt):"
  echo "  Requires pretrain-apt-seed{1,2} to have finished."
  for task in stand walk run; do
    for s in "${SEEDS[@]}"; do
      submit "adapt-apt-${task}-seed${s}" \
        scripts/adapt.sbatch apt "${task}" "${s}"
    done
  done
}

phase_list () {
  cat <<'EOF'
Full job matrix (28 runs):

  Pretraining (6 runs, 500K reward-free steps each)
    pretrain.sbatch p2e 1        pretrain.sbatch p2e 2
    pretrain.sbatch apt 1        pretrain.sbatch apt 2
    pretrain.sbatch random 1     pretrain.sbatch random 2

  C1 from-scratch (4 runs, 500K steps each; walk reused)
    c1_scratch.sbatch stand 1    c1_scratch.sbatch stand 2
    c1_scratch.sbatch run 1      c1_scratch.sbatch run 2

  Frozen-readout adaptation, C2 + C3 (12 runs, 250K steps each)
    adapt.sbatch random {stand,walk,run} {1,2}
    adapt.sbatch p2e    {stand,walk,run} {1,2}

  Frozen-readout adaptation, C4 (6 runs, 250K steps each)
    adapt.sbatch apt    {stand,walk,run} {1,2}

Existing assets reused (no new job): C1/walk = the two 500K walker_walk
runs (seeds 1 & 2); the 1.1M walker_walk run is the long-budget reference.
EOF
}

case "${1:-}" in
  smoke)        phase_smoke ;;
  phase1-gate)  phase1_gate ;;
  phase1-rest)  phase1_rest ;;
  c1)           phase_c1 ;;
  phase2-adapt) phase2_adapt ;;
  phase3-adapt) phase3_adapt ;;
  list)         phase_list ;;
  *)
    echo "Usage: $0 <smoke|phase1-gate|phase1-rest|c1|phase2-adapt|phase3-adapt|list>"
    echo "Run 'list' to see the full job matrix."
    exit 1 ;;
esac
