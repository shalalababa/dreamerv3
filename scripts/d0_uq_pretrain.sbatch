#!/bin/bash
#SBATCH --job-name=dv3_uq
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Latent-UQ Stage 0 dump on ONE Phase-4 pretrain run = the Gate D0
# calibration addendum host (EVPI note App. A.4 item 5; Latent-UQ brief
# Sec. 6 Stage 0). Pure inference over the retained milestone snapshots --
# no freeze dependency, submittable now. p2e runs dump the trained disag
# signal; apt/random runs dump rollout error + density only (cross-seed
# disagreement is computed later in latent_uq_analysis from pred_*).
# Requires a frozen probe set for the domain (cup_v1 / finger_v1 exist;
# walker has no pilots, hence no probe set -- excluded).
#
# Inference-only: fine on CPU too (submit with
#   sbatch --partition=<cpu partition> --gres=none --export=...,PLATFORM=cpu).
#
# Required env:
#   PRE_RUN   e.g. pretrain_p2e_cup_seed1
#   PROBESET  e.g. $RUNROOT/probesets/cup_v1
# Optional: MILESTONES ("100000 ... 500000"), PLATFORM (cuda), REF_REPLAY
#           (defaults to the run's own replay inside latent_uq)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${PRE_RUN:?set PRE_RUN}"; : "${PROBESET:?set PROBESET}"
MILESTONES="${MILESTONES:-100000 200000 300000 400000 500000}"
LOGDIR="$RUNROOT/$PRE_RUN"
[ -d "$PROBESET" ] || { echo "ERROR: no probe set at $PROBESET"; exit 2; }
OUT="$LOGDIR/latent_uq/$(basename "$PROBESET")"
if [ -f "$OUT/index.json" ]; then
  echo "SKIP: $OUT already dumped"; exit 0
fi
echo "[$(date)] latent_uq PRE_RUN=$PRE_RUN probeset=$PROBESET milestones=$MILESTONES"

python -m probing.latent_uq --probeset "$PROBESET" \
  --run_logdir "$LOGDIR" --milestones $MILESTONES \
  --output "$OUT" \
  ${PLATFORM:+--platform "$PLATFORM"} \
  ${REF_REPLAY:+--ref_replay "$REF_REPLAY"}

echo "[$(date)] latent_uq $PRE_RUN DONE -> $OUT"
