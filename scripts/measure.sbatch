#!/bin/bash
#SBATCH --job-name=dv3_measure
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Phase 5: driver measurement at each retained milestone of one pretrain run
# (PREREG_phase5a Sec. 3). Per milestone:
#   collect (frozen-policy eval rollouts, 40 eps, seed 0)
#     -> features (horizons 1 5 20) -> probe (fwd/geom/retdec/regime R^2)
#     -> value_sensitive measure (VSA, if the domain critic exists)
# plus, once per run, buffer-side drivers (cov / occ_phys / occ_rew) on the
# pretraining-replay prefixes via probing/measure_drivers.py.
#
# Required env:
#   PRE_RUN  pretrain run id, e.g. pretrain_p2e_cup_seed1
#   TASK     dmc task, e.g. dmc_cup_catch
#   REF_REPLAY  fixed per-domain reference buffer (goal pilot replay;
#               walker: pretrain_random_walker_seed1/replay)
# Optional:
#   MILESTONES  default "100000 200000 300000 400000 500000"
#   CRITIC      critic npz from `value_sensitive fit-critic`; VSA skipped
#               (and retdec used per PREREG Sec. 4) if unset/missing
#   EPISODES    default 40
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${PRE_RUN:?set PRE_RUN}"; : "${TASK:?set TASK}"; : "${REF_REPLAY:?set REF_REPLAY}"
MILESTONES="${MILESTONES:-100000 200000 300000 400000 500000}"
EPISODES="${EPISODES:-40}"
LOGDIR="$RUNROOT/$PRE_RUN"
MEAS="$LOGDIR/measure"
NEAREST="$LOGDIR/ckpt_snapshots/nearest.json"
[ -f "$NEAREST" ] || { echo "ERROR: no $NEAREST"; exit 2; }
mkdir -p "$MEAS"
echo "[$(date)] measure PRE_RUN=$PRE_RUN TASK=$TASK milestones=$MILESTONES"

snapshot_of() {  # snapshot_of <milestone> -> path (empty if >5000 steps off)
  python - "$NEAREST" "$1" <<'EOF'
import json, sys
entries = json.load(open(sys.argv[1]))
ms = int(sys.argv[2])
hit = min(entries, key=lambda e: abs(e['milestone'] - ms))
if hit['milestone'] != ms or hit['abs_error'] > 5000:
    sys.exit(0)
print(hit['snapshot'])
EOF
}

for MS in $MILESTONES; do
  OUT="$MEAS/ms$MS"
  if [ -f "$OUT/probe/probe_results.csv" ]; then
    echo "SKIP done: $OUT"
    continue
  fi
  SNAP="$(snapshot_of "$MS")"
  if [ -z "$SNAP" ]; then
    echo "SKIP milestone $MS: no snapshot within 5000 steps (excluded per PREREG Sec.1)"
    continue
  fi
  [ -f "$SNAP/done" ] || { echo "ERROR: $SNAP has no 'done' marker"; exit 2; }
  mkdir -p "$OUT"
  echo "[$(date)] milestone $MS  snapshot=$SNAP"
  python -m probing.collect --run_logdir "$LOGDIR" --checkpoint "$SNAP" \
    --output "$OUT/traj.npz" --episodes "$EPISODES" --mode eval --seed 0
  python -m probing.features --traj "$OUT/traj.npz" --run_logdir "$LOGDIR" \
    --checkpoint "$SNAP" --output "$OUT/feats.npz" --horizons 1 5 20 --seed 0
  python -m probing.probe --traj "$OUT/traj.npz" --features "$OUT/feats.npz" \
    --output "$OUT/probe"
  if [ -n "${CRITIC:-}" ] && [ -f "$CRITIC" ]; then
    python -m probing.value_sensitive measure --traj "$OUT/traj.npz" \
      --features "$OUT/feats.npz" --critic "$CRITIC" --task "$TASK" \
      --output "$OUT/vsa.json"
  else
    echo "no critic ($CRITIC) -> VSA skipped for $MS (retdec fallback, PREREG Sec.4)"
  fi
done

if [ ! -f "$MEAS/drivers_buffer.json" ]; then
  python -m probing.measure_drivers --replay "$LOGDIR/replay" --task "$TASK" \
    --ref_replay "$REF_REPLAY" --milestones $MILESTONES \
    --output "$MEAS/drivers_buffer.json"
fi

touch "$MEAS/MEASURE_DONE"
echo "[$(date)] measure $PRE_RUN DONE"
