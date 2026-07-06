#!/bin/bash
#SBATCH --job-name=dv3_d0meas
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Gate D0 measurement for ONE completed D0 run (submit with
# --dependency=afterok:<train jobid>; submit_d0.sh does this):
#   1. d0/sweep.py: held-out eval episodes -> per-state signal npz
#      (qfull/qhalf/udyn + derived) into $RUNROOT/d0_cells/<RUN_ID>.npz.
#   2. Calibration-addendum partials (note App. A.4 item 5): build the
#      run's leave-one-out probe set from its SIBLING seeds' replays if
#      missing (never from the measured run's own buffer -- that would
#      leak training data into the held-out error), then probing/latent_uq.
#      The same leave-one-out protocol is used for ALL doses including E1
#      so probe-window provenance is uniform across the dose axis
#      [prov.: confirm protocol at freeze].
#
# Required env:
#   RUN_ID    e.g. d0_cup_d8x1_seed1
#   TASK      dmc task (for the probe-set builder)
#   PROBESET  probe-set dir, e.g. $RUNROOT/probesets/d0_cup_d8x1_wo1_v1
#   SIBLINGS  labeled replay list for the builder when PROBESET is absent,
#             e.g. "s2=$RUNROOT/d0_cup_d8x1_seed2/replay s3=..._seed3/replay"
# Optional: EPISODES (50), ACTIONS (8), ROLLOUTS (16), SWEEP_SEED
#           (default 100+train seed suffix), SKIP_UQ=1 (sweep only)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUN_ID:?set RUN_ID}"; : "${TASK:?set TASK}"; : "${PROBESET:?set PROBESET}"
EPISODES="${EPISODES:-50}"; ACTIONS="${ACTIONS:-8}"; ROLLOUTS="${ROLLOUTS:-16}"
LOGDIR="$RUNROOT/$RUN_ID"
[ -f "$LOGDIR/TRAINING_DONE" ] || { echo "ERROR: $RUN_ID not trained"; exit 2; }
SWEEP_SEED="${SWEEP_SEED:-$((100 + ${RUN_ID##*seed}))}"
CELLS="$RUNROOT/d0_cells"
mkdir -p "$CELLS"

# --- 1. signal sweep (App. A.4 signals 1-4; >=50 episodes, N>=5k: A.5) ---
if [ ! -f "$CELLS/$RUN_ID.npz" ]; then
  python -m d0.sweep --run_logdir "$LOGDIR" \
    --output "$CELLS/$RUN_ID.npz" --episodes "$EPISODES" \
    --actions "$ACTIONS" --rollouts "$ROLLOUTS" --seed "$SWEEP_SEED"
else
  echo "SKIP sweep: $CELLS/$RUN_ID.npz exists"
fi

# --- 2. calibration partials (App. A.4 item 5) --------------------------
if [ "${SKIP_UQ:-0}" != "1" ]; then
  if [ ! -d "$PROBESET" ]; then
    : "${SIBLINGS:?PROBESET absent and no SIBLINGS to build it from}"
    # Protocol mirrors the frozen v1 sets (burn_in 32, eval_steps 8, seed 0)
    # so E1 partials stay comparable to the Phase-4 addendum numbers.
    python -m probing.probeset --task "$TASK" \
      --replay $SIBLINGS --output "$PROBESET" \
      --windows_per_source 640 --burn_in 32 --eval_steps 8 --seed 0
  fi
  python -m probing.latent_uq --probeset "$PROBESET" \
    --run_logdir "$LOGDIR" --checkpoints "$LOGDIR/ckpt" \
    --output "$LOGDIR/latent_uq/$(basename "$PROBESET")"
fi

touch "$LOGDIR/D0_MEASURE_DONE"
echo "[$(date)] d0 measure $RUN_ID DONE -> $CELLS/$RUN_ID.npz"
