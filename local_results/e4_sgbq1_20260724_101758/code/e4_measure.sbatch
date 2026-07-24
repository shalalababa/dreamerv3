#!/bin/bash
#SBATCH --job-name=dv3_e4
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# E4: regime- and reward-stratified WM prediction error, one domain per job
# (prereg/PREREG_e4_stratified_error_20260714.md). Sweeps every fitted WM
# run matching GLOB, scores the frozen per-domain E4 probe set at each
# run's final checkpoint (inference only), then collates one CSV.
#
# Per-run outputs land in <run>/e4_<probeset_id>/{summary.json,errors.npz};
# runs with an existing summary are skipped, so the job is resumable.
#
# Required env:
#   PROBESET  frozen E4 probe-set dir (probing/stratified_error build-probeset)
#   GLOB      run-dir glob under $RUNROOT, e.g. 'ax1wm_finger_*'
# Optional:
#   HORIZONS  default "1 5 20"
#   EP_BATCH  default 4
#   COLLATE   output CSV (default $RUNROOT/e4_<probeset_id>.csv)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${PROBESET:?set PROBESET}"; : "${GLOB:?set GLOB}"
HORIZONS="${HORIZONS:-1 5 20}"
EP_BATCH="${EP_BATCH:-4}"
PROBESET_ID="$(basename "$PROBESET")"
COLLATE="${COLLATE:-$RUNROOT/e4_${PROBESET_ID}.csv}"
[ -f "$PROBESET/FROZEN" ] || { echo "ERROR: $PROBESET is not FROZEN"; exit 2; }

n_done=0; n_run=0; n_skip=0
for logdir in "$RUNROOT"/$GLOB; do
  [ -f "$logdir/config.yaml" ] || continue
  [ -d "$logdir/ckpt" ] || { echo "SKIP no ckpt: $logdir"; n_skip=$((n_skip+1)); continue; }
  run_id="$(basename "$logdir")"
  if [ -f "$logdir/e4_${PROBESET_ID}/summary.json" ]; then
    echo "SKIP measured: $run_id"; n_done=$((n_done+1)); continue
  fi
  echo "[$(date)] E4 measure $run_id"
  python -m probing.stratified_error measure \
    --probeset "$PROBESET" --run_logdir "$logdir" \
    --horizons $HORIZONS --ep_batch "$EP_BATCH"
  n_run=$((n_run+1))
done
echo "[$(date)] measured $n_run new, $n_done already done, $n_skip skipped"

python -m probing.stratified_error collate \
  --runroot "$RUNROOT" --glob "$GLOB" \
  --probeset_id "$PROBESET_ID" --output "$COLLATE"
echo "[$(date)] E4 sweep DONE -> $COLLATE"
