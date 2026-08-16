#!/bin/bash
#SBATCH --job-name=dv3_lewm_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# LeWM probe: one ridge_probe measure per grafted-donor WM, on Slurm.
# (PREREG_lewm_20260812, the probe/E4 leg; command copied from its line 106.)
#
# One run per job, so a single failure cannot strand the rest -- the Vast lanes
# abort a whole lane on first failure, Slurm does not, and that difference is
# the point of running this leg here.
#
# Required env (supplied by the wave's generated submit_rcc.sh via --export):
#   RUN_LOGDIR  the fitted WM to probe
#   PROBESET    frozen probe-set dir
#   OUTPUT      json to write
# Optional:
#   EP_BATCH    default 4 -- KEEP UNIFORM across a panel (see ridge_probe.py:
#               batched RSSM forwards are not batch-shape-invariant on GPU)
#   PLATFORM    default cuda
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUN_LOGDIR:?set RUN_LOGDIR}"; : "${PROBESET:?set PROBESET}"; : "${OUTPUT:?set OUTPUT}"
EP_BATCH="${EP_BATCH:-4}"
PLATFORM="${PLATFORM:-cuda}"

[ -d "$RUN_LOGDIR" ]     || { echo "ERROR: no run dir: $RUN_LOGDIR" >&2; exit 2; }
[ -f "$RUN_LOGDIR/config.yaml" ] || { echo "ERROR: no config.yaml in $RUN_LOGDIR" >&2; exit 2; }
[ -d "$PROBESET" ]       || { echo "ERROR: no probeset: $PROBESET" >&2; exit 2; }
# The frozen-probeset gate the E4 instruments all enforce: scoring against an
# unfrozen probe set silently makes runs incomparable across a panel.
[ -f "$PROBESET/FROZEN" ] || { echo "ERROR: $PROBESET is not FROZEN" >&2; exit 2; }

# Idempotent: re-running the wave must not redo finished work. Checked on the
# OUTPUT's own parseability, not merely its existence -- a job killed
# mid-write leaves a truncated json that `test -f` would call done.
if [ -f "$OUTPUT" ] && python -c "import json,sys; json.load(open(sys.argv[1]))" "$OUTPUT" 2>/dev/null; then
  echo "SKIP already measured: $OUTPUT"
  exit 0
fi

mkdir -p "$(dirname "$OUTPUT")"
echo "[$(date)] ridge_probe measure run=$RUN_LOGDIR probeset=$PROBESET"
DREAMER_DISABLE_GPUFLAGS=1 python -m probing.ridge_probe measure \
  --run_logdir "$RUN_LOGDIR" \
  --probeset "$PROBESET" \
  --output "$OUTPUT" \
  --platform "$PLATFORM" --ep_batch "$EP_BATCH"
rc=$?
echo "[$(date)] done rc=$rc -> $OUTPUT"
exit $rc
