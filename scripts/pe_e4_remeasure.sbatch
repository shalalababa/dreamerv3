#!/bin/bash
#SBATCH --job-name=pe_e4_remeasure
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# LeWM pe panel re-measure on rtx6000 -- the ops half of the P-J1-rep
# pin-swap. (2026-08-16, requested by the Papers-1-3 chat.)
#
# WHY. The LeWM rep primary is paired [pe - jp] on rew_nll_in. The jp panel
# was measured fresh on Midway3 rtx6000; the pe values in the read bundle are
# HISTORICAL (csv sha 9907d733...), produced on unknown hardware weeks
# earlier. A one-cell diff (pe_e4_devdiff, 16 Aug) priced that term at
# -0.0994 nats (-0.208%) on rew_nll_in -- and because the term is pe-side and
# systematic, it shifts every pairwise delta in the same direction instead of
# cancelling. Re-measuring pe beside jp removes it.
#
# ALL 16, NOT THE REMAINING 15. The request was for the 15 not yet
# re-measured. This does all 16 because one job is one GPU, and the ridge
# devcheck established that this instrument is bitwise-deterministic WITHIN a
# GPU but shifts across architectures: 16-in-one-job gives a pe panel that is
# device-uniform BY CONSTRUCTION, whereas 15-plus-the-earlier-devdiff-cell
# would be 15 cells on this card and 1 on whichever rtx6000 the 2 Aug job
# landed on. The extra cell costs ~2 minutes and removes a seam that would
# otherwise sit inside the amended panel. The devdiff output is left exactly
# where it is; it stays the diagnostic it was.
#
# NON-DESTRUCTIVE: --output points into pe_e4_remeasure/<cell>/, a fresh side
# directory. The registered e4_fingerpx_v1_pe.csv and its pinned sha are NOT
# touched -- swapping the pin is a dated amendment for the owning chat, and
# nothing here presumes it. If the amendment is declined, this directory is
# simply unused.
#
# Command is the step-9 / devdiff form verbatim: same probeset, same
# --horizons 1 5 20, same --ep_batch 4. Only --run_logdir and --output vary.
#
# Resumable: a cell whose summary.json already parses is skipped.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUNROOT:?set RUNROOT}"
PROBESET="$RUNROOT/e4_probesets/fingerpx_v1"
[ -d "$PROBESET" ] || { echo "ERROR: no probeset $PROBESET" >&2; exit 2; }
[ -f "$PROBESET/FROZEN" ] || { echo "ERROR: $PROBESET is not FROZEN" >&2; exit 2; }
DEST="$RUNROOT/pe_e4_remeasure"
mkdir -p "$DEST"

host="$(hostname)"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "[device] host=$host gpu=$gpu driver=$drv job=${SLURM_JOB_ID:-none}"
echo "[device] all 16 pe cells below are measured on THIS gpu"
echo

cells=()
for side in 0 1; do
  for seed in 1 2 3 4 5 6 7 8; do
    cells+=("ax1wm_finger_pepxq1ms${side}_seed${seed}")
  done
done

n_ok=0; n_skip=0; n_fail=0; failed=""
for c in "${cells[@]}"; do
  d="$RUNROOT/$c"
  out="$DEST/$c"
  if [ ! -d "$d" ]; then
    echo "FAIL   $c: no run dir"; n_fail=$((n_fail+1)); failed="$failed $c"; continue
  fi
  if [ -f "$out/summary.json" ] && \
     python -c "import json,sys; json.load(open(sys.argv[1]))" "$out/summary.json" 2>/dev/null; then
    echo "SKIP   $c (already re-measured)"; n_skip=$((n_skip+1)); continue
  fi
  echo "[$(date)] MEASURE $c"
  mkdir -p "$out"
  python -m probing.stratified_error measure \
    --probeset "$PROBESET" --run_logdir "$d" \
    --horizons 1 5 20 --ep_batch 4 --output "$out"
  rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$out/summary.json" ]; then
    python - "$out/summary.json" "$c" "$host" "$gpu" "$drv" "${SLURM_JOB_ID:-none}" "$d" "$PROBESET" <<'PY'
import hashlib, json, sys
sm, cell, host, gpu, drv, job, run, probeset = sys.argv[1:9]
json.dump({'cell': cell, 'summary': sm, 'run_logdir': run, 'probeset': probeset,
           'hostname': host, 'gpu_name': gpu, 'driver_version': drv,
           'slurm_job_id': job,
           'command': ('python -m probing.stratified_error measure --probeset '
                       f'{probeset} --run_logdir {run} --horizons 1 5 20 '
                       f'--ep_batch 4 --output {sm.rsplit("/", 1)[0]}'),
           'summary_sha256': hashlib.sha256(open(sm, 'rb').read()).hexdigest()},
          open(sm + '.device.json', 'w'), indent=1)
PY
    echo "OK     $c"; n_ok=$((n_ok+1))
  else
    echo "FAIL   $c (rc=$rc)"; n_fail=$((n_fail+1)); failed="$failed $c"
  fi
done

echo
echo "[$(date)] pe re-measure: ok=$n_ok skipped=$n_skip failed=$n_fail (of ${#cells[@]})"
[ -n "$failed" ] && echo "FAILED CELLS:$failed"
echo
echo "NEXT: ops pulls $DEST and hands the 16-cell pe table to the owning chat."
echo "      The pin swap is THEIR dated amendment; nothing here changes the"
echo "      registered csv or its sha."
[ "$n_fail" -eq 0 ] || exit 1
exit 0
