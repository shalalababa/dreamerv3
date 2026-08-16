#!/bin/bash
#SBATCH --job-name=b1p_ridge_bundle
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --nodelist=midway3-0278
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# B1' per-fit ridge panel -- ALL 16 CELLS IN ONE JOB. (PREREG_b1prime_20260815,
# registered per-fit form at line 186.)
#
# WHY BUNDLED. Submitted as 16 separate jobs this panel eats the entire
# 12-job QOS cap and still leaves 4 behind, blocking every other leg. The
# operational convention here is to bundle short work into one job: the
# walltime limit is 36 h and a ridge measure is minutes, so all 16 sit
# comfortably inside one allocation.
#
# It is also STRICTLY BETTER for this particular panel. The primary compares
# these 16 ridge AUROCs against 16 pinned anchor literals, and ridge AUROC
# carries a device term of up to 0.113 across GPU architectures
# (ops/waves/lewm_probe_devcheck/FINDING.md) against a B1' MDE80 of 0.06-0.20.
# One job = one GPU = one device for the whole panel, by construction rather
# than by hoping the scheduler is consistent.
#
# --nodelist=midway3-0278 is the anchor's OWN node: the anchor bundle
# regen_ridge_e4_20260810_220849 carries its producer logs
# (refit_ridge_panel_53243601_{0..3}.out) and sacct puts all four array tasks
# there. Same hardware as the anchor, by construction.
#
# Command is the registered per-fit form verbatim: no --platform, no
# --ep_batch (those are the Vast lane's additions; the anchor had neither).
#
# Resumable: a cell whose output already parses as json is skipped, so a
# re-submit after a drain or a failure picks up where it stopped.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUNROOT:?set RUNROOT}"
PROBESET="$RUNROOT/e4_probesets/finger_v1"
[ -d "$PROBESET" ] || { echo "ERROR: no probeset $PROBESET" >&2; exit 2; }
[ -f "$PROBESET/FROZEN" ] || { echo "ERROR: $PROBESET is not FROZEN" >&2; exit 2; }

host="$(hostname)"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "[device] host=$host gpu=$gpu driver=$drv job=${SLURM_JOB_ID:-none}"
echo "[device] every cell below is measured on THIS gpu"
echo

runs=()
for armp in "" f; do
  for side in 0 1; do
    for seed in 1 2 3 4; do
      runs+=("ax1wm_finger_${armp}sk6q1s${side}_seed${seed}")
    done
  done
done

n_ok=0; n_skip=0; n_fail=0; failed=""
for r in "${runs[@]}"; do
  d="$RUNROOT/$r"
  out="$d/ridge_probe_finger_v1.json"
  if [ ! -f "$d/config.yaml" ]; then
    echo "FAIL   $r: no config.yaml"; n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  if [ -f "$out" ] && python -c "import json,sys; json.load(open(sys.argv[1]))" "$out" 2>/dev/null; then
    echo "SKIP   $r (already measured)"; n_skip=$((n_skip+1)); continue
  fi
  echo "[$(date)] MEASURE $r"
  python -m probing.ridge_probe measure \
    --run_logdir "$d" --probeset "$PROBESET" --output "$out"
  rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$out" ]; then
    # Sidecar device witness: the probe json itself records no device, which
    # is what forced a day of sacct archaeology to identify the anchor's GPU.
    # Outside the instrument, so no registered artifact changes.
    python - "$out" "$host" "$gpu" "$drv" "${SLURM_JOB_ID:-none}" "$d" "$PROBESET" <<'PY'
import hashlib, json, sys
out, host, gpu, drv, job, run, probeset = sys.argv[1:8]
json.dump({'output': out, 'run_logdir': run, 'probeset': probeset,
           'hostname': host, 'gpu_name': gpu, 'driver_version': drv,
           'slurm_job_id': job,
           'output_sha256': hashlib.sha256(open(out, 'rb').read()).hexdigest()},
          open(out + '.device.json', 'w'), indent=1)
PY
    echo "OK     $r"; n_ok=$((n_ok+1))
  else
    echo "FAIL   $r (rc=$rc)"; n_fail=$((n_fail+1)); failed="$failed $r"
  fi
done

echo
echo "[$(date)] ridge bundle: ok=$n_ok skipped=$n_skip failed=$n_fail  (of ${#runs[@]})"
[ -n "$failed" ] && echo "FAILED CELLS:$failed"
[ "$n_fail" -eq 0 ] || exit 1
exit 0
