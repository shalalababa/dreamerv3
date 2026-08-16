#!/bin/bash
#SBATCH --job-name=dv3_ridge
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# One `ridge_probe measure` per job, in the REGISTERED per-fit form:
#
#   python -m probing.ridge_probe measure --run_logdir <wm_dir> \
#     --probeset $RUNROOT/e4_probesets/<probeset> --output <wm_dir>/<name>.json
#
# Deliberately NO --platform and NO --ep_batch. Both are registered as
# defaults (PREREG_b1prime_20260815 line 186: "--platform cuda" is the VAST
# lane's addition only, because `gpu` is not a backend name in that JAX
# build). The B1' anchor panel was produced by exactly this form, and the
# whole point of this job is to be byte-comparable with that anchor.
#
# Required env: RUN_LOGDIR, PROBESET, OUTPUT
#
# DEVICE WITNESS. The probe's own json records no device, which is what forced
# a day of sacct archaeology to establish that the B1' anchor was measured on
# midway3-0278 (a V100). Ridge AUROC/R2 carries a device term of up to 0.113
# across architectures (ops/waves/lewm_probe_devcheck/FINDING.md), so a panel
# that cannot say which GPU produced it cannot be compared to anything. This
# writes a SIDECAR `<OUTPUT>.device.json` -- outside the instrument, so no
# registered artifact changes and no amendment is needed; stamping the field
# inside summary.json is the registered-amendment route.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUN_LOGDIR:?set RUN_LOGDIR}"; : "${PROBESET:?set PROBESET}"; : "${OUTPUT:?set OUTPUT}"

[ -d "$RUN_LOGDIR" ] || { echo "ERROR: no run dir: $RUN_LOGDIR" >&2; exit 2; }
[ -f "$RUN_LOGDIR/config.yaml" ] || { echo "ERROR: no config.yaml in $RUN_LOGDIR" >&2; exit 2; }
[ -d "$PROBESET" ] || { echo "ERROR: no probeset: $PROBESET" >&2; exit 2; }
[ -f "$PROBESET/FROZEN" ] || { echo "ERROR: $PROBESET is not FROZEN" >&2; exit 2; }

# Idempotent on the output's own parseability, not on mere existence: a job
# killed mid-write leaves a truncated json that `test -f` would call done.
if [ -f "$OUTPUT" ] && python -c "import json,sys; json.load(open(sys.argv[1]))" "$OUTPUT" 2>/dev/null; then
  echo "SKIP already measured: $OUTPUT"; exit 0
fi

mkdir -p "$(dirname "$OUTPUT")"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
gpu_drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "[device] host=$(hostname) gpu=${gpu_name:-unknown} driver=${gpu_drv:-unknown} job=${SLURM_JOB_ID:-none}"

echo "[$(date)] ridge_probe measure run=$RUN_LOGDIR probeset=$PROBESET"
python -m probing.ridge_probe measure \
  --run_logdir "$RUN_LOGDIR" \
  --probeset "$PROBESET" \
  --output "$OUTPUT"
rc=$?

# Written only on success: a sidecar beside a missing or failed output would
# assert provenance for a measurement that does not exist.
if [ "$rc" -eq 0 ] && [ -f "$OUTPUT" ]; then
  python - "$OUTPUT" "$(hostname)" "${gpu_name:-unknown}" "${gpu_drv:-unknown}" \
           "${SLURM_JOB_ID:-none}" "$RUN_LOGDIR" "$PROBESET" <<'PY'
import hashlib, json, sys
out, host, gpu, drv, job, run, probeset = sys.argv[1:8]
json.dump({
    'output': out, 'run_logdir': run, 'probeset': probeset,
    'hostname': host, 'gpu_name': gpu, 'driver_version': drv,
    'slurm_job_id': job,
    'output_sha256': hashlib.sha256(open(out, 'rb').read()).hexdigest(),
}, open(out + '.device.json', 'w'), indent=1)
PY
  echo "[device] witness -> $OUTPUT.device.json"
fi
echo "[$(date)] done rc=$rc -> $OUTPUT"
exit $rc
