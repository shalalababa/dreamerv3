#!/bin/bash
#SBATCH --job-name=se_dose_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# se_probe passes for the B1 dose ladder — PREREG_nfi_dose_20260821 §1/§4.
#
# SIXTEEN cells, all load-bearing: every dose run feeds the theta_1
# trend primary, and review #28 M7 requires ALL SIXTEEN probes in ONE
# CPU job — the seeds are blocked by dose, so a host split would
# confound device class with the dose axis exactly.
#
# FROZEN DEFAULTS, DEFAULT OUTPUT — both load-bearing, same as the Amendment-2
# D-only probes:
#   * every knob except --run_logdir stays at the parser default, because the
#     reader compares these outputs against statistics computed with exactly
#     those values;
#   * --output is OMITTED so the tool writes <run>/se_probe. The amendment
#     reserves --output <run>/se_probe_snap<pct> for SNAPSHOT passes and warns
#     they must never take the default path; this is the mirror case -- the
#     final-ckpt pass is the one that owns it.
#
# CPU, hence caslake: se_probe defaults to --platform cpu. That is also what
# satisfies §4's "all probe passes on one device class" for free -- every pass
# runs on the same CPU partition, so the 16-Aug single-GPU rule is met without
# needing to pin a GPU, and it matches how the Amendment-2 probes were run.
#
# Resumable: a cell whose se_probe/ already parses is skipped.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUNROOT:?set RUNROOT}"

echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}"
echo "[note] platform=cpu (se_probe default); frozen defaults; default output"
echo

runs=(se_dose10_s52 se_dose10_s53 se_dose10_s54 se_dose10_s55
      se_dose25_s56 se_dose25_s57 se_dose25_s58 se_dose25_s59
      se_dose50_s60 se_dose50_s61 se_dose50_s62 se_dose50_s63
      se_dose75_s64 se_dose75_s65 se_dose75_s66 se_dose75_s67)
n_ok=0; n_skip=0; n_fail=0; failed=""

for r in "${runs[@]}"; do
  d="$RUNROOT/$r"
  out="$d/se_probe"
  if [ ! -f "$d/config.yaml" ]; then
    echo "FAIL   $r: no config.yaml"; n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  latest="$(cat "$d/ckpt/latest" 2>/dev/null || true)"
  if [ -z "$latest" ] || [ ! -f "$d/ckpt/$latest/done" ]; then
    echo "FAIL   $r: ckpt/latest missing or not a completed save ('$latest')"
    n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  if ls "$out"/*.json >/dev/null 2>&1; then
    echo "SKIP   $r (se_probe already present)"; n_skip=$((n_skip+1)); continue
  fi
  echo "[$(date)] PROBE $r  (ckpt $latest)"
  python -m uncfield.se_probe --run_logdir "$d"
  rc=$?
  if [ "$rc" -eq 0 ] && ls "$out"/*.json >/dev/null 2>&1; then
    echo "OK     $r"; n_ok=$((n_ok+1))
  else
    echo "FAIL   $r (rc=$rc)"; n_fail=$((n_fail+1)); failed="$failed $r"
  fi
done

echo
echo "[$(date)] se_probe: ok=$n_ok skipped=$n_skip failed=$n_fail (of ${#runs[@]})"
[ -n "$failed" ] && echo "FAILED CELLS:$failed"
exit $([ "$n_fail" -eq 0 ] && echo 0 || echo 1)
