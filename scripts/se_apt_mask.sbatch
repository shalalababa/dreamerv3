#!/bin/bash
#SBATCH --job-name=se_apt_mask
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# APT-column mask instrument passes — PREREG_nfi_apt_20260821 §2.
#
# Every APT run (the seed-9 SMOKE and later the registered se_apt_s68..75)
# needs its own `uncfield.se_apt_mask` pass writing <run>/se_apt_mask/ before
# either the smoke gate or the registered read can load it: se_apt_read
# consumes se_apt_mask.json + .npz and does NOT compute them.
#
# FROZEN DEFAULTS, DEFAULT OUTPUT — load-bearing, same convention as the
# se_probe passes: every knob except --run_logdir stays at the parser default
# (n_eval 512 is the prereg's hard S pin; apt_knn/apt_logc are asserted against
# the run config), and --output is omitted so the tool writes <run>/se_apt_mask.
#
# CPU, hence caslake: se_apt_mask defaults to --platform cpu, which also
# satisfies the one-device-class rule for the whole panel for free (16-Aug
# single-GPU lesson) — every pass runs on the same CPU partition.
#
# Resumable: a cell whose se_apt_mask/se_apt_mask.json already parses is
# skipped.
#
# Usage:  RUNROOT=... RUNS="se_apt_smoke" sbatch scripts/se_apt_mask.sbatch
#         RUNROOT=... sbatch scripts/se_apt_mask.sbatch          # the 8 wave runs
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUNROOT:?set RUNROOT}"

echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}"
echo "[note] platform=cpu (se_apt_mask default); frozen defaults; default output"
echo

if [ -n "${RUNS:-}" ]; then
  read -r -a runs <<< "$RUNS"
else
  runs=(se_apt_s68 se_apt_s69 se_apt_s70 se_apt_s71
        se_apt_s72 se_apt_s73 se_apt_s74 se_apt_s75)
fi
n_ok=0; n_skip=0; n_fail=0; failed=""

for r in "${runs[@]}"; do
  d="$RUNROOT/$r"
  if [ ! -f "$d/config.yaml" ]; then
    echo "FAIL   $r: no config.yaml"; n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  latest="$(cat "$d/ckpt/latest" 2>/dev/null || true)"
  if [ -z "$latest" ] || [ ! -f "$d/ckpt/$latest/done" ]; then
    echo "FAIL   $r: ckpt/latest missing or not a completed save ('$latest')"
    n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  if [ -f "$d/se_apt_mask/se_apt_mask.json" ] && \
     python -c "import json,sys; json.load(open(sys.argv[1]))" \
            "$d/se_apt_mask/se_apt_mask.json" >/dev/null 2>&1; then
    echo "SKIP   $r (se_apt_mask already present)"; n_skip=$((n_skip+1)); continue
  fi
  echo "=== $r"
  if python -m uncfield.se_apt_mask --run_logdir "$d"; then
    n_ok=$((n_ok+1))
  else
    echo "FAIL   $r: se_apt_mask rc=$?"; n_fail=$((n_fail+1)); failed="$failed $r"
  fi
  echo
done

echo "[$(date)] se_apt_mask: ok=$n_ok skipped=$n_skip failed=$n_fail (of ${#runs[@]})"
[ -n "$failed" ] && echo "[failed]$failed"
exit $(( n_fail > 0 ))
