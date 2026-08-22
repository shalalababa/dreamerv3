#!/bin/bash
#SBATCH --job-name=se_onset_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A.out
# ---------------------------------------------------------------------------
# se_probe passes for the Track-B ONSET wave —
# PREREG_trackB_onset_20260822 §1. ALL 16 passes in ONE CPU job
# (scale ≡ seed block: a host split would confound device with the
# tested axis — the B1 rule). Frozen se_probe defaults; default
# --output. SKIP RULE = ckpt-identity (the 22-Aug B1 s53 lesson):
# skip IFF the existing probe records the run's CURRENT ckpt/latest;
# latest-vs-newest mismatch = stale pointer, cell refused.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
RUNS_ROOT="${RUNS_ROOT:?set RUNS_ROOT to the wave runroot}"
runs=()
for s in $(seq 96 99);   do runs+=("se_ons25_s$s"); done
for s in $(seq 100 103); do runs+=("se_ons30_s$s"); done
for s in $(seq 104 107); do runs+=("se_ons34_s$s"); done
for s in $(seq 108 111); do runs+=("se_ons38_s$s"); done
n_ok=0; n_skip=0; n_fail=0
for r in "${runs[@]}"; do
  d="$RUNS_ROOT/$r"
  out="$d/se_probe"
  latest="$(cat "$d/ckpt/latest" 2>/dev/null || true)"
  # newest = newest ckpt DIRECTORY (R1-m18: a stray file must not win)
  newest="$(find "$d/ckpt" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1)"
  if [ -z "$latest" ] || [ -z "$newest" ]; then
    echo "NO-CKPT $r (latest='$latest' newest='$newest') — cannot verify identity, refusing"
    n_fail=$((n_fail+1)); continue
  fi
  if [ "$latest" != "$newest" ]; then
    echo "STALE-POINTER $r (latest=$latest newest=$newest) — fix ckpt/latest first"
    n_fail=$((n_fail+1)); continue
  fi
  probed="$(python -c "import json;print(json.load(open('$out/se_probe.json'))['ckpt'].rstrip('/').split('/')[-1])" 2>/dev/null || true)"
  if [ -n "$probed" ] && [ "$probed" = "$latest" ]; then
    echo "SKIP   $r (probe matches ckpt/latest $latest)"
    n_skip=$((n_skip+1)); continue
  fi
  if [ -n "$probed" ]; then
    echo "STALE  $r (probed $probed != latest $latest) — quarantining"
    mv "$out" "$d/_INVALID_STALE_CKPT_se_probe_$(date +%Y%m%d%H%M%S)"
  fi
  echo "PROBE  $r"
  if python -m uncfield.se_probe --run_logdir "$d"; then
    n_ok=$((n_ok+1))
  else
    echo "FAIL   $r"; n_fail=$((n_fail+1))
  fi
done
echo "[$(date)] se_probe: ok=$n_ok skipped=$n_skip failed=$n_fail (of ${#runs[@]})"
[ "$n_fail" -eq 0 ]
