#!/bin/bash
#SBATCH --job-name=se_b4_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A.out
# ---------------------------------------------------------------------------
# B4 instrument passes — PREREG_trackB_alea_20260822 (rev R3-M4).
# ALL 4 runs in ONE CPU job (single-device rule): per run, se_probe
# then se_alea_mask (--platform cpu, S=512 default pins).
# SKIP RULE = ckpt-identity (the 22-Aug B1 s53 lesson; NEVER
# "the json parses"): skip IFF the instrument json records the
# run's CURRENT ckpt/latest AND that is the newest ckpt DIRECTORY;
# a stale pointer refuses the cell; a stale probe is quarantined.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
RUNS_ROOT="${RUNS_ROOT:?set RUNS_ROOT to the wave runroot}"
n_ok=0; n_skip=0; n_fail=0
for s in 120 121 122 123; do
  d="$RUNS_ROOT/se_alea_s$s"
  latest="$(cat "$d/ckpt/latest" 2>/dev/null || true)"
  newest="$(find "$d/ckpt" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1)"
  if [ -z "$latest" ] || [ -z "$newest" ]; then
    echo "NO-CKPT s$s — cannot verify identity, refusing"
    n_fail=$((n_fail+1)); continue
  fi
  if [ "$latest" != "$newest" ]; then
    echo "STALE-POINTER s$s (latest=$latest newest=$newest)"
    n_fail=$((n_fail+1)); continue
  fi
  ok=1
  for inst in se_probe se_alea_mask; do
    out="$d/$inst"
    probed="$(python -c "import json;print(json.load(open('$out/$inst.json'))['ckpt'].rstrip('/').split('/')[-1])" 2>/dev/null || true)"
    if [ -n "$probed" ] && [ "$probed" = "$latest" ]; then
      echo "SKIP   s$s/$inst (matches ckpt/latest $latest)"
      continue
    fi
    if [ -n "$probed" ]; then
      echo "STALE  s$s/$inst (probed $probed != latest $latest) — quarantining"
      mv "$out" "$d/_INVALID_STALE_CKPT_${inst}_$(date +%Y%m%d%H%M%S)"
    fi
    echo "RUN    s$s/$inst"
    if ! python -m "uncfield.$inst" --run_logdir "$d" --platform cpu; then
      echo "FAIL   s$s/$inst"; ok=0; break
    fi
  done
  if [ "$ok" = 1 ]; then n_ok=$((n_ok+1)); else n_fail=$((n_fail+1)); fi
done
echo "[$(date)] b4 instruments: ok=$n_ok failed=$n_fail (of 4)"
[ "$n_fail" -eq 0 ]
