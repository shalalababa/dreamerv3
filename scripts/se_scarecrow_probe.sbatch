#!/bin/bash
#SBATCH --job-name=se_scare_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A.out
# ---------------------------------------------------------------------------
# se_probe passes for the A2 SCARECROW wave —
# PREREG_trackA_scarecrow_20260822 §3. ALL 20 passes in ONE CPU job
# (arm ≡ seed block: a host split would confound device with arm,
# the B1 dose-wave rule). Frozen se_probe defaults; default --output
# (<run>/se_probe; snapshot passes are the only --output users).
#
# SKIP RULE (22-Aug B1 lesson, se_dose10_s53): a cell is skipped IFF
# its existing se_probe.json PARSES **AND records the run's CURRENT
# ckpt/latest** — parse-success alone kept a stale-checkpoint probe
# alive through the repair pass and cost a read refusal. Stale
# probes are quarantined, never overwritten in place.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
RUNS_ROOT="${RUNS_ROOT:?set RUNS_ROOT to the wave runroot}"
runs=()
for s in $(seq 76 79);  do runs+=("sc_ctrl_s$s");  done
for s in $(seq 80 83);  do runs+=("sc_scare_s$s"); done
for s in $(seq 84 87);  do runs+=("sc_scare2_s$s"); done
for s in $(seq 88 91);  do runs+=("sc_pen1_s$s");  done
for s in $(seq 92 95);  do runs+=("sc_pen2_s$s");  done
n_ok=0; n_skip=0; n_fail=0
for r in "${runs[@]}"; do
  d="$RUNS_ROOT/$r"
  out="$d/se_probe"
  latest="$(cat "$d/ckpt/latest" 2>/dev/null || true)"
  # #34 M15 (the A1 append-verify lesson): ckpt/latest itself can be a
  # frozen stale pointer — cross-check against the NEWEST ckpt subdir
  # and refuse the cell rather than probe a wrong checkpoint
  newest="$(ls "$d/ckpt" 2>/dev/null | grep -v '^latest' | sort | tail -1)"
  if [ -n "$latest" ] && [ -n "$newest" ] && [ "$latest" != "$newest" ]; then
    echo "STALE-POINTER $r (latest=$latest newest=$newest) — fix ckpt/latest first"
    n_fail=$((n_fail+1)); continue
  fi
  probed="$(python -c "import json,sys;print(json.load(open('$out/se_probe.json'))['ckpt'].rstrip('/').split('/')[-1])" 2>/dev/null || true)"
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
