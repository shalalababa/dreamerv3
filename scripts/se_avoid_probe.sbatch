#!/bin/bash
#SBATCH --job-name=se_avoid_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# se_probe + se_region_probe passes for the A1 avoidance wave — PREREG_nfi_avoidance_20260821 §4.
#
# EIGHT cells x TWO instruments: every A1 run needs BOTH the standard
# se_probe (misprice gate + rankx secondary) and the NEW
# se_region_probe (delivery gate + mechanism), and review #27 M7 pins
# both to the SAME job on the same device class, frozen defaults,
# default outputs.
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

runs=(se_avoid_het_s44 se_avoid_het_s45 se_avoid_het_s46 se_avoid_het_s47
      se_avoid_flat_s48 se_avoid_flat_s49 se_avoid_flat_s50 se_avoid_flat_s51)
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
  cell_ok=1
  for tool in se_probe se_region_probe; do
    out="$d/$tool"
    if ls "$out"/*.json >/dev/null 2>&1; then
      echo "SKIP   $r/$tool (already present)"; continue
    fi
    echo "[$(date)] $tool $r  (ckpt $latest)"
    python -m "uncfield.$tool" --run_logdir "$d"
    rc=$?
    if [ "$rc" -ne 0 ] || ! ls "$out"/*.json >/dev/null 2>&1; then
      echo "FAIL   $r/$tool (rc=$rc)"; cell_ok=0
    fi
  done
  if [ "$cell_ok" -eq 1 ]; then
    echo "OK     $r"; n_ok=$((n_ok+1))
  else
    n_fail=$((n_fail+1)); failed="$failed $r"
  fi
done

echo
echo "[$(date)] probes: ok=$n_ok skipped=$n_skip failed=$n_fail (of ${#runs[@]})"
[ -n "$failed" ] && echo "FAILED CELLS:$failed"
exit $([ "$n_fail" -eq 0 ] && echo 0 || echo 1)
