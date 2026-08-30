#!/bin/bash
#SBATCH --job-name=se_eva_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# RUNTIME, reconciled against the realized run (review M4): job 56095696
# did 12 cells in 41 min => ~3.4 min/cell at the registered config, NOT the
# ~10 min/cell projected pre-run.  Rev 5 adds the split-half IV, the
# circular-shift permutation null and leg E: measured +30% on the step-0
# config (98 s vs 75 s), so ~4.5 min/cell => ~55 min for 12.  The wall stays
# at 8 h deliberately: the house failure mode is an under-provisioned wall.
#
# NOTE: leg E is EXPLORATORY (review B9) and struck from the freeze; and
# `controls_pass` gates instrument integrity only -- `primary_resolved` is
# reported separately, because an unresolvable primary must NOT suppress a
# cell's e-process certificates.
#SBATCH --output=%x_%A.out
# ---------------------------------------------------------------------------
# EVA-at-scale — anytime-valid claim audit on the 12 deployed cells.
# Same shape as scripts/se_lic_probe.sbatch (which this follows deliberately):
# BOTH ARMS, ONE JOB, ONE HOST, ONE DEVICE (--platform cpu).
#   gauss arm : se_alea_s120..123      (B4)
#   det   arm : se_cheetah_seed10..17  (Stage-1; the 5.26x headline)
#
# WHY CPU: the whole family must be measured on one device (16 Aug: 0.113
# AUROC drift across GPUs on bf16 recurrent forwards; 22 Aug: five drivers on
# one card model).  Measured ~1 min/cell here, so CPU costs nothing.
#
# SKIP RULE = CKPT-IDENTITY (22-Aug B1 s53 lesson): skip iff the json records
# the run's CURRENT ckpt/latest AND that is the newest ckpt DIRECTORY; a stale
# pointer REFUSES the cell; stale/pre-spec outputs are quarantined.
#
# Usage:  sbatch --export=NONE,GAUSS_ROOT=...,DET_ROOT=... \
#                scripts/se_eva_probe.sbatch
#         ONE --export (review M15): a second --export REPLACES the first,
#         so the two-flag form silently dropped NONE and leaked the
#         module-managed PATH/LD_LIBRARY_PATH — which segfaults at the
#         first JAX touch ~30 s in.  Submit from a clean shell.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

ARMS="${ARMS:-both}"
N_STREAMS="${N_STREAMS:-6}"
T_MAX="${T_MAX:-8000}"
T_MAX_H="${T_MAX_H:-4000}"
HORIZONS="${HORIZONS:-2,4,8,15}"
N_CAL="${N_CAL:-1000}"
N_LAT="${N_LAT:-32}"
BLK="${BLK:-240}"
SEED="${SEED:-0}"

CELLS=()
if [ "$ARMS" = "both" ] || [ "$ARMS" = "gauss" ]; then
  : "${GAUSS_ROOT:?set GAUSS_ROOT to the b4_alea runroot}"
  for s in 120 121 122 123; do CELLS+=("$GAUSS_ROOT/se_alea_s$s"); done
fi
if [ "$ARMS" = "both" ] || [ "$ARMS" = "det" ]; then
  : "${DET_ROOT:?set DET_ROOT to the Stage-1 runroot}"
  for s in 10 11 12 13 14 15 16 17; do
    CELLS+=("$DET_ROOT/se_cheetah_seed$s")
  done
fi
echo "[$(date)] se_eva_probe: ${#CELLS[@]} cells, arms=$ARMS, "\
"streams=$N_STREAMS T=$T_MAX T_h=$T_MAX_H h=$HORIZONS"

n_ok=0; n_skip=0; n_fail=0
for d in "${CELLS[@]}"; do
  name="$(basename "$d")"
  if [ ! -d "$d" ]; then
    echo "MISSING  $name"; n_fail=$((n_fail+1)); continue
  fi
  latest="$(cat "$d/ckpt/latest" 2>/dev/null || true)"
  # `|| true`: without it, find on a missing ckpt/ exits nonzero under
  # pipefail and kills the job, making the refusal branch unreachable.
  newest="$( { find "$d/ckpt" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
               2>/dev/null || true; } | sort | tail -1)"
  if [ -z "$latest" ] || [ -z "$newest" ]; then
    echo "NO-CKPT  $name — refusing"; n_fail=$((n_fail+1)); continue
  fi
  if [ "$latest" != "$newest" ]; then
    echo "STALE-POINTER $name (latest=$latest newest=$newest) — refusing"
    n_fail=$((n_fail+1)); continue
  fi
  if [ ! -f "$d/ckpt/$latest/done" ]; then
    echo "NO-DONE  $name — refusing"; n_fail=$((n_fail+1)); continue
  fi
  # the audit CONSUMES the replay stream, so its presence is not optional
  nchunk="$( { ls "$d/replay"/*.npz 2>/dev/null || true; } | wc -l)"
  if [ "$nchunk" -lt 50 ]; then
    echo "NO-REPLAY $name ($nchunk chunks) — the audit consumes the stream"
    n_fail=$((n_fail+1)); continue
  fi

  out="$d/se_eva_probe"
  probed="$(LIC_J="$out/se_eva_probe.json" python -c 'import json,os;print(json.load(open(os.environ["LIC_J"]))["ckpt"].rstrip("/").split("/")[-1])' 2>/dev/null || true)"
  spec_ok=1
  if [ -n "$probed" ]; then
    LIC_J="$out/se_eva_probe.json" LIC_H="$HORIZONS" python -c '
import json, os, sys
d = json.load(open(os.environ["LIC_J"]))
want = [int(x) for x in os.environ["LIC_H"].split(",") if x.strip()]
sys.exit(0 if (d.get("instrument") == "se_eva_probe"
               and d.get("horizons") == want and d.get("revision", 0) >= 4) else 1)' \
      || spec_ok=0
  fi
  if [ -n "$probed" ] && [ "$probed" = "$latest" ] && [ "$spec_ok" = 1 ]; then
    echo "SKIP     $name (matches ckpt/latest $latest)"
    n_skip=$((n_skip+1)); continue
  fi
  if [ -n "$probed" ]; then
    echo "STALE/PRE-SPEC $name — quarantining"
    mv "$out" "$d/_INVALID_se_eva_probe_$(date +%Y%m%d%H%M%S)"
  fi

  echo "RUN      $name"
  if python -m uncfield.se_eva_probe --run_logdir "$d" --platform cpu \
        --n_streams "$N_STREAMS" --t_max "$T_MAX" --t_max_h "$T_MAX_H" \
        --horizons "$HORIZONS" --n_cal "$N_CAL" --n_lat "$N_LAT" \
        --blk "$BLK" --seed "$SEED"; then
    # REVIEW B13: the controls are GATES.  A cell whose SYNTHG certified,
    # whose planted overclaim failed to certify, or which could not declare
    # two scalings, has not measured anything -- refuse it here rather than
    # leaving a human to notice in the json.
    if LIC_J="$out/se_eva_probe.json" python -c '
import json, os, sys
d = json.load(open(os.environ["LIC_J"]))
sys.exit(0 if d.get("controls_pass") else 1)'; then
      n_ok=$((n_ok+1))
    else
      echo "CONTROLS-FAILED $name — quarantining (see .controls in the json)"
      mv "$out/se_eva_probe.json" "$out/CONTROLS_FAILED_se_eva_probe.json"
      n_fail=$((n_fail+1))
    fi
  else
    echo "FAIL     $name"; n_fail=$((n_fail+1))
  fi
done

echo "[$(date)] cells: ok=$n_ok skipped=$n_skip failed=$n_fail of ${#CELLS[@]}"
[ "$n_fail" -eq 0 ]
