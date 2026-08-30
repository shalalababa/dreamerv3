#!/bin/bash
#SBATCH --job-name=se_lic_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%A.out
# ---------------------------------------------------------------------------
# The B4-checkpoint DEPLOYED READ — responsiveness-license probe.
#
# BOTH ARMS, ONE JOB, ONE HOST, ONE DEVICE (--platform cpu).
#   gauss arm : se_alea_s120..123   (B4, PREREG_trackB_alea_20260822)
#   det   arm : se_cheetah_seed10..17 (Stage-1; where the 5.26x headline lives)
#
# WHY ONE JOB AND WHY CPU.  The whole family must be measured on ONE device:
# the E4 ridge probe shifted up to 0.113 AUROC across GPUs on bf16 recurrent
# forwards (16 Aug), and rented boxes run five different drivers on the same
# card model (22 Aug), so a wave whose arms are split across devices cannot be
# compared.  CPU makes the single-device rule free, costs ~2-6 h here, and is
# exactly what the B4 instrument wave already did (scripts/se_b4_probe.sbatch).
# bf16 is still active on CPU (embodied/jax/nets.py:12), which is why the
# instrument computes claim and license inside ONE jitted function and uses
# the Amendment-2 cross-instrument tolerance 5e-2, never 1e-4.
#
# SKIP RULE = CKPT-IDENTITY (the 22-Aug B1 s53 lesson; NEVER "the json
# parses"): skip a cell IFF its se_lic_probe.json records the run's CURRENT
# ckpt/latest AND that pointer names the newest ckpt DIRECTORY.  A stale
# pointer REFUSES the cell (it does not silently probe an old model — the
# 22-Aug pull-pass-2 defect probed 105k-step checkpoints while 494k ones sat
# beside them and reported OK).  A stale probe output is quarantined.
#
# Usage:
#   sbatch --export=NONE scripts/se_lic_probe.sbatch          # submit from a
#                                                             # CLEAN shell:
#   (--export=ALL leaks the module-managed PATH/LD_LIBRARY_PATH into the job
#    and segfaults at the first JAX touch ~30 s in.)
# Env:
#   GAUSS_ROOT  runroot holding se_alea_s120..123      (required unless ARMS=det)
#   DET_ROOT    runroot holding se_cheetah_seed10..17  (required unless ARMS=gauss)
#   ARMS        both|gauss|det          (default both)
#   OUT_ROOT    collate output dir      (default $REPO/local_results/uncfield/se_lic)
#   HORIZONS    default 1,2,4,8,15      N_EVAL default 512    BURN_IN default 16
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

ARMS="${ARMS:-both}"
N_EVAL="${N_EVAL:-512}"
BURN_IN="${BURN_IN:-16}"
HORIZONS="${HORIZONS:-1,2,4,8,15}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-$REPO/local_results/uncfield/se_lic}"
mkdir -p "$OUT_ROOT"

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
echo "[$(date)] se_lic_probe: ${#CELLS[@]} cells, arms=$ARMS, S=$N_EVAL, h=$HORIZONS"

n_ok=0; n_skip=0; n_fail=0; n_done=0; DONE_JSONS=()
for d in "${CELLS[@]}"; do
  name="$(basename "$d")"
  if [ ! -d "$d" ]; then
    echo "MISSING  $name ($d)"; n_fail=$((n_fail+1)); continue
  fi
  latest="$(cat "$d/ckpt/latest" 2>/dev/null || true)"
  # `|| true` is load-bearing: without it, `find` on a missing ckpt/ exits
  # nonzero under `set -o pipefail` and kills the WHOLE JOB, making the
  # NO-CKPT refusal branch below unreachable (review F11).
  newest="$( { find "$d/ckpt" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
               2>/dev/null || true; } | sort | tail -1)"
  if [ -z "$latest" ] || [ -z "$newest" ]; then
    echo "NO-CKPT  $name — cannot verify identity, refusing"
    n_fail=$((n_fail+1)); continue
  fi
  if [ "$latest" != "$newest" ]; then
    echo "STALE-POINTER $name (latest=$latest newest=$newest) — refusing"
    n_fail=$((n_fail+1)); continue
  fi
  # the done-marker is completion truth, never a progress counter
  if [ ! -f "$d/ckpt/$latest/done" ]; then
    echo "NO-DONE  $name (ckpt/$latest has no done marker) — refusing"
    n_fail=$((n_fail+1)); continue
  fi

  out="$d/se_lic_probe"
  probed="$(python -c "import json;print(json.load(open('$out/se_lic_probe.json'))['ckpt'].rstrip('/').split('/')[-1])" 2>/dev/null || true)"
  # a pre-spec output (no horizon grid / no legs block) is never skippable
  spec_ok=1
  if [ -n "$probed" ]; then
    LIC_JSON="$out/se_lic_probe.json" LIC_HORIZONS="$HORIZONS" python -c '
import json, os, sys
d = json.load(open(os.environ["LIC_JSON"]))
want = [int(x) for x in os.environ["LIC_HORIZONS"].split(",") if x.strip()]
sys.exit(0 if (d.get("instrument") == "se_lic_probe" and "legs" in d
               and d.get("horizons") == want) else 1)' || spec_ok=0
  fi
  if [ -n "$probed" ] && [ "$probed" = "$latest" ] && [ "$spec_ok" = 1 ]; then
    echo "SKIP     $name (matches ckpt/latest $latest)"
    DONE_JSONS+=("$out/se_lic_probe.json")
    n_done=$((n_done+1)); n_skip=$((n_skip+1)); continue
  fi
  if [ -n "$probed" ] && [ "$spec_ok" = 0 ]; then
    echo "PRE-SPEC $name — quarantining and re-running"
    mv "$out" "$d/_PRE_SPEC_se_lic_probe_$(date +%Y%m%d%H%M%S)"
  elif [ -n "$probed" ]; then
    echo "STALE    $name (probed $probed != latest $latest) — quarantining"
    mv "$out" "$d/_INVALID_STALE_CKPT_se_lic_probe_$(date +%Y%m%d%H%M%S)"
  fi

  echo "RUN      $name"
  if python -m uncfield.se_lic_probe --run_logdir "$d" --platform cpu \
        --n_eval "$N_EVAL" --burn_in "$BURN_IN" --horizons "$HORIZONS" \
        --seed "$SEED"; then
    DONE_JSONS+=("$out/se_lic_probe.json")
    n_done=$((n_done+1)); n_ok=$((n_ok+1))
  else
    echo "FAIL     $name"; n_fail=$((n_fail+1))
  fi
done

echo "[$(date)] cells: ok=$n_ok skipped=$n_skip failed=$n_fail of ${#CELLS[@]}"

# ---- L4 cross-seed min-license.  Collated PER ARM: the min is only
# meaningful across seeds sharing an architecture and a training protocol,
# and the two arms differ in disag_head.  Zero extra compute.
#
# EXPECTED-N GATE (review F12): the min is MONOTONE in the number of runs,
# so a partial collate silently reports a looser certificate under the same
# filename.  Each arm is collated only at its full seed count, and the
# matched-n subset size is stamped so the two arms are comparable.
# `n_done` is a counter, not `${#DONE_JSONS[@]}` — an empty array expansion
# under `set -u` errors on older bash (review F15).
N_MATCH="${N_MATCH:-4}"
if [ "$n_done" -gt 1 ]; then
  for arm in gauss det; do
    case "$arm" in
      gauss) expect=4 ;;
      det)   expect=8 ;;
    esac
    [ "$ARMS" = "both" ] || [ "$ARMS" = "$arm" ] || continue
    sel=()
    for j in "${DONE_JSONS[@]}"; do
      h="$(LIC_J="$j" python -c 'import json,os;print(json.load(open(os.environ["LIC_J"]))["disag_head"])' 2>/dev/null || true)"
      [ "$h" = "$arm" ] && sel+=("$j")
    done
    n_sel="${#sel[@]}"
    if [ "$n_sel" -ne "$expect" ]; then
      echo "-- L4 collate SKIPPED: $arm has $n_sel/$expect cells "
      echo "   (a partial collate is n-biased under a monotone min; F12)"
      continue
    fi
    echo "-- L4 collate: $arm ($n_sel seeds, n_match=$N_MATCH) --"
    python -m uncfield.se_lic_probe --collate "${sel[@]}" \
      --expect_n "$expect" --n_match "$N_MATCH" \
      --collate_out "$OUT_ROOT/se_lic_collate_$arm.json"
  done
fi

# realized-counter completion predicate, not a submit ledger
[ "$n_fail" -eq 0 ]
