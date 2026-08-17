#!/bin/bash
#SBATCH --job-name=se_grad_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# se_probe passes for the SE gradient arm — PREREG_nfi_scale_exhibit_amend3 §5.
#
# EIGHT cells, not four: the bundle spec gives se_probe to hetero AND flat
#   hetero  se_het_s28..31    -> the MANIFESTATION-CHECK GATE input (§4)
#   flat    se_flat_s36..39   -> the amplitude-dose descriptive
#   ungated2                  -> explicitly NOT probed
#
# The hetero probes are load-bearing in a way the flat ones are not: §4 makes
# the read's validity conditional on them (distractor p_perm < .05 in >= 3 of 4
# hetero runs), and registers that FEWER THAN 3 VALID probes is outcome cell
# iv, "manipulation check NOT-ADJUDICABLE" -- an instrument failure, never a
# finding. So a probe that fails here must fail loudly and be re-run, not be
# quietly absent from the bundle.
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

runs=(se_het_s28 se_het_s29 se_het_s30 se_het_s31
      se_flat_s36 se_flat_s37 se_flat_s38 se_flat_s39)
n_ok=0; n_skip=0; n_fail=0; failed=""

for r in "${runs[@]}"; do
  d="$RUNROOT/$r"
  out="$d/se_probe"
  if [ ! -f "$d/config.yaml" ]; then
    echo "FAIL   $r: no config.yaml"; n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  # The probe reads the final checkpoint via <run>/ckpt/latest. A run whose
  # latest does not resolve to a COMPLETED save must not be probed silently --
  # that would feed the manipulation-check gate an unfinished model.
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
# A hetero failure is materially worse than a flat one -- it feeds the
# registered validity gate. Surface that distinction rather than a bare count.
for f in $failed; do
  case "$f" in se_het_*) echo "NOTE: $f is a HETERO cell -- fewer than 3 valid hetero probes puts the read in outcome cell iv";; esac
done
[ "$n_fail" -eq 0 ] || exit 1
exit 0
