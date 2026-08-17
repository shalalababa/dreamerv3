#!/bin/bash
#SBATCH --job-name=se_donly_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# D-only se_probe passes — PREREG_nfi_scale_exhibit_amend2_20260814 §4.
#
# The amendment pins this as a bundling precondition, in its own words:
#   "D-only runs need their per-run `se_probe` pass (final ckpt, frozen
#    defaults, default output) AFTER training, before bundling; ... the
#    wave's done_when checks do not cover either."
# So it is genuinely owed work, not a nicety, and the wave checker reporting
# 12/12 DONE says nothing about it.
#
# FROZEN DEFAULTS, DEFAULT OUTPUT — both load-bearing:
#   * every knob except --run_logdir is left at the parser's default
#     (n_eval 512, burn_in 16, rollouts 256, horizon 15, topk 10,
#     n_perm 1000, ep_batch 64, platform cpu, seed 0), because the D-only
#     reader compares these outputs against Stage-1 statistics computed with
#     exactly those values;
#   * --output is OMITTED so the tool writes <run>/se_probe. The amendment
#     warns that snapshot passes must use --output <run>/se_probe_snap<pct>
#     and "NEVER the default output"; this is the mirror case — the FINAL
#     ckpt pass is the one that owns the default path.
#
# CPU, hence caslake rather than gpu: se_probe defaults to --platform cpu, so
# it needs no GPU. That also disposes of the device question raised when the
# D-only runs trained on a rented 5060 Ti while Stage-1's probes ran on
# Midway3 — the probe is device-independent by construction, and both sides
# of the comparison are CPU.
#
# 4 cells, bundled: each is minutes-to-an-hour of CPU and the 12-job QOS cap
# makes one job the house pattern for short work.
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

runs=(se_donly_s10 se_donly_s11 se_donly_s12 se_donly_s13)
n_ok=0; n_skip=0; n_fail=0; failed=""

for r in "${runs[@]}"; do
  d="$RUNROOT/$r"
  out="$d/se_probe"
  if [ ! -f "$d/config.yaml" ]; then
    echo "FAIL   $r: no config.yaml"; n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  # The probe reads the final checkpoint via <run>/ckpt/latest; a run whose
  # latest does not resolve to a completed save must not be probed silently.
  latest="$(cat "$d/ckpt/latest" 2>/dev/null || true)"
  if [ -z "$latest" ] || [ ! -f "$d/ckpt/$latest/done" ]; then
    echo "FAIL   $r: ckpt/latest missing or not a completed save ('$latest')"
    n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  if [ -f "$out/se_probe.json" ] || ls "$out"/*.json >/dev/null 2>&1; then
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
[ "$n_fail" -eq 0 ] || exit 1
exit 0
