#!/bin/bash
#SBATCH --job-name=se_noboot_probe
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# NOBOOT se_probe passes — PREREG_nfi_scale_exhibit_amend4_20260820 §1.
#
# The amendment pins this as a bundling precondition, in its own words:
#   "Per-run `se_probe` at the final checkpoint (frozen defaults, default
#    output, one device class) — the §5-shaped bundle: replay/, scores,
#    config, metrics, ckpt/latest, se_probe/."
# So it is genuinely owed work, not a nicety, and the wave checker reporting
# 4/4 DONE says nothing about it: ops/waves/se_noboot/spec.yaml states outright
# that "se_probe passes are NOT part of this wave". The frozen reader opens
# <run>/se_probe/se_probe_dims.npz and <run>/se_probe/se_probe.json per run
# (se_noboot_read.py:74-79), so without this pass the read cannot execute.
#
# FROZEN DEFAULTS, DEFAULT OUTPUT — both load-bearing:
#   * every knob except --run_logdir is left at the parser's default
#     (n_eval 512, burn_in 16, rollouts 256, horizon 15, topk 10,
#     n_perm 1000, ep_batch 64, platform cpu, seed 0), because the NOBOOT
#     reader's degeneracy gate compares these outputs against Stage-1
#     per-run statistics loaded live via --stage1_runs, which were computed
#     with exactly those values;
#   * --output is OMITTED so the tool writes <run>/se_probe — the path the
#     reader hardcodes. Snapshot passes use --output <run>/se_probe_snap<pct>
#     and never the default; the FINAL ckpt pass is the one that owns it.
#
# CPU, hence caslake rather than gpu: se_probe defaults to --platform cpu, so
# it needs no GPU. That also disposes of the device question raised by the
# NOBOOT runs training on a rented 5060 Ti while Stage-1's probes ran on
# Midway3 — the probe is device-independent by construction, and both sides
# of the comparison are CPU. One job = one node = "one device class".
#
# 4 cells, bundled: each is minutes-to-an-hour of CPU and the 12-job QOS cap
# makes one job the house pattern for short work.
#
# ARM IDENTITY. These are the REGENERATED runs (the 21-Aug wave ran the
# bootstrapped arm under noboot run_ids — see ops/waves/se_noboot/
# INCIDENT_20260821.md). The spoiled originals are quarantined as
# _SPOILED_BOOTON_se_noboot_s4*, which the run list below cannot reach.
# Verified 22 Aug before submission: config.yaml carries
# `expl: {..., disag_bootstrap: false, ...}` on all four, and each run's
# ckpt/latest resolves to its only completed save.
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

runs=(se_noboot_s40 se_noboot_s41 se_noboot_s42 se_noboot_s43)
n_ok=0; n_skip=0; n_fail=0; failed=""

for r in "${runs[@]}"; do
  d="$RUNROOT/$r"
  out="$d/se_probe"
  if [ ! -f "$d/config.yaml" ]; then
    echo "FAIL   $r: no config.yaml"; n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  # The arm is defined by this flag; a run wearing the run_id while carrying
  # the bootstrapped config is exactly the 21-Aug incident, and the reader
  # would refuse it later. Refuse it here instead, anchored to the expl block.
  if ! python3 - "$d/config.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
sys.exit(0 if cfg["agent"]["expl"]["disag_bootstrap"] is False else 1)
PY
  then
    echo "FAIL   $r: agent.expl.disag_bootstrap is not False (wrong arm)"
    n_fail=$((n_fail+1)); failed="$failed $r"; continue
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
