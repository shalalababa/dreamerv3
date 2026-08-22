#!/bin/bash
#SBATCH --job-name=se_b4_gate_mask
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# B4 FREEZE GATE -- det se_mask baseline. PREREG_trackB_alea_20260822 §0.
#
# The amendment states the reason in its own words: the manipulation check
# "rests on a quantity (`se_mask`-form delta of the intrinsic under distractor
# permutation) that has NEVER been measured on a full Stage-1 det run (the only
# measurements are a 2e4-step smoke and the APT arm -- both with POSITIVE
# deltas)". If this baseline does not show a reliably NEGATIVE distractor
# delta, the manipulation check is re-specified in relative-share form by dated
# amendment BEFORE the B4 wave is submitted.
#
# STATUS: descriptive instrument-calibration, NOT a registered read. It is
# value-aware by design (it informs the check's FORM) and the prereg licenses
# it explicitly because it runs before any B4 data exist.
#
# FROZEN DEFAULTS: only --run_logdir is passed. n_eval 512, burn_in 16,
# n_perm 1000, n_boot 2000, ep_batch 64, platform cpu, seed 0 all stay at the
# parser default, and --output is omitted so the tool writes <run>/se_mask.
#
# CPU, hence caslake: se_mask defaults to --platform cpu and needs no GPU.
#
# WRITES INTO SHARED SUBSTRATE. se_cheetah_seed10-17 are the Stage-1 dirs that
# every --stage1_runs consumer reads. This pass ADDS <run>/se_mask and touches
# nothing else, but that is exactly the "producers write outputs into input
# trees" case that has cost this project twice, so: a run whose se_mask/
# already parses is SKIPPED rather than overwritten, and nothing else in the
# run dir is written.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUNROOT:?set RUNROOT}"
STAGE1="${STAGE1:-$RUNROOT}"

echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}"
echo "[note] platform=cpu (se_mask default); frozen defaults; default output"
echo "[note] PREREG_trackB_alea_20260822 §0 -- calibration, not a registered read"
echo

n_ok=0; n_skip=0; n_fail=0; failed=""

for s in 10 11 12 13 14 15 16 17; do
  r="se_cheetah_seed$s"
  d="$STAGE1/$r"
  out="$d/se_mask"
  if [ ! -f "$d/config.yaml" ]; then
    echo "FAIL   $r: no config.yaml"; n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  latest="$(cat "$d/ckpt/latest" 2>/dev/null || true)"
  if [ -z "$latest" ] || [ ! -f "$d/ckpt/$latest/done" ]; then
    echo "FAIL   $r: ckpt/latest missing or not a completed save ('$latest')"
    n_fail=$((n_fail+1)); failed="$failed $r"; continue
  fi
  if ls "$out"/*.json >/dev/null 2>&1; then
    echo "SKIP   $r (se_mask already present)"; n_skip=$((n_skip+1)); continue
  fi
  echo "[$(date)] MASK $r  (ckpt $latest)"
  python -m uncfield.se_mask --run_logdir "$d" --platform cpu
  rc=$?
  if [ "$rc" -eq 0 ] && ls "$out"/*.json >/dev/null 2>&1; then
    echo "OK     $r"; n_ok=$((n_ok+1))
  else
    echo "FAIL   $r (rc=$rc)"; n_fail=$((n_fail+1)); failed="$failed $r"
  fi
done

echo
echo "[$(date)] se_mask gate: ok=$n_ok skipped=$n_skip failed=$n_fail (of 8)"
[ -n "$failed" ] && echo "FAILED CELLS:$failed"
[ "$n_fail" -eq 0 ] || exit 1
exit 0
