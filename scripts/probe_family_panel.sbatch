#!/bin/bash
#SBATCH --job-name=dv3_pfam
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# Tier-0 EXPLORATORY: misalignment probe-family dumps (probing/
# probe_family_dump.py; Algorithm_Ideation_20260819 §3.3; consumed by
# analysis/tier0_probe_family.py locally).
#
# ONE job = ONE GPU for the WHOLE 80-run panel — the single-device probe
# discipline (probe-panels-single-gpu). Do not split across jobs.
#
# Panel: finger q1 task / fq1 reward-free / rlq1 own-label / srd0q1+srd1q1
# random stamps, both sides, seeds 1-8 (80 runs). Resumable: runs with an
# existing dump are skipped.
#
# Optional env: EP_BATCH (default 8), PROBESET (default finger_v1).
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

PROBESET="${PROBESET:-$RUNROOT/e4_probesets/finger_v1}"
EP_BATCH="${EP_BATCH:-8}"
[ -f "$PROBESET/FROZEN" ] || { echo "ERROR: $PROBESET is not FROZEN"; exit 2; }
PROBESET_ID="$(basename "$PROBESET")"

n_run=0; n_done=0; n_skip=0
for arm in q1 fq1 rlq1 srd0q1 srd1q1; do
  for logdir in "$RUNROOT"/ax1wm_finger_${arm}s[01]_seed[1-8]; do
    [ -d "$logdir/ckpt" ] || { echo "SKIP no ckpt: $logdir"; n_skip=$((n_skip+1)); continue; }
    run_id="$(basename "$logdir")"
    if [ -f "$logdir/probe_family_${PROBESET_ID}/probe_family_dump.npz" ]; then
      echo "SKIP dumped: $run_id"; n_done=$((n_done+1)); continue
    fi
    echo "[$(date)] probe-family dump $run_id"
    python -m probing.probe_family_dump \
      --run_logdir "$logdir" --probeset "$PROBESET" \
      --platform cuda --ep_batch "$EP_BATCH"
    n_run=$((n_run+1))
  done
done
echo "[$(date)] dumped $n_run new, $n_done already done, $n_skip skipped"
echo "Pull with: rsync the probe_family_${PROBESET_ID} dirs, then run"
echo "  python -m analysis.tier0_probe_family --dumps '<local>/ax1wm_finger_*/probe_family_${PROBESET_ID}' --output artifacts/algo_tier0_20260820/probe_family"
