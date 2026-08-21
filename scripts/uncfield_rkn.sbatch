#!/bin/bash
#SBATCH --job-name=rkntrackc
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A.out
# ---------------------------------------------------------------------------
# Track-C RKN-class coherent operator — PREREG_trackC_rkn_20260821.md
# (review #33 applied). ONE sequential CPU job: selfcheck, then train
# 4 Kalman-structured members on the FROZEN pilot2 episodes
# (~8.5 min/member measured locally — #33 F21), then the exploit
# search + consistency rows (seconds/member on this operator), then
# the heldout row (incl. the diagonal exact floor) and the
# model-in-the-loop CEI audit. 4h = wide margin.
# Outputs: local_results/uncfield/rkn_pilot2/{ensemble.pkl,
#   train_info.json, search_m0..3.json, results_m0..3.npz,
#   heldout.json, audit.json}
# Collect per results-sync v2 before the registered adjudication.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
# #33 F15: the 75 MB training input is gitignored — fail fast if the
# sync did not stage it
test -f "$REPO/local_results/uncfield/pilot2/episodes.pkl" || {
  echo "FATAL: local_results/uncfield/pilot2/episodes.pkl not staged"
  exit 1
}
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
export JAX_PLATFORMS=cpu
echo "[$(date)] rkn selfcheck"
python -m uncfield.rknwm --selfcheck
echo "[$(date)] rkn train"
python -m uncfield.rknwm --train --tag rkn_pilot2 --verbose
echo "[$(date)] rkn search"
python -m uncfield.rknwm --search --tag rkn_pilot2
echo "[$(date)] rkn heldout + audit"
python -m uncfield.rknwm --heldout --tag rkn_pilot2
python -m uncfield.rknwm --audit --tag rkn_pilot2
echo "[$(date)] DONE"
