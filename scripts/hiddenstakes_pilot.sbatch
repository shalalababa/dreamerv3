#!/bin/bash
#SBATCH --job-name=dv3_hspilot
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# HiddenStakes pilot (execute-first start-gate; user GO 21 Aug 2026).
# NON-REGISTERED pilot — no prereg, descriptive read only; the
# paper-assignment decision (Paper-2 upgrade vs new paper) is OPEN and
# waits on this pilot (research_notes/paper_hiddenstakes/
# Design_HiddenStakes_20260821.md).
#
# Grid: dmc_cheetah_run, size1m proprio, 5e5 steps,
#   p_stay in {0.95, 0.5} x seeds {0, 1}   (4 runs)
# Fixed dose: hiddenstakes dim=4 scale=1.0 cost=0.0 obs_noise=0.5.
# p_stay is the information axis (0.5 = leak-immune no-VoI control:
# h_{t+1} is unpredictable regardless of what the agent infers about
# h_t; build-review finding 9 — reward leaks h at scale 1.0, so
# obs_noise must NOT be the axis).
#
# Question: does p_stay=0.95 beat p_stay=0.5 on eval return by more
# than the computable hazard-exposure difference (hazards are
# anticipatable/waitable only at 0.95)?
#
# Required env: HS_PSTAY (0.95|0.5), SEED (0|1). Optional: STEPS.
# Submit-all (login node, clean shell — module-leakage gotcha):
#   for p in 0.95 0.5; do for s in 0 1; do
#     HS_PSTAY=$p SEED=$s sbatch scripts/hiddenstakes_pilot.sbatch
#   done; done
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${HS_PSTAY:?set HS_PSTAY (0.95|0.5)}"; : "${SEED:?set SEED (0|1)}"
case "$HS_PSTAY" in 0.95|0.5) ;; *) echo "ERROR: HS_PSTAY '$HS_PSTAY' not in {0.95, 0.5}"; exit 2;; esac
case "$SEED" in 0|1) ;; *) echo "ERROR: SEED '$SEED' not in {0, 1} (pilot grid)"; exit 2;; esac
STEPS="${STEPS:-5e5}"
PTAG="$(echo "$HS_PSTAY" | tr -d '.')"
RUN_ID="hspilot_p${PTAG}_seed${SEED}"
LOGDIR="$RUNROOT/hspilot/$RUN_ID"
if [ -f "$LOGDIR/HSPILOT_DONE" ]; then echo "skip $RUN_ID (done)"; exit 0; fi
mkdir -p "$LOGDIR"

echo "[$(date)] hspilot $RUN_ID p_stay=$HS_PSTAY seed=$SEED steps=$STEPS"
python dreamerv3/main.py \
  --logdir "$LOGDIR" --configs dmc_proprio \
  --task dmc_cheetah_run --seed "$SEED" --run.steps "$STEPS" \
  --env.dmc.render False --run.save_every 600 \
  --hiddenstakes.dim 4 --hiddenstakes.p_stay "$HS_PSTAY" \
  --hiddenstakes.scale 1.0 --hiddenstakes.cost 0.0 \
  --hiddenstakes.obs_noise 0.5
touch "$LOGDIR/HSPILOT_DONE"
echo "[$(date)] hspilot $RUN_ID rc=0 status=DONE"
