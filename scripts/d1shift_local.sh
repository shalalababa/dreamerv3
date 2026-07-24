#!/bin/bash
# ---------------------------------------------------------------------------
# Route-B shift-consequence probe: local single-GPU driver (5090).
# CORRECTED-INSTRUMENT campaign (labeler d1fix_20260724):
# prereg/PREREG_d1_relabel_20260724.md — freeze-commit the prereg +
# repaired labeler + read BEFORE the first relabel pass. Original wave
# (defective labels, preserved): $ROOT/d1_labels + PREREG_d1_shift_20260723.
#
# Stages (run in this order; `labels` refuses to start before `smoke`):
#   ./scripts/d1shift_local.sh pilots   # skips: the 6 pilots exist
#   ./scripts/d1shift_local.sh smoke    # 5-state xpol + phys smokes
#   ./scripts/d1shift_local.sh labels   # 18 relabel passes (3 arms x 6)
# Then:
#   python -m analysis.d1_shift_read --labels $D1S_ROOT/d1_labels_fix \
#       --output artifacts/d1_shift_fix_<date>
#
# Env: activated dv3 conda env; D1S_ROOT (default ~/d1shift_local).
# All dials below are pinned by the prereg — do not override.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${D1S_ROOT:-$HOME/d1shift_local}"
LABELS="$ROOT/d1_labels_fix"
mkdir -p "$ROOT" "$LABELS"
STAGE="${1:?usage: d1shift_local.sh pilots|smoke|labels}"

DIALS="--states 200 --horizon 100 --label_every 25 --actions 8 --rollouts 16 --seed 0 --ref_stride 5 --oracle_all"
MASS=1.3

task_of() { case "$1" in cup) echo dmc_cup_catch;; finger) echo dmc_finger_turn_hard;; esac; }
sib_of()  { case "$1" in 21) echo 22;; 22) echo 23;; 23) echo 21;; esac; }

python -c "import jax; print('jax devices:', jax.devices())"

if [ "$STAGE" = pilots ]; then
  for dom in cup finger; do for seed in 21 22 23; do
    RUN="d1s_${dom}_seed${seed}"
    LOG="$ROOT/$RUN"
    if [ -f "$LOG/TRAINING_DONE" ]; then echo "skip $RUN (done)"; continue; fi
    mkdir -p "$LOG"
    echo "[$(date)] pilot $RUN (task $(task_of "$dom"), dose e1)"
    python dreamerv3/main.py --logdir "$LOG" \
        --configs dmc_proprio d0_probe \
        --task "$(task_of "$dom")" --seed "$seed" \
        --env.dmc.render False --run.steps 1e5
    touch "$LOG/TRAINING_DONE"
  done; done
  echo "[$(date)] all 6 pilots done"

elif [ "$STAGE" = smoke ]; then
  SRUN="$ROOT/d1s_cup_seed21"
  BEH="$ROOT/d1s_cup_seed22/ckpt"
  [ -f "$SRUN/TRAINING_DONE" ] || { echo "train d1s_cup_seed21 first"; exit 1; }
  [ -e "$BEH" ] || { echo "behavior ckpt $BEH missing"; exit 1; }
  python -m d0.oracle_labels --run_logdir "$SRUN" \
      --behavior_checkpoint "$BEH" \
      --output "$LABELS/d1s_cup_seed21_xpol_smoke.npz" \
      --states 5 --horizon 100 --label_every 25 --actions 8 --rollouts 16 \
      --seed 0 --ref_stride 5 --oracle_all
  python -m d0.oracle_labels --run_logdir "$SRUN" \
      --mass_scale "$MASS" \
      --output "$LABELS/d1s_cup_seed21_phys_smoke.npz" \
      --states 5 --horizon 100 --label_every 25 --actions 8 --rollouts 16 \
      --seed 0 --ref_stride 5 --oracle_all
  echo "SMOKE OK: both shift arms exercised the real MuJoCo path (5"
  echo "states each; determinism assert covers CRN under mass scaling;"
  echo "smoke npz excluded from the read by filename). Any instrument"
  echo "fix now requires a dated amendment BEFORE 'labels'."

elif [ "$STAGE" = labels ]; then
  [ -f "$LABELS/d1s_cup_seed21_xpol_smoke.npz" ] &&
  [ -f "$LABELS/d1s_cup_seed21_phys_smoke.npz" ] || {
    echo "smoke npz missing — run './scripts/d1shift_local.sh smoke' first"; exit 1; }
  for dom in cup finger; do for seed in 21 22 23; do
    RUN="d1s_${dom}_seed${seed}"
    LOG="$ROOT/$RUN"
    [ -f "$LOG/TRAINING_DONE" ] || { echo "pilot $RUN not trained"; exit 1; }
    BEH="$ROOT/d1s_${dom}_seed$(sib_of "$seed")/ckpt"
    OUT="$LABELS/${RUN}_base.npz"
    if [ ! -f "$OUT" ]; then
      echo "[$(date)] labeling $RUN base"
      python -m d0.oracle_labels --run_logdir "$LOG" --output "$OUT" $DIALS
    fi
    OUT="$LABELS/${RUN}_xpol.npz"
    if [ ! -f "$OUT" ]; then
      echo "[$(date)] labeling $RUN xpol (behavior $BEH)"
      python -m d0.oracle_labels --run_logdir "$LOG" --output "$OUT" \
          --behavior_checkpoint "$BEH" $DIALS
    fi
    OUT="$LABELS/${RUN}_phys.npz"
    if [ ! -f "$OUT" ]; then
      echo "[$(date)] labeling $RUN phys (mass x$MASS)"
      python -m d0.oracle_labels --run_logdir "$LOG" --output "$OUT" \
          --mass_scale "$MASS" $DIALS
    fi
  done; done
  echo "[$(date)] all 18 label passes done; next:"
  echo "  python -m analysis.d1_shift_read --labels $LABELS --output artifacts/d1_shift_$(date +%Y%m%d)"

else
  echo "unknown stage $STAGE"; exit 1
fi
