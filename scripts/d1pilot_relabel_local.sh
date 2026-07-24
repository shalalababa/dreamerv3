#!/bin/bash
# ---------------------------------------------------------------------------
# Amendment 1 to the D1 relabel campaign: 18 additional corrected-label
# passes on the SIX GATE-D1 PILOTS (cluster runs d1pilot_{cup,finger}_e1_
# seed{1,2,3}, confirmed alive 24 Jul). Registration:
# prereg/PREREG_d1_relabel_amend1_20260724.md — freeze-commit BEFORE the
# first d1pilot label pass. The frozen base driver (d1shift_local.sh) is
# untouched; this is its companion.
#
# Stages (in order):
#   ./scripts/d1pilot_relabel_local.sh pull    # prints/executes the rsync
#   ./scripts/d1pilot_relabel_local.sh smoke   # 5-state base + xpol smokes
#   ./scripts/d1pilot_relabel_local.sh labels  # 18 passes (3 arms x 6)
# Then ONE read execution over BOTH cohorts (see the amendment):
#   python -m analysis.d1_shift_read --labels $D1S_ROOT/d1_labels_fix \
#       --output artifacts/d1_shift_fix_<date>
#
# Env: activated dv3 conda env; D1S_ROOT (default ~/d1shift_local);
# REMOTE = cluster ssh host, REMOTE_RUNROOT = cluster runroot (pull only).
# Dials are pinned by the base prereg — do not override.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${D1S_ROOT:-$HOME/d1shift_local}"
LABELS="$ROOT/d1_labels_fix"
mkdir -p "$ROOT" "$LABELS"
STAGE="${1:?usage: d1pilot_relabel_local.sh pull|smoke|labels}"

DIALS="--states 200 --horizon 100 --label_every 25 --actions 8 --rollouts 16 --seed 0 --ref_stride 5 --oracle_all"
MASS=1.3

sib_of() { case "$1" in 1) echo 2;; 2) echo 3;; 3) echo 1;; esac; }

if [ "$STAGE" = pull ]; then
  : "${REMOTE:?set REMOTE=<cluster ssh host>}"
  : "${REMOTE_RUNROOT:?set REMOTE_RUNROOT=<cluster runroot with d1pilot_*>}"
  for dom in cup finger; do for seed in 1 2 3; do
    RUN="d1pilot_${dom}_e1_seed${seed}"
    echo "[$(date)] pulling $RUN"
    rsync -av --include 'ckpt' --include 'ckpt/**' \
        --include 'config.yaml' --exclude '*' \
        "$REMOTE:$REMOTE_RUNROOT/$RUN/" "$ROOT/$RUN/"
    [ -e "$ROOT/$RUN/ckpt" ] || { echo "PULL FAILED: $RUN has no ckpt"; exit 1; }
    [ -f "$ROOT/$RUN/config.yaml" ] || { echo "PULL FAILED: $RUN has no config"; exit 1; }
  done; done
  echo "[$(date)] all 6 pilot checkpoints pulled"

elif [ "$STAGE" = smoke ]; then
  SRUN="$ROOT/d1pilot_cup_e1_seed1"
  BEH="$ROOT/d1pilot_cup_e1_seed2/ckpt"
  [ -e "$SRUN/ckpt" ] || { echo "pull d1pilot ckpts first"; exit 1; }
  [ -e "$BEH" ] || { echo "behavior ckpt $BEH missing"; exit 1; }
  python -m d0.oracle_labels --run_logdir "$SRUN" \
      --output "$LABELS/d1pilot_cup_e1_seed1_base_smoke.npz" \
      --states 5 --horizon 100 --label_every 25 --actions 8 --rollouts 16 \
      --seed 0 --ref_stride 5 --oracle_all
  python -m d0.oracle_labels --run_logdir "$SRUN" \
      --behavior_checkpoint "$BEH" \
      --output "$LABELS/d1pilot_cup_e1_seed1_xpol_smoke.npz" \
      --states 5 --horizon 100 --label_every 25 --actions 8 --rollouts 16 \
      --seed 0 --ref_stride 5 --oracle_all
  echo "SMOKE OK: pulled checkpoints load and label through the real"
  echo "MuJoCo path (own-policy + behavior-driven). Smoke npz excluded"
  echo "from the read by filename. Any instrument fix now requires a"
  echo "dated amendment BEFORE 'labels'."

elif [ "$STAGE" = labels ]; then
  [ -f "$LABELS/d1pilot_cup_e1_seed1_base_smoke.npz" ] &&
  [ -f "$LABELS/d1pilot_cup_e1_seed1_xpol_smoke.npz" ] || {
    echo "smoke npz missing — run './scripts/d1pilot_relabel_local.sh smoke' first"; exit 1; }
  for dom in cup finger; do for seed in 1 2 3; do
    RUN="d1pilot_${dom}_e1_seed${seed}"
    LOG="$ROOT/$RUN"
    [ -e "$LOG/ckpt" ] || { echo "pilot $RUN not pulled"; exit 1; }
    BEH="$ROOT/d1pilot_${dom}_e1_seed$(sib_of "$seed")/ckpt"
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
  echo "[$(date)] all 18 d1pilot label passes done; run the SINGLE read"
  echo "over both cohorts once the d1s passes are also complete:"
  echo "  python -m analysis.d1_shift_read --labels $LABELS --output artifacts/d1_shift_fix_$(date +%Y%m%d)"

else
  echo "unknown stage $STAGE"; exit 1
fi
