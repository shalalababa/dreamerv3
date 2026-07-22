#!/bin/bash
# ---------------------------------------------------------------------------
# Route-A R2 probe: local single-GPU driver (5090 / WSL2).
# Registration: prereg/PREREG_gate_d1_r2probe_20260722.md — the prereg +
# read script MUST be freeze-committed BEFORE the first pilot starts.
#
# Stages (run in this order; `labels` refuses to start before `smoke`):
#   ./scripts/d1r2_local.sh pilots   # 12 x 1e5-step d0-probe runs + watcher
#   ./scripts/d1r2_local.sh smoke    # 5-state extended-labeler smoke
#   ./scripts/d1r2_local.sh labels   # 24 labeling passes (200 states each)
# Then:
#   python -m analysis.gate_d1_r2_read --labels $D1R2_ROOT/d1_labels \
#       --output artifacts/gate_d1_r2_<date>
#
# Env: activated dv3 conda env; D1R2_ROOT (default ~/d1r2_local).
# All dials below are pinned by the prereg — do not override.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${D1R2_ROOT:-$HOME/d1r2_local}"
LABELS="$ROOT/d1_labels"
mkdir -p "$ROOT" "$LABELS"
STAGE="${1:?usage: d1r2_local.sh pilots|smoke|labels}"
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

DIALS="--states 200 --horizon 100 --label_every 25 --actions 8 --rollouts 16 --seed 0 --ref_stride 5"

task_of() { case "$1" in cup) echo dmc_cup_catch;; finger) echo dmc_finger_turn_hard;; esac; }
dosecfg_of() { case "$1" in e1) echo "";; e4) echo d0_dose3;; esac; }

python -c "import jax; print('jax devices:', jax.devices())"

if [ "$STAGE" = pilots ]; then
  for dose in e1 e4; do for dom in cup finger; do for seed in 11 12 13; do
    RUN="d1r2_${dom}_${dose}_seed${seed}"
    LOG="$ROOT/$RUN"
    if [ -f "$LOG/TRAINING_DONE" ]; then echo "skip $RUN (done)"; continue; fi
    mkdir -p "$LOG"
    echo "[$(date)] pilot $RUN (task $(task_of "$dom"), dose ${dose})"
    python -m probing.checkpoint_watcher --run_logdir "$LOG" \
        --stop_step 100000 --stop_file "$LOG/TRAINING_DONE" \
        --poll_seconds 20 &
    WPID=$!
    python dreamerv3/main.py --logdir "$LOG" \
        --configs dmc_proprio d0_probe $(dosecfg_of "$dose") \
        --task "$(task_of "$dom")" --seed "$seed" \
        --env.dmc.render False --run.steps 1e5
    touch "$LOG/TRAINING_DONE"
    wait "$WPID" || true
    python -m probing.checkpoint_watcher --select --run_logdir "$LOG" \
        --milestones 25000
  done; done; done
  echo "[$(date)] all 12 pilots done"

elif [ "$STAGE" = smoke ]; then
  SRUN="$ROOT/d1r2_cup_e1_seed11"
  [ -f "$SRUN/TRAINING_DONE" ] || { echo "train d1r2_cup_e1_seed11 first"; exit 1; }
  python -m d0.oracle_labels --run_logdir "$SRUN" \
      --output "$LABELS/d1r2_cup_e1_seed11_late_smoke.npz" \
      --states 5 --horizon 100 --label_every 25 --actions 8 --rollouts 16 \
      --seed 0 --ref_stride 5
  echo "SMOKE OK: determinism assert + extended fields exercised on the"
  echo "real MuJoCo path (5 states; excluded from the read by filename)."
  echo "Any instrument fix now requires a dated amendment BEFORE 'labels'."

elif [ "$STAGE" = labels ]; then
  [ -f "$LABELS/d1r2_cup_e1_seed11_late_smoke.npz" ] || {
    echo "smoke npz missing — run './scripts/d1r2_local.sh smoke' first"; exit 1; }
  for dose in e1 e4; do for dom in cup finger; do for seed in 11 12 13; do
    RUN="d1r2_${dom}_${dose}_seed${seed}"
    LOG="$ROOT/$RUN"
    [ -f "$LOG/TRAINING_DONE" ] || { echo "pilot $RUN not trained"; exit 1; }
    OUT="$LABELS/${RUN}_late.npz"
    if [ ! -f "$OUT" ]; then
      echo "[$(date)] labeling $RUN late"
      python -m d0.oracle_labels --run_logdir "$LOG" --output "$OUT" $DIALS
    fi
    EARLY=$(python - "$LOG" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1] + '/ckpt_snapshots/nearest.json'))
rows = [r for r in rows if r['milestone'] == 25000 and r['snapshot']]
print(rows[0]['snapshot'] if rows else '')
PY
)
    [ -n "$EARLY" ] || { echo "no early snapshot for $RUN"; exit 1; }
    OUT="$LABELS/${RUN}_early.npz"
    if [ ! -f "$OUT" ]; then
      echo "[$(date)] labeling $RUN early ($EARLY)"
      python -m d0.oracle_labels --run_logdir "$LOG" --checkpoint "$EARLY" \
          --output "$OUT" $DIALS
    fi
  done; done; done
  echo "[$(date)] all 24 label passes done; next:"
  echo "  python -m analysis.gate_d1_r2_read --labels $LABELS --output artifacts/gate_d1_r2_$(date +%Y%m%d)"

else
  echo "unknown stage $STAGE"; exit 1
fi
