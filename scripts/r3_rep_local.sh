#!/bin/bash
# ---------------------------------------------------------------------------
# Competence-repair wave driver (PREREG_competence_repair_20260730.md —
# freeze-commit the prereg + d0/train_consumer_model.py + the cm-extended
# d0/oracle_labels.py + analysis/repair_read.py + this driver BEFORE the
# trainer runs and before any repaired label exists).
#
# Substrate: the 32 executed R3 training runs ($R3_ROOT/r3_{cup,finger}_
# {e1,e4}_seed{31..38}, late ckpt) + the 64 COMMITTED R3 label files
# (default $R3_ROOT/r3_labels — the committed wave's own labels dir; a
# bitwise-identical synced copy is equally valid, override R3_COMMITTED).
# 32 repaired passes, LATE cells only, dials IDENTICAL to the committed
# R3 wave (labeler seed 0) + --consumer_model: the committed labels ARE
# the paired control — no control passes are run.
#
# Stages (in order; each refuses to run out of order):
#   ./scripts/r3_rep_local.sh gate    # registered existence gate (32 run
#                                     # dirs + 64 committed labels); any
#                                     # miss writes SUBSTRATE_GONE
#                                     # (registered outcome) and stops
#   ./scripts/r3_rep_local.sh train   # CPU: LORO ridge model from the
#                                     # committed labels; records
#                                     # sha256(model) to
#                                     # rep_labels/CONSUMER_MODEL_SHA256
#                                     # BEFORE any labeling
#   ./scripts/r3_rep_local.sh smoke   # ONE full-dial repaired pass
#                                     # (cup e1 seed31 late — a wave
#                                     # pass) + the reader's --detgate
#                                     # (value-blind episode/step/g_all)
#   ./scripts/r3_rep_local.sh labels  # the remaining 31 passes
# Then ONE read:
#   python -m analysis.repair_read --labels $R3_ROOT/rep_labels \
#       --committed $R3_COMMITTED --model $R3_ROOT/rep_model/consumer_model.npz \
#       --output <artifact dir>
#
# Env: activated dv3 env; R3_ROOT (default ~/r3_local). Do NOT edit
# scripts/r3_local.sh or scripts/r3_xc_local.sh — the executed R3 wave's
# and the xc wave's drivers stay frozen; this is a registered sibling.
# All dials pinned by the prereg — do not override.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${R3_ROOT:-$HOME/r3_local}"
COMMITTED="${R3_COMMITTED:-$ROOT/r3_labels}"
LABELS="$ROOT/rep_labels"
MODEL="$ROOT/rep_model/consumer_model.npz"
mkdir -p "$ROOT" "$LABELS"
STAGE="${1:?usage: r3_rep_local.sh gate|train|smoke|labels}"
DOMS="cup finger"
SEEDS="31 32 33 34 35 36 37 38"
DOSES="e1 e4"

# Labeler seed 0 is REGISTERED — identical to the committed R3 wave:
# that identity (plus the chooser touching nothing upstream of the
# saved row) is what makes the committed labels the paired control.
DIALS="--states 200 --horizon 100 --label_every 25 --actions 8 --rollouts 16 --seed 0 --ref_stride 5 --oracle_all"

rep_pass() {  # dom dose seed
  local dom="$1" dose="$2" seed="$3"
  local RUN="r3_${dom}_${dose}_seed${seed}" LOG="$ROOT/r3_${dom}_${dose}_seed${seed}"
  local OUT="$LABELS/rep_${dom}_${dose}_seed${seed}_late.npz"
  [ -f "$OUT" ] && { echo "skip $OUT (exists)"; return; }
  [ -f "$LOG/TRAINING_DONE" ] || { echo "ERROR: $RUN not trained"; exit 1; }
  echo "[$(date)] repair label $RUN late -> $OUT"
  python -m d0.oracle_labels --run_logdir "$LOG" \
      --consumer_model "$MODEL" \
      --output "$OUT" $DIALS
}

if [ "$STAGE" = gate ]; then
  # REGISTERED EXISTENCE GATE, all-or-nothing: every run dir with
  # TRAINING_DONE + late ckpt AND every one of the 64 committed label
  # files. Any miss => the SUBSTRATE-GONE outcome is recorded (marker
  # consumed by the frozen reader) and the wave never starts.
  MISSING=""
  for dom in $DOMS; do for dose in $DOSES; do for seed in $SEEDS; do
    LOG="$ROOT/r3_${dom}_${dose}_seed${seed}"
    for need in "$LOG/TRAINING_DONE" "$LOG/ckpt"; do
      [ -e "$need" ] || MISSING="$MISSING $need"
    done
    for mat in early late; do
      F="$COMMITTED/r3_${dom}_${dose}_seed${seed}_${mat}.npz"
      [ -f "$F" ] || MISSING="$MISSING $F"
    done
  done; done; done
  if [ -n "$MISSING" ]; then
    printf 'missing:%s\n' "$MISSING" > "$LABELS/SUBSTRATE_GONE"
    echo "EXISTENCE GATE FAILED (all-or-nothing). Marker written:"
    echo "  $LABELS/SUBSTRATE_GONE"
    echo "Run the frozen reader on $LABELS to record the registered"
    echo "SUBSTRATE-GONE outcome; do not train or label."
    exit 1
  fi
  rm -f "$LABELS/SUBSTRATE_GONE"
  touch "$LABELS/REP_GATE_OK"
  echo "[$(date)] existence gate OK: 32/32 runs + 64/64 committed labels"

elif [ "$STAGE" = train ]; then
  [ -f "$LABELS/REP_GATE_OK" ] || { echo "run gate first"; exit 1; }
  if [ -f "$MODEL" ] && [ -f "$LABELS/CONSUMER_MODEL_SHA256" ]; then
    echo "skip train ($MODEL exists; sha recorded)"; exit 0
  fi
  mkdir -p "$(dirname "$MODEL")"
  # Deterministic CPU training on the committed labels (registered as
  # part of the intervention; the trainer prints only lambda choices).
  python -m d0.train_consumer_model --labels "$COMMITTED" --output "$MODEL"
  # Registered protocol: the model sha is recorded BEFORE any labeling;
  # the frozen reader asserts every pass stamped exactly this sha.
  sha256sum "$MODEL" | awk '{print $1}' > "$LABELS/CONSUMER_MODEL_SHA256"
  echo "[$(date)] consumer model trained; sha256 recorded:"
  cat "$LABELS/CONSUMER_MODEL_SHA256"

elif [ "$STAGE" = smoke ]; then
  [ -f "$LABELS/REP_GATE_OK" ] || { echo "run gate first"; exit 1; }
  [ -f "$LABELS/CONSUMER_MODEL_SHA256" ] || { echo "run train first"; exit 1; }
  # Registered smoke: ONE full-dial repaired pass on the registered cell
  # (cup e1 seed31 late — one of the 32 wave passes), then the frozen
  # reader's value-blind determinism gate against the committed label
  # (episode/step/g_all ONLY; no estimand exists outside those arrays'
  # identity, and none is computed).
  rep_pass cup e1 31
  python -m analysis.repair_read --detgate "$LABELS" --committed "$COMMITTED"
  touch "$LABELS/REP_DETGATE_OK"
  echo "[$(date)] determinism gate OK; the wave is authorized"

elif [ "$STAGE" = labels ]; then
  [ -f "$LABELS/REP_GATE_OK" ] || { echo "run gate first"; exit 1; }
  [ -f "$LABELS/CONSUMER_MODEL_SHA256" ] || { echo "run train first"; exit 1; }
  [ -f "$LABELS/REP_DETGATE_OK" ] || { echo "run smoke first"; exit 1; }
  for dom in $DOMS; do for dose in $DOSES; do for seed in $SEEDS; do
    rep_pass "$dom" "$dose" "$seed"
  done; done; done
  echo "[$(date)] repair labels done (32 passes incl. the smoke pass)"

else
  echo "unknown stage $STAGE"; exit 1
fi
