#!/bin/bash
# ---------------------------------------------------------------------------
# R3 cross-checkpoint consumer wave driver (PREREG_r3_amend2_20260730.md —
# freeze-commit the amendment + the xc-extended d0/oracle_labels.py +
# analysis/r3_xconsumer_read.py + this driver BEFORE any xc label exists).
#
# Substrate: the 32 executed R3 training runs ($R3_ROOT/r3_{cup,finger}_
# {e1,e4}_seed{31..38}, ckpt_early + ckpt). Per run, FOUR label passes:
# eval checkpoint in {early, late} x overlaid consumer heads in
# {early, late}; heads == eval is the CONTROL and still goes through the
# overlay code path (identical counters/RNG schedule => per-state paired
# contrasts; the reader's pairing guard asserts it). 128 passes total,
# labeler seed 1 (fresh states — never paired against the committed R3
# labels), every other dial identical to R3.
#
# Stages (in order; each refuses to run out of order):
#   ./scripts/r3_xc_local.sh gate      # registered existence gate,
#                                      # all-or-nothing; failure writes the
#                                      # SUBSTRATE_GONE marker (registered
#                                      # outcome) and stops the wave
#   ./scripts/r3_xc_local.sh smoke     # 4 dosed smokes (cup e4 seed31,
#                                      # 5 states): both directions + both
#                                      # controls
#   ./scripts/r3_xc_local.sh pairgate  # ONE full-dial pair (cup e1 seed31
#                                      # late: control + swapped), then the
#                                      # reader's --pairgate check — value-
#                                      # blind (episode/step/g_all only)
#   ./scripts/r3_xc_local.sh labels    # the remaining 126 passes
# Then ONE read:
#   python -m analysis.r3_xconsumer_read --labels $R3_ROOT/xc_labels \
#       --output <artifact dir>
#
# Env: activated dv3 env; R3_ROOT (default ~/r3_local). Do NOT edit
# scripts/r3_local.sh — the executed R3 wave's driver stays frozen.
# All dials pinned by the amendment — do not override.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${R3_ROOT:-$HOME/r3_local}"
LABELS="$ROOT/xc_labels"
mkdir -p "$ROOT" "$LABELS"
STAGE="${1:?usage: r3_xc_local.sh gate|smoke|pairgate|labels}"
DOMS="cup finger"
SEEDS="31 32 33 34 35 36 37 38"
DOSES="e1 e4"

# Labeler seed 1 is REGISTERED (fresh state draw, distinct from the R3
# wave's seed 0); the reader pins it from every file's meta.
DIALS="--states 200 --horizon 100 --label_every 25 --actions 8 --rollouts 16 --seed 1 --ref_stride 5 --oracle_all"

ckpt_of() {  # run-logdir maturity -> checkpoint path
  case "$2" in early) echo "$1/ckpt_early";; late) echo "$1/ckpt";; esac
}

xc_pass() {  # dom dose seed evalmat headsmat [--smoke] dials...
  local dom="$1" dose="$2" seed="$3" emat="$4" hmat="$5"; shift 5
  local RUN="r3_${dom}_${dose}_seed${seed}" LOG="$ROOT/r3_${dom}_${dose}_seed${seed}"
  local OUT="$LABELS/xc_${dom}_${dose}_seed${seed}_${emat}_h${hmat}.npz"
  if [ "${1:-}" = "--smoke" ]; then
    OUT="$LABELS/xc_${dom}_${dose}_seed${seed}_${emat}_h${hmat}_smoke.npz"
    shift
  fi
  [ -f "$OUT" ] && { echo "skip $OUT (exists)"; return; }
  [ -f "$LOG/TRAINING_DONE" ] || { echo "ERROR: $RUN not trained"; exit 1; }
  echo "[$(date)] xc label $RUN eval=$emat heads=$hmat -> $OUT"
  python -m d0.oracle_labels --run_logdir "$LOG" \
      --checkpoint "$(ckpt_of "$LOG" "$emat")" \
      --consumer_checkpoint "$(ckpt_of "$LOG" "$hmat")" \
      --output "$OUT" "$@"
}

if [ "$STAGE" = gate ]; then
  # REGISTERED EXISTENCE GATE, all-or-nothing: every one of the 32 run
  # dirs must still hold TRAINING_DONE + ckpt_early + ckpt. Any miss =>
  # the SUBSTRATE-GONE outcome is recorded (marker consumed by the
  # frozen reader) and the wave never starts — no partial substrate.
  MISSING=""
  for dom in $DOMS; do for dose in $DOSES; do for seed in $SEEDS; do
    LOG="$ROOT/r3_${dom}_${dose}_seed${seed}"
    for need in "$LOG/TRAINING_DONE" "$LOG/ckpt_early" "$LOG/ckpt"; do
      [ -e "$need" ] || MISSING="$MISSING $need"
    done
  done; done; done
  if [ -f "$LABELS/SUBSTRATE_GONE" ]; then
    # A registered-outcome marker is never silently erased: re-running
    # gate after a transient mount hiccup must not let the wave proceed
    # as if the outcome never fired. Manual removal + a dated
    # DEVIATIONS entry are required to retry.
    echo "SUBSTRATE_GONE marker already present at $LABELS — refusing"
    echo "to re-run the gate. If the miss was transient, remove the"
    echo "marker MANUALLY and record a dated DEVIATIONS entry first."
    exit 1
  fi
  if [ -n "$MISSING" ]; then
    printf 'missing:%s\n' "$MISSING" > "$LABELS/SUBSTRATE_GONE"
    echo "EXISTENCE GATE FAILED (all-or-nothing). Marker written:"
    echo "  $LABELS/SUBSTRATE_GONE"
    echo "Run the frozen reader on $LABELS to record the registered"
    echo "SUBSTRATE-GONE outcome; do not label."
    exit 1
  fi
  touch "$LABELS/XC_GATE_OK"
  echo "[$(date)] existence gate OK: 32/32 runs with ckpt_early + ckpt"

elif [ "$STAGE" = smoke ]; then
  [ -f "$LABELS/XC_GATE_OK" ] || { echo "run gate first"; exit 1; }
  # Registered smoke gate: 4 dosed (e4) smokes on cup seed31, 5 states —
  # both swap directions AND both controls; the reader asserts the four
  # files. Inspect stdout before pairgate/labels.
  for emat in early late; do for hmat in early late; do
    xc_pass cup e4 31 "$emat" "$hmat" --smoke \
        --states 5 --horizon 100 --label_every 25 --actions 8 \
        --rollouts 16 --seed 1 --ref_stride 5 --oracle_all
  done; done
  touch "$LABELS/XC_SMOKE_OK"
  echo "[$(date)] xc smoke OK (cup e4 seed31: 2 directions + 2 controls)"

elif [ "$STAGE" = pairgate ]; then
  [ -f "$LABELS/XC_SMOKE_OK" ] || { echo "run smoke first"; exit 1; }
  # Registered pairing gate: the ONE full-dial pair (cup e1 seed31,
  # eval=late; control hlate + swapped hearly — these ARE two of the 128
  # wave passes), then the frozen reader's value-blind identity check
  # (episode/step/g_all ONLY; no estimand exists outside those arrays'
  # identity, and none is computed).
  xc_pass cup e1 31 late late $DIALS
  xc_pass cup e1 31 late early $DIALS
  python -m analysis.r3_xconsumer_read --pairgate "$LABELS"
  touch "$LABELS/XC_PAIRGATE_OK"
  echo "[$(date)] pairing gate OK; the wave is authorized"

elif [ "$STAGE" = labels ]; then
  [ -f "$LABELS/XC_GATE_OK" ] || { echo "run gate first"; exit 1; }
  [ -f "$LABELS/XC_SMOKE_OK" ] || { echo "run smoke first"; exit 1; }
  [ -f "$LABELS/XC_PAIRGATE_OK" ] || { echo "run pairgate first"; exit 1; }
  for dom in $DOMS; do for dose in $DOSES; do for seed in $SEEDS; do
    for emat in early late; do for hmat in early late; do
      xc_pass "$dom" "$dose" "$seed" "$emat" "$hmat" $DIALS
    done; done
  done; done; done
  echo "[$(date)] xc labels done (128 passes incl. the pairgate pair)"

else
  echo "unknown stage $STAGE"; exit 1
fi
