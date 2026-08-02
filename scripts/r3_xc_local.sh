#!/bin/bash
# ---------------------------------------------------------------------------
# R3 cross-checkpoint consumer wave driver — AMENDED 2026-08-02
# (PREREG_r3_xc_amend1_20260802.md; parent PREREG_r3_amend2_20260730.md).
#
# The parent two-pass design (control + swapped passes, cross-invocation
# g_all pairing at atol 1e-5) is RETIRED UNEXECUTED: the determinism
# probe (artifacts/r3_xc_detprobe_20260801/) measured invocation-level
# g_all drift up to 95.0 on identical states, so its pairgate could
# never pass. The amended wave uses the labeler's --dual_chooser mode:
# ONE pass per (run, eval side) evaluates BOTH choosers — control
# (eval-own heads) and shadow (overlaid other-maturity heads) — on the
# same states, candidates, and g_all. 64 passes, no cross-invocation
# float comparison anywhere.
#
# Substrate: the 32 executed R3 training runs ($R3_ROOT/r3_{cup,finger}_
# {e1,e4}_seed{31..38}, ckpt_early + ckpt). Labeler seed 1 (fresh
# states — never paired against the committed R3 labels), every other
# dial identical to R3.
#
# Stages (in order; each refuses to run out of order):
#   ./scripts/r3_xc_local.sh gate      # registered existence gate,
#                                      # all-or-nothing; failure writes the
#                                      # SUBSTRATE_GONE marker (registered
#                                      # outcome) and stops the wave
#   ./scripts/r3_xc_local.sh smoke     # 2 dual smokes (cup e4 seed31,
#                                      # 5 states, one per eval side —
#                                      # each smokes BOTH choosers)
#   ./scripts/r3_xc_local.sh dualgate  # ONE full-dial dual pass (cup e1
#                                      # seed31 eval=late), then the
#                                      # reader's --dualgate check —
#                                      # value-blind (meta pins,
#                                      # identities, drift bound; no
#                                      # estimand)
#   ./scripts/r3_xc_local.sh labels    # the remaining 63 passes
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
STAGE="${1:?usage: r3_xc_local.sh gate|smoke|dualgate|labels}"
DOMS="cup finger"
SEEDS="31 32 33 34 35 36 37 38"
DOSES="e1 e4"

# Labeler seed 1 is REGISTERED (fresh state draw, distinct from the R3
# wave's seed 0); the reader pins it from every file's meta.
DIALS="--states 200 --horizon 100 --label_every 25 --actions 8 --rollouts 16 --seed 1 --ref_stride 5 --oracle_all --dual_chooser"

ckpt_of() {  # run-logdir maturity -> checkpoint path
  case "$2" in early) echo "$1/ckpt_early";; late) echo "$1/ckpt";; esac
}

other_mat() { case "$1" in early) echo late;; late) echo early;; esac; }

xc_pass() {  # dom dose seed evalmat [--smoke] dials...
  local dom="$1" dose="$2" seed="$3" emat="$4"; shift 4
  local hmat; hmat="$(other_mat "$emat")"
  local RUN="r3_${dom}_${dose}_seed${seed}" LOG="$ROOT/r3_${dom}_${dose}_seed${seed}"
  local OUT="$LABELS/xc_${dom}_${dose}_seed${seed}_${emat}_dual.npz"
  if [ "${1:-}" = "--smoke" ]; then
    OUT="$LABELS/xc_${dom}_${dose}_seed${seed}_${emat}_dual_smoke.npz"
    shift
  fi
  [ -f "$OUT" ] && { echo "skip $OUT (exists)"; return; }
  [ -f "$LOG/TRAINING_DONE" ] || { echo "ERROR: $RUN not trained"; exit 1; }
  echo "[$(date)] xc dual label $RUN eval=$emat consumer=$hmat -> $OUT"
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
  # The amended wave must start clean: two-pass _xc1 files from any
  # earlier manual experimentation would be dead weight (the reader
  # rejects them) but their presence deserves an explicit look.
  if ls "$LABELS"/xc_*_h*.npz >/dev/null 2>&1; then
    echo "WARNING: two-pass (_h*) xc files present under $LABELS — the"
    echo "amended reader ignores them; consider archiving them away."
  fi
  touch "$LABELS/XC_GATE_OK"
  echo "[$(date)] existence gate OK: 32/32 runs with ckpt_early + ckpt"

elif [ "$STAGE" = smoke ]; then
  [ -f "$LABELS/XC_GATE_OK" ] || { echo "run gate first"; exit 1; }
  # Registered smoke gate: 2 dosed (e4) dual smokes on cup seed31,
  # 5 states — one per eval side; each dual pass smokes BOTH choosers,
  # covering both swap directions. Inspect stdout before dualgate.
  for emat in early late; do
    xc_pass cup e4 31 "$emat" --smoke \
        --states 5 --horizon 100 --label_every 25 --actions 8 \
        --rollouts 16 --seed 1 --ref_stride 5 --oracle_all --dual_chooser
  done
  touch "$LABELS/XC_SMOKE_OK"
  echo "[$(date)] xc dual smoke OK (cup e4 seed31: both eval sides)"

elif [ "$STAGE" = dualgate ]; then
  [ -f "$LABELS/XC_SMOKE_OK" ] || { echo "run smoke first"; exit 1; }
  # Registered pre-wave gate: ONE full-dial dual pass (cup e1 seed31,
  # eval=late — this IS one of the 64 wave passes), then the frozen
  # reader's value-blind integrity check (meta pins, finiteness,
  # identities, chooser bounds, candidate-drift bound; NO estimand).
  xc_pass cup e1 31 late $DIALS
  python -m analysis.r3_xconsumer_read --dualgate "$LABELS"
  touch "$LABELS/XC_DUALGATE_OK"
  echo "[$(date)] dual gate OK; the wave is authorized"

elif [ "$STAGE" = labels ]; then
  [ -f "$LABELS/XC_GATE_OK" ] || { echo "run gate first"; exit 1; }
  [ -f "$LABELS/XC_SMOKE_OK" ] || { echo "run smoke first"; exit 1; }
  [ -f "$LABELS/XC_DUALGATE_OK" ] || { echo "run dualgate first"; exit 1; }
  for dom in $DOMS; do for dose in $DOSES; do for seed in $SEEDS; do
    for emat in early late; do
      xc_pass "$dom" "$dose" "$seed" "$emat" $DIALS
    done
  done; done; done
  echo "[$(date)] xc dual labels done (64 passes incl. the dualgate pass)"

else
  echo "unknown stage $STAGE"; exit 1
fi
