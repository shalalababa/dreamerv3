#!/bin/bash
#SBATCH --job-name=dv3_tm2r3
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# TM2-R3 opportunity/competence cross-family replication
# (prereg/PREREG_tm2_competence_20260730.md — freeze-commit the prereg +
# probing/tdmpc2_oracle_labels.py + probing/tdmpc2_r3_train.py +
# analysis/tm2_r3_read.py + the tdmpc2_compat dose patch BEFORE any run).
#
# Grid: {cup, finger} x {e1, e4} x seeds 51-58 (disjoint from every prior
# cohort), each run labeled at two maturities (early = ckpt_early.pt
# ~2.5e4 realized steps, late = ckpt_late.pt at 1e5) = 32 fresh ONLINE
# runs, 64 oracle_all label passes. e1 = no distractor; e4 = d0_dose3
# (dim 32, scale 3.0) composed exactly as DreamerV3 composes it.
#
# Stages (TM2R3_STAGE; ordering enforced by marker files; `labels`
# refuses to start before BOTH smoke gates):
#   train    one run/job:   TM2R3_DOM, TM2R3_DOSE, TM2R3_SEED
#   apismoke once:          random-init API smoke (no checkpoint, no
#                           outcome) -> labels/tm2r3_apismoke_smoke.npz
#   smoke    once:          4 dosed real smokes, {cup,finger} x {e1,e4}
#                           on seed-51 LATE ckpts, 5 states each
#   labels   one pass/job:  TM2R3_DOM, TM2R3_DOSE, TM2R3_SEED, TM2R3_MAT
# Then ONE read:
#   python -m analysis.tm2_r3_read --labels $TM2R3_ROOT/labels --output ...
#
# WAVE-1 first-update stages (PREREG_p2_wave1_firstupdate_20260821;
# LATE ckpts, seeds 51-58 ONLY, outputs to $TM2R3_ROOT/wave1_labels):
#   wv1dup   once:          the 2 registered CRN null gates
#                           (finger_e1_seed51 + cup_e4_seed58) ->
#                           wv1dup_*_late.npz + WV1_DUP_OK marker
#   wv1      one pass/job:  TM2R3_DOM, TM2R3_DOSE, TM2R3_SEED
#                           (refuses before WV1_DUP_OK)
# Registered ordering (procedural, prereg Ops): submit the 4 PILOT
# cells first (cup_e1_seed51 cup_e4_seed52 finger_e1_seed53
# finger_e4_seed54), let the analyst run
#   python -m analysis.p2_wave1_read --pilot ...   (dispersion-only)
# and record it, then submit the remaining 28. The wave proceeds
# regardless of the pilot unless the registered re-power trigger fires.
# Then ONE read: python -m analysis.p2_wave1_read --labels ... .
#
# WAVE-2 W1-PROTOCOL labeling stages
# (PREREG_p2_wave2_freshcohort_20260821: "labeling exactly per
# PREREG_w1_wave_20260809" — same flags, --w1_repeats 8,
# --env_seed 20260809 LOAD-BEARING, standard dials; outputs to the
# registered $RUNROOT/w1_labels dir beside the look-1 cohort, names
# disjoint by seed; seeds 59-66 ONLY, the exact complement of wv1):
#   w2smoke  once:          W1-protocol 5-state smoke on fresh LATE
#                           ckpts (finger_e1_seed59 main +
#                           cup_e1_seed59 dup_cand — the 20260809
#                           wave's registered cheap-insurance smoke
#                           mirrored) -> *_smoke.npz + W2_SMOKE_OK
#   w2dup    once:          2 duplicate-null gate passes, mirror of
#                           look-1's cup_e1+finger_e1 at the cohort's
#                           first seed -> w1tm2dup_*_seed59_late.npz
#                           + W2_DUP_OK
#   w2labels one pass/job:  TM2R3_DOM, TM2R3_DOSE, TM2R3_SEED
#                           (refuses before W2_DUP_OK) ->
#                           w1tm2_{dom}_{dose}_seed{59..66}_late.npz
# Then ONE read: python -m analysis.w1_read_fresh tm2 ... plus the
# pre-named pooled secondary via analysis/w1_read_pooled.py.
#
# Submit-all examples (login node, clean shell — module-leakage gotcha):
#   for d in cup finger; do for e in e1 e4; do for k in $(seq 51 58); do
#     TM2R3_STAGE=train TM2R3_DOM=$d TM2R3_DOSE=$e TM2R3_SEED=$k \
#       sbatch scripts/tm2r3.sbatch
#   done; done; done
#   TM2R3_STAGE=apismoke sbatch scripts/tm2r3.sbatch
#   TM2R3_STAGE=smoke sbatch scripts/tm2r3.sbatch
#   for d in cup finger; do for e in e1 e4; do for k in $(seq 51 58); do
#     for m in early late; do
#       TM2R3_STAGE=labels TM2R3_DOM=$d TM2R3_DOSE=$e TM2R3_SEED=$k \
#         TM2R3_MAT=$m sbatch scripts/tm2r3.sbatch
#   done; done; done; done
#   TM2R3_STAGE=wv1dup sbatch scripts/tm2r3.sbatch
#   # pilot 4 first, then the rest:
#   for cell in cup:e1:51 cup:e4:52 finger:e1:53 finger:e4:54; do
#     IFS=: read d e k <<<"$cell"
#     TM2R3_STAGE=wv1 TM2R3_DOM=$d TM2R3_DOSE=$e TM2R3_SEED=$k \
#       sbatch scripts/tm2r3.sbatch
#   done
#   TM2R3_STAGE=w2smoke sbatch scripts/tm2r3.sbatch
#   TM2R3_STAGE=w2dup sbatch scripts/tm2r3.sbatch
#   for d in cup finger; do for e in e1 e4; do for k in $(seq 59 66); do
#     TM2R3_STAGE=w2labels TM2R3_DOM=$d TM2R3_DOSE=$e TM2R3_SEED=$k \
#       sbatch scripts/tm2r3.sbatch
#   done; done; done
#
# All labeling dials pinned by the preregs — do not override.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"

TM2_CONDA_ENV="${TM2_CONDA_ENV:-/scratch/midway3/$USER/conda_envs/tdmpc2}"
TDMPC2_CHECKOUT="${TDMPC2_CHECKOUT:-/scratch/midway3/$USER/tdmpc2}"
export TDMPC2_ROOT="$TDMPC2_CHECKOUT/tdmpc2"
CONDA_ENV="$TM2_CONDA_ENV" activate_env
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Env-setup existence check (registered ordering step): pinned checkout.
TM2R3_PIN="e9f59321933cbc8e11a002b842adc7d4ffae8ff1"
[ -d "$TDMPC2_ROOT" ] || { echo "ERROR: no TDMPC2_ROOT at $TDMPC2_ROOT (run scripts/tdmpc2_env_setup.sh)"; exit 2; }
HEAD="$(git -C "$TDMPC2_CHECKOUT" rev-parse HEAD)"
[ "$HEAD" = "$TM2R3_PIN" ] || { echo "ERROR: tdmpc2 checkout $HEAD != pin $TM2R3_PIN"; exit 2; }

TM2R3_ROOT="${TM2R3_ROOT:-$RUNROOT/tm2r3}"
LABELS="$TM2R3_ROOT/labels"
WV1_LABELS="$TM2R3_ROOT/wave1_labels"
# Wave-2 W1-protocol labels: the PREREG_w1_wave_20260809 registered dir
# (KEEP_PENDING-protected), shared with the look-1 cohort — the pooled
# secondary reads both cohorts from one dir; names disjoint by seed.
W1_LABELS="${W1_LABELS_DIR:-$RUNROOT/w1_labels}"
mkdir -p "$TM2R3_ROOT" "$LABELS"
STAGE="${TM2R3_STAGE:?train|apismoke|smoke|labels|wv1dup|wv1|w2smoke|w2dup|w2labels}"

DIALS="--states 200 --horizon 100 --label_every 25 --actions 8 --seed 0 --ref_stride 5 --oracle_all"
task_of() { case "$1" in cup) echo dmc_cup_catch;; finger) echo dmc_finger_turn_hard;; *) echo "ERROR: dom $1" >&2; exit 2;; esac; }
# Registered-grid validation: a typo'd cell would train/label a stray
# run whose file then hard-blocks the ONE read (unregistered-seed
# assert) until manually removed — refuse at submission instead.
check_cell() {
  case "${TM2R3_DOM:-}" in cup|finger) ;; *) echo "ERROR: TM2R3_DOM '${TM2R3_DOM:-}' not in {cup,finger}"; exit 2;; esac
  case "${TM2R3_DOSE:-}" in e1|e4) ;; *) echo "ERROR: TM2R3_DOSE '${TM2R3_DOSE:-}' not in {e1,e4}"; exit 2;; esac
  case "${TM2R3_SEED:-}" in 5[1-8]|59|6[0-6]) ;; *) echo "ERROR: TM2R3_SEED '${TM2R3_SEED:-}' not in 51..66 (registered grids: W1 51-58, fresh cohort 59-66 per PREREG_p2_wave2_freshcohort_20260821)"; exit 2;; esac
}

if [ "$STAGE" = train ]; then
  : "${TM2R3_DOM:?cup|finger}"; : "${TM2R3_DOSE:?e1|e4}"; : "${TM2R3_SEED:?51..58}"
  check_cell
  RUN="tm2r3_${TM2R3_DOM}_${TM2R3_DOSE}_seed${TM2R3_SEED}"
  LOG="$TM2R3_ROOT/$RUN"
  if [ -f "$LOG/TM2R3_TRAIN_DONE" ]; then echo "skip $RUN (done)"; exit 0; fi
  echo "[$(date)] train $RUN"
  python -m probing.tdmpc2_r3_train --logdir "$LOG" \
      --task "$(task_of "$TM2R3_DOM")" --dose "$TM2R3_DOSE" \
      --seed "$TM2R3_SEED" --steps 100000 --early_steps 25000
  echo "[$(date)] train $RUN rc=0 status=DONE"

elif [ "$STAGE" = apismoke ]; then
  # Registered smoke gate (i): random-init model, no checkpoint touched,
  # no outcome revealed (PREREG_tdmpc2_f2 precedent), on the DOSED env —
  # exercises the OU snapshot path + planner + torch-RNG CRN plumbing.
  OUT="$LABELS/tm2r3_apismoke_smoke.npz"
  [ -f "$OUT" ] && { echo "skip apismoke (exists)"; exit 0; }
  python -m probing.tdmpc2_oracle_labels --smoke_random_init \
      --task dmc_cup_catch --dose e4 \
      --states 2 --horizon 10 --label_every 25 --actions 8 --seed 0 \
      --ref_stride 5 --oracle_all --output "$OUT"
  echo "[$(date)] apismoke OK -> $OUT"

elif [ "$STAGE" = smoke ]; then
  # Registered smoke gate (ii): 4 dosed REAL smokes across {dom}x{dose}
  # on the seed-51 LATE checkpoints, 5 states each (the reader asserts
  # these four files + the apismoke exist).
  [ -f "$LABELS/tm2r3_apismoke_smoke.npz" ] || { echo "run apismoke first"; exit 1; }
  for dom in cup finger; do for dose in e1 e4; do
    RUN="tm2r3_${dom}_${dose}_seed51"
    LOG="$TM2R3_ROOT/$RUN"
    OUT="$LABELS/${RUN}_late_smoke.npz"
    [ -f "$LOG/TM2R3_TRAIN_DONE" ] || { echo "train $RUN first"; exit 1; }
    [ -f "$OUT" ] && { echo "skip $OUT (exists)"; continue; }
    python -m probing.tdmpc2_oracle_labels \
        --run_logdir "$LOG" --checkpoint "$LOG/ckpt_late.pt" \
        --states 5 --horizon 100 --label_every 25 --actions 8 --seed 0 \
        --ref_stride 5 --oracle_all --output "$OUT"
  done; done
  touch "$LABELS/SMOKE_OK"
  echo "[$(date)] smoke OK (cup+finger x e1+e4 late; inspect stdout before labels)"

elif [ "$STAGE" = labels ]; then
  [ -f "$LABELS/SMOKE_OK" ] || { echo "run smoke first"; exit 1; }
  [ -f "$LABELS/tm2r3_apismoke_smoke.npz" ] || { echo "run apismoke first"; exit 1; }
  : "${TM2R3_DOM:?cup|finger}"; : "${TM2R3_DOSE:?e1|e4}"; : "${TM2R3_SEED:?51..58}"; : "${TM2R3_MAT:?early|late}"
  check_cell
  case "$TM2R3_MAT" in early|late) ;; *) echo "ERROR: TM2R3_MAT '$TM2R3_MAT' not in {early,late}"; exit 2;; esac
  RUN="tm2r3_${TM2R3_DOM}_${TM2R3_DOSE}_seed${TM2R3_SEED}"
  LOG="$TM2R3_ROOT/$RUN"
  OUT="$LABELS/${RUN}_${TM2R3_MAT}.npz"
  [ -f "$OUT" ] && { echo "skip $OUT (exists)"; exit 0; }
  [ -f "$LOG/TM2R3_TRAIN_DONE" ] || { echo "ERROR: $RUN not trained"; exit 1; }
  echo "[$(date)] label $RUN $TM2R3_MAT"
  python -m probing.tdmpc2_oracle_labels \
      --run_logdir "$LOG" --checkpoint "$LOG/ckpt_${TM2R3_MAT}.pt" \
      $DIALS --output "$OUT"
  echo "[$(date)] label $RUN $TM2R3_MAT rc=0 status=DONE"

elif [ "$STAGE" = wv1dup ]; then
  # WAVE-1 CRN null gates (PREREG_p2_wave1_firstupdate_20260821): the 2
  # registered pinned cells, act_cold1 := act_mode, every paired
  # quantity exactly zero (the frozen reader refuses otherwise). Runs
  # on trained LATE checkpoints; both smoke gates must have passed.
  [ -f "$LABELS/SMOKE_OK" ] || { echo "run smoke first"; exit 1; }
  mkdir -p "$WV1_LABELS"
  for cell in finger:e1:51 cup:e4:58; do
    IFS=: read -r dom dose k <<<"$cell"
    RUN="tm2r3_${dom}_${dose}_seed${k}"
    LOG="$TM2R3_ROOT/$RUN"
    OUT="$WV1_LABELS/wv1dup_${dom}_${dose}_seed${k}_late.npz"
    [ -f "$LOG/TM2R3_TRAIN_DONE" ] || { echo "ERROR: $RUN not trained"; exit 1; }
    [ -f "$OUT" ] && { echo "skip $OUT (exists)"; continue; }
    echo "[$(date)] wv1dup $RUN"
    python -m probing.tdmpc2_oracle_labels \
        --run_logdir "$LOG" --checkpoint "$LOG/ckpt_late.pt" \
        --states 400 --horizon 100 --label_every 25 --actions 8 \
        --seed 0 --ref_stride 5 \
        --wv1_repeats 4 --wv1_dup --env_seed 20260821 \
        --output "$OUT"
  done
  touch "$WV1_LABELS/WV1_DUP_OK"
  echo "[$(date)] wv1dup OK (finger_e1_seed51 + cup_e4_seed58)"

elif [ "$STAGE" = wv1 ]; then
  # WAVE-1 first-update cell pass (PREREG_p2_wave1_firstupdate_20260821):
  # LATE checkpoints, seeds 51-58 ONLY (the fresh cohort 59-66 belongs
  # to Wave 2's W1-protocol labeling, never to wv1), 400 states,
  # repeats 4, env_seed 20260821 — all reader-pinned.
  [ -f "$WV1_LABELS/WV1_DUP_OK" ] || { echo "run wv1dup first (registered ordering)"; exit 1; }
  : "${TM2R3_DOM:?cup|finger}"; : "${TM2R3_DOSE:?e1|e4}"; : "${TM2R3_SEED:?51..58}"
  check_cell
  case "$TM2R3_SEED" in 5[1-8]) ;; *) echo "ERROR: wv1 seeds are 51-58 only (PREREG_p2_wave1_firstupdate_20260821; 59-66 = Wave 2, not wv1)"; exit 2;; esac
  RUN="tm2r3_${TM2R3_DOM}_${TM2R3_DOSE}_seed${TM2R3_SEED}"
  LOG="$TM2R3_ROOT/$RUN"
  OUT="$WV1_LABELS/wv1tm2_${RUN#tm2r3_}_late.npz"
  [ -f "$OUT" ] && { echo "skip $OUT (exists)"; exit 0; }
  [ -f "$LOG/TM2R3_TRAIN_DONE" ] || { echo "ERROR: $RUN not trained"; exit 1; }
  echo "[$(date)] wv1 $RUN"
  python -m probing.tdmpc2_oracle_labels \
      --run_logdir "$LOG" --checkpoint "$LOG/ckpt_late.pt" \
      --states 400 --horizon 100 --label_every 25 --actions 8 \
      --seed 0 --ref_stride 5 \
      --wv1_repeats 4 --env_seed 20260821 \
      --output "$OUT"
  echo "[$(date)] wv1 $RUN rc=0 status=DONE"

elif [ "$STAGE" = w2smoke ]; then
  # WAVE-2 W1-protocol smoke (PREREG_p2_wave2_freshcohort_20260821 via
  # PREREG_w1_wave_20260809's registered cheap-insurance smoke, mirrored
  # onto the fresh cohort): 1 main + 1 dup_cand pass, 5 states, on
  # trained fresh LATE ckpts. *_smoke.npz never matches the derived
  # reader's regexes, so these files are inert at read time.
  mkdir -p "$W1_LABELS"
  for spec in "finger:e1:59:" "cup:e1:59:--w1_dup_cand"; do
    IFS=: read -r dom dose k dupflag <<<"$spec"
    RUN="tm2r3_${dom}_${dose}_seed${k}"
    LOG="$TM2R3_ROOT/$RUN"
    tag=""; [ -n "$dupflag" ] && tag="dup"
    OUT="$W1_LABELS/w1tm2${tag}_${dom}_${dose}_seed${k}_late_smoke.npz"
    [ -f "$LOG/ckpt_late.pt" ] || { echo "ERROR: $RUN has no ckpt_late.pt"; exit 1; }
    [ -f "$OUT" ] && { echo "skip $OUT (exists)"; continue; }
    echo "[$(date)] w2smoke $RUN ${tag:-main}"
    python -m probing.tdmpc2_oracle_labels \
        --run_logdir "$LOG" --checkpoint "$LOG/ckpt_late.pt" \
        --states 5 --horizon 100 --label_every 25 --actions 8 --seed 0 \
        --ref_stride 5 --oracle_all \
        --w1_repeats 8 --env_seed 20260809 $dupflag \
        --output "$OUT"
  done
  touch "$W1_LABELS/W2_SMOKE_OK"
  echo "[$(date)] w2smoke OK (finger_e1_seed59 main + cup_e1_seed59 dup; inspect stdout before w2dup)"

elif [ "$STAGE" = w2dup ]; then
  # WAVE-2 duplicate-null gate passes: the mirror of look-1's tm2 dup
  # cells (cup_e1 + finger_e1 at the cohort's first seed), full dials,
  # --w1_dup_cand. The derived reader's dup_gate REFUSES the whole read
  # unless every estimand is exactly 0 on these — run them before the
  # 32-pass spend so a CRN breakage costs 2 passes, not 34.
  [ -f "$W1_LABELS/W2_SMOKE_OK" ] || { echo "run w2smoke first"; exit 1; }
  # W2DUP_DOMS (20260823, ops): optional domain filter so the two dup
  # passes can run as two CONCURRENT invocations on one GPU (a single
  # pass measured 51% util / 302 MiB, so one card carries both). Dials,
  # seeds and outputs are untouched -- this changes execution shape
  # only. Default is the original sequential "cup finger".
  for dom in ${W2DUP_DOMS:-cup finger}; do
    RUN="tm2r3_${dom}_e1_seed59"
    LOG="$TM2R3_ROOT/$RUN"
    OUT="$W1_LABELS/w1tm2dup_${dom}_e1_seed59_late.npz"
    [ -f "$LOG/ckpt_late.pt" ] || { echo "ERROR: $RUN has no ckpt_late.pt"; exit 1; }
    [ -f "$OUT" ] && { echo "skip $OUT (exists)"; continue; }
    echo "[$(date)] w2dup $RUN"
    python -m probing.tdmpc2_oracle_labels \
        --run_logdir "$LOG" --checkpoint "$LOG/ckpt_late.pt" \
        $DIALS \
        --w1_repeats 8 --w1_dup_cand --env_seed 20260809 \
        --output "$OUT"
  done
  # STRICTER than the previous unconditional touch: the marker now
  # asserts BOTH registered dup passes exist, so a partial run (or one
  # half of a concurrent pair) can never open the label gate.
  if [ -f "$W1_LABELS/w1tm2dup_cup_e1_seed59_late.npz" ] && \
     [ -f "$W1_LABELS/w1tm2dup_finger_e1_seed59_late.npz" ]; then
    touch "$W1_LABELS/W2_DUP_OK"
    echo "[$(date)] w2dup OK (cup+finger e1 seed59)"
  else
    echo "[$(date)] w2dup partial (this invocation: ${W2DUP_DOMS:-cup finger}); W2_DUP_OK NOT set"
  fi

elif [ "$STAGE" = w2labels ]; then
  # WAVE-2 main W1-protocol pass (PREREG_p2_wave2_freshcohort_20260821):
  # LATE checkpoints, seeds 59-66 ONLY — the exact complement of wv1's
  # fence, so the two waves cannot cross in either direction. All dials
  # reader-pinned (env_seed 20260809 LOAD-BEARING, repeats 8,
  # states 200, horizon 100).
  [ -f "$W1_LABELS/W2_DUP_OK" ] || { echo "run w2dup first"; exit 1; }
  : "${TM2R3_DOM:?cup|finger}"; : "${TM2R3_DOSE:?e1|e4}"; : "${TM2R3_SEED:?59..66}"
  check_cell
  case "$TM2R3_SEED" in 59|6[0-6]) ;; *) echo "ERROR: w2labels seeds are 59-66 only (PREREG_p2_wave2_freshcohort_20260821; 51-58 = look-1, already read — relabeling them would collide with the look-1 cohort)"; exit 2;; esac
  RUN="tm2r3_${TM2R3_DOM}_${TM2R3_DOSE}_seed${TM2R3_SEED}"
  LOG="$TM2R3_ROOT/$RUN"
  OUT="$W1_LABELS/w1tm2_${RUN#tm2r3_}_late.npz"
  [ -f "$OUT" ] && { echo "skip $OUT (exists)"; exit 0; }
  [ -f "$LOG/ckpt_late.pt" ] || { echo "ERROR: $RUN has no ckpt_late.pt"; exit 1; }
  echo "[$(date)] w2labels $RUN"
  python -m probing.tdmpc2_oracle_labels \
      --run_logdir "$LOG" --checkpoint "$LOG/ckpt_late.pt" \
      $DIALS \
      --w1_repeats 8 --env_seed 20260809 \
      --output "$OUT"
  echo "[$(date)] w2labels $RUN rc=0 status=DONE"

else
  echo "unknown stage $STAGE"; exit 1
fi
