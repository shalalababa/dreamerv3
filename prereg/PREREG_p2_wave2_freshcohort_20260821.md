# PREREG — Paper-2 Wave 2: fresh-cohort TM2 replication (21 Aug 2026)

**Status: FROZEN at user commit. ONE read execution** (of the frozen
reader on this cohort; plus one pre-named pooled secondary invocation —
see below). Motivation: `Rescue_Synthesis_20260819.md` §4.2 — S2
(TM2 split-selected opportunity **+0.0617 [+0.0219, +0.1235]**, p=.016)
is the ONLY de-confounded positive in the program and rides a single
32-cluster read; a same-size replication fails ~1 in 3, so the number
must carry its replication status either way. Reviewer: ONE Opus pass,
user pre-approved, triggered at build.

## Design — a FRESH COHORT, not an extension

- **32 new TM2 training runs**: the PREREG_w1_wave_20260809 TM2 grid
  verbatim with only the training seeds replaced — cup/finger × e1/e4 ×
  **seeds 59–66**, LATE checkpoints only. Pipeline (review B5): the ops
  submit guard `scripts/tm2r3.sbatch` seed case is WIDENED
  5[1-8] → 5[1-8]|59|6[0-6] (an ops script, not a frozen instrument;
  sha256 39f96ecb8d58c2c04a7208492… — updated 21 Aug when the Wave-1
  wv1/wv1dup submission stages were added to the same script; the
  train/labels stages this registration uses are unchanged, and the
  new wv1 stage hard-refuses seeds 59–66, so the two waves cannot
  cross); everything else in the training pipeline is unchanged.
- **Labeling exactly per PREREG_w1_wave_20260809**: same flags, same
  `--w1_repeats`, same dup-gate passes, same `--env_seed 20260809`
  (LOAD-BEARING, not cosmetic — review M2: w1_read.py:67 pins
  EXPECT_ENV_SEED = 20260809 as a hard gate inherited byte-identically
  by the derived reader, so a changed seed would REFUSE; the
  fresh-cohort primary is a within-pass estimand unaffected by the
  shared marks — the pooled secondary is the leg that pays, see below),
  names `w1tm2_{dom}_{dose}_seed{59..66}_late.npz` (+ the mirrored
  dup-gate passes). ~36 label passes; total 101–184 GPU-h including
  training.
- The archived W1 cells are **not re-read, not re-labeled, and not
  pooled into any primary**.

## Estimands and decision rules

- **PRIMARIES, VERBATIM from the frozen rule — via a DERIVED reader**
  (review B2: the executed frozen `analysis/w1_read.py` hardcodes seeds
  5[1-8] in FILE_RES/DUP_RES and cannot read the fresh cohort; it is NOT
  edited). `analysis/w1_read_fresh.py` is a literal copy whose ONLY
  changes are the two tm2 seed-regex groups 5[1-8] → (?:59|6[0-6])
  (2 changed lines; selfcheck PASS re-run; sha256
  4c2a98c45eab0e8f023f7e31…; original 24994d110aad4da59453011b… — the
  diff is auditable byte-for-byte). Every dial pin, estimand function,
  and dup gate is byte-identical. This registration licenses ONE
  execution of the derived reader on the fresh cohort's 32 cells.
  P-W1a and P-W1b fire under exactly the thresholds frozen in
  PREREG_w1_wave_20260809; P-W1c reports as there specified.
- Power at n=32 (from the look-1 dispersions): P-W1a **0.67**, P-W1b
  **0.76** — disclosed; this is a replication attempt, not a powered
  discovery wave, and the read reports realized MDEs.
- **Pooled secondary (no decision weight, pre-named)**: ONE invocation
  of `analysis/w1_read_pooled.py` (review B3 — a second literal-copy
  derived reader: union seed regex, EXPECT_CELLS 64, output renamed
  w1pooled_{family}.json so nothing overwrites; 4 changed lines; sha256
  c80ad8f1b27b41966c2e7bdf…) over the 64 combined cells. **The two
  cohorts share env_seed 20260809 and therefore share CRN marks and the
  env/OU stream (review M2): the pooled interval is reported as a
  DESCRIPTIVE pooled point with its provenance split, not as an
  independent-64 interval, and the 0.93/0.99 power figures (which assume
  independence) apply to it only nominally.** It carries no fire/no-fire
  authority.
- **Registered consequences, symmetric**:
  - P-W1a fires fresh ⇒ S2 is replication-supported; the +0.062-scale
    claim may appear in the abstract with both cohorts cited.
  - P-W1a does not fire fresh ⇒ **+0.062 is thereafter reported with its
    replication failure attached, everywhere** — no pooling rescue, no
    third cohort.
  - P-W1b's anti-harvest direction: fresh fire strengthens S3's
    behavioral leg (subject to Wave 1's attribution); fresh non-fire is
    reported beside the look-1 value.

## Gates (refusal = read not consumed)

The frozen reader's own gates apply unchanged (they are part of the
frozen instrument); additionally the bundle must show: 32 fresh
checkpoints with training-audit echoes (trainer_version, realized LATE
steps) matching the registered dials; seeds ∈ {59..66} only; no file
name colliding with the look-1 cohort's names (the seed ranges are
disjoint by construction — a collision is a wiring error and refuses).

## Ops

32 TM2 trainings (seeds 59–66; re-run until drained) → labeling passes
per PREREG_w1_wave_20260809 verbatim → bundle + sha manifest →
**[ME] ONE read (fresh cohort) + the pre-named pooled secondary
invocation**, reported together.
