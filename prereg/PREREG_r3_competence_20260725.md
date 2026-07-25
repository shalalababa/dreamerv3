# PREREG: R3 — consumer-competence factorial (frozen 2026-07-25)

Paper-2 plan v4 §4 (`EVPI_Plan_Revision_20260724.md`) executed: the
corrected relabel read landed 25 Jul
(`artifacts/d1_relabel_20260725/` — corrected negative at 12 clusters;
opportunity large, achieved ≈ 0), so R3's design constants are in hand
and this registration is now licensed. This file +
`analysis/r3_read.py` (selfcheck PASS) + `scripts/r3_local.sh` are
committed BEFORE any R3 pilot or label exists.

## Question

Where the shift-consequence and calibration nulls hold, is that because
the oracle purchase has nothing to offer (an information-content fact),
or because opportunity exists and the consumer cannot harvest it (a
competence fact)? The corrected relabel's registered descriptives point
to the second (opportunity +0.6..+6.3 per cell, achieved ≈ 0
everywhere) — R3 turns that descriptive into a registered factorial
with maturity and dose axes on FRESH runs and PRE-SIZED power.

## Estimands (per labeled state, from the corrected labeler's g_all)

- opportunity = max_m G[m] − G[m_now]
- achieved = G[m_real] − G[m_now]
- implementation gap = opportunity − achieved

Consumer axis note: the trained consumer (m_real) vs the oracle
consumer (argmax g_all) contrast IS the gap — no new machinery. The
cross-checkpoint consumer (early WM × late critic and vice versa) from
plan v4 needs a labeler extension and its own smoke: it is NOT part of
this registration and would be a dated amendment before any such label.

## Design

- Grid: {cup, finger} × {e1, e4} × seeds 31–38 (disjoint from all
  prior D1 seeds), each run labeled at TWO maturities — early (2.5e4
  snapshot, `ckpt_early`) and late (1e5 final). 32 training runs, 64
  label passes, base arm only (no shift arms).
- Dose levels: e1 = no distractor (dim 0); e4 = `d0_dose3` (dim 32,
  scale 3.0) — the Gate-D1 dose levels, from the run's own config at
  labeling time (the repaired OU/calibration snapshot path,
  DEVIATIONS 24-Jul audit items 4–5).
- **Reacher third domain: conditional GO** (portfolio default GO;
  task `dmc_reacher_hard`, regime threshold verified against
  `artifacts/gate0_20260702/gate0_reacher/`): run as a COMPLETE cohort
  ({e1,e4} × 8 seeds × 2 maturities) or not at all — the frozen read
  accepts absent-but-never-partial (mirrors the relabel Amendment-1
  cohort rule).
- Training: two-phase in one logdir (train to 2.5e4 → copy `ckpt` to
  `ckpt_early` → resume to 1e5). The resume boundary at 2.5e4 is
  common to every run (arm-invariant by construction; disclosed).
  Checkpointing is clock-based with NO exit-time save (train loop
  fact, verified 25 Jul), so the driver pins `save_every` to 60 s and
  records each snapshot's REALIZED step (`ckpt_early/STEP`,
  `ckpt/STEP`); realized steps are disclosed in the artifact, and
  "early/late" are defined as these realized snapshots (expected
  within ~1–2K steps of the nominal 2.5e4/1e5). Phase B resumes FROM
  the early snapshot, so the late state is the early state's own
  continuation.
- Labeling dials: IDENTICAL to the relabel campaign (states 200,
  horizon 100, label_every 25, actions 8, rollouts 16, labeler seed 0,
  ref_stride 5, `--oracle_all`), labeler `d1fix_20260724`.
- Ordering: freeze-commit → `pilots` → `smoke` (REGISTERED GATE: e1
  AND e4 × early AND late smokes on real dosed envs — the repaired
  distractor snapshot path has never labeled a real dosed agent; any
  fix ⇒ dated amendment BEFORE labels) → `labels` → ONE read.

## Power (pre-sized from the corrected relabel, as plan v4 requires)

Measured 25 Jul on corrected base labels (12 runs): per-run gap sd
≈ 1.0 (cup) / 1.3 (finger), pooled mean gap ≈ 1.07, with 2/12 finger
floor runs included. At n = 32 clusters the detectable pooled gap is
≈ 1.96·1.2/√32 ≈ 0.42 — half the observed level; with the reacher
cohort (n = 48) ≈ 0.34. The 6-cluster fragility documented in the
11-Jul review cannot recur at this size.

## Registered decision rules (frozen in `analysis/r3_read.py`)

Run-clustered percentile bootstrap (cluster = training run; both
maturities of a run share the cluster), B = 10K, `default_rng(0)`:

- **P-R3a (opportunity exists)**: pooled mean opportunity CI entirely
  > 0.2 (threshold ≈ the smallest non-floor per-run base-cell
  opportunity observed 25 Jul — an existence bar, not a triviality).
- **P-R3b (competence failure)**: pooled mean gap CI entirely > 0.
- **P-R3c (maturity buys competence)**: per-run paired late−early
  achieved value, pooled CI entirely > 0.

Floor policy (frozen): a (run, maturity) cell with all-zero G is a
floor cell; floor cells are INCLUDED in all primaries (conservative —
they shrink opportunity and gap toward 0) and reported per domain; a
domain with > 50% floor cells is flagged floor-limited (disclosure;
primaries stay pooled). Registered secondaries (descriptive, never
decisional): dose effect on opportunity (does distraction change
information content?), maturity effect on opportunity, per-domain
splits, floor fractions.

## Consequence map (frozen)

- **P-R3a + P-R3b fire** ⇒ the consumer-competence failure is a
  registered RESULT: opportunity exists and is not harvested — Paper
  2's headline becomes opportunity/competence decomposition (with
  P-R3c scoping whether maturity closes the gap). The 25-Jul
  shift-consequence negative is then interpretable as a competence
  ceiling, not an information ceiling.
- **P-R3a fails** ⇒ no detectable opportunity on fresh runs at
  pre-sized power: the nulls are an information-content fact; plan
  v4's Option 5 (narrow diagnostic paper) becomes the honest scope.
- **P-R3a fires, P-R3b fails** ⇒ the consumer harvests what exists —
  a calibration-success branch; any calibration claim would still be a
  NEW registration (Route A stays closed as a resource decision).
- Regardless: 24+24 stay frozen; imag stays retired.

## Costs and venue

32 (–48) pilots × 1e5 proprio steps + 64 (–96) oracle_all label passes.
Pilots ≈ 1–1.5 h each, passes ≈ 1–2 h each on a single GPU —
cluster-lane or Vast-queue scale, driver also runs on the 5090
(sequential ≈ 1–2 weeks; cluster parallel ≈ 1–2 days wall-clock).

## Disclosure

Known at freeze: everything through the 25-Jul read wave, including the
relabel's opportunity/achieved constants used ONLY for power sizing and
the P-R3a threshold. Unknown: every R3 quantity — no seed-31–38 run,
no early-maturity corrected label, and no e4 corrected label exists
anywhere. The prior Gate-D1 e4 cells were defective-labeler cells on
checkpoints that no longer exist; nothing here re-reads them.
