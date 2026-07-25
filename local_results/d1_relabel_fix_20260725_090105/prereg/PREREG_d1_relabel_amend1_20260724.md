# PREREG AMENDMENT 1 to the D1 relabel campaign (frozen 2026-07-24)

Amends `PREREG_d1_relabel_20260724.md` (frozen at commit 83f7fba0).
Trigger: the registered optional target of plan v4 §5 item 4 — an
existence check on the cluster runroot CONFIRMED (owner, 24 Jul) that
the six Gate-D1 pilot runs `d1pilot_{cup,finger}_e1_seed{1,2,3}` are
alive with final checkpoints. This amendment is committed BEFORE any
corrected d1pilot label exists and BEFORE the single read execution of
the campaign (guard below).

## What is added

18 additional corrected-instrument label passes (labeler
`d1fix_20260724`, identical pinned dials + `--oracle_all`) on the six
Gate-D1 pilots, arms {base, xpol sibling-rotation 1→2→3→1 within
domain, phys mass ×1.3}, into the SAME `d1_labels_fix` root.
Companion driver: `scripts/d1pilot_relabel_local.sh`
(pull → smoke → labels; the frozen base driver is untouched). The
pulled checkpoints label through the same real-MuJoCo path already
smoked by the base campaign; the amendment's own 5-state base + xpol
smokes only verify pull integrity (any INSTRUMENT fix would still
require a further dated amendment before labels).

Maturity disclosure: the d1pilot runs are the SAME 1e5-step maturity
as the shift pilots (`run.steps 100000` in their configs) — this
amendment adds CLUSTERS, not a maturity axis. Re-adjudication of the
phase×dose constants still requires R3 (early+late, e1+e4); nothing
here changes that.

## Registered analysis structure (coded in `analysis/d1_shift_read.py`
## before any corrected label is read; selfcheck extended)

1. **PRIMARY — UNCHANGED.** The frozen 2-look rule on the d1s cohort
   (6 clusters) still solely determines `fired_arms`. This amendment
   does not touch it.
2. **REPLICATION family (new):** the same rule (paired arm−base per
   run, cluster bootstrap B=10K rng 0, CI>0) on the d1pilot cohort
   alone — 2 additional looks, disclosed.
3. **POOLED-12 estimate (new, headline magnitude):** the same
   estimator over all 12 clusters, reported with CI for both arms.
   It is the registered EFFECT-SIZE report; it is not a third
   fire/no-fire look.
4. Registered secondaries of the base prereg extend per cohort:
   defect-impact A/B now includes the ORIGINAL Stage-1B cells
   (18-Jul defective labels on the SAME checkpoints — matched-
   checkpoint, still distributional since trajectories re-sample);
   corrected e1 base constants on 12 runs; opportunity/achieved/gap
   decomposition from g_all on 12 runs (better R3 power inputs);
   C1-style calibration per cohort per arm (descriptive; the Gate-D1
   resource decision is frozen and is NOT re-adjudicated here).

## Cross-cohort consequence map (frozen)

- **Both cohorts fire an arm** (primary + replication, same arm) ⇒
  consequence leg REVIVED, strong form (replicated at 12 clusters).
- **Exactly one cohort fires an arm** ⇒ revived-weak iff the other
  cohort's same-arm point estimate is positive (directional
  consistency); otherwise UNRESOLVED-HETEROGENEOUS (no revival; the
  pooled CI is the honest summary).
- **Neither cohort fires either arm** ⇒ corrected negative at 12
  clusters — materially stronger than the base prereg's 6-cluster
  negative (the review's small-cluster fragility objection is
  discharged to the extent 12 clusters allow; t-interval sensitivity
  reported as before).
- Unchanged regardless: 24+24 frozen; Route A closed; imag retired;
  Gate-D1/R2/ladder remain instrument-invalidated.

## Ordering guard

The campaign's read (`analysis/d1_shift_read.py`) runs ONCE, after
both cohorts' passes land. Fallback (pre-specified): if the pull or
any d1pilot pass fails irrecoverably, the read runs on the d1s cohort
alone and the base prereg applies unamended (the read accepts an
absent-but-never-partial d1pilot cohort). Label passes of the base
campaign may already be running; no corrected label of either cohort
has been READ at this freeze, and no d1pilot corrected label EXISTS.

## Cost

+18 passes ≈ +1–2 days sequential on the 5090 (after the base 18);
pull is ~6 small checkpoint dirs. Within the standing compute policy.
