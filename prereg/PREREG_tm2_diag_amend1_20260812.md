# AMENDMENT 1 to PREREG_tm2_diag_20260811 — inner-split constant-label redraw — 2026-08-12

## What happened (disclosure)

Cluster-side, the probe pass `tm2diag_aware_s1_seed1` crashed at the
registered fail-closed refusal
(`tm2_legibility_probe.ridge_r2_grouped`, review-#26 gate: "all inner
alpha scores degenerate - refusing") while every aware-s0 pass
completed. Mechanism, confirmed by the operator's diagnosis: the probe
is CROSS-SIDE (an s1-trained fit probes the side0 episodes); the
seeded 16-episode side0 sample is positive-sparse (535 positive rows)
with episode-clustered rewards, and outer fold 2's seeded 3-episode
inner-test draw contains ZERO positives — constant labels ⇒ every
alpha's inner R² is NaN (ss_tot = 0) ⇒ refusal. Because the episode
sample and folds are seeded with the fixed instrument RNG, **every
s1-fit probe cell fails identically** (all consume the same side0
sample); the registered wave would arrive half-empty at the read and
refuse at the inventory gate.

**Value-awareness at this amendment (disclosed):** no probe OUTPUT has
been read (crashed cells wrote no json — the write is after run_probe;
completed s0 jsons exist cluster-side, unread; the ONE registered read
has not run). What was observed: the crash trace and the operator's
fold-level diagnosis (episode ids, per-fold positive-row counts,
positive totals 535/5770 by side) — data-property facts consistent
with the already-registered side-0 sparsity note ("~5.4% positive"),
not estimand values. Zero estimands observed. This mirrors the ZC
Amendment-1 precedent (pre-outcome instrument repair after a
fail-closed crash).

## The repair (frozen in `probing/tm2_legibility_probe.py`)

In `ridge_r2_grouped`, the inner alpha-selection split — TRAINING
episodes only; outer test folds are never touched — is redrawn (same
seeded RNG stream) when its inner-test labels are CONSTANT
(`np.unique(yv).size < 2`, exactly the ⟺ condition for the all-NaN
refusal, since `_ridge_fit_eval` returns NaN iff ss_tot = 0), up to
`MAX_INNER_REDRAWS = 20`; if every redraw is constant-label the run
still REFUSES (fail-closed preserved — the all-zeros-labels selfcheck
leg still refuses). The NaN-filter backstop assert is unchanged. The
probe output gains an audit field `inner_redraws` (per-fold counts).

**RNG-stream identity (the property that keeps completed outputs
valid):** a valid first draw consumes exactly one `rng.permutation`,
byte-identical to the pre-amendment path. All 16 completed s0-fit
probe jsons therefore remain valid under the amended instrument —
their draws were valid (they completed), so the amended code would
reproduce them bit-for-bit modulo GPU forward nondeterminism, which is
unchanged by this amendment. Selfcheck pins: the pre-amendment
reference outputs captured 12 Aug BEFORE the patch
(aligned-continuous auroc 0.9999374902328488 / r2 0.9997666377576077;
aligned-binary auroc 0.997932125854411) must reproduce to <1e-12 with
zero redraws, AND a pinned sparse episode-clustered fixture (label rng
102 — the realized side0 failure shape) must trigger redraws
([3, 5, 1, 5]) and COMPLETE where the pre-amendment code refused.
`analysis/tm2_diag_read.py` is untouched (it ignores the new audit
field); its selfcheck re-PASSES.

## Alternatives considered (rejected)

- Larger `--episodes` for s1-fit passes: changes the estimand's sample
  per arm asymmetrically and only shrinks, not removes, the degenerate
  draw probability.
- Accepting DIAGNOSTICS-BLOCKED for the s1 half: the both-sides legs
  of the registered read REQUIRE both sides; the wave would be void
  where a split-level repair with an identity guarantee exists.
- Stratified inner split (force a positive-bearing episode into
  inner-test): changes the split distribution for ALL runs, breaking
  the identity property for the completed s0 outputs.

## Registered continuation

Re-run ONLY the crashed cells (all s1-fit probe passes; the registered
per-fit command, unchanged) under the amended instrument; completed s0
jsons are NOT re-run (identity property above). The read then proceeds
per the base registration, ONE execution, with `inner_redraws`
reported in the RECORD as an instrument note. No decision rule,
threshold, or estimand changes.

## Discipline

Committed BEFORE any s1-fit probe output exists and before the ONE
read. Probe + reader selfchecks re-PASS (12 Aug). No reviewer
(fail-closed-crash repair with zero estimands observed; ZC Amendment-1
precedent).
