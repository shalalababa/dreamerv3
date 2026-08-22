# Det se_mask baseline (B4 freeze gate §0) — READ: the registered
gate FAILS, and the failure is a FINDING, 22 Aug 2026

**Status: instrument-calibration pass on the 8 existing Stage-1 det
checkpoints (PREREG_trackB_alea_20260822 §0; zero GPU; ops-executed
on RCC, jsons pulled to
`local_results/uncfield/det_semask_baseline_20260822/`). This is
the registered pre-freeze de-risk, NOT a registered read. Its
consequence is binding: B4 does NOT submit until the manipulation
check is re-specified by dated amendment.**

## The result (all 8 cells, distractor channel)

Two statistics per cell, both from `uncfield.se_mask`
(batch-permutation of the distractor across S=512 anchors):

| seed | base intrinsic | Δ intrinsic (total) | Δ/base | Δ own-dim (decoded, normalized) | own BCa | p_reduce(own) |
|---|---|---|---|---|---|---|
| 10 | 1.400e-4 | +1.27e-7 | +0.09% | −1.07e-4 | [−2.55e-4, +4.60e-5] | .080 |
| 11 | 1.664e-4 | +6.06e-7 | +0.36% | +2.04e-4 | [+4.52e-5, +3.82e-4] | .986 |
| 12 | 1.284e-4 | −3.55e-7 | −0.28% | +1.22e-4 | [−2.00e-5, +2.69e-4] | .940 |
| 13 | 1.717e-4 | +4.93e-8 | +0.03% | −2.58e-5 | [−1.96e-4, +1.32e-4] | .383 |
| 14 | 1.347e-4 | +4.96e-8 | +0.04% | −9.65e-7 | [−1.54e-4, +1.45e-4] | .494 |
| 15 | 1.777e-4 | +4.29e-7 | +0.24% | +2.51e-5 | [−1.30e-4, +1.81e-4] | .622 |
| 16 | 1.369e-4 | +3.93e-7 | +0.29% | −1.73e-5 | [−1.51e-4, +1.20e-4] | .387 |
| 17 | 1.323e-4 | −1.75e-7 | −0.13% | +1.16e-4 | [−2.82e-5, +2.46e-4] | .949 |

- **Total intrinsic delta: |Δ| ≤ 0.36% of base in 8/8, signs mixed
  (2/8 negative).** The det disagreement reward's LEVEL is
  insensitive to whether the realized distractor value is
  state-consistent.
- **Own-dim decoded delta: 4/8 negative, 7/8 CIs straddle zero, the
  one CI excluding zero is POSITIVE.** Even the decoder-projected
  distractor variance does not respond to permutation.

## Reading: the misprice is a LEVEL property, not a COUPLING property

This is mechanism-coherent, and in hindsight predictable from the
program's own results: the OU distractor's NEXT value is
unpredictable REGARDLESS of its current value, so ensemble
disagreement about it does not depend on the current value being
the true one — permuting the current value changes nothing. The
θ₁ = 5.26× misprice (registered, Stage-1) is real, but it lives in
the ALLOCATION of ensemble dispersion across dims (a level/share
quantity), not in any per-state coupling an intervention can break.
This coheres with duplicate-immunity and the stochasticity-farming
account: the reward farms the aleatoric floor, which is
state-independent.

## Consequences (each actioned)

1. **B4 HELD (ops holding the staged wave — correct).** The
   registered manipulation check (masked-delta < 0 in 4/4) would
   fail with near-certainty REGARDLESS of whether the aleatoric
   repair works — the wave would land RAW-MISPRICE-ABSENT by
   instrument-form mismatch. §0 fired exactly as written.
   Amendment drafted:
   `prereg/PREREG_trackB_alea_amend1_20260822.md` (share-form
   re-specification; awaits user GO vs the park option).
2. **B2 (APT-IMMUNE) scope sentence owed [writing chat]:** the APT
   immunity was established with the SAME interventional form that
   the det arm now shows is blind to level-type pricing. B2's cell
   stands as COUPLING-immunity (its velocity control fired −0.028
   in 8/8, so the instrument is alive on coupling); but the
   det-vs-APT proxy-specificity contrast mixes forms (det = θ₁
   level form; APT = mask coupling form) and must be worded as
   such — "APT shows no interventional footprint; the level-form
   comparison is carried by θ₁, which has no APT analog."
3. **B5 prereg scope sentence added (pre-freeze, legal):** a
   distractor-permutation null on TM2's Q-std licenses
   coupling-immunity only; this det baseline is the precedent.
4. **Zero-compute payoff:** "the misprice is a level property, not
   a coupling property" is itself a taxonomy-paper paragraph
   (dissociates the intervention axis from the allocation axis),
   obtained for 8 CPU passes on existing data.

Provenance: 8/8 dup0 bitwise-no-op anchors PASS in the pulled
jsons; instrument = the frozen `uncfield/se_mask.py` (no edits).

---

## CORRECTION (R-A1-M12, 22 Aug, same day — the amendment review):
the null is GUARANTEED BY CONSTRUCTION, not an empirical mechanism
finding

With gates/mod off, the distractor is a PURE EXOGENOUS AR(1)
(state- and action-independent, initialized at stationarity). Batch
permutation of its windows across anchors therefore resamples from
the IDENTICAL joint law — the population delta is zero BY
EXCHANGEABILITY for ANY statistic, regardless of how the reward
prices the channel. The section "Reading: the misprice is a LEVEL
property, not a COUPLING property" over-interprets: what this
baseline demonstrates is a FORM/ESTIMAND MISMATCH (the mask form is
inert-by-construction for exogenous channels), and "there was no
coupling to break" is the construction, not a discovery about the
ensemble. The mixed signs and ≤0.36% magnitudes are the arithmetic
of a zero-mean statistic.

Corrected consequences:
1. B4 amendment: unchanged in substance (re-specifying the form was
   right), wording corrected; the retained masked distractor rows
   are a CONSTRUCTION-INERT exchangeability control (a materially
   nonzero value = instrument defect), not a "level-vs-coupling
   dissociation replication".
2. **B2 (APT) — STRONGER than the original flag:** the APT read's
   distractor-mask null was ALSO construction-inert (same exogenous
   channel, same permutation form) — it could not have come out any
   other way. APT-IMMUNE therefore rests on (i) the velocity
   control firing (the instrument detects real couplings) and (ii)
   the absence of any INFORMATIVE distractor intervention in that
   read. [writing chat] the proxy-specificity contrast must be
   restated on the level-form θ₁ (which has no APT analog) and the
   dup-substitution rows (law-changing, informative).
3. **B5 (TM2) — the registered fire channel (distractor batch
   permutation on Q-std) was structurally unable to fire.**
   Re-specified pre-freeze (PREREG_trackB_tm2 rev 2): fire channel
   = distractor MEAN-SUBSTITUTION (law-changing, level-sensitive);
   the permutation is retained as a built-in exchangeability-null
   calibration row; velocity mean-substitution added as the
   form-matched specificity comparator.
4. The "level property" language survives only where it was always
   licensed: θ₁ = 5.26× is an allocation fact (se_probe, share
   form). Statements about what interventions the pricing responds
   to require law-CHANGING interventions.
