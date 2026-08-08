# Correction addendum (2026-08-08) — pr_head_input is a trace-ratio artifact

Labeled correction to RECORD.md's mechanism reading (review finding D8,
verified from this artifact's own csv; no numbers change, one interpretation
does).

**What stands.** The seed-paired WITHIN-BLOCK contrasts: [rde − X2]
`pr_deter` s0 −8.21 [−9.40, −6.89], s1 −8.05 [−8.83, −7.21] — rde trunks are
materially narrower in deter-space on BOTH sides at better reward legibility.
The frontier (fpxpx broad-illegible → X2 broad-swamped → rde narrow-legible)
survives on within-block deltas. The witness-gate provenance is untouched.

**What is corrected.** The co-localisation reading ("head-input breadth
collapses on exactly the behavioral-deficit side s1 while s0 keeps breadth")
leaned on `pr_head_input`, which is the PR of a **concatenation** and is
dominated by whichever block has the larger trace:

- corr(deter_std, pr_head_input) = **−0.811** over all 48 runs;
- rde s1: tr_deter 102–129 vs tr_stoch 22–23 ⇒ pr_head ≈ pr_deter (small);
- rde s0: tr_deter 0.5–1.6 (6/8 seeds) vs tr_stoch 9–14 ⇒ pr_head ≈ pr_stoch
  (large).

So the s1 "collapse" 44→9 is a **block-scale composition fact** (`h`
monopolising variance), not directions being lost; and by `pr_deter` rde s1
(5.69–7.97) is *broader* than the six non-basin-crossing s0 seeds (1.53–2.31)
— the cross-side ordering inverts at block level. The RECORD's own
self-correction (corr(deter_std, pr_deter) = −0.08; +0.35 pooled / +0.07
within-side for the AUC link) already pointed here.

**Corrected mechanism sentence.** rde narrows the deter-space representation
on both sides; the behavioral deficit does not co-localise with a breadth
collapse in any block-honest reading. "Transfer needs breadth, not just
legibility" remains supported only in the weaker within-block form (narrower
deter-space at better legibility, no lift); the side-resolved co-localisation
claim is withdrawn. Figure `fig_feature_pr.png`'s annotation ("remove recon:
−35 PR …") refers to pr_head and inherits this caveat.

**Also noted:** rde s0 `deter_std` ≈ 0.030 with pr_deter 1.5–2.3 is close to
rank-1; the latent-alive witness margin is 1.8× its threshold — a partial-
collapse account of s0 should be checked before any replay-side test
(review §7 block 1).
