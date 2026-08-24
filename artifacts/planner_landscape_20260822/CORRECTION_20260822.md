# CORRECTION + SUPERSESSION NOTE — 22 Aug 2026 (Fable review, memo
on the rescue decision; original RESULTS.md left unedited per house
rule)

1. **Statistic misdescribed (Fable F4).** The RESULTS.md sentence
   "34.8% of the pooled top-decile states are POLICY states" is
   WRONG as written: `comp[a]` in `planner_landscape.py` is the
   fraction OF ARM a's OWN STATES that fall in the pooled top
   decile, not the top decile's composition. The true composition
   is far MORE policy-dominated (≈97–99% policy in 6/8 runs; seed11
   ≈59% cem_distractor). Direction of the conclusion unaffected
   (understated); the number must not be quoted as written.
2. **The FLAT resolution is RETRACTED (Fable F1/F2/F5).** The
   declared primary rule returned AMBIGUOUS; the secondary-metric
   resolution to "flat" leaned on a level-confounded statistic. The
   re-analysis (verified 22 Aug,
   `artifacts/planner_share_decomp_20260822`): the policy is the
   per-run argmax on EVERY key (velocity 8/8 — a global
   disagreement-LEVEL factor), while in COMPOSITION terms the
   dedicated planner steered the distractor's share above both
   passive anchors in **8/8 runs** (mean share 0.0691 vs policy
   0.0414). A level-reaching, share-holding attacker projects to
   ≈0.0275 ≈ 1.64× the policy's realized 0.0168 — inside the
   observed headroom band. The wave also contains a FAILED RELATIVE
   POSITIVE CONTROL (the greedy CEM lost to the amortized policy on
   all three of its own objectives), so "no reachable peak" was
   never the licensed summary.
3. **Decision superseded.** The trained-probe decline and the PCM
   wave are both WITHDRAWN (PCM is level-confounded — it would fire
   identically on velocity). Replacement slate (registered builds in
   flight): C = localized-channel planner wave on the A1
   hetero/flat checkpoints; A = policy-seeded refinement arm with a
   built-in positive control; B = imagination-trained adversarial
   actor (gates the security paragraph's final wording).
4. The mechanism sentence "the amortized policy already sits at the
   reachable maximum of its own mispriced field" is **withdrawn
   unscoped** (quantifier mismatch; contradicted in-house by A1's
   localized-variant avoidance). Licensed wording = the Tier-0 /
   Tier-0-descriptive forms in the planner read's correction note.
