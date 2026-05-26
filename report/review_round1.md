# Editorial Review Package — Round 1

**Manuscript:** Reward-Free World-Model Pretraining for Transferable Control Representations
**Venue frame:** workshop-tier ML / unsupervised RL (course mini-paper, ICLR format)
**Panel:** EIC + Methodology (R1) + Domain (R2) + Perspective (R3) + Devil's Advocate (DA)
**Scoring:** Originality 20% · Methodological Rigor 25% · Evidence Sufficiency 25% · Argument Coherence 15% · Writing Quality 15%

> Reviewers reviewed independently and did not modify the manuscript.

---

## Phase 0 — Field analysis & panel configuration

- Primary field: model-based / unsupervised reinforcement learning. Secondary: representation analysis (probing).
- Paradigm: empirical, mechanism-focused; small-n controlled comparison on one DMC domain.
- Maturity: honest, well-caveated mini-study. The risk is not over-claiming (the draft is already cautious) but *presentation rigor* and *isolating the causal story*.
- Panel: R1 = RL evaluation-statistics specialist; R2 = world-models/unsupervised-RL domain expert; R3 = transfer/representation-learning generalist; DA = causal-inference skeptic.

---

## R1 — Methodology & Statistics

**Summary.** Clean experimental design with a genuinely useful frozen-vs-full calibration. The statistical presentation, however, overstates what two seeds can support.

**Strengths.** Shared body isolates the exploration objective; primary/secondary metric split is appropriate; the §F calibration is the right control and is interpreted correctly (agreement on primary, flip on secondary).

**Weaknesses (with fixes).**
1. *(MAJOR)* No engagement with modern RL evaluation statistics. With 2 seeds there is no power for significance, yet the text reports "$2$–$3\times$" and two-decimal correlations, implying precision the design cannot deliver. **Fix:** cite and follow the rliable guidance (Agarwal et al., 2021) at least in spirit — report per-seed points as the evidence, state explicitly that no significance is claimed, and present the $n{=}6$ correlations as a qualitative trend (scatter / sign), not as authoritative coefficients.
2. *(MINOR)* The "early-adaptation AUC" metric is a normalized area (mean return over the first 125K). It conflates *initial level* and *slope*. **Fix:** define it precisely and note the conflation; it does not change conclusions but a reader needs to know what the number is.
3. *(MINOR)* The final-return comparison mixes C1-at-240K-of-500K against adapt-at-end-of-250K. The budgets are matched in env steps, but state this explicitly so the comparison is not misread.

**Scores:** Rigor 60 · Evidence 62.

---

## R2 — Domain & World Models

**Summary.** Correctly positions DreamerV3, Plan2Explore, APT, and URLB. Two method adaptations are under-flagged.

**Strengths.** The RSSM/frozen-readout cut is well motivated; the related-work framing is not a laundry list.

**Weaknesses (with fixes).**
1. *(MAJOR)* The implemented C3 is not canonical Plan2Explore. Original P2E predicts the next *encoder embedding*; here the ensemble predicts the next *deterministic posterior feature* of the RSSM. This is defensible (and the draft explains the sampled-latent degeneracy), but the manuscript should state plainly that **results are for an adapted P2E variant**, not the original, and likewise that APT's entropy is computed in the *world-model latent* rather than a separate contrastive/random encoder. **Fix:** one explicit sentence each in §3, and hedge cross-paper comparisons accordingly.
2. *(MINOR)* Budget caveat is buried. 500K vs URLB's ~2M is a real external-validity threat to the "random pretraining hurts" claim (random may simply need longer). **Fix:** connect this caveat directly to the C2 result, not only to absolute magnitudes.

**Scores:** Originality 66 · Domain contribution 64.

---

## R3 — Perspective & Impact

**Summary.** The mechanism framing is the paper's most interesting contribution, but the "so what" is underdeveloped.

**Weaknesses (with fixes).**
1. *(MAJOR)* Missing amortization economics. The headline is "$2$–$3\times$ faster early, but from-scratch catches up by 240K." A reader will ask: when is a 500K reward-free pretraining phase worth it if C1 reaches the same place? **Fix:** add the break-even argument — pretraining amortizes only across *multiple* downstream tasks; give the rough arithmetic (reward-free steps spent once vs. adaptation steps saved per task × number of tasks).
2. *(MINOR)* External validity to pixels/harder bodies is asserted-away in one clause. The coverage→transfer story may be specific to proprio where current-state probing saturates. **Fix:** say so as a scoped claim.

**Scores:** Originality 64 · Argument coherence 74.

---

## DA — Devil's Advocate

**Strongest counter-argument (the paper's central inference is near-tautological).**
The paper's RQ2 conclusion is that *data coverage* predicts transfer. But there are only three pretrained conditions, and "coverage" is operationally almost the same variable as "is the pretraining policy an active explorer." Random barely moves (flat, low coverage); Plan2Explore and APT move (rising, high coverage). Any metric that separates "moves" from "doesn't move" will correlate with transfer here. So the correlation cannot isolate coverage as the *causal* mediator; it largely restates the exploratory-vs-passive split the conditions were built to create. The draft admits this in one sentence but then proceeds to headline coverage. Either isolate coverage from "policy activeness," or demote the claim to "the data distribution induced by exploration predicts transfer," which is weaker but defensible.

**Issue list.**
- *(MAJOR)* Coverage/activeness collinearity (above). Elevate the "task-relevant regime" alternative from a clause to a co-equal hypothesis: the benefit may be reaching upright/moving states at all, not breadth per se.
- *(MAJOR)* "Transferable representations" framing vs the §F finding that from-scratch reaches the same asymptote and the frozen-readout final ordering even *inverts*. The honest content is a *sample-efficiency / initialization* claim. Ensure title, abstract, and conclusion do not let "transferable representation" imply an asymptotic-quality benefit that the data does not show.
- *(MINOR)* The future-state probe ($k{=}20$) has $R^2 \approx 0$ for all conditions yet appears with a $+0.84$–$0.92$ correlation; that correlation rides on the sign of near-zero numbers and is not meaningful. The draft should say this as plainly as it already says it for the saturated current-state probe.

**Observations (non-defects).** The state\_h0 artifact is correctly flagged; the P2E timing analysis is a real, non-obvious result; the disclosure and limitations are unusually candid for a course report.

**No CRITICAL issues.** The claims are hedged and the data is real; the problems are framing and isolation, not validity. Per Checkpoint Rule #4, a non-Accept decision is therefore permissible and appropriate, but Reject is not warranted.

---

## Phase 2 — Editorial Decision

**Decision: MAJOR REVISION** (close to minor; all items are addressable by writing, no new experiments required).

**Weighted score (mean of panel):** Originality ≈ 65, Rigor ≈ 60, Evidence ≈ 62, Coherence ≈ 74, Writing ≈ 80 → **≈ 66/100**. A competent, honest mini-study whose claims slightly outrun its evidence in *presentation*; tightening the framing lifts it cleanly into acceptable range.

**Consensus across reviewers.** (1) The correlation cannot isolate coverage as causal (R1 evidence, DA core). (2) Statistical presentation overstates precision for $n{=}2$/$n{=}6$ (R1). (3) Method deviations from canonical P2E/APT must be flagged (R2). (4) The "transferable representation" claim must be scoped to sample efficiency given §F (DA, R3).

**Revision Roadmap (prioritized; all writing-only).**
- **P1-a (MAJOR, R1):** Add RL-evaluation-statistics framing (cite Agarwal et al., 2021); disclaim significance; soften "2–3×" to a per-seed-visible range; present $n{=}6$ correlations as a qualitative trend with explicit caveat.
- **P1-b (MAJOR, DA/R1):** Make the coverage↔activeness collinearity a first-class limitation; elevate "task-relevant regime reached" to a co-equal alternative hypothesis; demote the causal reading of coverage.
- **P2-a (MAJOR, R2):** State explicitly that C3/C4 are *adapted* P2E/APT variants (deterministic-posterior target; world-model-latent entropy) and hedge cross-paper comparison.
- **P2-b (MAJOR, R3):** Add the amortization / break-even argument for when reward-free pretraining is worthwhile.
- **P3-a (MINOR, DA):** Flag the $k{=}20$ future-state correlation as riding on near-zero magnitudes (parallel to the state\_h0 artifact).
- **P3-b (MINOR, R1/R2):** Define early-AUC precisely; tie the 500K-budget caveat to the C2 result; confirm matched-budget for the final-return comparison.
- **P3-c (MINOR, DA):** Scope "transferable representations" in title/abstract/conclusion to a sample-efficiency/initialization benefit.

No new runs are required; the roadmap is satisfiable by revision alone.
