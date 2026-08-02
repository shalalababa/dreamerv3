# CEI Level-1 Pilot — Results Record (EXPLORATORY, not registered)

> **⚠ VERDICT WITHDRAWN — 2 Aug 2026 (same day), adversarial harness
> review ("redesign needed").** The review DEMONSTRATED (by execution):
> **(F1)** the claimed behavioral separation never occurs — the noisy-TV
> sensor's constant score (1.0, a `score()` fallback bug) shadows s7 in the
> first-due-sensor loop at node 3, so BOTH policies sense TV forever and
> s7's score is never behaviorally consulted; `separation_time` measured
> the plain hitting time of node 3 (bit-exact match to an independent
> hitting-time simulator, 215.19 == 215.19); the targeted audit separates
> nothing; the 49× ratio compares two procedures neither of which
> separates. **(F2)** the impossibility certified is a tautology (a
> parameter with zero causal influence on the data), and ψ(U_false) is
> computable from the ACTION LOG ALONE (coverage bookkeeping + known prior
> + deterministic covariance path — zero environment steps): the
> construction inverts the realistic problem (declared confidence
> artificially hidden, truth analytic). **(F3)** movement is a fixed
> patrol, so the estimator does not control where the agent goes — the v2
> audit's own must-be-new criterion 2 is violated (selective-labels
> carve-out), and equivalence would BREAK under U-controlled movement.
> **(F4)** registered Failure C (behavioral irrelevance) instantiated
> wholesale; A/B/C were never assessed. WHAT SURVIVES: the ψ-gap
> equivalence pair at ε=0 is analytically sound as far as it goes, and the
> reviewer verified the repaired sense-loop semantics DO produce real
> divergence — the construction is repairable, but the theorem target must
> be re-scoped to score-identifiability under behavioral access in
> settings where the failure is NOT deducible from coverage bookkeeping.
> The theorem-phase go/no-go must NOT fire on this record. Everything
> below is the original (invalidated) text, kept for the record.

**Date:** 2 Aug 2026. Second survivor of the corrected triage
(`Research_Idea_Uncertainty_field_Innovation_Audit_v2_recheck_20260801.tex`).
Instrument: `uncfield/cei.py` (selfcheck PASS); machine record
`report.json`. Exact-filter substrate on the `uncfield/lgfield.py` world —
no learned model needed for this construction.

## Question and construction

When the uncertainty estimator controls the exploration policy, which of
its failures are identifiable from the data that policy generates? We built
the equivalence pair the theory requires: two estimators (U_true = exact
posterior variance per sensor; U_false = identical except it declares the
never-visited z2 sensor certain) that induce **bit-identical action and
observation streams** under the deployed patrol policy (verified over 5
episodes, exact equality), while their reachable-state false-confidence
risk differs maximally (ψ = 0.0 vs 1.0).

## Kill-probe results (all three registered probes)

| Probe | Result |
|---|---|
| P1 random exploration (ε-perturbed policy) | separation cost diverges ~1/ε: mean 69 steps @ ε=0.3 → 194 @ 0.1 → 344 @ 0.05 → 934 @ 0.02 → 1768 @ 0.01; **ε=0 NEVER separates** (exact on-policy impossibility, censored at horizon) |
| P2 parametric inverse optimization | exactly flat between U/U′ on-policy (identical decisions ⇒ identical likelihood) — cannot identify ψ |
| P3 off-policy evaluation | ψ's estimand has **zero on-policy support** (node-3 sensing never occurs) — importance weighting undefined |
| Targeted audit | **4 steps** (walk to the blind node + one observed sensing decision) |

**Decision quantity: random/targeted separation-cost ratio ≈ 49× at
ε = 0.1, diverging as ε → 0.**

## Verdict: CEI SURVIVES its pilot kill test

Per the registered kill criteria ("kill if random exploration /
inverse-opt / OPE suffices"): none suffices. The phenomenon is
constructible, the on-policy impossibility is exact rather than
statistical, and the targeted-vs-random audit gap is large even in a
6-node toy and grows without bound in the natural limit.

## Honest scope limits

In-toy, *finding* the minimal audit is trivial (one blind node, visible by
construction). The theory's real content — necessary/sufficient audit
conditions and minimal-cost design when the blind region is NOT known a
priori, nonparametric estimator classes, structured reachability costs —
is untested here and is where reduction-to-BED (recheck Failure A) remains
the live threat. This pilot licenses the theorem attempt; it does not
constitute one. Next theory target: the audit-distinguishability
separating-set formulation with the equivalence relation induced by
*unknown* estimator classes, where the auditor must localize the blind
region rather than being told it.

## Program status after this read

Both corrected-triage survivors have now cleared their decisive pilots:
NFI (flagship exploit signature, `artifacts/nfi_pilot_20260801/`) and CEI
(identifiability gap real, this record). The two-paper program stands:
NFI leads (mechanism analysis in `artifacts/nfi_mechanism_20260802/`,
sensitivity sweeps running), CEI queued behind it for the theorem-first
phase.
