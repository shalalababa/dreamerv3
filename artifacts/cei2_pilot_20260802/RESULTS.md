# CEI v2 Pilot — Results Record (EXPLORATORY, not registered)

**Date:** 2 Aug 2026. Supersedes the WITHDRAWN v1 pilot
(`artifacts/cei_pilot_20260802/`, invalidation note at top). Instrument:
`uncfield/cei2.py`; design + review lineage:
`research_notes/Design_CEI_v2_20260802.md` (§7b build addendum, §7c
adversarial review record). **Review discipline honored: the instrument
was adversarially reviewed BEFORE this read** — verdict "fix before
run/read"; 3 BLOCKING (B1 movement-claim re-scope, B2 commitment
deadlock, B3 censored-mean E3) + 6 non-blocking, all fixed; core
construction verified-and-held by execution; selfcheck re-PASS; then this
run. Machine record `report.json` (copy; source
`local_results/uncfield/cei2/`).

## Construction (one paragraph)

World = lgfield with s7's true noise R7 = 1.0 unknown to the auditor.
U_false = exact Bayes under assumed R̂7 = 0.01 (a **coherent liar**);
U_true = calibrated. Both drive a committed uncertainty-greedy policy
(post-update release). Falseness is **calibration-level, not
coverage-level**: both estimators visit node 3 and read s7 — U_false
once, U_true 19 times. Kalman covariance paths are observation-free, so
actions are ANCILLARY for R7 (bit-identical logs across worlds; ψ is
provably not computable from the action log — v1's F2 tautology
discharged) while the two estimators' logs differ (v1's Failure-C
discharged). Scope (review B1): the falseness controls **collection
intensity at a fixed stop**, not the route — the selective-labels
carve-out is *weakened to its statistical form* (support under both
hypotheses, bounded LR, IPW defined), not discharged; genuine route
control = v2.1 coupled arm (design note §7c).

## E1 — passive identification asymmetry (the impossibility, empirically)

n_s7 and ψ are deterministic (observation-blind policy): **U_false: 1
read, ψ = 0.5; U_true: 19 reads, ψ = 0.** The logLR column (R7_TRUE vs
R̂7, exact sequential predictives) across 5 env seeds:

| estimator | logLR across seeds |
|---|---|
| U_false | −0.10, −0.24, −0.25, −0.33, **+0.12** (bounded; one seed mildly *favors* the false model) |
| U_true | +579, +1001, +1048, +506, +594 (decisive every seed) |

Distributional statement (review N3): single-read P(|logLR| > 1.5) =
5.3%, P(> log 19) = 1.0%, median 0.29 — the impossibility is
in-probability (bounded Fisher information), not almost-sure.

## E3 — separation cost decomposition (review-B3 redesign)

Mechanism fact first: **a single ε-attributed late read is decisive**
(mean signed logLR per late read ≈ 40–52 nats ≫ log 19 = 2.94), because
by then U_false's own bookkeeping has collapsed its predictive to
var ≈ 0.02 while the data has spread ≈ 1.5. So the separation cost is
entirely the cost of *getting one read*, i.e. ≈ 1/hazard.
(`derived_sep_steps` in report.json assumed linear LR drift and
underestimates when one read suffices — documented formula artifact; use
1/h and the direct validation below.)

| ε | hazard (reads/step, t>100) | 1/h (≈ separation steps) | direct validation |
|---|---|---|---|
| 0.50 | 1.36e-3 | 733 | 40/40 crossed, median 387 |
| 0.35 | 2.63e-4 | 3,808 | 38/40 crossed, median 3,745 (**agrees with 1/h**) |
| 0.25 | 4.15e-5 | 24,120 | 20/40 crossed by t=20k (consistent: t_sim < 1/h) |
| 0.15 | 5.03e-6 | 198,990 | 7/40 crossed, median 25 = early-window/luck stratum only |

Hazard scaling: **h ∼ ε^4.6** (fit over the grid) — from the node-0
parking state a read needs ~3 ε-moves against the greedy pull plus an
ε-sense, so separation diverges *faster* than the 1/ε originally
registered. Early-window "luck" column: crossings at t ≤ 100 mix the ~1%
single-deterministic-read luck with early ε-reads; at large ε it is
mostly the latter (label caveat).

**Targeted audit (matched criteria, review N1): 3 walk steps + m\* forced
reads = 6 (ML pick) / 7 (log-19 criterion).** Passive-vs-targeted gap:
~10² at ε = 0.5 → ~3×10⁴ at ε = 0.15, unbounded as ε → 0.

## E4 — audit cost scaling (Thm-2 empirical shadow)

m*(power 0.95, ML) vs falseness closeness R̂/R: 0.01→**3**, 0.1→**7**,
0.3→**21**, 0.5→**51**, 0.7→**173**. Monotone, χ²-flavored blow-up as
the lie approaches the truth — the audit-cost curve the theorem phase
must reproduce.

## L2 blind-spot demo (from selfcheck, review-hardened)

Coherent liar replays Bayes-coherently to residual 1.4e-17 (**passes**
the query-access audit; its wrong R̂ is recovered as if legitimate) while
a parity-alternating over-contractor fails at residual 0.083 with ψ = 0 —
**the L2 verdict runs opposite to ψ**. Scope (review N4): only
non-constant incoherence is flaggable (a constant double-contractor is
absorbed as R̂/2); a coherent Q-liar is flagged-but-misdiagnosed; the
mean channel is invisible to both L2 and ψ. Blind-spot theorem must be
scoped to miscalibration inside the fitted (R̂ | known-dynamics) family.

## Verdict vs the registered kill-probes

Kill criterion "random exploration / inverse methods / OPE suffice":
**none suffices.** Passive on-policy identification has bounded LR
(U_false's own collection starves the informative reads); ε-exploration
separates only at cost diverging ~ε^-4.6 while the targeted audit is 7
steps flat; and each refutation datum is individually decisive once
forced — the sharpest form of confidence-starves-refutation. **CEI v2
survives its pilot. Theorem phase licensed** on: (i) bounded-LR
partial-identification impossibility at L1, (ii) intervention cost
O(diameter + m*(γ)) with the E4 scaling, (iii) the L2
coherent-class blind spot (scoped as above).

## Honest scope limits

Collection-intensity (not movement) control — criterion-2 satisfaction
deferred to the v2.1 coupled arm (dynamics coupling = identification-
DELAY regime, needs own design pass). Parametric falseness family (R̂
scalar, one sensor); nonparametric equivalence classes open. Single
world/geometry; the ε^4.6 exponent is geometry-specific (the general
statement is "graph-distance-driven polynomial in 1/ε", not the
exponent). ψ variance-scoped (mean-channel falseness invisible by
definition). Localization (auditor not told which sensor) untested —
Thm-3 territory.

## Program status after this read

NFI flagship (mechanism read done; sensitivity sweeps pending on RCC) +
CEI v2 pilot survived post-review ⇒ the two-paper program stands with the
L2 blind-spot theorem as the bridge claim: NFI's self-consistency
diagnostic catches operator incoherence; CEI proves what that audit
family cannot catch (coherent miscalibration) and prices the intervention
that can. Next: NFI sweep read (gated on its instrument review) →
claim-freeze lit audit; CEI theorem phase + v2.1 coupled arm design.
