## ⚠ AMENDMENT (9 Aug 2026) — drift-baseline artifact in the conjunction statistic

Read through this note first. The 12-lens review of 9 Aug
(`research_notes/paper5_uncertainty_field/reviews/Review_FullRecord_and_Draft_20260809.md`
§1, independently verified by recomputation — `artifacts/nfi_review_response_20260809/`)
established that this record's FLAGSHIP CONJUNCTION verdict is an artifact of the
move-only drift correction: measured drifts are NEGATIVE in every member, so
`carried_dh_adj = carried_dh − drift·n_actions` is an additive CREDIT (median +0.054
to +0.141 on flagged cycles, vs threshold 0.08). On the RAW realized-ΔH statistic the
conjunction counts are **0 / 0 / 0 / 1** (vs reported 13/61/0/61); 52 of m3's 61
flagged cycles have raw ΔH ≤ 0 (belief entropy ROSE). Recomputing the published
verdict tree on the raw statistic gives ensemble **EXPLOIT-SURVIVES-CIG-ONLY**, not
CIG+PBIM. The 1-Aug harness review's sign analysis ("this correction raised the bar")
was backwards for negative drifts. **UNAFFECTED:** the carried (prefix-conditioned
EIG) leg — every carried number, the 31× chord exhibit, the member heterogeneity of
the carried leg, and the planted-null/positive battery verdicts as battery outcomes.
The conjunction/PBIM-leg quantities in this record are superseded by the raw-statistic
tables in the review-response record.

---

# NFI Level-1 Pilot — Results Record (EXPLORATORY, not registered)

> **Naming note (claim-freeze audit, 2 Aug 2026):** the verdict tier
> "EXPLOIT-SURVIVES-CIG+PBIM" denotes survival of the two TRANSPLANTED
> PRINCIPLES (prefix-conditioned carried-belief accounting; potential-based
> realized-dH accounting), not of the published estimators — CIG's
> estimator scores parameter-information over open-loop rollouts with no
> belief object, and PBIM's guarantee is a terminal correction this
> protocol never encounters. Paper-facing labels: "carried" / "potential".
> Two fidelity arms (gamma-discounted Ng-form recompute; CIG-faithful
> disagreement-kernel) were mandated by the audit and are recorded
> separately. See research_notes/NFI_ClaimFreeze_Audit_20260802.md.


**Date:** 1 Aug 2026. **Status: PILOT.** Design: gitignored
`research_notes/Pilot_NFI_Design_20260801.md` (pre-outcome, incl. Amendment 1
after a 4-BLOCKING adversarial harness review; pilot1 was killed unread on
that review; this is pilot2 on the fixed harness). Thresholds were pilot
defaults, not frozen; nothing here is a registered claim. Machine records:
`summary.json`, `config.json` (this dir); full per-cycle arrays + frozen
ensemble + episodes in `local_results/uncfield/pilot2/`; code `uncfield/`
at this commit (selfcheck PASS: 4-model planted battery, 626 cycles,
coherence 1.9e-11).

## Question

Does a planner searching a FROZEN learned belief world model discover
repeatable, evidence-neutral action cycles with positive predicted
information return — and does any exploit survive prefix-conditioned
information gain (CIG-style carried accounting) and potential-based shaping
(realized-ΔH accounting), the two published defenses?

## Setup (config.json)

Linear-Gaussian field (8-dim; 4 static / 4 dynamic components), 6-node
graph, duplicate static sensors at nodes 1/4, noisy-TV (C=0) channel at
node 3; exact Kalman referee (observation-free true gains). 300 random
episodes × 600 steps; GRU(64) belief WM, diagonal z-head + obs head;
4-member ensemble (losses ≈ −1.03 final, converging smoothly from 10.3);
frozen; 626-cycle sweep (closed walks ≤6 nodes × sense variants), 30
burn-in + 8 scored loops, ML-observation imagination; EPS_TRUE=0.02,
EPS_PRED=0.08; per-member move-only drift baseline subtracted from ΔH
rates; flagship requires carried-EIG ∧ drift-adjusted-ΔH farming on the
same cycle.

## Result

**ENSEMBLE VERDICT: EXPLOIT-SURVIVES-CIG+PBIM (3 of 4 members; cumulative-
severity rule).**

| member | verdict | neutral | naive | cig | pbim | **both** | transient | TV max eig | TV max ΔH_adj |
|---|---|---|---|---|---|---|---|---|---|
| 0 | CIG+PBIM | 484/626 | 81 | 81 | 46 | **13** | 151 | −0.07 | +0.32 |
| 1 | CIG+PBIM | 484/626 | 260 | 253 | 127 | **61** | 292 | +0.87 | +0.10 |
| 2 | CIG-ONLY | 484/626 | 153 | 145 | 0 | **0** | 0 | −0.00 | −0.01 |
| 3 | CIG+PBIM | 484/626 | 102 | 100 | 331 | **61** | 294 | +0.55 | +0.39 |

Representative conjunction exploits (true rate certified by the exact
referee; rates in nats/loop, post-burn-in steady window):

- m1 `c447_1-4-1-4-1-4-1_all` — **the designed duplicate-pair chord loop**:
  true +0.014, predicted EIG **+0.424** (~31× true), ΔH_adj +0.098.
- m3 `c184_0-1-4-3-4-5-0_only_s7` — repeated static-z2 reads: true +0.013,
  EIG **+0.457**, ΔH_adj +0.223.
- m3 `c593_3-4-3-4-3-4-3_only_s6_tv` — **the noisy-TV loop**: true exactly
  0.000, EIG +0.419, ΔH_adj +0.178, naive +0.960.

Noisy-TV farming in carried (prefix-conditioned) accounting: m1 +0.87,
m3 +0.55 nats/loop on a channel carrying zero field information.

## Secondary reads

- **Heterogeneity is real and informative**: same data, same architecture,
  4 seeds → three exploitable accountants and one (m2) with zero
  drift-adjusted-ΔH farming and a clean TV channel (its 145 CIG-tier flags
  are promise-only). Exploitability is seed-dependent, not universal.
- **Holonomy mechanism (R1)**: sym-KL-holonomy vs farming correlations are
  weak/mixed (m0 +0.10/−0.02; m1 −0.03/+0.02; m2 −0.55/−0.61; m3
  −0.13/−0.21) ⇒ the simple "loop holonomy magnitude predicts farming"
  hypothesis is NOT supported at this scale; the commutator residue stays
  a diagnostic (no promotion). The dΦ-channel and per-cycle mechanism
  analysis are the follow-up.
- **Transient tier** engages heavily on exploiting members (151/292/0/294)
  — saturating farming coexists with steady farming.
- Drift baselines are small and negative (−0.002 to −0.020/action) —
  steady exploits are not drift artifacts (the correction *raises* the bar
  they cleared).
- R3 adaptivity residue: top cycles started off-warmup-node in all
  members ⇒ null this run (instrument limitation, noted).

## Caveats (pilot-grade)

Small GRUs at modest training; a skeptic reads "your model is merely
miscalibrated." The NFI claim is precisely that a planner *weaponizes*
whatever miscalibration exists — but the paper-grade version needs:
training-length/capacity sweeps (does the exploit die with better models,
and how fast), sampled-observation imagination robustness, seed replication
of the data set, threshold freeze + registration, executed-policy (not just
imagined) harvesting, and the claim-freeze literature audit (Caron et al.
2507.02639 positioning; CIG/PBIM papers read in full). Single environment
family; diagonal z-head.

## Decision (per the design-note verdict tree)

Flagship-signature branch fires ⇒ next: mechanism analysis (dΦ-channel,
per-cycle anatomy of exploited vs clean loops, member-2 contrast as the
natural control), sensitivity sweeps, then claim-freeze audit +
registration before any confirmatory run. CEI pilot remains queued behind
this read per the corrected triage.
