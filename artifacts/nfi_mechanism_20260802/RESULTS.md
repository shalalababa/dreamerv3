# NFI Mechanism Analysis — Results Record (EXPLORATORY, not registered)

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


> **⚠ AMENDMENT — 2 Aug 2026 (same day), adversarial instrument review
> ("fix before read"; all findings verified by execution).** The record
> below is amended, not withdrawn: **claims (i) and (ii) SURVIVE attack**
> — the Bayes-gain formula is exactly the Kalman entropy drop (the
> plausible alternative 0.5·log(1+t/R) is the wrong one), the coherent
> DiagExactFilter adapter calibrates the ratio to **exactly 1.000** on
> every sensed cycle, denominators are non-degenerate everywhere (min
> R_implied 0.031, ≤0 in 0/32 accounts), TV-licensed-exactly-0 is
> structural, and mechanism.json bit-reproduces from code + saved
> ensemble (421/421 fields). **Claim (iii) — the anatomy-3 dichotomy —
> does NOT survive:** (a) the quoted m0 off-support exhibit INVERTS under
> the matched-walk baseline (sensed +0.037 vs the SAME walk move-only
> +0.048 — walk drift, not sensing leakage); leakage is genuinely
> supported for m1/m3 only (e.g. m3: sensed +0.082 vs matched move-only
> −0.057, landing on STATIC components — damped-mode relaxation
> excluded). (b) The "duplicate double-count" anatomy is WITHDRAWN: its
> exhibit c451 is non-neutral (true +0.038 > EPS_TRUE) and not an exploit
> for any member, with realized contraction below the referee's true gain
> (mostly legitimate information); the actually-exploited duplicate loop
> c447 shows a 50/50 on/off-support split — no clean dichotomy exists.
> (c) Name conflation corrected: the 11.6× headline is c447
> (1-4-1-4-1-4-1 all); the "on +0.060 / off −0.006" figures are c451 (the
> designed chord, actual ratio 5.4×). (d) Wording: TV "+0.84 realized" is
> the carried EIG PROMISE — the traced belief EXPANDS off-support
> (−0.15/loop; dh_adj +0.009 = eig-only exploit); m3's TV both-exploit
> status is created by the registered drift adjustment; TV farming fires
> 2/4 members (m0/m2 are TV-coherent). (e) No thresholdable cycle-level
> discriminator: m2's non-dh-exploited cycles score 3.8–4.5 inside the
> exploited range — the ratio separates exploited MODELS' cycles from
> planted-coherent models, not exploited cycles from clean ones within a
> model. (f) m3 caveat: its steady-state z-head re-inflates z2 variance
> (licensing 0.29–0.42 nats/read on referee-neutral cycles), so m3's
> dominant incoherence is heads-vs-referee with only 1.4–1.8× update-side
> excess; "localized to the update operator" is cleanest for m0/m1/m2.
> Read the anatomy sections below through this amendment.

**Date:** 2 Aug 2026. Follow-up to `artifacts/nfi_pilot_20260801/` (flagship
signature EXPLOIT-SURVIVES-CIG+PBIM, 3/4 members). Instrument:
`uncfield/mechanism.py` on the frozen pilot2 ensemble; machine record
`mechanism.json` (per-member obs-head calibration fingerprints + 8 traced
cycles each: top conjunction exploits, the designed chord loop, top TV
cycle, 3 move-only clean controls).

## Three exploit anatomies found (H-M1/H-M2 both fire; distinct classes)

1. **Belief-update over-contraction vs the model's OWN heads (H-M1,
   dominant).** Per sense action we compare realized 3-pt-GH EIG against
   the head-consistent Bayes gain 0.5·log(tot_var/R_implied) computed from
   the model's z-head projection + obs-head total variance. On exploited
   cycles the ratio is **1.4–11.6×** (m1 chord-loop: EIG +0.075 vs
   own-heads-licensed +0.006 = 11.6×; m0 s7-loops 2.6–4.4×; m3 1.4–1.8×
   at high absolute scale). The GRU's recurrent update contracts more than
   its own generative model can justify — the internal chain-rule
   incoherence NFI's potential-characterization targets. Notably
   R_implied ≤ 0 occurred **nowhere** (0/8 sensors, all members): the
   heads are individually sane; the *update operator* is the incoherent
   part.
2. **Pure fictitious information on the TV channel.** For C=0 the
   head-consistent gain is EXACTLY 0, yet realized EIG reaches **+0.839**
   (m1) and +0.272 (m3): the update operator has learned a spurious
   "noise observations contract the field estimate" association. The
   cleanest exhibit in the study.
3. **Off-support leakage (H-M2).** On the s7-farming loops, predicted
   contraction lands on components the loop's sensors never measure at
   rates comparable to or larger than on-support contraction (m0: off
   +0.037 vs on +0.025; m3: off +0.082 vs on +0.042). The duplicate-pair
   chord loop shows the complementary anatomy: on-support over-contraction
   (duplicate double-count) with ~zero leakage (m1: on +0.060, off −0.006).

## Clean-member contrast (H-M3)

Member 2 shows the SAME defect classes at sub-threshold amplitude (ratios
3.8–4.5 but tiny absolute rates), a *conservative* chord ratio (0.6 — EIG
below its own license), and a coherent TV channel (EIG −0.001).
Exploitability is a matter of degree, not kind; warmup obs-head calibration
fingerprints do NOT separate the members (sd_pred 0.27–0.31 across all
four) — the difference lives in the update operator, not the heads.

## Candidate practical deliverable (for the claim-freeze audit)

The **self-consistency ratio** (realized EIG ÷ own-heads Bayes gain) is
computable WITHOUT any referee — both terms come from the model itself —
and separates exploited from clean cycles in this pilot. If it predicts
exploitability across the sensitivity sweeps, it is the paper's practical
diagnostic (an intrinsic coherence check for information-seeking agents).
To test in the sweep read; not a claim yet.

## Caveats

Pilot-grade: 8 traced cycles/member, ML-observation imagination, one env
family; the ratio's denominator degenerates when heads imply near-zero
gain (reported ∞ on TV — there the NUMERATOR is the finding). Holonomy
(sym-KL) correlation remains unsupported per the pilot read; the dΦ channel
is definitionally the dh rate and was not separately informative.
