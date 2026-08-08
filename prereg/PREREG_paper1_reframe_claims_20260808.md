# PREREG: Paper-1 reframe claim licences (claim-freeze, 2026-08-08)

Claim-level registration on EXISTING data (no new compute). Purpose: the
7-Aug review (Review_FullRecord_20260807.md §3.1 items 1–3, §5) identified
three already-computed facts whose claim licences the existing registrations
deliberately withheld (the scratch-anchor prereg is descriptive-only; the
random pretrain runs were never registered as a comparator). This file
freezes the claim wordings and the exact estimators BEFORE any manuscript
uses them. All inputs existed and were known before this freeze — this is
disclosed; the registration licenses *wording*, not discovery. Estimators
and numbers: `artifacts/review_response_20260808/zero_compute.json`
(deterministic re-derivation; scripts archived there).

## C-R1 — random-policy floor (licensed claim)

**Claim:** "A uniform-random policy attains AUC100k = 83.5 ± 15.1 on
finger_turn_hard (5 seeds × 96 episodes; `expl.mode: random` is
`uniform(-1,1)`), and every reward-free frozen-transfer cell in this study
(fq1s0 83.2, fq1s1 79.4, fq2s0 66.4, fq2s1 84.2) lies inside that band: the
reward-free null is *no learning*, not merely no benefit of composition."
Estimator: AUC100k per `analysis/adaptation_auc.py` over
`pretrain_random_finger_seed{1..5}` scores; band = mean ± 2 sd.
Scope: finger; descriptive comparator (protocols differ: pretrain logging vs
frozen adaptation); no CI-based decision attaches.

## C-R2 — negative transfer vs random initialization (licensed claim)

**Claim:** "Under a protocol that differs from from-scratch training only by
`--run.from_checkpoint` (both unfrozen, matched budget/architecture),
reward-free-pretrained initializations reach 0.54×/0.61× (s0/s1) of the
from-scratch anchor; the unpaired seed-cluster bootstrap CIs of the
difference are entirely negative (s0 [−94.1, −43.8]; s1 [−82.0, −33.8]).
Reward-supervised pretraining on the same buffers reaches 2.3–3.3×."
Estimator: `ax1uzfq1s{0,1}` (n=8 each) vs `scratch` (n=8), AUC100k,
B=10,000 rng(0) unpaired bootstrap; task arms `ax1uztq1s{0,1}` for the
upper anchor. Sensitivity to report alongside: Welch t. Scope: finger, 1m,
125K-step window; the from_checkpoint-only protocol identity is the licence
for "than random initialization".

## C-R3 — scale-free form of the interaction (licensed claim)

**Claim:** "The occupancy × supervision interaction survives a log transform:
ratio-of-ratios 1.75× [1.24, 2.49], exact sign-flip permutation p = .009,
13/16 seeds (aware occupancy-ratio 1.79 vs reward-free 0.96); the pixel 2×2
is null on the log scale too (0.94× [0.60, 1.55]) — neither result is an
additive-floor artifact." Estimator: per-seed
Δlog = [log A(t,s1) − log A(t,s0)] − [log A(f,s1) − log A(f,s0)], W0 n=16.

## Non-claims (frozen)

- No claim that reward-free pretraining is harmful beyond this protocol
  family/domain; C-R2 is a finger, reconstruction-family, 125K-window fact.
- No cross-modality behavioral comparison (D3 budget asymmetry).
- These licences do not alter any registered verdict; they add wordings for
  facts the registered artifacts already contain.

Freeze: commit this file before any manuscript text asserts C-R1/2/3.
