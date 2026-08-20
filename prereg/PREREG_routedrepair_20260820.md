# PREREG — Routed repair kill (algorithm Tier-1), 20 Aug 2026

**Status: FROZEN at user commit. ONE read execution. KILL-scale test.**
Candidate: `reviews/Algorithm_Ideation_20260819.md` §3.4 (P 0.60 vs full
fine-tune / 0.25 vs scratch). Self-scooping fence as in
PREREG_mindose_reacher_20260820: standalone-algorithm-paper material.

## The question

Given a SUNK reward-free-pretrained trunk (known in-house to be a
liability under full fine-tuning), does an rgo-configured re-fit —
reward-head gradients into the representation, recon/dynamics gradients
stopped/down-weighted, the registered rgo arm config — repair it? This is
a setting NO in-house arm has tested (no apt-init + rgo-refit cell exists
in the record). Distinct from the killed "gradient-filtered transfer"
(0.10): that concerned filtering during reward-available pretraining;
this is repair of a sunk reward-free init.

## Pinned baselines (look-0, artifacts cited)

- Scratch adaptation anchor **147.6823** (sd 33.224)
  (`artifacts/review_response_20260808/zero_compute.json`).
- apt-unfrozen (full fine-tune of the same donor class), side1:
  **89.8984625**, [apt_s1 − scratch] CI [−82.02, −33.78] — the baseline
  is in-house-confirmed to LOSE to scratch (same artifact).
- Mechanism risk, stated: spectral competition predicts pre-installed
  high-λ recon features may hold trunk capacity against the reward
  channel — the apt init could actively block routing. Either outcome
  extends the account.

## Design — 16 cells, finger q1 side1

- **Repair arm (8)**: donors = fq1s1 seeds 1–8 WM checkpoints (inventory
  20 Aug: 8/8 present). rgo-configured re-fit on the frozen q1 side1
  buffer, **500000 updates** (house fit length — mechanism-comparability
  first; the shorter-refit cost-efficiency version is follow-up material
  licensed only by a PROCEED) → standard unfrozen adapt.
- **Shrink-and-perturb free arm (8)**: same donors, θ → 0.5·θ + ε,
  ε ~ N(0, 0.01²) per parameter, then standard full fine-tune adapt.
  DESCRIPTIVE-ONLY (single (λ, σ) point, no tuning, no α claim) — the
  ~85%-published baseline rides along so a reviewer's first question has
  an answer.

## Estimands and decision rules (frozen kill criteria)

- **D1 (the 0.60 claim)**: two_sample [repair_adapt − apt_unfrozen_s1
  pin] — one_sample vs the pinned 89.8984625 (n=8), permutation + BCa,
  α=.05. FIRE ⇒ routing beats full fine-tuning of the same sunk trunk.
- **D2 (the salvage bar)**: repair_adapt mean vs the scratch anchor:
  kill-level criterion **mean ≥ 122.68** (scratch − 25 AUC), reported
  with the one_sample CI vs 147.6823. This is a NEIGHBORHOOD criterion,
  not an equivalence claim — n=8 cannot license TOST at ±25 and the read
  must say so.
- **PROCEED** iff D1 fires AND D2's bar is met. **DROP** otherwise; if
  D1 fires but D2 fails, record "routing helps but does not reach
  scratch's neighborhood" — DROP for the salvage claim, one sentence in
  the paper. No seed extension, no second look.
- S&P arm: reported descriptively next to both.

## Gates

Donor ckpt shas recorded and matched to the fq1 run dirs; refit counters
== 500000 ×8; adapt witnesses ×16; config byte-identity within arms;
STRICT modal n_ep; within-invocation comparisons; refusal on any failure.

## Reader

`analysis/routedrepair_read.py` — frozen (selfcheck PASS) **before any
job is submitted**; one_sample machinery; literal pins {89.8984625,
147.6823, 122.68, 500000, seeds 1–8}.

## Ops

Donor sha inventory → 8 refits + 8 S&P inits → 16 adapts (~25 GPU-h
beyond the refits; refits 8 × ~2 h) → collate + witness → bundle + sha
manifest → **[ME] ONE read**.
