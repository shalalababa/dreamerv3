# LABELED POST-READ ANALYSIS — parametric fit of the crowding curves
# (non-registered, descriptive, ZERO decision weight; 2026-08-15)

Inputs: the 8 pinned arm×σ mean AUROCs (B2 read nz1/nz2 + σ-ladder
read nzs2/nzs8). Fitted by least squares on MEANS (no per-fit spread —
widths are therefore lower bounds on the true widths; a writing-phase
refit should use the per-fit jsons in the two bundles with bootstrap
CIs).

## Structural fact (model-free)

Past the knee the task curve MERGES INTO the apt curve:
task 0.6794 vs apt 0.6805 at σ=4; 0.6786 vs 0.6751 at σ=8. The
decomposition task = apt + rescued is therefore read directly off the
data: rescued component R(σ) = task − apt = {0.195, 0.046, −0.001,
+0.004}; shared component C(σ) = apt − 0.5 = {0.218, 0.208, 0.181,
0.175}. What crowding removes from the task arm is EXACTLY the
support the apt arm never had — and what remains is common to both.

## Two-component logistic fit (in ln σ)

- **Rescued component**: R0 = 0.195, knee **σ\* = 1.94**, width 0.03
  (≈ a step; rss 1.4e-5). The rescued reward-relevant support dies in
  a narrow band just below σ=2 — consistent with a tight effective
  eigenvalue λ_rew(1+βa²) being crossed once.
- **Shared component** (scale pinned at 0.218): knee **σ\* ≈ 29**
  (EXTRAPOLATED — right-censored: only a 20% decline is observed by
  σ=8; the conservative statement is σ\*_shared > 8), width ≈ 1.0.
- **Knee ratio**: point ≈ 15, conservative bound > 4.1. Under a
  noise-power mapping g ∝ σ², this bounds
  **λ_corr / λ_rew_eff > 17** (point ≈ 220): the reward-correlated
  bulk support sits more than an order of magnitude above the rescued
  reward-relevant support in effective eigenvalue — the quantitative
  form of "reward-relevant features are low-variance".

## What this is for

The writing-phase "quantitative form" paragraph + one figure (two
curves, two knees, the merge). It converts the ordering theory into
three fitted numbers and a bound. NOT a registered claim; if a
reviewer demands inferential status for the knee locations, the
per-fit refit with bootstrap CIs is the first step, and a
finer σ grid in (1, 2] would be the (unplanned) confirmatory wave.

Script: `POST_READ_parametric_fit.py` (this directory).
