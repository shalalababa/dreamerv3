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

## Per-fit bootstrap (2026-08-15, run on user GO; B=5000, rng 0,
## 0 fit failures; draws in POST_READ_parametric_bootstrap.json)

Resampling the 8 fits within every (arm, σ) cell and refitting:

- rescued knee σ\*: point **1.92**, 95% CI **[0.42, 2.06]** — the
  knee is below σ=2.06 in ≥97.5% of draws; the soft LOWER tail comes
  from the σ=1 task cell's side-mixed heterogeneity (sd 0.104:
  side0 0.838 vs side1 0.989 — the same side stratification the B1'
  review surfaced on the capacity axis; a side-stratified refit is
  the next refinement if a figure needs a tighter lower edge).
- rescued width: 95% CI [0.005, 0.57] — consistent with
  sharp-to-moderately-sharp; the near-step point estimate is not an
  artifact of the means-only fit but its sharpness is not pinned.
- shared knee: right-censored beyond the ladder in **100% of draws**
  (P(σ\*_C > 8) = 1.0); 95% CI [11.2, 1034] — the upper edge is
  extrapolation noise, the LOWER edge is the usable number.
- knee ratio σ\*_C/σ\*_R: 95% CI **[6.1, 837]**, P(ratio > 4) = 1.0.
- eigenvalue separation under g ∝ σ²: **λ_corr/λ_rew_eff > 37 at 95%
  bootstrap confidence** (upgrades the conservative >17 of the
  means-only fit).

## What this is for

The writing-phase "quantitative form" paragraph + one figure (two
curves with bootstrap bands, two knees, the merge). Quotable form:
"the rescued reward-relevant support dies below σ ≈ 2 while the
shared support's knee lies beyond the sampled range in every
bootstrap draw; the implied eigenvalue separation exceeds 37×."
NOT a registered claim; a finer σ grid in (1, 2] would be the
(unplanned) confirmatory wave if a reviewer demands inferential
knees.

Scripts: `POST_READ_parametric_fit.py` (means fit) + the bootstrap
block recorded in `POST_READ_parametric_bootstrap.json` (inputs =
the 64 per-fit jsons of the two sha-verified bundles).
