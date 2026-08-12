# PREREG: R1 legibility-gating model — revision + registered out-of-sample predictions — 2026-08-11

User GO 2026-08-11 ("take the theory addendum, but still try to find
ways to prove lambda works … until we have decisive evidence that
lambda doesn't work"). This document (a) freezes the minimal revised
model R1, (b) registers its out-of-sample predictions for the #21 dose
read and the approved family/domain waves, and (c) keeps the original
λ-competition mechanism ALIVE in two-regime form with two registered
discriminating designs whose outcomes are the operationalized
"decisive evidence" criterion. This document licenses no claims by
itself; each prediction is graded when its wave reads, and no executed
verdict changes.

## Provenance and honesty block (value-aware, disclosed)

R1 is constructed POST-HOC, after the interior-regime results. Its
predictions below are therefore its out-of-sample test — that is the
point of this freeze. Known at freeze: every executed read through
2026-08-11 (in particular: the W0/E3v2 interaction chain; rgo
sufficiency + sgb tight null; rew-NLL level separation 1.2 vs 2.3;
finger #9 ABSENT-OR-NONLINEAR w/ apt AUROC below chance; cup TOST
parity vs finger violation; spectral-v2 P-SM1 cup 21/31 vs finger
45/47; capacity-robust interaction to 25m; U1–U4 no-flips; f_R
saturation w/ unmoved mediator; breadth informative null at 4.84-PR
verified separation; s×w_r both-axes-inert; TD-MPC2 17-Jul aware
+35.3* / free +25.8 ns / contrast +9.5 ns; P2E below scratch; rif ≈
apt).

**Critical timing disclosure**: the #21 dose bundle
(`dose_task_20260811_203441`) has LANDED on local disk at freeze time
but is UNOPENED — no file inside it (including its manifest) has been
read in this session before this document was written; the ONE
registered dose read executes immediately after this freeze,
same-session, with this file's sha256 recorded in the dose read
RECORD. Unknown at freeze: every #21 outcome quantity, every
family-wave (TM2-powered/LeWM/bridge), domain-wave, and λ-trial
quantity.

## The model (R1)

Feature j (of the pretraining representation) becomes USEFUL transfer
support iff its reward-gradient signal-to-noise clears a per-feature
threshold:

    include(j)  ⟺  SNR_j = a_j · s_j > θ_j

where a_j = the reward-alignment signal accumulated during
pretraining (for occupancy-manipulated task pretraining, a_j ∝ f, the
occupancy of target-relevant experience), s_j = per-feature signal
scale, θ_j = per-feature noise floor. **Independent gating: no
cross-feature competition, no shared capacity term in the tested
(interior) regime. The feature spectrum is a READOUT of legibility
(P-SM1's confirmed signature), not a cause of inclusion.** With
heterogeneous ratios r_j = θ_j/s_j, the included-set size at occupancy
f is N·F_r(f) (F_r = the CDF of r_j), and behavioral payoff is a
concave, non-decreasing function of included legible support.

Consistency with the ledger (why R1 and not the original): the breadth
null, s×w_r inertness, and capacity-robustness to 25m are PREDICTIONS
of no-competition; f_R inertness is consistent (spectrum moved,
per-feature alignment did not); everything confirmed on the g-side
(arm ordering, rgo sufficiency, NLL levels, dose existence, cup/finger
dissociation as threshold-crossing at ceiling) is retained.

## Part A — registered predictions

**#21 dose read (`PREREG_dose_task_20260810`, graded at its ONE read):**

- **A1 (G1 gate)**: passes — level3 − level0 > 0 (the main effect
  exists).
- **A2 (monotonicity, strong form)**: the dose response is monotone
  non-decreasing. A decisive NON-MONOTONE / falling-limb activation
  pattern (P-DR1 branch) FALSIFIES independent gating (more legible
  occupancy can never hurt under R1; a falling limb requires
  competition or interference).
- **A3 (shape, strong form)**: STEP must NOT win the P-DR2 CV
  comparison decisively. Homogeneous thresholds (a step) are the
  registered alternative gating structure; heterogeneous-θ R1 predicts
  a smooth activation (CDF composed with concave payoff).
- **A4 (shape, weak directional)**: between the smooth forms, R1 leans
  CONCAVE — ln(f) preferred over linear — with disclosed low
  resolution (4 levels over a 6× occupancy range; a linear CV win at
  this resolution is a weak loss, not a falsification).
- **A5 (scope note, registered)**: #21 does NOT discriminate R1 vs
  λ-competition — at ample capacity both predict the same interior
  behavior. #21 tests the gating STRUCTURE (heterogeneous vs
  homogeneous vs absent). λ-discrimination is Part B's job.
- **A6 (E4 membership, descriptive)**: target-relevant membership
  (P-DR3 curve) rises with f; no membership loss at high f.

**Family waves (graded at their future registered reads):**

- **P-F1 (TM2 powered contrast)**: the aware−free contrast in TD-MPC2
  is ATTENUATED relative to Dreamer's task−apt contrast (ratio < 1)
  while remaining ≥ 0. Mechanism: TM2 encoder gradients are
  reward/TD-carried even in weak arms, and the free arm
  (consistency-only) includes more control-relevant support than pixel
  reconstruction. Premise check (diagnostics wave, runs first): TM2
  free-arm reward-legibility (rew-NLL analog) sits CLOSER to its aware
  arm than Dreamer apt sits to Dreamer task.
- **P-J1 (LeWM / JEPA-family arm)**: frozen-transfer adaptation value
  ordering: task-aware > LeWM > apt (strict LeWM > apt = the primary
  directional), with LeWM ≥ TM2-free-analog unscored (cross-family
  level comparison, descriptive only). Basis: latent-prediction
  objectives include more reward-alignable support than pixel
  reconstruction (the +25.8 TM2-free hint is disclosed as known).
- **P-D1 (third domain, sparse)**: in a sparse-reward third domain
  (reacher_hard or acrobot-sparse; named at wave registration) the
  task×transfer interaction is PRESENT (CI > 0).
- **P-D2 (dense domain, discriminating)**: in a dense-reward domain
  (walker_run-class) the interaction is ATTENUATED (point estimate
  below the finger interaction; directional, power-disclosed at
  registration). Dense reward → recon features already cross θ →
  smaller legibility bottleneck.

## Part B — λ kept alive: two-regime form + the decisive trials

Registered position: the three λ-side negatives kill three
OPERATIONALIZATIONS in the interior regime; they do NOT test
competition where it must bind if it exists. λ-competition is retained
as a live boundary-regime hypothesis: gating in the interior,
competition possibly at the capacity boundary. "Decisive evidence"
(the user's criterion) is operationalized as the outcomes of the two
designs below — both have mediators set BY CONSTRUCTION (the f_R
failure mode is structurally excluded):

- **B1 (capacity-descent)**: shrink capacity DOWNWARD from the
  standard size (e.g. 12m → 1m → ~400k, or latent-dim constriction at
  fixed trunk; descent respects the standing 25m upper cap). Side-by-
  side predictions: **λ-competition** ⇒ the task×transfer interaction
  GROWS as capacity falls, and E4 membership composition reshuffles
  toward high-λ features in reward-free arms (spectral-v2 rank shift);
  **R1** ⇒ proportional degradation of both arms, interaction flat-to-
  shrinking, membership composition stable. AXIS1_SIZE plumbing + E4 +
  spectral-v2 are the instruments; full design at its own
  registration.
- **B2 (nuisance-injection)**: add high-variance distractor channels
  (D0 distractor wrapper) at graded load. **λ-competition** ⇒
  load×arm interaction (reward-free arms lose target features to
  crowding; a_j² rescues task arms); **R1** ⇒ a main effect of load at
  most (noise floor θ rises for both arms), NO load×arm interaction.
- **Adjudication (registered)**: both trials λ-negative ⇒ λ-competition
  is DECISIVELY dead in-scope and R1 stands alone (the paper says so
  with full force). Either trial λ-positive ⇒ the two-regime model is
  adopted (competition confirmed at its boundary + gating in the
  interior; the three interior nulls become predictions of the
  two-regime form — the theory gets STRONGER than the original).
  Split/partial outcomes are reported as-is; no wording beyond the
  branch definitions is licensed.

## Discipline

Frozen before the #21 dose read executes (same session; sha256 of this
file recorded in the dose read RECORD). Each Part-A/B prediction is
graded ONCE, at the ONE registered read of its wave, by that wave's
frozen reader — this document adds no estimators and modifies no
frozen rules. No reviewer on this document (predictions doc, not an
instrument; precedent PREREG_theory_predictions_20260717).
