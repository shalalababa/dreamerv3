# PREREG — NFI Scale Exhibit (SE wave), DRAFT v2 (12 Aug 2026, post-review)

**Status: DRAFT v2 — freeze after smoke FILL slots.** v1 reviewed same
day (instrument reviewer, Fable): 4 BLOCKING / 8 MAJOR / 9 MINOR, all
adopted below (adjudication in
`artifacts/nfi_assessment_response_20260812/RESULTS.md` §2). Plan:
`research_notes/paper5_uncertainty_field/Plan_AssessmentResponse_20260812.md`
§A. Purpose: the NFI external-validity leg — the fictitious-information
chain at standard world-model scale, ground truth **by construction**.
NOT gating ICLR.

## 1. Substrate (pinned)

DreamerV3 (this repo), size1m, dmc proprio `cheetah_run` [FILL smoke:
proprio-key inventory; fallback walker_walk]. Objective: the repo's
deployed latent-ensemble disagreement module (`dreamerv3/explore.py`,
`expl.mode: p2e`, `disag_task: False` — asserted at agent.py:104), the
UNMODIFIED training path. Training steps [FILL: size1m default]. **Seeds
0–7 pre-committed** (8 runs; no conditional extension clause). Arms are
independent samples — never cross-arm seed-paired (1-Aug GPU
nondeterminism rule).

## 2. Planted channels (appended observation keys)

Precise information statements (review M-1):
- **N (noisy-TV)**: `embodied/envs/distractor.py` OU wrapper, dim 8,
  scale 1.0, `basesd` PINNED from the smoke run (no online calibration
  window — the zero statement then holds from step 0), wrapper seed =
  training seed. **Zero information about task state and reward**; the
  OU latent itself is learnable, so early disagreement on N is partly
  legitimate — the registered claim is FINITE-TIME NON-DECAY (M4), not
  instant-zero.
- **D-family (redundancy ladder) — FLAGSHIP = D0**:
  - **D0**: exact duplicate (ε = 0) of one pinned proprio source key
    [FILL smoke: key id + dim]. Zero marginal information given the
    source key, and no stochastic component whatsoever — if the
    objective allocates value to D0, no noisy-TV reading survives.
  - **D1**: duplicate + i.i.d. N(0, (0.05·refsd)²); **D2**: same at
    ε = 0.5·refsd. The ε-ladder is the registered discriminator
    (review B3): redundancy farming predicts D0 > floor and/or
    D-share NOT ∝ ε²; noisy-TV predicts share ∝ ε² with D0 at floor.
- **C (constant ≡ 0, dim 4)**: anchor for the DECODER-PROJECTION floor
  only (its projected share is estimated and can fail); C is NOT a
  masking floor (mean-masking a constant is a bitwise no-op — review
  B1).

All channels appended in the single Stage-1 arm. Wrapper selfcheck =
construction checks (D-family emissions regress exactly onto recorded
source + registered-sd residual; N/D consume only their own RNG) +
smoke cross-correlation diagnostic (review minor 3).

## 3. Arms

**Stage 1: planted arm ONLY, seeds 0–7 = 8 runs** (the v1 unwrapped
control arm fed no registered measurement and carried a dimensionality
confound — deleted per review B4; all Stage-1 primaries are within-arm).
**Conditional D-only arm** (channels D0/D1/D2 without N, seeds 0–3,
4 runs) PRE-AUTHORIZED, trigger: N-share fires while all D-shares are
at floor (cross-channel interference check, review M-8).
**Stage 2 (behavioral, trigger: P-SE1 fires on any non-C channel):
gated vs ungated at MATCHED observation space** — both arms carry
identical appended channels; the gated arm emits them only when a
pinned proprio coordinate crosses a threshold (else 0), the ungated arm
always. Diversion contrast = gate-vs-no-gate at identical obs
dimensionality, encoder, and loss mass (review B4.2). Gated channels
remain zero-MARGINAL-information because the gating coordinate is
itself observed (stated per review minor 7). 2 × 4 seeds = 8 runs.

## 4. Attribution instruments (registered architecture, review B2)

- **PRIMARY (intervention-free, defined identically in replay and
  imagination): decoder-projected per-key disagreement.** For each
  disag-ensemble member m, push its predicted next latent through the
  frozen decoder's per-key heads (plumbing exists: per-key heads
  agent.py:277-281; imagined-feature decoding precedent
  agent.py:469-479); d_k = Var_m(μ_k^m) averaged over key-k dims,
  normalized per key [FILL smoke: per-dim refsd² vs decoder-licensed
  variance — pin one]. Planted share S = Σ_planted d_k / Σ_all d_k,
  reported per channel. This reads out the DEPLOYED ensemble — no
  retraining (the v1 `disag_target` fallback is WITHDRAWN: it changes
  the trained objective and forfeits the "standard objective"
  wording).
- **SECONDARY (replay-only, causal): interventional masks**, signed Δ
  registered with the directional hypothesis (removal of a farmed key
  REDUCES disag); the inflation side reported separately as a
  shift-artifact diagnostic (review M-2). Mask forms pinned per
  channel: N → batch-permutation mask; D-family →
  source-substitution (D := source value, deleting exactly the
  fictitious content); masks applied over the full filtering window
  [FILL smoke: window length], in raw obs space before symlog, mask
  statistics from a pinned replay slice (review M-3).

## 5. Registered measurements

- **P-SE1 (accounting, PRIMARY).** On N_eval = 512 replay states per
  run at the FINAL checkpoint: per-channel decoder-projected share vs
  the **permutation-null floor** (key-label permutation within run,
  planted/real labels reassigned dim-matched, ≥1000 permutations,
  α = .05 — this is the registered permutation-primary; seed-level BCa
  intervals are reporting-only, review M-6/B1) AND vs the comparator
  θ₁ = own source key's share (D-family) / nearest-dim real key by
  pinned rule (N) (review M-4). **FIRES** per channel if share exceeds
  the permutation null; the FLAGSHIP wording additionally requires the
  D-family discriminator (§2).
- **P-SE2 (ranking, PRIMARY).** M = 256 imagined rollouts (trained p2e
  policy, imag_length 15, replay-start selection rule [FILL smoke]),
  ranked by imagined intrinsic return; statistic = mean planted
  decoder-projected share of the top-k = 10 vs the all-rollout mean
  (dimension-fair continuous form, review B2), permutation test over
  rollout labels. **FIRES** if top-k share exceeds the mean, p < .05.
- **M3 (behavioral, Stage 2 SECONDARY).** Occupancy share of the gate
  region, gated vs ungated arm, + task-return delta under identical
  eval. Directional: diversion toward the gate region.
- **M4 (persistence, SECONDARY).** P-SE1 statistic at 25/50/100%
  checkpoints vs the permutation-null floor at each (falsifiable
  floor, review B1): registered weak form = planted share remains
  above the null at 100%.
- Exploratory (unregistered): M5 coherence fingerprint; encoder-mask
  attribution comparisons.

## 6. Outcome map (frozen wordings at freeze; review M-5 — exhaustive)

Cells over {P-SE1, P-SE2} × {D-family fires w/ discriminator, N fires}:
1. **SE1 ∧ SE2 ∧ D-discriminator**: FLAGSHIP — "a standard disagreement
   objective at standard scale allocates epistemic value to channels
   carrying zero marginal information by construction — including an
   exact duplicate — and its planner's imagined-rollout ranking
   concentrates on them."
2. SE1 ∧ SE2, N only (D at floor, share ∝ ε²): model-internal
   noisy-TV accounting exhibit (beyond the behavioral literature via
   per-key attribution); flagship wording NOT licensed; triggers the
   D-only arm (§3) before writing.
3. SE1 only (accounting without ranking): "the objective misprices;
   the planner does not yet concentrate" — accounting-level exhibit,
   ranking claim dropped.
4. SE2 only (ranking without per-state accounting): report as
   surprising dissociation, no flagship claim, flag for design review.
5. All planted shares at the permutation null: informative null —
   "not exhibited by ensemble-disagreement objectives at this scale
   under these channels"; core paper unaffected.
6. Instrument-invalid: decoder projection degenerate at smoke (e.g.,
   D0 collapsed by the encoder) AND interventional masks degenerate —
   report as build failure, no claim. (D0-degeneracy alone at smoke =
   registered sub-branch: D0 dropped, flagship rests on the ε-ladder
   discriminator only.)

## 7. Differentiation clause (REGISTERED, named foils — review M-7)

Known and NOT claimed: behavioral noisy-TV attraction (Burda et al.
2019 RND + large-scale curiosity; Schmidhuber's formulation); the
claim that disagreement handles stochasticity is the FOIL, not the
finding (Pathak, Gandhi & Gupta ICML 2019; Sekar et al. 2020
Plan2Explore — the deployed objective's own robustness argument);
aleatoric-aware curiosity measuring noise-channel intrinsic reward
(Mavor-Parker et al. ICML 2022 — nearest prior; our delta = per-key
attribution certified against BY-CONSTRUCTION-zero channels incl. an
exact duplicate, inside the deployed objective's own accounting);
stochasticity-robust curiosity at scale (Jarrett et al. 2023);
ensemble-disagreement ≠ calibrated epistemic uncertainty in principle
(Bengs/Hüllermeier line — ours is the certified in-objective exhibit).
Novel content claimed ONLY as: (i) redundancy/duplicate farming (D0 +
ladder), (ii) per-key model-internal attribution certified against
by-construction-zero channels, (iii) planner-ranking concentration
(P-SE2). **Pre-freeze registered action: one scoped search, 2024–26
"disagreement/curiosity exploration + redundant or distractor
observation channels", to certify the D-niche is still open.**

## 8. Instruments to build (inline; review fixes folded)

`embodied/envs/planted.py` (Duplicate ε-ladder + Constant + gate;
construction selfchecks §2); `uncfield/se_probe.py` (decoder-projected
per-key disagreement + interventional masks + P-SE1/P-SE2 reads +
fixture battery incl. a share-statistic mutant killed by the
permutation null); `scripts/uncfield_se.sbatch` (8 runs + probe passes
+ manifest collect). Smoke = 1 planted run, reduced steps → fills all
[FILL] → freeze commit → full submission.

## 9. Compute

Stage 1: 8 × size1m [FILL walltime]. Conditional D-only: +4.
Stage 2: +8. All RCC.

## 10. Standing-rule compliance

Permutation-primary pinned (§5, key-label/rollout-label schemes); BCa
reporting-only; fit counters verified in the read; results-sync v2
manifests; never-delete-without-archive; arms never cross-seed-paired;
frozen reader executes the read in the Paper-5 chat.
