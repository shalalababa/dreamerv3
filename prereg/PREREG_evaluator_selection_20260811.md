# PREREG: evaluator-selection follow-ups (ensemble repair + laws) — 2026-08-11

User GO 2026-08-11. Follow-ups to the Goodhart partial read
(`artifacts/goodhart_partial_read_20260811/`: LOTTERY-REPLICATES,
ρ=+0.165 p=.124, top-1 regret 87.4% — E2 sole evaluator, 88-policy
within-family pool). Three legs, one pool, one reader.

## Honesty block

Value-aware where stated: the E2 scores and the pool's real returns are
read data (leg 3a resamples them; disclosed). Unknown: every score by
any NEW evaluator (legs 1–2 primary quantities) and the action-distance
covariate (leg 3b).

## Pool and provenance (unchanged)

The 88-policy pool of `PREREG_goodhart_partial_20260811` + Amendment 1
(pinned sidecar `artifacts/goodhart_partial_reg_20260811/pool_ids.json`;
loadable = the 88 s×w_r policies; within-family scope carried verbatim —
cross-class wording stays unlicensed). Gates identical: pinned-id
identity, real ≡ archived final10 (1e-6), pool floor 88.

## Leg 1 — 16-evaluator ensemble (the repair test)

- **Evaluators**: the 16 restored ORIGINAL-weight finger q1 task fits
  (`ax1wm_finger_q1s{0,1}_seed{1..8}` — bitwise originals per the 10-Aug
  census; heterogeneous across sides/seeds, disclosed). E2
  (= `q1s1_seed1`) reuses its EXECUTED 88 score jsons verbatim
  (same registered protocol; reuse disclosed — scores are values, no
  bitwise pairing involved); the other 15 evaluators run fresh:
  **15 × 88 = 1320 scoring passes**, protocol verbatim from the
  Goodhart registration (20 starts × burn 16 × horizon 64, seed 0,
  `INIT_REPLAY=$RUNROOT/pilot_goal_finger_seed1/replay`), outputs
  `$RUNROOT/goodhart_partial/EV_<fit_name>/eval_<policy_id>.json` +
  one collate per evaluator (same collate command).
- **Registered estimators** (reader recomputes from rows):
  per-evaluator z-scored `m_return` (z within evaluator, removing scale
  differences); **ensemble score = mean over the 16 z-scores**;
  penalized score = mean_z − 1.0·std_z (λ=1 primary; 0.5/2
  descriptive). Spearman ρ vs real + label-permutation p (B=10K rng 0)
  + top-k regret + the exact random-pick mean regret, per the Goodhart
  reader's machinery.
- **Rules (per score variant; ensemble-mean = E-R1 primary, penalized =
  E-R2 secondary)**:
  - **ENSEMBLE-RESCUES**: p < .05 ∧ ρ ≥ 0.4 ∧ top-1 regret frac ≤
    0.5 × random-pick mean frac.
  - **ENSEMBLE-IMPROVES**: p < .05 ∧ ρ > the MAXIMUM single-evaluator ρ
    (the 16 singles are computed as a descriptive panel), but RESCUES
    fails.
  - **ENSEMBLE-FLAT**: otherwise. (Grading order: RESCUES > IMPROVES >
    FLAT.)
- Consequence wording (review M11 corrected): RESCUES ⇒ the lottery
  is a single-model phenomenon, ensembling is the constructive lever
  (coheres with the Paper-5 penalty theme; #28 gains a validated
  instrument candidate). **FLAT licenses only the literal fact**:
  'the 16-evaluator ensemble did not clear the max-single bar; no
  impossibility claim beyond "ensembling did not rescue selection on
  this pool"' — the IMPROVES bar (ρ_ens > max of 16 noisy singles)
  is deliberately conservative FOR the positive claim (E[max ρ̂] > ρ)
  and must not be converted into negative evidence. The singles MEAN
  is recorded alongside the max in every graded output. Honesty
  note: E2's executed ρ (+0.165) is KNOWN at freeze and enters both
  the IMPROVES bar and 1/16 of the ensemble score (disclosed).

## Leg 2 — singles heterogeneity panel (descriptive, no verdict)

The 16 single-evaluator ρ's and top-1 regrets: spread, side/seed
structure. Context for E-R1's IMPROVES bar; licenses no wording.

## Leg 3 — laws (registered analyses)

- **3a (selection-pressure curve; VALUE-AWARE, descriptive only — no
  verdict)**: on the EXECUTED E2 scores: E[top-1 regret frac] vs pool
  size n ∈ {5, 10, 20, 40, 88} by uniform subsampling (B=2000 rng 0),
  with the matching random-pick baseline per n. Output = the curve,
  disclosed as post-read descriptive.
- **3b (distribution-shift error law)**: covariate = action-distribution
  distance dist(p) = ||(μ_a, σ_a)_policy − (μ_a, σ_a)_evaluator||₂,
  from per-policy and per-evaluator replay action stats (producer
  command registered below; runs cluster-side; if the stats cannot be
  produced, 3b is SKIPPED-disclosed, never improvised). Outcome =
  per-policy rank discrepancy |rank(m_return) − rank(real)| under E2.
  **Rule (review M10 wording)**: Spearman ρ(discrepancy, dist) > 0 ∧
  two-sided permutation p < .05 (B=10K rng 0; no CI is computed for
  this leg) ⇒ **SHIFT-LAW-SUPPORTED** (evaluation validity decays
  with policy divergence — the OPE-facing finding); else NULL
  (reported). Missing/partial stats ⇒ leg SKIPPED-disclosed; legs
  1–2 are never aborted by a leg-3 input problem (review M9).
  Producer (registered):
  `python - <<'EOF'` one-pass over each policy's `replay/*.npz` and the
  evaluator's `replay` computing per-dim action mean/std, dumped to
  `action_stats.json {name: [mu..., sigma...]}` — exact script shipped
  in the freeze-commit at
  `artifacts/evaluator_selection_reg_20260811/make_action_stats.py`.

Reader: `analysis/evaluator_selection_read.py`, frozen with this file,
selfcheck before any new score exists. ONE read execution (runs when
the 15-evaluator bundle + optional action stats land). Reviewer:
2026-08-11 batch review (ONE reviewer, user-approved) BEFORE
freeze-commit.
