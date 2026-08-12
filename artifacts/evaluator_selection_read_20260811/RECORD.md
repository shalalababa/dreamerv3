# READ RECORD — evaluator-selection follow-ups (ensemble + laws) — 2026-08-11

- Registration: `prereg/PREREG_evaluator_selection_20260811.md` (frozen
  e1ffe5ab) + Amendment 1
  (`prereg/PREREG_evaluator_selection_amend1_20260811.md`, leg-3b
  producer repair — evaluator buffer path + flat-dir fallback; covariate
  producer only).
- Inputs: bundle `evaluator_selection_20260811_192338` (sha-manifest
  verified, 0 non-OK): 15 fresh evaluator collates
  (`EV_ax1wm_finger_q1s{0,1}_seed*/pool_metrics.json`, 88 eval jsons
  each) + **E2 = the EXECUTED
  `artifacts/goodhart_partial_read_20260811/E2_pool_metrics.json`
  (registered reuse — the bundle's E2/ dir carries raw score jsons
  only, no new collate consumed)** + `action_stats.json` (Amendment-1
  producer). Pool: pinned sidecar (88 loadable, Goodhart Amendment-1
  scope). Reader: `analysis/evaluator_selection_read.py` (frozen
  e1ffe5ab), selfcheck PASS pre-execution. ONE execution.

## Verdicts (registered rules verbatim)

| Leg | Result | Verdict |
|---|---|---|
| **E-R1** ensemble mean (16 evaluators, z-scored) | ρ = **+0.347**, perm p = **.0009**; top-1 regret **13.4%** (random-pick mean 78.7%) | **ENSEMBLE-FLAT** (ρ < 0.4 bar; ρ < max-single 0.493) |
| **E-R2** penalized (mean_z − 1.0·std_z) | ρ = +0.347, p = .0009 (λ 0.5/2 ≈ same) | **ENSEMBLE-FLAT** |
| **Leg 3b** shift law (rank discrepancy vs action-distance, under E2) | ρ = −0.086, p = .431 | **NULL** |

Licensed FLAT wording (registered, M11 form): the 16-evaluator
ensemble did not clear the max-single bar; no impossibility claim
beyond "ensembling did not rescue selection on this pool." Singles
mean +0.211 recorded alongside max +0.493 per the rule.

## Registered descriptive outputs

- **Leg 2 singles panel (no verdict licensed)**: extreme evaluator
  heterogeneity — `q1s1_seed5` ρ +0.493 / top-1 regret 1.7%;
  `q1s1_seed4` ρ +0.419 / top-1 regret **0.0%**; then a cliff
  (`q1s1_seed3` +0.332 but 87% regret; `q1s0_seed6` +0.251 with 100%
  regret). E2 (`q1s1_seed1`, the executed evaluator) sits mid-pack at
  +0.165 / 87.4%.
- **Leg 3a pressure curve (VALUE-AWARE descriptive, executed E2
  scores)**: E[top-1 regret frac] RISES with pool size — 45.3% (n=5) →
  57.6% (10) → 69.5% (20) → 78.5% (40) → 87.4% (88), tracking at/above
  the random-pick baseline at each n — selection pressure monotonically
  worsens E2's pick.

## Post-read observations (labeled, no registered weight)

- The ensemble misses RESCUES on the ρ ≥ 0.4 bar by 0.053 while
  clearing its other two conjuncts (p=.0009; regret 13.4% ≤ 0.5×78.7%)
  — a near-miss worth noting alongside the licensed FLAT.
- Practical contrast recorded in registered outputs: top-1 regret
  87.4% (E2 alone, executed) vs 13.4% (ensemble) vs 1.7%/0.0% (best
  two singles) — evaluator IDENTITY dominates; the "lottery" behaves
  like a property of unlucky single evaluators rather than of
  WM-evaluation per se. Any claim-grade version of this needs its own
  registration (candidate: prospective evaluator-selection rule tested
  on a fresh pool — #28 scope).
- Leg-3b's null says the s×w_r pool's rank errors are not explained by
  action-distribution distance to the evaluator's training buffer (or
  the (μ,σ) covariate is too coarse).
