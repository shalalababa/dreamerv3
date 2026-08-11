# READ RECORD — anti-harvest zero-compute stage (TM2 W1 labels) — 2026-08-11

- Registration: `prereg/PREREG_antiharvest_zc_20260811.md` (frozen commit
  e1ffe5ab) + **Amendment 1**
  (`prereg/PREREG_antiharvest_zc_amend1_20260811.md`): the first
  execution attempt crashed at the load-gate identity witness — the
  reader upcast float32 `qfull` to float64 before the head-mean argmax,
  flipping 61/6400 near-tie states vs the labeler's float32
  `plugin_choice`; zero estimand values were output before the crash.
  Fix: selection layer in storage precision; identity then exact
  6400/6400. ONE execution after the amendment = this one.
- Reader: `analysis/antiharvest_zc_read.py` (Amendment-1 form), sha256
  `2d54c7f3266f577c01ec672669f2f0404a56e51ddcef1d674162fad3afb8e898`.
  Selfcheck PASS pre-execution (incl. the new float32 near-tie leg).
- Inputs: 32 non-dup `w1tm2_*.npz` from
  `local_results/w1_labels_20260811_090727/w1_labels/` (bundle
  MANIFEST.sha256
  `278a83aa43d4733b3840c34e968288bcfb4eea3bc4cf9b52a4e7d4d1e1c8d61e`,
  verified at landing). Gates all passed: 32 cells, 2 dup files present
  and skipped, per-file pins (repeats 8, states 200, actions 8,
  env_seed 20260809, `_w1` version, dup_cand all False, shapes,
  NaN-free), sel_q ≡ m_now exact on all 6400 states.

## Verdicts (pooled 32 cells, cluster = cell; BCa 95% B=10K + cluster sign-flip)

| Leg | Estimand | Mean [95% CI] | perm p | Verdict |
|---|---|---|---|---|
| **P-Z1** | d_mode = g(plan) − g(prior mode) | **−0.261 [−0.612, −0.109]** | **.0076** | **PLANNER-BELOW-MODE** |
| **P-Z2** | d_qsel = g(Q-argmax cand) − ḡ(cands) | +0.016 [−0.017, +0.051] | .391 | NULL (uninformative) |
| **P-Z3** | d_rep(λ=1) = g(penalty-sel) − g(Q-sel) | −0.0006 [−0.034, +0.034] | .972 | NULL |

- **P-Z1 fires**: six MPPI iterations produce a first action
  significantly WORSE than the un-optimized prior MODE — optimization
  is strictly harmful at the first step, in both domains (cup −0.309,
  finger −0.213). Implied ordering: mode (≈ +0.04 vs candidate mean) >
  prior samples > planner (−0.301 vs candidate mean, the W1 harvest).
- **P-Z2 value-awareness outcome (against-interest note discharged)**:
  the registered a-priori-expected branch was Q-HARVESTS; it did NOT
  land — the Q-argmax candidate is indistinguishable from the candidate
  mean. Context from the descriptive: pooled per-state Spearman(q, g)
  = **+0.015** (25/32 cells finite; 7 G-degenerate, the known W1
  degeneracy) — the Q-ensemble carries essentially NO ranking signal
  among prior candidates. Neither value-curse nor value-harvest at the
  candidate level.
- **P-Z3**: disagreement penalty moves nothing (λ 0.5/2 descriptives
  also null). The constructive-lever hypothesis gets no support at the
  candidate level.
- Descriptive: d_rep_vs_plan = +0.276 [+0.116, +0.618], p=.0064 — the
  penalty-selected CANDIDATE beats the PLANNER (anti-harvest restated
  from the candidate side; coherent with W1).

## Reading

The anti-harvest is **planner-search-specific**: the deficit appears
between the prior and the planner's output (below even the unoptimized
mode), while candidate-level value selection is flat (P-Z2, P-Z3 null,
ρ(q,g)≈0). Value-error is NOT demonstrated to suffice. This redirects
the mechanism wave's weight to **P-N2 (optimization-pressure dose) and
P-N3 (warm-start inertia)** — exactly the W2 arms — and makes the W2
prior-mode/iters→0 reference a live cross-stage coherence check.
Scope: realization-conditional (one env snapshot per state, W1 m1
disclosure carries), within-family TM2, 32 cells.
