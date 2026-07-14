# Evaluator-Goodhart sprint registration (frozen 2026-07-14)

Registers the first empirical pass of the "Goodhart's Simulator" program
(innovation audit 13 Jul; new study, exploratory by design — this file
fixes the pool, evaluator, metrics, and the advance/demote criterion
before any pool-level number exists).

## Design (adaptivity levels 1–2 of the audit's ladder)

- **Evaluator M:** one reward-aware learned world model + reward/continue
  heads. Canonical first choice: `adapt_ax1q1s1_finger_seed1_ckpt500000`
  (task-mode adapt on the 10-Jul arm). The evaluator's own run is
  excluded from the pool.
- **Policy pool (level 1 — variants never used to train M):** every
  finger adapt checkpoint under the runroot with a completed
  `scores.jsonl` (all modes/sides/seeds; ≈100+ runs). Real return =
  mean of the last 10 eval episodes (already measured by the frozen
  adapt protocol; no new environment interaction).
- **M-score of a policy π:** mean imagined return over 20 fixed
  real-prefix starts (burn-in 16, horizon 64, seed 0), π acting through
  the observation interface (M decodes obs, π filters with its own
  RSSM), continue-head termination at 0.5
  (`probing/wm_evaluator.py score`; mechanics validated e2e locally
  2026-07-14 on two W0-snapshot adapt runs, throwaway numbers).
- **Metrics (`collate`):** pool Spearman(M-score, real), pairwise
  inversion rate, top-k regret for k ∈ {1,3,5} (real return foregone by
  trusting M's top-k), selection-vs-correlation contrast = top-1 regret
  compared with what pool-level correlation would predict.

## Registered advance/demote criterion (audit's kill rule, made concrete)

- **Advance** (levels 3–4: evaluator-guided fine-tuning, adversarial
  pairs; then a second domain): selection amplifies error — top-1
  regret ≥ 10% of the best real return DESPITE pool Spearman ≥ 0.6, or
  inversion rate in M's top quintile exceeds the pool-wide rate by ≥ 2×.
- **Continue-but-weaker:** high regret with LOW pool correlation (< 0.6)
  — the evaluator is just bad; try a stronger evaluator (more adapt
  steps / online model) before claiming a Goodhart effect.
- **Demote** (per the audit): selection does not amplify error (top-k
  regret consistent with the pool correlation), or failures are fully
  explained by trivial OOD actions flagged by standard uncertainty.

Level-2 note: selection by M *is* the treatment here; no new training.
All numbers exploratory for the eventual paper; this registration exists
to prevent post-hoc metric shopping, not to support confirmatory
language.

## Ordering

At freeze: no pool-level evaluation exists (only the two-run local
mechanics validation named above); `scripts/goodhart.sbatch` unsubmitted.
