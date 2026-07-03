"""Gate D0 four-signal analysis pipeline (EVPI note App. A).

Modules:
  signals   -- per-state signal math from (N, K, M) ensemble Q matrices;
               single source of truth for real sweeps and synthetic tests.
  analysis  -- aggregation discipline: quadrant occupancy against matching
               dose-zero thresholds, episode-clustered Spearman, reversal
               rates, and the pre-registered P1-P4 evaluation.
  synthetic -- synthetic-belief generator with a controllable dissociation
               dose, for end-to-end pipeline validation before any run.
  selfcheck -- runnable end-to-end check: python -m d0.selfcheck
  sweep     -- offline checkpoint sweep producing the per-state signal table.
"""
