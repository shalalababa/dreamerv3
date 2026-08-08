# PREREG: TD-MPC2 family boundary — planner de-confound (mpc: False) — 2026-08-08

Review-resolution wave #24 (user GO 2026-08-08). The registered family-scope
limitation ("reconstruction-based WMs") labels ONE of at least seven
coincident TD-MPC2/DreamerV3 differences (review D21); the cheapest single
de-confound is the readout algorithm: TD-MPC2 adapts through an MPPI
planner, DreamerV3 through a learned actor-critic. If the interaction
appears without the planner, the boundary is the adaptation mechanism, not
the family. Known at freeze: TM2 F1 n=16 (interaction +22.9 ns, aware
simple +35.5, aware ≈ 2.2–2.5× free levels). Unknown: all mpc-False
outcomes.

## Design — 64 adapt jobs on EXISTING TD-MPC2 fits, no new fits

- Re-adapt the existing **64** TD-MPC2 fit checkpoints ({aware, free} ×
  {s0, s1} × **16 fit seeds** — CORRECTED per batch review M6; the n=16
  seed-paired primary requires all four cells × 16, and the reader
  asserts exactly 16 pairs) with `probing.tdmpc2_adapt --mpc False`
  (flag added 2026-08-08; sets `cfg.mpc = False` so action selection is
  the learned policy prior, matching DreamerV3's readout class). All
  other knobs identical to the F1 wave (125K steps, eval cadence,
  seed_steps 5000).
- DONOR PREFLIGHT (before freeze): the 64 `tm2wm_*` fit dirs are
  cluster-only (local copies weight-free) under a DELETE-AFTER glob —
  confirm survival; new wave dirs `adapt_tm2mf*` are KEEP-PENDING
  protected (cleanup edit in the same commit; the pre-existing
  DELETE-AFTER `adapt_tm2*` glob would otherwise match them).
- SMOKE GATE: one job; assert the run's cfg dump shows `mpc: false`
  (the dump now records `cfg.mpc`, not a literal — batch-review B2),
  scores flow, and the checkpoint loaded (same asserts as the F1 smoke).
- Disclosed residual confounds: parameters, horizon, symlog, seed_steps,
  eval-episode logging — this wave isolates the READOUT axis only.

## Registered read (`analysis/review_waves_read.py tm2mpc`, frozen with this file; selfcheck PASS)

- **P-MF1 (primary): interaction [B_aware − B_free] at mpc False**, n=16
  seed-paired, exact sign-flip permutation primary + bootstrap CI.
  - Fires positive ⇒ the interaction exists in the TD-MPC2 family once
    the planner is removed ⇒ **family boundary re-attributed to the
    adaptation mechanism**; the "reconstruction-based" scope wording is
    replaced by "learned-readout" scope (registered consequence — a
    WORDING change to the boundary, no Paper-1 primary moves).
  - Null ⇒ the boundary survives the readout de-confound; the family
    wording stands, one confound retired.
- **P-MF2 (secondary): aware simple** at mpc False (n=16), same machinery.
- Descriptive: mpc-False vs mpc-True level ratio per arm (planner value),
  aware/free level ratio.
- QC: standard AUC reader + modal-n_ep rule; scores are TD-MPC2 eval
  episodes both arms (within-family comparisons only — the D21
  logging-asymmetry caveat bars cross-family level statements).

Freeze ordering: commit this file + reader before the smoke job.
