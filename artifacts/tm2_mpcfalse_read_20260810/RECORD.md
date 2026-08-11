# TM2 mpc:False read (#24 planner de-confound) — 2026-08-10

Registration: `prereg/PREREG_tm2_mpcfalse_20260808.md` (frozen 1a36a65a,
08-08). Reader: `analysis/review_waves_read.py tm2mpc` (frozen same
commit; selfcheck PASS pre-execution). Bundle
`tm2_mpcfalse_20260809_152905` (sha-verified). ONE execution;
`read.json` unedited. 64/64 adapt rows, no sub-modal exclusions
(exclusions.csv rows are unrelated stale runs).

## Verdict: **P-MF1 NULL — the family boundary survives the readout de-confound**

- **P-MF1** interaction [B_aware − B_free] at mpc:False, n=16 seed-paired:
  **+2.05** [−32.4, +35.5], exact sign-flip perm p = .910 (7/16 positive).
- **P-MF2** aware simple at mpc:False: **+0.70** [−28.3, +31.5], p = .966.

Registered consequence: the "reconstruction-based WMs" family-scope
wording STANDS; the readout-algorithm confound is retired (one of the
seven D21 coincident differences eliminated as the explanation).

**[POST-READ observation, descriptive]**: at mpc:True (F1 archived) the
aware simple was +35.3 CI>0; at mpc:False it is +0.7 (p=.97). The
adaptation value TD-MPC2's aware trunk provided was expressed through the
MPPI planner, not the learned policy prior — mechanism color for the
boundary discussion (the registered mpc-False/True level-ratio
descriptive was not emitted by the frozen reader; noted, not computed
post-hoc here).
