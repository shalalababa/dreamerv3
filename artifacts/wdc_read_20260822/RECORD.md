# P2 WDC — diverse-candidate variant, ONE read (22 Aug 2026)

Registration: PREREG_p2_dcand_20260821.md (frozen 8828ab6f). Bundle:
wdc_20260822_171431 (manifest verified 0 failed; 16/16 cells + 2/2
registered dup-gate passes; all reader gates passed — env_seed
20260822, R=8/S=200/M=8 pins, dcands pinned-stream recomputation,
candidate-0 CRN witness, duplicate-null exact zero).

## Verdict: **FLATNESS-IS-POLICY**

| quantity | value |
|---|---|
| ceiling (policy set P, 8 policy candidates) | 0.117 [0.000, 0.326] · noise floor 0.011 |
| ceiling (diverse set D, P + 4 uniform) | **0.710 [0.310, 1.259]** · noise floor 0.015 |
| Δceiling (paired D − P, PRIMARY, n=16) | **+0.593 [+0.265, +1.167], perm p = .0037** |
| registered BAR (net ceiling_D must exceed) | 0.20 — exceeded ~3.5× |
| split-selected opp vs candidate-0 (secondary, descriptive) | +0.075 [−0.011, +0.171], p = .152 |

## What this settles (registered consequences, verbatim)

- **The substrate holds resolvable decision value that the policy
  generator hides.** Adding just 4 uniform candidates per state raises
  the per-state value ceiling from ≈noise (0.117, CI touching 0) to
  0.710 — well above the 0.20 substrate bar, net of a matched noise
  floor.
- Registered consequence: **the flatness verdict is re-scoped to
  (environment × trained policy × candidate generator) in every
  draft**, and the environments lens's "empty room" reading is
  RETIRED. Flatness was a property of where a converged actor samples,
  not of the environment.
- The descriptive split-selected opportunity (vs the candidate-0
  baseline) does not move significantly — the extra ceiling is about
  the value RANGE candidates span, not (yet) about what the frozen
  selection chain harvests from it; harvesting under diversity is a
  follow-up question, not licensed here.
- Read alongside Stage B (same day, IN-VIVO-CALIBRATED): the estimator
  chain is proven able to see real value at slope 1, and WDC shows
  real value exists off the policy's candidate manifold. The Paper-2
  story gains a constructive leg: the instrument works, the value is
  there, the converged policy neither samples nor harvests it.
