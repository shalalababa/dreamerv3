# AMENDMENT 1 to PREREG_goodhart_partial_20260811 — 2026-08-11

Triggered by the user's execution preflight (2026-08-11), BEFORE any
evaluator score exists — value-blindness intact (the only unknowns
remain the m_return scores; everything below is inventory).

## Findings at preflight

1. **E1 (p2elong online model) is MISSING** — the registered E1-drop
   branch fires: E2 (`ax1wm_finger_q1s1_seed1`, restored original
   weights) is the SOLE evaluator; the single-evaluator disclosure
   attaches to every output, and the "weaker-of-evaluators" rule
   degenerates to E2's verdict (as registered).
2. **18 pinned pool members have no loadable checkpoint**: `p2eof`
   seeds 1–5, `p2eou` seeds 1–5, `rif` seeds 1–8 (run dirs/scores
   survive; the policy checkpoints do not — the rif/p2e adapt path did
   not retain final checkpoints, unlike the axis1 adapt path). The
   loadable pool is the 88 s×w_r policies — below the registered floor
   of 95, so the frozen reader refuses as designed.

## Amended rule (the ONLY change): pool floor 95 → 88

Basis, disclosed: the 88 s×w_r policies alone preserve the registered
ranking substrate — archived final10 spans 0.0–904.5 (sd 181; p25 94.6,
median 163.2, p75 249.4), i.e. the FULL spread of the original 106-pool
(the lost 18 sat mid/low: max 342.3). What is lost is POLICY-CLASS
diversity, not range: no untrained-trunk (`rif`) and no external-method
(`p2eou`/`p2eof`) members remain, so the pool is homogeneous-by-family
(s×w_r task-arm policies, one buffer side, 11 cells). Registered
consequence for wording: any P-GH1/P-GH2 verdict is scoped "on a
within-family policy pool"; cross-class ranking ability (e.g. spotting
an untrained policy) is NOT tested and no such wording is licensed.

Regeneration of the 18 (fresh adapts) is REJECTED for this wave: fresh
policies are new pool members, not recoveries; the class-diversity gain
does not justify re-plumbing checkpoint retention on the rif/p2e path;
and the ranking question is answerable on the 88. A diverse-class pool,
if the prospective-curation thread wants one, is a future registration.

## Mechanics

- The 106-id sidecar is UNCHANGED (committed provenance); the 18
  unloadable ids are recorded in `dropped_missing_ckpt` by the existing
  gate machinery.
- Reader change (pre-execution, dated comment): `POOL_FLOOR = 88`;
  selfcheck floor legs re-pinned accordingly. No other rule, threshold,
  or gate changes.
- Scoring: 88 passes under E2 only (half the registered 212), same
  protocol verbatim.
