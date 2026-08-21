# PREREG — Listen-or-Leap exact-referee testbed (21 Aug 2026)

**Status: FROZEN at user commit. ONE execution of `--run`.** Fence:
**Paper-2 appendix** (ThreeAxes item 11 — "the one stop-lift worth
asking for on Paper 2's behalf"; a small tabular POMDP whose belief MDP
is EXACTLY solvable, so the estimator chain can be graded against
ground-truth EVSI in an environment that genuinely holds value of
information). Registration-lite is NOT taken: the full estimand set and
branch map are frozen here, because even a testbed read can flatter the
chain if its verdict rule is chosen after the numbers. Reviewer: ONE
Opus pass (shared with the HiddenStakes/testbed review), user
pre-approved, triggered at build.

## Design (all pinned in `analysis/listen_or_leap.py`, selfcheck PASS)

Tiger-style POMDP (hidden z ∈ {L,R}; LISTEN cost c / accuracy q;
OPEN-L/R pay +10 / −20; horizon 6; unopened pays 0 — listening at the
last step is the walk-away action). Exact finite-horizon DP over
(steps_left, tally) — verified in the selfcheck against an
**independent history-space recursion** (unnormalized joint weights;
no shared belief/posterior helpers — review finding 3; the reviewer
additionally reproduced 0 mismatches at 1e-9 over 25 (q,c) pairs).
Grid: q ∈ {.6,.75,.9} × c ∈ {.05,.3,1.0} × 4 seeds = 36 cells; 200
states/cell, R=8 CRN repeats, M=3 candidates; run seed **20260827**.

**Seed-rotation disclosure (review finding 1)**: the original seed
20260821's state prefix was pre-read by the selfcheck's mini-run
(which labels an exact prefix of the registered states; at that seed
the 40-state prefix showed verdict CHAIN-MISCALIBRATED, slope 1.011
[0.825, 1.178], bias +0.408 — recorded here as non-registered
value-aware compute). The run seed is therefore ROTATED to 20260827,
whose states are disjoint from everything computed pre-freeze, and the
selfcheck mini-run is pinned to a further-disjoint seed (RUN_SEED+1)
with no verdict assertion.

The chain under test replays the program's estimator shape verbatim:
split selection on even repeats, evaluation on odd, VoI-blind plug-in
consumer (reduces to myopic-best-open; at k=0 the two opens tie and
argmax picks OPEN-L by index — disclosed conventions), optimal-policy
followers. Labeled states are the optimal policy's pre-open visitation
(depths 0..4, so steps_left ≥ 2; a deliberately VoI-rich state set —
the calibration claim is conditional on this distribution, disclosed).

## Estimands and decision rules

- Substrate gate: mean exact EVSI at (q=.9, c=.05) must be > 0
  (else the env is miswired — refusal, not a finding). Degenerate
  referee spread or a degenerate bootstrap REFUSES (wiring, not a
  finding — review finding 7).
- **PRIMARY — DECISIVE statistic (review finding 4)**: cell-level
  recovery slope of rec on exact, **percentile bootstrap over the 36
  cells, B=4000** (measured unbiased: 0.9995 ± 0.0081 over 60 null
  replicates; coverage of 1 ≈ 88–92%, disclosed).
  - **CHAIN-TRACKS-REFEREE**: slope CI excludes 0 AND covers 1.
  - **CHAIN-MISCALIBRATED**: slope CI excludes 0, misses 1 ⇒ the
    measured slope is carried as a correction bound in the appendix.
  - **CHAIN-BLIND**: slope CI includes 0 on a positive-EVSI substrate.
- **Bias diagnostic (DEMOTED from the conjunction — review finding
  2)**: rec − exact is mean-zero by construction (the odd half is
  independent of selection), so it is a split/CRN-integrity diagnostic
  with NO adjudication weight. Registered null operating
  characteristics (measured pre-freeze at non-registered seeds, 60
  replicates): BCa-CI-excludes-0 fires 15% (nominal 5%), perm 11.7%;
  the retired three-leg conjunction would have branded a perfectly
  calibrated chain CHAIN-MISCALIBRATED ~27% of the time. The
  diagnostic is reported with this table cited.
- EVSI-by-config table reported descriptively (the dose–response of
  true VoI in q and c).

## Ops

CPU-only, minutes, local. Output path is CODE-PINNED to
`artifacts/p2_lol_20260821/` (the ONE-read guard is path-pinned so a
fresh --output cannot defeat it); `lol_read.json` + RECORD.md authored
at read. **[ME] ONE execution** after freeze.
