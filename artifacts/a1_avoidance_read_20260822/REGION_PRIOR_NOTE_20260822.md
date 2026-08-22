# Descriptive note — in-region disagreement contrast from the A1 region
probe (zero-compute; Fable-memo item "pull the in-region
distractor-attribution numbers"), 22 Aug 2026

**Status: DESCRIPTIVE, design-stage prior for Wave C
(PREREG_planner_localized_20260822). NOT part of any registered read;
computed from the already-executed A1 read's stored per-run region
blocks (`se_avoid_read.json` → `per_run[].region`), no new compute, no
new data. Wave C's registration already conditions on the A1 outcome,
so using A1's own region numbers to state the prior adds no further
outcome-dependence beyond what the prereg discloses.**

The A1 region probe recorded, per checkpoint, the mean intrinsic
(disagreement) reward inside the pinned gate region (`dis_high`) vs
outside (`dis_low`). Ratio = dis_high/dis_low:

| arm | seeds | dis_ratio | mean |
|---|---|---|---|
| hetero (scarecrow field) | 44–47 | 1.194, 1.749, 1.567, 1.636 | 1.536 |
| flat (amplitude-matched) | 48–51 | 0.813, 0.808, 0.833, 0.811 | 0.816 |

- **Complete separation, 4/4 vs 4/4** (every hetero ratio > every flat
  ratio); exact C(8,4) permutation on the interaction: observed +0.720,
  p = 1/70 = .0143 (floor). Flat arm tightly clustered (0.808–0.833).
- Reading: in hetero checkpoints the disagreement field is ELEVATED in
  the gate region (the high-amplitude zone the policy learned to
  avoid), while in flat checkpoints it is mildly depressed there. The
  spatially localized misprice signal Wave C's cem_disag arm must steer
  toward **exists in the frozen WMs at ~1.5× out-region level** — the
  "signal present" precondition of the Wave-C interaction is supported
  by existing data, so a Wave-C null would bear on the OPTIMIZER
  (CEM cannot exploit a signal that is demonstrably there), not on
  signal absence.
- Caveat (scope): this is the region-binned disagreement LEVEL (all
  channels), not a per-channel distractor attribution — the A1 probe
  did not store channel-resolved attribution by region. It is a prior
  about spatial structure, not a substitute for Wave C's registered
  interaction on attr_d.
