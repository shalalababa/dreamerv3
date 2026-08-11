# W1 adjudication wave read (Paper 2) — 2026-08-11

Registration: `prereg/PREREG_w1_adjudication_constraints_20260809.md` +
`prereg/PREREG_w1_wave_20260809.md` (both frozen 35983241, 08-09, before
any label existed). Reader: `analysis/w1_read.py` (frozen same commit;
selfcheck re-PASS immediately before execution; bundled copy in the
label bundle byte-identical to the committed instrument, as are the two
labelers). Bundle: `w1_labels_20260811_090727` (sha-manifest verified,
162 files; 68/68 labels: 32 dv3 + 32 tm2 + 4 duplicate gates). ONE
execution per family. Duplicate-null gates passed in both families
(n_dup_passes=2 each; any violation would have refused the read).

## dv3 family — **LEAK-WAS-ALL**

- **P-W1a split-selected opportunity: +0.0048** [−0.070, +0.075],
  perm p = .898 — NO FIRE. With independent follower marks and
  split-selection, the archived within-pass "+1.30 opportunity" is
  GONE, exactly as the cross-fit collapse predicted.
- **P-W1b consumer harvest (deployed actor): +0.024** [−0.033, +0.089],
  p = .476 — NO FIRE.
- **P-W1c leak decomposition**: ρ_full −0.010 (p=.41); ρ_value −0.012
  (p=.36); ρ_reward-only +0.138 (p=.037 on only 8 non-degenerate
  cells; CI NaN — cluster bootstrap degenerate at n=8, disclosed;
  below the registered decisive form).
- **Verdict: LEAK-WAS-ALL** — the duplicate-leak witness (ρ=+0.47
  inside identical-action groups) accounted for the entirety of the
  archived dv3 "signal". The strongest registered negative.

## tm2 family — **LEAK-WAS-ALL**, with two disclosed structure facts

- **P-W1a FIRES: +0.062** [+0.022, +0.123], p = .016 — a small but
  genuine split-selected opportunity exists in TM2 (selection on even
  repeats transfers to odd repeats).
- **P-W1b does NOT fire — and is significantly NEGATIVE: −0.301**
  [−0.745, −0.123], p = .008. The warm-started deployed planner's
  action evaluates WORSE than the average candidate branch.
  [POST-READ observation, no wording licensed]: an anti-harvest of the
  deployed consumer is a surprising structure fact — the planner
  actively moves away from what the ground-truth probe rewards —
  worth a mechanism look in the Route-B discussion, own registration
  if pursued.
- **P-W1c**: ρ_full +0.075 (p=.32); reward-only leg n=1
  (uninformative); ρ_value +0.075 (p=.32) ⇒ **Verdict: LEAK-WAS-ALL**.

## Consequence (Paper 2)

The W1 adjudication closes the loop the two reviews opened: on
constraint-hardened, de-confounded labels (independent marks, split
selection, actual consumers, seeded envs), **the opportunity/achieved
dissociation does not survive in either family** — dv3 shows nothing;
tm2 shows a small real max-over-candidates edge that the deployed
consumer not only fails to harvest but significantly anti-harvests,
with no held-out ground-truth correlation. The Route-B framing
(dissociation + instrument crisis) now rests on: construct invalidity
of the original estimand (reviews) + W1's clean negative adjudication +
the duplicate-leak mechanism. NaN CIs on degenerate legs disclosed;
permutation is primary per the standing rule.
