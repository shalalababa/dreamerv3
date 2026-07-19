# Scaling Option C (volume rider) read — registered direction FAILS descriptively, but the realized design broke occupancy matching (2026-07-18)

Snapshot: `local_results/scaling_option_c_20260718_175258/` (24 runs:
{task, apt} × {q1v200, q1v400} × 6 seeds, side1 only, size1m). The
registered read (`PREREG_scaling_pilot_20260717.md` §Option C) is
DIRECTIONAL prose only — "no volume effect at fixed fraction above the
estimation floor" (P-B4 hook) — no frozen script, no CI-bearing
decision. Numbers below computed with the frozen conventions (AUC100k
from the untouched pipeline, 24/24 QC; B=10K bootstrap seed 0,
robustness only); `optc_read.json` here.

## CENTRAL AUDIT FINDING — the v400 pair's side order came out inverted

| pair | side0 occ | side1 occ | run side |
|---|---|---|---|
| q1v200 | 0.0536 | **0.3234** | side1 (hi — as intended) |
| q1v400 | **0.2899** | 0.0938 | side1 (**LOW** — inverted) |

The plain `search` subcommand does not enforce the side0=low
convention (only `search-rpair` does); v200 happened to come out
low-first, v400 high-first. The registered "volume ladder on the hi
side" was therefore NOT realized: the v400 cells ran on a 400-episode
**occ 0.094** buffer vs v200's 200-episode **occ 0.323** buffer.
Volume is confounded with occupancy(↓ 3.4×) and composition (v400s1
mixture dominated by zero-occ bins of apt/goal/random pilots; both
pairs cov-matched internally, dcov ≤ 0.008; same ref_replay —
`pilot_goal_finger_seed1` — as the original q1, per prereg).

## Descriptive results

| cell | mean AUC100k | sd |
|---|---|---|
| task v200s1 (200 eps, occ .32) | 204.3 | 90.6 |
| task v400s1 (400 eps, occ .09) | **432.9** | 242.5 |
| apt v200s1 | 64.4 | 19.8 |
| apt v400s1 | 86.3 | 17.4 |

Paired within arm (robustness, not decisional): task v400−v200 =
+228.6 [+38.0, +443.4], 4/6 (per-seed spread 88..749); apt +21.8
[+0.2, +44.4], 4/6.

## What can and cannot be concluded

1. **The registered direction fails descriptively** — something about
   the 400-episode buffers strongly improves task-arm transfer — but
   the registered "at fixed fraction" clause was not realized, so this
   is NOT a clean volume effect and P-B4 is NOT adjudicated here.
2. **The anomaly is the interesting part:** the winning cell has
   *fewer* in-regime frames (≈37.6K vs ≈64.7K), a *lower*
   rewarded-frame fraction (.094 vs .323), and *more* total data +
   source diversity — and wins by 2×. Neither the fraction account nor
   the count account predicts that; composition/diversity or total
   data volume is doing heavy lifting. Also revises the occupancy
   bracket: occ .094 evidently sits ABOVE finger's binding threshold
   (previous bracket: null at .054, works at .19+) — at 400 episodes.
3. **Reward-free volume buys ≈ nothing:** apt stays near the floor at
   both volumes (+21.8 marginal) — consistent with every reward-free
   null; whatever volume buys, it flows through the reward-capable
   objective.

## Cheap decisive follow-up (already on disk)

The built `q1v400/side0` (occ .290, 400 eps) was never run. Running
{task, apt} × 6 seeds on it (12 jobs, `ax1v4*q1v400s0`) yields BOTH
missing contrasts at once: the occ-matched volume contrast (v400s0
.29/400eps vs v200s1 .32/200eps) and the within-v400 occupancy
contrast (s0 .29 vs s1 .094 at fixed 400 eps + fixed pool). Needs a
dated amendment registered before those outcomes exist.

## Provenance

`auc.csv` (24 rows, canonical), `optc_read.json` (cells, paired
deltas, realized-pair audit). Pairs provenance in the snapshot's
`axis1_finger/pairs_v{200,400}.json` + build manifest (v200).
