# Scaling pilot registration — Option B, size12m (draft 2026-07-17; freeze after timing smoke)

Registers the READ-2-authorized scaling pilot (execution plan
`research_notes/Plan_ScalingPilot_Exec_20260717.md`; scope = Option B
capacity primary per `Scaling_Pilot_Design_Brief_20260716.tex`, framed
as pilot-for-the-flagship, NOT a scaling-law study) **before any
size12m outcome exists**. Theory predictions tested: **P-B5**
(capacity buys representation, not compression) and **P-B6-lite**
(low-support side stays null at 12m), plus the primary three-way
(`prereg/PREREG_theory_predictions_20260717.md`).

FILL AT FREEZE (from the timing smoke; the smoke is protocol
validation, its run dirs are deleted and carry no evidentiary weight):

- measured 12m fit upd/s: ____ ; chosen AXIS1_BUNDLE_TIME: ____
- update count: 500000 unless walltime forces a registered reduction —
  if reduced, BOTH 12m arms use the SAME count (gradient-equalized
  within the 12m stratum; the three-way survives because the 1m
  stratum is internally equalized at 500K) and the number is recorded
  here: ____
- reward-direction variance-rank prior
  (`probing/reward_direction_rank.py --side .../axis1_finger/q1/side1`,
  stated prior only, no decision role): rank = ____ , linear R² = ____

## Design

Finger q1 pair, seeds 1–8 paired across all four cells; 12m fits
mirror the existing 1m protocol (offline fit then frozen-readout
adapt, `AXIS1_SIZE=size12m` applied to BOTH stages — a size mismatch
fails checkpoint loading, which is the built-in audit guard).

**Buffer for the 12m cells (registered protocol difference, found by
the 2026-07-18 timing smoke):** the frozen q1 chunks byte-preserve the
source runs' `dyn/*` replay-context latents at size1m dims (512/32×4),
which cannot feed a size12m agent (`_apply_replay_context` rebuilds
the RSSM carry from them — the smoke failed at the compiled-shape
check, as designed). The 12m cells therefore fit
`axis1_finger/q1_ctx12`: a copy with `dyn/*` ZERO-INITIALIZED at
2048/32×16 (`probing/resize_replay_context.py`, selfcheck PASS;
obs/action/reward/flags byte-preserved; stepid re-encoded so
Replay.update refreshes the context with the 12m agent's own latents
after the first pass — the same contract every built buffer already
relies on). Submitted via `AXIS1_QUAD_SUFFIX=_ctx12`, which changes
ONLY the replay path — run/WM naming stays as registered
(DRYRUN-verified). Disclosure: the historical 1m fits started from the
source agents' stored latents, the 12m fits start from zeros; the
difference is stratum-internal (both 12m arms share the same resized
pair, both 1m arms shared the original), so the three-way contrast
differences it out; noted as a scope caveat on any cross-stratum LEVEL
comparison (which is descriptive-only here anyway).

| cell | mode string | jobs | source |
|---|---|---|---|
| task@12m | `ax1s12q1s{0,1}` | 16 | new (`AXIS1_ID_PREFIX=ax1s12`, `AXIS1_EXPL_MODE=task`) |
| apt@12m | `ax1fs12q1s{0,1}` | 16 | new (`AXIS1_ID_PREFIX=ax1fs12`, `AXIS1_EXPL_MODE=apt`) |
| task@1m | `ax1q1s{0,1}` | — | historical (10-Jul read, bit-checked) |
| apt@1m | `ax1fq1s{0,1}` | — | historical (P0 corrective, bit-checked) |

size12m = deter 2048 / hidden 256 / classes 16 / depth 16 / units 256
vs size1m 512/64/4/4/64 (configs.yaml). WM run names
`ax1wm_finger_{s12|fs12}q1s<side>_seed<k>` (note: `wm_infix` strips
`ax1`, so the apt arm is `fs12…` — E4 globs must use
`ax1wm_finger_*s12*` to catch both arms).

## Outcomes and decision rule

Endpoints from the untouched `analysis/adaptation_auc.py` (AUC100k
inferential; final10 = near-asymptotic endpoint, directional only).
B(k) = endpoint(s1) − endpoint(s0) within seed k. CIs:
cluster-bootstrap percentile 95% (B=10,000, default_rng seed 0);
decisions on the CI alone; sensitivity suite robustness-only. Read
script frozen pre-outcome: `analysis/scaling_read.py` (selfcheck PASS;
bit-checks the 1m strata against the original paired JSONs at read
time).

- **PRIMARY (sole confirmatory, AUC100k): mean over seeds 1–8 of
  [B_task@12m − B_apt@12m] − [B_task@1m − B_apt@1m].**
  - CI < 0 ⇒ supervision effect shrinks with capacity — substitution
    begins (membership account); parameterizes the 9b ladder.
  - CI includes 0 ⇒ capacity does not substitute at 12× — compression
    account unchallenged; the interaction is scale-robust (Paper-1
    strengthens).
  - CI > 0 ⇒ supervision effect grows — outside both registered
    accounts; theory revision before any follow-up design.
- **S1 (P-B5 dissociation, registered directional, no decision):** the
  same three-way on final10; prediction = shrinkage larger (more
  negative) on final10 than on AUC100k. The paired difference
  [three-way(final10) − three-way(AUC100k)] is reported with CI.
- **S2 (P-B6-lite):** B_apt@12m simple effect; prediction = CI includes
  0 on both descriptive sides (reward-free transfer stays null at 12m;
  capacity does not substitute for absent support).
- Descriptive: all 8 cell means on both endpoints; level shifts (bigger
  models may lift all arms — the contrasts carry the inference);
  interaction-at-12m and interaction-at-1m contrasts.
- **Registered caveat:** the 1m task stratum seeds 1–8 is the known
  optimistic draw (+159 vs honest pooled +98.7, W0). Robustness
  variant with pooled-1–16 1m strata reported alongside, never
  decision-bearing.
- Audit (per-run): saved WM config `dyn.rssm.deter == 2048` (size
  audit) + `expl.mode` per arm; ADAPT_DONE; QC ≥20 eps ≤100K unchanged;
  fit stdout (where retained) shows `Static replay: .../q1_ctx12/side<s>`.
- E4 measure pass over the 12m fits (descriptive membership readout:
  does apt@12m reward-NLL move toward task@12m?) uses the frozen
  `finger_v1` probe set, glob `ax1wm_finger_*s12*`; no probe-set
  change.

## Option C rider (buffer volume; submit after B is queued)

Volume ladder at size1m, side1, task+apt: episodes {200, 400} ×
{task, apt} × 6 seeds = 24 jobs (`ax1v2*/ax1v4*` prefixes). Episode
count is set at the **search** stage (`build_controlled_replay search
--n_episodes` writes a new pairs file per volume level), then
`build --which q1` against it; the search must reuse the original q1's
`--ref_replay` (confirm from `_submit_runlists/` or the pairs.json
provenance) so occupancy stays matched. Verify ≥400 qualifying
episodes first; if short, degrade the ladder and record realized
counts here. Theory hook: P-B4 (fraction-not-count). Contrast: within
size1m/side1, task and apt AUC100k vs episode count — registered
directional (no volume effect at fixed fraction above the estimation
floor); details finalized in a dated amendment before the rider
submits if any of this section changes.

## Accounting, risks, scope

- 32 GPU jobs Option B (AXIS1_BUNDLE_SIZE=1) + 24 rider + 1 deleted
  smoke; concurrent with Amendment-1 (32) and stamping (48) — within
  policy.
- OOM / walltime risks are what the smoke measures; the update-count
  fallback rule is registered above.
- The 25m third point is explicitly NOT part of this registration; any
  25m decision is a new dated registration parameterized by this read.

## Disclosure and ordering

Known at freeze: every outcome through 17 Jul 2026 (P0/W0/W1/W2/W3, P3
+ E4, battery, Goodhart, TD-MPC2 read, Phase-5a) — in particular both
1m strata entering the three-way are fully known; the three-way is
prospective only through the unknown 12m cells. Unknown: every 12m
number (no 12m fit or adapt has ever run; the timing smoke uses seed
99, its dirs are deleted, and it is excluded by rule from every
analysis). Ordering: AXIS1_SIZE plumbing, `probing/resize_replay_context.py`, the
AXIS1_QUAD_SUFFIX hook, `analysis/scaling_read.py`,
`probing/reward_direction_rank.py`, and this file are committed
together BEFORE the Option B wave is submitted; the smoke may precede
the commit (it produces no outcome-relevant information beyond
walltime/memory/shape feasibility, disclosed above — the first smoke
attempt failed at the context-shape check with zero training progress
and is what motivated the resize protocol).
