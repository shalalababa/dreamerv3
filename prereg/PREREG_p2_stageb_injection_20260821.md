# PREREG — Paper-2 Stage B: in-vivo certified injection (21 Aug 2026)

**Status: FROZEN at user commit. ONE read execution.** Motivation:
`Rescue_Ideation_ThreeAxes_20260821.md` item 8 — the in-vivo positive
control: Stage A (Tier-1 bundle) calibrates the estimator chain on
archived arrays; Stage B plants a KNOWN reward signal into the LIVE env
so the calibration covers the real probe/selection path end-to-end. The
designed dissociation is **probe-harvests / consumer-does-not** (the
frozen critic and policy know nothing of the stamp). Reviewer: ONE Opus
pass, user pre-approved, triggered at build.

## Design — 8 existing dv3 R3 LATE cells, stamped env, no new training

- Instrument: `embodied/envs/rewardstamp.py` (NEW; selfcheck PASS) —
  ARM-ON-RESTORE one-step stamp: every labeler branch begins with
  `restore_env`, whose `oracle_set_state` call arms the wrapper;
  exactly the next non-reset step's reward gains
  **delta · clip(a[0], −1, 1)**, then it disarms. Consequences by
  construction: each candidate branch's G and the probe's r_real carry
  planted(s,m) = delta·clip(cands[s,m,0]) exactly once (EXACT referee,
  computable from stored cands); follower steps are never stamped;
  Vhat comes from the stamp-ignorant frozen critic. Reward does not
  enter physics and the policy is frozen, so state visitation is
  bit-identical to an unstamped pass; the base trajectory's
  restore-resume step is also stamped but its reward is never stored
  (disclosed). **Verified code fact (review 4.1)**: the dv3 encoder
  EXCLUDES the reward key (`dreamerv3/agent.py:39` exclude tuple), so
  follower beliefs cannot see the stamp and "state visitation
  bit-identical" is a code fact, not an argument.
- **Exact stamp-count identity (review B1, registered gate)**: per
  labeled state the protocol stamps exactly 2 determinism probes + M
  op_real steps + R·(M+1) branch first-steps + 1 base resume, so
  stamped_steps must equal S·(3 + M + R·(M+1)) = 200·83 = **16600**
  per pass — an end-to-end wiring proof; the reader refuses any other
  value.
- **delta = 2.0** (pinned). Cells: cup/finger × e1/e4 × **seeds 31–32**
  (8), dv3 R3 LATE checkpoints (RCC-confirmed). Dials: R=8, S=200,
  horizon 100, `--env_seed 20260823` (fresh). Labeler flag
  `--stamp_delta 2.0` on the W1 path; version stamps '_w1sb' (the
  frozen w1_read cannot ingest these files — endswith('_w1') fails —
  and this wave's reader refuses everything else). Names
  `wsb_{dom}_{dose}_seed{31,32}_late.npz`.
- **Dup gates**: `wsbdup_finger_e1_seed31_late.npz`,
  `wsbdup_cup_e4_seed32_late.npz` (full filenames — the reader refuses
  anything else) — identical candidates receive identical stamps, so
  every estimand stays EXACTLY zero under the stamp too; violation
  refuses.
- Meta witnesses: stamp.delta == 2.0, stamp.kind ==
  'first_step_dim0_clip', stamp.stamped_steps == 16600 (the exact
  identity above).

## Estimands and decision rules

Reader `analysis/wsb_read.py` FROZEN with this file (selfcheck PASS:
gain 1/0.4/0 fixtures → CALIBRATED/BIASED/BLIND; underspread NO-CALL; 6
refusal legs; ONE-read guard). Cluster = cell (n=8),
permutation-primary + BCa at α=.05.

- **Admissibility, registered before any data**: median over states of
  sd_m(planted) ≥ **0.20** (G units — raised from 0.10 at build review
  5.7: an admitted plant must DOMINATE the archived substrate ceiling
  0.2026, not merely exist; pre-freeze context from the archived
  analogue cells puts the realized median near 0.75, disclosed per
  §1.1 governance), else **NO-CALL-UNDERSPREAD** — and on that branch
  the reader computes and emits NOTHING else (review 5.7: the
  registered remedy re-registers a diversified plant on the same
  cells, which must stay value-blind).
- **PRIMARY — recovery slope**: per cell, pooled within-state-centered
  OLS of all-repeat candidate means on planted(s,m); the chain recovers
  the plant iff slope = 1.
  - **IN-VIVO-CALIBRATED**: slope CI excludes 0 AND covers 1 ⇒ the
    end-to-end chain (env, snapshot/restore, CRN marks, G accounting)
    registers real per-candidate value at the planted scale; the
    "calibrated instrument found nothing" sentence gains its in-vivo
    leg.
  - **IN-VIVO-BIASED**: CI excludes 0, not 1 ⇒ the chain attenuates or
    inflates real signal; the measured slope becomes a correction
    factor that every magnitude claim must carry.
  - **CHAIN-BLIND**: CI includes 0 ⇒ the chain cannot see a real
    planted signal in vivo — instrument-invalid for magnitude claims;
    the measurement-paper floor survives but all ceiling statements
    are struck.
- Slope disclosure (review 5.5): the estimand is formally 1 + β where
  β is the substrate's own action-dim-0 value gradient — measured
  pre-freeze on the archived analogue cells at −0.007 ± 0.024
  (disclosed §1.1-governance context; essentially nothing).
- Witnesses (reported only on the admissible branch, adjudicating
  nothing): **probe slope** — r_real regressed on planted (review 5.6:
  the archived substrate has candidate-constant r_real, so this is an
  essentially confound-free referee separating "stamp never landed"
  from "G accounting destroys it" if the primary reads CHAIN-BLIND);
  split-selected opportunity on the stamped arrays (n=8, low-powered —
  disclosed); consumer tilt delta·clip(actor[0]) − mean_m planted (a
  positive fire is a WIRING ALARM — the frozen consumer cannot adapt —
  not a finding).

## Gates (refusal = read not consumed)

Exactly the 8 registered cells + 2 registered dup passes; stamp meta
pins (delta/kind + stamped_steps == 16600 exact); env_seed 20260823,
repeats 8, states 200, horizon 100, label_every 25, M 8; version
'_w1sb'; filename ↔ run_id identity; finite arrays (g, cands, actor,
r_real, real_scores — review B2: a NaN would otherwise silently
disable the admissibility gate) + finite realized spread; no
unregistered wsb* files; duplicate-null exact zero.

## Ops

dv3 R3 LATE checkpoints (RCC) → 2 dup passes → 8 cells
(`--w1_repeats 8 --stamp_delta 2.0 --env_seed 20260823 --oracle_all`)
→ bundle + sha manifest → **[ME] ONE read**. Cost ≈ 10 W1-scale passes
(UNVERIFIED estimate; ops re-derives from the first measured pass).
