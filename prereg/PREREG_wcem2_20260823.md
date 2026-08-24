# PREREG — WCEM fresh-cohort anti-harvest test (wcem2; 23 Aug 2026)

**Status: FROZEN at user commit. ONE read execution. LAST look on the
anti-harvest question for this substrate — no third cohort, ever.**
Reviewer: ONE Opus pass, user pre-approved, triggered at build.

## Motivation (registered with the trigger stated honestly)

The executed WCEM look-1 (artifacts/wcem_read_20260823) landed
CEM-ACTOR-INDISTINGUISHABLE with point **−0.4803** and BCa CI
**[−1.155, +0.001]** — the upper bound grazing zero from below, on 8
cells, with the planner-competence gate PASSING (+4.13 pooled). This
registration asks the anti-harvest question properly, once, on fresh
cells, and retires it either way. **Winner's-curse caveat**: the
look-1 point is a boundary observation and may be inflated toward
negative; the honest fire probability is ~50–60%, not the naive-power
~80%. Every branch is valuable: a fire upgrades the flagship from
"the value surface is exhausted" to "pushing past the operating point
is actively misleading" (Goodhart-at-deployment) and MEETS item-3's
trigger (reward-only CEM attribution un-parks — **this SUPERSEDES the
23-Aug post-look-1 "parked permanently" record** in STUDY_LEDGER/TODO,
review M1: the ORIGINAL 21-Aug parking was conditional on exactly an
anti-harvest fire, and this registration restores that conditional
with a fresh registered test; the look-1 VERDICT itself is untouched);
a no-fire retires the lean permanently and re-affirms the permanent
parking, and tightens the indistinguishability certificate ~1.7×
(two-sided MDE80 0.87 → ~0.50; the reader's -QUALIFIED threshold uses
the ONE-SIDED MDE80 ≈0.45-at-look-1-sd — distinct numbers, both
correct).

## Design — 24 fresh cells, zero training

- Cells: cup/finger × e1/e4 × **seeds 33–38** — the unused dv3 R3
  LATE checkpoints (the registered W1 cohort runs 31–38; look-1 used
  31–32; WDC used 31–34 under its own protocol/names — no collision).
- Labeler: identical to look-1 (`--w1_repeats 8 --cem_consumer
  --oracle_all`), **fresh `--env_seed 20260825`**. **Producer
  code-identity (review M2)**: verified at review — the repo drift
  since the look-1 freeze 8828ab6f (agent.py +101/−4, configs.yaml
  +17) is CEM-inert (advd/penalty_mix branches only; d0/
  oracle_labels.py and analysis/w1_read.py unchanged); the producer
  MUST run at or after repo commit **790c1393** and ops RECORDS the
  realized producer commit sha in the bundle NOTES (an identity claim
  in a frozen registration is a recorded sha, not prose — the NOBOOT
  lesson); names
  `wcem_{dom}_{dose}_seed{33..38}_late.npz` + dup gates
  `wcemdup_finger_e1_seed33_late.npz`,
  `wcemdup_cup_e4_seed38_late.npz`.
- **Execution regime**: the approved concurrent regime
  (MEM_FRACTION + tasks/GPU) is permitted under the SAME four
  conditions as look-1 (uniform across every pass incl. dups and the
  optional smoke; regime recorded in bundle NOTES; dup gates must
  pass UNDER the production regime; no override on a dup violation).
  ~26 passes ≈ 3 concurrent rounds ≈ a few hours wall.

## Estimands and decision rules (frozen)

Reader `analysis/wcem2_read.py` — DERIVED from the frozen executed
`analysis/wcem_read.py` (untouched): same load machinery, CEM pins
(iters 4 / samples 64 / horizon 6 / elites 8 / std 0.5), CRN witness,
dup-null exact-zero, competence gate; parameterized cohort constants
+ the one-sided primary. Selfcheck PASS (5 branches + 12 refusals
incl. the 6 inherited gate legs — cem pin/version/run_id/CRN
witness/w1.cem/dup-null — review m5).

- **PRIMARY (ONE-SIDED, α=.05)**: d = per-cell mean[g_cem − g_actor],
  n=24 clusters. **ANTI-HARVEST-CONFIRMED** iff one-sided MC
  sign-flip p < .05 (N=200000, seed 20260820) AND the 95% BCa CI
  upper bound < 0 (conservative conjunction, disclosed). Fire
  consequences: Goodhart-at-deployment upgrade; item-3 trigger MET.
- **CEM-OUTPERFORMS-FRESH**: the symmetric two-sided positive
  surprise (perm p < .05 AND CI low > 0) — reported, licenses
  nothing further.
- **No fire ⇒ ANTI-HARVEST-RETIRED, PERMANENTLY** (this is the LAST
  look; the look-1 lean is thereafter described as noise in every
  draft). `-QUALIFIED` appended iff realized one-sided MDE80 >
  |look-1 point| = 0.4803 (power shortfall disclosed; retirement
  stands regardless — pre-priced, the U1a pattern).
- **NO-CALL-PLANNER-INCOMPETENT**: pooled mean[cem_score − q_best]
  < 0 on the fresh cells — no adjudication (conservative against the
  flagship, as in look-1).
- **POOLED (descriptive, NO decision weight, pre-named here)**: point
  + BCa CI over the 32 combined cells, loaded from both label
  bundles (independent env_seed marks); mirrors the w1_read_pooled
  precedent. It cannot rescue or veto anything; a look-1 load failure is RECORDED in the output (pooled_descriptive.error), never raised — the primary is unaffected.

## Gates (refusal = read not consumed)

All look-1 reader gates inherited on both cohorts (cem pins, env_seed
per cohort, repeats/states/horizon, version `_wcem`, run_id↔filename
identity, CRN witness bit-identity, finiteness, action box, dup-null
EXACT zero, cell counts 24/8, no unregistered wcem* files); PLUS
`--look1_read` must be the executed look-1 json whose
cem_minus_actor.point equals the pinned −0.4803125 AND whose
n_cells==8 and verdict is the executed CEM-ACTOR-INDISTINGUISHABLE
(review m2: a one-field stub cannot satisfy the sequencing proof);
ONE-read guard PATH-PINNED to artifacts/wcem2_read_20260823
(review B1).

## Ops

R3 LATE checkpoints seeds 33–38 (RCC/instance — same substrate as
look-1) → 2 dup passes + 24 cells (`--w1_repeats 8 --cem_consumer
--env_seed 20260825 --oracle_all`; ordering waiver as look-1 if run
concurrently — the gate, not the order, is load-bearing) → bundle
(labels + NOTES recording the regime + sha manifest; counts asserted
against the reader constants) → **[ME] ONE read**:
`python -m analysis.wcem2_read --labels <fresh>/labels
--look1_labels local_results/p2_wcem_20260823_210000/labels
--look1_read artifacts/wcem_read_20260823/wcem_read.json
--output artifacts/wcem2_read_20260823` — the output path is
CODE-PINNED in the reader (review B1: labels are not reproducible
job-to-job, so the path-pinned ONE-read guard IS the enforcement of
LAST-look permanence; a second draw cannot be read anywhere).
**Ops paths (review m6)**: wave dir `ops/waves/p2_wcem2/` and bundle
`local_results/p2_wcem2_<ts>` — NEVER regenerate into
`ops/waves/p2_wcem/`, whose untracked generated/ tree is the executed
look-1 wave's only provenance. Cost ≈ 26 look-1-scale passes under
the concurrent regime (UNVERIFIED; ops re-derives).
