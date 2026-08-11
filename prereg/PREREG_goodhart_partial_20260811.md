# PREREG: Goodhart fixed-code partial re-test on the surviving policy pool — 2026-08-11

User GO 2026-08-10/11 after the recoverability inventory. Context: the
D18 defect (`wm_evaluator.py` fed policies doubly-symlogged observations;
one-line symexp fix landed 2026-08-08) suspended the Goodhart track's
conclusions ("selection ≈ lottery", sprint Spearman +0.153 / p2elong
+0.209 w/ top-1 regret 78.4%). The registered closure — re-score the
ORIGINAL pools with fixed code — is **unrecoverable**: all 64 original
`adapt_*` policy/evaluator run dirs are destroyed (user inventory
2026-08-10), and Midway training is not job-to-job deterministic, so no
regeneration can reproduce the original pool members.

**Registered consequence, effective regardless of this wave's outcome:
the archived sprint and p2elong Goodhart conclusions are PERMANENTLY
WITHDRAWN** (suspended → unrecoverable, same class as #27 apt-drift).
This registration does NOT close them. It answers the forward question:
**can the FIXED evaluator rank policies at all?** — on a surviving pool,
with a surviving evaluator. Outcomes license instrument wording for
future use (e.g. #28 prospective curation) plus at most a SUGGESTIVE
sentence about whether the withdrawn lottery finding was plausibly
bug-driven. No numeric comparison to the archived pool metrics is
licensed (different pools).

**Design-sketch correction (disclosed)**: the 2026-08-10 chat sketch
("re-score 26 surviving original-weight WMs against archived AUCs")
confused the two Goodhart roles — pool members are POLICIES, not WMs;
the surviving WM checkpoints cannot reconstitute the destroyed policy
pool. This registration replaces that sketch.

## Known at freeze / unknown

Known (disclosed): the entire Paper-1 record; the archived REAL returns
of every pool member (committed bundle csvs — final10 + auc100k); the
withdrawn Goodhart numbers. Unknown: **every fixed-code evaluator score
(`m_return`)** — none has ever been computed. Value-blindness applies to
the evaluator-score side, which is the only new measurement.

## Pool (pinned, committed sidecar)

`artifacts/goodhart_partial_reg_20260811/pool_ids.json`
(sha256 c83c185d8f380a1e…): the **106 surviving finger turn_hard adapt
policies** with live run dirs (checkpoints + scores.jsonl) — s×w_r wave
88 (`ax1swr*`/`ax1swv*`) + random-init 8 (`rif`) + Plan2Explore 10
(`p2eof`/`p2eou`) — with their ARCHIVED real returns. Excluded by
construction: `ax1ter*`/`rndte` (turn_easy — different objective, real
returns not comparable), all `tm2*` (non-Dreamer agent API). Real-return
spread: final10 0–904.5, sd 171 (ranking is meaningful). Composition
disclosure: 83% of the pool is s×w_r task-arm policies; Spearman is
pool-composition-dependent — one more reason no archived-number
comparison is licensed.

## Evaluators (two; both must pass `require_task`)

- **E1 (primary)**: the p2elong online model — the SAME evaluator
  weights as the withdrawn p2elong leg. Preflight: record the exact run
  dir + checkpoint; if it no longer exists, E1 is DROPPED with
  disclosure (E2 stands alone).
- **E2**: `ax1wm_finger_q1s1_seed1` — the restored ORIGINAL-weight task
  fit (bitwise-original per the 10-Aug census), the closest surviving
  analogue of the July sprint evaluator (`adapt_ax1q1s1_finger_seed1`,
  an adapt OF this fit).

## Scoring protocol (July-matched; registered verbatim)

Per (evaluator, policy):
`python -m probing.wm_evaluator score --evaluator_run <EVAL> --policy_run $RUNROOT/<policy_id> --init_replay $RUNROOT/pilot_goal_finger_seed1/replay --n_starts 20 --burn_in 16 --horizon 64 --seed 0 --output $RUNROOT/goodhart_partial/<Ek>/eval_<policy_id>.json`
then per evaluator:
`python -m probing.wm_evaluator collate --evals '$RUNROOT/goodhart_partial/<Ek>/eval_*.json' --runroot $RUNROOT --real_window 10 --output $RUNROOT/goodhart_partial/<Ek>/pool_metrics.json`
(2 × 106 = 212 scoring passes; CPU-viable.) The `goodhart_partial` dir
is bundled with a sha manifest. **Hold extension**: the 106 policy run
dirs must survive on cluster until scoring completes — the swave/rif/
p2e KEEP_PENDING entries' release condition is extended in the same
commit (their reads verified 08-10, but this wave now consumes the
dirs).

## Registered decision rules (frozen reader; B=10K rng 0)

Per evaluator, on the collate's rows (reader recomputes everything from
`rows[]`; the collate's own summary numbers are cross-checked, never
trusted):

- **Gates**: scored ids ≡ the pinned 106 (a policy whose checkpoint is
  gone at scoring time is recorded and dropped; pool floor n ≥ 95, else
  the read REFUSES); collate `real` values must equal the pinned
  archived final10 within 1e-6 per policy (provenance witness — same
  scores.jsonl tail; mismatch REFUSES); evaluator identity fields
  recorded (run dir + checkpoint path).
- **P-GH1 (ranking, primary)**: Spearman ρ(m_return, real) with
  label-permutation p (B=10K, rng 0, two-sided).
  - **RANKING-RECOVERED**: p < .05 AND ρ ≥ 0.4 — the fixed evaluator
    ranks; suggestive that the withdrawn lottery finding was
    bug-inflated; instrument usable for prospective curation (with its
    own registration).
  - **WEAK-RANKING**: p < .05 AND 0 < ρ < 0.4 — fixed-code ranking at
    the withdrawn-era magnitude; the bug was not the main story.
  - **LOTTERY-REPLICATES**: p ≥ .05 — the evaluator cannot rank even
    with fixed code; the lottery conclusion re-establishes itself
    cleanly attributable to WM evaluation (forward wording only).
  - **ANTI-RANKING**: p < .05 with ρ < 0 — registered anomaly,
    disclosed, no wording.
- **P-GH2 (selection, secondary)**: top-1 regret fraction vs the EXACT
  random-pick mean regret fraction (computed from the pinned reals).
  **SELECTION-USEFUL** iff top1_regret_frac ≤ 0.5 × random-pick mean;
  top-3/5 regret + inversion rate descriptive.
- **Cross-evaluator rule**: verdicts per evaluator; any forward
  instrument wording takes the WEAKER of the two (conservative). E1
  dropped ⇒ E2 alone, disclosed.

Reader: `analysis/goodhart_partial_read.py`, frozen with this file,
selfcheck (all four P-GH1 branches + gate trips + regret arithmetic
fixtures) before any evaluator score exists. Plain `python -m` (gates
are asserts). ONE read execution. No reviewer (single-instrument
re-scoring on just-reviewed governance patterns; two frozen thresholds;
skip disclosed per standing policy — reviewers reserved for
decision-heavy new instruments).
