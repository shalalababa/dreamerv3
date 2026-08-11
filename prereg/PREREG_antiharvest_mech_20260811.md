# PREREG: anti-harvest mechanism wave (TM2 W2 passes) — 2026-08-11

User GO 2026-08-11 ("explore all explanations"). Companion to
`PREREG_antiharvest_zc_20260811.md` (zero-compute stage on executed
labels); this wave runs NEW label passes to discriminate the mechanism
behind the W1 tm2 anti-harvest (deployed-planner harvest −0.301*,
`artifacts/w1_read_20260811/`). Handoff/continuation mismatch is
structurally excluded (identical planner-continuation for every branch
— labeler construction), leaving three candidate mechanisms:
**optimizer's curse against the learned value** (the planner's action is
model-favored beyond its truth), **optimization-pressure causation**
(more MPPI iterations → worse ground truth), and **warm-start inertia**
(the deployed t0=False plan is stale). Known at freeze: everything
archived incl. the W1 numbers. Unknown: every W2 quantity.

## Design — 34 label passes on the SAME TM2 substrate

- Labeler: `probing/tdmpc2_oracle_labels.py --w2_repeats 8` (built +
  selfchecked with this registration: `_w2_label_state_tm2`,
  `Tm2Oracle.q_of` / `set_iterations`, `_W2Mock`, `selfcheck_w2_tm2`;
  `_w2` stdout redaction in `d0/oracle_labels.py` `save_rows`).
- Cells: the same 32 tm2r3 checkpoints as W1
  ({cup, finger} × {e1, e4} × seeds 51–58, `ckpt_late.pt`) + 2
  duplicate-null gate passes (one per domain, `--w2_dup_cand`,
  finger/cup e1 seed 51). `--env_seed 20260811` (new; W2 estimands are
  within-pass only), `--oracle_all`, states 200, horizon 100, M=8.
- Per state the pass records: the W1-style candidate panel
  (`g_all_rep (8, 8)`, `qfull`), four planner first-action variants —
  **warm** (t0=False + warm-start buffer, the deployed convention),
  **cold** (t0=True, default 6 iterations), **p_low** (t0=True,
  iterations=1), **p_high** (t0=True, iterations=12) — each with its
  ground-truth repeats (`g_{warm,cold,plow,phigh}_rep (8,)`, marks
  SHARED with the candidate branches per repeat: CRN) and its
  Q-ensemble value (`q_*`, the same learned-value metric as qfull).
  The iterations override applies to the ACTION DRAW only (restored in
  a finally block); every ground-truth rollout continues at default
  iterations — variants differ in the first action alone. Variant
  action-draw mark tags are 10/11/12 (review M7: tags ≥ 101 collide
  with `_w1_seed`'s t·101 term across labeled states; repeats ≤ 8
  asserted). Registered command defaults: `--states 200 --horizon
  100 --label_every 25 --actions 8` (explicit, review NIT 24); the
  operator must refuse to overwrite an existing output file (NIT 25:
  `[ -e <out> ] && exit 1` before each pass).
- Output: `$RUNROOT/w2_labels/w2tm2_{dom}_{cell}_seed{k}_late.npz`
  (+ `w2tm2dup_…`). Registered command per cell:
  `python -m probing.tdmpc2_oracle_labels --run_logdir <dir> --checkpoint <dir>/ckpt_late.pt --output $RUNROOT/w2_labels/w2tm2_<dom>_<cell>_seed<k>_late.npz --oracle_all --w2_repeats 8 --env_seed 20260811`
  (dup gates add `--w2_dup_cand` AND use the dup output name
  `$RUNROOT/w2_labels/w2tm2dup_<dom>_e1_seed51_late.npz` — review
  #17: the plain name would overwrite the real seed-51 cell file). Cost ≈ 1.5× a W1 pass per cell.
- KEEP note: `w2_labels` added to `runroot_cleanup.sh` KEEP_PENDING in
  the freeze-commit.
- Smoke gate: ONE cell with `--states 4`; assert `iters_realized` =
  [1, 6, 12], the file round-trips, and — the KNOB-VALIDITY form
  (review M6: distinctness alone is not diagnostic, the three cold
  draws use different RNG marks and TD-MPC2's eval-mode gumbel
  selection makes actions differ even with an inert knob) — that with
  a FIXED mark, `follower_act(obs, t0=True)` at iterations=1 vs 12
  differs at ≥1 state. Smoke output deleted before the wave (never
  collated). The reader additionally gates on the per-state
  `iters_realized` witness and `meta.planner.{mpc:true, iterations:6}`
  in every file.

## Registered decision rules (pooled over 32 cells, cluster = cell; BCa 95% B=10K rng 0 + cluster sign-flip permutation; a fire needs CI excluding 0 AND p < .05)

Per state: g(x) = mean over repeats; gbar = mean over the 8 candidates
of g; q(x) = mean over Q-heads.

- **G0 (in-wave replication GATE)**: harv_warm = g(warm) − gbar pooled.
  Gate passes iff CI < 0 AND p < .05. A significantly POSITIVE harvest
  instead reads **REPLICATION-SIGN-REVERSED** (review #12), also
  zero-weighting the mechanism legs. Otherwise FAILS ⇒ **REPLICATION-FAILED**:
  all mechanism legs reported with ZERO decision weight (the W1
  anti-harvest fact itself is untouched — it has its own read; this
  wave simply loses its premise).
- **P-N1 (curse signature, primary)**: within state, over the 9 actions
  {8 candidates, warm}: z_q = standardized q, z_g = standardized g;
  gap = z_q − z_g; stat = gap(warm) − mean over candidates of gap.
  (Algebraic note, review #13: because both z-scores are mean-zero over
  the same 9 actions, this equals (9/8)·gap(warm) exactly — the
  statistic IS warm's own standardized model-minus-truth score, scaled;
  the "baseline" phrasing is presentational.) **Degenerate-state
  exclusion (review B2, registered)**: a state whose 9 ground-truth
  returns are (near-)tied (std ≤ 1e-12; W1 batch-review B2 convention)
  carries no truth content and is EXCLUDED from P-N1 — otherwise every
  tied state would contribute a systematic positive with zero evidence
  (the executed W1 read saw 6/32 cells fully G-degenerate). Cells with
  zero retained states drop from the P-N1 pool; **coverage floor: if
  retained states < 25% of all states pooled, P-N1 = INSTRUMENT-LIMITED
  (cannot fire)**, recorded with the coverage fraction.
  CI > 0 ∧ p < .05 on the retained pool ⇒ **CURSE-CONFIRMED** (the
  planner's action is model-overvalued relative to ground truth).
- **P-N2 (pressure dose)**: per state, OLS slope of g over ln(iters) at
  the three cold points {(1, g(p_low)), (6, g(cold)), (12, g(p_high))};
  pooled slope CI < 0 ∧ p < .05 ⇒ **PRESSURE-CAUSAL**. The prior mode
  (candidate 0) is the iters→0 reference, descriptive only.
- **P-N3 (inertia)**: d_inertia = g(warm) − g(cold) pooled.
  CI < 0 ∧ p < .05 ⇒ **INERTIA-CONTRIBUTES**.
- **Verdict map (G0 passing)**:
  - P-N1 ∧ P-N2 ⇒ **OPTIMIZERS-CURSE (pressure-causal)** — with P-N3
    additionally noted if it fires;
  - P-N3 alone (¬P-N1 ∧ ¬P-N2) ⇒ **INERTIA-DRIVEN**;
  - P-N3 ∧ (P-N1 ∨ P-N2) ⇒ **MIXED**;
  - P-N1 alone or P-N2 alone ⇒ **PARTIAL-CURSE** (named leg reported);
  - none ⇒ **UNRESOLVED** (anti-harvest real but unattributed —
    disclosed; no mechanism wording licensed).
- **Duplicate-null gate**: on dup passes the candidate panel must be
  exactly zero-spread within repeat (as W1); violation ⇒ the read
  REFUSES (INSTRUMENT-INVALID).
- Descriptives: q-vs-g Spearman over the 9-action sets; per-domain
  splits; g(p_low/cold/p_high) means with the candidate-0 reference;
  cross-stage coherence sentence vs the ZC read (no verdict).

Reader: `analysis/w2_mech_read.py`, frozen with this file, selfcheck
(planted fixtures per branch + verdict-map legs + load_cells
round-trip with run_id/knob-witness pin trips + the coverage
INSTRUMENT-LIMITED leg) before any pass runs. Load gates: filename↔
run_id identity, horizon/label_every/actions pins, iters_realized
witness, planner meta pins, dup-file meta pins + one dup per domain. ONE read execution. Reviewer: 2026-08-11 batch review
(ONE reviewer, user-approved) BEFORE freeze-commit.
