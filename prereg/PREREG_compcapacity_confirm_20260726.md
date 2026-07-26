# PREREG: Paper-3 confirmatory 25m point — A→B transition (frozen 2026-07-26)

The Paper-3 (composition×capacity flagship) confirmatory registration
the launch plan gated on the Scaling B read
(`Plan_CompCapacity_Launch_20260724.md` queue item 2). The B read
landed 26 Jul (`artifacts/scaling_optionb_20260726/`): PRIMARY
three-way includes 0, P-E1 holds, 12m = regime A. The frozen
consequence map (`PREREG_compcapacity_theory_20260724.md`, committed
746d0b13 pre-read) then selects the target: **"the 25m point targets
the A→B transition (task-lo rise)."** This file +
`analysis/compcapacity_read.py` (selfcheck PASS) are committed BEFORE
any size25m run exists.

## Buy decision and rule reconciliation (disclosed)

The launch plan's draft buy rule ("buy 25m iff the 1m→12m contrast is
sign-stable...") was a placeholder superseded by the later-frozen
consequence map, which conditions on the P-E1/regime adjudication
instead; that condition resolved to regime-A ⇒ the 25m point's
question is the transition location, and the buy defaults to GO under
the portfolio policy. Power (from the B read, seeds 1–8): task-lo@12m
97.0 ± 33.9 ⇒ the unpaired 8+8 primary detects a rise of ≈33 AUC
units — a quarter of the full transition magnitude (the 12m side gap
is 139.2). Well-powered for any theory-relevant movement.

## Design: 32 jobs, mirroring Option B at size25m

- Grid: {task, apt} × {s0, s1} × seeds 1–8 (apt cells serve the
  ordering-law guard and the interaction@25m descriptive).
- `size25m` (existing `configs.yaml` block: deter 3072 / hidden 384 /
  classes 24 / depth 24 / units 384) via `AXIS1_SIZE=size25m` for
  BOTH fit and adapt stages (a size mismatch fails checkpoint
  loading — the built-in audit guard, as at 12m).
- Buffer: `axis1_finger/q1_ctx25` — the frozen q1 pair resized with
  the committed tool
  (`python -m probing.resize_replay_context --input
  $RUNROOT/axis1_finger/q1 --output $RUNROOT/axis1_finger/q1_ctx25
  --deter 3072 --stoch 32 24`, per side as at ctx12), submitted via
  `AXIS1_QUAD_SUFFIX=_ctx25`. Zero-init ctx protocol is IDENTICAL to
  ctx12 ⇒ 12m↔25m level comparisons are protocol-clean (unlike
  1m↔12m, disclosed at 12m); this is what licenses the primary.
- Run ids `ax1s25q1s{0,1}` (task) / `ax1fs25q1s{0,1}` (apt), WM names
  `ax1wm_finger_{s25|fs25}q1s<side>_seed<k>`; E4 globs must use
  `ax1wm_finger_*s25*` (wm_infix strips `ax1`, same gotcha as 12m).
- Updates 500000, STEPS 1.25e5, frozen readout — the Option B
  protocol with only the size axis moved. REGISTERED FALLBACK
  (walltime): if 500000 updates do not fit the queue limit, BOTH 25m
  arms use the same reduced count (gradient-equalized within the
  stratum) and the realized count is recorded in the ARTIFACT (not
  edited here — lesson from the Option B FILL-slot lapse: this
  registration carries no blank slots; all realized values go in the
  artifact record).
- E4 pass after adaptation: frozen `finger_v1` probe set,
  `GLOB='ax1wm_finger_*s25*'` — registered descriptive membership
  counterpart (A→B predicts task@25m,s0 reward-NLL moves from the 12m
  value 5.77 toward the membership band; apt reward-NLL remains
  absent-by-design under the reward_aware gate, disclosed at 12m).

Submission (clean shell, per the Option B loop with
`AXIS1_SIZE=size25m` and `AXIS1_QUAD_SUFFIX=_ctx25`; build q1_ctx25
first, FROZEN marker before any fit).

## Registered read (frozen `analysis/compcapacity_read.py`)

Seeds 1–8 explicit (canonical-csv seed-99 smoke rows excluded by
rule). AUC100k. B = 10K bootstrap, `default_rng(0)`.

- **PRIMARY (sole confirmatory): task-lo rise R = mean[task@25m,s0] −
  mean[task@12m,s0], two-sample seed-cluster bootstrap; FIRES iff
  CI > 0.**
- **GUARD (decisional for the ordering law): apt rise (25m−12m, sides
  pooled). FORBIDDEN pattern iff the apt-rise CI > 0 while the
  primary does not fire ⇒ P-E1 REFUTED as registered** (Paper 3
  pivots to measuring where the model fails; Paper 1's legibility
  framing weakens per the frozen consequence map).
- Secondaries (never decisional): interaction@25m; B_task@25m vs 12m
  shrinkage + decomposition (rise-of-lo vs change-of-hi — A→B
  predicts shrinkage via the lo cell); apt@25m side simple; cell
  means; sensitivity t-intervals.

## Consequence map (frozen)

- **PRIMARY fires (no forbidden)** ⇒ the A→B transition is DETECTED
  between 12× and 25×: the flagship headline becomes "capacity
  repairs the dose deficit before the legibility deficit, in the
  predicted order"; the E4 counterpart calibrates the membership form
  of the transition; any 50m/interpolation point is a NEW dated
  registration parameterized by this read.
- **Does not fire (no forbidden)** ⇒ regime A persists through 25×:
  the transition bracket widens (the G1 crossing sits above 25m); the
  flagship reports the two-point bounded null + scale-robustness as
  its capacity-law result; a wider ladder is a new registration.
- **Forbidden pattern** ⇒ ordering law refuted (see guard).
- Either way: Paper 1 cites the 25m cells descriptively at most (the
  self-scooping guard from the launch plan stands — Paper 1 makes no
  capacity-law claim).

## Disclosure and ordering

Known at freeze: everything through the 26-Jul Scaling B + 12m E4
read (all 1m/12m cells, membership levels, the consequence-map
selection itself). Unknown: every size25m quantity — no size25m fit,
adapt, or E4 number exists anywhere (the config block exists unused).
Ordering: this file + `analysis/compcapacity_read.py` committed BEFORE
the ctx25 build and submission; realized upd/s, walltime, and any
update-count fallback recorded in the artifact at read time.
