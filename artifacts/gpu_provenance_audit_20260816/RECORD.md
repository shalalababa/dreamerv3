# GPU-provenance audit of executed measurements — 2026-08-16

Diagnostic. Triggered by `ops/waves/lewm_probe_devcheck/FINDING.md`: the E4
ridge probe is byte-identical re-run on the same GPU but moves up to **0.113
AUROC** across GPU models. Question asked: does that invalidate anything
already measured?

## The structural hazard

Midway3's `gpu` partition is **heterogeneous**:

| nodes | GPU |
|---|---|
| midway3-0277 … 0286 | **V100** |
| midway3-0294 | **A100** |

No submission script in this repo has ever passed `--constraint`, so Slurm was
free to scatter any panel across both models. Nothing recorded which it got:
`e4_*/summary.json` stores `run_id`, `expl_mode`, `probeset_sha256`,
`deter_std` — and **no device, host, or platform field**. That absence is why
this had to be reconstructed from `sacct` job windows and file mtimes rather
than read off the outputs.

## Method

`sacct` (retention reaches back to 2023, so it survives the deleted `.out`
files) for every job 2026-07-01 → 2026-08-16; node → GPU model; each
`*/e4_*/summary.json` mtime matched against job windows.

**Limit of the method, stated plainly:** a file written while any RCC job
happened to be running can be attributed to it by coincidence, so this
identifies *A100 exposure* reliably (the A100 job list is short and each
attribution below was confirmed against job duration) but does **not** certify
every other panel as uniform. Vast-measured panels are indistinguishable from
V100-measured ones by this method.

## Finding 1 — the td (dose) panel's E4 is split down the middle

| panel | s1–s8 | s9–s16 |
|---|---|---|
| `ax1wm_finger_td0` | **A100** | V100 |
| `ax1wm_finger_td1` | **A100** | V100 |
| `ax1wm_finger_td2` | **A100** (s5,s6 non-Slurm) | V100 |
| `ax1wm_finger_td3` | **A100** (s5,s6 non-Slurm) | V100 |

Two jobs did it:

* `53290838` — **midway3-0294 (A100)**, 2026-08-11 20:01:46→20:12:38
* `53330472` — midway3-0285 (V100), 2026-08-13 13:07:39→13:18:38

Corroborated by arithmetic, not just window overlap: td0 seeds 1–8 were
written at 20:02:12…20:04:29 in even ~20 s steps, and 32 cells × ~20 s ≈ 10.7
min ≈ each job's elapsed (10:52 / 10:59).

Four cells — td2/td3 seeds 5,6, written 2026-08-13 22:07 — match **no** Slurm
job. A job named `dose_e4_fix4_cuda` was cancelled 21 min earlier without ever
being assigned a node, so these were produced outside Slurm and their device is
unknown. They are the two seeds where the arm-to-arm balance is not established.

## Impact: the registered dose verdicts do NOT rest on this

`artifacts/dose_ext_read_20260813` pools seeds 1–16 — precisely across the
split — and passed by 0.0007 in p (perm .029297 vs Pocock .0294), with the
seeds-9–16 subset at +47.6 against the pooled +102.6. That looks alarming until
the estimand is checked:

* **G1 and P-DR1 are `auc100k`**, computed from the *adapts'* `scores.jsonl`.
  Every td adapt in both seed blocks ran on **Vast**, not RCC — so the
  governing contrasts are not carried by the V100/A100 split at all.
* **P-DR3 membership (1.79 / 1.39 / 1.00 / 0.99) IS E4-derived**, and its two
  halves come from different GPU models. The record already labels it
  "descriptive, never adjudicated".
* `deter_std` and any other E4 diagnostic on td inherits the same split.

**No registered decision is invalidated.** What is not safe is comparing td
seeds 1–8 against seeds 9–16 on any E4-derived quantity, or quoting a pooled
E4 spread as if the 16 cells were exchangeable.

## Finding 2 — three training-side splits (weaker)

| family | A100 | V100 |
|---|---|---|
| `orth_obj_b00…b07` | b01 | b00, b02–b07 |
| `axis1_bundle_20260812_002356_00*` | 001 | 002–005 (006–008 failed) |
| `pilot_*_seed1` (exploration-pretraining) | seed1 ×5 | seeds 2,3 on V100 |

Training is stochastic and seed-varying, so a device difference is one more
between-run nuisance rather than a systematic offset — *unless* it aligns with
an arm. For the two bundle families the device sits at shard granularity; if a
shard is arm-homogeneous the concern is real, if arm-crossed it is not. **Not
resolved here — worth checking the shard contents before either is leaned on.**
`pilot_goal_reacher` is split seed1-vs-seeds2,3 outright.

Everything else on the A100 either FAILED or CANCELLED (`pixel_swamp_e4`,
four `adapt_bundle_2026070*`, `tm2_diag`) and produced nothing, or was a
seed99 smoke.

## Fixes applied

1. **`wavegen.py` now emits `--constraint=v100` on every Slurm stage** by
   default (`sbatch: {constraint: a100}` to choose otherwise, `""` to opt out).
   Two selfchecks pin both directions. This closes the hazard going forward:
   the system can no longer scatter a panel across GPU models by accident.

## Recommended, not done here

2. **Record device provenance in the instruments.** `ridge_probe measure`
   should stamp GPU model + hostname into its json. It is additive and cannot
   change a computed value, but it touches a measurement instrument mid-study,
   so it belongs in a registered amendment rather than an ops commit — the same
   shape as the earlier "bitwise-inert witness" amendment.
3. **Decide how the LeWM/E4 reads should treat td's E4 columns** given the
   seed-block split, and whether td2/td3 seeds 5,6 need re-measuring on the
   same model as their arm-mates to restore balance.
