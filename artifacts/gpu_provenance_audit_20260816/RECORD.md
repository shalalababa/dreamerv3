# GPU-provenance audit of executed measurements — 2026-08-16

Diagnostic. Triggered by `ops/waves/lewm_probe_devcheck/FINDING.md`: the E4
ridge probe is byte-identical re-run on the same GPU but moves up to **0.113
AUROC** across GPU models. Question asked: does that invalidate anything
already measured?

## The structural hazard

Midway3's `gpu` partition is **heterogeneous — three models, not two**:

| nodes | GPU |
|---|---|
| midway3-0277 … 0281 | **V100** |
| midway3-0282 … 0286 | **RTX6000** |
| midway3-0294 | **A100** |

> **CORRECTION (same day, before this record was used).** The first version of
> this audit said "0277–0286 = V100". That came from spot-checking 0277 and
> 0294 with `scontrol` and extrapolating across the `sinfo` grouping, whose
> feature column was truncated in the output I read. Enumerating every node
> (`sinfo -N -o '%N|%f'`) shows half that range is RTX6000. The audit's
> conclusion is unchanged — the same four td panels are the only mixed ones —
> but the second model in the td split is **RTX6000, not V100**, and the
> figures below are the re-run with three classes. A spot check plus a range
> assumption is not an enumeration; this is the second time in this audit that
> the honest move was to enumerate rather than infer.

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
| `ax1wm_finger_td0` | **A100** | **RTX6000** |
| `ax1wm_finger_td1` | **A100** | **RTX6000** |
| `ax1wm_finger_td2` | **A100** (s5,s6 non-Slurm) | **RTX6000** |
| `ax1wm_finger_td3` | **A100** (s5,s6 non-Slurm) | **RTX6000** |

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

## Follow-up 1 — the B1′ anchor's device is POSITIVELY established

The earlier note recorded the anchor panel `regen_ridge_e4_20260810_220849` as
"device not positively established; A100 excluded by enumeration, V100-vs-Vast
undecidable by sacct". It is decidable, and it is decided: the bundle carries
its own producer logs, `refit_ridge_panel_53243601_{0..3}.out`, i.e. a Slurm
array job.

```
53243601_0..3   refit_ridge_panel   midway3-0278   COMPLETED   gpu
                                    2026-08-10 20:32 → 20:45
```

**midway3-0278 is a V100, and all four array tasks ran on that one node**, so
the anchor panel is internally device-uniform as well. B1′ therefore takes the
cheap resolution — pin the fresh ridge panel to `--constraint=v100` (already
the generated default) — rather than a 32-cell anchor re-measure, *provided*
model-pinning is sufficient (see the open question below).

## Follow-up 2 — cudnn 5003 is an ARCHITECTURE fact, not a broken node

The 12 pixel probe jobs of 2026-08-15 separate perfectly:

| node | model | outcome |
|---|---|---|
| midway3-0280 | V100 | **9/9 FAILED**, `<unknown cudnn status: 5003>` |
| midway3-0282 | RTX6000 | **3/3 COMPLETED** clean |

So "RCC Slurm cannot run pixel workloads" was too broad: `cudnnConvolutionForward`
fails on Volta in this JAX build and runs fine on Turing. Pixel legs can stay on
RCC pinned to `--constraint=rtx6000`; only the V100s are excluded.

This also makes the blanket `--constraint=v100` default **wrong for pixel
waves** — it would route them to the one architecture that fails. Fixed by
pinning `constraint: rtx6000` on `lewm_probe_rcc`, with a selfcheck.

## Open question (measurement queued)

The devcheck established the ACROSS-ARCHITECTURE term (0.113) and a zero
same-GPU noise floor. It never tested **same model, different node** — which is
exactly what `--constraint=v100` buys B1′. Job `53411135`
(`scripts/ridge_devnode_check.sbatch`) re-measures three anchor cells on a V100
that is *not* 0278 and byte-compares against the anchor jsons:

* **identical** → `--constraint=v100` is sufficient; B1′ pins and proceeds
* **different** → model pinning is not enough; B1′ needs `--nodelist`, or the
  anchor must be re-measured beside the new panel

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
