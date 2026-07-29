# PREREG R3 Amendment 1: reader seed-grid repair + integrity guards (frozen 2026-07-29)

Amends `PREREG_r3_competence_20260725.md` (freeze commit 8b3380ce).
Registered BEFORE the one R3 read executes: no read output exists
anywhere (verified — zero reader outputs in the bundle, no
`analysis_out/`, no `r3.json` in the repo), and no decisional quantity
(no mean/CI of opportunity, achieved, or gap) has been computed by
anyone or anything. Discovery and the audit below were value-blind.

## The defect

`analysis/r3_read.py` (frozen at 8b3380ce) line 51 had
`EXPECT_SEEDS = tuple(range(1, 9))` and its docstring said "seeds
1-8", contradicting the registration itself (prereg §Design: "seeds
31–38 (disjoint from all prior D1 seeds)") and all 64 real label
files. `check_grid` therefore trips `core cell missing: cup/e1/seed1`
before touching any label value — the frozen instrument cannot run on
the registered grid at all. Proven with a key-level harness (filenames
only, no label data loaded). The build-time selfcheck could not catch
this class of defect: it synthesizes its fixture FROM `EXPECT_SEEDS`,
so reader and fixture were wrong together.

## Registered repair (applied in `analysis/r3_read.py`; selfcheck PASS)

Mechanical seed-literal corrections — the fix is fully determined by
the prereg text and the crash; nothing here can be outcome-motivated:

1. `EXPECT_SEEDS = tuple(range(31, 39))`; docstring grid note 1-8 → 31-38.
2. Selfcheck literals that referenced in-grid seeds updated so every
   branch still exercises: floor_runs seeds (1–5)/(1–4) → (31–35)/(31–34);
   deletion-trip keys seed 3 → 33 (reacher), seed 2 → 32 (cup).

Hardening guards added on the pre-freeze-review model (each verified
NON-TRIPPING on the real bundle by a value-blind integrity audit that
checked shapes, finiteness, and index ranges per file — never any
aggregate of label values):

3. **Unregistered-seed guard** (`check_grid`): any parsed seed outside
   31–38 asserts. Without it, a validly-named stray file (e.g. a
   leftover `seed310`/prior-campaign seed) with both maturities would
   silently pool into all three primaries (demonstrated synthetically:
   an injected stray run shifted a synthetic P-R3a point with no error).
4. **Finiteness guard** (`validate_arrays`): non-finite `g_all`/`g_now`
   asserts. The labeler fills unevaluated `g_all` targets with NaN and
   meta does not record `--oracle_all`, so a non-oracle_all pass is
   otherwise silent — `nanmax` would compute opportunity over a
   candidate subset (deflating P-R3a) and `nanmean` would average opp
   and ach over different state subsets. All-finite `g_all` is the
   reader-detectable signature that `--oracle_all` was in effect
   (all 64 files verified all-finite).
5. **m_real bounds guard**: negative indices otherwise wrap silently
   (numpy) and would corrupt `achieved` without error (all 64 files
   verified in [0,8)).
6. **Shape/n_states guards**: per-file row-count consistency; all
   cells must have the registered 200 states (per-cell means pool with
   equal weight, so a truncated file would otherwise enter at full
   weight; all 64 files verified at exactly 200).

## What does NOT change

Every decision rule is verbatim-unchanged: P-R3a pooled opportunity
CI > 0.2, P-R3b pooled gap CI > 0, P-R3c paired late−early achieved
CI > 0; run-clustered percentile bootstrap B=10K `default_rng(0)`,
cluster = training run with both maturities sharing the cluster; floor
policy (inclusion, per-domain fractions, >50% flag); core all-or-nothing
per run; reacher cohort all-or-nothing; verdict branches and the
consequence map. An independent audit confirmed the frozen decisional
logic matches the prereg exactly and that the estimands match the
labeler's schema (`g_now = G[m_now]`, `g_all[m] = G[m]` under
oracle_all) and the 25-Jul relabel decomposition. Reader sha256:
defective frozen-era `3ef90e4895b29ebce0374d1dc59b5acfdd53a8972090f1f8b8aab7330c9c9f64`,
amended `cf52724ef3db81a6e9b51f77fef13cdeb1f185809f48dc27f641346d7b8dfa5f`.

## Provenance disclosures (from the value-blind bundle audit)

- Bundle `local_results/r3_pilot_smoke_labels_20260727_222940/`:
  training + smoke ran on the Vast box (`/workspace`); the 64 label
  passes ran on RCC midway3 after an rsync (all label meta paths under
  `/scratch/midway3/...`; bundle committed from an RCC login node).
  The prereg pins ordering and labeling dials, not host; every npz
  meta carries the registered dials (states 200, horizon 100,
  label_every 25, actions 8, rollouts 16, labeler seed 0, ref_stride 5,
  `d1fix_20260724`) and the correct per-cell dose (e1 dim 0 / e4
  dim 32 scale 3.0).
- The labels-stage stdout log is absent from the bundle (only
  pilots/smoke/worker logs shipped); bundle assembly reset all file
  mtimes, so smoke-before-labels ordering rests on the worker log
  (smoke DONE 27 Jul 00:15:25 UTC, `SMOKE_OK` present in the labels
  dir) rather than filesystem timestamps. Vast-side ordering is fully
  evidenced: freeze commit 25 Jul 15:47 UTC → pilots → smoke.
- Smoke gate = the registered four passes ({e1,e4} × {early,late}, real
  dosed envs — e4 smokes show the dim-32 distractor obs space) with
  gate semantics "all passes exit 0 + inspection"; the 5-state smoke
  deltas are informational only (e1-early mean delta_real −0.400,
  the other three +0.000).
- Training: 32/32 two-phase with the continuation property verified
  (every phase-B resume loads phase-A's final snapshot; zero
  mid-training restarts); realized snapshots early 22,992–24,960 /
  late 98,016–100,000, inside the disclosed ~1–2K envelope.

## Ordering

This amendment + the amended `analysis/r3_read.py` are committed
BEFORE the one registered read executes on
`runroot_light/r3_local/r3_labels`. Everything above was established
without computing any registered quantity.
