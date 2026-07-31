# PREREG: R3 candidate-aware actionability ladder — L0–L4 on the corrected labels (built 2026-07-30; freeze-commit 2026-07-31)

Registered-descriptive, NON-decisional addendum on the EXISTING
committed R3 labels — no new runs, no new labeling, one local CPU read.
This is the successor of the instrument-invalidated D1 ladder
(`PREREG_d1_ladder_20260723.md`): the 24-Jul invalidation was entirely
in the LABELS (stale-carry candidate branches + non-common policy RNG;
repaired labeler `d1fix_20260724`), not in the ladder machinery, so the
nested-LORO ridge machinery, the criteria metric, and the
plant-a-signal selfcheck design carry over and are re-registered here
on the corrected R3 campaign — with one NEW rung (L4, candidate-aware)
and one explicit scope change (below). Closes the residue of 11-Jul
editorial objection 4: which observable features predict the per-state
implementation gap? This file + `analysis/r3_ladder_read.py` (selfcheck
PASS) are committed BEFORE any per-state feature→target statistic on
the corrected labels exists anywhere.

## Question

The committed R3 read established THAT the gap exists (opportunity is
not harvested) at the cell level. This ladder asks WHERE the per-state
gap is legible: in scalar uncertainty summaries (L1/L2), only in the
full belief state (L3), only with candidate-set information (L4), or
nowhere (L0 best). **Registered prediction** (soft — this is
descriptive): if any rung fires, the ordering is L4 ≥ L3 > L1/L2; the
mechanistically interesting outcomes are a compression signature
(belief state predicts what scalars cannot) or a candidate signature
(candidate geometry adds prediction beyond the belief state), because
each points the competence-repair intervention at a different module.
**Genuine risk**: all rungs stay flat (per-state gap is noise around a
run-level constant — the gap would then be a policy-level, not
state-level, phenomenon), or floor states dominate the ranks. A flat
outcome is itself informative and is registered as such (NO PER-STATE
SIGNATURE branch); nothing is re-run on that branch.

## Estimands (two registered target families; looks counted below)

Per labeled state, from the corrected labeler's `g_all` — definitions
match `analysis/r3_read.py` `cell_stats` exactly (identity asserted in
the reader's selfcheck against a hand-computed case):

- **opportunity** = max_m G[m] − G[m_now]
- **implementation gap** = opportunity − achieved, achieved =
  G[m_real] − G[m_now]

Achieved is not a separate target family (gap = opportunity − achieved
makes a third family redundant); imag quantities stay retired and are
never read.

**Look accounting (registered):** 2 targets × 4 cells × 4 correlational
rungs (L1–L4) = 32 Spearman looks, + 8 paired [L4−L3] contrasts, + 8 L0
cell means = 48 looks total, all descriptive, no multiplicity
correction, none decisional. The reader emits this count in its output
(`looks` block).

## Design

- **Data**: the 64 committed R3 label npz
  (`r3_{cup|finger}_{e1|e4}_seed{31..38}_{early|late}.npz`, labeler
  `d1fix_20260724`), i.e. the label set of the executed R3 read
  (`artifacts/r3_competence_20260729/`). Bundle
  `local_results/r3_competence_20260729_105146/labels/` or the RCC
  labels dir — byte-pinned in git at the read commit 4f74b187 (the
  bundle predates the manifests/ policy; the RCC dir was verified
  md5-identical in the read record), AND machine-gated by the reader:
  `read()` recomputes a sha256 digest over the 64 files and asserts it
  equals the frozen `COMMITTED_LABELS_DIGEST` constant (the same
  constant frozen in `analysis/repair_read.py` — both reads consume
  this exact substrate). Either dir is valid iff the digest matches. `*_smoke.npz` skipped; reacher files (none committed;
  the conditional-GO cohort ran absent) are OUT OF SCOPE and skipped by
  the loader — a future reacher ladder would be its own registration.
- **Loader guards (machine-checked gate — this read has no new labeling,
  so the registered gate is grid + dial identity, all asserted before
  any statistic)**: `FILE_RE`/`EXPECT_SEEDS`(31–38)/`LABELER_VERSION`/
  `validate_arrays`/`_cluster_boot`/`B_BOOT`/`RNG_SEED` are imported
  from the frozen `analysis/r3_read.py` with import-pin asserts (the
  `r3_reacher_read` pattern); plus: full 64-file grid all-or-nothing,
  duplicate trip, n_states = 200, dial identity from every file's meta
  (states 200, horizon 100, label_every 25, labeler seed 0, actions 8,
  rollouts 16, ref_stride 5, mass_scale 1.0, no behavior checkpoint),
  train_seed = filename seed, checkpoint = the named cell's
  `ckpt_early` (early) / `ckpt` (late), dose identity (e1 dim 0; e4
  dim 32 scale 3.0), array shapes (qfull (200,5,8), cands (200,8,A),
  deter (200,512)), finiteness, and m_now/m_real bounds.
- **Cells (frozen): {early, late} × {e1, e4}, domains POOLED — 16 run
  clusters per cell** (cluster = one training run at one maturity = one
  label file). Rationale: (i) maturity and dose are the registered R3
  axes along which repair targeting could differ; domain never was
  (domain splits were descriptive secondaries), and the R3 primaries
  pool domains; (ii) 16 clusters vs the old ladder's 6 stabilizes both
  the cluster CI and the nested λ selection (16 outer LORO folds);
  (iii) splitting by domain would halve clusters to 8 and double the
  look count. Fits and standardization never cross cell boundaries;
  per-domain rung splits are NOT computed (keeps the look count at the
  registered 48).
- **Rungs, per cell × target** (frozen in the reader):
  - **L0** constants: always (pooled per-state mean, run-clustered
    bootstrap CI via the imported `_cluster_boot`) vs never (0);
    best-constant = max of the two.
  - **L1** = LORO OLS on F0 = the 5 `d0.signals.compute(qfull)` scalars
    (uq, aflip, evpi_plugin, evpi_split, gap).
  - **L2** = LORO OLS on F1 = F0 + udyn + logdens + udyn_resid.
    logdens = −ln(mean distance of the k=10 nearest reference latents,
    per-run standardized deter space, same-episode points within ±10
    steps excluded) — reimplemented locally following the
    `gate_d1_r2_read.knn_logdens` definition; udyn_resid = within-run
    OLS residual of udyn on [1, logdens].
  - **L3** = nested-LORO ridge on [deter (512, standardized) + F1],
    intercept unpenalized (centered-target ridge on standardized
    features); λ ∈ {1e1, 1e2, 1e3, 1e4} chosen PER OUTER FOLD by inner
    leave-one-training-run-out mean Spearman (never sees the held run).
  - **L4 (NEW — the candidate-aware rung)** = the same nested-LORO
    ridge on [deter + F1 + G], where **G is the frozen 7-feature
    candidate-geometry / Q-dispersion set** computed from qfull
    (N,5,8), cands (N,8,A), m_now only — never from g_all/g_now/m_real
    (no target leakage; m_real is part of the gap estimand and is
    excluded): q_cand_std (std over candidates of head-mean Q),
    q_cand_range (max−min of head-mean Q), hspread_argmax (across-head
    std at the head-mean argmax candidate), hspread_mean (mean
    across-head std over candidates), act_pd_mean and act_pd_max
    (mean/max pairwise Euclidean distance among the M candidate action
    vectors), act_greedy_dist (‖cands[argmax head-mean Q] −
    cands[m_now]‖). The head-mean top-2 margin is deliberately NOT in
    G — it is already in the L4 stack as F0's `gap`
    (`d0.signals.advantage_gap`).
  - L5 (privileged physical state) remains not computable from stored
    labels; noted as a gap, not silently skipped (carried disclosure).
- **REGISTERED SCOPE CHANGE (explicit, never silent)**:
  `PREREG_d1_ladder_20260723.md` froze "no other features
  (candidate-geometry excluded for scope)". That exclusion is LIFTED
  here, for this registration only: repair targeting needs to know
  whether gap legibility lives in the candidate set (consumer side) vs
  the belief state (representation side), and the corrected labels are
  the first instrument that can answer it. The old registration is
  superseded for this read; nothing else in it is reopened (its labels
  and outputs stay invalidated).
- **Metric per rung**: held-out pooled Spearman + run-clustered
  percentile bootstrap CI, B = 10K, `default_rng(0)` (resample the 16
  clusters, recompute pooled rho on fixed cross-fitted predictions —
  no refit inside the bootstrap). The [L4−L3] contrast is
  bootstrap-PAIRED: one joint pass draws each cluster resample once and
  scores both rungs on the same draw (with fresh `default_rng(0)` and
  equal cluster counts, the marginal L3/L4 CIs are identical to their
  single-rung bootstraps; the selfcheck asserts the degenerate
  same-predictions case gives an exactly-zero contrast with zero-width
  CI).
- **Reuse decisions (registered)**: `analysis.gate_d1_read` /
  `gate_d1_r2_read` are NOT imported — the r2 loader pins the defective
  labeler `r2_ext_20260722` and gd1 carries the invalidated program's
  decision thresholds; coupling this reader to either module is an
  instrument hazard. The Spearman+cluster-CI metric (the only part of
  `criteria()` this design uses; lift and net-gain are Δ-target
  quantities with no meaning for opp/gap targets) and `knn_logdens` are
  minimally reimplemented instead, with identity legs in the selfcheck
  (_rho ≡ `scipy.stats.spearmanr` including ties). Shared-frozen
  machinery that IS safe (`r3_read`'s grid constants, guards,
  `_cluster_boot`, `cell_stats`) is imported with pins.
- **Floor policy (carried from R3, frozen)**: floor label files
  (all-zero G) are INCLUDED — their per-state targets are exact zeros,
  which is real signal for a rank metric — and each cell reports its
  floor fraction; committed domain-level fractions (cup 25%, finger
  18.75%) imply no cell can be floor-dominated.
- **Ordering**: freeze-commit this file + `analysis/r3_ladder_read.py`
  → ONE read:

```
python -m analysis.r3_ladder_read \
  --labels local_results/r3_competence_20260729_105146/labels \
  --output artifacts/r3_ladder_20260730
```

(the reader's substrate gate does the byte verification itself — the
frozen `COMMITTED_LABELS_DIGEST` must match or the read refuses; the
RCC labels dir is an equally valid `--labels` iff it matches.)

## Power (pre-sized from named committed sources)

- **Machinery noise floor**: the frozen reader's own selfcheck runs the
  untrimmed B=10K code path on synthetic grids at EXACTLY the
  registered geometry (16 clusters × 200 states per cell) — null cells
  are machine-checked to stay below |ρ| = 0.2 at every rung, and
  planted signals of each class are machine-checked to clear their
  thresholds (deter→L3/L4, scalar→L1/L2, candidate-geometry→L4-not-L3).
  The signature thresholds (0.2 / 0.1) are carried verbatim from
  `PREREG_d1_ladder_20260723.md`, where they sat above the same
  machinery's noise floor at only 6 clusters; 16 clusters is strictly
  more conservative.
- **Target scale**: the committed R3 read
  (`artifacts/r3_competence_20260729/r3.json`) gives pooled gap +1.293
  CI [+0.723, +1.987] at 32 clusters — the between-run spread that the
  per-state ladder must beat to be interesting is large and known;
  per-state predictability is exactly the unknown this ladder measures.

## Registered non-decisional heuristics (frozen in `analysis/r3_ladder_read.py`)

Per (cell, target):

- **COMPRESSION SIGNATURE** iff L3 Spearman ≥ 0.2 with cluster CI > 0
  while L2 Spearman < 0.1 (carried verbatim from the 23-Jul ladder).
- **CANDIDATE SIGNATURE** (new) iff the paired [L4−L3] contrast ≥ 0.1
  with paired cluster CI > 0 AND L4 Spearman ≥ 0.2 with cluster CI > 0
  (the margin 0.1 mirrors the compression rule's L2 bar; the L4-level
  clause prevents a "signature" made of two noise rungs).

Both are labeled descriptive heuristics, not gates. Their ONLY
consequence is a RESOURCE one: they inform TARGETING of the
competence-repair intervention registered the same day
(`PREREG_competence_repair_20260730.md` — sibling reference, no
dependency in either direction; each read executes and stands alone):
candidate signature → consumer-side repair; compression-only →
read-out/representation-side repair; no signature → the repair
registration's default targeting stands.

## Consequence map (frozen)

- **Candidate signature (any cell)** ⇒ the per-state gap is legible in
  candidate-set information beyond the belief state; repair targeting
  points at the consumer (resource only). Paper 2 gains a
  "where the gap is legible" descriptive section.
- **Compression signature only** ⇒ legibility lives in the belief state
  but not scalars; read-out-side repair targeting (resource only).
- **No signature anywhere** ⇒ the implementation gap is not
  state-legible at this information level: a registered descriptive
  boundary (the gap is a policy-level constant, not a per-state
  opportunity map); the repair registration proceeds with its default
  targeting.
- **Regardless**: the 24+24 stay frozen; Route A stays closed; imag
  stays retired; the R3 primaries (P-R3a/b/c) and every other
  registered verdict stand exactly as committed; no outcome here
  authorizes new runs or labels by itself.

## Costs

Zero new runs/labels. ONE local CPU read, ≈ 20–45 min single box
(nested ridge: 2 rungs × 8 cell-targets × 976 fits of ~527 dims on
~3000 rows; bootstrap on cached predictions ≈ 5 s per look × 48).
Selfcheck ≈ 10–15 min (three full-grid legs at the untrimmed B=10K).

## Disclosure

**Known at freeze**: also, at the freeze-commit date (31 Jul), the
executed R3 reacher read (**REPLICATES**,
`artifacts/r3_reacher_20260731/` — cell-level reacher quantities only;
it touches no cup/finger per-state quantity and no decision rule here).
Every committed cell-level quantity in
`artifacts/r3_competence_20260729/` — P-R3a pooled opportunity +1.247
[+0.700, +1.908] FIRES; P-R3b pooled gap +1.293 [+0.723, +1.987]
FIRES; P-R3c paired late−early achieved +0.014 [−0.177, +0.215] null;
dose margins (opp e1 +1.510, e4 +0.984), maturity margins (opp early
+1.193, late +1.301), per-domain splits (cup opp +0.982 / ach +0.065 /
gap +0.918; finger opp +1.512 / ach −0.157 / gap +1.669), floor
fractions (cup 25%, finger 18.75%). Also known: the ENTIRE
invalidated-era record (Gate-D1, R2 probe, 23-Jul ladder incl. its L3
outputs in `artifacts/d1_ladder_20260723/`) — all of it computed on
defective labels; it supplies NO prior on any corrected-label value and
is not re-read here.

**Unknown at freeze — and kept unknown by the build**: every per-state
feature→target statistic on the corrected labels (every rung Spearman,
every contrast, every signature, every per-(maturity×dose)-cell target
mean — the committed artifact reports margins, not the 2×2 cells; even
those cell means have never been computed). The reader's build touched
real label files for schema/integrity inspection only (field names,
shapes, dtypes, meta keys); its selfcheck runs on synthetic fixtures
exclusively; no estimand statistic of any kind was computed on real
data before this freeze.
