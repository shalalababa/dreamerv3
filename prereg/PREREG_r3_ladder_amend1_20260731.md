# PREREG AMENDMENT 1: R3 candidate-aware ladder — within-run rung metric (registered 2026-07-31, BEFORE the one read)

Parent: `prereg/PREREG_r3_ladder_20260730.md` (sha256
`320d2404143a4da5dfd0bf893b01ffdaf486c4432ea37501cc0d9df1142748c8`),
frozen at commit `6271c770` together with `analysis/r3_ladder_read.py`
(sha256 `ea4271d5359ed09a27ad5008f4eb20461bb0e502a5c52795fd97a3118f811251`).
**Status at this amendment: the parent's ONE read has NOT been
executed.** No per-state feature→target statistic on the corrected R3
labels exists anywhere: the review below was barred from real-label
contents (hashing/listing only), and every demonstration it produced
ran on synthetic data. The parent Disclosure section's "unknown at
freeze" list is still fully unknown.

## Trigger — independent post-freeze review (31 Jul)

An independent adversarial review of the committed bundle (two
reviewers with distinct lenses — registration↔code alignment and
statistics/leakage — plus one adversarial refuter per finding; 12
agents; run after the freeze-commit, before the read; the reviewers had
no access to the build-time review notes) CONFIRMED one blocking defect
in the registered metric, reproduced with the module's own
`crossfit_ols`/`crossfit_ridge`/`rho_boot` on synthetic grids at the
registered geometry (16 clusters × 200 states):

1. **LORO intercept artifact (mean reversal).** Every rung's held-out
   predictions carry the training runs' mean as intercept, so under any
   between-run target spread the held-out prediction LEVELS anti-order
   with held-run means (measured corr(held-run target mean, held-run
   prediction mean) = −0.998). Pooled Spearman on a pure within-run
   null with run effects of 0.5/1.0/1.5/2.0× the within-run sd:
   −0.235/−0.497/−0.608/−0.654, each with run-cluster bootstrap CI
   entirely below zero — "significant" associations on pure null.
   Conversely a genuinely legible scalar (held-out within-run rho
   +0.145, above the 0.1 bar) measured pooled L2 of −0.033 to −0.179 —
   satisfying the compression signature's L2 < 0.1 clause while scalars
   truly predict.
2. **Domain shortcut.** Cells pool cup+finger; deter trivially
   separates the two embodiments; the committed R3 read established
   domain-level target mean differences (gap finger +1.669 vs cup
   +0.918; opp +1.512 vs +0.982). A domain-separable deter plus a
   domain-level target offset yields positive pooled L3 rho with ZERO
   per-state signal (demonstrated +0.30, CI > 0), which together with
   the negative L2 bias from (1) fires the compression signature on
   nothing.
3. **Contrast miscalibration.** The paired pooled [L4−L3] CI excludes
   zero on pure noise under differential shrinkage (e.g. −0.033
   [−0.076, −0.004] at the real 520-vs-527 feature shapes).

The parent's own Question section defines the estimand as per-state
legibility versus "noise around a run-level constant" — the pooled
metric cannot distinguish these two registered branches. And the
heterogeneity regime is not hypothetical: it is committed as TRUE of
this data (R3 pooled gap +1.293, CI [+0.723, +1.987]; the domain means
above). Executing the read as-frozen would spend the one execution on
an instrument that cannot answer its own registered question.

## The change (one substitution; everything else stands)

**Metric per rung (L1–L4)**: the **mean per-run held-out Spearman** —
Spearman rho computed WITHIN each run (its 200 held-out predictions vs
its 200 targets), averaged over the cell's 16 run clusters — with the
run-resampled percentile bootstrap CI (resample the 16 clusters with
replacement, recompute the nanmean of the resampled per-run rhos on
fixed cross-fitted predictions; B = 10K, `default_rng(0)`). Within-run
ranking is invariant to run-level and domain-level additive target
structure and to the LORO intercept, so it estimates exactly the
registered estimand. **The paired [L4−L3] contrast** is the mean over
runs of (rho4_r − rho3_r), bootstrap-PAIRED on the same cluster draws
(one `integers(0, n, n)` draw per replicate shared by both rungs, so
the marginal L3/L4 CIs are identical to their single-rung bootstraps —
machine-checked degenerate-case leg carried over).

**Unchanged**: all fitting and cross-fitting (LORO OLS; nested-LORO
ridge, λ grid, per-outer-fold inner selection — the inner selection
already scored candidates within-run, so it is consistent with the
amended metric); features and rung definitions (incl. the frozen
7-feature G set); cells and pooling; targets; thresholds (0.2 / 0.1
bars); signature definitions and verdict strings (now evaluated on the
within-run metric); floor policy at the state level; B = 10K and
`default_rng(0)`; the substrate digest gate
(`COMMITTED_LABELS_DIGEST`); the look accounting (32 rung Spearmans + 8
paired contrasts + 8 L0 means = 48).

**Registered nan policy (now explicit — the parent left it undefined)**:
a run whose targets are rank-constant — exactly the all-floor label
files; committed file-level floor fractions (cup 25%, finger 18.75%)
imply ~3–4 nan runs per 16-cluster cell — has no within-run ordering to
predict: its per-run rho is nan and is excluded from the cell mean,
with the count reported per rung as `n_nan_runs`. Bootstrap draws
resample all 16 clusters and take the nanmean of the draw; a draw with
no live cluster yields a nan replicate, dropped by `nanpercentile`
(P ≤ (4/16)^16 ≈ 5e-10 per replicate at the committed floor fractions;
fail-safe — a nan CI can never satisfy CI > 0). Floor STATES inside
mixed runs remain included: their exact-zero targets are real
within-run rank signal (the parent's floor-inclusion rationale, now
applied at the only level where it measures per-state structure).

**Pooled demotion**: the parent's pooled Spearman is carried POINT-ONLY
per rung as `pooled_diagnostic` (and `pooled_diagnostic_diff` for
contrasts) — 40 explicitly NON-evidential diagnostic values, disclosed
in the artifact's `looks` block. No CI is computed for them: the review
demonstrated their cluster-bootstrap CIs lack nominal meaning under run
heterogeneity. No signature, threshold, or verdict reads them; they are
retained solely so the artifact documents what the parent's metric
would have reported.

## Selfcheck additions (machine-checked; also errata for the parent's Power section)

The review also CONFIRMED that the parent's Power section overstated
its selfcheck coverage: null cells were bounded only at L3/L4 (L1/L2
never asserted), the deter-plant leg never asserted L4 ≥ 0.2, and the
synthetic grids use deter dim 16, not the registered 512, with only
cluster-homogeneous targets. That paragraph is hereby corrected, and
the coverage gaps are closed with new machine-checked legs frozen in
the amended reader:

- **Run-heterogeneous null leg** (run effects 1.5× within-run sd):
  every rung bounded — |L1|, |L2| < 0.1 AND |L3|, |L4| < 0.2 — and no
  signature fires anywhere; on the SAME grid the pooled diagnostics
  document the mean-reversal bias (pinned assert: minimum pooled L2/L3
  ≤ −0.15; measured at build: within-run |ρ| ≤ 0.05 at every rung while
  pooled reaches −0.57).
- **Domain-offset null leg** (finger target offset 0.9× within-run sd +
  domain-separable deter): no signature, all within-run rungs bounded;
  the pooled L3 diagnostic documents the domain shortcut (pinned
  assert: ≥ +0.15; measured at build: pooled L3 +0.26 to +0.32 on zero
  per-state signal — the review's predicted magnitude — while
  within-run |ρ| ≤ 0.03).
- **Registered-dimension leg**: nested-LORO ridge at p = 512 (the real
  deter width) on a 16-run run-heterogeneous null — within-run metric
  bounded below 0.2, pooled diagnostic reproduces the negative bias at
  full dimension (pinned assert ≤ −0.10; measured −0.16 within +0.005;
  single rung-cell scale keeps selfcheck runtime sane — the metric's
  invariance is dimension-independent, the noise floor is what this leg
  pins at the real width).
- **Deter-plant leg** now also asserts L4 ≥ 0.2 with CI > 0 (deter is
  in L4's stack; the parent asserted only L3).
- **All-null grid leg** now bounds every rung including L1/L2, in every
  cell.
- **Floor-run nan accounting**: the planted grid's all-floor file
  yields `n_nan_runs` = 1 in its cell at every rung and in the paired
  contrast.

## Additional registered disclosures (folded from the same review; all refuter-CONFIRMED, none blocking)

- **`act_greedy_dist` is degenerate-by-construction on the real
  labels**: the labeler defines m_now := plugin choice = argmax over
  candidates of head-mean Q (`d0/oracle_labels.py:278–280`) — the
  identical argmax the feature recomputes — so the feature is
  identically 0 on every real file. L4 therefore carries 6 live G
  features plus one inert column (a constant column standardizes to
  zeros and contributes nothing to the ridge). The registered G set is
  UNCHANGED (this is a disclosure, not a redefinition); the read record
  must restate it.
- **`lambda_per_fold`** in the artifact JSON (per-outer-fold chosen λ
  for L3/L4) is carried d1_ladder fitting diagnostics — registered here
  as non-evidential metadata.
- **Digest-gate strictness**: `labels_digest` hashes ALL non-smoke npz
  in `--labels`, so any extra file (e.g. a later cohort's labels
  synced into the same directory) refuses the read even though the 64
  in-scope files are intact. This any-extra-file trip is intended
  registered behavior; the read runs against the freeze-time local
  bundle `local_results/r3_competence_20260729_105146/labels` (digest
  re-verified MATCH on 31 Jul: 64 files, `02efe6c5…`).
- The bootstrap's `nanpercentile` nan-drop policy, previously implicit,
  is now the registered policy stated above.

## Ordering & costs

This file + the amended `analysis/r3_ladder_read.py` (sha256
`aa776f1c58a813801f0451f1295a9b1110eb4bbcd8b58b353fbf0c2f36701791`,
selfcheck PASS on exactly these bytes, EXIT=0) are committed BEFORE the
read. Then the parent's registered command executes ONCE, unchanged:

```
python -m analysis.r3_ladder_read \
  --labels local_results/r3_competence_20260729_105146/labels \
  --output artifacts/r3_ladder_20260730
```

Costs: the read gets slightly CHEAPER (each bootstrap replicate
averages ≤16 per-run rhos instead of ranking 3200 pooled states);
selfcheck measured 7m08s wall including the p = 512 leg. Zero new runs or
labels, unchanged consequence map (resource-only repair targeting), and
nothing in the parent's consequence or disclosure sections is reopened
beyond what is stated here.
