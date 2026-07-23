# REGISTERED-DESCRIPTIVE ADDENDUM: D1 actionability ladder (2026-07-23)

Paper-2 / Route-B addendum on the EXISTING R2 labels — no new runs, no
new labeling. Registered per plan v3
(`EVPI_Plan_Revision_20260723.md` §2, "actionability ladder", adapted
from the external review's proposal). This file + the frozen script
`analysis/d1_ladder_read.py` are committed BEFORE the ladder's new
quantity (belief-state-level predictability) is computed.

## Question

Which failure class does the D1/R2 calibration null belong to?

- **Statistic failure / representation-compression**: the full belief
  state predicts Δ_real but the scalar summaries do not (⇒ a
  constructive "build a better uncertainty read-out" result; would
  authorize designing the support-diverse-ensemble arm).
- **No exploitable heterogeneity**: even the belief state cannot beat
  the best constant allocation (⇒ the negative is fundamental at this
  information level; strengthens the boundary framing).

## Disclosure of what is already known at freeze

The R2 read (`artifacts/gate_d1_r2_20260723/`) is complete: all
scalar-level (F0/F1) cell Spearmans are KNOWN (≈ 0 everywhere, max
+0.085), as are mean-Δ descriptives. The ladder's L1/L2 rows
re-present those known numbers for the table. The ONLY new quantity is
L3 (belief-state ridge), unknown at freeze. Imag-op labels are
excluded (defective per Amendment 1 §A); real op only.

## Pinned design

- Data: the 24 R2 npz files; cells {early, late} × {e1, e4}, domains
  pooled, 6 runs/cell; target `delta_real`.
- **L0** constants: always (mean Δ_real, cluster CI) vs never (0);
  best-constant = max of the two.
- **L1** = LORO OLS on F0 (5 scalars) — frozen R2 machinery, known.
- **L2** = LORO OLS on F1 (8 scalars) — frozen R2 machinery, known.
- **L3** = LORO RIDGE on [deter (512, standardized) + F1 scalars],
  intercept unpenalized (implemented as centered-target ridge on
  standardized features); λ ∈ {1e1, 1e2, 1e3, 1e4} chosen PER OUTER
  FOLD by inner leave-one-training-run-out mean Spearman (nested;
  never sees the held-out run). No other model class, no other
  features (candidate-geometry excluded for scope; privileged physical
  state not stored).
- Metrics per level per cell: the frozen Gate-D1 `criteria()` outputs
  (held-out pooled Spearman + run-clustered bootstrap CI, lift, net
  gain at price 0), B = 10K, rng seed 0 — unchanged machinery.
- L5 (privileged state) not computable from stored labels; noted as a
  gap, not silently skipped.

## Registered interpretation heuristic (NON-decisional, disclosed)

"Compression signature" in a cell iff L3 Spearman ≥ 0.2 with cluster
CI > 0 while L2 Spearman < 0.1. This is a labeled descriptive
heuristic, not a gate: its only consequence is a RESOURCE one
(authorizes designing the support-diverse-ensemble arm; plan v3 §2).
No outcome of this ladder unfreezes the 24+24, revives Route A, or
changes any registered verdict.

## Ordering

1. Freeze-commit this file + `analysis/d1_ladder_read.py`.
2. `python -m analysis.d1_ladder_read --labels <R2 labels dir>
   --output artifacts/d1_ladder_<date>`.

Compute [prov. estimate]: minutes on CPU (ridge 520 dims × 1K rows ×
4 λ × nested LORO × 4 cells; bootstrap on cached predictions).
