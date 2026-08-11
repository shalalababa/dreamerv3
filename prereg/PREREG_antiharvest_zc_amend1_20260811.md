# AMENDMENT 1 to PREREG_antiharvest_zc_20260811 — selection-layer precision (2026-08-11)

Status at amendment: the ONE registered execution was attempted after
the freeze-commit and **crashed inside `load_cells` on the 4th file
alphabetically** (`w1tm2_cup_e1_seed52_late.npz`) at the review-B3
identity gate `sel_q ≡ m_now`. **Zero estimand values were output or
observed** — the process died at a load-time assert; one cell's
statistics (seed51) had been computed in-process but were never
displayed or written. This is a pre-outcome instrument repair,
disclosed here; the registered rules, estimand definitions, and grading
are UNCHANGED.

## Defect (reader, not data)

The labeler's `plugin_choice` (d0/oracle_labels.py:280) computed
`argmax(qfull.mean(heads))` on the **native float32** array; `m_now`
stores that result. The frozen reader upcast `qfull` to float64 before
evaluating the same expression. The two head-means differ at the ~1e-6
level (float32 summation rounding), and at knife-edge near-ties the
argmax flips. Census on the 32 registered files (diagnostic run,
2026-08-11):

- storage-precision (float32) identity `qfull.mean(1).argmax(1) == m_now`:
  **exact on 6400/6400 states**;
- float64-upcast path: **61/6400 states flip (0.95%)**, spread over 20
  of 32 files; every flip is a near-tie (float64 top-2 head-mean gap
  ≤ 1.9e-5, many exactly 0 in float64), i.e. the flips carry no value
  information — they are tie-breaks.

The registered honesty block says "sel_q ≡ m_now by construction"; the
construction is the labeler's float32 computation. The gate failed
because the reader did not reproduce the registered construction, not
because the identity is broken.

## Registered fix (this amendment)

1. `load_cells` keeps `qfull` in **storage precision** (no float64
   upcast). The identity gate is unchanged in FORM
   (`qfull.mean(1).argmax(1) == m_now`, must hold exactly on every
   state) and now runs in the construction's precision.
2. `cell_stats` computes the **selection layer** — head-mean `q`,
   head-std `qstd` (ddof=0 unchanged), `sel_q`, and P-Z3's
   `sel_p = argmax(q − λ·qstd)` — in the dtype of the `qfull` it
   receives (storage float32 at execution). `sel_p` was never computed
   anywhere before (no identity exists); its precision is pinned to
   match `sel_q` for coherence. All ESTIMAND arithmetic (g-side
   differences d_mode/d_qsel/d_rep, pooling, BCa, permutation) remains
   float64 as before — estimand VALUES are unaffected by this
   amendment except through tie-breaks at sub-2e-5 near-ties, which now
   resolve to the registered construction (m_now) instead of an
   unregistered float64 re-derivation. Spearman(q, g) is unaffected
   (float32→float64 is order-preserving; identical ranks).
3. Selfcheck extended with a **defect-class leg**: a planted float32
   near-tie fixture whose float64 argmax differs from its float32
   argmax, npz round-tripped, asserting the reader-path expression
   reproduces the planted m_now while the float64 path demonstrably
   flips.

No other change. The ONE execution follows this amendment.

Portable lesson (recorded): an identity witness against a stored
labeler decision must be evaluated in the precision the labeler used —
upcasting before an argmax is an instrument change, not a no-op.
