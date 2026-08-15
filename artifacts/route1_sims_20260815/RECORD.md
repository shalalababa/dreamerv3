# Route-1 simulation suite — record (15 Aug 2026)

**Status: NON-REGISTERED demonstrative simulation suite** (Paper-2
Route-1 upgrade, user GO 15 Aug; ledger entry "PAPER-2 ROUTE-1 EXECUTED
08-15"). Not a registered read: no estimand is measured on experiment
outputs — every quantity is computed on synthetic draws with known
ground truth (true effects zero unless stated). Kept in artifacts/ as
the reproducibility archive for the Paper-2 restructure's taxonomy /
corrected-estimator / diagnostic-battery numbers.

- Script: `analysis/route1_sims.py` (canonical; numpy-only, seed
  20260815, deterministic — same-seed re-run bit-identical; full run
  ~8 s CPU). Selfcheck 12/12 PASS: closed-form anchors (E[max₂ Z] exact
  1/√π; E[max₈ Z] 1.4236; EVPI tie 0.39894; plug-in K=5 exact-eval
  0.22053 and noisy-tie τ=1.066 → 0.3990 — both matching the verified
  2-Jul-2026 synthetic check), zero-bias legs, planted-leak mutant
  (power 0.98 / clean size 0.03), determinism gate.
- Results: `route1_sims_results.json` (this folder; produced by
  `python analysis/route1_sims.py --out route1_sims_results.json`).
- Interpretation + headline tables: gitignored drafting docs in
  `research_notes/paper2_evpi/route1/` (`Route1_Taxonomy_20260815.md`,
  `Route1_Sim_Results_20260815.md`, `Route1_Diagnostic_Protocol_20260815.md`).

## Headline results

- **E1 pedestal (channel A)**: bias of max-of-k noisy estimates matches
  σ/√m·E[max_k Z] at every (k, m) cell (k up to 64, m up to 16).
- **E2 reuse (channel B)**: k=8, m=4 — reuse +0.712 ± 0.002 vs fresh
  re-eval +0.006 (ns) and split-selection +0.004 (ns), true value 0.
- **E3 leakage (channel C)**: Spearman ρ(probe, outcome) at zero true
  signal = 0.234 / 0.480 / 0.727 at shared-variance fractions
  0.25 / 0.50 / 0.75 (clean probe ≈ 0). The archived duplicate-group
  ρ = +0.47 is consistent with ≈half the variance shared.
- **E4 composed, calibrated to the archived case** (k=8, m=1, σ=0.876
  calibrated to the archived identity value 1.247): pedestal +1.246
  ± 0.002 from a TRUE ZERO; ×2.01 under doubled noise (archived ×2.21);
  cross-fit collapse −0.007 ± 0.004 (archived −0.105). Level agreement
  is calibrated-in; shape/scaling/collapse are the evidence.
- **E5 corrections**: split/fresh ≈ 0 exactly; average-before-max
  ∝ 1/√m; analytic debias residual +0.055 ± 0.003 (σ̂ correlates with
  the in-pass max — quote the residual when this fallback is used).
- **E6 diagnostic battery (400 replicate studies)**: duplicate-candidate
  null size .045 / power .98 (50% shared variance); permutation-null
  referencing coverage .99 at nominal .95, single-study power .43 vs a
  planted 1σ-best (a reference correction, not a detector); cross-fit
  delta (parametric-bootstrap 95th-pct threshold at the study's own σ̂)
  size .000 (conservative) / power .89.

## Claim discipline

Demonstrations are "consistent with" the archived forensics, never a
claim that the archived numbers decompose exactly this way; no claim
about any specific published external result. Gaussian iid noise
throughout; heavy tails raise the pedestal (direction transfers,
magnitudes do not).
