# E2 / Gate F decision — 2026-07-11

Inputs: `local_results/e2_gatef_20260711_092716/adapt_scores/` (4 runs, all
`ADAPT_DONE`, 4/4 QC pass). AUC extraction re-derived locally with the frozen
`analysis/adaptation_auc.py` — bit-identical to the cluster-side
`analysis/auc_e2.csv`.

## Data

| run | AUC₁₀₀ₖ | final10 | note |
|---|---|---|---|
| adapt_goalwm_finger_seed1_ckpt100000 | 92.5 | 7.6 | oracle (goal-pilot WM) |
| adapt_goalwm_finger_seed2_ckpt100000 | 56.1 | 296.6 | oracle |
| adapt_goalwm_finger_seed3_ckpt100000 | 56.5 | 100.1 | oracle |
| adapt_p2elong_finger_seed1_ckpt500000 | 82.5 | 99.2 | 496K adapt steps (full budget probe); last-32 eps mean 31.5 |

## Frozen rule (addendum §E2, registered 7 Jul)

Oracle mean AUC₁₀₀ₖ vs μ+2σ of the frozen 5a finger rows
(`artifacts/phase5a_20260707/auc.csv`, n=75): μ = 83.41, σ = 24.21 →
threshold **131.83**. Oracle mean = **68.34 < 131.83 ⇒ the rule fires**:
finger is a reported-null domain *in the observational population*.

## Interpretation under the post-Axis-1 amendment (registered 10 Jul, before
any Gate-F outcome existed; results generated 11 Jul)

Axis-1 (unblinded 10 Jul) already established that the frozen-readout
instrument CAN express signal in finger: the high-occupancy controlled
buffer reaches 268 AUC₁₀₀ₖ / 388 final10. Gate F therefore resolves the
*source* of the 5a finger null rather than demoting the domain outright:

1. **Not a readout ceiling.** A buffer exists (ax1q1s1) under which the same
   instrument adapts.
2. **Not an adapt-budget ceiling.** `p2elong` gives the p2e WM 4× the
   registered budget (496K steps) and still ends ~99 final10 (last-32 mean
   31.5) — flat, no slow climb.
3. **Buffer deficiency, including for the natural oracle.** The goal-pilot
   WM (highest natural occupancy, 0.43, but only 100K env steps of online
   training) also fails the threshold — natural online buffers at this scale
   do not produce transferable finger WMs; the curated 200-episode/500K-update
   high-occ offline fit does.

**Disposition:** the 5a observational finger population remains
reported-null (per the frozen rule, unchanged). Finger continues as an
intervention-signal domain (Axis-1 Q1 positive; E3 dose grid + rpairs).
This is the exact split the 10-Jul amendment pre-registered.
