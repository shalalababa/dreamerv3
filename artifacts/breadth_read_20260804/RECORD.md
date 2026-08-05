# Breadth-causal wave (q1d) — ONE read record (2026-08-04)

Registered ONE read of the matched-f_R diversity-pair wave via the
frozen reader `analysis/breadth_read.py` (sha 73cc1c24…), under
`PREREG_breadth_wave_20260803.md` (freeze commit `7bbf3eb1`),
adjudicating the pre-feasibility theory freeze
`PREREG_breadth_theory_20260803.md` (commit `481aced9`).

## VERDICT: **P-BD1 INFORMATIVE NULL** (registered branch, frozen consequence fires)

Reader verdict text (verbatim): "P-BD1 INFORMATIVE NULL: no breadth
effect detected at an instrument-verified separation of 4.84 PR units
— the lambda-input reading of the volume anomaly loses its principal
support (frozen consequence: the anomaly requires a non-breadth
account). P-BD2 ordering lo < v200s1 <= hi FAILS (253.7 / 187.7 /
243.9; weak point form, non-confirmatory)."

- **PRIMARY (P-BD1): hi-div − lo-div = −9.8 [−89.9, +72.2]** (AUC100k,
  8v8 seed-cluster bootstrap B=10K rng 0). Does not fire; not
  REVERSED. Breadth effects larger than +72 AUC are excluded at the
  97.5th percentile — the null is not vacuous: an effect on the scale
  of the composition contrasts is ruled out at this separation.
- **P-BD2 (weak ordering): FAILS — but per the REGISTERED reading rule
  (review MAJOR-6), a FAIL is UNINFORMATIVE about breadth.** Both
  curated cells sit far ABOVE the natural anchor (hi +56.2 [−13.7,
  +126.1]; lo +66.0 [−5.3, +137.3] descriptive) — the curated-draw
  offset direction and magnitude the f_R wave measured (+49.7/+56.9)
  reproduced on two fresh cells.
- **P-BD3 (gross-collapse screen): both cells MEMBER** (hi rew-NLL
  1.046 [0.923, 1.175]; lo 0.927 [0.849, 1.019]; both ≤ 1.5); no
  inclusion-leak, no hi-cell anomaly. The manipulation did not damage
  legibility — the null is not an artifact of a broken cell.

## Gates (all reader-enforced FATAL; all PASS)

f_lo 0.30358 / f_hi 0.30350 in [0.27, 0.34]; **|Δf| = 8e-05** (bar
0.02 — the f_R match is essentially exact); diversity_pr 7.4952 /
12.3360, **separation 4.8407 ≥ 3.0**, straddling the frozen reference
9.2300; **observable-support gate: proprio-block PR 3.0359 / 3.7718,
separation 0.7359 ≥ 0.5** — as-built values bit-consistent with the
registration round-trip pins; n_episodes == 200 both sides
(loader-enforced); version/domain/side pins hold.

## Consequences (frozen map + registered readings)

1. **The theory's λ/support-breadth input loses its principal causal
   support.** The correlational support-breadth finding (P-E4a) does
   NOT elevate to causal through the diversity manipulation: an
   instrument-verified 4.84-PR (full-index) / 0.74-PR
   (observable-block) separation at exactly matched f_R and volume
   produced no transfer difference. Frozen consequence: the volume
   anomaly requires a non-breadth account.
2. Combined with the f_R read (mediator never moved) the two curation
   waves now bracket the λ-axis from both directions: manipulating
   f_R did not move diversity; manipulating diversity did not move
   AUC. The causal ledger of the theory stands asymmetric:
   **composition/g-side causally supported (rgo sufficiency at
   proprio; rde competition-removal at pixel); λ/breadth-side = one
   registered causal null.**
3. Flagship factorial: the breadth axis now has neither a falling
   side (f_R saturation) nor a causal rise (this null) — it demotes
   to descriptive in the flagship design.
4. Registered scope of the null (disclosed at registration, review
   B1): the manipulated full index is latent-majority (trace shares
   0.38/0.47); the model-visible (proprio-block) dose was 0.736 PR —
   above the pre-registered 0.5 gate but a ~23% relative swing. The
   null is decisive against the full-index λ reading as registered;
   a hypothetical much-larger visible-block dose is outside this
   wave's reach (registered limitation, not an escape hatch).

## Post-read exploratory (labeled; verdict-untouchable)

- **The curation lift is now a 4-cell regularity**: curated 200-ep
  cells beat the natural pooled draw by ≈ +50 to +66 across wildly
  different f_R (0.30, 0.41, 0.73) and diversity (7.5–12.3) levels
  (f_R wave: +49.7/+56.9; this wave: +66.0/+56.2). Whatever curation
  does (e.g., MIN_REW_EPISODE floors dropping dead episodes), it is
  worth a registered account before any external claim touches the
  natural-vs-curated comparison.
- Per-seed spreads are large in both cells (120–370), consistent with
  the wide primary CI; nll_out descriptives tiny both sides
  (0.039/0.049).

## Provenance (verified before the read)

- Bundle `local_results/breadth_q1d_20260804_190158/`; committed
  manifest ≡ bundle manifest; **sha256 sweep 176/176 OK, 0 FAILED**.
- Run head `a2d19c72` — freeze `7bbf3eb1` is an ancestor; run head is
  an ancestor of local HEAD; tree clean except untracked slurm logs
  (including this wave's own gate/instrument job logs).
- Instrument shas byte-identical to frozen finals: breadth_read.py
  73cc1c24…, curate_frew.py f58a6513…, wave-prereg fc24f4e0…, theory
  prereg eb31b3a2… (local files re-hashed at read time).
- **Wave identity: the bundle's `q1d/div_pairs.json` is BIT-IDENTICAL
  to the registered pair (sha 48032a7598ac…ba0)** — the build consumed
  exactly the registered search output (the check-pairs core
  assertion, verified directly here).
- Cohorts complete: 16 fits + 16 adapts (bd modes, seeds 1–8 both
  sides; `adapt_ax1bdq1ds1_finger_seed8`, missing at the rde bundle's
  earlier AUC pass, is present — the job had simply not finished);
  frozen comparator = the archived volume_repl bundle's 12 v200s1
  rows; E4 csv complete 2×8.
- Note (not a deviation): the cluster-side preflight/HALT gate job
  logs were not bundled; every HALT condition is registered as
  reader-enforced (the B1/B2 review fixes made them fatal in the
  frozen reader precisely so gate evidence is re-derived at read
  time), and all passed here on the bundle's own as-built jsons.

## Read execution

```
python -m analysis.breadth_read \
  --auc local_results/breadth_q1d_20260804_190158/analysis/auc/auc.csv \
  --auc_frozen local_results/volume_repl_20260729_204854/analysis/auc/auc.csv \
  --spectral_lo local_results/breadth_q1d_20260804_190158/q1d/spectral_side0.json \
  --spectral_hi local_results/breadth_q1d_20260804_190158/q1d/spectral_side1.json \
  --spectral_ref artifacts/volume_repl_20260729/diversity_q1v200_s1.json \
  --block_lo local_results/breadth_q1d_20260804_190158/q1d/block_side0.json \
  --block_hi local_results/breadth_q1d_20260804_190158/q1d/block_side1.json \
  --e4 local_results/breadth_q1d_20260804_190158/e4/e4_finger_v1_bd.csv \
  --output artifacts/breadth_read_20260804/
```

Outputs: `breadth.json` (full, incl. per-seed cell values),
`read_stdout.txt`. Executed once, 2026-08-04.
