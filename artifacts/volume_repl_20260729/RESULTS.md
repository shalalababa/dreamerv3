# Volume-anomaly replication (P-E4a) + diversity secondary — READ 2026-07-29

Registered: `prereg/PREREG_volume_repl_20260726.md` (freeze commit
aaa31bc9, before any fresh-seed volume run existed). ONE read executed
cluster-side (`analysis/read/volume_repl.json` in bundle
`local_results/volume_repl_20260729_204854/`); diversity secondary
measured on the login node per the registered post-commit allowance
(bundle `local_results/spectral_diversity_pr_20260729_204814/`).

## Verdict

**P-E4a REPLICATES — and the diversity direction holds. Support
breadth (episode diversity) enters the theory as a λ-structure input
distinct from occupancy fraction; the P-E4b diversity leg engages.**

| quantity | estimate [95% CI] | n | status |
|---|---|---|---|
| **PRIMARY: fresh (seeds 7–12) v400s1 − v200s1** | **+331.0 [+169.3, +502.7]** | 6+6 | **FIRES** |
| pooled-12 (descriptive) | +279.8 [+150.9, +405.2] | 12+12 | — |
| original batch restated (descriptive) | +228.6 [+34.8, +414.1] | 6+6 | — |
| s0 fresh contrast (descriptive) | +194.8 [+65.2, +336.9] | 6+6 | — |

No winner's curse: the fresh-batch effect (+331) is LARGER than the
original (+229) — the anomaly strengthens out of sample. The power
disclosure said the 6+6 contrast detects ≈207 (calibrated to the
observed magnitude, the right bar for a curse check); the observed
fresh effect clears it comfortably.

## Diversity secondary (P-E4b operational form, registered direction)

Pinned instrument `spectral_v1_1_20260724` (version gate verified in
both outputs), run on the two frozen s1 buffers:

| buffer | diversity_pr | f_rewarded (occupancy) | episodes | spectrum_pr | r2_oof |
|---|---|---|---|---|---|
| q1v200/side1 | 9.230 | .323 | 200 | 44.90 | .399 |
| q1v400/side1 | **10.633** | .094 | 400 | 30.17 | .274 |

**diversity_pr(v400s1) > diversity_pr(v200s1)** — the registered
direction. Notably the generic whole-buffer spectrum_pr goes the
OTHER way (30.2 < 44.9): it is specifically the registered
episode-diversity index, not spectral richness at large, that
separates the buffers in the predicted direction — breadth, not
occupancy, is what the 400-episode buffer buys (its occupancy is 3.4×
LOWER yet it wins by +331). The "diversity fails while primary fires"
branch (which would have forced a non-spectral account) does NOT
trigger.

## Cohorts and integrity

- All-or-nothing cohorts complete: v200s1 seeds 1–12, v400s1 1–12,
  v400s0 1–12, v200s0 fresh 7–12 (first-ever cells for that buffer,
  occ .054). 42 volume rows, qc_pass 1/1, milestone 500000, finger
  only; zero volume-run exclusions (the 24 fresh jobs all completed).
- Reader `analysis/volume_repl_read.py` byte-identical to freeze
  commit aaa31bc9 (bundle snapshot ≡ repo HEAD); selfcheck PASS
  before verification; local re-execution on the bundled auc.csv
  **bit-identical** to the cluster json (only the recorded csv path
  differs).
- Independent spot restatement of the original batch from raw csv
  rows: +228.6, exactly the prereg's disclosed value.
- `probing/spectral_measure.py` bundle snapshot ≡ repo; occupancies
  from the instrument (.323/.094) match the registered calibration
  values.
- Provenance notes: the spectral bundle ships only the v400 stdout
  log (v200 log absent; the v200 json carries the full version gate
  and metadata, non-decisional); the v200 json's free-form `tag`
  reads `volume_q1v200_s1` vs `q1v400_s1` (cosmetic naming drift).
- The shared collector csv incidentally contains partial rows from
  in-flight U1–U3 and 25m runs; per the one-read protocol, no value
  from those modes was read or computed — only the registered volume
  modes were consumed (the frozen reader filters to them by
  construction).

## Frozen consequences applied

- The 18-Jul anomaly is REAL: support breadth becomes a λ-structure
  input in the theory, distinct from occupancy fraction; the flagship
  (Paper 3) gains its support-breadth axis.
- The full conditional-on-occupancy regression form of P-E4b remains
  UNREGISTERED (needs more buffers; a new dated registration if
  pursued).
- Rep-level exclusion from the 18-Jul read stands (the behavioral gap
  was the registered replication target and it replicated).
