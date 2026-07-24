# Route-A R2 decision probe — READ (2026-07-23)

> **INSTRUMENT-INVALIDATED (2026-07-24).** The shared D1 real-operation
> labeler contained two confirmed implementation defects (stale-carry
> candidate branches: obs_t never assimilated, prevact = a_{t-1} not the
> candidate; and policy-RNG not common across branches), so delta_real /
> G labels realize an unintended estimand. The read below remains a
> faithful record of the registered procedure ON THOSE LABELS, but its
> decision-value conclusions cannot be interpreted as established.
> Details: analysis/DEVIATIONS.md 2026-07-24 deviation entry; repair +
> relabel registration: prereg/PREREG_d1_relabel_20260724.md.
> Audit addendum (same day): the e4 (dosed) cells carry an additional
> defect — the OU distractor's process state was not in the branch
> snapshot, so CRN was broken in the distractor observation dims
> (DEVIATIONS 24-Jul audit entry, item 2).


Registration: `prereg/PREREG_gate_d1_r2probe_20260722.md` + Amendment 1
(`prereg/PREREG_gate_d1_r2probe_amend1_20260723.md`, written BEFORE this
read executed). Frozen read: `analysis/gate_d1_r2_read.py` (unchanged
since freeze; bit-identical to the copy shipped with the results).
Labels: `local_results/d1_routea_r2_save60_20260723_103849/runroot_light/
d1_labels/` (24 files, full grid). Canonical output:
`gate_d1_r2.json` in this directory.

## VERDICT — R2 DOES NOT FIRE. ROUTE A CLOSED.

No cell reaches C1 (Spearman ≥ 0.3 with run-clustered bootstrap CI > 0)
under ANY of the 16 cell×op×feature-set combinations — the null is
rule-robust: it holds under the original 12-look rule AND the amended
6-look real-op rule (which is the registered decisional one). Per the
frozen consequence map: **Route A closed; Route B absorbs the package
(this probe = registered negative); the 24+24 D0 runs stay frozen
permanently.**

Best value in any predicted cell: early×e1 real F1 ρ = +0.085
[−0.009, +0.149] — a fifth of threshold, CI includes 0.

## Registered read (C1 per cell; threshold 0.3)

| cell (op, fset)        | ρ      | 95% CI            | C1  |
|------------------------|--------|-------------------|-----|
| early×e1 real F0       | +0.033 | [+0.000, +0.085]  | fail|
| early×e1 real F1       | +0.085 | [−0.009, +0.149]  | fail|
| early×e4 real F0       | +0.006 | [−0.012, +0.104]  | fail|
| early×e4 real F1       | −0.005 | [−0.042, +0.103]  | fail|
| late×e4 real F0        | +0.004 | [−0.040, +0.023]  | fail|
| late×e4 real F1        | +0.007 | [−0.022, +0.026]  | fail|
| late×e1 real F0 (bnd)  | +0.035 | [+0.003, +0.065]  | fail (predicted) |
| late×e1 real F1 (bnd)  | +0.026 | [−0.009, +0.057]  | fail (predicted) |
| imag cells (8)         | −0.032…+0.071 | all fail   | descriptive-defective |

- **Boundary-replication cell PASSES its check**: late×e1 real
  reproduces the Gate-D1 near-zero on this platform (predicted FAIL,
  failed; no anomaly flag). The local port is validated.
- Imag-op cells are DESCRIPTIVE-DEFECTIVE per Amendment 1 §A (op_imag
  candidate-misalignment defect, confirmed in code pre-read).
- C2/C3 descriptors fail everywhere as well (net gain at price 0 spans
  0 in every real cell; no lift CI > 1).

## The registered PREDICTION fails, not just the gate

The estimand relocation itself is falsified as specified: calibration
does not appear EARLY (ρ ≤ +0.085 at ~25.6K steps, either dose) and
does not appear under DOSE (ρ ≈ 0 in both e4 cells). The phase-change
story survives only at the error level (Stage-0); it does NOT
translate into per-state operation-value predictability anywhere on
the {phase × dose} grid with these summaries and this amortizer.

Per-feature Spearmans (real cells, |ρ| max over grid): uq .037,
aflip .035, evpi_plugin .041, evpi_split .041, gap .032, udyn .033,
logdens .057, **udyn_resid .029 — the Stage-1A density-residual asset
is a clean null as an ÊVSI feature (R1 adjudicated)**. `_boot`
companion labels move nothing (R3: CRN rollouts already near-
deterministic; boot column descriptive as registered).

## Population-level structure (descriptive; the interesting part)

Pooled mean Δ_real per cell (6 runs/cell):

| cell      | mean Δ_real | cup        | finger     |
|-----------|-------------|------------|------------|
| early×e1  | **−0.208**  | −0.095     | −0.320     |
| early×e4  | **−0.146**  | +0.010     | −0.302     |
| late×e1   | +0.068      | −0.083     | +0.220     |
| late×e4   | **+0.313**  | +0.300     | +0.327     |

- **Early real purchases HURT on average** (both doses). Post-hoc
  reading (not registered): op_real re-ranks candidates by
  r + γ·V̂(s′) with the agent's OWN value head; early, V̂ is unreliable,
  so the purchase actively misleads — the value of information depends
  on the consumer's competence to use it.
- **Dose RAISES mean purchase value late** (+0.313 vs +0.068) while
  per-state calibration stays at zero — the "value exists on average
  but no signal can see it" pattern again, now with a dose knob that
  moves the average.
- finger late×e1 +0.220 replicates Gate-D1's finger +0.222 across
  platform and seeds; cup flips sign (−0.083 vs +0.083) — consistent
  with the large between-run heterogeneity flagged in the external
  review (cup run means here: −0.19, −0.06, 0.00).
- change_rate_real ≈ 0.85 in all cells — the operation almost always
  changes the chosen action; the null is not "the op does nothing".

## Integrity & provenance

- All 24 label files pass the frozen asserts: pinned dials,
  labeler_version `r2_ext_20260722`, dose meta (e1 dim 0; e4 dim 32
  scale 3.0), CRN identity, early steps ∈ [10K, 40K] (realized:
  25,648–25,664; |error| ≤ 664), n = 200 states/file.
- Single platform: all 12 pilots + smoke + 24 label passes on one CUDA
  instance (`/workspace` runroot `d1r2_r2probe_save60`); commit
  `a5079f42`. Instruments bit-identical repo ↔ results `code/` copy
  (read, labeler incl. the two transfer-guard patches `fd66addb`
  `f2798309`, driver, prereg).
- Attempt 1 discarded outcome-blind (early snapshots ~41.4K, outside
  registered window; step metadata only); full-grid rerun = this data.
  Amendment 1 §B–C + DEVIATIONS 23-Jul entry hold the process record.
- Smoke npz present in the labels dir; excluded by the frozen filename
  rule.
- Known design limitation (disclosed post-hoc, from the external
  review): early/late cells label states from each checkpoint's OWN
  on-policy distribution, so checkpoint age is confounded with state
  distribution; and the e4 doses are reward-irrelevant nuisance by
  construction (Amendment 1 §D). Neither rescues the null — the
  nuisance-free early×e1 cell is as dead as the rest.

## Consequences

1. **Route A CLOSED** (registered). Paper 2 = Route B, absorbing:
   attractor mechanism + phase change + cross-policy boundary +
   Stage-1A density-residual (error-level) + Gate-D1 fail + this probe
   as a broad registered negative + the population-level Δ structure
   above.
2. **R1 closed** (density-residual is not an ÊVSI feature), **R3
   closed** (labels are effectively deterministic; boot column adds
   nothing), R4/R5 moot as Route-A moves.
3. Gate-D1's imag cells reclassified uninterpretable (Amendment 1 §A);
   any future use of the imagined op requires the fixed-action or
   expanded-search repair first.
4. Plan revision for Paper 2: `research_notes/other research/
   EVPI_Plan_Revision_20260723.md` (assessment of
   `D1_GPT_Analysis_20260723.md` folded in).
