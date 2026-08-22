# PREREG — Track-B onset ladder (the amplitude-onset of the misprice), 22 Aug 2026

**Program:** Track B (user GO 22 Aug "do all the builds"; revised
same-day per review R1 BEFORE any compute or freeze — grid moved
onto the transition window, wave layout round-balanced, reader
gates made fatal; all revisions pre-outcome, value-aware only in
that they use the already-executed B1/A1 reads, disclosed below).
The B1 dose read (`artifacts/b1_dose_read_20260822`) refuted the
registered falling-dose law and revealed, DESCRIPTIVELY, a
threshold-onset hump: θ₁ ≈ 0.11 at scale 0.10 (under-priced),
BIMODAL at 0.25 (0.09/0.13 vs 1.52/1.55 — two runs off, two near
parity), PEAK 8.41 at 0.50, 6.56 at 0.75, anchors 0.359→~7.1 and
1.0→5.26. Those anchors place the rise essentially complete by
≈0.36 and put the seed-level phase transition at ≈0.25. This wave
registers the RISING LIMB on fresh data ACROSS that window — the
direction B1's post-hoc curve cannot claim.

## 1. Wave (16 runs, seeds 96–111, one wave/site, 5e5 steps)

Stage-1 config with `DISTRACTOR_SCALE` ∈ **{0.25, 0.30, 0.34,
0.38}** × 4 seeds each: 0.25 → 96–99 (`se_ons25_s*`), 0.30 →
100–103 (`se_ons30_s*`), 0.34 → 104–107 (`se_ons34_s*`), 0.38 →
108–111 (`se_ons38_s*`). **0.25 is simultaneously a fresh-seed
replication of B1's bimodal cell** (registered as such: its
crossing fraction tests whether the 2-on/2-off split is a
reproducible seed-level phase transition). Everything else
Stage-1-verbatim (the B1 pin block applies unchanged: p2e/ens
8/scale 1000/bootstrap True+0.8, planted position/0.0976, dim 8,
basesd 1.215, theta 0.1, gates/mod off, penalty inert). **Wave
layout (R1-B1): one instance, 8 lanes × 2 rounds, declaration
order in `ops/waves/b_onset/spec.yaml` fixed so each ROUND holds
exactly two runs of every scale and each LANE holds two different
scales — the primary is monotone in scale, so a round≡scale layout
would let any monotone wall-clock drift reproduce it.** Per-run
`se_probe` post-training — ALL 16 in ONE CPU job
(`scripts/se_onset_probe.sbatch`, the B1 device rule) **with the
ckpt-identity skip rule + newest-DIRECTORY resolution + refuse on
missing `ckpt/latest` (the 22-Aug B1 s53 lesson)**. `replay/`
REQUIRED in the bundle (coverage leg). B1's cells and the
Stage-1/flat anchors are CROSS-WAVE DESCRIPTIVE only (1-Aug rule).

## 2. Registered read (frozen reader `uncfield/se_onset_read.py`,
built + selfchecked BEFORE this compute; ONE execution)

- **DEGENERACY GATE** first (B1 pattern): per-run REAL-dim level +
  intrinsic ≥ 1/10 of the gated Stage-1 minima
  (`stage1_reference_gated`, reference = the pinned
  `uncfield_se_stage1_20260813_193324` runs, n ≥ 4 asserted).
- **PRIMARY (confirmatory):** θ₁(distractor) INCREASES in scale
  across the fresh 16 — Spearman ρ(θ₁, scale) > 0, one-sided
  POSITIVE label permutation (100,000 draws, rng seed 2026_08_22,
  average ranks); **FIRES iff p ≤ .05 AND ρ > 0.** θ₁
  provenance-recomputed per run (the B1 gate). ESTIMAND unchanged
  from B1 (#28 M5): θ₁ is the RELATIVE per-unit-data-variance
  misprice; per-run floor fields emitted; floor accounting mirrors
  B1 with the direction FLIPPED (a binding floor DEFLATES low-scale
  θ₁ — ANTI-conservative for this positive primary): a FIRE with
  any floored run among the LOADED set is NOT-ADJUDICABLE; a
  NON-CONFIRMED outcome stands under a binding floor. A degenerate
  (constant-θ₁) panel is NOT-ADJUDICABLE, never a fire.
- **CLAIM SCOPE (R1-B4):** the fire licenses ONLY "the rising limb
  replicates on fresh data." A positive Spearman cannot separate a
  threshold from a smooth monotone rise — the SHAPE stays
  descriptive (below), and no "threshold"/"switches on" wording is
  licensed by the primary alone.
- **Identity gates are FATAL (R1-B3):** any gate failure (grid,
  scale↔seed map, pin block, θ₁ provenance, final-ckpt,
  newest-ckpt-dir, duplicates) REFUSES the whole read — the B1 s53
  incident path: operator repairs/re-probes, then re-executes (the
  one-execution guard survives because refusal precedes output).
  The partial path (flagged subset, ≥14 usable, ≥3/scale) covers
  runs MISSING from the glob, never gate-failing runs;
  `missing_seeds` is computed against the registered 16 and any
  subset is FLAGGED in the primary statement. Fit counters are
  reported per run and a non-OK flag ANNOTATES the primary
  statement (disclosure-carried, 7-Aug standing rule — a registered
  decision, not an exclusion). Checkpoint provenance is RECORDED
  per run (`ckpt_final_ok`, `newest_ckpt_checked`) and aggregated
  (`probe_ckpt_flag`), so the artifact shows whether the gates were
  live or vacuous. The reader dumps its full argv into the output.
- **SECONDARY (registered, no fire):** the transition profile —
  per-scale θ₁ rows; **crossing fraction** frac(θ₁ > 1) per scale
  (0.25 = the B1-bimodality replication cell; the window predicts
  an intermediate crossing fraction somewhere inside); a
  descriptive logistic midpoint of θ₁ > 1 vs scale — grid MLE over
  BOTH slope signs with `boundary_pinned` / `in_data_window` /
  `monotone_direction` flags and a run-resample percentile interval
  (R1-M6/M7), never a fire; per-scale BCa (indicative, n=4,
  per-scale rng streams); **coverage proxy + score tails per scale
  delivered in the per-scale rows** plus Spearman(coverage, scale)
  with the shared helper's left-tail p, descriptive (R1-M8; B1's
  registered coverage trend was −0.728); un-normalized
  decomposition per run.

## 3. Outcome map

- **ONSET-CONFIRMED**: the rising limb is REGISTERED on fresh data
  across the transition window. Combined with B1's
  registered-descriptive under-pricing at 0.10–0.25, the taxonomy's
  amplitude axis gains a REGISTERED RISING LIMB; the transition
  SHAPE (threshold vs smooth) remains a descriptive question
  carried by the crossing fractions and midpoint — any
  "threshold-onset" wording in papers must cite those as
  descriptive, not this fire.
- **ONSET-NOT-CONFIRMED**: the hump was wave-specific structure OR
  the limb is flat inside [0.25, 0.38]; the taxonomy keeps both
  curves descriptively; grid note — a null inside [0.25, 0.38]
  cannot exclude a transition outside it.
- Validity cells as usual (NOT-ADJUDICABLE / GLOBAL-COLLAPSE / the
  flipped floor rule).

## 4. Fences & placement

Track-B taxonomy paper. Paper 5 keeps only its (reworded, per the
B1 duty) one-sentence dose observation. No Paper-5 content depends
on this wave.

Freeze = review R1 adjudicated (done 22 Aug — 4 BLOCKING + 5 MAJOR
+ 12 minor, all resolved pre-freeze; grid/layout/reader revised
BEFORE compute) + commit of this file + `uncfield/se_onset_read.py`
+ `scripts/se_onset_probe.sbatch` + `ops/waves/b_onset/spec.yaml` —
then the 16 submissions (post-A2-freeze checkout; BIND-CHECK live).
