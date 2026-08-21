# PREREG — SE amplitude-DOSE ladder (Track B1), 21 Aug 2026

**Program:** Track B of `Plan_FollowupPrograms_20260821.md` (taxonomy
sequel; user GO 21 Aug "build everything"). ONE wave, TWO registered
reads: (i) the misprice-vs-amplitude curve — registering, on fresh
data, the direction the 12-run POST-HOC observation (θ₁ 5.26× at
scale 1.0 → ~7× at 0.359, review #25 rec-2 provenance) could not
claim; (ii) the exploration-cost curve = the cost-regime seed,
registered DESCRIPTIVE (a null is publishable robustness — this is
why the cost question was kept out of Paper 5).

## 1. Wave

Stage-1 config with **`DISTRACTOR_SCALE` ∈ {0.10, 0.25, 0.50, 0.75}**
× 4 seeds each = 16 runs × 5e5 steps, one wave, one site.
`disag_bootstrap` stays **True** (Stage-1 default — the dose axis
moves amplitude and NOTHING else; the bootstrap axis is NOBOOT's).
Seeds: 0.10 → 52–55 (`se_dose10_s*`), 0.25 → 56–59 (`se_dose25_s*`),
0.50 → 60–63 (`se_dose50_s*`), 0.75 → 64–67 (`se_dose75_s*`); all
disjoint from prior. **Full Stage-1 pin block on every run (review
#28 M1): task, expl.mode p2e, disag_ens 8, disag_scale 1000,
disag_bootstrap True + prob 0.8, planted.source_key position (= the
θ₁ DENOMINATOR) + basesd 0.0976, distractor.theta 0.1 (the OU
timescale — amplitude must be the ONLY thing that moves).** Per-run
`se_probe` post-training — **ALL 16 passes in ONE CPU job via
`scripts/se_dose_probe.sbatch` (#28 M7: dose ≡ seed block, so a host
split would confound device with the tested axis exactly).**
**`replay/` is REQUIRED in the dose bundle** (#28 B1; the coverage
leg degrades to REPLAY-UNAVAILABLE, never aborts). Existing cells
(Stage-1 at 1.0, n=8; flat at 0.359, n=4) are CROSS-WAVE DESCRIPTIVE
anchors only — never inside the test (1-Aug cross-site rule).

## 2. Registered reads (frozen reader `uncfield/se_dose_read.py`,
built + selfchecked BEFORE this compute; ONE execution)

- **DEGENERACY GATE** (NOBOOT pattern, adjudicated first): per-run
  REAL-dim attribution level + intrinsic_mean ≥ 1/10 of the live
  Stage-1 minima (real keys are dose-independent, so a level failure
  is instrument death, never dose) → else GLOBAL-COLLAPSE, trend not
  adjudicable.
- **PRIMARY (confirmatory):** θ₁(distractor) DECREASES in scale —
  Spearman ρ(θ₁, scale) < 0, one-sided negative label-permutation
  test (**100,000 vectorized draws** — #28 m1 — rng seed 20260821,
  ties by average ranks); **FIRES iff p ≤ .05 AND ρ < 0.** θ₁
  provenance-recomputed from each npz (the NOBOOT gate). **ESTIMAND
  SCOPE (#28 M5, review-verified on the shipped bundles): θ₁ is the
  RELATIVE, per-unit-data-variance misprice; the absolute decoded
  disagreement FALLS with falling dose (3.44× from scale 1.0→0.359)
  while its normalizer falls faster (4.53×) — the licensed sentence
  is "quieter fictitious channels are priced more favorably per unit
  of information they don't carry", NEVER "attract more
  disagreement"; the un-normalized decomposition is emitted per run.
  NORMALIZER-FLOOR clause (#28 M2, quantitatively resolved): the
  floor does NOT bind at dose 0.10 (2.3× margin, review-computed and
  bundle-validated), and a binding floor would DEFLATE low-dose θ₁ —
  conservative for this one-sided primary. Registered rule: per-run
  floor fields are emitted; a FIRE stands under a binding floor; a
  NON-CONFIRMED outcome with any floored run is NOT-ADJUDICABLE.
  PARTIAL PATH (#28 M6/B2): duplicates fatal; a missing/invalid/
  collapsed run routes to the flagged-subset primary (≥14 usable and
  ≥3 per dose required, dose balance reported); GLOBAL-COLLAPSE only
  at ≥2 collapsed runs.**
- **DESCRIPTIVE (registered, never fire):** per-dose θ₁ / coverage /
  level / SCORE-TAIL curves (score tail = mean of the final 100
  scores.jsonl entries — implemented, #28 M4; per-dose BCa at n=4 is
  INDICATIVE only, #28 m5); the COST estimand = the coverage proxy
  (pinned: Σ over position+velocity dims of log std over the ENTIRE
  replay; DISCLOSED as a new uncalibrated instrument — the score tail
  is its behavioral companion) + its Spearman vs scale, emitted
  unconditionally; per-key absolute levels incl. the θ₁ denominator;
  anchor-cell overlay. Grid note (#28 m9): the wave's top dose is
  0.75, so a NOT-CONFIRMED outcome cannot separate "no law" from
  "saturation above 0.75". Degeneracy premise softened (#28 m8):
  real-key levels are behavior-mediated and move ~±20% with dose; a
  10× collapse is instrument death — that is the gate.
- Identity gates: per-run scale on the registered grid to 1e-9;
  dose↔seed map enforced; dim/basesd pins; gates/mod off; bootstrap
  ON; duplicates fatal; fit counters.

## 3. Outcome map

- **DOSE-LAW-CONFIRMED**: "the relative misprice grows as the
  channel's amplitude falls" becomes a REGISTERED result — a
  registered column of the taxonomy (quieter fictitious channels are
  priced more favorably per unit of information they don't carry),
  with the candidate normalizer-vs-disagreement mechanism left open.
- **DOSE-LAW-NOT-CONFIRMED**: the post-hoc observation is demoted to
  wave-specific; the taxonomy keeps the curve descriptively.
- Validity cells as usual. The cost curve carries NO fire either way.

## 4. Placement & fences

Track-B paper (ICML 2027 if B1+B2+B3 land by Dec; else NeurIPS 2027);
cites Paper 5's arXiv. Paper 5 itself keeps only its already-drafted
one-sentence dose observation with the review-#25 rec-2 provenance
clause — this wave's results do NOT enter Paper 5.

Freeze = review #28 ADJUDICATED (2B/7M/10m ALL adopted 21 Aug: the
coverage-abort and missing-run-crash blockers, the full pin block,
floor accounting + estimand scoping, score tails implemented, partial
paths, the one-job probe driver, selfcheck hardening incl.
weak-trend/doctored-θ₁/pin-mutant fixtures; the reviewer resolved the
floor confound quantitatively and double-sourced the power analysis —
power ≥ 0.99 under the extrapolated law) + commit of: this file,
`uncfield/se_dose_read.py`, `scripts/se_dose_probe.sbatch` — then the
16 submissions.
