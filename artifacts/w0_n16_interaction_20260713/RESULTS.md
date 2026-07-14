# W0 registered read — fully-paired n=16 interaction (Amendment 2 §A) + buffer battery

Read executed 2026-07-13 against snapshot
`local_results/w0_finger_q1_apt_s9_16_20260713_223349/` (16 apt finger-Q1
adapt runs, seeds 9–16 × sides {0,1}) per
`prereg/PREREG_axis1_corrective_amendment2_20260713.md` §A, frozen before
these outcomes existed. Analysis machinery identical to the P0 read
(`analysis.adaptation_auc` frozen extraction; cluster bootstrap B=10,000,
`np.random.default_rng(0)`, percentile CI; script `n16_interaction_read.py`
in this directory). Seeds-1–8 deltas bit-checked against the frozen paired
JSONs in `artifacts/p0_axis1_corrective_20260713/paired/` and
`artifacts/phase6_axis1_20260710/paired/` (asserts pass).

## Audit (Amendment 1 §C per-run rule) — PASS 16/16

- Saved `config.yaml: expl.mode == apt` in all 16 WM fits
  (`ax1wm_finger_fq1s{0,1}_seed{9..16}`).
- Fit loss lines in all 16 job logs contain only observation-decoder +
  dynamics losses (`con, dyn, rep, position, velocity, touch,
  target_position, dist_to_target`), byte-identical component set to the
  audited P0 seed-1–8 finger logs; **no `rew`/`value` term anywhere in the
  16 logs** (whole-file grep).
- QC: 16/16 adapt runs complete (`ADAPT_DONE`, 112 episodes, last step
  112,112 > 100K — identical coverage convention to the P0 batch);
  16/16 pass the frozen AUC QC, 0 exclusions.

## Primary confirmatory (decision on this CI alone) — INTERACTION CONFIRMED

Finger Q1 occupancy × reward-supervision interaction, per-seed
Δ_task − Δ_apt on AUC₁₀₀ₖ, fully paired over seeds 1–16:

| quantity | value |
|---|---|
| mean | **+98.72** |
| cluster-bootstrap 95% CI (B=10K, seed 0) | **[+45.81, +158.83] — excludes 0** |
| positive seeds | 13/16 |
| exact sign-flip permutation (2¹⁶) | p = 0.00272 |
| Wilcoxon | p = 0.00269 |
| paired t CI | [+35.2, +162.2] |
| d_z | 0.83 |
| leave-one-seed-out mean range | [+81.0, +110.4] |

Per-seed interaction deltas (seeds 1–16): +106.8, +364.1, +89.5, +300.8,
+2.6, +218.0, +141.2, +35.2 | −20.0, +77.3, +61.4, −76.3, −47.1, +106.9,
+63.9, +155.3.

**Registered subsidiary (labeled):** fresh-batch seeds 9–16 interaction
+40.17 [−12.77, +90.68], 5/8 positive, perm p=0.203 — directionally
consistent, attenuated relative to seeds 1–8 (+157.28), exactly the
winner's-curse pattern disclosed in the P0 read. The attenuation is
entirely in the aware arm (cells 267.9→163.1 / 108.6→132.4); the apt arm
is flat in both batches. Honest effect-size estimate is the pooled
+98.7 [+45.8, +158.8], not the seeds-1–8 +157.

## Secondary — reward-free simple effect is now a tight null at n=16

Pooled seeds 1–16 apt (s1−s0): **−3.71 [−16.97, +9.76]**, 8/16 positive
(fresh batch alone: −9.46 [−22.01, +2.69], 3/8). Cell means: apt-lo 78.9 /
87.4, apt-hi 80.9 / 78.0 (batches 1–8 / 9–16) — all four apt cells sit on
the ~80 online-pretrain floor. The P0 headline null (+2.04 [−22.0, +24.2])
tightens by ~2× in CI width. High-occupancy buffers do nothing for
reward-free world-model fits in finger Q1; the benefit exists only under
reward-capable objectives (aware simple effect pooled 1–16:
+95.02 [+44.20, +152.91], 12/16).

## Buffer battery (descriptive; hi sides only — lo invocations were not run)

`battery/` holds the four pair JSONs from
`local_results/buffer_battery_20260713_224257/`. The log contains only
`hi:` lines; per-side occ/cov/mixture for the lo sides was recovered from
the build manifests. **Re-run request:** the same battery command on the
four `side0` (lo) directories to complete dwell/diversity/reward stats
(login node, minutes).

| pair | hi occ | hi rew-frac | occ↔rew corr (ep-level) | entries/ep | dwell mean | hi sources (eff. n) | lo sources |
|---|---|---|---|---|---|---|---|
| finger q1 | 0.323 | 0.323 | **+1.00** | 3.6 | 91.2 | 19 (15.2) | **1 (p2e3 only)** |
| finger q2 | 0.101 | 0.101 | +1.00 | 1.9 | 53.0 | **1 (apt1 only)** | 1 (p2e2 only) |
| cup q1 | 0.173 | 0.172 | **−0.51** | 25.1 | 6.9 | 19 (11.9) | 1 (p2e2, 95%) |
| cup q2 | 0.100 | 0.507 | **−0.70** | 13.7 | 7.3 | 18 (9.1) | 1 (apt3, 98%) |

Three mechanistic findings:

1. **In finger, occupancy ≡ reward density** (episode-level correlation
   1.0; reward-frame fraction equals occupancy fraction). The confirmed
   interaction is literally *reward-dense data × reward-capable
   objective* — the "hidden curriculum" of the Q1 buffer is reward
   exposure, not regime dynamics per se. This sharpens the P3
   (objective–data alignment) framing and makes the label-relocation
   control in the gradient-path factorial the critical arm (it separates
   reward-as-supervision-signal from reward-correlated states).
2. **In cup the correlation flips sign** (−0.51 / −0.70; cup q2 hi has 51%
   reward frames at only 10% occupancy). High-occupancy selection dilutes
   reward in cup — a clean account of why the interaction is
   finger-specific (cup q1 interaction was −104.9 ns at seeds 1–8) and why
   cup's directional reward-free benefit, if real, cannot be
   reward-mediated. Regime structure also differs in kind: finger = few
   long regime bouts (dwell ~91 steps), cup = many brief contacts
   (dwell ~7 steps).
3. **Constructional confound in the original Axis-1 buffers:** every lo
   side is mono-source (200 episodes from a single collector) while three
   of four hi sides are ~19-source mixtures. Source diversity co-varies
   with occupancy. The apt-arm flatness bounds the reward-free effect of
   diversity at ≈0, but within the task arm these buffers cannot separate
   occupancy/reward-density from source diversity. **E3v2
   (within-collector pairs, both arms — Amendment 2 §B, submitted next)
   is exactly the design that removes this confound**; its role is
   upgraded from replication to de-confounded identification.

## Consequences (per the registered interpretations)

- Amendment 2 §A decision rule fires **positive**: the interaction is
  confirmed at n=16 on the decision CI alone. Paper-1 headline
  (observational reversal + occ × reward-supervision interaction) keeps
  its confirmatory support with the honest pooled effect size.
- The P0 pivot stands unchanged; nothing in this read reopens E3
  as-designed.
- The support-scaling pilot and P3 flagship status remain gated on the
  E3v2 within-collector replication (READ 2), which now also carries the
  de-confounding burden identified by the battery.
- Correction to `artifacts/p0_axis1_corrective_20260713/RESULTS.md` audit
  prose (dated note added there): the finger/cup loss-key lists were
  swapped; finger logs carry the 8-component set (finger turn_hard has
  `touch/target_position/dist_to_target` observables). Component *sets*
  and the no-`rew`/`value` verification were and are correct.
