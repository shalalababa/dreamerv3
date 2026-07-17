# TD-MPC2 cross-family read + Goodhart ensemble + D1 Pilot E1 (2026-07-17)

Snapshots: `local_results/tdmpc2_20260717_154136/` (32 TD-MPC2
fit+adapt runs), `local_results/goodhart_ensemble_20260717_132942/`
(3 evaluator sweeps + ensemble pool), `local_results/d1pilot_e1_20260717_132936/`
(6 online D0-instrumented runs). Prereg ordering intact:
`prereg/PREREG_tdmpc2_crossfamily_20260716.md` committed in `b3f05f76`,
before either results commit (`712f6eb9`, `19c57377`).

## 1. TD-MPC2 cross-family replication — PRIMARY does NOT fire; aware simple effect does

Registered read per `PREREG_tdmpc2_crossfamily_20260716.md`. AUC100k by
the untouched `analysis/adaptation_auc.py` (32/32 runs pass QC, 112
eval episodes ≤100K each; `auc.csv` here). Audit: 32/32 `TM2_FIT_DONE`,
32/32 `ADAPT_DONE`; saved fit configs show the registered manipulation
exactly — free: `reward_coef 0.0, value_coef 0.0`; aware: `0.1, 0.1`
(official defaults); `consistency_coef 20` in both, all seeds.

B_arm(k) = AUC100k(s1) − AUC100k(s0), seeds 1–8 paired, cluster
bootstrap B=10,000 `default_rng(0)`, decisions on CI alone
(`tdmpc2_read.py` here, run on `auc.csv`):

| contrast | mean | 95% CI | verdict |
|---|---|---|---|
| **PRIMARY [B_aware − B_free]** | +9.5 | [−43.2, +51.4] | **does NOT fire** |
| SEC-1 B_aware simple effect | +35.3 | [+4.8, +69.6] | positive (as predicted) |
| SEC-2 B_free simple effect | +25.8 | [−8.7, +63.2] | null by CI (predicted null; not tight) |

Sensitivity (robustness only): primary t=+0.36, exact sign-flip perm
p=0.734, d_z=+0.13, LOO range [−2.9, +32.2] — no seed-driven artifact;
the null is not fragile. Per-seed diffs: +54.8, −149.8, +36.9, +18.1,
+26.6, +34.6, −41.7, +96.4 (sd ≈ 72 ⇒ ~±50 detectable at n=8; a true
effect at DreamerV3 scale (+84..+99) would likely have been seen, but a
smaller real effect would not — reported as scope limitation, not
evidence of absence, per the frozen language).

Registered consequence: **family-scope limitation** — headline language
restricted to *reconstruction-based world models*; TD-MPC2 panel
reported as boundary, not external-validity leg.

Interpretation-map position: closest to map cell 3 ("PRIMARY null with
aware+ and free+"), except the free simple effect is indeterminate
(+25.8, CI spans 0) rather than confirmed-positive. So: an occupancy
benefit exists in the aware arm, and the data cannot resolve whether it
is supervision-dependent in this family.

Descriptive cell means (AUC100k, never compared numerically to
DreamerV3): aware s0 212.0 (sd 51), aware s1 247.3 (sd 30), free s0
88.0 (sd 26), free s1 113.8 (sd 38). The striking unregistered
descriptive fact is the **arm main effect**: aware ≈ 2.2× free in level
on BOTH sides (+128.7 pooled). Reward/value supervision during offline
fitting massively improves frozen-representation adaptability in
TD-MPC2 too — mechanism-coherent with "support must be legible to the
objective" — but the occupancy×supervision interaction specifically is
what did not replicate at n=8.

## 2. Goodhart ensemble (unregistered follow-up; decision-support for ensemble-or-demote)

Three independent evaluator seeds rescored the 295-policy pool; the
ensemble pools their M-return means (`pool_metrics_ensemble_mean.json`,
n=296 rows — includes one policy not in every single-evaluator log
tail; discrepancy is 1 policy, immaterial).

| evaluator | spearman(M-return, real) | inversion rate | top-1 regret |
|---|---|---|---|
| s1 | +0.136 | 0.455 | 572.8 |
| s2 | +0.135 | 0.455 | 287.5 |
| s3 | +0.138 | 0.454 | 571.9 |
| **ensemble mean** | **+0.186** (p=0.0013) | **0.439** | **571.9** (selected 192.7 vs best 764.6) |

Ensembling lifts rank correlation slightly (0.136 → 0.186, still very
weak), leaves the pairwise inversion rate essentially unchanged, and
does not improve top-1 selection at all — the ensemble picks a
192.7-return policy from a pool whose best is 764.6. Selection remains
≈ lottery. Points to **demote**: report the Goodhart track as a
negative/boundary result with the ensemble as its closing control
(decision is the study owner's; nothing here was registered).

## 3. D1 Pilot E1 — pipeline complete, 6/6 clean

`d0` pipeline (dose E1 = 0.0, the zero-distractor base cell per
`d0/synthetic.py DOSES`), 100K env steps online, full instrumentation
(disag + valens0–4 critic ensemble), cup + finger × seeds 1–3. All six
`finished rc=0 status=DONE`; 96 episodes each to step 96,096;
checkpoints committed in `712f6eb9`.

| run | mean score (all eps) | last-10 |
|---|---|---|
| cup e1 s1/s2/s3 | 616 / 493 / 505 | 946 / 943 / 925 |
| finger e1 s1/s2/s3 | 170 / 150 / 153 | 390 / 150 / 344 |

Cup near ceiling by 100K; finger in the normal online range; valens
losses training sane (1.78–1.88) throughout. This validates the
D0/D1 substrate at pilot scale. Gate D1 itself
(`PREREG_gate_d1_stage1a_20260714.md` §B) still requires Stage-1B
oracle labels (`d0/oracle_labels.py`) on these checkpoints before any
unfreeze decision; the 24+24 stay frozen.

## Files

- `auc.csv` — frozen `adaptation_auc.py` output over the TD-MPC2 runroot.
- `tdmpc2_read.py` — exact contrast script (registered machinery: B=10K
  percentile bootstrap, `default_rng(0)`, decisions on CI alone).
