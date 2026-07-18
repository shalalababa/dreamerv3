# P3 Amendment-1 read: pooled n=16 re-test — PRIMARY FIRES (2026-07-17)

Snapshot `local_results/p3_amend1_20260717_223311/` (32 runs: rgo/sgb ×
2 sides × seeds 9–16). Read per `prereg/PREREG_p3_amendment1_20260717.md`
(frozen pre-outcome, committed b3f05f76 before the wave was submitted),
executed exactly as registered: the frozen `analysis/p3_factorial_read.py`
contrast machinery (committed 2026-07-16) reused with SEEDS = 1..16 —
the only declared change (`amend1_read.py` here imports that machinery;
merged table `auc_pooled_1_16.csv` here). Seeds-1–8 rows come from the
bit-checked P3 wave artifact and are asserted equal to the frozen
`p3_read.json` deltas at read time (bit-check PASS).

## Audit

32/32 new runs pass QC (96 eps ≤100K); config audit 32/32 (rgo =
task/reward_grad T/repval_loss T/repval_grad F; sgb = task/F/T/F —
exactly the registered arm rows); ADAPT_DONE present on all.

## Registered read (AUC100k, B=10K seed-0 cluster bootstrap, decision on CI alone)

| contrast | mean | 95% CI | verdict |
|---|---|---|---|
| **PRIMARY pooled 1–16 [B_rgo − B_sgb]** | **+51.5** | **[+9.0, +94.6]** | **CI > 0 — FIRES** |
| Subsidiary fresh-batch 9–16 | +52.7 | [−4.6, +112.6] | ns alone; point estimate replicates |
| Secondary rgo simple pooled 1–16 | +56.7 | [+19.5, +91.6] | excludes 0 |
| Secondary sgb simple pooled 1–16 | +5.2 | [−13.7, +24.8] | tight placebo null |

Sensitivity (robustness-only): primary perm p=0.0409, t p=0.0395,
Wilcoxon 0.065, d_z=0.56, LOO [+41.7, +60.4] all positive — not
seed-driven. 11/16 seeds positive; per-seed deltas span −83..+198
(noisy per-seed, stable mean).

**No winner's curse this time:** fresh-batch point estimate +52.7 vs
seeds-1–8 +50.2 — the effect size REPLICATES in fresh seeds (contrast
with the W0 full-arm attenuation +157→+40). Batch means: rgo 65.0→48.4,
sgb 14.8→−4.3 — both drift down, the difference is stable. The honest
effect size for the paper is the pooled +51.5 [+9.0, +94.6].

## Registered consequence

The reward-head REPRESENTATION GRADIENT ALONE carries part of the
interaction: the 17-Jul descriptive claim (rgo own-CI>0) upgrades to
**confirmatory**. Combined with the fired S2/S3 binding results, the
licensed mechanism statement becomes: *frame-bound reward-prediction
gradients through the WM trunk are a sufficient carrier of the
occupancy × supervision interaction; label–state binding is necessary
for the full effect.* The "support must be legible to the objective"
headline gains its gradient-path leg.

- **P-A1 (theory prereg, first registered prediction adjudicated):
  CONFIRMED — CI > 0 with point estimate +51.5 inside the registered
  +40..+70 band.** P-A2 (slow-crossing fallback) is moot for this
  contrast. (Registered predictions file committed b3f05f76, before
  any seed-9–16 outcome existed.)
- **Idea 9a (reward-head-only pretraining recipe) is UN-EMBARGOED**
  per the 17-Jul triage condition (rgo>sgb confirmed at n=16).
- Editorial band condition "pooled n=16" LANDS. Band ledger: n=16 ✓,
  TD-MPC2 ✗ (family-scope limitation, same-day read),
  full/unfrozen calibration — pending, orthogonal test — pending.
- Both-paths/interplay branch is superseded as registered: rgo is a
  confirmed sufficient carrier; vgo's null stands (and the same-day
  wiring audit, `artifacts/vgo_wiring_audit_20260717/`, reclassifies
  it as case-(b) attenuation — extended-fit discriminator remains the
  open follow-up).

## Still open on this thread

- E4 measure pass over the 32 new fits (descriptive membership panel;
  cluster job: `E4_DOMAINS=finger E4_GLOB_finger='ax1wm_finger_rgo*'`
  then `'ax1wm_finger_sgb*'` — joins the existing factorial panel).
- Files here: `amend1_read.py` (driver), `amend1_read.json` (full
  output incl. sensitivity), `auc_amend1.csv` (frozen AUC rows, seeds
  9–16), `auc_pooled_1_16.csv` (merged read table).
