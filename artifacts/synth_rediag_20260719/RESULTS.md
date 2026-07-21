# Synth re-diagnosis — ADAPT LOTTERY CONFIRMED; Phase-B″ authorized (2026-07-19)

Snapshot: `local_results/optc_synth_rediag_20260719_175551/` (24
adapt-only jobs: task arm, fit seeds 1–4 × adapt seeds 101–103 × 2
sides, reusing the existing 500K synth checkpoints). Read exactly as
registered in `prereg/PREREG_synth_rediag_20260718.md` with the
pre-frozen `analysis/synth_rediag_read.py`.

## Pipeline integrity

- Local canonical AUC recompute vs cluster csv: 24/24 rows
  bit-identical, all pass QC; archived here. Local read json
  bit-identical to the cluster-side json. No seed-99 contamination
  (the frozen MODE_RE admits only `ax1rd<fit>q1s<side>` adapt rows,
  adapt seeds 101–103).

## Registered decomposition (one-way random effects per side, task arm)

| side | fit means (seeds 1–4) | MS_between | MS_within | σ²_b | σ²_w | within share |
|---|---|---|---|---|---|---|
| s0 | 267 / 230 / 150 / 331 | 17,068 | 26,300 | 0 (clamped) | 26,300 | **1.000** |
| s1 | 357 / 267 / 256 / 116 | 29,536 | 13,970 | 5,189 | 13,970 | **0.729** |

**Pooled within share = 0.865 ≥ 0.5 ⇒ the registered adapt-lottery
prediction CONFIRMS ⇒ Phase-B″ (adapt-averaged re-test) AUTHORIZED for
registration.**

The raw grid is stark — same fit, three adapt seeds: s0 fit1
[327, 16, 459]; s0 fit3 [255, 7, 189]; s1 fit2 [476, 73, 251]. Adapt
AUC on synth swings by up to ~30× across adapt seeds holding the world
model fixed. On side0 the between-fit mean square is *smaller* than the
within-fit mean square (σ²_b clamps to 0): fit identity explains
nothing there.

## Interpretation (registered branch)

The synth Phase-B collapse-only pattern (interaction +44 CI ±130, only
the shuffle collapse decisive) is now attributed to adapt-stage
stochasticity, not fit-quality differences: with per-cell n=8 single
adapt draws, the interaction contrast was drowned by a lottery the
proprio domains do not exhibit at this severity. This matches the
18-Jul diagnosis closure (in-subspace hypothesis refuted at model
level; adapt-stage variance the leading hypothesis — now confirmed).
Phase-B″ design implication: score each fit by the MEAN over ≥3 adapt
seeds before forming the interaction; at within-share .865, averaging
3 adapt seeds cuts the dominant variance component to ~1/3.

## Provenance

- `auc.csv` — canonical local recompute, 24 rows (bit-identical).
- `synth_rediag.json` — full read output (local; identical to cluster).
- Read command: `python -m analysis.synth_rediag_read --auc <csv>
  --output <dir>`.
