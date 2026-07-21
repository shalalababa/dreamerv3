# Option C corrective read — C2 FIRES: P-B4 strong form REFUTED (2026-07-19)

Snapshot: `local_results/optc_synth_rediag_20260719_175551/` (12 new
fit+adapt jobs on the never-run `q1v400/side0` buffer, occ .290; modes
`ax1v4q1v400s0` task / `ax1fv4q1v400s0` apt, seeds 1–6). Read exactly as
registered in `prereg/PREREG_scaling_optc_amend1_20260718.md` with the
pre-frozen `analysis/optc_amend_read.py`.

## Pipeline integrity

- Local canonical AUC recompute (untouched `analysis/adaptation_auc.py`)
  vs cluster csv: **36/36 snapshot rows bit-identical** (12 corrective +
  24 re-diagnosis), all pass QC. The 12 corrective rows archived here.
- Historical v200s1/v400s1 rows taken from the frozen
  `artifacts/scaling_optc_20260718/auc.csv` with the in-script overlap
  bit-check, as registered.
- Local read json bit-identical to the cluster-side json.

## Registered read (B=10K cluster bootstrap seed 0; decision on CI alone)

| contrast | mean | 95% CI | registered verdict |
|---|---|---|---|
| **C1 task [v400s0 − v400s1]** (occ .290 vs .094, fixed 400-ep volume) | +23.3 | [−248.1, +301.9] | CI spans 0 ⇒ **occupancy saturates below .094 at this volume** (threshold moved with volume — estimation-floor account) |
| **C2 task [v400s0 − v200s1]** (≈fraction-matched, count ×2 — the P-B4 test) | **+251.9** | **[+120.4, +396.0]** | CI > 0, 6/6 positive ⇒ **volume matters at ≈fixed fraction ⇒ P-B4 fraction-not-count strong form REFUTED** |
| C1 apt analog (descriptive) | −4.9 | [−32.2, +15.4] | floor, as expected |
| C2 apt analog (descriptive) | +16.9 | [−4.6, +43.0] | floor, as expected |

Sensitivity (robustness only), C2: paired t=3.28, exact sign-flip perm
p=.031, Wilcoxon p=.031, d_z=1.34, LOO range [+191, +301] — not
seed-driven. C1 is wide-noise (d_z=.06, deltas −528..+604).

Cell means (task): v400s0 456±193, v400s1 433±242, v200s1 204±91; apt
v400s0 81±30. The 18-Jul anomaly ("winning cell has FEWER in-regime
frames") resolves: at 400 episodes the task arm sits ≈440–456 regardless
of occupancy .094 vs .290 — episode volume/diversity, not in-regime
fraction, drives adaptation benefit at this scale, and the occupancy
threshold is not a fixed fraction.

## Theory ledger

**P-B4 (fraction-not-count strong form) = first REFUTED adjudicated
prediction** (ledger: P-A1 landed, P-B1 landed, P-B4 refuted). The
registered composition caveat stands: v400s0 vs v200s1 differ in episode
composition as well as count (stated limit in the amendment); within
that limit the registered interpretation map fires on the REFUTED
branch. The spectral-competition model needs a volume term (estimation
floor / effective-rank growth with data) — flagged for the theory
addendum; occupancy remains necessary in the confirmed regime-gated
sense (P0/W0/E3v2 unaffected — those held volume fixed by construction).

## Provenance

- `auc.csv` — canonical local recompute, 12 corrective rows
  (bit-identical to cluster).
- `optc_amend_read.json` — full frozen-read output (local; verified
  identical to the cluster-side json).
- Read command: `python -m analysis.optc_amend_read --auc <cluster csv>
  --output <dir>`.
