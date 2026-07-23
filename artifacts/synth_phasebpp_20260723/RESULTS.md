# Synth Phase-B″ — READ (2026-07-23)

Registration: `prereg/PREREG_synth_phasebpp_20260720.md` (frozen 20 Jul
with `analysis/synth_phasebpp_read.py`, before any ax1rb* outcome).
Results: `local_results/synth_phasebpp_20260722_092523/` (96 adapt-only
jobs, {task, apt} × {s0, s1} × fit seeds 1–8 × fresh adapt seeds
{104, 105, 106}). Canonical read: local frozen script on
`analysis/auc/auc.csv`; output `synth_phasebpp.json` here — **identical
to the cluster-side json** (`analysis/read/synth_phasebpp.json`),
field-for-field.

## VERDICT — PRIMARY DOES NOT FIRE. SYNTH INTERACTION LEG CLOSED.

**I = mean_fit [B_task − B_apt] = +30.5, 95% CI [−18.7, +100.2]**
(cluster bootstrap over 8 fit seeds, B=10K, rng 0; perm p = 0.53,
decision on the CI alone).

Registered consequence executes: **no Phase C; synth remains a
shuffle-collapse-only exhibit** in the paper, and the domain-scope
statement records that the interaction was not detectable under
lottery-corrected adaptation at this design's power (decidable
magnitude ≳ ~90 AUC units; the observed +30.5 is well below it).

## Registered secondaries / descriptives

| quantity | value |
|---|---|
| task simple (s1 − s0) | +6.0 [−52.8, +80.2] |
| apt simple (s1 − s0)  | −24.5 [−48.6, −0.8] (nominally negative) |
| within-fit adapt SD   | task_s0 48.9, task_s1 53.7, apt_s0 31.4, apt_s1 36.4 |

- **The design behaved as registered:** predicted CI half-width ≈ ±87
  (0.65× Phase-B's ±133 under share-0.865 with 3-seed averaging);
  realized ≈ ±60. Adapt averaging delivered the promised variance
  reduction — the null is a power-honest null, not a noisy repeat.
- The apt simple's nominally negative CI (apt adapts WORSE from the
  hi-occupancy side in synth) is a descriptive curiosity only; no
  registered status.

## Integrity

- Full 96-cell grid present, QC pass asserted per row; fresh adapt
  seeds 104–106 never used before (no revealed outcome enters);
  re-diagnosis modes (`ax1rd*`) cannot parse under the frozen RUN_RE.
- Freeze ordering held: prereg + read committed 20 Jul (results dir
  ships the bit-identical prereg copy); jobs submitted after.
- Local frozen read == cluster json exactly (no ulp drift this time).

## Where this leaves the synth domain (Paper 1)

Synth's contribution is now fixed: (1) **shuffle collapse −226.6***
(Phase B, stands) = binding-necessity in a fully controlled domain;
(2) the interaction is **not detectable** there under the corrected
design — scope statement: the reward-legibility interaction is a
property of the real-task domains (finger; cup mirror-pattern), not
reproduced in the synthetic reach construction at n=8 fits × 3 adapt
seeds. The adapt-lottery decomposition (share 0.865) + this
power-honest null close the thread cleanly.
