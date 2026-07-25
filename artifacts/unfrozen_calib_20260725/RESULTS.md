# Unfrozen-adaptation calibration — FIRES, effect AMPLIFIED ×4.3 (2026-07-25)

Snapshot: `local_results/unfrozen_calib_20260725_172314/` (32 adapt-only
jobs: {rgo, sgb} × {s0, s1} × fit seeds 1–8 from the EXISTING
E4-verified 500K fits, `AXIS1_ADAPT_CONFIG=unfrozen_readout` — same
initialization as the frozen wave, nothing frozen; first unfrozen-adapt
runs ever made in this project). Read exactly as registered in
`prereg/PREREG_unfrozen_calib_20260723.md` with the pre-frozen
`analysis/unfrozen_calib_read.py` (selfcheck PASS).

## Pipeline integrity

- 32/32 cells present (ADAPT_DONE + scores.jsonl each); bundle read
  code md5-identical to the frozen repo version; local read IDENTICAL
  to the cluster-side json on every key.
- exclusions.csv lists only the unrelated in-flight `ax1(f)s12*`
  Scaling-B rows (owner-confirmed still running; not this wave).

## Registered read (cluster bootstrap over 8 fit seeds, B=10K rng 0)

| quantity | mean | 95% CI | notes |
|---|---|---|---|
| **PRIMARY: unfrozen rgo − sgb** | **+267.2** | **[+206.7, +331.4]** | **FIRES** (perm p=.0078 = the n=8 minimum — every seed positive; d_z=2.77; LOO [242.9, 284.4]) |
| side s0 simple | +243.5 | [+140.9, +343.3] | fires individually |
| side s1 simple | +291.0 | [+207.3, +365.5] | fires individually |
| rgo level change (unfrozen − frozen, paired) | +235.2 | [+178.1, +292.1] | unfreezing helps rgo enormously |
| sgb level change | +30.4 | [+20.3, +41.5] | unfreezing helps sgb a little |
| frozen same-seed effect (seeds 1–8, disclosed baseline) | +62.5 | [+32.8, +94.5] | from committed `p3_amend1` csv |
| **calibration ratio (unfrozen/frozen)** | **4.28** | — | **amplification, not attenuation** |

## Registered consequence applied — band leg LANDS, stronger than required

The registered prediction was "persists, plausibly attenuated." The
outcome is the strong form: under full adaptation the effect is
**4.3× larger**. Frozen-readout was not merely "not the artifact" — it
was the CONSERVATIVE measurement. Mechanism-consistent reading (for
the paper's calibration paragraph): the reward-legible support is a
better *initialization*, not just a better frozen feature bank —
fine-tuning compounds from it (rgo +235) while the illegible sgb
features must be substantially relearned (+30). The practical
fine-tuning regime — the one practitioners actually use — shows the
LARGER legibility effect; frozen-readout understates the phenomenon it
was designed to isolate.

Band-ledger state after this read: n=16 ✓ · TD-MPC2 ✗ (registered
family-scope) · **unfrozen calibration ✓ (amplified)** · orthogonal ✓
(objective-specific). The 40–55% claim band's conditions are now fully
adjudicated: three of four legs land, with the TD-MPC2 leg replaced by
the explained-boundary treatment.

## Provenance

- `unfrozen_calib.json` — full read output.
- `auc.csv` — canonical AUC rows consumed (ufz modes).
- Read command: `python -m analysis.unfrozen_calib_read --auc <csv>
  --output <dir>`.
