# Spectral domains read — P-SM1 NOT confirmed; domain account refuted as registered (2026-07-25)

Snapshot: `local_results/spectral_domains_20260725_090929/` (4 buffer
measurements under `spectral_v1_1_20260724` + one compare). Read exactly
as registered in `prereg/PREREG_spectral_domains_20260724.md` +
Amendment 1 (`_amend1_20260724.md`) with the pre-frozen
`probing/spectral_measure.py` (selfcheck PASS).

## Pipeline integrity

- All 4 matched buffers (cup/finger × q1 s0/s1) measured under the
  amended version `spectral_v1_1_20260724`; the compare version gate
  accepted all inputs (no pre-amendment json survived into the read).
- **θ estimable on ALL FOUR buffers** (n_rewarded 10,732–64,699 ≫
  MIN_REWARDED=50): `dropped_sides = []`, `inestimable = []`, and the
  Amendment-1 cross-buffer-θ secondary was never engaged — the
  amendment's fallback machinery was needed for the crashed buffer set
  encountered mid-pass, but every buffer in the REGISTERED primary grid
  is own-θ. (The pre-amendment partial-outcome disclosure stands; no
  compare verdict ever ran before this one.)
- Local `compare` re-execution on the 4 measure jsons is IDENTICAL to
  the cluster-side `spectral_compare.json` (deterministic, verified).

## Registered read (P-SM1: rank_cup < rank_finger on EVERY evaluable matched side)

| side | rank_cup | rank_finger | holds |
|---|---|---|---|
| hi (s1-occ side) | 492 | 455 | **NO — reversed** |
| lo | 582 | 640 | yes |

**Verdict: P-SM1 NOT confirmed.** The reward-direction variance-rank
ordering fails on the hi side; the registered form required every
evaluable matched side.

## Registered consequences applied

- The spectral account of the DOMAIN (cup/finger) boundary is
  **refuted as registered**: Paper 1 presents cup/finger as an
  unexplained scope condition (the honest fallback), and the theory's
  cup section (E5) loses its measurement support — honesty-box entry
  for `Theory_SpectralTransfer_20260717.tex`.
- Untouched by this outcome (per the frozen map): the arm-level results
  — interaction, shuffle collapse, sgb/rgo separation, Amendment-1
  sufficiency — do not depend on the cross-domain account.
- Combined with the 24-Jul pixel X2 out-of-sample failure (monotone
  form), the spectral model is now 0-for-2 on its DOMAIN-level
  predictions while its arm-level alignment reading survived the same
  day's orthogonal-objective test — the theory's live content is the
  within-domain objective-alignment law, not cross-domain λ ordering.

## Descriptive secondaries (never decisional)

- θ-regression quality varies widely: r2_oof cup .64/.73 vs finger
  .40/.25 — the reward direction is itself better-estimated in cup;
  rank comparisons inherit this asymmetry (disclosed, not modeled).
- diversity_pr and λ_need do not order the domains consistently either
  (cup hi 7.7 < finger hi 9.2 but cup lo 9.8 > finger lo 8.1).
- f_rewarded spans 0.05 (finger lo) to 0.32 (finger hi) — the hi/lo
  occupancy construction is visible in the buffers as intended.

## Provenance

- `spectral_compare.json` — compare verdict object (primary +
  secondaries + integrity fields).
- `cup_q1_s{0,1}.json`, `finger_q1_s{0,1}.json` — per-buffer
  measurements (version, θ, spectrum, ratio profiles).
- Read commands: `python -m probing.spectral_measure measure ...` per
  buffer (cluster, v1_1), then one `compare --inputs <4 jsons>`.
