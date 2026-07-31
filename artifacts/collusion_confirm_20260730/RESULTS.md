# Paper-4 collusion confirmatory READ — ALL FOUR PRIMARIES FIRE (2026-07-30)

ONE execution of the frozen read (`analysis/collusion_confirm_read.py`,
selfcheck PASS immediately before), per
`prereg/PREREG_collusion_confirm_20260730.md` (freeze commit
`42bbe487`). Bundle
`local_results/collusion_confirm_20260730_200834/` (slurm
52831447): MANIFEST.sha256 verify OK 16/16; bundled
pilot.py/designb.py/reader/prereg snapshots byte-identical to the
freeze commit; worktree copies unmodified since freeze. All six
cohorts complete (loader-enforced): Design B seeds 1000–1099, Design A
2000–2099 / 3000–3399.

## Gates — all pass

exact_replay 100/100 (G-CLEX); baseline reproduction G-CLBASE: Design-B
online Δ_profit 0.853, fingerprint 0.76; Design-A baseline Δ 0.852,
fingerprint 0.65 (both in [0.70, 0.90], both fp > 0.5); n_fp = 76 ≥ 30
(G-CLFP); manipulation gate G-CLA-MANIP: forbid occupancy shift
−0.485 [−0.510, −0.459] — passed.

## Primaries (95% bootstrap CIs, B=10K, rng 0)

| claim | estimand | point [CI] | verdict |
|---|---|---|---|
| **P-CLB1** | fp-cond paired I_dp at d=1 | **−0.104 [−0.165, −0.046]** | **FIRES** |
| **P-CLB2** | fp-cond paired I_fp at d=1 | **−0.592 [−0.724, −0.447]** | **FIRES** |
| **P-CLA1** | forbid−base fingerprint rate | **−0.650 [−0.740, −0.550]** | **FIRES** |
| **P-CLA2** | forbid−base Δ_profit | **+0.135 [+0.113, +0.157]** | **FIRES** |

**Registered verdict (verbatim from the frozen reader):** P-CLB1+P-CLB2
FIRE — punishment-phase experience is a composition-specific driver of
both the profit level and the punishment structure at matched update
mass; the RQ2 composition leg lands and the **RQ3 coupling wave is
authorized**. P-CLA1+P-CLA2 FIRE (dissociation) — removing
exploration-generated undercuts collapses the punishment structure
(fingerprint 0.65 → 0.00) while the price level RISES (0.852 → 0.987):
an experimentally induced failure-to-learn (Abada–Lambin) regime;
punishment-generating exploration is necessary for GENUINE collusion,
not for supra-competitive prices.

## Registered secondaries (never decisional)

- **Dose ladder (fp-cond), monotone ✓** — I_dp: d.25 −0.006 ns /
  d.50 −0.050 [−0.088, −0.017] / d1 −0.104*; I_fp: −0.039 ns /
  −0.118 [−0.197, −0.053] / −0.592*. The mid dose fires at
  confirmatory power (calibration had it mostly inert at n=20) —
  graded, not threshold-only.
- Unconditional twins: I_dp −0.105 [−0.155, −0.055]; Δ_price twin
  (fp-cond) −0.049 [−0.088, −0.010] — both same-signed, CI<0.
- **Design-A arms**: forbid λ=0.5 dp +0.057 [+0.030, +0.085], fp
  +0.190 [+0.070, +0.310] (mild bias mildly HELPS collusion — λ
  response is non-monotone, consistent with the smoke's fp 1.00);
  force λ=1 dp −0.208 [−0.248, −0.168] (forcing undercuts suppresses
  profit), fp −0.030 ns; force λ=0.5 dp −0.026 ns.
- Confound audit (disclosed): forbid λ=1 coverage_late −0.540
  [−0.580, −0.502] — the manipulation is blunt as registered; forbid
  arm converged 100/100 at the window minimum (the degenerate price
  ratchet disclosed pre-freeze; its Δ spread ≈ 0, CIs driven by
  baseline variance as disclosed).
- Convergence: designB 0.81, base 0.73, force100 0.82, force50 0.83,
  forbid50 0.75 (no exclusions, as registered). Exo label descriptive:
  0.974 (sd 0.072) — degenerate as calibrated.
- RQ1-observational (midrank Spearman, Design-A baseline): occ↔Δ
  +0.73, occ↔fp +0.45, cov↔Δ −0.20, cov↔fp −0.49 — the correlational
  associations the causal arms supersede.

## Consequence map lands (frozen entries)

- RQ2 composition leg CONFIRMED on both outcomes ⇒ **RQ3 coupling wave
  authorized** as the next registration.
- The dissociation exhibit lands ⇒ the two-outcome measurement
  argument (Δ vs fingerprint) is load-bearing for the paper; the
  fingerprint-conditional convention is vindicated.

## Provenance

- `collusion_confirm.json` — full machine record (this dir).
- Bundle: `local_results/collusion_confirm_20260730_200834/`
  (results/*.csv ×6, code+prereg snapshots, slurm log, MANIFEST
  verified) pinned by `manifests/collusion_confirm_20260730_200834.sha256`.
- Read command: registered form (six csv paths) →
  `analysis_out/collusion_confirm_20260730/`, copied here unmodified.
