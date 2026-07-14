# Stage-1A registered read — density-residual instrument PASSES the full battery (2026-07-14)

Read per `prereg/PREREG_gate_d1_stage1a_20260714.md` §A (frozen minutes
before this run; instrument implemented and synthetic-validated today,
never previously run on real dumps). Data: the existing Stage-0 dumps
(`local_results/evpi_stage0_20260710_221924/raw/`), protocol otherwise
identical to the frozen Stage-0 read (horizons {1,5}, headline anchor
h=5 final checkpoint, stream-clustered bootstrap n=300, 2-SE verdicts).

## Decision (registered rule): **PASS**

- **Within cells: 10/10 sub-cells flip TRACKS_DENSITY → TRACKS_ERROR**
  at the final (500K) checkpoint (rule required ≥8/10, zero
  TRACKS_DENSITY). Final-checkpoint Δ = pcorr(D_res,E|ρ) −
  pcorr(D_res,ρ|E): cup_p2e +0.17..+0.43, finger_p2e +0.14..+0.21, all
  > 2 clustered SEs.
- **Cross cells: no regression** — all four TRACKS_ERROR;
  finger_random_cross improves from AMBIGUOUS (raw) to TRACKS_ERROR
  (Δ +0.96, SE 0.04).
- **Registered secondary (early-checkpoint preservation): PASS** — all
  cells TRACKS_ERROR at the 100K checkpoint under the residual, as
  under raw.

Honest effect sizes: the residual's raw error correlation is modest in
the within cells (Spearman +0.03..+0.23 at h=5 final) versus the cross
ensembles (+0.16..+0.31). The instrument rescues the *sign* and the
verdict decisively; per-state precision remains diagnostic-grade
(consistent with the 12-Jul revision's SNR caveat — this does not
license per-state gating by itself; Gate D1's calibration/lift/net-value
criteria still stand between here and the 24+24 unfreeze).

## Consequences (per the registered mapping)

- Density-residual disagreement is confirmed as the **Stage-1A primary
  instrument**; it feeds Gate-D1 ÊVSI features and is Route B's
  "calibration fix" asset (the theory addendum §6 mechanism story —
  late-phase D → leverage/density complement — now has its predicted
  intervention working on all 14 cells).
- Route B package now holds: 10/10 attractor replication (Stage 0) +
  phase change + cross-policy boundary + **a registered, passing
  calibration fix**; remaining Route-B item = one downstream
  consequence (decision-relevant demo).
- Next infra per §C: `d0/oracle_labels.py` (restorable-state Δ_o
  labels) for Gate D1 proper.

Per-cell analyses + memos in this directory (`<cell>/analysis.json`,
`<cell>/memo.md`; `instrument: density_residual`, resid_folds 5, seed 0).
