# Wave C (localized-channel planner) — REGISTERED READ, 22 Aug 2026

**Prereg:** `PREREG_planner_localized_20260822.md` (rev per review
R2, frozen pre-compute). **Reader:**
`uncfield/se_planner_loc_read.py`, ONE execution 22 Aug on the
verified bundle `wavec_planloc_20260822_182736` (manifest
byte-exact vs the committed copy; 8/8 A1 checkpoints, frozen
se_planner_probe unchanged, one CUDA device — all 8 cells on ONE
RCC GPU, job 54313393; ckpt gates green incl. the step floor on
the repointed checkpoints).

## OUTCOME: **NO-LOCALIZED-TRANSMISSION-WEAK-OPTIMIZER** (the scoped null — registered NO tier movement)

- **PRIMARY does not fire, decisively:** Δ = attr_d(cem_disag) −
  attr_d(policy) is NEGATIVE in 4/4 hetero runs (−0.0168, −0.0156,
  −0.0109, −0.0070) AND 4/4 flat runs (−0.0074, −0.0045, −0.0091,
  −0.0021); interaction −0.0068 in the WRONG direction, exact
  C(8,4) p = .971; hetero-positive count 0/4. The de-normalized
  robustness row agrees (−0.0011) — the Jensen normalizer
  asymmetry is not masking anything.
- **MATCHED-STRENGTH CONTROL FAILS (the registered scope
  determinant):** realized intrinsic(cem_disag)/intrinsic(policy)
  = 0.74, 0.89, 0.75, 0.81 on the hetero wing (bar 0.9 in ≥3/4;
  policy > random in 4/4, so no degeneracy). The from-scratch
  greedy CEM is AGAIN a weaker harvester than the amortized
  incumbent — exactly the ambient wave's failure mode,
  reproduced in the localized regime. **Per the registration, this
  null licenses only "this optimizer, at this strength, could
  not"; NO tier movement in either direction.**
- S1 region entry: mixed (+0.02, −0.17, −0.27, +0.12) — the
  planner does not systematically enter the avoided region.
- **S2 (PRE-DECLARED this wave) — the share/composition steering
  REPLICATES on fresh data, 8/8:** share_d(cem_disag) >
  share_d(policy) in every run (hetero 0.075/0.059, 0.044/0.039,
  0.058/0.049, 0.088/0.042; flat 0.060/0.038, 0.050/0.046,
  0.073/0.045, 0.071/0.044). The ambient wave's value-aware 8/8
  share observation is now a REGISTERED-descriptive fact on fresh
  checkpoints: the deployed-objective planner steers the
  COMPOSITION of attribution toward the distractor even while
  losing on level.
- S3 steerability ceiling: negative in 4/4 (even cem_distractor
  cannot beat max(policy, random)) — coherent with the weak
  optimizer.
- 0 excluded; fit counters OK 8/8; full 4+4.
- **Protocol deviation (disclosed):** the registered SMOKE=1
  hetero pre-cell was not run separately — ops executed the 8
  cells directly (8/8 rc=0); the hetero env path (mod quadruple)
  is validated post-hoc by the clean s44–s47 outputs.

## Reading (composed with Wave A, same day)

The localized-channel question ("can deployment planning cash the
misprice where avoidance leaves it unclaimed?") remains OPEN at
matched strength: the only attacker that has probed the spatial
channel is one that fails its own strength control. Wave A's
matched-strength attacker probed the AMBIENT axis and landed a
tight stationarity null. The two compose into: Tier-1 stationarity
on the refinement axis (Wave A, gates green) + an optimizer-scoped
localized null + a now-registered composition-steering regularity
(8/8 fresh). A matched-strength probe of the LOCALIZED channel is
exactly what the Wave-B adversarial amortized actor would be —
Wave B's GO condition (A+C read) is now met; it remains a user
opt-in.

Artifacts: `se_planner_loc_read.json` (this dir). Data:
`local_results/wavec_planloc_20260822_182736` (manifest committed).
The A1-region prior note stands: the in-region signal exists
(1.54×, complete separation) — this wave shows the CEM cannot
harvest it, not that it is absent.
