# A1 registered avoidance read — 22 Aug 2026

**Prereg:** `PREREG_nfi_avoidance_20260821.md` (review #27). ONE
execution of the frozen `se_avoid_read` ON RCC (same site as probe
production, per the 22-Aug pull-policy decision). Bundle
`a1_avoidance_20260822_101130` verified in place (16,419 files OK);
globs matched 4+4; instrument CLEAN; ckpt gates OK.

## Outcome: **AVOIDANCE-CONFIRMED — the registered primary FIRES**

- **Primary:** occupancy(hetero2) − occupancy(flat2) = **−0.0598**,
  one-sided negative exact permutation **p = 0.0143** (the C(8,4)=70
  assignment floor — complete separation), fires=True. Occupancy
  0.4599 (hetero) vs 0.5197 (flat).
- **Adjudication chain, in registered order:** degeneracy gate
  clean → misprice gate 4/4 valid hetero probes → gradient-delivery
  gate 4/4 adjudicable → validity CLEAN → primary FIRES → gated
  secondary (ranking) Δ=−0.0655, p=0.0143, fires (licensed only
  because the primary fired).
- Task return: two-sided score delta −6.7, p=0.37 — behaviorally
  neutral on the task, as in both prior waves.
- Mechanism blocks: descriptive-only per registration.

## What this settles

The twice-exploratory reversal (M3 gated −0.079; gradient-wave
hetero −0.052) is now a **REGISTERED, direction-registered,
de-confounded confirmatory result on fresh seeds**: spatially
structured fictitious information REPELS the agent that misprices
it hardest (θ₁≈7× in the avoided region). Third complete separation
in three independent waves. **A2 (SCARECROW) is now un-gated** —
the Track-A follow-up's trigger condition is met.

## Disclosure (carried per the prereg's provenance duty)

All eight cells were probed on REPOINTED checkpoints
(494,432–498,896 of 5e5): `DV3_RSYNC_OPTS --append-verify` froze the
mutable, constant-length `ckpt/latest` at the first-pull value, so
four cells (s44, s47, s48, s51) were initially probed at
105k–133k steps. Those outputs were quarantined in place
(`_INVALID_STALE_CKPT_*`, kept as evidence), pointers preserved as
`ckpt/latest.stale_20260822`, all cells repointed and re-probed
BEFORE this read (before/after table:
`$RUNROOT/CKPT_LATEST_REPOINT_20260822.md`; no outcome values were
consulted in the repair). The panel is uniform but is not the
literal final save on any cell (the true final saves died with the
instances); the probed checkpoints are 98.9–99.8% of training.
Fixed at source in 015e555a (pull pass 2 now carries ckpt/latest).
