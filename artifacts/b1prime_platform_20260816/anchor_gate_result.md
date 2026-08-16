# B1' Phase-A anchor gate: 16/16 BITWISE IDENTICAL

Job 53420696, midway3-0278 (Tesla V100, driver 535.216.03), `--platform cuda`,
no `--ep_batch`. Re-measures written to `$RUNROOT/b1p_anchor_recheck/`;
nothing pinned was touched. Pins are `ridge_finger_q1refit.csv` from
bundle `regen_ridge_e4_20260810_220849` (job 53243601, 10 Aug, same node).

The collate selects `alpha_0.001`; the pin reproduces at that alpha and
only that alpha, for all 16 cells.

| run | pinned auroc | re-measured auroc | d(auroc) | d(r2) | d(auroc_in) |
|---|---|---|---|---|---|
| `ax1wm_finger_q1s0_seed1` | 0.9455077864173864 | 0.9455077864173864 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_q1s0_seed2` | 0.8644163417838757 | 0.8644163417838757 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_q1s0_seed3` | 0.8829461405809119 | 0.8829461405809119 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_q1s0_seed4` | 0.7998172164096323 | 0.7998172164096323 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_q1s1_seed1` | 0.9604135558241971 | 0.9604135558241971 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_q1s1_seed2` | 0.8934252293923794 | 0.8934252293923794 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_q1s1_seed3` | 0.9789085822904775 | 0.9789085822904775 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_q1s1_seed4` | 0.9789873013009671 | 0.9789873013009671 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_fq1s0_seed1` | 0.4091586287343572 | 0.4091586287343572 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_fq1s0_seed2` | 0.4438698925194391 | 0.4438698925194391 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_fq1s0_seed3` | 0.4041109059652788 | 0.4041109059652788 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_fq1s0_seed4` | 0.4177901160693131 | 0.4177901160693131 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_fq1s1_seed1` | 0.3540836151377431 | 0.3540836151377431 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_fq1s1_seed2` | 0.4741590494216728 | 0.4741590494216728 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_fq1s1_seed3` | 0.3364297445452 | 0.3364297445452 | +0.0e+00 | +0.0e+00 | +0.0e+00 |
| `ax1wm_finger_fq1s1_seed4` | 0.3564225693023456 | 0.3564225693023456 | +0.0e+00 | +0.0e+00 | +0.0e+00 |

**max |delta| across auroc, r2 and auroc_in_regime = 0.000e+00 (exactly zero).**

## What this discharges

Four candidate terms were folded into this one comparison, and all four are
zero on the pinned estimand:

1. **node** -- same node as the anchor (0278), so this was always expected;
2. **env** -- the wheel-dating said the jax stack is unchanged since
   2026-05-09; this confirms it empirically rather than by inventory;
3. **`--platform cuda` vs whatever the anchor used** -- backend selection is
   inert, as argued but not previously demonstrated;
4. **the anchor's unlogged, unrecoverable `ep_batch`** -- Phase A ran with NO
   `--ep_batch` and still reproduced the pins bitwise, so that difference is
   inert for this instrument. This was the reason the gate was kept after the
   wheel-dating removed the original one, and it is the term this result
   actually retires.

Gate verdict: **MATCH**. Under the decision note that means the ridge panel
proceeds as registered and the narrow pin-swap amendment is not needed.

Phase B measured all 16 B1' cells in the same allocation, same GPU, same
flag: `ok=16 skipped=0 failed=0`. Ops has not inspected those values.
