# Tier-1 calibration bundle — ONE main read (21 Aug 2026)

Registration: `prereg/PREREG_p2_tier1_calibration_20260821.md` (frozen
at commit 8828ab6f). Executed once: `--component main` on the committed
`local_results/w1_labels_20260811_090727/w1_labels` (68 npz, per-file
shas in `p2t1_main.json` provenance) against the published
`artifacts/w1_read_20260811` anchors. Both identity gates PASSED
(recomputed P-W1a == published bit-exactly; published == disclosed
constants). **The `--component ab` leg is PENDING ops** (one eval-score
glob per domain, RCC).

## Verdicts (frozen rules)

| component | dv3 | tm2 |
|---|---|---|
| CCD positive control | **CCD-PC-FAILS** (licensed via FLOOR-BELOW-BAR — but see the sensitivity note) | **CCD-PC-FIRES** (+0.0547 [+0.0060,+0.1049], p=.037) |
| CCD placement | ABOVE-PERFECT (noise-scale, uninformative — all curve points ≈ 0) | **BELOW-RANDOM** (published −0.301 below every chooser incl. random — as pre-stated; consistent with S3) |
| C7 probe rule | **PROBE-BLIND** (evsi −0.055 [−0.134,+0.038], p=.24) | **PROBE-BLIND** (evsi +0.006 [−0.029,+0.061], p=.80) |
| Stage A | zero-identity PASS; MDS80 = 0.10 sd = **0.142 opportunity units**; FLOOR-BELOW-BAR | zero-identity PASS; detection ≡ 1 (base already fires; descriptive) |
| CE harvest power | **MDE80 = 0.10** (cell sd 0.180 [0.089, 0.249]) | **MDE80 = 0.50** (cell sd 0.78 [0.29, 1.23]) |

## What the bundle bought

1. **The positive control exists (tm2).** A σ=0 synthetic selector
   harvests +0.0547* from the tm2 substrate through the exact frozen
   chain — the chain CAN register harvest when a competent selector
   exists. The curve decays only mildly to σ=8 (+0.0401*) because the
   raw-G-unit σ grid is small against tm2's tail-state spreads
   (disclosed); the uniform-random endpoint −0.0054 confirms ≈0 truth
   under random choice.
2. **The probe rule identifies nothing (both families).** The
   FullRecord:1963 "unmeasured term" is now measured: one-real-step
   probe identification adds ≈0 over plug-in, and the identification
   gap ≈ the entire split-selected opportunity (tm2 gap +0.056 vs
   P-W1a +0.062) — what a perfect selector could get, the probe does
   not get.
3. **The instrument floor is measured on real noise (dv3).** Planted
   true spread at 0.142 opportunity units is detected with rate 1.0
   (0.071 units: rate 0.14 — MDS80 bracket (0.071, 0.142]); selector
   efficiency 0.74–0.94 across the grid.
4. **Harvest power events now have numbers.** Every W2/P-W1b "power
   event" sentence must quote MDE80 with its band (registered
   consequence): dv3 0.10, tm2 0.50 — note tm2's S3 anti-harvest
   (−0.301*) fired below its own prospective MDE80.

## SENSITIVITY NOTE (post-read, labeled; does not alter the executed verdicts)

The frozen BAR = 0.2026 came from the ideation-doc substrate screen.
The repo-tracked reproduction (`artifacts/substrate_screen_20260821/`,
non-registered archived compute) **could not reproduce that ceiling
figure under any definition tried** (per-state decision-sd: dv3 0.084 /
tm2 0.056; per-cell wdc-form: ~0 / 0.040), while the pedestal
reproduces bit-exactly. Under the reproduced (lower) ceilings, dv3's
Stage-A floor (0.142) sits ABOVE the substrate ceiling, and the frozen
M7 conjunction would have read **CCD-PC-UNDERPOWERED** instead of
licensing CCD-PC-FAILS. Consequence for drafts: the dv3
substrate-empty statement must NOT rest on the null+calibration
conjunction; it rests DIRECTLY on the screen's measured ceilings
(≈0.06–0.08 against a single-draw pedestal of 1.54/0.46), which is the
stronger and cleaner statement. This caveat travels with every use of
the dv3 CCD verdict.
