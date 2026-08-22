# P2 Stage B — in-vivo certified injection, ONE read (22 Aug 2026)

Registration: PREREG_p2_stageb_injection_20260821.md (frozen 8828ab6f).
Bundle: wsb_20260822_171432 (manifest verified 0 failed; 8/8 cells +
2/2 registered dup passes; stamp meta pins delta=2.0 /
first_step_dim0_clip / stamped_steps == 16600 exact identity; env_seed
20260823; version '_w1sb'; duplicate-null exact zero).

## Verdict: **IN-VIVO-CALIBRATED**

| quantity | value |
|---|---|
| admissibility: median sd_m(planted) | 0.764 ≥ 0.20 — admitted (near the 0.75 pre-freeze context) |
| PRIMARY recovery slope (n=8) | **1.0045 [0.9966, 1.0243], perm p = .0081** — excludes 0, covers 1 |
| probe slope (witness) | 1.00005 [1.0000, 1.0001] |
| split-selected opportunity on stamped arrays (witness) | +0.720 [+0.461, +1.078], p = .0081 |
| consumer tilt (wiring alarm) | −0.039 [−0.078, −0.005], p = .118 — **no alarm** |

## What this settles

- **The end-to-end chain (live env, snapshot/restore, CRN marks, G
  accounting, selection estimator) recovers a real planted
  per-candidate signal at slope ≈ 1** — the in-vivo leg of the
  "calibrated instrument found nothing" sentence is bought. Magnitude
  claims need no correction factor.
- The probe referee reads the plant essentially exactly (slope
  1.00005), and the designed dissociation holds: the split-selected
  opportunity FIRES on stamped arrays (+0.72 — when value exists, the
  estimator finds it) while the stamp-ignorant frozen consumer shows
  no positive tilt (point negative, ns) — probe-harvests /
  consumer-does-not, exactly as constructed.
- Together with Stage A (Tier-1, 21 Aug) this completes the
  calibration ladder: archived-array calibration + live-environment
  calibration, both clean. Every "the instrument found nothing on the
  unstamped substrate" claim now carries an in-vivo positive control.
