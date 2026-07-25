# Pixel swamping diagnostic — P-SW1 SWAMPING-CONSISTENT FIRES (2026-07-25)

Snapshot: `local_results/pixel_swamping_20260725_104532/` (resubmitted
E4 sweep after the Amendment-1 measure repair: 32/32 pixel fits scored
on the frozen `fingerpx_v1` probe set, 128 csv rows, zero skips). Read
exactly as registered (`prereg/PREREG_pixel_swamping_20260724.md` +
`_amend1_20260725.md`) with the pre-frozen
`analysis/pixel_swamping_read.py` (selfcheck PASS, frozen before any
pixel E4 number existed).

## Pipeline integrity

- Bundle code snapshots are md5-IDENTICAL to the frozen repo versions
  (both `pixel_swamping_read.py` and the amended
  `stratified_error.py`).
- Sweep complete: "measured 32 new, 0 already done, 0 skipped"; the
  probe set's FROZEN marker asserted by the sbatch.
- Local read on the snapshot csv reproduces the cluster-side json on
  every decisional field; the single differing value is
  `trivial_floor` at 5.6e-16 (cross-machine float summation,
  disclosed).
- Registered gate: trivial-predictor floor 0.671 < 2.0 ⇒ the read is
  INFORMATIVE (the swamping threshold was reachable).

## Registered read (seed-clustered bootstrap B=10K rng 0)

| quantity | value | rule |
|---|---|---|
| **task-arm in-regime rew-NLL, h0, sides pooled** | **23.34 [19.41, 27.29] nats** | CI entirely ≥ 2.0 ⇒ **P-SW1 SWAMPING-CONSISTENT** |
| trivial floor | 0.671 | < 2.0 ⇒ informative |
| proprio anchors (same task/labels/head class) | membership 1.02–1.21; non-membership 2.03–2.31 | frozen record |

The pixel task arm is not merely in the non-membership band — it sits
an order of magnitude above it. Reward information is absent from the
pixel trunk: the task arm's own reward head, trained on true reward
WITH trunk gradients, cannot predict in-regime reward at all.

## Descriptive context (never decisional)

- Side split: s0 (lo-occupancy) 24.5–58.0 nats; s1 (hi-occupancy)
  4.0–7.3 nats. The verdict is SIDE-ROBUST: every one of the 16
  task-arm runs individually sits ≥ 2.0. The hi-occupancy side retains
  more reward-predictive signal (occupancy helps even here), but never
  approaches the proprio membership band.
- Out-of-regime rew-NLL ≈ 4.5 (mostly zero-reward frames — easier than
  in-regime for a reward-blind trunk, as expected).
- d_errin at h0 spans −1.15..+16.27 across runs (dynamics in/out
  differences exist descriptively; no registered claim).
- Apt-arm true-reward NLL: absent-by-design (reward_aware gate; the
  read asserts the gate held on all 16 apt runs) — the disclosed
  non-readout.

## Registered consequences applied (frozen in the base prereg)

- **Paper 1's pixel boundary paragraph is written as PREDICTED
  STRUCTURE**: pixel reconstruction pushes the competition bar above
  every arm's inclusion bar (the bar framework's g > G2 regime; the
  ordering law P-E1 extended upward). The X2 all-cells-floor null is
  mechanism-complete: no cell — including task-hi — included θ, so no
  cell adapts fast. The membership→transfer link is NOT challenged
  (there was no membership to fail).
- **G-X3 stays NO-GO; no further pixel arms pre-deadline.**
- Theory bookkeeping: the X2 prereg's registered escape (non-monotone
  swamping regime) is now CONFIRMED as the surviving branch — the
  within-domain competition model gains a third leg (proprio arm
  ordering, orthogonal objective-specificity, pixel g > G2) the same
  week its cross-domain leg (P-SM1) was refuted.

## Provenance

- `pixel_swamping.json` — full read output (statistic/floor/sides/
  d_errin/verdict).
- `e4_fingerpx_v1.csv` — collated E4 rows (32 runs × 4 horizons).
- Read command: `python -m analysis.pixel_swamping_read --e4_csv <csv>
  --probeset <fingerpx_v1> --output <dir>`.
