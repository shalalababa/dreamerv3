# B-onset (Track B dose-onset ladder) — REGISTERED READ, 24 Aug
2026: **ONSET-CONFIRMED — the primary FIRES** (Track B's first
registered fire)

**Prereg:** `PREREG_trackB_onset_20260822.md` (R1-revised grid
{0.25, 0.30, 0.34, 0.38}) + Amendment 1
(`PREREG_trackB_onset_amend1_20260823.md`, pooled continuation).
**Reader:** `uncfield/se_onset_read.py` (frozen; RCC copy
sha-verified byte-identical to HEAD incl. se_read/se_dose_read),
ONE execution on RCC 24 Aug 09:15, output
`se_onset_read.json` (stamp git 8e91040 = the RCC checkout;
reader bytes = frozen HEAD).

## Verification chain (before the read)

- Committed manifest `manifests/b_onset_lite_20260824_084500.sha256`
  == RCC lite `MANIFEST.sha256` byte-exact; lite bundle verified in
  place 209/209 OK.
- The read consumed the FULL bundle
  `b_onset_20260824_084500/runroot` (the lite tree omits `replay/`,
  which the coverage leg opens): all 207 reader-opened files hashed
  DIRECTLY in the full tree against the committed manifest — 0
  mismatches; `replay/` verified per-run by npz count vs the full
  manifest (16/16 exact) + 3 witness hashes per run (48/48 OK).
- Amendment conditions verified by ops from producer logs (bundle
  NOTES.txt): instance-15 set is EXACTLY the four scale-complete
  round-1 runs; the 12 continuation runs all on instance 25, the
  four interrupted round-2 cells re-run CLEAN (no resume, first
  scores row step 16001); replay chunk counts show no stale-file
  contamination.

## Primary

**θ₁(distractor) increases in scale across the fresh onset window:
Spearman ρ = +0.897, one-sided positive label permutation
p = 3.0e-5 (100k draws, seed 20260822), n = 16/16 usable — FIRES.**
No collapsed runs, no floored runs, probe-ckpt identity OK,
fit counters 16/16 OK (realized 496k–499k of 5e5; standing-rule
check done). Per-scale θ₁:

| scale | θ₁ (4 seeds) |
|---|---|
| 0.25 | 0.113, 0.086, 0.130, 0.090 |
| 0.30 | 5.72, 4.55, 5.33, 3.80 |
| 0.34 | 7.34, 6.81, 7.62, 9.56 |
| 0.38 | 7.56, 7.60, 8.41, 9.31 |

## Registered descriptives

- **Logistic midpoint 0.275, 95% run-resample interval
  [0.27, 0.28], slope boundary-pinned (≈ a step), rising, in-window.**
  The misprice does not grow smoothly — it SWITCHES ON in a ±0.005
  window around scale ≈ 0.275.
- **Cost curve: coverage vs scale ρ = −0.097 (descriptive p .37)** —
  replay coverage does not trend with scale, so the onset is not a
  data-coverage artifact.
- **Anchor composition** (fresh window + B1 + Stage-1): 0.10 → 0.11;
  0.25 → bimodal {0.09, 0.13, 1.52, 1.55}; 0.30 → ~4.9; 0.34 → ~7.8;
  0.38 → ~8.2; 0.50 → 8.41; 0.75 → 6.56; flat@0.359 → 7.1;
  1.0 → 5.26. **B1's unexplained bimodal 0.25 cell is resolved: 0.25
  sits AT the threshold and its runs straddled it.** Global dose
  shape (descriptive): step-on at ≈0.275 → peak ≈0.4–0.5 → mild
  decline toward 1.0 — consistent with B1's registered REVERSED
  direction on its higher-scale grid (+0.728). The dose story is
  THRESHOLD-GATED, not dose-proportional.

## Amendment-1 compliance

- Pooled panel = 4 (instance 15, vast 48423745) + 12 (instance 25,
  vast 48475297); device is a balanced block (one instance-15 run
  per scale). Per-run devices: s96/s100/s104/s108 = instance 15;
  all other seeds (97–99, 101–103, 105–107, 109–111) = instance 25.
- **Registered same-box sensitivity (cl.4, separate dated
  computation `analysis/onset_samebox_sensitivity_20260824.py`,
  reader untouched): continuation-box-only Spearman on the 12
  instance-25 runs = ρ = +0.885, p = 3.1e-4** — descriptive-never-
  a-fire by registration (n = 12 < MIN_USABLE 14), and it
  REPRODUCES the pooled primary: the fire is not device-driven
  (`samebox_sensitivity.json`).
- Disclosed residual (cl.3, carried): a device × scale INTERACTION
  is not excluded by the balance — only the main effect is. Given
  the same-box reproduction and the ~50× effect size at the step,
  this residual is disclosed, not remediated.

## Disclosure: bundle-time gate sweep

The manifest commit (40c05600) notes "reader gates PASS at
n_perm=1000" — a bundle-time PER-RUN gate sweep by ops (read_run's
per-run machinery at its 1000-perm default; consistent with the
lite bundle's "with se_probe/: 16/16" completeness line). Before
this execution, no `se_onset_read.json` existed anywhere searched
on RCC (checkout artifacts/, scratch, output guard path clear), and
the cross-run primary (label permutation at seed 20260822, 100k
draws) exists only in this read. Per-run θ₁ values were computable
at bundle time; the registered decision structure (bars, grid,
direction) was frozen 22–23 Aug, before the wave ran.

## Consequences

- **Track B's dose axis lands its first registered fire, in the
  strongest available form**: a sharp, tightly-localized onset
  threshold (≈0.275 ± 0.005) for the misprice, reproduced
  same-box, with coverage ruled out as the carrier and the B1
  reversal composing into a coherent global dose shape.
- [writing chat] the B1 "direction reversed" paragraph upgrades to
  the two-window synthesis: rising limb + threshold onset (this
  read) then gentle decline (B1 grid + anchors); B1's bimodal 0.25
  cell gets its explanation; the NFI/misprice sections gain the
  threshold framing (misprice is switch-like in distractor scale).
- No further onset compute registered or needed.

Artifacts: `se_onset_read.json` (the consumed execution),
`samebox_sensitivity.json` (amendment cl.4).
