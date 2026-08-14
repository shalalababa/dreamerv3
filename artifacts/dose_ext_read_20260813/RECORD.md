# #21 dose wave — pooled look-2 (seeds 1–16) — ONE registered read — 2026-08-13

- Registration: `prereg/PREREG_dose_task_20260810.md` (7ecfb4ca…6e4abe)
  + `prereg/PREREG_dose_task_amend1_20260811.md` (961fca4e…7332cc,
  freeze-committed 8cd62845 — the Pocock two-look power extension,
  value-aware disclosure of look-1 inside). Reader
  `analysis/dose_task_read.py` VERBATIM, bitwise pin verified:
  0251271c13493c81ec709ede1f63d4fd3ed35a657467995b8bcd6ae38c59d9d7.
- Bundle: `local_results/dose_extension_20260813_222739` —
  MANIFEST.sha256 verified (650 entries, all OK; identical to the
  committed manifest) before any file was opened.
- Selfcheck PASS re-run immediately before execution. THIS is look
  2 — the LAST look (registered; no further extension).

## Pre-read gates (registered in the amendment; outputs here)

- **64/64 dual witness**: the literal registered block PASSED
  (update==total and ckpt 500000 for every `ax1wm_finger_td{0..3}_seed{1..16}`).
- **Pooled-integrity gate: NOT VOID** — read.json shows
  `excluded_submodal == []`, `modal_n_ep == 96` (the pinned look-1
  modal), `n_per_cell == {16,16,16,16}`; `excluded_qc == []`.
- **Operator-flagged adapts** (td2 s15/s16, td3 s11/s12/s16,
  "interrupted while marked done"): adjudicated BENIGN pre-read on
  witness columns only — ALL 64 adapts (flagged and not, both looks)
  share the identical realized endpoint (max step 112112, 112
  episodes, strictly increasing steps, 96 episodes at the 100k
  estimand window, qc_pass). The interruptions postdate the final
  logged episode. Disclosure: while checking, secondary columns
  (auc50k/auc125k/final10) of the 5 flagged rows were displayed; the
  governing pooled auc100k estimand was not.
- **E4 row gate**: realized `e4_dose_filtered.csv` = e4_dose.csv
  (260 rows) minus EXACTLY the four `ax1wm_finger_td1_seed99` smoke
  rows (256 kept; all 64 registered fits present).
  sha_pre 33cfb199a865653f7d956ef23cd30f00ccefd06cd3505a6d74158bc82e167c3c,
  sha_post ad2704c5a26a6f1efd11e569ca69829f46d9c3d3ac862e920ca49c00770ab027.
  **Instrument note (disclosed)**: the amendment's LITERAL filter
  block has a substring defect — `_seed9` matches `_seed99`, so as
  written it would have KEPT the contamination. The operator's
  realized filter enforced the gate's registered INTENT (seed99
  removal; `kept_ids == names` verified here independently). The
  read consumes the verified filtered csv.

## Verdict (amendment thresholds GOVERNING; reader's internal .05
## strings recorded but non-governing)

**G1-POOLED PASSES — the wave regains decision weight.**
G1 = level3 − level0 = **+102.6** [BCa +24.8, +200.2], perm
**p = .029297 < .0294** (Pocock K=2 boundary), n=16v16, occupancy
gates green (realized 0.0536/0.1305/0.2490/0.3234, all within ±0.02
of nominal; dose_manifest sha de8d31e5…8b4bf6). The pass is by
0.0007 in p — reported plainly; the honest α accounting stands as
registered (family-wise ≤ .0794 union bound, ≈ .06 realized).
DOSE-GRID-RETIRED does not fire.

- **P-DR1 (adjudicated at corrected thresholds)**: interior contrasts
  vs level0 — level1 +143.3 [+63.3, +227.7] perm p=.0025; level2
  +125.7 [+48.2, +197.1] perm p=.0032. BH (m=2: .0147/.0294): BOTH
  survive ∧ both CIs exclude 0 ⇒ **activation EARLY — the occupancy
  effect is present by the 0.131 cell** (re-derived; agrees with the
  reader's label).
- **P-DR2 (DEMOTED — descriptive verbatim, no confirmatory shape
  wording licensed)**: reader reports "THRESHOLD-AT-0.092: step form
  wins the calibrated CV comparison" (fold cuts uniform at 0.0920;
  margin .068 vs null crit .041; runner-up log). Confirmatory shape
  claims require a fresh single-look registration.
- **Fresh-only replication (REPLICATION-DESCRIPTIVE)**: seeds-9–16
  subset G1 = **+47.6** [−52.7, +119.0], perm p=.302, n=8v8 —
  sign-consistent, ~half magnitude, underpowered as disclosed (G1
  power ~0.50 at +69 on the registered basis). Computed via the
  registered block with the row-shape access CONFORMED (recorded:
  `load_auc` returns `(kept, modal, excluded, excluded_qc)`; rows
  keyed level/seed/auc) — output bit-identical to the operator's
  `fresh_only_replication.json`.
- Pooled cell means (descriptive): 141.7 / 285.0 / 267.4 / 244.3.
- P-DR3 membership (descriptive, never adjudicated): in-band from
  level1 up (1.79 / 1.39 / 1.00 / 0.99 vs bar 1.5); D1-leak overlap
  disclosed in `leak_overlap` (buffers predate the --holdout guard).

## R1 scoreboard consequences (registered in Amendment 1 / R1 docs)

- **A1′ PASSES** (G1-pooled at the corrected alpha) — R1's A1
  recovers at look 2; the executed look-1 miss remains on the
  scoreboard as look 1.
- **A2 (monotonicity, strong form): NOT FALSIFIED** — the registered
  falsifier (a decisive NON-MONOTONE / falling-limb P-DR1 pattern)
  did not fire; the descriptive decline after level1
  (285.0 → 267.4 → 244.3) is non-decisive at these thresholds.
- **A3/A4: remain UNADJUDICATED** at this look (P-DR2 demotion).

## Post-read notes (labeled, non-registered)

- The wave's story is now: a real, EARLY-activating occupancy effect
  (both interior contrasts decisively positive at n=16) with a
  descriptive step-then-plateau/decline shape — consistent with the
  R1 legibility-gating picture of a threshold rather than a
  dose-proportional response. The near-boundary G1 p is a direct
  consequence of the plateau: level3 is the WEAKEST interior cell,
  so the registered endpoint (3 vs 0) understates the interior
  effect its own P-DR1 contrasts show at p≈.003.
- This was the registered LAST look for this grid; any shape-level
  confirmation is a fresh registration on a new realization.
