# U2/U3/U4 unfrozen stress reads — 2026-07-30

Registered: `prereg/PREREG_unfrozen_u2_20260727.md`,
`PREREG_unfrozen_u3_20260727.md`, `PREREG_unfrozen_u4_20260727.md`
(committed with the U-wave freeze BEFORE any `ax1uz*` run existed).
Reader: `analysis/unfrozen_stress_read.py` subcommands u2/u3/u4 (frozen
pre-outcome; selfcheck PASS). ONE execution per wave, run cluster-side
against the canonical collated csv; the jsons here are those executions'
outputs verbatim. U1 is NOT read here (stragglers still draining at
read time; its registration is untouched).

## Verdicts

| Wave | Primary | Result | Verdict |
|---|---|---|---|
| U2 stamped unfrozen | P-U2 [B_srd − B_sid] | **−1.27 [−25.50, +21.02]**, 5/8 pos | **PROTOCOL-ROBUST** (no flip; CI straddles 0, so the CI<0 outside-accounts branch also does not engage) |
| U3 orthogonal unfrozen | P-U3 [rgo − sgb] on spin | **+5.32 [−13.60, +23.62]**, 5/8 pos; pooled level 67.9 ≥ 20 floor gate | **PROTOCOL-ROBUST** (above floor, no flip) |
| U4 pixel unfrozen 2×2 | P-U4a floor relief (gate) | **+1.66 [−10.61, +13.44]**, 5/8 pos ⇒ does not fire | **FLOOR-CENSORED** — interaction unadjudicable per the registered guard |

All statistics: per-seed cluster bootstrap B=10K rng 0, AUC100k,
domain finger, milestone 500000, seeds 1–8, qc asserted.

## Registered consequences (frozen maps, applied as written)

- **U2**: the installed stamp (stamp-NLL previously shown in the
  included band) is inert even under full fine-tuning — "inclusion ≠
  usefulness" upgrades from a frozen-readout fact to a TWO-PROTOCOL
  fact; the UNREAL/RaMP-style refutation strengthens (aux-stamped
  directions are not even a useful initialization); P-B1's standing
  improves. Registered prediction (no flip) LANDS.
- **U3**: objective-specificity holds as an INITIALIZATION fact, not
  just a frozen-feature fact — the band's orthogonal leg upgrades to
  two-protocol standing; the strict "support must be legible to the
  objective" headline strengthens.
- **U4**: reported, no adjudication; the frozen pixel floor still
  binds; no further pixel-unfrozen compute pre-deadline; G-X3 stays
  NO-GO; P-SW1 (fit-time swamping fact) untouched. The P-SW1-derived
  no-differential-rescue prediction (P-U4b) is neither confirmed nor
  challenged — the censoring guard fired first.

## Secondaries / descriptors (never decisional)

- U2 simples: B_srd +0.17 [−15.48, +16.27]; B_sid +1.44 [−14.92, +20.88].
  Level changes vs frozen (`artifacts/stamping_20260718/auc.csv`):
  sid s0 +32.89* [+24.22, +42.98] 8/8, sid s1 +26.76* [+10.59, +46.72];
  srd0 s0 +13.60* [+2.24, +26.06]; srd0 s1 +2.15 ns, srd1 s0 +22.74 ns,
  srd1 s1 +1.58 ns — the placement CONTROLS gained most from
  unfreezing; the stamped cells gained less (directionally consistent
  with the stamp as dead weight; descriptor only).
- U3 side simples: s0 +7.96 ns, s1 +2.68 ns. Level changes vs frozen
  (`artifacts/orthogonal_obj_20260725/auc.csv`, seeds 1–8): rgo0
  +36.36* [+5.85, +64.07] 7/8, rgo1 +15.60 ns, sgb0 +7.52 ns, sgb1
  +19.71 ns — unfreezing helps on the spin objective too, but NOT
  differentially by arm.
- U4 cell means (AUC100k): t0 85.8, t1 80.6, f0 87.8, f1 76.3 — inside
  the frozen X2 floor range (77–83); unlike proprio (calibration
  +235/+30), pixel fine-tuning does not move levels. P-U4b descriptor
  (+6.36 [−26.85, +41.94]) reported per registration, NOT adjudicated.

## Verification (chain of custody)

- **Bundles** (first wave under results-sync policy v2):
  `local_results/unfrozen_u2_20260730_102053` (19,384 files),
  `unfrozen_u3_20260730_103614` (13,131), `unfrozen_u4_20260730_101152`
  (264). `scripts/bundle_manifest.sh verify` PASS 3/3 against
  source-generated MANIFEST.sha256; committed copies =
  `manifests/unfrozen_u{2,3,4}_2026…sha256`.
- **Code identity**: bundle snapshots of `unfrozen_stress_read.py` +
  `adaptation_auc.py` byte-identical (sha256) to the frozen committed
  readers in all three bundles.
- **Csv identity**: one canonical auc.csv, byte-identical across all
  three bundles (sha256 0617543b70d4…ca45); copied here as `auc.csv`.
  1495 rows; all 112 U2/U3/U4 rows present (48+32+32, n=8 per cell);
  qc_pass asserted by the reader. `exclusions.csv` (11): in-flight U1
  stragglers (6× ax1uz{t,f}q1 seeds 6–8) + in-flight 25m (4) + 1
  speedtest junk id — NONE in U2/U3/U4 scope. The shared canonical csv
  also carries partial in-flight U1/25m rows; per the one-read
  discipline they were not consumed decisionally (each subcommand
  loads only its own registered modes).
- **Re-execution**: all three reads re-executed locally on the bundle
  csv — every statistic in all three jsons reproduces EXACTLY (only
  the two machine-path provenance fields differ).
- **Config audits** (3 independent agent passes, one per wave):
  U2 48/48, U3 32/32, U4 32/32 — zero mismatches on any registered
  dial. Verified per run: unfrozen signature (`agent.frozen_wm: false`
  + `run.from_checkpoint_regex '^(enc|dyn|dec)/'`; U3 additionally
  `orthreward.task: finger_spin`; U4 additionally `model_obs: image` +
  `dmc.image: true`), task/seed/steps/logdir, WM checkpoint path
  matching the run id's arm/side/seed with ckpt step 500000, replay
  cell + manifest.json present (prereg existence condition), scores
  complete (uniform structural endpoint 112112 = last 16-env×1001-step
  episode wave under the 125k budget — arm-comparable), rc=0 ×112,
  "offline WM already fit" in every log ⇒ NO refits triggered (refit
  disclosure clean). Caveat disclosed: the parent sbatch REPLAY /
  AXIS1_ARM / AXIS1_EXPL_MODE export strings are not byte-recorded in
  the light bundles (stage 1 skipped everywhere, so REPLAY was only a
  manifest-existence guard); their values are established indirectly
  via the WM checkpoint paths and the WM configs' fit flags (rgo ⇒
  reward_grad=True/repval_grad=False, sgb ⇒ both False; fpxpx ⇒ apt
  WM), all matching 32-48/32-48.
- Baselines for level-change descriptors = the committed artifact csvs
  named in the registrations (stamping_20260718, orthogonal_obj_20260725
  seeds-1–8 rows, pixel_x2_20260724).

## Files

- `unfrozen_u2.json`, `unfrozen_u3.json`, `unfrozen_u4.json` — the
  cluster-side read outputs (verbatim).
- `auc.csv`, `exclusions.csv` — canonical collated csv + exclusions.
