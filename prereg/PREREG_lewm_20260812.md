# PREREG: LeWM / JEPA-family arm (P-J1) — 2026-08-12

User GO 2026-08-12 ("proceed with all paper 1 GO remaining stuff";
slate decided 11 Aug: LeWM in-plan tier 2 — "train LeWM on our
existing replay buffers, transplant its encoder into our adaptation
protocol via the frozen_enc/partial-init machinery built for the pe
pixel arm"). Grades `PREREG_theory_R1_gating_20260811` **P-J1**
(latent-prediction objectives include more reward-alignable support
than pixel reconstruction), operationalized per
`PREREG_theory_R1_gating_amend2_20260812` (the pixel substrate is
floor-censored behaviorally — pe read; the representation legs carry
the primary weight).

External instrument: **LeWM** (LeCun et al., arXiv 2603.19312, MIT
license), pinned checkout `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
(`scripts/lewm_env_setup.sh`); reward-free BY CONSTRUCTION (objective =
next-embedding prediction + Gaussian latent regularizer; no reward
term exists). Data interface = stable-worldmodel 0.1.1 HDF5
(exporter `probing/lewm_export.py`, schema validated against the
vendored 0.1.1 source, selfcheck PASS locally).

## Honesty block

Known at freeze: every executed read through 2026-08-12; the anchors
this wave consumes — pe P-PE1 rew-NLL **23.032 [20.143, 25.676]
NO-RELIEF** + pe behavioral **−8.54 [−25.97, +9.62]** vs X2
(`artifacts/pe_pixel_read_20260802/pe_pixel.json`); X2 task baseline
cells (committed `artifacts/pixel_x2_20260724/auc.csv`); swamping band
23.34 [19.41, 27.29] + trivial floor 0.6713; dv3 apt pixel legibility
(fpxpx) has NOT been ridge-probed — that anchor is measured inside
this wave. The pe per-cell e4 csv is a LOCAL (untracked) file already
read on 2 Aug — value-aware input, disclosed; pinned by sha256
`9907d7337a84c02c0820473c0455cd033dce200f5fde069bfde119d6892cc6e8`
(`local_results/pe_pixel_20260802_165601/e4_fingerpx_v1_pe.csv`; the
reader refuses on sha mismatch; the csv ships in this wave's bundle).
Unknown: every LeWM / distill / jp-fit quantity.

**Registered instrument caveat (the reviewer's first objection,
pre-answered):** the graft tests a DISTILLED student of LeWM's
encoder, not LeWM's ViT weights (framework boundary: PyTorch ViT vs
JAX CNN). Three mitigations are part of the design: (a) the
distillation fidelity gate below; (b) a **distill-free leg** (P-J1-emb)
that probes the RAW LeWM embeddings directly against the recon-WM
features under the identical estimator; (c) the stage-2 protocol is
byte-identical to the executed pe arm, so the recon-vs-JEPA comparison
is donor-swap-only.

## Design — per (side ∈ {0,1} × seed ∈ 1..8): one LeWM, one distill, one graft-fit, one adapt

Substrate: the frozen finger pixel pair `axis1_finger/pxq1m/side{0,1}`
(KEEP_SUBSTRATE) — the same reward-free data the fpxpx (recon) donors
consumed. All commands registered; FILL slots are marked and resolve
at the registered smokes BEFORE the corresponding full stage (fill
record `artifacts/lewm_fill_<date>/FILL.md`, written pre-stage).

0. **Env + selfchecks (registered gates)**: `scripts/lewm_env_setup.sh`
   (prints the pinned sha + format-registration smoke);
   `python -m probing.lewm_export --selfcheck` (lewm env);
   `python -m probing.distill_encoder selfcheck` and
   `python -m probing.embed_probe --selfcheck` (dreamer env; both PASS
   locally 12 Aug, re-run cluster-side).
1. **Export ×2**:
   `python -m probing.lewm_export --side_dir $RUNROOT/axis1_finger/pxq1m/side<S> --output $RUNROOT/lewm_data/finger_pxq1m_side<S> --holdout 8`
2. **LeWM training ×16** (lewm env, single GPU each; run dirs
   `$RUNROOT/lewm_finger_s<S>_seed<F>`): `train.py` from the pinned
   checkout on the TRAIN split
   `$RUNROOT/lewm_data/finger_pxq1m_side<S>.h5` (the literal file — a
   directory reference would trip the reader's multiple-h5 ambiguity
   branch, review C9; the `_holdout.h5` sits alongside and must never
   be named), seed <F>, `img_size=64`, `frameskip=1`,
   `keys_to_load=[pixels,action]`, repo-default epochs.
   **FILL-AT-SMOKE**: the literal Hydra override spelling — audited
   fields `data.dataset.name/frameskip/keys_to_load`, **plus
   `data.dataset.num_steps` and any dataset/`transform` composition
   (review C3: the pixel-preprocessing path is part of the pin)** —
   `img_size`, `seed`, `output_model_name` — is pinned by ONE
   registered smoke training on side0 before the 16 full runs.
   Completion witness per run: `config.yaml` + `lewm_weights.ckpt`
   exist.
2b. **Embedding-pipeline POSITIVE CONTROL (registered gate, review
   C3):** at the smoke, `LEWM_CHECKOUT=<checkout> python -m
   probing.lewm_embed verify --episode <one pxq1m side0 chunk>
   --preproc <candidate> --output $RUNROOT/lewm_emb/` compares this
   wave's preprocessing against the pinned checkout's OWN
   training-time image-preprocessor and writes `embed_control.json`
   (pass ⟺ max|diff| < 1e-5). The passing `--preproc` value is THE
   PIN for every subsequent embed pass; `verify` fails loudly if the
   checkout preprocessor cannot be located (interface adjustment =
   dated amendment BEFORE any full training). **The frozen reader
   refuses the whole read unless embed_control passes and every LeWM
   embedding manifest carries the pinned preproc** — an instrument
   false-negative can never become the registered strengthened-negative
   claim.
3. **Embed ×32** (lewm env):
   `LEWM_CHECKOUT=<checkout> python -m probing.lewm_embed embed --run_dir $RUNROOT/lewm_finger_s<S>_seed<F> --side_dir $RUNROOT/axis1_finger/pxq1m/side<S> --preproc <pinned> --output $RUNROOT/lewm_emb/s<S>_seed<F>`
   and `... embed --run_dir ... --probeset $RUNROOT/e4_probesets/fingerpx_v1 --preproc <pinned> --output $RUNROOT/lewm_emb/s<S>_seed<F>_probeset`.
4. **Distill ×16** (dreamer env, GPU):
   `python -m probing.distill_encoder train --ref_run $RUNROOT/ax1wm_finger_fpxpxq1ms<S>_seed<F> --side_dir $RUNROOT/axis1_finger/pxq1m/side<S> --emb_dir $RUNROOT/lewm_emb/s<S>_seed<F> --holdout_list $RUNROOT/lewm_data/finger_pxq1m_side<S>.manifest.json --output $RUNROOT/lewm_distill_finger_s<S>_seed<F>`
   (ref_run = the seed/side-matched fpxpx donor — architecture source
   only; its WEIGHTS are never loaded: the distill selfcheck +
   `fidelity.json` + fresh-init code path witness this). **Fidelity
   gate (registered): holdout R² ≥ 0.5 on all 16**, else the GRAFT
   legs return INSTRUMENT-LIMITED (the emb leg is unaffected).
5. **Anchor probes ×16** (registered smoke first: ONE fpxpx pass must
   exit 0 — pixel-WM support check):
   `python -m probing.ridge_probe measure --run_logdir $RUNROOT/ax1wm_finger_fpxpxq1ms<S>_seed<F> --probeset $RUNROOT/e4_probesets/fingerpx_v1 --output $RUNROOT/lewm_probe/fpx_s<S>_seed<F>.json`
6. **Embedding probes ×16** (dreamer env, CPU):
   `python -m probing.embed_probe --emb $RUNROOT/lewm_emb/s<S>_seed<F>_probeset/probeset_emb.npz --embed_manifest $RUNROOT/lewm_emb/s<S>_seed<F>_probeset/embed_manifest.json --probeset $RUNROOT/e4_probesets/fingerpx_v1 --run_id lewm_finger_s<S>_seed<F> --output $RUNROOT/lewm_probe/`
7. **Stage-2 graft fits + adapts ×16** (the pe loop verbatim with the
   donor swapped; clean shell):

```bash
for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=36:00:00 \
    --job-name=adapt_ax1jppxq1ms${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1jppxq1ms${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_jppxq1ms${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/pxq1m/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_BASE_CONFIG=pixel_wm,AXIS1_INIT_WM=lewm_distill_finger_s${side}_seed${f} \
    $REPO/scripts/axis1.sbatch
done; done
```

   Smoke gate first (pe precedent): ONE 2000-update fit under the
   NON-MATCHING name `ax1wm_finger_jpsmoke_seed99` (review C14: a
   `jppxq1ms0_seed99` smoke would match the E4 glob; the jpsmoke name
   matches neither the glob nor the reader regexes; removed by hand
   after the gate) must show the `PARTIAL_INIT ... counters_reset=0`
   line and pass `check_frozen_enc.py` against
   `$RUNROOT/lewm_distill_finger_s0_seed99/ckpt/distill-0` — a seed-99
   distill donor built for the smoke only.
8. **Integrity sweep** (pe form) + the machine-readable producer
   (review C2 — the pe block only tee'd a text log; the reader
   requires json). Registered verbatim:

```bash
cd $RUNROOT && python - <<'PY'
import json, subprocess, glob, os
out = {}
for side in (0, 1):
  for f in range(1, 9):
    wm = f'ax1wm_finger_jppxq1ms{side}_seed{f}'
    dn = f'lewm_distill_finger_s{side}_seed{f}'
    s2 = sorted(glob.glob(f'{wm}/ckpt/*/done'))
    rc = 1
    if s2:
      rc = subprocess.call(
          ['python', os.path.expandvars('$REPO/scripts/check_frozen_enc.py'),
           os.path.dirname(s2[-1]), f'{dn}/ckpt/distill-0', '500000'])
    out[wm] = (rc == 0)
json.dump(out, open('jp_enc_checks.json', 'w'), indent=1)
bad = [k for k, v in out.items() if not v]
print(json.dumps(out, indent=1))
print('QUARANTINE:', bad if bad else 'none')
PY
```

   (run with `REPO` exported per `scripts/env.sh` convention; any
   failure quarantines that run BEFORE the read; the sweep stdout is
   also tee'd to `$RUNROOT/jp_integrity_sweep.log`).
9. **E4 pass** (the pe DIRECT form verbatim — review C1 BLOCKING: the
   `e4-measure` stage does NOT forward COLLATE and would have
   OVERWRITTEN the committed swamping csv):
   `PROBESET=$RUNROOT/e4_probesets/fingerpx_v1 GLOB='ax1wm_finger_jppx*' COLLATE=$RUNROOT/e4_fingerpx_v1_jp.csv sbatch $REPO/scripts/e4_measure.sbatch`
10. **AUC collate + fit witness**: `python -m analysis.adaptation_auc
    --runroot $RUNROOT --output analysis_out_jp/`; fit counters via the
    domains-wave witness block with regex
    `^ax1wm_finger_jppxq1ms[01]_seed[1-8]$` →
    `jp_fit_counters.json` / `jp_ckpt_steps.json`.
11. **Bundle**: `e4_fingerpx_v1_jp.csv` + the pe e4 csv (sha above) +
    `auc.csv` + 16 `fidelity.json` + `jp_enc_checks.json` +
    `jp_fit_counters.json` + `jp_ckpt_steps.json` +
    `embed_control.json` + 32 probe jsons + 16+16 embed manifests +
    jp fit/adapt config.yamls + the sweep log; rsync + sha256
    manifest. Operational notes: submission loops re-run until drained
    where capped; the shared `one_sample` statistical core lives in
    `analysis/domains_read.py` and is frozen jointly by three
    registrations (review B10). **Probeset-provenance disclosure
    (review C15):** at read time the RECORD states whether
    `fingerpx_v1`'s episodes overlap the `pxq1m` training episodes
    (from the two manifests' episode lists) — symmetric across arms
    either way, but it belongs in the record (8-Aug probe-leak
    precedent).

## Registered decision rules (frozen reader `analysis/lewm_read.py`)

ONE read execution. Gates, fail-closed: pe-csv sha256 equals the pin;
`embed_control.json` pass = true AND every LeWM embed manifest carries
the pinned preproc (review C3 — the read REFUSES otherwise); 16/16
`jp_enc_checks` true; 16/16 fit counters `update == total` and ckpt
step 500000; e4 grids complete (both sides × 8 seeds at h=0,
`reward_aware=1`, finite `rew_nll_in`) for BOTH the jp and pe csvs;
auc rows complete/unduplicated/qc-passed; probe jsons: 16 embed_probe
(probeset_id fingerpx_v1, dims consistent) + 16 fpxpx ridge_probe
(`witness_match is not False`, no reward_override, run_id matches the
donor pattern); fidelity gate as above.

**Cross-era pairing disclosure (review C7):** P-J1-rep pairs new-era jp
fits against the 2-Aug executed pe cells by (seed, side). This is a
REGISTERED EXCEPTION to the same-era preference the TM2 bridge wave
enforces, justified: both arms consumed the SAME frozen `pxq1m` buffer
sides with seed/side-matched donor lineage; the estimand is a
level/paired difference, not any bitwise or derived pairing (which is
what the Midway3 ban targets); under cross-era nondeterminism the
pairing may buy no variance reduction, but the estimator stays
unbiased and the sign-flip test stays exact under the null of equal
distributions. Re-running 16 pe fits same-era was considered and
rejected as pure duplication of an executed protocol.

Estimands (permutation primary + BCa, standing rule; paired exact
sign-flip where paired):

- **P-J1-rep (PRIMARY, graft leg)**: per-seed paired delta
  `[pe − jp]` on side-mean in-regime rew-NLL (h=0), n = 8, one-sample
  BCa + exact sign-flip permutation.
  - CI > 0 ∧ p < .05 ⇒ **JEPA-MORE-LEGIBLE-THAN-RECON** (P-J1's
    "LeWM > apt" lands at the representation level: the JEPA-distilled
    encoder supports reward prediction better than the recon-pretrained
    encoder under the identical protocol).
  - CI < 0 ∧ p < .05 ⇒ **RECON-MORE-LEGIBLE** (P-J1 violated; reported).
  - else ⇒ **NO-SEPARATION** (MDE note).
  - Band descriptors (never decisional, pe vocabulary): jp
    seed-clustered CI hi < 2.0 ⇒ INCLUSION-RESTORED note; hi < 19.41 ⇒
    below the swamping band note.
- **P-J1-emb (distill-free representation leg)**: per-(side, seed)
  paired delta `[auroc(LeWM emb) − auroc(fpxpx feat)]` at the headline
  alpha (`alpha_0.001`, pooled AUROC vs reward > 0), n = 16, one-sample
  BCa + exact sign-flip.
  - CI > 0 ∧ p < .05 ⇒ **EMB-MORE-LEGIBLE** (objective-level
    confirmation, no distill proxy).
  - CI < 0 ∧ p < .05 ⇒ **EMB-LESS-LEGIBLE**; else **EMB-NO-SEPARATION**.
  - Degenerate-AUROC (None) rows REFUSE the leg (reported).
- **P-J1-beh (secondary, floor-disclosed)**: per-seed paired lift of
  the jp adapt cells over the committed X2 task baseline — the pe
  P-PE2 CELL/PAIRING STRUCTURE with the house one_sample estimator
  (BCa + exact sign-flip; review C11: the executed pe read used a
  percentile bootstrap, so this is estimator-upgraded, not verbatim;
  the quoted pe anchor −8.54 [−25.97, +9.62] is a percentile CI).
  CI > 0 ⇒ **LIFTS** (would be the spectacular outcome); CI < 0 ∧
  p < .05 ⇒ **BELOW-X2**; else **FLOOR-CONSISTENT** (the a-priori
  likely branch — the pixel behavioral floor is protocol-robust, U4).
- **P-J1 overall grading (per amend2)**: primary carrier = the two
  representation legs. BOTH fire positive ⇒ P-J1 CONFIRMED (strict
  form). EXACTLY ONE fires ⇒ P-J1 PARTIAL (the RECORD states which,
  and the distill-proxy caveat governs interpretation when only the
  graft leg fires). NEITHER ⇒ P-J1 NOT CONFIRMED (informative for the
  headline: the reward-free null is then objective-GENERAL across
  recon and latent-prediction pretraining at this substrate — itself a
  registered strengthening of the Paper-1 negative result, stated in
  exactly that form). **Conditionality (reviews C3/C8):** the
  strengthened-negative form is licensed ONLY because the read cannot
  execute without the embedding-pipeline positive control passing
  (reader gate) — a mis-preprocessed pipeline refuses instead of
  producing the claim. P-J1-beh never upgrades a verdict; a LIFTS
  outcome is reported as a discovery requiring its own confirmatory
  wave before any claim.

Descriptives (never decisional): per-cell NLL/AUROC/AUC panels,
LeWM-vs-TM2-free cross-family level note (R1's unscored clause),
fidelity R² panel, jp-vs-pe adapt cell means, and the FULL alpha-grid
AUROC curve for both probe arms (review C12: a single fixed alpha
compares feature spaces of very different dimension — the curve is the
registered capacity-confound sensitivity, reported never decisional).

## Power + risks

- P-J1-rep n=8 exact sign-flip: min two-sided p = .0078; MDE₈₀ paired
  d_z ≈ 1.156 (exact noncentral-t; review C10 — the decision rule
  itself is permutation+BCa) — an honest low-n primary; the pe−X2 NLL
  scale (points apart ≈ 0.3 vs band widths ≈ 5) makes large paired
  effects plausible but NOT assured; NO-SEPARATION licenses nothing.
- P-J1-emb n=16 exact: MDE₈₀ d_z ≈ 0.749.
- Registered risk: LeWM training instability on 64px DMC data (its
  paper claims pixel stability; if ≥ 2 of 16 trainings fail their
  completion witness, the wave pauses and the failure is disclosed
  BEFORE any substitution; no silent seed swaps).
- Registered risk: ridge_probe on pixel WMs is untested — the stage-5
  smoke gates it; a failure BLOCKS the emb leg's fpxpx side (not the
  graft legs) and is disclosed.

## Discipline

This file + `scripts/lewm_env_setup.sh` + `probing/lewm_export.py` +
`probing/lewm_embed.py` + `probing/distill_encoder.py` +
`probing/embed_probe.py` + `analysis/lewm_read.py` (selfcheck PASS) +
runroot_cleanup KEEPs are freeze-committed BEFORE any LeWM artifact
exists. ONE read execution. Reviewer: 2026-08-12 build batch (Opus 5,
sequential build→verification per the 12-Aug policy).
