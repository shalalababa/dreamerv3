# PREREG: pixel replication of the core interaction (X1 + X2) — frozen 2026-07-19

**Status: FROZEN at commit time. Amendments go in new dated files in this
directory, never edits to this one.** Registered per the gated plan
(research_notes/Plan_PixelReplication_Exec_20260718.md, Stage X1/X2);
G-X1 (feasibility) passed on the 19-Jul X0 smoke — throughput, storage,
pair machinery, and learnability all green (§7).

## 1. Claim under test

The occupancy × reward-supervision interaction confirmed in proprio
(W0 +98.7, n=16, perm p=.0027; E3v2 within-collector +84.0, 14/14
collectors) replicates when the world model trains on IMAGES ONLY.
Theory (Theory_SpectralTransfer_20260717.tex) predicts the effect is at
least as large in pixels: reconstruction competition g_k is stronger when
observations are dominated by task-irrelevant visual detail, so
reward-gradient rescue has more to rescue (**P-C2**, directional).

Domain: finger (dmc_finger_turn_hard) only. This is ONE registered
replication, not a re-run of P0–P3.

## 2. Instruments (all committed before any outcome exists)

- **Image-only WM:** `pixel_wm` config = size1m + `agent.model_obs: image`
  + train_ratio 256. Proprio keys stay in obs/replay for regime labeling
  but never enter enc/dec. **Audit anchor:** each wm run's saved
  config.yaml must show `model_obs: image` and the arm's `expl.mode`;
  `analysis/pixel_repl_read.py --audit_runroot` checks all 32
  (non-inferential).
- **Pair search:** `build_controlled_replay search-matched` (NEW,
  built + selfchecked 2026-07-19, synthetic data only). Rationale: the
  pooled `search` selects sides independently and its best X0 pair fully
  confounded collector composition with occupancy (side0 = one pure
  collector, side1 = five others) — the exact confound E3v2 had to fix in
  proprio; `search-within` de-confounds but a single collector holds only
  ~4–6 high-occ image episodes, below any usable K. `search-matched`
  pools collectors on the high side and forces the low side to repeat the
  high side's per-collector episode counts (**collector_l1 = 0 by
  construction**), with the frozen matching semantics (|dcov| ≤ cov_tol =
  cov_match_frac × raw source-coverage range; docc ≥ 3× median bootstrap
  occupancy noise) and **side0 = low occupancy by construction** (the
  Option C side inversion cannot recur). Note: the binned `source_l1`
  metric mechanically saturates for ANY occupancy pair (bins differ by
  design); `collector_l1` is the controlled quantity and is recorded in
  the manifest's confound deltas.
- **Read:** `analysis/pixel_repl_read.py`, committed with this file,
  selfchecked (fires/null/missing-cell/smoke-exclusion branches).

## 3. X1 — registered pair build (login node, before any X2 submission)

Pool = ALL image-bearing finger pretrain sources on scratch at build time
(currently 6: random_finger_seed{100,104,105}, p2e_finger_seed{99,101},
apt_finger_seed101, ~496 modal-length episodes; if additional RENDER
sources exist by build time they enter the index and are enumerated in
the fill record). **Every source must carry the image key** (the 19-Jul
X0 first-attempt failure was an image-less episode in the buffer):

```bash
# 0) verify image keys (fails loudly on any image-less source):
python -c "import numpy as np,glob,sys; [sys.exit('no image: '+d) for d in sys.argv[1:] if 'image' not in np.load(sorted(glob.glob(d+'/*.npz'))[0])]" \
  $RUNROOT/pretrain_{random_finger_seed100,random_finger_seed104,random_finger_seed105,p2e_finger_seed99,p2e_finger_seed101,apt_finger_seed101}/replay
# 1) index (image-bearing sources only):
python -m probing.build_controlled_replay index \
  --replay random100=... random104=... random105=... p2e99=... p2e101=... apt101=... \
  --task dmc_finger_turn_hard --output $RUNROOT/pixel_x0/episodes_x1.json
# 2) collector-matched pair (K ladder below):
python -m probing.build_controlled_replay search-matched \
  --index $RUNROOT/pixel_x0/episodes_x1.json \
  --ref_replay $RUNROOT/pretrain_apt_finger_seed101/replay \
  --n_episodes 20 --output $RUNROOT/pixel_x0/pairs_pxq1m.json
# 3) build:
python -m probing.build_controlled_replay build \
  --index $RUNROOT/pixel_x0/episodes_x1.json \
  --pairs $RUNROOT/pixel_x0/pairs_pxq1m.json \
  --which q1 --output_root $RUNROOT/axis1_finger/pxq1m
```

- **K ladder (registered):** n_episodes 20, else 16, else 12 — first K
  that is non-SHORT AND meets the quality gate. **Quality gate (buffer
  properties, decidable pre-outcome):** hi-side occ ≥ 0.30 and lo-side
  occ ≤ 0.05 (mirroring the realized proprio contrast ≈.32/.05), docc ≥
  occ_sep_min, dcov ≤ cov_tol, collector_l1 = 0. If K=12 fails the gate,
  X1 is BLOCKED and reported — no silent loosening of any threshold.
- **Fill record:** the realized index composition, chosen K, pair stats
  (dcov/docc/side occs/mixtures), and manifest confound deltas are
  recorded in `artifacts/pixel_x1_pair_<date>/PAIR.md` BEFORE the first
  X2 submission; `$RUNROOT/axis1_finger/pxq1m/manifest.json` is the
  canonical audit object. No re-search after any X2 job starts.

## 4. X2 — main wave (32 fit+adapt jobs)

{task, apt} × {side0, side1} × seeds 1–8; 500,000 updates; adapt =
frozen readout, 1.25e5 steps, task-mode (adaptation is supposed to see
reward). Same pixel_wm config both arms and sides (capacity and
gradient-count equalized by construction).

```bash
# task arm (16 jobs):
AXIS1_BASE_CONFIG=pixel_wm AXIS1_ID_PREFIX=ax1px AXIS1_EXPL_MODE=task \
  AXIS1_DOMAINS=finger AXIS1_QUADS=pxq1m AXIS1_SEEDS="1 2 3 4 5 6 7 8" \
  AXIS1_BUNDLE_SIZE=1 AXIS1_BUNDLE_TIME=24:00:00 \
  ./scripts/submit_all.sh axis1-bundles
# reward-free arm (16 jobs):
AXIS1_BASE_CONFIG=pixel_wm AXIS1_ID_PREFIX=ax1fpx AXIS1_EXPL_MODE=apt \
  AXIS1_DOMAINS=finger AXIS1_QUADS=pxq1m AXIS1_SEEDS="1 2 3 4 5 6 7 8" \
  AXIS1_BUNDLE_SIZE=1 AXIS1_BUNDLE_TIME=24:00:00 \
  ./scripts/submit_all.sh axis1-bundles
```

Run naming (parses under the frozen `adaptation_auc` RUN_RE): adapt modes
`ax1pxpxq1ms{0,1}` (task) / `ax1fpxpxq1ms{0,1}` (apt); wm runs
`ax1wm_finger_pxpxq1ms{side}_seed{k}` / `ax1wm_finger_fpxpxq1ms{side}_seed{k}`.
Walltime basis (X0 smoke): 20K updates in 16–40 min ⇒ 500K ≈ 7–17 h;
adapt ≈ 0.5 h at fps/policy ~80–120 ⇒ single-job bundles at 24:00:00.
Buffers ≈ 31 MB/pair at K=10 ⇒ ≤ ~0.2 GB at K=20 (storage gate trivially
met). Submit from a clean shell (--export=ALL module-leakage gotcha).

## 5. Frozen analysis (analysis/pixel_repl_read.py)

- **Primary (sole inferential endpoint):** interaction
  I = mean over seeds of [(task: s1−s0) − (apt: s1−s0)] on **AUC100k
  only**; cluster bootstrap over seeds B=10,000 percentile CI,
  `np.random.default_rng(0)`; FIRES iff CI > 0. Decision on the CI alone.
- **P-C2 (directional secondary, never a gate):** evaluated only if the
  primary fires; "meets" iff the point estimate ≥ +84.0 (lower edge of
  the proprio anchor band [+84.0, +98.7] — same task, same reward scale,
  same AUC100k window; reported directionally, never pooled with proprio).
- **Robustness-only descriptives:** arm simples with own CIs, exact
  sign-flip permutation p, d_z, leave-one-seed-out range.
- **QC:** frozen `adaptation_auc` rule (≥20 episodes ≤100K steps ⇒
  qc_pass); the reader asserts qc_pass on every consumed row.
- **Decision tree (frozen):**
  - FIRES ⇒ P-C2 read as above; **G-X3: distractor-amplification arm GO**
    (X3 gets its own prereg amendment before any run).
  - Does not fire ⇒ **G-X3 NO-GO**; registered next step is the
    reconstruction-swamping diagnostic (E4-style stratified reward-NLL
    over the 32 pixel fits — theory-consistent in the extreme-g_k limit),
    which gets its own registration before it is read. The null is
    reported as a registered family-scope boundary either way.

## 6. Exclusions

- Seed-99 X0 smoke runs (modes `ax1pxpxq1imgs*`): excluded by rule,
  cannot parse under the frozen mode regex, and are deleted after timing.
- The abandoned image-less `axis1_finger/pxq1` buffers (built from the
  pre-fix index; never produced a valid fit).

## 7. Ordering statement / disclosure

Known at freeze: all proprio outcomes (W0, E3v2, P3/Amendment-1,
stamping, Option C, synth) and the TD-MPC2 F0/F1 design. X0 smoke
sightings, disclosed fully: both seed-99 fits converged (wm_total ≈ 11.3
and 11.5 at 20K updates; loss set {con, dyn, image, rep, rew}, no proprio
heads); during adapt-log QC, episode scores ≈ 966 (s0) and ≈ 974 (s1)
were observed in the final log windows — single seed, smoke-scale
collector-confounded pair, **no AUC or side contrast was computed** and
none informs this design beyond "the pixel model can learn finger". No
`pxq1m` pair exists; no X1/X2 outcome exists. `search-matched` and the
read script have run on synthetic selfcheck data only. ICLR clock: X2
submits by ~mid-Aug for reads by early Sept.
