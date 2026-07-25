# Project Index

Annotated map of the study's files. Companion to `STUDY_LEDGER.md`
(all experiments → results → decisions). Tracked unless marked
**[gitignored]**.

## The four record surfaces (read these first)

- `STUDY_LEDGER.md` — every experiment: registration → record →
  verdict → decision. The index.
- `prereg/` — the immutable registration record. One dated file per
  registered experiment/amendment; NEVER edited after its freeze
  commit (fixes = new dated amendment files, committed BEFORE
  outcomes exist). Historical exception: `analysis/PREREG_phase5a.md`
  (pre-dates the prereg/ convention).
- `artifacts/` — one directory per read: RESULTS.md (or
  DECISION/AUDIT/PAIR.md for early/aux waves) + machine json + csvs.
  The results record. Invalidation notes are prepended, never
  overwrite history.
- `analysis/DEVIATIONS.md` — chronological log: true deviations from
  frozen specs (top section) + code notes/registrations (bottom
  section). The decision/process record.

Plans and status: `research_notes/` **[gitignored]** — see below.

## Analysis reads (`analysis/`)

Frozen pre-outcome readers, one per registered experiment; each has a
`--selfcheck` with planted-positive/null cases.

- `adaptation_auc.py` — collector: adapt-run scores → AUC csv (all waves consume this).
- `gate_d1_read.py`, `gate_d1_r2_read.py`, `d1_ladder_read.py`, `d1_shift_read.py` — Paper-2 D1 series (shift read = the relabel re-reader, version `d1fix_20260724`; read 25 Jul: corrected negative).
- `r3_read.py` — R3 consumer-competence factorial read (frozen 25 Jul pre-outcome; opportunity/achieved/gap primaries; driver `scripts/r3_local.sh`).
- `p3_factorial_read.py` — P3 gradient-path factorial + Amendment 1.
- `stamping_read.py`, `scaling_read.py`, `optc_amend_read.py` — stamping + scaling pilot + Option-C corrective.
- `synth_phaseb_read.py`, `synth_diagnosis_read.py`, `synth_rediag_read.py`, `synth_phasebpp_read.py` — synth series.
- `tdmpc2_amend_read.py`, `tdmpc2_f2_read.py` — TD-MPC2 family boundary.
- `pixel_repl_read.py` — pixel X2 (read 24 Jul: null, G-X3 NO-GO).
- `pixel_swamping_read.py` — pixel swamping diagnostic read (frozen 25 Jul pre-outcome; task-arm rew-NLL level + trivial two-hot floor guard; awaits the resubmitted E4 csv).
- `unfrozen_calib_read.py`, `vgo_extended_read.py` — Paper-1 band-ledger waves (pending submission).
- `orthogonal_obj_read.py` — orthogonal-objective band leg (read 25 Jul: objective-specific).
- `buffer_battery.py`, `collate_drivers.py`, `fit_mixed_effects.py` — shared statistics/collation helpers.

## Instruments

- `d0/` — Gate-D0/D1 measurement stack: `oracle_labels.py` (the
  restorable-state labeler; current version `d1fix_20260724`),
  `sweep.py` (offline signal sweep + `load_config`/`load_frozen_agent`),
  `signals.py`/`analysis.py`/`synthetic.py`/`selfcheck.py`.
- `probing/` — training-adjacent instruments:
  `offline_fit.py` (static-replay WM fits — read-only on buffers),
  `build_controlled_replay.py` (occupancy-controlled buffer builder),
  `stratified_error.py` (E4 NLL panels; err_diff = out−in),
  `latent_uq.py` + `latent_uq_analysis.py` (Stage 0/1A; the
  density-deconfounded residual), `wm_evaluator.py` (Goodhart, demoted),
  `tdmpc2_*.py` (family-boundary bridge), `probeset.py`/`regimes.py`/
  `resize_replay_context.py` (probe sets, regime labels, ctx rebuilds),
  `spectral_measure.py` (24 Jul: λ-spectrum / reward-direction
  variance-rank pass on raw buffers; P-SM1,
  PREREG_spectral_domains_20260724).
- `collusion/` — Paper 4 CPU harness (24 Jul, v2 25 Jul): `env.py`
  (Calvano logit duopoly, equilibrium anchors), `pilot.py` (coupled
  Q-learning sessions, IR fingerprint, coverage/coverage_late,
  undercut + distance punishment labels, --alpha/--beta knobs;
  selfcheck PASS), `designb.py` (recorded sessions + float-exact
  offline stream replay + volume-controlled composition interventions;
  selfcheck PASS). Pilot records: artifacts/collusion_stage1_20260725
  + collusion_stage23_20260725.
- `embodied/envs/distractor.py` — OU distractor wrapper (dose arms;
  restorable via oracle_get/set_state).
- `embodied/envs/orthreward.py` — orthogonal-objective reward override
  (finger spin-on-turn_hard).
- `embodied/envs/synthpred.py` — synthetic predictable-heads env.

## Core training code (modified upstream DreamerV3)

- `dreamerv3/agent.py` — arm gradient gating (`reward_grad`/
  `repval_grad` sg placement), valens critic ensemble + disag probe
  (measurement-only), `_d0_signals` (per-state Q matrices), expl modes.
- `dreamerv3/configs.yaml` — named configs: `dmc_proprio`, `pixel_wm`,
  `d0_probe`, `d0_dose1/2/3`, `expl_random/p2e/apt`, `size1m/12m/...`,
  `frozen_readout`, `unfrozen_readout`, `orth_spin(_frozen)`.
- `dreamerv3/main.py` — env construction (orthreward/distractor
  wrapping), replay, drivers.
- `embodied/` — upstream runtime; study-relevant details:
  `jax/internal.py` (transfer-guard arming on agent construction),
  `jax/agent.py` (policy-RNG counter, `load(regex)` partial loads),
  `core/replay.py`, `core/streams.py` (Consec).

## Cluster / execution (`scripts/`)

- `axis1.sbatch` (+`_bundle`) — fit+adapt pipeline; derives arm flags
  from `AXIS1_ARM`; `AXIS1_ADAPT_CONFIG` selects readout config.
- `adapt.sbatch`, `pretrain.sbatch`, `measure.sbatch`, `pilot.sbatch`
  (+bundles) — earlier-phase drivers.
- `e4_measure.sbatch` — E4 NLL passes (`E4_DOMAINS`, `E4_GLOB_<dom>`).
- `d0*.sbatch`, `submit_d0.sh` — Gate-D0 grids.
- `tdmpc2.sbatch` + `tdmpc2_env_setup.sh` — family-boundary runs.
- `goodhart.sbatch` — demoted track.
- `d1r2_local.sh`, `d1shift_local.sh` — local 5090 drivers (R2; shift
  relabel: pilots→smoke→labels into `d1_labels_fix/`).
- `d1pilot_relabel_local.sh` — relabel Amendment-1 companion driver
  (pull cluster d1pilot ckpts → smoke → 18 passes; same
  `d1_labels_fix/`).
- `submit_all.sh`, `run_status.py`, `env.sh` — submission wrapper
  (clean-shell rule: --export=ALL module-leakage gotcha), status,
  environment.

## Plans, theory, reviews — `research_notes/` [gitignored]

- `Roadmap_20260718.md` — the live status board (bands A–D + standing
  tracks), updated every read.
- `Research_Plan_v4_20260711.tex` — Paper-1 plan of record (v4).
- `Theory_SpectralTransfer_20260717.tex` (+ decoder-free addendum
  18 Jul + comp×capacity addendum
  `Theory_CompCapacity_Addendum_20260724.tex`) — spectral-competition
  model behind the P-A/P-B/P-E predictions.
- `Direction_Review_TheoryMapping_20260724.md` — ledger sweep + method
  mapping (λ/βa²/g table) + SOTA sequencing.
- `Note_ExplainedBoundary_20260724.md` — draft-section note: TD-MPC2
  boundary as predicted structure + "inclusion ≠ usefulness" named
  claim.
- Paper 3/4 launches: `Plan_CompCapacity_Launch_20260724.md`,
  `Plan_Collusion_Launch_20260724.md`,
  `Design_Collusion_Pilot_20260724.md` (pilot design of record).
- `Research_Branch_Ideas_Triage_20260717.tex` — branch-idea scoring;
  band definitions for the 40–55% claim strength.
- Exec plans: `Plan_Stamping_20260717.md`,
  `Plan_ScalingPilot_Exec_20260717.md`,
  `Plan_PixelReplication_Exec_20260718.md`,
  `Plan_TDMPC2_FamilyBoundary_Exec_20260718.md`,
  `Plan_PredictableHeads_20260717.md`.
- Editorial reviews: `Research_Editorial_Review_v3/v4_20260711.tex`,
  `Research_Editorial_Review_EVPI_20260711.tex`.
- `other research/` — Paper 2 (EVPI): `EVPI_theory_note_20260702.tex`
  + addendum, plan revisions v2–v4
  (`EVPI_Plan_Revision_20260712/23/24.md` — v4 current), external
  reviews (`D1_GPT_Analysis_20260723/24.md`), estimator checks; plus
  `Idea_*.tex` briefs (side-project triage).
- `Research_Other_Ideas_20260701.tex` — field scans / future
  directions.
- `archive/` — superseded docs.

## Data [gitignored]

- `local_results/` — pulled result bundles from cluster/local boxes
  (each with `code/` snapshot + `logs/` + csvs/npz; the artifacts/
  records cite these as provenance).
- Cluster runroot (not in repo): see the DreamerV3-cluster-layout
  memory / `scripts/env.sh` for paths.
