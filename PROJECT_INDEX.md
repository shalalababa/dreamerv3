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
- `r3_read.py` — R3 consumer-competence factorial read (frozen 25 Jul pre-outcome; Amendment 1 29 Jul: seed-grid repair + value-blind guards, committed pre-read; READ 29 Jul: P-R3a+P-R3b fire, P-R3c null ⇒ consumer-competence failure, `artifacts/r3_competence_20260729/`).
- `r3_reacher_read.py`, `r3_doubling_read.py` — R3 band-mover wave 1 (frozen 30 Jul pre-outcome, freeze 596a7f93). Reacher READ 31 Jul: REPLICATES (P-RRa opp +0.828\* + P-RRb gap +0.813\* fire, P-RRc null ⇒ three-domain fact; floors 3.1%), `artifacts/r3_reacher_20260731/`. Doubling: awaiting bundle.
- `r3_ladder_read.py` — candidate-aware ladder L0–L4 on the committed R3 labels (registered-descriptive; `PREREG_r3_ladder_20260730` + pre-read metric amendment `PREREG_r3_ladder_amend1_20260731` after the independent review caught the pooled-metric confound; READ EXECUTED 31 Jul: **NO PER-STATE SIGNATURE** — gap not state-legible, repair default targeting stands; `artifacts/r3_ladder_20260730/`).
- `stage0_secondk_read.py` — Stage-0 second-k density-proxy robustness (frozen 31 Jul pre-outcome, `PREREG_stage0_secondk_20260730`; existence-gated re-dump k∈{5,20}, calibrated two-mode integrity guard; SUBSTRATE-GONE = valid outcome).
- `tm2_r3_read.py` — TM2-R3 TD-MPC2 cross-family replication read (frozen 31 Jul pre-outcome, `PREREG_tm2_competence_20260730`; with `probing/tdmpc2_oracle_labels.py` labeler port + `tdmpc2_r3_train.py` + `scripts/tm2r3.sbatch`).
- `r3_xconsumer_read.py` — cross-checkpoint consumer read (frozen 31 Jul pre-outcome, `PREREG_r3_amend2_20260730`; per-state-paired via counter-restored overlay; pairing QUARANTINE gate).
- `repair_read.py` — competence-repair read (frozen 31 Jul pre-outcome, `PREREG_competence_repair_20260730`; with `d0/train_consumer_model.py` LORO trainer; committed labels = per-state control via determinism gate).
- `p3_factorial_read.py` — P3 gradient-path factorial + Amendment 1.
- `stamping_read.py`, `scaling_read.py`, `optc_amend_read.py` — stamping + scaling pilot + Option-C corrective.
- `synth_phaseb_read.py`, `synth_diagnosis_read.py`, `synth_rediag_read.py`, `synth_phasebpp_read.py` — synth series.
- `tdmpc2_amend_read.py`, `tdmpc2_f2_read.py` — TD-MPC2 family boundary.
- `pixel_repl_read.py` — pixel X2 (read 24 Jul: null, G-X3 NO-GO).
- `pixel_swamping_read.py` — pixel swamping diagnostic read (frozen 25 Jul pre-outcome; task-arm rew-NLL level + trivial two-hot floor guard; awaits the resubmitted E4 csv).
- `unfrozen_calib_read.py` (read 25 Jul: fires ×4.3 amplified), `vgo_extended_read.py` (running) — Paper-1 band-ledger waves.
- `scaling_read.py` (read 26 Jul: scale-robust), `compcapacity_read.py` (25m A→B point, frozen pre-outcome), `volume_repl_read.py` (P-E4a; READ 29 Jul: REPLICATES +331*, diversity direction holds — `artifacts/volume_repl_20260729/`) — capacity axis.
- `unfrozen_stress_read.py` — U1–U4 unfrozen stress waves (frozen 27 Jul pre-outcome; subcommands u1–u4; reward-free-null / stamping / orthogonal / pixel protocol stress tests from the frozen-protocol audit; READ u2/u3/u4 30 Jul: robust/robust/floor-censored, no flips — `artifacts/unfrozen_stress_u234_20260730/`; u1 pending).
- `orthogonal_obj_read.py` — orthogonal-objective band leg (read 25 Jul: objective-specific).
- `collusion_confirm_read.py` — Paper-4 confirmatory read (frozen 30 Jul pre-outcome; P-CLB1/2 composition + P-CLA1/2 dissociation primaries, exact-replay/baseline/population/manipulation gates, strict csv hygiene).
- `pe_pixel_read.py` — pretrained-encoder pixel arm read (frozen 30 Jul pre-outcome; P-PE1 graded inclusion-restoration vs pinned swamping baseline + P-PE2 lift vs committed X2 cells + conditional P-PE3; companion gate `scripts/check_frozen_enc.py`).
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
- `collusion/` — Paper 4 CPU harness (24 Jul; v3/v2 30 Jul): `env.py`
  (Calvano logit duopoly, equilibrium anchors), `pilot.py` (coupled
  Q-learning sessions, IR fingerprint, coverage/coverage_late,
  undercut + terminal/exogenous distance punishment labels,
  --alpha/--beta knobs, Design-A biased-exploration interventions
  --explore_mode forbid/force + --explore_mix at matched budget;
  selfcheck PASS), `designb.py` (recorded sessions + float-exact
  offline stream replay + graded volume-controlled down-weighting w/
  size-matched random controls + Δ_price columns; selfcheck PASS).
  Records: artifacts/collusion_stage1_20260725 +
  collusion_stage23_20260725 + collusion_calib_20260730 (calibration:
  terminal reference frozen, Design-A dissociation smoke);
  confirmatory sweep registered `PREREG_collusion_confirm_20260730`.
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
- `bundle_manifest.sh` — bundle sha256 manifests (generate at source /
  verify before reads; see `manifests/README.md`).
- `runroot_cleanup.sh` — $RUNROOT dry-run classifier (KEEP/DELETE
  tiers from the 30-Jul record sweep; `CONFIRM=DELETE` removes
  DELETE-SAFE only).

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
- Paper-1 writing pack (30 Jul, adversarially reviewed):
  `Paper1_Distillation_20260730.md` (prof deliverable),
  `Paper1_ResponseCurves_20260730.md` (+`figures_paper1_20260730/`),
  `Paper1_TheoryAssumptions_20260730.md`,
  `Paper1_ProseHygiene_20260730.md`,
  `Paper1_PracticalCorollary_20260730.md`,
  `Theory_HighFR_Prediction_20260730.md` (companion to the committed
  `prereg/PREREG_highfr_theory_20260730.md` freeze).
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
- `Research_Methodology_Directions_Claims_20260801.tex` — standing
  methodology note: direction / research-question / claim taxonomy,
  six-level claim-granularity ladder, and the lit-check placement rule
  (shallow check to start, exhaustive audit at claim-freeze); governs
  new-direction selection for Papers 4/5+.
- `Research_Idea_Uncertainty_field_20260727.tex` (external review, 7
  directions, recommends Belief Hydrodynamics) +
  `Research_Idea_Uncertainty_field_Fable_20260727.tex` (companion:
  Riccati/determinism backbone, constitutive-laws program, repo
  bridges, composition deltas + costed pilot) — Paper-5 seed
  (solo project, own clock, finish-early; at most late-stage
  prof review — the notes' "collaboration" framing is superseded
  30 Jul).
- `Audit_FrozenProtocol_20260726.md` (+ `_claims_*.json`) — 47-claim
  frozen-vs-unfrozen protocol audit; U1–U4 stress-wave design.
- `archive/` — superseded docs.

## Data [gitignored]

- `local_results/` — result bundles rsync'd from cluster/local boxes
  (each with `code/` snapshot + `logs/` + csvs/npz; the artifacts/
  records cite these as provenance). Untracked + gitignored since
  30 Jul (bundles committed before then remain in git history): bundles
  move by direct rsync, and each registered bundle's bytes are pinned
  by a tracked `manifests/<bundle>.sha256` (generated at the source via
  `scripts/bundle_manifest.sh generate`, verified before any frozen
  read via `... verify <bundle> manifests/<bundle>.sha256`).
- Cluster runroot (not in repo): see the DreamerV3-cluster-layout
  memory / `scripts/env.sh` for paths.
