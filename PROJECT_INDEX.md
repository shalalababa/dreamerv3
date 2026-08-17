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
- `r3_xconsumer_read.py` — cross-checkpoint consumer read (parent `PREREG_r3_amend2_20260730` frozen 31 Jul, two-pass form RETIRED UNEXECUTED after the 2-Aug detprobe FAIL; AMENDED 2 Aug by `PREREG_r3_xc_amend1_20260802`: 64 `_dual` passes, within-pass d_pair estimand from `--dual_chooser` labels (`_xc2` exact pin), QUARANTINE = within-pass integrity, value-blind `--dualgate`).
- `repair_read.py` — competence-repair read (frozen 31 Jul pre-outcome, `PREREG_competence_repair_20260730`; with `d0/train_consumer_model.py` LORO trainer; AMENDED 1 Aug by `PREREG_competence_repair_amend1_20260801` after the registered QUARANTINE: primary = within-pass shadow pairing `g_all[m_real] − g_all[m_real_probe]`, committed labels demoted to state-identity reference + provenance digest — no cross-job float comparison anywhere).
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
- `highfr_read.py` — high-f_R falling-limb read (frozen 2 Aug pre-buffer, `PREREG_highfr_wave_20260802` adjudicating `PREREG_highfr_theory_20260730`; P-HF1 falling limb vs archived v200s1 rows + monotone-rise-refutation / saturation branches + P-HF2 membership×diversity dissociation + P-HF3 interior max; instrument gates on `spectral_v1_1` f_rewarded).
- `buffer_battery.py`, `collate_drivers.py`, `fit_mixed_effects.py` — shared statistics/collation helpers.

## Instruments

- `d0/` — Gate-D0/D1 measurement stack: `oracle_labels.py` (the
  restorable-state labeler; current version `d1fix_20260724`),
  `sweep.py` (offline signal sweep + `load_config`/`load_frozen_agent`),
  `signals.py`/`analysis.py`/`synthetic.py`/`selfcheck.py`.
- `probing/` — training-adjacent instruments:
  `offline_fit.py` (static-replay WM fits — read-only on buffers),
  `build_controlled_replay.py` (occupancy-controlled buffer builder),
  `curate_frew.py` (2 Aug: deterministic high-f_R window curation →
  pairs json for the frozen builder; falling-limb wave),
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
- `uncfield/` — Paper-5 NFI/CEI harness (1–7 Aug): `lgfield.py`
  (LG world + exact Kalman referee; family-1), `dcfield.py` (7 Aug:
  discrete-chain world + exact 256-state joint Bayes referee w/
  observation-independence certificate; family-2), `planner.py`
  (world-parameterized cycle-sweep planner, three accounting modes,
  verdict tiers — thresholds shared across families), `learnedwm.py`
  (GRU + additive-LSTM belief WMs + planted adapters), `dcwm.py`
  (categorical belief WMs + dc planted battery), `pilot.py`/`sweeps.py`
  (family-1 pilots + sensitivity queue), `family2.py`/`family2_read.py`
  (generality factorial runner + frozen reader,
  `PREREG_nfi_family2_20260807`), `mechanism.py`/`ratio_research.py`/
  `gamma_rescore.py`/`cig_kernel.py`/`p1_rescore.py` (mechanism +
  fidelity arms + PRIME-P1), `cei2.py` (CEI v2 calibration-falseness
  pilot; `cei.py` = withdrawn v1, warning header), `residues.py`
  (family-1 diagnostics).
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

**Reorganized 7 Aug 2026** to paper-first/genre-second. Paths cited in
`prereg/`, `artifacts/`, `STUDY_LEDGER.md`, `analysis/DEVIATIONS.md`,
and source-file comments were NOT rewritten (the first two are
immutable; the frozen readers' shas are integrity checks) — resolve any
pre-7-Aug research_notes path via `research_notes/PATHMAP.md`.

Root: `README.md` (layout + conventions), `PATHMAP.md` (reorg
redirects), `TODO.md` (shared working list), `Roadmap_20260718.md` (the
live status board, bands A–D + standing tracks, updated every read).

- `paper1_wm_transfer/` — `Paper1_FullRecord_20260816.md` (complete
  research record; entry doc). `plan/` = `Research_Plan_v4_20260711.tex`
  (plan of record). `theory/` = `Theory_SpectralTransfer_20260717.tex`
  + decoder-free addendum 18 Jul + `Theory_HighFR_Prediction_20260730.md`
  (companion to the committed `prereg/PREREG_highfr_theory_20260730.md`
  freeze) — the spectral-competition model behind the P-A/P-B/P-E
  predictions. `exec/` = per-wave plans (`Plan_Stamping_20260717.md`,
  `Plan_ScalingPilot_Exec_20260717.md` + `Scaling_Pilot_Design_Brief_20260716.tex`,
  `Plan_PixelReplication_Exec_20260718.md`,
  `Plan_TDMPC2_FamilyBoundary_Exec_20260718.md`,
  `Plan_PredictableHeads_20260717.md`). `writing/` = the 30-Jul
  adversarially-reviewed pack (`Paper1_Distillation` = prof deliverable,
  `Paper1_ResponseCurves`, `Paper1_TheoryAssumptions`,
  `Paper1_ProseHygiene`, `Paper1_PracticalCorollary`,
  `Note_ExplainedBoundary_20260724.md` = TD-MPC2 boundary as predicted
  structure + the "inclusion ≠ usefulness" named claim). `audits/` =
  `Audit_FrozenProtocol_20260726.md` (+ `_claims_*.json`; 47-claim
  frozen-vs-unfrozen audit, U1–U4 stress-wave design) +
  `Direction_Review_TheoryMapping_20260724.md` (ledger sweep + λ/βa²/g
  method mapping + SOTA sequencing). `figures/paper1_20260730/` +
  `figures/theory_20260804/`.
- `paper2_evpi/` — `plan/` = `EVPI_Plan_Revision_20260712/23/24.md`
  (v4 = 24 Jul current); `theory/` = `EVPI_theory_note_20260702.tex` +
  addendum + memo; `design/` = competence-repair note + estimator
  checks (`evpi_estimator_check_20260702.py`); `writing/` = Route-B
  outline + `draft/` (the LaTeX sections); `reviews/` =
  `D1_GPT_Analysis_20260723/24.md`, Route-A revision options, econ
  reader review, `Research_Editorial_Review_EVPI_20260711.tex`.
- `paper3_comp_capacity/` — `Plan_CompCapacity_Launch_20260724.md` +
  `Theory_CompCapacity_Addendum_20260724.tex` (comp×capacity addendum
  to the spectral model; P-E1–P-E4).
- `paper4_collusion/` — `Plan_Collusion_Launch_20260724.md`,
  `Design_Collusion_Pilot_20260724.md` (pilot design of record),
  innovation audit, venue scan.
- `paper5_uncertainty_field/` — `ideation/` =
  `Research_Idea_Uncertainty_field_20260727.tex` (external review, 7
  directions, recommends Belief Hydrodynamics) + the Fable companion
  (Riccati/determinism backbone, constitutive-laws program, repo
  bridges, costed pilot) + object-vs-field connections note; `theory/`
  = `Theory_CEI_20260804.tex`; `design/` = `Design_CEI_v2_20260802.md`,
  `Pilot_NFI_Design_20260801.md`; `audits/` = innovation audits ×3,
  `NFI_ClaimFreeze_Audit_20260802.md`, `NFI_ScoopRescan_20260804.md`.
  Solo project, own clock, finish-early; at most late-stage prof review
  — the notes' "collaboration" framing is superseded 30 Jul.
- `cross_cutting/` — `methodology/` =
  `Research_Methodology_Directions_Claims_20260801.tex` (standing note:
  direction / research-question / claim taxonomy, six-level
  claim-granularity ladder, lit-check placement rule — shallow check to
  start, exhaustive audit at claim-freeze; governs new-direction
  selection for Papers 4/5+) + the critical-review prompt. `ideation/`
  = `Research_Other_Ideas_20260701.tex` (field scans / future
  directions), `Research_Branch_Ideas_Triage_20260717.tex` (branch-idea
  scoring; band definitions for the 40–55% claim strength), the 13-Jul
  innovation audit, and the five `Idea_*.tex` side-project briefs.
  `reviews/` = `Research_Editorial_Review_v3/v4_20260711.tex`.
- `meetings/` — raw discussion notes + `Meeting_Response_ProfDiscussion_20260729.md`.
- `slides/` — all decks, chronological: dated `research_update_*` /
  `group_meeting_*` plus the per-paper `personal_research_paper_N_*`.
- `archive/` — superseded docs: the brainstorming series, plan v1/v2 +
  criticism, runbook v1/v2, `Research_Plan_v3_20260701.tex`,
  `Addendum_Post5a_Enhancements_20260707.tex`,
  `Runbook_Addendum_RepConvergence_20260703.tex`, and the reorg plan.

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
- `temp_files/ops/` **[gitignored]** — cluster/instance operations
  notes (grouped 7 Aug 2026; `temp_files/` root is scratch):
  `instance_initiation_with_template.md` (cloud-instance setup, env
  creation, repo/buffer sync, verification), `instance_sample_commands.md`
  (launch / monitor / snapshot-refresh / result-sync patterns),
  `instance_helper_design_notes.md` (per-GPU-lane queue design),
  `template_onstart_script.sh` (Vast onstart template), `SU_usage.txt`
  (RCC service-unit burn + per-GPU task runtimes), `ssh-keys.txt`.
  Submission gotchas (`--export=ALL` module leakage; GPU jobs not
  job-to-job deterministic on Midway3) live in the same memory.
