# Study Ledger

One entry per experiment/read: registration → record → verdict →
decision. This is the INDEX; the authoritative documents are the
per-experiment `prereg/` files (immutable after their freeze commit),
the `artifacts/<wave>/` records (RESULTS.md + machine json), and
`analysis/DEVIATIONS.md` (chronological deviations + code
notes/registrations). Plans live in `research_notes/` (gitignored);
the live status board is `research_notes/Roadmap_20260718.md`. See
`PROJECT_INDEX.md` for the file map.

Maintenance rule: append/update the relevant row in the same session
that writes an artifact record. Flags: ⚠ = later qualified;
✖ = instrument-invalidated (see the 2026-07-24 DEVIATIONS entry).

## Paper 1 — causal WM-transfer (ICLR 2027, branch causal-wm-transfer)

| Date | Experiment | Registration | Record | Verdict → decision |
|---|---|---|---|---|
| 07-02 | Gate 0 (pilot probe sets, cup/finger/reacher) | note App. A | `artifacts/gate0_20260702/GATE0_DECISION.md` | pilots pass → probe-set protocol frozen |
| 07-07 | Phase 5a (reward-free WM transfer, unblinded) | `analysis/PREREG_phase5a.md` | `artifacts/phase5a_20260707/` | cup reward-free transfer large → Axis-1 program launched |
| 07-10 | Phase 6 / Axis-1 controlled buffers | plan v3/v4 | `artifacts/phase6_axis1_20260710/` (+ `axis1_2026*` AUC snapshots) | occupancy-controlled q1 pairs established |
| 07-11 | E2 Gate F; E3 precheck; expl smoke | per-file | `e2_gatef_*/DECISION.md`, `e3_precheck_*/PRECHECK.md` | gates passed → E3 wave authorized |
| 07-11/12 | E1 reacher replication (driver-level) | corrective prereg + amds | `artifacts/e1_reacher_final_20260712/` | registered driver-level read done |
| 07-13 | P0 corrective refit (reward-free finger Q1) | `PREREG_axis1_corrective_*` | `artifacts/p0_axis1_corrective_20260713/` | **NULL → pivot fired** (plan v4, Act-1-only) |
| 07-13 | W0 fully-paired n=16 interaction | corrective Amendment 2 §A | `artifacts/w0_n16_interaction_20260713/` | **CONFIRMED +98.7\*, perm p=.0027** |
| 07-14 | E4 stratified error + Goodhart sprint | `PREREG_e4_stratified_error_20260714`, `PREREG_goodhart_sprint_20260714` | `artifacts/e4_goodhart_20260714/` | mechanism = reward-predictability of transferred features; Goodhart continue-but-weaker |
| 07-16 | W1 (E3v2 within-collector) + W2 + W3 | registered reads | `artifacts/w123_e3v2_20260716/` | **W1 REPLICATES +84.0, 14/14, p=.00012, de-confounded**; apt −5.1 = 3rd reward-free null; W2 cup +93 directional-only; W3 Q2 null → scaling pilot authorized |
| 07-17 | P3 gradient-path factorial | `PREREG_p3_gradient_path_20260714` | `artifacts/p3_wave_20260717/` | primary rgo−sgb +50 NOT confirmed at n=8; **shuffle collapse FIRES** (binding necessary); rgo +65 own-CI>0; vgo dead; E4 rew-NLL LEVEL separates arms |
| 07-17 | vgo wiring audit | — (audit) | `artifacts/vgo_wiring_audit_20260717/AUDIT.md` | case (b) replay-grounded → P-B2 shifts to attenuation form |
| 07-17 | TD-MPC2 cross-family n=8 + Goodhart ensemble + D1 pilot | `PREREG_tdmpc2_crossfamily_20260716` | `artifacts/tdmpc2_goodhart_d1pilot_20260717/` | [B_aware−B_free] +9.5 ns; ensemble does not fix top-1 regret (demote-leaning); D1 pilots 6/6 |
| 07-17 | **P3 Amendment 1, pooled n=16** | `PREREG_p3_amendment1_20260717` | `artifacts/p3_amend1_20260717/` | **PRIMARY FIRES +51.5 [+9.0,+94.6]; rgo +56.7\*; sgb +5.2 tight null → reward-gradient path CONFIRMED sufficient carrier** |
| 07-17 | Theory predictions frozen | `PREREG_theory_predictions_20260717` | (spectral model, `research_notes/Theory_SpectralTransfer_20260717.tex`) | P-A1 landed (Amend-1); P-B1 stands (stamping); P-B2 pending vgo-x; P-B4 strong REFUTED (Opt-C corrective); P-B5 pending scaling; P-C2 premise-failed 07-24 (pixel X2 null — monotone form fails, swamping regime survives pending `PREREG_pixel_swamping_20260724`) |
| 07-18 | Stamping | `PREREG_stamping_20260717` | `artifacts/stamping_20260718/` | primary does NOT fire → **P-B1 strong form STANDS** |
| 07-18 | Scaling Option C (volume rider) | `PREREG_scaling_pilot_20260717` | `artifacts/scaling_optc_20260718/` | ⚠ realized design ≠ registered (side inversion; DEVIATIONS 07-18); P-B4 not adjudicated; behavioral anomaly recorded |
| 07-19 | Option C corrective | `PREREG_scaling_optc_amend1_20260718` | `artifacts/optc_corrective_20260719/` | **C2 FIRES: P-B4 strong form REFUTED** |
| 07-18 | Synth Phase B | `PREREG_synth_phaseb_20260717` | `artifacts/synth_phaseb_20260718/` | collapse-only (interaction null; shuffle −226.6\*); Phase C not authorized |
| 07-18/19 | Synth diagnosis + re-diagnosis | `PREREG_synth_diagnosis_20260718`, `PREREG_synth_rediag_20260718` | `artifacts/synth_diagnosis_20260718/`, `synth_rediag_20260719/` | in-subspace refuted at model level; adapt lottery confirmed → B″ authorized |
| 07-23 | Synth Phase B″ | `PREREG_synth_phasebpp_20260720` | `artifacts/synth_phasebpp_20260723/` | NULL +30.5 [−18.7,+100.2] → **synth interaction leg CLOSED** (shuffle-collapse-only exhibit) |
| 07-19 | TD-MPC2 F1 (n=16 pooled) | `PREREG_tdmpc2_amend1_20260718` | `artifacts/tdmpc2_f1_20260719/` | null at doubled power; P-C1 PARTIAL; **family-scope limitation registered** (headline restricted to reconstruction-based WMs); G-F2 GO |
| 07-22 | TD-MPC2 F2 (E4 mechanism port) | `PREREG_tdmpc2_f2_20260720` | `artifacts/tdmpc2_f2_20260722/` | **P-F2a FIRES +1.717, lottery 0 → never-included-like**; G-F3 NO-GO; TD-MPC2 track complete |
| 07-23/24 | E4 passes: Amend-1 fits + Opt-C volume + sgb gap | E4 prereg (descriptive) | `artifacts/e4_amend1_optionc_20260723/` (+§A′) | rgo mechanism REPLICATES (1.07 vs 1.21); volume buys stability not level; **sgb gap closed 07-24: separation replicates at n=16 both arms (2.03 vs 1.07 hi side)** |
| 07-19 | Pixel X1 pair build | `PREREG_pixel_repl_20260719` | `artifacts/pixel_x1_pair_20260719/PAIR.md` | pair built → X2 submitted |
| 07-24 | **Pixel X2 read** | `PREREG_pixel_repl_20260719` | `artifacts/pixel_x2_20260724/` | **interaction does NOT fire** (+3.6 [−29.9,+40.9], p=.87; both simples null; all four cells on a common AUC floor; audit 32/32 clean) → **G-X3 NO-GO**; P-C2 premise-failed (monotone form fails; swamping regime = surviving branch); pixel leg = registered scope boundary; swamping diagnostic REGISTERED same day |
| 07-25 | **Orthogonal-objective test** (full grid, seeds 1–16) | `PREREG_orthogonal_obj_20260723` | `artifacts/orthogonal_obj_20260725/` | validity gate passed (45.3 ≥ 20); **primary rgo−sgb +1.0 [−16.1,+17.0] does NOT fire ⇒ objective-specific support** (same fits: +45.0\* on the ORIGINAL objective; ratio .023); band leg LANDED; strict headline reading stands (scope does not widen); local read ≡ cluster json |
| 07-25 | **Pixel swamping diagnostic** | `PREREG_pixel_swamping_20260724` + Amend 1 (07-25 E4 repair after `KeyError: 'image'` crash — measure keyed off model enc/dec spaces, proprio numbers bit-identical, read frozen pre-outcome) | `artifacts/pixel_swamping_20260725/` | **P-SW1 SWAMPING-CONSISTENT FIRES**: task-arm in-regime rew-NLL 23.3 [19.4, 27.3] nats ≫ 2.0 bar (floor 0.67 informative; side-robust — all 16 runs ≥ 2.0; s0 24–58 / s1 4.0–7.3); reward info ABSENT from pixel trunk ⇒ **pixel boundary = predicted structure (g > G2 regime), mechanism-complete**; membership→transfer link unchallenged; G-X3 stays NO-GO, no further pixel arms |
| 07-25 | **Spectral domains read** | `PREREG_spectral_domains_20260724` + Amend 1 | `artifacts/spectral_domains_20260725/` | **P-SM1 NOT confirmed** (hi side REVERSED: rank cup 492 vs finger 455; lo holds; all 4 buffers own-θ, no fallback engaged) ⇒ spectral DOMAIN account refuted as registered; cup/finger = unexplained scope condition; theory E5 cup section loses measurement support; arm-level results untouched per frozen map |

Standing Paper-1 decisions: headline = "support must be legible to the
objective" (17 Jul editorial); claim band 40–55% via n=16 ✓ + TD-MPC2 ✗
(F1 retests) + unfrozen calibration (registered) + orthogonal test ✓
(**objective-specific**, 07-25 — strengthens the strict reading);
**Goodhart DEMOTED** (18 Jul, re-confirmed
22 Jul — negative/boundary exhibit only); 9a un-embargoed (Amend-1).

## Paper 2 — EVPI / value-of-computation (off ICLR path)

**⚠ 2026-07-24 INSTRUMENT INVALIDATION**: the shared D1 real-op labeler
carried two defects (stale-carry candidate branches; policy-RNG not
common across branches) + audit found a third (xpol trajectory
prevact). ALL D1 decision-value verdicts below marked ✖ are withdrawn
as evidence (faithful reads of an unintended estimand); frozen
RESOURCE decisions stand. DEVIATIONS 07-24 entries = tracked record.

| Date | Experiment | Registration | Record | Verdict → decision |
|---|---|---|---|---|
| 07-10 | Latent-UQ Stage 0 | Gate-D0 App. A | `artifacts/latent_uq_stage0_20260710_221232/DECISION.md` | attractor bias found (disagreement → density-tracking at convergence); phase change + cross-policy boundary |
| 07-14 | Stage 1A (density-residual instrument) | `PREREG_gate_d1_stage1a_20260714` | `artifacts/latent_uq_stage1a_20260714/` | registered battery PASSES (10/10 flip + 4 cross cells) → renamed **density-deconfounded disagreement residual** (diagnostic, not per-state gate; audit-verified leakage-free) — STANDS |
| 07-18 | Gate D1 (Stage-1B labels, 24+24 unfreeze) | Stage-1A prereg criteria | `artifacts/gate_d1_20260718/` | ✖ FAILS 0/4 → 24+24 frozen (resource decision stands; verdict uninterpretable) |
| 07-23 | Route-A R2 decision probe | `PREREG_gate_d1_r2probe_20260722` + Amend 1 (imag demoted pre-read) | `artifacts/gate_d1_r2_20260723/` | ✖ NULL 16/16 → **Route A CLOSED** (resource decision stands); imag-op defect confirmed independently (retired — stands) |
| 07-23 | Actionability ladder | `PREREG_d1_ladder_20260723` | `artifacts/d1_ladder_20260723/` | ✖ no compression signature → ensemble arm stays gated (off-by-default stands) |
| 07-24 | Shift-consequence probe | `PREREG_d1_shift_20260723` + Amend 1 | `artifacts/d1_shift_20260724/` | ✖ neither arm fires (xpol also carries the prevact defect) |
| 07-24 | Instrument correction + relabel | `PREREG_d1_relabel_20260724` (+ audit additions) | → 07-25 read | labeler `d1fix_20260724` built + registered; 18 relabel passes on existing shift pilots |
| 07-24 | Relabel Amendment 1 — d1pilot cohort | `PREREG_d1_relabel_amend1_20260724` | → 07-25 read | Gate-D1 pilots confirmed alive (same 1e5 maturity) ⇒ +18 passes; replication family + pooled-12 headline; frozen d1s primary UNCHANGED |
| 07-24 | Relabel Amendment 2 — schema-drift loader fix | `PREREG_d1_relabel_amend2_20260724` | → 07-25 read | amend-1 smoke crashed (`model_obs` absent from 18-Jul configs); `load_config` backfills absent keys from defaults; backfill logs on the real campaign show ONLY registered keys (7-key subset early, 8 later — checkout drift, all inert) |
| 07-25 | **R3 consumer-competence factorial REGISTERED** | `PREREG_r3_competence_20260725` | — pending runs | {cup,finger}×{e1,e4}×seeds 31–38, early(2.5e4)+late(1e5) maturities, base arm, oracle_all; P-R3a opp>0.2 / P-R3b gap>0 / P-R3c maturity-buys-competence; reacher conditional-GO (all-or-nothing cohort); power pre-sized (n=32 detects gap 0.42 = half observed); driver `scripts/r3_local.sh` (dosed-smoke GATE before labels), read `analysis/r3_read.py` selfcheck PASS |
| 07-25 | **Corrected-instrument relabel READ** (both cohorts, ONE execution) | relabel prereg + Amend 1 + Amend 2 | `artifacts/d1_relabel_20260725/` | **neither arm fires in ANY family** (d1s: xpol +0.24 [−0.33,+0.77]; phys −0.19 [−0.34,−0.06] — negative, no fire under one-sided rule, sign flips in replication; pooled-12 both null) ⇒ **corrected negative at 12 clusters — the consequence leg fails ON THE CORRECTED INSTRUMENT**; attenuation objection discharged up to power (t-intervals disclosed); defect-impact: old labels inflated base/phys (+0.20/+0.15 → −0.01/−0.20); **opportunity/achieved: opportunity large everywhere (xpol finger +6.3), achieved ≈ 0 ⇒ implementation gap = R3's target, parameterized**; next = R3 registration |

Paper-2 plan: `research_notes/other research/EVPI_Plan_Revision_20260724.md`
(v4). Constants (early −0.208\*/late×e4 +0.313\*) = hypotheses to
re-test, not results. 24+24 permanently frozen; imag op retired.

## Registered / running, no read yet

- **Scaling Option B** (running): `PREREG_scaling_pilot_20260717` → `analysis/scaling_read.py`; 12m E4 pass waits on it (P-B5).
- **Unfrozen calibration** (to submit, 32 jobs): `PREREG_unfrozen_calib_20260723` → `analysis/unfrozen_calib_read.py`.
- **vgo extended discriminator** (to submit, 16 fresh 1.5M fits + adapt): `PREREG_vgo_extended_20260723` → `analysis/vgo_extended_read.py`; adjudicates P-B2 attenuation form.
- **Scaling-B completeness blocker**: latest exclusions snapshot (07-25 orthogonal bundle) lists 4 `ax1(f)s12*` adapt runs missing scores.jsonl (`fs12q1s0_seed8`, `fs12q1s1_seed7`, `s12q1s1_seed2`, `s12q1s1_seed7` — set CHANGED from the X2 snapshot; seed6 completed since). Re-run or explain before the Scaling B read.
- **Comp×capacity theory predictions** (registered 07-24, adjudicated by Scaling B + 12m E4 + volume replication): `PREREG_compcapacity_theory_20260724` (P-E1 ordered un-nulling / forbidden pattern; P-E2 task-lo-first; P-E3 metric split; P-E4 diversity-as-λ); derivations in `research_notes/Theory_CompCapacity_Addendum_20260724.tex`; MUST be committed before the Scaling B read.
- **Paper 4 collusion pilot — stages 1–3 COMPLETE** (07-25): stage 1 GO (`artifacts/collusion_stage1_20260725/`: Δ_profit 0.831 in published range, fingerprint 68/100; failure-to-learn minority ⇒ fingerprint-conditional measurement). **Stages 2+3 same day (`artifacts/collusion_stage23_20260725/`, local CPU, harness v2 + designb, selfchecks PASS, v1 columns validated vs cluster csv): regime label = DISTANCE form recommended (undercut form saturates 26% + inverts vs fingerprint); RQ1 feasibility RESOLVED — `coverage_late` varies 0.22–0.95 and is orthogonal to occupancy (corr −0.12), matched pairs constructible, no new knob needed (β stays 4e-6; β sweep matches Calvano statics); Design-B VIABLE — offline stream replay float-exact 20/20, punishment-drop collapses collusion (Δ 0.825→0.508, fp 0.70→0.20) BUT size-matched random-drop shows most is data volume (→0.615/0.40; composition increment −0.107, 10/20) ⇒ confirmatory prereg must register graded volume-controlled interventions.** Next: confirmatory prereg (reference-choice + intervention ladder from these logs).

## Cross-cutting audits (2026-07-24, DEVIATIONS entries)

Labeling stack: 3 defects fixed pre-freeze (xpol prevact, distractor OU
snapshot, FromDM._done). Training side: clean (arm sg wiring exact;
reward hard-excluded from the model; valens/disag measurement-only;
caveat: con-head grads reach trunk — constant target in DMC,
arm-invariant, do not change mid-study). Upstream surfaces: clean
(offline_fit replay read-only by construction; replay_context
foreign-latent transient is arm-invariant; return math canonical; AUC =
online wrapper-emitted reward; E4 err_diff = out−in; Stage-1A residual
out-of-fold). No known open defects on the decision path.
