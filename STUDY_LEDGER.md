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
| 07-17 | Theory predictions frozen | `PREREG_theory_predictions_20260717` | (spectral model, `research_notes/Theory_SpectralTransfer_20260717.tex`) | P-A1 landed (Amend-1); P-B1 stands (stamping); P-B2 pending vgo-x; P-B2 attenuation NOT supported 07-26 (vgo absence-like at 3×; weakest reading only); P-B4 strong REFUTED (Opt-C corrective); P-B5 sign-consistent underpowered (07-26 scaling S1d; E4 leg supports representation half); P-B6-lite CONFIRMED 07-26; P-C2 premise-failed 07-24 → swamping regime CONFIRMED 07-25 (P-SW1) |
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
| 07-26 | **vgo extended-fit discriminator** | `PREREG_vgo_extended_20260723` | `artifacts/vgo_extended_20260726/` | **does NOT fire** (diff-in-diff −2.2 [−35.2,+40.2], p=.93; vgo gain +5.8 ns ≈ sgb generic +8.0 ns) ⇒ **P-B2 attenuation form NOT supported — vgo absence-like at 3×**; value gradients are NOT a carrier even with 3× time (sharpens the reward-gradient sufficiency claim); no further extended-fit arms without new theory; **PAPER-1 EXPERIMENTAL PROGRAM COMPLETE** |
| 07-26 | **Scaling Option B + 12m E4 pass** | `PREREG_scaling_pilot_20260717` (+ compcapacity predictions committed pre-read) | `artifacts/scaling_optionb_20260726/` | **PRIMARY three-way includes 0** (−30.8 [−154.5,+83.4]) ⇒ **capacity does not substitute at 12× — interaction SCALE-ROBUST** (12m interaction fires +126.5\* 7/8 seeds; robust variant vs honest W0 baseline +27.8 ns, same branch); **P-B6-lite CONFIRMED** (apt@12m +12.7 ns — 4th reward-free null, now at 12×); P-B5 sign-consistent underpowered; **P-E1 HOLDS** (no forbidden pattern; 12m = regime A ⇒ 25m targets A→B/task-lo rise); P-E2/P-E3 premise-idle; E4@12m: task hi rew-NLL 1.30 IN the 1m membership band, lo 5.8 out; d_errin ≈ 0 (arm-invariance again); apt rew-NLL absent-by-design (gate) |
| 07-25 | **Unfrozen-adaptation calibration** | `PREREG_unfrozen_calib_20260723` | `artifacts/unfrozen_calib_20260725/` | **FIRES, AMPLIFIED ×4.3**: unfrozen rgo−sgb +267.2 [+206.7,+331.4] (p=.0078 = n=8 min, every seed positive, d_z=2.77, both sides fire) vs frozen same-seed +62.5; level changes rgo +235.2\* / sgb +30.4\* ⇒ reward-legible support = better INITIALIZATION, frozen-readout was the CONSERVATIVE protocol; **band leg lands in the strong form — all four band conditions now adjudicated** |
| 07-25 | **Spectral domains read** | `PREREG_spectral_domains_20260724` + Amend 1 | `artifacts/spectral_domains_20260725/` | **P-SM1 NOT confirmed** (hi side REVERSED: rank cup 492 vs finger 455; lo holds; all 4 buffers own-θ, no fallback engaged) ⇒ spectral DOMAIN account refuted as registered; cup/finger = unexplained scope condition; theory E5 cup section loses measurement support; arm-level results untouched per frozen map |
| 07-30 | **U2 stamped unfrozen** (48 jobs) | `PREREG_unfrozen_u2_20260727` | `artifacts/unfrozen_stress_u234_20260730/` | **PROTOCOL-ROBUST**: P-U2 [B_srd−B_sid] −1.3 [−25.5,+21.0], no flip — the installed stamp is inert even under full fine-tuning ⇒ **"inclusion ≠ usefulness" = TWO-PROTOCOL fact**; UNREAL/RaMP refutation strengthens; P-B1 standing improves; descriptor: sid controls gained most from unfreezing (+32.9*/+26.8*), stamped cells less; config audit 48/48 |
| 07-30 | **U3 orthogonal unfrozen** (32 jobs) | `PREREG_unfrozen_u3_20260727` | `artifacts/unfrozen_stress_u234_20260730/` | **PROTOCOL-ROBUST**: P-U3 rgo−sgb on spin +5.3 [−13.6,+23.6], pooled level 67.9 ≥ 20 gate — no flip ⇒ **objective-specificity = INITIALIZATION fact** (two-protocol standing); strict headline strengthens; rgo0 level +36.4* from unfreezing but not differential by arm; config audit 32/32 |
| 07-30 | **U4 pixel unfrozen 2×2** (32 jobs) | `PREREG_unfrozen_u4_20260727` | `artifacts/unfrozen_stress_u234_20260730/` | **FLOOR-CENSORED** (registered guard): P-U4a relief +1.7 [−10.6,+13.4] does not fire — pixel cells stay at the X2 floor (76–88 vs frozen 77–83) even fine-tuned, unlike proprio (+235/+30) ⇒ interaction unadjudicable; frozen pixel floor binds; no further pixel-unfrozen compute pre-deadline; G-X3 NO-GO unchanged; P-SW1 untouched; config audit 32/32 |

Standing Paper-1 decisions: headline = "support must be legible to the
objective" (17 Jul editorial); claim band 40–55% — **all four conditions
adjudicated 07-25**: n=16 ✓ + TD-MPC2 ✗ (explained boundary) +
unfrozen calibration ✓ (**AMPLIFIED ×4.3** — frozen-readout was the
conservative protocol) + orthogonal test ✓ (**objective-specific**);
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
| 07-25 | **R3 consumer-competence factorial REGISTERED** | `PREREG_r3_competence_20260725` | → 07-29 read | {cup,finger}×{e1,e4}×seeds 31–38, early(2.5e4)+late(1e5) maturities, base arm, oracle_all; P-R3a opp>0.2 / P-R3b gap>0 / P-R3c maturity-buys-competence; reacher conditional-GO (all-or-nothing cohort); power pre-sized (n=32 detects gap 0.42 = half observed); driver `scripts/r3_local.sh` (dosed-smoke GATE before labels), read `analysis/r3_read.py` selfcheck PASS |
| 07-25 | **Corrected-instrument relabel READ** (both cohorts, ONE execution) | relabel prereg + Amend 1 + Amend 2 | `artifacts/d1_relabel_20260725/` | **neither arm fires in ANY family** (d1s: xpol +0.24 [−0.33,+0.77]; phys −0.19 [−0.34,−0.06] — negative, no fire under one-sided rule, sign flips in replication; pooled-12 both null) ⇒ **corrected negative at 12 clusters — the consequence leg fails ON THE CORRECTED INSTRUMENT**; attenuation objection discharged up to power (t-intervals disclosed); defect-impact: old labels inflated base/phys (+0.20/+0.15 → −0.01/−0.20); **opportunity/achieved: opportunity large everywhere (xpol finger +6.3), achieved ≈ 0 ⇒ implementation gap = R3's target, parameterized**; next = R3 registration |
| 07-29 | R3 Amendment 1 — reader seed-grid repair | `PREREG_r3_amend1_20260729` | commit bbe86983 (pre-read) | frozen reader had `EXPECT_SEEDS`=1–8 vs registered 31–38 (crash before any value touched; selfcheck structurally blind — fixture built FROM the constant); value-blind repair + 4 integrity guards (unregistered-seed, finiteness = oracle_all signature, m_real bounds, n_states=200), decision rules verbatim-unchanged; 3-agent value-blind bundle audit clean (64/64 labels exact dials/doses, smoke gate 4/4 dosed, 32/32 two-phase continuation verified) |
| 07-29 | **R3 consumer-competence READ** (ONE execution, RCC) | `PREREG_r3_competence_20260725` + Amend 1 | `artifacts/r3_competence_20260729/` | **P-R3a FIRES** (opp +1.247 [+0.700,+1.908] > 0.2) **+ P-R3b FIRES** (gap +1.293 [+0.723,+1.987] ≈ 3× the pre-sized detection bar); **P-R3c null** (+0.014 [−0.177,+0.215]) ⇒ **CONSUMER-COMPETENCE FAILURE registered: opportunity exists, is not harvested, and maturity does not close the gap** — 25-Jul shift negative = competence ceiling, NOT information ceiling; Paper-2 headline = opportunity/competence decomposition; domain-robust (cup gap +0.917\*, finger +1.669\*; finger achieved −0.157 CI<0 = harvest slightly counterproductive); opportunity survives dose (e4 +0.98\*) and maturity (late +1.30\*); floors 25%/19% not limiting; local verification re-execution bit-identical; 24+24 frozen / Route A closed / imag retired reaffirmed |
| 07-31 | **R3 reacher third-domain READ** (ONE execution, local; manifest verify OK 14,554 files == committed pin; code snapshots ≡ frozen; selfcheck PASS pre-read) | `PREREG_r3_reacher_20260729` (freeze 596a7f93) | `artifacts/r3_reacher_20260731/` | **REPLICATES — P-RRa FIRES** (opp +0.828 [+0.565,+1.114] > 0.2) **+ P-RRb FIRES** (gap +0.813 [+0.554,+1.097]); **P-RRc null** (+0.047 [−0.023,+0.121]) ⇒ **the opportunity/competence decomposition is a THREE-DOMAIN fact (cup, finger, reacher); the conference-band task axis LANDS**; maturity-does-not-buy-competence replicates per domain; opportunity survives dose (e1 +0.99\*, e4 +0.66\*) and maturity (early +0.68\*, late +0.97\*); pooled achieved +0.015 [−0.038,+0.071] tight null; **floors 1/32 (3.1%) — the registered sparse-floor risk did NOT materialize** (gate bar 50%); 24+24 / Route A / imag invariants reaffirmed |
| 07-31 | **R3 candidate-aware ladder READ** (ONE execution, local CPU 21m30s; parent freeze 6271c770 + pre-read Amendment 1 commit 5a16c728 triggered by the independent review; reader ≡ HEAD at execution; in-read digest gate matched frozen 02efe6c5…) | `PREREG_r3_ladder_20260730` + `PREREG_r3_ladder_amend1_20260731` | `artifacts/r3_ladder_20260730/` | **NO PER-STATE SIGNATURE (registered flat branch): the implementation gap is NOT state-legible at this information level — policy-level constant, not a per-state opportunity map**; no compression signature (L2 ≥ L3/L4 in the e1 cells — scalars already carry the little legibility there is) and no candidate signature ([L4−L3] ≈ 0 everywhere, max +0.040 with CI lower bound +0.000); weak sub-bar legibility real (L2 ≈ +0.16\* both e1 cells; early_e4 L4 +0.163\*) but all rungs < 0.2; late_e4 flat at every rung ⇒ **competence-repair proceeds with its registered DEFAULT targeting**; amendment VALIDATED on real data (pooled diagnostics non-evidential: early_e1 L1 sign-flip +0.100\* within vs −0.130 pooled; late_e4 L3 −0.014 within vs −0.248 pooled — the confound was live); nan runs (all-floor) 5/10/1/7 ⇒ live clusters 11/6/15/9, floors concentrate in e4+early; 48 registered looks + 40 pooled diagnostic points; json sha 3cff90dc… |

Paper-2 plan: `research_notes/other research/EVPI_Plan_Revision_20260724.md`
(v4). Constants (early −0.208\*/late×e4 +0.313\*) = hypotheses to
re-test, not results. 24+24 permanently frozen; imag op retired.

## Paper 3 — composition×capacity flagship (own venue clock)

| Date | Experiment | Registration | Record | Verdict → decision |
|---|---|---|---|---|
| 07-29 | **Volume-anomaly replication P-E4a + diversity secondary** (ONE read, cluster-side) | `PREREG_volume_repl_20260726` | `artifacts/volume_repl_20260729/` | **PRIMARY FIRES: fresh v400s1−v200s1 +331.0 [+169.3,+502.7]** (> the original +228.6 — no winner's curse; pooled-12 +279.8\*; s0 fresh +194.8\* same direction) ⇒ **anomaly REAL — support breadth enters the theory as a λ-structure input distinct from occupancy**; diversity direction HOLDS (diversity_pr v400s1 10.63 > v200s1 9.23 under the pinned `spectral_v1_1_20260724` gate, while generic spectrum_pr goes the other way and v400's occupancy is 3.4× lower) ⇒ P-E4b diversity leg engages; non-spectral-account branch not triggered; conditional-on-occupancy regression form stays unregistered; local re-execution bit-identical |

## Paper 4 — algorithmic collusion (MARL/econ venue, own clock, Paper-4 chat)

| Date | Experiment | Registration | Record | Verdict → decision |
|---|---|---|---|---|
| 07-25 | Pilot stages 1–3 (baseline reproduction, measurement calibration, Design-B viability) | design memo `Design_Collusion_Pilot_20260724.md` | `artifacts/collusion_stage1_20260725/`, `collusion_stage23_20260725/` | stage 1 GO (Δ 0.831 in published range, fp 68/100; failure-to-learn minority ⇒ fp-conditional measurement); distance label recommended (undercut saturates + inverts); RQ1 resolved via `coverage_late`; Design B VIABLE (float-exact 20/20) but deletion confounds volume (composition increment −0.107) ⇒ graded volume-controlled interventions required |
| 07-30 | Confirmatory-prereg calibration | (calibration, pilot seeds only) | `artifacts/collusion_calib_20260730/` | terminal reference frozen (exo degenerate: mean .981, 96%>0.9); ladder d.25 inert / d1 fp-cond −0.134; **Design-A smoke: DISSOCIATION found (forbid λ=1 ⇒ Δ +0.987 w/ fingerprint 0/10, degenerate attractor) ⇒ P-CLA primaries re-specified pre-freeze**; 2 adversarial review passes (10+11 findings, 2 blocking reader-corruption defects) all fixed |
| 07-30 | **Confirmatory sweep READ** (ONE execution; freeze `42bbe487`; manifest 16/16, snapshots ≡ freeze) | `PREREG_collusion_confirm_20260730` | `artifacts/collusion_confirm_20260730/` | **ALL FOUR PRIMARIES FIRE — P-CLB1 −0.104 [−0.165,−0.046] + P-CLB2 −0.592 [−0.724,−0.447]: punishment-phase experience is a composition-specific driver of BOTH profit level and punishment structure at matched update mass (dose ladder monotone, mid-dose fires); P-CLA1 −0.650* + P-CLA2 +0.135*: forbidding exploratory undercuts collapses the fingerprint (0.65→0.00) while Δ RISES (0.852→0.987) ⇒ experimentally induced Abada–Lambin failure-to-learn regime — punishment-generating exploration necessary for GENUINE collusion, not for supra-competitive prices. All gates pass (exact 100/100, n_fp=76, manip −0.485*). ⇒ RQ2 composition leg LANDS; RQ3 coupling wave AUTHORIZED; two-outcome measurement argument load-bearing** |

## Registered / running, no read yet

- **R3 reacher third-domain replication** (REGISTERED 07-29; **READ
  07-31: REPLICATES — see the Paper-2 table;
  `artifacts/r3_reacher_20260731/`**):
  `PREREG_r3_reacher_20260729` → `analysis/r3_reacher_read.py`
  (selfcheck PASS; imports the frozen r3_read machinery, import surface
  pinned). 16 fresh two-phase runs + 32 oracle_all passes on the idle
  instance (`DOMS=reacher` pilots → `SMOKE_DOM=reacher` smoke →
  labels); machine-checked reacher smoke gate at driver AND reader;
  registered FLOOR gate (>50% all-zero-G ⇒ inconclusive, not failure);
  P-RRa opp>0.2 / P-RRb gap>0 / P-RRc maturity; verdicts REPLICATES /
  NO-OPPORTUNITY / HARVESTED / FLOOR-LIMITED; n=16 detects gap ≈0.59
  vs observed cup/finger +1.29. Supersedes the executed R3 prereg's
  conditional-GO clause; cup/finger primaries never recomputed.
- **R3 candidate-count doubling test M=8→16** (REGISTERED 07-29, to
  submit after freeze-commit; closes editorial objection-6 residue):
  `PREREG_r3_doubling_20260729` → `analysis/r3_doubling_read.py`
  (selfcheck PASS incl. _load fixture leg + dial-identity guards from
  meta on both sides). 1 smoke + 32 late-cell M=16 passes on the
  existing R3 checkpoints (`--actions 16`, all other dials pinned);
  CELL-level pairing only (candidate sampler shares the agent RNG ⇒
  trajectories diverge — disclosed by design); P-DBLa monotone
  consistency (CI entirely <0 ⇒ ANOMALY, quarantine) / P-DBLb
  materiality vs 0.6465 (= 50% of the committed gap 1.29304688,
  truncated conservatively); T4 signed-bias theory anchor verified
  (Prop biassign). **08-01 READ (ONE execution, local): CONSISTENT +
  MATERIAL TRUNCATION DISCLOSED** — inc_opp (opp16−opp8) +0.634
  [−0.210, +1.439] n=32 (no anomaly; CI upper > 0.6465 ⇒ the M=8
  truncation can exceed half the gap — quantified in the paper;
  conservative direction unaffected). Decisive secondaries: **gap16
  REFIRES bigger** (+1.893 [+1.141, +2.715] vs committed M=8 gap
  1.293) while **inc_ach ≈ 0** (+0.081 [−0.117, +0.255]) — doubling
  the candidate set grows opportunity but the consumer harvests none
  of it ⇒ the headline gap GROWS under candidate-set refinement
  (mechanism-coherent with opportunity-without-competence); per-domain
  inc: cup +0.25 ns, finger +1.02 ns wide. Bundle
  `local_results/r3_doubling_20260801_212204/` (32+smoke m16 labels);
  → `artifacts/r3_doubling_20260801/r3_doubling.json`.
- **R3 candidate-aware ladder REGISTERED 07-30/31** (Paper-2
  objection-4 residue; registered-DESCRIPTIVE, never decisional):
  `PREREG_r3_ladder_20260730` → `analysis/r3_ladder_read.py` (selfcheck
  PASS; sha256 ea4271d5…). L0–L4 on the committed R3 labels (substrate
  byte-gated by frozen `COMMITTED_LABELS_DIGEST` 02efe6c5…); NEW L4 =
  frozen 7-feature candidate-geometry/Q-dispersion set (old ladder's
  candidate-geometry exclusion LIFTED as an explicit registered scope
  change); nested-LORO ridge, 16 clusters/cell, 48 looks counted;
  heuristics: compression signature (carried) + candidate signature
  (new) — consequence resource-only (repair targeting). ONE local CPU
  read after freeze-commit. Solo pre-freeze review (reviewer wave lost
  to session limits — see DEVIATIONS): exact version pin + digest gate
  added. **AMENDED 07-31 pre-read** (`PREREG_r3_ladder_amend1_20260731`
  → reader sha aa776f1c…, selfcheck PASS 7m08s): the deferred
  independent review (2 reviewers + refuters, 12 agents, value-blind)
  CONFIRMED the pooled rung metric confounds between-run/domain
  structure with per-state legibility (LORO mean-reversal −0.24…−0.65
  on null w/ CIs<0; domain shortcut +0.30 on zero signal) ⇒ metric =
  mean per-run held-out Spearman (pooled demoted to point-only
  diagnostics; explicit all-floor nan policy, ~3–4 nan runs/cell
  expected); new machine-checked heterogeneity + p=512 legs; +4
  registered disclosures (act_greedy_dist ≡ 0 by construction — L4 has
  6 live G features; Power-section erratum; lambda_per_fold
  diagnostics; nanpercentile policy). Read still unexecuted at
  amendment (value-blindness intact); executed after the amendment
  commit — **READ 07-31 late: NO PER-STATE SIGNATURE (flat branch);
  repair default targeting stands; amendment validated on real data —
  see the Paper-2 table row.**
- **Stage-0 second-k robustness REGISTERED 07-30/31** (the Stage-0
  memos' own unexecuted caveat; gates externalizing TRACKS_DENSITY):
  `PREREG_stage0_secondk_20260730` → `analysis/stage0_secondk_read.py`
  (selfcheck PASS; sha256 3c297e18…) + one-line dumper patch
  (`probing/latent_uq.py` meta gains ep_batch). k2 ∈ {5,20}, 10 p2e
  cells all-or-nothing; substrate NOT on disk (anchors discarded at
  dump time) ⇒ registered EXISTENCE GATE (gate JSON w/ replay chunk
  counts) + SUBSTRATE-GONE branch; two-mode integrity-guard ladder
  (tolerance / fallback rank-floor 0.9) selected by a k=10 CALIBRATION
  dump (bf16 + categorical sampling ⇒ cross-backend drift is real;
  --platform cuda pinned); P-K1 robust iff 20/20 TRACKS_DENSITY,
  AMBIGUOUS counts as flip. 2 reviewer passes (2 BLOCKING: broken
  selfcheck float-equality asserts; prereg/reader design divergence
  from an interrupted fixer — reader design adopted, prereg rewritten
  to match). **31 Jul EXECUTION: existence gate PASSED (SUBSTRATE OK,
  10/10 runs, replay chunks 1520–1760) but the k=10 calibration hit the
  registered QUARANTINE branch (guard_mode=invalid: 7 tolerance
  failures AND err_position/err_steps/err_velocity rank corr
  0.74/0.84/0.81 < floor 0.9) ⇒ full passes correctly refused, no
  verdict. AUDIT
  (`artifacts/stage0_secondk_quarantine_20260731/AUDIT.md`): committed
  dumps were computed on Vast (/workspace); re-dump ran on RCC — dials
  and probeset sha identical, true_* arrays matched (pipeline
  faithful), only model-path arrays diverged ⇒ cross-MACHINE
  backend/env change, the registered drift mechanism beyond the rank
  floor. NOT evidence against TRACKS_DENSITY (frozen consequence map).
  Next: Vast-matched smoke retry (gate retry, no amendment needed) or,
  if the env is unreconstructable, quarantine-terminal ⇒ permanent
  single-k qualifier.** **08-01 VAST-MATCHED RETRY + ONE READ (executed
  ON the instance): K-SENSITIVE.** Smoke `--smoke_check` PASSED on the
  matched env (guard_mode=fallback, the registered branch — 5 tolerance
  failures but rank floor met; RCC had refused outright); full passes
  10 cells × k∈{5,20} × 5 ckpts, integrity 0 failures/1450 arrays;
  verdict **18/20 TRACKS_DENSITY, 2 flips both finger @k20**
  (seed2, seed5 AMBIGUOUS; all k5 and all cup verdicts track) ⇒ frozen
  consequence: every external TRACKS_DENSITY statement carries the
  k=10 qualifier + enumerates the flipped cells. Record
  `artifacts/stage0_secondk_20260801/READ_RECORD.md` (bundle
  `local_results/stage0_secondk_20260801_212204/`; **02 Aug provenance
  pulled + VERIFIED, chain closed**: json bytes = instance-recorded
  sha 75ab1bd2…, json verdict ≡ synced stdout, instance reader sha
  3c297e18… = registered freeze sha = local file, instance git head
  003d7b82 = committed ancestor on branch; disclosure: the recommended
  twice-run probe was not executed — partially covered by same-session
  smoke pass + read).
- **TM2-R3 TD-MPC2 cross-family replication REGISTERED 07-30/31**
  (Paper-2 LONG POLE; the ICML band-mover):
  `PREREG_tm2_competence_20260730` → labeler port
  `probing/tdmpc2_oracle_labels.py` (selfcheck PASS; 8eeb5949…; planner
  warm-start snapshot + torch-RNG CRN replace belief carries; smoke
  stdout estimand redaction) + `probing/tdmpc2_r3_train.py` (official
  OnlineTrainer subclass, early ckpt at 2.5e4) + compat dose patch
  (exact main.py distractor mirror; byte-identical for existing
  callers) + `scripts/tm2r3.sbatch` (4 stages, checkout pin, grid
  validation) + `analysis/tm2_r3_read.py` (selfcheck PASS; 00e83fc4…;
  r3_read import pins, FLOOR gate, dosed-smoke authenticity + task
  pins). Grid: {cup,finger}×{e1,e4}×seeds 51–58×{2.5e4,1e5} = 32 online
  runs + 64 oracle_all passes ≈ 130–230 GPU·h. P-TM2a/b/c mirror
  P-R3a/b/c; HARVESTED-BY-PLANNER = registered family-boundary branch.
  2 reviewer passes 0-blocking; all MINOR/NOTE hardening applied.
- **R3 Amendment 2 — cross-checkpoint consumer REGISTERED 07-30/31**
  (parent 8b3380ce re-entry clause): `PREREG_r3_amend2_20260730` →
  `--consumer_checkpoint` overlay in `d0/oracle_labels.py` (full
  selfcheck PASS; 3613f488…; counter snapshot/restore ⇒ control and
  swapped passes state-identical — per-state pairing; regex
  '^(rew|con|valens\d+)/'; no-op/partial-overlay guard; consumer-arm
  stdout REDACTION (value-blindness leak fixed pre-freeze); config
  comparer) + `analysis/r3_xconsumer_read.py` (selfcheck PASS;
  b4952946…; pairing QUARANTINE gate, smoke authenticity, max_steps
  pin) + `scripts/r3_xc_local.sh` (existence gate refuses to erase a
  prior SUBSTRATE_GONE marker). 128 passes on the R3 run dirs (kept in
  runroot_cleanup KEEP_PENDING); P-XC1 head-harm / P-XC2 head-deficit,
  prediction HEADS-IRRELEVANT (per P-R3c + reacher). 2 reviewer passes
  ×2 rounds (2 BLOCKING fixed: stdout estimand leak; stale disclosure —
  reacher read now named known-at-freeze).
- **Competence-repair intervention REGISTERED 07-30/31** (the R3
  consequence map's mandated NEW registration; the headline follow-up):
  `PREREG_competence_repair_20260730` → `d0/train_consumer_model.py`
  (LORO per-run ridge on cmfeat1 observables; selfcheck PASS
  deterministic across PYTHONHASHSEED; 2405c44e…; live ALLOWED_KEYS
  whitelist + source-scan) + `--consumer_model` chooser (cast-first f32)
  + `analysis/repair_read.py` (selfcheck PASS; bec84fc6…; determinism
  gate episode/step/m_now/g_all vs the COMMITTED labels as per-state
  control — zero control passes; frozen COMMITTED_LABELS_DIGEST +
  trainer-source sha + 3-way model-sha protocol) +
  `scripts/r3_rep_local.sh`. 32 repaired passes (~32–64 GPU·h);
  P-REP1 paired achieved CI>0, P-REP2 materiality 0.6465 (point label);
  verdicts REPAIRED / PARTIAL / NOT-REPAIRABLE-FROM-OBSERVABLES /
  HARMFUL. 2 reviewer passes ×2 rounds (1 BLOCKING fixed: salted-hash
  selfcheck nondeterminism). **08-01 EXECUTED → registered QUARANTINE
  (determinism gate; instrument outcome, no verdict computed, 31/32
  passes never run, no read executed).** Existence gate + trainer
  PASSED (32 LORO models, sha `be802f1e…` recorded pre-labeling);
  smoke detgate tripped (episode/step ok, m_now/g_all drift); audit:
  chooser EXONERATED (no-consumer re-pass drifts identically) — root
  cause = **labeling stack is not job-to-job deterministic on midway3**
  (same-day same-code pair: deter max 1.36, g_all per-state rank ρ
  0.02, m_now flips 48/200; committed-pass bitwise reproduction was
  never a property this substrate had). Bundle
  `local_results/r3_repair_quarantine_20260801_210414/`; audit record
  `artifacts/r3_repair_quarantine_20260801/AUDIT.md`. Recommended
  path: within-pass shadow-pairing amendment
  (`g_all[m_real] − g_all[m_real_probe]` inside each rep pass; model
  reusable; zero extra compute). **xc wave EXPOSED
  to the same hazard (pairing gate atol 1e-5 vs measured cross-job
  drift ~30–46) — one-cell same-job reproducibility probe required
  before submitting; TM2 not exposed (within-pass asserts).**
  **08-01 late AMENDMENT 1 REGISTERED (user GO):**
  `PREREG_competence_repair_amend1_20260801` (aefe13da…) — parent
  committed-paired primary RETIRED unmeasurable-on-substrate; amended
  primary = per-state within-pass [`g_all[m_real] −
  g_all[m_real_probe]`] (g_now cancels; both choosers already stored
  by the frozen labeler), same P-REP1/P-REP2 (bar 0.6465, bridge
  disclosure) + verdict map; gates: episode/step state identity vs
  committed ⇒ QUARANTINE, within-pass identities (oracle_all, m_real
  max ghat, NEW m_real_probe max real_scores + finiteness), NO
  cross-job float gate anywhere; sensitivity leg excl. the
  pre-amendment smoke cell + registered FRAGILITY DISCLOSURE
  (materiality is a point threshold — deterministic flip leg in
  selfcheck); driver byte-unchanged (comments stale by design; smoke
  stage re-mints the detgate marker); model-restore rule (never
  retrain — sha mismatch would crash the read). Reader amended
  (selfcheck PASS; c92b2fd7…). ONE reviewer (Opus 5): 0 BLOCKING /
  4 MAJOR / 7 MINOR / 4 NIT; all MAJORs fixed pre-commit — drift
  figure corrected to ≈50% (`m_real` 101/200; 25% was `m_now`), the
  three audit scripts ARCHIVED VERBATIM
  (`artifacts/r3_repair_quarantine_20260801/audit_scripts/`; none
  loads chooser fields from the rep file; smoke npz git-pinned
  dd3138e7 sha 0d549b6b…), selfcheck source-pinning e2e numerics +
  episode-half legs (d_pair-from-ghat and episode-drop mutations both
  verified CAUGHT on a mirror); MINORs applied (scoped
  no-cross-job-float wording, ghat/real_scores finiteness guards,
  non-read precedent clause for within-pass gate aborts, audit_nocm
  file disclosure, model-restore rule); skipped by design:
  branch-varying JSON schema NIT, mechanical one-read guard
  (procedural, parent-inherited). Remaining: smoke detgate re-run →
  31 passes → ONE read.

- **Unfrozen stress waves U1–U4** (REGISTERED 07-27; motivated by the frozen-protocol audit `research_notes/Audit_FrozenProtocol_20260726.md`): `PREREG_unfrozen_u{1,2,3,4}_20260727` → `analysis/unfrozen_stress_read.py` (four subcommands, selfcheck PASS) + `orth_spin_unfrozen` config. **U2/U3/U4 READ 07-30** (see Paper-1 table: robust / robust / floor-censored — no flips anywhere; `artifacts/unfrozen_stress_u234_20260730/`). **U1 READ 08-01 (ONE execution, local): PROTOCOL-ROBUST — the strongest branch.** P-U1a interaction +145.0 [+22.9, +276.5] n=8 PERSISTS unfrozen; P-U1b apt occupancy +10.7 [−7.1, +26.5] — the reward-free null HOLDS under fine-tuning ⇒ hi-occupancy reward-free support is not merely frozen-illegible, it is not an initialization advantage either; headline + null family generalize beyond the frozen protocol. Unfrozen levels: task +225/+221 vs frozen (both sides). Bundle `local_results/unfrozen_u1_20260801_212204/` (32/32, exclusions all foreign waves) → `artifacts/unfrozen_u1_20260801/unfrozen_u1.json`. **U1–U4 family COMPLETE: no flips anywhere — the frozen-protocol scope objection is fully discharged.** Frozen registrations untouched under every branch — flips would have changed generality language only.
- **Pretrained-encoder pixel arm REGISTERED 07-30** (GO'd; the swamping account's lever prediction): `PREREG_pe_pixel_20260730` → `analysis/pe_pixel_read.py` (selfcheck PASS; sha256 34380e0f…) + code: `agent.frozen_enc` (optimizer module filtering), `offline_fit` partial-init w/ counter reset (the skip trap), `AXIS1_INIT_WM` plumbing, `scripts/check_frozen_enc.py` integrity gate (training witness via OFFLINE_FIT_PROGRESS; committed selfcheck; sha256 00acbdff…). Design: 16 stage-2 pixel fits (task-mode full, enc loaded '^enc/' from seed/side-matched fpxpx donors + frozen) + 16 frozen-readout adapts + E4 pass (COLLATE to a NEW csv). P-PE1 graded (CI<2.0 INCLUSION-RESTORED / <19.41 PARTIAL / else NO-RELIEF vs pinned 23.34 baseline); P-PE2 lift vs committed X2 cells; P-PE3 conditional occupancy split. Smoke gate + 16-run integrity sweep with persisted log. 2 adversarial reviews (0 blocking; all 4 MINOR hardening items applied incl. the counter-skip witness + subshell-safe sweep gate). Donors protected in runroot_cleanup (KEEP-PENDING).
- **From-scratch baseline anchor REGISTERED 07-30** (GO'd; descriptive class — no decision rules): `PREREG_scratch_anchor_20260730`. 8 jobs, `pilot.sbatch MODE=goal` (plain dmc_proprio, zero code change), RUN_ID `adapt_scratch_finger_seed<k>_ckpt0` → collates as mode `scratch` (no frozen reader consumes it; 10 readers verified mode-pinned). Purpose: absolute-scale anchor for Fig 2a; %-of-anchor reporting never load-bearing. Reviewed (0 blocking; env.sh sourcing + staging checks added). **31 Jul LANDED: 8/8 complete + QC-pass (112 episodes each; config audit 8× from_checkpoint ''/frozen_wm false; bundle `local_results/scratch_anchor_20260731_170618/` manifest-verified). Anchor: AUC50k 113.2 (sd 42.0), AUC100k 147.7 (33.2), AUC125k 174.0 (39.2), final10 319.9 (151.7; seed-8 late collapse to 0.0, descriptive). % anchoring entered `Paper1_ResponseCurves`: W1 +84.0 = 57% of anchor AUC100k, Amend-1 +51.5 = 35%; task_s1 267.9 = 1.81×; reward-free plateau 0.53–0.68× (denominators only, protocols differ — no claim licensed). Fig 2a build blocked on a scores-only pull of per-arm adapt scores.jsonl (none local). **08-01: pull landed (`local_results/fig2a_scores_20260801_122853/`) + Fig 2a BUILT post-U1/25m-reads (`research_notes/figures_paper1_20260730/fig2a_anchoring.png` + data json + archived generating script): unfrozen-protocol arms (frozen-era per-episode scores no longer on scratch; U1/U2 no-flip reads license the substitution, captions must say "unfrozen") + scratch anchor; recomputed scratch AUC100k 147.68 matches the registered anchor exactly; unfrozen task clears scratch 2.3–3.3×, every reward-free arm sits at/below the scratch curve.**
- **High-f_R falling-limb theory predictions FROZEN 07-30** (pre-buffer; GO'd wave, registration to follow): `PREREG_highfr_theory_20260730` — P-HF1 falling limb / P-HF2 support-starvation dissociation (membership intact + diversity_pr lower) / P-HF3 interior maximum, + registered failure modes (monotone rise refutes the breadth-patch trade-off form). Derivation note `Theory_HighFR_Prediction_20260730.md` (disk-only). No high-f_R buffer exists; 25m + U1 not consulted.
- **Paper-3 confirmatory 25m point** (REGISTERED 07-26, to submit after freeze-commit): `PREREG_compcapacity_confirm_20260726` → `analysis/compcapacity_read.py` (selfcheck PASS). Primary = task-lo rise (25m−12m, s0 task cells, two-sample seed bootstrap, CI>0 fires); GUARD = forbidden pattern (apt rises while task-lo floored ⇒ P-E1 refuted); 32 jobs at `size25m` (existing config block) on `q1_ctx25` (build via resize tool, deter 3072 stoch 32×24) + E4 pass `ax1wm_finger_*s25*`; power: detects rise ≈33 vs full transition 139; no blank slots — realized values go in the artifact. **08-01 READ (ONE execution, CLUSTER-SIDE by the staged driver; adopted — reader/collator/prereg shas verified byte-identical to local frozen files; 32/32 done): PRIMARY DOES NOT FIRE** — task-lo rise −12.2 [−45.7, +17.7] (t-sensitivity agrees) ⇒ regime A persists through 25m, transition bracket widens to [12m, >25m] (G1 crossing above 25m), bounded null, no ordering violation; **guard clean** (apt rise +7.0 ns — forbidden pattern NOT triggered, P-E1 not refuted); composition interaction PERSISTS at 25m (+113.8 [+68.9, +155.4], 8/8 pos; b_task_25m +119.2*; task-hi change −32.2 ns). E4 membership pass still running (feeds the THEORY read P-E2/P-E3, not this one — frozen CLI consumes auc.csv only). Record `artifacts/compcapacity_25m_20260801/` (json + code shas); bundle `local_results/compcapacity_25m_full_20260801_214345/` (scores-only per sync policy — small by design). **02 Aug USER RESOURCE DECISION (standing, like 24+24): capacity axis CAPPED at 25m — no >25m runs will be registered** (25m = 4×5090 for a full week; single-wave wall-clock at that scale is the affordability limit) ⇒ the A→B transition is permanently right-censored; bracket [12m, >25m] is FINAL; the flagship factorial redesigns to composition × capacity ≤ 25m with **capacity-robustness of the composition effect** as the flagship claim (interaction persists at every tested size; +113.8* at the cap) and the right-censored transition reported as a bounded scaling fact. Theory adjudication (P-E1–E4) unaffected — it consumes existing waves only. **02 Aug E4 membership counterpart LANDED (registered descriptive; bundle `local_results/e4_25m_20260801_225152/`, provenance verified — code shas match local, head committed ancestor; 128/128 rows):** task@25m,s0 in-regime rew-NLL (h=0) **mean ≈ 8.06 nats (per-seed 3.77–10.81) vs 12m reference 5.77** — the membership form did NOT move toward the band ⇒ **regime A persists through 25m on BOTH metrics (AUC flat + membership not approaching): the bounded null is metric-consistent**; apt rows rew-NLL absent-by-design (reward_aware gate as disclosed); no decision rule touched (csv + record in `artifacts/compcapacity_25m_20260801/`); P-E2/P-E3 theory read stays future (Scaling-B completeness blocker open).
- ~~Volume-anomaly replication P-E4a~~ → READ 07-29 (see Paper 3 table).
- **Scaling-B completeness blocker**: latest exclusions snapshot (07-25 orthogonal bundle) lists 4 `ax1(f)s12*` adapt runs missing scores.jsonl (`fs12q1s0_seed8`, `fs12q1s1_seed7`, `s12q1s1_seed2`, `s12q1s1_seed7` — set CHANGED from the X2 snapshot; seed6 completed since). Re-run or explain before the Scaling B read.
- **Comp×capacity theory predictions** (registered 07-24, adjudicated by Scaling B + 12m E4 + volume replication): `PREREG_compcapacity_theory_20260724` (P-E1 ordered un-nulling / forbidden pattern; P-E2 task-lo-first; P-E3 metric split; P-E4 diversity-as-λ); derivations in `research_notes/Theory_CompCapacity_Addendum_20260724.tex`; MUST be committed before the Scaling B read.
- ~~Paper 4 pilot / calibration / confirmatory sweep~~ → READ 07-30, ALL FOUR PRIMARIES FIRE (see the Paper 4 section table).

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
