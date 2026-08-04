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
| 08-02 | **Pretrained-encoder pixel arm** (16 frozen-enc fits + 16 adapts + E4) | `PREREG_pe_pixel_20260730` | `artifacts/pe_pixel_read_20260802/` | **NO-RELIEF (registered honest negative)**: P-PE1 in-regime rew-NLL 23.03 [20.14, 25.68] ≡ the swamping baseline 23.34 [19.41, 27.29] — removing the encoder from the gradient competition entirely moved reward-NLL by NOTHING; P-PE2 no lift off the X2 floor (−8.5 [−26.0, +9.6]); P-PE3 premise-idle ⇒ the swamping account's causal lever FAILS its first intervention (measured fact P-SW1 stands, NOT re-adjudicated); U4's pretrained-encoder decision resolved — pixel floor survives encoder pretraining; integrity sweep 16 PASS w/ training witness (counter-skip trap excluded); no further pixel arms queued (account revision first) |

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
| 08-02 | **Competence-repair READ under Amendment 1** (ONE execution, local; bundle `local_results/r3_repair_20260802_143054/` manifest 0-mismatch; code shas ≡ committed finals incl. labeler 6561eff9…; REP_GATE/REP_DETGATE OK, 32/32 passes; model sha be802f1e… matches the frozen pin; state-identity violations 0; selfcheck PASS pre-read) | `PREREG_competence_repair_20260730` + `PREREG_competence_repair_amend1_20260801` | `artifacts/repair_read_20260802/` | **NOT-REPAIRABLE-FROM-OBSERVABLES: P-REP1 +0.070 [−0.092,+0.245] straddles 0** (n=32; sensitivity leg n=31 excl smoke cell +0.074 — NOT fragile); achieved_rep +0.011 / in-pass probe −0.060 / committed cross-era −0.039, all ≈0; chooser change rate 83.3% (the LORO model re-picks constantly — it just adds no value); residual gap fraction 99.3%; cross-era drift context exactly as the audit predicted (probe-vs-committed agreement 32.6%, opportunity max\|Δ\| 97.0 — gates nothing) ⇒ **a LINEAR probe on stored observables (deter, action, qfull, udyn) does not close the competence gap; representational account strengthened (support must be legible cross-link); scope-limited by registration — NOT "no repair possible"** |
| 08-02 | **Cross-consumer (xc) READ under Amendment 1 — within-pass dual-chooser** (ONE execution, local; bundle `local_results/r3_xc_20260802_135102/` manifest 0-mismatch; code shas ≡ committed finals; XC_GATE/SMOKE/DUALGATE OK; 64/64 dual labels; integrity violations 0; cands_max_absdiff 0.0 — in-process RNG replay EXACT where cross-job floats drift O(100), vindicating the within-pass redesign; selfcheck PASS pre-read) | `PREREG_r3_xc_amend1_20260802` (parent `PREREG_r3_amend2_20260730` retired unexecuted) | `artifacts/r3_xc_read_20260802/` | **MIXED — off-map CI excursion: P-XC1 late-eval +0.028 [−0.216,+0.244] straddles (matches the registered HEADS-IRRELEVANT prediction); P-XC2 early-eval −0.155 [−0.311,−0.032] ENTIRELY NEGATIVE where the registered fire direction was + ("mature heads rescue") ⇒ verdict MIXED per the frozen map: disclosed, NO headline claim, interpretation deferred.** Direction domain-consistent (cup −0.146\*, finger −0.164 straddling same sign); chooser disagreement 75–85% (swap real, no SUSPECT-NO-OP); opportunity matched across sides (1.37/1.36). Mature heads transplanted onto an early agent REDUCE achieved value — consistent with heads-features-occupancy co-adaptation (interface mismatch caveat registered symmetric to the HEAD-DEFICIT caveat); any interpretive follow-up (e.g. interface-controlled swap) = NEW registration; regardless-clauses reaffirmed (committed reads stand, labels never re-read, 24+24 frozen, Route A closed, imag retired) |
| 08-02 | **XC2 excursion decomposition** (EXPLORATORY post-read on the adjudicated xc bundle; plan fixed in chat pre-execution; d_pair anchor matches the read exactly both sides) | — (exploratory; cannot alter the MIXED verdict) | `artifacts/xc2_decomposition_20260802/` | **The excursion's significant carrier is the PROBE-PICK term: real_term −0.152 [−0.269,−0.052] run-clustered (harm story confirmed); now_term +0.004 ns (self-consistency story rejected at run level). DIRECTIONAL: late→early damages (−0.152\*), early→late is value-neutral (+0.018 everywhere) — generic swap noise can't do this (now_term IS side-symmetric: non-floor +3.0/+2.2 both sides = optimizer's-curse decorrelation, mechanical).** Floors 93%/88.5% (17/32 early runs constant-opp); non-floor 7% (opp≈19.6) carries d_pair −4.27 = real −1.27 vs now +3.00; floors DILUTE (excursion would be ≈−0.30 without them). Concentration: finger/e1 holds the 4 worst runs (real_term to −1.15); per-domain real_term finger −0.240\*, cup −0.063. ⇒ interface-controlled swap (aligner-mediated transplant) now WELL-POSED as a NEW registration (separates interface-mismatch vs value-knowledge-mismatch — the registered caveat's two branches) but **DEFERRED (user 2 Aug: revisit at Route-B §6 drafting or on reviewer pressure; no default GO)**; meanwhile the anatomy folds into Route-B §6 as disclosed MIXED + excursion, no new compute; nothing here is confirmatory |
| 08-03 | **TM2-R3 TD-MPC2 cross-family replication** (ONE read; manifest 370/370; RCC head 481aced9 clean; all 5 instrument shas byte-identical; SMOKE_OK + API smoke + 4 dosed smokes reader-enforced; 32 runs TRAIN_DONE, 64/64 labels; FILL audit executed at read — 64/64 planner blocks exact, num_q 5 / latent_dim 512, steps 25000/100001 ×32; **disclosed deviation: the registered pre-labels FILL amendment file was never created — data bundled, audit discharged at read, no decision quantity touched**) | `PREREG_tm2_competence_20260730` | `artifacts/tm2_r3_read_20260803/` | **CROSS-FAMILY REPLICATION (strongest registered branch): P-TM2a opportunity +0.546 [+0.325,+0.808] FIRES (bar 0.2) + P-TM2b gap +0.530 [+0.315,+0.787] FIRES — the 512-sample 6-iteration MPPI planner leaves the one-real-step oracle opportunity on the table (pooled achieved +0.016 [−0.036,+0.074], a tight null: it harvests essentially NOTHING); P-TM2c maturity null (−0.034) = the dv3 maturity null reproduced.** Same shape as DreamerV3 in every registered secondary: dose GROWS opportunity (e1 +0.304 → e4 +0.789), both domains fire same-direction (cup gap +0.777\*, finger +0.283\*), floors pass (cup 0%, finger 15.6% < 50%). Effect ≈42% of the dv3 gap, above the ≈0.42 pre-sized bar ⇒ **decomposition = TWO-FAMILY regularity incl. a categorically stronger consumer class; HARVESTED-BY-PLANNER boundary did NOT materialize; Paper 2's strongest generality leg lands; the user's venue rule "ICML if TD-MPC2 repl fires" has its condition MET (call stays ~Dec)** |

Paper-2 plan: `research_notes/other research/EVPI_Plan_Revision_20260724.md`
(v4). Constants (early −0.208\*/late×e4 +0.313\*) = hypotheses to
re-test, not results. 24+24 permanently frozen; imag op retired.

## Paper 3 — composition×capacity flagship (own venue clock)

| Date | Experiment | Registration | Record | Verdict → decision |
|---|---|---|---|---|
| 07-29 | **Volume-anomaly replication P-E4a + diversity secondary** (ONE read, cluster-side) | `PREREG_volume_repl_20260726` | `artifacts/volume_repl_20260729/` | **PRIMARY FIRES: fresh v400s1−v200s1 +331.0 [+169.3,+502.7]** (> the original +228.6 — no winner's curse; pooled-12 +279.8\*; s0 fresh +194.8\* same direction) ⇒ **anomaly REAL — support breadth enters the theory as a λ-structure input distinct from occupancy**; diversity direction HOLDS (diversity_pr v400s1 10.63 > v200s1 9.23 under the pinned `spectral_v1_1_20260724` gate, while generic spectrum_pr goes the other way and v400's occupancy is 3.4× lower) ⇒ P-E4b diversity leg engages; non-spectral-account branch not triggered; conditional-on-occupancy regression form stays unregistered; local re-execution bit-identical |
| 08-03 | **High-f_R falling-limb wave** (ONE read; manifest 151/151; instrument shas ≡ amendment finals; both f-gates PASS: side1 0.7248, side0 0.4088 included) | `PREREG_highfr_wave_20260802` + amend1 + amend2 (freezes `39a15880`/`be60afbe`/`080aa7b2`) | `artifacts/highfr_read_20260803/` | **SATURATION (registered branch) — P-HF1 does NOT fire: hi-f − v200s1 = +49.7 [−8.4, +102.8]**; monotone-rise refutation does not fire (chain 187.7→244.6→237.5 non-monotone); P-HF2 membership INTACT (side1 rew-NLL 1.178 [1.119,1.237] ≤1.5) but diversity_lower FALSE ⇒ dissociation NOT adjudicated; P-HF3 weak HOLDS (argmax interior f=0.4088). Disclosed riders: CI excludes the registered predicted band −88..−108; **mediator never moved — diversity_pr(side1) 10.26 > ref 9.23 (this pool's high-f tail is BROADER, not starved)**; both curated cells above the natural buffer (+49.7/+56.9, straddling) with source_l1=1.0 tier-pure composition (mid vs high) — no f_R attribution licensed ⇒ trade-off form neither supported nor refuted; flagship breadth axis gains NO falling side; grid-densification option does NOT trigger |

## Paper 4 — algorithmic collusion (MARL/econ venue, own clock, Paper-4 chat)

| Date | Experiment | Registration | Record | Verdict → decision |
|---|---|---|---|---|
| 07-25 | Pilot stages 1–3 (baseline reproduction, measurement calibration, Design-B viability) | design memo `Design_Collusion_Pilot_20260724.md` | `artifacts/collusion_stage1_20260725/`, `collusion_stage23_20260725/` | stage 1 GO (Δ 0.831 in published range, fp 68/100; failure-to-learn minority ⇒ fp-conditional measurement); distance label recommended (undercut saturates + inverts); RQ1 resolved via `coverage_late`; Design B VIABLE (float-exact 20/20) but deletion confounds volume (composition increment −0.107) ⇒ graded volume-controlled interventions required |
| 07-30 | Confirmatory-prereg calibration | (calibration, pilot seeds only) | `artifacts/collusion_calib_20260730/` | terminal reference frozen (exo degenerate: mean .981, 96%>0.9); ladder d.25 inert / d1 fp-cond −0.134; **Design-A smoke: DISSOCIATION found (forbid λ=1 ⇒ Δ +0.987 w/ fingerprint 0/10, degenerate attractor) ⇒ P-CLA primaries re-specified pre-freeze**; 2 adversarial review passes (10+11 findings, 2 blocking reader-corruption defects) all fixed |
| 07-30 | **Confirmatory sweep READ** (ONE execution; freeze `42bbe487`; manifest 16/16, snapshots ≡ freeze) | `PREREG_collusion_confirm_20260730` | `artifacts/collusion_confirm_20260730/` | **ALL FOUR PRIMARIES FIRE — P-CLB1 −0.104 [−0.165,−0.046] + P-CLB2 −0.592 [−0.724,−0.447]: punishment-phase experience is a composition-specific driver of BOTH profit level and punishment structure at matched update mass (dose ladder monotone, mid-dose fires); P-CLA1 −0.650* + P-CLA2 +0.135*: forbidding exploratory undercuts collapses the fingerprint (0.65→0.00) while Δ RISES (0.852→0.987) ⇒ experimentally induced Abada–Lambin failure-to-learn regime — punishment-generating exploration necessary for GENUINE collusion, not for supra-competitive prices. All gates pass (exact 100/100, n_fp=76, manip −0.485*). ⇒ RQ2 composition leg LANDS; RQ3 coupling wave AUTHORIZED; two-outcome measurement argument load-bearing** |

## Paper 5 — uncertainty field / No Free Information (own clock, Paper-5 chat)

| Date | Experiment | Registration | Record | Verdict → decision |
|---|---|---|---|---|
| 08-01 | **NFI Level-1 pilot** (EXPLORATORY; harness `uncfield/`, 4-model planted selfcheck battery PASS; 1 adversarial harness review "trustworthy after fixes", 4 BLOCKING fixed pre-read, pilot1 killed unread) | design note `Pilot_NFI_Design_20260801.md` + Amendment 1 (gitignored; pre-outcome) | `artifacts/nfi_pilot_20260801/` | **FLAGSHIP SIGNATURE FIRES: ensemble EXPLOIT-SURVIVES-CIG+PBIM (3/4 members; conjunction exploits 13/61/0/61)** — planner finds evidence-neutral cycles (referee-certified, incl. the designed duplicate-pair loop at 31× true gain and the zero-information noisy-TV loop) farming predicted info through BOTH prefix-conditioned EIG and drift-adjusted realized-ΔH; member heterogeneity real (m2 = clean-accountant control); holonomy-predicts-farming NOT supported (commutator residue stays diagnostic) ⇒ next: mechanism analysis + sensitivity sweeps, then claim-freeze audit + registration; NOT a registered claim yet |
| 08-02 | **NFI mechanism analysis** (instrumented replay of 8 cycles/member on the frozen pilot2 ensemble) | — (exploratory follow-up) | `artifacts/nfi_mechanism_20260802/` (⚠ **AMENDED same day, post-review — read through the prepended amendment**) | **Anatomies (1)+(2) SURVIVE adversarial review** (formula = exact Kalman entropy drop; coherent-adapter calibration exactly 1.000; denominators non-degenerate; record bit-reproduces 421/421): (1) update-operator over-contraction 1.4–11.6× own-heads license (cleanest m0/m1/m2; m3 = heads-vs-referee dominant); (2) TV fictitious info — but +0.84 = carried EIG PROMISE not realized contraction (eig-only; fires 2/4 members). **Anatomy (3) dichotomy AMENDED AWAY**: m0 off-support exhibit inverts under matched-walk baselines (walk drift); leakage real for m1/m3 only (static components — relaxation excluded); duplicate-double-count WITHDRAWN (exhibit c451 non-neutral non-exploit; c447/c451 name conflation corrected; c447 = 50/50 split); no thresholdable cycle-level discriminator ⇒ self-consistency ratio = model-level diagnostic candidate only |
| 08-02 | **CEI Level-1 pilot** (equivalence-pair construction + 3 registered kill probes; selfcheck PASS) | design in module docstring (exploratory) | `artifacts/cei_pilot_20260802/` (invalidation note prepended) | ✖ **VERDICT WITHDRAWN same day — harness review "redesign needed"**: TV-sensor shadowing meant NO behavioral separation ever occurs (separation curve = plain node-3 hitting time, bit-exact); ψ deducible from the action log alone (coverage bookkeeping — impossibility tautological as built); fixed patrol = selective-labels carve-out (v2-audit criterion 2) + registered Failure C. Survives: ε=0 ψ-gap pair analytically sound; repaired sense-loop verified to produce real divergence ⇒ **CEI v2 redesign required** (U-controlled movement + failure non-deducible from coverage bookkeeping + explicit access model + A/B/C assessment); theorem-phase go/no-go NOT licensed by this record |
| 08-02 | **CEI v2 pilot** (calibration-falseness redesign: coherent liar R̂₇=0.01 vs true 1.0, committed uncertainty-greedy policy; **reviewed BEFORE read** — "fix before run/read", B1 movement→collection-intensity re-scope + B2 deadlock fix + B3 hazard redesign + N1–N6, core construction verified-and-held; selfcheck re-PASS) | `Design_CEI_v2_20260802.md` §§7b–7c (pre-outcome + review record) | `artifacts/cei2_pilot_20260802/` | **SURVIVES post-review: passive identification bounded (U_false 1 s7 read, logLR ∈ [−0.33,+0.12] across seeds, ψ=0.5 vs U_true 19 reads, logLR +506..+1048, ψ=0; in-probability: P(1-read LR>log19)=1%); ε-separation cost = 1/hazard with h∼ε^4.6 (733→199k steps over ε .5→.15; one late read decisive at ~46 nats) vs targeted audit 7 steps flat (gap 10²→3×10⁴, unbounded); E4 audit-cost curve m* 3/7/21/51/173 for R̂/R .01–.7; L2 blind spot: coherent liar passes replay audit at 1e-17 vs parity operator flagged at .083 with ψ dissociation OPPOSITE** ⇒ theorem phase licensed (bounded-LR impossibility at L1; intervention cost O(diam+m*(γ)); L2 blind spot scoped to (R̂\|known-dynamics) family); v2.1 coupled arm (route control) parked with prototype facts |
| 08-02 | **NFI sensitivity sweeps** (8-job queue on RCC: ymode sample ×3, train {1k,10k,30k}, dataseed {1,2}, hid {32,128}; bundle manifest-verified; **instrument reviewed BEFORE read** — "fix before read": verdict leg CLEARED (classify/aggregation verbatim-shared with pilot, axes verified genuine, resume-guard summary bit-reproduced), self-consistency column EXCLUDED (B2: 34/40 cells null/mistargeted), mechanism record amended same day, sweeps.py empty-ensemble hole patched) | — (exploratory; the "just undertrained" decider) | `artifacts/nfi_sweeps_20260802/` | **STRONG OBJECTION REFUTED: 37/40 members ≥ CIG-ONLY across every axis (CIG-farming persists at 10× training); flagship CIG+PBIM conjunction robust across obs-mode (3/3 sampled seeds), fresh data (2/2 dataseed pilots), and capacity-UP (hid128 = only unanimous 4/4 flagship) — but TRAINING-DOSE-SENSITIVE: flagship members 0/4 @1k → 3/4 @3k → 1/4 @10k → 1/4 @30k = realized-ΔH farming is a mid-training phenomenon (attenuates, does not vanish); exploit GROWS with capacity (hid32 = only NO-EXPLOIT member)** ⇒ headline must scope the conjunction by training dose; claim-freeze lit audit UNBLOCKED; optional follow-ups: per-member ratio re-search; hid128×30k interaction |
| 08-02 | **NFI defense-fidelity arms** (γ-discounted Ng-form rescore, REQUIRED + CIG-faithful disagreement kernel; both built+selfchecked same day; **arms review "fix before read" — machinery certified bit-exact but BOTH registered read quantities replaced; amended protocol REGISTERED pre-read, memo §6.3**; kernel re-run w/ walk-matched twins) | memo §§1-C1, 5.3, 6.3 (protocol governs) | `artifacts/nfi_gamma_kernel_20260802/` | **γ-arm (gates 0.0 ×8): the exploit is a RETURN-PREFERENCE phenomenon, not an unbounded-stream one — telescoped preference for the farming future is γ-INVARIANT in sign (m0 +3.64/+0.66/+0.12, m1 +8.13/+0.83/+0.08, m3 +2.72/+0.72/+0.19 at γ=1/.995/.99) while the reward STREAM self-limits at rate (1−γ)·(entropy displacement): m0/m1 exploit streams fully taxed at γ≤.995, but member-heterogeneous — 34/61 of m3's conjunction cycles farm level-neutrally and their streams SURVIVE γ=.99 outright. Kernel arm: registered THIRD outcome — estimator-insensitive-in-regime (matched-twin per-action surpluses one narrow band, twins beat exploits as often as not; no Prop-2(ii) decay; open-loop divergence dominates) ⇒ C1's principle-vs-estimator scoping now EMPIRICALLY grounded; refutation reading forbidden** ⇒ both arms publishable both ways; remaining pre-freeze arm = P1 (PRIME penalty, §6.2 spec) |
| 08-02 | **NFI claim-freeze literature audit** (24-agent workflow: 4 deep-readers CIG/PBIM/Caron/BiasedDreams + 4 claim-collision searches + fresh-lit scoop scan + adversarial refuters on every serious collision + synthesis; refuters corrected 7 claimed collisions downward) | — (audit; memo = pre-registration input) | `research_notes/NFI_ClaimFreeze_Audit_20260802.md` (gitignored memo) | **ZERO KILLS, ZERO SCOOPS — all four claims freeze, NONE as worded**: C1 rescope (both defenses implemented principle-faithful but estimator-unfaithful — CIG scores parameter-info w/o any belief object, PBIM's guarantee is a terminal correction we never encounter ⇒ claim = "survives the transplanted principles, outside the proven scope of the published estimators"; **REQUIRED pre-freeze: γ∈{.99,.995} Ng-form recompute from saved beliefs — deficit straddles EPS_PRED, numerically threatens the flagship count**; CIG-faithful kernel arm strongly recommended); C2 rescope (dissociation is ours; Pan et al. owns capability-axes template; gradient-steps-not-data qualifier mandatory); C3 rescope (split m3 out of the sane-heads sentence; **Bayer et al. 2021 conditioning gap = biggest citation risk**, capacity contrast runs OPPOSITE to amortization-gap lit); C4 blocked-as-worded (Chen 2605.06915 May-26 priority on the diagnostic concept; "self-consistency ratio" name OWNED by Schmitt ICML-24 → rename "Bayes-license ratio"; claim only if the queued ratio re-search predicts exploit counts, else discussion-level); battery adequacy: PRIME (18 Jul, aleatoric-penalized IG) must be re-scored or carved out; LPM + learnable-novelty = registered scope sentences; **relabel EXPLOIT-SURVIVES-CIG+PBIM → principle-level names**; RECOMMEND freeze+register NOW (don't wait for hid128×30k); re-scan scoops after 4 Aug + OpenReview pass pre-submission |

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
  reacher read now named known-at-freeze). **02 Aug DETERMINISM PROBE
  FAIL (`artifacts/r3_xc_detprobe_20260801/`, diagnostic — no
  registered gate ran, no wave label exists): control-config overlay
  pass run twice, identical dials — episode/step EXACT but g_all
  max|Δ|=95.0 and m_now flips 64/200 (32%) ⇒ the registered two-pass
  pairing (pairgate atol 1e-5) is UNEXECUTABLE on this substrate; the
  128-pass wave is NOT submitted (would have QUARANTINED at the
  pairgate). Same substrate fact as the repair parent. Path forward
  queued for GO: xc Amendment 1 = within-pass dual-chooser pairing
  (one pass computes g_all once, evaluates control + overlaid choosers
  on shared features; 64 passes, half cost; labeler extension + amended
  reader + dated amendment before any wave label). Disclosure:
  `--consumer_checkpoint` has now run in CONTROL config only (one cell
  ×2, labeler seed 0 ≠ wave seed 1); no swapped pass, no
  cross-checkpoint statistic exists; probe touched integrity
  quantities only (stdout redaction active).** **02 Aug xc AMENDMENT 1
  BUILT on user GO (pending freeze-commit; ONE reviewer run same day):
  `PREREG_r3_xc_amend1_20260802` (sha ef9e31b4…) + dual-extended
  `d0/oracle_labels.py` (`--dual_chooser`, version `_xc2`; DualHeads
  cached head swap w/ witness read-back + identical-heads refusal;
  per-state gates: candidate identity ≤1e-4, env-reward CRN ≤1e-6,
  RNG-consumption identity; RNG schedule identical to a non-dual pass;
  default paths byte-identical, full selfcheck PASS incl. new xc8 dual
  legs + all pre-existing legs; sha b437a6a7…) + amended
  `analysis/r3_xconsumer_read.py` (64-file `_dual` grid, exact `_xc2`
  pin — `_xc1`/plain files trip; consumer-must-be-other-maturity pin;
  QUARANTINE re-scoped to within-pass integrity; SUSPECT-NO-OP honesty
  valve; value-blind `--dualgate` replaces pairgate; selfcheck PASS;
  sha 81793426…) + amended driver `scripts/r3_xc_local.sh` (stages
  gate→smoke[2 dual]→dualgate→labels[63]; sha 772ef3cd…). Estimand =
  the parent's achieved contrast realized within-pass
  (d_pair = [g_all[m_real_x]−g_all[m_now_x]] −
  [g_all[m_real]−g_all[m_now]]); decision rules/prediction/consequence
  map inherited verbatim; ≈0.55–0.6× parent compute (~70–150 GPU·h).
  Regression: repair_read + r3_read selfchecks still PASS (labeler
  default paths untouched). **Review DONE 08-02 (ONE reviewer, Opus 5):
  0 BLOCKING / 3 MAJOR / 5 MINOR / 8 NIT — logic/inheritance/ordering/
  value-blindness all confirmed machine-true (incl. jax swap semantics
  against agent internals, RNG bookkeeping, guard-exemption of the
  device ops, three-way probe exclusion, dualgate blindness); all 3
  MAJORs were selfcheck-strength gaps, ALL FIXED + verified caught on
  mirrors: M1 CI-endpoint fire rules now discriminated by
  nondegenerate-CI legs (both endpoint mutants killed); M2 the RNG
  replay now exercised with a REAL counting oracle (schedule-invariance
  final-counter equality + over-consuming-shadow trip; reset-deletion
  and guard-deletion mutants killed); M3 DualHeads itself now executed
  (jax-free identical-heads-refusal leg + jax-backed two-store swap +
  lossy-store witness trip; both guard-deletion mutants killed). MINORs
  applied: m_now/m_real ranges restored to the QUARANTINE class,
  DUAL_CANDS_ATOL import-pinned to the labeler, dual_allow_identical
  stamped + pinned False in DIALS, shadow-executes-CONTROL-cands leg
  (threshold-straddling sub-atol drift; cands_x mutant killed),
  process-count guard; NITs: SystemExit conversions, flag dependency,
  shadow array shape checks, dead fields dropped, side_diffs reuse,
  prereg wording (labeler-vs-reader enforcement split; probe seed-0).
  Post-review shas: prereg 05cd8772…, oracle_labels 6561eff9…,
  r3_xconsumer_read 2908a78c…, driver 772ef3cd… (unchanged). All four
  selfchecks + driver dry-run PASS. COMMIT-READY.** **08-02 COMMITTED
  (4decd451) + wave RAN + READ — MIXED (off-map XC2 negative
  excursion); see the Paper-2 table row 08-02 and
  `artifacts/r3_xc_read_20260802/`.**
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
  (procedural, parent-inherited). **08-02 detgate re-run OK + 31 passes
  RAN + READ — NOT-REPAIRABLE-FROM-OBSERVABLES (P-REP1 straddles, not
  fragile); see the Paper-2 table row 08-02 and
  `artifacts/repair_read_20260802/`.**

- **Unfrozen stress waves U1–U4** (REGISTERED 07-27; motivated by the frozen-protocol audit `research_notes/Audit_FrozenProtocol_20260726.md`): `PREREG_unfrozen_u{1,2,3,4}_20260727` → `analysis/unfrozen_stress_read.py` (four subcommands, selfcheck PASS) + `orth_spin_unfrozen` config. **U2/U3/U4 READ 07-30** (see Paper-1 table: robust / robust / floor-censored — no flips anywhere; `artifacts/unfrozen_stress_u234_20260730/`). **U1 READ 08-01 (ONE execution, local): PROTOCOL-ROBUST — the strongest branch.** P-U1a interaction +145.0 [+22.9, +276.5] n=8 PERSISTS unfrozen; P-U1b apt occupancy +10.7 [−7.1, +26.5] — the reward-free null HOLDS under fine-tuning ⇒ hi-occupancy reward-free support is not merely frozen-illegible, it is not an initialization advantage either; headline + null family generalize beyond the frozen protocol. Unfrozen levels: task +225/+221 vs frozen (both sides). Bundle `local_results/unfrozen_u1_20260801_212204/` (32/32, exclusions all foreign waves) → `artifacts/unfrozen_u1_20260801/unfrozen_u1.json`. **U1–U4 family COMPLETE: no flips anywhere — the frozen-protocol scope objection is fully discharged.** Frozen registrations untouched under every branch — flips would have changed generality language only.
- **Pretrained-encoder pixel arm REGISTERED 07-30** (GO'd; the swamping account's lever prediction): `PREREG_pe_pixel_20260730` → `analysis/pe_pixel_read.py` (selfcheck PASS; sha256 34380e0f…) + code: `agent.frozen_enc` (optimizer module filtering), `offline_fit` partial-init w/ counter reset (the skip trap), `AXIS1_INIT_WM` plumbing, `scripts/check_frozen_enc.py` integrity gate (training witness via OFFLINE_FIT_PROGRESS; committed selfcheck; sha256 00acbdff…). Design: 16 stage-2 pixel fits (task-mode full, enc loaded '^enc/' from seed/side-matched fpxpx donors + frozen) + 16 frozen-readout adapts + E4 pass (COLLATE to a NEW csv). P-PE1 graded (CI<2.0 INCLUSION-RESTORED / <19.41 PARTIAL / else NO-RELIEF vs pinned 23.34 baseline); P-PE2 lift vs committed X2 cells; P-PE3 conditional occupancy split. Smoke gate + 16-run integrity sweep with persisted log. 2 adversarial reviews (0 blocking; all 4 MINOR hardening items applied incl. the counter-skip witness + subshell-safe sweep gate). Donors protected in runroot_cleanup (KEEP-PENDING). **08-02 READ (ONE execution): NO-RELIEF — see Paper-1 table row 08-02 + `artifacts/pe_pixel_read_20260802/`.** Bundle `local_results/pe_pixel_20260802_165601/` (manifest 152/152; git head bcc73387 = local commit, all 7 code shas byte-identical, zero drift; integrity sweep 16 PASS 0 QUARANTINE — training witness 500000/500000 counters-reset on all 16, excluding the counter-skip masquerade). Independent anchor: hand-mean of the 16 h0 `rew_nll_in` = 23.0324 ≡ reader point. Interpretation flag (non-verdict, in RECORD): the donor encoder is REWARD-FREE trained, so the null is also readable as a 4th reward-free-transfer null (legibility-coherent) rather than against spectral competition per se — separating those needs a task-trained-donor arm, NEW registration, NOT queued. fpxpx donors' KEEP-PENDING condition now met — release = user cleanup decision (releasing forecloses donor-dependent pe follow-ups). *(Superseded same day: the rde registration below re-holds the donors as its calibration reference.)*
- **Recon-detached pixel arm (rde) REGISTERED 08-02** (pending freeze-commit; user GO conditional on story value — both live branches end the pixel arc decisively): `PREREG_rde_pixel_20260802` (sha 502c36c2…) → `analysis/rde_pixel_read.py` (selfcheck PASS; sha 6a845655…) + code: `agent.recon_grad` (decoder trains on sg(latents) — trunk shaped by rew/con/repval/dyn only; the exact complement of pe), `AXIS1_ARM=rde` case, additive `deter_std` in `stratified_error` (measure moments + collate column; cross-wave-safe: additive only, all pending readers consume by column name), fpxpx cleanup hold transferred to the rde calibration pass. Design: 16 pixel fits `ax1wm_finger_rdepxq1ms<side>_seed<f>` (X2-task-identical except recon_grad False, NO donors) + 16 adapts + fpxpx calibration E4 pass (per-run `--output e4_fingerpx_v1cal` — swamping summaries never touched) + E4 pass to NEW csv. P-RD1 graded as P-PE1 (same pins 23.34/[19.41,27.29]/2.0/1.5/0.6713) **plus the latent-alive witness: deter_std ≥ 0.10×median(fpxpx calibration) on ALL 16 fits, else a numeric fire reads DEGENERATE-MASQUERADE** (registered trap: collapsed trunk → marginal reward head → false sub-2.0 NLL; pe was immune via recon-trained donors, rde is not); P-RD2/P-RD3 = P-PE2/P-PE3 with rde modes. Verdict map: COMPETITION-CONFIRMED / INCLUDED-BUT-USELESS / DEGENERATE-MASQUERADE / PARTIAL-RELIEF / NO-RELIEF(+COLLAPSE-QUALIFIED disclosure) — NO-RELIEF retires the causal competition account (both lever directions failed) and fixes the pixel boundary as a modality-level scope fact. Design-rationale disclosure: task-trained-donor variant REJECTED (near-tautological given P-SW1). Local freeze verification archived (`artifacts/rde_pixel_reg_20260802/`): image→trunk grads 114/7528 → **0.0 exact** under the flag, dec unchanged, reward path bit-identical, moment math exact. Smoke gate + config-audit sweep + calibration ordering registered. **Review DONE 08-02 (ONE reviewer Opus 5, user-approved): 2 BLOCKING / 6 MAJOR / 6 MINOR / 5 NIT — ALL B+M fixed + re-verified**: B1 cleanup-glob deletion hazard (`ax1wm_finger_rd*`/`adapt_ax1rd*` in DELETE-SAFE matched every rde run ⇒ digit-anchored `rd[0-9]*`, machine-verified vs rediag names); **B2 trivial-floor false fire — the fire region [0.6713, 2.0) contained the best-constant-predictor level, so a marginal head (the MOST LIKELY degenerate outcome) read COMPETITION-CONFIRMED and the pre-review selfcheck's Case-1 fixture was itself an instance ⇒ floor made DECISIONAL (fire needs CI entirely < 0.671345852414428) + NOT-BETTER-THAN-CONSTANT registered outcome**; M1 COLLAPSE-QUALIFIED extends to PARTIAL; M2 negative-lift + M3 NO-RELIEF-with-lift disclosure branches; M4 four surviving mutants (inverted fire bound, partial-bound-vs-point, lift-point-vs-CI, calib mean-vs-median) killed by new fixtures — six mutants total CAUGHT on mirror copies; M5 config audit fail-closed + asserted PASS count; M6 smoke rerouted through axis1.sbatch (ARM-case + --export exercised) w/ machine asserts. MINORs: Chan centered-moment combiner (unit-tested; shipped function imported by the harness), harness rev 2 (training=True, all-8-component bit-identical comparison), witness scope + cross-arm-margin disclosures, audit +expl.mode/model_obs; NITs: full-precision pins (23.338872993169502 [19.409267325402837, 27.285688932142705]), member_band ≤, calib --horizons 1, COLLATE-echo check. Final shas: prereg 70d52a30…, reader 864f0556…, stratified_error c382a6e1…, cleanup 922df763… (agent/configs/axis1 unchanged from pre-review). Both selfchecks + harness re-PASS. **Pending [YOU] freeze-commit.**
- **From-scratch baseline anchor REGISTERED 07-30** (GO'd; descriptive class — no decision rules): `PREREG_scratch_anchor_20260730`. 8 jobs, `pilot.sbatch MODE=goal` (plain dmc_proprio, zero code change), RUN_ID `adapt_scratch_finger_seed<k>_ckpt0` → collates as mode `scratch` (no frozen reader consumes it; 10 readers verified mode-pinned). Purpose: absolute-scale anchor for Fig 2a; %-of-anchor reporting never load-bearing. Reviewed (0 blocking; env.sh sourcing + staging checks added). **31 Jul LANDED: 8/8 complete + QC-pass (112 episodes each; config audit 8× from_checkpoint ''/frozen_wm false; bundle `local_results/scratch_anchor_20260731_170618/` manifest-verified). Anchor: AUC50k 113.2 (sd 42.0), AUC100k 147.7 (33.2), AUC125k 174.0 (39.2), final10 319.9 (151.7; seed-8 late collapse to 0.0, descriptive). % anchoring entered `Paper1_ResponseCurves`: W1 +84.0 = 57% of anchor AUC100k, Amend-1 +51.5 = 35%; task_s1 267.9 = 1.81×; reward-free plateau 0.53–0.68× (denominators only, protocols differ — no claim licensed). Fig 2a build blocked on a scores-only pull of per-arm adapt scores.jsonl (none local). **08-01: pull landed (`local_results/fig2a_scores_20260801_122853/`) + Fig 2a BUILT post-U1/25m-reads (`research_notes/figures_paper1_20260730/fig2a_anchoring.png` + data json + archived generating script): unfrozen-protocol arms (frozen-era per-episode scores no longer on scratch; U1/U2 no-flip reads license the substitution, captions must say "unfrozen") + scratch anchor; recomputed scratch AUC100k 147.68 matches the registered anchor exactly; unfrozen task clears scratch 2.3–3.3×, every reward-free arm sits at/below the scratch curve.**
- **High-f_R falling-limb theory predictions FROZEN 07-30** (pre-buffer; GO'd wave, registration to follow): `PREREG_highfr_theory_20260730` — P-HF1 falling limb / P-HF2 support-starvation dissociation (membership intact + diversity_pr lower) / P-HF3 interior maximum, + registered failure modes (monotone rise refutes the breadth-patch trade-off form). Derivation note `Theory_HighFR_Prediction_20260730.md` (disk-only). No high-f_R buffer exists; 25m + U1 not consulted.
- **High-f_R falling-limb wave REGISTERED 08-02** (pending freeze-commit; adjudicates `PREREG_highfr_theory_20260730`): `PREREG_highfr_wave_20260802` (pre-review sha256 acd6d7c9…) + `probing/curate_frew.py` (selfcheck PASS; pre-review sha256 6251c958…) + `analysis/highfr_read.py` (selfcheck PASS; pre-review sha256 876a2bb7…) — post-review finals at the end of this bullet. Design: quad `axis1_finger/q1f` — two curated 200-episode buffers from the frozen episode index (side0 target f_R 0.60, side1 target 0.80; deterministic contiguous-window rule, sides disjoint, NO coverage matching — breadth is measured not controlled; frozen builder materializes; SHORT ⇒ halt + dated amendment); task arm seeds 1–8 per side at the identical default Axis-1 protocol as the v200/v400 cells (16 fits + 16 adapts) + E4 h0 pass (`finger_v1`, GLOB `ax1wm_finger_hfq1f*`). Gates: `spectral_v1_1_20260724`; side1 f∈[0.60,0.85] else the read refuses; side0 f∈[0.55,0.65] + sep≥0.08 else excluded non-fatal. PRIMARY (sole confirmatory, P-HF1): mean[ax1hfq1fs1]−mean[ax1v2q1v200s1] AUC100k two-sample seed-cluster bootstrap B=10K rng0, FIRES iff CI entirely <0. Branches: monotone-rise ⇒ breadth-patch trade-off REFUTED (revision required); straddle+no-diversity-separation ⇒ saturation (uninformative). P-HF2: side1 fits' h0 in-regime rew-NLL CI ≤1.5 member / ≥2.0 inclusion-alternative (pixel-swamping constants) × diversity_pr(side1) < 9.230033291492585 (frozen v200s1). P-HF3 weak: argmax of matched-200-ep grid {v200s0 .054 n6, v200s1 .3232 n12, side0, side1} not at highest f (grid correction disclosed: freeze's matched-volume clause selects v200s0, not v400s1). Comparators = archived volume_repl bundle rows (12/6, hard-asserted). Power (restated post-review against the registered POOLED-12 comparator, mean 187.72 sd 77.18 — the against-interest cohort): MDE ≈69 vs predicted −88..−108. **Review DONE 08-02 (ONE reviewer, Opus 5): 2 BLOCKING / 5 MAJOR / 12 MINOR / 7 NIT — ALL BLOCKINGs+MAJORs fixed + verified**: B1 E4 command dropped invalid HORIZONS=0 (h0 emitted unconditionally by `stratified_error`, 0 crashes it — reviewer machine-verified); B2 side0 made never-fatal end-to-end (qc-fail/incomplete cohort ⇒ EXCLUDED with recorded reason, not abort); M1 three decision-rule mutants (point-vs-CI membership, chain-missing-lo, refute-on-ci[1]) now killed by new selfcheck legs — verified caught on mirror copies; M2 reported-mean≡members invariant added (window-shift + argmin-tie + doubled-tol mutants all caught; tie fixture rebuilt float-exact dyadic); M3 spectral side/domain pin (swap trips); M4 source-composition rider registered (f_R inseparable from collector composition as-realized; manifest mixture+source_l1 disclosed pre-fit; P-HF2 carries mechanism attribution); M5 known P-HF2 comparator level disclosed (archived v200s1 E4 h0 mean 0.827, deep in-band). MINORs applied: pooled-12 power restatement, P-HF3 near-tautology + monotone-decreasing + 2-point-chain disclosures, verdict wording "max-not-at-highest-f", collation+read commands registered verbatim, domain/milestone row filter, spectrum_pr registered descriptive, F_V200S0 provenance comment, halt rule spelled out, floor-valve scope note. Final shas: prereg bbc8f25a…, curate_frew 49b7cf6f…, highfr_read 77def16e…; both selfchecks PASS. **08-02 FREEZE-COMMITTED (39a15880) + registered curation ran on RCC → SHORT (`artifacts/highfr_short_20260802/`, provenance verified — shas match committed finals, head = the freeze commit): side1 OK via registered fallback at f=0.7293 (in-gate, inside the freeze's range); side0 target 0.60 unreachable disjointly (post-removal best window 0.4130 — the pool's high-occ tail is ~200 episodes deep) ⇒ REGISTERED HALT pre-fit, no buffer/fit/outcome exists. Amendment 1 QUEUED for GO: re-target side0 → 0.41 (gate [0.36,0.46]); side1/primary/decision rules unchanged; grid {0.054, 0.3232, ≈0.413, ≈0.729}.** **08-02 AMENDMENT 1 BUILT on user GO (pending freeze-commit): `PREREG_highfr_wave_amend1_20260802` (sha 98e1e122…) + amended `analysis/highfr_read.py` (F_LO_RANGE→[0.36,0.46] + fixtures/range legs; selfcheck PASS; sha dc474355…); `curate_frew.py` UNCHANGED (targets are CLI args). side0 re-registered as MID-GRID point (never load-bearing, as before); separation gate vacuous under amended ranges (retained as belt); deterministic curation outcome knowable in advance (side1 reproduces 0.7293 exactly, side0 lands 0.4130 at dev 0.003). Registered scope note: GRID DENSIFICATION IS OUT — a 0.60-mean window exists only OVERLAPPING side1; adding non-disjoint mid cells (or new collection) is an outcome-contingent NEW registration, warranted only if the falling limb fires AND a peak-location estimate is wanted for the flagship (the freeze claims no peak location). NO fresh reviewer (two-constant amendment on the just-reviewed instrument, per standing reviewer policy).** **08-02 AMENDMENT 1 FREEZE-COMMITTED (be60afbe) + curation OK in-gate (side1 0.7293 fallback reproduced; side0 0.4098 dev 0.0012 — a CLOSER window than the SHORT diagnostic's 0.4130 max-mean window; Amendment 1's ≈0.4130 prediction corrected in Amendment 2, no decision consequence) → BUILD FAILED pre-fit: FileNotFoundError on donor chunk; preflight needed=2046 missing=26 (chunks dated 07-01; scratch purge of never-materialized donor chunks — this wave's tail picks episodes no prior curation ever built) ⇒ AMENDMENT 2 BUILT (pending freeze-commit): `PREREG_highfr_wave_amend2_20260802` (sha bdd27a92…) + `curate_frew.py` `--require_chunks` availability filter (pool = buildable episodes, covering-chunk rule ≡ builder's load_span, eid numbering preserved — index never rewritten; excluded eids + missing files recorded in pairs json; targets/tol/fallback/SHORT/gates unchanged, re-armed on filtered pool; selfcheck REPRODUCES the field failure e2e incl. interval-overlap oracle + FileNotFoundError leg + filtered pair builds clean; PASS; sha 4b14eb47…). Builder + reader UNCHANGED. Record `artifacts/highfr_chunkgap_20260802/`. NO fresh reviewer (per standing policy). Donor-longevity flag: wave becomes donor-independent post-build, but grid-densification option needs the donors — copy off scratch to keep it open.** **→ AMENDMENT 2 FREEZE-COMMITTED (080aa7b2), wave ran clean, READ 08-03 — SATURATION, see Paper 3 table (`artifacts/highfr_read_20260803/`).**
- **Breadth-causal wave GO 08-03** (user; **reviewers pre-approved in advance** for this wave): the missing causal leg of the theory's λ/support input — two 200-ep cells MATCHED on f_R (target 0.32, the v200s1 level) and maximally SEPARATED in diversity_pr. Theory predictions FROZEN pre-feasibility (`PREREG_breadth_theory_20260803`, sha256 eb31b3a2…; committed before any feasibility scan/search/buffer exists): P-BD1 primary hi-div − lo-div > 0 (CI>0) at matched f_R + matched volume; P-BD2 ordering lo < v200s1 ≤ hi (anomaly attribution); P-BD3 membership intact BOTH cells (≤1.5 band, anchor 0.827) w/ registered inclusion-leak alternative; failure modes: informative NULL (λ-input reading loses principal support — anomaly needs non-breadth account), REVERSED (λ-monotonicity refuted), **manipulation-validity HALT gates (min instrument diversity separation + max |Δf| + target band) REQUIRED in the wave registration — the high-f_R mediator lesson made an explicit gate**; composition rider inherited (source_l1 disclosed pre-fit; no pure-diversity coefficient). Feasibility instrument BUILT same day (`probing/curate_frew.py` sha256 931bdbf7…, additive modes, frew legs regression-PASS): `scan-moments` = one chunk pass → per-episode rewarded-frame sufficient statistics EXACTLY in diversity_pr's feature space (obs_keys+symlog+f64), availability rule built in (episodes touching purged chunks recorded+excluded ⇒ buildable by construction); `search-div` = deterministic PR-greedy (objective-aligned own-PR seeds — greedy never evicts its seed, caught by the planted-family selfcheck) + least-damage occupancy-repair exchange, corridor + min_rew drops recorded. Selfcheck PASS incl.: pooled-moment PR ≡ materialized-frames participation_ratio (1e-8) AND ≡ frozen spectral instrument on BUILT buffers (1e-6, the binding path); planted narrow/broad families recovered with forced repair + mixed occ classes; determinism; SHORT; chunk-availability avoidance. Run-name family `bdq1d` (fits `ax1wm_finger_bdq1ds{0,1}_seed*`, adapts `adapt_ax1bdq1ds{0,1}_*`) glob-checked CLEAN vs runroot_cleanup DELETE-SAFE. Ordering: [YOU] freeze-commit → RCC login-node scan+search (value-blind) → IF qualifying pair: wave registration + ONE reviewer (pre-approved) → freeze → build → spectral gates → 16 fits + 16 adapts → E4 → ONE read. Donor-copy URGENT (purge active; this wave needs the pool's chunks). **→ 08-03 THEORY FREEZE COMMITTED (481aced9) pre-feasibility → FEASIBILITY STRONG: 7424 scanned (288 unavailable), pair occ 0.3038/0.3037 (matched 1e-4), search diversity_pr 7.495 / 12.336 (separation 4.841, STRADDLES ref 9.23); feature-space audit: dim 524 ≡ frozen instrument obs_keys incl. dyn/deter → WAVE REGISTERED (`PREREG_breadth_wave_20260803` + frozen `analysis/breadth_read.py`; quad q1d, modes ax1bdq1ds0/1, seeds 1–8, archived v200s1 anchor) → REVIEW DONE (ONE Opus 5): 2 BLOCKING / 4 MAJOR / 8 MINOR / 5 NIT, ALL B+M FIXED + machine-verified. B1 = diversity_pr is 97.7% `dyn/deter` COLLECTOR LATENTS the fitted WM never models (not in obs_space; replay-carry overwritten; probeset builder excludes /-keys) — a fire would have licensed "support breadth" off a WM-invisible quantity ⇒ block-pr decomposition (moments+replay modes, ≡ instrument 1e-9 on built buffers) + OBSERVABLE-SUPPORT GATE proprio-block PR sep ≥ 0.5 (HALT+reader) + pre-freeze block disclosure + "observable support" vocabulary. B2 = rebuild-onto-existing-quad silently doubles a side and f/PR gates are duplication-INVARIANT (the 2-Aug chunkgap incident's rebuild path would have passed all gates, burned 32 jobs, died at the read) ⇒ test-!-e + check-pairs preflight (sha identity + decision-OK + member-chunk re-check) + n_episodes==200 HALT witness + registered recovery rule. M3 wave identity sha-pinned (div_pairs.json committed w/ registration); M4 f-band fixtures de-confounded (message-verified); M5 P-BD3 = gross-collapse screen only (finger reward ≡ regime indicator ⇒ trivial member; historical NLLs 0.75–1.18 never near 2.0; nll_out descriptives); M6 P-BD2 curation-offset rule (+50/+57 prior; FAIL-uninformative/HOLD-strong). 12/12 mutants CAUGHT (clean control PASS); both selfchecks re-PASS. Shas: wave-prereg interim 91722494…, reader 73cc1c24…, curate_frew f58a6513…. NEXT: instrument commit → RCC round-trip (archive pairs+meta into `artifacts/breadth_reg_20260803/`, moments-mode block-pr on the pair, replay-mode on the reference) → fill --expect_sha + fold block numbers → [YOU] freeze-commit → preflight/build/gates/submit.** **→ 08-03 ROUND-TRIP EXECUTED: observable-support pre-check PASSES — pair proprio-block PRs 3.0359/3.7718, separation 0.736 ≥ 0.5; n_rew matched to 0.03% (sample-size confound dead); proprio ordering STRADDLES the reference (3.036 < 3.250 < 3.772) at ~half the full index's relative swing; latent-majority by trace (shares 0.38/0.47) = what the gate guards; **reference replay-mode pr_full 9.230033291492585 BIT-EXACT vs the frozen pin — live real-data validation of block-pr ≡ spectral_measure**. Wave identity pinned: div_pairs.json sha 48032a75… in check-pairs --expect_sha + freeze_shas. FINAL shas: wave-prereg fc24f4e0…, reader 73cc1c24…, curate_frew f58a6513…. READY: [YOU] freeze-commit → preflight → build → gates → 16+16 → E4 → ONE read.**
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
