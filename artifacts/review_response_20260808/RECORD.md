# Paper-1 review response — verification + zero-compute analyses (2026-08-08)

Labeled **EXPLORATORY / post-read**. Response to the 7-Aug full adversarial
review (`research_notes/paper1_wm_transfer/reviews/Review_FullRecord_20260807.md`).
**No registered verdict is re-executed**; where a sensitivity would change how a
verdict should be *worded*, that is recorded here and in the record's
correction log, not by editing frozen artifacts. Deterministic: rng(0),
B=10,000 (sims noted per-block). Scripts archived in `scripts/`.

## 1. Verification of the review's [A]/[A✓] items — ALL CONFIRMED

| Item | Status | Note |
|---|---|---|
| **D1 probe-set holdout (BLOCKING)** | **CONFIRMED [V], exact** | `d1_overlap.json`. finger q1s1 **7/60** (4 pilot_p2e + 2 pilot_random + 1 pilot_goal; s0 and both q2 sides 0); cup q1s1 **1/60**, q2s1 **8/60** (lo 0); pixel **8/20 (s0), 5/20 (s1)** of buffer episodes are probe episodes. Mechanism: `stratified_error.cmd_build` has no exclusion; axis1 pools *include* the pilot seed1 runs; the pixel pool *is* the probe set's six pretrain runs. §2.7's "held out by construction" is false. Proprio leak one-sided (hi side); pixel leak both sides. |
| D8 pr_head trace-ratio artifact | CONFIRMED, exact | corr(deter_std, pr_head_input) = **−0.811** (n=48); rde s1 tr_deter 102–129 vs tr_stoch 22 ⇒ pr_head≈pr_deter; s0 opposite; by pr_deter rde s1 (5.7–8.0) > s0 non-crossers (1.5–2.3). Feature-PR RECORD gets a correction addendum. |
| D11 membership band run-level | CONFIRMED, exact | rgo s0 h0 1.29–2.74 (4/16 ≥ 2.0) overlaps sgb s1 1.90–2.52 (3/16 < 2.0). |
| D14 snapshot rows collated qc_pass=1 | CONFIRMED, exact | All four cases reproduced (89.75/n=64 vs 163.20/n=96; 29.53/n=32 vs 80.18; 169.13/n=48 vs 199.57; 58.95/n=80 vs 63.84). Complete-bundle selection was by bundle choice, not a guard. |
| D15 synth contaminated row | CONFIRMED, exact | `adapt_ax1rbf2q1s0_synth_seed104` n_ep_100k=80 (modal 96), qc_pass=1. |
| D16 duplicate-row exposure | CONFIRMED | Canonical csvs carry hundreds of duplicate (mode,seed) pairs across domains/milestones; `optc_amend_read.py:40` `startswith(("ax1v","ax1fv"))` also matches `ax1vgo*`. No reader gates on `n_ep_100k` (appears only in schema constants). |
| D17 Amendment-1 reader | CONFIRMED (structural) | `artifacts/p3_amend1_20260717/amend1_read.py` has no selfcheck; imports `contrast`/`load_b_arm` from `p3_factorial_read`. |
| D18 Goodhart double-symlog | CONFIRMED [V] | Decoder vec heads are `symlog_mse` (`rssm.py:299`); `MSE.pred()` returns the symlog-space mean (`outs.py:135`); `wm_evaluator.py:154` feeds it to `PI.enc`, which symlogs again (`rssm.py:218`). Fixed in code (symexp before re-encoding); **re-collate required before any Goodhart claim ships** (instance list). |
| D19 pixel capacity | CONFIRMED (config) | `pixel_wm` inherits size1m (depth 4 / hidden 64 / deter 512) vs the repo's own `dmc_vision` recipe at depth 64 — orders of magnitude smaller CNN. Pixel floor has two live non-swamping accounts (capacity + D3's ¼ adapt budget). |
| D20 synth `to_target` exposure | CONFIRMED | `to_target` in obs_space; reward is an exact threshold on it; Phase-B″ fit configs have no `model_obs` override (default `.*` ⇒ exposed); the reader's own gate text names the never-run remedy. Synth interaction leg restated as shuffle-collapse exhibit only. |
| D21 TD-MPC2 logging asymmetry | CONFIRMED | `tdmpc2_adapt.py` logs *eval* episodes to scores.jsonl; DreamerV3 logs training episodes; both feed the same AUC reader. Cancels within family; taints cross-family *level* comparisons only. |

## 2. Zero-compute analyses (§3.1) — `zero_compute.json`

1. **Random-policy floor (finger): AUC100k 83.48 ± 15.10** (5 seeds × 96 eps,
   exact). Every reward-free frozen cell sits inside the ±2sd band
   (fq1s0 83.2, fq1s1 79.4, fq2s0 66.4, fq2s1 84.2). The reward-free null is
   *no learning*, not "no benefit".
2. **Reward-free pretraining < random init, CI-solid**: apt-unfrozen 79.2 (s0)
   / 89.9 (s1) vs scratch 147.68 ± 33.22 ⇒ **0.537× / 0.609×**; unpaired
   bootstrap CIs **entirely negative**: s0 [−94.1, −43.8], s1 [−82.0, −33.8].
   Task-unfrozen 333.5 / 489.2 = 2.3–3.3× scratch. Claim licence requires the
   fresh dated registration (drafted, awaiting user commit).
3. **Interaction survives the log transform**: W0 n=16 ratio-of-ratios
   **1.749× [1.241, 2.487]**, perm p = .0090, 13/16 (aware occ-ratio 1.789 vs
   free 0.955). Pixel 2×2 log-null 0.943× [0.603, 1.545] — neither result is a
   scale artifact. Kills the additive-floor objection.
4. d_errin **by-key split: NOT a re-collate** (review detail refuted —
   `errors.npz` stores only nll/rew_nll totals). Moved to instance measure list.
5. Target-angle audit: needs buffer chunks (cluster). Instance list.
6. Replay-context rule-out (D9): record correction — the 12m stratum runs
   zero-initialised context and the interaction still fires +126.5*.
7. **Volume 2×2 completed at n=12** (cells exact: 162.0 / 187.7 / 406.5 /
   467.5). Volume fires at both occupancies (+194.8 paired-6 lo / +279.8
   paired-12 hi, CI>0 both); occupancy fires at neither (+9.2 [−70.3, +96.3]
   at 200ep; hi−lo −61.1 at v400 in registered C1 orientation).
   **Registered C1 (+23.3, n=6) flips sign to −61.1 at n=12** (v400 pair is
   side-inverted: v400s0 = occ .290, v400s1 = .094). §4.14's "occupancy
   saturates below .094" sentence is retired.
8. **rew_nll_out panel** (h0): hi side rgo 0.123 < sh 0.200 < sgb 0.281 <
   vgo 0.290 < **rl 0.376** — same ordering in-regime and out ⇒ D6 confirmed:
   rl is the most confidently-wrong arm out-of-regime, exactly as relocation
   implies; the record must report both columns.

### D1 leak-excluded sensitivity (pixel, computable locally)

Recomputed `rew_nll_in` from per-episode `errors.npz` excluding each side's
own-buffer leaked probe episodes (8 of 120 rows on s0, 5 on s1):

| arm | side | full | leak-excluded | Δ |
|---|---|---|---|---|
| rde | s0 | 11.54 | 11.57 | +0.02 |
| rde | s1 | 3.41 | **4.96** | **+1.55** |
| pe | s0 | 41.29 | 41.47 | +0.18 |
| pe | s1 | 4.78 | **6.67** | **+1.89** |

Pooled read metrics: rde 7.47 → **8.26**; pe 23.03 → **24.07**. Both registered
verdicts unchanged (PARTIAL-RELIEF: still ≪ baseline, ≫ 2.0 bar; NO-RELIEF:
still ≈ baseline). The leak *direction* is as the review predicted (hi-side
membership numbers were flattered). X2's own errors.npz is not bundled locally
— the X2 anchor (23.34) recompute is on the instance list, as is the proprio
leak-excluded pass (finger_v1 probe npz lives on the cluster).

## 3. Statistics re-adjudication — `stats_readjudication.json`

Full battery (percentile / BCa / bootstrap-t / exact sign-flip permutation /
t / Wilcoxon / sign / LOO-fires / TOST ±24.7) over the decisional statistics.
Reproduces the review's Part III essentially exactly:

- **U1 P-U1a**: perm .0859, t .077, sign .727, boot-t ns, LOO-fires 5/8 —
  fires on percentile CI (+BCa) only. **Downgrade to SUGGESTIVE carried.**
- **P3 rgo simple n=8**: perm .094, BCa ns, LOO 2/8 — "own CI>0 / dominant
  single carrier" wording retired; **pooled n=16 rgo simple +56.7 survives
  everything** (perm .0125, BCa/boot-t fire, LOO 16/16) — cite that.
- **Carrier +51.5 n=16**: perm .0409, BCa + boot-t fire, but **fails BH**
  (q=.05) in the assembled 21-primary family (same five survivors as the
  review: W1, W0, Phase-5a occ_phys, P0 2×2, unfrozen-calib level) and fails
  Pocock two-look spending (α_realized ≈ 15.8% by simulation, 20k sims).
  **Estimand split adopted**: the LEVEL contrast (rgo>sgb trunks: +45.0* frozen,
  +267.2 unfrozen 8/8, E4 1.21 vs 2.30 with replication) is the safe claim; the
  interaction-carrier attribution is marginal evidence and must be worded so.
- W0 interaction n=16: solid on every test (perm .0027, sign .021, LOO 16/16).
- Scaling 12m interaction: perm .0156, boot-t ns, sign .070 — solid-ish;
  survives nothing-fancy but keep permutation as primary.
- Synth shuffle collapse: perm .0312, boot-t ns [−361,+15] — carried with the
  D20 restatement.
- TM2 aware simple n=16: perm .070 — suggestive only, as recorded.
- **TOST vs ±24.7**: power-honest nulls are **W0 apt simple (p=.005)** and the
  **orthogonal primary (p=.003)**; stamping (p=.117), vgo (dd CI ±40), U1b,
  three-way are **not** equivalence-licensed ⇒ wording: stamping excludes only
  effects > ~40% of rgo; vgo "inert" → "no differential growth detected".
- **W0 batch effect**: batch1 +157.3 vs batch2 +40.2, Welch p=.048 (aware
  .023; apt .438). Pooled headline must disclose two-batch structure.
  P3 carrier batches homogeneous (p=.958) but sgb-s0 level shifts (p=.002).
- **Pairing buys nothing across arms**: corr(B_task,B_apt) = −0.094 (n=8);
  sd(diff) 127.2 vs 124.1 if independent. Within-arm r: task +0.21, apt −0.18,
  rgo↔sgb s1 −0.27. Report d alongside d_z; cross-wave "paired" contrasts are
  unjustified-as-paired (disclosure).
- **Floor censoring**: pixel-X2 all cells 80.95 (sd 26.6) vs proprio apt 81.30
  (sd 18.2), Welch p=.951 — statistically identical populations; U4's own gate
  applied to proprio apt: +4.68 [−7.9, +17.4] ⇒ would be CENSORED (task
  control +223.1 [+170.8, +275.8] lifts). The asymmetric labels
  ("floor-censored" vs "tight null") are retired in favour of the
  within-experiment-headroom statement; random-init control (wave #15) is the
  discharge experiment.
- **Ladder adjacency**: no adjacent pair distinguishable (full−rgo p=.141,
  rgo−rl .719, …) — "monotone in supervision quality" demoted to descriptive.
- **W1 ICC ≈ 0.012** (σ²_between ≈ 93 vs within 7344): 14 collectors give no
  external validity beyond finger-Q1-one-pipeline; analysis itself sound.
- **Phase-5a**: finger contributes ~zero (β_cov −0.128, β_occ −0.027,
  between-run sd 10.5 vs cup 227.2); cup cov↔occ r = −0.965 ⇒ joint betas are
  an arbitrary rotation (my refit's cup betas differ from the review's under
  an equivalent spec — which is itself the point). Licensed wording: "the
  observational association does not survive intervention", a null, not a
  reversal.
- **Coverage sim** (3k sims): percentile CI covers 86.4% at n=8 on W0 deltas
  (t 93.1%; exact permutation type-I 6.1%) — the "95%" percentile interval is
  an ~86–90% interval at n=8. **Permutation-primary + BCa adopted for every
  future read; historical fired verdicts carry the table above.**

## 4. Consequences ledger (what changes where)

- Record corrections R1–R8 + §2.5/§2.6/§4.21 + D-item disclosures → applied to
  `Paper1_FullRecord_20260807.md` with a dated correction log (writing side).
- Code: `wm_evaluator` symlog fix; `spectral_measure` dyn/-exclusion + block
  split; duplicate-row/qc guards — see repo diff of 2026-08-08.
- Fig 2a rebuilt from `unfrozen_calib_20260725` + U1 arms with correct labels.
- Wave decisions, measure passes, and the negative-transfer claim registration
  → `research_notes/paper1_wm_transfer/reviews/Review_Resolution_20260808.md`.
