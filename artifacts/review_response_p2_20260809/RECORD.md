# Paper-2 review resolution — verification record (2026-08-09)

Response to BOTH Paper-2 reviews:
`research_notes/paper2_evpi/reviews/Review_FullRecord_20260807.md`
(internal, 12 lenses) and `.../GPT_Diagnosis_20260809.md` (external,
independent). LABELED POST-READ ANALYSES throughout — every registered
read was executed long before either review; nothing here re-executes a
frozen reader or touches a verdict file. Scripts in `scripts/`, outputs
in this directory.

## Verification outcomes (my own reproductions, this machine)

### Code claims — ALL CONFIRMED [V]

| claim | verified at |
|---|---|
| G is ONE stochastic rollout (no averaging; `--rollouts` feeds D0 signals) | `d0/oracle_labels.py` rollout_return |
| `policy(..., mode)` never reads `mode`; follower samples | `dreamerv3/agent.py:195-212` |
| D1: TM2 `m_now` = imported plug-in argmax over `model.pi` samples scored by `model.Q`; the MPPI planner appears ONLY as `follower_act` (continuation + base trajectory) | `probing/tdmpc2_oracle_labels.py` `_cand_q`/`follower_act`/label loop |
| D2: repair features — cols 4–7 `np.full` constants; belief latent `np.tile`d across candidates; cols 2–3 argmax-equivalent to col 0 ⇒ within-state argmax sees qm, qs, action only (4 DoF) | `d0/train_consumer_model.py` consumer_features |
| D3: `plugin_choice = argmax(qfull.mean(0))`; CONSUMER_REGEX = `^(rew|con|valens\d+)/` explicitly excludes the deployed critic `val` | `d0/oracle_labels.py` |
| D3b: dmc config has NO `use_seed` (only dmlab/synth do) ⇒ `suite.load(domain, task)` unseeded (OS entropy) | `dreamerv3/main.py` make_env + `embodied/envs/dmc.py` + configs.yaml |
| D10: `label_run`'s `rng` arg appears only in the signature | AST scan |
| D3c: stdout redaction only for versions ending `_xc1/_xc2/_cm1`; default path prints `mean delta_real` (≡ per-pass achieved) | `d0/oracle_labels.py:570-582` |

### Data claims — ALL REPRODUCED (`p2_verify_A.json`)

- **A1 primaries EXACT**: pooled opportunity 1.24703125 ≡ registered;
  gap 1.29304688 ≡ registered (validates cohort + arithmetic).
- **A2 selection columns**: max−mean = **1.268** (opportunity ≡ 98.3%
  of it); m_now−mean = **+0.021** [−0.049, +0.082]; m_real−mean =
  **−0.025** [−0.088, +0.030]; m_now mean oracle rank **4.508** vs
  chance 4.5. Matches both reviews.
- **A3 permutation null** (candidate→slot shuffle within state, rng 0):
  null opportunity **+1.233 [+0.700, +1.865]** vs observed +1.247 — the
  exchangeability null REPRODUCES the headline; null achieved −0.007.
- **A4 state floors**: 84.4% of states have ALL candidates identical in
  G; 90.6% have opportunity exactly 0; P(opp>0) = 9.4%; top 1% of
  states carry **51.4%** of total opportunity.
- **A5 non-floor (opp>0) cells** (41 cells / 27 runs): achieved
  **+3.39 [+1.40, +6.02]** (CI>0 — the consumer DOES harvest where
  opportunity exists); m_now-vs-random **−2.76 [−3.88, −1.64]** (the
  plug-in baseline is significantly WORSE than a random candidate).
  (Review's variant −2.97/+2.97 used a slightly different cell pooling;
  direction and significance identical.)
- **A6 within-state dissociation** (non-degenerate states, cell-pooled,
  27 run clusters): Spearman(q-mean, g_all) = **−0.017 [−0.048, +0.017]
  null**; Spearman(real_scores, g_all) = **+0.023 [+0.001, +0.047]
  CI>0**. The value ensemble cannot rank its own candidates; the
  one-real-step probe score can (a little).
- **A7 g_all_boot**: opportunity **+2.881 [+1.924, +3.959]**, gap
  +2.951 — adding the bootstrap term's extra noise MORE THAN DOUBLES
  measured "opportunity" (selection-bias signature; no information
  account predicts it).
- **A8 variogram**: σ(δ) flat — 4.74 at δ≤0.005, 4.57 at 0.05–0.3,
  4.32 at δ>1.2; σ(0)=0 exactly (CRN perfect on identical actions).
  Returns do not depend on the action taken.

### THE CROSS-FIT — REPRODUCED (`p2_verify_B_crossfit.json`)

Select oracle argmax on the committed R3 late pass, evaluate that slot
on the repair wave's independent re-pass (same 32 cells, identical
(episode, step) index):

| | within-pass | cross-fitted |
|---|---|---|
| opportunity | **+1.3014 [+0.626, +2.149]** (≡ registered opp8) | **−0.105 [−0.290, +0.066]** |
| (baseline m_now_A variant) | — | −0.060 [−0.204, +0.069] |
| gap | +1.340 | −0.066 [−0.295, +0.112] |

**Low-drift half** (16 cells below median candidate drift; drift down
to 0.019 action units): within +0.434, cross-fitted **+0.011 [−0.084,
+0.093]** — the collapse is NOT a drift artifact.
⇒ **P-R3a and P-R3b do not survive out-of-sample selection.** The
within-pass "opportunity" is the selection bias of a max over one noisy
realization per candidate.

### Stage-1A (verify-before-acting items) — CONFIRMED

- **[V] Comparator flip**: cup seed1 final ckpt (anchor h=5) — RAW
  pcorr(D,E|ρ) **+0.272** / pcorr(D,ρ|E) **+0.434** (TRACKS_DENSITY);
  RESIDUAL **+0.267** / **−0.166** (TRACKS_ERROR). The error term is
  unchanged by residualization; the verdict flip is entirely the
  comparator term over-correcting into negative territory. Honest
  statistic = pcorr(D_resid, E|ρ) alone.
- **[V-code] M1**: `density_residualize` runs ONCE (latent_uq_analysis
  `:528`) before `row_stats`; the bootstrap resamples a fixed residual
  — isotonic fitting variance never enters the SE. (1.98× understatement
  magnitude remains [A], simulation-based.)
- **[V-structural] M4**: cross mode sets D = normalized member-variance
  of decoded predictions and E = member-mean error on the same targets
  ⇒ E = D + squared-bias ≥ D algebraically; the four cross-cell
  TRACKS_ERROR verdicts are an identity, not a finding.

## Consequences (registered in the record, dated 2026-08-09)

1. **P-R3a / P-RRa / P-TM2a (opportunity) and P-R3b / P-RRb / P-TM2b
   (gap): CONSTRUCT-INVALID as competence estimands** — the registered
   arithmetic is correct and reproduced, but the quantity is the
   expected max of M noisy realizations, reproduced by an
   exchangeability null and eliminated by cross-fitting. (TM2 partial
   survival: bias-corrected +0.34 [+0.19, +0.51] per review — real
   action-dependence, but crosses the registered 0.2 bar and rides
   ~2.3 distinct candidates/state.)
2. **TM2 planner claim struck** (D1): the planner was never the
   consumer; the ICML venue condition (TD-MPC2 replication) is NOT met.
3. **Repair verdict re-scoped** (D2): what was tested is a 4-DoF tilt
   of the plug-in score; NOT-REPAIRABLE-FROM-OBSERVABLES is not
   licensed (the belief latent never entered the choice).
4. **Cross-invocation lesson corrected** (D3b): the labeler env is
   unseeded — no cross-invocation state pairing is possible by
   construction; the bf16-drift story is superseded (quarantine
   DECISIONS stand).
5. **Value-blindness disclosure** (D3c): main-wave stdout printed
   `mean delta_real` per pass before the ONE reads.
6. **P-R3c → NOT ESTIMABLE** (84% exact-zero states manufacture the
   tightness); ladder NO-PER-STATE-SIGNATURE → UNDERPOWERED (target
   reliability 0.086; bar 0.2 ≈ 69% of the √r ceiling); XC2 asymmetry →
   dilution artifact (both sides negative under the same lens);
   Stage-1A wording → comparator-flip; cross-cell TRACKS_ERROR →
   algebraic identity; probe-distribution column → mislabel (M5).
7. **The identified positive results** (labeled, post-review): non-floor
   achieved CI>0 in all three waves; within-state dissociation (probe
   score ranks, value ensemble does not); both order-statistic-immune.
8. **Must-cite additions**: Berseth (arXiv:2508.01329); Brennan,
   Kharroubi, O'Hagan & Chilcott (Med. Decis. Making 27(4), 2007);
   Smith & Winkler (2006, optimizer's curse); Fenwick, Claxton &
   Sculpher (MDM 28(1), 2008, value of implementation); Muppidi et al.
   (2026, trainable planning-budget gate).

## Review credibility

Every claim checked (8 code, 9 data-arithmetic, 3 Stage-1A) confirmed —
100%, consistent with the Paper-1 review. The two reviews are fully
independent (different provenance, two days apart) and agree wherever
they overlap; neither contradicts the other anywhere material.
