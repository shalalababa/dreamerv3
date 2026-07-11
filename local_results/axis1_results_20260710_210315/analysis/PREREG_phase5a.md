# Phase 5a pre-registration — analysis pipeline freeze

**Frozen: 2026-07-06.** Plan v3 §3.5 / power section; runbook v2 Phase 5/5a.
This file fixes the outcome definition, driver definitions (including the
value-sensitive-accuracy recipe), model formulas, and inference procedure
**before any adaptation outcome was observed**. The local snapshot available
at freeze time (`local_results/runroot_snapshot_20260705_101044/runroot_light`)
contains no `adapt_*` run directories; no adapt `scores.jsonl` was read while
writing this spec or the accompanying code. Any later change to this spec is a
**logged deviation** (append to `analysis/DEVIATIONS.md` with date + reason);
the headline analysis is whatever this file says.

## 1. Unit of analysis

Dose-response row = (mode, domain, seed, milestone) with
mode ∈ {p2e, apt, random}, domain ∈ {cup, finger, walker},
seed ∈ {1..5}, milestone ∈ {100, 200, 300, 400, 500}K.
`pretrain_run` = mode×domain×seed (45 groups; the random-effect grouping).
Model-side drivers are measured at the retained snapshot nearest the milestone
(`ckpt_snapshots/nearest.json`; require `abs_error` ≤ 5000 steps, else exclude
and log). Adapt run ids follow `adapt_<mode>_<dom>_seed<k>_ckpt<milestone>`.

## 2. Outcome — early-adaptation AUC

From the adapt run's `scores.jsonl` (one line per episode:
`{"step": <env step at episode end>, "episode/score": <return>}`):

- **AUC_W = mean of `episode/score` over episodes with `step` ≤ W.**
  DMC episodes have fixed length, so this equals the normalized
  area under the score-vs-step curve.
- **Primary W = 100,000.** Secondary ("early") W = 50,000.
  Robustness W = 125,000 (all episodes of the 125K adapt).
- Descriptive: `final10` = mean of the last 10 episodes.
- **QC:** an adapt run enters the analysis only if it has ≥ 20 episodes with
  step ≤ 100,000; otherwise the row is excluded and logged.

Implemented in `analysis/adaptation_auc.py` (frozen with this spec).

## 3. Drivers, measured at each milestone K

### Buffer-side (pretraining replay prefix)

Prefix = the first frames of the pretraining replay corresponding to K env
steps, in chunk-successor-chain order (`probing/probeset.chain_streams`). With
S streams of lengths L_s, take the first `round(K · L_s / ΣL)` frames of each
stream (exact prefix for single-stream runs).

1. **Coverage `cov`** — k-NN particle entropy (knn = 12, logc = 1.0; exact
   self-exclusion, as `probing/gate0_compose.knn_entropy`) on the domain's
   body-state coverage keys (`probing/regimes.py`), standardized by the
   **fixed per-domain reference scaler**: mean/std of the same keys over the
   full Gate-0 `pilot_goal_<dom>_seed1` replay. Estimated on a uniform random
   subsample of min(3000, prefix) frames, numpy seed 0.
2. **Occupancy `occ_phys`** — fraction of prefix frames in R^phys
   (`regimes.py` fn/threshold/direction), computed on the full prefix.
3. **Occupancy `occ_rew`** — fraction of prefix frames with logged reward > 0
   (R^reward; sparse-reward domains, so reward > 0 ⇔ in reward regime).

Implemented in `probing/measure_drivers.py`.

### Model-side (checkpoint), on frozen-policy eval rollouts

Per (run, milestone): `probing/collect.py` fresh eval-mode rollouts from the
milestone snapshot (**40 episodes, max_steps 1000, seed 0**) →
`probing/features.py` (horizons 1 5 20, seed 0) → `probing/probe.py`
(defaults; test split = trailing 30% of episodes). Rollouts are from the
checkpoint's own frozen exploration policy — this is the runbook's named
mechanism; the state-distribution shift across checkpoints is acknowledged and
the Rep-Convergence fixed probe set (frozen latent dumps) is the
distribution-controlled robustness view for geometry.

4. **Forward accuracy `fwd`** — held-out `r2_ridge` at
   (model=rssm, site=imag5, target=state, horizon=5, rf=1). Secondary h ∈ {1, 20}.
5. **Latent geometry `geom`** — held-out `r2_ridge` at
   (model=rssm, site=posterior, target=state, horizon=0, rf=1).
6. **Value-sensitive accuracy `vsa`** — recipe in §4 (k=5 primary).
7. Secondary/descriptive: `retdec` = `r2_ridge`
   (rssm, posterior, return_to_go, h=0, rf=1) — this is variant (d.i) of the
   runbook menu, reported alongside but **not** the headline VSA;
   regime decodability (rssm, posterior, regime/in_regime, h=0).

## 4. Value-sensitive accuracy — frozen recipe (runbook Phase 5 item 3, option d.ii)

Chosen: **k-step rollout error weighted by ‖∂V/∂s‖** — a genuine
forward-model property, most cleanly separated from coverage/decodability.

- **State space:** the domain's body-state observation vector s (coverage
  keys, flattened), standardized by the same fixed goal-pilot reference
  scaler as §3.
- **Critic V̂ (per domain, fit once, frozen):** MLP 256×2 (tanh), Adam
  lr 1e-3, 4000 steps, batch 512, seed 0; input s; target = discounted
  return-to-go (γ = 0.99, computed within episodes, episodes delimited by
  `is_first`) on the domain's Gate-0 **goal pilot replay**
  (`pilot_goal_<dom>_seed1`) — task-labeled, disjoint from every Phase-4+
  training replay (legitimacy per runbook Phase 6b: analysis-time use of
  logged rewards). Episode-level 70/30 train/held-out split, **stratified
  across collection time** (train = episodes with index mod 10 < 7): the goal
  pilot trains online, so a leading/trailing split would confound the critic
  eval with policy drift.
  **Usability gate: held-out critic R² > 0.2**; if a domain's critic fails
  the gate, that domain's VSA driver is replaced by `retdec` (d.i) — this
  substitution is pre-registered here, not a deviation.
- **Weights:** w_t = ‖∇_s V̂(s_{t+k})‖₂ at the **true** state, normalized to
  mean 1 over the evaluation set.
- **Error:** e_t = ‖ŝ_{t+k} − s_{t+k}‖² / dim(s) in the standardized space,
  where ŝ_{t+k} is the ridge readout (probe.py lambda grid, 20% val split)
  from the `imag{k}` features, fit on the leading 70% of episodes and
  evaluated on the trailing 30% (same split convention as probe.py).
- **VSA_k = − Σ w·e / Σ w** over the evaluation episodes (higher = better).
  k = 5 primary; k ∈ {1, 20} secondary. The unweighted −mean(e) is reported
  as a crosscheck (it is the MSE view of driver 4).
- **Domain notes:** walker (the control domain) has no Gate-0 goal pilot and
  no R^phys spec, so: no walker critic is fit (walker VSA = the retdec
  substitution above), occ_phys is undefined (occ_rew only), and walker's
  fixed reference scaler for coverage is the full
  `pretrain_random_walker_seed1` replay.

Implemented in `probing/value_sensitive.py` (`fit-critic`, `measure`).

## 5. Standardization

Within each domain, z-score the outcome and every driver across all
(mode, seed, milestone) rows of that domain's dose-response grid. Effect
sizes are therefore std-AUC per std-driver, comparable across domains.

## 6. Models (statsmodels MixedLM, REML; `analysis/fit_mixed_effects.py`)

- **Primary population:** cup + finger pooled (the decoupled domains).
  **Walker is fit separately** and reported as the collinear control
  contrast, never pooled into the primary estimate.
- **M0 (dose-response):** `auc_z ~ 1 + milestone_c`, random intercept for
  `pretrain_run`; milestone_c = (milestone − 300000)/100000.
- **M1 (per-driver, primary):** `auc_z ~ 1 + driver_z`, random intercept for
  `pretrain_run`; one model per driver. **Primary endpoints: cov and
  occ_phys.** fwd/geom/vsa are the model-level panel.
- **M2 (joint, secondary):** `auc_z ~ cov_z + occ_phys_z + fwd_z + geom_z +
  vsa_z`, random intercept for `pretrain_run`. VIFs reported; if max VIF > 10
  the joint model is reported as unstable and M1 + the intervention phases
  carry the claim (expected on walker — that is the point of the control).
- **Inference:** cluster bootstrap over `pretrain_run` (resample groups with
  replacement, B = 1000, seed 0), percentile 95% CIs on each β. Wald CIs
  reported as a secondary column. No driver enters or leaves the model based
  on results; all pre-named models are reported.
- **Paired contrasts (Phases 6/6b):** within-seed AUC deltas across matched
  cells; mean Δ, cluster-bootstrap CI over seeds (B = 10,000, seed 0), and
  exact two-sided sign test. Primary Phase-6 endpoint: the
  coverage-matched/occupancy-different contrast under R^phys.

## 7. Exclusions

Excluded rows are enumerated in `exclusions.csv` emitted by the pipeline:
missing adapt dir / scores.jsonl, < 20 episodes in the primary window,
snapshot `abs_error` > 5000, probe job returning None (insufficient data).
No other exclusion rule exists.

## 8. Blindness / ordering statement

Written and frozen 2026-07-06, after Phase-4 pretraining completed but with
adapt outcomes unread (not present locally; none inspected on the cluster).
Descriptive plots of adaptation curves may be produced after this freeze;
the model formulas above do not change in response to them. Anything
exploratory is labeled exploratory in the paper and never headlines.
