# Research Procedure — Reward-Free World-Model Pretraining for Transferable Representations

## A. Experimental Design

**Conditions (4).** C1 No-pretraining (lower bound); C2 Random pretraining; C3 Plan2Explore pretraining (latent-disagreement intrinsic reward); C4 APT pretraining (particle-entropy intrinsic reward).

**Seeds.** 2 per condition. Escalate to 3 only on a triggered basis (see §H).

**Pretraining budget.** 500K environment steps for all DMC proprio runs (C2/C3/C4 pretraining and C1 from-scratch). Held fixed.

**Downstream tasks (3).** Walker `stand`, `walk`, `run` — shared body and dynamics, spanning a behavior gradient. Pretraining is task-free; tasks enter only at Stage 2.

**Adaptation budget `A`.** A single fixed value used identically for every condition × seed × task. Suggested `A ≈ 150K` steps (a fixed fraction of the 500K pretraining budget, consistent with URLB's shorter adaptation phase). Primary metric is **early-adaptation AUC** computed over the first ~half of `A`.

**Adaptation protocol.** Frozen-readout is primary: the pretrained world model (encoder + RSSM) is frozen; only a fresh task head (actor + critic + reward predictor) is trained. A small full-adaptation sub-experiment calibrates this against the standard URLB protocol (§F).

### Experimental matrix

| Condition | Stage 1 — Pretraining (500K, reward-free, task-free) | Stage 2 — per downstream task |
|---|---|---|
| C1 No-pretrain | none | DreamerV3 from scratch, `A` steps |
| C2 Random | WM trained on uniform-random-policy data | Frozen-readout, `A` steps |
| C3 Plan2Explore | WM + exploration policy, latent-disagreement reward | Frozen-readout, `A` steps |
| C4 APT | WM + exploration policy, particle-entropy reward | Frozen-readout, `A` steps |

### Existing assets

- Walker `walk`, 2 seeds @ 500K + 1 seed @ 1.1M (from the UL project, reward-driven from scratch).
- The two 500K walk seeds **are** condition C1 / task=`walk`. Compute early-AUC from their initial `A`-step segment; **no new C1/walk run needed.**
- The 1.1M walk seed is an auxiliary budget-sufficiency reference (see §H, trigger 8).
- Shared infrastructure from the UL project: DreamerV3 on RCC, linear-probe pipeline, held-out trajectory collection with MuJoCo physics-state logging.

---

## B. Phase 0 — Infrastructure (mostly shared with UL project)

**Implement / verify.**
- DreamerV3 (JAX) runs and checkpoints correctly on RCC for DMC proprio Walker.
- Held-out trajectory collector logs observation, action, `qpos`, `qvel` per step (already specified for the UL project).
- Probe pipeline (linear / small-MLP regression on frozen features) is callable on extracted latents.

**Gate before Phase 1.** A 500K Walker run reproduces and checkpoints cleanly; the trajectory collector produces physics-state-aligned trajectories.

---

## C. Phase 1 — Reward-Free Pretraining

### Implement

1. **Random data collection (C2).** Uniform random action sampling; the world model trains on this replay stream with its standard reconstruction + KL objective and no actor.
2. **Plan2Explore intrinsic reward (C3).** An ensemble of one-step latent predictors (small MLPs) predicting the next deterministic posterior feature target (`deter` + categorical probabilities, not the sampled one-hot stochastic latent) from current state + action. Intrinsic reward = scaled ensemble disagreement (variance of predictions). The ensemble trains on bootstrapped replay data; the exploration actor is trained in imagination to maximize discounted intrinsic reward.
3. **APT intrinsic reward (C4).** A particle-based entropy estimate over world-model latents: per-latent reward ∝ `log(c + mean k-NN distance)` in latent space, computed over a batch / latent buffer. The exploration actor maximizes discounted intrinsic reward. (APT's intrinsic reward is typically simpler to implement than P2E's ensemble.)

### Train / Run

- 6 pretraining runs: C2, C3, C4 × 2 seeds, 500K steps each. C1 has no pretraining stage.
- Persist for each run: WM checkpoint (encoder + RSSM + decoder), exploration actor (C3/C4), and the **full pretraining replay buffer** (needed for coverage analysis in §G).

### Test / verify (gate before scaling to all conditions)

- Run **C3 seed 0 first** as the implementation smoke test. Confirm: WM losses converge; ensemble disagreement is non-zero and varies across states; the exploration actor's intrinsic return rises then plateaus as the model's uncertainty is reduced.
- Then run **C4 seed 0**: confirm the entropy reward is non-degenerate and the policy spreads coverage rather than collapsing.
- Only after both smoke tests pass, launch the remaining seeds.

### Branch — see §H triggers 1, 2.

---

## D. Phase 2 — Frozen-Readout Adaptation (answers RQ1 + RQ2 for C1/C2/C3)

### Implement

- **Frozen-readout harness.** Load a pretrained WM; freeze encoder + RSSM parameters (no gradient). Instantiate a fresh **task head**: actor, critic, and a reward predictor (reward is task-specific; pretraining was reward-free, so no pretrained reward head exists). Train only the head, using DreamerV3's imagination procedure rolled through the *frozen* RSSM, on the downstream task reward.
- C1 needs no harness — it is standard DreamerV3 trained from scratch.

### Train / Run

- C1: from-scratch DreamerV3 on `stand` and `run`, 2 seeds each (walk reused from existing assets). Run to ≥`A` steps; running to 500K additionally yields a full-task-performance reference at modest extra cost.
- C2, C3: frozen-readout adaptation, 3 tasks × 2 seeds = 12 runs, `A` steps each.
- Log full learning curves (return vs. environment steps) for every run.

### Test / verify

- Primary metric: **early-adaptation AUC** over the first ~half of `A`, per (condition, seed, task).
- Secondary: final return at `A` steps.
- Sanity check: C2 (random pretraining) should generally fall between C1 and C3; if not, inspect before proceeding.

### Checkpoint result

After Phase 2 the project has a **complete RQ1 + RQ2 result for none / random / P2E** — a presentable study even if everything after this stops.

### Branch — see §H triggers 3, 4, 5.

---

## E. Phase 3 — APT Condition (completes RQ3)

### Train / Run

- C4 frozen-readout adaptation: 3 tasks × 2 seeds = 6 runs, `A` steps each.

### Test / verify

- Compare C4 vs. C3 on early-adaptation AUC across tasks and seeds.
- RQ3 logic: if C3 (uncertainty) and C4 (coverage) yield **similar** adaptation gains, the "active uncertainty" story is weak; if they **differ**, the contrast between their probe and coverage results (§G) indicates which representational property drives transfer.

### Branch — see §H trigger 7.

---

## F. Phase 4 — Protocol-Calibration Sub-Experiment

Purpose: confirm whether frozen-readout ranking agrees with the standard URLB full-adaptation ranking.

### Train / Run

- Full-adaptation runs (continue training the WM during adaptation) for the contrast that matters most: **C3 (P2E)** on task `walk`, 2 seeds. C1/walk full-adaptation already exists (from-scratch is inherently "full"). This is the minimal 2-run version.
- Optional richer version: add C2 and C4 full-adaptation on `walk` (+4 runs) to compare the full 4-way ordering.

### Test / verify

- Compare the C1→C3 gap under frozen-readout vs. under full-adaptation on `walk`.
- **Agreement** → frozen-readout is validated; lean on it for all conclusions.
- **Disagreement in ordering** → report it as a finding ("condition X helps as an initialization but not as a fixed representation"); expand calibration to all four conditions to characterize it.

### Branch — see §H trigger 6.

---

## G. Representation Analysis (run per condition as soon as its models exist)

Two distinct trajectory sources, deliberately separated:

- **Common held-out set** — trajectories from a single fixed reference policy (e.g., uniform random) with physics state logged. Encoded through *each* condition's frozen WM, so all conditions are probed on the *same underlying states*. Isolates representation quality from data distribution.
- **Per-condition pretraining replay** — the data each condition actually collected (already on disk from Phase 1). Used only for coverage.

### Probes and metrics

1. **Linear state probe** — predict simulator state variables (`qpos`/`qvel`-derived: torso height, velocities, joint angles) from frozen RSSM posterior states on the common set. Report R². Include **future-state probes at horizon k** and at least one **derived dynamical quantity** (gait phase, time-to-fall), since current-state probing is near-trivial in the proprio setting.
2. **Linear reward probe** — per downstream task, predict task reward from the frozen latent on the common set. Report R² / accuracy.
3. **Latent-space coverage** — particle entropy or histogram over a low-dimensional projection of the *true* simulator state, computed on each condition's pretraining replay.
4. **World-model open-loop prediction error** — feed initial steps, roll the RSSM prior forward, measure k-step prediction error on the common set.
5. **Latent visualization** — PCA / UMAP of the latent space colored by task variables; qualitative inspection of behavior separability.

### Test / verify

- All metrics reported per (condition, seed) with seed spread shown, never as single point estimates.

---

## H. Synthesis and Decision Points

### Synthesis (the analysis that answers the RQs)

- Build one table/plot keyed by (condition, seed): early-adaptation AUC alongside every representation metric from §G.
- **Correlate adaptation AUC against each representation metric across conditions and seeds.** This correlation — not side-by-side tables — is what converts RQ2 from "restated" to "answered."
- RQ3 verdict combines the C3-vs-C4 adaptation gap with the divergence (or not) of their coverage vs. probe profiles.

### Decision points — trigger → diagnosis → action

1. **Pretraining run diverges or crashes.** → Implementation bug or unstable intrinsic reward. → Fix on the seed-0 smoke test before launching remaining seeds; do not scale a broken pipeline.
2. **P2E ensemble disagreement collapses to ~0 early.** → Ensemble members too correlated. → Check independent initialization / bootstrapping of ensemble members; if disagreement stays degenerate, the intrinsic signal is uninformative — document and treat C3 cautiously.
3. **2-seed adaptation curves overlap within seed noise.** → Underpowered comparison. → Add a 3rd seed for the affected conditions before drawing any conclusion.
4. **All four conditions look indistinguishable (probes and curves).** → Several candidate causes; check in order: (a) 500K pretraining is short vs. URLB's ~2M — extend the most promising condition's pretraining as a diagnostic; (b) downstream task too easy at budget `A` — shorten `A` or weight analysis toward `run`; (c) proprio current-state probe saturated — shift emphasis to future-state and derived-quantity probes; (d) intrinsic rewards not actually changing the collected data — inspect coverage stats of the pretraining replays.
5. **C1 (no pretraining) matches or beats pretrained conditions.** → Either pretraining genuinely doesn't help here (a legitimate, reportable result) or `A` is generous enough to wash out the initialization. → Re-examine with a shorter early-AUC window; if the gap is absent even early, report the null honestly.
6. **Frozen-readout and full-adaptation disagree on ordering (Phase 4).** → Representation-as-fixed-features vs. representation-as-initialization diverge. → Report as a finding; expand the calibration to all four conditions on `walk`.
7. **C3 ≈ C4 on adaptation (RQ3).** → "Active uncertainty" story is weak. → Lean on the coverage vs. uncertainty-reduction decomposition of the *collected data* to explain; this is one of the proposal's anticipated outcomes, not a failure.
8. **The 1.1M-step walk run is much better than the 500K walk runs.** → 500K may undertrain. → At minimum, caveat the budget choice; consider extending pretraining for the headline conditions.
9. **Probe results swing depending on trajectory source.** → Representation quality and data distribution are confounded. → Already mitigated by the common-held-out-set design in §G; if reporting per-condition probe sets too, present them clearly as a separate, distribution-inclusive view.

---

## I. Compute Estimate

Anchored to ~50 SU per 500K-step DMC proprio run; frozen-readout adaptation at budget `A` is roughly half a full run (WM forward passes remain; WM gradient updates are skipped).

| Item | Runs | Approx. SU |
|---|---|---|
| Pretraining (C2/C3/C4 × 2 seeds) | 6 | ~300 |
| C1 from-scratch (`stand`, `run` × 2 seeds; `walk` reused) | 4 | ~200 |
| Frozen-readout adaptation (C2/C3/C4 × 2 seeds × 3 tasks) | 18 | ~360 |
| Protocol calibration (C3 full-adapt on `walk` × 2 seeds; minimal) | 2 | ~80 |
| Probe collection, probing, analysis | — | ~70 |
| **Total** | | **~1000** |

Levers if the budget is tight: drop to 2 downstream tasks (cuts adaptation and from-scratch by ~1/3); keep the minimal 2-run calibration rather than the richer version. Seed count is the main lever on the expensive pretraining line — escalate to 3 seeds only on trigger 3.

---

## J. Execution Order (graceful degradation)

Each phase ends at a coherent, presentable state.

1. Phase 0 — infrastructure verified.
2. Phase 1 — implement P2E + APT; smoke-test on seed 0; run all 6 pretraining jobs.
3. Phase 2 — frozen-readout harness; run C1/C2/C3 adaptation; run probe suite. **→ complete RQ1 + RQ2 (none/random/P2E).**
4. Phase 3 — C4 adaptation + probes. **→ completes RQ3.**
5. Phase 4 — protocol-calibration runs.
6. Phase 5 — synthesis, correlation analysis, write-up.
