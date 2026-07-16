# TD-MPC2 cross-family replication registration (frozen 2026-07-16)

Registers the external-validity leg of Paper 1 (plan v4; v4 re-review
demand) **before any TD-MPC2 outcome exists**: does the confirmed
occupancy × reward-supervision interaction (W0 n=16, W1 within-collector
replication) hold outside the DreamerV3 model family, in a
decoder-free, implicitly-value-shaped world model?

## Design

- **Model family:** official TD-MPC2 (github.com/nicklashansen/tdmpc2,
  pinned commit `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`), model size 5,
  default hyperparameters except as listed. No architecture changes.
- **Data:** the frozen finger Q1 buffer pair (`axis1_finger/q1`,
  identical npz chunks used by every DreamerV3 arm), bridged by
  `probing/tdmpc2_bridge.py` (canonical dm_control-spec obs
  concatenation; DreamerV3 post-action rows shifted to TD-MPC2's
  prev-action convention; content sha256 in the bridge manifest).
- **2×2 cells:** side ∈ {s0 low-occ, s1 high-occ} × arm ∈
  {**aware** = official objective (consistency + reward + value losses
  shape the representation), **free** = `reward_coef=0, value_coef=0`
  (latent self-consistency only; reward/Q heads receive zero gradient
  and stay at init; parameter counts and update mechanics identical)}.
  The policy update never shapes the encoder in either arm (official
  code detaches zs), so the manipulation is exactly
  reward/value-supervision of the representation.
- **Protocol (mirrors the frozen Axis-1 stages):** offline fit 500K
  gradient updates on the static buffer (`probing/tdmpc2_offline_fit.py`,
  batch 256 × horizon 3), then frozen-representation online adaptation:
  encoder + latent dynamics loaded and FROZEN, fresh reward/Q/policy
  heads trained for 125K env steps (action repeat 1, same embodied DMC
  wrapper family that produced the buffers), MPC planning through the
  frozen dynamics (`probing/tdmpc2_adapt.py`). Eval bursts: 16 episodes
  every 16K steps, written to scores.jsonl in the frozen pipeline's
  format; AUC100k computed by the untouched `analysis/adaptation_auc.py`
  (QC rule unchanged; ≥20 eps ≤100K — the cadence yields 112).
- **Units:** paired seeds 1–8 (fit + adapt RNG; buffers fixed). 2 sides
  × 2 arms × 8 seeds = 32 jobs
  (`SLURM_TIME=12:00:00 ./scripts/submit_all.sh tdmpc2-bundles`).
- **Run naming:** WM `tm2wm_finger_<arm>q1s<side>_seed<k>`, adapt
  `adapt_tm2<arm>q1s<side>_finger_seed<k>_ckpt500000` (parses under the
  frozen RUN_RE; `tm2*` modes are excluded from every DreamerV3
  population model by the frozen `--modes` filters and are never pooled
  with DreamerV3 cells).

## Outcomes and decision rule

Let B_arm(k) = AUC100k(s1) − AUC100k(s0) within seed k. All CIs:
cluster-bootstrap percentile 95% (B=10,000, numpy default_rng seed 0)
over seeds; decisions on the CI alone; the standard sensitivity suite
(paired t, exact sign-flip permutation, Wilcoxon, d_z, LOO) is
robustness-only.

- **PRIMARY (sole confirmatory):** mean over seeds 1–8 of
  [B_aware − B_free]. CI > 0 ⇒ the interaction **replicates
  cross-family** (headline gains the external-validity leg). CI < 0 or
  spanning 0 ⇒ reported as a family-scope limitation of the claim
  (headline language restricted to "reconstruction-based world models"
  per the registered interpretation below).
- **Named secondaries:** (1) aware simple effect B_aware (prediction:
  positive — TD-MPC2's representation is natively reward/value-shaped,
  the analog of the DreamerV3 task arm); (2) free simple effect B_free
  (prediction: null, the third family-form of the reward-free null).
- Descriptive: cell means and adaptation curves; no window other than
  AUC100k is inferential (50K/125K robustness, final10 descriptive —
  frozen stats rules).
- **Never compared numerically across families:** TD-MPC2 and DreamerV3
  AUC values live on different achievable-return scales under frozen
  representations; only the within-family contrast structure is
  compared qualitatively.

## Interpretation map (registered in advance)

- PRIMARY fires with aware+ / free≈0: the objective–data-alignment
  mechanism is family-general — Paper-1 headline keeps its general
  language; TD-MPC2 panel enters the main text.
- PRIMARY fires with free also positive: occupancy helps TD-MPC2 even
  reward-free — family difference in the reward-free null; reported
  as-is (consistency loss may extract more from rare-regime data than
  ELBO reconstruction).
- PRIMARY null with aware+ and free+ (both benefit): occupancy benefit
  is supervision-independent in this family — scope limitation, and a
  pointer that the DreamerV3 interaction is decoder/ELBO-specific.
- PRIMARY null with both ≈0: finger Q1 occupancy does not transfer
  through TD-MPC2's frozen representation at all — protocol-level
  scope note (frozen-rep MPC may bottleneck elsewhere); Gate-F-style
  floor check against the DreamerV3 lo cells before interpreting.

## Disclosure and ordering

Known at freeze: every DreamerV3 outcome to date (P0/W0/W1/W2/W3 reads,
E4 panel, battery), including that the same buffers produce interaction
+84..+99 in the DreamerV3 family — the *direction* of the primary is
therefore an informed prediction, and the test is prospective only
through the unknown TD-MPC2 outcomes. Unknown: every TD-MPC2 number (no
TD-MPC2 fit or adapt has ever been run on any study buffer; no
`tm2*` logdir exists; the pinned checkout was cloned 2026-07-16 and the
pipeline validated only by numpy-level selfchecks —
`tdmpc2_bridge selfcheck` — and syntax checks; no torch environment
exists locally). Required before submission: one cluster smoke fit
(≤2K updates) + one short adapt (≤5K steps) on a scratch copy to
validate plumbing end-to-end; its outputs are protocol validation,
discarded, and carry no evidentiary weight.
