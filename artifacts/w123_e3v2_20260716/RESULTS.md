# W1 (E3v2 within-collector) + W2 + W3 registered reads (2026-07-16)

Snapshot `local_results/w1_w2_w3_20260716_163331/runroot_light`. Reads per
`prereg/PREREG_axis1_corrective_amendment2_20260713.md` §B (W1), §C (W2),
§D (W3) — all frozen 13 Jul before any outcome existed. Analysis =
`w123_read.py` (this dir; reuses the W0 machinery verbatim: AUC100k from
frozen `analysis/adaptation_auc.py`, B=10,000 seed-0 percentile
bootstraps, exact sign-flip permutations).

## Audit (Amendment 1 §C rule) — PASS 248/248

- 248 adapt runs, 248 WM fits, 248 job logs. Every apt-arm fit
  (`ax1fw*`, `ax1fq1*`) logs the pure reward-free loss set (finger
  8-component / cup 5-component, no `rew`); every task-arm fit
  (`ax1w*`, `ax1q2*`) logs the same set + `rew` and nothing else (no AC
  losses in the fit stage). Saved WM configs: `agent.expl.mode` = apt/task
  as required; every `offline_fit` consumed exactly the registered pair
  dir+side (verified from the job-log REPLAY line); all adapts task-mode,
  125K steps, ADAPT_DONE. QC 248/248 (≥20 eps ≤100K; all runs have 96).
- Bit-consistency: cup-Q1 apt seeds 1–8 identical to the P0 record
  (16/16 rows and the frozen paired-JSON deltas); finger-Q2 rows overlap
  the 10-Jul record (16/16) and the aware-1–16 table (20/20) exactly.
  The 12 previously-missing finger-Q2 runs are the new §D completions.
- Pair feasibility at build scale (n_episodes=100/side, frozen
  `search-within` criteria): finger 14/15 collectors built (p2e3 SHORT —
  no cov-matched occ-separated pair), cup 9/10 expected (apt2 SHORT;
  p2e collectors infeasible as known from the frozen scan). Drops are
  build-time (pre-outcome) feasibility decisions. Finger docc 0.19–0.36
  at dcov ≤ cov_tol 0.011; cup docc 0.10–0.37 at dcov ≤ 0.021.
  **source_l1 ≡ 0 by construction** — both sides of every pair come from
  a single collector run: the source-diversity confound identified by the
  W0 battery is eliminated by design.

## W1 PRIMARY (finger, §B): the interaction REPLICATES on independent, de-confounded buffers

Per-collector interaction Δ̂_c = mean over nested seeds {1,2} of
[(s1−s0)_task − (s1−s0)_apt] on AUC100k; registered decision = exact
two-sided sign-flip permutation across collector runs.

- **Δ̂ mean +84.0, positive in 14/14 collectors, perm p = 0.00012**
  (the two-sided floor for n=14: the observed configuration is the
  extreme point). Robustness: hierarchical collector-level bootstrap CI
  [+48.2, +124.0]; t CI [+48.6, +119.5]; d_z = 1.37; Wilcoxon p = 0.0001;
  LOO [+73.8, +88.7]. Figure: `e3v2_interaction_forest.png`.
- Positive in every collector family (p2e +62 / apt +101 / random +85) —
  robust across collection policies, at occupancy doses 0.19–0.36.
- Labeled per-arm simple effects (non-decisional): task arm +78.9
  (13/14, perm p = 0.0002); **apt arm −5.1 (8/14, perm p = 0.57)** — the
  reward-free null now stands in its third independent form (P0 seeds
  1–8, W0 pooled n=16, and now within-collector across 14 buffers).
- Descriptive: no dose trend across collectors within the 0.19–0.36
  docc range (Spearman(Δ̂_c, docc) = +0.25, p = 0.39) — consistent with a
  threshold/requirement reading rather than a linear dose effect.

**Registered interpretation fires:** the occupancy × reward-supervision
interaction replicates on independent buffers ⇒ the
objective–data-alignment claim gains buffer-level causal support
(Paper-1 headline + branch-paper seed), and **the
causal-support-scaling pilot is authorized** (was gated on exactly this
replication). Magnitude +84 is consistent with W0's honest pooled +98.7,
not the winner-cursed +157.

## W1 SECONDARY (cup, §B): does NOT replicate — noisy, directionally negative

- Δ̂ mean −140.4, positive 3/9, perm p = 0.31; hier CI [−391, +91].
  Seed-level deltas swing ±1000 (cup adapt is high-variance); nothing
  decisional here.
- Labeled sub-structure (descriptive): the *apt* simple effect is
  positive across collectors (+213, 6/9, perm p = 0.031, labeled
  non-decisional) while the task simple effect is null (+73, p = 0.48) —
  the mirror image of finger. This is exactly what the W0 battery +
  E4 mechanism predict: in cup, occupancy anti-correlates with reward
  density (corr −0.5/−0.7), so reward supervision cannot convert hi-occ
  data into reward-predictive features; any benefit must be (and only
  appears as) reward-free/data-composition. Domain specificity is now a
  coherent mechanism story, not an anomaly.

## W2 (§C, cup-Q1 apt pooled 1–16): CI includes 0 ⇒ directional-only

- Pooled within-seed contrast **+93.1 [−33.0, +217.1]**, 12/16 positive,
  perm p = 0.18 ⇒ per the registered rule, "reward-free occupancy
  benefits cup" is **not claimed**; reported directional-only.
- Fresh batch 9–16: +35.2 [−147, +207] (5/8) — the seeds-1–8 +151 was
  again the optimistic draw. Cell means s0 530 / s1 623 (both far above
  finger's ~80 floor; cup transfers regardless — occupancy composition
  moves it at most modestly).
- Convergence note (descriptive only): W2 (+93, 12/16) and the W1-cup
  apt simple effect (+213, 6/9) point the same direction on independent
  buffers. If the cup reward-free story is wanted as a claim, it needs
  its own confirmatory wave; on current registered tests it is
  directional.

## W3 (§D, finger-Q2 aware completion, descriptive)

Full 16-seed table: pooled −5.4 [−41.9, +28.3], 9/16 positive; fresh
9–16 −9.2; cell means s1 129.6 vs s0 135.0. The 10-Jul finger-Q2 null
stands at n=16 — coverage manipulation at matched occupancy does nothing
in finger, sharpening the contrast with the Q1 occupancy axis.

## Consequences

1. **Paper-1 headline is now fully supported at the registered
   standard:** observational reversal (5a) + corrective reward-free null
   (P0/W0) + confirmed interaction (W0 n=16) + **independent
   de-confounded buffer-level replication (this read)**.
2. **Scaling pilot authorized** by the registered gate.
3. P3 factorial keeps flagship status; its registered prediction (rgo
   carries the benefit) now has two independent supports (E4 reward-NLL
   + this replication). P3 results due shortly.
4. Cup: registered secondary not claimed; the mechanism-consistent
   mirror pattern (apt +, task 0) recorded as descriptive; any cup claim
   requires a new registered wave.
5. Files: `w123_read.py` / `w123_read.json` (full per-collector table,
   seed deltas, all CIs), `auc_w123.csv` (248 rows), `w123_audit.json`
   (per-run audit), `e3v2_interaction_forest.png`.
