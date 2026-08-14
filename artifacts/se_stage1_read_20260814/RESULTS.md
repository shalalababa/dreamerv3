# SE Stage-1 REGISTERED READ — 14 Aug 2026

**Prereg:** `PREREG_nfi_scale_exhibit_20260812.md` v2.2 (frozen fc85fdee)
+ Amendment A1 (per-family SE2 fields, pre-read). **Reader:** the frozen
`uncfield/se_read.py` (built+reviewed #22 pre-read, selfcheck PASS),
executed ONCE on the 8-run bundle; output `se_read.json` here. **This
is the NFI external-validity leg (Paper A); not ICLR-gating.**

## VERDICT — OUTCOME CELL 3 (frozen map §6), N-ONLY QUALIFIER

**P-SE1 FIRES on the noisy-TV channel in 8/8 seeds; all three
duplicates are an informative null at exactly source-parity; P-SE2
(planner ranking) does NOT fire — accounting misprices, ranking does
not concentrate.** Frozen cell-3 wording licensed: "the objective
misprices; the planner does not yet concentrate." Flagship (cell-1)
wording NOT licensed (requires D-discriminator + SE2). Registered
qualifier: the misprice is the STOCHASTIC zero-information channel,
not redundancy.

## Registered numbers (all from `se_read.json`, frozen reader)

**P-SE1 (PRIMARY, cross-seed ≥7/8 + BH q=.05 over 4 channels):**
- **distractor (N): 8/8, exact binomial p=.0039, BH rank-1 (.0039 ≤
  .0125) → FIRES.** Per-run permutation p = .001 (floor) in every run;
  Fisher 3.5e-16. Share-vs-real 0.611 BCa [0.592, 0.625] (null median
  ≈ .32 at 8-vs-17 dims). **θ₁ = 5.26× the source key's own share,
  BCa [5.02, 5.46]** — the deployed disagreement objective assigns a
  by-construction zero-information OU channel 5¼ times the epistemic
  value of the real position key, in its own accounting.
- **dup0 / dup1 / dup2 (D-family): 0/8 each, p=1.0 — at (below) the
  floor.** Per-run p's .90–.99: duplicates carry LESS per-dim
  disagreement than real keys (ensemble members agree on perfectly
  predictable copies). **θ₁: dup0 = 1.0000 [0.9989, 1.0004]** — the
  exact duplicate is priced EXACTLY at par with its source; dup1
  1.000 [1.000, 1.001]; dup2 0.973 [0.968, 0.978].
- **D-ladder discriminator (§2):** shares .0884 / .0885 / .0861 — no
  ε² growth (excess ratio (D2−D0)/(D1−D0) = −62 vs +100 predicted for
  ε²-noise farming); the ε-noise slightly LOWERS normalized share
  (it inflates the per-dim normalizer, not the disagreement).
  **Neither redundancy farming NOR ε-noise farming: duplicates are
  simply not farmed in this substrate.**
- planted_const: diagnostic only as registered; calibration 0.0;
  raw (unfloored) normalizer exactly 0.0 → raw-normalized cal
  undefined, reported None (the §5-registered honesty).

**P-SE2 (PRIMARY, ≥7/8): 1/8 — does NOT fire.** Registered delta
(top-k − all-rollout fire-share) = **−0.0170 BCa [−0.0277, −0.0064]**:
top-ranked imagined rollouts carry slightly LESS fire-channel share.
Per-family (Amendment A1): D-family .252 top vs .263 base; noise .450
vs .456 — both negative. Mechanism-coherent reading (descriptive): OU
disagreement is ≈state-independent, so the fictitious channel adds a
near-constant intrinsic PEDESTAL that inflates the level everywhere
while contributing little rollout-to-rollout ranking variance — the
ranking is driven by real-key disagreement on top of a mispriced
baseline. (This is reporting, not a registered claim.)

**Pooled fire+real share (DESCRIPTIVE only, §5): 0.711 [0.697, 0.723].**

## Validity & provenance (all gates green)

- Bundle `local_results/uncfield_se_stage1_20260813_193324`: 76/76
  sha256 OK; committed manifest identical
  (`manifests/uncfield_se_stage1_20260813_193324.sha256`).
- Runs executed on the Vast per-GPU-lane queue (4 lanes, QUEUE_DONE
  failures=0), NOT RCC. Probe passes executed cluster-side with the
  frozen defaults (n_eval 512 realized in 8/8 — validity floor 256;
  probe seed 0; n_perm 1000 — confirmed by the reader's exact
  p-equality provenance gate on every channel of every run).
- Code provenance: bundle `code/` shas MATCH local frozen instruments
  bit-for-bit (se_probe, se_read, se_mask, planted, main) at bundle
  head d1b797b2 — the post-A1, post-#22 code.
- Fit counters (standing rule): 8/8 last metrics step 497,584–499,568
  of 5e5 (99.5–99.9%, logging-cadence gap only), config steps 5e5 ✓,
  flag OK ✓ — no realized-training truncation (the compcapacity
  failure mode is absent).
- Reader gates: train seeds 10–17 distinct ✓ (seed_check OK); final-
  checkpoint gate OK ×8; calibration acceptance max 0.049 on real keys
  (bar <0.5 — 10× margin); no weak-gate runs; instrument_flag CLEAN.
- Disclosed gap: the probes' stdout (probe-window stream counts) was
  not captured in the bundle logs; realized S=512 is recorded in every
  json and the sampler is the standard stream-stratified
  collect_windows. se_mask NOT run: its registered decision role is
  outcome cell 6 only, and the instrument is CLEAN.

## Registered consequences

1. **D-only arm TRIGGERED (§3, pre-authorized):** the trigger is met
   verbatim — "N-share fires while all D-shares are at floor."
   4 runs (D0/D1/D2 WITHOUT N, seeds 10–13) as the cross-channel
   interference check. Prior note: θ₁=1.0000 makes interference
   unlikely, but the check is registered.
2. **Stage 2 (behavioral) TRIGGERED (§3):** "P-SE1 fires on any non-C
   channel" is met. Gated-vs-ungated at matched observation space,
   2×4 seeds. The gate coordinate/threshold was never pinned in v2.2 —
   requires a data-independent pinning amendment BEFORE submission
   (proposal: gate_key=position, index 0, threshold 0.0 — chosen from
   the smoke inventory only). Given P-SE2's negative delta, Stage 2
   now carries the accounting-vs-behavior dissociation question.
3. **M4 persistence (SECONDARY): NOT EXECUTABLE from this bundle** —
   ckpt_snapshots/ and replay/ were not synced. The weak form ("fire-
   channel share above the null at 100%") is established by the
   primary itself (distractor 8/8 at final). The 25/50% trajectory
   needs the snapshot checkpoints + final replay buffer. **SALVAGE
   (urgent if the Vast instances still exist):** run per run
   `python -m uncfield.se_probe --ckpt <snap25|snap50>
   --output <run>/se_probe_snap<pct>` (the #22-B1 protocol — NEVER
   default --output) or sync ckpt_snapshots/ + replay/; if the
   instances are gone, M4 is registered-unexecutable-as-designed and
   carries as a disclosure (never-delete lesson: the bundle sync list
   must include what registered secondaries need).

## NFI paper (Paper A) wording this read licenses

- "At standard world-model scale, the deployed ensemble-disagreement
  objective (Plan2Explore-style, unmodified) allocates 5.3× a real
  key's epistemic value to a channel carrying zero task information by
  construction, in 8/8 seeds (per-run permutation p=.001, cross-seed
  binomial p=.0039), in its own decoder-projected accounting" — the
  model-internal noisy-TV ACCOUNTING exhibit (the §7 differentiation:
  prior work is behavioral; per-key attribution against
  by-construction-zero channels is the delta).
- "Exact duplicates are priced exactly at par with their source
  (θ₁ = 1.000 [0.999, 1.000]); redundancy is not farmed" — a clean,
  tight informative null that SCOPES the fictitious-information
  mechanism: it requires irreducible stochasticity, not mere
  zero-marginal-information.
- "The planner's imagined-rollout ranking does not concentrate on the
  mispriced channel (top-k share delta −0.017 [−0.028, −0.006])" —
  the accounting/ranking dissociation, consistent with the core
  paper's objective-level (not planner-level) hacking story.
- NOT licensed: any redundancy-farming claim; the cell-1 flagship
  wording; any Stage-2 behavioral claim (not yet run).
