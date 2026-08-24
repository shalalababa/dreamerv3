# Algorithm Tier-0 kills — EXPLORATORY (20 Aug 2026)

**Label: EXPLORATORY throughout.** Post-read analyses on existing artifacts
plus two new measurement jobs; no registered estimand consumed, no frozen
instrument modified, nothing here is a confirmatory claim. Candidates and
priors from `research_notes/paper1_wm_transfer/reviews/Algorithm_Ideation_
20260819.md`; slate approved by the user 20 Aug ("build everything for
tier 0").

## 1. Support-gate (ideation §3.2, P prior 0.45) — **KILLED: DROP**

`analysis/tier0_supportgate.py` → `supportgate/supportgate_loo.json`.
Leave-one-out threshold selection on the FB read's own 16 (frac_pos,
zero-shot return) cells, fallback F = the 113.682 floor.

- always-trust 179.79 / never-trust 113.68 / oracle 191.79.
- Gate in-sample 185.07 at τ=0.314 — **pure overfit**: τ sits just above
  the catastrophic cell's frac_pos. **LOO value 177.98 < always-trust** —
  the threshold does not generalize.
- **The gate variable cannot see the failure mode it exists for**: the
  worst cell (1|1, return 0.2) has frac_pos 0.3105 with 8 healthy cells
  gated below it. No frac_pos threshold catches it without also refusing
  half the healthy pool.
- Break-even note: the sweep's break-even (105) inherits the in-sample
  overfit; the honest side-level break-even is ~117.3 (side0's deployed
  mean) — i.e., gating pays only if the fallback beats thin-support
  zero-shot, which the floor-valued fallback does not.

Verdict: **DROP as specified.** A selective-prediction wrapper for FB
zero-shot needs a signal that detects catastrophic z-inference failure;
frac_pos is a side detector, and nothing else in the recorded diagnostics
(z_norm, frac_pos) separates cell 1|1 from its healthy siblings. Recorded
as a negative worth one sentence in the algorithm paper, no build.

## 2. Legibility-gated compute allocation (ideation §3.6, prior 0.35) — **PROCEED (as coarse triage)**

`analysis/tier0_alloc_retro.py` → `alloc_retro/alloc_retro.json`.
27 arm-level (probe rew_nll_in, transfer auc100k) pairs assembled from 8
finger-proprio E4 csvs + the consolidated AUC table + the dose read
(licensed unit: arm-level means; sources cited per pair in the json).

- Spearman(probe, −transfer): **+0.701** full pool; **+0.242** competitive
  subpool (transfer > scratch 147.68, n=13) — the ideation doc's
  within-pool resolution caveat is REAL.
- Scheduler sim (spend-to-95%-of-best): gated **1** vs random median 14
  [IQR 7–21] full pool; gated **1** vs 7 [4–10] competitive; gated **2**
  vs 8 drop-best-probe sensitivity. **SC1+SC2+SC3 all pass.**
- Criteria v2 disclosed in the script header: v1's reward_aware subpool
  turned out vacuous (the E4 column marks the probeset, not the arm);
  replaced with the competitive subpool BEFORE any verdict was adopted.
- Honest limits: the win is front-loading the legible tail, not fine
  ranking (the Q2 task arms — the pool's best transfers — rank 16th–17th
  of 27 by probe); one domain; probe and transfer share arms with the
  waves that built the screen.

Verdict: **PROCEED as the coarse-triage instantiation** (promote-never-
eliminate), pitched as an allocation tool, never a ranker. Next signal:
include in the algorithm paper's retrospective section; a live-sweep demo
only if the paper needs it.

## 3. Misalignment probe family (ideation §3.3, prior ~0.30) — **AWAITING GPU PANEL**

Instruments built + selfchecked:
- `probing/probe_family_dump.py` — GPU stage; per-frame reward-head
  predicted mean + identity NLL on the frozen finger_v1 probeset, E4
  posterior path (h=0), every arm including reward-free (untrained head =
  the control). Verified `.pred()` against agent.py:330.
- `analysis/tier0_probe_family.py` — CPU stage; family {constant, identity,
  affine, isotonic(±), regime_affine, mlp}; held-out Gaussian NLL, even/odd
  episode split; **info_gain = constant − best-recoder** is the metric (the
  constant member absorbs the base-rate offset every arm gets for free).
  Synthetic selfcheck PASS: aligned +8.84 / sign-flipped +1.58 (recodes) /
  noise +0.001 (nothing to recode).
- `scripts/probe_family_panel.sbatch` — ONE job = ONE GPU for the whole
  80-run panel (q1, rlq1, srd0q1, srd1q1, shq1 × 2 sides × seeds 1–8; all
  verified present with ckpts on RCC 20 Aug). Resumable.

**Panel revision 20 Aug (pre-outcome — only the q1 dumps existed):** the
first panel attempt (job 53738652) died on its first fq1 run —
**reward-free (apt-mode) fits carry NO reward head at all** (0 `/rew/`
ckpt entries vs 15 in every reward-bearing arm; lazy ninjax params + the
rew loss never runs under reward_free), so the rhat path would need to
create params inside a pure function, which nj refuses. My dump script's
premise ("the untrained head is the control") was structurally wrong.
Decision: **fq1 leaves the panel as structural-NA** — "cannot recode" is
true by construction there, which is itself a finding (the screen's
premise, a head to probe, does not exist for reward-free fits) but not a
measurable control; **shq1 enters as the second negative control** (a
TRAINED head on shuffled/mis-bound labels — recoding should not recover
true reward, and unlike fq1 that is an empirical claim). srd0/srd1 remain
the primary no-information controls; srd1 retained per the user's
instruction.

Prediction on file before the discriminating arms run: rl recodes toward
the task anchor (info_gain up, driven by sign/regime members); srd0/srd1
and shq1 do not (info_gain ≈ 0). Honest prior ~0.30 (the own-label read
already shows the most favorable recoding failing to clear the house
band; this family is richer, in its own currency).

## 4. Pixel-parity cell (ideation §4.1) — **SPEC READY, needs one GPU job**

One scratch pixel adaptation cell at harness parity, deciding whether the
pixel harness can score any pixel algorithm (and discharging the
Limitation-4 capacity/budget confound either way):

- Base: `pixel_wm` config (agent.model_obs=image), task = finger (the
  pixel study's domain), SCRATCH (no from_checkpoint).
- Parity overrides vs the study's pixel cells: `run.train_ratio: 1024`
  (study ran 256 = ¼ budget) and CNN at the standard depth
  (`.*\.depth: 32`; study inherited size1m's depth 4 ⇒ ~14K encoder,
  ~250× under standard). RSSM stays size1m (the complaint on file is the
  encoder, and changing one lever keeps the cell interpretable).
- Same adaptation protocol/steps as the pixel behavioral cells; scored by
  the same adaptation_auc; compare vs the existing floored pixel scratch
  numbers and the 113.682 floor.
- Either outcome informative: still floored ⇒ pixel boundary needs
  re-litigating before any pixel algorithm claim; clears ⇒ the X2/pe/rde
  behavioral nulls were partly harness facts (disclosure updates).
- Routing: RCC rtx6000 (pixel routing rule) or one 5060 Ti lane;
  train_ratio 1024 ⇒ ~4× the study's pixel cell cost (~6–12 h, one job).

## GPU needs (user to submit)

1. `sbatch scripts/probe_family_panel.sbatch` on RCC (repo + env as for
   e4_measure; ~80 short inference passes, well under the 12 h limit;
   single job, do not split).
2. The pixel-parity cell per §4 (ops chat can wire it into the existing
   pixel submission machinery; exact overrides above).

## §5 Probe-family panel READ (22 Aug) — SCREEN-INSENSITIVE, candidate KILLED

Bundle pfam_panel_20260822_212317 (161/161 sha-verified; 80/80 dumps,
10 arms × 8 seeds; ONE device, Tesla V100 — the reader's
single_device witness true, per the 16-Aug probe-panel rule).

**Result: the family has NO discriminating power on this substrate.**
Per-seed decomposition (probe_family.json):
- EVERY arm × side × seed sits at an identical **+1.25 info gain via
  regime_affine** — q1 (true-reward-trained), rl (polarity-flipped),
  shq1 (binding destroyed), srd0/srd1 (random stamp) all alike, to
  2 decimals. That gain is available from the family's `in_regime`
  INPUT alone (the regime indicator predicts reward by itself); it
  carries no information about the head.
- Head-output channels are dead everywhere: identity is WORSE than
  the constant base-rate member in all 10 cells (+0.06..+1.41 vs
  −2.047), and affine collapses onto constant (−2.0475 ≈ −2.0469) —
  i.e. even the q1 head's predicted mean has ~zero linear relation to
  per-frame true reward on in-regime eval frames.
- The three elevated cells (rl_s1 one seed, shq1_s0 one seed, shq1_s1
  one seed; gain 6.24 each, isotonic) are ONE shared numeric artifact,
  not signal: best_nll ≈ −8.29 = 0.5·log(2π·1e-8), the fit_score
  sigma clamp — a degenerate zero-train-residual isotonic fit hitting
  the floor. Identical magnitude across unrelated arms; arm-nonspecific.

**Verdict**: the registered prediction (rl recodes toward the task
anchor; shq1/srd do not) is UNMEASURABLE at tier-0 — the screen
cannot even separate q1 from random-stamp controls, so it has no
power over the rl question. The §3.3 misalignment-aware-readout
candidate (honest prior ~0.30) is **KILLED as unmeasurable in this
currency/domain** (not refuted): per-frame in-regime reward in finger
is base-rate + regime-indicator dominated, and every recoding the
family can express saturates on that structure. Consistent with the
8-Aug own-label read (own_in_mean 1.690 vs bar 1.5, claimable False)
— two independent instruments now fail to find recoverable
misalignment structure. No build; recorded negative. fq1 remains
structural-NA as registered (no reward head exists in reward-free
fits). Panel cost: one V100-day; the tier-0 fence (exploratory, no
registered α) held throughout.
