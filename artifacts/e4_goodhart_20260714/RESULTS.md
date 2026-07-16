# E4 stratified-error + Goodhart-sprint reads (2026-07-14)

Snapshot `local_results/e4_goodhart_20260714_223140/`. E4 read per
`prereg/PREREG_e4_stratified_error_20260714.md` (descriptive only, no
decision role); Goodhart read per
`prereg/PREREG_goodhart_sprint_20260714.md` (registered advance/demote
rule). Probe sets frozen as registered (60 eps/domain from the Gate-0
pilot replays; finger occ 0.091 / rew 0.091; cup occ 0.155 / rew 0.198).

## E4 — the mechanism panel reshapes the interaction story

Coverage this batch: finger Q1 both arms (task seeds 1–8, apt seeds
9–16 — unpaired across arms, identical buffers), finger Q2 + cup Q1/Q2
task arm. Cup apt fits not yet measured (completes the dissociation leg
when the remaining sweep lands). All numbers = paired hi−lo means over
8 seeds unless noted; h ∈ {0,1,5,20} consistent throughout (signature 4
holds; magnitudes attenuate at h20).

1. **Signature 1/2 (relative allocation) holds 8/8 — but as differential
   improvement, not zero-sum.** Finger Q1: hi-occ side has lower
   in-regime error (d_errin ≈ −0.13..−0.24, 8/8 at every horizon) and
   err_diff = (out−in) rises with occupancy (+0.05..+0.14, 8/8 at
   h≤5). The registered "higher err_out" leg FAILS: the hi side is
   better out-of-regime too (d_errout ≈ −0.03..−0.19), just less so.
   Same qualitative pattern in cup Q1 (d_errin −0.03, 8/8; err_diff
   +0.02, 8/8). Finger Q2 (low-cov side): errors higher everywhere
   (0/8 neg) — coverage loss hurts globally, as expected.
2. **Arm-invariance — the pivotal negative result.** The in-regime
   modeling advantage of hi-occ buffers is the SAME in the reward-free
   arm (apt d_errin −0.13..−0.24) as in the reward-aware arm (task
   −0.13..−0.21). The apt fits model the regime just as much better —
   yet transfer nothing (W0 tight null −3.7). ⇒ *Better regime
   dynamics/observation modeling is not the carrier of the confirmed
   interaction.*
3. **Reward predictability of the representation tracks the behavioral
   interaction across cells.** Task-arm reward-head NLL (same head
   capacity, trained on the same data): finger Q1 hi 0.191 vs lo 0.242
   (better on hi at every horizon) — exactly the cell where the
   interaction lives; cup Q1 flat (+0.005) — where the interaction is
   null; cup Q2 better on the reward-dense side (−0.10) — it follows
   reward-frame density, not occupancy, where the two dissociate
   (battery corr −0.5/−0.7 in cup vs +1.0 in finger).
4. **Signature 3's obs-decoder version fails informatively:** the task
   arm's observation-decoding advantage over apt concentrates on
   UNREWARDED frames (allocation gap +0.006..+0.05) — reward
   supervision does not buy better obs decoding of reward states.
   Combined with (2)–(3): the interaction's mechanism is
   **reward-predictability of the transferred features (enc/dyn), not
   observation or dynamics accuracy anywhere.** This sharpens the P3
   factorial prediction: the reward-head gradient arm (`rgo`) should
   carry the benefit, and E4's reward-head NLL on the factorial fits is
   the direct diagnostic (sgb-arm fits should show apt-like reward NLL).

Caveats: descriptive throughout (registered as such); task-vs-apt
contrasts unpaired (seed batches 1–8 vs 9–16; buffers identical, seeds
affect only fit RNG); cup-apt measure pass outstanding.

## Goodhart sprint — registered verdict: neither advance trigger fires; continue-but-weaker

Pool = 63 finger adapt checkpoints, evaluator
`adapt_ax1q1s1_finger_seed1` (excluded from pool), 20 starts × horizon
64 (`goodhart_pool_metrics.json`):

- Pool Spearman(M-score, real) = **+0.153** (p=0.23); inversion rate
  0.44; above-floor subpool (real ≥ 50, n=41): Spearman 0.32 (p=.04).
- **Top-1/3/5 regret = 0.0** — the evaluator's top pick IS the true
  best policy (764.6).
- Registered triggers: (1) regret ≥10% despite Spearman ≥0.6 — NO;
  (2) top-quintile inversion ≥2× pool — NO (0.59 vs 0.44 = 1.34×).

**Fragility note (descriptive, the program's teaser):** M's rank-2
policy has real return **0.0** (`ax1q1s1_seed3`, M-score 10.17 vs the
winner's 10.59). Leave-one-out: with the true best removed from the
pool, the evaluator selects that zero-return policy — 100% regret
(565.1 foregone). Zero top-k regret is carried entirely by one policy;
selection at the top is extremely high-variance even when it happens to
succeed.

**Registered path forward:** the pool correlation (0.15) is far below
the 0.6 the advance trigger presumes, so per the "continue-but-weaker"
branch the next step is a stronger evaluator before any Goodhart claim:
candidates = a longer-trained/online model (p2elong or goal-pilot
final), or a small evaluator ensemble with disagreement flags (the
rank-2 failure is exactly what an ensemble + targeted real queries
should catch — the certification protocol of the eventual paper). One
job re-run; no design change.
