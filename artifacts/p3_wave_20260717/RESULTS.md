# P3 factorial + mechanism-wave reads (2026-07-17)

Snapshots `local_results/p3_20260717_001302/` (80 factorial runs) and
`local_results/cloud_roadmap_20260716_200152/` (E4 factorial + cup-apt
CSVs, lo-side battery, Goodhart p2elong). P3 read executed by the
**pre-frozen** `analysis/p3_factorial_read.py` (committed before any
factorial outcome existed), unmodified, per
`prereg/PREREG_p3_gradient_path_20260714.md`.

## Audit

- P3: 80/80 QC (96 eps ≤100K each); frozen-script audit PASS 80/80 on
  the registered rule (saved WM config flags exactly match the arm
  table: sgb task/F/T/F, rgo task/T/T/F, vgo task/F/T/T, sh & rl
  task/T/T/T). Transform arms' replay paths verified from all 32 fit
  logs (`axis1_finger/q1_sh|q1_rl/side{0,1}` as registered).
  Gradient-arm fit stdout was not retained by the cloud workers
  (shard logs only) — the config-flag audit is the registered per-run
  rule and is satisfied; the 34 retained adapt logs all show `rew` in
  fit lines and no AC keys.
- Cup-apt E4 coverage: 32/48 targets measured (fq1 12/16 paired seeds;
  fq2 4/8 — `ax1wm_cup_fq2s1_seed{7,8}` have no ckpt; descriptive leg,
  no decision role).

## P3 registered read — primary does NOT fire; binding necessity DOES

B_arm = within-seed AUC100k(s1) − AUC100k(s0), seeds 1–8, B=10K seed-0
CIs, decisions on CI alone:

| contrast | mean | 95% CI | verdict |
|---|---|---|---|
| **PRIMARY** B_rgo − B_sgb | +50.2 | [−12.8, +111.2] | **not confirmed** (5/8, perm p=.195) |
| S1 B_vgo − B_sgb | −15.8 | [−56.8, +27.4] | null |
| **S2** B_sh − B_full | **−127.9** | [−201.0, −51.5] | **fires** (1/8 pos, perm p=.023) |
| S3 B_rl − B_full | −106.5 | [−187.9, −30.3] | fires (perm p=.039) |
| S4 placebo B_sgb − B_apt | +12.7 | [−17.4, +45.4] | ≈0 ✓ design validates |

**Registered branch: "PRIMARY and S1 both null ⇒ the interaction needs
BOTH gradient paths (or their interplay)."** With S2 firing: the
benefit requires frame-level reward↔state binding — within-episode
shuffling (which preserves episode-level reward density, the battery's
+1.0 correlate) collapses it. The "density-not-binding" strange-claim
branch is ruled out.

Descriptive simple effects (labeled): rgo **+65.0 [+4.2, +120.9]**
(own CI excludes 0), rl +52.8 [+14.2, +102.9], sh +31.4 ns, sgb +14.8
ns, vgo −1.0; known: full(1–8) +159.3 (winner-cursed; honest pooled
+98.7), apt +2.0. Ordering full > rgo > rl > sh > sgb > apt is monotone
in reward-supervision quality. Additivity: full − (rgo+vgo−sgb) =
+110 with a wide CI [−6.3, +204.5] — superadditivity suggestive, not
established. Caveats: n=8 throughout; S2/S3/additivity use the 1–8
full-arm record as registered+disclosed (the shuffle collapse holds
even against the honest +98.7: 31 vs 99).

## E4 factorial pass (registered descriptive) — the rep-level diagnostic lands

In-regime reward-head NLL, h=0, hi side (all task arms train the head;
only gradient flow differs):

full **1.02** < rgo **1.21** ≪ sh 2.03 < sgb 2.30 ≈ vgo 2.31 < rl 3.43

- The reward-gradient arms (full, rgo) are the only ones whose
  *representation* supports reward prediction — sgb ≈ vgo confirms head
  training alone does nothing to the rep (the 14-Jul E4 prediction,
  confirmed at the representation level).
- **d_errin is arm-invariant across all five arms** (−0.11..−0.24, 8/8
  everywhere, h0–h20) — regime/obs modeling is still not the carrier,
  now shown under five gradient wirings.
- **rl refines the mechanism:** worst true-reward NLL (3.43 — trained
  to put reward elsewhere) yet retains behavioral benefit +52.8. So
  reward-predictability of the *true* labels is not the whole carrier
  either: reward-linked gradient shaping helps per se, and correct
  binding is required only for the full effect.
- Code note: the hi−lo reward-NLL *delta* is negative in every task arm
  (more reward frames train the head better regardless of wiring) —
  the discriminating quantity is the LEVEL, not the delta.

## Cup-apt E4 (dissociation leg complete)

cup fq1 d_errin −0.035 (12/12), fq2 −0.005 (4/4) — matches the cup
task-arm values from 14 Jul. Arm-invariance of the regime-modeling
advantage now holds in **all four domain × arm cells**, including the
domain where the behavioral interaction is null.

## Lo-side battery (closes the 13-Jul request)

- All four lo sides are **mono-source** (n=1, eff 1.0) — the diversity
  confound of the original Axis-1 buffers is now documented on both
  sides; it was already *defeated* by W1's within-collector design.
- finger occ↔reward corr **+1.00 on lo sides too** — occupancy ≡ reward
  density is a task property, not a side artifact.
- cup q1 lo: occ 0.013 with reward-frame frac **0.222** — the low-occ
  side is reward-rich; the cup occ⊥reward dissociation documented
  cross-side.

## Goodhart p2elong rerun — neither trigger fires again; ensemble-or-demote

Pool n=95, evaluator p2elong (496K online model):
Spearman +0.209 (p=.042), inversion rate 0.428; **top-1 regret 599.7 =
78.4%** (picked policy real 164.9 = 66th percentile; true best ranks
49/95 under M); top-quintile inversions 0.661 = **1.54×** pool.

- Trigger 1: NO (regret huge but Spearman 0.21 < 0.6). Trigger 2: NO
  (1.54× < 2×).
- **Baseline check: random-pick regret is 81.3% — selection by this
  evaluator ≈ lottery.** Per the registered demote criterion, regret is
  consistent with (not amplified beyond) the near-zero correlation.
- Two-evaluator pattern realizes the 14-Jul LOO teaser: at r≈0.15–0.21,
  top-1 outcomes are draws (evaluator 1: lucky zero regret; evaluator
  2: 78%). No fair Goodhart test exists in this pool without a
  materially better evaluator.
- **Path (registered options): one final attempt with the small
  evaluator ensemble (named 14 Jul); if pool correlation still cannot
  support trigger 1's premise, demote per the audit's kill rule.**

## Consequences

1. **Paper-1 mechanism section (as registered, reported as-is):** the
   interaction is a joint gradient-path effect — reward path is the
   dominant single carrier (descriptive CI > 0; rep-level reward
   predictability tracks it) but does not confirmably exceed placebo at
   n=8; the value path alone is inert; frame-level label binding is
   necessary (registered confirmatory S2). Regime-modeling accuracy is
   invariant across 5 arms × 2 domains × 2 objectives.
2. **Registered next step (Amendment 1 to the P3 prereg, frozen
   2026-07-17 before any extension outcome):** rgo+sgb seed-9–16
   extension (32 jobs) → pooled n=16 primary re-test, exactly the W0
   playbook. `prereg/PREREG_p3_amendment1_20260717.md`.
3. Goodhart: ensemble run or demote — one job, then the fork closes.
4. Files: `p3_read.json`, `auc_p3.csv`, `e4_factorial_finger_v1.csv`,
   `e4_cup_apt_v1.csv`, `battery_lo/`, `goodhart_p2elong_pool_metrics.json`.
