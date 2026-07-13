# Amendment 2 to PREREG_axis1_corrective_20260711.md — frozen 2026-07-13

New file per this directory's amendment rule. Registered after the P0
corrective read (`artifacts/p0_axis1_corrective_20260713/RESULTS.md`) and
**before any outcome exists** for: reward-free (apt) seeds 9–16 in any
cell, any within-collector (E3v2) run in either arm, or the reward-aware
finger-Q2 seed-extension completion.

## Context and disclosure of known outcomes at freeze

Known when this file is frozen: all seed-1–8 outcomes in both arms (P0
read: reward-free finger Q1 **null** +2.04 [−22.04, +24.16]; registered
2×2 interaction +157.28 [+76.51, +240.68], 8/8 seeds); the reward-aware
seed-9–16 extension outcomes for finger Q1 (+30.71 [−15.51, +75.59], ns)
and all cup cells (read 13 Jul as labeled supplementary). **Unknown:** all
reward-free seed-9–16 outcomes (finger Q1 jobs running, unread), all
within-collector outcomes, cup-Q1 apt 9–16 (not yet run), finger-Q2 aware
completion. Every decision rule below rests on at least one unknown
component; partially-informed components are disclosed inline.

## A. Fully-paired n=16 interaction read (the protocol's registered
## re-pointing of the seed extension)

- **Runs:** `adapt_ax1fq1s{0,1}_finger_seed{9..16}_ckpt500000` (16 jobs,
  submitted 13 Jul, running; same buffers, apt fits, per-run audit rule of
  Amendment 1 §C applies).
- **Primary confirmatory contrast:** the occupancy × reward-supervision
  interaction in finger Q1, fully paired over seeds 1–16:
  per-seed Δ_task − Δ_apt on AUC₁₀₀ₖ, mean with cluster-bootstrap
  percentile 95% CI (B=10,000, seed 0). **Decision on this CI alone.**
- Disclosure: the aware-arm deltas (all 16) and the apt-arm seeds 1–8
  deltas are known; the test is prospective only through the 8 unknown
  apt-arm seed-9–16 outcomes. The fresh-batch interaction (seeds 9–16
  only) is reported alongside as a labeled subsidiary, same machinery.
- Sensitivity suite (robustness only, never decisions): paired t CI, exact
  sign-flip permutation, Wilcoxon, d_z, leave-one-seed-out range.
- Reward-free simple effect at n=16 (pooled 1–16 paired CI) is reported as
  a secondary precision update of the P0 headline null.

## B. E3v2 within-collector replication — BOTH arms (supersedes the
## single-arm framing of Amendment 1 §B; units unchanged)

The P0 decision tree deprioritized E3v2 *as designed* (reward-free
occupancy dose). This amendment re-purposes the registered units for the
new headline: **independent buffer-level replication of the
interaction.** Each within-collector cov-matched high/low-occupancy pair
(Amendment 1 §B units, frozen search criteria, builder `search-within`)
is fitted under both objectives.

- **Naming:** task arm = `ax1w<collector>s<side>` (as reserved in
  Amendment 1); apt arm = `ax1fw<collector>s<side>` (reserved here; `f` =
  reward-free per the ax1/ax1f convention). WM runs
  `ax1wm_<dom>_w…`/`ax1wm_<dom>_fw…`. Both parse under the frozen RUN_RE
  and are excluded from dose-response population models by the frozen
  `--modes` filter. Training seeds nested (1–2), never inferential units.
- **Units:** finger — all feasible collector runs (15 expected from the
  frozen feasibility scan); cup — the random+apt collectors (10 expected;
  p2e scope note stands). Buffer size ≈100 episodes/side as searched.
- **Primary confirmatory (finger):** per-collector interaction
  Δ̂_c = mean over nested seeds of [(s1−s0)_task − (s1−s0)_apt] on
  AUC₁₀₀ₖ; **exact two-sided sign-flip permutation across collector
  runs** on {Δ̂_c}. Secondary: the same in cup. Reported alongside
  (labeled, non-decisional): per-arm simple effects with the same
  permutation machinery; hierarchical collector-level bootstrap CI.
- **Interpretation registered in advance:** interaction replicates on
  independent buffers ⇒ the objective–data-alignment claim gains
  buffer-level causal support (Paper-1 headline + branch-paper seed);
  interaction fails ⇒ the seeds-1–8 interaction is attributed to the
  specific Q1 buffer construction; the branch paper ("hidden curriculum")
  is demoted to a Paper-1 mechanism note. This read also gates the
  causal-support-scaling pilot (authorized only on replication).
- Submission: `submit_all.sh axis1-within-bundles` now requires
  `AXIS1_EXPL_MODE=apt|task` and derives the prefix (`ax1fw`/`ax1w`);
  both arms use identical buffers, seeds, updates (500K), and adapt
  protocol (125K steps, task-mode by design).

## C. Cup Q1 reward-free extension, seeds 9–16 (named secondary)

Motivated by the known seeds-1–8 descriptive signal (+151.04
[−17.75, +327.88], 7/8 positive; final10 descriptive CI excludes zero).
Runs `adapt_ax1fq1s{0,1}_cup_seed{9..16}_ckpt500000` (16 jobs).
**Decision rule:** pooled seeds-1–16 within-seed paired contrast on
AUC₁₀₀ₖ, cluster-bootstrap percentile 95% CI (B=10,000, seed 0); CI
excludes 0 ⇒ "reward-free occupancy benefits cup" is claimed as a named
secondary finding (never pooled with the finger family); otherwise
directional-only. Prospective through the 16 unknown outcomes.

## D. Reward-aware finger-Q2 completion, seeds 9–16 (descriptive only)

The 12 missing `adapt_ax1q2s{0,1}_finger_seed{9..16}` runs complete the
extension table for symmetry. No confirmatory or secondary role; reported
descriptively.

## E. Ordering statement

At freeze: A-runs submitted and running, unread; B-, C-, D-runs not
submitted; `search-within` outputs not yet generated on the cluster; no
`ax1w*`/`ax1fw*` logdir exists (verified by run-id grep over the runroot
snapshots to date).
