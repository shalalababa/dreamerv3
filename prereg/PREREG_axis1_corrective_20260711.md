# PREREG: Axis-1 reward-free corrective refit — frozen 2026-07-11

**Status: FROZEN at commit time. Amendments go in new dated files in this
directory, never edits to this one.** This directory (`prereg/`) is tracked
in git precisely so registration timestamps are provable; the narrative
plan documents live in gitignored `research_notes/` and are NOT the
registration record.

## 1. Why this exists (discovered implementation deviation)

The 2026-07-11 editorial code audit found that all 64 Axis-1 offline WM
fits ran `probing.offline_fit --configs dmc_proprio` with the default
`expl.mode: task` (`scripts/axis1.sbatch`; confirmed in every saved
`wm_audit/*/config.yaml`). Under `mode: task` the agent sets
`reward_free=False` and trains `losses['rew']` on logged task rewards with
`reward_grad: true` (gradients into the representation) plus
`repval_loss/repval_grad` (dreamerv3/agent.py). The Phase-4 **online**
pretrains ran `expl.mode ∈ {p2e, apt, random}` ⇒ `reward_free=True` ⇒
neither loss existed. In finger the physical regime is definitionally the
reward condition (probing/regimes.py), so reward/value supervision was
**differential by intervention side** (high-occ ≈ 0.32 reward-bearing frame
fraction vs ≈ 0.05).

Consequently the 10-Jul Axis-1 read (`artifacts/phase6_axis1_20260710/`)
is **reclassified as the reward-aware arm**: the finger Q1 +159 effect
conflates (a) physical-regime experience, (b) positive reward labels,
(c) reward/value-supervised representation learning. It does not by itself
support "reward-free occupancy rescues transfer". Logged as a true
deviation in `analysis/DEVIATIONS.md` (2026-07-11).

## 2. Corrective runs (registered before any is launched)

- **Fits:** refit ALL four existing buffer pairs (cup/finger × Q1/Q2, the
  identical materialized buffers — the buffers are not contaminated, only
  the fitting objective was) with `--expl.mode apt` appended to the same
  `offline_fit` invocation. Rationale for `apt` (over `random`/`p2e`):
  it reproduces exactly the reward-free replay-loss set of the Phase-4
  online pretrains — decoder + dynamics/representation + continuation —
  with no additional replay-side representation-shaping loss (`p2e`
  additionally trains the disagreement ensemble on non-stop-gradient
  features; `random` drops the continuation head). Task reward enters no
  loss term (`agent.py: reward_free=True` path). APT's intrinsic reward
  affects only the (discarded) imagination actor-critic.
- **Everything else identical to the 10-Jul arm:** 500,000 updates,
  `dmc_proprio`, same seeds **1–8**, same frozen-readout adapt stage
  (1.25e5 steps; the adapt stage was and remains task-mode by design —
  adaptation is supposed to see reward).
- **Run naming:** mode strings `ax1fq1s0/1`, `ax1fq2s0/1` (f = reward-free
  fit), run ids `adapt_ax1f<q>s<side>_<dom>_seed<k>_ckpt500000`, WM runs
  `ax1wm_<dom>_f<q>s<side>_seed<k>`. These parse under the frozen `RUN_RE`
  and are excluded from all dose-response population models by the frozen
  `--modes` filter. 64 jobs.
- **Config guard:** `scripts/axis1.sbatch` now *requires*
  `AXIS1_EXPL_MODE` (no default). `apt` = this protocol;
  `task` reproduces the reward-aware arm. The flag is recorded in each run's
  saved config.yaml (`expl.mode`), which is the audit anchor.
- **E3 (dose grid, r-pairs) and the seed extension are PAUSED** until this
  read; when they run, they run under `--expl.mode apt` with their
  already-registered names (`ax1d*`, `ax1r*` — no runs exist under those
  names, so they denote reward-free fits from the start; any queued
  reward-aware jobs are cancelled, and the 9–16 extension is re-pointed at
  whichever arm the corrective read says needs precision).

## 3. Frozen analysis

- **Primary endpoint (one number per domain-quadrant):** within-seed paired
  contrast s1−s0 on **AUC₁₀₀ₖ only**, frozen machinery
  (`fit_mixed_effects paired`, cluster bootstrap B=10,000 seed 0 + exact
  sign test). AUC₅₀ₖ/AUC₁₂₅ₖ robustness panels; `final10` descriptive.
  No "significant at all windows" claims.
- **Pre-registered secondary suite** (motivated by the 10-Jul reward-aware
  deltas, registered here BEFORE any corrective outcome exists): paired
  t CI, exact sign-flip permutation test, Wilcoxon signed-rank,
  standardized paired effect d_z, and leave-one-seed-out mean range.
- **2×2 read (occupancy × reward supervision):** for each domain-quadrant,
  compare the corrective (reward-free) paired contrast against the 10-Jul
  reward-aware contrast; the interaction estimate = difference of
  within-seed deltas (same seeds, same buffers), bootstrap over seeds.
  Labeled *corrective replication* and *reward-supervision interaction* —
  never pooled with, or presented as, the original confirmatory family.
- **Decision tree (frozen):**
  - Finger Q1 (reward-free) CI excludes 0, same sign ⇒ headline claim
    "reward-free regime occupancy is required for transfer in
    regime-gated domains" is supported; proceed to redesigned E3 for
    identification of occupancy vs composition.
  - Finger Q1 null or reversed ⇒ the occupancy-rescue claim is
    unsupported as reward-free; the paper pivots to (i) the observational
    sign reversal under intervention and (ii) the occupancy × reward-
    supervision interaction; E3 as designed is deprioritized.
  - Cup Q2 (coverage) is read the same way (directional claim only unless
    the CI excludes 0).
- **QC identical to PREREG_phase5a** (≥20 episodes in the 100K window;
  exclusions enumerated).

## 4. Ordering statement

As of this freeze: no corrective run exists; no E3 adapt outcome exists;
the Axis-1 seed-9–16 extension has not produced outcomes. The 10-Jul
reward-aware results are known (they motivated this protocol) and are the
only Axis-1 outcomes read to date.
