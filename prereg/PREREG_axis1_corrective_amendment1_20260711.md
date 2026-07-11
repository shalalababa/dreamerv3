# Amendment 1 to PREREG_axis1_corrective_20260711.md — frozen 2026-07-11

New file per this directory's amendment rule (the original protocol is not
edited). Registered in response to the v4 editorial re-review
(`Research_Editorial_Review_v4_20260711.tex`), **before any corrective
(`ax1f*`) or E3 outcome exists**.

## A. Inferential hierarchy for the corrective read (closes re-review §4)

- **Sole headline confirmatory contrast: finger Q1** (s1−s0, AUC₁₀₀ₖ).
- **Pre-named secondary causal contrast: cup Q2** (coverage).
- **Specificity checks (reported, not confirmatory): cup Q1, finger Q2.**
- **Primary estimator and decision criterion:** the frozen paired machinery
  — mean within-seed delta with cluster-bootstrap percentile 95% CI
  (B=10,000, seed 0). The decision at every branch of the protocol's
  decision tree is taken on **this CI alone** (excludes 0 or not). The
  sensitivity suite (paired t, exact sign-flip permutation, Wilcoxon, d_z,
  leave-one-out) is robustness reporting only — never an alternative
  decision route. The exact sign test is reported per the original PREREG
  but does not control decisions.
- No multiplicity correction between the headline and the secondary; they
  are reported as one confirmatory + one secondary, never as a family of
  four discoveries.

## B. E3 v2 replication unit (closes re-review §3) — preferred option adopted

**Unit = collector run.** One high/low-occupancy buffer pair is formed
*within* each Phase-4 collector run (episodes from that run only, both
sides), coverage-matched under the frozen criteria (cov_match_frac 0.05,
overlap cap, frozen search seed); collection policy, collector identity,
and source mixture are thereby identical across sides *by construction*.

- **Feasibility (measured 11 Jul from the frozen episode indices**, top-100
  vs bottom-100 episodes by per-episode occupancy, before coverage
  matching): finger — all 15 collector runs give within-run Δocc
  0.20–0.36 (bottom-100 ≈ 0.000); cup — random collectors 0.31–0.37,
  apt 0.11–0.15, p2e only 0.05–0.06.
- **Units:** finger — all 15 collector runs. cup — the 10 random+apt runs
  (p2e collectors cannot express the contrast internally; recorded as a
  scope note, not silently dropped). Buffer size ≈100 episodes/side
  [prov. until the builder lands; fixed before any E3v2 run].
- **Training seeds are nested** (1–2 per pair) and are never the
  inferential unit.
- **Primary small-cluster inference:** one delta per collector run
  (averaging over nested seeds); **exact sign-flip permutation across
  collector runs** on those deltas (two-sided), reported with the mean and
  a hierarchical (collector-level) bootstrap CI as secondary. No asymptotic
  random-effects model carries the claim at ≤15 clusters.
- Mode strings: `ax1w<collector>s<side>` reserved (`w` = within-run;
  collector index per the frozen episode-index source order); registered
  now, excluded from dose-response population models by the frozen
  `--modes` filter like all intervention modes.
- The previously registered dose grid (`ax1d*`) and exclusion r-pairs
  (`ax1r*`) are demoted to **buffer resamples / robustness** (per the
  re-review's third option): same-direction checks, no population claim.

## C. Launcher verification record (closes re-review §§1–2)

- Bundle-path defect confirmed and fixed 11 Jul: `submit_axis1_bundle`
  builds an explicit `--export` list (deliberately no `--export=ALL`) that
  lacked `AXIS1_EXPL_MODE`; added (with `AXIS1_SAVE_EVERY_UPDATES`),
  value-validated at submit time (`apt|task`), and threaded explicitly
  through the bundle child invocation.
- Flag-path defect found by the required smoke test and fixed:
  `offline_fit` rejects `--expl.mode` (config key is nested); corrected to
  `--agent.expl.mode`.
- **Smoke test (11 Jul, local CPU, debug config, real 400-step cup
  replay; evidence in `artifacts/smoke_axis1_expl_20260711/`):**
  - `--agent.expl.mode apt`: saved config `expl.mode: apt`; WM losses
    {con, dyn, position, rep, velocity}; **no `rew` loss, no `rew` head
    in the parameter table**.
  - `--agent.expl.mode task`: saved config `expl.mode: task`; same losses
    **plus `rew`**, with a `rew` head present.
- Per-run audit rule for the 64 corrective jobs: saved
  `config.yaml: expl.mode == apt` AND no `rew` term in the
  `OFFLINE_FIT_PROGRESS`/log loss line; any violation excludes the run and
  is logged as a deviation.
