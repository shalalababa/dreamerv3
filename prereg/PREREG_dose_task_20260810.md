# PREREG: task-arm dose–response on the E3 occupancy grid (#21) — 2026-08-10

Review-resolution wave #21 (GO conditional on the donor/buffer preflight —
**user confirmed 2026-08-10: the `axis1_finger` E3 dose pairs' chunks
survive**). The E3 grid (4 occupancy levels, one buffer per level, built
2026-07 by `build_controlled_replay build-dose`) was registered REWARD-FREE
(`ax1d*` names, `AXIS1_EXPL_MODE=apt` enforced in `submit_all.sh`) and was
demoted to robustness-only when P0 nulled the reward-free question
(`PREREG_axis1_corrective_amendment1_20260711.md`). This registration is a
NEW wave under NEW names on the SAME buffers, in the **task** arm — the arm
where the occupancy effect lives — answering the interior-shape question
the record leaves open: the occupancy response is measured at f≈0.054 and
f≈0.323 (large), and at 0.41/0.73 (saturated, high-f_R wave); nothing is
known between 0.054 and 0.323. Threshold-like vs graded onset is the
remaining discriminating shape fact.

Reviewed pre-freeze (ONE reviewer, Opus 5, user-approved): 1 BLOCKING +
11 MAJOR + 7 minor/NIT — ALL blocking/major findings applied below
(calibrated shape test, fold-internal change-point, NON-MONOTONE branch,
real dual fit-witness, env-leakage guard, registered submission command,
manifest-sourced occupancies, D1-leak disclosure, grid-realization scope,
G1 power disclosure).

**Unit of inference (review #11)**: there is exactly ONE buffer
realization per level, so the fit seed is the replication unit *within
one buffer draw per level*. Amendment 1's no-population-claim demotion
is NOT overridden: every verdict below is scoped "on this grid
realization"; occupancy-as-population-cause language remains unlicensed.

Known at freeze: the whole Paper-1 record incl. the 10-Aug read batch
(q1 task contrasts, highfr saturation, s×w_r BOTH-AXES-INERT, breadth
null). Unknown: every td* outcome.

## Design — 32 fits + 32 adapts + E4 pass (finger only)

- **Cells**: `axis1_finger/dose/level{0,1,2,3}` — nominal occupancies
  {0.054, 0.131, 0.249, 0.323}. ALL FOUR levels fitted fresh in-wave
  (same-construction comparability + in-wave replication of the known
  occupancy effect; archived q1 values never enter a rule). Realized
  occupancies are read by the reader DIRECTLY from the grid's
  `manifest.json` (`levels[*].occ_recomputed`; sha256 recorded) — never
  hand-transcribed; each must lie within ±0.02 of nominal or the read
  REFUSES.
- **Fits**: task arm, 500K updates, `dmc_proprio` size1m — the Axis-1
  corrective protocol with exactly one change: `AXIS1_EXPL_MODE=task`
  (review #23 wording fix). Names `ax1wm_finger_td<level>_seed<k>`,
  seeds 1–8.
- **REGISTERED SUBMISSION COMMAND (review #9 — defaults are wrong for
  this wave; run exactly this):**
  `AXIS1_DOSE_TASK=1 AXIS1_EXPL_MODE=task AXIS1_DOMAINS=finger AXIS1_SEEDS="1 2 3 4 5 6 7 8" AXIS1_LEVELS="0 1 2 3" ./scripts/submit_all.sh axis1-dose-bundles`
  The AXIS1_DOSE_TASK path additionally REFUSES stale protocol knobs
  (`AXIS1_ARM`≠full or non-empty `AXIS1_SIZE`/`AXIS1_BASE_CONFIG`/
  `AXIS1_WR`/`AXIS1_INIT_WM` — review #8: task mode disables the
  sbatch's implicit arm guard).
- **Adapts**: standard frozen adapt, 125K steps,
  `adapt_ax1td<level>_finger_seed<k>_ckpt500000` (mode `ax1td<level>`),
  collate via `adaptation_auc` to a NEW csv.
- **E4 measure per fit** (default output dir, F5 lesson):
  `python -m probing.stratified_error measure --probeset $RUNROOT/e4_probesets/finger_v1 --run_logdir $RUNROOT/ax1wm_finger_td<l>_seed<k>`
  → `e4_finger_v1/` inside each NEW td run dir. Collate:
  `python -m probing.stratified_error collate --runroot $RUNROOT --glob 'ax1wm_finger_td*' --probeset_id finger_v1 --output <bundle>/e4_dose.csv`.
- **Names glob-checked (machine-verified)** vs `runroot_cleanup.sh`:
  td names match NO DELETE-SAFE (94) or DELETE-AFTER (5) glob; new
  KEEP_PENDING entries `ax1wm_*_td[0-3]_seed*` (domain-broad, review
  #9) + `adapt_ax1td[0-3]_*` in the same commit.
- **Fit witnesses (dual, real — review #4)**: (1) counters:
  `OFFLINE_FIT_PROGRESS` update == total for all 32, extracted with the
  registered command
  `awk -F= '$1=="update"{u=$2} $1=="total_updates"{t=$2} END{print u, t}'`
  (never a bare `total` substring match — `wm_total` is a loss);
  (2) checkpoint steps: `--ckpt_steps` json produced by the registered
  one-liner over the fit dirs —
  `for d in $RUNROOT/ax1wm_finger_td*_seed*; do ck=$(cat $d/ckpt/latest 2>/dev/null); echo "$(basename $d) ${ck##*-}"; done`
  (the `elements.Checkpoint` dir suffix `-000000500000` parses to the
  step) — every consumed fit must be at 500000. Either witness failing
  REFUSES the read; discrepancies between them are disclosed
  (10-Aug overlay lesson).
- **Preflights (halt + dated amendment on failure)**: (a) chunks —
  PASSED (user, 08-10); (b) manifest occupancy gate (above, enforced by
  the reader); (c) glob dry-run vs cleanup with the new KEEP entries;
  (d) composition rider — per-level source mixture / source_l1 recorded
  PRE-FIT and disclosed: **every occupancy manipulation in this program
  changes composition** (review §5); this wave measures the occupancy
  response SHAPE on this grid, it does not isolate occupancy from
  composition (the unconfounded axis was the s-wave: BOTH-AXES-INERT).
- **D1-leak disclosure (review #10)**: the dose buffers were built
  BEFORE the `--holdout` guard (28de08d1), so `finger_v1` probeset
  episodes may sit inside the level buffers, with level-dependent
  fractions. Before the read, the per-level overlap counts (probeset
  manifest picked-ids ∩ level report episode-ids) are computed
  cluster-side and recorded in the artifact (`--leak_overlap`); P-DR3
  is descriptive-only, and its curve is reported WITH this leak
  disclosure, never as held-out error.

## Registered decision rules (reader frozen with this file; B=10K rng 0; permutation-primary + BCa per the 08-08 standing rule)

Let `y` = auc100k (modal-n_ep rule; milestone-500000 rows only; qc
standard; qc and sub-modal exclusions both recorded), cells ℓ ∈ {0..3}
with manifest occupancies f_ℓ. Contrast machinery = house `two_sample`
(BCa 95% + two-sided permutation; one run per seed per cell makes the
plain bootstrap identical to the seed-cluster form — noted per review
#14).

- **G1 (in-wave replication GATE)**: level3 − level0, BCa CI > 0 AND
  perm p < .05. FAILS ⇒ **GRID-UNINFORMATIVE** — worded "the known
  occupancy effect does not reproduce on dose-constructed buffers AT
  THIS POWER" (G1 power at the wave's own noise basis: ≈0.73 for a true
  +100, ≈0.50 at +69 — review #12); P-DR1/P-DR2 reported with ZERO
  decision weight (stdout labels them so).
- **P-DR2 (PRIMARY, shape adjudication — uses all 32 runs; review
  #1/#2 form)**: leave-one-seed-out CV (8 folds) of three forms:
  **A** y = a + b·f; **B** y = a + b·ln f; **C** step y = a + b·1[f≥c],
  with the change-point c chosen INSIDE each training fold by an inner
  leave-one-seed-out CV over the three interior gap midpoints (no
  selection leakage; reported cut = modal per-fold cut, ties to the
  smallest). Winner = lowest CV-MSE. **Decisiveness is
  permutation-calibrated**: margin = 1 − CVMSE_winner/CVMSE_runner-up;
  null = B=1000 (rng 0) within-seed permutations of the level labels
  (destroys shape, preserves seed effects), full procedure re-run;
  **decisive iff margin > the null's 95th percentile**, else
  **TIE-UNDISCRIMINATED** (no shape wording). Registered basis: the
  reviewer's simulation showed the uncalibrated 5%-margin form fires
  THRESHOLD on a pure null ~19% and on graded truth ~16% of the time;
  the calibrated form restores ~5% nominal. Wording map (all suffixed
  "on this grid realization"): GRADED-LINEAR / GRADED-LOG (coheres
  with the archived log-interaction 1.749*) / THRESHOLD-AT-c.
- **P-DR1 (secondary, activation point)**: contrasts level1−level0 and
  level2−level0, BH q=.05 across the two; a survivor needs
  BH-significance AND a CI excluding 0. Wording: EARLY (both) / MID
  (only level2) / **NON-MONOTONE (only level1 — no activation wording
  licensed; review #3)** / LATE (neither, G1 passing). **Power
  disclosure**: MDE ≈ 69 at n=8v8; interior nulls are weak evidence —
  the shape primary is P-DR2 precisely because it pools all 32 runs.
- **P-DR3 (descriptive, no verdict)**: E4 h0 in-regime rew-NLL per
  level (member band ≤ 1.5, anchor 0.827) — the membership onset curve
  alongside the behavioral one, reported WITH the D1-leak disclosure
  above; divergence between the onsets reported, never adjudicated.

Reader: `analysis/dose_task_read.py`, frozen with this file, selfcheck
PASS before any outcome exists (planted log/linear/step recovery under
the calibrated procedure, null-at-wave-noise non-decisiveness, all gate
trips, loader round-trips, BH mutants). Invocation is plain
`python -m analysis.dose_task_read` (gates are asserts; never -O).
ONE read execution.
