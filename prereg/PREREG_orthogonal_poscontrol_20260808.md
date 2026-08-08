# PREREG: orthogonal-test positive control (turn_easy leg) — 2026-08-08

Review-resolution wave #18, cheap leg (user GO 2026-08-08). The registered
orthogonal null (U3/orthogonal_obj: rgo−sgb on spin = +1.0, level gate
45.3 ≥ 20) has **no demonstration that any trunk difference is detectable
on the transfer objective** — and the random policy scores 83.5 on
turn_hard, so the spin level 45.3 is plausibly near-floor. Without a
positive control the objective-specificity claim is attackable as a floor
artifact. Known at freeze: everything incl. U3; unknown: all new outcomes.

## Design — 16 adapt jobs + 5 floor runs; DONOR PREFLIGHT REQUIRED

- **DONOR PREFLIGHT (batch-review B4, before freeze-commit):** the
  `ax1wm_finger_rgoq1s*`/`sgbq1s*` fit dirs are in the cleanup script's
  DELETE-SAFE list and a CONFIRM=DELETE cleanup has run; local copies are
  weight-free skeletons. Confirm cluster survival of seeds 1–4 ckpts
  BEFORE freezing. If gone, this wave becomes 32 refits + 16 adapts on
  the frozen q1 buffers (which survive under KEEP_SUBSTRATE) — that is a
  materially different registration and must be a dated amendment, not a
  silent change. Any surviving donors are re-protected in
  `runroot_cleanup.sh` KEEP-PENDING in the same commit.
- Adapt the existing `ax1wm_finger_rgoq1s{0,1}` and `ax1wm_finger_sgbq1s{0,1}`
  fits (seeds 1–4 each cell = 16 jobs) on **`finger:turn_easy`**
  (`task=dmc_finger_turn_easy`; the env exists via dm_control suite.load
  with identical obs keys/shapes and reward `dist_to_target <= 0`;
  note the repo's existing orthogonal mechanism `orthreward.task` is a
  DIFFERENT thing — reward substitution on the turn_hard env — and is
  not used here) — the
  *near*-orthogonal objective: same domain and reward *channel family*
  (`dist_to_target`-based target reaching, wider target) where turn_hard
  trunk differences SHOULD express if they are detectable on any
  non-identical objective. Frozen protocol, standard adapt/collate,
  mode names `ax1terrgoq1s*` / `ax1tersgbq1s*` (glob-checked vs
  `runroot_cleanup.sh` DELETE-SAFE before submit — standing rule).
- Note: turn_easy reward lives in the same observable the rgo/sgb labels
  thresholded; spin's lives in `velocity`. So this leg tests "adjacent
  objective, same channel" — the informative middle point between
  identical (45.0*) and orthogonal (+1.0).
- **Floor arm `rndte` (batch-review M8): 5 random-policy runs ON
  turn_easy** (`expl_random`, 125K steps, seeds 1–5) — turn_easy's target
  is 0.07 vs turn_hard's 0.03, so the turn_hard floor band does NOT
  transport; the read's floor test uses this wave's own measured band
  (mean ± 2 sd of the 5 runs).
- The full spin-relabel positive control (fit rgo/sgb on spin-relabeled
  buffers) is NOT registered here — it needs a verified spin-reward
  formula from stored keys; separate registration if this leg is
  ambiguous.

## Registered read (`analysis/review_waves_read.py orthpos`, frozen with this file; selfcheck PASS)

- **P-OP1 (primary): per-(seed,side) rgo−sgb LEVEL contrast on turn_easy**
  (n=8 pairs), exact sign-flip permutation primary + bootstrap CI reported.
  A SIGNIFICANTLY NEGATIVE contrast is an explicit non-predicted branch:
  reported as-is, adversarial review before any use, no U3 consequence.
  - Fires positive ⇒ **positive control passes**: trunk differences are
    detectable on an adjacent objective ⇒ the spin null is informative
    about *orthogonality*, U3 stands strengthened.
  - Null AND pooled turn_easy level at/below the upper edge of the
    MEASURED turn_easy floor band (`rndte` runs) ⇒ **floor-confound
    confirmed** — the orthogonal null is downgraded to unadjudicated
    (registered consequence; U3's wording changes to "not adjudicable at
    this level").
  - Null with levels above the measured band ⇒ trunk specificity is even
    narrower than "objective-specific" — report as a sharpening
    (specific to the *exact* objective), flagged for adversarial review
    before external use.
- Descriptive: per-cell means, comparison to turn_hard levels.
- QC standard + modal-n_ep rule.

Freeze ordering: commit this file + reader before any turn_easy job.
