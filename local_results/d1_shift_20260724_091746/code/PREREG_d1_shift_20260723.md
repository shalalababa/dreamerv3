# PREREG: Route-B shift-consequence probe (frozen 2026-07-23)

Paper-2 / Route-B downstream-consequence leg, per plan v3
(`EVPI_Plan_Revision_20260723.md` §2 "deferred arms" — owner GO under
the standing portfolio policy). Adapted from the external review's
crossed-design proposal (`D1_GPT_Analysis_20260723.md`), repurposed:
this is explicitly **NOT a Route-A revival** — Route A is closed
(`artifacts/gate_d1_r2_20260723/`) and no outcome of this probe
reopens it or unfreezes the 24+24. This file + the frozen read + the
labeler shift extension are committed BEFORE any shift pilot or label
exists.

## Registered prediction

From Stage-0's cross-policy boundary (uncertainty stays error-tracking
under support shift) and the attractor mechanism: **the real
(privileged-lookahead) purchase gains average value when the labeled
states carry task-relevant uncertainty the converged WM lacks** —
states visited under a different policy (xpol) or dynamics the WM was
never trained on (phys). The R2/Gate-D1 own-policy converged cells are
the boundary where that value was small and unpredictable.

## Design: 6 fresh e1 pilots × 3 label arms (single GPU)

- Pilots: {cup, finger} × seeds {21, 22, 23}, run ids
  `d1s_{dom}_seed{s}`, config `dmc_proprio d0_probe` (dose ZERO only —
  the nuisance doses are excluded by design per Amendment 1 §D),
  `--env.dmc.render False`, 1e5 steps, FINAL checkpoints only (no
  early snapshots; the phase axis was adjudicated by R2).
- Fresh seeds because the R2 pilot checkpoints were not retained in
  the pulled copy (`runroot_light` is stripped); fresh runs keep every
  number prospective and platform-uniform.
- Label arms per run (labeler_version `shift_ext_20260723`):
  - **base**: own policy, unshifted — the paired baseline.
  - **xpol**: the sibling seed's final checkpoint DRIVES the base
    trajectory (rotation 21→22→23→21 within domain); the run's own
    agent remains the evaluation agent for beliefs, d0 evals,
    operations, and rollouts.
  - **phys**: all dm_control body masses ×1.3 at label time (model
    params are outside `physics.get_state()`, so CRN
    snapshot/restore is untouched; the labeler's determinism assert
    verifies this on the real path, and the smoke stage runs both
    arms before any label).
- Dials: identical to R2's pinned set (states 200, horizon 100,
  label_every 25, actions 8, rollouts 16, labeler seed 0,
  ref_stride 5). 18 label passes total.

## Instruments (committed with this file)

- `d0/oracle_labels.py` shift extension (`--behavior_checkpoint`,
  `--mass_scale`; behavior=None path bit-identical to the r2_ext
  labeler; selfcheck PASS incl. behavior-driven trajectory +
  mass-scale plumbing).
- `analysis/d1_shift_read.py` (frozen read; selfcheck PASS: planted
  xpol lift fires xpol only, null grid fires nothing, missing cell
  trips). Calibration criteria imported unchanged from the frozen
  Gate-D1/R2 machinery.
- `scripts/d1shift_local.sh` (driver; pilots → smoke → labels with
  ordering enforced).

## Registered decision rule

- PRIMARY, one per shift arm (2 looks, disclosed): D_r =
  mean(Δ_real | arm) − mean(Δ_real | base) per run (paired by
  domain×seed; 6 runs, domains pooled). **Arm FIRES iff the
  run-clustered bootstrap 95% CI (B = 10K, rng 0) of mean(D_r) is
  entirely > 0.**
- SECONDARY (registered descriptors, never decisional): C1-style
  calibration (F0/F1, frozen machinery) within each arm's pooled
  cell; per-domain splits; change rates; per-feature Spearmans. A C1
  pass in a shift cell is reported as mechanism-consistent and would
  motivate a separately registered follow-up; it does NOT revive
  Route A and adjudicates nothing here.

## Consequence map (frozen)

- **≥ 1 arm fires** ⇒ Route B gains its downstream-consequence leg:
  "computation/lookahead purchases regain value exactly where support
  shifts away from the training distribution" — the paper's
  actionable complement to the boundary negative. Which arm(s) fired
  scopes the claim (policy-support vs dynamics shift).
- **Neither fires** ⇒ the consequence leg fails; Route B rests on
  mechanism + boundary; this probe is reported as a registered
  negative. No further shift arms without a new registration.
- Either way: 24+24 stay frozen; Route A stays closed; the imagined
  op stays retired (real op only here).

## Disclosure

Known at freeze: everything in `artifacts/gate_d1_r2_20260723/` and
the Gate-D1 record (own-policy converged Δ_real means: late×e1 +0.07
pooled; dose cells excluded here as nuisance). Unknown: every
shifted-cell quantity (no xpol or phys label has ever been produced by
any version of this pipeline). The mass factor 1.3 was chosen at the
design stage without any shifted-label evidence. Compute [prov.
estimates]: 6 pilots ~1–2 h each + 18 label passes ~20–40 min each ⇒
~0.7–1.5 days sequential on the 5090.
