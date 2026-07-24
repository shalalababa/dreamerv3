# D1 shift-consequence probe — READ (2026-07-24)

> **INSTRUMENT-INVALIDATED (2026-07-24).** The shared D1 real-operation
> labeler contained two confirmed implementation defects (stale-carry
> candidate branches: obs_t never assimilated, prevact = a_{t-1} not the
> candidate; and policy-RNG not common across branches), so delta_real /
> G labels realize an unintended estimand. The read below remains a
> faithful record of the registered procedure ON THOSE LABELS, but its
> decision-value conclusions cannot be interpreted as established.
> Details: analysis/DEVIATIONS.md 2026-07-24 deviation entry; repair +
> relabel registration: prereg/PREREG_d1_relabel_20260724.md.
> Audit addendum (same day): the xpol cells carry a THIRD defect — on
> behavior-driven trajectories the eval belief's prevact was the eval
> agent's own counterfactual sample, not the executed behavior action
> (DEVIATIONS 24-Jul audit entry, item 1).


Frozen read `analysis/d1_shift_read.py` executed on
`local_results/d1_shift_20260724_091746/runroot_light/d1shift_local/d1_labels`
(18 label passes, 6 fresh e1 pilots × {base, xpol, phys}).
Registration: `prereg/PREREG_d1_shift_20260723.md` (+ on-time
Amendment 1, smoke-stage transfer-guard fix, plumbing only).
Machine record: `d1_shift.json` (this directory).

## VERDICT — NEITHER ARM FIRES (registered negative)

> Shift consequence does not fire: purchase value does not detectably
> rise under either shift arm — the consequence leg fails; Route B
> rests on mechanism + boundary alone.

The registered prediction (real-purchase value rises where the labeled
states carry uncertainty the converged WM lacks — cross-policy support
or shifted dynamics) is NOT supported. Frozen consequences: Paper 2's
downstream-consequence leg is closed; the paper rests on mechanism
(context-legible constants: early −0.208* harmful, late×e4 +0.313*
valuable) + boundary (state-illegibility: Gate-D1 0/4, R2 16/16 null,
ladder no-signature). No further shift arms without a new
registration. 24+24 stay frozen; Route A stays closed; imag op stays
retired.

## Primary (2 registered looks, run-clustered bootstrap, B=10K, rng 0)

D_r = mean(Δ_real | arm) − mean(Δ_real | base), paired per run,
6 runs pooled; FIRES iff 95% CI entirely > 0.

| arm  | D pooled | 95% CI            | fires | cup    | finger |
|------|----------|-------------------|-------|--------|--------|
| xpol | −0.006   | [−0.853, +0.772]  | NO    | +0.207 | −0.218 |
| phys | −0.055   | [−0.483, +0.325]  | NO    | −0.320 | +0.210 |

Per-run D_r — xpol: cup −0.94/+0.51/+1.05, finger −1.67/−0.02/+1.03;
phys: cup −0.97/+0.05/−0.03, finger +0.66/0.00/−0.03.

Structure notes (descriptive):
- Point estimates ≈ 0 in both arms, and the per-domain signs are
  OPPOSITE within each arm (cup/finger mirror in both) — no coherent
  direction anywhere, not a near-miss.
- Power honesty: the phys CI excludes effects > +0.33 — a genuine
  bound. The xpol CI is wide (±0.81 at n=6 runs with per-run spread
  −1.7..+1.1); the xpol null excludes only large effects (> +0.77),
  so it is a registered negative, not a tight zero.

## Secondary (registered descriptors, never decisional)

C1-style calibration (frozen Gate-D1/R2 machinery) fails in ALL THREE
arms — no exploitable state-level calibration emerges even under
support/dynamics shift:

| arm  | F1 Spearman | 95% CI            | gate  | pooled Δ_real |
|------|-------------|-------------------|-------|---------------|
| base | +0.058      | [−0.022, +0.150]  | FAIL  | +0.200        |
| xpol | −0.022      | [−0.065, +0.015]  | FAIL  | +0.194        |
| phys | +0.040      | [−0.000, +0.083]  | FAIL  | +0.145        |

- This extends the state-illegibility boundary: purchase value stays
  state-illegible even where the WM's support assumptions are broken.
  Consistent with (and strengthens) the ladder's
  no-exploitable-heterogeneity result.
- base-arm pooled Δ_real +0.200 on fresh final-checkpoint e1 pilots is
  sign-consistent with Gate-D1/R2 late×e1 (+0.07/+0.068), somewhat
  larger (different seeds, 1e5-step pilots).
- Per-feature Spearmans all |ρ| ≤ 0.08 in every arm (incl. udyn_resid,
  the R1-adjudicated feature: −0.079 in phys — wrong sign, tiny).

## Integrity / provenance

- 18/18 registered cells present, n=200 each; smoke npz excluded by
  filename rule. change_rate_real 0.815–0.900 across cells (matches
  R2's ≈0.85 — the op is not saturated or dead).
- Label meta conforms: labeler_version `shift_ext_20260723`; xpol
  behavior_checkpoint follows the registered rotation (21→22→23→21,
  verified in meta); mass_scale 1.3 in phys cells, 1.0 elsewhere;
  pinned dials (200/100/25/8/16/seed 0/ref_stride 5) in every cell.
- Code snapshot shipped with the results
  (`local_results/.../code/`) diffs CLEAN against the repo instrument
  (oracle_labels.py, d1_shift_read.py, prereg, driver — all identical).
- Freeze ordering: registration commit `d68277f1` → smoke failure →
  Amendment-1 fix commits `f8f3e1e7`/`cf0eeead` (23 Jul 21:46/22:32
  CDT, before any shift label) → smoke PASS both arms (log shows the
  CRN determinism assert covered the mass-scaled path) → labels
  complete 24 Jul 07:17 UTC → this read. Read selfcheck re-run PASS
  immediately before execution.
- Descriptive caveats: (1) finger seeds 22/23 base cells have mean
  Δ_real = 0.000 exactly (reward-floor states dominate at 1e5-step
  finger pilots), which depresses paired sensitivity in those finger
  cells; disclosed as a design limit (steps were pinned at
  registration). (2) One scipy ConstantInputWarning during the
  secondary pass (a constant input in one per-cell Spearman;
  secondaries are non-decisional, primaries unaffected).

## Paper-2 status after this read

Experimental program COMPLETE — nothing left to run. All legs
adjudicated: Gate-D1 0/4 (18 Jul), R2 null 16/16 → Route A closed
(23 Jul), ladder no-signature → ensemble gated (23 Jul), shift
consequence null both arms (this read). Remaining work is writing:
mechanism + boundary paper with the context-level constants as the
positive result and this probe as the registered
negative-under-shift.
