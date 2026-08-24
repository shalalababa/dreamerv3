# Wave A (policy-seeded refinement) — REGISTERED READ, 22 Aug 2026

**Prereg:** `PREREG_planner_refine_20260822.md` (rev per review R2,
frozen pre-compute). **Reader:** `uncfield/se_refine_read.py`, ONE
execution 22 Aug on the verified bundle
`wavea_refine_20260822_182736` (manifest byte-exact vs the
committed copy; 8/8 runs; all probes full-mode, one CUDA device —
all 8 cells on ONE RCC GPU, job 54313394 — post-revision
instrument fields present).

## OUTCOME: **POLICY-STATIONARY** (the full-wording Tier-1 null — both controls passed)

- **HARD INSTRUMENT GATE (in-plan) PASSES 8/8:** the warm-started
  refiner improves its own objective in-model over the policy's
  own plan in every run.
- **REALIZED-STRENGTH SCOPE PASSES 8/8:** realized
  intrinsic(refine_disag) / intrinsic(policy) = 0.94–1.06 — the
  refiner MATCHES the incumbent's realized harvest (the ambient
  wave's from-scratch CEM ran ~0.63×; the Fable-F2 objection is
  discharged by construction AND empirically).
- **PRIMARY: Δ = attr_d(refine_attr) − attr_d(policy) — a TIGHT
  null.** Per-run: −0.00142, +0.00022, −0.00023, −0.00085,
  −0.00100, +0.00244, +0.00014, −0.00109; 3/8 positive, exact
  sign-flip p = .668; mean −0.00022, BCa [−0.00079, +0.00110].
  Against policy attr_d levels of 0.011–0.022, the interval bounds
  any effect within ~±10% of the policy's own attribution.

**Registered claim (controller-class, rev A-M3 wording):** greedy
H=12 replanning seeded at the deployed policy's own imagined plan,
at MATCHED realized strength, does not increase realized distractor
attribution. This is the Tier-1 stationarity rung of the pinned
ladder — landed with its gates green.

## The registered predicted-vs-realized dissociation (reported, not adjudicated — and it is LOUD)

Per-run model-predicted attr in-plan gains: 0.017–0.027 — the WM
predicts headroom at the warm start COMPARABLE TO THE POLICY'S
ENTIRE ATTRIBUTION LEVEL (~+0.02 on levels of ~0.011–0.022), and
the refiner moves substantially to chase it (first-action
displacement 0.44–0.72; candidate clip fraction 0.41–0.47 — the
actor saturates ±1, so roughly half the search ball is clipped,
disclosed scope). **Realized gain: ≈ 0.** The imagined attribution
landscape overstates reachable attribution by an order of
magnitude under execution — an off-support-imagination gap. For
the containment story this is a mechanism, not a nuisance: an
attacker optimizing the model's imagined exposure harvests nothing
real. (Registered descriptive; Wave B's imagined-vs-realized
secondary is the designed follow-up.)

## Integrity rows

- Reproduction vs the ambient published values (report-only,
  cross-device shifts expected per the 16-Aug rule): 6/8 within
  ~±5% (s11 0.0112/0.0120, s12 0.0140/0.0144, s13 0.0125/0.0129,
  s14 0.0139/0.0143, s16 0.0114/0.0121, s17 0.0159/0.0164); two
  larger deviations s10 (0.0221 vs 0.0357) and s15 (0.0106 vs
  0.0163) — different GPU class (RCC vs instance), no gate, both
  runs' within-run contrasts unaffected (all comparisons are
  same-device, same-invocation).
- refine_disag−policy attr row: −0.0009…+0.0021 (≈0 — refining the
  DISAG objective also leaves attr unchanged, coherent).
- 0 excluded, 0 degenerate, 0 cem-invalid; fit counters OK 8/8;
  ckpt gates green (final ckpt, newest-dir).
- **Protocol deviation (disclosed):** the registered SMOKE=1
  pre-array cell was not run — ops executed the 8 cells directly
  (8/8 rc=0). The smoke's de-risk purpose is discharged post-hoc
  by the clean full outputs; no bearing on the estimand.

## Consequences

- The Paper-5 security paragraph's Tier-1 rung is LANDED on the
  refinement axis: matched-strength stationarity, gates green.
  Tier-2 wording still requires Wave B (unchanged).
- Composes with Wave C (same day): the from-scratch CEM remains a
  weak optimizer (its null stays optimizer-scoped), so Wave A is
  the load-bearing null; the predicted-vs-realized gap is the new
  mechanism paragraph.

Artifacts: `se_refine_read.json` (this dir). Data:
`local_results/wavea_refine_20260822_182736` (manifest committed).
