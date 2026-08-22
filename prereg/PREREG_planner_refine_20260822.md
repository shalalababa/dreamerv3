# PREREG — Wave A: policy-seeded refinement (the transmission-rescue
slate, part 2 of 3), 22 Aug 2026

**Program:** the Fable-review rescue slate (PCM cancelled as
level-confounded; this wave is its legitimate successor; revised
same-day per review R2 BEFORE any compute). Question: does greedy
H-step replanning SEEDED AT the trained policy's own imagined plan
increase realized distractor attribution? **Claim scope (rev
A-M3):** REFINE_STD bounds the per-step ACTION deviation, not the
trajectory deviation (which compounds over 2000 steps under a
different controller) — so the null is a CONTROLLER-CLASS
stationarity statement ("this refinement class does not increase
it"), never a neighborhood/'local maximum of the field' statement.
The ambient wave's from-scratch greedy CEM lost to the amortized
policy on every objective (a failed relative positive control —
Fable F2); this design starts the planner AT the policy's plan.

**VALUE-AWARE DISCLOSURE:** priors informed by the executed ambient
wave (Δ_vs_policy 1/8; share decomposition 8/8) — and
attr_d(policy)/attr_d(random) for these EXACT 8 checkpoints are
already published (`se_planner_read_20260822`), which this design
turns into a free integrity control (below) rather than leaving as
an undisclosed overlap. Fresh computation: no refinement probe has
run on any checkpoint.

## 1. Wave (1 registered GPU smoke + 8 GPU probe jobs on the
Stage-1 checkpoints, seeds 10–17; NO training)

`uncfield/se_refine_probe.py` (built 22 Aug; CPU smoke 11 s;
`--selfcheck` unit-tests the REAL refine_loop — rev A-M4: the warm
plan is candidate 0 of iteration 0 EXACTLY (eps[0]=0), best_first ≥
warm_score by construction, refinement progresses on a smooth toy
objective, deterministic under fixed keys; the (H, adim) warm-plan
slice is shape-asserted in plan_fn at trace time): arms random
(normalizer source) / policy / **refine_attr** (CEM warm-started at
ONE STOCHASTIC SAMPLE of the policy's H-step imagined action plan —
disclosed (rev A-m8); init std REFINE_STD = 0.2 pinned, objective =
normalized distractor attribution) / **refine_disag** (the same
warm-started CEM on the raw disag objective — the instrument-reach
arm). Registered constants otherwise identical to the ambient wave
(4×500, N=256, K=32, I=3, H=12); measurement = the frozen #32 form
(online states, pre-action pairing, CRN per arm, elite diagnostics,
reset_hits==0), plus per-step `warm_score` / `best_final` /
`refine_disp` (first-action displacement from the warm start) /
`clip_frac` (candidate saturation at ±1 — the evidence behind the
std-0.2 scope, rev A-M8), per-arm task return, and raw norms in
the npz. Driver `scripts/se_refine.sbatch` (array 0–7, one CUDA
device per task; SMOKE=1 writes to a separate dir, #32 m2).
**Registered GPU smoke (rev A-M7): one SMOKE=1 task on seed 10
BEFORE the array — the warm-start path has only run on CPU.**

## 2. Registered read (frozen `uncfield/se_refine_read.py`,
selfcheck PASS; ONE execution)

- Validity: Stage-1 pins, seeds 10–17, registered constants incl.
  refine_std, occupancy-gate constants pin + final-ckpt +
  newest-ckpt-dir cross-check (rev A-M5), probe-json run-dir
  identity, backend/device pins, float64-exact provenance (attr +
  intrinsic + occupancy), n_steps/reset_hits, CEM sanity both
  refine arms, **per-run degeneracy floor intrinsic(policy) >
  intrinsic(random)** (rev A-M1 — a dead WM makes every ratio
  vacuous; ambient base rate 8/8); defective → excluded,
  denominator fixed at 8; < 7 valid ⇒ NOT-ADJUDICABLE.
- **HARD INSTRUMENT GATE (rev A-B1/A-M2, adjudicated first,
  horizon-fair):** the refine_disag arm's IN-PLAN control —
  model-predicted `best_final − warm_score > 0` on average in
  ≥ 6/8 valid runs. Because candidate 0 IS the warm plan, this
  cannot fail for horizon/discount reasons; failure means the
  search cannot improve its own objective in-model ⇒
  **NOT-ADJUDICABLE-INSTRUMENT-WEAK**.
- **REALIZED-STRENGTH SCOPE (rev A-M2 — wording, never validity):**
  realized intrinsic(refine_disag) ≥ 0.9 × intrinsic(policy) in
  ≥ 6/8. Pass ⇒ the null ships with full stationarity wording;
  fail ⇒ the null is **POLICY-STATIONARY-GREEDY-EXEC-SCOPED**
  ("under greedy H-step execution" — the undiscounted 12-step
  replanner may legitimately shed realized intrinsic over 2000
  steps without being weak in-plan).
- **PRIMARY:** Δ = attr_d(refine_attr) − attr_d(policy); FIRES
  (**REFINEMENT-TRANSMISSION**) iff Δ > 0 in ≥ 7/8 AND exact
  sign-flip p ≤ .05 (sign-flip and BCa run over the valid set;
  counts keep the fixed denominator 8 — disclosed, rev A-m5). The
  NULL is the registered Tier-1 stationarity claim (scope above).
  **Predicted-vs-realized reporting (rev A-B1):** the refine_attr
  arm's in-plan gains are reported alongside — a null with large
  model-predicted gains is a predicted-but-unrealized dissociation
  (off-support imagination), a DIFFERENT publishable finding from
  model-predicted stationarity; reported, never adjudicated.
- Registered descriptives (emitted on EVERY path, rev A-m4): SHARE
  rows per arm (pre-declared), refine_disag−policy attribution row,
  occupancy / intrinsic / task-return rows, BCa on Δ, refinement
  displacement + clip fraction, and the **reproduction row** (rev
  A-M6): per-run attr_d(policy) and attr_d(random) against the
  ambient wave's published per-seed values (report-only; cross-GPU
  shifts are a measured phenomenon — the 16-Aug single-device rule —
  so the tolerance is descriptive, and a gross mismatch flags a
  mis-pointed checkpoint or env-reconstruction drift before Δ is
  believed).

## 3. Fences

Track-B transmission section + the Paper-5 tier ladder (pinned in
PREREG_planner_localized_20260822 §2). The refinement planner is an
instrument. Scope: stationarity under H=12/std-0.2 GREEDY
refinement of this controller class — never "reachable maximum"
(retired program-wide per the Fable memo), never a neighborhood
claim.

Freeze = review R2 adjudicated (done 22 Aug — 1 BLOCKING + 8 MAJOR
+ 8 minor on this wave, all resolved pre-freeze) + commit of this
file + `uncfield/se_refine_probe.py` + `uncfield/se_refine_read.py`
+ `scripts/se_refine.sbatch` — then GPU smoke → the 8 probe jobs.
