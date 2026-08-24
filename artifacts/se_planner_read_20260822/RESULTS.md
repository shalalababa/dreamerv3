# Planner-in-the-loop registered read — 22 Aug 2026

**Prereg:** `PREREG_nfi_planner_20260821.md` (review #32). ONE
execution of the frozen `se_planner_read` on a composite read root:
planner overlay bundle `b3_planner_20260822_121225` (manifest 27/27
OK) symlinked over the verified Stage-1 bundle's config/ckpt/metrics
(the probes ran on the Stage-1 run dirs; both trees
manifest-verified; composition documented here). 8/8 runs valid,
zero excluded, CEM sanity passed on all planner arms, ckpt gates OK,
single CUDA device per panel, registered constants pinned.

## Outcome: **UNSTEERABLE** — the transmission axis closes at the
strongest optimizer

- **Steerability gate FAILS decisively:** ceil =
  attr_d(cem_distractor) − max(attr_d(policy), attr_d(random))
  positive in **1/8** runs (per-run ceilings −0.020…+0.004). The
  DEDICATED distractor-chasing planner cannot push distractor
  attribution above the passive anchors.
- **The #32-B1 conjunctive primary earned its keep:** Δ_vs_real =
  +0.0008, 7/8 positive, sign-flip p = 0.031 — the
  comparator-suppression pattern the review predicted from the
  smoke — while Δ_vs_policy = −0.0082, **1/8 positive, p = 0.996**.
  The pre-review reader (Δ_vs_real alone) would have printed
  TRANSMISSION on an artifact; the registered read correctly does
  not adjudicate a fire through a failed gate.
- **Mean distractor attribution per arm:** policy **0.0168** >
  cem_distractor 0.0118 > cem_disag 0.0086 > cem_real 0.0078 >
  random 0.0057. **The trained policy's own visitation carries the
  highest mispriced attribution of every arm** — 43% above the
  planner explicitly optimizing for it. Descriptive companion rows:
  realized intrinsic policy 1.9e-4 > cem_distractor 1.2e-4 >
  cem_disag 1.2e-4 > cem_real 0.9e-4 > random 0.8e-4 (greedy H=12
  planning also harvests LESS disagreement than the amortized
  policy); occupancy 0.55–0.75 across arms.

## Registered consequences

1. **The dissociation chain closes at every level:** accounting
   (θ₁ 5.26×, 8/8) → ranking (anti-transmitted) → trained-policy
   behavior (M3, no diversion) → value gradients (no follow) →
   **greedy deployment planning (UNSTEERABLE)**. Five levels, no
   transmission anywhere. Paper 5's security paragraph upgrades
   from "containment by architectural accident" to "containment
   robust even to explicit greedy deployment optimization" — a
   stronger, now fully evidenced, negative.
2. **Track-D is retired** (registered: UNSTEERABLE does not trigger
   it — #32 m11).
3. Per the fence, the result routes to the Track-B taxonomy's
   transmission paragraph; Paper 5 carries no Act-2 deployment
   section.
4. Registered scoping: "unsteerable" is a statement about H=12
   greedy CEM over this frozen WM in this env class; the
   policy-tops-every-planner observation is descriptive material
   for the mechanism discussion (the amortized objective
   concentrates visitation where its own mispriced disagreement
   lives), not a claim.

Protocol note: the registered SMOKE=1 GPU task's mechanics/timing
role was superseded by events (the wave ran and validates
end-to-end); the smoke had no adjudication role by registration
("cannot amend the fire rule").

---

## CORRECTION / RE-SCOPING — 22 Aug 2026 (Fable review; the
registered read json above is untouched and stands)

The "Registered consequences" item 4's mechanism sentence is
**withdrawn as unscoped**: a share/level re-analysis of this read's
own published json (`artifacts/planner_share_decomp_20260822`,
VALUE-AWARE, disclosed) shows the policy is the per-run argmax on
EVERY key (velocity 8/8) — a global disagreement-LEVEL fact — while
in COMPOSITION terms the dedicated planner raised the distractor's
SHARE above both passive anchors in 8/8 runs. The licensed wording:

- **Tier 0 (registered, this read):** under the registered H=12
  greedy CEM, distractor attribution could not be pushed above the
  passive anchors (ceiling positive 1/8); the dissociation chain
  closes at a fifth level AT THIS OPTIMIZER STRENGTH.
- **Tier 0-descriptive (value-aware, must be disclosed):** the
  trained exploration policy's visitation carried the highest
  ABSOLUTE attribution on every channel alike (consistent with its
  disagreement-maximizing training); the dedicated planner DID
  steer composition (distractor share 8/8 above anchors) at a lower
  overall level; the registered null is about absolute exposure in
  a comparison whose incumbent is itself the stronger, amortized
  optimizer of the same field.
- "Cannot amplify exposure" unqualified is NEVER licensed by this
  program. Stronger tiers await the registered C/A/B rescue slate
  (localized-channel wave; policy-seeded refinement; adversarial
  amortized actor — the last gates the security paragraph's final
  wording).
