# PREREG — Wave C: localized-channel planner wave (the
transmission-rescue slate, part 1 of 3), 22 Aug 2026

**Program:** the Fable-review rescue slate (memo adjudicated 22 Aug;
ledger row; PCM cancelled as level-confounded; revised same-day per
review R2 BEFORE any compute). Question: the AMBIENT planner wave
returned UNSTEERABLE — but A1 (registered, p=.0143, third complete
separation) shows that in the SPATIALLY LOCALIZED variant the
trained policy AVOIDS the region where the misprice is largest
(θ₁≈7× there), leaving spatial headroom (registered occupancy gap
0.4599 vs 0.5197 — an occupancy fact; nothing yet links its size to
the attribution flip the primary needs, which is why the primary is
the test). Does deployment-time planning cash the misprice where
the policy's avoidance leaves it unclaimed? Prior sharpener (zero
compute, disclosed): the A1 region rows show the disagreement field
elevated ~1.54× INSIDE the gate region for hetero (complete
separation vs flat 0.82; `REGION_PRIOR_NOTE_20260822.md`) — the
signal the CEM must steer toward exists in these WMs.

**VALUE-AWARE DISCLOSURE (binding):** the directional priors here
are informed by executed reads — A1's avoidance, the ambient
planner wave's Δ_vs_policy (1/8 positive) and its share
decomposition (planner steered composition 8/8;
`artifacts/planner_share_decomp_20260822`), and the A1 region
rows. The DATA are fresh: no planner probe has ever run on these 8
checkpoints.

**Region scope (rev C-M3):** the pinned occupancy gate is the
smoke-median of position[0]; under the mod map m =
clip((x−lo)/(hi−lo), 0, 1) the gate region is m ∈ (0.370, 1] and
the out-region m ∈ [0, 0.370) — the gate region is the
ABOVE-MEDIAN-AMPLITUDE HALF of the state distribution, not an
amplitude-zero/amplitude-one dichotomy.

**Normalizer scope (rev C-M1):** attribution is per-run-normalized
by the run's OWN realized distractor std; the hetero distractor is
heteroscedastic, so by Jensen its global normalizer is
systematically larger than the flat arm's (flat scale was matched
to E[m], not sqrt(E[m²])) — compressing hetero Δ toward zero and
biasing the INTERACTION MAGNITUDE positive. The registered
defenses: criterion (ii) (the within-wing sign count) is immune to
the compression, and the reader emits a de-normalized robustness
row (Δ·mean(norm_d²)) plus per-run normalizer records. The
interaction magnitude is NOT interpretable across wings on its own.

## 1. Wave (8 GPU probe jobs + 1 registered smoke; NO training)

The FROZEN `uncfield/se_planner_probe.py` UNCHANGED (registered
constants: 4×500 per arm, N=256, K=32, I=3, H=12; all five arms on
one CUDA device per job) on the 8 A1 checkpoints:
`se_avoid_het_s44..47` (hetero: the mod-quadruple scarecrow field)
and `se_avoid_flat_s48..51` (amplitude-matched flat comparator).
Driver `scripts/se_planner_loc.sbatch` (array 0–7; SMOKE=1 writes
to a separate `se_planner_smoke` dir, #32 m2). **Registered smoke
(rev C-M7): one SMOKE=1 task on se_avoid_het_s44 BEFORE the array —
the hetero env path (mod quadruple) has never been exercised by
this probe.** The A1 read already validated these checkpoints
(misprice gate 4/4, gradient-delivery 4/4); the repointed-ckpt
disclosure carries over, and the reader now GATES on it (below).

## 2. Registered read (frozen reader
`uncfield/se_planner_loc_read.py`, built + selfchecked BEFORE this
compute; ONE execution)

- **Validity per run:** A1 pin blocks per arm (hetero: mod
  quadruple to 1e-9 at scale 1.0; flat: scale 0.35891393, no mod;
  both: Stage-1 expl pins, bootstrap True, penalty inert), seeds
  44–51 distinct, registered planner constants, occupancy-gate
  constants pin (rev C-M2), probe-json run-dir identity (rev
  C-m4), smoke off, one CUDA device, json==npz float64-exact
  provenance, n_steps==2000, reset_hits==0, CEM sanity on ALL
  planner arms (registered decision, rev C-m2: strict — a
  cem_real/cem_distractor failure voids the run because the S2/S3
  secondaries need those arms and partial-arm runs invite
  selection), **final-ckpt + newest-ckpt-dir cross-check + step
  floor 490,000 (rev C-B2 — these exact run dirs are where the
  stale ckpt/latest incident happened; probed ckpt + step recorded
  per run)**. Defective runs → excluded_runs, read persists.
  **PRIMARY REQUIRES THE FULL 4+4** (the #34-B3 lesson): anything
  less ⇒ NOT-ADJUDICABLE.
- **PRIMARY (localized transmission, conjunctive):** per run,
  Δ = attr_d(cem_disag) − attr_d(policy). FIRES iff BOTH
  (i) the hetero−flat interaction is positive: mean Δ(hetero) −
  mean Δ(flat) > 0 with exact C(8,4) label permutation p ≤ .05
  (floor 1/70), AND (ii) Δ > 0 in ≥ 3/4 hetero runs. **Power note
  (rev C-M5): a fire requires near-complete separation; this
  design has no power against moderate effects — a null is not
  evidence of absence of a moderate interaction.** (Ambient Δ was
  positive 1/8 — the localized prediction is that the hetero field
  flips it; the flat arm is the in-wave control for everything
  non-spatial.)
- **MATCHED-STRENGTH CONTROL (rev C-B1, the Fable-F2 repair):**
  registered scope determinant, adjudicated alongside the primary:
  realized intrinsic(cem_disag) ≥ 0.9 × intrinsic(policy) in ≥ 3/4
  hetero runs AND intrinsic(policy) > intrinsic(random) in all
  hetero runs (degeneracy floor). It does NOT gate the fire; it
  splits the null cell (below). In the ambient wave the CEM lost to
  the policy on its own objectives — a null from an attacker weaker
  than the incumbent licenses only "this optimizer, at this
  strength, could not."
- **REGISTERED SECONDARIES (no fire; pre-declared THIS time, so the
  share axis is registered rather than value-aware):**
  S1 region entry: occ(cem_disag) − occ(policy) per hetero run
  (does the planner ENTER the region the policy avoids?) with the
  flat contrast alongside — **scope note (rev C-m1): in hetero
  runs attr_d is amplitude-weighted, so Δ and region entry are
  mechanically coupled; S1 is a DECOMPOSITION of the primary, not
  independent evidence**; S2 the SHARE estimand: share_d(cem_disag)
  and share_d(cem_distractor) vs share_d(policy), per run, both
  wings (composition steering — observed 8/8 ambient, now
  pre-declared); S3 steerability row on hetero:
  attr_d(cem_distractor) − max(attr_d(policy), attr_d(random));
  S4 per-arm occupancy/intrinsic/level rows + per-wing BCa
  (indicative, n=4) + the de-normalized robustness row.
- **Outcome map:**
  - **LOCALIZED-TRANSMISSION** (fires) — deployment optimization
    cashes the misprice where avoidance leaves it unclaimed; the
    containment story becomes two-regime and Track-D's retirement
    is REOPENED as a registered consequence.
  - **NO-LOCALIZED-TRANSMISSION-MATCHED** (no fire, control
    passes) — a matched-strength attacker could not cash
    registered spatial headroom: the strongest null this wave can
    produce. Tier consequence: supports the containment paragraph
    at **Tier-1** ("greedy H=12 CEM at matched realized strength
    does not exploit the localized channel"); **advancement to any
    Tier-2 wording remains gated on Wave B** (the Fable memo pins
    Tier-2 to the adversarial-actor wave, never to C).
  - **NO-LOCALIZED-TRANSMISSION-WEAK-OPTIMIZER** (no fire, control
    fails) — scoped strictly to "this optimizer at this strength";
    NO tier movement in either direction.
  - NOT-ADJUDICABLE. Either branch composes with A1; the executed
    ambient UNSTEERABLE read stands regardless.

**Tier ladder (pinned here; rungs referenced from the memo + the
ambient RESULTS correction):** Tier 0 = registered null at this
optimizer strength; Tier 0-descriptive = value-aware share/level
disclosure; Tier 1 = local-maximum/matched-strength stationarity
(Waves A + C nulls); Tier 2 = matched-strength security wording —
ships ONLY if Wave B nulls. "Cannot amplify exposure" unqualified
is NEVER licensed by this program.

## 3. Fences & placement

Track-B transmission section (two-regime paragraph) + feeds the
Paper-5 security-paragraph wording ONLY through the ladder above.
The probe is the frozen instrument; no claims about planner
competence beyond its registered constants.

Freeze = review R2 adjudicated (done 22 Aug — 2 BLOCKING + 6 MAJOR
+ 7 minor on this wave, all resolved pre-freeze) + commit of this
file + `uncfield/se_planner_loc_read.py` +
`scripts/se_planner_loc.sbatch` — then smoke → the 8 probe jobs
(post-A2-freeze checkout).
