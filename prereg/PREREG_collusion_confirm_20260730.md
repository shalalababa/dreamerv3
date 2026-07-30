# PREREG: Paper-4 collusion confirmatory sweep — RQ2 composition drivers (frozen 2026-07-30)

Registers the confirmatory experiment the three pilot stages and the
30-Jul calibration read prepare
(`artifacts/collusion_stage1_20260725/`,
`artifacts/collusion_stage23_20260725/`,
`artifacts/collusion_calib_20260730/`; design of record
`research_notes/Design_Collusion_Pilot_20260724.md`). Question (RQ2 of
the 2-Jul brief): holding the learner fixed, does punishment-phase
experience — as a COMPOSITION property at matched data volume /
matched exploration budget — drive (a) the supra-competitive profit
level and (b) the punishment impulse-response structure? This file +
`analysis/collusion_confirm_read.py` (selfcheck PASS) are committed
BEFORE any confirmatory session exists.

## Instruments (pinned)

`collusion/pilot.py` `collusion_pilot_v3_20260730` +
`collusion/designb.py` `collusion_designb_v2_20260730`, both selfcheck
PASS at freeze. v3/v2 validated additions-only: the local v3 rerun of
baseline seeds 0–99 reproduces `baseline_v2_sessions100.csv` on all 15
shared columns bit-exactly (calibration §1); the v1 `probe` path is
untouched. Environment/learner constants unchanged from the pilot
(Calvano baseline: α=0.15, β=4e-6, δ=0.95, m=15, ξ=0.1, memory-1,
Calvano Q-init, conv=100K-stable greedy profile, cap 2M).

## Frozen measurement choices (calibration read, pre-outcome)

- **Punishment label (primary): distance form, TERMINAL reference**
  (`punish_occ_dist`: a period is punishment-phase iff the lower price
  sits ≥1 grid step below the session's terminal greedy path minimum).
  Calibration §2: the exogenous (monopoly-reference) variant is
  degenerate — occupancy mean 0.981, 96% of sessions above 0.9, no
  fp/no-fp separation (0.980 vs 0.984), and its graded d=1 arm deletes
  ~99.6% of the stream (level destruction, not regime composition).
  The terminal label is well-behaved (mean 0.495, sd 0.136, no
  saturation); its mild outcome-anchoring (corr +0.70 with Δ_profit;
  reference from the session's own converged path) is DISCLOSED as a
  limitation. Exogenous is a registered descriptive column only.
- **Undercut+W5 label: descriptive only** (stage-2a: saturates 26%,
  inverts against the fingerprint).
- **Fingerprint-conditional analysis**: Design-B primaries computed on
  sessions whose ONLINE run fires the impulse-response fingerprint (fp
  measured pre-intervention ⇒ conditioning is pre-treatment for the
  stream edits). Stage-1 basis: high-Δ-without-fingerprint =
  Abada–Lambin failure-to-learn; collusion claims are
  fingerprint-conditional.

## Design B — offline composition at matched volume (seeds 1000–1099)

100 fresh recorded sessions. Per session: exact-replay anchor, then
for each dose d ∈ {0.25, 0.5, 1.0}: TARGET arm down-weights
distance-labeled punishment transitions to w=1−d; CONTROL arm
down-weights a size-matched random subset to the same w (identical
total effective update mass by construction — asserted in code; the
stage-3 binding lesson that deletion confounds composition with
volume). d=1 is Q-identical to deletion under the distance label
(selfcheck-proven). Retrained Q0 is evaluated against the partner's
online final Q (greedy Δ + impulse response). Calibration §3
disclosure: d=0.25 was behaviorally inert and d=0.5 mostly inert on
Δ_profit in the calibration cohort (tabular argmax readout ⇒
threshold-like dose response) — the ladder is registered as
descriptive structure; the primary sits at the top dose only.

```
python -m collusion.designb graded --sessions 100 --seed0 1000 \
  --doses 0.25,0.5,1.0 --reference terminal --output confirm_designB.csv
```
(≈2 core-h; embarrassingly parallel by any --sessions/--seed0 split —
sessions are seed-deterministic; the reader takes a csv glob.)

## Design A — generation-time intervention (phenomenon-preserving)

Coupled online learning with the exploratory-action DISTRIBUTION
biased at matched exploration budget (ε schedule untouched; both
agents symmetrically): `forbid` = exploratory prices never strictly
below both previous prices (no exploration-generated undercuts);
`force` = exploratory undercut whenever one exists; mix λ =
biased-draw probability. Five arms × 100 fresh sessions:

```
python -m collusion.pilot run --sessions 100 --seed0 2000 --output confirm_designA_base.csv
python -m collusion.pilot run --sessions 100 --seed0 3000 --explore_mode forbid --output confirm_designA_forbid100.csv
python -m collusion.pilot run --sessions 100 --seed0 3100 --explore_mode forbid --explore_mix 0.5 --output confirm_designA_forbid50.csv
python -m collusion.pilot run --sessions 100 --seed0 3200 --explore_mode force --output confirm_designA_force100.csv
python -m collusion.pilot run --sessions 100 --seed0 3300 --explore_mode force --explore_mix 0.5 --output confirm_designA_force50.csv
```
(≈3 core-h total.) Seed ranges are disjoint from every pilot and
calibration cohort (0–99, 200–219, designb 0–19, smokes 0–9).

**Design-A prediction re-specification (disclosed).** The draft
prediction was "forbid lowers Δ_profit". The calibration smoke (§4;
n=10 for the λ=1 arms, n=5 for the λ=0.5 arms, PILOT seeds, run
before this freeze) showed the opposite profit direction with a
collapsed fingerprint: forbid λ=1 ⇒ Δ +0.987 vs baseline +0.844 with
fingerprint 0/10 and occupancy 0.000 — an induced failure-to-learn
regime. The registered primaries were therefore re-specified BEFORE
freeze as a DISSOCIATION; this ordering (smoke → re-specification →
freeze) is the disclosure. Two further smoke facts are disclosed
rather than modeled: the forbid λ=1 dynamics were DEGENERATE
(all 10 sessions bit-identical Δ, sd 0, convergence at the 100K
window minimum — a deterministic price ratchet); if that replicates
at n=100 the forbid side of the two-sample CIs contributes ~zero
variance and the CIs are driven by baseline spread (valid,
disclosed). And the forbid arm collapses coverage_late (0.004)
jointly with occupancy: the manipulation is blunt, so the registered
claim is the coarse generation-time one, with the coverage confound
audited as a secondary, never partialled out.

## Registered read (frozen `analysis/collusion_confirm_read.py`)

Bootstrap B=10K, `default_rng(0)`, 95% percentile CIs. Strict csv
hygiene: non-finite cells and non-canonical booleans HALT the read
(loader/parse asserts), they never adjudicate.

**Primaries (the only confirmatory claims):**

- **P-CLB1**: fingerprint-conditional paired increment at top dose,
  mean[dp_punish_d100 − dp_random_d100]; **FIRES iff CI entirely < 0**
  ⇒ punishment-phase experience drives the profit level
  composition-specifically (beyond volume).
- **P-CLB2** (co-primary): same for the fingerprint indicator; FIRES
  iff CI < 0 ⇒ composition carries the punishment STRUCTURE.
- **P-CLA1**: forbid(λ=1) − baseline on the FINGERPRINT rate,
  two-sample; **FIRES iff CI < 0** ⇒ punishment-generating exploration
  is necessary for punishment structure.
- **P-CLA2** (co-primary): forbid(λ=1) − baseline on Δ_profit; **FIRES
  iff CI entirely > 0** ⇒ the price level is maintained/raised while
  the structure collapses — the induced failure-to-learn half of the
  dissociation.

No multiplicity correction across the four (disclosed); each verdict
is stated separately and none is decisional for another.

**Gates (verdict-blocking, in order):**

- **G-CLEX**: exact_replay on ALL 100 Design-B sessions; any failure ⇒
  INSTRUMENT HALT, no verdicts.
- **G-CLBASE**: both baselines reproduce the phenomenon (mean Δ_profit
  ∈ [0.70, 0.90] AND fingerprint rate > 0.5), else
  INCONCLUSIVE-COHORT — no confirmatory verdicts.
- **G-CLFP**: fingerprint-conditional n ≥ 30, else POPULATION FLOOR.
  DISCLOSED REDUNDANCY: on the registered n=100 path G-CLBASE already
  implies n_fp ≥ 51, so G-CLFP cannot trip alone; it is retained as a
  guard for any future amended cohort.
**Claim-specific gate:**

- **G-CLA-MANIP**: forbid(λ=1) lowers `punish_occ_dist` (CI < 0), else
  P-CLA1/2 are reported MANIPULATION-FAILED (uninterpretable, not
  negative).

Archival semantics (must match the reader exactly): under G-CLEX
nothing is computed (instrument halt, no estimates); under
G-CLBASE/G-CLFP failure all four primaries carry `issued=false` and
`fires=null` with their estimates still archived; under G-CLA-MANIP
failure P-CLA1/2 carry `manipulation_failed=true` and `fires=null`.
A fire flag (`true`/`false`) is only ever emitted for an issued,
interpretable claim.

**Registered secondaries (never decisional, all implemented in the
frozen reader):** full dose ladder I(d) (fp-conditional +
unconditional) + monotone-direction flag; unconditional top-dose
twins; Δ_price twins (`dpr_*` columns / Design-A `delta_price`);
force and λ=0.5 arms (force = sufficiency direction, directional
only); per-arm occupancy + coverage_late shifts (the Design-A
confound audit); convergence rates (no session excluded on
convergence — pilot precedent, avoids selection); exogenous-label
descriptive summary; RQ1-observational midrank-Spearman associations
on the Design-A baseline (the correlational complement the causal
arms supersede).

## Power (disclosed, calibration §5)

P-CLB1: fp-cond paired sd at top dose 0.287 (n=20 calibration cohort)
⇒ expected n_fp≈68 detects ≈0.097; calibration point −0.134;
unconditional n=100 detects 0.080. P-CLB2: fp-cond calibration I_fp
−0.500 (sd 0.650) ⇒ n_fp≈68 detects ≈0.221. P-CLA1/2: two-sample
100v100 at baseline Δ sd 0.122 detects ≈0.048; smoke-magnitude
fingerprint effects (−0.80) are far above any power concern.

## Consequence map (frozen)

- **P-CLB1+P-CLB2 fire** ⇒ punishment-phase experience is a
  composition-specific causal driver of both outcomes at matched
  update mass — the RQ2 composition leg lands; RQ3 (coupling wave) is
  authorized as the next registration.
- **P-CLB1 only** ⇒ profit level tracks composition, structure
  survives ⇒ structure-from-elsewhere discussion (Abada–Lambin
  adjacent); RQ3 still authorized.
- **P-CLB2 only** ⇒ the punishment structure tracks composition while
  the profit level survives — the composition claim lands on the
  collusion signature itself (arguably the stronger mechanism
  reading); RQ3 still authorized.
- **Neither Design-B primary** ⇒ registered negative: at matched
  volume, punishment-phase share is NOT the driver — the paper's
  empirical core becomes the volume-vs-composition decomposition plus
  the Design-A results; no re-run at other doses without a new dated
  amendment.
- **P-CLA1+P-CLA2 fire** ⇒ the dissociation is confirmed at n=100:
  punishment-generating exploration is necessary for GENUINE collusion
  (structure) while its removal produces collusion-mimicking
  supra-competitive prices — the paper gains the induced
  failure-to-learn exhibit and the two-outcome measurement argument
  becomes load-bearing.
- **P-CLA1 without P-CLA2** ⇒ structure collapse without the price
  rise: the necessity claim stands, the mimicry claim does not.
- **P-CLA2 without P-CLA1** ⇒ outside the registered pattern; new
  registration required for any interpretation.
- **Design B and Design A both null** ⇒ registered double negative:
  neither replay composition nor generation-time exploration bias
  moves either outcome — the experience-composition hypothesis for
  collusion fails at this scale and the scope note says so.

## Disclosure and ordering

Known at freeze: all three pilot stages, the β sweep, and the full
30-Jul calibration read (baseline v3 rerun seeds 0–99, graded ladder
seeds 0–19 under BOTH references, Design-A smokes forbid/force ×
λ∈{1, 0.5} — all on pilot seed ranges, disjoint from every
confirmatory cohort; one calibration job re-run after a silent death,
disclosed in the artifact). The Design-A primary directions were
re-specified from the smoke BEFORE this freeze (see the Design-A
section). Unknown: every confirmatory-seed quantity. Ordering: this
file + `analysis/collusion_confirm_read.py` + harness v3/v2 +
`artifacts/collusion_calib_20260730/` committed together BEFORE
submission; ONE read, in the Paper-4 owning chat, when all six csvs
are complete (all-or-nothing per cohort, loader-enforced).
