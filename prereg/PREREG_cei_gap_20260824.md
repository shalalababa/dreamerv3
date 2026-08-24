# PREREG — CEI graded-access pipeline (GAP), 24 Aug 2026 (rev 2:
one Opus review, 2 BLOCKING / 5 MAJOR / 6 minor applied pre-freeze;
rev 1 never froze or ran)

**Provenance:** the CEI algorithm-ideation panel's recommended build
(`reviews/Algorithm_Ideation_CEI_20260820.md` §6: candidates #1 + #2
merged; user GO 24 Aug). **Fence (the panel's own):** the deliverable
is a PROCEDURE — the staging + certificate layer over the access
lattice — never a new test (Wald owns the SPRT, KCG the bound,
Mehra/Bar-Shalom innovation monitoring; the manuscript concedes each
in print). Repairs review items H7 (the operator class is
OPERATIONALLY defined: operators whose full L3 query behaviour —
every sensed key's score path — conforms to a single-r̂-vector exact
Kalman; the class over which Stages B/C's guarantees hold), H12, and
"lattice asserted then abandoned". The itemized access-cost ledger
(below) is the panel's #7 (logging spec / read-floor conservation)
fused in: every access the pipeline consumed is counted, per tier.
The panel's two manuscript prerequisites (B4 zero-row fix, B1
coupling-gloss fix) are WRITING-CHAT edits on the same critical path
and are queued there; they do not gate this instrument.

## Instrument

`uncfield/cei_gap.py` (numpy-only; imports the executed cei2
machinery unchanged). Stages: **A** = L2 replay-coherence screen +
the L3 counterfactual score-query battery: identifies the declared
r̂ per sensed key (2 queries/key) and checks conformance of EVERY
sensed key's score path against an exact Kalman at the identified
vector (review #3: an s7-only comparison is blind outside the z2
marginal; under the all-key form, parity-refusal is a property —
0/20 conformance seeds hide it — not seed luck, review #4).
**Access scope, disclosed (review #7): L3 includes instantiating
the operator at its declared prior** (fresh-copy query access);
querying only the deployed mid-trajectory operator is NOT claimed.
The battery touches no world data; its world-independence (zero
falsification power) is asserted by re-running it under worlds with
s7 noise ∈ {0.01, 1.0, 48} and requiring bit-identical output
(review #6). **B** = the count-only power certificate:
`power − size ≤ √(κ·n/2)`, κ = κ(γ) of the REGISTERED lie class, on
the passive log's s7 count alone; the realized passive LR
(declaration r̂ vs r̂(1+γ)) decides only when the ceiling allows the
target AND clears ln 19 — otherwise escalate (vacuity never
certifies). The ln 19 criterion is a confidence convention and
carries no per-verdict error control; in this suite it only gates
escalation, and the flip arm's ~0.2% REJECTED-PASSIVE probability
under the honest law is disclosed (review #15). **C** = the priced
top-up: BFS walk + a Wald SPRT between the DECLARATION (H0: world
noise = r̂) and H1 = r̂(1+γ), data from the real world. Wald's error
guarantee applies when the world lies inside the pair; for a world
far outside it (the suite's 100× liar) the SPRT is a decision
procedure whose speed comes from the ACTUAL per-read KL (≈47/read),
not the registered γ — disclosed in the headline (review #12).

**Registered constants (in code):** δ = 0.05, target power = 0.95,
γ = 0.2 (the manuscript's pilot point; κ(0.2) = 0.00783, Wald bound
d(.05,.95)/κ = 338.4 — rem:constants; the manuscript's sequential
simulation ≈349, the panel's 20-Aug re-verification 346.56, review
#8), T_ROLLOUT = 1000, **flip arm** = an HONEST operator in a
HIGH-NOISE world (world R₇ = declaration r̂₇ = 48, T = 1200): under
this policy the passive count is deterministic at n = ⌈19·r̂₇⌉, so
no honest operator in the R₇ = 1 world can ever reach the n ≥ 207
ceiling flip (review #1 — rev 1's long-horizon arm was dead on
arrival; reads stop forever at n = 19). Suite = {U_true, U_false
(r̂₇ = 0.01), ParityContractKalman} + the flip arm. n_trials = 4000,
n_mc = 200000, all RNG seeds literal in code. `--light` writes to
`report_light.json` and can never consume the registered guard
(review #2).

## Registered outcome cells and bars (ONE execution of `--run`;
in-code guard refuses a second)

- **C-GAP-STAGED (primary):** verdict vector = (U_true →
  CERTIFIED-PRICED; U_false → CAUGHT-PRICED with mean forced reads
  ≤ 12 and walk 3; Parity → FLAGGED-FREE with 0 forced reads). Any
  deviation names the cell INSTRUMENT-DEFECT; no re-run without a
  dated amendment.
- **C-GAP-SATURATE:** the pilot SPRT (1.0 vs 1.2, data 1.0, seed
  12): mean reads ∈ [338.4, 389.2] (≤ 1.15× the Wald bound),
  err_rate ≤ 0.065 (Wald's nominal α ≤ 1/19 = 0.0526 + 3σ MC at
  4000 trials ≈ 0.063; review #9), n_capped = 0. Fixed-sample
  comparator m*(0.95) ∈ [600, 710]; `fixed_over_sequential` ≥ 1.7
  (computed in the report, review #11). **Reconciliation rule
  (review #14):** seed-12 `sprt_pilot` is THE C-GAP-SATURATE
  estimand; the same quantity inside the seed-11/16 pipeline rows is
  a descriptive price estimate — discrepancies beyond MC noise are
  flagged descriptively and carry no adjudication weight.
- **C-GAP-CERT-SOUND:** zero `violated` rows over
  n ∈ {1, 4, 16, 64, 100, 150} — the grid where the ceiling BINDS
  (review #10: the 1.0 cap made n = 256/653 vacuous; grid moved
  pre-freeze).
- **C-GAP-FLIP:** the flip arm's certificate flips:
  `ceiling_allows_target = True` at its realized count
  (deterministic, n = ⌈19·48⌉ = 913 ≥ 207; anything else =
  INSTRUMENT-DEFECT). The arm's VERDICT row is DESCRIPTIVE: at
  n = 913 the honest-law P(|LR| ≥ ln 19) ≈ 0.88 (review #15
  measured), so CLEARED-PASSIVE is expected but a ~12% escalation
  to an honest stage-C certification (in the R₇ = 48 world) and a
  ~0.2% REJECTED-PASSIVE are within-design outcomes, reported as
  realized.
- **Descriptives:** the E4 m*-ladder under the matched ML criterion
  (expectation 0.01→3, 0.1→7); the itemized access-cost ledger per
  operator (L3 queries counted as executed + L2 declared-path
  length + forced reads + walk — review #5: no hardcoded counts);
  κ_actual of the suite lie (world-side commentary).

## Value-aware disclosure (review #17)

The reviewer, verifying bars mechanically pre-freeze, executed
several registered computations at their registered seeds and
reported: sprt_pilot mean 355.4175 / err 0.036 / capped 0; fixed
m* = 649; the ladder 3/7/12/54; U_false stage-B LR 0.00057; the
soundness rows at the OLD grid. All pass their bars AS WRITTEN
BEFORE the disclosure; no bar has been moved toward a disclosed
value (the one grid change, C-GAP-CERT-SOUND, moves to cells whose
values are NOT yet computed). The rev-1→rev-2 diff is fully
reviewer-driven and pre-outcome for every adjudicated bar.

## Scope (registered disclosures)

1. One-sided lie class (over-trust: under H1 the world is (1+γ)r̂,
   so r̂ < R); the two-point form is the theorem's own
   composite-to-boundary reduction.
2. Stage A inherits cei2's scoped blind spots verbatim (review N4):
   constant incoherence absorbed by the fit; a Q-liar flagged but
   misdiagnosed; the mean channel out of view.
3. The soundness exhibit is over the LR-threshold test class on the
   marginal log (the bound dominates the marginal by convexity); an
   exhibit of the theorem, not a new bound.
4. The licensed headline: *certifying an honest declaration against
   a γ = 0.2 lie costs ≈ 350 forced reads (saturating Wald);
   refuting THIS coherent liar costs ≈ 3 forced reads + 3 walk
   steps — fast because its actual lie leaks ≈ 47 nats/read, far
   beyond the registered γ; incoherence is caught for free — and
   the pipeline knows which case it is in before paying.* No
   deployed-regime claim: everything is the known-noise-family,
   frozen-R̂ regime (the manuscript's declared vulnerability
   stands).

## Process

ONE Opus review pre-freeze (done; this rev). Selfcheck must PASS
post-revision. The registered run executes locally (CPU, ~3–5 min
measured by the reviewer; review #13) exactly once; `report.json`
is the record; RESULTS.md + ledger entry follow in the same
session.
