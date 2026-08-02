# PREREG high-f_R wave Amendment 1: side0 re-target 0.60 → 0.41 (frozen 2026-08-02)

Amends `PREREG_highfr_wave_20260802.md` (freeze commit `39a15880`).
Executes that registration's own halt clause: the registered curation
ran post-commit and came out SHORT
(`artifacts/highfr_short_20260802/`, provenance verified — code shas
match the committed post-review files, RCC head = the freeze commit):

- side1 (target 0.80): OK via the REGISTERED fallback — top-200
  window, realized mean occupancy **0.7293**, inside its instrument
  gate [0.60, 0.85] and inside the theory freeze's manipulation range
  f_R ∈ [0.60, 0.80] (`PREREG_highfr_theory_20260730`, which binds
  the HIGH cell only).
- side0 (target 0.60): SHORT — the pool's high-occupancy tail is
  ~200 episodes deep; after side1 removes the top 200, the best
  remaining 200-window mean is **0.4130** (miss 0.187 ≫ tol 0.05).
  Two disjoint cells at ≥ 0.60 do not exist in this pool.

The halt fired BEFORE any buffer/fit: no transfer outcome of any kind
exists; every number above is a registered pre-fit disclosable (pool
occupancies only).

## Amended design (everything else unchanged)

- **side0 target: 0.41** (tol 0.05 unchanged). Deterministic
  consequence, knowable in advance: the same descending search
  reproduces side1 exactly (fallback top-200, 0.7293 — the window
  rule and pool are unchanged), then side0 = the remainder's best
  window at dev 0.003, realized mean **≈ 0.4130**.
- **side0 instrument gate: f_rewarded ∈ [0.36, 0.46]** (was
  [0.55, 0.65]). The separation gate (side1 − side0 ≥ 0.08) is
  retained as a belt but is now vacuously satisfied by the ranges
  themselves (minimum gap 0.60 − 0.46 = 0.14); its selfcheck leg is
  replaced by range-exclusion legs on both sides of the amended
  window.
- **side0 role, re-registered honestly:** side0 is no longer a second
  in-range manipulation point (the pool cannot supply one disjointly);
  it is a MID-GRID point between the natural buffer (0.3232) and the
  hi-f cell — it feeds the P-HF3 grid, the monotone-rise chain, and
  the descriptive secondary, exactly as before, and remains never
  load-bearing for the primary. The P-HF3 grid becomes
  **{0.054, 0.3232, ≈0.413, ≈0.729}** — a denser monotone chain than
  the original design.
- UNCHANGED: side1 cell and its gates; the PRIMARY (P-HF1, sole
  confirmatory) and both registered branches (monotone-rise
  refutation; saturation); P-HF2 in full; all cohorts, seeds, config,
  bootstrap machinery, comparators, power statement; the
  source-composition rider; the E4 pass; every command except the
  curation targets:

```
python -m probing.curate_frew search-frew \
  --index $RUNROOT/axis1_finger/episodes.json \
  --n_episodes 200 --targets 0.41 0.80 --tol 0.05 \
  --output $RUNROOT/axis1_finger/frew_pairs.json
```

(This overwrites the SHORT json on scratch; the SHORT run is archived
in `artifacts/highfr_short_20260802/`.) Build → spectral measure →
gates → submit → E4 → ONE read proceed verbatim from the parent
registration.

## Code changes (this amendment)

`analysis/highfr_read.py` only: `F_LO_RANGE = (0.36, 0.46)` +
docstring/fixture updates + the selfcheck range legs (selfcheck PASS;
sha in the ledger). `probing/curate_frew.py` is UNCHANGED (targets are
CLI arguments; `FALLBACK_MIN` applies to the highest target only).

## Registered scope note: grid densification is OUT of this amendment

A 0.60-mean window DOES exist in the full pool (contiguous windows
straddling the tail boundary) — but only OVERLAPPING side1's episodes.
Whether to add such non-disjoint mid-grid cells (or collect new
high-occupancy episodes) is an OUTCOME-CONTINGENT follow-up decision:
it earns its cost only if the falling limb fires and a peak-location
estimate is wanted for the flagship's support-breadth axis — a claim
the theory freeze explicitly does NOT make (P-HF3 is the weak form; no
peak location is frozen). Any densification is a NEW dated
registration; nothing in this wave's decision rules depends on it.

## Disclosure and ordering

Known at this freeze: everything through 2026-08-02 including the
SHORT diagnostics above and side1's realized occupancy 0.7293 (a
registered pre-fit gate quantity that passed its unchanged gates —
knowing it cannot steer side0's re-target, which is forced by the
pool's structure and lands at the only reachable value). Still no
buffer, no fit, no adaptation, no E4 row, no transfer outcome exists.
Ordering: commit this file + the amended reader → curation re-run →
build → spectral → gates → fits/adapts → E4 → ONE read
(`analysis.highfr_read`, once).
