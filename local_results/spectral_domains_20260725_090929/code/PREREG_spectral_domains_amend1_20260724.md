# PREREG AMENDMENT 1: spectral pass — θ-inestimable buffers (frozen 2026-07-24)

Amends `PREREG_spectral_domains_20260724.md`. Trigger: the first real
measurement run crashed on a buffer whose loaded sample contains no
(or constant) rewarded frames — the centered cross-moment E[r·x] is
identically zero and no reward direction exists to estimate. The
original instrument asserted instead of handling this case.

## Disclosure (partial outcomes known at this freeze)

A few buffers had already measured successfully before the crash;
their printed rank/λ_need/f lines were SEEN. Not seen: any
cross-domain `compare` verdict (never ran), the failing buffer's
spectrum, and all remaining buffers. This amendment changes ONLY the
handling of buffers that cannot yield an own-θ estimate; the primary
comparison statistic (median variance rank, cup < finger) and the
confirmation rule are UNCHANGED for every buffer the original
instrument could measure. The version gate below forces
re-measurement of the already-seen buffers with identical estimators
(deterministic: same numbers, new metadata fields), so the seen
values constrain nothing that this amendment alters.

## Registered changes (instrument v1_1, selfcheck extended, PASS)

1. **θ-estimability rule:** a buffer yields an own-θ record iff its
   loaded sample has ≥ 50 rewarded frames (MIN_REWARDED) and
   non-constant reward. Otherwise the pass emits a SPECTRUM-ONLY
   record (spectrum, PR, f_rewarded, n_rewarded, diversity; rank/
   λ_need/ratio = null) instead of crashing. f_rewarded ≈ 0 on a lo
   side is itself a registered bookkeeping fact (zero label energy —
   the a² = s²f input to the task-lo floor).
2. **Primary population:** `compare` uses own-θ-estimable records
   only. A matched side with no own-θ-estimable records in BOTH
   domains is DROPPED from the primary and disclosed in the verdict
   (`dropped_sides`); the primary then requires the ordering to hold
   on every remaining evaluable matched side. At least one evaluable
   matched side is required (else no verdict).
3. **Cross-buffer θ secondary (new, descriptive):** own-θ records now
   store θ; `measure --theta_from <same-domain json>` evaluates that
   θ against a sparse buffer's spectrum (θ is a task property, Σ a
   buffer property; feature-space identity asserted — same obs_keys
   and window; cross-domain import refused). These records are
   reported in `secondary_crossbuffer`, never in the primary.
4. **Version gate:** records are stamped `spectral_v1_1_20260724`;
   `compare` refuses any other version — the pre-amendment jsons
   (which lack the estimability fields and would otherwise be
   silently mis-classified) must be re-measured.

## Execution addendum

Re-run `measure` for ALL buffers (including the previously successful
ones) with the amended script; for each θ-inestimable buffer,
optionally add one `--theta_from` pass sourcing the same domain's
reward-richest own-θ json (registered convention: the same-domain
json with the largest n_rewarded). Then one `compare` over the full
v1_1 set, as before. Freeze-commit this amendment + the amended
script BEFORE any v1_1 measurement of a real buffer.
