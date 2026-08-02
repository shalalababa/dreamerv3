# PREREG high-f_R wave Amendment 2: chunk-availability filter (frozen 2026-08-02)

Amends `PREREG_highfr_wave_20260802.md` (freeze `39a15880`) as amended
by `PREREG_highfr_wave_amend1_20260802.md` (freeze `be60afbe`).

## What happened (post-Amendment-1 execution facts, 2026-08-02)

The amended curation ran on RCC and both sides came out OK, in-gate,
pre-fit:

- side1 (target 0.80): mean **0.7293**, registered fallback —
  reproduces the SHORT run exactly, as predicted.
- side0 (target 0.41): mean **0.4098**, dev 0.0012, no fallback —
  inside its amended gate [0.36, 0.46]. **Correction to Amendment 1's
  prediction note:** Amendment 1 predicted ≈0.4130 by assuming the
  SHORT diagnostic's window would be re-selected; that number was the
  remainder's MAX-mean window (closest to the old target 0.60). The
  registered rule minimizes |mean − target| and found a closer window
  at 0.4098. Every registered constant (target, tol, gate, window
  rule) was honored; the misprediction has no decision consequence and
  is recorded here for the audit trail.

Then the frozen builder FAILED pre-fit:

- `FileNotFoundError` on a donor replay chunk
  (`pilot_p2e_finger_seed1/replay/20260701T215001F…-251.npz`).
- User preflight over the emitted pair: **needed_chunks = 2046,
  missing_chunks = 26**, affected donors include
  `pilot_p2e_finger_seed1` (full list archived in
  `artifacts/highfr_chunkgap_20260802/`).
- Missing chunks are dated 2026-07-01. Consistent cause: scratch purge
  of chunk files never read since collection — prior q1 curations
  materialized only their selected episodes' chunks (refreshing access
  times); this wave's tail selects episodes never before built, whose
  chunks sat untouched for a month. Cause is a FILESYSTEM-AVAILABILITY
  fact. No frames were fit, no adaptation ran, no transfer outcome of
  any kind exists or was read. The failed build's partial output (side0
  was written before the crash) is deleted before rebuild.

## Amended design (everything else unchanged)

The registered curation command gains `--require_chunks`
(`probing/curate_frew.py`, this amendment's only code change):

- **Pool definition:** the search pool is restricted to episodes whose
  covering chunk files ALL exist on disk at curation time. The
  covering-chunk rule is byte-identical to the builder's `load_span`
  (searchsorted over cumulative stream lengths), so a filtered
  selection is buildable by construction.
- **eid contract preserved:** the index is NEVER rewritten — episodes
  are dropped from the search pool only, so eid numbering (what the
  frozen builder resolves `members` against) is unchanged.
- **Auditable:** excluded eids (with their missing files) and the
  missing-file list are recorded in the output pairs json
  (`availability` block) and `criteria.n_excluded`.
- **Everything re-arms unchanged on the filtered pool:** targets
  (0.41, 0.80), tol 0.05, FALLBACK_MIN 0.60 (highest target only),
  SHORT ⇒ halt + dated amendment, disjointness by removal,
  determinism. If the filtered pool cannot support a side, the
  registered halt fires again.
- Realized means will shift relative to the unbuildable pair; the
  registered instrument gates bind exactly as before: side1
  f ∈ [0.60, 0.85] else the read refuses (theory freeze's [0.60, 0.80]
  binds the high cell); side0 f ∈ [0.36, 0.46] else excluded — never
  fatal.

Amended curation command:

```
python -m probing.curate_frew search-frew \
  --index $RUNROOT/axis1_finger/episodes.json \
  --n_episodes 200 --targets 0.41 0.80 --tol 0.05 \
  --require_chunks \
  --output $RUNROOT/axis1_finger/frew_pairs.json
```

Build → spectral → gates → submit → E4 → ONE read proceed verbatim
from the parent registration.

## Code changes (this amendment)

`probing/curate_frew.py` only: `missing_chunk_eids()` +
`search(..., exclude=)` + `--require_chunks` + selfcheck legs that
REPRODUCE the field failure end-to-end (delete a donor chunk covering a
picked episode → flagged set matches an independent interval-overlap
oracle → unfiltered build raises `FileNotFoundError` → filtered search
avoids every flagged eid, is deterministic, and its pair builds clean
through the frozen builder with matching recomputed occupancies —
proving eid numbering intact). Selfcheck PASS; sha in the ledger.
Builder (`build_controlled_replay.py`) and reader (`highfr_read.py`)
UNCHANGED. No fresh reviewer (availability filter on the just-reviewed
instrument, field failure reproduced in the selfcheck, per the standing
reviewer policy).

## Value-blindness and ordering

The filter consults file existence and index occupancies only — no
frames, no fits, no outcomes. Ordering: commit this file +
`curate_frew.py` → on RCC: delete the partial `q1f` build output,
archive the unbuildable `frew_pairs.json` + missing-chunk list into
`artifacts/highfr_chunkgap_20260802/` → curation re-run with
`--require_chunks` → build → spectral → gates → fits/adapts → E4 →
ONE read (`analysis.highfr_read`, once).

## Donor-longevity note (non-binding)

A successful build makes the WAVE donor-independent (the builder writes
fresh self-contained chunks). But the outcome-contingent grid-
densification option (Amendment 1 scope note) depends on this pool's
donor chunks surviving on scratch. If that option should stay open,
copy the donor replay dirs off scratch before the purge takes more;
storage-cost decision is the user's.
