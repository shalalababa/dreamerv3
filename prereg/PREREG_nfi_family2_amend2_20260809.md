# PREREG — NFI Family-2 Read, Amendment 2 (9 Aug 2026, night)

**Amends:** `PREREG_nfi_family2_20260807.md` (freeze `8f6b4de8`) as already
amended by `PREREG_nfi_family2_amend1_20260809.md` (G-STAMP dirty gate +
compensating control).

**Trigger:** v2-delta review findings §4.2–§4.3
(`research_notes/paper5_uncertainty_field/reviews/Review_v2Delta_20260809.md`):
the Amendment-1 dirty gate and its compensating-control checker were
correctly specified but under-enforced against malformed inputs — three
live bypasses (a stamp that OMITS the `dirty` key; a reproduction record
lacking `host_git`; a reproduction record matching on cell names without
member indices).

## 1. Registered gate changes (`uncfield/family2_read.py`)

All changes are **strictly stricter**: any record that passes the hardened
gates passes the previous gates; no decision rule, threshold, or verdict
logic is touched.

1. **G-STAMP dirty-flag presence pin:** every cell stamp must CONTAIN the
   `dirty` key; a stamp lacking it HALTs
   (`G-STAMP: <cell> stamp lacks the dirty flag`). A runner that drops
   the field can no longer bypass the Amendment-1 gate.
2. **`_reproduction_ok` hardening:** the compensating-control record must
   (a) carry a non-empty `host_git`, and (b) contain passing entries for
   the EXACT claim-carrying (cell, member) pairs
   {(dc_gru, 1), (dc_lstm, 3)} — cell names alone no longer suffice.
   Per-entry requirements unchanged (match_names, match_neutral,
   max_rate_dev < 1e-6, control_passes).
3. **Selfcheck fixture battery extended** with three mutants that must be
   DETECTED: reproduction record missing `host_git`; reproduction record
   with a wrong member index; stamp missing the `dirty` flag
   (`dirty_missing` tamper kind).

**Registered boundary (unchanged from Amendment 1):** the reader cannot
verify that the reproducing HOST was clean beyond the recorded
`host_git`; a `host_clean` field is NOT required because the genuine
2026-08-09 `reproduction.json` predates this amendment and does not carry
it — requiring it would retroactively invalidate a genuine record.
Future reproduction records SHOULD stamp `host_clean: true`.

## 2. Validation executed (same session, before this registration's commit)

- `family2_read --selfcheck` PASS (full fixture battery incl. the three
  new mutants).
- The REAL registered read re-executed under the hardened gates:
  **no HALT; P-F2 FIRES; every member and ensemble verdict identical**
  to `artifacts/nfi_family2_read_20260807/` (incl. the Amendment-1
  re-read).
- `confirm_read` full run: all recorded outcomes reproduce (unaffected
  code path; its legacy-trace eval was sandboxed in the same session —
  empty `__builtins__` — with identical output).

## 3. Process disclosure

The code changes were implemented ~hours BEFORE this registration, in the
same session, as part of the v2-delta review remediation
(`artifacts/nfi_review_response_20260809/RESULTS.md` §8). No registered
read consumed the hardened gates between implementation and this
registration; the re-read in §2 is a verification re-run, not a new
registered decision. This amendment brings the frozen-reader edit under
the standing rule ("never edit executed frozen readers" without a
registered amendment), following the Amendment-1 precedent. Verdict
consequences of this amendment: NONE (verified identical).

## 4. Scope

Governs every future execution of `uncfield/family2_read.py` on the
family-2 record or successor bundles. The P-F2/P-F2b decision rules,
thresholds (0.02/0.08/30/8), verdict tiers, and all Amendment-1
provisions remain frozen and unchanged.
