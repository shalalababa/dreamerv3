# PREREG Amendment 1 — NFI family-generality factorial (9 Aug 2026)

Amends `PREREG_nfi_family2_20260807.md` (freeze `8f6b4de8`). Occasioned
by the 9-Aug full-record review (§14): the parent prereg's §5 review
record states "a dirty or wrong-commit stamp fails the G-STAMP gate,"
but the frozen reader implemented only stamp-uniformity + threshold
pins — the dirty check was never coded, all three landed stamps carry
`dirty: true` (RCC checkout; untracked scratch/job files are the
expected cause but the flag cannot distinguish them from modified
tracked files remotely), and the 7-Aug read therefore proceeded where
the registered gate as worded would have HALTED. This amendment
regularizes that gate rather than leaving it aspirational.

## Amended G-STAMP rule (binding from this amendment's commit)

G-STAMP = (a) identical `stamp.git` across the three cells, within the
freeze-descendant set with empty decision-path diff; (b) pinned
thresholds (0.02/0.08/30/8); (c) **dirty flag: any `stamp.dirty: true`
HALTs the read UNLESS the compensating control is on record** at
`artifacts/nfi_family2_read_20260807/reproduction.json`:

> Clean-HEAD end-to-end model-level reproduction of at least one
> conjunction-bearing member per dc cell (dc_gru m1, dc_lstm m3): a
> full `search_exploits` re-run from the bundled `ensemble.pkl` on a
> clean checkout, **executed device-matched to the registered runs
> (`JAX_PLATFORMS=cpu`)**, must match the bundled npz on `names` and
> `neutral` exactly and on every rate column to max dev < 1e-6.
> (Execution note, disclosed: the first control attempt ran on the
> local GPU and deviated at 1.68e-6 — the documented jax GPU-vs-CPU
> effect — while reproducing names/neutral bitwise, true_rate at
> 5.3e-15, and both member verdicts + conjunction counts exactly; the
> gate correctly HALTED on it, which is itself evidence the gate is
> live. The threshold is not loosened; the control is device-matched.)

Rationale: the reproduction bounds precisely what a dirty remote tree
could have corrupted that the (committed-tree) decision-path diff and
G-REPRO cannot — a modified `planner.py`/`dcfield.py`/`dcwm.py` on the
executing host changing how rates or neutrality were produced. It does
not cover training/episode generation (disclosed residual; episode
generation is seed-deterministic and the dc episode sha is G-DATA-
pinned). The 9-Aug review independently executed this control for
dc_gru m1 (bitwise names/neutral, rates at float64 round-off) before
this amendment codified it.

## Not amended

No decision rule changes: P-F2, P-F2b, the pinned thresholds, the
verdict tree, G-REPRO, and G-DATA are untouched. The 7-Aug verdicts are
expected to reproduce identically under the amended gate; the amendment
exists to make the gate real, not to alter outcomes.

## Disclosure ledger

- The original read's G-STAMP handling (uniformity + pins only, dirty
  disclosed post-hoc) is recorded in the 9-Aug amendment prepended to
  the read record.
- G-REPRO's boundary, stated explicitly (review §15): it re-derives
  verdicts from the runner's saved rate columns; it can NOT catch a
  remote mutation of how those rates were computed — that is what the
  compensating control above is for — nor of EPS_TRUE/STEADY/N_BURN
  as used at run time (pinned via the stamp instead).
- Reader selfcheck hardened same day against gate/decision mutants at
  the `run_read` level (fixture battery; the 9-Aug review's 17
  surviving mutants incl. the CIG_LEVEL primary flip are now killed).
