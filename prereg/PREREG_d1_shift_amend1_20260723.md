# AMENDMENT 1 to PREREG_d1_shift_20260723 (2026-07-23, smoke stage)

Registered smoke-fix flow (prereg "Ordering and smoke": any instrument
fix ⇒ dated amendment BEFORE the labels stage). Written after the xpol
smoke failed and BEFORE any shift label exists; no label content has
been produced or seen.

## The failure and the fix (plumbing only)

The xpol smoke crashed with
`Disallowed device-to-host transfer: shape=(512), dtype=BF16` at the
first belief-latent pull. Cause: every agent construction re-arms the
JAX transfer guard (`embodied/jax/internal.py`:
`jax_transfer_guard='disallow'`), and the xpol arm builds a SECOND
agent (the behavior policy) AFTER the `'allow'` line that the R2 run
placed after the first agent's load — so the guard was re-armed for
the labeling loop. Single-agent paths (R2, base, phys) were unaffected,
which is why the R2 wave and its smoke never hit this.

Fix (`d0/oracle_labels.py` `main_real`): the
`jax.config.update('jax_transfer_guard', 'allow')` line is moved to
AFTER the behavior-agent block, i.e. after the LAST `make_agent` call.
No label semantics, dials, meta fields, or the `labeler_version`
change; the behavior=None path is behaviorally identical (the same
update, applied at a later line with no intervening transfers).
Labeler selfcheck re-run PASS.

## Everything else unchanged

Grid, arms, dials, decision rule, consequence map, and the frozen read
are untouched. Ordering resumes at the smoke stage: rerun
`./scripts/d1shift_local.sh smoke` (both arms), then `labels`.
