# PREREG: cup parity equivalence re-test (TOST successor to P-C3) — 2026-08-10

User GO 2026-08-10. Successor registration to `PREREG_cup_refit_20260808.md`,
whose P-C3 read executed 2026-08-10 with verdict **AMBIGUOUS by the letter**
(`artifacts/cup_refit_read_20260810/`): the FIRES branch required the cup
Δ_dom CI to include 0, and it excluded 0 by ~0.001 AUROC at near-ceiling
per-fit precision. The registered catch-all correctly licensed no wording —
"CI includes 0" is a null-hypothesis-significance criterion misused as an
equivalence criterion, and this registration replaces it with the standard
equivalence form on the SAME frozen band.

## Honesty block — what is known at freeze (VALUE-AWARE, disclosed)

This is a post-outcome registration. Known: the executed read's cup
Δ_dom = +0.0015, 95% BCa CI [+0.0009, +0.0028]; finger Δ_dom = +0.534
[+0.483, +0.557]; all arm levels. Given these, the TOST outcome is close
to deterministic. The value of registering is NOT blinding — it is (a)
making the licensing rule for any future "parity" wording explicit,
citable, and non-weakenable, (b) holding the finger-violation leg to a
STRICTER bar than the original (CI entirely > the band edge +0.10, not
merely > 0), and (c) keeping the original P-C3 verdict untouched
(AMBIGUOUS stands in the record; this is a successor claim, never an
overwrite). No new compute; no new measures; the same bundle, gates, and
amendments (`PREREG_refit_reads_amend{1,2}_20260810.md`) apply verbatim.

## Registered rule (P-C3-EQ)

Inputs: the SAME per-fit pooled AUROC values the executed reads consumed
(bundle `regen_ridge_e4_20260810_220849`, loaded by the frozen
`analysis.cup_refit_read.load_rows`, identical gates). Pairing and
clustering identical to the executed reads: Δ per (side, seed) cell,
seed clusters (n=16 pairs, 8 clusters), B=10K, rng 0.

- **Equivalence band: ±0.10 AUROC** — the band frozen in
  `PREREG_cup_refit_20260808` (|Δ_dom| ≤ 0.10), not chosen today.
- **Cup leg (TOST at α=.05)**: BOTH of
  1. the **90%** seed-cluster BCa CI for cup Δ_dom lies entirely within
     (−0.10, +0.10);
  2. both one-sided cluster-level sign-flip permutation tests reject:
     p(Δ ≥ +0.10) < .05 and p(Δ ≤ −0.10) < .05 (shifted sign-flip,
     cluster flips; resolution floor 1/257 at 2^8 patterns, disclosed).
- **Finger leg (violation, STRICTER than the original)**: the finger
  Δ_dom **95%** BCa CI lies entirely **above +0.10** (the parity band
  edge) — cup must be equivalent-within-the-band while finger exceeds
  the band outright.
- **P-C3-EQ ESTABLISHED** iff cup leg AND finger leg both pass ⇒
  licensed wording: "cup apt features support reward prediction
  equivalent to cup task features within ±0.10 AUROC (TOST), while the
  finger contrast exceeds the band" — the parity-with-violation
  dissociation, equivalence-grade.
- **NOT-ESTABLISHED** otherwise ⇒ no parity wording; AMBIGUOUS remains
  the operative P-C3 record.

Reader: `analysis/cup_parity_tost.py`, frozen with this file, selfcheck
before the read. ONE execution. No reviewer (20-line successor on the
just-executed frozen loader; skip disclosed per standing policy).
