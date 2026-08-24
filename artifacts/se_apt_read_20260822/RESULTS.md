# B2 APT-column registered read — 22 Aug 2026

**Prereg:** `PREREG_nfi_apt_20260821.md` + Amendment 1 (smoke gate
adjudicated 21 Aug; SIGN fire rule; comparator path closed). ONE
execution of the frozen `se_apt_read` on the verified bundle
`b2_apt_20260822_100924` (manifest 13,151/13,151 OK). 8/8 runs
loaded, seeds 68–75, zero invalid, fit OK, ckpt gates OK, dup0
vector-exact zero, no spread rescope, no positive-resample anomalies.

## Outcome: **cell ii — APT-IMMUNE** (primary does NOT fire, and the
null is sharp)

- **Distractor (the single fire channel):** delta_apt_mean negative
  in **1/8** (attained binomial p 0.996). Per-run deltas +0.0013,
  −0.0016, +0.0007, +0.0005, +0.0001, +0.0007, +0.0020, +0.0018 —
  |delta| ≈ 0.05–0.1% of the base APT level (~2.0). Batch-permuting
  the planted irreducible-stochasticity channel does essentially
  nothing to the k-NN particle entropy.
- **Velocity calibration control (the mechanism row):** −0.0278,
  negative in **8/8 runs**, an order of magnitude larger than any
  planted delta — APT DOES price real cross-window coupling. The
  instrument has teeth; the distractor null is a property of the
  objective, not of the mask.
- **Duplicates:** substitution and fresh-resample deltas both ≈ 0
  (|mean| ≤ 2e-4); mechanical spread component nil; dup0 exact-zero
  anchor holds ×8.

## Taxonomy consequence (Track-B matrix)

The channels × proxies matrix now has its full contrasting column:
**k-NN particle entropy is immune to BOTH zero-information
structures** (irreducible stochasticity AND redundancy) while
pricing real coupling — against ensemble disagreement, which farms
irreducible stochasticity at 5.26× (Stage-1, 8/8) yet is
duplicate-immune (exact parity ×2 reads). The accounting pathology
is PROXY-SPECIFIC, not a universal cost of epistemic bonuses — a
natural in-repo objective simply does not have the failure. This is
arguably the matrix's most load-bearing cell: it converts the
Paper-5 phenomenon from "epistemic rewards misprice noise" to
"parametric-uncertainty proxies misprice noise; nonparametric
state-space proxies need not", and it sharpens the repair question
(what does disagreement buy that APT doesn't, and at what pricing
cost?).

Registered scope note: APT here is the replay-anchor PROXY of the
trained functional (training scored imagined rollouts; #29 C-m11),
population-intervention estimand, S=512 pinned.
