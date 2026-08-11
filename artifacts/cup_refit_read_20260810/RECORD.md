# Cup regeneration read (P-C3 adjudication) — 2026-08-10

Registration: `prereg/PREREG_cup_refit_20260808.md` (frozen 825de241,
08-08) + Amendments 1–2 (`PREREG_refit_reads_amend{1,2}_20260810.md`).
Reader: `analysis/cup_refit_read.py` (selfcheck PASS; same frozen code as
the finger execution). Bundle: `regen_ridge_e4_20260810_220849`
(sha-verified). ONE execution (`--domain cup`, consuming
`artifacts/finger_refit_read_20260810/read.json`); output `read.json`.

## Gates

32/32 runs (16 task, 16 apt); witness 32/32 True; probeset `cup_v1`
sha ca5948468f39… uniform; wiring correct; measured checkpoints 64/64 at
step 500000. Counter preflight failed → checkpoint-step witness
(Amendment 1). Substrate (Amendment 2): task s0 = original weights,
task s1 = fresh, apt = fresh-named (no comparator). cup_v1 in-regime
AUROC undefined (0 rewarded in-regime frames) ⇒ P-C3 runs on the
registered unqualified pooled AUROC.

## P-C3 verdict: **AMBIGUOUS (registered catch-all — no P-C3 wording licensed)**

- cup Δ_dom = **+0.0015** [+0.0009, +0.0028], perm p = .008
- finger Δ_dom = **+0.534** [+0.483, +0.557] (CI > 0 ⇒ violation leg holds)
- FIRES requires the cup CI to include 0: it does not (by ~0.001 AUROC).
  REFUTED requires cup magnitude overlapping finger: it does not (0.0015
  vs 0.534). ⇒ registered catch-all: AMBIGUOUS, disclosed as
  instrument-limited.
- **[POST-READ observation, no wording licensed]**: the pattern is
  descriptively parity-like — cup task 0.99896 [0.99890, 0.99901] vs apt
  0.99743 [0.99615, 0.99809]: BOTH cup arms carry near-ceiling linear
  reward signal, and the "significant" Δ of +0.0015 (1.5% of the ±0.10
  parity band, ~350× smaller than finger's Δ) is a ceiling-proximity
  artifact of extreme per-fit precision, while finger apt sits below
  chance (0.401). The domain DISSOCIATION the theory predicted is
  numerically enormous; only the literal CI∋0 clause failed. Any parity
  claim would need a registered equivalence-band re-test (TOST-style)
  — flagged as a candidate one-line amendment registration, NOT claimed
  here.

## Replication rider (descriptive)

- Task s0: bit-identical to the archived panel (weight recovery — see
  Amendment 2), not a population draw.
- Task s1 (genuinely fresh draws, byte-identical buffers): regen
  in/out/all = 2.92e-06 / 0.2241 / 0.1893 vs archived 2.13e-06 / 0.2245 /
  0.1897 — **population-consistent** (differences at the 4th decimal on
  identical data; the registered "divergence disclosed" clause has nothing
  to disclose). This is the strongest regeneration-validity evidence in
  the wave.
