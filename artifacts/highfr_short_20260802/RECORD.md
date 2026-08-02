# High-f_R wave — registered curation SHORT (2026-08-02)

Registration `PREREG_highfr_wave_20260802.md` (freeze commit
`39a15880`). The registered curation step ran on RCC AFTER the freeze
commit and came out **SHORT**; per the registration ("either cell
SHORT ⇒ the wave HALTS before any fit and a dated amendment is
required — no outcome exists at that point") the wave halted at the
registered point: **no buffer was built, no fit exists, no transfer
outcome of any kind exists.** All quantities in this record are
registered pre-fit disclosables (pool occupancies only).

## Result (`frew_pairs_SHORT.json`)

- **target 0.80: OK via the registered fallback** — top-200 window,
  realized mean occupancy **0.7293** (≥ FALLBACK_MIN 0.60; inside the
  side1 instrument gate [0.60, 0.85]). The freeze's manipulation range
  f_R ∈ [0.60, 0.80] is satisfied by this cell.
- **target 0.60: SHORT** — after the hi cell removes the top 200
  episodes, the best remaining 200-window mean is **0.4130** (= the
  remainder's top window), missing 0.60 by 0.187 ≫ tol 0.05.
- Pool structure implied: episodes ranked 1–200 average 0.729;
  ranked 201–400 average 0.413 — the high-occupancy tail is ~200
  episodes deep, so two disjoint cells at ≥0.60 do not exist.

## Provenance (verified locally)

- `code.sha256`: all three shas (prereg `bbc8f25a…`, curate_frew
  `49b7cf6f…`, highfr_read `77def16e…`) match the local committed
  post-review files byte-for-byte.
- `rcc_git_head.txt` `39a15880` = the f_R freeze commit itself,
  verified ancestor on `causal-wm-transfer`.
- Ordering held: commit → curation → registered halt. Value-blindness
  intact by construction (curation touches index occupancies only).

## Path forward (queued for GO — f_R wave Amendment 1)

Re-target side0 to the reachable mid window: **target 0.41 (tol
0.05)** — realized will be ≈0.4130 (the remainder's top window,
deterministic). Amended side0 instrument gate [0.36, 0.46]; everything
else unchanged: side1 = the fallback top-200 cell (0.7293, PRIMARY,
gates already satisfied), separation 0.316 ≥ 0.08, P-HF1 untouched,
P-HF3 grid becomes {0.054, 0.3232, ≈0.413, ≈0.729} (a denser
monotone-chain/grid than the original design; side0 changes role from
"second in-range manipulation point" to "mid-grid point between
natural and hi-f" — disclosed in the amendment). The freeze is
unaffected (its [0.60, 0.80] range binds the HIGH cell only).
Alternative rejected: dropping side0 entirely (submit plumbing wants
two sides, and the mid-grid point is what the monotone-rise branch
feeds on). Build = dated prereg amendment + two reader constants +
selfcheck ranges + curation re-run; no new instruments.
