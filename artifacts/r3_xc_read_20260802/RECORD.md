# Cross-consumer (xc) READ — Amendment 1 within-pass dual-chooser (2026-08-02)

ONE registered execution of `analysis.r3_xconsumer_read` (never re-run).
Registration: `prereg/PREREG_r3_xc_amend1_20260802.md` (freeze commit
`4decd451`; parent `PREREG_r3_amend2_20260730` retired unexecuted after
the determinism probe). Outputs here: `r3_xconsumer.json` (full
registered output incl. per-run values) + `read_stdout.txt` (verbatim).

## Provenance (verified pre-read)

- Bundle `local_results/r3_xc_20260802_135102/`: `sha256sum -c
  MANIFEST.sha256` → 0 mismatches (169 files); repo copy
  `manifests/r3_xc_20260802_135102.sha256`.
- Code shas in the bundle ≡ committed post-review finals byte-for-byte:
  reader `2908a78c…`, labeler `6561eff9…`, driver `772ef3cd…`, prereg
  `05cd8772…`.
- Driver markers: `XC_GATE_OK`, `XC_SMOKE_OK` (2/2 dual smokes),
  `XC_DUALGATE_OK`, `SUBSTRATE_GONE: False`; 64/64 dual labels
  (2 domains × {e1,e4} × seeds 31–38 × {early,late}).
- Reader selfcheck PASS immediately before the read (all guard/verdict
  branches).
- RCC head `52907499…` not in local history (RCC-side commit); binding
  provenance = the byte-exact code-sha match above.

## Result

- **P-XC1 (late eval, consumer = early heads):** +0.028
  [−0.216, +0.244], n = 32 — straddles 0. Matches the registered
  prediction (HEADS-IRRELEVANT direction).
- **P-XC2 (early eval, consumer = late heads):** **−0.155
  [−0.311, −0.032]** — CI entirely NEGATIVE. The registered fire
  direction was positive ("mature heads rescue"); an entirely-negative
  CI is an unregistered-direction excursion.
- Fires: none. Off-direction flags: `xc2_neg = true`.
- **VERDICT (frozen map): MIXED — off-map CI excursion. Disclosed, no
  headline claim, interpretation deferred.**

## Integrity (all clean)

- `integrity_violations: []` across 64 files (g_now identity, chooser
  ranges, candidate drift, finiteness).
- **`cands_max_absdiff_max = 0.0`** — within-process RNG replay
  reproduced every candidate set bit-identically across all 64 passes
  (bound was 1e-4; cross-job floats drifted at O(100) in the probe).
  The within-pass redesign premise is fully vindicated.
- Chooser disagreement 75–85% per side (the head swap genuinely
  changes choices; SUSPECT-NO-OP valve silent).

## Registered secondaries (descriptive)

- Per-domain: cup xc1 +0.133 [−0.063, +0.365], xc2 −0.146
  [−0.313, −0.022] (entirely negative); finger xc1 −0.078
  [−0.510, +0.291], xc2 −0.164 [−0.423, +0.031] (straddles, same
  sign). The XC2 negative direction is domain-consistent; cup alone
  clears zero.
- Opportunity level matched across eval sides (early 1.373, late
  1.364) — the swap changes choices, not opportunity, as designed.

## Interpretation (deferred by registration — notes only)

Mature heads transplanted onto an early-eval agent REDUCE achieved
value relative to the early agent's own heads; immature heads on a
late agent do nothing detectable. Registered-symmetric caveat: the
consumer heads read features from a different encoder era, so
interface mismatch (not value-knowledge deficit) can produce exactly
this sign — the same feature-mismatch lower-bound caveat the parent
registered for HEAD-DEFICIT fires. Coherent with the program's
co-adaptation account (heads × features × occupancy;
"support must be legible"). Any interpretive follow-up (e.g. an
interface-controlled swap) is a NEW dated registration.

## Consequence map (inherited, reaffirmed)

Committed R3/reacher/doubling/second-k/ladder/repair reads stand;
committed labels never re-read; 24+24 frozen; imag retired; Route A
closed. No headline claim enters Paper 2 from this read; the MIXED
verdict + excursion are disclosed wherever xc is described.
