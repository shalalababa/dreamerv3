# Competence-repair READ — Amendment 1 within-pass shadow pairing (2026-08-02)

ONE registered execution of `analysis.repair_read` (never re-run).
Registration: `prereg/PREREG_competence_repair_20260730.md` +
`prereg/PREREG_competence_repair_amend1_20260801.md` (parent primary
retired unmeasurable-on-substrate after the quarantine audit; amended
primary = within-pass consumer-vs-probe chooser contrast). Outputs
here: `repair.json` (full registered output) + `read_stdout.txt`.

## Provenance (verified pre-read)

- Bundle `local_results/r3_repair_20260802_143054/`: `sha256sum -c
  MANIFEST.sha256` → 0 mismatches (120 files); repo copy
  `manifests/r3_repair_20260802_143054.sha256`.
- Code shas ≡ committed finals byte-for-byte: reader `c92b2fd7…`,
  driver `0c640fd7…`, labeler `6561eff9…`, trainer `2405c44e…`.
- Markers: `REP_GATE_OK`, `REP_DETGATE_OK` (amended semantics),
  `CONSUMER_MODEL_SHA256`; 32/32 repaired passes; consumer model sha
  **`be802f1e…` matches the frozen pin** (model NOT retrained, per the
  registered model-restore rule).
- State-identity violations: 0 (episode/step exact vs committed on all
  32 cells — the component the audit showed job-invariant).
- Reader selfcheck PASS immediately before the read.

## Result

- **P-REP1 (paired [achieved_rep − achieved_probe]):** +0.070
  [−0.092, +0.245], n = 32 — straddles 0. No fire.
- P-REP2 (materiality 0.6465): not adjudicated (P-REP1 did not fire);
  point +0.070 is ~9× below the bar regardless.
- HARMFUL: no (CI not entirely negative).
- **Sensitivity leg (excl. pre-amendment smoke cell cup/e1/31):**
  +0.074 [−0.096, +0.257], n = 31, same fire status — **NOT fragile.**
- **VERDICT: NOT-REPAIRABLE-FROM-OBSERVABLES** — a LINEAR probe on
  stored observables (deter, candidate action, qfull summaries, udyn),
  trained on oracle returns from sibling runs (LORO), does not
  detectably repair the consumer. Scope-limited by registration: NOT a
  claim that no repair is possible.

## Registered secondaries (descriptive)

- achieved_rep +0.011 [−0.159, +0.170]; in-pass achieved_probe −0.060;
  committed cross-era achieved_orig −0.039 [−0.179, +0.113] — all ≈ 0.
- **Chooser change rate 83.3%** — the consumer model re-picks on most
  states; it changes choices without adding value (mirrors the xc
  finding the same day: choice churn ≠ value).
- **Residual gap fraction 99.3%** of the in-pass opportunity — the
  competence gap is essentially untouched.
- Per-domain (cup +0.032, finger +0.109) and per-dose (e1 +0.082,
  e4 +0.059) all straddle.
- Cross-era drift context (gates nothing, exactly as the audit
  predicted): probe-vs-committed realized-choice agreement 32.6%
  (≈ the audit's ~50% flip scale), opportunity max cell |Δ| 97.0
  (O(100) float drift — same magnitude as the xc detprobe's 95).

## Consequence

The R3 chain now reads: opportunity large and robust (3 domains) →
achieved ≈ 0 → maturity doesn't close it (P-R3c null) → the gap is not
state-legible at the stored-observable level (ladder flat) → **a linear
observable-based repair doesn't close it (this read)** → head
transplants don't rescue it either (xc MIXED, same day). The
representational account ("support must be legible cross-link") is
strengthened from the consumer side. Regardless-clauses reaffirmed:
committed reads stand, committed labels never re-read, 24+24 frozen,
Route A closed, imag retired. Any nonlinear/finetuned-repair follow-up
is a NEW dated registration (not queued by default).
