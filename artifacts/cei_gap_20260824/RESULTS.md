# CEI graded-access pipeline (GAP) — REGISTERED RUN, 24 Aug 2026:
**ALL FOUR CELLS PASS — the access lattice is operationalized**

**Prereg:** `PREREG_cei_gap_20260824.md` rev 2 (one Opus review,
2 BLOCKING / 5 MAJOR / 6 minor applied pre-freeze; value-aware
disclosure of reviewer-executed bar checks carried in the prereg).
**Instrument:** `uncfield/cei_gap.py`
(sha256 43deb93f…, recorded at execution; prereg sha 50f8be9f…),
selfcheck PASS post-revision; ONE execution of `--run` (guard
enforced; `--light` writes to its own path), local CPU ~4 min.
Record: `report.json` (copied here from
`local_results/uncfield/cei_gap/`).

## C-GAP-STAGED (primary) — PASSES

| operator | verdict | free tier (L3 queries / L2 path) | priced tier (forced reads + walk) |
|---|---|---|---|
| U_true (honest) | **CERTIFIED-PRICED** | 350 / 1001 | 345.9 + 3 |
| U_false (coherent liar, r̂₇=0.01) | **CAUGHT-PRICED** | 350 / 1001 | **2.7 + 3** |
| ParityContract (incoherent) | **FLAGGED-FREE** | 350 / 1001 | **0** |

The registered verdict vector exactly. The licensed headline holds
with realized numbers: *certifying an honest declaration against a
γ=0.2 lie costs ≈346 forced reads (saturating Wald); refuting this
coherent liar costs ≈3 forced reads + 3 walk steps (its actual lie
leaks κ≈47.2 nats/read, far beyond the registered γ — disclosed);
incoherence is caught for free — and the pipeline knows which case
it is in before paying.*

## C-GAP-SATURATE — PASSES

Pilot SPRT (1.0 vs 1.2, data 1.0, seed 12, 4000 trials):
**mean reads 355.42 = 1.050× the Wald bound 338.6** (bar ≤ 1.15×);
median 286, q90 649; err_rate 0.036 ≤ 0.065; n_capped 0.
Fixed-sample comparator **m*(0.95) = 649** (bar [600, 710]; the
manuscript's 653); **fixed/sequential = 1.826** (bar ≥ 1.7). The
sequential audit saturates the bound from inside the auditor class,
as the manuscript claims — now as a runnable procedure, not a
remark.

## C-GAP-CERT-SOUND — PASSES

Zero violations on the binding grid: (n, ceiling, best realized
power−size over the LR-threshold class) = (1, .063, .022) /
(4, .125, .088) / (16, .250, .195) / (64, .500, .391) /
(100, .626, .477) / (150, .766, .566). The count-only certificate
is sound everywhere it binds, with margin.

## C-GAP-FLIP — PASSES (bar); descriptive branch realized

Honest operator in the high-noise world (R₇ = r̂₇ = 48):
**n = 913 passive reads, ceiling_allows_target = TRUE** (the
deterministic registered bar). Descriptive verdict row: realized
passive LR = 2.356 < ln 19 = 2.944 — the pre-disclosed ~12%
undecided branch REALIZED — so the pipeline escalated and certified
the honest declaration in its own world at 349.96 forced reads:
**a live exhibit that an allowing ceiling never certifies by
itself; only a decided test or paid reads do.**

## Descriptives

- E4 m*-ladder reproduced under the matched ML criterion:
  r̂/R 0.01→3, 0.1→7, 0.2→12, 0.5→54.
- κ_actual of the suite lie = 47.2 nats/read (world-side
  commentary; explains the 2.7-read catch).
- Access-cost ledger counted as executed (review #5): identical
  free-tier price for every operator (350 queries + the declared
  path), with the entire between-operator cost difference in the
  priced tier — the read-floor conservation exhibit (panel
  candidate #7, fused).

## Consequences

- **CEI's "lattice asserted then abandoned" review finding is
  discharged constructively**: the lattice now runs end-to-end as
  one procedure with itemized access costs, exhibits the theorem
  constants (355 ≈ Wald 339, 649 ≈ 653, ratio 1.83), and repairs
  H7 (the operator class = all-key L3-conformance-identifiable
  Kalman family) and H12.
- Scope carried: known-noise-family, frozen-R̂, one-sided lie
  class; the deployed-regime vulnerability stands exactly as the
  manuscript declares. No new test is claimed anywhere.
- [writing chat] fold into the CEI venue draft: the pipeline
  section (procedure + the four-cell table + the headline
  sentence), the H7 operational definition, and the flip arm's
  vacuity exhibit; cite `artifacts/cei_gap_20260824/`.

Artifacts: `report.json` (the one execution).
