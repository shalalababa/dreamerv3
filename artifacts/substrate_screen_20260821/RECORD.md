# Substrate screen — NON-REGISTERED ARCHIVED COMPUTE (21 Aug 2026)

Status: shipped by user decision (21 Aug, "neglect that house
discipline… ship"); estimand first computed pre-registration during
the ThreeAxes ideation, so this can never be a registered read;
adjudicates nothing. Script `analysis/substrate_screen.py` (selfcheck
PASS: ddof=1 estimator unbiased at planted 0/.25/4; duplicate arms
between-variance bitwise zero; pedestal positive on noise). Substrate:
the committed W1 label bundle (64 main + 4 dup files).

## Results

| | dv3 | tm2 |
|---|---|---|
| ceiling_state (mean per-state decision-sd, G units) | **0.0838** | **0.0559** |
| ceiling_cell (frozen wdc_read form) | **0.0000** | **0.0395** (2 cells: 0.55, 0.69; 30 cells 0) |
| frac states S² ≤ 0 | 92.3% | 97.3% |
| frac states all-G-zero | 55.7% | 42.1% |
| single-draw pedestal | 1.5399 | 0.4591 |
| dup arms (planted truth-zero) | between-var ≡ 0 bitwise; ceilings exactly 0 | same |

## Reproduction adjudication (carry at every citation)

- The **pedestal reproduces the ideation-doc values BIT-EXACTLY**
  (1.5399 / 0.4591) — data handling identical.
- The ideation-doc **ceilings +0.2026 / +0.1184 do NOT reproduce**
  under any definition tried (per-state mean sd, sqrt-of-pooled-S²,
  per-cell, ddof 0/1) — same status as the already-withdrawn "12–29×".
  The reproducible ceilings are LOWER: the substrate is at least as
  flat as claimed, and the flatness verdict is *strengthened*.
- **All inflation ratios are DROPPED**, including the interim
  7.6×/3.9× (their denominators were the unreproduced ceilings). This
  supersedes the 21-Aug decision to quote 7.6×/3.9× — the
  lead-with-ceiling-vs-bar half of that decision stands; the ratio
  half is voided by the reproduction failure. Quote the pedestal and
  the ceiling table; never a ratio.

## Consequences for frozen pins

- `PREREG_p2_tier1_calibration_20260821` (EXECUTED): BAR = 0.2026 was
  pinned pre-outcome and the read stands as executed; the sensitivity
  (dv3 licensing flips to UNDERPOWERED under reproduced ceilings) is
  recorded in `artifacts/p2_tier1_20260821/RECORD.md` and travels with
  the dv3 CCD verdict.
- `PREREG_p2_dcand_20260821` (NOT yet run): its BAR = 0.20 stays as
  frozen; under the reproduced ceilings the bar is CONSERVATIVE for
  the FLATNESS-IS-POLICY branch (harder to fire), so no amendment is
  needed — noted, not amended.
