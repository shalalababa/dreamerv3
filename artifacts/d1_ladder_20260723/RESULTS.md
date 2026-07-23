# D1 actionability ladder — READ (2026-07-23)

Registration: `prereg/PREREG_d1_ladder_20260723.md` (registered-
descriptive addendum; freeze commit `d68277f1` 13:51, read executed
15:01 — ordering verified). Frozen script: `analysis/d1_ladder_read.py`
(unchanged). Data: the 24 existing R2 labels
(`local_results/d1_routea_r2_save60_20260723_103849/`), real op only.
Canonical output: `d1_ladder.json`.

## VERDICT — NO COMPRESSION SIGNATURE. NO EXPLOITABLE PER-STATE
## HETEROGENEITY UP TO THE FULL BELIEF STATE.

The registered heuristic (L3 ρ ≥ 0.2 with CI > 0 while L2 < 0.1) fires
in NO cell. L3 — nested-LORO ridge on the full 512-d belief state +
all scalars — is as blind as the scalar summaries everywhere:

| cell     | L1 ρ (scalars) | L2 ρ (+density) | L3 ρ (belief state) |
|----------|----------------|-----------------|---------------------|
| early×e1 | +0.033         | +0.085          | −0.042 [−.089,−.007]|
| early×e4 | +0.006         | −0.005          | +0.035 [+.002,+.044]|
| late×e1  | +0.035         | +0.026          | −0.003 [−.047,+.047]|
| late×e4  | +0.004         | +0.007          | +0.012 [−.018,+.056]|

Two L3 CIs exclude zero but at |ρ| ≤ 0.04 — statistically nonzero,
practically nothing, one of them NEGATIVE. λ selection was active
across folds (10–10⁴ chosen), so this is not a degenerate-regularizer
artifact.

**Diagnosis (per the registered failure-class map): the D1/R2
calibration null is NOT a scalar-summary compression artifact — it is
no-exploitable-heterogeneity at the information level of the agent's
own belief state.** The external review's ladder logic, level 4
adjudicated. Level 5 (privileged physical state) is not stored in the
labels; it remains the one unexcluded information level (recorded gap,
not silently skipped).

**Resource consequence (registered): the support-diverse-ensemble arm
stays GATED** — its motivating scenario (belief carries signal that
summaries lose) did not materialize.

## The structure that IS there (L0 constants, cluster CIs)

| cell     | always-buy mean Δ_real | 95% CI            |
|----------|------------------------|-------------------|
| early×e1 | **−0.208**             | [−0.472, −0.003]  |
| early×e4 | −0.146                 | [−0.593, +0.145]  |
| late×e1  | +0.068                 | [−0.237, +0.543]  |
| late×e4  | **+0.313**             | [+0.051, +0.627]  |

Early dose-zero buying is significantly HARMFUL; late dosed buying is
significantly VALUABLE — the first cells in this project where a
constant allocation policy separates from zero with CI. Meanwhile no
gated policy at any ladder level beats the best constant anywhere
(net-gain CIs all straddle or fall below 0).

**Paper-2 reading: the value of the lookahead purchase is legible at
the CONTEXT level (training phase, support/dose regime) and illegible
at the STATE level — even to the agent's own full belief state.** The
best allocation policy in this family is "never early, always late
under shift-like load," not a per-state gate. This is the constructive
complement to the Gate-D1/R2 negatives and dovetails with the pending
shift-consequence probe (which tests the context-level claim
prospectively under true support shift).

## Notes

- L1/L2 rows reproduce the R2 read's known F0/F1 numbers exactly (same
  frozen machinery), as disclosed at freeze.
- Executed by the owner from the frozen script after the freeze
  commit; working tree bit-clean at verification.
- Consequence map recap: nothing here unfreezes the 24+24, revives
  Route A, or alters any registered verdict; the only decision taken
  is resource-side (ensemble arm remains gated).
