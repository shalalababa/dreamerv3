# AMENDMENT 2 to PREREG_trackB_alea_20260822 (+Amendment 1) —
cross-instrument tolerance recalibration after a registered
refusal, 23 Aug 2026

**What happened:** the ONE registered execution of
`uncfield/se_b4_read.py` on the verified `b4_alea_20260823_035808`
bundle REFUSED all 4 runs at the R-A1-M7a cross-instrument share
gate (mask raw-share ratios vs se_probe `key_share` ratios,
registered tolerance 1e-4 relative) and, per the R-A1-M13 rule,
wrote a NOT-ADJUDICABLE side file WITHOUT consuming the execution.
No adjudication-bearing number was produced or seen beyond the
gate's own diagnostic.

**Diagnosis (all-keys divergence table, computed from the refused
bundle's instrument jsons):** ratios-to-position diverge by
+1.11/−0.77/−1.48/−1.92 % (distractor) and +1.20/+0.99/−0.72/−2.27
% (velocity) across the four seeds — MIXED SIGN, ~1–2 % — while
the position-correlated dup channels sit at ±0.05–0.29 % (their
shared noise cancels in the ratio). This is the signature of
numerical-regime divergence between two DIFFERENT compiled graphs
over bf16 forwards (the instruments compute the same member
variances in differently-fused XLA programs; the 16-Aug measured
phenomenon, same family), NOT of a form/norms drift — an M6-class
drift is one-directional and structured (1.53× on the affected
key), and the within-instrument share provenance (float64-exact)
PASSED on all 4 runs. The registered 1e-4 assumed f32/f64 rounding
(R-A1-m21) and was mis-calibrated for bf16 graph divergence.

**Re-specification:** `XPROV_RTOL` 1e-4 → **5e-2** relative. The
gate's registered purpose — catching gross form drift (M6-class
≥ 50 %), wrong-checkout instruments, and cross-attached outputs —
is fully retained at 5e-2; anchor-set identity is carried by the
dedicated seed/n_eval pins (R-A1-M8), not by this gate. No
estimand, bar, cell, or any other gate changes.

**Value-aware disclosure:** the recalibration follows contact with
wave data, limited to the gate's own divergence diagnostics; it
touches an integrity tolerance only. Procedure = the B1-s53 house
pattern (registered refusal → repair → re-execute); the reader is
edited pre-execution (the execution was never consumed) with this
dated amendment as the record.
