# Response to Reviewers (Round 1) + Verification Re-Review

## Part A — Response to Reviewers (R → A → C traceability)

Format: **R** = reviewer concern, **A** = author response, **C** = change in `final_report.tex`. We accepted all framing/rigor concerns; we pushed back (with reasoning) where a concern would have required experiments we cannot run, marking those as Acknowledged Limitations rather than silently dropping them.

| # | Source | R (concern) | A (response) | C (change) | Status |
|---|--------|-------------|--------------|-----------|--------|
| P1-a | R1 | $n{=}2$/$n{=}6$ cannot support "2–3×" + two-decimal correlations | Agree; we never intended significance | Added rliable framing + explicit "no significance claim" to metric def (§3); softened "statistically indistinguishable"→"essentially identical"; correlations now read for sign/size only (§5.2); cite \citet{agarwal2021rliable} | ADDRESSED |
| P1-b | DA, R1 | Coverage is near-collinear with "policy is active"; cannot claim coverage is causal | Agree — this is the central honest caveat | Elevated to first-class limitation in abstract, §5.2 (3rd caveat), and §6; gave the "task-relevant regime reached" alternative equal weight in §5.3 | ADDRESSED |
| P2-a | R2 | C3/C4 deviate from canonical P2E/APT | Agree; deviations are deliberate | Added explicit "adapted variants" sentence in §3 (deterministic-posterior target; world-model-latent entropy) + hedge on cross-paper comparison | ADDRESSED |
| P2-b | R3 | Missing amortization economics | Agree; important for the "so what" | Added break-even paragraph to §6 Implications (one-time pretraining cost vs.\ per-task early saving $\sim$$10^5$ steps) | ADDRESSED |
| P3-a | DA | $k{=}20$ future-state correlation rides on near-zero $R^2$ | Agree; parallel to the state\_h0 artifact | Added explicit caveat in §5.2 | ADDRESSED |
| P3-b | R1, R2 | Define early-AUC; tie 500K caveat to C2; confirm matched budget | Agree | Early-AUC defined as normalized AUC (§3); 500K caveat now tied to the C2 negative result (§6); Table 2 caption states C1 measured at the same step budget | ADDRESSED |
| P3-c | DA, R3 | "Transferable representations" must be scoped given §F | Agree | Abstract + §6 now state the benefit is sample efficiency, not a higher ceiling; title retained but no longer implies asymptotic gains | ADDRESSED |

**Pushback / Acknowledged Limitations (not silently dropped):** Items P1-b and P2-b have a residual that *cannot* be resolved by writing — isolating coverage from "reached the task-relevant regime" requires adapting from intermediate checkpoints (needs checkpoint retention), and tightening the amortization estimate requires multi-task budgeting experiments. Both are stated as explicit limitations and future work rather than over-claimed.

## Part B — Verification Re-Review (Stage 3')

Independent re-check of the revised manuscript against the Round-1 roadmap.

| Roadmap item | Verified in revised draft? | Note |
|---|---|---|
| P1-a statistical framing | YES | rliable cited; no-significance statement present; precision claims softened |
| P1-b coverage confound | YES | Now the headline caveat in three places; not buried |
| P2-a method deviation | YES | Stated plainly in §3 |
| P2-b amortization | YES | Break-even paragraph present and quantified at order-of-magnitude |
| P3-a $k{=}20$ caveat | YES | Present and parallel to state\_h0 |
| P3-b metric/budget | YES | All three sub-points present |
| P3-c scoping | YES | Abstract + discussion scoped to sample efficiency |

**New issues introduced by the revision:** none. The revision added only framing and caveats; no claims were strengthened beyond the evidence, no data or numbers changed, no scope creep.

**Residual issues:** the fundamental power limits ($n{=}2$, single domain, coverage/activeness collinearity) remain, but are now correctly disclosed as Acknowledged Limitations rather than papered over. These are inherent to the study's budget and are appropriate to carry into a course mini-report.

**Re-review scores (vs.\ Round 1):** Originality 65→66, Rigor 60→70, Evidence 62→70, Coherence 74→80, Writing 80→82. Weighted ≈ **66 → 73**.

**Decision: ACCEPT.** Delta is driven by honest reframing, not new claims; no P0/CRITICAL issues remain. Per the convergence criterion (no P0 remaining, improvement concentrated in disclosure), the revision loop is **converged** — a second revision round is not warranted.
