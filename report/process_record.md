# Paper Creation Process Record

Project: *Reward-Free World-Model Pretraining for Transferable Control Representations*
Course: CMSC 35401 (Active Representation Learning). Pipeline: ARS academic-pipeline, Stages 2–6.

## 1. Creation journey

| Phase | What happened |
|-------|---------------|
| Implementation | Exploration modes (random/P2E/APT) + frozen-readout added to a DreamerV3 fork (`dreamerv3/explore.py`, `agent.py`); probing/coverage/synthesis pipeline built (`probing/`). |
| Runs | 6 reward-free pretraining + 4 C1 from-scratch + 18 frozen-readout adaptation + 2 full-adaptation calibration runs on RCC; all completed. |
| Analysis (§G/§H) | Common held-out probe set, coverage + coverage-over-time, AUC↔representation correlations. |
| Stage 2 WRITE | ICLR-format draft from results/figures (`final_report.tex`, `final_report.bib`). |
| Stage 2.5 INTEGRITY | Data traced to `results/`; 4 core citations web-verified; structure linted. PASS. |
| Stage 3 REVIEW | 5-reviewer panel → Major Revision; no CRITICAL; roadmap of 7 writing-only items (`review_round1.md`). |
| Stage 4 REVISE | All 7 items addressed; rliable citation added; `response_to_reviewers.md`. |
| Stage 3' RE-REVIEW | All items verified; no new issues; ACCEPT; converged. |
| Stage 4.5 FINAL INTEGRITY | New citation verified; numbers unchanged; 7-mode failure checklist clean. PASS (zero issues). |
| Stage 5 FINALIZE | Package complete and lint-clean; PDF compile delegated to user (no local engine). |

## 2. Collaboration quality evaluation (6 dimensions, 1–100, evidence-based)

Honesty first — scores cite specific evidence; no inflation.

1. **Problem framing & ownership — 95.** The user supplied a complete, well-structured `research_procedure.md` (RQs, conditions, trigger table). Direction was theirs throughout.
2. **Methodological rigor of the collaboration — 90.** Gated execution (smoke-test C3 before scaling), a pre-registered primary metric, and a deliberately requested protocol-calibration control. Lost points only because the budget (2 seeds) caps statistical power by design.
3. **Critical vigilance — 95.** The user independently diagnosed the P2E sampled-latent degeneracy from the metrics, and explicitly corrected an over-strong causal claim ("early exploration *did the work*") down to a hypothesis. This is high cognitive vigilance, not passive delegation.
4. **Iteration & convergence — 88.** Multiple genuine refine cycles (gate inspection → scaling; coverage-over-time added on request; review→revision converging in one round).
5. **Division of labor clarity — 90.** AI handled implementation, analysis tooling, and drafting; the user owned design, parameter decisions (A=250K, seeds, scope), result interpretation, and go/no-go gates.
6. **Integrity & honesty of the artifact — 92.** Every number traces to logs; citations verified; the saturated-probe and $k{=}20$ artifacts are flagged rather than sold; the coverage→transfer claim is demoted to a hypothesis with the confound stated up front.

**Overall ≈ 92/100 (Zone 1, deep collaboration).** The pattern was user-led with AI as a capable executor and critic, not autopilot.

## 3. AI self-reflection

**What went well.** Catching the P2E target bug early (sampled one-hot → aleatoric floor) saved a meaningless 500K run; the frozen-readout vs full-adaptation calibration turned an apparent C3 deficit into an interpretable protocol finding; the report foregrounds its own limitations.

**What I got wrong, and the correction.** I initially framed the P2E mechanism as a proven early-exploration decomposition. The user correctly flagged this as unprovable from the data; the coverage-over-time analysis then *refuted* even the "early burst" sub-hypothesis (coverage rises monotonically, late). The final text states only what the data supports.

**7-mode AI-research failure audit (Stage 2.5 + 4.5).**

| Mode | Verdict | Basis |
|------|---------|-------|
| 1 Citation hallucination | PASS | 5 core refs web-verified; remainder canonical; guessed page numbers stripped. |
| 2 Implementation bug as result | PASS | The one real bug (P2E target) was caught and fixed before the headline runs. |
| 3 Hallucinated results | PASS | Every reported value recomputed from `results/`. |
| 4 Shortcut reliance | N/A | No held-out shortcut available to exploit. |
| 5 Bug-as-insight | PASS | state\_h0 and $k{=}20$ correlations explicitly labeled artifacts. |
| 6 Methodology fabrication | PASS | Methods match the committed code (`explore.py`, `agent.py`, `probing/`). |
| 7 Pipeline frame-lock | PASS | Reports the unfavorable C1 catch-up, the frozen-readout artifact, and the coverage confound, not only the favorable early-AUC. |

No suspected modes; pipeline did not block.

## 4. Outstanding item

PDF not compiled (no LaTeX engine in this environment). Compile with a TeX install or Overleaf:
```
cd report && pdflatex final_report && bibtex final_report && pdflatex final_report && pdflatex final_report
# or: latexmk -pdf final_report.tex   # or: tectonic final_report.tex
```
Figures resolve via the relative path `../results/presentation_figures/`; keep the `report/` and `results/` trees in place when compiling.
