# Final report

ICLR-2026-formatted report for the probing study (sequential generative latent
vs. static VAE). Source lives in `iclr2026/` alongside the style files so the
`\usepackage`/`\input`/`.bst` references resolve without path tweaks.

- `iclr2026/report.tex` — the paper (main body ≤ 5 pages; appendices after the
  references).
- `iclr2026/references.bib` — bibliography.
- `iclr2026/figures/` — figures used by the report.

## Build

```bash
cd final_report/iclr2026
pdflatex report
bibtex   report
pdflatex report
pdflatex report
```
Or upload the `iclr2026/` folder to Overleaf and compile `report.tex`. The
`\iclrfinalcopy` macro is on, so author name and page numbers are shown (course
report, not an anonymous submission).

## Notes
- Main body: Introduction, Background/Setup, Probing methodology, Results
  (5 findings, 2 figures, 1 table), Discussion.
- Appendix (not counted toward the 5 pages): open-loop prediction, the
  generative-vs-probe figure, the null/unreliable targets, and reproducibility
  details.
- If the main body runs slightly over 5 pages after compiling, move Figure 2
  (receptive field + gait phase) or Table 1 to the appendix, or trim the
  Background section.
