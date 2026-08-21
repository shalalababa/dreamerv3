# Null fixture rebuild — Gate 0 (21 Aug 2026)

Status: NON-REGISTERED ARCHIVED COMPUTE (Gate 0 of
Rescue_Synthesis_20260819; user decision 21 Aug: rebuild rather than
strike). Script `analysis/null_fixture.py` (selfcheck PASS: closed
form E[max_8 z] ≈ 1.4236 both estimand forms; ×2-noise doubling; the
sigma estimator is signal-free under planted spread; zero-noise →
exactly zero; pinned-stream determinism). Substrate for calibration:
the committed W1 label bundle.

## The exhibit

Fixture with **true per-candidate opportunity IDENTICALLY ZERO by
construction**, single-draw noise calibrated PER STATE from the
archived arrays (within-candidate across-repeat sd — contains no
between-candidate signal, verified in the selfcheck). The unmodified
R3-era single-draw estimator returns:

| | dv3 | tm2 |
|---|---|---|
| calibrated noise sd (mean) | 1.464 | 2.337 |
| **opportunity (max − chosen, chance chooser)** | **+2.099** (± 0.062 over 50 draws) | **+3.320** (± 0.097) |
| opportunity (max − mean identity form) | +2.087 | +3.333 |
| ×2 evaluation noise | +4.174 | +6.640 |
| true opportunity | 0.000 | 0.000 |

The ×2-noise rows double exactly — the draft's "adding evaluation
noise doubles the effect" selection-bias signature reproduces on the
fixture itself.

## Supersession note (carry into the draft edit)

The draft's "+5.6" (Paper2_Draft_20260814.tex:169, :635) had NO
artifact and its fixture parameters were never recorded — it is
SUPERSEDED by this artifact's numbers. The contribution-list sentence
should quote **+2.10 (dv3) / +3.32 (tm2)** (or the family relevant in
context) with this artifact cited. The rebuilt exhibit is stronger
than the lost one: its noise is calibrated from the archived data, so
the zero-truth fixture lands at the archived headline's order of
magnitude — the route1-sims recoverable-from-zero exhibit, on real
noise calibration. (The lost "+5.6" is bracketed by the tm2 ×2-noise
value 6.64; its exact provenance is unrecoverable and no longer
matters.)
