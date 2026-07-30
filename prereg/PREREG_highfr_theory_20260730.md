# PREREG — high-f_R falling-limb theory predictions (frozen 2026-07-30)

Compact frozen form of `research_notes/Theory_HighFR_Prediction_20260730.md`
(dated derivation note, disk-only). Committed BEFORE any high-f_R
curated buffer, fit, or adaptation run exists, per the registered
ordering for this wave (theory note → this freeze → wave registration
→ curated buffers). Adjudication happens under a separate future wave
registration that will cite this file; nothing here sets n, estimator
details, or gates — direction and dissociation forms only.

## Predictions

Anchors (existing, read): v200 s1 f_R=0.3232 / diversity_pr 9.230;
v400 s1 f_R=0.0937 / diversity_pr 10.633
(`artifacts/volume_repl_20260729/`, instrument pinned
`spectral_v1_1_20260724`). Target manipulation: curated draws at
matched 200-episode volume, same collector pool, f_R ∈ [0.60, 0.80].

- **P-HF1 (falling limb):** task-arm transfer from the curated
  high-f_R buffer is WORSE than from the natural f≈0.32 buffer at
  matched volume — per-seed [B_hi-f − B_q1s1] CI entirely < 0.
- **P-HF2 (mechanism dissociation):** the high-f_R task-arm trunk
  remains IN the reward-NLL membership band while its buffer's
  diversity_pr is LOWER than the f≈0.32 buffer's — the deficit is
  support starvation with inclusion intact. Alternative registered:
  if the trunk leaves the band, the deficit is an inclusion effect
  and the breadth trade-off account gains no support.
- **P-HF3 (interior maximum, weak form):** over the f values measured
  at matched volume, the maximum transfer is interior (not at the
  highest-f cell). No symmetric f(1−f) form and no exact peak
  location are frozen.

## Registered failure modes

- Monotone rise through f≈0.8 ⇒ the trade-off form of the breadth
  patch (`Theory_CompCapacity_Addendum_20260724.tex` §4) is REFUTED —
  revision required, not annotation.
- P-HF1 CI straddles 0 AND diversity_pr does not separate ⇒
  saturation reading; compatible, uninformative, reported as such.

## Disclosure

Known at freeze: all reads through 2026-07-30 (incl. volume
replication P-E4a/b, U2/U3/U4 stress reads). Not known: any high-f_R
buffer or run (none exists); the 25m and U1 reads are pending and
were NOT consulted. Apt-arm cells in the future wave are controls
with the standard reward-free-null expectation, not predictions of
this file.
