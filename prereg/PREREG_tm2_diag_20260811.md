# PREREG: TM2 legibility diagnostics (P-F1 premise check) — 2026-08-11 (post-review REVISED form)

User GO 2026-08-11 (family slate, diagnostics-first — the f_R lesson).
Registered premise check for `PREREG_theory_R1_gating_20260811` P-F1:
*"TM2 free-arm reward-legibility sits CLOSER to its aware arm than
Dreamer apt sits to Dreamer task."* Premise tier — it licenses or
re-scopes the POWERED TM2 wave; it adjudicates no paper claim.
**This file replaces the pre-review draft in full** (batch review
findings 1/2/5: the draft's reader consumed a nonexistent
ridge_probe schema, its honesty block was false, and its R²-ratio
mixed two incommensurable unbounded metrics).

## Honesty block (corrected — review B-2)

**The dv3 anchor is ALREADY EXECUTED AND READ**: the 08-10 finger
refit read (`artifacts/finger_refit_read_20260810/read.json`) pooled
ridge-probe AUROCs over the same 32 fits this check would have used —
task 0.93477 (n=16), apt 0.40107 (n=16), finger_v1 probeset,
witness_match 32/32. **gap_dv3 = 0.53370 is therefore a KNOWN
constant, pinned below; the registered rule reduces to thresholds on
gap_TM2 alone**, and NO new dv3 compute runs. Also known: TM2
behavioral values (aware +35.3*, free +25.8 ns) and R1's frozen P-F1
prediction (which precedes this check — premise-peeking cannot tune
it). Unknown: every TM2-side probe quantity (the TM2 instrument has
never run on any real fit).

## Substrate + inventory gate (fail-closed)

- TM2 fits: `$RUNROOT/tm2wm_finger_{aware|free}q1s{0,1}_seed{1..8}/`
  (`tm2_ckpt.pt` + `TM2_FIT_DONE` + `config.yaml` audit; free arm =
  reward_coef 0, value_coef 0 — consistency-only). **Inventory
  command ([YOU], runs first):**
  `for a in aware free; do for s in 0 1; do for k in 1 2 3 4 5 6 7 8; do d=$RUNROOT/tm2wm_finger_${a}q1s${s}_seed${k}; [ -f $d/tm2_ckpt.pt ] && [ -f $d/TM2_FIT_DONE ] && echo OK $d || echo MISS $d; done; done; done`
  plus `ls -la $RUNROOT/tm2_data/finger_q1_side{0,1}.pt`.
- **Gate** (reader-enforced, re-checked after any degenerate-label
  exclusion): ≥6 loadable fits PER ARM with BOTH sides represented in
  each arm, and both `tm2_data` side files present; else
  **DIAGNOSTICS-BLOCKED** (reported; powered-wave decision returns to
  the user).

## Instrument (frozen with this file)

`probing/tm2_legibility_probe.py` (selfcheck PASS). Per fit: encode 16
rng-0-sampled episodes of the **OPPOSITE side's** bridged `.pt`
(cross-side asserted fail-closed — the fit consumed its entire own
side; cross-side is the held-out analog of dv3's pilot-replay
probesets), episode-grouped 4-fold ridge CV with RANDOM grouped inner
alpha selection (refuses on degenerate inner scores; review #26).
**Registered scalar = fold-centered pooled-OOF AUROC of the ridge
predictions against reward>0** (bounded, base-rate-free, chance 0.5 —
the same metric family as the pinned dv3 anchor, which is
ridge_probe's AUROC; review #5/#27/#29; fold-centering removes the
train-mean fold-composition artifact, selfcheck-caught). Held-out R²
recorded as secondary descriptive only. Registered command per fit:
`python -m probing.tm2_legibility_probe --wm_run $RUNROOT/tm2wm_finger_<arm>q1s<side>_seed<k> --data $RUNROOT/tm2_data/finger_q1_side<1-side>.pt --output $RUNROOT/tm2_diag/tm2wm_finger_<arm>q1s<side>_seed<k>.json`
(≤32 probe passes; GPU-minutes each; NO dv3 passes).

Instrument caveat (disclosed): the TM2 scalar (grouped-CV
fold-centered OOF AUROC on cross-side episodes) and the dv3 anchor
(ridge_probe pooled-OOF AUROC on the finger_v1 probeset) are the same
metric on the same bounded scale but not the same estimator or probe
data; residual estimator-level offsets are the known limitation of
the cross-family ratio, mitigated by AUROC's boundedness and
base-rate freedom (the review-killed R² form had unbounded
denominator terms).

## Registered rule (thresholds on gap_TM2; denominator PINNED)

gap_TM2 = **side-balanced** mean of within-side (aware − free) AUROC
means (review #20; the unbalanced pooled gap + its house two-sample
BCa 95% CI are recorded alongside — a SEED-variance CI, n_data = 2
side buffers per arm, disclosed). Pinned anchor: gap_dv3 = 0.53370.

- **DEGENERATE** (fail-closed): gap_TM2 non-finite.
- **PREMISE-STRONG**: 0 ≤ gap_TM2 < 0.267 (ratio < 0.5).
- **PREMISE-CONSISTENT**: 0.267 ≤ gap_TM2 < 0.534 (ratio in [0.5, 1)),
  or gap_TM2 < 0 with CI including 0 (compression to
  indistinguishability counts toward the premise).
- **PREMISE-VIOLATED**: gap_TM2 ≥ 0.534 (ratio ≥ 1) — P-F1's premise
  fails; the powered wave's attenuation prediction must be
  re-registered or dropped BEFORE that wave freezes.
- **PREMISE-INVERTED**: gap_TM2 < 0 with CI upper < 0 (free MORE
  legible than aware beyond noise) — reported; powered wave re-scoped.
- Per-fit degenerate labels (AUROC nan: probe episodes with no
  positive reward rows) ⇒ fit EXCLUDED-disclosed; the ≥6-per-arm gate
  re-checked after exclusion (review #17). Point rule governs
  (premise tier); the CI is context, disclosed.

Reader: `analysis/tm2_diag_read.py` (REWRITTEN post-review; selfcheck
PASS incl. a real-file loader round-trip with audit/duplicate/nan
gates — review finding-1 class covered). ONE read execution when the
TM2 jsons land. Reviewer: 2026-08-11 batch review (ONE, Opus 5) —
findings applied in this revised form; re-review not required (the
redesign implements the reviewer's own prescribed fixes 1/2/5 +
16–20/26–29).
