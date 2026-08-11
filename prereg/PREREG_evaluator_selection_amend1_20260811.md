# AMENDMENT 1 to PREREG_evaluator_selection_20260811 — leg-3b producer repair (2026-08-11)

Status at amendment: the registered leg-3b producer
(`artifacts/evaluator_selection_reg_20260811/make_action_stats.py`,
shipped in freeze-commit e1ffe5ab) failed on its first invocation with
`evaluator replay unreadable`. **No scoring pass, collate, or reader
output is involved** — the producer emits the leg-3b COVARIATE only
(per-dim action means/stds of replay buffers; no outcome quantity), so
this is a value-blind, pre-outcome instrument repair. Legs 1–2 and the
leg-3b decision rule are UNCHANGED; the registered SKIPPED-disclosed
branch (review M9) remains in force if inputs are still unreadable
after this fix.

## Defects (both in the producer/usage, not the design)

1. The registered usage example pointed `--evaluator` at the FIT run
   dir (`$RUNROOT/ax1wm_finger_q1s1_seed1`). Offline WM fits CONSUME an
   existing buffer and retain no `replay/` of their own — the
   evaluator's action distribution is that of its TRAINING buffer,
   which for E2 (`ax1wm_finger_q1s1_seed1`) is
   **`$RUNROOT/axis1_finger/q1/side1`** per the submit plumbing
   (`scripts/submit_all.sh`: `REPLAY=$root/side${side}`,
   `root=$RUNROOT/axis1_<dom>/q1`).
2. `stats_of` hardcoded a `replay/` subdirectory. Adapt policy run
   dirs have one (unchanged behavior); the axis1 q1 side buffers are
   FLAT dirs of episode npz chunks and never matched.

## Registered fix

`stats_of` now falls back to `<dir>/*.npz` when `<dir>/replay/*.npz`
is empty (policy-side behavior byte-identical when `replay/` exists),
and the registered invocation becomes:

```
python make_action_stats.py --evaluator $RUNROOT/axis1_finger/q1/side1 \
    --policies "$RUNROOT"/adapt_ax1sw*_finger_seed*_ckpt500000 \
    --output action_stats.json
```

(The policy glob covers exactly the 88-member s×w_r pool of the
Goodhart partial Amendment 1; policies whose replay was pruned land in
the producer's `missing` list and degrade leg 3b with the registered
missing-count disclosure — they do NOT abort anything.) The estimand,
distance definition, decision rule, and E2 identity are unchanged.
