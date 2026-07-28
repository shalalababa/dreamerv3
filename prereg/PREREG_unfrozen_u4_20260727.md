# PREREG U4: pixel unfrozen 2×2 — floor relief + the swamping-derived prediction (frozen 2026-07-27)

Unfrozen stress wave 4 (audit: pixel flip-risk MEDIUM after P-SW1 —
the swamping result removes the canonical flip mechanism, but
floor-censoring relief is real: unfreezing lifts levels +235/+30 in
proprio, so the frozen pixel floor could be masking differences).
This file + `analysis/unfrozen_stress_read.py u4` (selfcheck PASS)
are committed BEFORE any `ax1uz*px*` run exists.

## Question and registered predictions

Does unfrozen adaptation lift the pixel cells off the X2 floor, and
if so, does an interaction express? Registered predictions:

- **P-U4a (floor relief)**: YES — unfreezing lifts levels (the
  proprio calibration lifted every arm; pixel encoders have the most
  to gain from task-driven fine-tuning).
- **P-U4b (the P-SW1-derived prediction, the theory's stake)**: NO
  interaction — reward information is absent from every pixel trunk
  (task-arm rew-NLL 23.3 ≫ bands, `artifacts/pixel_swamping_20260725/`),
  so fine-tuning must re-carve reward features from scratch, and the
  re-carving starts from the SAME reward-free position in all four
  cells ⇒ no differential rescue. A fired interaction would rescope
  the pixel boundary toward a frozen-readout construct and force an
  initialization-geometry amendment to the swamping account (P-SW1
  itself, a fit-time fact, is untouched either way).

## Design: 32 unfrozen adapt jobs from the EXISTING pixel fits

- Grid: {task, apt} × {s0, s1} × fit seeds 1–8 — the full X2 2×2, so
  the unfrozen interaction is measurable (not only the task-arm floor
  relief). `AXIS1_BASE_CONFIG=pixel_wm` +
  `AXIS1_ADAPT_CONFIG=unfrozen_readout`; everything else identical to
  the X2 adapt protocol (TASK `dmc_finger_turn_hard`, adapt task-mode,
  STEPS 1.25e5).
- WM runs `ax1wm_finger_pxpxq1ms<side>_seed<f>` (task) /
  `ax1wm_finger_fpxpxq1ms<side>_seed<f>` (apt) — the 32 X2 fits
  (config-audited 32/32 in the X2 read). Existence + refit disclosure
  as in U1 (buffer manifest:
  `ls $RUNROOT/axis1_finger/pxq1m/manifest.json`); the 24 h walltime
  covers a triggered pixel refit (the X2 envelope), unlike a 12 h
  adapt-only slot.
- RUN_ID `adapt_ax1uzpxpxq1ms<side>_finger_seed<f>_ckpt500000` /
  `adapt_ax1uzfpxpxq1ms<side>_finger_seed<f>_ckpt500000`.

Submit loop (clean shell):

```
for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=24:00:00 \
    --job-name=adapt_ax1uzpxpxq1ms${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1uzpxpxq1ms${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_pxpxq1ms${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/pxq1m/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_BASE_CONFIG=pixel_wm,AXIS1_ADAPT_CONFIG=unfrozen_readout \
    $REPO/scripts/axis1.sbatch
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=24:00:00 \
    --job-name=adapt_ax1uzfpxpxq1ms${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1uzfpxpxq1ms${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_fpxpxq1ms${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/pxq1m/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=apt,AXIS1_BASE_CONFIG=pixel_wm,AXIS1_ADAPT_CONFIG=unfrozen_readout \
    $REPO/scripts/axis1.sbatch
done; done
```

## Registered read (frozen `analysis/unfrozen_stress_read.py u4`)

- **P-U4a (floor relief)**: per-seed mean over the four cells of
  [unfrozen − frozen] level, paired per cell against the committed X2
  baseline (`artifacts/pixel_x2_20260724/auc.csv`); cluster bootstrap
  B=10K rng 0; LIFTS iff CI > 0.
- **CENSORING GUARD (registered)**: if P-U4a does not fire, the
  interaction read is declared FLOOR-CENSORED — uninformative
  (reported, no adjudication; the frozen floor still binds).
- **P-U4b (interaction under unfreezing)**: per-seed
  [B_task − B_apt], B = s1−s0; adjudicated only if P-U4a fires.
  CI entirely > 0 ⇒ **FLIP**. Else ⇒ **PROTOCOL-ROBUST** (the
  P-SW1-derived prediction lands).
- Descriptors: cell means; per-cell level changes.

## Consequence map (frozen)

- **Relief + no interaction** ⇒ the swamping account gains a
  behavioral confirmation in the second protocol: the pixel boundary
  ("pixels raise the competition bar past every arm") holds for both
  frozen and fine-tuned transfer; the paper's pixel paragraph cites
  both.
- **FLIP** ⇒ the pixel boundary is re-annotated frozen-scoped; a
  dated theory note (initialization geometry vs trunk content) is
  required before any further pixel arm; G-X3 stays NO-GO regardless
  (resource decision).
- **Floor-censored** ⇒ reported; no further pixel-unfrozen compute
  pre-deadline.

## Disclosure and ordering

Known at freeze: the X2 read (all cell floors), P-SW1 (task-arm
rew-NLL 23.3, floor 0.67), the proprio unfrozen calibration. Unknown:
every unfrozen pixel quantity — no pixel WM has ever been adapted
unfrozen. Ordering: committed with the U-wave freeze BEFORE any
submission. Compute: 32 adapt-only jobs at the 24 h X2 envelope
(adapt-only completes well inside it; the envelope exists so a
triggered refit can also complete).
