# PREREG: unfrozen-adaptation calibration (frozen 2026-07-23)

The band-ledger "full/unfrozen adaptation calibration" leg (Research
Plan v4 revised queue item 4: "unfrozen-WM adaptation on a subset to
calibrate what frozen-readout misses"; editorial 17 Jul band condition).
This file + `analysis/unfrozen_calib_read.py` (selfcheck PASS) + the
`unfrozen_readout` config + the `AXIS1_ADAPT_CONFIG` sbatch knob are
committed BEFORE any ax1ufz* run exists.

## Question and registered prediction

Is the confirmed rgo−sgb transfer effect (+51.5* pooled, Amendment 1)
an artifact of freezing the WM at adaptation time? Registered
prediction (from the mechanism: the advantage lives in the transferred
representation's reward-legible support, which unfrozen fine-tuning
inherits as an initialization): **the effect persists under full
adaptation, plausibly attenuated** (initialization effects wash out
slower than 1.25e5 steps).

## Design: 32 adapt-only jobs from the EXISTING frozen fits

- Grid: {rgo, sgb} × {s0, s1} × fit seeds 1–8 (the fits whose 500K
  checkpoints exist and were E4-verified 22 Jul); TASK
  `dmc_finger_turn_hard`; adapt seed = fit seed (P3 convention);
  STEPS 1.25e5; `AXIS1_ADAPT_CONFIG=unfrozen_readout` (same
  enc/dyn/dec initialization as frozen_readout, nothing frozen).
- RUN_ID `adapt_ax1ufz<arm>q1s<side>_finger_seed<f>_ckpt500000`
  (mode `ax1ufz<arm>q1s<side>`; the frozen-wave modes cannot parse
  under the read's regex and vice versa).
- Frozen baseline for the paired descriptors = the committed
  `artifacts/p3_amend1_20260717/auc_pooled_1_16.csv` (seeds 1–8
  subset; KNOWN at freeze, disclosed).

Submit loop (clean shell; module-leakage gotcha applies):

```
for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do for arm in rgo sgb; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=12:00:00 \
    --job-name=adapt_ax1ufz${arm}q1s${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1ufz${arm}q1s${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_${arm}q1s${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/q1/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_ARM=${arm},AXIS1_ADAPT_CONFIG=unfrozen_readout \
    $REPO/scripts/axis1.sbatch
done; done; done
```

(`AXIS1_EXPL_MODE`/`AXIS1_ARM` serve the fit-skip guard and, if a fit
were ever incomplete, would refit under the correct registered arm.)

## Registered read (frozen `analysis/unfrozen_calib_read.py`)

Seed-level effect d(seed) = mean over sides of [rgo − sgb] AUC100k.

- **PRIMARY: mean over the 8 fit seeds of d(seed); cluster bootstrap
  over seeds, B = 10,000 percentile CI, `default_rng(0)`; FIRES iff
  CI > 0.** Decision on the CI alone.
- Registered descriptors (never decisional): per-side simples;
  per-arm level change (unfrozen − frozen, paired by cell — does
  unfreezing raise absolute performance); frozen same-seed effect +
  attenuation ratio (the calibration number; the seeds-1–8 frozen
  effect is itself noisy — disclosed); exact sign-flip permutation,
  d_z, leave-one-seed-out range.

## Consequence map (frozen)

- **FIRES** ⇒ the band-ledger leg lands: the effect survives full
  adaptation; frozen-readout is a magnifier, not the mechanism; the
  attenuation ratio is reported as the calibration number.
- **Does not fire** ⇒ registered scope limitation: the confirmed
  effect is established for frozen-readout transfer; full-adaptation
  claims are not licensed at this n (power ≈ the original seeds-1–8
  wave, which needed pooling to 16 to confirm — disclosed; a fired
  Amendment-style extension to seeds 9–16 would need a dated
  amendment BEFORE those outcomes exist).
- Either way the paired level-change numbers calibrate "what
  frozen-readout misses" for the paper.

## Disclosure and ordering

Known at freeze: all outcomes through 23 Jul (frozen-readout effects
at seeds 1–16 incl. Amendment 1 +51.5*; E4 panels; every read in
`analysis/DEVIATIONS.md`). Unknown: every unfrozen quantity — no
unfrozen-adapt run has ever been made in this project. Ordering: this
file + the read + the config/sbatch plumbing committed BEFORE
submission.
