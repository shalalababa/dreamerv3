# PREREG: vgo extended-fit discriminator (frozen 2026-07-23)

Band-C optional arm (roadmap: "extended-fit vgo discriminator — own
prereg; attenuation form P-B2 after the case-(b) audit";
`artifacts/vgo_wiring_audit_20260717/`). This file +
`analysis/vgo_extended_read.py` (selfcheck PASS) are committed BEFORE
any ax1x* run exists.

## Question and registered prediction

The vgo arm (value-gradients-only: `--agent.reward_grad False`) is
behaviorally dead at 500K fit updates, and the wiring audit
reclassified it as case (b): the value-gradient path is replay-grounded
— P-B2 therefore takes its **attenuation form**: the path exists but is
SLOW. Registered discriminating prediction: **if attenuation is right,
3× fit length grows vgo's transfer beyond generic-extension gains; if
vgo is absence-like, it does not.** sgb (no representation-gradient
path at all) is the generic-extension control: any sgb gain from
longer fitting is generic by construction.

## Design: 16 fresh 1.5M-update fits + standard frozen-readout adapt

- Grid: {vgo, sgb} × fit seeds 1–8 × **s1 only** (the hi-occupancy
  side, where every confirmed effect lives); UPDATES = 1,500,000;
  **FRESH WM dirs** `ax1wm_finger_x<arm>q1s1_seed<f>` — the frozen
  500K fits are NEVER touched (no resume into registered dirs).
- Adapt: standard frozen_readout, TASK `dmc_finger_turn_hard`, adapt
  seed = fit seed, STEPS 1.25e5. RUN_ID
  `adapt_ax1x<arm>q1s1_finger_seed<f>_ckpt1500000` (the read asserts
  milestone 1,500,000).
- 500K baselines = the committed `artifacts/p3_wave_20260717/auc_p3.csv`
  (KNOWN at freeze, disclosed).

Submit loop (clean shell; 24 h wall time for the 3× fit):

```
for f in 1 2 3 4 5 6 7 8; do for arm in vgo sgb; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=24:00:00 \
    --job-name=adapt_ax1x${arm}q1s1_finger_seed${f}_ckpt1500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1x${arm}q1s1_finger_seed${f}_ckpt1500000,WM_RUN=ax1wm_finger_x${arm}q1s1_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/q1/side1,UPDATES=1500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_ARM=${arm} \
    $REPO/scripts/axis1.sbatch
done; done
```

## Registered read (frozen `analysis/vgo_extended_read.py`)

- **PRIMARY: D(seed) = [vgo_1.5M − vgo_500K] − [sgb_1.5M − sgb_500K]
  (AUC100k, paired by seed); mean over the 8 seeds; cluster bootstrap
  B = 10,000 percentile CI, `default_rng(0)`; FIRES iff CI > 0.**
  Decision on the CI alone.
- Descriptors (never decisional): per-arm extension gains with own
  CIs; levels; permutation/d_z/loo.

## Consequence map (frozen)

- **FIRES** ⇒ P-B2 attenuation form SUPPORTED: the value-gradient path
  is present but slow — the theory's gradient-pathway account gains a
  time-constant parameter; a magnitude-matched follow-up (does vgo at
  k× approach rgo at 1×?) would need its own registration.
- **Does not fire** ⇒ attenuation NOT supported at 3× updates; vgo
  stays dead in the tested range (absence-like); the paper reports the
  discriminator as run-and-negative and P-B2 keeps only its weakest
  reading. **No further extended-fit arms without new theory.**

## Disclosure and ordering

Known at freeze: vgo/sgb 500K outcomes (auc_p3.csv), the wiring audit,
all reads through 23 Jul. Unknown: every >500K-update fit quantity —
no fit in this project has ever run past 500K updates. Ordering: this
file + the read committed BEFORE submission. Compute: 16 jobs × ≤24 h
(fit ≈ 3× the 500K fit time + standard adapt).
