# PREREG — pretrained-encoder pixel arm (frozen 2026-07-30)

The registered prediction-confirmation arm for the pixel boundary
(GO'd 30 Jul; the 29-Jul meeting response's "single best
reviewer-defusing experiment"). This file + the frozen reader
`analysis/pe_pixel_read.py` (selfcheck PASS) + the code additions
listed below are committed BEFORE any `pe` run exists.

## Question and registered predictions

The swamping account says the pixel boundary is a fit-time gradient
competition: reconstruction's bid for encoder capacity swamps the
reward head (task-arm in-regime rew-NLL 23.34 [19.41, 27.29] ≫ the
2.0-nat bar; `artifacts/pixel_swamping_20260725/`, P-SW1). Its lever
prediction: anything that removes reconstruction from the encoder's
gradient competition lowers effective g and restores inclusion via
λ_j(1+βa²) > g_k. This wave pulls that lever inside the codebase: a
pretrained, FROZEN visual encoder feeding the same WM, with the
reward-carrying objective fitting dyn+heads on top.

- **P-PE1 (representation, primary):** the pe-arm task fits' in-regime
  rew-NLL (h=0, sides pooled, seed-clustered bootstrap B=10K rng 0 —
  the P-SW1 estimator) leaves the swamping regime. Graded, frozen:
  CI entirely < 2.0 ⇒ **INCLUSION-RESTORED**; else CI entirely below
  the committed baseline's lower bound 19.41 ⇒ **PARTIAL-RELIEF**;
  else **NO-RELIEF**. (Membership-band descriptor CI < 1.5 reported,
  never decisional. Trivial floor 0.6713 transports — same frozen
  probeset.)
- **P-PE2 (behavior):** frozen-readout adaptation lifts off the
  committed X2 floor: per-seed paired [pe − X2-task] level contrast,
  mean over the two side cells, vs `artifacts/pixel_x2_20260724/auc.csv`
  (the U4 relief estimator); CI entirely > 0 ⇒ LIFTS.
- **P-PE3 (conditional secondary, adjudicated only if P-PE2 fires):**
  the occupancy split re-engages: per-seed [s1 − s0] within the pe
  arm, CI > 0 fires. If P-PE2 does not fire: premise-idle.

Registered verdict map (in the frozen reader, verbatim):
INCLUSION-RESTORED + LIFTS ⇒ **PREDICTION-CONFIRMED** (the boundary
converts to confirmed structure); INCLUSION-RESTORED without lift ⇒
**INCLUDED-BUT-USELESS AT PIXEL** (membership necessary-not-sufficient
in the pixel regime — stamping-pattern analogue); PARTIAL-RELIEF ⇒
directional support only, no headline change; NO-RELIEF ⇒ registered
honest negative — the encoder-competition lever fails its first
intervention and the swamping account needs revision before further
pixel arms. P-SW1 itself is NOT re-adjudicated under any branch.

## Design: staged pretrained-frozen encoder, 16 fit+adapt jobs

- **Stage 1 (exists, no new compute):** the reward-free pixel fits
  `ax1wm_finger_fpxpxq1ms<side>_seed<f>` (apt-mode `pixel_wm`,
  encoder trained by reconstruction+dynamics, no reward head in ckpt)
  are the ENCODER DONORS, seed/side-matched. They are protected in
  `scripts/runroot_cleanup.sh` (KEEP-PENDING) until this wave's read
  verifies.
- **Stage 2 (new):** 16 fits `ax1wm_finger_pepxq1ms<side>_seed<f>`,
  side ∈ {0,1} × seeds 1–8: `pixel_wm`, task-mode, FULL arm (default
  reward_grad/repval_grad — identical objective to the X2 task fits),
  same buffer `axis1_finger/pxq1m/side<side>`, UPDATES 500000, PLUS
  `--run.from_checkpoint <donor latest ckpt> --run.from_checkpoint_regex
  '^enc/' --agent.frozen_enc True` (via the driver's `AXIS1_INIT_WM`).
  Only the 12 CNN encoder params load; everything else fresh-inits;
  the encoder receives no updates.
- **Adapts:** 16 standard frozen-readout adapts
  `adapt_ax1pepxq1ms<side>_finger_seed<f>_ckpt500000` (modes
  `ax1pepxq1ms0/1` — collision-free, no frozen reader consumes them),
  TASK dmc_finger_turn_hard, STEPS 1.25e5.
- **E4 pass:** frozen machinery, `PROBESET fingerpx_v1`,
  `GLOB 'ax1wm_finger_pepxq1m*'`, default horizons; **COLLATE
  overridden to `$RUNROOT/e4_fingerpx_v1_pe.csv`** so the committed
  swamping csv path is never rewritten.

### Code registered with this wave (committed at this freeze)

1. `dreamerv3/configs.yaml`: `agent.frozen_enc: False` default.
2. `dreamerv3/agent.py`: `frozen_enc` drops `enc` from the optimizer
   module list (the `frozen_wm` mechanism one level finer; enc params
   become constants in `nj.grad` — forward pass unchanged).
3. `probing/offline_fit.py`: honors `run.from_checkpoint(+_regex)` for
   partial init, then RESETS the update/batch/action counters to 0 —
   without the reset, a 500k-update donor makes the fit loop start at
   500k and silently skip all training (the known trap; the fit log
   prints `PARTIAL_INIT ... counters_reset=0`). Skipped on resume.
4. `scripts/axis1.sbatch`: `AXIS1_INIT_WM` resolves the donor's latest
   done ckpt and appends the three flags (regex default `^enc/`,
   `AXIS1_INIT_REGEX` overridable).
5. `scripts/check_frozen_enc.py`: post-fit integrity gate — asserts
   enc params BYTE-IDENTICAL to donor, a TRAINING WITNESS
   (`OFFLINE_FIT_PROGRESS` update ≥ expected — written only inside the
   fit loop, so param diffs cannot fake it; this is what catches a
   counter-carryover skip, whose failure direction would masquerade as
   NO-RELIEF), and reward head present. Committed `--selfcheck` builds
   its own fixtures: pass + 4 trip legs (enc-drift, missing-witness,
   short-witness, no-rew).
6. `scripts/runroot_cleanup.sh`: the fpxpx donor fits moved
   DELETE-SAFE → KEEP-PENDING until this wave's read verifies.

Local verification at freeze (CPU, pixel debug config): frozen_enc
removes enc from the optimizer modules; `'^enc/'` load copies exactly
the 12 CNN keys and leaves 216 non-enc params fresh; counter reset
works. End-to-end enc immutability under real training = the smoke
gate below (deliberately cluster-side, real fit path).

### Smoke gate (machine-checked; BEFORE the 16 fits)

One short stage-2 fit named `ax1wm_finger_pesmoke_seed99` (name chosen
so the E4 glob `ax1wm_finger_pepxq1m*` can never match it; the seed-99
cleanup globs cover it after the gate):

```
DONOR=$(find $RUNROOT/ax1wm_finger_fpxpxq1ms0_seed1/ckpt -mindepth 2 -maxdepth 2 -name done -printf '%h\n' | sort | tail -1)
python -m probing.offline_fit \
  --static_replay $RUNROOT/axis1_finger/pxq1m/side0 --updates 2000 \
  --logdir $RUNROOT/ax1wm_finger_pesmoke_seed99 \
  --configs pixel_wm --task dmc_finger_turn_hard --seed 99 \
  --agent.expl.mode task --env.dmc.render True \
  --run.from_checkpoint $DONOR --run.from_checkpoint_regex '^enc/' \
  --agent.frozen_enc True 2>&1 | tee $RUNROOT/ax1wm_finger_pesmoke_seed99_fit.log
SMOKE=$(find $RUNROOT/ax1wm_finger_pesmoke_seed99/ckpt -mindepth 2 -maxdepth 2 -name done -printf '%h\n' | sort | tail -1)
python scripts/check_frozen_enc.py "$SMOKE" "$DONOR" 2000
grep 'PARTIAL_INIT.*counters_reset=0' $RUNROOT/ax1wm_finger_pesmoke_seed99_fit.log
```

GATE: `check_frozen_enc.py` prints PASS (expected_updates=2000) AND
the grep on the persisted smoke log finds the `PARTIAL_INIT ...
counters_reset=0` line. The 16 full fits are submitted only after the
gate passes. After the wave completes, `check_frozen_enc.py` runs for
ALL 16 stage-2 fits against their donors with an AGGREGATE exit code
and a persisted record (sweep block below); any failure quarantines
that run before the read.

### Submit (clean shell; after the smoke gate)

```
for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=36:00:00 \
    --job-name=adapt_ax1pepxq1ms${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1pepxq1ms${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_pepxq1ms${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/pxq1m/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_BASE_CONFIG=pixel_wm,AXIS1_INIT_WM=ax1wm_finger_fpxpxq1ms${side}_seed${f} \
    $REPO/scripts/axis1.sbatch
done; done
# after all 16 fits: integrity sweep (aggregate exit code + persisted
# record — the sweep log ships with the wave bundle as the registered
# integrity evidence)
{ for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do
    S2=$(find $RUNROOT/ax1wm_finger_pepxq1ms${side}_seed${f}/ckpt -mindepth 2 -maxdepth 2 -name done -printf '%h\n' | sort | tail -1)
    D=$(find $RUNROOT/ax1wm_finger_fpxpxq1ms${side}_seed${f}/ckpt -mindepth 2 -maxdepth 2 -name done -printf '%h\n' | sort | tail -1)
    python scripts/check_frozen_enc.py "$S2" "$D" 500000 \
      || echo "QUARANTINE side${side} seed${f}"
  done; done; } 2>&1 | tee $RUNROOT/pe_integrity_sweep.log
# gate via the persisted log (a FAIL flag inside the tee pipeline would
# live in a subshell and silently not propagate):
grep -q QUARANTINE $RUNROOT/pe_integrity_sweep.log \
  && { echo "INTEGRITY SWEEP FAILED — do not run the read"; exit 1; }
grep -c '^PASS' $RUNROOT/pe_integrity_sweep.log   # expect 16
# E4 pass (separate collate file — never overwrite the swamping csv)
PROBESET=$RUNROOT/e4_probesets/fingerpx_v1 GLOB='ax1wm_finger_pepxq1m*' \
  COLLATE=$RUNROOT/e4_fingerpx_v1_pe.csv sbatch $REPO/scripts/e4_measure.sbatch
```

## Registered read (ONE execution)

```
python -m analysis.pe_pixel_read --auc <canonical auc.csv> \
  --e4 <e4_fingerpx_v1_pe.csv> --output <dir>
```

Reader guards (selfcheck-verified): complete 2×8 grids on both the
adapt and E4 sides, unregistered-seed assert, duplicate assert,
`reward_aware=1` on every pe E4 row, finite NLLs, qc_pass, complete
committed baseline. Pinned anchors in the reader: swamping baseline
23.34 [19.41, 27.29], bars 2.0/1.5, floor 0.6713.

## Power and cost

n=8 per side (16 total). P-PE2 mirrors the U4 relief estimator, whose
realized CI half-width was ~12 AUC at the same n — detectable lift
≳15 AUC against cell means 77–83. P-PE1's baseline sits ~21 nats
above the swamping bar; even the partial-relief bound (19.41) is >2×
any plausible measurement noise (swamping CI half-width ~4). Compute:
1 smoke (~1–2 h) + 16 fit+adapt jobs at the 36 h pixel envelope +
1 E4 pass. A frozen-RANDOM-encoder control (same flags, no
`AXIS1_INIT_WM` donor) is named here as the natural follow-up control
if PE1 fires — NOT part of this wave.

## Disclosure and ordering

Known at freeze: every read through 30 Jul (incl. U4 FLOOR-CENSORED
and the committed swamping/X2 anchors pinned above). Not known: any
pe fit, adapt, E4 row, or smoke — none exists. The 25m and U1 reads
are pending and were not consulted. Committed with the code additions
BEFORE the smoke runs; reader sha256 recorded at freeze commit.
