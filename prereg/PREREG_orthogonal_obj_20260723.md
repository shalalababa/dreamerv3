# PREREG: orthogonal-objective generalization test (frozen 2026-07-23)

The band-ledger "one orthogonal generalization test" leg (editorial
17 Jul). This file + `analysis/orthogonal_obj_read.py` (selfcheck PASS)
+ the `orthreward` wrapper + `orth_spin_frozen` config are committed
BEFORE any ax1og* run exists. **Running this test lands the band item
under EITHER outcome** — the registered interpretation map below covers
both directions; this is a scope-measuring instrument, not a
hoped-for-positive.

## Question and instrument

Do the reward-legibly-selected features transfer to a DIFFERENT
objective in the same domain? Instrument: `embodied/envs/orthreward.py`
— the dm_control **finger:spin** reward (sparse: hinge_velocity ≤ −15,
via the same `Physics.hinge_velocity` method and `_SPIN_VELOCITY`
constant dm_control uses — no re-implemented reward math) evaluated ON
THE `finger_turn_hard` ENV. Observation space, dynamics, action space,
and episode structure are identical to the original task, so the frozen
enc/dyn/dec checkpoints load exactly; only the reward channel changes.
Wrapper validated locally pre-freeze: 300-step exact agreement with the
threshold rule; injected hinge velocity −20 ⇒ wrapper reward ==
`dm_control.suite.finger.Spin.get_reward` == 1.0 on shared physics;
`make_env` config path smoked (`orth_spin` wiring, obs keys unchanged).

## Design: frozen-readout adapt under the orthogonal objective

- Grid (registered conditional — fit EXISTENCE is not an outcome):
  {rgo, sgb} × {s0, s1} × fit seeds **1–16** (64 jobs) if the sgb
  seeds 9–16 fits exist at submission time
  (`ls -d $RUNROOT/ax1wm_finger_sgbq1s0_seed{9..16}
  $RUNROOT/ax1wm_finger_sgbq1s1_seed{9..16}` → 16 dirs); registered
  FALLBACK: seeds 1–8 (32 jobs) with the power caveat disclosed. The
  frozen read asserts the realized grid is exactly one of the two and
  records which.
- Frozen readout (`AXIS1_ADAPT_CONFIG=orth_spin_frozen`) — the
  objective is the ONLY axis varied relative to the confirmed wave.
  TASK stays `dmc_finger_turn_hard`; adapt seed = fit seed; STEPS
  1.25e5.
- RUN_ID `adapt_ax1og<arm>q1s<side>_finger_seed<f>_ckpt500000`.

Submit loop (clean shell; seeds list per the existence check):

```
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"   # or 1..8 fallback
for side in 0 1; do for f in $SEEDS; do for arm in rgo sgb; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=12:00:00 \
    --job-name=adapt_ax1og${arm}q1s${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1og${arm}q1s${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_${arm}q1s${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/q1/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_ARM=${arm},AXIS1_ADAPT_CONFIG=orth_spin_frozen \
    $REPO/scripts/axis1.sbatch
done; done; done
```

## Registered read (frozen `analysis/orthogonal_obj_read.py`)

- **VALIDITY GATE (registered): pooled mean AUC100k over all
  orthogonal rows < 20.0 ⇒ "FLOOR — uninformative"** — the sparse spin
  objective was not reachable at this adaptation budget and the
  objective-axis prediction is NOT adjudicated (guards against reading
  a floor as an objective-specificity null).
- **PRIMARY: seed-level d(seed) = mean over sides of [rgo − sgb] on
  the orthogonal objective; cluster bootstrap over fit seeds,
  B = 10,000 percentile CI, `default_rng(0)`; FIRES iff CI > 0.**
- Registered interpretation map (frozen; both outcomes land the leg):
  - FIRES ⇒ the retained features are **objective-general** within
    the domain: reward-legibility governs INCLUSION at fit time, but
    the included support serves other objectives — the headline's
    scope statement widens accordingly.
  - Does not fire (above floor) ⇒ **objective-specific support**,
    consistent with the strict headline reading; power bounded by the
    realized grid (disclosed).
- Descriptors (never decisional): same-seed original-task effects
  (committed `auc_pooled_1_16.csv`), d_z comparison and orth/orig
  ratio — with the registered SCALE CAVEAT (turn and spin AUC units
  are not commensurable; no cross-task difference is ever decisional);
  side simples; permutation/d_z/loo.

## Disclosure and ordering

Known at freeze: all original-task outcomes (seeds 1–16 both arms);
the E4 panels; the sgb-9–16 fit-existence question (open — hence the
conditional grid). Unknown: every orthogonal-objective quantity — no
agent has ever been trained or evaluated under the orthreward wrapper
beyond the pre-freeze env-level smoke (no learning involved).
Ordering: this file + read + wrapper + configs committed BEFORE
submission; the realized grid and the existence-check output are
recorded in the artifact. Compute: 32–64 adapt-only jobs, ≤ 12 h each.
