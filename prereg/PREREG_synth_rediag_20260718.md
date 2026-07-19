# Synth re-diagnosis registration — adapt-stage variance decomposition (frozen 2026-07-18)

The synth diagnosis (`artifacts/synth_diagnosis_20260718/`) closed the
Phase-B′ gate: in-subspace refuted at the model level — synth HAS
occupancy- and supervision-gated representation structure (P-D2
separation 3.93; P-D3 lo-side ratio 3.74) yet the behavioral
interaction was null with enormous per-seed spread (task deltas
−256..+375). The registered instruction is "re-diagnose". This note
registers the re-diagnosis instrument and its directional prediction
**before any re-diagnosis outcome exists**. Descriptive-only; the only
decision gated is a resource one (whether Phase-B″ gets registered).

## Hypothesis under test

**Adapt-stage lottery:** frozen-readout adaptation on sparse-reward
synth is dominated by adapt-time variance (exploration luck of the
fresh AC), washing out a real representation-level effect. The
alternative: fit-level (trunk) heterogeneity dominates — different
fits genuinely differ in transferable quality.

## Instrument (24 adapt-only jobs; no new fits)

Task arm, both sides, fit seeds 1–4, THREE fresh adapt seeds
{101, 102, 103} per fit, re-using the existing 500K-update WM
checkpoints (axis1.sbatch skips a completed fit). Run naming parses
under the frozen RUN_RE with the csv `seed` column = ADAPT seed and
the mode string encoding the fit:
`adapt_ax1rd<fitseed>q1s<side>_synth_seed<adaptseed>_ckpt500000`.
Submit loop (clean shell; mirrors the `submit` helper's export style):

```
for side in 0 1; do for f in 1 2 3 4; do for a in 101 102 103; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=12:00:00 \
    --job-name=adapt_ax1rd${f}q1s${side}_synth_seed${a}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1rd${f}q1s${side}_synth_seed${a}_ckpt500000,WM_RUN=ax1wm_synth_q1s${side}_seed${f},TASK=synth_reach,SEED=${a},REPLAY=$RUNROOT/axis1_synth/q1/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task \
    $REPO/scripts/axis1.sbatch
done; done; done
```

## Registered read (frozen script `analysis/synth_rediag_read.py`, committed pre-outcome)

Per side, one-way random-effects decomposition over the 4×3 grid
(fits × adapt seeds) of AUC100k: σ²_within (adapt-seed, MS_w) and
σ²_between (fit, max(0, (MS_b − MS_w)/3)); the WITHIN SHARE =
σ²_w/(σ²_w+σ²_b), pooled across sides by mean. Existing adapts (seed =
fit seed) are excluded from the decomposition (different convention),
reported alongside descriptively.

- **Directional prediction (adapt-lottery):** pooled within share
  ≥ 0.5.
- Adjudication (resource gate): share ≥ 0.5 ⇒ lottery confirmed ⇒
  **Phase-B″ authorized for registration** — a re-test of the Phase-B
  interaction with adapt-seed-averaged cells (its own frozen file,
  design + job count there). Share < 0.5 ⇒ fit-level heterogeneity
  dominates ⇒ no Phase-B″; the trunk-quality question needs a
  different instrument (e.g. a dense-reward readout probe), which
  would get its own registration.

## Disclosure and ordering

Known at freeze: all outcomes through 18 Jul incl. the Phase-B read
and the closed-gate diagnosis. Unknown: every re-diagnosis quantity —
no synth adaptation has ever been run twice from one checkpoint; no
`ax1rd*` run exists. Ordering: this file +
`analysis/synth_rediag_read.py` committed BEFORE the 24 jobs are
submitted.
