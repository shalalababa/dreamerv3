# Synth Phase-B″ registration — adapt-averaged interaction re-test (frozen 2026-07-20)

Authorized by the registered re-diagnosis gate
(`PREREG_synth_rediag_20260718.md` → `artifacts/synth_rediag_20260719/`:
pooled within-fit share **0.865 ≥ 0.5**, adapt lottery confirmed; side0
between-fit component clamps to 0, same-fit adapt swings up to ~30×).
This file registers the re-test design and its read **before any
ax1rb\* outcome exists**.

## Design (96 adapt-only jobs; no new fits)

Re-test of the Phase-B PRIMARY-1 interaction with adapt-seed-averaged
cells: {task, apt} × {s0, s1} × fit seeds 1–8, **three fresh adapt
seeds {104, 105, 106}** per fit, reusing the existing 500K synth WM
checkpoints (`axis1.sbatch` skips completed fits).

- **Fresh adapt seeds by design:** seeds 101–103 (task fits 1–4) were
  revealed in the re-diagnosis read; no revealed adapt outcome enters
  the B″ statistics. Fit identity is not an outcome.
- **Shuffle arm not re-tested:** PRIMARY-2 (shuffle collapse −226.6)
  fired decisively in Phase-B and stands.
- Run naming (frozen RUN_RE; csv `seed` column = ADAPT seed; the mode
  string encodes arm + fit): task
  `adapt_ax1rb<fit>q1s<side>_synth_seed<a>_ckpt500000` with
  `WM_RUN=ax1wm_synth_q1s<side>_seed<fit>`; apt
  `adapt_ax1rbf<fit>q1s<side>_synth_seed<a>_ckpt500000` with
  `WM_RUN=ax1wm_synth_fq1s<side>_seed<fit>`. Re-diagnosis modes
  (`ax1rd*`) cannot parse under the B″ reader and vice versa.

Submit loop (clean shell; mirrors the re-diagnosis export style;
`AXIS1_EXPL_MODE` matches each arm's original fit mode as the
fit-skip guard):

```
for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do for a in 104 105 106; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=12:00:00 \
    --job-name=adapt_ax1rb${f}q1s${side}_synth_seed${a}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1rb${f}q1s${side}_synth_seed${a}_ckpt500000,WM_RUN=ax1wm_synth_q1s${side}_seed${f},TASK=synth_reach,SEED=${a},REPLAY=$RUNROOT/axis1_synth/q1/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task \
    $REPO/scripts/axis1.sbatch
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=12:00:00 \
    --job-name=adapt_ax1rbf${f}q1s${side}_synth_seed${a}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1rbf${f}q1s${side}_synth_seed${a}_ckpt500000,WM_RUN=ax1wm_synth_fq1s${side}_seed${f},TASK=synth_reach,SEED=${a},REPLAY=$RUNROOT/axis1_synth/q1/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=apt \
    $REPO/scripts/axis1.sbatch
done; done; done
```

## Registered read (frozen `analysis/synth_phasebpp_read.py`, committed with this file)

Fit-level AUC100k = mean over the 3 adapt seeds; B_arm(fit) =
fit-level(s1) − fit-level(s0);

- **PRIMARY: interaction I = mean over fit seeds of [B_task − B_apt]**;
  cluster bootstrap over fit seeds B=10,000 percentile CI,
  `default_rng(0)`; **FIRES iff CI > 0**. Decision on the CI alone.
- Secondary (robustness/descriptive): arm simples with own CIs; exact
  sign-flip permutation, d_z, leave-one-fit-out range; per-cell
  within-fit adapt-seed SD (re-check against the re-diagnosis
  decomposition: with share .865 and 3-seed averaging, the expected CI
  half-width is ≈0.65× Phase-B's ±133, i.e. ≈±87).
- **Registered power statement:** this design decides interactions of
  magnitude ≳ ~90 AUC units. The between-fit noise floor (share .135)
  bounds what any amount of adapt averaging can resolve at 8 fit seeds
  (≈±49 at infinite adapt seeds).
- **Consequence map (frozen):** FIRES ⇒ the Phase-B null was
  lottery-masked; synth rejoins as a positive interaction domain
  (Phase-C eligibility restored, own registration). Does not fire ⇒
  the synth interaction leg is CLOSED at this design: no Phase-C;
  synth remains a shuffle-collapse-only exhibit in the paper, and the
  domain-scope statement records that the interaction was not
  detectable under lottery-corrected adaptation.

## Disclosure and ordering

Known at freeze: all outcomes through 20 Jul, incl. Phase-B (+44.2
interaction, CI ±133), the diagnosis (gate closed), the re-diagnosis
decomposition (share .865, and the 24 revealed adapt outcomes at seeds
101–103 for task fits 1–4), and the four 19-Jul reads. Unknown: every
ax1rb\* quantity — no B″ adapt run exists; adapt seeds 104–106 have
never been used anywhere. Ordering: this file +
`analysis/synth_phasebpp_read.py` committed BEFORE the 96 jobs are
submitted.
