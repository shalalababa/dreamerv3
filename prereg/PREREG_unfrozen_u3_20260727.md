# PREREG U3: orthogonal objective unfrozen — the objective-specificity stress test (frozen 2026-07-27)

Unfrozen stress wave 3 (audit critic upgrade medium→HIGH): the
orthogonal-objective verdict ("objective-specific support",
`artifacts/orthogonal_obj_20260725/`) is a frozen null measured in the
exact system where the frozen protocol understated the paired positive
contrast ×4.3. This file + `analysis/unfrozen_stress_read.py u3`
(selfcheck PASS) + the `orth_spin_unfrozen` config block
(`dreamerv3/configs.yaml` — orth_spin + unfrozen_readout as one
export-safe token) are committed BEFORE any `ax1uzog*` run exists.

## Question and registered prediction

Does the rgo−sgb advantage stay null under an ORTHOGONAL objective
when nothing is frozen? Genuinely open, with a mild theory lean toward
robustness: the ×4.3 amplification shows legible support is better
initialization FOR ITS OWN objective; whether that geometry helps an
orthogonal reward is exactly what the frozen protocol could not ask.
A flip would mean part of the rgo advantage is objective-agnostic
domain competence — a real scope change to "support must be legible
to the objective."

## Design: 32 unfrozen adapt jobs from the EXISTING rgo/sgb fits

- Grid: {rgo, sgb} × {s0, s1} × fit seeds 1–8 (the frozen orthogonal
  wave's seeds-1–8 core; seeds 9–16 NOT included — power is
  calibrated by the unfrozen calibration, which detected at n=8 with
  d_z 2.77). `AXIS1_ADAPT_CONFIG=orth_spin_unfrozen`; everything else
  identical to the frozen orthogonal wave (TASK stays
  `dmc_finger_turn_hard`; the orthreward wrapper supplies the spin
  reward; STEPS 1.25e5).
- WM runs `ax1wm_finger_{rgo|sgb}q1s<side>_seed<f>` (E4-verified,
  same fits as the unfrozen calibration). Existence + refit
  disclosure as in U1 (buffer manifest:
  `ls $RUNROOT/axis1_finger/q1/manifest.json`).
- RUN_ID `adapt_ax1uzog<arm>q1s<side>_finger_seed<f>_ckpt500000`
  (modes `ax1uzogrgo*/ax1uzogsgb*` — distinct from the frozen wave's
  `ax1og*` and from `ax1ufz*`).

Submit loop (clean shell):

```
for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do for arm in rgo sgb; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=12:00:00 \
    --job-name=adapt_ax1uzog${arm}q1s${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1uzog${arm}q1s${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_${arm}q1s${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/q1/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_ARM=${arm},AXIS1_ADAPT_CONFIG=orth_spin_unfrozen \
    $REPO/scripts/axis1.sbatch
done; done; done
```

## Registered read (frozen `analysis/unfrozen_stress_read.py u3`)

- **VALIDITY GATE** (mirrors the frozen wave): pooled unfrozen
  orthogonal AUC100k < 20.0 ⇒ FLOOR — uninformative, no adjudication.
- **P-U3 (specificity stress)**: per-seed mean over sides of
  [rgo − sgb] on the spin objective; cluster bootstrap B=10K rng 0.
  CI entirely > 0 (above floor) ⇒ **FLIP**. Else (above floor) ⇒
  **PROTOCOL-ROBUST** (no flip); a CI entirely < 0 is additionally
  reported as OUTSIDE REGISTERED ACCOUNTS (sgb beats rgo under spin
  fine-tuning — audited, reported as-is; the reader prints this
  branch explicitly).
- Descriptors (never decisional): side simples; per-cell level
  changes vs the frozen orthogonal wave (committed baseline =
  `artifacts/orthogonal_obj_20260725/auc.csv`, seeds 1–8 rows).

## Consequence map (frozen)

- **PROTOCOL-ROBUST** ⇒ objective-specificity holds as an
  initialization fact, not just a frozen-feature fact — the band's
  orthogonal leg upgrades to two-protocol standing and the strict
  headline reading strengthens further.
- **FLIP** ⇒ the band's "orthogonal ✓ (objective-specific)" is
  re-annotated protocol-scoped; the headline's scope section
  distinguishes frozen-feature specificity (confirmed) from
  initialization generality (new fact); the theory gains a registered
  task: separate reward-direction content from domain-geometry
  content in the λ decomposition.
- **FLOOR** ⇒ reported; no further orthogonal-unfrozen compute
  without a redesign.

## Disclosure and ordering

Known at freeze: the frozen orthogonal read (+1.0 ns, above floor),
the unfrozen calibration on the SAME fits under the ORIGINAL
objective (+267.2*). Unknown: every unfrozen orthogonal quantity — no
agent has ever fine-tuned under the orthreward wrapper. Ordering:
committed with the U-wave freeze (incl. the config block) BEFORE any
submission. Compute: 32 adapt-only jobs ≤ 12 h.
