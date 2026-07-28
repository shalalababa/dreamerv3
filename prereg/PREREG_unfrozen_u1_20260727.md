# PREREG U1: apt+task unfrozen 2×2 — the reward-free-null stress test (frozen 2026-07-27)

The decisive wave of the unfrozen stress program motivated by the
frozen-protocol audit
(`research_notes/Audit_FrozenProtocol_20260726.md`): a frozen null
cannot license "useless as initialization" (linear-probe vs fine-tune
gap), and the reward-free (apt) null family is the program's most
exposed claim set — E4 shows apt fits DO model the hi-occupancy
regime (d_errin arm-invariant), so the information is in the weights;
the frozen readout cannot use it; fine-tuning might. This file +
`analysis/unfrozen_stress_read.py` (subcommand `u1`, selfcheck PASS)
are committed BEFORE any `ax1uz*` run exists.

## Question and registered predictions

Does the W0 occupancy×supervision interaction persist under FULL
adaptation, and does the reward-free occupancy null survive it?
Registered prediction (theory): the interaction persists (every
unfrozen datum to date points to growth — rgo level +235 vs sgb +30);
the apt-null outcome is genuinely open and DISCRIMINATING: if the
needed direction is ABSENT from apt fits (competition/exclusion), apt
gains only generic-extension-sized level and its occupancy null holds;
if present-but-illegible, apt's occupancy benefit expresses.

## Design: 32 unfrozen adapt jobs from the EXISTING W0 1m fits

- Grid: {task, apt} × {s0, s1} × fit seeds 1–8;
  `AXIS1_ADAPT_CONFIG=unfrozen_readout` (the proven config; same WM
  initialization as frozen_readout, nothing frozen); everything else
  IDENTICAL to the frozen W0 adapt protocol (TASK
  `dmc_finger_turn_hard`, adapt seed = fit seed, STEPS 1.25e5,
  UPDATES=500000 milestone).
- WM runs: `ax1wm_finger_q1s<side>_seed<f>` (task) /
  `ax1wm_finger_fq1s<side>_seed<f>` (apt). EXISTENCE CONDITION
  (registered): verify all 32 WM dirs exist before submitting
  (`ls -d $RUNROOT/ax1wm_finger_{,f}q1s{0,1}_seed{1..8}`). If any fit
  is missing, the axis1.sbatch fit-skip guard refits it under the
  correct registered arm — any refit is DISCLOSED in the artifact and
  the frozen-vs-unfrozen paired descriptors for that cell carry a
  provenance caveat (the unfrozen-only primaries are unaffected).
  The existence check ALSO covers the buffer manifests (axis1.sbatch
  hard-requires them even on the adapt-only path):
  `ls $RUNROOT/axis1_finger/q1/manifest.json`.
- RUN_ID `adapt_ax1uztq1s<side>_finger_seed<f>_ckpt500000` (task) /
  `adapt_ax1uzfq1s<side>_finger_seed<f>_ckpt500000` (apt) — modes
  `ax1uzt*/ax1uzf*` parse under the frozen `adaptation_auc` RUN_RE and
  collide with no existing reader's regex.

Submit loop (clean shell; module-leakage gotcha applies):

```
for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=12:00:00 \
    --job-name=adapt_ax1uztq1s${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1uztq1s${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_q1s${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/q1/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_ADAPT_CONFIG=unfrozen_readout \
    $REPO/scripts/axis1.sbatch
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=12:00:00 \
    --job-name=adapt_ax1uzfq1s${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1uzfq1s${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_fq1s${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/q1/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=apt,AXIS1_ADAPT_CONFIG=unfrozen_readout \
    $REPO/scripts/axis1.sbatch
done; done
```

## Registered read (frozen `analysis/unfrozen_stress_read.py u1`)

AUC100k, seeds 1–8, domain finger, milestone 500000; per-seed paired
cluster bootstrap B=10K `default_rng(0)`; decisions on CIs alone.

- **P-U1a (interaction persists unfrozen)**: per-seed
  [B_task − B_apt] with B = s1−s0 within seed; FIRES iff CI > 0.
- **P-U1b (the null stress; two-sided adjudication)**: B_apt
  (per-seed apt s1−s0). CI entirely > 0 ⇒ **FLIP** — the frozen apt
  null was protocol-bound. Else ⇒ **PROTOCOL-ROBUST** (no flip); a CI
  entirely < 0 is additionally reported as OUTSIDE REGISTERED
  ACCOUNTS (apt occupancy actively harmful unfrozen — audited,
  reported as-is, no further adjudication; the reader prints this
  branch explicitly).
- Registered descriptors (never decisional): per-cell level changes
  vs the frozen counterparts (committed baseline =
  `artifacts/scaling_optionb_20260726/auc.csv`, 1m finger rows seeds
  1–8, paired per cell); the absence-vs-illegible discriminator (apt
  level gain vs the disclosed unfrozen-calibration anchors rgo +235.2
  / sgb +30.4 — sgb-band ⇒ absence, rgo-band ⇒ present-but-illegible).

## Consequence map (frozen)

- **a fires + b robust** ⇒ strongest outcome: headline AND null family
  generalize beyond the frozen protocol; the paper reports two-protocol
  robustness and the audit's high-risk flags close.
- **b flips (a fires)** ⇒ registered rescope: every "reward-free
  transfer null" sentence gains a frozen-readout qualifier; the
  headline keeps legibility as the STRONGER axis but drops exclusivity
  ("only under reward-capable objectives" is withdrawn); the
  initialization-geometry extension becomes a registered theory task.
- **b flips + a does not fire** ⇒ full rescope: the interaction is a
  frozen-protocol construct; Paper-1's claims stand as registered
  (protocol-scoped) but the generality section is rewritten, and the
  12m "regime A" designation (Paper 3's 25m targeting) is re-examined
  in a dated note before the 25m read.
- **a does not fire + b robust** ⇒ the interaction is
  frozen-protocol-scoped while the null family is robust; report both
  protocols side by side.
- No frozen registration changes under any branch — frozen claims
  were protocol-scoped by registration; what changes is generality
  language (the audit's prose-hygiene list executes either way).

## Disclosure and ordering

Known at freeze: the full frozen record through 27 Jul incl. the
unfrozen calibration (rgo/sgb ONLY) and the audit. Unknown: every
unfrozen task-arm and apt-arm quantity — no task-mode or apt WM has
ever been adapted unfrozen. Ordering: this file + the reader + the
`orth_spin_unfrozen` config addition (U3's, same commit) BEFORE any
U-wave submission. Compute: 32 adapt-only jobs ≤ 12 h.
