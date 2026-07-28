# PREREG U2: stamped fits unfrozen — the inclusion≠usefulness stress test (frozen 2026-07-27)

Unfrozen stress wave 2 (audit:
`research_notes/Audit_FrozenProtocol_20260726.md`, critic upgrade
medium→HIGH): included-but-frozen-unusable is exactly the
probe-vs-fine-tune configuration — the stamp is demonstrably INSTALLED
(stamp-NLL in the included band, `artifacts/stamping_20260718/`), and
UNREAL-style auxiliary tasks help precisely in end-to-end training, so
refuting them with frozen nulls is the canonical protocol confound.
This file + `analysis/unfrozen_stress_read.py u2` (selfcheck PASS) are
committed BEFORE any `ax1uz{srd0,srd1,sid}*` run exists.

## Question and registered prediction

Can full fine-tuning repurpose an installed synthetic stamp that the
frozen readout could not use? Registered prediction (spectral model):
NO — the stamped directions are reward-aligned to a SYNTHETIC signal
uncorrelated with the true task, so they provide no
initialization-geometry advantage toward the true reward; [B_srd −
B_sid] stays null unfrozen. A flip would be the UNREAL-style rescue.

## Design: 48 unfrozen adapt jobs from the EXISTING stamped fits

- Grid: {srd0, srd1, sid} × {s0, s1} × fit seeds 1–8 — the FULL
  frozen stamping grid, so the frozen primary is mirrored exactly
  (srd pooled over both function draws; sid = placement control).
- WM runs `ax1wm_finger_<code>q1s<side>_seed<f>`, code ∈ {srd0, srd1,
  sid}; REPLAY = the stamped buffer of the SAME cell
  (`$RUNROOT/axis1_finger/q1_<code>/side<side>`) — identical to the
  frozen wave except `AXIS1_ADAPT_CONFIG=unfrozen_readout`. Existence
  condition + refit disclosure rule as in U1, extended to the stamped
  buffer manifests:
  `ls $RUNROOT/axis1_finger/q1_{srd0,srd1,sid}/manifest.json`.
- RUN_ID `adapt_ax1uz<code>q1s<side>_finger_seed<f>_ckpt500000`
  (modes `ax1uzsrd0*/ax1uzsrd1*/ax1uzsid*`).

Submit loop (clean shell):

```
for code in srd0 srd1 sid; do for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do
  sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION \
    --gres=$SLURM_GRES --time=12:00:00 \
    --job-name=adapt_ax1uz${code}q1s${side}_finger_seed${f}_ckpt500000 \
    --export=ALL,REPO=$REPO,RUNROOT=$RUNROOT,CONDA_ENV=$CONDA_ENV,MANIFEST=$MANIFEST,RUN_ID=adapt_ax1uz${code}q1s${side}_finger_seed${f}_ckpt500000,WM_RUN=ax1wm_finger_${code}q1s${side}_seed${f},TASK=dmc_finger_turn_hard,SEED=${f},REPLAY=$RUNROOT/axis1_finger/q1_${code}/side${side},UPDATES=500000,STEPS=1.25e5,AXIS1_EXPL_MODE=task,AXIS1_ADAPT_CONFIG=unfrozen_readout \
    $REPO/scripts/axis1.sbatch
done; done; done
```

## Registered read (frozen `analysis/unfrozen_stress_read.py u2`)

- **P-U2 (stamp-usability stress)**: per-seed [B_srd − B_sid], B =
  s1−s0 within seed, srd averaged over the two function draws (exact
  mirror of the frozen primary); cluster bootstrap B=10K rng 0.
  CI entirely > 0 ⇒ **FLIP**. Else ⇒ **PROTOCOL-ROBUST** (no flip);
  a CI entirely < 0 is additionally reported as OUTSIDE REGISTERED
  ACCOUNTS (stamp actively interferes unfrozen — audited, reported
  as-is; the reader prints this branch explicitly, mirroring the
  frozen stamping registration's CI<0 clause).
- Descriptors (never decisional): per-arm B contrasts; per-cell level
  changes vs the frozen wave (committed baseline =
  `artifacts/stamping_20260718/auc.csv`).

## Consequence map (frozen)

- **PROTOCOL-ROBUST** ⇒ "inclusion ≠ usefulness" upgrades from a
  frozen-readout fact to a TWO-PROTOCOL fact; the UNREAL/RaMP
  refutation strengthens (aux-stamped directions are not even a
  useful initialization); P-B1's standing improves.
- **FLIP** ⇒ P-B1 and "inclusion ≠ usefulness" rescope to
  frozen-readout as registered in the audit; the UNREAL-refutation
  framing is weakened in the draft (the stamp helps end-to-end even
  though it is task-uncorrelated — an initialization/regularization
  effect the theory does not currently model; dated theory note
  required before any follow-up arm).
- Either way the frozen stamping registration is untouched.

## Disclosure and ordering

Known at freeze: the full frozen stamping record (primary null, E4
panels incl. stamp-NLL levels, true-label panels). Unknown: every
unfrozen stamped quantity — no stamped WM has ever been adapted
unfrozen. Ordering: committed with the U-wave freeze BEFORE any
submission. Compute: 48 adapt-only jobs ≤ 12 h.
