# PREREG: TM2 bridge wave — powered P-F1 contrast + reconstruction-bridge arm (P-F2) — 2026-08-12

User GO 2026-08-12 ("proceed with all paper 1 GO remaining stuff"; slate
decided 11 Aug: family plan = diagnostics → powered TM2 contrast, with
the recon bridge as "the best third-family substitute per unit cost —
it explains WHY families differ"). One wave, three arms, one read:

- **P-F1 (frozen in `PREREG_theory_R1_gating_20260811`)**: the
  aware−free contrast in TD-MPC2 is ATTENUATED relative to Dreamer's
  task−apt contrast (ratio < 1) while remaining ≥ 0. Operationalized
  per `PREREG_theory_R1_gating_amend2_20260812` as the adaptation-value
  LEVEL contrast against the pinned Dreamer anchor **+108.34** (finger
  q1, seeds 1–8, both sides, auc100k, qc-passed; computed 12 Aug from
  committed `artifacts/p0_axis1_corrective_20260713` csvs).
- **P-F2 (new, this wave's registered primary)**: ADDING observation
  reconstruction to the TM2 free arm re-opens the contrast — the rec
  arm transfers WORSE than the free arm ([free − rec] > 0). This is
  the family-level causal test: if the family boundary is carried by
  the reconstruction objective (R1: recon bids encoder capacity toward
  reconstructable features, diluting reward-alignable support), then
  importing recon into TM2 must reproduce the deficit inside the
  planner family.

## Relationship to the registered diagnostics wave (timing disclosure)

`PREREG_tm2_diag_20260811` (legibility diagnostics on the EXISTING
17-Jul aware/free fits) is registered but its cluster chain has NOT run
and its read has NOT executed at this freeze. This wave's estimands do
not consume the diagnostics read; freezing now is strictly more
pre-outcome. If the diagnostics read lands before this wave's read,
that is disclosed in the RECORD; neither read gates the other. P-F1's
"premise check" clause in R1 refers to the diagnostics wave and is
graded there.

## Honesty block

Known at freeze: the 17-Jul TM2 2×2 (aware simple +35.3 CI>0, free
+25.8 ns, contrast [B_aware−B_free] +9.5 [−43, +51] at n=8/arm-cell;
aware LEVEL ≈ 2.2× free); TM2 mpc:False read (planner carries
adaptation value); the Dreamer anchor +108.34; every executed read
through 2026-08-12. Unknown: every bridge-arm quantity and every
fresh aware/free quantity of this wave. **Value-aware note**: the
17-Jul contrast direction (positive, small) is known; the powered
re-run is registered anyway because (a) n=8→16 halves the CI, (b) the
three arms must be same-era to be comparable (Midway3 runs are not
job-to-job deterministic; cross-era pairing is banned — standing rule),
and (c) the rec arm requires the new decoder-attached fit script, so
aware/free must re-run under the identical mechanics for parameter
parity.

## Design — 3 arms × 2 sides × 8 seeds = 48 fit+adapt jobs

Substrate: the frozen finger q1 pair (`axis1_finger/q1/side{0,1}`,
KEEP_SUBSTRATE) via the existing bridge blobs
(`tm2_data/finger_q1_side{0,1}.pt`, KEEP_PENDING).

- **Fit instrument**: `probing/tdmpc2_recon_fit.py` (this
  freeze-commit; the executed `tdmpc2_offline_fit.py` is untouched).
  Official pinned TD-MPC2 (`e9f5932…`), subclassed to attach a decoder
  MLP (latent_dim → 2×mlp_dim → obs_dim) in EVERY arm (parameter
  parity; it trains only when recon_coef > 0, exactly as the free arm
  carries an untrained reward head). Arms, by loss coefficients only:
  - `aware`: official objective (consistency 20 + reward .1 + value .1),
    recon 0.
  - `free`: reward 0, value 0, recon 0 (consistency-only).
  - `rec`:  reward 0, value 0, consistency 20 (dynamics still trains),
    **recon_coef = 20.0** (registered constant; the consistency slot's
    magnitude). Recon = MSE(decode(encode(obs)), obs) over the sampled
    horizon; the gradient reaches the ENCODER. **Verified by SOURCE
    INSPECTION against the pin** (review B1): the official `_update`
    zero-grads only AFTER `optim.step()`, so the pre-accumulated recon
    grads survive into one combined step. The script's selfcheck
    (free-vs-rec encoders diverge, aware/free decoders at the replayed
    reference init, free/rec reward heads identical) is a CLUSTER gate
    that has NOT run at freeze (CUDA required on the build machine) —
    it is the registered pre-wave smoke, not an executed fact.
  - Honest scale note: recon MSE is on the raw bridged proprio vector
    (TD-MPC2 does not normalize state obs); no symlog. The
    recon-vs-consistency gradient balance at coef 20/20 is a
    registered design choice, not tuned (no pilot was run).
  - Grad-clip coupling disclosure (review B5): the official
    `clip_grad_norm_(model.parameters(), 20)` now spans the decoder in
    every arm; with raw-scale recon the rec arm may clip more often
    than aware/free. Adam's per-parameter normalization makes a
    uniform clip factor largely self-cancelling; the fit logs
    `grad_norm` per arm so the realized norms are inspectable, and the
    decoder-parity gate is a MODULE-parity guarantee, not a
    clip-behavior parity guarantee.
- **Checkpoint compatibility**: saved `model` dict strips decoder keys
  (official-class strict load verified in selfcheck) so
  `probing/tdmpc2_adapt.py` (frozen `_encoder`+`_dynamics`, fresh
  reward/Q/pi heads, MPPI planning, 1.25e5 steps) runs UNCHANGED.
- **Submission (registered command; review B4 — walltime pinned so a
  mid-adapt timeout cannot force an append-mode scores.jsonl rerun)**:
  `SLURM_TIME=12:00:00 TM2_WAVE=bridge TM2_ARMS="aware free rec" TM2_SEEDS="1 2 3 4 5 6 7 8" ./scripts/submit_all.sh tdmpc2-bundles`
  (re-run until drained — MAX_JOBS caps each invocation)
  → run ids `tm2wm_finger_b{aware,free,rec}q1s{0,1}_seed{1..8}` +
  `adapt_tm2b{aware,free,rec}q1s{0,1}_finger_seed{1..8}_ckpt500000`
  (name-collision-free: the 17-Jul fits carry no `b` infix). If an
  adapt IS truncated anyway, the recovery is delete-the-run-dir-and-
  resubmit (after archiving), never FORCE over an existing
  scores.jsonl; the reader's strict n_ep gate refuses the append
  signature regardless.
- **Pre-wave gate (registered, cluster)**:
  `TDMPC2_ROOT=$TDMPC2_CHECKOUT/tdmpc2 python -m probing.tdmpc2_recon_fit --selfcheck`
  must print SELFCHECK PASS in the tdmpc2 env (CUDA required — the
  selfcheck cannot run on the local build machine; this is the
  registered smoke, TM2-pipeline precedent; review B6: TDMPC2_ROOT is
  required for the command to be literally executable).
- **Collate + witness + bundle**:
  - `python -m analysis.adaptation_auc --runroot $RUNROOT --output analysis_out_tm2b/`
  - Witness producer (literal):

```bash
cd $RUNROOT && python - <<'PY'
import glob, json, os, re
out = {}
pat = re.compile(r'^tm2wm_finger_b(aware|free|rec)q1s[01]_seed[1-8]$')
for d in sorted(glob.glob('tm2wm_finger_b*')):
  b = os.path.basename(d)
  if not pat.match(b): continue
  kv = open(os.path.join(d, 'TM2_FIT_PROGRESS')).read().strip()
  m = re.match(r'^update=(\d+)/(\d+)$', kv)
  out[b] = dict(update=int(m.group(1)), total=int(m.group(2)),
                done=os.path.exists(os.path.join(d, 'TM2_FIT_DONE')))
json.dump(out, open('tm2b_fit_counters.json', 'w'), indent=1)
print(len(out), 'fits witnessed')
PY
```

  - Bundle: `auc.csv` + 48 fit `config.yaml`s + 48 adapt
    `config.yaml`s (runroot_light) + `tm2b_fit_counters.json`; rsync +
    sha256 manifest per results-sync policy v2.

## Registered decision rules (frozen reader `analysis/tm2_bridge_read.py`)

ONE read execution. Gates, fail-closed:

- Exact inventory: 48 auc rows (arm × side × seed), no duplicates,
  qc_pass, `n_ep_100k` STRICTLY equal to the modal count (review
  X1/B4: super-modal = the append-on-restart duplication signature).
- Fit witness: all 48 `update == total` ∧ `done` true.
- Fit config gates (the audit json): per arm, exact registered
  coefficients (aware: reward .1 / value .1 / recon 0; free: 0/0/0;
  rec: 0/0/20.0), consistency_coef identical across all 48,
  `decoder_params` identical across all 48 and > 0, data task
  `dmc_finger_turn_hard`, and the `data` manifest's **`source_dir`**
  (the key `probing/tdmpc2_bridge.py` actually writes — review B2)
  contains the run's side.
- Adapt config gates: `pretrained` names the arm/side/seed-matched WM
  run, `mpc` is True, `steps` = 125000, `frozen_prefixes` =
  ['_encoder', '_dynamics'], `n_frozen_tensors` > 0, task matches.

Estimands (auc100k; paired per (seed, side) cell, n = 16; one-sample
BCa 95% + EXACT sign-flip permutation, permutation primary):

- **P-F2 (primary)**: paired [free − rec] deltas.
  - CI > 0 ∧ p < .05 ⇒ **RECON-REOPENS** (the family boundary is
    carried by the objective; the bridge lands).
  - CI < 0 ∧ p < .05 ⇒ **RECON-HELPS** (violates the account; fact
    reported).
  - else ⇒ **INDETERMINATE** (MDE note from realized sd; licenses
    nothing).
- **P-F1 (powered contrast)**: paired [aware − free] deltas; ratio =
  values / **108.33852499999999** (pinned denominator; recipe pinned
  per review B9: aware = `artifacts/p0_axis1_corrective_20260713/
  auc_aware_1_16.csv` modes `ax1q1s{0,1}` seeds 1–8 qc-passed, mean
  188.2233; apt = `auc_p0.csv` modes `ax1fq1s{0,1}` seeds 1–8, mean
  79.884775; anchor = difference of arm-level means. The ratio CI
  treats the anchor as error-free — anticonservative, disclosed).
  - CI > 0 ∧ p < .05 ∧ ratio CI upper < 1 ⇒ **ATTENUATED-POSITIVE**
    (P-F1 confirmed in full).
  - CI > 0 ∧ p < .05, ratio upper ≥ 1 ⇒ **POSITIVE-NOT-RESOLVED**
    (contrast exists; attenuation not resolved; ratio point reported).
  - CI < 0 ∧ p < .05 ⇒ **INVERTED** (R1-violating; reported).
  - else ⇒ **NULL-CONTRAST** — registered as NON-REFUTATION, not
    confirmation (review B8): a CI spanning zero does not establish
    "≥ 0", and only ATTENUATED-POSITIVE licenses the attenuation
    claim; NULL-CONTRAST reports the MDE note and licenses nothing.
- Secondaries (descriptive, never decisional): [aware − rec], the
  closure fraction ([aware−rec] − [aware−free]) / 108.34, per-arm ×
  side cell means, per-side simples, comparison of the fresh aware/free
  cells against the 17-Jul values (labeled cross-era, non-inferential).

R1 Part-A grading of P-F1 happens in this read's RECORD against the
frozen text; P-F2 additionally feeds the family-scope section (a
RECON-REOPENS verdict upgrades the TD-MPC2 limitation from "boundary
registered" to "boundary explained — objective-carried").

## Power

Paired n=16, α=.05: the 17-Jul per-cell contrast sd was not persisted
per-cell; from CI half-width 47 at n=8-per-arm-cell design, sd ≈ 66 ⇒
MDE₈₀ ≈ 0.749 × 66 ≈ +49 in auc units at n=16 (disclosed: an
INDETERMINATE P-F2 below that scale licenses nothing; the read reports
the realized-sd MDE).

## Discipline

Shared-core note (review B10): the reader imports `one_sample` from
`analysis/domains_read.py` — a statistical core frozen jointly by the
three 12-Aug registrations; any change to it requires dated amendments
to all three. This file + `probing/tdmpc2_recon_fit.py` +
`analysis/tm2_bridge_read.py` (selfcheck PASS) + the tdmpc2.sbatch /
submit_all.sh TM2_WAVE=bridge switch + runroot_cleanup KEEPs are
freeze-committed BEFORE any bridge run exists. ONE read execution.
Reviewer: 2026-08-12 build batch (Opus 5, sequential
build→verification per the 12-Aug policy).
