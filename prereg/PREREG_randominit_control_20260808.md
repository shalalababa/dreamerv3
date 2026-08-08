# PREREG: random-init WM control (frozen protocol) — 2026-08-08

Review-resolution wave #15 (user GO 2026-08-08). The study has no
untrained-trunk control: without it, "reward-free transfer is null" cannot be
distinguished from "the frozen protocol scores ~80 for anything". Known at
freeze (disclosed): the entire Paper-1 record, the random-policy floor
83.48 ± 15.10, apt frozen cells ≈ 79–84, apt-unfrozen 79.2/89.9, scratch
147.7. Unknown: every outcome of the new arm.

## Design — 8 new adapt jobs, no fits

- **Arm `rif` (random-init frozen):** standard adapt protocol with
  `run.from_checkpoint=''` and `agent.frozen_wm: True` — the frozen-WM
  optimizer masking is checkpoint-independent (verified:
  `embodied/run/train.py:87` skips loading on empty path; `agent.py:133`
  masks regardless), so the frozen trunk is the seed-`k` random
  initialization. finger, size1m, seeds 1–8, 125K steps, standard
  collate (`adaptation_auc`). Everything else identical to the frozen
  Axis-1 adapt stage (task expl mode, fresh heads/actor-critic).
- **Unfrozen counterpart = the existing scratch anchor** (8 runs,
  `scratch_anchor_20260731`) — random-init unfrozen IS from-scratch
  training; reused, disclosed, no new jobs.
- SMOKE NOTE (batch-review m14): `adapt_*seed99*` is a DELETE-SAFE
  cleanup glob — bundle the smoke run's bit-identical-trunk evidence
  off-runroot BEFORE any cleanup.
- SMOKE GATE before the wave: one job (seed 99), assert the run trains
  (scores.jsonl grows), the trunk params are bit-identical at ckpt to the
  init (frozen verified), and the config audit shows
  `from_checkpoint: ''` + `frozen_wm: true` + `expl.mode: task`.

## Registered read (frozen rules; permutation-primary per the 08-08 standing rule)

Reader: `analysis/review_waves_read.py randominit` (frozen with this
file; boundary-crossing selfcheck PASS 2026-08-08). Inputs: this wave's auc.csv rows
(mode `rif`, n=8, milestone 0) + baselines via `--baseline_auc` as a LIST
of archived csvs (no single archived csv carries both populations —
verified 2026-08-08): `ax1fq1s0/1` complete rows from
`local_results/volume_repl_20260729_204854/analysis/auc/auc.csv` +
`scratch` rows from `local_results/unfrozen_u1_20260801_212204/auc/auc.csv`;
max-n_ep row wins across files (explicit D14 policy); the random floor
band is a frozen constant in the reader (83.48 ± 2×15.10).

- **P-RI1 (primary): `rif` vs pooled apt-frozen** (both sides pooled to one
  apt population, n=32 runs vs 8). Two-sided exact permutation
  (label-shuffle between groups, 100K draws rng0) on the mean difference +
  unpaired seed-cluster bootstrap CI (B=10K rng0, reported).
  - perm p ≥ .05 **AND |Δ| ≤ 15.1** (one floor-sd; batch-review M5 —
    ns alone can be pure low power at n=8 vs 32) ⇒ **reward-free ≈
    untrained** — the null family upgrades to "indistinguishable from an
    untrained network". perm ns with |Δ| > 15.1 ⇒ UNDERPOWERED, no
    wording upgrade, CI reported.
  - `rif` significantly BELOW apt ⇒ apt features carry *some* frozen
    value; the null-family wording stays "no benefit of composition",
    not "no learning" (C-R1's random-POLICY band claim is unaffected).
  - `rif` significantly ABOVE apt ⇒ negative transfer even frozen —
    report as-is (strongest form; not predicted).
- **P-RI2 (secondary): `rif` vs the random-policy floor band** — inside
  [53.3, 113.7] or not (descriptive band membership, no CI rule).
- Sensitivity: Welch t, BCa (registered as report-only).
- QC: standard ≥20-episode bar **plus** the 08-08 rule: n_ep_100k must
  equal the modal count, else the row is excluded and disclosed.

## Consequences (frozen)

- Discharges the §19 floor-censoring asymmetry with data (the same control
  reads on both the pixel and proprio floors conceptually; pixel variant
  NOT registered here).
- No headline change under any branch; wording changes only as specified.

Freeze ordering: commit this file + the frozen reader BEFORE the smoke job.
