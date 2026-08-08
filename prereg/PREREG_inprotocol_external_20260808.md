# PREREG: external reward-free method in-protocol (Plan2Explore) — 2026-08-08

Review-resolution wave #20 (user GO 2026-08-08). Converts limitation-item 7
("no external method baselines") from rebuttal to result at minimal cost:
the Phase-4 **online Plan2Explore runs already exist**
(`pretrain_p2e_finger_seed1..5` — full online reward-free training, not our
offline fit protocol), and Plan2Explore is the canonical published
frozen-model reward-free-transfer positive (Sekar et al. 2020). Known at
freeze: the whole record incl. the reward-free nulls and scratch anchor.
Unknown: every outcome of the new adapts.

## Design — 10 new adapt jobs, no fits/training

- **Arm `p2eof` (p2e online, frozen):** standard frozen adapt
  (`frozen_wm: True`, `from_checkpoint = pretrain_p2e_finger_seed<k>/ckpt`),
  finger turn_hard, seeds 1–5 (the 5 existing donors; adapt seed = donor
  seed), 125K steps.
- **Arm `p2eou` (p2e online, unfrozen):** same donors, `frozen_wm: False`,
  seeds 1–5. (10 jobs total.)
- Protocol note (disclosed, scope-limiting): these donors trained ONLINE
  with p2e intrinsic reward for 500K steps (CORRECTED per batch review
  M7 — the earlier draft said 1.1M; snapshot configs show
  steps=500000) — still a representative of
  published reward-free pretraining than our offline apt fits; they are
  reconstruction+dynamics objectives, NOT successor-structured, so the
  objective-taxonomy defence predicts they stay null.
- SMOKE GATE: one job asserts checkpoint loads (`enc|dyn|dec` regex),
  config audit (`frozen_wm`, `expl.mode: task`), scores flow.
- DONOR PREFLIGHT (before freeze-commit): confirm
  `pretrain_p2e_finger_seed{1..5}/ckpt` exist on scratch (local bundles
  carry manifests only; purge risk is live). If any donor is gone, the
  wave re-registers on the survivors as a dated amendment — never a
  silent seed-set change.

## Registered read (`analysis/review_waves_read.py external`, frozen with this file; selfcheck PASS)

- Baselines: `--baseline_auc` list = the volume_repl csv (`ax1fq1s*`) +
  the U1 csv (`scratch`), max-n_ep merge (same convention as the
  random-init read).
- **P-X1 (primary): `p2eof` vs pooled apt-frozen** (n=5 vs 32), exact
  permutation two-sided + unpaired bootstrap CI. Null ⇒ the published
  method's own online representation is also frozen-null in-protocol —
  the apt arm is a fair family representative. Positive ⇒ family-scope
  correction: our offline apt fits under-represent the method; report and
  scope the reward-free null to offline fits.
- **P-X2 (secondary): `p2eou` vs scratch** (n=5 vs 8), same machinery.
  Below-scratch ⇒ the negative-transfer claim (C-R2) extends to the
  canonical external method; ≈/above ⇒ C-R2 stays scoped to offline apt
  fits (disclosed limitation).
- Descriptive: band membership vs the random-policy floor.
- QC: standard + modal-n_ep rule. n=5 is small — CIs reported, TOST not
  attempted (disclosed power limit; this is a scope probe, not an
  equivalence claim).

Freeze ordering: commit this file + reader before the smoke job.
