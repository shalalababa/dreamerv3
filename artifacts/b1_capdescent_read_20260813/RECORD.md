# B1 capacity-descent λ trial — ONE registered read — 2026-08-13

- Registration: `prereg/PREREG_capdescent_20260811.md` (sha256
  87f427af987644317a658d1fcf1743153fcd00e97324030b018c8c706b95c2ac,
  freeze-committed 8cd62845 on 2026-08-12, before any sub-1m fit
  existed anywhere in the portfolio). Part-B predictions frozen
  earlier in `PREREG_theory_R1_gating_20260811` (98015404…5bb8).
- Bundle: `local_results/b1_capdescent_20260813_221524` —
  MANIFEST.sha256 verified (101 entries, all OK; identical to
  committed manifest) before any file was opened. 32 fit + 32 adapt
  run dirs, 32 ridge jsons, auc/e4 csvs, witness jsons.
- Reader: `analysis/capdescent_read.py` (sha256 dbca4e5f…416c4),
  selfcheck PASS re-run immediately before execution. THIS is the
  wave's single read. All gates passed: 32-name dual witness
  (update==total per name, ckpt 500000), per-fit config gates (deter
  256/128 by size, expl mode by arm), per-adapt gates (registered WM
  from_checkpoint, frozen, **policy.units == 64** readout pin), QC +
  modal n_ep, exact inventory, ridge witness_match.

## Verdict (registered rules)

**INSTRUMENT-LIMITED: both sizes FLOOR-CENSORED — the descent
overshot; a shallower grid is a new registration.**

- **Floor gate** (registered constant 113.7 = random-floor 83.5 +
  2×15.1): task-arm mean auc100k = **73.6 at sk3 (300k)** and **97.5
  at sk1 (100k)** — BOTH below the floor. At these capacities even
  the reward-legible task arm cannot learn the behavior, so the cells
  carry no competition information (both λ and R1 predict collapse);
  P-B1b (the DID discriminator) is **not computed** per the frozen
  rule.
- **P-B1a (per-size, secondary, reported)**: I(sk3) = −5.6
  [−33.6, +16.3] p=.685; I(sk1) = +4.1 [−26.0, +80.9] p=.955 — tight
  nulls at both sizes, exactly the both-arms-collapsed shape the
  floor gate exists to label.
- **P-B1c (legibility, unconditional but headroom-gated):
  APT-FLOOR-CENSORED** — mean apt AUROC at sk3 = 0.413 < 0.55 (the
  disclosed-likely outcome; the executed size1m apt anchor was
  0.401). The legibility DID cannot discriminate λ from R1 here.
  Panel means: task 0.847 (sk3) / 0.697 (sk1); apt 0.413 / 0.392.

## What this licenses (per the frozen texts)

- **λ-negative mapping (Amendment 1): INSTRUMENT-LIMITED counts as
  NEITHER λ-positive nor λ-negative.** B1 contributes no weight to
  the Part-B joint rule; the λ adjudication now rests entirely on
  B2 (`PREREG_nuisance_20260811`).
- A shallower capacity grid (between 300k and 1m) would be a NEW
  registration — NOT authorized by this read; and the standing 25m
  cap governs the other direction.

## Post-read notes (labeled, non-registered)

- The floor result is itself a calibration fact: the transfer
  interaction machinery needs task-arm learnability, and that is lost
  somewhere between size1m (task mean ≈ 188 executed) and 300k.
  Non-monotonicity of the task means (73.6 at 300k vs 97.5 at 100k)
  is within-noise (per-cell sd basis ~77 on n=8) — no size ordering
  claim.
- Task-arm LEGIBILITY survives the descent far better than behavior
  (AUROC 0.847/0.697 vs behavioral floor) — representation-level
  reward legibility degrades gracefully while control collapses.
  Descriptive lead only; any use needs its own registration.
