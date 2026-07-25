# Orthogonal-objective read — does NOT fire above floor ⇒ objective-specific support (2026-07-25)

Snapshot: `local_results/orthogonal_obj_20260725_090734/` (64 adapt-only
jobs: {rgo, sgb} × {s0, s1} × fit seeds 1–16, frozen readout under the
`orth_spin_frozen` objective — dm_control finger:spin sparse reward
evaluated ON the finger_turn_hard env; only the reward channel differs
from the confirmed wave). Read exactly as registered in
`prereg/PREREG_orthogonal_obj_20260723.md` with the pre-frozen
`analysis/orthogonal_obj_read.py` (selfcheck PASS).

## Pipeline integrity

- Realized grid = the FULL registered conditional (seeds 1–16, 64
  jobs): the sgb 9–16 fit-existence condition was satisfied at
  submission; the frozen read asserts and records the grid.
- Local read on the snapshot csv is IDENTICAL to the cluster-side json
  (every key equal; `orthogonal_obj.json` here is that object).
- **VALIDITY GATE PASSED**: pooled orthogonal AUC100k level 45.3 ≥
  floor 20.0 — the sparse spin objective IS reachable at this
  adaptation budget; the read is informative (not a floor artifact).

## Registered read (cluster bootstrap over fit seeds, B=10K rng 0)

| quantity | mean | 95% CI | notes |
|---|---|---|---|
| **PRIMARY: rgo − sgb on orthogonal objective** | **+1.0** | **[−16.1, +17.0]** | **does NOT fire** (perm p=.909, d_z=.03, LOO [−3.0, +5.8]) |
| same-seed rgo − sgb on ORIGINAL objective | +45.0 | [+21.5, +68.7] | CI > 0 (descriptor; frozen `auc_pooled_1_16.csv`) |
| ratio orth/orig | 0.023 | — | registered scale caveat applies |
| side simples s0 / s1 | −9.3 / +11.4 | both include 0 | descriptive |

## Registered interpretation applied (this outcome lands the band leg)

**Objective-specific support.** The same 64 checkpoints that carry a
+45-point rgo advantage on the reward that selected their features
carry a +1-point advantage under an orthogonal reward in the same
domain, same dynamics, same observations. Per the frozen map: the
strict headline reading stands — reward-legibility governs inclusion
AND the included support serves the selecting objective specifically,
not the domain generally. The headline's scope statement does NOT
widen. Power is bounded by the realized grid (full 16 seeds — the
stronger of the two registered forms) and disclosed via the CI.

Theory bookkeeping: this is the alignment-law reading — the spectral
model's θ-direction story predicts retained features concentrate on the
reward direction that shaped g; an orthogonal θ' has no claim on them.
(Adjacent registered spectral prediction P-SM1 failed the same day —
domain-boundary account, separate leg; see
`artifacts/spectral_domains_20260725/`. The two outcomes are
independent registrations.)

## Scaling-B side observation (flagged to the runner)

The bundle's `exclusions.csv` (unrelated `ax1(f)s12*` Scaling-B adapt
rows) now lists 4 runs missing scores.jsonl: `ax1fs12q1s0_seed8`,
`ax1fs12q1s1_seed7`, `ax1s12q1s1_seed2`, `ax1s12q1s1_seed7` — a
DIFFERENT set from the X2-bundle snapshot (seed6 has since completed;
two seed7 rows appeared). These must be complete or explained before
the Scaling B read executes.

## Provenance

- `orthogonal_obj.json` — full read output (grid/gate/primary/
  descriptors/verdict).
- `auc.csv`, `exclusions.csv` — canonical AUC rows consumed.
- Read command: `python -m analysis.orthogonal_obj_read --auc <csv>
  --output <dir>`.
