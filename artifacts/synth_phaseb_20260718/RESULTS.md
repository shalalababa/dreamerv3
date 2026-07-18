# Synth Phase B read — collapse-only branch fires; Phase C NOT authorized pending diagnosis (2026-07-18)

Snapshot: `local_results/synth_phaseb_20260718_133456/` (48 fit+adapt
runs: task/apt/sh × sides × seeds 1–8, domain synth). Read exactly as
registered in `prereg/PREREG_synth_phaseb_20260717.md` with the
pre-frozen `analysis/synth_phaseb_read.py`. AUC by the untouched
frozen pipeline (48/48 QC, 0 exclusions; no cluster-side csv — the
local canonical run is the sole source). Audit: **48/48 pass, 0
problems** (arm loss-flag rows, `Static replay: .../axis1_synth/...`
paths, ADAPT_DONE, QC).

## Registered read (B=10K cluster bootstrap seed 0; decisions on CI alone)

| contrast | mean | 95% CI | verdict |
|---|---|---|---|
| **PRIMARY-1 [B_task − B_apt]** (interaction) | +44.2 | [−88.0, +178.4] | **does NOT fire** (5/8, perm p=.57) |
| **PRIMARY-2 [B_sh − B_task]** (binding) | **−226.6** | **[−341.5, −96.6]** | **FIRES** (2/8 pos, perm p=.031) |
| S1 B_apt simple | +12.6 | [−20.0, +44.6] | null (4th reward-free null, engineered domain) |
| desc B_task simple | +56.8 | [−50.1, +164.8] | ns — enormous per-seed variance (deltas span −256..+375) |

**Registered branch: collapse-only** — "occupancy does not gate
transfer in synth; boundary condition; Phase C only after diagnosing
which DMC ingredient synth lacks." No Phase C registration until then.

Cell means (AUC100k; synth scale, never pooled with DMC):

| arm | s0 (occ .03) | s1 (occ .30) | benefit |
|---|---|---|---|
| task | 65.8 (sd 81) | 122.6 (sd 114) | +56.8 |
| apt | 30.9 (sd 26) | 43.5 (sd 43) | +12.6 |
| sh | **216.8 (sd 159)** | 47.0 (sd 34) | **−169.8** |

## What the pattern says (interpretation, not registered)

1. **The binding result is real but mechanically DIFFERENT from
   finger's.** In P3, shuffle attenuated the benefit toward zero
   (+31.4 vs full +159). Here shuffle **inverts** it: the sh-lo cell
   (216.8) is the highest of all six cells while sh-hi sits on the apt
   floor. Label–state binding matters in synth, but through a
   different route than DMC — report as replication of "binding
   matters," not of the DMC signature.
2. **Leading diagnosis for the null interaction (mechanism-coherent,
   checkable):** the synth obs includes `to_target` — the
   reward-aligned coordinate is a *directly observed, high-variance,
   reconstruction-decoded* key. That is the theory's **in-subspace
   regime** (the cup/E5 cell: λ_j ≥ g_k ⇒ the predictive objective
   already includes the reward direction ⇒ supervision has nothing to
   rescue ⇒ no interaction). The engineered domain landed in the wrong
   spectral cell by obs design, not by a failure of the law.
   **Checkable without new training:** an E4-style reward-NLL measure
   pass over the existing 48 synth WM checkpoints — if in-subspace,
   *apt* fits already predict reward well (low NLL), unlike DMC apt
   (2.3 vs rgo 1.2).
3. **Phase C lever if the diagnosis confirms:** hide the reward
   direction from the model — drop `to_target` from the model's inputs
   via the new `agent.model_obs` regex (built 18 Jul for the pixel
   study; e.g. `model_obs: '(?!to_target).*'`-style or generator-side
   key drop), pushing λ_j into low-variance structure. A small
   registered Phase-B′ (16–32 jobs) would then re-test the interaction
   before any full Phase C factorial.
4. The sh-lo anomaly (216.8, sd 159) is unexplained at n=8 — flagged,
   not interpreted.

## Registered consequences

- Synth does NOT enter Paper 1 as third domain. It enters (if at all)
  as a boundary-condition observation: binding matters even where
  occupancy does not gate.
- Phase C (P-B3/P-B4 host) remains unregistered; diagnosis first.
- Theory ledger: no registered synth prediction existed in
  `PREREG_theory_predictions_20260717.md` (Phase B predates the
  in-subspace check); the post-hoc in-subspace reading is labeled
  post-hoc here and would need its own prospective test (the Phase-B′
  above) to count for the theory.

## Provenance

- `auc.csv` — canonical frozen-pipeline AUC (sole source).
- `synth_phaseb_read.json` — full frozen-read output.
- Read command: `python -m analysis.synth_phaseb_read --auc auc.csv
  --runroot <snapshot>/runroot_light --output <dir>`.
