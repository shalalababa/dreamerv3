# Stage-0 second-k — registered QUARANTINE + instrument audit (2026-07-31)

Registered branch of `PREREG_stage0_secondk_20260730.md` (frozen at
commit `6271c770`): the k=10 calibration smoke failed BOTH guard modes
⇒ `guard_mode = invalid` ⇒ QUARANTINE — no full passes ran, no verdict
computed. Per the frozen consequence map this is an **instrument
outcome**: "a mismatch means the single-variable property failed
(substrate changed under the pass), **not that the claim is fragile**."
The committed Stage-0 record (10/10 TRACKS_DENSITY) and its DECISION.md
stand untouched; the single-k qualifier remains in force.

Bundle: `local_results/stage0_secondk_quarantine_20260731_231323/`
(manifest verified, EXIT=0). Substrate existence gate itself PASSED
(`secondk_substrate_gate.json`: verdict "SUBSTRATE OK", all 10 runs,
replay chunks 1520–1760).

## Audit findings (value-blind: only gate + k=10 calibration quantities inspected; no k2 estimand touched — the k5 smoke was never opened)

Detailed failure record re-extracted locally by re-running the frozen
reader's `check_smoke` on the bundle against the committed root
`local_results/evpi_stage0_20260710_221924/raw`:

**Tolerance failures (7)** — every model-path float array:
`density_knn` (max_abs 4.48), `disag_steps` (2.2e-4), `err_position`
(0.021), `err_steps` (1.76), `err_velocity` (1.36), `pred_position`
(0.267), `pred_velocity` (1.61). The `true_*` arrays (numpy-computed
from the sha-pinned probeset, backend-independent) all MATCHED —
probeset and pipeline are faithful; only quantities computed through
the model/backend diverge.

**Fallback (rank ≥ 0.9) failures (3)** — the per-state horizon-error
arrays: `err_position` 0.7356, `err_steps` 0.8426, `err_velocity`
0.8101. `pred_*`, `disag_steps`, and `density_knn` kept ranks ≥ 0.9
(density shifted up to 4.48 in log-density at sparse points but stayed
rank-coherent).

## Root cause

**Cross-machine environment change.** The committed Stage-0 dumps were
computed on a Vast.ai instance (`/workspace/...` checkpoint and replay
paths in the committed `meta.json`; Vast tooling in the committed job
logs' PATH). The re-dump ran on RCC Midway3
(`/scratch/midway3/...`). Every registered dial is identical between
the two metas (probeset sha `9327e960…`, seed 0, burn_in 32,
eval_steps 8, ref_windows 2048, knn 10, disag_ens 8; the only meta
diffs are the machine-local paths, `milestone` labeling, and the new
dumper's `ep_batch` record). Different GPU class + JAX/cuDNN build ⇒
bf16 numerics + categorical draw flips, compounding over the 8-step
open-loop rollouts — exactly the drift mechanism the prereg's two-mode
guard was registered for, but at cross-machine scale, which exceeds
even the rank floor for the per-state horizon-error arrays.
Replay-byte drift is not required to explain the pattern (ranks of
`density_knn` held) but is not excluded by this comparison alone.

## Paths forward (the smoke is a GATE, retryable after an operational
fix with disclosure — the ONE read remains unexecuted and binding)

1. **RECOMMENDED — Vast-matched retry**: stage the cup-seed1 runroot
   (checkpoint snapshot + replay) on a Vast instance matching the
   committed environment (`/workspace/conda_envs/dreamerv3`, same GPU
   class as the 10-Jul lane) and re-run the two smoke dumps +
   `--smoke_check` there. If the calibration passes (tolerance or
   fallback), run the full 102 passes on that instance and sync back.
   Cost: one instance + ~3–10 GPU·h + runroot staging.
2. If a matched environment cannot be reconstructed (image/driver
   gone): QUARANTINE is terminal for this instrument — the registered
   outcome is the recorded audit + the **permanent single-k
   qualifier** on the Stage-0 externalization (the same consequence as
   SUBSTRATE-GONE, reached via the calibration branch).

No prereg amendment is required for path 1 (gate retry under corrected
environment, disclosed here); any change to guard thresholds or modes
WOULD require a dated amendment and is not proposed.
