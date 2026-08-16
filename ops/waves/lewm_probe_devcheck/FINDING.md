# E4 ridge probe: deterministic within a GPU, NOT comparable across GPUs

**2026-08-16.** Diagnostic, not a registered read. Nothing here grades a
hypothesis; it establishes a measurement constraint that the LeWM E4 read
must respect.

## Question

The fpx probe panel was being measured on two different GPUs: 3 cells landed
on RCC Slurm before `cudnn status: 5003` killed the rest, and the other 13 ran
on a Vast RTX 5060 Ti. Are the two halves comparable?

## Design

Re-measure on the Vast box the three cells RCC had already measured
(`s0_seed6`, `s1_seed1`, `s1_seed3`) — the only cells where a same-cell
cross-device pair can exist — with the identical command and identical
`ep_batch 4`. Then repeat those three **again on the same Vast GPU**, because a
single measurement per device cannot separate a device effect from ordinary
run-to-run nondeterminism.

Outputs went to side directories (`lewm_probe_devcheck/`,
`lewm_probe_devcheck_rep/`) so no panel value could be overwritten by the
check. The RCC originals are preserved in `lewm_probe_rcc_slurm/`.

## Result

Same GPU, run twice: **byte-identical**. Not "close" — the same sha256.

| cell | dev-run-1 | dev-run-2 | RCC |
|---|---|---|---|
| s0_seed6 | `9b6dc8bdbb86c1b0` | `9b6dc8bdbb86c1b0` | `323305d8e10ff47d` |
| s1_seed1 | `6d3cd8a2b9a962d8` | `6d3cd8a2b9a962d8` | `8dcc7e693f049422` |
| s1_seed3 | `ac8c72949f45820d` | `ac8c72949f45820d` | `1afa7e4e8a26b394` |

Across 3 cells × 3 alphas, max |difference|:

| metric | cross-device | same-device |
|---|---|---|
| `auroc` | 2.12e-02 | **0** |
| `auroc_in_regime` | 1.13e-01 | **0** |
| `r2` | 2.12e-02 | **0** |
| `r2_in_regime` | 1.66e+01 | **0** |

The noise floor is exactly zero, so **every bit of the cross-device gap is the
device**. It is not small: `auroc_in_regime` on `s0_seed6` moves 0.743 → 0.817,
and on `s1_seed3` 0.432 → 0.521. Differences are almost all one-signed
(Vast > RCC in 33 of 36 comparisons) — a systematic shift, not jitter.

## Mechanism

Everything downstream of the forward pass is device-independent by
construction: `episode_folds` is `arange(N) % k` (no RNG), and the ridge solve,
AUROC, R² and NLL are float64 NumPy on the CPU. The forward pass is not. It
runs under `compute_dtype: bfloat16` — an 8-bit mantissa — through a
**recurrent** `dyn.observe` over T steps, and `post['stoch']` is a categorical
sample (`stoch 32 × classes 4`) where a near-tie logit can flip a one-hot
outright. Different GPU architectures pick different fused kernels and
reduction orders; bf16 rounding differences compound through the recurrence and
can become discrete at the sampling step.

The forward-pass signature is directly visible: `deter_std_recomputed` differs
across devices at ~1e-5 (7.9e-06 / 9.3e-05 / 5.0e-05) while
`deter_std_summary`, which is not recomputed, matches to the last digit.

That a ~1e-5 perturbation in the features moves an AUROC by 0.11 is itself
worth recording: at `alpha` as low as 1e-4 over `feat_dim 640`, with
`r2_in_regime ≈ −1084`, the fit is badly ill-conditioned. The device difference
does not create that instability, it reveals it. The probe's own json already
carries `"exploratory": "... not frozen; no registered decision consumes this"`.

## Consequence (applied)

The three RCC-measured cells were **replaced in `lewm_probe/` by their Vast
measurements**, so all 16 panel cells now come from one GPU
(`probe_e4: 16/16 DONE`, verified). The RCC values are not lost — they remain
in `lewm_probe_rcc_slurm/`, and the Vast copies also remain in
`lewm_probe_devcheck/`.

**Standing constraint: never mix GPUs within a probe panel.** A panel split
across devices carries a device term of the same order as the effects these
probes are used to detect. Where a leg must move hardware mid-flight, re-measure
the already-done cells on the new device rather than pooling — it is cheap
(minutes per cell) and, as shown here, exactly reproducible.
