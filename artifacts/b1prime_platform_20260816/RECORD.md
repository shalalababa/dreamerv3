# B1′ ridge: why it crashed, and why `--platform cuda` is the anchor-faithful form

**2026-08-16, ops.** Diagnostic evidence behind
`artifacts/b1prime_platform_decision_20260816.md`. Logs in this directory are
the producers' own; nothing here is transcribed.

## Confirmed mechanism

`gpu_platform_probe_53420572.out` (midway3-0278, Tesla V100, driver
535.216.03, jax/jaxlib 0.4.33) exercises the *same call path* the ridge probe
takes — `jax.config.update('jax_platforms', …)` then `jax.devices()`:

| `jax_platforms` | result |
|---|---|
| `gpu` | **RuntimeError: Unable to initialize backend 'rocm' … no attribute 'GpuAllocatorConfig'** |
| `cuda` | `[CudaDevice(id=0)]` |
| *unset* | `[CudaDevice(id=0)]` |
| `rocm` | same RuntimeError |

The `gpu` and `rocm` failures are the *same error*: `'gpu'` expands to the GPU
platforms, rocm is in that expansion, rocm's init calls
`_xla.GpuAllocatorConfig()` which this jaxlib build does not export, and an
explicitly-requested platform that fails is fatal rather than skipped. Unset
lets JAX pick and it picks CUDA.

Chain to the observed failure:
`probing/ridge_probe.py:430` defaults `--platform` to `'gpu'` →
`embodied/jax/internal.py:34` (`platform and jax.config.update('jax_platforms',
platform)`) → `embodied/jax/agent.py:72` calls `jax.devices()` → the above.

## What this rules out

* **Not the node.** `v100_backend_probe_53415197.out` (0278) and
  `rtx_backend_probe_53415597.out` (0284, Quadro RTX 6000) both bring CUDA up
  as-is, with `JAX_PLATFORMS=''` and with `=cuda`. Same driver on both.
* **Not the architecture.** V100 (cc 7.0) and RTX 6000 (cc 7.5) behave
  identically.
* **Not node-specific bad state.** `ridge_devnode_check_53411135.out` hit the
  byte-identical traceback on **0279**, a different V100.
* **Not an env change.** Every `jax`, `jaxlib`, `jax_cuda12_plugin`,
  `jax_cuda12_pjrt` and `nvidia_cudnn_cu12` dist-info dates to **2026-05-09**,
  three months before the 10-Aug anchor. The only site-packages change since
  is 2026-08-15 10:32 — the `vastai` CLI and its networking dependency tree
  (aiohttp, rich, cryptography, borb, qrcode …). Nothing numeric.

## The inference that matters for the flag

The anchor panel (job 53243601, 10 Aug) succeeded: sacct puts all four array
tasks on **midway3-0278** and its logs report `JAX devices (1): [cuda:0]`.
The wheels then are the wheels now. But `jax_platforms='gpu'` is *fatal* with
these wheels.

**Therefore the anchor cannot have run with `jax_platforms='gpu'`.** Its
effective platform was `cuda` or unset — both of which resolve to
`CudaDevice`, exactly what its log records.

So `--platform cuda` does not deviate from the anchor: it **reproduces** the
anchor's backend, and the bare default form — which v1 (53411405) used and
which crashed — is the one that would not have. The flag makes the new panel
*more* faithful to the pinned literals, not less.

Consistent with this: the anchor's producer (`refit_ridge_panel.sbatch`, no
longer in the repo) printed `ridge amended default ep_batch` in its own logs,
so that invocation was not bare. Note `internal.py:34` only fires when
`platform` is truthy, so an empty platform string leaves `jax_platforms`
unset — a sufficient explanation, though not one this evidence can single out.

**Left open deliberately** (user's call, 16 Aug): the exact form the anchor's
vanished producer used. It is harmless — every future ridge invocation passes
`--platform cuda` explicitly, so the default-alias path is not reachable
again.

## Files

| file | what it is |
|---|---|
| `gpu_platform_probe_53420572.out` | the four-value comparison above (0278) |
| `v100_backend_probe_53415197.out` | backend init on 0278, V100 |
| `rtx_backend_probe_53415597.out` | backend init on 0284, RTX 6000 |
| `ridge_devnode_check_53411135.out` | same crash on 0279 (node-independence) |
| `b1p_ridge_bundle_53411405.out` | v1's failure: 16/16 cells, nothing measured |
