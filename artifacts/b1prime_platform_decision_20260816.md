# B1′ ridge `--platform cuda` — decision note (owning chat, 16 Aug 2026)

**Decision: NOT an amendment.** The flag is applied as a registered
contingency, disclosed here and in the read RECORD.

## Mechanism (ops diagnosis, b1p_ridge_bundle_53411405.out, midway3-0278)

`probing/ridge_probe.py --platform` defaults to `'gpu'`, which
`embodied/jax/internal.py:34` feeds to `jax.config.update('jax_platforms',
…)`. In jaxlib 0.4.33 the `gpu` alias expands to {cuda, rocm}; the rocm
member calls `_xla.GpuAllocatorConfig()`, absent from this build, and an
explicitly-requested platform list treats a failing member as fatal
(the default path skips it — why bare `import jax` works on the same
node). Every LeWM/devcheck probe passes `--platform cuda` and works; the
B1′ per-fit form inherited the default and died at backend init, before
any measurement.

## Why this is not an amendment

1. **The prereg registers the flag.** PREREG_b1prime_20260815's
   registered per-fit measure form carries: "(on Vast lanes add
   `--platform cuda` — the σ-ladder operational lesson: the default
   `gpu` is not a backend name in that JAX build)". The trigger
   condition of that clause — this build's default `gpu` alias being
   unusable — has now been shown to hold on RCC as well. Applying the
   registered contingency where its condition holds is execution, not
   registration change.
2. **Backend selection, not arithmetic.** The anchor's successful
   10-Aug run resolved the default to the cuda backend; `--platform
   cuda` selects the same backend explicitly. No kernel, precision, or
   numerics surface is touched. The stage crashed at init, so the
   change is trivially value-blind (zero outputs existed).

## Conditions and the one open tail

- **Condition (pending)**: probe 53420572 (all four `--platform` values
  through the same `jax.config` path on one node, plus jax wheel dates)
  confirms the mechanism. If it lands differently, this note is void
  and the question reopens.
- **The real tail is the env, not the flag**: the anchor panel ran the
  *same default form successfully on 10 Aug*, so if the wheels changed
  since, the anchor and the new panel would differ in software stack —
  same-hardware (`--nodelist=midway3-0278`) no longer implies
  same-numerics. **Gate before the B1′ panel is consumed**: re-measure
  3 anchor cells on 0278 under the *current* env (job 53411135's cells,
  rerun on 0278 or accepted as the joint node+env test) and compare to
  the pinned values. Identical ⇒ node and env terms both zero, proceed.
  Different ⇒ re-measure all 32 anchor cells on 0278/current-env and
  swap the pinned tuples via a **narrow dated amendment** (the
  LeWM-pe pattern) — that branch IS an amendment, this flag is not.

## Disclosure chain

This note (committed pre-outcome) → sidecar device witness beside every
ridge json (records the actual command incl. the flag) → B1′ read
RECORD cites both.
