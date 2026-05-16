# Methods and theory

This note derives the static-VAE objective implemented in `vae.py` and states
what the probing `R²` (in `probe.py`) does and does not measure. It is the
theory companion to the Week-7/8 implementation.

## 1. The static VAE objective

### 1.1 Evidence lower bound

The static VAE is a latent-variable model of a single observation frame `x`:

```
p_θ(x, z) = p(z) · p_θ(x | z),     p(z) = N(0, I).
```

The marginal log-likelihood `log p_θ(x) = log ∫ p(z) p_θ(x|z) dz` is
intractable. Introduce a variational posterior `q_φ(z|x)` and apply Jensen's
inequality to the importance-weighted form:

```
log p_θ(x) = log E_{q_φ(z|x)} [ p(z) p_θ(x|z) / q_φ(z|x) ]
          ≥  E_{q_φ(z|x)} [ log p_θ(x|z) ]  −  KL( q_φ(z|x) ‖ p(z) )
          =: ELBO(x; θ, φ).
```

The inequality is **tight up to the posterior gap**: subtracting the ELBO
from the exact log-likelihood gives

```
log p_θ(x) − ELBO(x; θ, φ) = KL( q_φ(z|x) ‖ p_θ(z|x) )  ≥  0,
```

because `q_φ(z|x) p_θ(x) = p_θ(x,z) = p_θ(z|x) p_θ(x)` makes the two ratios
inside the expectation differ exactly by `p_θ(z|x)/q_φ(z|x)`. So maximising
the ELBO simultaneously fits the model and drives `q_φ` toward the true
posterior.

### 1.2 β-VAE form (what the code minimises)

`vae.py` minimises the per-frame loss

```
L_β(x) = E_{q_φ(z|x)}[ −log p_θ(x|z) ]  +  β · KL( q_φ(z|x) ‖ p(z) )
       = recon(x)                       +  β · kl(x).
```

`β = 1` is exactly the negative ELBO. `β` is a knob on the rate–distortion
trade-off: larger `β` compresses the latent harder (higher distortion, lower
rate). The default is `β = 1` to keep the model a standard VAE (Lecture 6);
`free_nats` optionally floors the KL to mitigate posterior collapse.

### 1.3 KL for diagonal Gaussians

With `q_φ(z|x) = N(μ, diag σ²)` and `p(z) = N(0, I)` in dimension `d`, the KL
has the closed form used in `StaticVAE.loss`:

```
KL( N(μ, diag σ²) ‖ N(0, I) ) = ½ Σ_{i=1..d} ( σ_i² + μ_i² − 1 − log σ_i² ).
```

This is the diagonal case of the general Gaussian KL
`½[ tr(Σ_p⁻¹Σ_q) + (μ_p−μ_q)ᵀΣ_p⁻¹(μ_p−μ_q) − d + log(detΣ_p/detΣ_q) ]`
with `Σ_p = I`, `μ_p = 0`.

### 1.4 Reparameterisation

The gradient of `E_{q_φ(z|x)}[·]` w.r.t. `φ` uses the reparameterised sample

```
z = μ(x) + σ(x) ⊙ ε,     ε ~ N(0, I),
```

which moves the stochasticity off the parameter path and yields a
low-variance unbiased estimator of `∇_φ ELBO` (`reparam` in `vae.py`).

### 1.5 Observation model — matched to DreamerV3

DreamerV3 decodes proprioceptive vectors with a **symlog-MSE** loss, where
`symlog(x) = sign(x)·log(1+|x|)`. The static VAE uses the identical decoder
output (`symlog_mse`), so

```
−log p_θ(x|z) = ½ ‖ symlog(x) − d(z) ‖²  +  const,
```

i.e. a unit-variance Gaussian observation model in symlog space. Because both
the RSSM and the VAE use this same `p(x|z)`, the **held-out predictive
log-likelihoods reported by `features.py` are computed identically** and are
directly comparable across the two representations.

### 1.6 Why this is the right control

DreamerV3's RSSM is trained (per `rssm.py`) with a reconstruction term plus a
KL between a *categorical* posterior `q(s_t | h_t, x_t)` and a *categorical*
prior `q(s_t | h_t)` predicted from the recurrent state `h_t`. The static VAE
keeps the encoder, the decoder, and the observation model **identical** and
removes only (a) the recurrent state `h_t` and (b) the dynamics/prior. The KL
of the VAE is against a fixed `N(0,I)` instead of a learned temporal prior.
Hence a difference in probe performance between the two latents is
attributable to **sequential structure**, not to architecture or
preprocessing.

## 2. What probe R² measures

For a probe `g` from the (fixed) ridge or small-MLP family, fit on a
representation `z` to predict a target `y`, the held-out score is

```
R² = 1 − Σ (y − g(z))² / Σ (y − ȳ_train)² ,
```

with the baseline `ȳ_train` the *train-set* mean (so an uninformative
representation scores ≈ 0, and `R² < 0` is possible on held-out data).

* `R²` is **monotone in decodable information under a fixed probe family.**
  It is not the mutual information `I(y; z)`; it is a lower bound on the
  variance of `y` explainable by a smooth function of `z`. For the optimal
  predictor, `MMSE = Var(y)·(1 − R²*)` and, for Gaussian-ish `(y,z)`,
  `I(y; z) ≥ −½ log(1 − R²*)`. A probe gives `R²_probe ≤ R²*`, so the bound
  is loose — which is why the experiment relies on **relative** comparisons
  under one fixed probe family rather than absolute information values.
* Fixing the probe family is what makes the comparison a *controlled
  measurement*: encoder, posterior, prior and VAE latents are all scored by
  the same ridge/MLP, so differences reflect the representation.

### 2.1 The three contrasts, formalised

Let `R²(site, target, k, K)` be the score of `site` predicting `target` at
horizon `k` with VAE receptive field `K`.

* **Per-frame content** — `R²(posterior, state, 0, ·) − R²(vae_latent,
  state, 0, 1)`. Both sites see exactly one observation frame; the gap
  isolates what the recurrent state adds at zero horizon.
* **Predictive content** — `R²(posterior, state, k, ·)` as `k` grows. The
  decay rate quantifies how much of the *future* trajectory the latent
  encodes.
* **Predictive model (prior)** — `R²(prior, state, 0)` and
  `R²(imag_k, state, k)`. The prior/imagined feature never consumed the
  observation at the predicted step, so a non-zero score isolates what the
  learned **dynamics** (not the encoder) has captured.
* **Post-hoc aggregation** — `R²(vae_latent, state, 0, K)` as `K` grows
  through `{1,4,16,all}`. This measures how much of the recurrence gap a
  *static* model recovers simply by concatenating past latents — turning
  "static vs sequential" into an empirical curve rather than a framing claim.

Held-out generative likelihood (§1.5) is reported alongside `R²` so that
generative quality and probe quality can be correlated across configurations
and seeds (Week-9 analysis).
