"""Intrinsic rewards for reward-free DreamerV3 runs."""

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
sg = jax.lax.stop_gradient


class Disag(nj.Module):
  """One-step latent prediction ensemble.

  head='det' (default, Stage-1-verbatim): members output point
  predictions, trained by MSE; reward = var of predictions.
  head='gauss' (Track-B4 repair arm, PREREG_trackB_alea_20260822):
  members additionally output per-dim log-variances. DESIGN NOTE
  (from the 22-Aug pre-registration toy): training mu by pure NLL
  FREEZES mu on noisy dims (1/sigma^2 gradient scaling), so var-of-mu
  gets WORSE, not better — therefore mu is trained by the SAME MSE
  as det and sigma by NLL against STOP-GRADIENTED residuals, with
  the logvar head reading a STOP-GRADIENTED copy of the shared trunk
  (rev R3-M1: without that sg, the sigma-NLL gradient reaches the
  trunk and — via per-tensor AGC on shared kernels — rescales mu's
  updates; with it, every parameter mu depends on receives ONLY the
  MSE gradient, so 'mu training is det-identical in loss AND
  gradient path' holds by construction). The repair lives in the
  DEPLOYED REWARD: var(mu) normalized per dim by the ensemble-mean
  predicted aleatoric variance. predict() returns the mu block with
  the SAME shape either way (se_probe/member_vars compatible).
  Note (rev R3-N2): logvar clipping is a hard clip — a member pinned
  at a bound has zero restoring gradient (one-way ratchet); the
  instrument records bound-fractions so this is observable.
  """

  ensemble: int = 8
  units: int = 256
  layers: int = 2
  act: str = 'silu'
  norm: str = 'rms'
  head: str = 'det'
  logvar_min: float = -8.0
  logvar_max: float = 6.0

  def __init__(self, target_dim):
    assert self.head in ('det', 'gauss'), self.head
    self.target_dim = target_dim

  def _member(self, idx, x):
    for j in range(self.layers):
      x = self.sub(f'm{idx}h{j}', nn.Linear, self.units)(x)
      x = nn.act(self.act)(self.sub(f'm{idx}n{j}', nn.Norm, self.norm)(x))
    mu = self.sub(f'm{idx}out', nn.Linear, self.target_dim)(x)
    if self.head == 'gauss':
      # separate logvar head off a STOP-GRADIENTED trunk (rev R3-M1)
      lv = self.sub(f'm{idx}lv', nn.Linear, self.target_dim)(sg(x))
      return jnp.concatenate([mu, lv], -1)
    return mu

  def _raw(self, feat, action):
    x = jnp.concatenate([nn.cast(feat), nn.cast(action)], -1)
    return jnp.stack([self._member(i, x) for i in range(self.ensemble)], 0)

  def predict(self, feat, action):
    """Per-member next-latent MEAN predictions, shape
    (ensemble, ..., target_dim) for BOTH heads."""
    raw = self._raw(feat, action)
    if self.head == 'gauss':
      return raw[..., :self.target_dim]
    return raw

  def predict_logvar(self, feat, action):
    """gauss head only: per-member predicted log-variances."""
    assert self.head == 'gauss'
    raw = self._raw(feat, action)
    return jnp.clip(raw[..., self.target_dim:],
                    self.logvar_min, self.logvar_max)

  def loss(self, feat, action, target, bootstrap=False, bootstrap_prob=0.8):
    """det: MSE. gauss: the SAME MSE on mu (identical dispersion
    dynamics) + sigma-NLL against STOP-GRADIENTED residuals (sigma
    learns the noise floor without touching mu). Identical bootstrap
    masking either way."""
    raw = f32(self._raw(sg(feat), sg(action)))
    target = sg(f32(target))[None]
    if self.head == 'gauss':
      mu = raw[..., :self.target_dim]
      logvar = jnp.clip(raw[..., self.target_dim:],
                        self.logvar_min, self.logvar_max)
      mse = ((mu - target) ** 2).mean(-1)
      sig = 0.5 * (logvar + sg((target - mu) ** 2)
                   / jnp.exp(logvar)).mean(-1)
      err = mse + sig
    else:
      err = ((raw - target) ** 2).mean(-1)
    if bootstrap:
      keep = jax.random.bernoulli(nj.seed(), bootstrap_prob, err.shape)
      err *= keep.astype(err.dtype) / jnp.maximum(f32(bootstrap_prob), 1e-6)
    return err.mean(0)

  def reward(self, feat, action):
    """Intrinsic reward. det: variance of member predictions
    (Stage-1-verbatim). gauss: the ALEATORIC-NORMALIZED disagreement
    — per-dim var(mu) over members divided by the ensemble-mean
    predicted noise variance (the B4 repair: dims whose dispersion
    is licensed by predicted irreducible noise stop paying)."""
    if self.head == 'gauss':
      raw = f32(self._raw(feat, action))
      mu = raw[..., :self.target_dim]
      logvar = jnp.clip(raw[..., self.target_dim:],
                        self.logvar_min, self.logvar_max)
      alea = jnp.exp(logvar).mean(0)
      return (mu.var(0) / (alea + 1e-8)).mean(-1)
    pred = f32(self.predict(feat, action))
    return pred.var(0).mean(-1)


def apt_reward(feat, knn=12, logc=1.0):
  """APT k-NN entropy reward for latents shaped (N, T, D)."""
  feat = f32(feat)
  k = min(knn, max(feat.shape[0] - 1, 1))

  def per_step(z):                                   # z: (N, D)
    dist2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    dist = jnp.sqrt(jnp.maximum(dist2, 0.0) + 1e-12)
    negd, _ = jax.lax.top_k(-dist, k + 1)            # nearest, includes self
    knn_dist = -negd[:, 1:]                          # drop self (distance 0)
    return jnp.log(logc + knn_dist.mean(-1))

  return jax.vmap(per_step, in_axes=1, out_axes=1)(feat)
