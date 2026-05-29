"""Intrinsic rewards for reward-free DreamerV3 runs."""

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
sg = jax.lax.stop_gradient


class Disag(nj.Module):
  """One-step latent prediction ensemble."""

  ensemble: int = 8
  units: int = 256
  layers: int = 2
  act: str = 'silu'
  norm: str = 'rms'

  def __init__(self, target_dim):
    self.target_dim = target_dim

  def _member(self, idx, x):
    for j in range(self.layers):
      x = self.sub(f'm{idx}h{j}', nn.Linear, self.units)(x)
      x = nn.act(self.act)(self.sub(f'm{idx}n{j}', nn.Norm, self.norm)(x))
    return self.sub(f'm{idx}out', nn.Linear, self.target_dim)(x)

  def predict(self, feat, action):
    """Per-member next-latent predictions, shape (ensemble, ..., target_dim)."""
    x = jnp.concatenate([nn.cast(feat), nn.cast(action)], -1)
    return jnp.stack([self._member(i, x) for i in range(self.ensemble)], 0)

  def loss(self, feat, action, target, bootstrap=False, bootstrap_prob=0.8):
    """Mean squared prediction error, averaged over ensemble members."""
    pred = f32(self.predict(sg(feat), sg(action)))
    target = sg(f32(target))[None]
    err = ((pred - target) ** 2).mean(-1)
    if bootstrap:
      keep = jax.random.bernoulli(nj.seed(), bootstrap_prob, err.shape)
      err *= keep.astype(err.dtype) / jnp.maximum(f32(bootstrap_prob), 1e-6)
    return err.mean(0)

  def reward(self, feat, action):
    """Intrinsic reward: variance of the ensemble's predictions."""
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
