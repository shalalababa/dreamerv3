"""Reward-free exploration objectives for world-model pretraining.

Two intrinsic-reward mechanisms, selected by `agent.expl.mode`:

  * Disag       -- Plan2Explore (C3). An ensemble of one-step latent
                   predictors; the intrinsic reward is the disagreement
                   (variance) of their predictions of the next stochastic
                   latent. Disagreement is high in states the world model is
                   still uncertain about, so the exploration actor is pulled
                   toward novel dynamics.
  * apt_reward  -- APT (C4). A particle-based entropy estimate over
                   world-model latents: per-state reward proportional to
                   log(c + mean k-NN distance) within the batch of imagined
                   latents. Maximising it spreads the policy's state coverage.

Both are consumed by `dreamerv3/agent.py` inside the imagination rollout, in
place of the task reward head.
"""

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
sg = jax.lax.stop_gradient


class Disag(nj.Module):
  """Plan2Explore one-step latent-disagreement ensemble.

  Each ensemble member is a small MLP that predicts the next stochastic
  latent from the current model state and action. Members are initialised
  independently (distinct ninjax paths), so their predictions diverge
  precisely where the training data was sparse -- that spread is the
  intrinsic reward.
  """

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

  def loss(self, feat, action, target):
    """Mean squared one-step prediction error, averaged over the ensemble.

    Inputs are stop-gradient'd: the ensemble adapts to the world model's
    representation, it must not perturb the world model in return.
    """
    pred = f32(self.predict(sg(feat), sg(action)))
    target = sg(f32(target))[None]
    return ((pred - target) ** 2).mean(-1).mean(0)

  def reward(self, feat, action):
    """Intrinsic reward: variance of the ensemble's predictions."""
    pred = f32(self.predict(feat, action))
    return pred.var(0).mean(-1)


def apt_reward(feat, knn=12, logc=1.0):
  """APT particle-entropy reward over a batch of latents.

  `feat` is (N, T, D). At each horizon step the N parallel imagined latents
  form the particle set; a particle's reward is log(c + mean distance to its
  k nearest neighbours), a non-parametric estimate of latent-space entropy.
  Returns (N, T).
  """
  feat = f32(feat)
  # Clamp k to the particle count (matters only for tiny debug batches; real
  # imagination batches have B*K ~ 1000 particles).
  k = min(knn, max(feat.shape[0] - 1, 1))

  def per_step(z):                                   # z: (N, D)
    dist2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    dist = jnp.sqrt(jnp.maximum(dist2, 0.0) + 1e-12)
    negd, _ = jax.lax.top_k(-dist, k + 1)            # nearest, includes self
    knn_dist = -negd[:, 1:]                          # drop self (distance 0)
    return jnp.log(logc + knn_dist.mean(-1))

  return jax.vmap(per_step, in_axes=1, out_axes=1)(feat)
