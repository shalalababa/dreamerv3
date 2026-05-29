"""Static frame-level VAE over DMC proprio observations."""

import pickle

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax.nets as nn
from embodied.jax import DictHead
from dreamerv3 import rssm

f32 = jnp.float32


class StaticVAE(nj.Module):

  latent: int = 128
  beta: float = 1.0
  free_nats: float = 0.0
  dec_layers: int = 3
  dec_units: int = 64
  act: str = 'silu'
  norm: str = 'rms'
  logstd_min: float = -8.0
  logstd_max: float = 8.0

  def __init__(self, obs_space, enc_kw, winit='trunc_normal_in'):
    assert obs_space, obs_space
    assert all(len(s.shape) == 1 for s in obs_space.values()), obs_space
    self.obs_space = dict(obs_space)
    self.veckeys = sorted(self.obs_space.keys())
    self.enc = rssm.Encoder(self.obs_space, **enc_kw, name='enc')
    self.mean = nn.Linear(self.latent, winit=winit, name='mean')
    self.logstd = nn.Linear(self.latent, winit=winit, name='logstd')
    self.decmlp = nn.MLP(
        self.dec_layers, self.dec_units, act=self.act, norm=self.norm,
        winit=winit, name='decmlp')
    outputs = {k: 'symlog_mse' for k in self.veckeys}
    self.dechead = DictHead(
        self.obs_space, outputs, winit=winit, name='dechead')

  def encode(self, obs):
    """obs: dict of (B, d_k) arrays -> (tokens, mean, logstd)."""
    bsize = jax.tree.leaves(obs)[0].shape[0]
    reset = jnp.zeros((bsize,), bool)
    _, _, tokens = self.enc({}, obs, reset, training=False, single=True)
    tokens = nn.cast(tokens)
    mean = f32(self.mean(tokens))
    logstd = jnp.clip(f32(self.logstd(tokens)), self.logstd_min,
                      self.logstd_max)
    return tokens, mean, logstd

  def decode(self, z):
    """z: (B, latent) -> dict of per-key Output distributions."""
    x = self.decmlp(nn.cast(z))
    return self.dechead(x)

  def reparam(self, mean, logstd):
    std = jnp.exp(logstd)
    eps = jax.random.normal(nj.seed(), mean.shape, f32)
    return mean + std * eps, std

  def loss(self, obs, training=True):
    tokens, mean, logstd = self.encode(obs)
    z, std = self.reparam(mean, logstd)
    recons = self.decode(z)
    recon = jnp.zeros(mean.shape[0], f32)
    for key in self.veckeys:
      recon += recons[key].loss(f32(obs[key]))
    kl = 0.5 * (jnp.square(std) + jnp.square(mean) - 1.0 - 2.0 * logstd)
    kl = kl.sum(-1)
    kl_used = jnp.maximum(kl, self.free_nats) if self.free_nats else kl
    loss = (recon + self.beta * kl_used).mean()
    metrics = {
        'loss': loss,
        'recon': recon.mean(),
        'kl': kl.mean(),
        'elbo': -(recon + kl).mean(),
        'latent_std': std.mean(),
        'latent_absmean': jnp.abs(mean).mean(),
        'active_units': (std.mean(0) < 0.5).sum().astype(f32),
    }
    return loss, metrics

  def featurize(self, obs):
    """Return probe-site features for a batch of frames."""
    tokens, mean, logstd = self.encode(obs)
    recons = self.decode(mean)
    recon = jnp.zeros(mean.shape[0], f32)
    for key in self.veckeys:
      recon += recons[key].loss(f32(obs[key]))
    return {
        'enc': f32(tokens),
        'mean': mean,
        'logstd': logstd,
        'recon_nll': recon,
    }


def build(obs_space, enc_kw, *, latent=128, beta=1.0, free_nats=0.0,
          dec_layers=3, dec_units=64, act='silu', norm='rms', name='vae'):
  """Construct a StaticVAE with the given hyper-parameters."""
  return StaticVAE(
      obs_space, enc_kw, latent=latent, beta=beta, free_nats=free_nats,
      dec_layers=dec_layers, dec_units=dec_units, act=act, norm=norm,
      name=name)


def save(path, state, meta):
  state = jax.tree.map(lambda x: np.asarray(x), state)
  with open(path, 'wb') as f:
    pickle.dump({'state': state, 'meta': meta}, f)


def load(path):
  with open(path, 'rb') as f:
    blob = pickle.load(f)
  state = {k: jnp.asarray(v) for k, v in blob['state'].items()}
  return state, blob['meta']
