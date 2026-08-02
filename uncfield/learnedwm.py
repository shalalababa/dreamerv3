"""Learned belief world model for the NFI pilot (jax GRU, CPU-scale).

The model is a recurrent belief filter over the LG field: it consumes
(action, node, observation) streams and maintains a belief vector b, with
  - a z-head:   b -> (mu[DZ], logvar[DZ])   diagonal-Gaussian field estimate
  - an obs-head: (b, a) -> (y_mu, y_logvar) generative model for imagination.
Trained on random-policy episodes (targets: post-step latent z_after and the
incoming observation), then FROZEN. Every model exposes the same numpy API:

  init_belief() -> b
  step(b, action, node, y) -> b'        (y may be None)
  zstats(b) -> (mu, logvar)
  obs_pred(b, action) -> (y_mu, y_logvar)
  entropy(b) -> float                    (nats, diagonal Gaussian)

Also here: ExactFilterAdapter (the planted-NULL model: the true Kalman filter
behind the same API — a coherent model on which the exploit detector must
find nothing) and CorruptedModel (the planted-POSITIVE model: predicted
variance shrinks a bit on every sense regardless of evidence — an
attractor-style false-contraction the detector must flag).
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import lgfield as lg

HID = 64
D_IN = lg.N_ACTIONS + lg.N_NODES + 2      # action 1h + node 1h + [y, has_obs]
LOGVAR_MIN, LOGVAR_MAX = -12.0, 6.0


# ---------------------------------------------------------------- parameters

def _glorot(key, shape):
    fan = np.sqrt(6.0 / (shape[0] + shape[1]))
    return jax.random.uniform(key, shape, minval=-fan, maxval=fan)


def init_params(seed):
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 12)
    p = {}
    for i, g in enumerate(("z", "r", "h")):
        p[f"W{g}"] = _glorot(ks[3 * i], (D_IN, HID))
        p[f"U{g}"] = _glorot(ks[3 * i + 1], (HID, HID))
        p[f"b{g}"] = jnp.zeros(HID)
    p["zh_W1"] = _glorot(ks[9], (HID, 64))
    p["zh_b1"] = jnp.zeros(64)
    p["zh_W2"] = _glorot(ks[10], (64, 2 * lg.DZ))
    p["zh_b2"] = jnp.zeros(2 * lg.DZ)
    p["oh_W1"] = _glorot(ks[11], (HID + lg.N_ACTIONS, 64))
    p["oh_b1"] = jnp.zeros(64)
    p["oh_W2"] = _glorot(jax.random.PRNGKey(seed + 999), (64, 2))
    p["oh_b2"] = jnp.zeros(2)
    return p


# ------------------------------------------------------------------- network

def gru_step(p, b, x):
    z = jax.nn.sigmoid(x @ p["Wz"] + b @ p["Uz"] + p["bz"])
    r = jax.nn.sigmoid(x @ p["Wr"] + b @ p["Ur"] + p["br"])
    h = jnp.tanh(x @ p["Wh"] + (r * b) @ p["Uh"] + p["bh"])
    return (1.0 - z) * b + z * h


def z_head(p, b):
    h = jnp.tanh(b @ p["zh_W1"] + p["zh_b1"])
    out = h @ p["zh_W2"] + p["zh_b2"]
    mu, logvar = out[..., : lg.DZ], out[..., lg.DZ:]
    return mu, jnp.clip(logvar, LOGVAR_MIN, LOGVAR_MAX)


def obs_head(p, b, a_onehot):
    h = jnp.tanh(jnp.concatenate([b, a_onehot], -1) @ p["oh_W1"] + p["oh_b1"])
    out = h @ p["oh_W2"] + p["oh_b2"]
    return out[..., 0], jnp.clip(out[..., 1], LOGVAR_MIN, LOGVAR_MAX)


def make_input(action, node, y):
    x = np.zeros(D_IN, np.float32)
    x[action] = 1.0
    x[lg.N_ACTIONS + node] = 1.0
    if y is not None:
        x[-2] = y
        x[-1] = 1.0
    return x


def _gauss_nll(target, mu, logvar):
    return 0.5 * (logvar + (target - mu) ** 2 / jnp.exp(logvar)
                  + jnp.log(2.0 * jnp.pi))


def seq_loss(p, xs, a1h, zt, ymask, ytrue):
    """xs (T,D_IN); a1h (T,n_act); zt (T,DZ); ymask/ytrue (T,)."""
    b0 = jnp.zeros(HID)

    def step(b, inp):
        x, a, z, m, y = inp
        # obs head predicts the INCOMING observation from the prior belief
        ymu, ylv = obs_head(p, b, a)
        onll = m * _gauss_nll(y, ymu, ylv)
        b_next = gru_step(p, b, x)
        mu, lv = z_head(p, b_next)
        znll = jnp.sum(_gauss_nll(z, mu, lv))
        return b_next, znll + onll

    _, losses = jax.lax.scan(step, b0, (xs, a1h, zt, ymask, ytrue))
    return jnp.mean(losses)


def batch_loss(p, batch):
    return jnp.mean(jax.vmap(lambda *seq: seq_loss(p, *seq))(*batch))


# ------------------------------------------------------------------ training

def episodes_to_arrays(episodes):
    """List of rollout_random record-lists -> stacked training tensors."""
    xs, a1h, zt, ym, yt = [], [], [], [], []
    for recs in episodes:
        x_seq, a_seq, z_seq, m_seq, y_seq = [], [], [], [], []
        for r in recs:
            y = r["y"]
            x_seq.append(make_input(r["action"], r["node"], y))
            a = np.zeros(lg.N_ACTIONS, np.float32)
            a[r["action"]] = 1.0
            a_seq.append(a)
            z_seq.append(r["z_after"])
            m_seq.append(1.0 if y is not None else 0.0)
            y_seq.append(y if y is not None else 0.0)
        xs.append(x_seq)
        a1h.append(a_seq)
        zt.append(z_seq)
        ym.append(m_seq)
        yt.append(y_seq)
    return tuple(jnp.asarray(np.array(v, np.float32)) for v in (xs, a1h, zt, ym, yt))


def train_model(episodes, seed, steps=3000, batch=64, lr=1e-3, verbose=False):
    data = episodes_to_arrays(episodes)
    n = data[0].shape[0]
    params = init_params(seed)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def update(p, s, batch_):
        loss, grads = jax.value_and_grad(batch_loss)(p, batch_)
        upd, s = opt.update(grads, s)
        return optax.apply_updates(p, upd), s, loss

    rng = np.random.default_rng(seed)
    loss = np.nan
    for i in range(steps):
        idx = rng.integers(0, n, size=batch)
        mb = tuple(d[idx] for d in data)
        params, opt_state, loss = update(params, opt_state, mb)
        if verbose and (i % 500 == 0 or i == steps - 1):
            print(f"    [seed {seed}] step {i} loss {float(loss):.4f}")
    return params, float(loss)


def save_ensemble(path, ensemble):
    with open(path, "wb") as f:
        pickle.dump([jax.device_get(p) for p in ensemble], f)


def load_ensemble(path):
    with open(path, "rb") as f:
        return [jax.tree_util.tree_map(jnp.asarray, p) for p in pickle.load(f)]


# ------------------------------------------------------- frozen model wrapper

class LearnedModel:
    """Frozen single-member model behind the pilot's numpy API."""

    def __init__(self, params):
        self.p = params
        self._step = jax.jit(lambda p, b, x: gru_step(p, b, x))
        self._zh = jax.jit(lambda p, b: z_head(p, b))
        self._oh = jax.jit(lambda p, b, a: obs_head(p, b, a))

    def init_belief(self):
        return np.zeros(HID, np.float32)

    def step(self, b, action, node, y):
        x = make_input(action, node, y)
        return np.asarray(self._step(self.p, jnp.asarray(b), jnp.asarray(x)))

    def zstats(self, b):
        mu, lv = self._zh(self.p, jnp.asarray(b))
        return np.asarray(mu), np.asarray(lv)

    def obs_pred(self, b, action):
        a = np.zeros(lg.N_ACTIONS, np.float32)
        a[action] = 1.0
        ymu, ylv = self._oh(self.p, jnp.asarray(b), jnp.asarray(a))
        return float(ymu), float(ylv)

    def entropy(self, b):
        _, lv = self.zstats(b)
        return float(0.5 * np.sum(np.log(2.0 * np.pi * np.e) + lv))


class ExactFilterAdapter:
    """Planted-NULL model: the exact Kalman filter behind the learned-WM API.

    Belief = [mu (DZ), vec(P) (DZ*DZ)]. A coherent information accountant:
    the exploit detector must find no positive evidence-neutral cycles here.
    """

    def init_belief(self):
        return np.concatenate([lg.MU0, lg.P0.flatten()]).astype(np.float64)

    @staticmethod
    def _unpack(b):
        return b[: lg.DZ].copy(), b[lg.DZ:].reshape(lg.DZ, lg.DZ).copy()

    def step(self, b, action, node, y):
        mu, p = self._unpack(b)
        ref = lg.KalmanReferee.__new__(lg.KalmanReferee)
        ref.mu, ref.p = mu, p
        if action >= lg.N_NODES:
            s = lg.SENSORS[action - lg.N_NODES]
            ref.update(s, y)
        ref.predict()
        return np.concatenate([ref.mu, ref.p.flatten()])

    def zstats(self, b):
        mu, p = self._unpack(b)
        return mu, np.log(np.diag(p) + lg.JITTER)

    def obs_pred(self, b, action):
        mu, p = self._unpack(b)
        s = lg.SENSORS[action - lg.N_NODES]
        return float(s.c @ mu), float(np.log(s.c @ p @ s.c + s.r))

    def entropy(self, b):
        # Full joint entropy (not the diagonal surrogate): the adapter is the
        # coherence reference, so its accounting must match the referee's.
        _, p = self._unpack(b)
        return lg.entropy(p)


class DiagExactFilterAdapter(ExactFilterAdapter):
    """Planted-NULL variant using the DIAGONAL-SUM entropy surrogate — the
    functional the learned members are actually scored with. Certifies
    false-positive immunity for that surrogate, not just full-logdet."""

    def entropy(self, b):
        _, lv = self.zstats(b)
        return float(0.5 * np.sum(np.log(2.0 * np.pi * np.e) + lv))


class CorruptedModel:
    """Planted-POSITIVE model: wraps a base model; every sense action shrinks
    the predicted logvar by delta regardless of evidence (attractor-style
    false contraction). Belief = [base_b, sense_count].

    floor=True clips the corruption at LOGVAR_MIN — the SATURATING variant
    (farming completes inside the burn window; must be caught by the
    transient tier, not the steady tiers)."""

    def __init__(self, base, delta=0.05, floor=False):
        self.base = base
        self.delta = delta
        self.floor = floor

    def init_belief(self):
        return np.concatenate([self.base.init_belief(), [0.0]])

    def step(self, b, action, node, y):
        inner = self.base.step(b[:-1], action, node, y)
        count = b[-1] + (1.0 if action >= lg.N_NODES else 0.0)
        return np.concatenate([inner, [count]])

    def zstats(self, b):
        mu, lv = self.base.zstats(b[:-1])
        lv = lv - self.delta * b[-1]
        if self.floor:
            lv = np.maximum(lv, LOGVAR_MIN)
        return mu, lv

    def obs_pred(self, b, action):
        return self.base.obs_pred(b[:-1], action)

    def entropy(self, b):
        _, lv = self.zstats(b)
        return float(0.5 * np.sum(np.log(2.0 * np.pi * np.e) + lv))


def ensemble_disagreement(models, beliefs):
    """Variance of z-head means across ensemble members (scalar summary)."""
    mus = np.stack([m.zstats(b)[0] for m, b in zip(models, beliefs)])
    return float(np.mean(np.var(mus, axis=0)))
