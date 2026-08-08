"""Learned belief world models for the DISCRETE-CHAIN family (family 2).

Mirror of learnedwm.py for dcfield: a recurrent belief filter consumes
(action, node, symbol) streams; heads are categorical —
  z-head:   b -> 8 Bernoulli logits (per-chain state estimate)
  obs-head: (b, a) -> 2 symbol logits (generative model for imagination).
Recurrent CELLS are imported from learnedwm (gru_step / lstm_step /
cell_step / belief_width) so the family factorial varies architecture with a
single shared implementation. Entropy surrogate mirrors family 1's diagonal
sum: total belief entropy = sum of per-chain Bernoulli entropies (nats).

Frozen numpy API consumed by the world-parameterized planner (categorical
observation kind):
  init_belief() -> b
  step(b, action, node, y) -> b'     (y = symbol int, or None for moves)
  obs_probs(b, action) -> (N_SYMBOLS,) predictive symbol distribution
  entropy(b) -> float                 (nats)

Planted battery (roles mirror learnedwm's):
  ExactChainAdapter     planted NULL — exact 256-state joint filter,
                        full-joint Shannon entropy
  MarginalChainAdapter  planted NULL #2 — same filter scored with the
                        per-chain-marginal-sum surrogate the learned members
                        actually use
  CorruptedDCModel      planted POSITIVE — every sense action lowers claimed
                        entropy by delta regardless of evidence (floor=True:
                        saturating variant for the transient tier)
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import dcfield as dc
from .learnedwm import (_glorot, belief_width, cell_step)

HID = 64
D_IN = dc.N_ACTIONS + dc.N_NODES + dc.N_SYMBOLS + 1   # act 1h + node 1h + sym 1h + has_obs


# ---------------------------------------------------------------- parameters

def init_params_dc(seed, hid=HID, cell="gru"):
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 16)
    p = {}
    if cell == "gru":
        for i, g in enumerate(("z", "r", "h")):
            p[f"W{g}"] = _glorot(ks[3 * i], (D_IN, hid))
            p[f"U{g}"] = _glorot(ks[3 * i + 1], (hid, hid))
            p[f"b{g}"] = jnp.zeros(hid)
    elif cell == "lstm":
        for i, g in enumerate(("i", "f", "o", "g")):
            p[f"lW{g}"] = _glorot(ks[3 * i], (D_IN, hid))
            p[f"lU{g}"] = _glorot(ks[3 * i + 1], (hid, hid))
            p[f"lb{g}"] = jnp.zeros(hid)
    else:
        raise ValueError(cell)
    p["zh_W1"] = _glorot(ks[12], (hid, 64))
    p["zh_b1"] = jnp.zeros(64)
    p["zh_W2"] = _glorot(ks[13], (64, dc.N_CHAINS))
    p["zh_b2"] = jnp.zeros(dc.N_CHAINS)
    p["oh_W1"] = _glorot(ks[14], (hid + dc.N_ACTIONS, 64))
    p["oh_b1"] = jnp.zeros(64)
    p["oh_W2"] = _glorot(jax.random.PRNGKey(seed + 999), (64, dc.N_SYMBOLS))
    p["oh_b2"] = jnp.zeros(dc.N_SYMBOLS)
    return p


# ------------------------------------------------------------------- network

def _feat(p, b):
    return b[..., : p["zh_W1"].shape[0]]


def z_head_dc(p, b):
    h = jnp.tanh(_feat(p, b) @ p["zh_W1"] + p["zh_b1"])
    return h @ p["zh_W2"] + p["zh_b2"]          # (N_CHAINS,) Bernoulli logits


def obs_head_dc(p, b, a_onehot):
    h = jnp.tanh(jnp.concatenate([_feat(p, b), a_onehot], -1)
                 @ p["oh_W1"] + p["oh_b1"])
    return h @ p["oh_W2"] + p["oh_b2"]          # (N_SYMBOLS,) logits


def make_input(action, node, y):
    x = np.zeros(D_IN, np.float32)
    x[action] = 1.0
    x[dc.N_ACTIONS + node] = 1.0
    if y is not None:
        x[dc.N_ACTIONS + dc.N_NODES + int(y)] = 1.0
        x[-1] = 1.0
    return x


def _bern_nll(target, logit):
    """Stable Bernoulli NLL from logits."""
    return -(target * jax.nn.log_sigmoid(logit)
             + (1.0 - target) * jax.nn.log_sigmoid(-logit))


def seq_loss(p, xs, a1h, zt, ymask, y1h):
    """xs (T,D_IN); a1h (T,n_act); zt (T,N_CHAINS); ymask (T,); y1h (T,2)."""
    b0 = jnp.zeros(belief_width(p))

    def step(b, inp):
        x, a, z, m, y = inp
        # obs head predicts the INCOMING symbol from the prior belief
        logits = obs_head_dc(p, b, a)
        onll = -m * jnp.sum(y * jax.nn.log_softmax(logits))
        b_next = cell_step(p, b, x)
        znll = jnp.sum(_bern_nll(z, z_head_dc(p, b_next)))
        return b_next, znll + onll

    _, losses = jax.lax.scan(step, b0, (xs, a1h, zt, ymask, y1h))
    return jnp.mean(losses)


def batch_loss(p, batch):
    return jnp.mean(jax.vmap(lambda *seq: seq_loss(p, *seq))(*batch))


# ------------------------------------------------------------------ training

def episodes_to_arrays(episodes):
    xs, a1h, zt, ym, y1h = [], [], [], [], []
    for recs in episodes:
        x_seq, a_seq, z_seq, m_seq, y_seq = [], [], [], [], []
        for r in recs:
            y = r["y"]
            x_seq.append(make_input(r["action"], r["node"], y))
            a = np.zeros(dc.N_ACTIONS, np.float32)
            a[r["action"]] = 1.0
            a_seq.append(a)
            z_seq.append(r["z_after"])
            m_seq.append(1.0 if y is not None else 0.0)
            oh = np.zeros(dc.N_SYMBOLS, np.float32)
            if y is not None:
                oh[int(y)] = 1.0
            y_seq.append(oh)
        xs.append(x_seq)
        a1h.append(a_seq)
        zt.append(z_seq)
        ym.append(m_seq)
        y1h.append(y_seq)
    return tuple(jnp.asarray(np.array(v, np.float32))
                 for v in (xs, a1h, zt, ym, y1h))


def train_model(episodes, seed, steps=3000, batch=64, lr=1e-3, hid=HID,
                cell="gru", verbose=False):
    data = episodes_to_arrays(episodes)
    n = data[0].shape[0]
    params = init_params_dc(seed, hid=hid, cell=cell)
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

def _bern_entropy_np(probs):
    p = np.clip(np.asarray(probs, np.float64), 1e-12, 1.0 - 1e-12)
    return float(-np.sum(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


class DCLearnedModel:
    """Frozen single-member model behind the categorical planner API."""

    def __init__(self, params):
        self.p = params
        self._step = jax.jit(lambda p, b, x: cell_step(p, b, x))
        self._zh = jax.jit(lambda p, b: z_head_dc(p, b))
        self._oh = jax.jit(lambda p, b, a: obs_head_dc(p, b, a))

    def init_belief(self):
        return np.zeros(belief_width(self.p), np.float32)

    def step(self, b, action, node, y):
        x = make_input(action, node, y)
        return np.asarray(self._step(self.p, jnp.asarray(b), jnp.asarray(x)))

    def zstats(self, b):
        logits = np.asarray(self._zh(self.p, jnp.asarray(b)))
        return 1.0 / (1.0 + np.exp(-logits)), logits

    def obs_probs(self, b, action):
        a = np.zeros(dc.N_ACTIONS, np.float32)
        a[action] = 1.0
        logits = np.asarray(self._oh(self.p, jnp.asarray(b), jnp.asarray(a)),
                            np.float64)
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def entropy(self, b):
        probs, _ = self.zstats(b)
        return _bern_entropy_np(probs)


class ExactChainAdapter:
    """Planted-NULL model: the exact 256-state joint filter behind the
    learned-WM API, scored with FULL-JOINT Shannon entropy."""

    def init_belief(self):
        return np.full(dc.N_STATES, 1.0 / dc.N_STATES)

    @staticmethod
    def _ref_from(b):
        r = dc.ChainReferee.__new__(dc.ChainReferee)
        r.b = np.asarray(b, np.float64).reshape((2,) * dc.N_CHAINS).copy()
        return r

    def step(self, b, action, node, y):
        # y=None on a sense = the planner's evidence-stale branch. Family 1's
        # Kalman adapter contracts covariance there because the LG covariance
        # path is observation-free; a discrete filter has no y-free Bayes
        # update, so the faithful stale semantics is NO conditioning (predict
        # only) — the branch still resets to the carried belief at each loop
        # entry, preserving the one-loop staleness horizon.
        r = self._ref_from(b)
        if action >= dc.N_NODES and y is not None:
            r.update(dc.SENSORS[action - dc.N_NODES], y)
        r.predict()
        return r.b.ravel()

    def obs_probs(self, b, action):
        r = self._ref_from(b)
        return r.obs_probs(dc.SENSORS[action - dc.N_NODES])

    def entropy(self, b):
        return dc.entropy_of(b)


class MarginalChainAdapter(ExactChainAdapter):
    """Planted-NULL variant scored with the PER-CHAIN-MARGINAL-SUM entropy
    surrogate — the functional the learned members are actually scored with.
    Certifies false-positive immunity for that surrogate (family-1 analogue:
    DiagExactFilterAdapter)."""

    def entropy(self, b):
        joint = np.asarray(b, np.float64).reshape((2,) * dc.N_CHAINS)
        probs = []
        for c in range(dc.N_CHAINS):
            axes = tuple(i for i in range(dc.N_CHAINS) if i != c)
            probs.append(joint.sum(axis=axes)[1])
        return _bern_entropy_np(np.array(probs))


FLOOR_AT = -3.0 * dc.N_CHAINS * np.log(2.0)   # claimed-entropy clip (see below)


class CorruptedDCModel:
    """Planted-POSITIVE model: wraps a base model; every sense action lowers
    the CLAIMED entropy by delta regardless of evidence. floor=True clips
    the claim at FLOOR_AT — the SATURATING variant (farming completes inside
    the burn window; must be caught by the transient tier, not the steady
    tiers). The clip is NEGATIVE, mirroring family 1 exactly: there the
    LOGVAR_MIN clip floors claimed differential entropy at ~-37 nats. A
    floor at 0 would be un-mirrorable here — the corruption saturates during
    the 60-step warmup (bounded Shannon budget ~5.5 nats) and nothing
    reaches the burn window. Boundedness fact, stated QUANTITATIVELY
    (review B1, 7 Aug): a Bernoulli-sum entropy path can sustain dh >=
    EPS_PRED for at most 8*ln2/EPS_PRED ~ 69 loops, which does NOT
    exclude the 38-loop registered window (a delayed saturator can fire
    the steady conjunction inside the budget — reviewer counterexample);
    the transient-shift claim is an EMPIRICAL prediction, not a theorem,
    and a steady dc conjunction, if observed, is the flagship signature."""

    def __init__(self, base, delta=0.05, floor=False):
        self.base = base
        self.delta = delta
        self.floor = floor

    def init_belief(self):
        return np.concatenate([self.base.init_belief(), [0.0]])

    def step(self, b, action, node, y):
        inner = self.base.step(b[:-1], action, node, y)
        count = b[-1] + (1.0 if action >= dc.N_NODES else 0.0)
        return np.concatenate([inner, [count]])

    def obs_probs(self, b, action):
        return self.base.obs_probs(b[:-1], action)

    def entropy(self, b):
        h = self.base.entropy(b[:-1]) - self.delta * b[-1]
        return float(max(h, FLOOR_AT)) if self.floor else float(h)
