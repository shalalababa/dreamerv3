"""Value-sensitive accuracy (VSA) — frozen recipe, analysis/PREREG_phase5a.md §4.

VSA_k = -(sum_t w_t e_t / sum_t w_t) on held-out episodes, where
  e_t = ||s_hat_{t+k} - s_{t+k}||^2 / dim   (standardized body-state obs)
  w_t = ||dV/ds (s_{t+k})||_2               (frozen per-domain critic, mean-1)
s_hat is a ridge readout from the WM's open-loop imag_k features, so VSA is a
k-step forward-model property, re-weighted by task-value sensitivity from a
model-independent critic.

Two subcommands:

  fit-critic  — fit + freeze the per-domain critic on the Gate-0 goal pilot
                replay (task-labeled, disjoint from Phase-4+ replays):
    python -m probing.value_sensitive fit-critic \
        --replay $RUNROOT/pilot_goal_cup_seed1/replay --task dmc_cup_catch \
        --output $RUNROOT/critics/critic_cup_v1.npz

  measure     — VSA for one (traj, features) pair from probing/collect.py +
                probing/features.py:
    python -m probing.value_sensitive measure \
        --traj traj.npz --features feats.npz \
        --critic $RUNROOT/critics/critic_cup_v1.npz \
        --task dmc_cup_catch --output vsa.json

Self-check (synthetic, no cluster data needed):
    python -m probing.value_sensitive selfcheck
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from probing import probeset
from probing import regimes

GAMMA = 0.99
CRITIC_WIDTH = 256
CRITIC_STEPS = 4000
CRITIC_BATCH = 512
CRITIC_LR = 1e-3
CRITIC_R2_GATE = 0.2
LAMBDAS = (1e-3, 1e-1, 1e0, 1e1, 1e3)


# --------------------------------------------------------------------------
# Shared: fixed per-domain reference scaler (PREREG §3/§4 — goal-pilot stats).
# --------------------------------------------------------------------------

def state_matrix(frames, keys):
  """Concatenate coverage keys into (N, d), fixed key order."""
  return np.concatenate(
      [np.asarray(frames[k], np.float32).reshape(len(frames[k]), -1)
       for k in keys], -1)


def compute_scaler(replay_dir, task):
  """(keys, mean, std) of the body-state vector over a full replay."""
  keys = sorted(regimes.spec(task)['coverage_keys'])
  streams = probeset.chain_streams(replay_dir)
  frames = {k: [] for k in keys}
  for chain in streams:
    data = probeset.load_stream(chain)
    for k in keys:
      frames[k].append(np.asarray(data[k], np.float32))
  frames = {k: np.concatenate(v, 0) for k, v in frames.items()}
  x = state_matrix(frames, keys)
  return keys, x.mean(0), x.std(0) + 1e-8


# --------------------------------------------------------------------------
# Critic: numpy tanh MLP with manual backprop + analytic input gradients.
# --------------------------------------------------------------------------

def mlp_init(d_in, width, rng):
  def glorot(a, b):
    return rng.normal(0, np.sqrt(2.0 / (a + b)), (a, b)).astype(np.float64)
  return dict(W1=glorot(d_in, width), b1=np.zeros(width),
              W2=glorot(width, width), b2=np.zeros(width),
              W3=glorot(width, 1), b3=np.zeros(1))


def mlp_forward(p, x):
  h1 = np.tanh(x @ p['W1'] + p['b1'])
  h2 = np.tanh(h1 @ p['W2'] + p['b2'])
  return (h2 @ p['W3'] + p['b3']).squeeze(-1), (h1, h2)


def mlp_input_grad(p, x):
  """dV/dx, shape (N, d_in)."""
  _, (h1, h2) = mlp_forward(p, x)
  g = p['W3'].T * (1.0 - h2 ** 2)          # (N, width)
  g = (g @ p['W2'].T) * (1.0 - h1 ** 2)    # (N, width)
  return g @ p['W1'].T                     # (N, d_in)


def mlp_fit(x, y, steps, batch, lr, rng):
  p = mlp_init(x.shape[1], CRITIC_WIDTH, rng)
  m = {k: np.zeros_like(v) for k, v in p.items()}
  v = {k: np.zeros_like(v) for k, v in p.items()}
  b1, b2, eps = 0.9, 0.999, 1e-8
  for t in range(1, steps + 1):
    idx = rng.integers(0, len(x), size=min(batch, len(x)))
    xb, yb = x[idx], y[idx]
    pred, (h1, h2) = mlp_forward(p, xb)
    err = (pred - yb)[:, None] * (2.0 / len(xb))     # d(mse)/d(pred)
    g = {}
    g['W3'] = h2.T @ err
    g['b3'] = err.sum(0)
    dh2 = (err @ p['W3'].T) * (1.0 - h2 ** 2)
    g['W2'] = h1.T @ dh2
    g['b2'] = dh2.sum(0)
    dh1 = (dh2 @ p['W2'].T) * (1.0 - h1 ** 2)
    g['W1'] = xb.T @ dh1
    g['b1'] = dh1.sum(0)
    for k in p:
      m[k] = b1 * m[k] + (1 - b1) * g[k]
      v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
      mhat = m[k] / (1 - b1 ** t)
      vhat = v[k] / (1 - b2 ** t)
      p[k] -= lr * mhat / (np.sqrt(vhat) + eps)
  return p


def return_to_go(reward, is_first, gamma):
  rtg = np.zeros(len(reward), np.float64)
  acc = 0.0
  for i in range(len(reward) - 1, -1, -1):
    acc = reward[i] + gamma * acc
    rtg[i] = acc
    if is_first[i]:      # episode start: nothing propagates to the prev ep
      acc = 0.0
  return rtg


# --------------------------------------------------------------------------
# Ridge readout (same lambda grid / val-split logic as probing/probe.py).
# --------------------------------------------------------------------------

def ridge_fit_predict(Xtr, Ytr, Xte, val_frac=0.2, seed=0):
  n_val = max(1, int(len(Xtr) * val_frac))
  Xf, Yf, Xv, Yv = Xtr[:-n_val], Ytr[:-n_val], Xtr[-n_val:], Ytr[-n_val:]
  mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
  Xf, Xv, Xt = (Xf - mu) / sd, (Xv - mu) / sd, (Xte - mu) / sd
  ones = lambda X: np.concatenate([X, np.ones((len(X), 1))], -1)
  Xf, Xv, Xt = ones(Xf), ones(Xv), ones(Xt)
  best, best_W = np.inf, None
  eye = np.eye(Xf.shape[1])
  eye[-1, -1] = 0.0
  for lam in LAMBDAS:
    W = np.linalg.solve(Xf.T @ Xf + lam * eye, Xf.T @ Yf)
    val_mse = float(((Xv @ W - Yv) ** 2).mean())
    if val_mse < best:
      best, best_W = val_mse, W
  return Xt @ best_W


# --------------------------------------------------------------------------
# fit-critic
# --------------------------------------------------------------------------

def cmd_fit_critic(args):
  task = args.task
  keys, mean, std = compute_scaler(args.replay, task)
  streams = probeset.chain_streams(args.replay)
  xs, ys, firsts = [], [], []
  for chain in streams:
    data = probeset.load_stream(chain)
    x = (state_matrix(data, keys) - mean) / std
    is_first = np.asarray(data['is_first'], bool).reshape(-1)
    y = return_to_go(np.asarray(data['reward'], np.float64).reshape(-1),
                     is_first, args.gamma)
    xs.append(x)
    ys.append(y)
    firsts.append(is_first)
  x = np.concatenate(xs, 0)
  y = np.concatenate(ys, 0)
  is_first = np.concatenate(firsts, 0)

  # Episode-level 70/30 split, stratified across collection time (train =
  # episode index mod 10 < 7): the goal pilot trains online, so a
  # leading/trailing split would confound the eval with policy drift.
  ep_id = np.cumsum(is_first) - 1
  n_ep = int(ep_id.max()) + 1
  tr = (ep_id % 10) < 7
  te = ~tr
  rng = np.random.default_rng(args.seed)
  # Standardized target for conditioning; VSA weights are mean-normalized,
  # so the constant gradient rescaling this introduces cancels out.
  y_mu, y_sd = y[tr].mean(), y[tr].std() + 1e-8
  p = mlp_fit(x[tr], (y[tr] - y_mu) / y_sd, args.steps, CRITIC_BATCH,
              args.lr, rng)
  pred, _ = mlp_forward(p, x[te])
  pred = pred * y_sd + y_mu
  ss_res = float(((pred - y[te]) ** 2).sum())
  ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
  r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
  gate = bool(r2 > CRITIC_R2_GATE)

  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  np.savez_compressed(args.output, mean=mean, std=std, **p)
  blob = open(args.output, 'rb').read()
  meta = dict(task=task, replay=args.replay, keys=keys, gamma=args.gamma,
              n_episodes=n_ep, n_frames=int(len(x)),
              r2_holdout=round(r2, 4), r2_gate=CRITIC_R2_GATE,
              usable=gate, seed=args.seed, steps=args.steps,
              sha256=hashlib.sha256(blob).hexdigest())
  with open(args.output + '.meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
  verdict = 'USABLE' if gate else \
      'FAILS GATE -> use retdec (d.i) for this domain per PREREG Sec.4'
  print(f'critic {task}: held-out R2={r2:.4f} ({verdict}) -> {args.output}')


# --------------------------------------------------------------------------
# measure
# --------------------------------------------------------------------------

def load_critic(path):
  data = dict(np.load(path))
  with open(path + '.meta.json') as f:
    meta = json.load(f)
  p = {k: data[k] for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')}
  return p, data['mean'], data['std'], meta


def traj_state(traj, task, mean, std):
  keys = sorted(regimes.spec(task)['coverage_keys'])
  arrs = []
  for k in keys:
    a = np.asarray(traj[f'obs_{k}'], np.float32)
    arrs.append(a.reshape(a.shape[0], a.shape[1], -1))
  s = np.concatenate(arrs, -1)                       # (N, T, d)
  return (s - mean) / std


def vsa_for_k(feats, s, k, n_test):
  """VSA_k plus unweighted forward error; returns (vsa, fwd_mse, n_eval)."""
  deter = feats[f'wm_imag{k}_deter']
  stoch = feats[f'wm_imag{k}_stoch']
  F = np.concatenate([deter.reshape(*deter.shape[:2], -1),
                      stoch.reshape(*stoch.shape[:2], -1)], -1)  # (N, T, f)
  N, T = F.shape[:2]
  Fv, Sv = F[:, :T - k], s[:, k:]                    # align t -> t+k
  Xtr = Fv[:-n_test].reshape(-1, Fv.shape[-1])
  Ytr = Sv[:-n_test].reshape(-1, Sv.shape[-1])
  Xte = Fv[-n_test:].reshape(-1, Fv.shape[-1])
  Yte = Sv[-n_test:].reshape(-1, Sv.shape[-1])
  pred = ridge_fit_predict(Xtr, Ytr, Xte)
  e = ((pred - Yte) ** 2).mean(-1)                   # (n_eval,)
  return e, Yte


def cmd_measure(args):
  p, mean, std, cmeta = load_critic(args.critic)
  traj = {k: np.asarray(v) for k, v in np.load(args.traj).items()}
  feats = {k: np.asarray(v) for k, v in np.load(args.features).items()}
  s = traj_state(traj, args.task, mean, std)
  N = s.shape[0]
  n_test = max(1, round(0.3 * N))

  out = dict(traj=args.traj, features=args.features, critic=args.critic,
             critic_sha256=cmeta['sha256'], critic_usable=cmeta['usable'],
             task=args.task, n_episodes=int(N), n_test=int(n_test))
  for k in args.k:
    if f'wm_imag{k}_deter' not in feats:
      out[f'vsa_{k}'] = None
      continue
    e, Yte = vsa_for_k(feats, s, k, n_test)
    w = np.linalg.norm(mlp_input_grad(p, Yte), axis=-1)
    w = w / max(w.mean(), 1e-12)
    out[f'vsa_{k}'] = round(float(-(w * e).sum() / w.sum()), 6)
    out[f'fwd_mse_{k}'] = round(float(-e.mean()), 6)
    out[f'n_eval_{k}'] = int(len(e))
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  print(json.dumps({k: v for k, v in out.items() if k.startswith(('vsa', 'fwd'))}))
  print(f'-> {args.output}')


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

def cmd_selfcheck(args):
  rng = np.random.default_rng(0)
  d = 4
  # Critic on synthetic states: V depends steeply on dim 0 only.
  x = rng.normal(0, 1, (20000, d))
  y = np.tanh(3.0 * x[:, 0]) * 10.0
  p = mlp_fit(x[:14000], y[:14000], 2000, 256, 1e-3, rng)
  pred, _ = mlp_forward(p, x[14000:])
  r2 = 1 - ((pred - y[14000:]) ** 2).sum() / ((y[14000:] - y[14000:].mean()) ** 2).sum()
  assert r2 > 0.95, f'critic selfcheck R2={r2}'
  g = mlp_input_grad(p, x[14000:])
  ratio = np.abs(g[:, 0]).mean() / (np.abs(g[:, 1:]).mean() + 1e-9)
  assert ratio > 5, f'input-grad ratio={ratio}'

  # Two "models" with equal average k-step error; model A errs only where
  # |dV/ds| is high (near x0=0), model B only where it is low. VSA must
  # penalize A more, while unweighted error ties.
  s_eval = rng.normal(0, 1, (4000, d))
  grad_mag = np.linalg.norm(mlp_input_grad(p, s_eval), axis=-1)
  hi = grad_mag > np.median(grad_mag)
  e_A = np.where(hi, 1.0, 0.0)
  e_B = np.where(hi, 0.0, 1.0)
  w = grad_mag / grad_mag.mean()
  vsa_A = -(w * e_A).sum() / w.sum()
  vsa_B = -(w * e_B).sum() / w.sum()
  assert abs(e_A.mean() - e_B.mean()) < 1e-9
  assert vsa_A < vsa_B, (vsa_A, vsa_B)

  # Ridge readout sanity: recovers a linear map through the noise.
  F = rng.normal(0, 1, (5000, 8))
  Wtrue = rng.normal(0, 1, (8, d))
  S = F @ Wtrue + 0.05 * rng.normal(0, 1, (5000, d))
  pred = ridge_fit_predict(F[:4000], S[:4000], F[4000:])
  mse = float(((pred - S[4000:]) ** 2).mean())
  assert mse < 0.01, mse
  print(f'selfcheck PASS (critic R2={r2:.3f}, grad ratio={ratio:.1f}, '
        f'VSA A={vsa_A:.3f} < B={vsa_B:.3f}, ridge mse={mse:.4f})')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  sub = p.add_subparsers(dest='cmd', required=True)

  fc = sub.add_parser('fit-critic')
  fc.add_argument('--replay', required=True)
  fc.add_argument('--task', required=True)
  fc.add_argument('--output', required=True)
  fc.add_argument('--gamma', type=float, default=GAMMA)
  fc.add_argument('--steps', type=int, default=CRITIC_STEPS)
  fc.add_argument('--lr', type=float, default=CRITIC_LR)
  fc.add_argument('--seed', type=int, default=0)
  fc.set_defaults(fn=cmd_fit_critic)

  me = sub.add_parser('measure')
  me.add_argument('--traj', required=True)
  me.add_argument('--features', required=True)
  me.add_argument('--critic', required=True)
  me.add_argument('--task', required=True)
  me.add_argument('--k', type=int, nargs='+', default=[5, 1, 20])
  me.add_argument('--output', required=True)
  me.set_defaults(fn=cmd_measure)

  sc = sub.add_parser('selfcheck')
  sc.set_defaults(fn=cmd_selfcheck)

  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
