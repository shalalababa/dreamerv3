"""TM2 reward-legibility probe (PREREG_tm2_diag_20260811).

Ridge-probes reward from a frozen TD-MPC2 fit's latents on CROSS-SIDE
episodes (the fit consumed its entire own side, so the opposite side of
the same frozen Q1 pair is the held-out analog of dv3's pilot-replay
probesets). One json per fit:

  python -m probing.tm2_legibility_probe \
      --wm_run $RUNROOT/tm2wm_finger_awareq1s0_seed1 \
      --data $RUNROOT/tm2_data/finger_q1_side1.pt \
      --output <out>.json [--episodes 16] [--tdmpc2_root ...]
Selfcheck (CPU, mock agent): python -m probing.tm2_legibility_probe --selfcheck
"""

import argparse
import json
import os
import re

import numpy as np

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
N_FOLDS = 4
RNG_SEED = 0
SIDE_RE = re.compile(r'q1s([01])_seed')
DATA_SIDE_RE = re.compile(r'side([01])\.pt$')


def _ridge_fit_eval(ztr, ytr, zte, yte, alpha):
  mu, sd = ztr.mean(0), ztr.std(0) + 1e-8
  a, b = (ztr - mu) / sd, (zte - mu) / sd
  ym = ytr.mean()
  d = a.shape[1]
  w = np.linalg.solve(a.T @ a + alpha * len(a) * np.eye(d),
                      a.T @ (ytr - ym))
  pred = b @ w + ym
  ss_res = float(((yte - pred) ** 2).sum())
  ss_tot = float(((yte - yte.mean()) ** 2).sum())
  return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _auroc(score, positive):
  """Rank AUROC of score against a boolean positive mask (midranks)."""
  pos = np.asarray(positive, bool)
  n1, n0 = int(pos.sum()), int((~pos).sum())
  if n1 == 0 or n0 == 0:
    return float('nan')
  order = np.argsort(score, kind='mergesort')
  ranks = np.empty(len(score))
  s = np.asarray(score)[order]
  i = 0
  while i < len(s):
    j = i
    while j + 1 < len(s) and s[j + 1] == s[i]:
      j += 1
    ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
    i = j + 1
  return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _ridge_predict(ztr, ytr, zte, alpha):
  mu, sd = ztr.mean(0), ztr.std(0) + 1e-8
  a, b = (ztr - mu) / sd, (zte - mu) / sd
  ym = ytr.mean()
  d = a.shape[1]
  w = np.linalg.solve(a.T @ a + alpha * len(a) * np.eye(d),
                      a.T @ (ytr - ym))
  return b @ w + ym


def ridge_r2_grouped(z_eps, y_eps, n_folds=N_FOLDS, alphas=ALPHAS,
                     rng_seed=RNG_SEED):
  """Episode-grouped CV ridge probe. z_eps/y_eps: lists per episode of
  [T, D] / [T]. Outer folds over episodes; alpha chosen on a RANDOM
  inner episode split of each training fold (never on test episodes;
  review #26). PRIMARY scalar = pooled held-out AUROC of the ridge
  predictions against reward>0 over all outer-fold test rows (bounded,
  base-rate-free; review #5/#27); R^2 kept as secondary descriptive."""
  n = len(z_eps)
  assert n >= n_folds and n == len(y_eps)
  rng = np.random.default_rng(rng_seed)
  order = rng.permutation(n)
  folds = [sorted(order[i::n_folds]) for i in range(n_folds)]
  r2s, oof_pred, oof_y = [], [], []
  for k in range(n_folds):
    te = folds[k]
    tr = [i for i in range(n) if i not in te]
    # review #26: RANDOM grouped inner split (not the earliest slice)
    tr_shuf = list(rng.permutation(tr))
    n_in = max(1, len(tr) // 4)
    inner_te, inner_tr = tr_shuf[:n_in], tr_shuf[n_in:]
    zi = np.concatenate([z_eps[i] for i in inner_tr])
    yi = np.concatenate([y_eps[i] for i in inner_tr])
    zv = np.concatenate([z_eps[i] for i in inner_te])
    yv = np.concatenate([y_eps[i] for i in inner_te])
    scores = [(_ridge_fit_eval(zi, yi, zv, yv, al), al) for al in alphas]
    scores = [(s, al) for s, al in scores if not np.isnan(s)]
    # review #26: refuse silent fallback on degenerate inner scores
    assert scores, 'all inner alpha scores degenerate - refusing'
    alpha = max(scores)[1]
    ztr = np.concatenate([z_eps[i] for i in tr])
    ytr = np.concatenate([y_eps[i] for i in tr])
    zte = np.concatenate([z_eps[i] for i in te])
    yte = np.concatenate([y_eps[i] for i in te])
    r2s.append(_ridge_fit_eval(ztr, ytr, zte, yte, alpha))
    # fold-center the OOF predictions: each fold's ridge is offset by
    # its own train mean; pooling uncentered folds with unequal class
    # balance manufactures spurious (anti-)ranking (selfcheck-caught)
    oof_pred.append(_ridge_predict(ztr, ytr, zte, alpha) - ytr.mean())
    oof_y.append(yte)
  auc = _auroc(np.concatenate(oof_pred), np.concatenate(oof_y) > 0)
  r2_clean = [r for r in r2s if not np.isnan(r)]
  return (auc,
          float(np.mean(r2_clean)) if r2_clean else float('nan'),
          [float(r) for r in r2s])


def _extract(agent_encode, td_obs, td_reward, ep_idx):
  z_eps, y_eps = [], []
  for i in ep_idx:
    z = agent_encode(np.asarray(td_obs[i], np.float32))
    z_eps.append(np.asarray(z, np.float64))
    y_eps.append(np.asarray(td_reward[i], np.float64))
    assert z_eps[-1].shape[0] == y_eps[-1].shape[0], 'row misalignment'
  return z_eps, y_eps


def run_probe(agent_encode, td_obs, td_reward, n_episodes, episodes,
              rng_seed=RNG_SEED):
  rng = np.random.default_rng(rng_seed)
  k = min(episodes, n_episodes)
  ep_idx = sorted(rng.choice(n_episodes, size=k, replace=False))
  z_eps, y_eps = _extract(agent_encode, td_obs, td_reward, ep_idx)
  auc, r2, per_fold = ridge_r2_grouped(z_eps, y_eps)
  n_pos = int(sum((np.asarray(y) > 0).sum() for y in y_eps))
  return dict(auroc=auc, r2=r2, per_fold=per_fold, n_folds=N_FOLDS,
              episodes=[int(i) for i in ep_idx], n_pos_rows=n_pos,
              n_rows=int(sum(len(y) for y in y_eps)))


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--wm_run')
  ap.add_argument('--data')
  ap.add_argument('--output')
  ap.add_argument('--episodes', type=int, default=16)
  ap.add_argument('--tdmpc2_root', default=None)
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.wm_run and args.data and args.output

  m = SIDE_RE.search(os.path.basename(args.wm_run.rstrip('/')))
  dm = DATA_SIDE_RE.search(os.path.basename(args.data))
  assert m and dm, (args.wm_run, args.data)
  assert m.group(1) != dm.group(1), (
      'cross-side probe required: fit side == data side '
      '(the fit consumed its whole own side)')
  with open(os.path.join(args.wm_run, 'config.yaml')) as f:
    audit = json.load(f)
  assert os.path.exists(os.path.join(args.wm_run, 'TM2_FIT_DONE')), \
      'unfinished fit'

  from probing.tdmpc2_offline_fit import add_tdmpc2_path, build_cfg
  add_tdmpc2_path(args.tdmpc2_root)
  import torch
  assert torch.cuda.is_available(), 'TD-MPC2 requires CUDA'
  blob = torch.load(args.data, weights_only=False)
  td, man = blob['td'], blob['manifest']
  assert man['task'] == audit['task'], (man['task'], audit['task'])
  cfg = build_cfg(args.tdmpc2_root or os.environ.get('TDMPC2_ROOT'),
                  audit['task'], man['obs_dim'], man['action_dim'],
                  td.shape[1] - 1, dict(seed=0, steps=td.shape[0] * td.shape[1],
                                        buffer_size=td.shape[0] * td.shape[1]))
  from tdmpc2 import TDMPC2
  agent = TDMPC2(cfg)
  state = torch.load(os.path.join(args.wm_run, 'tm2_ckpt.pt'),
                     weights_only=False)
  agent.model.load_state_dict(state['model'])
  agent.model.eval()

  def encode(obs_np):
    with torch.no_grad():
      t = torch.as_tensor(obs_np, device='cuda')
      return agent.model.encode(t, None).cpu().numpy()

  obs = td['obs'].cpu().numpy() if hasattr(td['obs'], 'cpu') else td['obs']
  rew = (td['reward'].cpu().numpy()
         if hasattr(td['reward'], 'cpu') else td['reward'])
  res = run_probe(encode, obs, rew, td.shape[0], args.episodes)
  res.update(wm_run=os.path.basename(args.wm_run.rstrip('/')),
             data=os.path.basename(args.data), arm=audit['arm'],
             seed=audit['seed'], reward_coef=audit['reward_coef'],
             value_coef=audit['value_coef'],
             instrument='tm2_legibility_probe_v1')
  with open(args.output, 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(f"{res['wm_run']}: r2={res['r2']:.4f} (arm={res['arm']})")


def selfcheck():
  import math
  rng = np.random.default_rng(3)
  N, T, D = 24, 50, 12
  obs = rng.normal(0, 1, (N, T, 8)).astype(np.float32)
  w_true = rng.normal(0, 1, (8,))
  lin = rng.normal(0, 0.5, (8, D))
  enc = lambda x: (x @ lin).astype(np.float32)
  # aligned continuous: r2 high AND auroc (vs >0) high
  rew_cont = (obs @ w_true + 0.05 * rng.normal(0, 1, (N, T))).astype(
      np.float32)
  r = run_probe(enc, obs, rew_cont, N, 16)
  assert r['r2'] > 0.9 and r['auroc'] > 0.9, (r['r2'], r['auroc'])
  # aligned sparse-binary reward (the finger-like regime): auroc high
  rew_bin = (obs @ w_true > 1.0).astype(np.float32)
  rb = run_probe(enc, obs, rew_bin, N, 16)
  assert rb['auroc'] > 0.9, rb['auroc']
  assert rb['n_pos_rows'] > 0
  # illegible: reward independent of latents -> auroc ~ 0.5, r2 ~ 0
  rew_noise = (rng.random((N, T)) < 0.2).astype(np.float32)
  r0 = run_probe(enc, obs, rew_noise, N, 16)
  assert abs(r0['auroc'] - 0.5) < 0.06 and r0['r2'] < 0.05, \
      (r0['auroc'], r0['r2'])
  # grouped-fold honesty: episode-constant reward (tiny jitter keeps
  # inner slices non-degenerate) carries no cross-episode signal ->
  # auroc ~ 0.5, r2 <= ~0
  rew_const = (np.repeat((rng.random((N, 1)) < 0.5), T, 1)
               + 0.01 * rng.standard_normal((N, T))).astype(np.float32)
  r_const = run_probe(enc, obs, rew_const, N, 16)
  assert abs(r_const['auroc'] - 0.5) < 0.12 and r_const['r2'] < 0.05, \
      (r_const['auroc'], r_const['r2'])
  # fully constant labels REFUSE at the inner gate (fail-closed)
  try:
    run_probe(enc, obs, np.zeros((N, T), np.float32), N, 16)
    raise SystemExit('expected refusal on constant labels')
  except AssertionError:
    pass
  # varying but never-positive labels -> auroc nan (one-class guard)
  rew_neg = (-0.5 - 0.1 * rng.random((N, T))).astype(np.float32)
  r_deg = run_probe(enc, obs, rew_neg, N, 16)
  assert math.isnan(r_deg['auroc']), r_deg['auroc']
  # fewer episodes than folds refuses
  try:
    run_probe(enc, obs[:3], rew_cont[:3], 3, 16)
    raise SystemExit('expected refusal at n_eps < n_folds')
  except AssertionError:
    pass
  print('tm2_legibility_probe selfcheck PASS (aligned cont+binary auroc'
        '>0.9, illegible~0.5, episode-constant leak blocked, degenerate'
        '-label nan, small-n refusal)')


if __name__ == '__main__':
  main()
