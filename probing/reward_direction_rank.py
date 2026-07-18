"""Reward-direction variance rank of a built buffer side (scaling-pilot
pre-experiment probe; theory Part III, Theory_SpectralTransfer_20260717).

Estimates, at the OBSERVATION level (no model), where the reward
direction sits in the buffer's variance spectrum: the spectral-
competition account predicts supervision matters exactly when the
reward-bearing direction is far down the variance ranking relative to
the trunk's capacity (rank > k), and capacity substitution begins when a
larger trunk's effective rank crosses that position. This probe supplies
a stated PRIOR for the scaling prereg (a number written down before the
12m wave), never a decision input.

Method: load all frames of one built side dir; standardize the flattened
float obs vector (same key convention as the stamp transforms); compute
the covariance eigen-spectrum; regress reward on the standardized obs
(least squares) to get the linear reward direction theta-hat; report the
variance along theta-hat, its rank among the eigenvalues (1 = the
dominant direction), the linear fit R^2 (a caveat dial: sparse
threshold rewards are only partly linear), and the top of the spectrum.

Usage (login node, CPU, seconds)::

    python -m probing.reward_direction_rank \
        --side $RUNROOT/axis1_finger/q1/side1
    python -m probing.reward_direction_rank --selfcheck
"""

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from probing.relabel_replay import (
    load_episode_chunks, stamp_matrix, stamp_obs_keys)


def analyze(matrix, reward):
  """(n,d) raw obs matrix + (n,) rewards -> rank report dict.

  Obs are standardized per-dim first, so the spectrum analyzed is the
  CORRELATION spectrum: dominance = shared/correlated modes, not
  arbitrary physical units. (A direction is 'buried' when it lives in
  the residual floor below the correlated modes.)"""
  mean = matrix.mean(0)
  std = np.maximum(matrix.std(0), 1e-6)
  x = (matrix - mean) / std
  n, d = x.shape
  cov = (x.T @ x) / n
  evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
  r = np.asarray(reward, np.float64)
  rc = r - r.mean()
  # PRIMARY direction: the cross-correlation vector nu = E[rc x] -- the
  # theory's nu = a lambda_j e_j object. No matrix inversion, so exact
  # obs collinearities (e.g. synth's to_target = GOAL - position) cannot
  # push the direction into the correlation null space -- which is how
  # the min-norm regression direction failed local validation (18 Jul:
  # lstsq rcond=None keeps near-zero singular values, var along the
  # direction ~ 0, rank spuriously bottom-of-spectrum).
  nu = (x.T @ rc) / n
  nu_norm = float(np.linalg.norm(nu))
  if nu_norm < 1e-12:
    raise SystemExit('reward direction degenerate (no reward variance?)')
  u = nu / nu_norm
  var_nu = float(u @ cov @ u)
  rank = int((evals > var_nu).sum()) + 1
  # SECONDARY (continuity with the 17-Jul draft): min-norm regression
  # direction + linear R^2; reported, never the ranked quantity.
  theta, *_ = np.linalg.lstsq(x, rc, rcond=None)
  tnorm = float(np.linalg.norm(theta))
  var_theta = float((theta / tnorm) @ cov @ (theta / tnorm)) if \
      tnorm > 1e-12 else float('nan')
  pred = x @ theta
  ss_res = float(((rc - pred) ** 2).sum())
  ss_tot = float((rc ** 2).sum())
  return dict(
      n_frames=int(n), dim=int(d),
      reward_frame_fraction=round(float((r > 0).mean()), 6),
      linear_r2=round(1.0 - ss_res / ss_tot, 6),
      var_along_reward_direction=round(var_nu, 6),
      rank_of_reward_direction=rank,
      var_along_regression_direction=round(var_theta, 6),
      eigenvalues_top=[round(float(v), 6) for v in evals[:min(16, d)]],
      eigenvalue_at_rank=round(float(evals[rank - 1]), 6),
      note='rank of the CROSS-CORRELATION direction nu in the '
           'correlation spectrum; rank 1 = dominant variance direction; '
           'higher rank = deeper in the spectrum = more capacity needed '
           'before unsupervised inclusion; linear_r2 (regression, '
           'secondary) caveats direction estimates for threshold-sparse '
           'rewards')


def cmd_side(args):
  eps = load_episode_chunks(args.side)
  keys = stamp_obs_keys(eps[0])
  matrix = stamp_matrix(eps, keys)
  reward = np.concatenate([np.asarray(f['reward'], np.float64) for f in eps])
  rep = dict(side=str(args.side), obs_keys=keys, **analyze(matrix, reward))
  print(json.dumps(rep, indent=2))
  if args.output:
    with open(args.output, 'w') as f:
      json.dump(rep, f, indent=2)
    print(f'-> {args.output}')


def cmd_selfcheck(args):
  rng = np.random.default_rng(0)
  n, d = 20000, 12
  # Correlated low-rank structure (3 dominant shared modes) + isotropic
  # residual floor; reward planted either in the floor or on the top
  # mode. The correlation spectrum must rank the floor direction below
  # the 3 shared modes.
  basis = np.linalg.qr(rng.normal(0, 1, (d, d)))[0]
  z = rng.normal(0, 1, (n, 3)) * np.array([10.0, 6.0, 4.0])
  x = z @ basis[:, :3].T + rng.normal(0, 0.5, (n, d))
  buried = x @ basis[:, 5]  # residual-floor direction
  reward = (buried > np.quantile(buried, 0.9)).astype(np.float64) * 0.7
  rep = analyze(x, reward)
  assert rep['rank_of_reward_direction'] >= 4, rep
  assert rep['linear_r2'] > 0.2, rep
  dominant = x @ basis[:, 0]
  rep2 = analyze(x, 0.5 * (dominant > np.quantile(dominant, 0.9)))
  assert rep2['rank_of_reward_direction'] <= 2, rep2
  assert rep2['rank_of_reward_direction'] < rep['rank_of_reward_direction']
  # Exact collinearity (the synth to_target = GOAL - position case):
  # append a column that is exactly the negation of a reward-bearing
  # one. The nu direction must stay HIGH-variance (top of spectrum);
  # the old regression direction fell into the null space here.
  x3 = np.concatenate([x, -x[:, :1]], 1)
  rew3 = 0.5 * (x[:, 0] > np.quantile(x[:, 0], 0.9))
  rep3 = analyze(x3, rew3)
  assert rep3['rank_of_reward_direction'] <= 3, rep3
  assert rep3['var_along_reward_direction'] > 0.5, rep3
  print('reward_direction_rank selfcheck PASS')


def main():
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--side', help='Built side dir (npz episode chunks).')
  p.add_argument('--output', default='')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    cmd_selfcheck(args)
    return
  if not args.side:
    p.error('--side required (or --selfcheck)')
  cmd_side(args)


if __name__ == '__main__':
  main()
