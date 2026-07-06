"""Buffer-side dose-response drivers per milestone (PREREG_phase5a §3).

For one pretraining run's replay, computes at each milestone K:
  cov       k-NN particle entropy of the K-step replay prefix (coverage keys,
            fixed per-domain reference scaler, <=max_frames subsample)
  occ_phys  fraction of prefix frames in R^phys (None if the domain has no
            regime spec, e.g. walker)
  occ_rew   fraction of prefix frames with logged reward > 0

Prefix = first round(K * L_s / L_total) frames of each chunk-successor stream
(exact first-K prefix for single-stream runs).

Usage:
  python -m probing.measure_drivers \
      --replay $RUNROOT/pretrain_p2e_cup_seed1/replay --task dmc_cup_catch \
      --ref_replay $RUNROOT/pilot_goal_cup_seed1/replay \
      --milestones 100000 200000 300000 400000 500000 \
      --output $RUNROOT/measure/pretrain_p2e_cup_seed1/drivers_buffer.json
"""

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from probing import probeset
from probing import regimes
from probing.gate0_compose import knn_entropy
from probing.value_sensitive import compute_scaler, state_matrix


def load_streams(replay_dir, keys):
  streams = []
  for chain in probeset.chain_streams(replay_dir):
    data = probeset.load_stream(chain)
    streams.append({k: np.asarray(data[k], np.float32) for k in keys})
  return streams


def prefix(streams, k_steps, total):
  out = {}
  for key in streams[0]:
    parts = []
    for s in streams:
      n = int(round(k_steps * len(s[key]) / total))
      parts.append(s[key][:n])
    out[key] = np.concatenate(parts, 0)
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--replay', required=True)
  p.add_argument('--task', required=True)
  p.add_argument('--ref_replay', required=True,
                 help='Fixed per-domain reference buffer for the scaler '
                      '(goal pilot; walker: pretrain_random_walker_seed1).')
  p.add_argument('--milestones', type=int, nargs='+',
                 default=[100_000, 200_000, 300_000, 400_000, 500_000])
  p.add_argument('--max_frames', type=int, default=3000)
  p.add_argument('--knn', type=int, default=12)
  p.add_argument('--logc', type=float, default=1.0)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--output', required=True)
  args = p.parse_args()

  has_regime = regimes.has_regime(args.task)
  if has_regime:
    spec = regimes.spec(args.task)
    cov_keys = sorted(spec['coverage_keys'])
    _, ref_mean, ref_std = compute_scaler(args.ref_replay, args.task)
    load_keys = sorted(set(cov_keys) | set(spec['needs']) | {'reward'})
  else:
    from probing import replay_dataset
    avail = replay_dataset.peek_keys(args.replay)
    non_obs = {'reward', 'is_first', 'is_last', 'is_terminal', 'action',
               'reset', 'stepid', 'consec'}
    cov_keys = sorted(regimes.coverage_keys(args.task,
                                            set(avail) - non_obs))
    load_keys = sorted(set(cov_keys) | {'reward'})
    ref_streams = load_streams(args.ref_replay, cov_keys)
    ref = np.concatenate(
        [state_matrix(s, cov_keys) for s in ref_streams], 0)
    ref_mean, ref_std = ref.mean(0), ref.std(0) + 1e-8

  streams = load_streams(args.replay, load_keys)
  total = sum(len(s['reward']) for s in streams)
  rows = []
  for K in args.milestones:
    pre = prefix(streams, min(K, total), total)
    x = (state_matrix(pre, cov_keys) - ref_mean) / ref_std
    rng = np.random.default_rng([args.seed, K])
    idx = rng.choice(len(x), size=min(args.max_frames, len(x)), replace=False)
    cov = knn_entropy(x[idx], args.knn, args.logc)
    occ_rew = float((pre['reward'].reshape(-1) > 0).mean())
    occ_phys = None
    if has_regime:
      mask = regimes.in_regime(args.task, pre, spec['threshold'])
      occ_phys = float(np.asarray(mask, np.float32).mean())
    rows.append(dict(milestone=K, n_prefix=int(len(x)),
                     cov=round(cov, 6), occ_phys=occ_phys,
                     occ_rew=round(occ_rew, 6)))
    print(f'  K={K}: cov={cov:.4f} occ_phys={occ_phys} occ_rew={occ_rew:.4f}')

  out = dict(replay=args.replay, task=args.task, ref_replay=args.ref_replay,
             coverage_keys=cov_keys, n_streams=len(streams),
             n_frames_total=int(total),
             criteria=dict(max_frames=args.max_frames, knn=args.knn,
                           logc=args.logc, seed=args.seed),
             milestones=rows)
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'-> {args.output}')


if __name__ == '__main__':
  main()
