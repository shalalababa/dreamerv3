"""Reward-label transforms for built controlled buffers (P3 factorial).

Produces a transformed COPY of a built Axis-1 pair directory (side0/side1 +
manifest.json) in which only the per-frame `reward` array is changed; every
other key is byte-preserved and `stepid` is re-encoded for the fresh chunk
uuids (probing/build_controlled_replay.write_episode_chunk -- stale stepids
silently disable Replay.update).

Transforms (per episode, deterministic given --seed):

  shuffle    permute the reward values WITHIN each episode.  Preserves the
             episode's reward density exactly (and hence the episode-level
             occ<->reward-density correlation the 13-Jul battery measured);
             destroys the frame-level reward<->state binding.
  relocate   move all positive reward mass to OUT-OF-REGIME frames of the
             same episode (values preserved, uniformly chosen destinations
             without replacement).  Destroys the binding AND redirects it:
             if reward supervision allocates capacity to labeled frames,
             the labeled frames are now the out-of-regime ones (E4
             measures exactly this).  Episodes with more rewarded frames
             than out-of-regime frames keep the overflow in-regime
             (counted in the manifest; ~impossible in finger Q1).

Reward-free (apt) fits never read `reward` (it is not a decoder key), so
transformed buffers are only meaningful under reward-aware arms; the apt
arm on the original buffer doubles as the transform-invariance control.

Usage (login node; one invocation per pair x transform)::

    python -m probing.relabel_replay transform \
        --input $RUNROOT/axis1_finger/q1 --task dmc_finger_turn_hard \
        --kind shuffle --seed 0 --output $RUNROOT/axis1_finger/q1_sh
    python -m probing.relabel_replay transform \
        --input $RUNROOT/axis1_finger/q1 --task dmc_finger_turn_hard \
        --kind relocate --seed 0 --output $RUNROOT/axis1_finger/q1_rl

    python -m probing.relabel_replay selfcheck
"""

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from probing import regimes
from probing import replay_dataset
from probing.build_controlled_replay import ZERO_UUID, write_episode_chunk


def load_episode_chunks(side_dir):
  """Built side dir -> list of raw per-episode frame dicts (all keys)."""
  paths = replay_dataset.list_chunks(side_dir)
  if not paths:
    raise SystemExit(f'No .npz chunks in {side_dir}')
  episodes = []
  for path in sorted(paths):
    succ = pathlib.Path(path).stem.split('-')[2]
    if succ != str(ZERO_UUID):
      raise SystemExit(
          f'{path}: chunk has a successor; this tool requires the built-'
          f'buffer contract (one chunk per episode, succ=0).')
    with np.load(path) as data:
      episodes.append({k: np.asarray(data[k]) for k in data.files})
  return episodes


def transform_reward(reward, in_regime, kind, rng):
  """Return (new_reward, stats). Values are preserved as a multiset."""
  reward = np.asarray(reward)
  new = np.zeros_like(reward)
  pos = np.flatnonzero(reward > 0)
  stats = dict(n_frames=len(reward), n_rewarded=int(len(pos)))
  if kind == 'shuffle':
    new = reward[rng.permutation(len(reward))]
    stats['rewarded_in_regime'] = int((new > 0)[in_regime].sum())
    return new, stats
  if kind == 'relocate':
    if len(pos) == 0:
      stats['rewarded_in_regime'] = 0
      stats['overflow_in_regime'] = 0
      return new, stats
    out_idx = np.flatnonzero(~np.asarray(in_regime, bool))
    n_out = min(len(pos), len(out_idx))
    dest = rng.choice(out_idx, size=n_out, replace=False)
    values = reward[pos][rng.permutation(len(pos))]
    new[dest] = values[:n_out]
    overflow = len(pos) - n_out
    if overflow:  # more rewarded frames than out-of-regime frames
      in_idx = np.setdiff1d(np.flatnonzero(np.asarray(in_regime, bool)), dest)
      spill = rng.choice(in_idx, size=overflow, replace=False)
      new[spill] = values[n_out:]
    stats['rewarded_in_regime'] = int((new > 0)[np.asarray(in_regime, bool)].sum())
    stats['overflow_in_regime'] = int(overflow)
    return new, stats
  raise ValueError(f'unknown kind {kind!r}')


def cmd_transform(args):
  spec = regimes.spec(args.task)
  in_manifest_path = os.path.join(args.input, 'manifest.json')
  if not os.path.exists(in_manifest_path):
    raise SystemExit(f'{args.input}: no manifest.json (not a built pair dir).')
  with open(in_manifest_path) as f:
    src_manifest = json.load(f)
  if os.path.exists(os.path.join(args.output, 'manifest.json')):
    raise SystemExit(f'{args.output} already materialized; refusing to '
                     f'overwrite (transforms are registered artifacts).')

  sides = sorted(d for d in os.listdir(args.input)
                 if d.startswith('side') and
                 os.path.isdir(os.path.join(args.input, d)))
  if not sides:
    raise SystemExit(f'{args.input}: no side*/ directories.')
  rng = np.random.default_rng(args.seed)
  side_stats = {}
  for side in sides:
    episodes = load_episode_chunks(os.path.join(args.input, side))
    out_dir = os.path.join(args.output, side)
    os.makedirs(out_dir, exist_ok=True)
    ep_stats = []
    for frames in episodes:
      vals = spec['fn'](frames)
      below = spec['direction'] == 'below'
      in_r = (vals < spec['threshold']) if below else (vals > spec['threshold'])
      new_reward, stats = transform_reward(
          frames['reward'], in_r, args.kind, rng)
      assert sorted(new_reward.tolist()) == sorted(
          np.asarray(frames['reward']).tolist()), 'reward multiset changed'
      out_frames = dict(frames)
      out_frames['reward'] = new_reward.astype(frames['reward'].dtype)
      write_episode_chunk(out_dir, out_frames)
      stats['occ'] = round(float(in_r.mean()), 6)
      ep_stats.append(stats)
    side_stats[side] = dict(
        n_episodes=len(ep_stats),
        reward_frames=int(sum(s['n_rewarded'] for s in ep_stats)),
        rewarded_in_regime=int(sum(s['rewarded_in_regime'] for s in ep_stats)),
        overflow_in_regime=int(sum(s.get('overflow_in_regime', 0)
                                   for s in ep_stats)),
        occ_frame=round(float(np.average(
            [s['occ'] for s in ep_stats],
            weights=[s['n_frames'] for s in ep_stats])), 6))
    print(f'{side}: {side_stats[side]}')

  manifest = dict(src_manifest)
  manifest['reward_transform'] = dict(
      kind=args.kind, seed=args.seed, task=args.task,
      source_pair=os.path.abspath(args.input),
      regime=dict(name=spec['name'], threshold=float(spec['threshold']),
                  direction=spec['direction']),
      sides=side_stats,
      note='only the reward array differs from the source pair; all other '
           'keys byte-preserved, stepid re-encoded for the fresh chunk uuids')
  with open(os.path.join(args.output, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
  print(f'-> {args.output} ({args.kind}, seed {args.seed})')


def cmd_selfcheck(args):
  import tempfile
  rng = np.random.default_rng(7)
  with tempfile.TemporaryDirectory() as tmp:
    # Fake built pair: cup-like episode chunks, in-regime = small
    # ball-cup distance (position[:2] vs [2:4]).
    src = os.path.join(tmp, 'q1')
    for side, base in (('side0', 0.8), ('side1', 0.2)):
      out = os.path.join(src, side)
      os.makedirs(out)
      for _ in range(3):
        n = 50
        cup = rng.normal(0, 0.1, (n, 2)).astype(np.float32)
        offs = np.where(rng.random(n) < 0.4, 0.01, base)[:, None]
        pos = np.concatenate([cup, cup + offs], 1).astype(np.float32)
        dist = np.linalg.norm(pos[:, :2] - pos[:, 2:4], axis=1)
        frames = dict(
            position=pos,
            velocity=rng.normal(0, 1, (n, 4)).astype(np.float32),
            action=rng.normal(0, 1, (n, 2)).astype(np.float32),
            reward=(dist < 0.05).astype(np.float32),
            is_first=np.eye(1, n, 0, dtype=bool)[0],
            stepid=np.zeros((n, 20), np.uint8))
        write_episode_chunk(out, frames)
    with open(os.path.join(src, 'manifest.json'), 'w') as f:
      json.dump(dict(task='dmc_cup_catch', sides=[]), f)

    for kind in ('shuffle', 'relocate'):
      out_pair = os.path.join(tmp, f'q1_{kind[:2]}')
      ns = argparse.Namespace(input=src, output=out_pair, seed=0,
                              task='dmc_cup_catch', kind=kind)
      cmd_transform(ns)
      spec = regimes.spec('dmc_cup_catch')
      for side in ('side0', 'side1'):
        orig = load_episode_chunks(os.path.join(src, side))
        new = load_episode_chunks(os.path.join(out_pair, side))
        assert len(orig) == len(new)
        for o, t in zip(orig, new):
          assert sorted(o['reward'].tolist()) == sorted(t['reward'].tolist())
          for key in ('position', 'velocity', 'action', 'is_first'):
            assert np.array_equal(o[key], t[key]), key
          assert not np.array_equal(o['stepid'], t['stepid'])
          if kind == 'relocate' and (t['reward'] > 0).any():
            in_r = regimes.in_regime('dmc_cup_catch', t)
            assert not (t['reward'][in_r] > 0).any(), \
                'relocated reward left in regime'
      with open(os.path.join(out_pair, 'manifest.json')) as f:
        m = json.load(f)
      assert m['reward_transform']['kind'] == kind
      assert m['reward_transform']['sides']['side0']['reward_frames'] > 0
    # Shuffle actually changed at least one episode's reward layout.
    orig = load_episode_chunks(os.path.join(src, 'side0'))
    shuf = load_episode_chunks(os.path.join(tmp, 'q1_sh', 'side0'))
    assert any(not np.array_equal(o['reward'], s['reward'])
               for o, s in zip(orig, shuf))
    # Refuses overwrite.
    try:
      cmd_transform(argparse.Namespace(
          input=src, output=os.path.join(tmp, 'q1_sh'), seed=0,
          task='dmc_cup_catch', kind='shuffle'))
      raise AssertionError('overwrote an existing transform dir')
    except SystemExit:
      pass
  print('relabel_replay selfcheck PASS')


def main():
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)

  t = sub.add_parser('transform')
  t.add_argument('--input', required=True,
                 help='Built pair dir (side0/side1 + manifest.json).')
  t.add_argument('--output', required=True)
  t.add_argument('--task', required=True)
  t.add_argument('--kind', required=True, choices=['shuffle', 'relocate'])
  t.add_argument('--seed', type=int, default=0)
  t.set_defaults(fn=cmd_transform)

  sc = sub.add_parser('selfcheck')
  sc.set_defaults(fn=cmd_selfcheck)

  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
