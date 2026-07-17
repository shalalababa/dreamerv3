"""Direct buffer synthesis for the synth predictable-heads domain.

Builds Axis-1-compatible buffer PAIRS (side0/side1 + manifest.json) for
`synth_reach` (embodied/envs/synthpred.py) with exact dials and no
collector training (Plan_PredictableHeads_20260717.md):

  * occupancy: scripted controller reaches the target region, dwells
    until the episode's in-regime frame count hits the target, then
    leaves; occupancy-zero episodes actively avoid the region. The
    count uses the same regime test as the env reward, so realized
    occupancy tracks the dial to within boundary-transit frames.
  * label density f: fraction of in-regime frames labeled;
  * conditional predictability p: fraction of the labels placed on
    in-regime frames (the rest are resampled onto out-of-regime frames
    -- marginal label count preserved, usable label energy ~ p * a^2);
  * amplitude s: label value.

Labels are painted AFTER the rollout (trajectories are identical across
label settings given the same seed/occ dials). Binding and head-identity
arms REUSE probing/relabel_replay.py transforms on the built pair
(shuffle / stamp_rand / stamp_iid) -- validated by the selfcheck.

Chunks carry the full replay key set (obs + action + flags + stepid +
zero dyn/* context latents, refreshed by Replay.update during the fit
thanks to write_episode_chunk's stepid re-encoding). dyn/* dims default
to size1m (deter 512, stoch 32x4); pass --latent_deter/--latent_stoch
to match a different AXIS1_SIZE.

Usage (login node, CPU, seconds-minutes)::

    python -m probing.synth_buffers build-pair \
        --occ_hi 0.30 --occ_lo 0.03 --density 1.0 --predictability 1.0 \
        --episodes 100 --seed 0 --output $RUNROOT/axis1_synth/q1

    python -m probing.synth_buffers selfcheck
"""

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from embodied.envs.synthpred import SynthPred, GOAL
from probing import regimes
from probing.build_controlled_replay import write_episode_chunk

KP, KD, NOISE = 3.0, 1.0, 0.1


def _policy_to(target, pos, vel, rng):
  return np.clip(KP * (target - pos) - KD * vel +
                 NOISE * rng.standard_normal(2), -1.0, 1.0)


def _far_waypoint(rng, min_goal_dist=0.5):
  while True:
    w = rng.uniform(-0.95, 0.95, 2)
    if np.linalg.norm(w - GOAL) >= min_goal_dist:
      return w


def roll_episode(env, n_target, threshold, rng):
  """One scripted episode; returns frames dict (length+1 rows).

  n_target: in-regime frame budget (0 = avoid the region entirely)."""
  obs = env.step({'reset': True, 'action': np.zeros(2, np.float32)})
  rows, actions = [obs], []
  n_in, waypoint = 0, _far_waypoint(rng)
  while True:
    pos = np.asarray(obs['position'], np.float64)
    vel = np.asarray(obs['velocity'], np.float64)
    if n_in < n_target:
      a = _policy_to(GOAL, pos, vel, rng)
    else:
      if np.linalg.norm(waypoint - pos) < 0.15:
        waypoint = _far_waypoint(rng)
      a = _policy_to(waypoint, pos, vel, rng)
      if n_target == 0 and np.linalg.norm(GOAL - pos) < 2.5 * threshold:
        a = np.clip(-KP * (GOAL - pos) - KD * vel, -1.0, 1.0)
    a = a.astype(np.float32)
    actions.append(a)
    obs = env.step({'reset': False, 'action': a})
    rows.append(obs)
    if np.linalg.norm(obs['to_target']) < threshold:
      n_in += 1
    if obs['is_last']:
      break
  actions.append(np.zeros(2, np.float32))  # post-action convention
  frames = {k: np.stack([np.asarray(r[k]) for r in rows])
            for k in rows[0]}
  frames['action'] = np.stack(actions)
  return frames


def paint_labels(frames_list, threshold, density, predictability, amplitude,
                 rng):
  """Replace the reward channel with painted labels; returns stats."""
  in_masks = []
  for frames in frames_list:
    dist = np.linalg.norm(np.asarray(frames['to_target'], np.float64), axis=1)
    m = dist < threshold
    m[0] = False  # is_first frames never carry reward
    in_masks.append(m)
  flat_in = np.concatenate(in_masks)
  n_frames = len(flat_in)
  offsets = np.cumsum([0] + [len(m) for m in in_masks])
  in_idx = np.flatnonzero(flat_in)
  first_rows = offsets[:-1]
  out_idx = np.setdiff1d(np.flatnonzero(~flat_in), first_rows)
  n_labels = int(round(density * len(in_idx)))
  n_true = int(round(predictability * n_labels))
  n_noise = n_labels - n_true
  chosen_in = rng.choice(in_idx, size=n_true, replace=False) \
      if n_true else np.array([], int)
  chosen_out = rng.choice(out_idx, size=n_noise, replace=False) \
      if n_noise else np.array([], int)
  flat_reward = np.zeros(n_frames, np.float64)
  flat_reward[chosen_in] = amplitude
  flat_reward[chosen_out] = amplitude
  for i, frames in enumerate(frames_list):
    frames['reward'] = flat_reward[offsets[i]:offsets[i + 1]].astype(
        np.float32)
  return dict(
      n_frames=int(n_frames), n_in_regime=int(len(in_idx)),
      occ_frame=round(float(flat_in.mean()), 6),
      n_labels=int(n_labels), labels_in_regime=int(n_true),
      labels_out_regime=int(n_noise),
      density_realized=round(n_labels / max(len(in_idx), 1), 6),
      predictability_realized=round(n_true / max(n_labels, 1), 6),
      amplitude=float(amplitude))


def realized_in(frames, threshold):
  dist = np.linalg.norm(np.asarray(frames['to_target'], np.float64), axis=1)
  return int((dist < threshold).sum())


def build_side(out_dir, occ, args, seed, spec):
  def make_env(s):
    return SynthPred(
        'reach', length=args.length, distractors=args.distractors,
        distractor_scale=args.distractor_scale, coupling=args.coupling,
        radius=args.radius, seed=s)

  n_target = int(round(occ * (args.length + 1)))
  budget = n_target
  if n_target > 0:
    # Calibration pass: the controller needs a few frames to exit the
    # region after the budget is spent; measure that overshoot on probe
    # episodes (separate env/rng seeds, discarded) and shrink the dwell
    # budget so realized occupancy tracks the dial.
    probe_env, probe_rng = make_env([seed, 9]), np.random.default_rng([seed, 9])
    over = [realized_in(roll_episode(probe_env, n_target,
                                     spec['threshold'], probe_rng),
                        spec['threshold']) - n_target
            for _ in range(3)]
    budget = max(n_target - int(round(np.mean(over))), 1)
  env = make_env(seed)
  rng = np.random.default_rng([seed, 1])
  frames_list = [roll_episode(env, budget, spec['threshold'], rng)
                 for _ in range(args.episodes)]
  stats = paint_labels(frames_list, spec['threshold'], args.density,
                       args.predictability, args.amplitude,
                       np.random.default_rng([seed, 2]))
  os.makedirs(out_dir, exist_ok=True)
  deter = np.zeros(args.latent_deter, np.float32)
  stoch = np.zeros(tuple(args.latent_stoch), np.float32)
  for frames in frames_list:
    n = len(frames['reward'])
    frames['dyn/deter'] = np.tile(deter, (n, 1))
    frames['dyn/stoch'] = np.tile(stoch, (n,) + (1,) * stoch.ndim)
    frames['stepid'] = np.zeros((n, 20), np.uint8)  # re-encoded on write
    write_episode_chunk(out_dir, frames)
  stats.update(n_episodes=args.episodes, occ_target=float(occ))
  return stats


def cmd_build_pair(args):
  if os.path.exists(os.path.join(args.output, 'manifest.json')):
    raise SystemExit(f'{args.output} already materialized; refusing to '
                     f'overwrite (buffers are registered artifacts).')
  spec = regimes.spec('synth_reach')
  assert abs(spec['threshold'] - args.radius) < 1e-9, \
      'env radius must match the synth regime spec threshold'
  sides = {}
  for side, occ, seed_off in (('side0', args.occ_lo, 0),
                              ('side1', args.occ_hi, 1)):
    stats = build_side(os.path.join(args.output, side), occ, args,
                       [args.seed, seed_off], spec)
    sides[side] = stats
    print(f'{side}: {stats}')
  manifest = dict(
      task='synth_reach',
      generator=dict(
          tool='probing/synth_buffers.py', seed=args.seed,
          episodes=args.episodes, length=args.length,
          density=args.density, predictability=args.predictability,
          amplitude=args.amplitude,
          env=dict(distractors=args.distractors,
                   distractor_scale=args.distractor_scale,
                   coupling=args.coupling, radius=args.radius),
          latent_dims=dict(deter=args.latent_deter,
                           stoch=list(args.latent_stoch))),
      regime=dict(name=spec['name'], threshold=float(spec['threshold']),
                  direction=spec['direction']),
      sides=sides,
      sources={'scripted': dict(episodes=2 * args.episodes)},
      note='directly synthesized buffer (no collector training); '
           'trajectories are label-independent; labels painted post-hoc '
           'with dialed density/predictability/amplitude')
  with open(os.path.join(args.output, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
  print(f'-> {args.output} (occ {args.occ_lo}/{args.occ_hi}, '
        f'f={args.density}, p={args.predictability})')


def cmd_selfcheck(args):
  import tempfile
  from probing.relabel_replay import load_episode_chunks, cmd_transform
  with tempfile.TemporaryDirectory() as tmp:
    ns = argparse.Namespace(
        occ_hi=0.30, occ_lo=0.02, density=0.8, predictability=0.75,
        amplitude=1.0, episodes=4, length=200, seed=0, distractors=4,
        distractor_scale=1.0, coupling=0.0, radius=0.1,
        latent_deter=16, latent_stoch=[4, 2],
        output=os.path.join(tmp, 'q1'))
    cmd_build_pair(ns)
    with open(os.path.join(tmp, 'q1', 'manifest.json')) as f:
      m = json.load(f)
    for side, occ in (('side0', 0.02), ('side1', 0.30)):
      s = m['sides'][side]
      assert abs(s['occ_frame'] - occ) < 0.01, (side, s)
      assert s['density_realized'] == round(
          s['n_labels'] / max(s['n_in_regime'], 1), 6)
      # exact up to integer rounding of round(p * n_labels)
      assert abs(s['predictability_realized'] - 0.75) <= \
          0.5 / max(s['n_labels'], 1) + 1e-9, s
      eps = load_episode_chunks(os.path.join(tmp, 'q1', side))
      assert len(eps) == 4
      for e in eps:
        assert len(e['reward']) == 201
        assert e['is_first'][0] and not e['is_first'][1:].any()
        assert e['is_last'][-1] and not e['is_last'][:-1].any()
        assert e['reward'][0] == 0.0
        np.testing.assert_allclose(
            e['to_target'], (GOAL - e['position']).astype(np.float32),
            atol=1e-6)
        assert e['dyn/deter'].shape == (201, 16)
        assert e['dyn/stoch'].shape == (201, 4, 2)
        vals = e['reward'][e['reward'] > 0]
        assert vals.size == 0 or np.allclose(vals, 1.0)
      # painted label accounting matches the manifest
      dist = np.concatenate(
          [np.linalg.norm(e['to_target'], axis=1) for e in eps])
      lab = np.concatenate([e['reward'] for e in eps]) > 0
      assert int(lab.sum()) == s['n_labels']
      assert int((lab & (dist < 0.1)).sum()) == s['labels_in_regime']
    # determinism: identical dials/seed reproduce identical rewards
    ns2 = argparse.Namespace(**{**vars(ns), 'output': os.path.join(tmp, 'q1b')})
    cmd_build_pair(ns2)
    a = load_episode_chunks(os.path.join(tmp, 'q1', 'side1'))
    b = load_episode_chunks(os.path.join(tmp, 'q1b', 'side1'))
    for x, y in zip(a, b):
      assert np.array_equal(x['reward'], y['reward'])
      assert np.array_equal(x['position'], y['position'])
    # the whole relabel chain works on the built pair (binding arm reuse)
    cmd_transform(argparse.Namespace(
        input=os.path.join(tmp, 'q1'), output=os.path.join(tmp, 'q1_sh'),
        seed=0, task='synth_reach', kind='shuffle'))
    sh = load_episode_chunks(os.path.join(tmp, 'q1_sh', 'side1'))
    for x, y in zip(a, sh):
      assert sorted(x['reward'].tolist()) == sorted(y['reward'].tolist())
  print('synth_buffers selfcheck PASS')


def main():
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)

  b = sub.add_parser('build-pair')
  b.add_argument('--occ_hi', type=float, required=True,
                 help='side1 target frame occupancy (e.g. 0.30).')
  b.add_argument('--occ_lo', type=float, required=True,
                 help='side0 target frame occupancy (e.g. 0.03; 0 = avoid).')
  b.add_argument('--density', type=float, default=1.0)
  b.add_argument('--predictability', type=float, default=1.0)
  b.add_argument('--amplitude', type=float, default=1.0)
  b.add_argument('--episodes', type=int, default=100)
  b.add_argument('--length', type=int, default=1000)
  b.add_argument('--seed', type=int, default=0)
  b.add_argument('--distractors', type=int, default=8)
  b.add_argument('--distractor_scale', type=float, default=1.0)
  b.add_argument('--coupling', type=float, default=0.0)
  b.add_argument('--radius', type=float, default=0.1)
  b.add_argument('--latent_deter', type=int, default=512)
  b.add_argument('--latent_stoch', type=int, nargs='+', default=[32, 4])
  b.add_argument('--output', required=True)
  b.set_defaults(fn=cmd_build_pair)

  sc = sub.add_parser('selfcheck')
  sc.set_defaults(fn=cmd_selfcheck)

  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
