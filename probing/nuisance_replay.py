"""Nuisance-injection replay transform (PREREG_nuisance_20260811, B2).

Copies a built pair (side0/side1 + manifest.json) appending a
high-variance distractor obs key `distractor` [T, dims] ~ N(0, sigma^2)
to every episode chunk, deterministically per (sigma, dims, filename).
Per side, `--holdout` chunks (rng 0 over the sorted list) are diverted
to heldout/side{k}/ (WITH nuisance) and excluded from the fit sides —
the probe episodes. Same holdout filenames at every sigma level ⇒
matched probe experience across loads.

  python -m probing.nuisance_replay build \
      --input $RUNROOT/axis1_finger/q1 --output $RUNROOT/axis1_finger/q1_nz1 \
      --sigma 1.0 --dims 16 --holdout 16
Selfcheck: python -m probing.nuisance_replay selfcheck
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np

TOOL_VERSION = 'nuisance_replay_v1'


def _chunk_rng(sigma, dims, fname):
  h = hashlib.sha256(
      f'{TOOL_VERSION}|{sigma:.6g}|{dims}|{fname}'.encode()).digest()
  return np.random.default_rng(int.from_bytes(h[:8], 'little'))


def _transform_chunk(src, dst, sigma, dims):
  with np.load(src, allow_pickle=False) as z:
    data = {k: z[k] for k in z.files}
  assert 'distractor' not in data, (src, 'already transformed')
  t = len(data['reward'])
  rng = _chunk_rng(sigma, dims, os.path.basename(src))
  data['distractor'] = rng.normal(0.0, sigma, (t, dims)).astype(np.float32)
  np.savez_compressed(dst, **data)


def build(args):
  assert not os.path.exists(args.output), \
      f'refusing to overwrite {args.output}'
  with open(os.path.join(args.input, 'manifest.json')) as f:
    manifest = json.load(f)
  hold_rng = np.random.default_rng(0)
  holdout_files = {}
  for side in ('side0', 'side1'):
    src_dir = os.path.join(args.input, side)
    files = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(src_dir, '*.npz')))
    assert len(files) > args.holdout >= 1, (side, len(files))
    held = sorted(hold_rng.choice(len(files), size=args.holdout,
                                  replace=False).tolist())
    held_names = [files[i] for i in held]
    holdout_files[side] = held_names
    fit_dir = os.path.join(args.output, side)
    held_dir = os.path.join(args.output, 'heldout', side)
    os.makedirs(fit_dir)
    os.makedirs(held_dir)
    for name in files:
      dst_dir = held_dir if name in held_names else fit_dir
      _transform_chunk(os.path.join(src_dir, name),
                       os.path.join(dst_dir, name), args.sigma, args.dims)
  manifest = dict(manifest)
  manifest['distractor'] = dict(
      tool=TOOL_VERSION, sigma=args.sigma, dims=args.dims,
      holdout_per_side=args.holdout, holdout_files=holdout_files,
      source=os.path.abspath(args.input))
  with open(os.path.join(args.output, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
  n_fit = {s: len(glob.glob(os.path.join(args.output, s, '*.npz')))
           for s in ('side0', 'side1')}
  print(f'-> {args.output} sigma={args.sigma} dims={args.dims} '
        f'fit chunks {n_fit} heldout {args.holdout}/side')


def selfcheck(_args=None):
  import tempfile
  rng = np.random.default_rng(3)
  with tempfile.TemporaryDirectory() as tmp:
    src = os.path.join(tmp, 'q1')
    for side in ('side0', 'side1'):
      os.makedirs(os.path.join(src, side))
      for i in range(8):
        t = 30
        np.savez_compressed(
            os.path.join(src, side, f'ep{i:03d}.npz'),
            position=rng.normal(0, 1, (t, 4)).astype(np.float32),
            reward=(rng.random(t) < 0.2).astype(np.float32),
            is_first=np.eye(1, t, 0, dtype=bool)[0])
    with open(os.path.join(src, 'manifest.json'), 'w') as f:
      json.dump(dict(task='dmc_finger_turn_hard'), f)

    ns = argparse.Namespace(input=src, output=os.path.join(tmp, 'nz1'),
                            sigma=1.0, dims=16, holdout=2)
    build(ns)
    # holdout diverted + disjoint from fit side
    m = json.load(open(os.path.join(tmp, 'nz1', 'manifest.json')))
    for side in ('side0', 'side1'):
      held = set(m['distractor']['holdout_files'][side])
      fit = {os.path.basename(p) for p in
             glob.glob(os.path.join(tmp, 'nz1', side, '*.npz'))}
      hdir = {os.path.basename(p) for p in
              glob.glob(os.path.join(tmp, 'nz1', 'heldout', side,
                                     '*.npz'))}
      assert held == hdir and not (held & fit) and len(fit) == 6
    # originals preserved bit-identical; nuisance sigma realized
    src_z = np.load(os.path.join(src, 'side0', 'ep000.npz'))
    name0 = sorted(os.listdir(os.path.join(tmp, 'nz1', 'side0'))
                   + os.listdir(os.path.join(tmp, 'nz1', 'heldout',
                                             'side0')))[0]
    where = ('side0' if os.path.exists(
        os.path.join(tmp, 'nz1', 'side0', name0))
        else os.path.join('heldout', 'side0'))
    out_z = np.load(os.path.join(tmp, 'nz1', where, 'ep000.npz'))
    assert (out_z['position'] == src_z['position']).all()
    assert (out_z['reward'] == src_z['reward']).all()
    assert out_z['distractor'].shape == (30, 16)
    allnz = np.concatenate(
        [np.load(p)['distractor'].ravel() for p in
         glob.glob(os.path.join(tmp, 'nz1', '**', '*.npz'),
                   recursive=True)])
    assert abs(allnz.std() - 1.0) < 0.05, allnz.std()
    # determinism: rebuild -> identical arrays; sigma changes -> differs
    ns2 = argparse.Namespace(input=src, output=os.path.join(tmp, 'nz1b'),
                             sigma=1.0, dims=16, holdout=2)
    build(ns2)
    a = np.load(os.path.join(tmp, 'nz1', where, 'ep000.npz'))['distractor']
    b = np.load(os.path.join(tmp, 'nz1b', where, 'ep000.npz'))['distractor']
    assert (a == b).all(), 'not deterministic'
    ns4 = argparse.Namespace(input=src, output=os.path.join(tmp, 'nz2'),
                             sigma=4.0, dims=16, holdout=2)
    build(ns4)
    m2 = json.load(open(os.path.join(tmp, 'nz2', 'manifest.json')))
    assert m2['distractor']['holdout_files'] == \
        m['distractor']['holdout_files'], 'holdout not level-invariant'
    c = np.load(os.path.join(tmp, 'nz2', where, 'ep000.npz'))['distractor']
    assert not (a == c).all() and abs(c.std() - 4.0) < 0.2
    # refuses overwrite; refuses double transform
    try:
      build(ns)
      raise SystemExit('expected overwrite refusal')
    except AssertionError:
      pass
  print('nuisance_replay selfcheck PASS (holdout diverted+level-'
        'invariant, originals bit-identical, sigma realized, '
        'deterministic rebuild, overwrite refusal)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  sub = p.add_subparsers(dest='cmd', required=True)
  b = sub.add_parser('build')
  b.add_argument('--input', required=True)
  b.add_argument('--output', required=True)
  b.add_argument('--sigma', type=float, required=True)
  b.add_argument('--dims', type=int, default=16)
  b.add_argument('--holdout', type=int, default=16)
  b.set_defaults(fn=build)
  s = sub.add_parser('selfcheck')
  s.set_defaults(fn=selfcheck)
  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
