"""Export a frozen pixel replay buffer to the stable-worldmodel HDF5
format LeWM trains on (PREREG_lewm_20260812).

Input: a build_controlled_replay side dir (ONE npz chunk per episode,
succ=0 — the frozen chunk contract), each chunk carrying 'image'
[T, H, W, C] uint8 and 'action' [T, A]. Output: <output>.h5 with flat
columns 'pixels' + 'action' and ep_len/ep_offset indices (HDF5Writer
schema, stable-worldmodel 0.1.1), plus <output>_holdout.h5 with the
LAST --holdout episodes (sorted filename order — deterministic), and
<output>.manifest.json recording the split, episode counts, and the
sha256 of both written h5 files. The holdout is the distillation
fidelity set; it never enters LeWM training or distill training.

Runs in the lewm conda env (needs stable_worldmodel + h5py; numpy only
otherwise — no repo deps).

Usage:
  python -m probing.lewm_export --side_dir $RUNROOT/axis1_finger/pxq1m/side0 \
      --output $RUNROOT/lewm_data/finger_pxq1m_side0 --holdout 8
Selfcheck: python -m probing.lewm_export --selfcheck
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for blk in iter(lambda: f.read(1 << 20), b''):
      h.update(blk)
  return h.hexdigest()


def episode_files(side_dir):
  files = sorted(glob.glob(os.path.join(side_dir, '*.npz')))
  assert files, f'no npz chunks under {side_dir}'
  return files


def load_episode(path):
  with np.load(path) as z:
    assert 'image' in z, f'{path}: no image key (pixel buffer required)'
    assert 'action' in z, f'{path}: no action key'
    img = np.asarray(z['image'])
    act = np.asarray(z['action'], np.float32)
  assert img.dtype == np.uint8 and img.ndim == 4, (path, img.dtype,
                                                   img.shape)
  # review C4: the two properties LeWM's reader relies on for its
  # HWC->CHW permute + value-range assumptions
  assert img.shape[-1] in (1, 3), (path, 'channels-last expected',
                                   img.shape)
  assert img.shape[1] == img.shape[2], (path, 'non-square frames',
                                        img.shape)
  assert len(img) == len(act), (path, len(img), len(act))
  return img, act


def write_h5(path, files):
  from stable_worldmodel.data.formats.hdf5 import HDF5Writer
  n_frames = 0
  with HDF5Writer(path, mode='error') as w:
    for f in files:
      img, act = load_episode(f)
      w.write_episode(dict(pixels=img, action=act))
      n_frames += len(img)
  return n_frames


def run(args):
  files = episode_files(args.side_dir)
  assert args.holdout > 0 and args.holdout < len(files), \
      (args.holdout, len(files))
  train_files = files[:-args.holdout]
  hold_files = files[-args.holdout:]
  os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.',
              exist_ok=True)
  train_path = args.output + '.h5'
  hold_path = args.output + '_holdout.h5'
  n_train = write_h5(train_path, train_files)
  n_hold = write_h5(hold_path, hold_files)
  manifest = dict(
      side_dir=os.path.abspath(args.side_dir),
      train_h5=os.path.abspath(train_path),
      holdout_h5=os.path.abspath(hold_path),
      train_h5_sha256=sha256_file(train_path),
      holdout_h5_sha256=sha256_file(hold_path),
      n_train_episodes=len(train_files), n_train_frames=n_train,
      n_holdout_episodes=len(hold_files), n_holdout_frames=n_hold,
      train_files=[os.path.basename(f) for f in train_files],
      holdout_files=[os.path.basename(f) for f in hold_files],
      split_rule='last-N sorted filenames = holdout (deterministic)')
  with open(args.output + '.manifest.json', 'w') as f:
    json.dump(manifest, f, indent=1)
  print(f'{len(train_files)} train eps ({n_train} frames) -> {train_path}')
  print(f'{len(hold_files)} holdout eps ({n_hold} frames) -> {hold_path}')


def selfcheck():
  import shutil
  import tempfile
  import h5py
  tmp = tempfile.mkdtemp(prefix='lewm_export_selfcheck_')
  try:
    side = os.path.join(tmp, 'side0')
    os.makedirs(side)
    rng = np.random.default_rng(0)
    lens = [5, 7, 6, 5]
    for i, n in enumerate(lens):
      np.savez(os.path.join(side, f'ep{i:03d}.npz'),
               image=rng.integers(0, 255, (n, 8, 8, 3), np.uint8),
               action=rng.normal(0, 1, (n, 2)).astype(np.float32))
    args = argparse.Namespace(side_dir=side,
                              output=os.path.join(tmp, 'out'), holdout=1)
    run(args)
    with h5py.File(os.path.join(tmp, 'out.h5'), 'r') as f:
      assert list(f['ep_len'][:]) == lens[:-1], list(f['ep_len'][:])
      assert list(f['ep_offset'][:]) == [0, 5, 12], list(f['ep_offset'][:])
      assert f['pixels'].shape == (18, 8, 8, 3) and \
          f['pixels'].dtype == np.uint8
      assert f['action'].shape == (18, 2)
      # round-trip content of episode 1
      with np.load(os.path.join(side, 'ep001.npz')) as z:
        assert np.array_equal(f['pixels'][5:12], z['image'])
        assert np.allclose(f['action'][5:12], z['action'])
    with h5py.File(os.path.join(tmp, 'out_holdout.h5'), 'r') as f:
      assert list(f['ep_len'][:]) == [lens[-1]]
    man = json.load(open(os.path.join(tmp, 'out.manifest.json')))
    assert man['holdout_files'] == ['ep003.npz']
    assert set(man['train_files']).isdisjoint(man['holdout_files'])
    # overwrite refusal (mode='error')
    try:
      run(args)
      raise SystemExit('overwrite was not refused')
    except FileExistsError:
      pass
    # image-less chunk refusal
    side2 = os.path.join(tmp, 'side1')
    os.makedirs(side2)
    np.savez(os.path.join(side2, 'ep000.npz'),
             action=np.zeros((4, 2), np.float32))
    np.savez(os.path.join(side2, 'ep001.npz'),
             image=rng.integers(0, 255, (4, 8, 8, 3), np.uint8),
             action=np.zeros((4, 2), np.float32))
    try:
      run(argparse.Namespace(side_dir=side2,
                             output=os.path.join(tmp, 'out2'), holdout=1))
      raise SystemExit('image-less chunk was not refused')
    except AssertionError:
      pass
    # non-uint8 image refusal (review C4: the mutant that deletes the
    # dtype gate must fail HERE, not survive on an all-uint8 fixture)
    side3 = os.path.join(tmp, 'side2')
    os.makedirs(side3)
    np.savez(os.path.join(side3, 'ep000.npz'),
             image=rng.random((4, 8, 8, 3)).astype(np.float32),
             action=np.zeros((4, 2), np.float32))
    np.savez(os.path.join(side3, 'ep001.npz'),
             image=rng.integers(0, 255, (4, 8, 8, 3), np.uint8),
             action=np.zeros((4, 2), np.float32))
    try:
      run(argparse.Namespace(side_dir=side3,
                             output=os.path.join(tmp, 'out3'), holdout=1))
      raise SystemExit('float image was not refused')
    except AssertionError:
      pass
    # non-square refusal
    side4 = os.path.join(tmp, 'side3')
    os.makedirs(side4)
    np.savez(os.path.join(side4, 'ep000.npz'),
             image=rng.integers(0, 255, (4, 8, 6, 3), np.uint8),
             action=np.zeros((4, 2), np.float32))
    np.savez(os.path.join(side4, 'ep001.npz'),
             image=rng.integers(0, 255, (4, 8, 8, 3), np.uint8),
             action=np.zeros((4, 2), np.float32))
    try:
      run(argparse.Namespace(side_dir=side4,
                             output=os.path.join(tmp, 'out4'), holdout=1))
      raise SystemExit('non-square image was not refused')
    except AssertionError:
      pass
    # review C4: reader-side round trip through the REAL HDF5Dataset
    # (needs torch — runs in the lewm env; skipped where torch absent,
    # in which case the cluster-side selfcheck is the registered gate)
    try:
      import torch
      have_torch = hasattr(torch, 'from_numpy')  # False under stubs
    except ImportError:
      have_torch = False
    if have_torch:
      from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
      ds = HDF5Dataset(path=os.path.join(tmp, 'out.h5'),
                       keys_to_load=['pixels', 'action'],
                       frameskip=1, num_steps=2)
      item = ds[0]
      px = np.asarray(item['pixels'])
      assert px.shape == (2, 3, 8, 8), \
          ('reader did not emit [T,C,H,W] pixels', px.shape)
      print('HDF5Dataset read-back OK')
    else:
      print('SKIP HDF5Dataset read-back (no torch here; cluster '
            'selfcheck is the registered gate)')
    print('SELFCHECK PASS')
  finally:
    shutil.rmtree(tmp, ignore_errors=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--selfcheck', action='store_true')
  ap.add_argument('--side_dir')
  ap.add_argument('--output')
  ap.add_argument('--holdout', type=int, default=8)
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  for req in ('side_dir', 'output'):
    assert getattr(args, req), f'--{req} required'
  run(args)


if __name__ == '__main__':
  main()
