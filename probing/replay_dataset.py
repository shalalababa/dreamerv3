"""Read DreamerV3 replay-buffer chunks as a flat per-frame dataset.

DreamerV3 stores its replay buffer as a directory of compressed `.npz`
chunks (see `embodied/core/chunk.py`); every chunk holds `length` consecutive
transitions with one array per observation/action/flag key.

The static VAE is a per-frame model, so for VAE training we only need the
*set* of observation frames -- their temporal order is irrelevant. This
module flattens the chunks into `{key: (N, d)}` arrays. Training the VAE on
exactly these frames is what makes "the same observation stream" literal:
both the RSSM and the VAE see the identical input distribution.
"""

import glob
import os

import numpy as np


def find_replay_dir(run_logdir):
  """Return the replay sub-directory of a DreamerV3 run logdir."""
  cand = os.path.join(run_logdir, 'replay')
  if os.path.isdir(cand):
    return cand
  # Some runs nest replay under a replica index (e.g. replay/00000).
  nested = sorted(glob.glob(os.path.join(cand, '*')))
  nested = [p for p in nested if os.path.isdir(p)]
  if len(cand) and nested:
    return nested[0]
  raise FileNotFoundError(f'No replay directory under {run_logdir}')


def list_chunks(replay_dir):
  return sorted(glob.glob(os.path.join(replay_dir, '*.npz')))


def peek_keys(replay_dir):
  """Report the keys and per-frame shapes available in the first chunk."""
  chunks = list_chunks(replay_dir)
  if not chunks:
    raise FileNotFoundError(f'No .npz chunks in {replay_dir}')
  with np.load(chunks[0]) as data:
    return {k: tuple(data[k].shape[1:]) for k in data.files}


def load_frames(replay_dirs, keys, max_frames=None, rng=0, verbose=True):
  """Flatten replay chunks into a per-frame dataset.

  Args:
    replay_dirs: one replay dir, or a list (e.g. one per DreamerV3 seed).
    keys: observation keys to extract (each kept as float32 (N, d)).
    max_frames: optional cap; frames are uniformly subsampled if exceeded.
    rng: seed for the subsampling.

  Returns:
    dict {key: (N, d) float32}, with a shared frame ordering across keys.
  """
  if isinstance(replay_dirs, str):
    replay_dirs = [replay_dirs]
  buffers = {k: [] for k in keys}
  total = 0
  for replay_dir in replay_dirs:
    chunks = list_chunks(replay_dir)
    if not chunks:
      raise FileNotFoundError(f'No .npz chunks in {replay_dir}')
    for path in chunks:
      with np.load(path) as data:
        missing = [k for k in keys if k not in data.files]
        if missing:
          raise KeyError(f'{path} is missing keys {missing}; '
                         f'available: {sorted(data.files)}')
        n = data[keys[0]].shape[0]
        for k in keys:
          arr = np.asarray(data[k], np.float32)
          buffers[k].append(arr.reshape(arr.shape[0], -1))
        total += n
    if verbose:
      print(f'  {replay_dir}: {len(chunks)} chunks')
  frames = {k: np.concatenate(v, 0) for k, v in buffers.items()}
  n = next(iter(frames.values())).shape[0]
  assert all(v.shape[0] == n for v in frames.values()), \
      {k: v.shape for k, v in frames.items()}
  if verbose:
    print(f'Loaded {n} frames; keys: '
          f'{ {k: v.shape[1] for k, v in frames.items()} }')
  if max_frames and n > max_frames:
    idx = np.random.default_rng(rng).choice(n, max_frames, replace=False)
    idx.sort()
    frames = {k: v[idx] for k, v in frames.items()}
    if verbose:
      print(f'Subsampled to {max_frames} frames')
  return frames


def iterate_batches(frames, batch_size, rng):
  """Yield shuffled mini-batches {key: (B, d)} for one epoch."""
  n = next(iter(frames.values())).shape[0]
  order = np.random.default_rng(rng).permutation(n)
  for start in range(0, n - batch_size + 1, batch_size):
    idx = order[start:start + batch_size]
    yield {k: v[idx] for k, v in frames.items()}
