"""Build, freeze, and version held-out probe sets for latent dumps.

Background (Rep-Convergence idea brief, 2 Jul 2026, Sec. 4)
-----------------------------------------------------------
The representational-convergence study needs, per domain, a *fixed* set of a
few thousand held-out observation subsequences, each long enough to burn in
the RSSM recurrent state so posteriors are well-defined.  Every latent dump
(DreamerV3 now, TD-MPC2 in Phase 7b) encodes the *same* frozen windows, so
alignment across checkpoints/architectures is computed on identical inputs.
The probe set is built from the Gate-0 pilot buffers, which are disjoint from
the Phase-4+ training replays -- that is what makes it held-out.

Protocol
--------
* A probe *window* is ``length = burn_in + eval_steps`` contiguous steps from
  one replay stream, containing no episode boundary after its first step.
  Encoders reset state at the window start and burn in on the first
  ``burn_in`` steps; probe points are read from the last ``eval_steps`` steps.
* Windows are non-overlapping within a stream and stratified across the pilot
  buffers (``label=replay_dir``), sampled with a fixed seed.
* The output directory is content-versioned: ``manifest.json`` records every
  parameter plus the sha256 of ``probeset.npz``; ``--freeze`` (default) writes
  a ``FROZEN`` marker.  A frozen set is never overwritten, and consumers
  (``probing/latents.py``) verify the hash before encoding.

Usage
-----
Build and freeze one probe set per decoupled domain from the pilot buffers::

    python -m probing.probeset \
        --task dmc_cup_catch \
        --replay p2e=$RUN/pilot_p2e_cup/replay apt=$RUN/pilot_apt_cup/replay \
                 random=$RUN/pilot_random_cup/replay \
                 goal=$RUN/pilot_goal_cup/replay \
        --output $SCRATCH/probesets/cup_v1 \
        --windows_per_source 640 --burn_in 32 --eval_steps 8 --seed 0

Verify an existing set (also run by consumers)::

    python -m probing.probeset --verify --output $SCRATCH/probesets/cup_v1

Validation (synthetic end-to-end self-check, no data needed)::

    python -m probing.probeset --selfcheck
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from probing import regimes
from probing import replay_dataset

ZERO_UUID = '0' * 22


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--replay', nargs='+', default=[],
                 help='label=replay_dir pilot buffers (>=1).')
  p.add_argument('--task', default='',
                 help='dmc task, e.g. dmc_cup_catch (names the domain and '
                      'enables regime stats).')
  p.add_argument('--output', default='', help='Probe-set directory.')
  p.add_argument('--windows_per_source', type=int, default=640)
  p.add_argument('--burn_in', type=int, default=32,
                 help='Steps used to burn in recurrent state; not probed.')
  p.add_argument('--eval_steps', type=int, default=8,
                 help='Steps after burn-in from which probe points are read.')
  p.add_argument('--tag', default='v1', help='Human-readable version tag.')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--no-freeze', dest='freeze', action='store_false',
                 help='Skip the FROZEN marker (iteration only; consumers '
                      'refuse unfrozen sets by default).')
  p.add_argument('--verify', action='store_true',
                 help='Verify manifest/hash of --output and exit.')
  p.add_argument('--selfcheck', action='store_true',
                 help='Run the synthetic end-to-end self-check and exit.')
  return p.parse_args(argv)


def chain_streams(replay_dir):
  """Group chunk files into per-worker streams via successor UUIDs.

  Returns a list of lists of chunk paths, each list one contiguous stream in
  temporal order.  Chains are recovered from filenames alone
  (``time-uuid-succ-length.npz``); a missing successor ends the chain.
  """
  paths = replay_dataset.list_chunks(replay_dir)
  if not paths:
    raise FileNotFoundError(f'No .npz chunks in {replay_dir}')
  info = {}
  for path in paths:
    time, uuid, succ, length = pathlib.Path(path).stem.split('-')
    info[uuid] = dict(path=path, time=time, succ=succ, length=int(length))
  successors = {v['succ'] for v in info.values() if v['succ'] != ZERO_UUID}
  heads = sorted((u for u in info if u not in successors),
                 key=lambda u: info[u]['time'])
  streams, used = [], set()
  for head in heads:
    chain, uuid = [], head
    while uuid in info and uuid not in used:
      used.add(uuid)
      chain.append(info[uuid]['path'])
      uuid = info[uuid]['succ']
    streams.append(chain)
  leftover = [u for u in info if u not in used]
  if leftover:  # cycles or orphans; treat each as its own stream
    for u in sorted(leftover, key=lambda u: info[u]['time']):
      streams.append([info[u]['path']])
  return streams


def load_stream(chunk_paths):
  """Concatenate a chunk chain into {key: (N, ...)} arrays.

  Skips every key containing '/' (agent-internal annotations such as
  ``dyn/deter``): they are artifacts of the collecting run and must not be
  baked into a held-out probe set.
  """
  buffers = {}
  for path in chunk_paths:
    with np.load(path) as data:
      for k in data.files:
        if '/' in k:
          continue
        buffers.setdefault(k, []).append(np.asarray(data[k]))
  return {k: np.concatenate(v, 0) for k, v in buffers.items()}


def valid_starts(is_first, length):
  """Start indices of non-overlapping windows with no reset after step 0."""
  n = len(is_first)
  starts = []
  t = 0
  while t + length <= n:
    if is_first[t + 1:t + length].any():
      # Skip to the last boundary inside the window; a window may *start* at
      # (or after) a boundary but never straddle one.
      t += 1 + int(np.where(is_first[t + 1:t + length])[0][-1])
      continue
    starts.append(t)
    t += length
  return starts


def collect_windows(replay_dir, length, quota, rng, allow_fewer=False):
  """Sample `quota` non-overlapping valid windows from one pilot buffer.

  With ``allow_fewer`` (used by density-reference sampling in
  ``probing/latent_uq.py``), a short buffer yields all its valid windows
  instead of aborting; probe-set builds keep the strict behavior.
  """
  streams = chain_streams(replay_dir)
  candidates = []  # (stream_idx, start)
  loaded = []
  for si, chain in enumerate(streams):
    data = load_stream(chain)
    loaded.append(data)
    is_first = np.asarray(data['is_first'], bool).reshape(-1)
    candidates.extend((si, t) for t in valid_starts(is_first, length))
  if len(candidates) < quota:
    if allow_fewer and candidates:
      print(f'WARNING: {replay_dir}: only {len(candidates)} valid windows of '
            f'length {length} (asked for {quota}); using all of them.')
      quota = len(candidates)
    else:
      raise SystemExit(
          f'{replay_dir}: only {len(candidates)} valid windows of length '
          f'{length} (need {quota}); lower --windows_per_source or collect '
          f'longer pilots.')
  pick = rng.choice(len(candidates), quota, replace=False)
  pick.sort()
  windows, meta = {}, []
  for i in pick:
    si, t = candidates[i]
    for k, v in loaded[si].items():
      windows.setdefault(k, []).append(v[t:t + length])
    meta.append((si, t))
  n_chunks = sum(len(c) for c in streams)
  return windows, meta, dict(streams=len(streams), chunks=n_chunks,
                             candidates=len(candidates))


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
      h.update(block)
  return h.hexdigest()


def git_commit():
  try:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
        stderr=subprocess.DEVNULL).decode().strip()
  except Exception:
    return None


def build(args):
  out = args.output
  frozen_marker = os.path.join(out, 'FROZEN')
  if os.path.exists(frozen_marker):
    raise SystemExit(
        f'{out} is FROZEN; refusing to overwrite. Build a new version into a '
        f'new directory instead.')
  assert args.replay, 'Need at least one label=replay_dir entry.'
  assert args.burn_in >= 1 and args.eval_steps >= 1
  length = args.burn_in + args.eval_steps
  os.makedirs(out, exist_ok=True)
  rng = np.random.default_rng(args.seed)

  data, labels, starts, sources = {}, [], [], {}
  for item in args.replay:
    assert '=' in item, f'Expected label=dir, got {item!r}'
    label, directory = item.split('=', 1)
    windows, meta, stats = collect_windows(
        directory, length, args.windows_per_source, rng)
    for k, v in windows.items():
      data.setdefault(k, []).extend(v)
    labels.extend([label] * len(meta))
    starts.extend(meta)
    sources[label] = dict(replay_dir=os.path.abspath(directory), **stats,
                          windows=len(meta))
    print(f'  {label}: {len(meta)} windows from {stats["streams"]} streams '
          f'({stats["candidates"]} candidates)')

  counts = {len(v) for v in data.values()}
  assert len(counts) == 1, f'Inconsistent keys across buffers: {counts}'
  arrays = {}
  for k, v in data.items():
    arr = np.stack(v, 0)
    if arr.dtype in (np.float64,):
      arr = arr.astype(np.float32)
    arrays[k] = arr
  arrays['source_label'] = np.asarray(labels)
  arrays['stream_start'] = np.asarray(starts, np.int64)  # (S, 2) stream, t

  # Descriptive regime stats over the eval segment (never a filter).
  regime_stats = None
  if args.task and regimes.has_regime(args.task):
    spec = regimes.spec(args.task)
    if all(k in arrays for k in spec['needs']):
      S = arrays['is_first'].shape[0]
      tail = {k: arrays[k][:, args.burn_in:].reshape(
          S * args.eval_steps, -1) for k in spec['needs']}
      occ = regimes.in_regime(args.task, tail).reshape(S, args.eval_steps)
      per_window = occ.mean(1)
      regime_stats = {
          label: float(per_window[arrays['source_label'] == label].mean())
          for label in sources}
      arrays['regime_eval'] = regimes.regime_values(args.task, tail).reshape(
          S, args.eval_steps).astype(np.float32)

  npz_path = os.path.join(out, 'probeset.npz')
  np.savez_compressed(npz_path, **arrays)
  digest = sha256_file(npz_path)
  domain = regimes.domain_of(args.task) if args.task else 'unknown'
  probeset_id = f'{domain}_{args.tag}_{digest[:8]}'
  manifest = dict(
      probeset_id=probeset_id,
      task=args.task, domain=domain, tag=args.tag,
      windows=int(arrays['is_first'].shape[0]),
      length=length, burn_in=args.burn_in, eval_steps=args.eval_steps,
      windows_per_source=args.windows_per_source, seed=args.seed,
      sources=sources, regime_occupancy_eval=regime_stats,
      keys={k: list(v.shape[1:]) for k, v in arrays.items()},
      sha256=digest, git_commit=git_commit(),
      created=datetime.datetime.now().isoformat(timespec='seconds'),
      frozen=bool(args.freeze))
  with open(os.path.join(out, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
  if args.freeze:
    with open(frozen_marker, 'w') as f:
      f.write(digest + '\n')
  state = 'FROZEN' if args.freeze else 'UNFROZEN (pass --verify before use)'
  print(f'Probe set {probeset_id}: {manifest["windows"]} windows x '
        f'{length} steps ({args.burn_in} burn-in + {args.eval_steps} eval)')
  print(f'  sha256={digest[:16]}...  {state}\n  -> {out}')
  return manifest


def load(probeset_dir, require_frozen=True):
  """Load and verify a probe set; returns (arrays, manifest)."""
  manifest_path = os.path.join(probeset_dir, 'manifest.json')
  npz_path = os.path.join(probeset_dir, 'probeset.npz')
  with open(manifest_path) as f:
    manifest = json.load(f)
  frozen = os.path.exists(os.path.join(probeset_dir, 'FROZEN'))
  if require_frozen and not frozen:
    raise SystemExit(f'{probeset_dir} is not frozen; freeze it first or pass '
                     f'--allow_unfrozen (results will not be comparable).')
  digest = sha256_file(npz_path)
  if digest != manifest['sha256']:
    raise SystemExit(f'{probeset_dir}: probeset.npz sha256 mismatch '
                     f'(manifest {manifest["sha256"][:12]}..., file '
                     f'{digest[:12]}...); the frozen set was modified.')
  if frozen:
    with open(os.path.join(probeset_dir, 'FROZEN')) as f:
      marker = f.read().strip()
    if marker != digest:
      raise SystemExit(f'{probeset_dir}: FROZEN marker does not match file.')
  arrays = {k: np.asarray(v) for k, v in np.load(npz_path).items()}
  return arrays, manifest


def verify(probeset_dir):
  arrays, manifest = load(probeset_dir, require_frozen=False)
  S = arrays['is_first'].shape[0]
  L = manifest['length']
  assert arrays['is_first'].shape[1] == L, (arrays['is_first'].shape, L)
  assert not arrays['is_first'][:, 1:].any(), \
      'Episode boundary inside a window.'
  print(f'OK {manifest["probeset_id"]}: {S} windows x {L} steps, '
        f'sha256 verified, frozen={manifest["frozen"]}.')
  return manifest


def selfcheck():
  """Synthetic end-to-end check: chaining, windowing, freeze, tamper."""
  import tempfile
  tmp = tempfile.mkdtemp(prefix='probeset_selfcheck_')
  rng = np.random.default_rng(0)

  def write_chunks(replay_dir, n_streams=2, n_chunks=3, size=200, ep_len=130):
    os.makedirs(replay_dir, exist_ok=True)
    for si in range(n_streams):
      uuids = [f'{rng.integers(10**18):022d}' for _ in range(n_chunks)]
      offset = 0
      for ci in range(n_chunks):
        succ = uuids[ci + 1] if ci + 1 < n_chunks else ZERO_UUID
        idx = offset + np.arange(size)
        ball = rng.normal(size=(size, 2)).astype(np.float32)
        data = dict(
            # dim 0 counts steps within the stream (contiguity check); dims
            # 2:4 make the ball_in_cup regime quantity computable.
            position=np.stack(
                [idx, np.full(size, si), ball[:, 0], ball[:, 1]],
                -1).astype(np.float32),
            velocity=rng.normal(size=(size, 2)).astype(np.float32),
            action=rng.normal(size=(size, 1)).astype(np.float32),
            reward=np.zeros(size, np.float32),
            is_first=(idx % ep_len == 0),
            is_last=np.zeros(size, bool),
            is_terminal=np.zeros(size, bool))
        name = f'2026070{si}T00000{ci}F000000-{uuids[ci]}-{succ}-{size}.npz'
        np.savez_compressed(os.path.join(replay_dir, name), **data)
        offset += size

  for label in ('a', 'b'):
    write_chunks(os.path.join(tmp, label))
  out = os.path.join(tmp, 'set')
  args = parse_args([
      '--replay', f'a={tmp}/a', f'b={tmp}/b', '--output', out,
      '--windows_per_source', '6', '--burn_in', '8', '--eval_steps', '4',
      '--seed', '0', '--task', 'dmc_cup_catch'])
  manifest = build(args)

  arrays, _ = load(out)
  L = manifest['length']
  # 1. No window straddles an episode boundary, and contiguity survived
  #    chunk chaining (position[:, 0] counts steps within each stream).
  assert not arrays['is_first'][:, 1:].any()
  pos = arrays['position'][..., 0]
  assert np.allclose(np.diff(pos, axis=1), 1.0), 'stream chaining broke order'
  # 2. Stratification and shapes.
  assert arrays['is_first'].shape == (12, L)
  assert sorted(set(arrays['source_label'])) == ['a', 'b']
  # 3. Determinism: same seed -> same content hash.
  out2 = os.path.join(tmp, 'set2')
  args2 = parse_args([
      '--replay', f'a={tmp}/a', f'b={tmp}/b', '--output', out2,
      '--windows_per_source', '6', '--burn_in', '8', '--eval_steps', '4',
      '--seed', '0', '--task', 'dmc_cup_catch'])
  manifest2 = build(args2)
  assert manifest2['sha256'] == manifest['sha256'], 'not deterministic'
  # 4. Frozen set refuses rebuild.
  try:
    build(args)
    raise AssertionError('rebuilt a FROZEN set')
  except SystemExit as e:
    assert 'FROZEN' in str(e)
  # 5. Tamper detection.
  npz = os.path.join(out, 'probeset.npz')
  with open(npz, 'ab') as f:
    f.write(b'x')
  try:
    load(out)
    raise AssertionError('tampered file passed verification')
  except SystemExit as e:
    assert 'sha256 mismatch' in str(e)
  print('probeset selfcheck: PASS')


def main():
  args = parse_args()
  if args.selfcheck:
    selfcheck()
  elif args.verify:
    assert args.output, '--verify needs --output'
    verify(args.output)
  else:
    assert args.output, 'Need --output'
    build(args)


if __name__ == '__main__':
  main()
