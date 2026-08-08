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

Stamp transforms (stamping plan, Plan_Stamping_20260717.md; these REPLACE
the reward values instead of permuting them -- the multiset is NOT
preserved, but the per-side label MARGINALS are matched exactly):

  stamp_rand frozen random-function labels: g = fixed random MLP
             (64x2 tanh, weights from --fn_seed) on the standardized
             flattened obs vector (pooled standardization over both
             sides).  Per side, the top-n frames by g are labeled, where
             n = that side's TRUE rewarded-frame count (density matched
             exactly), with amplitude = that side's mean positive true
             reward (scale matched).  Learnable, state-dependent,
             reward-UNALIGNED.
  stamp_iid  state-independent control: per side, n frames chosen
             uniformly at random (--seed; exact-count placement, i.e.
             hypergeometric rather than literal Bernoulli, so the
             marginal is matched exactly), same amplitude rule.
             Nothing state-dependent to learn.

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
    python -m probing.relabel_replay transform \
        --input $RUNROOT/axis1_finger/q1 --task dmc_finger_turn_hard \
        --kind stamp_rand --fn_seed 0 --seed 0 \
        --output $RUNROOT/axis1_finger/q1_srd0

Emit stamp labels for an E4 probe set (mechanism panel (a); feeds
``stratified_error measure --reward_override``)::

    python -m probing.relabel_replay stamp-probeset \
        --manifest $RUNROOT/axis1_finger/q1_srd0/manifest.json \
        --side side1 --probeset $RUNROOT/e4_probesets/finger_v1 \
        --output $RUNROOT/e4_probesets/finger_v1_srd0_s1.npz

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


def transform_reward(reward, in_regime, kind, rng, scale=None):
  """Return (new_reward, stats). shuffle/relocate preserve the value
  multiset; `scale` preserves the SUPPORT (which frames are rewarded)
  and multiplies values by a constant — the s-axis label-energy knob
  (2026-08-08; see PREREG_swave_theory_correction_20260808)."""
  reward = np.asarray(reward)
  new = np.zeros_like(reward)
  pos = np.flatnonzero(reward > 0)
  stats = dict(n_frames=len(reward), n_rewarded=int(len(pos)))
  if kind == 'scale':
    if scale is None or not (scale > 0):
      raise SystemExit('kind=scale requires --scale > 0')
    new = (reward.astype(np.float64) * float(scale)).astype(reward.dtype)
    stats['rewarded_in_regime'] = int(
        (new > 0)[np.asarray(in_regime, bool)].sum())
    stats['scale'] = float(scale)
    return new, stats
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


STAMP_EXCLUDE = ('reward', 'action', 'stepid', 'is_first', 'is_last',
                 'is_terminal', 'consec')
STAMP_WIDTH = 64


def stamp_obs_keys(frames):
  """Float-valued observation keys of a frame dict, sorted (frozen order).

  Namespaced replay extras (dyn/*, log/*) are NOT observations and are
  excluded. The 17-Jul srd0/srd1 buffers predate this rule (their g
  consumed the stored dyn/* context latents; the manifests record the
  keys actually used, so manifest-driven reproduction is unaffected —
  see the 18-Jul DEVIATIONS entry and stamp-probeset --latent_replay)."""
  keys = sorted(
      k for k, v in frames.items()
      if k not in STAMP_EXCLUDE and '/' not in k and
      np.issubdtype(np.asarray(v).dtype, np.floating))
  if not keys:
    raise SystemExit('stamp: no float observation keys found')
  return keys


def recover_probe_latents(stepids, keys, replay_dirs):
  """Per-frame namespaced extras (dyn/*) for probe frames via stepid join.

  Probe sets preserve each frame's ORIGINAL 20-byte stepid, so the frame
  can be joined back to its source chunk row exactly. Rows are matched on
  the full 20-byte stepid (encoding-agnostic: no assumption about the
  uuid/index layout). Returns {key: (n_frames, ...) array}."""
  import glob as _glob
  flat = np.asarray(stepids, np.uint8).reshape(-1, 20)
  chunk_paths = []
  for d in replay_dirs:
    chunk_paths += sorted(_glob.glob(os.path.join(d, '*.npz')))
  if not chunk_paths:
    raise SystemExit(f'--latent_replay: no chunks under {replay_dirs}')
  row_of, cache = {}, {}
  for p in chunk_paths:
    with np.load(p) as z:
      sids = np.asarray(z['stepid'], np.uint8)
      arrs = {k: np.asarray(z[k]) for k in keys if k in z.files}
    if len(arrs) != len(keys):
      missing = sorted(set(keys) - set(arrs))
      raise SystemExit(f'{p}: chunk lacks {missing}')
    cache[p] = arrs
    for j, s in enumerate(sids):
      row_of[bytes(s)] = (p, j)
  out = {k: [] for k in keys}
  for s in flat:
    hit = row_of.get(bytes(s))
    if hit is None:
      raise SystemExit(
          'stepid join failed: a probe frame is not present in the given '
          '--latent_replay dirs (pass the SAME pilot replays the probe '
          'set was built from)')
    p, j = hit
    for k in keys:
      out[k].append(cache[p][k][j])
  return {k: np.stack(v) for k, v in out.items()}


def stamp_matrix(episodes, obs_keys):
  """List of per-episode frame dicts -> (n_frames, d) float64 matrix."""
  rows = []
  for frames in episodes:
    n = len(np.asarray(frames['reward']))
    rows.append(np.concatenate(
        [np.asarray(frames[k], np.float64).reshape(n, -1) for k in obs_keys],
        axis=1))
  return np.concatenate(rows, 0)


def stamp_mlp(fn_seed, dim, width=STAMP_WIDTH):
  """Frozen random MLP g: R^d -> R (weights deterministic in fn_seed, d)."""
  rng = np.random.default_rng(fn_seed)
  w1 = rng.normal(0.0, 1.0 / np.sqrt(dim), (dim, width))
  w2 = rng.normal(0.0, 1.0 / np.sqrt(width), (width, width))
  w3 = rng.normal(0.0, 1.0 / np.sqrt(width), (width, 1))
  def g(x):
    h = np.tanh(x @ w1)
    h = np.tanh(h @ w2)
    return (h @ w3)[:, 0]
  return g


def stamp_select(kind, gvals, n_frames, n_pos, seed, side_index):
  """Flat frame indices to stamp for one side.

  stamp_rand: top-n_pos frames by gvals (exact-count quantile threshold);
  stamp_iid:  n_pos frames uniform without replacement (state-independent).
  Returns (indices, threshold-or-None).
  """
  if kind == 'stamp_rand':
    order = np.argsort(-gvals, kind='stable')
    idx = order[:n_pos]
    return idx, float(gvals[idx].min()) if n_pos else None
  if kind == 'stamp_iid':
    rng = np.random.default_rng([seed, side_index])
    return rng.choice(n_frames, size=n_pos, replace=False), None
  raise ValueError(f'unknown stamp kind {kind!r}')


def cmd_transform_stamp(args, src_manifest, sides, spec):
  """Stamp path of cmd_transform: pair-level pass (pooled standardization,
  per-side marginal matching), then per-episode chunk writes."""
  episodes = {s: load_episode_chunks(os.path.join(args.input, s))
              for s in sides}
  obs_keys = stamp_obs_keys(episodes[sides[0]][0])
  mats = {s: stamp_matrix(episodes[s], obs_keys) for s in sides}
  pooled = np.concatenate([mats[s] for s in sides], 0)
  mean = pooled.mean(0)
  std = np.maximum(pooled.std(0), 1e-6)
  dim = pooled.shape[1]
  g = stamp_mlp(args.fn_seed, dim) if args.kind == 'stamp_rand' else None

  side_stats = {}
  for side_index, side in enumerate(sides):
    eps = episodes[side]
    rewards = np.concatenate(
        [np.asarray(f['reward'], np.float64) for f in eps])
    n_frames = len(rewards)
    pos = rewards > 0
    n_pos = int(pos.sum())
    if n_pos == 0:
      raise SystemExit(
          f'{side}: no positive true-reward frames; amplitude/density '
          f'matching impossible (registered fallback must be decided at '
          f'the prereg, not silently applied here).')
    amplitude = float(rewards[pos].mean())
    gvals = g((mats[side] - mean) / std) if g is not None else None
    idx, threshold = stamp_select(
        args.kind, gvals, n_frames, n_pos, args.seed, side_index)
    new_flat = np.zeros(n_frames, np.float64)
    new_flat[idx] = amplitude

    out_dir = os.path.join(args.output, side)
    os.makedirs(out_dir, exist_ok=True)
    offset, stamped_in_regime, occs, weights = 0, 0, [], []
    for frames in eps:
      n = len(np.asarray(frames['reward']))
      new_reward = new_flat[offset:offset + n]
      offset += n
      vals = spec['fn'](frames)
      below = spec['direction'] == 'below'
      in_r = (vals < spec['threshold']) if below else (vals > spec['threshold'])
      stamped_in_regime += int((new_reward > 0)[np.asarray(in_r, bool)].sum())
      occs.append(float(np.asarray(in_r, bool).mean()))
      weights.append(n)
      out_frames = dict(frames)
      out_frames['reward'] = new_reward.astype(
          np.asarray(frames['reward']).dtype)
      write_episode_chunk(out_dir, out_frames)
    assert offset == n_frames
    side_stats[side] = dict(
        n_episodes=len(eps), n_frames=n_frames,
        true_reward_frames=n_pos, n_stamped=n_pos,
        # Full precision: stamp_labels_from_manifest must reproduce the
        # written labels bit-for-bit (threshold ties / float32 casts).
        amplitude=amplitude,
        threshold=(float(threshold) if threshold is not None else None),
        stamped_in_regime=stamped_in_regime,
        true_rewarded_in_regime=int(sum(
            (np.asarray(f['reward']) > 0)[
                np.asarray(regimes.in_regime(args.task, f), bool)].sum()
            for f in eps)),
        occ_frame=round(float(np.average(occs, weights=weights)), 6))
    print(f'{side}: {side_stats[side]}')

  manifest = dict(src_manifest)
  manifest['reward_transform'] = dict(
      kind=args.kind, seed=args.seed,
      fn_seed=(args.fn_seed if args.kind == 'stamp_rand' else None),
      task=args.task,
      source_pair=os.path.abspath(args.input),
      regime=dict(name=spec['name'], threshold=float(spec['threshold']),
                  direction=spec['direction']),
      stamp=dict(obs_keys=obs_keys, dim=dim, width=STAMP_WIDTH,
                 # full precision (reproducibility contract, see above)
                 standardize_mean=[float(v) for v in mean],
                 standardize_std=[float(v) for v in std]),
      sides=side_stats,
      note='reward REPLACED by stamp labels (multiset NOT preserved; '
           'per-side density and amplitude matched to true labels '
           'exactly); all other keys byte-preserved, stepid re-encoded '
           'for the fresh chunk uuids')
  with open(os.path.join(args.output, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
  print(f'-> {args.output} ({args.kind}, seed {args.seed}'
        + (f', fn_seed {args.fn_seed}' if args.kind == 'stamp_rand' else '')
        + ')')


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
  if args.kind.startswith('stamp'):
    return cmd_transform_stamp(args, src_manifest, sides, spec)
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
          frames['reward'], in_r, args.kind, rng,
          scale=getattr(args, 'scale', None))
      if args.kind == 'scale':
        orig = np.asarray(frames['reward'], np.float64)
        assert np.array_equal(new_reward > 0, orig > 0), \
            'scale changed the reward support'
        assert np.allclose(np.asarray(new_reward, np.float64),
                           orig * args.scale), 'scale values wrong'
      else:
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
      scale=(float(args.scale) if args.kind == 'scale' else None),
      source_pair=os.path.abspath(args.input),
      regime=dict(name=spec['name'], threshold=float(spec['threshold']),
                  direction=spec['direction']),
      sides=side_stats,
      note='only the reward array differs from the source pair; all other '
           'keys byte-preserved, stepid re-encoded for the fresh chunk uuids')
  with open(os.path.join(args.output, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
  print(f'-> {args.output} ({args.kind}, seed {args.seed})')


def stamp_labels_from_manifest(tr, matrix):
  """Recompute stamp_rand labels for arbitrary frames.

  tr: the `reward_transform` manifest block of a stamp_rand pair;
  matrix: (n, d) raw obs matrix with the SAME obs_keys/column order.
  Returns {side: (n,) float32 labels} using each side's recorded
  threshold (inclusive) and amplitude.
  """
  if tr['kind'] != 'stamp_rand':
    raise SystemExit(f'stamp-probeset needs a stamp_rand manifest, '
                     f'got kind={tr["kind"]!r}')
  stamp = tr['stamp']
  if matrix.shape[1] != stamp['dim']:
    raise SystemExit(f'obs matrix dim {matrix.shape[1]} != recorded '
                     f'{stamp["dim"]} (obs_keys mismatch?)')
  mean = np.asarray(stamp['standardize_mean'], np.float64)
  std = np.asarray(stamp['standardize_std'], np.float64)
  g = stamp_mlp(tr['fn_seed'], stamp['dim'], stamp['width'])
  gvals = g((matrix - mean) / std)
  out = {}
  for side, s in tr['sides'].items():
    labels = np.where(gvals >= s['threshold'], s['amplitude'], 0.0)
    out[side] = labels.astype(np.float32)
  return out


def cmd_stamp_probeset(args):
  """Evaluate a frozen stamp function on an E4 probe set -> override npz
  for `stratified_error measure --reward_override` (mechanism panel (a):
  reward-head NLL against the labels the fit was actually trained on)."""
  with open(args.manifest) as f:
    tr = json.load(f)['reward_transform']
  npz_path = os.path.join(args.probeset, 'probeset_e4.npz')
  with open(os.path.join(args.probeset, 'manifest.json')) as f:
    ps_manifest = json.load(f)
  arrays = {k: np.asarray(v) for k, v in np.load(npz_path).items()}
  N, T = arrays['reward'].shape
  obs_keys = tr['stamp']['obs_keys']
  missing = [k for k in obs_keys if k not in arrays]
  latent_recovery = None
  if missing:
    if getattr(args, 'latent_replay', None) and all(
        '/' in k for k in missing):
      rec = recover_probe_latents(arrays['stepid'], missing,
                                  args.latent_replay)
      arrays.update(rec)  # flat (N*T, ...) — reshape below handles both
      latent_recovery = dict(keys=missing,
                             replay_dirs=[os.path.abspath(d)
                                          for d in args.latent_replay])
    else:
      raise SystemExit(
          f'probe set lacks stamp obs keys {missing}; for dyn/* keys '
          '(manifests from the 17-Jul wave, whose g consumed the stored '
          'context latents) pass --latent_replay <the pilot replay dirs '
          'the probe set was built from> to recover them by stepid join')
  matrix = np.concatenate(
      [np.asarray(arrays[k], np.float64).reshape(N * T, -1)
       for k in obs_keys], axis=1)
  labels = stamp_labels_from_manifest(tr, matrix)
  if args.side not in labels:
    raise SystemExit(f'side {args.side!r} not in manifest '
                     f'(have {sorted(labels)})')
  reward = labels[args.side].reshape(N, T)
  meta = dict(
      kind='stamp_rand', side=args.side, fn_seed=tr['fn_seed'],
      manifest=os.path.abspath(args.manifest),
      probeset_id=ps_manifest['probeset_id'],
      probeset_sha256=ps_manifest['sha256'],
      density=round(float((reward > 0).mean()), 6),
      amplitude=tr['sides'][args.side]['amplitude'],
      latent_recovery=latent_recovery)
  np.savez_compressed(args.output, reward=reward)
  with open(args.output + '.json', 'w') as f:
    json.dump(meta, f, indent=2)
  print(f'{args.output}: stamp labels {N}x{T}, density {meta["density"]} '
        f'(buffer-side occ-matched density '
        f'{tr["sides"][args.side]["n_stamped"] / tr["sides"][args.side]["n_frames"]:.6f})')


def cmd_transform_probeset(args):
  """Apply a LABEL transform (shuffle/relocate/scale) to a frozen E4 probe
  set -> override npz for `stratified_error measure --reward_override`.

  Own-label panels (2026-08-08, review D6/#11 + the s-wave scoring trap):
  a fit trained on transformed labels must be scored against labels drawn
  by the SAME process, or its NLL conflates mislocation with miscalibration.
  Per-episode transform with the stored in_regime mask; rng(seed) advances
  in row order, so the override is deterministic per (kind, seed)."""
  npz_path = os.path.join(args.probeset, 'probeset_e4.npz')
  with open(os.path.join(args.probeset, 'manifest.json')) as f:
    ps_manifest = json.load(f)
  # batch-review m17: verify the BYTES against the manifest sha before
  # transforming (same convention as stratified_error.load_e4).
  import hashlib
  digest = hashlib.sha256(open(npz_path, 'rb').read()).hexdigest()
  if digest != ps_manifest['sha256']:
    raise SystemExit(f'{npz_path}: sha256 mismatch vs manifest')
  arrays = np.load(npz_path)
  reward = np.asarray(arrays['reward'])
  in_regime = np.asarray(arrays['in_regime'], bool)
  N, T = reward.shape
  rng = np.random.default_rng(args.seed)
  new = np.zeros_like(reward)
  n_in = 0
  for i in range(N):
    new[i], stats = transform_reward(
        reward[i], in_regime[i], args.kind, rng, scale=args.scale)
    n_in += stats['rewarded_in_regime']
  meta = dict(
      kind=args.kind, seed=args.seed,
      scale=(float(args.scale) if args.kind == 'scale' else None),
      probeset_id=ps_manifest['probeset_id'],
      probeset_sha256=ps_manifest['sha256'],
      density=round(float((new > 0).mean()), 6),
      rewarded_in_regime=int(n_in),
      n_episodes=int(N), length=int(T))
  np.savez_compressed(args.output, reward=new)
  with open(args.output + '.json', 'w') as f:
    json.dump(meta, f, indent=2)
  print(f'{args.output}: {args.kind} override {N}x{T}, '
        f'density {meta["density"]}, rewarded_in_regime {n_in}')


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

    # --- scale kind (2026-08-08) ------------------------------------------
    out_pair = os.path.join(tmp, 'q1_sc')
    cmd_transform(argparse.Namespace(
        input=src, output=out_pair, seed=0, task='dmc_cup_catch',
        kind='scale', scale=2.45))
    for side in ('side0', 'side1'):
      orig = load_episode_chunks(os.path.join(src, side))
      new = load_episode_chunks(os.path.join(out_pair, side))
      for o, t in zip(orig, new):
        ov = np.asarray(o['reward'], np.float64)
        tv = np.asarray(t['reward'], np.float64)
        assert np.array_equal(ov > 0, tv > 0), 'scale moved the support'
        assert np.allclose(tv, ov * 2.45), 'scale factor wrong'
        for key in ('position', 'velocity', 'action', 'is_first'):
          assert np.array_equal(o[key], t[key]), key
    with open(os.path.join(out_pair, 'manifest.json')) as f:
      assert json.load(f)['reward_transform']['scale'] == 2.45

    # --- transform-probeset (2026-08-08) ----------------------------------
    ps_dir = os.path.join(tmp, 'ps')
    os.makedirs(ps_dir)
    N_, T_ = 6, 50
    ps_rew = (rng.random((N_, T_)) < 0.1).astype(np.float32)
    ps_inr = rng.random((N_, T_)) < 0.3
    np.savez_compressed(os.path.join(ps_dir, 'probeset_e4.npz'),
                        reward=ps_rew, in_regime=ps_inr)
    import hashlib
    sha = hashlib.sha256(
        open(os.path.join(ps_dir, 'probeset_e4.npz'), 'rb').read()).hexdigest()
    with open(os.path.join(ps_dir, 'manifest.json'), 'w') as f:
      json.dump(dict(probeset_id='selfcheck_ps', sha256=sha), f)
    for kind, kw in (('relocate', {}), ('scale', dict(scale=3.0)),
                     ('shuffle', {})):
      outp = os.path.join(tmp, f'ovr_{kind}.npz')
      cmd_transform_probeset(argparse.Namespace(
          probeset=ps_dir, kind=kind, seed=1,
          scale=kw.get('scale'), output=outp))
      ov = np.asarray(np.load(outp)['reward'])
      if kind == 'scale':
        assert np.allclose(ov, ps_rew * 3.0)
      else:
        for i in range(N_):
          assert sorted(ov[i].tolist()) == sorted(ps_rew[i].tolist())
      if kind == 'relocate':
        # relocated mass out of regime up to overflow
        spill = int(json.load(open(outp + '.json'))['rewarded_in_regime'])
        cap = sum(max(0, int((ps_rew[i] > 0).sum()) - int((~ps_inr[i]).sum()))
                  for i in range(N_))
        assert spill <= cap, 'relocate left avoidable mass in regime'
      # determinism
      outp2 = os.path.join(tmp, f'ovr_{kind}_b.npz')
      cmd_transform_probeset(argparse.Namespace(
          probeset=ps_dir, kind=kind, seed=1,
          scale=kw.get('scale'), output=outp2))
      assert np.array_equal(np.load(outp)['reward'],
                            np.load(outp2)['reward']), 'nondeterministic'

    # --- stamp kinds -------------------------------------------------------
    for kind, tag in (('stamp_rand', 'srd0'), ('stamp_iid', 'sid')):
      out_pair = os.path.join(tmp, f'q1_{tag}')
      cmd_transform(argparse.Namespace(
          input=src, output=out_pair, seed=0, fn_seed=0,
          task='dmc_cup_catch', kind=kind))
      with open(os.path.join(out_pair, 'manifest.json')) as f:
        tr = json.load(f)['reward_transform']
      for side in ('side0', 'side1'):
        orig = load_episode_chunks(os.path.join(src, side))
        new = load_episode_chunks(os.path.join(out_pair, side))
        n_true = sum(int((np.asarray(o['reward']) > 0).sum()) for o in orig)
        n_stamp = sum(int((np.asarray(t['reward']) > 0).sum()) for t in new)
        # Exact per-side density match; amplitude = mean positive reward.
        assert n_stamp == n_true == tr['sides'][side]['n_stamped']
        amp = tr['sides'][side]['amplitude']
        for o, t in zip(orig, new):
          for key in ('position', 'velocity', 'action', 'is_first'):
            assert np.array_equal(o[key], t[key]), key
          assert not np.array_equal(o['stepid'], t['stepid'])
          vals = t['reward'][t['reward'] > 0]
          assert np.allclose(vals, amp), (kind, side)
    # Determinism: identical seeds reproduce identical rewards.
    cmd_transform(argparse.Namespace(
        input=src, output=os.path.join(tmp, 'q1_srd0b'), seed=0, fn_seed=0,
        task='dmc_cup_catch', kind='stamp_rand'))
    a = load_episode_chunks(os.path.join(tmp, 'q1_srd0', 'side0'))
    bb = load_episode_chunks(os.path.join(tmp, 'q1_srd0b', 'side0'))
    assert all(np.array_equal(x['reward'], y['reward'])
               for x, y in zip(a, bb))
    # A second function draw places labels differently.
    cmd_transform(argparse.Namespace(
        input=src, output=os.path.join(tmp, 'q1_srd1'), seed=0, fn_seed=1,
        task='dmc_cup_catch', kind='stamp_rand'))
    c = load_episode_chunks(os.path.join(tmp, 'q1_srd1', 'side0'))
    assert any(not np.array_equal(x['reward'], y['reward'])
               for x, y in zip(a, c)), 'srd0 == srd1 placement'
    # Manifest reproducibility (the stamp-probeset path): recomputing the
    # labels from the recorded standardization/threshold/amplitude matches
    # the written buffers bit-for-bit.
    with open(os.path.join(tmp, 'q1_srd0', 'manifest.json')) as f:
      tr = json.load(f)['reward_transform']
    for side in ('side0', 'side1'):
      eps = load_episode_chunks(os.path.join(src, side))
      matrix = stamp_matrix(eps, tr['stamp']['obs_keys'])
      labels = stamp_labels_from_manifest(tr, matrix)[side]
      written = np.concatenate(
          [t['reward'] for t in
           load_episode_chunks(os.path.join(tmp, 'q1_srd0', side))])
      assert np.array_equal(labels, written.astype(np.float32)), side

    # --- namespaced-extras rule + stepid-join recovery ---------------------
    # (a) dyn/* never enters NEW stamp functions (18-Jul rule).
    lat_src = os.path.join(tmp, 'q1lat')
    for side in ('side0', 'side1'):
      out = os.path.join(lat_src, side)
      os.makedirs(out)
      for _ in range(2):
        n = 40
        frames = dict(
            position=rng.normal(0, 1, (n, 4)).astype(np.float32),
            velocity=rng.normal(0, 1, (n, 4)).astype(np.float32),
            reward=(rng.random(n) < 0.2).astype(np.float32),
            is_first=np.eye(1, n, 0, dtype=bool)[0],
            stepid=np.zeros((n, 20), np.uint8))
        frames['dyn/deter'] = rng.normal(0, 1, (n, 8)).astype(np.float32)
        write_episode_chunk(out, frames)
    with open(os.path.join(lat_src, 'manifest.json'), 'w') as f:
      json.dump(dict(task='dmc_cup_catch', sides=[]), f)
    probe_eps = load_episode_chunks(os.path.join(lat_src, 'side0'))
    assert 'dyn/deter' not in stamp_obs_keys(probe_eps[0]), \
        'namespaced extras leaked into stamp_obs_keys'
    cmd_transform(argparse.Namespace(
        input=lat_src, output=os.path.join(tmp, 'q1lat_srd0'), seed=0,
        fn_seed=0, task='dmc_cup_catch', kind='stamp_rand'))
    with open(os.path.join(tmp, 'q1lat_srd0', 'manifest.json')) as f:
      keys_new = json.load(f)['reward_transform']['stamp']['obs_keys']
    assert all('/' not in k for k in keys_new), keys_new
    # (b) recover_probe_latents: full-20-byte stepid join reproduces the
    # chunk rows exactly (the 17-Jul-manifest stamp-probeset path).
    sel = [(0, 5), (0, 31), (1, 12)]
    sids = np.stack([probe_eps[e]['stepid'][t] for e, t in sel])
    rec = recover_probe_latents(sids, ['dyn/deter'],
                                [os.path.join(lat_src, 'side0')])
    want = np.stack([probe_eps[e]['dyn/deter'][t] for e, t in sel])
    assert np.array_equal(rec['dyn/deter'], want)
    try:
      recover_probe_latents(np.full((1, 20), 255, np.uint8),
                            ['dyn/deter'], [os.path.join(lat_src, 'side0')])
      raise AssertionError('stepid join accepted an unknown frame')
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
  t.add_argument('--kind', required=True,
                 choices=['shuffle', 'relocate', 'scale',
                          'stamp_rand', 'stamp_iid'])
  t.add_argument('--seed', type=int, default=0)
  t.add_argument('--scale', type=float, default=None,
                 help='kind=scale only: multiply reward values by this '
                      'constant (support preserved).')
  t.add_argument('--fn_seed', type=int, default=0,
                 help='stamp_rand only: seed of the frozen random MLP '
                      '(srd0 -> 0, srd1 -> 1).')
  t.set_defaults(fn=cmd_transform)

  sp = sub.add_parser('stamp-probeset')
  sp.add_argument('--manifest', required=True,
                  help='manifest.json of a stamp_rand transformed pair.')
  sp.add_argument('--side', required=True, choices=['side0', 'side1'])
  sp.add_argument('--probeset', required=True,
                  help='E4 probe set dir (probeset_e4.npz + manifest.json).')
  sp.add_argument('--output', required=True, help='Output .npz path.')
  sp.add_argument('--latent_replay', nargs='*', default=None,
                  help='Pilot replay chunk dirs for stepid-join recovery '
                       'of dyn/* stamp inputs (17-Jul manifests).')
  sp.set_defaults(fn=cmd_stamp_probeset)

  tp = sub.add_parser('transform-probeset')
  tp.add_argument('--probeset', required=True,
                  help='E4 probe set dir (probeset_e4.npz + manifest.json).')
  tp.add_argument('--kind', required=True,
                  choices=['shuffle', 'relocate', 'scale'])
  tp.add_argument('--seed', type=int, default=0)
  tp.add_argument('--scale', type=float, default=None)
  tp.add_argument('--output', required=True, help='Output .npz path.')
  tp.set_defaults(fn=cmd_transform_probeset)

  sc = sub.add_parser('selfcheck')
  sc.set_defaults(fn=cmd_selfcheck)

  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
