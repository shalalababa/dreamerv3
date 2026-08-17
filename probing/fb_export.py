"""Export a curated proprio side buffer to FB (controllable_agent) format.

FB wave (FB_Wave_Plan_20260816; prereg PREREG_fb_20260817). Reads the
materialized side dir of a curated buffer (one episode per npz chunk —
the build_controlled_replay materialization contract) and writes a
single npz with episode-major arrays plus a manifest:

  obs    (E, T, 12) f32 — pinned key order OBS_KEYS (concat)
  action (E, T, A)  f32
  reward (E, T)     f32 — the env task reward stored at collection time
                          (reward-free collection still records it)

dv3 chunks store the POST-action convention (row t = obs_t paired with
the action chosen AT obs_t); controllable_agent stores the PREV action
(row t = obs_t with the action that LED here, zeros at reset). The
exporter applies the same post->prev shift the in-repo precedent uses
(probing/tdmpc2_bridge.py): action'[t] = action[t-1], action'[0] = 0
('prev_shifted_v1', recorded in the manifest and gated by the reader).
Row 0 is the reset row (reward must be 0, asserted); the arrays map
onto controllable_agent's (T+1)-row episode storage, episode length
T-1. Chunks must satisfy the built-buffer contract (one episode per
chunk, successor uuid = 0 in the filename — refused otherwise).

Usage:
  python -m probing.fb_export export --side_dir <.../q1/side0> \
      --output $RUNROOT/fb_data/finger_q1_side0.npz
Selfcheck: python -m probing.fb_export --selfcheck   (numpy-only)
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np

ZERO_UUID = '0' * 22  # matches probing/probeset.py / build_controlled_replay
ACTION_CONVENTION = 'prev_shifted_v1'
OBS_KEYS = ('position', 'velocity', 'touch', 'target_position',
            'dist_to_target')
OBS_DIMS = {'position': 4, 'velocity': 3, 'touch': 2,
            'target_position': 2, 'dist_to_target': 1}
OBS_DIM = sum(OBS_DIMS.values())  # 12


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def load_episode(path):
  succ = os.path.basename(path)[:-len('.npz')].split('-')[2]
  assert succ == ZERO_UUID, \
      (path, 'chunk has a successor - built-buffer contract violated '
       '(one episode per chunk, succ=0)')
  with np.load(path) as z:
    for k in OBS_KEYS + ('action', 'reward', 'is_first'):
      assert k in z, (path, 'missing key', k)
    parts = []
    for k in OBS_KEYS:
      a = np.asarray(z[k], np.float32)
      if a.ndim == 1:
        a = a[:, None]
      assert a.shape[1] == OBS_DIMS[k], (path, k, a.shape,
                                         'dim != pinned OBS_DIMS')
      parts.append(a)
    obs = np.concatenate(parts, axis=1)
    act = np.asarray(z['action'], np.float32)
    rew = np.asarray(z['reward'], np.float32).reshape(-1)
    isf = np.asarray(z['is_first'], bool).reshape(-1)
  assert obs.shape[1] == OBS_DIM, (path, obs.shape)
  assert isf[0] and not isf[1:].any(), \
      (path, 'not a single-episode chunk (is_first pattern)')
  assert len(obs) == len(act) == len(rew), (path, len(obs), len(act),
                                            len(rew))
  assert np.isfinite(obs).all() and np.isfinite(rew).all(), \
      (path, 'non-finite values')
  assert abs(float(rew[0])) < 1e-6, (path, 'nonzero reward at reset row',
                                     float(rew[0]))
  shifted = np.zeros_like(act)
  shifted[1:] = act[:-1]  # dv3 post-action -> prev-action (finding B1)
  return obs, shifted, rew


def run(args):
  assert 'heldout' not in os.path.abspath(args.side_dir), \
      (args.side_dir, 'refusing to export a heldout dir (probe-leak guard)')
  files = sorted(glob.glob(os.path.join(args.side_dir, '*.npz')))
  assert files, f'no npz chunks under {args.side_dir}'
  eps = [load_episode(p) for p in files]
  lengths = {e[0].shape[0] for e in eps}
  assert len(lengths) == 1, ('mixed episode lengths', sorted(lengths))
  T = lengths.pop()
  obs = np.stack([e[0] for e in eps])
  act = np.stack([e[1] for e in eps])
  rew = np.stack([e[2] for e in eps])
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  np.savez_compressed(args.output, obs=obs, action=act, reward=rew)
  manifest = dict(
      tool='fb_export_v1', side_dir=os.path.abspath(args.side_dir),
      n_episodes=int(obs.shape[0]), ep_len=int(T),
      obs_keys=list(OBS_KEYS), obs_dims={k: OBS_DIMS[k] for k in OBS_KEYS},
      obs_dim=OBS_DIM, action_dim=int(act.shape[2]),
      action_convention=ACTION_CONVENTION,
      heldout_guard=('heldout' not in os.path.abspath(args.side_dir)),
      reward_stats=dict(mean=float(rew.mean()), max=float(rew.max()),
                        frac_pos=float((rew > 0).mean())),
      sources=[dict(file=os.path.basename(p), sha256=sha256_file(p))
               for p in files],
      output_sha256=sha256_file(args.output))
  mpath = args.output + '.manifest.json'
  with open(mpath, 'w') as f:
    json.dump(manifest, f, indent=1)
  print(f'{obs.shape[0]} eps x {T} steps -> {args.output}')
  print(f'manifest -> {mpath}')


def selfcheck():
  import tempfile
  base = tempfile.mkdtemp(prefix='fb_export_sc_')
  side = os.path.join(base, 'side0')
  os.makedirs(side)
  rng = np.random.default_rng(0)
  T = 11
  for i in range(3):
    arrs = {k: rng.standard_normal((T, OBS_DIMS[k])).astype(np.float32)
            for k in OBS_KEYS}
    arrs['dist_to_target'] = arrs['dist_to_target'].reshape(T)
    isf = np.zeros(T, bool); isf[0] = True
    rew = (rng.random(T) > .9).astype(np.float32)
    rew[0] = 0.0
    act = rng.standard_normal((T, 2)).astype(np.float32)
    if i == 0:
      act0 = act.copy()
    np.savez(os.path.join(side, f'x-id{i}-{"0"*22}-{T}.npz'), action=act,
             reward=rew, is_first=isf, **arrs)
  out = os.path.join(base, 'out.npz')
  run(argparse.Namespace(side_dir=side, output=out))
  z = np.load(out)
  assert z['obs'].shape == (3, T, OBS_DIM) and z['action'].shape == (3, T, 2)
  assert (z['action'][0][0] == 0).all() and np.allclose(
      z['action'][0][1:], act0[:-1]), 'prev-action shift not applied'
  man = json.load(open(out + '.manifest.json'))
  assert man['obs_keys'] == list(OBS_KEYS) and man['n_episodes'] == 3
  assert man['action_convention'] == ACTION_CONVENTION
  # refusal legs: missing key; multi-episode chunk; wrong dim
  fails = 0
  for mut in ('missing', 'multifirst', 'wrongdim', 'resetrew', 'successor'):
    s2 = os.path.join(base, mut); os.makedirs(s2)
    arrs = {k: rng.standard_normal((T, OBS_DIMS[k])).astype(np.float32)
            for k in OBS_KEYS}
    isf = np.zeros(T, bool); isf[0] = True
    if mut == 'missing':
      arrs.pop('touch')
    if mut == 'multifirst':
      isf[5] = True
    if mut == 'wrongdim':
      arrs['velocity'] = arrs['velocity'][:, :2]
    rew2 = np.zeros(T, np.float32)
    if mut == 'resetrew':
      rew2[0] = 1.0
    succ = 'a' * 22 if mut == 'successor' else '0' * 22
    np.savez(os.path.join(s2, f'x-id0-{succ}-{T}.npz'),
             action=np.zeros((T, 2), np.float32), reward=rew2,
             is_first=isf, **arrs)
    try:
      run(argparse.Namespace(side_dir=s2, output=os.path.join(base,
                                                              mut + '.npz')))
    except AssertionError:
      fails += 1
  assert fails == 5, ('refusal legs', fails)
  print('fb_export selfcheck PASS (roundtrip + prev-shift + manifest + 5 refusals)')


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd')
  ex = sub.add_parser('export')
  ex.add_argument('--side_dir', required=True)
  ex.add_argument('--output', required=True)
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
  else:
    assert args.cmd == 'export'
    run(args)


if __name__ == '__main__':
  main()
