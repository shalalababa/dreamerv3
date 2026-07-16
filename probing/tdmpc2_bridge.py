"""Bridge a built DreamerV3 buffer side into a TD-MPC2 offline dataset.

Reads a built pair side dir (one .npz chunk per episode, succ=0 — the
built-buffer contract) and writes a single ``.pt`` file holding a
TensorDict of shape [n_episodes, ep_len] with keys:

  obs        [N, T, D]  canonical concat (tdmpc2_compat.OBS_ORDER)
  action     [N, T, A]  PREV-action convention (row t = action that LED
                        to obs_t; row 0 = zeros) — TD-MPC2's storage
                        convention, i.e. DreamerV3 rows shifted by one.
  reward     [N, T]     reward received on arrival at obs_t (row 0 = 0)
  terminated [N, T]     zeros (DMC study tasks are non-episodic)

This is exactly what ``common.buffer.Buffer.load`` expects. Convention
check: TD-MPC2's ``_prepare_batch`` uses obs[:] with action[1:]/
reward[1:], so slice t pairs obs_t with the action taken AT obs_t via the
t+1 row — the shift below reproduces the pairing its online trainer
produces (``to_td(obs, action, reward)`` appends the action WITH the obs
it produced).

Usage (cluster, tdmpc2 conda env):
  python -m probing.tdmpc2_bridge convert \
      --side_dir $RUNROOT/axis1_finger/q1/side1 \
      --task dmc_finger_turn_hard --output $RUNROOT/tm2_data/finger_q1_side1.pt
  python -m probing.tdmpc2_bridge selfcheck
"""

import argparse
import hashlib
import json
import os
import pathlib

import numpy as np

from probing.tdmpc2_compat import episode_obs_matrix

ZERO_UUID = '0' * 22  # build_controlled_replay.ZERO_UUID (kept standalone:
                      # that module pulls elements/embodied into the import
                      # chain, which the torch env does not carry)


def load_episode_chunks(side_dir):
  """Built side dir -> per-episode frame dicts (built-buffer contract:
  one chunk per episode, successor uuid = zero)."""
  from probing.replay_dataset import list_chunks  # numpy-only module
  paths = list_chunks(side_dir)
  if not paths:
    raise SystemExit(f'No .npz chunks in {side_dir}')
  episodes = []
  for path in sorted(paths):
    succ = pathlib.Path(path).stem.split('-')[2]
    if succ != ZERO_UUID:
      raise SystemExit(
          f'{path}: chunk has a successor; this tool requires the built-'
          f'buffer contract (one chunk per episode, succ=0).')
    with np.load(path) as data:
      episodes.append({k: np.asarray(data[k]) for k in data.files})
  return episodes


def episodes_to_arrays(episodes, task):
  """Raw dv3 episode dicts -> (obs [N,T,D], action [N,T,A], reward [N,T]).

  Applies the post->prev action shift and validates the dv3 episode
  contract (is_first at row 0 only, zero reward at row 0, uniform T).
  """
  lengths = {len(ep['reward']) for ep in episodes}
  if len(lengths) != 1:
    raise SystemExit(f'non-uniform episode lengths {sorted(lengths)}; '
                     'built buffers are modal-length by contract')
  obs_l, act_l, rew_l = [], [], []
  for i, ep in enumerate(episodes):
    is_first = np.asarray(ep['is_first'], bool)
    if not is_first[0] or is_first[1:].any():
      raise SystemExit(f'episode {i}: is_first not exactly at row 0')
    rew = np.asarray(ep['reward'], np.float32)
    if abs(float(rew[0])) > 1e-6:
      raise SystemExit(f'episode {i}: nonzero reward {rew[0]} at is_first row')
    act = np.asarray(ep['action'], np.float32)
    shifted = np.zeros_like(act)
    shifted[1:] = act[:-1]  # prev-action convention
    obs_l.append(episode_obs_matrix(ep, task))
    act_l.append(shifted)
    rew_l.append(rew)
  return (np.stack(obs_l), np.stack(act_l), np.stack(rew_l))


def convert(side_dir, task, output):
  import torch
  from tensordict.tensordict import TensorDict

  episodes = load_episode_chunks(side_dir)
  obs, act, rew = episodes_to_arrays(episodes, task)
  n, t = rew.shape
  td = TensorDict(dict(
      obs=torch.from_numpy(obs),
      action=torch.from_numpy(act),
      reward=torch.from_numpy(rew),
      terminated=torch.zeros(n, t, dtype=torch.float32),
  ), batch_size=(n, t))
  os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
  manifest = dict(
      source_dir=os.path.abspath(side_dir), task=task,
      n_episodes=int(n), ep_len=int(t), obs_dim=int(obs.shape[2]),
      action_dim=int(act.shape[2]),
      reward_sum=float(rew.sum()),
      content_sha256=hashlib.sha256(
          obs.tobytes() + act.tobytes() + rew.tobytes()).hexdigest(),
  )
  torch.save(dict(td=td, manifest=manifest), output)
  with open(output + '.manifest.json', 'w') as f:
    json.dump(manifest, f, indent=1)
  print(f'wrote {output}: {n} eps x {t} steps, obs_dim {obs.shape[2]}, '
        f'reward_sum {rew.sum():.2f}')
  return manifest


def selfcheck():
  """Numpy-level contract check (no torch needed): shift + validation."""
  t, n = 6, 2
  eps = []
  for i in range(n):
    eps.append(dict(
        position=np.arange(t * 4, dtype=np.float32).reshape(t, 4) + i,
        velocity=np.ones((t, 3), np.float32) * i,
        touch=np.zeros((t, 2), np.float32),
        target_position=np.zeros((t, 2), np.float32),
        dist_to_target=np.arange(t, dtype=np.float32),
        reward=np.array([0, 1, 0, 2, 0, 1], np.float32),
        action=np.stack([np.full(2, 10 * i + j, np.float32)
                         for j in range(t)]),
        is_first=np.eye(1, t, 0, dtype=bool)[0],
    ))
  obs, act, rew = episodes_to_arrays(eps, 'dmc_finger_turn_hard')
  assert obs.shape == (n, t, 12), obs.shape
  assert act.shape == (n, t, 2) and rew.shape == (n, t)
  assert (act[:, 0] == 0).all(), 'row 0 action must be zeros'
  assert (act[1][3] == np.full(2, 12.0)).all(), 'shift broken'  # ep1 act[2]
  assert obs[0, 2, 0] == 8.0, 'position not first obs block'
  assert obs[0, 2, 11] == 2.0, 'dist_to_target not last obs block'
  # contract violations must raise
  for mutate, msg in (
      (lambda e: e['reward'].__setitem__(0, 5.0), 'nonzero reward'),
      (lambda e: e['is_first'].__setitem__(2, True), 'is_first'),
  ):
    bad = [dict(ep) for ep in eps]
    bad[0] = {k: np.copy(v) for k, v in eps[0].items()}
    mutate(bad[0])
    try:
      episodes_to_arrays(bad, 'dmc_finger_turn_hard')
      raise AssertionError(f'expected SystemExit for {msg}')
    except SystemExit:
      pass
  print('SELFCHECK PASS')


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd', required=True)
  c = sub.add_parser('convert')
  c.add_argument('--side_dir', required=True)
  c.add_argument('--task', required=True)
  c.add_argument('--output', required=True)
  sub.add_parser('selfcheck')
  args = ap.parse_args()
  if args.cmd == 'selfcheck':
    selfcheck()
  else:
    if os.path.exists(args.output):
      raise SystemExit(f'{args.output} exists; refusing to overwrite a '
                       'materialized dataset')
    convert(args.side_dir, args.task, args.output)


if __name__ == '__main__':
  main()
