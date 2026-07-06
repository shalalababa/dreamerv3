"""Dump RSSM latents for a frozen probe set over retained checkpoints.

Background (Rep-Convergence idea brief, 2 Jul 2026, Sec. 4)
-----------------------------------------------------------
For each retained checkpoint of a DreamerV3 run, encode the frozen probe set
(``probing/probeset.py``) and store the latent matrices the convergence
analysis needs: posterior means of the RSSM (deterministic state plus
categorical posterior probabilities).  Encoding is pure inference over saved
checkpoints -- no training-code change.  The probe protocol (window length,
burn-in) comes from the probe set's manifest, so it cannot drift per run.

Per window, state is reset at the window start, burned in for the manifest's
``burn_in`` steps, and latents are read from the ``eval_steps`` tail
(``--store last`` keeps the final step only, one vector per window --- the
default and the primary probe point; ``--store tail`` keeps the whole tail).
As a secondary view for prior/open-loop alignment (the analog of TD-MPC2's
dynamics-rolled latents), ``--prior_horizon k`` additionally dumps the k-step
open-loop prior state started from the posterior at the end of burn-in.

Output layout (one directory per run x probe set)::

    <output>/
      index.json                    # checkpoints processed + probeset identity
      step<exactstep>/latents.npz   # post_deter (S,d), post_stoch (S,k*c),
                                    # [prior{k}_deter, prior{k}_stoch]
      step<exactstep>/meta.json

Usage
-----
Over the retained snapshot series of one pretraining run (milestones are
mapped to the nearest retained snapshot, as in Phase 5)::

    python -m probing.latents \
        --probeset $SCRATCH/probesets/cup_v1 \
        --run_logdir $RUN/pretrain_p2e_cup_seed1 \
        --milestones 100000 200000 300000 400000 500000 \
        --output $RUN/pretrain_p2e_cup_seed1/latents/cup_v1

Or on explicit checkpoints (e.g. the final ``ckpt``)::

    python -m probing.latents --probeset ... --run_logdir <RUN> \
        --checkpoints <RUN>/ckpt --output ...

Validation (after any debug run with a saved checkpoint)::

    python -m probing.probeset --selfcheck   # probe-set logic
    python -m probing.latents --probeset <set> --run_logdir <debug_run> \
        --checkpoints <debug_run>/ckpt --output /tmp/latents_smoke \
        --platform cpu
"""

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamerv3.main import make_agent
from probing import probeset as probeset_mod
from probing.checkpoint_watcher import resolve_checkpoints as _resolve
from probing.collect import load_run_config, load_frozen_agent

f32 = jnp.float32


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--probeset', required=True,
                 help='Frozen probe-set directory (probing/probeset.py).')
  p.add_argument('--run_logdir', required=True,
                 help='DreamerV3 run dir (config.yaml).')
  p.add_argument('--output', required=True, help='Output directory.')
  p.add_argument('--milestones', type=int, nargs='+',
                 default=[100000, 200000, 300000, 400000, 500000],
                 help='Mapped to nearest retained snapshots in '
                      '<run_logdir>/ckpt_snapshots (checkpoint_watcher).')
  p.add_argument('--snapshots_dir', default='',
                 help='Default <run_logdir>/ckpt_snapshots.')
  p.add_argument('--checkpoints', nargs='+', default=[],
                 help='Explicit checkpoint dirs; overrides --milestones.')
  p.add_argument('--store', default='last', choices=['last', 'tail'],
                 help="'last': one vector per window (final eval step); "
                      "'tail': all eval steps.")
  p.add_argument('--prior_horizon', type=int, default=0,
                 help='If >0, also dump the k-step open-loop prior from the '
                      'end of burn-in (secondary view).')
  p.add_argument('--ep_batch', type=int, default=64,
                 help='Probe windows per forward pass.')
  p.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  p.add_argument('--allow_unfrozen', action='store_true')
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def resolve_checkpoints(args):
  """Return [(name, ckpt_dir, exact_step, milestone)] to process."""
  return _resolve(args.run_logdir, args.milestones, args.snapshots_dir,
                  args.checkpoints)


def rssm_encode(model, obs, action_dict, reset, prior_horizon, burn_in):
  """Posterior deter/stoch-probs over a batch of probe windows.

  Mirrors probing/features.py: encoder tokens -> dyn.observe with prevact
  shifted right (zero action at the reset step).
  """
  B, T = reset.shape
  enc_carry = model.enc.initial(B)
  dyn_carry = model.dyn.initial(B)
  enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)
  prevact = {k: jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], 1)
             for k, v in action_dict.items()}
  dyn_carry, _, post = model.dyn.observe(
      dyn_carry, tokens, prevact, reset, training=False)
  out = {
      'post_deter': f32(post['deter']),
      'post_stoch': jax.nn.softmax(f32(post['logit']), -1).reshape((B, T, -1)),
  }
  if prior_horizon:
    # Open-loop k-step prior anchored at the end of burn-in (index b-1),
    # driven by the window's own logged actions; the resulting state at index
    # b-1+k is comparable with the posterior at the same index (for
    # k == eval_steps, the 'last' probe point).
    k, b = prior_horizon, burn_in
    carry = {'deter': post['deter'][:, b - 1], 'stoch': post['stoch'][:, b - 1]}
    acts = {ak: av[:, b - 1:b - 1 + k] for ak, av in action_dict.items()}
    _, feat, _ = model.dyn.imagine(carry, acts, k, training=False)
    out[f'prior{k}_deter'] = f32(feat['deter'][:, -1])
    out[f'prior{k}_stoch'] = jax.nn.softmax(
        f32(feat['logit'][:, -1]), -1).reshape((B, -1))
  return out


def main():
  args = parse_args()
  arrays, manifest = probeset_mod.load(
      args.probeset, require_frozen=not args.allow_unfrozen)
  burn_in, eval_steps = manifest['burn_in'], manifest['eval_steps']
  L = manifest['length']
  S = arrays['is_first'].shape[0]
  print(f'Probe set {manifest["probeset_id"]}: {S} windows x {L} steps '
        f'({burn_in} burn-in + {eval_steps} eval)')

  # Held-out sanity: the probe set must not come from the probed run itself.
  run_replay = os.path.realpath(os.path.join(args.run_logdir, 'replay'))
  for label, src in manifest['sources'].items():
    src_dir = os.path.realpath(src['replay_dir'])
    if src_dir == run_replay or src_dir.startswith(run_replay + os.sep):
      print(f'WARNING: probe-set source {label!r} is the replay of the run '
            f'being probed; latents are NOT held-out for this run.')

  todo = resolve_checkpoints(args)
  if not todo:
    raise SystemExit('No checkpoints to process.')
  print(f'Checkpoints: {[name for name, *_ in todo]}')

  os.makedirs(args.output, exist_ok=True)
  config = load_run_config(args.run_logdir, args.platform, args.output, False)
  # Latent dumps only use enc/dyn inference. Avoid train/report precompilation
  # and policy parameter copies, which are unnecessary here and can trip native
  # CUDA/XLA failures on cluster inference jobs.
  config = config.update({'jax': {
      'precompile': False,
      'enable_policy': False,
      'prealloc': False,
  }})
  agent = make_agent(config)
  jax.config.update('jax_transfer_guard', 'allow')

  exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
  obs_keys = sorted(k for k, v in agent.obs_space.items()
                    if k not in exclude and len(v.shape) <= 1)
  act_keys = sorted(agent.act_space.keys())
  missing = [k for k in obs_keys + act_keys if k not in arrays]
  assert not missing, f'Probe set lacks keys {missing} required by the agent.'
  model = agent.model

  # Reset at the window start only; the builder guarantees no episode
  # boundary inside a window, so the posterior at burn_in is well-defined.
  reset_np = np.zeros((S, L), bool)
  reset_np[:, 0] = True

  assert args.prior_horizon <= eval_steps, (
      f'--prior_horizon {args.prior_horizon} exceeds the probe set\'s '
      f'eval_steps {eval_steps}; the open-loop roll would leave the window.')

  def fn(obs, action_dict, reset):
    return rssm_encode(model, obs, action_dict, reset, args.prior_horizon,
                       burn_in)

  pure = nj.pure(fn)
  jit = jax.jit(lambda p, o, a, r, s: pure(p, o, a, r, seed=s))

  index = dict(
      run_logdir=os.path.abspath(args.run_logdir),
      probeset=os.path.abspath(args.probeset),
      probeset_id=manifest['probeset_id'], probeset_sha256=manifest['sha256'],
      burn_in=burn_in, eval_steps=eval_steps, store=args.store,
      prior_horizon=args.prior_horizon, checkpoints=[])

  for name, ckpt, exact_step, milestone in todo:
    load_frozen_agent(agent, ckpt)
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                          agent.params)
    feats = {}
    for lo in range(0, S, args.ep_batch):
      hi = min(lo + args.ep_batch, S)
      obs = {k: jnp.asarray(arrays[k][lo:hi], np.float32) for k in obs_keys}
      action_dict = {k: jnp.asarray(arrays[k][lo:hi], np.float32)
                     for k in act_keys}
      reset = jnp.asarray(reset_np[lo:hi])
      seed = jnp.array([args.seed, lo + 1], np.uint32)
      _, out = jit(params, obs, action_dict, reset, seed)
      for k, v in out.items():
        v = np.asarray(v, np.float32)
        if v.ndim == 3:  # (B, L, d) sequences: keep the eval tail only
          v = v[:, -1] if args.store == 'last' else v[:, burn_in:]
        feats.setdefault(k, []).append(v)
    feats = {k: np.concatenate(v, 0) for k, v in feats.items()}

    step_dir = os.path.join(args.output, name)
    os.makedirs(step_dir, exist_ok=True)
    np.savez_compressed(os.path.join(step_dir, 'latents.npz'), **feats)
    meta = dict(
        checkpoint=os.path.abspath(ckpt), exact_step=exact_step,
        milestone=milestone, probeset_id=manifest['probeset_id'],
        probeset_sha256=manifest['sha256'], store=args.store,
        burn_in=burn_in, eval_steps=eval_steps,
        shapes={k: list(v.shape) for k, v in feats.items()})
    with open(os.path.join(step_dir, 'meta.json'), 'w') as f:
      json.dump(meta, f, indent=2)
    index['checkpoints'].append(dict(
        name=name, exact_step=exact_step, milestone=milestone,
        path=os.path.join(name, 'latents.npz')))
    shapes = {k: v.shape for k, v in feats.items()}
    print(f'  {name} (milestone {milestone}): {shapes} -> {step_dir}')

  with open(os.path.join(args.output, 'index.json'), 'w') as f:
    json.dump(index, f, indent=2)
  print(f'Done: {len(index["checkpoints"])} checkpoints -> {args.output}')


if __name__ == '__main__':
  main()
