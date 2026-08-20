"""Tier-0 EXPLORATORY: per-frame reward-head dump for the misalignment
probe family (Algorithm_Ideation_20260819 §3.3).

GPU stage. For one run's frozen WM checkpoint, replays the frozen E4
probeset through the SAME posterior path as probing/stratified_error.py
cmd_measure (h=0 only) and writes per-frame arrays:

  probe_family_dump.npz:
    rhat      [N, T]  reward-head predicted mean (model.rew(...).pred())
    nll_id    [N, T]  reward-head identity NLL vs true reward (checksum
                      against the E4 csv's rew_nll_* for aware arms)
    reward    [N, T]  true reward from the probeset
    in_regime [N, T]  regime mask
    rewarded  [N, T]  reward > 0 mask
    is_first  [N, T]
  + probe_family_manifest.json (run_id, ckpt path, probeset sha,
    expl_mode, gpu_name — single-GPU panel discipline applies).

Unlike cmd_measure, rhat/nll_id are computed for EVERY arm including
reward_free ones (the untrained head is the control the family test
needs; its NLL is meaningless-by-design and labeled so).

The recoding family itself is fit CPU-side by analysis/tier0_probe_family.py
on these dumps. No registered instrument is modified; this file is additive.

Usage (torch not needed; JAX env, one GPU, same GPU for the whole panel):
  python -m probing.probe_family_dump --run_logdir <run> \
      --probeset $RUNROOT/e4_probesets/finger_v1 --platform cuda \
      [--checkpoint <ckpt>] [--output <dir>] [--ep_batch 8]
"""

import argparse
import json
import os

import numpy as np

from probing.stratified_error import load_e4


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run_logdir', required=True)
  ap.add_argument('--probeset', required=True)
  ap.add_argument('--platform', default='cuda')
  ap.add_argument('--checkpoint', default=None)
  ap.add_argument('--output', default=None)
  ap.add_argument('--ep_batch', type=int, default=8)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--allow_unfrozen', action='store_true')
  args = ap.parse_args()

  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from dreamerv3.main import make_agent
  from probing.collect import load_run_config, load_frozen_agent

  f32 = jnp.float32
  arrays, manifest = load_e4(args.probeset, not args.allow_unfrozen)
  N, T = arrays['is_first'].shape
  out_dir = args.output or os.path.join(
      args.run_logdir, f'probe_family_{manifest["probeset_id"]}')
  os.makedirs(out_dir, exist_ok=True)

  config = load_run_config(args.run_logdir, args.platform, out_dir, False)
  config = config.update({'jax': {
      'precompile': False, 'enable_policy': False, 'prealloc': False}})
  expl_mode = str(config.agent.expl.mode)
  agent = make_agent(config)
  ckpt = args.checkpoint or os.path.join(args.run_logdir, 'ckpt')
  load_frozen_agent(agent, ckpt)
  model = agent.model
  jax.config.update('jax_transfer_guard', 'allow')

  feed_keys = sorted(set(model.dec.obs_space.keys())
                     | set(model.enc.obs_space.keys()))
  isimg = {k: len(agent.obs_space[k].shape) == 3 for k in feed_keys}
  act_keys = sorted(agent.act_space.keys())
  missing = [k for k in feed_keys + act_keys if k not in arrays]
  assert not missing, f'probe set lacks keys {missing}'

  def fn(obs, action_dict, reward, reset):
    B = reset.shape[0]
    enc_carry = model.enc.initial(B)
    dyn_carry = model.dyn.initial(B)
    enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)
    prevact = {k: jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], 1)
               for k, v in action_dict.items()}
    dyn_carry, _, post = model.dyn.observe(
        dyn_carry, tokens, prevact, reset, training=False)
    inp = model.feat2tensor({'deter': post['deter'], 'stoch': post['stoch']})
    dist = model.rew(inp, 2)
    return {'rhat': f32(dist.pred()), 'nll_id': f32(dist.loss(reward))}

  pure = nj.pure(fn)
  jit = jax.jit(lambda p, o, a, r, re, s: pure(p, o, a, r, re, seed=s))
  params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), agent.params)

  reset_np = np.asarray(arrays['is_first'], bool)
  acc = {'rhat': [], 'nll_id': []}
  for lo in range(0, N, args.ep_batch):
    hi = min(lo + args.ep_batch, N)
    obs = {k: (jnp.asarray(arrays[k][lo:hi]) if isimg[k]
               else jnp.asarray(arrays[k][lo:hi], np.float32))
           for k in feed_keys}
    action_dict = {k: jnp.asarray(arrays[k][lo:hi], np.float32)
                   for k in act_keys}
    reward = jnp.asarray(arrays['reward'][lo:hi], np.float32)
    reset = jnp.asarray(reset_np[lo:hi])
    seed = jnp.array([args.seed, lo + 1], np.uint32)
    _, out = jit(params, obs, action_dict, reward, reset, seed)
    for k in acc:
      acc[k].append(np.asarray(out[k], np.float64))
    print(f'  episodes {lo}-{hi - 1}')

  gpu = jax.devices()[0]
  np.savez(
      os.path.join(out_dir, 'probe_family_dump.npz'),
      rhat=np.concatenate(acc['rhat'], 0).astype(np.float32),
      nll_id=np.concatenate(acc['nll_id'], 0).astype(np.float32),
      reward=np.asarray(arrays['reward'], np.float32),
      in_regime=np.asarray(arrays['in_regime'], bool),
      rewarded=np.asarray(arrays['reward'] > 0, bool),
      is_first=reset_np)
  with open(os.path.join(out_dir, 'probe_family_manifest.json'), 'w') as f:
    json.dump(dict(
        label='EXPLORATORY tier0 probe-family dump',
        run_id=os.path.basename(args.run_logdir.rstrip('/')),
        run_logdir=os.path.abspath(args.run_logdir),
        checkpoint=os.path.abspath(ckpt),
        expl_mode=expl_mode,
        probeset_id=manifest['probeset_id'],
        probeset_sha256=manifest['sha256'],
        device_kind=str(gpu.device_kind),
        platform=str(gpu.platform)), f, indent=1)
  print(f'-> {out_dir}')


if __name__ == '__main__':
  main()
