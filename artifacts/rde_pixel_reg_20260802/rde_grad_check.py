"""Local CPU verification for the rde arm (PREREG_rde_pixel_20260802, rev 2).

Constructs the REAL dreamerv3 Agent (bypassing the embodied.jax wrapper)
with synthetic pixel spaces at tiny sizes, traces the REAL Agent.loss
method under training=True, and checks gradient reachability of the
trunk (enc+dyn) from EVERY loss component under both recon_grad
settings. Also unit-checks the shipped _combine_deter_moments against
direct np.std (uneven batches, large offset, constant input).

REV 2 (post-review): training=False -> training=True; value/policy
output kernels un-zeroed alongside rew (zero-initialized outscale
blocks feature gradients AT INIT, W^T @ delta = 0, making a comparison
vacuous — reviewer finding m5); comparison now covers ALL loss
components, asserting bit-identical trunk grads across settings for
every non-image component; moment check now imports the SHIPPED
combiner instead of re-implementing the math (reviewer finding m4).

Expected:
  recon_grad=True : g(image->enc) > 0, g(image->dyn) > 0
  recon_grad=False: g(image->enc) == 0 exactly, g(image->dyn) == 0
                    exactly, g(image->dec) > 0 (decoder still trains),
                    every other component's (enc, dyn) grad norms
                    BIT-IDENTICAL to the recon_grad=True run.
"""

import os
import sys
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, '/home/rickybao/projects/dreamerv3')
os.chdir('/home/rickybao/projects/dreamerv3')

import numpy as np
import jax
import jax.numpy as jnp
import ninjax as nj
import elements
import ruamel.yaml as yaml

from dreamerv3.agent import Agent

B, T, H, W = 2, 6, 32, 32


def build(recon_grad):
  cfgs = yaml.YAML(typ='safe').load(
      open('dreamerv3/configs.yaml'))
  agent_cfg = elements.Config(cfgs['defaults']['agent'])
  agent_cfg = agent_cfg.update({
      'model_obs': 'image',
      'recon_grad': recon_grad,
      'dyn': {'rssm': {'deter': 64, 'hidden': 32, 'stoch': 4, 'classes': 8,
                       'blocks': 4}},
      'enc': {'simple': {'depth': 4, 'mults': [1, 2], 'layers': 1,
                         'units': 32}},
      'dec': {'simple': {'depth': 4, 'mults': [1, 2], 'layers': 1,
                         'units': 32, 'bspace': 4}},
      # outscale 1.0 (default 0.0 for rew/value): zero-initialized output
      # kernels block feature gradients AT INIT (W^T @ delta = 0); real
      # runs have nonzero kernels after the first update. Harness-only.
      'rewhead': {'units': 32, 'bins': 15, 'outscale': 1.0},
      'conhead': {'units': 32},
      'policy': {'layers': 1, 'units': 32, 'outscale': 1.0},
      'value': {'layers': 1, 'units': 32, 'bins': 15, 'outscale': 1.0},
      'imag_length': 3,
      'imag_last': 0,
  })
  obs_space = {
      'image': elements.Space(np.uint8, (H, W, 3)),
      'reward': elements.Space(np.float32, ()),
      'is_first': elements.Space(bool, ()),
      'is_last': elements.Space(bool, ()),
      'is_terminal': elements.Space(bool, ()),
  }
  act_space = {'action': elements.Space(np.float32, (3,), -1.0, 1.0)}
  inst = object.__new__(Agent)
  Agent.__init__(inst, obs_space, act_space, agent_cfg)
  return inst


def data(rng):
  obs = {
      'image': jnp.asarray(rng.integers(0, 255, (B, T, H, W, 3), np.uint8)),
      'reward': jnp.asarray(rng.normal(size=(B, T)).astype(np.float32)),
      'is_first': jnp.asarray(np.eye(T, dtype=bool)[0][None].repeat(B, 0)),
      'is_last': jnp.zeros((B, T), bool),
      'is_terminal': jnp.zeros((B, T), bool),
  }
  prevact = {'action': jnp.asarray(
      rng.uniform(-1, 1, (B, T, 3)).astype(np.float32))}
  return obs, prevact


def grads(recon_grad):
  inst = build(recon_grad)
  obs, prevact = data(np.random.default_rng(0))

  def fwd(obs, prevact):
    carry = (inst.enc.initial(B), inst.dyn.initial(B), inst.dec.initial(B))
    loss, (_, _, outs, _) = inst.loss(carry, obs, prevact, training=True)
    return {k: v.sum() for k, v in outs['losses'].items()}

  pure = nj.pure(fwd)
  params, comps = pure({}, obs, prevact, seed=0, create=True)

  def comp_grad(name):
    def scalar(p):
      _, out = pure(p, obs, prevact, seed=0)
      return out[name]
    g = jax.grad(scalar)(params)
    norm = lambda pre: float(np.sqrt(sum(
        float((v ** 2).sum()) for k, v in g.items() if k.startswith(pre))))
    return {pre: norm(pre) for pre in ('enc/', 'dyn/', 'dec/')}

  return sorted(comps.keys()), {c: comp_grad(c) for c in sorted(comps)}


def main():
  comps_on, g_on = grads(True)
  comps_off, g_off = grads(False)
  assert comps_on == comps_off, (comps_on, comps_off)
  print(f'loss components: {comps_on}')
  for c in comps_on:
    print(f'  {c:8s} on={g_on[c]}  off={g_off[c]}')

  assert g_on['image']['enc/'] > 0 and g_on['image']['dyn/'] > 0, g_on['image']
  assert g_off['image']['enc/'] == 0.0, ('recon->enc leak', g_off['image'])
  assert g_off['image']['dyn/'] == 0.0, ('recon->dyn leak', g_off['image'])
  assert g_off['image']['dec/'] > 0, ('decoder must still train', g_off)
  for c in comps_on:
    if c == 'image':
      continue
    assert g_on[c] == g_off[c], (
        f'{c}: trunk grads differ across recon_grad settings',
        g_on[c], g_off[c])
  assert g_off['rew']['enc/'] > 0 and g_off['rew']['dyn/'] > 0, g_off['rew']
  assert g_off['repval']['enc/'] > 0, ('repval path must stay open',
                                       g_off['repval'])
  print('image->trunk 0.0 exact under the flag; decoder still trains; '
        'ALL other components bit-identical across settings '
        '(rew/repval trunk paths verified open).')

  # deter_std: the SHIPPED combiner vs direct np.std.
  from probing.stratified_error import _combine_deter_moments
  rng = np.random.default_rng(1)
  states = rng.normal(size=(101, 16)) * rng.uniform(0.5, 2, 16) + 1e3
  splits = np.split(states, [40, 72])
  m = np.stack([np.stack([b.mean(0), ((b - b.mean(0)) ** 2).sum(0)])
                for b in splits])
  n = np.array([[len(b)] for b in splits], np.float64)
  ours = _combine_deter_moments(m, n)
  ref = float(states.std(0).mean())
  assert abs(ours - ref) < 1e-9 * ref, (ours, ref)
  const = np.full((4, 2, 16), 3.0); const[:, 1] = 0.0
  assert _combine_deter_moments(const, np.full((4, 1), 7.0)) == 0.0
  print('shipped _combine_deter_moments: uneven batches + offset 1e3 '
        'matches np.std; constant -> 0. PASS')

  print('ALL CHECKS PASS')


if __name__ == '__main__':
  main()
