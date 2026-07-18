"""Stage-1B oracle labels: restorable-state value-of-operation ground truth.

Implements PREREG_gate_d1_stage1a_20260714.md SC: at sampled decision
states of a frozen d0-probe run, measure the REALIZED value of an
information/computation purchase

    Delta_o(s) = V(decision after o) - V(decision now)

by paired execute/skip branches under common random numbers (identical
restored MuJoCo physics + wrapper RNG + agent belief; branches differ
only in the first action), with the decision evaluated by a high-budget
frozen-policy rollout from the restored state.

Operation pair (one-pair scope per the 12-Jul revision), at matched
candidate-evaluation budget:

  real  -- real-data purchase: probe each candidate action with ONE real
           env step from the restored state; re-rank candidates by
           r_real + disc * Vhat(s'_real), where Vhat(s') is the mean-over-
           heads max-over-candidates of the agent's one-step Q at the
           observed next state. Metered cost: M real env steps + M policy
           calls.
  imag  -- imagined-rollout purchase: C extra d0 policy evaluations at the
           same state (fresh imagination RNG each call); re-rank by the
           averaged Q matrix. Metered cost: 0 real steps + C policy calls
           (C = M for matched candidate-evaluation budget).

Decisions: 'now' = argmax_m mean-over-heads qfull (the plug-in rule the
D0 signals describe). Oracle: G(a) = undiscounted return of executing a
then following the frozen policy (eval mode) for --horizon steps from the
restored snapshot. Labels: delta_real = G(a_real) - G(a_now), delta_imag
= G(a_imag) - G(a_now).

Output: one npz per run with per-state signals (qfull/qhalf/udyn — the
EVSI feature set), candidates, branch choices, G values, deltas, meters,
and run/episode/step ids (cross-fitting by run happens in the Gate-D1
read; labels carry their run identity). Amortized EVSI-hat training is
downstream analysis, not this tool.

Usage (cluster, dv3 env, run trained with the d0_probe config):
  python -m d0.oracle_labels --run_logdir <dir> --output <run>.npz \
      --states 200 [--horizon 100] [--label_every 25] [--platform cpu]
  python -m d0.oracle_labels --selfcheck   (synthetic env+agent, no MuJoCo)

MuJoCo plumbing must be smoke-validated on one real d0 run before the
labeling sweep (snapshot/restore determinism assert runs per state).
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

import numpy as np


# --------------------------------------------------------------------------
# Restorable-state plumbing (wrapper-chain walkers; no env code changes)
# --------------------------------------------------------------------------

def _chain(env):
  """Yield the wrapper chain outermost-first (attrs .env / ._env)."""
  seen = set()
  node = env
  while node is not None and id(node) not in seen:
    seen.add(id(node))
    yield node
    node = getattr(node, 'env', None) or getattr(node, '_env', None)


def snapshot_env(env):
  """Snapshot everything that makes the future stochastic or bounded:
  MuJoCo physics state, dm_control step counter, and any wrapper RNGs
  (e.g. the OU distractor). Returns an opaque dict for restore_env."""
  snap = {}
  for node in _chain(env):
    dmenv = getattr(node, '_dmenv', None)
    if dmenv is not None and 'physics' not in snap:
      snap['physics'] = np.array(dmenv.physics.get_state(), np.float64)
      snap['step_count'] = int(getattr(dmenv, '_step_count', 0))
      snap['dm_node'] = dmenv
    rng = getattr(node, '_rng', None)
    if rng is not None and hasattr(rng, 'bit_generator'):
      snap.setdefault('rngs', []).append(
          (node, json.dumps(rng.bit_generator.state)))
    if hasattr(node, 'oracle_get_state'):  # synthetic/selfcheck envs
      snap['custom'] = (node, node.oracle_get_state())
  if 'physics' not in snap and 'custom' not in snap:
    raise SystemExit('snapshot_env: no dm_control physics or '
                     'oracle_get_state found in the wrapper chain')
  return snap


def restore_env(env, snap):
  if 'custom' in snap:
    node, state = snap['custom']
    node.oracle_set_state(state)
  if 'physics' in snap:
    dmenv = snap['dm_node']
    with dmenv.physics.reset_context():
      dmenv.physics.set_state(snap['physics'])
    if hasattr(dmenv, '_step_count'):
      dmenv._step_count = snap['step_count']
    if hasattr(dmenv, '_reset_next_step'):
      dmenv._reset_next_step = False
  for node, state in snap.get('rngs', ()):
    node._rng.bit_generator.state = json.loads(state)


# --------------------------------------------------------------------------
# Agent adapters (thin; the selfcheck substitutes synthetic equivalents)
# --------------------------------------------------------------------------

class AgentOracle:
  """Frozen agent behind the three calls the labeler needs."""

  def __init__(self, agent, act_space, discount):
    self.agent = agent
    self.act_keys = sorted(k for k in act_space if k != 'reset')
    assert len(self.act_keys) == 1, 'oracle assumes single continuous action'
    self.act_key = self.act_keys[0]
    self.act_shape = act_space[self.act_key].shape
    self.discount = discount

  def init(self):
    return self.agent.init_policy(batch_size=1)

  def policy(self, carry, obs, mode='eval'):
    aobs = {k: np.asarray(v)[None] for k, v in obs.items()
            if not k.startswith('log/')}
    carry, acts, out = self.agent.policy(carry, aobs, mode=mode)
    return carry, acts, out

  def d0_eval(self, carry, obs):
    """One d0 policy evaluation: (qfull [K,M], cands [M,A], act dict)."""
    carry2, acts, out = self.policy(carry, obs)
    q = np.asarray(out['d0/qfull'][0], np.float32)
    cands = np.asarray(out['d0/cands'][0], np.float32)
    act = {k: np.asarray(acts[k][0]) for k in self.act_keys}
    return carry2, q, cands, act

  def vec2act(self, vec):
    return {self.act_key: np.asarray(vec, np.float32).reshape(self.act_shape)}


def plugin_choice(qfull):
  """Decision-now rule: argmax over candidates of mean-over-heads Q."""
  return int(np.argmax(qfull.mean(0)))


# --------------------------------------------------------------------------
# Branches (all start from one snapshot + one belief carry: exact CRN)
# --------------------------------------------------------------------------

def rollout_return(env, oracle, carry, first_act, horizon, snap):
  """G(a): restore, execute a, then frozen policy for `horizon` steps."""
  restore_env(env, snap)
  obs = env.step({**first_act, 'reset': np.array(False)})
  total = float(obs['reward'])
  for _ in range(horizon - 1):
    if bool(obs['is_last']):
      break
    carry, acts, _ = oracle.policy(carry, obs)
    act = {k: np.asarray(acts[k][0]) for k in oracle.act_keys}
    obs = env.step({**act, 'reset': np.array(False)})
    total += float(obs['reward'])
  return total


def op_imag(oracle, carry, obs, qfull, budget):
  """Imagined purchase: `budget` extra d0 evaluations, averaged."""
  qs = [qfull]
  for _ in range(budget):
    _, q, _, _ = oracle.d0_eval(carry, obs)
    qs.append(q)
  return plugin_choice(np.mean(qs, 0)), dict(env_steps=0,
                                             policy_calls=budget)


def op_real(env, oracle, carry, obs, qfull, cands, snap):
  """Real purchase: one real probe step per candidate, re-rank by
  r_real + disc * Vhat(s'_real)."""
  m_count = cands.shape[0]
  scores = np.zeros(m_count, np.float32)
  for m in range(m_count):
    restore_env(env, snap)
    nobs = env.step({**oracle.vec2act(cands[m]), 'reset': np.array(False)})
    r = float(nobs['reward'])
    _, qn, _, _ = oracle.d0_eval(carry, nobs)
    vhat = float(qn.mean(0).max())
    scores[m] = r + oracle.discount * vhat
  return int(np.argmax(scores)), dict(env_steps=m_count,
                                      policy_calls=m_count), scores


# --------------------------------------------------------------------------
# Labeling sweep
# --------------------------------------------------------------------------

def label_run(env, oracle, n_states, horizon, label_every, max_steps,
              rng, run_id, oracle_all=False):
  rows = []
  zero_act = oracle.vec2act(np.zeros(int(np.prod(oracle.act_shape))))
  carry = oracle.init()
  obs = env.step({**zero_act, 'reset': np.array(True)})
  ep, t = 0, 0
  while len(rows) < n_states:
    if bool(obs['is_last']):
      carry = oracle.init()
      obs = env.step({**zero_act, 'reset': np.array(True)})
      ep += 1
      t = 0
      continue
    label_here = (t % label_every == 0 and t > 0
                  and t <= max_steps - horizon - 1)
    if label_here:
      snap = snapshot_env(env)
      carry_s = carry  # jax pytrees are immutable: safe belief snapshot
      _, qfull, cands, _ = oracle.d0_eval(carry_s, obs)
      # determinism assert: restore must reproduce the same next obs
      restore_env(env, snap)
      probe1 = env.step({**oracle.vec2act(cands[0]), 'reset': np.array(False)})
      restore_env(env, snap)
      probe2 = env.step({**oracle.vec2act(cands[0]), 'reset': np.array(False)})
      if abs(float(probe1['reward']) - float(probe2['reward'])) > 1e-6:
        raise SystemExit('restore_env is not deterministic; CRN broken')
      restore_env(env, snap)

      m_now = plugin_choice(qfull)
      m_imag, cost_imag = op_imag(oracle, carry_s, obs, qfull,
                                  budget=cands.shape[0])
      m_real, cost_real, real_scores = op_real(
          env, oracle, carry_s, obs, qfull, cands, snap)

      targets = {m_now, m_imag, m_real}
      if oracle_all:
        targets = set(range(cands.shape[0]))
      G = {m: rollout_return(env, oracle, carry_s, oracle.vec2act(cands[m]),
                             horizon, snap) for m in sorted(targets)}
      rows.append(dict(
          run_id=run_id, episode=ep, step=t,
          qfull=qfull, cands=cands,
          m_now=m_now, m_imag=m_imag, m_real=m_real,
          g_now=G[m_now], g_imag=G[m_imag], g_real=G[m_real],
          delta_imag=G[m_imag] - G[m_now],
          delta_real=G[m_real] - G[m_now],
          real_scores=real_scores,
          cost_imag_env=cost_imag['env_steps'],
          cost_imag_calls=cost_imag['policy_calls'],
          cost_real_env=cost_real['env_steps'],
          cost_real_calls=cost_real['policy_calls'],
      ))
      restore_env(env, snap)  # resume the base trajectory untouched
    carry, acts, _ = oracle.policy(carry, obs)
    act = {k: np.asarray(acts[k][0]) for k in oracle.act_keys}
    obs = env.step({**act, 'reset': np.array(False)})
    t += 1
  return rows


def save_rows(rows, output, meta):
  cols = {}
  for key in rows[0]:
    vals = [r[key] for r in rows]
    if isinstance(vals[0], str):
      cols[key] = np.array(vals)
    else:
      cols[key] = np.stack([np.asarray(v) for v in vals], 0)
  os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
  np.savez_compressed(output, meta=json.dumps(meta), **cols)
  print(f'wrote {output}: {len(rows)} labeled states; '
        f'mean delta_real {cols["delta_real"].mean():+.3f}, '
        f'mean delta_imag {cols["delta_imag"].mean():+.3f}')


# --------------------------------------------------------------------------
# Real-run entry point (mirrors d0/sweep.py loading)
# --------------------------------------------------------------------------

def main_real(args):
  import elements
  import ruamel.yaml as yaml
  from dreamerv3.main import make_agent, make_env
  from d0.sweep import load_config, load_frozen_agent

  out_dir = pathlib.Path(args.output).parent
  config, train_seed = load_config(args, out_dir)
  env = make_env(config, 0)
  agent = make_agent(config)
  ckpt = args.checkpoint or os.path.join(args.run_logdir, 'ckpt')
  agent = load_frozen_agent(agent, ckpt)
  disc = (1.0 if config.agent.contdisc else
          1 - 1 / config.agent.horizon)
  oracle = AgentOracle(agent, env.act_space, disc)
  rng = np.random.default_rng(args.seed)
  rows = label_run(env, oracle, args.states, args.horizon,
                   args.label_every, args.max_steps, rng,
                   run_id=os.path.basename(args.run_logdir.rstrip('/')),
                   oracle_all=args.oracle_all)
  save_rows(rows, args.output, meta=dict(
      run_logdir=args.run_logdir, checkpoint=str(ckpt),
      states=args.states, horizon=args.horizon,
      label_every=args.label_every, seed=args.seed,
      train_seed=train_seed, actions=args.actions, rollouts=args.rollouts,
      operation_pair='real_vs_imag_matched_candidate_budget'))


# --------------------------------------------------------------------------
# Selfcheck: synthetic restorable env + synthetic agent, no MuJoCo/jax
# --------------------------------------------------------------------------

class _SynthEnv:
  """1-D chain with hidden per-candidate rewards; restorable via
  oracle_get/set_state. Rewards depend only on (state, action sign/idx),
  so the oracle labels have a closed-form best action."""

  def __init__(self, seed=0):
    self._rng_state = np.random.default_rng(seed)
    self.pos = 0.0
    self.t = 0

  def oracle_get_state(self):
    return (self.pos, self.t, json.dumps(
        self._rng_state.bit_generator.state))

  def oracle_set_state(self, s):
    self.pos, self.t = s[0], s[1]
    self._rng_state.bit_generator.state = json.loads(s[2])

  def step(self, act):
    if act.get('reset'):
      self.pos, self.t = 0.0, 0
      return dict(reward=np.float32(0), is_last=np.array(False))
    a = float(np.asarray(act['action']).reshape(-1)[0])
    self.pos += a
    self.t += 1
    reward = np.float32(1.0 if a > 0.5 else (0.2 if a > 0 else 0.0))
    return dict(reward=reward, is_last=np.array(self.t >= 200))


class _SynthOracle(AgentOracle):
  """M=3 fixed candidates [0.9, 0.3, -0.5]; the plug-in Q ranks the
  MIDDLE candidate first (miscalibrated), real probes reveal the truth,
  imagined purchases just re-sample the same wrong belief."""

  CANDS = np.array([[0.9], [0.3], [-0.5]], np.float32)

  def __init__(self):
    self.act_keys = ['action']
    self.act_key = 'action'
    self.act_shape = (1,)
    self.discount = 0.0  # scores = observed reward only: exposes ranking

  def init(self):
    return None

  def policy(self, carry, obs, mode='eval'):
    acts = {'action': self.CANDS[1][None]}  # frozen policy plays cand 1
    q = np.array([[0.1, 0.8, 0.0], [0.1, 0.8, 0.0]], np.float32)
    out = {'d0/qfull': q[None], 'd0/cands': self.CANDS[None]}
    return carry, acts, out


def selfcheck():
  env = _SynthEnv()
  oracle = _SynthOracle()
  rng = np.random.default_rng(0)
  rows = label_run(env, oracle, n_states=5, horizon=10, label_every=7,
                   max_steps=200, rng=rng, run_id='synth')
  for r in rows:
    assert r['m_now'] == 1, r  # plug-in picks the miscalibrated middle
    assert r['m_imag'] == 1, r  # imagined purchase repeats the belief
    assert r['m_real'] == 0, r  # real probe finds the true best (a=0.9)
    # G(now): first step 0.2 then policy plays cand1 (+0.2 x 9) = 2.0
    # G(real): first step 1.0 then policy plays cand1 (+0.2 x 9) = 2.8
    assert abs(r['delta_real'] - 0.8) < 1e-5, r['delta_real']
    assert abs(r['delta_imag'] - 0.0) < 1e-5, r['delta_imag']
    assert r['cost_real_env'] == 3 and r['cost_imag_env'] == 0
  # snapshot/restore roundtrip determinism on the synthetic env
  s = snapshot_env(env)
  before = env.oracle_get_state()
  env.step({'action': np.ones(1), 'reset': False})
  restore_env(env, s)
  assert env.oracle_get_state()[:2] == before[:2]
  out = pathlib.Path(os.environ.get('TMPDIR', '/tmp')) / 'oracle_sc.npz'
  save_rows(rows, str(out), meta=dict(selfcheck=True))
  data = np.load(out, allow_pickle=False)
  assert data['delta_real'].shape == (5,)
  print('SELFCHECK PASS')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--run_logdir')
  p.add_argument('--output')
  p.add_argument('--checkpoint', default='')
  p.add_argument('--states', type=int, default=200)
  p.add_argument('--horizon', type=int, default=100)
  p.add_argument('--label_every', type=int, default=25)
  p.add_argument('--max_steps', type=int, default=1000)
  p.add_argument('--seed', type=int, default=0)
  # d0-signal knobs consumed by sweep.load_config (agent.d0.actions/
  # rollouts); same defaults as d0/sweep.py [prov.; confirm at freeze].
  p.add_argument('--actions', type=int, default=8,
                 help='Candidate actions M: policy mode + M-1 samples.')
  p.add_argument('--rollouts', type=int, default=16,
                 help='One-step rollouts R per (head, action).')
  p.add_argument('--oracle_all', action='store_true',
                 help='ground-truth every candidate, not just the chosen')
  p.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not args.run_logdir or not args.output:
    p.error('--run_logdir and --output required (or --selfcheck)')
  if args.platform:
    os.environ['JAX_PLATFORMS'] = args.platform
  main_real(args)


if __name__ == '__main__':
  main()
