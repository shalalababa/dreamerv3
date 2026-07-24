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

Output: one npz per run with per-state signals (qfull + udyn), the
belief latent (RSSM deter, float16) at each labeled state, candidates,
branch choices, G values (plain and value-bootstrapped), deltas, meters,
and run/episode/step ids (cross-fitting by run happens in the read;
labels carry their run identity). A strided base-trajectory latent
reference (ref_deter + ref_episode/ref_step) is stored per run so the
read can compute a kNN density proxy at the labeled states. Amortized
EVSI-hat training is downstream analysis, not this tool.

R2-probe extension (2026-07-22, PREREG_gate_d1_r2probe_20260722.md):
udyn + deter + ref arrays + bootstrapped G columns (g_*_boot = G_h +
Vhat(s_h) from the agent's own ensemble at the truncation state; plain
undiscounted G stays the primary label). The Gate-D1-era columns are
unchanged bit-for-bit in semantics.

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

def _own(node, name):
  """This node's OWN instance attribute, or None. Plain getattr is wrong
  here twice over: embodied Wrapper.__getattr__ DELEGATES lookups down
  the chain (an outer wrapper would report the inner env's attribute as
  its own, and restore would then write to the wrong node) and converts
  the terminal miss to ValueError, which getattr defaults and hasattr
  do not suppress."""
  return vars(node).get(name)


def _chain(env):
  """Yield the wrapper chain outermost-first (attrs .env / ._env)."""
  seen = set()
  node = env
  while node is not None and id(node) not in seen:
    seen.add(id(node))
    yield node
    node = _own(node, 'env') or _own(node, '_env')


def snapshot_env(env):
  """Snapshot everything that makes the future stochastic or bounded:
  MuJoCo physics state, dm_control step counter, and any wrapper RNGs
  (e.g. the OU distractor). Returns an opaque dict for restore_env."""
  snap = {}
  for node in _chain(env):
    dmenv = _own(node, '_dmenv')
    if dmenv is not None and 'physics' not in snap:
      snap['physics'] = np.array(dmenv.physics.get_state(), np.float64)
      snap['step_count'] = int(getattr(dmenv, '_step_count', 0))
      snap['dm_node'] = dmenv
    rng = _own(node, '_rng')
    if rng is not None and hasattr(rng, 'bit_generator'):
      snap.setdefault('rngs', []).append(
          (node, json.dumps(rng.bit_generator.state)))
    # method lookup on the CLASS: skips instance-dict misses AND the
    # wrapper delegation (synthetic/selfcheck envs only)
    if callable(getattr(type(node), 'oracle_get_state', None)):
      snap['custom'] = (node, node.oracle_get_state())
  if 'physics' not in snap and 'custom' not in snap:
    raise SystemExit('snapshot_env: no dm_control physics or '
                     'oracle_get_state found in the wrapper chain')
  return snap


def apply_mass_scale(env, scale):
  """Physics-shift arm: scale every body mass of the underlying
  dm_control model ONCE at setup. Model parameters are not part of
  physics.get_state(), so CRN snapshot/restore is unaffected. Refuses
  to run silently unshifted."""
  if float(scale) == 1.0:
    return
  for node in _chain(env):
    dmenv = _own(node, '_dmenv')
    if dmenv is not None:
      dmenv.physics.model.body_mass[:] = (
          dmenv.physics.model.body_mass * float(scale))
      return
  raise SystemExit('apply_mass_scale: no dm_control physics in the '
                   'wrapper chain; refusing to label unshifted')


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
    """One d0 policy evaluation:
    (carry, qfull [K,M], cands [M,A], act dict, extras)."""
    carry2, acts, out = self.policy(carry, obs)
    q = np.asarray(out['d0/qfull'][0], np.float32)
    cands = np.asarray(out['d0/cands'][0], np.float32)
    act = {k: np.asarray(acts[k][0]) for k in self.act_keys}
    extras = dict(udyn=float(np.asarray(out['d0/udyn'][0])),
                  deter=self.latent_of(carry2))
    return carry2, q, cands, act, extras

  def latent_of(self, carry):
    """Belief latent behind a policy carry: RSSM deter vector, float16.
    Agent carry layout is (enc_carry, dyn_carry, dec_carry, prevact);
    dyn_carry is the RSSM carry dict with 'deter' [B, D]."""
    import jax
    dyn = carry[1]
    with jax._src.config.explicit_device_get_scope():
      return np.asarray(jax.device_get(dyn['deter'][0]), np.float16)

  def vec2act(self, vec):
    return {self.act_key: np.asarray(vec, np.float32).reshape(self.act_shape)}


def plugin_choice(qfull):
  """Decision-now rule: argmax over candidates of mean-over-heads Q."""
  return int(np.argmax(qfull.mean(0)))


# --------------------------------------------------------------------------
# Branches (all start from one snapshot + one belief carry: exact CRN)
# --------------------------------------------------------------------------

def rollout_return(env, oracle, carry, first_act, horizon, snap):
  """G(a): restore, execute a, then frozen policy for `horizon` steps.
  Returns (G, G_boot): the plain undiscounted return (unchanged Gate-D1
  semantics) and the value-bootstrapped truncation variant G + Vhat(s_h)
  from the agent's own ensemble at the final state (R3 descriptive
  companion; G_boot == G when the episode ended before the horizon)."""
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
  if bool(obs['is_last']):
    return total, total
  _, qend, _, _, _ = oracle.d0_eval(carry, obs)
  return total, total + float(qend.mean(0).max())


def op_imag(oracle, carry, obs, qfull, budget):
  """Imagined purchase: `budget` extra d0 evaluations, averaged."""
  qs = [qfull]
  for _ in range(budget):
    _, q, _, _, _ = oracle.d0_eval(carry, obs)
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
    _, qn, _, _, _ = oracle.d0_eval(carry, nobs)
    vhat = float(qn.mean(0).max())
    scores[m] = r + oracle.discount * vhat
  return int(np.argmax(scores)), dict(env_steps=m_count,
                                      policy_calls=m_count), scores


# --------------------------------------------------------------------------
# Labeling sweep
# --------------------------------------------------------------------------

def label_run(env, oracle, n_states, horizon, label_every, max_steps,
              rng, run_id, oracle_all=False, ref_stride=5,
              behavior=None):
  """behavior: optional second AgentOracle that DRIVES the base
  trajectory (state visitation) while `oracle` remains the evaluation
  agent for beliefs, d0 evals, operations, and rollouts — the
  cross-policy shift arm. With behavior=None this is bit-identical to
  the r2_ext labeler path."""
  rows = []
  ref = dict(deter=[], episode=[], step=[])
  zero_act = oracle.vec2act(np.zeros(int(np.prod(oracle.act_shape))))
  carry = oracle.init()
  bcarry = behavior.init() if behavior is not None else None
  obs = env.step({**zero_act, 'reset': np.array(True)})
  ep, t = 0, 0
  while len(rows) < n_states:
    if bool(obs['is_last']):
      carry = oracle.init()
      if behavior is not None:
        bcarry = behavior.init()
      obs = env.step({**zero_act, 'reset': np.array(True)})
      ep += 1
      t = 0
      continue
    label_here = (t % label_every == 0 and t > 0
                  and t <= max_steps - horizon - 1)
    if label_here:
      snap = snapshot_env(env)
      carry_s = carry  # jax pytrees are immutable: safe belief snapshot
      _, qfull, cands, _, extras = oracle.d0_eval(carry_s, obs)
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
          udyn=np.float32(extras['udyn']),
          deter=np.asarray(extras['deter'], np.float16),
          m_now=m_now, m_imag=m_imag, m_real=m_real,
          g_now=G[m_now][0], g_imag=G[m_imag][0], g_real=G[m_real][0],
          delta_imag=G[m_imag][0] - G[m_now][0],
          delta_real=G[m_real][0] - G[m_now][0],
          g_now_boot=G[m_now][1], g_imag_boot=G[m_imag][1],
          g_real_boot=G[m_real][1],
          delta_imag_boot=G[m_imag][1] - G[m_now][1],
          delta_real_boot=G[m_real][1] - G[m_now][1],
          real_scores=real_scores,
          cost_imag_env=cost_imag['env_steps'],
          cost_imag_calls=cost_imag['policy_calls'],
          cost_real_env=cost_real['env_steps'],
          cost_real_calls=cost_real['policy_calls'],
      ))
      restore_env(env, snap)  # resume the base trajectory untouched
    # the evaluation agent's belief always tracks the observed stream
    carry, acts, _ = oracle.policy(carry, obs)
    if behavior is not None:
      bcarry, acts, _ = behavior.policy(bcarry, obs)
    if t % ref_stride == 0:
      # belief at obs_t (same step convention as the labeled rows)
      ref['deter'].append(oracle.latent_of(carry))
      ref['episode'].append(ep)
      ref['step'].append(t)
    act = {k: np.asarray(acts[k][0]) for k in oracle.act_keys}
    obs = env.step({**act, 'reset': np.array(False)})
    t += 1
  ref_arrays = dict(
      ref_deter=np.stack(ref['deter'], 0).astype(np.float16),
      ref_episode=np.asarray(ref['episode'], np.int64),
      ref_step=np.asarray(ref['step'], np.int64))
  return rows, ref_arrays


def save_rows(rows, output, meta, extra_arrays=None):
  cols = {}
  for key in rows[0]:
    vals = [r[key] for r in rows]
    if isinstance(vals[0], str):
      cols[key] = np.array(vals)
    else:
      cols[key] = np.stack([np.asarray(v) for v in vals], 0)
  for key, arr in (extra_arrays or {}).items():
    assert key not in cols, key
    cols[key] = arr
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
  import jax
  import ruamel.yaml as yaml
  from dreamerv3.main import make_agent, make_env
  from d0.sweep import load_config, load_frozen_agent

  out_dir = pathlib.Path(args.output).parent
  config, train_seed = load_config(args, out_dir)
  env = make_env(config, 0)
  apply_mass_scale(env, args.mass_scale)
  agent = make_agent(config)
  ckpt = args.checkpoint or os.path.join(args.run_logdir, 'ckpt')
  agent = load_frozen_agent(agent, ckpt)
  disc = (1.0 if config.agent.contdisc else
          1 - 1 / config.agent.horizon)
  oracle = AgentOracle(agent, env.act_space, disc)
  behavior = None
  if args.behavior_checkpoint:
    assert os.path.abspath(args.behavior_checkpoint) != os.path.abspath(
        str(ckpt)), 'behavior checkpoint must differ from the eval one'
    bagent = load_frozen_agent(make_agent(config),
                               args.behavior_checkpoint)
    behavior = AgentOracle(bagent, env.act_space, disc)
  # AFTER the last make_agent: every agent setup re-arms the guard
  # (embodied/jax/internal.py jax_transfer_guard='disallow'), so setting
  # it earlier is undone by the behavior agent's construction.
  jax.config.update('jax_transfer_guard', 'allow')
  rng = np.random.default_rng(args.seed)
  rows, ref_arrays = label_run(
      env, oracle, args.states, args.horizon,
      args.label_every, args.max_steps, rng,
      run_id=os.path.basename(args.run_logdir.rstrip('/')),
      oracle_all=args.oracle_all, ref_stride=args.ref_stride,
      behavior=behavior)
  save_rows(rows, args.output, meta=dict(
      run_logdir=args.run_logdir, checkpoint=str(ckpt),
      states=args.states, horizon=args.horizon,
      label_every=args.label_every, seed=args.seed,
      train_seed=train_seed, actions=args.actions, rollouts=args.rollouts,
      ref_stride=args.ref_stride, labeler_version='shift_ext_20260723',
      dose=dict(config.distractor),
      behavior_checkpoint=str(args.behavior_checkpoint or ''),
      mass_scale=float(args.mass_scale),
      operation_pair='real_vs_imag_matched_candidate_budget'),
      extra_arrays=ref_arrays)


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
    self._calls = 0

  def init(self):
    return None

  def policy(self, carry, obs, mode='eval'):
    acts = {'action': self.CANDS[1][None]}  # frozen policy plays cand 1
    q = np.array([[0.1, 0.8, 0.0], [0.1, 0.8, 0.0]], np.float32)
    out = {'d0/qfull': q[None], 'd0/cands': self.CANDS[None],
           'd0/udyn': np.array([0.25], np.float32)}
    return carry, acts, out

  def latent_of(self, carry):
    self._calls += 1
    return np.array([self._calls, 0.0], np.float16)


def selfcheck():
  env = _SynthEnv()
  oracle = _SynthOracle()
  rng = np.random.default_rng(0)
  rows, ref = label_run(env, oracle, n_states=5, horizon=10, label_every=7,
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
    # R2 extension: udyn + belief latent stored; bootstrapped G adds the
    # synthetic Vhat(end) = 0.8 to every branch (constant => same delta)
    assert abs(float(r['udyn']) - 0.25) < 1e-6, r['udyn']
    assert r['deter'].shape == (2,) and r['deter'].dtype == np.float16
    assert abs(r['g_now_boot'] - (r['g_now'] + 0.8)) < 1e-5, r
    assert abs(r['delta_real_boot'] - r['delta_real']) < 1e-5, r
    assert abs(r['delta_imag_boot'] - r['delta_imag']) < 1e-5, r
  # base-trajectory latent reference: labels at t=7..35, stride 5 => 8 pts
  assert ref['ref_deter'].shape == (8, 2), ref['ref_deter'].shape
  assert list(ref['ref_step']) == [0, 5, 10, 15, 20, 25, 30, 35]
  assert ref['ref_deter'].dtype == np.float16
  # snapshot/restore roundtrip determinism on the synthetic env
  s = snapshot_env(env)
  before = env.oracle_get_state()
  env.step({'action': np.ones(1), 'reset': False})
  restore_env(env, s)
  assert env.oracle_get_state()[:2] == before[:2]
  out = pathlib.Path(os.environ.get('TMPDIR', '/tmp')) / 'oracle_sc.npz'
  save_rows(rows, str(out), meta=dict(selfcheck=True), extra_arrays=ref)
  data = np.load(out, allow_pickle=False)
  assert data['delta_real'].shape == (5,)
  assert data['udyn'].shape == (5,) and data['deter'].shape == (5, 2)
  assert data['delta_real_boot'].shape == (5,)
  assert data['ref_deter'].shape == (8, 2)

  # --- shift extension: cross-policy behavior drives the trajectory ---
  class _SynthBehavior(_SynthOracle):
    def __init__(self):
      super().__init__()
      self._policy_calls = 0

    def policy(self, carry, obs, mode='eval'):
      self._policy_calls += 1
      carry, acts, out = super().policy(carry, obs, mode)
      return carry, {'action': self.CANDS[0][None]}, out

  env_b = _SynthEnv()
  oracle_b = _SynthOracle()
  beh = _SynthBehavior()
  rows_b, ref_b = label_run(env_b, oracle_b, n_states=5, horizon=10,
                            label_every=7, max_steps=200,
                            rng=np.random.default_rng(0),
                            run_id='synth', behavior=beh)
  assert beh._policy_calls > 30, beh._policy_calls  # behavior drove
  for r in rows_b:
    # eval-agent semantics unchanged on behavior-visited states:
    # beliefs, ops, and rollouts all come from the evaluation oracle
    assert r['m_now'] == 1 and r['m_real'] == 0, r
    assert abs(r['delta_real'] - 0.8) < 1e-5, r['delta_real']
  assert ref_b['ref_deter'].shape == (8, 2)  # ref latents = eval agent

  # --- shift extension: mass-scale plumbing ---
  apply_mass_scale(env_b, 1.0)  # no-op path never touches the chain
  try:
    apply_mass_scale(env_b, 1.3)
    raise AssertionError('mass_scale must refuse envs without physics')
  except SystemExit:
    pass

  class _N:
    pass

  mock, dm, phys, model = _N(), _N(), _N(), _N()
  model.body_mass = np.array([1.0, 2.0])
  phys.model = model
  dm.physics = phys
  mock._dmenv = dm
  apply_mass_scale(mock, 1.5)
  assert np.allclose(model.body_mass, [1.5, 3.0]), model.body_mass
  print('SELFCHECK PASS (incl. R2 extension: udyn/deter/boot/ref; '
        'shift extension: behavior-driven trajectory + mass-scale)')


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
  p.add_argument('--behavior_checkpoint', default='',
                 help='optional second ckpt whose policy DRIVES the '
                      'base trajectory (cross-policy shift arm); the '
                      'main checkpoint stays the evaluation agent')
  p.add_argument('--mass_scale', type=float, default=1.0,
                 help='scale all dm_control body masses at setup '
                      '(physics shift arm); 1.0 = unshifted')
  p.add_argument('--ref_stride', type=int, default=5,
                 help='Record a base-trajectory belief latent every N '
                      'steps (kNN density reference for the R2 read).')
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
