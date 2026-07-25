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
    # episode-done latches (FromDM._done): step() auto-resets when the
    # flag is set, so a branch that ends an episode would silently give
    # the NEXT branch (and the resumed base trajectory) a fresh episode
    # unless the flag is restored with the snapshot.
    done = _own(node, '_done')
    if done is not None:
      snap.setdefault('dones', []).append((node, bool(done)))
    # method lookup on the CLASS: skips instance-dict misses AND the
    # wrapper delegation (stateful wrappers like the OU distractor, plus
    # synthetic/selfcheck envs); a LIST — several chain nodes may carry
    # restorable state.
    if callable(getattr(type(node), 'oracle_get_state', None)):
      snap.setdefault('custom', []).append((node, node.oracle_get_state()))
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
  for node, state in snap.get('custom', ()):
    node.oracle_set_state(state)
  for node, done in snap.get('dones', ()):
    node._done = done
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

  def branch_carry(self, carry_post, cand_vec):
    """Candidate-conditioned belief for a branch: the posterior through
    obs_t (carry_post, returned by d0_eval at the labeled state) with
    the prevact slot replaced by the candidate — so the next policy call
    assimilates obs_{t+1} conditioned on the action actually taken.
    Carry layout: (enc_carry, dyn_carry, dec_carry, prevact)."""
    act = {k: np.asarray(v)[None] for k, v in self.vec2act(cand_vec).items()}
    return (*carry_post[:3], act)

  def rng_mark(self):
    """Policy-RNG position: the outer agent derives each policy call's
    seed from the global n_actions counter (embodied/jax/agent.py), so
    marking/restoring the counter gives every branch the same follower
    seed schedule (true CRN for policy sampling, not just env state)."""
    with self.agent.n_actions.lock:
      return int(self.agent.n_actions.value)

  def rng_reset(self, mark):
    with self.agent.n_actions.lock:
      self.agent.n_actions.value = mark

  def with_prevact(self, carry, acts):
    """Carry with the prevact slot replaced by the (batched) act dict
    actually EXECUTED in the environment. Needed on the cross-policy
    trajectory: the eval agent's own sampled action is counterfactual
    there, and beliefs must be conditioned on the executed action."""
    return (*carry[:3], {k: np.asarray(acts[k]) for k in self.act_keys})


def plugin_choice(qfull):
  """Decision-now rule: argmax over candidates of mean-over-heads Q."""
  return int(np.argmax(qfull.mean(0)))


# --------------------------------------------------------------------------
# Branches. CRN: every branch starts from one env snapshot, one
# candidate-conditioned belief (posterior through obs_t, prevact = the
# branch's candidate), and one policy-RNG mark — env state, belief, AND
# follower sampling noise are common across candidates.
# --------------------------------------------------------------------------

def rollout_return(env, oracle, carry, first_act, horizon, snap,
                   rng_mark=None):
  """G(a): restore, execute a, then frozen policy for `horizon` steps.
  `carry` must be the candidate-conditioned branch carry
  (oracle.branch_carry(carry_post, cand)): posterior through obs_t with
  prevact = a, so the first follower call assimilates obs_{t+1} under
  the action actually taken. Returns (G, G_boot): the plain
  undiscounted return (unchanged Gate-D1 semantics) and the
  value-bootstrapped truncation variant G + Vhat(s_h) from the agent's
  own ensemble at the final state (R3 descriptive companion; G_boot ==
  G when the episode ended before the horizon)."""
  if rng_mark is not None:
    oracle.rng_reset(rng_mark)
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


def op_real(env, oracle, carry_post, obs, qfull, cands, snap,
            rng_mark=None):
  """Real purchase: one real probe step per candidate, re-rank by
  r_real + disc * Vhat(s'_real). Each candidate's next-state value is
  evaluated from that candidate's branch carry (posterior through
  obs_t, prevact = the candidate), under a common policy-RNG mark."""
  m_count = cands.shape[0]
  scores = np.zeros(m_count, np.float32)
  for m in range(m_count):
    if rng_mark is not None:
      oracle.rng_reset(rng_mark)
    restore_env(env, snap)
    bcarry = oracle.branch_carry(carry_post, cands[m])
    nobs = env.step({**oracle.vec2act(cands[m]), 'reset': np.array(False)})
    r = float(nobs['reward'])
    _, qn, _, _, _ = oracle.d0_eval(bcarry, nobs)
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
      carry_post, qfull, cands, _, extras = oracle.d0_eval(carry_s, obs)
      # determinism assert: restore must reproduce the same next obs
      restore_env(env, snap)
      probe1 = env.step({**oracle.vec2act(cands[0]), 'reset': np.array(False)})
      restore_env(env, snap)
      probe2 = env.step({**oracle.vec2act(cands[0]), 'reset': np.array(False)})
      if abs(float(probe1['reward']) - float(probe2['reward'])) > 1e-6:
        raise SystemExit('restore_env is not deterministic; CRN broken')
      restore_env(env, snap)

      m_now = plugin_choice(qfull)
      # op_imag re-assimilates obs_t from carry_s each sample (fresh
      # candidate draws are the point there — no RNG reset).
      m_imag, cost_imag = op_imag(oracle, carry_s, obs, qfull,
                                  budget=cands.shape[0])
      rmark = oracle.rng_mark()
      m_real, cost_real, real_scores = op_real(
          env, oracle, carry_post, obs, qfull, cands, snap,
          rng_mark=rmark)

      targets = {m_now, m_imag, m_real}
      if oracle_all:
        targets = set(range(cands.shape[0]))
      G = {m: rollout_return(env, oracle,
                             oracle.branch_carry(carry_post, cands[m]),
                             oracle.vec2act(cands[m]), horizon, snap,
                             rng_mark=rmark) for m in sorted(targets)}
      g_all = np.full(cands.shape[0], np.nan, np.float32)
      g_all_boot = np.full(cands.shape[0], np.nan, np.float32)
      for m, (gv, gb) in G.items():
        g_all[m] = gv
        g_all_boot[m] = gb
      rows.append(dict(
          run_id=run_id, episode=ep, step=t,
          qfull=qfull, cands=cands,
          g_all=g_all, g_all_boot=g_all_boot,
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
      # d1fix audit: the eval carry's prevact is the eval agent's OWN
      # sampled action, but the trajectory executes the BEHAVIOR action
      # — condition the eval belief on the action actually taken.
      carry = oracle.with_prevact(carry, acts)
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
      ref_stride=args.ref_stride, labeler_version='d1fix_20260724',
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

  # Carry-free synthetic oracle: branch conditioning, prevact
  # substitution, and policy-RNG marking are no-ops (exercised by
  # _RecOracle in the selfcheck instead).
  def branch_carry(self, carry_post, cand_vec):
    return carry_post

  def with_prevact(self, carry, acts):
    return carry

  def rng_mark(self):
    return 0

  def rng_reset(self, mark):
    pass


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
    # per-candidate returns: filled exactly at the computed targets
    # ({m_real=0, m_now=m_imag=1} here), NaN elsewhere
    assert abs(r['g_all'][0] - r['g_real']) < 1e-5, r['g_all']
    assert abs(r['g_all'][1] - r['g_now']) < 1e-5, r['g_all']
    assert np.isnan(r['g_all'][2]), r['g_all']
    assert abs(r['g_all_boot'][0] - r['g_real_boot']) < 1e-5, r
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
  assert data['g_all'].shape == (5, 3)

  # oracle_all: every candidate's return is realized (no NaN cells)
  rows_all, _ = label_run(_SynthEnv(), _SynthOracle(), n_states=2,
                          horizon=10, label_every=7, max_steps=200,
                          rng=np.random.default_rng(0), run_id='synth',
                          oracle_all=True)
  for r in rows_all:
    assert not np.isnan(r['g_all']).any(), r['g_all']
    assert abs(r['g_all'][2] - 0.2 * 9) < 1e-5, r['g_all']  # a=-0.5 path

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

  # --- d1fix: carry-sensitive recurrent mock (candidate branches must
  # see the posterior through obs_t with prevact = the candidate) ---
  class _RecChainEnv:
    """Drifting chain: pos += a + 0.1 per step, obs exposes pos, all
    rewards zero. The drift makes pos_t distinct from pos_{t-1}, so a
    stale (pre-obs_t) carry is provably wrong in the q probe."""

    def __init__(self):
      self.pos, self.t = 0.0, 0

    def oracle_get_state(self):
      return (self.pos, self.t)

    def oracle_set_state(self, s):
      self.pos, self.t = s

    def step(self, act):
      if act.get('reset'):
        self.pos, self.t = 0.0, 0
        return dict(reward=np.float32(0), is_last=np.array(False),
                    pos=np.float32(self.pos))
      a = float(np.asarray(act['action']).reshape(-1)[0])
      self.pos += a + 0.1
      self.t += 1
      return dict(reward=np.float32(0), is_last=np.array(self.t >= 500),
                  pos=np.float32(self.pos))

  class _RecOracle(_SynthOracle):
    """carry = (seen_pos, prevact); q encodes the INCOMING carry
    (100*seen_pos + 10*prevact + obs pos), so a branch evaluated from
    a stale carry (obs_t skipped, or prevact != candidate) produces a
    provably wrong score. The frozen policy plays 0.0, so any nonzero
    prevact in an incoming carry marks a branch-start call."""

    def __init__(self):
      super().__init__()
      self.discount = 1.0
      self.branch_starts = []   # (seen_pos, prevact) at branch entries
      self.calls = []           # every incoming (seen_pos, prevact)
      self.rng_resets = []
      self._ctr = 7             # policy-RNG counter stand-in

    def init(self):
      return (np.float32(-1.0), np.float32(0.0))

    def with_prevact(self, carry, acts):
      return (carry[0],
              np.float32(np.asarray(acts['action']).reshape(-1)[0]))

    def policy(self, carry, obs, mode='eval'):
      self._ctr += 1
      seen, prev = float(carry[0]), float(carry[1])
      self.calls.append((seen, prev))
      if prev != 0.0:
        self.branch_starts.append((seen, prev))
      qv = 100.0 * seen + 10.0 * prev + float(obs['pos'])
      q = np.full((2, 3), qv, np.float32)
      out = {'d0/qfull': q[None], 'd0/cands': self.CANDS[None],
             'd0/udyn': np.array([0.25], np.float32)}
      acts = {'action': np.zeros((1, 1), np.float32)}
      return (np.float32(obs['pos']), np.float32(0.0)), acts, out

    def branch_carry(self, carry_post, cand_vec):
      return (carry_post[0],
              np.float32(np.asarray(cand_vec).reshape(-1)[0]))

    def rng_mark(self):
      return self._ctr

    def rng_reset(self, mark):
      self.rng_resets.append(mark)
      self._ctr = mark

  renv, roracle = _RecChainEnv(), _RecOracle()
  rrows, _ = label_run(renv, roracle, n_states=2, horizon=5,
                       label_every=7, max_steps=500,
                       rng=np.random.default_rng(0), run_id='rec')
  cands = _SynthOracle.CANDS.reshape(-1)
  for r in rrows:
    pos_t = 0.1 * r['step']
    # closed-form probe scores REQUIRE obs_t assimilated (100*pos_t
    # term) and prevact = candidate (10*c term); the stale-carry
    # convention would produce 100*pos_{t-1} + 10*0 instead.
    want = [100.0 * pos_t + 10.0 * c + (pos_t + c + 0.1) for c in cands]
    assert np.allclose(r['real_scores'], want, atol=1e-4), (
        r['step'], r['real_scores'], want)
  # every branch start (3 op_real probes + 1 rollout per label) came in
  # with (seen_pos = pos_t, prevact = candidate)
  assert len(roracle.branch_starts) == 8, roracle.branch_starts
  want_starts = []
  for r in rrows:
    pos_t = 0.1 * r['step']
    want_starts += [(pos_t, float(c)) for c in cands]   # op_real
    want_starts += [(pos_t, float(cands[r['m_real']]))]  # rollout
  for got, want in zip(roracle.branch_starts, want_starts):
    assert np.allclose(got, want, atol=1e-4), (got, want)
  # policy-RNG contract: one reset per branch, all to the mark taken
  # right before op_real (per label), marks distinct across labels
  assert len(roracle.rng_resets) == 8, roracle.rng_resets
  assert len(set(roracle.rng_resets)) == 2, roracle.rng_resets
  assert roracle.rng_resets[:4] == [roracle.rng_resets[0]] * 4
  assert roracle.rng_resets[4:] == [roracle.rng_resets[4]] * 4

  # --- d1fix audit: snapshot covers done-latches and EVERY custom-state
  # node in the chain (was: single 'custom' slot, last writer won) ---
  class _DoneWrap:
    def __init__(self, env):
      self.env = env
      self._done = False
      self.extra = 3

    def oracle_get_state(self):
      return self.extra

    def oracle_set_state(self, s):
      self.extra = s

    def step(self, act):
      return self.env.step(act)

  denv = _DoneWrap(_SynthEnv())
  dsnap = snapshot_env(denv)
  assert len(dsnap['custom']) == 2, dsnap['custom']  # wrapper + inner env
  denv._done = True
  denv.extra = 9
  denv.env.step({'action': np.ones(1), 'reset': False})
  restore_env(denv, dsnap)
  assert denv._done is False and denv.extra == 3, (denv._done, denv.extra)

  # --- d1fix audit: cross-policy trajectory must condition the EVAL
  # belief on the EXECUTED (behavior) action, not the eval agent's own
  # counterfactual sample ---
  class _RecBehavior(_RecOracle):
    def policy(self, carry, obs, mode='eval'):
      carry, acts, out = super().policy(carry, obs, mode)
      return carry, {'action': np.full((1, 1), 0.7, np.float32)}, out

  xoracle = _RecOracle()
  label_run(_RecChainEnv(), xoracle, n_states=1, horizon=5,
            label_every=7, max_steps=500, rng=np.random.default_rng(0),
            run_id='rec', behavior=_RecBehavior())
  n_beh = sum(1 for (s, p) in xoracle.calls if abs(p - 0.7) < 1e-6)
  assert n_beh >= 5, (
      'eval belief not conditioned on executed behavior action',
      xoracle.calls)

  # --- d1fix: real-adapter branch_carry / rng plumbing (structural) ---
  import threading

  class _Ctr:
    def __init__(self):
      self.lock = threading.Lock()
      self.value = 5

  ao = AgentOracle.__new__(AgentOracle)
  ao.agent = _N()
  ao.agent.n_actions = _Ctr()
  ao.act_keys, ao.act_key, ao.act_shape = ['action'], 'action', (1,)
  assert ao.rng_mark() == 5
  ao.rng_reset(11)
  assert ao.agent.n_actions.value == 11
  bc = ao.branch_carry(('e', 'd', 'c', {'action': 'old'}),
                       np.array([0.5], np.float32))
  assert bc[:3] == ('e', 'd', 'c')
  assert bc[3]['action'].shape == (1, 1)
  assert float(bc[3]['action'].reshape(())) == 0.5

  print('SELFCHECK PASS (incl. R2 extension: udyn/deter/boot/ref; '
        'shift extension: behavior-driven trajectory + mass-scale; '
        'd1fix: candidate-conditioned branch carries + policy-RNG CRN)')


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
