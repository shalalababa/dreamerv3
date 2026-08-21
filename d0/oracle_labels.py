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

Cross-checkpoint consumer extension (2026-07-30,
PREREG_r3_amend2_20260730.md): --consumer_checkpoint performs a SECOND,
regex-scoped load ('^(rew|con|valens\\d+)/') AFTER the main
eval-checkpoint load, replacing only the reward/continuation/valens
value heads with the consumer checkpoint's parameters. qfull — hence
m_now, op_imag, and the op_real chooser — then reads the EVAL agent's
features through the CONSUMER's heads, while the follower policy, the
WM (enc/dyn/dec), and disag stay the eval agent's. The RNG counters
the second load would clobber are snapshotted and restored (see
overlay_consumer), so a control pass (consumer = the eval checkpoint
itself, still through the overlay code path) and a swapped pass walk
IDENTICAL base trajectories and produce identical g_all under
--oracle_all: per-state paired contrasts. With the flag empty the
labeler is byte-identical to the d1fix_20260724 path (no meta keys
added, version unchanged); with an overlay active the meta version
becomes d1fix_20260724_xc1 and records consumer_checkpoint +
consumer_regex.

Competence-repair consumer-model extension (2026-07-30,
PREREG_competence_repair_20260730.md): --consumer_model <npz> loads a
LORO ridge model (d0/train_consumer_model.py, frozen feature map
'cmfeat1') and deploys it as an EXTERNAL chooser: at each labeled state
the labeler computes the frozen per-candidate features from quantities
d0_eval already returned (qfull, cands, deter, udyn) and overrides the
realized choice with argmax ghat. Everything else runs UNCHANGED —
op_real still probes every candidate (its own argmax is kept as
m_real_probe, its scores as real_scores), op_imag still runs, and with
--oracle_all (required) the rollout target set is all-M — so the base
trajectory, the policy-RNG schedule, and g_all are bit-identical to a
pass without the flag: the committed R3 labels themselves are the
paired control (the repair reader's determinism gate asserts it; the
chooser is pure numpy — no env step, no policy call, no RNG use).
Meta version becomes d1fix_20260724_cm1 and records the model path,
its sha256, the deployed run key, and the feature-map version; flag
empty = byte-identical default path.

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
import re
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

  def d0_eval(self, carry, obs, mode='eval'):
    """One d0 policy evaluation:
    (carry, qfull [K,M], cands [M,A], act dict, extras). mode='cemplan'
    (wcem labeled states only) additionally computes the CEM plan —
    every other call site keeps the default and pays nothing."""
    carry2, acts, out = self.policy(carry, obs, mode=mode)
    q = np.asarray(out['d0/qfull'][0], np.float32)
    cands = np.asarray(out['d0/cands'][0], np.float32)
    act = {k: np.asarray(acts[k][0]) for k in self.act_keys}
    extras = dict(udyn=float(np.asarray(out['d0/udyn'][0])),
                  deter=self.latent_of(carry2))
    if 'd0/cem_act' in out:
      # wcem consumer wave (PREREG_p2_cem_consumer_20260821): the CEM
      # plan computed in the same policy call. Absent unless
      # config.agent.d0.cem_iters > 0 AND mode == 'cemplan' — default
      # path byte-identical.
      extras['cem_act'] = np.asarray(out['d0/cem_act'][0], np.float32)
      extras['cem_score'] = float(np.asarray(out['d0/cem_score'][0]))
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


class CEMOracle(AgentOracle):
  """AgentOracle whose EXECUTED action is the CEM plan ('d0/cem_act'):
  the follower for the wcem consumer branch
  (PREREG_p2_cem_consumer_20260821). Shares the underlying agent (and
  therefore the n_actions RNG counter) with the eval oracle, so W1
  marks give the CEM follower the same CRN discipline as policy
  sampling. The belief carry's prevact slot is replaced by the CEM
  action actually executed."""

  def policy(self, carry, obs, mode='eval'):
    # Always plan: the CEM follower re-plans at every step (mode
    # 'cemplan' is what unlocks _cem_signals in the jitted policy —
    # review finding 2).
    carry, acts, out = super().policy(carry, obs, 'cemplan')
    assert 'd0/cem_act' in out, (
        'CEMOracle requires config.agent.d0.cem_iters > 0')
    cem = {self.act_key: np.asarray(out['d0/cem_act'], np.float32)}
    return (*carry[:3], cem), cem, out


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
            rng_mark=None, return_rewards=False):
  """Real purchase: one real probe step per candidate, re-rank by
  r_real + disc * Vhat(s'_real). Each candidate's next-state value is
  evaluated from that candidate's branch carry (posterior through
  obs_t, prevact = the candidate), under a common policy-RNG mark.

  return_rewards (xc Amendment 1): additionally return the raw
  per-candidate env rewards — the head-INDEPENDENT part of the score —
  so a dual-chooser replay can machine-check that its env work is
  identical to the control's. Default path byte-identical."""
  m_count = cands.shape[0]
  scores = np.zeros(m_count, np.float32)
  rewards = np.zeros(m_count, np.float32)
  for m in range(m_count):
    if rng_mark is not None:
      oracle.rng_reset(rng_mark)
    restore_env(env, snap)
    bcarry = oracle.branch_carry(carry_post, cands[m])
    nobs = env.step({**oracle.vec2act(cands[m]), 'reset': np.array(False)})
    r = float(nobs['reward'])
    rewards[m] = r
    _, qn, _, _, _ = oracle.d0_eval(bcarry, nobs)
    vhat = float(qn.mean(0).max())
    scores[m] = r + oracle.discount * vhat
  cost = dict(env_steps=m_count, policy_calls=m_count)
  if return_rewards:
    return int(np.argmax(scores)), cost, scores, rewards
  return int(np.argmax(scores)), cost, scores


# --------------------------------------------------------------------------
# Labeling sweep
# --------------------------------------------------------------------------

DUAL_CANDS_ATOL = 1e-4   # xc Amendment 1: max |cands_shadow - cands|
                         # (action units) — the shadow chooser must rank
                         # the SAME candidate set g_all was realized for

# W1 adjudication wave (PREREG_w1_adjudication_constraints_20260809):
# disjoint policy-RNG counter offsets. Each G rollout consumes at most
# `horizon` policy calls and the probe at most M, so 5e4/1e5 spacings
# guarantee non-overlapping follower seed schedules: the probe mark is
# independent of every evaluation mark (constraint 5), and each repeat
# r uses its own mark (constraint 1) with CRN across branches WITHIN a
# repeat (same mark_r for all candidates + the actor branch).
W1_PROBE_OFFSET = 50_000
W1_MARK_OFFSET = 100_000


def seed_env_task(env, seed):
  """Seed the dm_control task RNG explicitly (W1 constraint 6).

  The default config path never forwards a seed (`use_seed` is absent
  from the dmc env block), so `suite.load` draws OS entropy and no two
  labeler invocations share a physical trajectory. This walks the
  wrapper chain to the dm_control env and replaces the task RNG before
  the first reset. Returns the seed for meta stamping."""
  import numpy as _np
  for node in _chain(env):
    dmenv = _own(node, '_dmenv')
    if dmenv is not None and hasattr(dmenv, 'task'):
      dmenv.task._random = _np.random.RandomState(int(seed))
      return int(seed)
  raise SystemExit('seed_env_task: no dm_control env found in the '
                   'wrapper chain — refusing to run unseeded (W1 '
                   'constraint 6)')


def _w1_label_state(env, oracle, carry_post, obs, qfull, cands, snap,
                    m_now, extras, horizon, run_id, ep, t, w1, act0):
  """W1 adjudication labeling of ONE state
  (PREREG_w1_adjudication_constraints_20260809).

  Constraint 1: every candidate (and the actor branch) is evaluated
  R times from the same restored snapshot; repeat r uses follower mark
  base + (r+1)*W1_MARK_OFFSET — CRN across branches WITHIN a repeat,
  independent follower schedules ACROSS repeats.
  Constraint 3: dup_cand replaces the candidate set with M copies of
  the plug-in choice's candidate (true opportunity exactly 0).
  Constraint 4: the deployed actor's SAMPLED action at this state is
  stored and evaluated as its own branch under every repeat.
  Constraint 5: op_real runs under base + W1_PROBE_OFFSET — a mark
  disjoint from every evaluation mark — and per-candidate r_real is
  stored (leak decomposition happens in the frozen reader).
  """
  base = oracle.rng_mark()
  # Deployed-actor branch (batch-review B1): the action is act0 — the
  # sample d0_eval ALREADY drew from the deployed belief (carry_s +
  # obs_t) and the default path discards (review D3). Re-sampling from
  # carry_post here would double-assimilate obs_t (a belief the
  # deployed agent never holds). The no-op reset keeps the per-state
  # base mark as the selfcheck parser's delimiter.
  oracle.rng_reset(base)
  actor_vec = np.asarray(act0[oracle.act_key], np.float32).reshape(-1)
  cands_eval = np.asarray(cands, np.float32)
  if w1.get('dup_cand'):
    cands_eval = np.tile(cands_eval[m_now:m_now + 1],
                         (cands_eval.shape[0], 1))
  m_count = cands_eval.shape[0]
  # Probe under the INDEPENDENT mark (de-shared from all G marks).
  m_real, cost_real, real_scores, r_real = op_real(
      env, oracle, carry_post, obs, qfull, cands_eval, snap,
      rng_mark=base + W1_PROBE_OFFSET, return_rewards=True)
  repeats = int(w1['repeats'])
  g_rep = np.zeros((repeats, m_count), np.float32)
  g_actor_rep = np.zeros(repeats, np.float32)
  for r in range(repeats):
    mark_r = base + (r + 1) * W1_MARK_OFFSET
    for m in range(m_count):
      g_rep[r, m] = rollout_return(
          env, oracle, oracle.branch_carry(carry_post, cands_eval[m]),
          oracle.vec2act(cands_eval[m]), horizon, snap,
          rng_mark=mark_r)[0]
    g_actor_rep[r] = rollout_return(
        env, oracle, oracle.branch_carry(carry_post, actor_vec),
        oracle.vec2act(actor_vec), horizon, snap, rng_mark=mark_r)[0]
  # Leave the counter past every mark this state consumed, so the base
  # trajectory's schedule cannot collide with any branch schedule.
  oracle.rng_reset(base + (repeats + 1) * W1_MARK_OFFSET)
  return dict(
      run_id=run_id, episode=ep, step=t,
      qfull=qfull, cands=cands_eval,
      g_all_rep=g_rep, g_actor_rep=g_actor_rep, actor_act=actor_vec,
      udyn=np.float32(extras['udyn']),
      deter=np.asarray(extras['deter'], np.float16),
      m_now=m_now, m_real=m_real,
      real_scores=np.asarray(real_scores, np.float32),
      r_real=np.asarray(r_real, np.float32),
      dup_cand=bool(w1.get('dup_cand', False)),
      cost_real_env=cost_real['env_steps'],
      cost_real_calls=cost_real['policy_calls'])


def _wdc_label_state(env, oracle, carry_post, obs, qfull, cands, snap,
                     m_now, extras, horizon, run_id, ep, t, w1, act0):
  """WDC diverse-candidate labeling of ONE state
  (PREREG_p2_dcand_20260821). Additive on the W1 protocol (same mark
  discipline, same probe, same actor branch, same dup semantics), plus:

  - U UNIFORM candidates dcands ~ U[-1,1]^A drawn from the pinned
    per-state stream default_rng([env_seed, ep, t]) — deterministic and
    reproducible by the reader (a wiring gate recomputes them);
  - each uniform candidate is evaluated under the SAME repeat marks as
    the policy candidates (CRN across ALL branches within a repeat);
  - a candidate-0 WITNESS re-run per repeat: the same action under the
    same mark and snapshot must reproduce g_all_rep[:, 0] bit-exactly,
    else CRN is broken and the pass aborts (fail-fast; the reader
    re-checks the stored arrays).

  Under dup_cand the uniform set is ALSO replaced by copies of the
  plug-in candidate, so every estimand is exactly zero (the W1
  duplicate-null gate extends to the diverse set)."""
  base = oracle.rng_mark()
  oracle.rng_reset(base)
  actor_vec = np.asarray(act0[oracle.act_key], np.float32).reshape(-1)
  cands_eval = np.asarray(cands, np.float32)
  if w1.get('dup_cand'):
    cands_eval = np.tile(cands_eval[m_now:m_now + 1],
                         (cands_eval.shape[0], 1))
  m_count = cands_eval.shape[0]
  n_uni = int(w1['dcand_uniform'])
  drng = np.random.default_rng(
      [int(w1['env_seed']), int(ep), int(t)])
  dcands = drng.uniform(-1.0, 1.0,
                        (n_uni, cands_eval.shape[1])).astype(np.float32)
  if w1.get('dup_cand'):
    dcands = np.tile(cands_eval[:1], (n_uni, 1))
  m_real, cost_real, real_scores, r_real = op_real(
      env, oracle, carry_post, obs, qfull, cands_eval, snap,
      rng_mark=base + W1_PROBE_OFFSET, return_rewards=True)
  repeats = int(w1['repeats'])
  g_rep = np.zeros((repeats, m_count), np.float32)
  g_actor_rep = np.zeros(repeats, np.float32)
  g_dcand_rep = np.zeros((repeats, n_uni), np.float32)
  g_wit_rep = np.zeros(repeats, np.float32)
  for r in range(repeats):
    mark_r = base + (r + 1) * W1_MARK_OFFSET
    for m in range(m_count):
      g_rep[r, m] = rollout_return(
          env, oracle, oracle.branch_carry(carry_post, cands_eval[m]),
          oracle.vec2act(cands_eval[m]), horizon, snap,
          rng_mark=mark_r)[0]
    g_actor_rep[r] = rollout_return(
        env, oracle, oracle.branch_carry(carry_post, actor_vec),
        oracle.vec2act(actor_vec), horizon, snap, rng_mark=mark_r)[0]
    for u in range(n_uni):
      g_dcand_rep[r, u] = rollout_return(
          env, oracle, oracle.branch_carry(carry_post, dcands[u]),
          oracle.vec2act(dcands[u]), horizon, snap, rng_mark=mark_r)[0]
    g_wit_rep[r] = rollout_return(
        env, oracle, oracle.branch_carry(carry_post, cands_eval[0]),
        oracle.vec2act(cands_eval[0]), horizon, snap, rng_mark=mark_r)[0]
    if g_wit_rep[r] != g_rep[r, 0]:
      raise SystemExit(
          f'WDC CRN WITNESS VIOLATION at episode {ep} step {t} repeat '
          f'{r}: candidate-0 re-run {g_wit_rep[r]} != g_all_rep '
          f'{g_rep[r, 0]} — same action/mark/snapshot must be '
          'bit-identical; pass aborts')
  oracle.rng_reset(base + (repeats + 1) * W1_MARK_OFFSET)
  return dict(
      run_id=run_id, episode=ep, step=t,
      qfull=qfull, cands=cands_eval,
      g_all_rep=g_rep, g_actor_rep=g_actor_rep, actor_act=actor_vec,
      dcands=dcands, g_dcand_rep=g_dcand_rep, g_wit_rep=g_wit_rep,
      udyn=np.float32(extras['udyn']),
      deter=np.asarray(extras['deter'], np.float16),
      m_now=m_now, m_real=m_real,
      real_scores=np.asarray(real_scores, np.float32),
      r_real=np.asarray(r_real, np.float32),
      dup_cand=bool(w1.get('dup_cand', False)),
      cost_real_env=cost_real['env_steps'],
      cost_real_calls=cost_real['policy_calls'])


def _wcem_label_state(env, oracle, cem_oracle, carry_post, obs, qfull,
                      cands, snap, m_now, extras, horizon, run_id, ep, t,
                      w1, act0):
  """WCEM consumer labeling of ONE state
  (PREREG_p2_cem_consumer_20260821). The W1 protocol plus one extra
  branch per repeat: the CEM CONSUMER — first action = the CEM plan
  computed at the labeled state (extras['cem_act'], same policy call as
  qfull), followers = the CEM oracle (re-plans every step) — under the
  SAME repeat marks as every other branch, so g_cem_rep − g_actor_rep
  is a CRN-paired consumer contrast."""
  base = oracle.rng_mark()
  oracle.rng_reset(base)
  actor_vec = np.asarray(act0[oracle.act_key], np.float32).reshape(-1)
  cem_vec = np.asarray(extras['cem_act'], np.float32).reshape(-1)
  cands_eval = np.asarray(cands, np.float32)
  if w1.get('dup_cand'):
    cands_eval = np.tile(cands_eval[m_now:m_now + 1],
                         (cands_eval.shape[0], 1))
  m_count = cands_eval.shape[0]
  m_real, cost_real, real_scores, r_real = op_real(
      env, oracle, carry_post, obs, qfull, cands_eval, snap,
      rng_mark=base + W1_PROBE_OFFSET, return_rewards=True)
  repeats = int(w1['repeats'])
  g_rep = np.zeros((repeats, m_count), np.float32)
  g_actor_rep = np.zeros(repeats, np.float32)
  g_cem_rep = np.zeros(repeats, np.float32)
  g_wit_rep = np.zeros(repeats, np.float32)
  for r in range(repeats):
    mark_r = base + (r + 1) * W1_MARK_OFFSET
    for m in range(m_count):
      g_rep[r, m] = rollout_return(
          env, oracle, oracle.branch_carry(carry_post, cands_eval[m]),
          oracle.vec2act(cands_eval[m]), horizon, snap,
          rng_mark=mark_r)[0]
    g_actor_rep[r] = rollout_return(
        env, oracle, oracle.branch_carry(carry_post, actor_vec),
        oracle.vec2act(actor_vec), horizon, snap, rng_mark=mark_r)[0]
    g_cem_rep[r] = rollout_return(
        env, cem_oracle, cem_oracle.branch_carry(carry_post, cem_vec),
        cem_oracle.vec2act(cem_vec), horizon, snap, rng_mark=mark_r)[0]
    # CRN witness (review finding 6, mirroring _wdc_label_state): the
    # CEM path introduces new XLA work inside follower calls — this
    # clears it: same action/mark/snapshot must be bit-identical.
    g_wit_rep[r] = rollout_return(
        env, oracle, oracle.branch_carry(carry_post, cands_eval[0]),
        oracle.vec2act(cands_eval[0]), horizon, snap,
        rng_mark=mark_r)[0]
    if g_wit_rep[r] != g_rep[r, 0]:
      raise SystemExit(
          f'WCEM CRN WITNESS VIOLATION at episode {ep} step {t} repeat '
          f'{r}: candidate-0 re-run {g_wit_rep[r]} != g_all_rep '
          f'{g_rep[r, 0]} — pass aborts')
  oracle.rng_reset(base + (repeats + 1) * W1_MARK_OFFSET)
  return dict(
      run_id=run_id, episode=ep, step=t,
      qfull=qfull, cands=cands_eval,
      g_all_rep=g_rep, g_actor_rep=g_actor_rep, actor_act=actor_vec,
      g_cem_rep=g_cem_rep, cem_act=cem_vec, g_wit_rep=g_wit_rep,
      cem_score=np.float32(extras['cem_score']),
      q_best=np.float32(qfull.mean(0).max()),
      udyn=np.float32(extras['udyn']),
      deter=np.asarray(extras['deter'], np.float16),
      m_now=m_now, m_real=m_real,
      real_scores=np.asarray(real_scores, np.float32),
      r_real=np.asarray(r_real, np.float32),
      dup_cand=bool(w1.get('dup_cand', False)),
      cost_real_env=cost_real['env_steps'],
      cost_real_calls=cost_real['policy_calls'])


def label_run(env, oracle, n_states, horizon, label_every, max_steps,
              rng, run_id, oracle_all=False, ref_stride=5,
              behavior=None, consumer=None, dual=None, w1=None,
              cem_oracle=None):
  """behavior: optional second AgentOracle that DRIVES the base
  trajectory (state visitation) while `oracle` remains the evaluation
  agent for beliefs, d0 evals, operations, and rollouts — the
  cross-policy shift arm. With behavior=None this is bit-identical to
  the r2_ext labeler path.

  consumer: optional external chooser (competence repair,
  d0/train_consumer_model.make_chooser): (qfull, cands, deter, udyn)
  -> (m_hat, ghat). When set, the realized choice becomes argmax ghat;
  op_real still runs unchanged (its argmax is kept as m_real_probe).
  The chooser is pure numpy — no env step, no policy call, no RNG use
  — so the base trajectory and g_all are bit-identical to a
  consumer=None pass; only the saved row changes.

  dual (xc Amendment 1, PREREG_r3_xc_amend1_20260802): an object with
  .to_consumer()/.to_control() head-swap methods. Per labeled state
  the choosers are evaluated TWICE — once under the eval agent's own
  heads (control: m_now, m_real) and once under the overlaid consumer
  heads (shadow: m_now_x, m_real_x) — on the SAME states, the SAME
  candidate set, and the SAME g_all, entirely within this invocation:
  the shadow d0_eval replays the control's RNG counter window (candidate
  identity gated at DUAL_CANDS_ATOL, counter-consumption asserted) and
  the shadow op_real replays the control's rng_mark (per-candidate env
  rewards gated at 1e-6). The RNG schedule seen by the base trajectory
  and the G rollouts is IDENTICAL to a dual=None pass, and g_all is
  computed once, from the control pass, for the control candidates."""
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
      c0 = oracle.rng_mark() if dual is not None else None
      carry_post, qfull, cands, act0, extras = oracle.d0_eval(
          carry_s, obs,
          mode='cemplan' if (w1 or {}).get('cem') else 'eval')
      # determinism assert: restore must reproduce the same next obs
      restore_env(env, snap)
      probe1 = env.step({**oracle.vec2act(cands[0]), 'reset': np.array(False)})
      restore_env(env, snap)
      probe2 = env.step({**oracle.vec2act(cands[0]), 'reset': np.array(False)})
      if abs(float(probe1['reward']) - float(probe2['reward'])) > 1e-6:
        raise SystemExit('restore_env is not deterministic; CRN broken')
      restore_env(env, snap)

      m_now = plugin_choice(qfull)
      if w1 is not None:
        assert dual is None and consumer is None and behavior is None, (
            'W1 passes are registered standalone (PREREG_w1_'
            'adjudication_constraints_20260809): no dual/consumer/'
            'behavior overlay')
        assert bool(w1.get('cem')) == (cem_oracle is not None), (
            'w1.cem and cem_oracle must be set together')
        if w1.get('cem'):
          rows.append(_wcem_label_state(
              env, oracle, cem_oracle, carry_post, obs, qfull, cands,
              snap, m_now, extras, horizon, run_id, ep, t, w1, act0))
        else:
          w1_fn = (_wdc_label_state if w1.get('dcand_uniform')
                   else _w1_label_state)
          rows.append(w1_fn(
              env, oracle, carry_post, obs, qfull, cands, snap, m_now,
              extras, horizon, run_id, ep, t, w1, act0))
        restore_env(env, snap)  # resume the base trajectory untouched
      else:
        if dual is not None:
          # Shadow plugin chooser: replay the control d0_eval's RNG
          # window under the consumer heads. The policy net is outside
          # the overlay regex, so the candidate DRAWS repeat (same seeds)
          # up to float jitter — gated below; the Q values differ only
          # through the swapped heads (+ the same sample-noise class the
          # registered two-pass design had per pass).
          c1 = oracle.rng_mark()
          oracle.rng_reset(c0)
          dual.to_consumer()
          _, qfull_x, cands_x, _, _ = oracle.d0_eval(carry_s, obs)
          dual.to_control()
          if oracle.rng_mark() != c1:
            raise SystemExit(
                'dual shadow d0_eval consumed a different RNG count '
                f'({oracle.rng_mark()} != {c1}) — replay schedule broken')
          cdiff = float(np.max(np.abs(
              np.asarray(cands_x, np.float64) - np.asarray(cands,
                                                           np.float64))))
          if cdiff > DUAL_CANDS_ATOL:
            raise SystemExit(
                f'dual-chooser candidate drift {cdiff:.3e} > '
                f'{DUAL_CANDS_ATOL} at episode {ep} step {t}: the shadow '
                'chooser is not ranking the g_all candidate set — '
                'instrument invalid on this substrate')
          m_now_x = plugin_choice(qfull_x)
        # op_imag re-assimilates obs_t from carry_s each sample (fresh
        # candidate draws are the point there — no RNG reset).
        m_imag, cost_imag = op_imag(oracle, carry_s, obs, qfull,
                                    budget=cands.shape[0])
        rmark = oracle.rng_mark()
        if dual is not None:
          m_real, cost_real, real_scores, r_ctl = op_real(
              env, oracle, carry_post, obs, qfull, cands, snap,
              rng_mark=rmark, return_rewards=True)
          # Shadow op_real: SAME candidates, same env snapshot, same
          # rng_mark — only the value bootstrap reads the consumer heads.
          dual.to_consumer()
          m_real_x, _, real_scores_x, r_x = op_real(
              env, oracle, carry_post, obs, qfull_x, cands, snap,
              rng_mark=rmark, return_rewards=True)
          dual.to_control()
          if not np.allclose(r_ctl, r_x, atol=1e-6):
            raise SystemExit(
                f'dual-chooser env-reward drift across the op_real '
                f'replays at episode {ep} step {t}: '
                f'{np.max(np.abs(r_ctl - r_x)):.3e} > 1e-6 — CRN broken')
        else:
          m_real, cost_real, real_scores = op_real(
              env, oracle, carry_post, obs, qfull, cands, snap,
              rng_mark=rmark)
        m_probe, ghat = m_real, None
        if consumer is not None:
          # External chooser override AFTER all env/policy work of this
          # state: influences nothing downstream except the saved row
          # (with oracle_all the rollout target set is all-M regardless).
          m_hat, ghat = consumer(qfull, cands, extras['deter'],
                                 float(extras['udyn']))
          m_real = int(m_hat)

        targets = {m_now, m_imag, m_real}
        if consumer is not None:
          targets.add(m_probe)
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
        row = dict(
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
        )
        if consumer is not None:
          row.update(m_real_probe=m_probe,
                     ghat=np.asarray(ghat, np.float32))
        if dual is not None:
          row.update(m_now_x=m_now_x, m_real_x=m_real_x,
                     qfull_x=qfull_x,
                     real_scores_x=real_scores_x,
                     cands_max_absdiff=np.float32(cdiff))
        rows.append(row)
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
  # Value-blindness: consumer-arm passes (overlay '_xc1' / external
  # chooser '_cm1') must not print estimand means — delta_real IS the
  # registered per-state achieved, and paired stdout means would reveal
  # the primaries before the ONE read. Default path byte-identical.
  version = str(meta.get('labeler_version', ''))
  if version.endswith(('_xc1', '_xc2', '_cm1', '_w1', '_w2', '_wv1',
                       '_wdc', '_w1sb', '_wcem')):
    # '_wv1' added 21 Aug (PREREG_p2_wave1_firstupdate build-review B1):
    # wv1 rows carry no delta_real, so the else-branch both crashes AND
    # would be the exact leak this guard exists for. Additive: no
    # pre-existing version string ends '_wv1'. '_wdc' (diverse-candidate
    # wave, PREREG_p2_dcand_20260821) and '_w1sb' (Stage B stamped wave,
    # PREREG_p2_stageb_injection_20260821) added 21 Aug for the same two
    # reasons. Default path byte-identical.
    reason = ('WAVE-1 pass' if version.endswith('_wv1') else
              'WDC pass' if version.endswith('_wdc') else
              'WCEM pass' if version.endswith('_wcem') else
              'Stage-B stamped pass' if version.endswith('_w1sb') else
              'W1 pass' if version.endswith('_w1') else
              'W2 pass' if version.endswith('_w2') else 'consumer arm')
    print(f'wrote {output}: {len(rows)} labeled states '
          f'(estimand summary redacted: {reason})')
  else:
    print(f'wrote {output}: {len(rows)} labeled states; '
          f'mean delta_real {cols["delta_real"].mean():+.3f}, '
          f'mean delta_imag {cols["delta_imag"].mean():+.3f}')


# --------------------------------------------------------------------------
# Cross-checkpoint consumer overlay (PREREG_r3_amend2_20260730.md)
# --------------------------------------------------------------------------

# Exactly the heads _d0_signals consults for qfull (dreamerv3/agent.py:
# rew + con + the valens ensemble). Deliberately EXCLUDES pol, enc, dyn,
# dec, disag (trajectories, beliefs, and udyn stay the eval agent's),
# the main critic 'val', the slowvalens targets, and any normalizer.
CONSUMER_REGEX = r'^(rew|con|valens\d+)/'


def stamp_consumer(meta, consumer_checkpoint, max_steps=None, dual=False,
                   dual_allow_identical=False):
  """Version/provenance stamping for the overlay. The default path is
  byte-identical (same dict back: no new keys, version unchanged, so
  every existing frozen reader keeps matching); an active overlay —
  including the control arm, which passes the eval checkpoint itself —
  suffixes the version with '_xc1' and records the consumer checkpoint,
  regex, and max_steps (max_steps gates the labeled-state window, so a
  pass re-run with a different value would label a different window
  while staying internally consistent — the xc reader pins it). The xc
  reader pins the exact suffixed version string. dual (xc Amendment 1)
  stamps '_xc2' instead plus the dual fields; the amended reader pins
  '_xc2' exactly, so plain '_xc1' two-pass files can never enter the
  amended wave."""
  if not consumer_checkpoint:
    return meta
  stamped = dict(
      meta,
      labeler_version=meta['labeler_version'] + ('_xc2' if dual
                                                 else '_xc1'),
      consumer_checkpoint=str(consumer_checkpoint),
      consumer_regex=CONSUMER_REGEX)
  if dual:
    stamped['dual_chooser'] = True
    stamped['dual_cands_atol'] = DUAL_CANDS_ATOL
    stamped['dual_allow_identical'] = bool(dual_allow_identical)
  if max_steps is not None:
    stamped['max_steps'] = int(max_steps)
  return stamped


def _flat_cfg(d, prefix=''):
  """Flatten a nested config dict to dotted keys (local, jax-free
  equivalent of d0.sweep._flat for the selfcheck)."""
  out = {}
  for k, v in d.items():
    key = f'{prefix}{k}'
    if isinstance(v, dict):
      out.update(_flat_cfg(v, key + '.'))
    else:
      out[key] = v
  return out


def consumer_config_mismatches(eval_flat, cons_flat):
  """Arch-relevant mismatches between the eval and consumer runs' saved
  configs (flattened). The regex load path (embodied/jax/agent.py)
  skips the full-tree shape asserts, so head-architecture agreement is
  enforced HERE, manually and loudly: task (obs/act spaces) plus the
  entire agent.* subtree (sizes, head layouts, valens.k) must agree;
  run-identity keys (seed, logdir, ...) are irrelevant and ignored.
  Additionally both sides must have agent.valnorm.impl == 'none' — the
  overlay moves heads WITHOUT any value-normalizer state, which is only
  sound while the normalizer is the identity. Returns a list of
  human-readable mismatch strings (empty = compatible)."""
  keys = {k for k in list(eval_flat) + list(cons_flat)
          if k == 'task' or k.startswith('agent.')}
  out = []
  for k in sorted(keys):
    a = eval_flat.get(k, '<absent>')
    b = cons_flat.get(k, '<absent>')
    if a != b:
      out.append(f'{k}: eval={a!r} consumer={b!r}')
  for tag, flat in (('eval', eval_flat), ('consumer', cons_flat)):
    impl = flat.get('agent.valnorm.impl')
    if impl != 'none':
      out.append(
          f"{tag}: agent.valnorm.impl={impl!r} != 'none' (overlaid heads "
          'would detach from a learned value normalizer)')
  return out


def check_consumer_config(eval_run_logdir, consumer_ckpt):
  """File-level guard: the consumer checkpoint must sit inside a run dir
  whose config.yaml exists and is arch-compatible with the eval run's.
  Errors loudly on any mismatch; never labels on unverified heads."""
  cons_run = os.path.dirname(os.path.abspath(str(consumer_ckpt)).rstrip('/'))
  cons_cfg = os.path.join(cons_run, 'config.yaml')
  eval_cfg = os.path.join(eval_run_logdir, 'config.yaml')
  if not os.path.exists(cons_cfg):
    raise SystemExit(
        f'--consumer_checkpoint: no config.yaml next to it ({cons_cfg}); '
        'refusing to overlay unverified heads')
  if not os.path.exists(eval_cfg):
    raise SystemExit(f'eval run has no config.yaml ({eval_cfg})')
  import ruamel.yaml as yaml
  from d0.sweep import _backfill_defaults
  with open(eval_cfg) as f:
    ev = _backfill_defaults(yaml.YAML(typ='safe').load(f))
  with open(cons_cfg) as f:
    cn = _backfill_defaults(yaml.YAML(typ='safe').load(f))
  mism = consumer_config_mismatches(_flat_cfg(ev), _flat_cfg(cn))
  if mism:
    raise SystemExit(
        '--consumer_checkpoint: arch-relevant config mismatch between '
        f'{eval_run_logdir} and {cons_run}:\n  ' + '\n  '.join(mism))


def overlay_consumer(agent, consumer_ckpt, load_fn=None):
  """Second, regex-scoped load AFTER the main eval-checkpoint load.

  JAXAgent.load overwrites n_updates/n_batches/n_actions UNCONDITIONALLY
  from whichever checkpoint it loads (embodied/jax/agent.py:364-371),
  and every policy call's seed derives from (config.seed, n_actions)
  (agent.py:231-233, 405-408) — so without repair a control pass
  (consumer = eval ckpt) and a swapped pass (consumer = the other
  maturity) would start from DIFFERENT counter values and walk different
  base trajectories. The counters are therefore snapshotted after the
  main load and restored after the overlay: base trajectories and
  oracle rollouts depend only on pol + counters + env RNG, none of
  which the overlaid heads touch, which is what makes paired passes
  state-identical (asserted by the reader's pairing guard).

  load_fn is an injection point for the selfcheck; the default performs
  the elements checkpoint load of key 'agent' with regex=CONSUMER_REGEX
  (the embodied/run/train.py from_checkpoint bind pattern), resolving a
  latest-file checkpoint dir the same way d0.sweep.load_frozen_agent
  does. Returns the restored counter marks (for logging)."""
  names = ('n_updates', 'n_batches', 'n_actions')
  marks = {}
  for name in names:
    ctr = getattr(agent, name)
    with ctr.lock:
      marks[name] = int(ctr.value)
  if load_fn is None:
    def load_fn(path):
      import elements
      from functools import partial as bind
      if not os.path.exists(os.path.join(path, 'done')):
        latest = os.path.join(path, 'latest')
        if not os.path.exists(latest):
          raise SystemExit(
              f'--consumer_checkpoint: {path} is neither a completed save '
              'folder (no done file) nor a checkpoint dir (no latest file)')
        with open(latest) as f:
          path = os.path.join(path, f.read().strip())
      elements.checkpoint.load(
          path, dict(agent=bind(agent.load, regex=CONSUMER_REGEX)))
  load_fn(consumer_ckpt)
  # No-op-overlay guard: JAXAgent.load's regex path silently updates
  # nothing when zero keys match (the non-empty assert runs BEFORE
  # filtering), and a silent no-op would make control and swapped
  # passes byte-identical — fabricating the registered HEADS-IRRELEVANT
  # prediction with every guard green. Refuse unless the live param set
  # contains matched keys covering every consulted head family.
  params = getattr(agent, 'params', None)
  if params is not None:
    keys = list(params.keys())
    matched = [k for k in keys if re.match(CONSUMER_REGEX, k)]
    assert matched, (
        '--consumer_checkpoint: overlay regex matched ZERO live param '
        f'keys (regex {CONSUMER_REGEX}); param-layout drift — refusing '
        'rather than run a silent no-op overlay')
    for prefix in ('rew/', 'con/', 'valens0'):
      assert any(k.startswith(prefix) for k in matched), (
          f'--consumer_checkpoint: overlay covers no {prefix!r} key '
          f'(matched {sorted(matched)[:6]}...); param-layout drift — '
          'refusing rather than run a partial overlay')
  for name in names:
    ctr = getattr(agent, name)
    with ctr.lock:
      ctr.value = marks[name]
  print(f'Consumer overlay loaded from {consumer_ckpt} '
        f'(regex {CONSUMER_REGEX}); counters restored to {marks}.')
  return marks


class DualHeads:
  """xc Amendment 1 (PREREG_r3_xc_amend1_20260802): cached control and
  consumer head-param sets with a per-state swap into the LIVE policy
  store, so BOTH choosers are evaluated inside one invocation (the
  detprobe killed cross-invocation pairing: g_all max|drift| 95 on
  identical states, artifacts/r3_xc_detprobe_20260801/).

  Built BEFORE overlay_consumer runs (captures the eval agent's own
  heads), completed after it (captures the overlaid consumer heads),
  then reset to control — the pass runs under the eval agent's heads
  and swaps only around the shadow chooser evaluations. Swaps write
  into agent.policy_params — the store every policy/d0_eval call
  consults (embodied/jax/agent.py policy()) — AND agent.params for
  coherence, with a read-back assert on a witness key: a swap that
  silently fails to reach the consulted store would fabricate chooser
  agreement (the registered HEADS-IRRELEVANT prediction), so it
  refuses instead. Two cached head sets at MLP-head size are trivial.
  """

  def __init__(self, agent):
    import jax
    if jax.process_count() != 1:
      raise SystemExit('dual-chooser head swap is single-process only '
                       '(device_get/put on globally sharded params '
                       'would silently truncate)')
    self.agent = agent
    self.matched = sorted(
        k for k in agent.params if re.match(CONSUMER_REGEX, k))
    if not self.matched:
      raise SystemExit('dual-chooser: overlay regex matched no keys')
    self.ctl = self._snapshot()
    self.sw = None

  def _store(self):
    pp = getattr(self.agent, 'policy_params', None)
    return pp if pp is not None else self.agent.params

  def _snapshot(self):
    import jax
    with jax._src.config.explicit_device_get_scope():
      return {k: np.asarray(jax.device_get(self._store()[k]))
              for k in self.matched}

  def capture_consumer(self, allow_identical=False):
    self.sw = self._snapshot()
    differs = any(not np.array_equal(self.ctl[k], self.sw[k])
                  for k in self.matched)
    if not differs and not allow_identical:
      raise SystemExit(
          'dual-chooser: consumer and control head params are '
          'byte-identical — a shadow chooser cannot differ (consumer '
          'checkpoint == eval checkpoint?); refusing to fabricate '
          'chooser agreement. --dual_allow_identical is reserved for '
          'the registered identical-heads smoke.')

  def _apply(self, host):
    import jax
    for store in (self._store(), self.agent.params):
      for k in self.matched:
        store[k] = jax.device_put(host[k], store[k].sharding)
    witness = self.matched[0]
    with jax._src.config.explicit_device_get_scope():
      back = np.asarray(jax.device_get(self._store()[witness]))
    if not np.array_equal(back, host[witness]):
      raise SystemExit(
          'dual-chooser: head swap did not reach the live policy store')

  def to_consumer(self):
    assert self.sw is not None, 'capture_consumer() has not run'
    self._apply(self.sw)

  def to_control(self):
    self._apply(self.ctl)


# --------------------------------------------------------------------------
# External consumer-model chooser (PREREG_competence_repair_20260730.md)
# --------------------------------------------------------------------------

def check_consumer_model_flags(consumer_model, consumer_checkpoint,
                               behavior_checkpoint, oracle_all):
  """Registered flag discipline for --consumer_model: it composes with
  nothing (each combination would be an unregistered instrument) and
  requires --oracle_all (the determinism-gate pairing needs the all-M
  ground truth realized in every pass). No-op when the flag is empty."""
  if not consumer_model:
    return
  assert not consumer_checkpoint, (
      '--consumer_model combined with --consumer_checkpoint is '
      'unregistered; refusing')
  assert not behavior_checkpoint, (
      '--consumer_model combined with --behavior_checkpoint is '
      'unregistered; refusing')
  assert oracle_all, (
      '--consumer_model requires --oracle_all (registered pairing '
      'invariant: g_all fully realized in every pass)')


def stamp_consumer_model(meta, consumer_model, info):
  """Version/provenance stamping for the external chooser. Empty flag =
  the SAME dict back (byte-identical default path, mirrors
  stamp_consumer); active = exact '_cm1' version suffix + model
  provenance (path, sha256, deployed run key, feature-map version).
  The repair reader pins the suffixed string by EXACT equality."""
  if not consumer_model:
    return meta
  return dict(
      meta,
      labeler_version=meta['labeler_version'] + '_cm1',
      consumer_model=str(consumer_model),
      consumer_model_sha256=info['sha256'],
      consumer_model_run=info['run'],
      consumer_feature_map=info['feature_map_version'])


# --------------------------------------------------------------------------
# Real-run entry point (mirrors d0/sweep.py loading)
# --------------------------------------------------------------------------

def main_real(args):
  import elements
  import jax
  import ruamel.yaml as yaml
  from dreamerv3.main import make_agent, make_env
  from d0.sweep import load_config, load_frozen_agent

  check_consumer_model_flags(args.consumer_model, args.consumer_checkpoint,
                             args.behavior_checkpoint, args.oracle_all)
  consumer, cm_info = None, None
  if args.consumer_model:
    from d0.train_consumer_model import make_chooser
    consumer, cm_info = make_chooser(
        args.consumer_model,
        os.path.basename(args.run_logdir.rstrip('/')))
  out_dir = pathlib.Path(args.output).parent
  config, train_seed = load_config(args, out_dir)
  env = make_env(config, 0)
  w1 = None
  if getattr(args, 'w1_repeats', 0):
    assert args.env_seed is not None, (
        'W1 passes require --env_seed (constraint 6: the unseeded env '
        'path is retired for all new labeling)')
    assert args.oracle_all, 'W1 requires --oracle_all'
    assert not (args.consumer_checkpoint or args.consumer_model or
                args.behavior_checkpoint), (
        'W1 passes are registered standalone')
    w1 = dict(repeats=int(args.w1_repeats),
              dup_cand=bool(args.w1_dup_cand))
    if getattr(args, 'dcand_uniform', 0):
      assert not getattr(args, 'stamp_delta', 0.0), (
          'dcand + stamp composition is unregistered; refusing')
      assert not getattr(args, 'cem_consumer', False), (
          'dcand + cem composition is unregistered; refusing')
      w1.update(dcand_uniform=int(args.dcand_uniform),
                env_seed=int(args.env_seed))
    if getattr(args, 'cem_consumer', False):
      assert not getattr(args, 'stamp_delta', 0.0), (
          'cem + stamp composition is unregistered; refusing')
      w1.update(cem=True)
  else:
    assert not getattr(args, 'w1_dup_cand', False), (
        '--w1_dup_cand requires --w1_repeats > 0')
    assert not getattr(args, 'dcand_uniform', 0), (
        '--dcand_uniform requires --w1_repeats > 0 (WDC rides the W1 '
        'protocol; PREREG_p2_dcand_20260821)')
    assert not getattr(args, 'stamp_delta', 0.0), (
        '--stamp_delta requires --w1_repeats > 0 (Stage B rides the W1 '
        'protocol; PREREG_p2_stageb_injection_20260821)')
    assert not getattr(args, 'cem_consumer', False), (
        '--cem_consumer requires --w1_repeats > 0 '
        '(PREREG_p2_cem_consumer_20260821)')
  stampenv = None
  if getattr(args, 'stamp_delta', 0.0):
    from embodied.envs.rewardstamp import RewardStamp
    act_keys = sorted(k for k in env.act_space if k != 'reset')
    assert len(act_keys) == 1, act_keys
    stampenv = RewardStamp(env, delta=float(args.stamp_delta),
                           act_key=act_keys[0])
    env = stampenv
  if args.env_seed is not None:
    seed_env_task(env, args.env_seed)
  apply_mass_scale(env, args.mass_scale)
  agent = make_agent(config)
  ckpt = args.checkpoint or os.path.join(args.run_logdir, 'ckpt')
  agent = load_frozen_agent(agent, ckpt)
  dual = None
  if args.consumer_checkpoint:
    assert not args.behavior_checkpoint, (
        'consumer overlay combined with a behavior checkpoint is '
        'unregistered; refusing')
    assert args.oracle_all, (
        '--consumer_checkpoint requires --oracle_all (the pairing guard '
        'needs the all-M ground truth realized in every pass; a targeted '
        'pass would NaN g_all and waste the run)')
    # Both loads happen HERE, before any labeling; every RNG mark is
    # taken afterwards. The control arm passes the eval ckpt itself.
    check_consumer_config(args.run_logdir, args.consumer_checkpoint)
    if args.dual_chooser:
      # xc Amendment 1: snapshot the eval agent's own heads, overlay
      # the consumer's, snapshot those too, then run the pass under
      # CONTROL heads, swapping only around the shadow choosers.
      dual = DualHeads(agent)
      overlay_consumer(agent, args.consumer_checkpoint)
      dual.capture_consumer(allow_identical=args.dual_allow_identical)
      dual.to_control()
    else:
      overlay_consumer(agent, args.consumer_checkpoint)
  else:
    assert not args.dual_chooser, (
        '--dual_chooser requires --consumer_checkpoint')
  assert not args.dual_allow_identical or args.dual_chooser, (
      '--dual_allow_identical requires --dual_chooser')
  disc = (1.0 if config.agent.contdisc else
          1 - 1 / config.agent.horizon)
  oracle = AgentOracle(agent, env.act_space, disc)
  cem_oracle = (CEMOracle(agent, env.act_space, disc)
                if (w1 or {}).get('cem') else None)
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
      behavior=behavior, consumer=consumer, dual=dual, w1=w1,
      cem_oracle=cem_oracle)
  meta = dict(
      run_logdir=args.run_logdir, checkpoint=str(ckpt),
      states=args.states, horizon=args.horizon,
      label_every=args.label_every, seed=args.seed,
      train_seed=train_seed, actions=args.actions, rollouts=args.rollouts,
      ref_stride=args.ref_stride,
      labeler_version=(
          'd1fix_20260724_wdc' if (w1 or {}).get('dcand_uniform') else
          'd1fix_20260724_wcem' if (w1 or {}).get('cem') else
          'd1fix_20260724_w1sb' if (w1 is not None and
                                    stampenv is not None) else
          'd1fix_20260724_w1' if w1 is not None else 'd1fix_20260724'),
      dose=dict(config.distractor),
      behavior_checkpoint=str(args.behavior_checkpoint or ''),
      mass_scale=float(args.mass_scale),
      operation_pair='real_vs_imag_matched_candidate_budget')
  if w1 is not None:
    meta.update(w1=dict(w1), env_seed=int(args.env_seed),
                w1_probe_offset=W1_PROBE_OFFSET,
                w1_mark_offset=W1_MARK_OFFSET)
  elif args.env_seed is not None:
    meta.update(env_seed=int(args.env_seed))
  if stampenv is not None:
    meta.update(stamp=dict(delta=float(args.stamp_delta),
                           kind='first_step_dim0_clip',
                           stamped_steps=int(stampenv.stamp_count)))
  if (w1 or {}).get('cem'):
    d0cfg = config.agent.d0
    meta.update(cem=dict(iters=int(d0cfg.cem_iters),
                         samples=int(d0cfg.cem_samples),
                         horizon=int(d0cfg.cem_horizon),
                         elites=int(d0cfg.cem_elites),
                         std=float(d0cfg.cem_std)))
  meta = stamp_consumer(meta, args.consumer_checkpoint,
                        max_steps=args.max_steps,
                        dual=bool(args.dual_chooser),
                        dual_allow_identical=bool(
                            args.dual_allow_identical))
  meta = stamp_consumer_model(meta, args.consumer_model, cm_info)
  save_rows(rows, args.output, meta=meta, extra_arrays=ref_arrays)


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


class _W1Env:
  """Chaotic restorable env for the W1 fixture: the reward stream
  depends sensitively on the accumulated position, so branches with
  DIFFERENT first actions decorrelate under a COMMON follower schedule
  (the within-repeat noise the old estimand maxes over), while
  IDENTICAL first actions reproduce exactly (the duplicate-null gate).
  Candidate true values are equal by construction (chaotic rewards are
  effectively exchangeable across first actions)."""

  def __init__(self):
    self.pos, self.t = 0.0, 0

  def oracle_get_state(self):
    return (self.pos, self.t)

  def oracle_set_state(self, s):
    self.pos, self.t = s

  def step(self, act):
    if act.get('reset'):
      self.pos, self.t = 0.0, 0
      return dict(reward=np.float32(0), is_last=np.array(False))
    a = float(np.asarray(act['action']).reshape(-1)[0])
    self.pos += a
    self.t += 1
    reward = np.float32(np.sin(self.pos * 997.13) ** 2)
    return dict(reward=reward, is_last=np.array(self.t >= 400))


class _W1Oracle(AgentOracle):
  """Counter-seeded STOCHASTIC follower (the property every previous
  fixture lacked — §13.4 lesson): the sampled action is a deterministic
  function of the RNG counter, so marks give exact CRN and different
  marks give independent follower schedules. Records every rng_reset
  for the mark-disjointness assertions."""

  M = 4

  def __init__(self):
    self.act_keys = ['action']
    self.act_key = 'action'
    self.act_shape = (1,)
    self.discount = 0.9
    self._ctr = 7
    self.resets = []

  def init(self):
    return ('carry',)

  def rng_mark(self):
    return int(self._ctr)

  def rng_reset(self, mark):
    self._ctr = int(mark)
    self.resets.append(int(mark))

  def policy(self, carry, obs, mode='eval'):
    a = np.float32(np.sin(self._ctr * 12.9898) * 0.3)
    self._ctr += 1
    return carry, {'action': np.asarray([[a]], np.float32)}, {}

  def d0_eval(self, carry, obs, mode='eval'):
    cands = np.linspace(0.1, 0.4, self.M,
                        dtype=np.float32)[:, None]
    q = np.tile(np.arange(self.M, dtype=np.float32)[None], (3, 1))
    extras = dict(udyn=0.5, deter=np.zeros(2, np.float16))
    return carry, q, cands, {'action': np.zeros(1, np.float32)}, extras

  def latent_of(self, carry):
    return np.zeros(2, np.float16)

  def branch_carry(self, carry_post, cand_vec):
    return carry_post

  def with_prevact(self, carry, acts):
    return carry


def selfcheck_w1():
  """W1 legs (PREREG_w1_adjudication_constraints_20260809)."""
  # --- winner's-curse fixture: equal-value candidates, stochastic
  # follower. Within-repeat opportunity must be POSITIVE (the max-of-
  # noise bias is visible to this fixture — no previous fixture could
  # see it) while split-selection kills it.
  env, oracle = _W1Env(), _W1Oracle()
  rows, _ = label_run(env, oracle, n_states=24, horizon=12,
                      label_every=7, max_steps=400,
                      rng=np.random.default_rng(0), run_id='w1synth',
                      oracle_all=True, w1=dict(repeats=6))
  g = np.stack([r['g_all_rep'] for r in rows])          # (S, R, M)
  ga = np.stack([r['g_actor_rep'] for r in rows])       # (S, R)
  assert g.shape[1:] == (6, _W1Oracle.M) and ga.shape[1] == 6
  assert float(np.mean(g.std(1))) > 1e-3, 'follower not stochastic'
  assert float(np.mean(ga.std(1))) > 1e-3, 'actor branch not stochastic'
  within = float(np.mean(g.max(2) - g.mean(2)))
  sel = g[:, 0::2].mean(1).argmax(1)                    # select on even
  ho = g[:, 1::2].mean(1)                               # evaluate on odd
  split = float(np.mean(ho[np.arange(len(ho)), sel] - ho.mean(1)))
  if not within > 0.05:
    raise SystemExit(f'selfcheck FAIL: within-repeat max bias invisible '
                     f'({within}) — fixture lost its noise')
  if not abs(split) < 0.5 * within:
    raise SystemExit(f'selfcheck FAIL: split-selection did not remove '
                     f'the max bias (within {within}, split {split})')
  for r in rows:
    assert r['r_real'].shape == (_W1Oracle.M,)
    assert r['actor_act'].shape == (1,)
    # carry-sensitive fixture (batch-review B1): the actor branch must
    # carry d0_eval's OWN discarded sample (zeros in this fixture), not
    # a fresh policy() draw (sin(ctr) != 0 here) from a
    # double-assimilated belief.
    assert np.all(r['actor_act'] == 0.0), r['actor_act']
    assert not r['dup_cand']
  # mark discipline: per state, the first reset is the base; every
  # subsequent reset must be base + W1_PROBE_OFFSET (probe) or
  # base + k*W1_MARK_OFFSET (repeat k / final advance), and both the
  # probe and at least one repeat offset must occur for every state.
  base, seen_probe, seen_rep, n_states_seen = None, False, False, 0
  for v in oracle.resets:
    d = None if base is None else v - base
    if d is not None and (d == W1_PROBE_OFFSET or
                          (d > 0 and d % W1_MARK_OFFSET == 0)):
      seen_probe |= (d == W1_PROBE_OFFSET)
      seen_rep |= (d > 0 and d % W1_MARK_OFFSET == 0)
      continue
    # new state's base mark
    if base is not None:
      assert seen_probe and seen_rep, (base, seen_probe, seen_rep)
    base, seen_probe, seen_rep = v, False, False
    n_states_seen += 1
  assert seen_probe and seen_rep and n_states_seen == len(rows), (
      n_states_seen, len(rows))
  # --- duplicate-candidate NULL: identical actions must give
  # bit-identical rollouts within every repeat (gate, not estimate)
  env2, oracle2 = _W1Env(), _W1Oracle()
  rows2, _ = label_run(env2, oracle2, n_states=6, horizon=12,
                       label_every=7, max_steps=400,
                       rng=np.random.default_rng(0), run_id='w1dup',
                       oracle_all=True,
                       w1=dict(repeats=3, dup_cand=True))
  for r in rows2:
    g2 = r['g_all_rep']
    if not np.all(g2 == g2[:, :1]):
      raise SystemExit('selfcheck FAIL: duplicate-null violated — '
                       'within-repeat CRN broken')
    assert float(np.ptp(r['r_real'])) == 0.0, r['r_real']
    assert np.all(r['cands'] == r['cands'][0]), r['cands']
    assert r['dup_cand']
  # --- save/reload round-trip with the redacted (_w1) print path
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, 'w1.npz')
    save_rows(rows2, out, meta=dict(labeler_version='d1fix_20260724_w1'))
    z = np.load(out, allow_pickle=True)
    for key in ('g_all_rep', 'g_actor_rep', 'actor_act', 'r_real',
                'dup_cand'):
      assert key in z.files, key
    # Stage-B redaction branch (build-review 2.1): W1-shaped rows under
    # the '_w1sb' version must ALSO take the redacted path (the
    # else-branch would KeyError on the absent delta_real)
    save_rows(rows2, os.path.join(tmp, 'w1sb.npz'),
              meta=dict(labeler_version='d1fix_20260724_w1sb'))
  # --- flag guards: w1 with dual/consumer/behavior must refuse
  try:
    label_run(_W1Env(), _W1Oracle(), n_states=1, horizon=12,
              label_every=7, max_steps=400,
              rng=np.random.default_rng(0), run_id='w1bad',
              oracle_all=True, w1=dict(repeats=2), dual=object())
    raise SystemExit('selfcheck FAIL: w1+dual combo not refused')
  except AssertionError:
    pass
  print('SELFCHECK PASS (W1: stochastic-follower fixture — within-repeat '
        'max bias visible + split-selection kills it; duplicate-null '
        'exact; mark-offset discipline; actor branch + r_real stored; '
        '_w1 redacted save round-trip; standalone-flag guard)')


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

  # --- xc1: consumer-swap on the synthetic oracle. Same env class, same
  # pol (both play cand 1), different value heads: the swapped heads
  # rank candidate 2 first and score probe next-states by -10 x reward,
  # so every chooser flips while trajectories, labeled states, and (with
  # oracle_all) g_all are provably invariant — the pairing fact the xc
  # wave registers as a guard. Closed forms, horizon 10:
  # g_all = [2.8, 2.0, 1.8] at every labeled state. ---
  class _XcHeadsOracle(_SynthOracle):
    def __init__(self):
      super().__init__()
      self.discount = 1.0

    def policy(self, carry, obs, mode='eval'):
      carry, acts, out = super().policy(carry, obs, mode)
      base = -10.0 * float(obs['reward'])
      q = np.array([[base, base + 0.05, base + 0.1]] * 2, np.float32)
      return carry, acts, dict(out, **{'d0/qfull': q[None]})

  ctrl_oracle = _SynthOracle()
  ctrl_oracle.discount = 1.0  # matched op_real discount across the pair
  rows_c, _ = label_run(_SynthEnv(), ctrl_oracle, n_states=4, horizon=10,
                        label_every=7, max_steps=200,
                        rng=np.random.default_rng(0), run_id='xc',
                        oracle_all=True)
  rows_s, _ = label_run(_SynthEnv(), _XcHeadsOracle(), n_states=4,
                        horizon=10, label_every=7, max_steps=200,
                        rng=np.random.default_rng(0), run_id='xc',
                        oracle_all=True)
  for rc, rs in zip(rows_c, rows_s):
    # pairing invariants: identical labeled states, identical g_all
    assert rc['episode'] == rs['episode'] and rc['step'] == rs['step']
    assert np.allclose(rc['g_all'], rs['g_all'], atol=1e-6)
    assert np.allclose(rc['g_all'], [2.8, 2.0, 1.8], atol=1e-5), rc['g_all']
    # chooser difference: control keeps the miscalibrated middle + true
    # best; swapped heads pick candidate 2 everywhere
    assert rc['m_now'] == 1 and rc['m_real'] == 0, rc
    assert rs['m_now'] == 2 and rs['m_real'] == 2, rs
    # closed-form achieved (= g_all[m_real] - g_now): 0.8 vs 0.0
    ach_c = rc['g_all'][rc['m_real']] - rc['g_now']
    ach_s = rs['g_all'][rs['m_real']] - rs['g_now']
    assert abs(ach_c - 0.8) < 1e-5 and abs(ach_s - 0.0) < 1e-5, (ach_c, ach_s)

  # --- xc2: CONSUMER_REGEX scope (param-key classification) ---
  import re as _re
  for key, want in [
      ('rew/w', True), ('con/b', True), ('valens0/x', True),
      ('valens12/x', True), ('val/w', False), ('valnorm/x', False),
      ('slowvalens0/w', False), ('pol/w', False), ('enc/w', False),
      ('dyn/w', False), ('dec/w', False), ('disag/w', False),
      ('rewnorm/w', False)]:
    assert bool(_re.match(CONSUMER_REGEX, key)) == want, (key, want)

  # --- xc3: overlay_consumer counter snapshot/restore + regex-scoped
  # param surgery on a mock agent mirroring JAXAgent.load semantics
  # (counters overwritten unconditionally by whichever load runs) ---
  class _Ctr2:
    def __init__(self, v):
      self.lock = threading.Lock()
      self.value = v

  class _MockLoadAgent:
    def __init__(self):
      self.n_updates = _Ctr2(3)
      self.n_batches = _Ctr2(3)
      self.n_actions = _Ctr2(1234)
      self.params = {
          'enc/w': 'E0', 'dyn/w': 'Y0', 'dec/w': 'X0', 'pol/w': 'P0',
          'rew/w': 'R0', 'con/w': 'C0', 'valens0/w': 'V0',
          'valens1/w': 'V1', 'val/w': 'VAL0', 'slowvalens0/w': 'SV0',
          'disag/w': 'D0'}

    def load(self, data, regex=None):
      self.n_updates.value = data['counters']['updates']
      self.n_batches.value = data['counters']['updates']
      self.n_actions.value = data['counters']['actions']
      params = data['params']
      if regex:
        params = {k: v for k, v in params.items() if _re.match(regex, k)}
      self.params.update(params)

  mag = _MockLoadAgent()
  cons_data = {'params': {k: k + '_CONS' for k in mag.params},
               'counters': {'updates': 99, 'actions': 777}}
  calls = []

  def _mock_load(path):
    calls.append(path)
    mag.load(cons_data, regex=CONSUMER_REGEX)

  marks = overlay_consumer(mag, 'ckptB', load_fn=_mock_load)
  assert calls == ['ckptB']
  assert marks == dict(n_updates=3, n_batches=3, n_actions=1234), marks
  # counters restored despite the load's unconditional overwrite
  assert (mag.n_updates.value, mag.n_batches.value,
          mag.n_actions.value) == (3, 3, 1234)
  # only the head params were replaced
  for k in ('rew/w', 'con/w', 'valens0/w', 'valens1/w'):
    assert mag.params[k] == k + '_CONS', (k, mag.params[k])
  for k in ('enc/w', 'dyn/w', 'dec/w', 'pol/w', 'val/w',
            'slowvalens0/w', 'disag/w'):
    assert not mag.params[k].endswith('_CONS'), (k, mag.params[k])

  # control-equivalence: overlay-with-self leaves the ENTIRE mock state
  # byte-for-byte unchanged (params + counters)
  mag2 = _MockLoadAgent()
  before = (dict(mag2.params), mag2.n_updates.value,
            mag2.n_batches.value, mag2.n_actions.value)
  self_data = {'params': dict(mag2.params),
               'counters': {'updates': 99, 'actions': 777}}
  overlay_consumer(mag2, 'ckptSelf',
                   load_fn=lambda p: mag2.load(self_data,
                                               regex=CONSUMER_REGEX))
  after = (dict(mag2.params), mag2.n_updates.value,
           mag2.n_batches.value, mag2.n_actions.value)
  assert before == after, (before, after)

  # --- xc4: meta/version stamping. Empty flag = the SAME dict back
  # (byte-identical default path); active overlay = exact _xc1 version
  # + consumer keys, original dict never mutated. ---
  base_meta = dict(labeler_version='d1fix_20260724', seed=1)
  assert stamp_consumer(base_meta, '') is base_meta
  stamped = stamp_consumer(base_meta, '/x/r3_cup_e1_seed31/ckpt_early')
  assert stamped['labeler_version'] == 'd1fix_20260724_xc1'
  assert stamped['consumer_checkpoint'] == '/x/r3_cup_e1_seed31/ckpt_early'
  assert stamped['consumer_regex'] == CONSUMER_REGEX
  assert base_meta == dict(labeler_version='d1fix_20260724', seed=1)
  # max_steps is overlay-only meta (state-window dial; xc reader pins it)
  # Amendment 1: dual=True stamps _xc2 + the dual fields; dual=False
  # unchanged _xc1 (existing two-pass files can never enter the wave)
  stamped_d = stamp_consumer(base_meta, '/x/ckpt_early', dual=True)
  assert stamped_d['labeler_version'] == 'd1fix_20260724_xc2'
  assert stamped_d['dual_chooser'] is True
  assert stamped_d['dual_cands_atol'] == DUAL_CANDS_ATOL
  assert stamped_d['dual_allow_identical'] is False
  assert stamp_consumer(base_meta, '/x/ckpt_early', dual=True,
                        dual_allow_identical=True)[
                            'dual_allow_identical'] is True
  assert 'dual_chooser' not in stamp_consumer(base_meta, '/x/ckpt_early')
  stamped_ms = stamp_consumer(base_meta, '/x/ckpt_early', max_steps=1000)
  assert stamped_ms['max_steps'] == 1000
  assert 'max_steps' not in stamped

  # --- xc6: no-op-overlay guard — zero-match and partial-coverage
  # overlays must REFUSE (a silent no-op would fabricate the registered
  # HEADS-IRRELEVANT null); the mock layouts here are deliberately NOT
  # synthesized from CONSUMER_REGEX. ---
  for layout, needle in (
      ({'pol/w': 1, 'enc/w': 2}, 'ZERO live param'),
      ({'rew/w': 1, 'pol/w': 2}, "no 'con/'"),
      ({'rew/w': 1, 'con/w': 2, 'pol/w': 3}, "no 'valens0'")):
    magn = _MockLoadAgent()
    magn.params = dict(layout)
    try:
      overlay_consumer(magn, 'ckptN', load_fn=lambda p: None)
      raise SystemExit(f'selfcheck FAIL: no-op overlay not caught: {layout}')
    except AssertionError as e:
      assert needle in str(e), (needle, e)

  # --- xc7: consumer-arm stdout redaction — save_rows must not print
  # estimand means for _xc1/_cm1 passes (delta_real IS achieved; paired
  # stdout means would reveal the primaries pre-read); the default path
  # keeps the original line. ---
  import contextlib
  import io
  import tempfile
  with tempfile.TemporaryDirectory() as d:
    r = dict(episode=0, step=0, delta_real=0.5, delta_imag=0.1)
    for ver, redacted in (('d1fix_20260724', False),
                          ('d1fix_20260724_xc1', True),
                          ('d1fix_20260724_xc2', True),
                          ('d1fix_20260724_cm1', True)):
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        save_rows([r], os.path.join(d, f'x_{ver}.npz'),
                  meta=dict(labeler_version=ver))
      out = buf.getvalue()
      if redacted:
        assert 'redacted' in out and 'delta_real' not in out, out
      else:
        assert 'mean delta_real' in out, out

  # --- xc5: config-safety comparer (the regex load path skips shape
  # asserts, so this is the manual arch check) + missing-config trip ---
  assert _flat_cfg({'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}) == {
      'a.b': 1, 'a.c.d': 2, 'e': 3}
  ev_flat = {'task': 'dmc_cup_catch', 'agent.valens.k': 5,
             'agent.dyn.deter': 512, 'agent.valnorm.impl': 'none',
             'seed': 31, 'logdir': '/a'}
  cn_flat = dict(ev_flat, seed=99, logdir='/b')  # run-identity diffs ok
  assert consumer_config_mismatches(ev_flat, cn_flat) == []
  bad = dict(ev_flat, **{'agent.valens.k': 3})
  msgs = consumer_config_mismatches(ev_flat, bad)
  assert msgs and 'agent.valens.k' in msgs[0], msgs
  bad = dict(ev_flat, task='dmc_finger_turn_hard')
  assert any('task' in m for m in consumer_config_mismatches(ev_flat, bad))
  bad = dict(ev_flat)
  del bad['agent.dyn.deter']
  assert any('<absent>' in m
             for m in consumer_config_mismatches(ev_flat, bad))
  ema = dict(ev_flat, **{'agent.valnorm.impl': 'ema'})
  msgs = consumer_config_mismatches(ema, ema)
  assert msgs and all('valnorm' in m for m in msgs), msgs
  import tempfile
  with tempfile.TemporaryDirectory() as d:
    os.makedirs(os.path.join(d, 'runX', 'ckpt'))
    try:
      check_consumer_config(d, os.path.join(d, 'runX', 'ckpt'))
      raise AssertionError('missing consumer config.yaml must trip')
    except SystemExit as e:
      assert 'config.yaml' in str(e), e

  # --- xc8 (Amendment 1): dual-chooser within-pass pairing. The same
  # planted head difference as xc1, but evaluated as control + shadow
  # inside ONE pass: g_all computed once, both chooser index sets
  # recorded, closed-form within-pass paired delta
  # [g_all[m_real_x]-g_all[m_now_x]] - [g_all[m_real]-g_all[m_now]]
  # = (1.8-1.8) - (2.8-2.0) = -0.8. ---
  class _DualSynthOracle(_SynthOracle):
    """xc1's planted head difference, plus a REAL policy-RNG counter
    (reviewer M2): every policy call consumes one tick and
    rng_mark/rng_reset are genuine — so the replay bookkeeping and the
    RNG-count guard are exercised for real, not via no-ops."""

    def __init__(self):
      super().__init__()
      self.discount = 1.0
      self.consumer_active = False
      self._rng_ctr = 100

    def policy(self, carry, obs, mode='eval'):
      self._rng_ctr += 1
      carry, acts, out = super().policy(carry, obs, mode)
      if self.consumer_active:
        base = -10.0 * float(obs['reward'])
        q = np.array([[base, base + 0.05, base + 0.1]] * 2, np.float32)
        out = dict(out, **{'d0/qfull': q[None]})
      return carry, acts, out

    def rng_mark(self):
      return self._rng_ctr

    def rng_reset(self, mark):
      self._rng_ctr = mark

  class _SynthDual:
    def __init__(self, oracle):
      self.oracle = oracle
      self.swaps = 0

    def to_consumer(self):
      self.oracle.consumer_active = True
      self.swaps += 1

    def to_control(self):
      self.oracle.consumer_active = False

  d_oracle = _DualSynthOracle()
  d_swap = _SynthDual(d_oracle)
  rows_d, _ = label_run(_SynthEnv(), d_oracle, n_states=4, horizon=10,
                        label_every=7, max_steps=200,
                        rng=np.random.default_rng(0), run_id='xcd',
                        oracle_all=True, dual=d_swap)
  assert d_swap.swaps == 2 * len(rows_d), d_swap.swaps
  assert not d_oracle.consumer_active   # pass ends under control heads
  for rc, rd in zip(rows_c, rows_d):
    # base trajectory + g_all identical to the plain control pass
    assert rc['episode'] == rd['episode'] and rc['step'] == rd['step']
    assert np.allclose(rc['g_all'], rd['g_all'], atol=1e-6)
    # control choosers unchanged; shadow choosers = the planted flips
    assert rd['m_now'] == 1 and rd['m_real'] == 0, rd
    assert rd['m_now_x'] == 2 and rd['m_real_x'] == 2, rd
    assert float(rd['cands_max_absdiff']) == 0.0
    assert rd['qfull_x'].shape == rd['qfull'].shape
    assert rd['real_scores_x'].shape == rd['real_scores'].shape
    d_pair = ((rd['g_all'][rd['m_real_x']] - rd['g_all'][rd['m_now_x']])
              - (rd['g_all'][rd['m_real']] - rd['g_all'][rd['m_now']]))
    assert abs(d_pair - (-0.8)) < 1e-5, d_pair

  # RNG-schedule invariance for real (reviewer M2): a plain pass with
  # the SAME counting oracle class ends at the SAME counter value —
  # the dual pass's resets net out exactly.
  p_oracle = _DualSynthOracle()
  rows_p, _ = label_run(_SynthEnv(), p_oracle, n_states=4, horizon=10,
                        label_every=7, max_steps=200,
                        rng=np.random.default_rng(0), run_id='xcd',
                        oracle_all=True)
  assert p_oracle._rng_ctr == d_oracle._rng_ctr, (
      p_oracle._rng_ctr, d_oracle._rng_ctr)
  for rp, rd in zip(rows_p, rows_d):
    assert rp['episode'] == rd['episode'] and rp['step'] == rd['step']
    assert np.allclose(rp['g_all'], rd['g_all'], atol=1e-6)

  # ... and a shadow that over-consumes the replay window must trip
  # the RNG-count guard.
  class _GreedyShadow(_DualSynthOracle):
    def policy(self, carry, obs, mode='eval'):
      if self.consumer_active:
        self._rng_ctr += 1          # one extra tick under the shadow
      return super().policy(carry, obs, mode)

  go = _GreedyShadow()
  try:
    label_run(_SynthEnv(), go, n_states=1, horizon=10, label_every=7,
              max_steps=200, rng=np.random.default_rng(0), run_id='xcd',
              oracle_all=True, dual=_SynthDual(go))
    raise AssertionError('shadow RNG over-consumption must trip')
  except SystemExit as e:
    assert 'RNG count' in str(e), e

  # sub-atol candidate drift (reviewer m4): the shadow op_real must
  # execute the CONTROL candidate set. Candidate 1 sits exactly on the
  # synth reward threshold (a=0.5 -> 0.2); the shadow's drifted copy
  # (+5e-5, under the gate) would cross it (-> 1.0). Executing control
  # cands keeps both op_real reward sweeps identical, so the pass
  # completes; a shadow that executed its own cands would trip the
  # env-reward CRN gate.
  class _SubAtolOracle(_DualSynthOracle):
    CANDS = np.array([[0.9], [0.5], [-0.5]], np.float32)

    def policy(self, carry, obs, mode='eval'):
      carry, acts, out = super().policy(carry, obs, mode)
      if self.consumer_active:
        out = dict(out, **{'d0/cands': (self.CANDS + 5e-5)[None]})
      return carry, acts, out

  so = _SubAtolOracle()
  rows_sa, _ = label_run(_SynthEnv(), so, n_states=2, horizon=10,
                         label_every=7, max_steps=200,
                         rng=np.random.default_rng(0), run_id='xcd',
                         oracle_all=True, dual=_SynthDual(so))
  for r in rows_sa:
    assert 0 < float(r['cands_max_absdiff']) <= DUAL_CANDS_ATOL, r

  # candidate-drift gate: a shadow whose candidate draws move beyond
  # DUAL_CANDS_ATOL must refuse (the shadow would rank actions g_all
  # was never realized for)
  class _DriftCandsOracle(_DualSynthOracle):
    def policy(self, carry, obs, mode='eval'):
      carry, acts, out = super().policy(carry, obs, mode)
      if self.consumer_active:
        out = dict(out, **{'d0/cands': (self.CANDS + 0.01)[None]})
      return carry, acts, out

  dr_oracle = _DriftCandsOracle()
  try:
    label_run(_SynthEnv(), dr_oracle, n_states=1, horizon=10,
              label_every=7, max_steps=200, rng=np.random.default_rng(0),
              run_id='xcd', oracle_all=True, dual=_SynthDual(dr_oracle))
    raise AssertionError('candidate drift beyond atol must trip')
  except SystemExit as e:
    assert 'candidate drift' in str(e), e

  # env-reward drift gate: if the env stops replaying identical rewards
  # between the control and shadow op_real sweeps, CRN is broken and
  # the pass must refuse
  class _DriftEnv(_SynthEnv):
    drift = False

    def step(self, act):
      obs = super().step(act)
      if self.drift and not act.get('reset'):
        obs = dict(obs, reward=np.float32(float(obs['reward']) + 0.01))
      return obs

  class _EnvDriftDual(_SynthDual):
    def __init__(self, oracle, env):
      super().__init__(oracle)
      self.env = env

    def to_consumer(self):
      super().to_consumer()
      self.env.drift = True

    def to_control(self):
      super().to_control()
      self.env.drift = False

  de_env = _DriftEnv()
  de_oracle = _DualSynthOracle()
  try:
    label_run(de_env, de_oracle, n_states=1, horizon=10, label_every=7,
              max_steps=200, rng=np.random.default_rng(0), run_id='xcd',
              oracle_all=True, dual=_EnvDriftDual(de_oracle, de_env))
    raise AssertionError('env-reward drift across dual op_real must trip')
  except SystemExit as e:
    assert 'env-reward drift' in str(e), e

  # DualHeads guard logic (reviewer M3), jax-free half: the
  # identical-heads refusal is REAL code, exercised here via a
  # snapshot-injection subclass.
  class _FakeDual(DualHeads):
    def __init__(self, ctl, nxt):
      self.matched = sorted(ctl)
      self.ctl = {k: np.asarray(v) for k, v in ctl.items()}
      self._next = nxt
      self.sw = None

    def _snapshot(self):
      return {k: np.asarray(v) for k, v in self._next.items()}

  same = {'rew/w': np.ones(3)}
  try:
    _FakeDual(same, same).capture_consumer()
    raise AssertionError('identical-heads capture must refuse')
  except SystemExit as e:
    assert 'byte-identical' in str(e), e
  _FakeDual(same, {'rew/w': np.zeros(3)}).capture_consumer()
  _FakeDual(same, same).capture_consumer(allow_identical=True)

  # ... jax-backed half: the swap must land in the LIVE policy store
  # (the dict policy calls consult) and the witness read-back must
  # catch a store that swallows writes.
  import jax

  class _FakeAgent:
    pass

  ag = _FakeAgent()
  ag.params = {'rew/w': jax.device_put(np.full(3, 1.0)),
               'con/w': jax.device_put(np.full(2, 1.0)),
               'valens0/w': jax.device_put(np.full(2, 1.0)),
               'pol/w': jax.device_put(np.full(2, 9.0))}
  ag.policy_params = {k: jax.device_put(np.asarray(v) * 2)
                      for k, v in ag.params.items()}
  dh = DualHeads(ag)
  assert dh.matched == ['con/w', 'rew/w', 'valens0/w'], dh.matched
  for store in (ag.params, ag.policy_params):    # simulate the overlay
    for k in dh.matched:
      store[k] = jax.device_put(np.full_like(np.asarray(store[k]), 5.0))
  dh.capture_consumer()
  dh.to_control()
  assert float(np.asarray(jax.device_get(
      ag.policy_params['rew/w']))[0]) == 2.0
  assert float(np.asarray(jax.device_get(ag.params['rew/w']))[0]) == 2.0
  dh.to_consumer()
  assert float(np.asarray(jax.device_get(
      ag.policy_params['rew/w']))[0]) == 5.0
  assert float(np.asarray(jax.device_get(
      ag.policy_params['pol/w']))[0]) == 18.0    # non-head keys untouched

  class _LossyDual(DualHeads):
    def _store(self):
      return dict(self.agent.policy_params)      # writes vanish

  ld = _LossyDual(ag)
  ld.ctl = {k: np.zeros_like(v) for k, v in ld.ctl.items()}
  try:
    ld.to_control()
    raise AssertionError('lossy store must trip the witness read-back')
  except SystemExit as e:
    assert 'live policy store' in str(e), e

  # dual row schema round-trip
  out_d = pathlib.Path(os.environ.get('TMPDIR', '/tmp')) / 'oracle_xcd.npz'
  save_rows(rows_d, str(out_d),
            meta=dict(labeler_version='d1fix_20260724_xc2'))
  data_d = np.load(out_d, allow_pickle=False)
  assert data_d['m_now_x'].shape == (4,)
  assert data_d['m_real_x'].shape == (4,)
  assert data_d['qfull_x'].shape == data_d['qfull'].shape
  assert data_d['real_scores_x'].shape == (4, 3)
  assert data_d['cands_max_absdiff'].shape == (4,)

  # --- cm1: external consumer-model chooser (competence repair). A
  # ridge model whose only weight is -1 on the action column ranks the
  # a=-0.5 candidate first; the chooser must override ONLY the
  # realized-choice row fields while trajectory, labeled states, g_all,
  # the plug-in/imag picks, and the probe scores stay provably
  # invariant vs the consumer=None run (closed forms as xc1). ---
  from d0.train_consumer_model import (
      FEATURE_MAP_VERSION, make_chooser)
  with tempfile.TemporaryDirectory() as d:
    w = np.zeros(11)  # F = 8 + A(1) + D(2) for the synthetic oracle
    w[8] = -1.0       # candidate action column
    mpath = os.path.join(d, 'model.npz')
    np.savez(mpath,
             feature_map_version=FEATURE_MAP_VERSION,
             trainer_version='cm1',
             lambda_grid=np.array([1.0]),
             runs=np.array(['synth']),
             manifest=json.dumps(dict(files={}, per_model={'synth': []})),
             w_synth=w, b_synth=np.float64(0.0), mu_synth=np.zeros(11),
             sd_synth=np.ones(11), lambda_synth=np.float64(1.0))
    chooser, cm_info = make_chooser(mpath, 'synth')
    assert cm_info['run'] == 'synth' and len(cm_info['sha256']) == 64
    assert cm_info['feature_map_version'] == FEATURE_MAP_VERSION
    rows_p, _ = label_run(_SynthEnv(), _SynthOracle(), n_states=4,
                          horizon=10, label_every=7, max_steps=200,
                          rng=np.random.default_rng(0), run_id='synth',
                          oracle_all=True)
    rows_m, _ = label_run(_SynthEnv(), _SynthOracle(), n_states=4,
                          horizon=10, label_every=7, max_steps=200,
                          rng=np.random.default_rng(0), run_id='synth',
                          oracle_all=True, consumer=chooser)
    for rp, rm in zip(rows_p, rows_m):
      # pairing invariants: identical labeled states, identical g_all
      assert rp['episode'] == rm['episode'] and rp['step'] == rm['step']
      assert np.allclose(rp['g_all'], rm['g_all'], atol=1e-6)
      assert np.allclose(rm['g_all'], [2.8, 2.0, 1.8], atol=1e-5)
      # untouched decisions and probe machinery
      assert rm['m_now'] == 1 and rm['m_imag'] == 1, rm
      assert np.allclose(rm['real_scores'], rp['real_scores'])
      # the override: realized choice = argmax ghat; probe pick kept
      assert rp['m_real'] == 0 and rm['m_real'] == 2, (rp, rm)
      assert rm['m_real_probe'] == 0, rm['m_real_probe']
      assert np.allclose(rm['ghat'], [-0.9, -0.3, 0.5], atol=1e-6)
      assert rm['m_real'] == int(np.argmax(rm['ghat']))
      assert abs(rm['g_real'] - rm['g_all'][2]) < 1e-5, rm
      assert abs(rm['delta_real'] - (1.8 - 2.0)) < 1e-5, rm['delta_real']
      # consumer=None rows carry NO new fields (byte-identical schema)
      assert 'ghat' not in rp and 'm_real_probe' not in rp
    # save/reload: new columns round-trip with shapes and dtypes
    out_m = os.path.join(d, 'cm_rows.npz')
    save_rows(rows_m, out_m, meta=dict(selfcheck=True))
    data = np.load(out_m, allow_pickle=False)
    assert data['ghat'].shape == (4, 3) and data['ghat'].dtype == np.float32
    assert data['m_real_probe'].shape == (4,)
    assert np.array_equal(data['m_real'], np.argmax(data['ghat'], 1))

    # cm2: registered flag discipline — the chooser composes with
    # nothing and requires oracle_all; empty flag is a no-op.
    for bad in (dict(consumer_checkpoint='x'),
                dict(behavior_checkpoint='x'),
                dict(oracle_all=False)):
      kw = dict(consumer_checkpoint='', behavior_checkpoint='',
                oracle_all=True)
      kw.update(bad)
      try:
        check_consumer_model_flags('m.npz', **kw)
        raise SystemExit(f'selfcheck FAIL: flag combo not caught: {bad}')
      except AssertionError as e:
        assert 'unregistered' in str(e) or 'oracle_all' in str(e), e
    check_consumer_model_flags('', 'ck', 'bk', False)

    # cm3: meta/version stamping — empty flag returns the SAME dict
    # (byte-identical default path); active = exact _cm1 version + model
    # provenance; the original dict is never mutated.
    bm = dict(labeler_version='d1fix_20260724', seed=0)
    assert stamp_consumer_model(bm, '', None) is bm
    st = stamp_consumer_model(bm, mpath, cm_info)
    assert st['labeler_version'] == 'd1fix_20260724_cm1'
    assert st['consumer_model'] == mpath
    assert st['consumer_model_sha256'] == cm_info['sha256']
    assert st['consumer_model_run'] == 'synth'
    assert st['consumer_feature_map'] == FEATURE_MAP_VERSION
    assert bm == dict(labeler_version='d1fix_20260724', seed=0)

    # cm4: deploy guards — unknown run key trips at load.
    try:
      make_chooser(mpath, 'r3_cup_e1_seed31')
      raise SystemExit('selfcheck FAIL: unknown model run not caught')
    except AssertionError as e:
      assert 'no weights' in str(e), e

  print('SELFCHECK PASS (incl. R2 extension: udyn/deter/boot/ref; '
        'shift extension: behavior-driven trajectory + mass-scale; '
        'd1fix: candidate-conditioned branch carries + policy-RNG CRN; '
        'xc extension: head-swap trajectory/g_all invariance closed-form, '
        'CONSUMER_REGEX scope, counter snapshot/restore + regex surgery + '
        'control self-equivalence, meta stamping byte-identical default '
        '+ max_steps overlay stamp, no-op/partial-overlay guard trips, '
        'consumer-arm stdout redaction (_xc1/_xc2/_cm1), '
        'config-safety comparer + missing-config trip; xc Amendment 1 '
        '(dual chooser): within-pass control+shadow pairing closed-form '
        '(d_pair -0.8, g_all/base-trajectory invariance vs plain pass), '
        'candidate-drift + env-reward-drift gates trip, _xc2 stamping + '
        'dual fields + row schema round-trip; cm extension: '
        'external ridge chooser trajectory/g_all invariance closed-form '
        'with probe fields preserved + row schema round-trip, flag '
        'discipline (no composition, oracle_all required), _cm1 stamping '
        'byte-identical default, unknown-run guard)')
  selfcheck_w1()
  selfcheck_wdc()
  selfcheck_wcem()


def selfcheck_wdc():
  """WDC diverse-candidate legs (PREREG_p2_dcand_20260821)."""
  env, oracle = _W1Env(), _W1Oracle()
  rows, _ = label_run(env, oracle, n_states=10, horizon=12,
                      label_every=7, max_steps=400,
                      rng=np.random.default_rng(0), run_id='wdcsynth',
                      oracle_all=True,
                      w1=dict(repeats=4, dcand_uniform=3, env_seed=123))
  for r in rows:
    assert r['dcands'].shape == (3, 1), r['dcands'].shape
    assert r['g_dcand_rep'].shape == (4, 3)
    assert r['g_wit_rep'].shape == (4,)
    # CRN witness: candidate-0 re-run bit-identical to g_all_rep[:, 0]
    assert np.array_equal(r['g_wit_rep'], r['g_all_rep'][:, 0]), (
        r['g_wit_rep'], r['g_all_rep'][:, 0])
    # pinned per-state uniform stream: reproducible from (env_seed,ep,t)
    drng = np.random.default_rng([123, int(r['episode']), int(r['step'])])
    want = drng.uniform(-1.0, 1.0, (3, 1)).astype(np.float32)
    assert np.array_equal(r['dcands'], want), (r['dcands'], want)
    # uniform branches genuinely evaluated (chaotic env: distinct
    # actions give distinct returns within a repeat)
    assert float(np.ptp(np.concatenate(
        [r['g_dcand_rep'][0], r['g_all_rep'][0, :1]]))) > 1e-6
  # dup composition: uniform set collapses to the plug-in candidate too
  # — every diverse estimand exactly zero
  env2, oracle2 = _W1Env(), _W1Oracle()
  rows2, _ = label_run(env2, oracle2, n_states=4, horizon=12,
                       label_every=7, max_steps=400,
                       rng=np.random.default_rng(0), run_id='wdcdup',
                       oracle_all=True,
                       w1=dict(repeats=3, dcand_uniform=3, env_seed=123,
                               dup_cand=True))
  for r in rows2:
    assert np.all(r['dcands'] == r['cands'][0]), r['dcands']
    assert np.all(r['g_dcand_rep'] == r['g_all_rep'][:, :1]), (
        r['g_dcand_rep'], r['g_all_rep'])
    assert r['dup_cand']
  # save/reload round-trip under the redacted '_wdc' print path
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, 'wdc.npz')
    save_rows(rows, out, meta=dict(labeler_version='d1fix_20260724_wdc'))
    z = np.load(out, allow_pickle=True)
    for key in ('dcands', 'g_dcand_rep', 'g_wit_rep', 'g_all_rep'):
      assert key in z.files, key
  print('SELFCHECK PASS (WDC: uniform candidates pinned-stream '
        'reproducible + genuinely evaluated; candidate-0 CRN witness '
        'bit-identical; dup composition exactly zero on the diverse '
        'set; _wdc redacted save round-trip)')


class _W1CEMEvalOracle(_W1Oracle):
  """Eval oracle exposing a constant CEM plan in extras — but ONLY
  under mode='cemplan' (mirroring the real mode gate, review
  finding 2)."""

  CEM_ACT = 0.35
  CEM_SCORE = 1.5
  modes_seen = None

  def d0_eval(self, carry, obs, mode='eval'):
    carry, q, cands, act, extras = super().d0_eval(carry, obs, mode)
    if self.modes_seen is not None:
      self.modes_seen.append(mode)
    if mode == 'cemplan':
      extras['cem_act'] = np.asarray([self.CEM_ACT], np.float32)
      extras['cem_score'] = float(self.CEM_SCORE)
    return carry, q, cands, act, extras


class _W1CEMFollower(_W1Oracle):
  """CEM follower sharing the eval oracle's RNG counter (the real
  CEMOracle shares the agent's n_actions counter the same way) but
  acting through a DIFFERENT deterministic counter function."""

  def __init__(self, main):
    super().__init__()
    self._main = main

  def rng_mark(self):
    return self._main.rng_mark()

  def rng_reset(self, mark):
    self._main.rng_reset(mark)

  def policy(self, carry, obs, mode='eval'):
    a = np.float32(np.cos(self._main._ctr * 3.7) * 0.3)
    self._main._ctr += 1
    return carry, {'action': np.asarray([[a]], np.float32)}, {}


def selfcheck_wcem():
  """WCEM consumer legs (PREREG_p2_cem_consumer_20260821)."""
  def run(seed_env):
    env, oracle = _W1Env(), _W1CEMEvalOracle()
    oracle.modes_seen = []
    cem = _W1CEMFollower(oracle)
    rows, _ = label_run(env, oracle, n_states=8, horizon=12,
                        label_every=7, max_steps=400,
                        rng=np.random.default_rng(0), run_id='wcemsynth',
                        oracle_all=True, w1=dict(repeats=4, cem=True),
                        cem_oracle=cem)
    return rows, oracle
  rows, oracle0 = run(0)
  # mode gate: cemplan only at labeled states, never in op_real's or
  # rollout_return's bootstrap d0_evals (review finding 2)
  assert oracle0.modes_seen.count('cemplan') == len(rows), (
      oracle0.modes_seen.count('cemplan'), len(rows))
  assert 'eval' in oracle0.modes_seen
  for r in rows:
    assert r['cem_act'].shape == (1,) and r['cem_act'][0] == np.float32(
        _W1CEMEvalOracle.CEM_ACT)
    assert r['g_cem_rep'].shape == (4,)
    # competence diagnostics stored (review finding 1)
    assert r['cem_score'] == np.float32(_W1CEMEvalOracle.CEM_SCORE)
    assert r['q_best'] == np.float32(3.0)   # fixture qfull.mean(0).max()
    # CRN witness bit-identity (review finding 6)
    assert np.array_equal(r['g_wit_rep'], r['g_all_rep'][:, 0])
    # distinct consumer: different first action + follower stream must
    # decorrelate from the actor branch under the chaotic env
    assert float(np.ptp(r['g_cem_rep'] - r['g_actor_rep'])) > 1e-6 or (
        float(abs(r['g_cem_rep'][0] - r['g_actor_rep'][0])) > 1e-6)
  # counter-CRN determinism: a fresh identical run reproduces g_cem_rep
  rows2, _ = run(0)
  for a, b in zip(rows, rows2):
    assert np.array_equal(a['g_cem_rep'], b['g_cem_rep'])
    assert np.array_equal(a['g_all_rep'], b['g_all_rep'])
  # guard: w1.cem without cem_oracle refuses
  try:
    label_run(_W1Env(), _W1CEMEvalOracle(), n_states=1, horizon=12,
              label_every=7, max_steps=400,
              rng=np.random.default_rng(0), run_id='wcembad',
              oracle_all=True, w1=dict(repeats=2, cem=True))
    raise SystemExit('selfcheck FAIL: cem without cem_oracle not refused')
  except AssertionError:
    pass
  # save/reload round-trip under the redacted '_wcem' print path
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, 'wcem.npz')
    save_rows(rows, out,
              meta=dict(labeler_version='d1fix_20260724_wcem'))
    z = np.load(out, allow_pickle=True)
    for key in ('g_cem_rep', 'cem_act', 'g_all_rep', 'g_actor_rep',
                'g_wit_rep', 'cem_score', 'q_best'):
      assert key in z.files, key
  print('SELFCHECK PASS (WCEM: cem branch stored + distinct from actor '
        'under shared marks; mode gate = cemplan at labeled states '
        'only; competence diagnostics (cem_score/q_best) stored; CRN '
        'witness bit-identical; counter-CRN reproduction; '
        'cem-without-oracle guard; _wcem redacted save round-trip)')


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
  p.add_argument('--consumer_checkpoint', default='',
                 help='optional second ckpt whose rew/con/valens value '
                      'heads OVERLAY the eval agent after the main load '
                      '(cross-checkpoint consumer, '
                      'PREREG_r3_amend2_20260730); pol, WM, and disag '
                      'stay the eval checkpoint\'s; pass the eval ckpt '
                      'itself for the control arm')
  p.add_argument('--dual_chooser', action='store_true',
                 help='xc Amendment 1 (PREREG_r3_xc_amend1_20260802): '
                      'evaluate BOTH choosers within one pass — control '
                      '(eval-own heads: m_now/m_real) and shadow '
                      '(consumer heads: m_now_x/m_real_x) on the same '
                      'states, candidates, and g_all; requires '
                      '--consumer_checkpoint (the OTHER maturity) and '
                      '--oracle_all; stamps _xc2')
  p.add_argument('--dual_allow_identical', action='store_true',
                 help='permit byte-identical consumer/control heads in '
                      '--dual_chooser (the registered identical-heads '
                      'smoke ONLY — normally refused as a fabrication '
                      'guard)')
  p.add_argument('--consumer_model', default='',
                 help='optional LORO ridge-model npz '
                      '(d0/train_consumer_model.py) deployed as an '
                      'EXTERNAL chooser for the realized choice '
                      '(competence repair, '
                      'PREREG_competence_repair_20260730); requires '
                      '--oracle_all and composes with no other second-'
                      'checkpoint flag; base trajectory and g_all are '
                      'unaffected by construction')
  p.add_argument('--mass_scale', type=float, default=1.0,
                 help='scale all dm_control body masses at setup '
                      '(physics shift arm); 1.0 = unshifted')
  p.add_argument('--ref_stride', type=int, default=5,
                 help='Record a base-trajectory belief latent every N '
                      'steps (kNN density reference for the R2 read).')
  p.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  p.add_argument('--selfcheck', action='store_true')
  p.add_argument('--w1_repeats', type=int, default=0,
                 help='W1 adjudication wave (PREREG_w1_adjudication_'
                      'constraints_20260809): R independent-mark repeat '
                      'evaluations per (state, branch). 0 = off (default '
                      'path byte-identical).')
  p.add_argument('--w1_dup_cand', action='store_true',
                 help='W1 duplicate-candidate NULL pass: all M '
                      'candidates replaced by the plug-in choice '
                      '(true opportunity exactly 0). Requires '
                      '--w1_repeats > 0.')
  p.add_argument('--env_seed', type=int, default=None,
                 help='Seed the dm_control task RNG explicitly (W1 '
                      'constraint 6). REQUIRED for W1 passes.')
  p.add_argument('--dcand_uniform', type=int, default=0,
                 help='WDC diverse-candidate wave (PREREG_p2_dcand_'
                      '20260821): additionally evaluate N uniform '
                      'U[-1,1]^A candidates per state under the W1 '
                      'repeat marks, plus a candidate-0 CRN witness '
                      're-run per repeat. Requires --w1_repeats > 0; '
                      '0 = off (default path byte-identical).')
  p.add_argument('--stamp_delta', type=float, default=0.0,
                 help='Stage B certified injection (PREREG_p2_stageb_'
                      'injection_20260821): wrap the env in RewardStamp '
                      '(arm-on-restore one-step reward bonus delta * '
                      'clip(a[0])). Requires --w1_repeats > 0; 0.0 = '
                      'off (default path byte-identical).')
  p.add_argument('--cem_consumer', action='store_true',
                 help='WCEM consumer wave (PREREG_p2_cem_consumer_'
                      '20260821): enable the CEM planner over the '
                      'frozen RSSM (agent.d0.cem_* pins via '
                      'load_config) and evaluate a CEM consumer branch '
                      'per repeat alongside the actor branch. Requires '
                      '--w1_repeats > 0; off = default path '
                      'byte-identical.')
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
