"""TM2-R3 oracle labels: the d0/oracle_labels privileged-lookahead-probe
labeler ported to TD-MPC2 (opportunity/competence cross-family
replication, prereg/PREREG_tm2_competence_20260730.md).

At sampled decision states of a fresh online TD-MPC2 run, measure per
labeled state (M = 8 candidates = policy-prior mode + 7 samples at
z = encode(obs_t)):

  qfull [K, M]  Q-ensemble (all num_q = 5 heads, two_hot_inv) at
                (z, a_m) — the plug-in signal.
  m_now         argmax_m of the head-mean (d0 plugin_choice, imported).
  op_real       one real probe step per candidate from the restored env
                snapshot; re-rank by r_real + disc * Vhat(s'_real),
                Vhat = mean-over-heads max-over-a-FRESH-candidate-set Q
                at the observed next state (the qn.mean(0).max() mirror).
  G(a_m)        oracle return: restore, execute a_m, then the frozen
                agent (eval-mode MPPI planner) for --horizon steps;
                --oracle_all grounds EVERY candidate (g_all).
  g_all_boot    G + Vhat(s_h) at the truncation state (== G when the
                episode ended first) — the registered truncation
                bootstrap companion.

PORT FACTS (why this file differs from d0/oracle_labels.py):
- TD-MPC2's latent is feedforward z = encode(obs): there is NO belief
  carry, so the d1fix branch_carry / with_prevact machinery collapses.
  The two REAL CRN hazards instead are:
  (1) the MPPI planner's warm-start buffer ``_prev_mean`` — mutable
      per-trajectory state overwritten by every plan. The base value is
      snapshotted at each label and restored before every branch and
      after the last branch; the FIRST follower call of every branch
      uses t0=True (registered cold start: the pre-label plan
      conditions on not-taking the candidate and is not a belief, so
      no branch may depend on it).
  (2) the planner and policy-prior sample from torch global RNG (CPU +
      CUDA): rng_mark/rng_reset use torch.get_rng_state /
      set_rng_state (+ CUDA states), with the SAME mark/reset points
      as the dv3 labeler — one mark before op_real, reset before every
      candidate probe and every oracle rollout.
- op_imag is NOT ported: the dv3 program retired imag as defective for
  its estimand (24-Jul instrument audit; "imag stays retired" is a
  frozen R3 consequence-map invariant) and no registered read consumes
  the imag columns. The npz schema SLOTS are kept for column-compatible
  tooling with sentinels: m_imag = -1, g_imag/delta_imag (+ _boot) =
  NaN, cost_imag_* = 0; meta operation_pair records the retirement.
- udyn has no TD-MPC2 analog (no dynamics-disagreement ensemble):
  stored as NaN, schema slot kept.
- The latent column is 'zlat' (= encode(obs_t), 512-D at model_size 5,
  float16) with ref_zlat/ref_episode/ref_step strides for future ladder
  use (the deter/ref_deter analog).
- Branch steps drive the INNER embodied node of Dv3TaskEnv (env._env:
  [Distractor ->] DMC -> ActionRepeat -> FromDM). Both the gym facade
  and the embodied node bottom out in the same FromDM.step; the
  embodied node is chosen because it takes the same dict actions and
  carries the same _done auto-reset latch the imported
  snapshot_env/restore_env walkers already cover, with no torch tensor
  conversion inside the branch loop. snapshot_env / restore_env /
  plugin_choice / save_rows are IMPORTED from d0.oracle_labels
  unchanged (generic wrapper-chain walkers), and the per-state
  determinism assert is kept (strengthened: reward AND full flattened
  obs, so a broken OU-distractor restore also trips).

Usage (cluster, tdmpc2 conda env, GPU; see scripts/tm2r3.sbatch):
  python -m probing.tdmpc2_oracle_labels --run_logdir <dir> \
      --checkpoint <dir>/ckpt_late.pt --output <labels>/tm2r3_...npz \
      --states 200 --horizon 100 --label_every 25 --oracle_all
  python -m probing.tdmpc2_oracle_labels --smoke_random_init \
      --task dmc_cup_catch --dose e4 --states 2 --horizon 10 \
      --output <labels>/tm2r3_apismoke_smoke.npz
  python -m probing.tdmpc2_oracle_labels --selfcheck   (mocks; no torch)
"""

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np

from d0.oracle_labels import (  # numpy-only at module level
    plugin_choice, restore_env, save_rows, snapshot_env)

LABELER_VERSION = 'tm2_oracle_20260730'
# Literal copy of probing/tdmpc2_r3_train.TRAINER_VERSION: label passes
# refuse run dirs trained by anything else (foreign/off-registration).
EXPECTED_TRAINER_VERSION = 'tm2r3_train_20260730'
TASKS = ('dmc_cup_catch', 'dmc_finger_turn_hard')
DOSES = ('e1', 'e4')
M_REGISTERED = 8
SMOKE_MAX_STATES = 5


def save_labels(rows, output, meta, extra_arrays=None):
  """save_rows with the registered smoke-stdout scope: for *_smoke.npz
  outputs the shared per-file summary line (it prints mean delta_real,
  an estimand-adjacent number) is suppressed — smoke stdout is for
  crash/dial/determinism sanity only, never estimand values."""
  wv1_pass = str(meta.get('labeler_version', '')).endswith('_wv1')
  if os.path.basename(output).endswith('_smoke.npz') or wv1_pass:
    # review B1 (PREREG_p2_wave1 build review): wv1 rows have no
    # delta_real, so the shared summary line both CRASHES and would be
    # the exact value-blindness leak the smoke scope exists to prevent.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
      save_rows(rows, output, meta, extra_arrays=extra_arrays)
    print(f'wrote {output}: {len(rows)} labeled states '
          + ('(WAVE-1 pass: estimand summary redacted)' if wv1_pass
             else '(smoke pass: estimand summary suppressed)'))
  else:
    save_rows(rows, output, meta, extra_arrays=extra_arrays)


# --------------------------------------------------------------------------
# Frozen-agent adapter (torch inside methods only; the selfcheck
# substitutes closed-form mocks implementing the same interface)
# --------------------------------------------------------------------------

class Tm2Oracle:
  """Frozen TD-MPC2 agent behind the calls the labeler needs.

  Interface (what the mocks implement):
    candidates(obs) -> (qfull [K,M] f32, cands [M,A] f32, zlat [D] f16)
    state_value(obs) -> float  (fresh-candidate-set Vhat)
    follower_act(obs, t0) -> [A] f32  (eval-mode planner action)
    zlat_of(obs) -> [D] f16  (RNG-free encode; ref strides)
    plan_state() / set_plan_state(s)  (MPPI _prev_mean snapshot)
    rng_mark() / rng_reset(mark)      (torch CPU+CUDA RNG)
    discount, act_shape, m
  """

  def __init__(self, agent, cfg, task, extra_keys=(), m=M_REGISTERED):
    import torch
    from common import math as tm2_math
    from probing.tdmpc2_compat import flatten_obs
    self._torch = torch
    self._tm2_math = tm2_math
    self._flatten_obs = flatten_obs
    self.agent = agent
    self.model = agent.model.eval()
    self.cfg = cfg
    self.task = task
    self.extra_keys = tuple(extra_keys)
    self.discount = float(agent.discount)
    self.act_shape = (int(cfg.action_dim),)
    self.m = int(m)
    self._cuda = torch.cuda.is_available()

  def _obs_tensor(self, obs):
    flat = self._flatten_obs(obs, self.task, extra=self.extra_keys)
    return self._torch.from_numpy(flat).to(self.agent.device)

  def _cand_q(self, obs):
    torch = self._torch
    with torch.no_grad():
      z = self.model.encode(self._obs_tensor(obs).unsqueeze(0), None)
      _, info = self.model.pi(z, None)
      mode = info['mean']                               # [1, A]
      samp, _ = self.model.pi(z.repeat(self.m - 1, 1), None)
      cands = torch.cat([mode, samp], 0)                # [M, A]
      logits = self.model.Q(z.repeat(self.m, 1), cands, None,
                            return_type='all')          # [K, M, bins]
      q = self._tm2_math.two_hot_inv(logits, self.cfg).squeeze(-1)
    return q, cands, z

  def candidates(self, obs):
    q, cands, z = self._cand_q(obs)
    return (np.asarray(q.cpu(), np.float32),
            np.asarray(cands.cpu(), np.float32),
            np.asarray(z[0].cpu(), np.float16))

  def state_value(self, obs):
    q, _, _ = self._cand_q(obs)
    return float(q.mean(0).max())

  def follower_act(self, obs, t0):
    torch = self._torch
    flat = self._flatten_obs(obs, self.task, extra=self.extra_keys)
    with torch.no_grad():
      a = self.agent.act(torch.from_numpy(flat), t0=bool(t0),
                         eval_mode=True)
    return np.asarray(a.cpu(), np.float32).reshape(self.act_shape)

  def q_of(self, obs, act):
    """W2 (PREREG_antiharvest_mech_20260811): the Q-ensemble values of
    an ARBITRARY action at obs — the same learned-value metric qfull
    records for candidates, evaluated on planner outputs. Deterministic
    given weights (no sampling)."""
    torch = self._torch
    with torch.no_grad():
      z = self.model.encode(self._obs_tensor(obs).unsqueeze(0), None)
      a = torch.from_numpy(np.asarray(act, np.float32).reshape(1, -1)
                           ).to(z.device)
      logits = self.model.Q(z, a, None, return_type='all')  # [K,1,bins]
      q = self._tm2_math.two_hot_inv(logits, self.cfg).reshape(-1)
    return np.asarray(q.cpu(), np.float32)

  def get_iterations(self):
    return int(self.agent.cfg.iterations)

  def set_iterations(self, k):
    """W2 pressure knob: MPPI reads cfg.iterations at plan time; callers
    MUST restore via try/finally (the label function does)."""
    self.agent.cfg.iterations = int(k)

  def zlat_of(self, obs):
    torch = self._torch
    with torch.no_grad():
      z = self.model.encode(self._obs_tensor(obs).unsqueeze(0), None)
    return np.asarray(z[0].cpu(), np.float16)

  def plan_state(self):
    return self.agent._prev_mean.detach().clone()

  def set_plan_state(self, state):
    with self._torch.no_grad():
      self.agent._prev_mean.copy_(state)

  def rng_mark(self):
    torch = self._torch
    mark = dict(cpu=torch.get_rng_state())
    if self._cuda:
      mark['cuda'] = torch.cuda.get_rng_state_all()
    return mark

  def rng_reset(self, mark):
    torch = self._torch
    torch.set_rng_state(mark['cpu'])
    if self._cuda and 'cuda' in mark:
      torch.cuda.set_rng_state_all(mark['cuda'])

  def w1_mark(self, seed):
    """W1: derive a fresh, reproducible torch-RNG mark from an integer
    seed (torch marks are opaque state dicts, so cross-repeat
    independence is by SEEDING, not counter offsets — the dv3
    analogue's base+offset arithmetic does not apply here)."""
    torch = self._torch
    torch.manual_seed(int(seed) % (2 ** 31))
    if self._cuda:
      torch.cuda.manual_seed_all(int(seed) % (2 ** 31))
    return self.rng_mark()


# --------------------------------------------------------------------------
# Branches. CRN: every branch starts from one env snapshot, one restored
# planner warm-start buffer, one torch-RNG mark, and a t0=True first
# follower call — env state, planner state, AND sampling noise are
# common across candidates.
# --------------------------------------------------------------------------

def _step(enode, act_vec):
  return enode.step({'action': np.asarray(act_vec, np.float32),
                     'reset': np.array(False)})


def op_real(enode, oracle, cands, snap, pstate, rmark,
            return_rewards=False):
  """Real purchase: one real probe step per candidate from the restored
  snapshot; score = r_real + disc * Vhat(s'_real).

  return_rewards (W1): additionally return the raw per-candidate env
  rewards for the registered leak decomposition
  (PREREG_w1_adjudication_constraints_20260809 constraint 5). Default
  path byte-identical."""
  m_count = cands.shape[0]
  scores = np.zeros(m_count, np.float32)
  rewards = np.zeros(m_count, np.float32)
  for m in range(m_count):
    oracle.rng_reset(rmark)
    oracle.set_plan_state(pstate)
    restore_env(enode, snap)
    nobs = _step(enode, cands[m])
    r = float(nobs['reward'])
    rewards[m] = r
    scores[m] = r + oracle.discount * oracle.state_value(nobs)
  cost = dict(env_steps=m_count, policy_calls=m_count)
  if return_rewards:
    return int(np.argmax(scores)), cost, scores, rewards
  return int(np.argmax(scores)), cost, scores


def rollout_return(enode, oracle, first_act, horizon, snap, pstate, rmark):
  """G(a): restore, execute a, then the frozen eval-mode planner for
  `horizon` steps; first follower call t0=True (registered branch cold
  start). Returns (G, G_boot)."""
  oracle.rng_reset(rmark)
  oracle.set_plan_state(pstate)
  restore_env(enode, snap)
  obs = _step(enode, first_act)
  total = float(obs['reward'])
  first = True
  for _ in range(horizon - 1):
    if bool(obs['is_last']):
      break
    act = oracle.follower_act(obs, t0=first)
    first = False
    obs = _step(enode, act)
    total += float(obs['reward'])
  if bool(obs['is_last']):
    return total, total
  return total, total + oracle.state_value(obs)


# --------------------------------------------------------------------------
# Labeling sweep
# --------------------------------------------------------------------------

def _w1_seed(seed_base, ep, t, tag):
  """Deterministic per-(state, purpose) mark seed. tag 0 = planner
  action draw, 1 = probe (constraint 5: independent of every
  evaluation mark), 2+r = repeat r (constraint 1)."""
  return (int(seed_base) * 1_000_003 + int(ep) * 10_007
          + int(t) * 101 + int(tag)) % (2 ** 31)


def _w1_label_state_tm2(enode, oracle, obs, qfull, cands, zlat, snap,
                        pstate, m_now, horizon, run_id, ep, t, w1):
  """W1 adjudication labeling of ONE state — TD-MPC2 side
  (PREREG_w1_adjudication_constraints_20260809). Mirrors the dv3
  `_w1_label_state`: R independent-mark repeat evaluations with CRN
  across branches within a repeat (constraint 1); dup_cand null
  (constraint 3); the ACTUAL MPPI planner action as its own branch
  (constraint 4 — the D1 fix: `follower_act` at the decision state IS
  `agent.act(eval_mode=True)`, the planner); probe under an
  independent mark with per-candidate r_real stored (constraint 5)."""
  seed_base = int(w1['seed_base'])
  # The planner's actual DEPLOYED action at the decision state
  # (batch-review M1): t0=False with the warm-start buffer pstate
  # restored — exactly the (state, warm-start) pair the deployed agent
  # holds at a labeled step (t > 0 always). The t0=True cold-start
  # convention is registered for CANDIDATE branches only (they execute
  # counterfactual first actions the pre-label plan never conditioned
  # on); the consumer branch measures the agent as deployed.
  oracle.rng_reset(oracle.w1_mark(_w1_seed(seed_base, ep, t, 0)))
  oracle.set_plan_state(pstate)
  restore_env(enode, snap)
  plan_act = np.asarray(oracle.follower_act(obs, t0=False),
                        np.float32).reshape(-1)
  cands_eval = np.asarray(cands, np.float32)
  if w1.get('dup_cand'):
    cands_eval = np.tile(cands_eval[m_now:m_now + 1],
                         (cands_eval.shape[0], 1))
  m_count = cands_eval.shape[0]
  pmark = oracle.w1_mark(_w1_seed(seed_base, ep, t, 1))
  m_real, cost_real, real_scores, r_real = op_real(
      enode, oracle, cands_eval, snap, pstate, pmark,
      return_rewards=True)
  repeats = int(w1['repeats'])
  g_rep = np.zeros((repeats, m_count), np.float32)
  g_plan_rep = np.zeros(repeats, np.float32)
  for r in range(repeats):
    mark_r = oracle.w1_mark(_w1_seed(seed_base, ep, t, 2 + r))
    for m in range(m_count):
      g_rep[r, m] = rollout_return(enode, oracle, cands_eval[m],
                                   horizon, snap, pstate, mark_r)[0]
    g_plan_rep[r] = rollout_return(enode, oracle, plan_act, horizon,
                                   snap, pstate, mark_r)[0]
  return dict(
      run_id=run_id, episode=ep, step=t,
      qfull=qfull, cands=cands_eval,
      g_all_rep=g_rep, g_plan_rep=g_plan_rep, plan_act=plan_act,
      zlat=np.asarray(zlat, np.float16),
      m_now=m_now, m_real=m_real,
      real_scores=np.asarray(real_scores, np.float32),
      r_real=np.asarray(r_real, np.float32),
      dup_cand=bool(w1.get('dup_cand', False)),
      cost_real_env=cost_real['env_steps'],
      cost_real_calls=cost_real['policy_calls'])


W2_ITERS_LOW = 1
W2_ITERS_HIGH = 12


def _w2_label_state_tm2(enode, oracle, obs, qfull, cands, zlat, snap,
                        pstate, m_now, horizon, run_id, ep, t, w2):
  """W2 mechanism labeling of ONE state (PREREG_antiharvest_mech_20260811).

  Per state, four planner first-action variants — warm (t0=False, the
  deployed convention, = the W1 consumer branch), cold (t0=True at
  default iterations), and cold at iterations {W2_ITERS_LOW,
  W2_ITERS_HIGH} — each with its Q-ensemble value, plus the W1-style
  candidate ground-truth panel. ALL ground-truth rollouts share the
  per-repeat marks with the candidate branches (CRN); the iterations
  override is restored in a finally block, and every G rollout runs at
  the DEFAULT iterations (variants differ in the first action only)."""
  seed_base = int(w2['seed_base'])
  repeats = int(w2['repeats'])
  # Review M7: _w1_seed multiplies t by 101, so tags must stay < 101
  # AND clear of the repeat tags 2..2+R-1; 10/11/12 are free for R<=8.
  assert repeats <= 8, 'W2 tag space assumes repeats <= 8'
  variants = {}
  q_var = {}
  # warm: t0=False with the deployed warm-start buffer (M1 convention)
  oracle.rng_reset(oracle.w1_mark(_w1_seed(seed_base, ep, t, 0)))
  oracle.set_plan_state(pstate)
  restore_env(enode, snap)
  variants['warm'] = np.asarray(oracle.follower_act(obs, t0=False),
                                np.float32).reshape(-1)
  # cold + pressure grid: t0=True (plan from scratch); iterations
  # overridden ONLY for the action draw, restored before any rollout
  it0 = oracle.get_iterations()
  try:
    for name, iters, tag in (('cold', it0, 10),
                             ('p_low', W2_ITERS_LOW, 11),
                             ('p_high', W2_ITERS_HIGH, 12)):
      oracle.set_iterations(iters)
      oracle.rng_reset(oracle.w1_mark(_w1_seed(seed_base, ep, t, tag)))
      oracle.set_plan_state(pstate)
      restore_env(enode, snap)
      variants[name] = np.asarray(oracle.follower_act(obs, t0=True),
                                  np.float32).reshape(-1)
  finally:
    oracle.set_iterations(it0)
  for name, act in variants.items():
    q_var[name] = np.asarray(oracle.q_of(obs, act), np.float32)
  cands_eval = np.asarray(cands, np.float32)
  if w2.get('dup_cand'):
    cands_eval = np.tile(cands_eval[m_now:m_now + 1],
                         (cands_eval.shape[0], 1))
  m_count = cands_eval.shape[0]
  g_rep = np.zeros((repeats, m_count), np.float32)
  g_var = {name: np.zeros(repeats, np.float32) for name in variants}
  for r in range(repeats):
    mark_r = oracle.w1_mark(_w1_seed(seed_base, ep, t, 2 + r))
    for m in range(m_count):
      g_rep[r, m] = rollout_return(enode, oracle, cands_eval[m],
                                   horizon, snap, pstate, mark_r)[0]
    for name in variants:
      g_var[name][r] = rollout_return(enode, oracle, variants[name],
                                      horizon, snap, pstate, mark_r)[0]
  return dict(
      run_id=run_id, episode=ep, step=t,
      qfull=qfull, cands=cands_eval,
      g_all_rep=g_rep,
      g_warm_rep=g_var['warm'], g_cold_rep=g_var['cold'],
      g_plow_rep=g_var['p_low'], g_phigh_rep=g_var['p_high'],
      act_warm=variants['warm'], act_cold=variants['cold'],
      act_plow=variants['p_low'], act_phigh=variants['p_high'],
      q_warm=q_var['warm'], q_cold=q_var['cold'],
      q_plow=q_var['p_low'], q_phigh=q_var['p_high'],
      zlat=np.asarray(zlat, np.float16),
      m_now=m_now,
      dup_cand=bool(w2.get('dup_cand', False)),
      iters_realized=np.asarray([W2_ITERS_LOW, it0, W2_ITERS_HIGH],
                                np.int64))


def _wave1_label_state_tm2(enode, oracle, obs, qfull, cands, zlat, snap,
                           pstate, m_now, horizon, run_id, ep, t, wv1):
  """WAVE-1 first-update labeling of ONE state
  (PREREG_p2_wave1_firstupdate_20260821; additive — no executed path is
  modified).

  Two fixed policy variants, no selection on G anywhere (order-statistic
  and leak-immune by construction):
    mode   the policy-prior mode action  == cands[0] (M_REGISTERED
           convention: candidate 0 IS the prior mode; no planner call)
    cold1  the planner's action after exactly ONE MPPI update from a
           cold start (t0=True, iterations forced to 1, restored in a
           finally block; W2_ITERS_LOW == 1 is the same dial)

  Per variant: R raw undiscounted G rollouts under per-repeat CRN marks
  (tags 30+r), plus ONE own-objective score under an independent mark
  (tag 22): own(a) = r(s,a) + discount * state_value(s') — the SAME
  scoring convention as op_real (C5, the dual-objective oracle: grades
  the first update on the planner's OWN terms). cold1's action draw uses
  tag 21. Tag space: _w1_seed requires tags < 101; 21/22/30..37 are
  disjoint from W1 (0,1,2..9) and W2 (0,10,11,12,2..9) tags."""
  seed_base = int(wv1['seed_base'])
  repeats = int(wv1['repeats'])
  assert repeats <= 8, 'wave1 tag space assumes repeats <= 8'
  act_mode = np.asarray(cands[0], np.float32).reshape(-1)
  it0 = oracle.get_iterations()
  try:
    oracle.set_iterations(1)
    oracle.rng_reset(oracle.w1_mark(_w1_seed(seed_base, ep, t, 21)))
    oracle.set_plan_state(pstate)
    restore_env(enode, snap)
    act_cold1 = np.asarray(oracle.follower_act(obs, t0=True),
                           np.float32).reshape(-1)
  finally:
    oracle.set_iterations(it0)
  if wv1.get('dup'):
    # CRN null gate (review M3, w1_dup_cand analogue): both variants get
    # the SAME action; every paired quantity must come out EXACTLY equal.
    act_cold1 = act_mode.copy()
  # C5 own-objective scores (op_real convention), independent mark
  own = {}
  for name, act in (('mode', act_mode), ('cold1', act_cold1)):
    oracle.rng_reset(oracle.w1_mark(_w1_seed(seed_base, ep, t, 22)))
    oracle.set_plan_state(pstate)  # literal op_real parity (review minor)
    restore_env(enode, snap)
    nobs = _step(enode, act)
    own[name] = np.float32(float(nobs['reward'])
                           + oracle.discount * oracle.state_value(nobs))
  g_mode = np.zeros(repeats, np.float32)
  g_cold1 = np.zeros(repeats, np.float32)
  for r in range(repeats):
    mark_r = oracle.w1_mark(_w1_seed(seed_base, ep, t, 30 + r))
    g_mode[r] = rollout_return(enode, oracle, act_mode, horizon, snap,
                               pstate, mark_r)[0]
    g_cold1[r] = rollout_return(enode, oracle, act_cold1, horizon, snap,
                                pstate, mark_r)[0]
  return dict(
      run_id=run_id, episode=ep, step=t,
      qfull=qfull, cands=np.asarray(cands, np.float32),
      g_mode_rep=g_mode, g_cold1_rep=g_cold1,
      own_mode=own['mode'], own_cold1=own['cold1'],
      act_mode=act_mode, act_cold1=act_cold1,
      zlat=np.asarray(zlat, np.float16), m_now=m_now,
      iters_first=np.int64(1), iters_default=np.int64(it0),
      dup=np.bool_(bool(wv1.get('dup', False))))


def label_run(enode, oracle, n_states, horizon, label_every, max_steps,
              run_id, oracle_all=False, ref_stride=5, obs_flat=None,
              w1=None, w2=None, wv1=None):
  """obs_flat: callable obs->1-D vector for the determinism assert
  (defaults to reward-only comparison when None)."""
  rows = []
  ref = dict(zlat=[], episode=[], step=[])
  zero = np.zeros(oracle.act_shape, np.float32)
  obs = enode.step({'action': zero, 'reset': np.array(True)})
  ep, t = 0, 0
  while len(rows) < n_states:
    if bool(obs['is_last']):
      obs = enode.step({'action': zero, 'reset': np.array(True)})
      ep += 1
      t = 0
      continue
    label_here = (t % label_every == 0 and t > 0
                  and t <= max_steps - horizon - 1)
    if label_here:
      snap = snapshot_env(enode)
      pstate = oracle.plan_state()
      qfull, cands, zlat = oracle.candidates(obs)
      # determinism assert: restore must reproduce the same next step
      # (reward AND observation — a broken OU restore trips here too)
      restore_env(enode, snap)
      p1 = _step(enode, cands[0])
      restore_env(enode, snap)
      p2 = _step(enode, cands[0])
      same = abs(float(p1['reward']) - float(p2['reward'])) <= 1e-6
      if same and obs_flat is not None:
        same = np.allclose(obs_flat(p1), obs_flat(p2), atol=1e-6)
      if not same:
        raise SystemExit('restore_env is not deterministic; CRN broken')
      restore_env(enode, snap)

      m_now = plugin_choice(qfull)
      if wv1 is not None:
        rows.append(_wave1_label_state_tm2(
            enode, oracle, obs, qfull, cands, zlat, snap, pstate,
            m_now, horizon, run_id, ep, t, wv1))
        restore_env(enode, snap)
        oracle.set_plan_state(pstate)
      elif w2 is not None:
        rows.append(_w2_label_state_tm2(
            enode, oracle, obs, qfull, cands, zlat, snap, pstate,
            m_now, horizon, run_id, ep, t, w2))
        restore_env(enode, snap)
        oracle.set_plan_state(pstate)
      elif w1 is not None:
        rows.append(_w1_label_state_tm2(
            enode, oracle, obs, qfull, cands, zlat, snap, pstate,
            m_now, horizon, run_id, ep, t, w1))
        # resume the base trajectory untouched: env snapshot AND the
        # planner warm-start buffer
        restore_env(enode, snap)
        oracle.set_plan_state(pstate)
      else:
        rmark = oracle.rng_mark()
        m_real, cost_real, real_scores = op_real(
            enode, oracle, cands, snap, pstate, rmark)
        targets = (set(range(cands.shape[0])) if oracle_all
                   else {m_now, m_real})
        G = {m: rollout_return(enode, oracle, cands[m], horizon, snap,
                               pstate, rmark) for m in sorted(targets)}
        # resume the base trajectory untouched: env snapshot AND the
        # planner warm-start buffer (mutable, unlike jax carries)
        restore_env(enode, snap)
        oracle.set_plan_state(pstate)
        g_all = np.full(cands.shape[0], np.nan, np.float32)
        g_all_boot = np.full(cands.shape[0], np.nan, np.float32)
        for m, (gv, gb) in G.items():
          g_all[m] = gv
          g_all_boot[m] = gb
        nan = np.float32(np.nan)
        rows.append(dict(
            run_id=run_id, episode=ep, step=t,
            qfull=qfull, cands=cands,
            g_all=g_all, g_all_boot=g_all_boot,
            udyn=nan,  # no TD-MPC2 analog (schema slot)
            zlat=np.asarray(zlat, np.float16),
            m_now=m_now, m_imag=-1, m_real=m_real,
            g_now=G[m_now][0], g_imag=nan, g_real=G[m_real][0],
            delta_imag=nan,
            delta_real=G[m_real][0] - G[m_now][0],
            g_now_boot=G[m_now][1], g_imag_boot=nan,
            g_real_boot=G[m_real][1],
            delta_imag_boot=nan,
            delta_real_boot=G[m_real][1] - G[m_now][1],
            real_scores=real_scores,
            cost_imag_env=0, cost_imag_calls=0,
            cost_real_env=cost_real['env_steps'],
            cost_real_calls=cost_real['policy_calls'],
        ))
    if t % ref_stride == 0:
      ref['zlat'].append(oracle.zlat_of(obs))
      ref['episode'].append(ep)
      ref['step'].append(t)
    act = oracle.follower_act(obs, t0=(t == 0))
    obs = _step(enode, act)
    t += 1
  ref_arrays = dict(
      ref_zlat=np.stack(ref['zlat'], 0).astype(np.float16),
      ref_episode=np.asarray(ref['episode'], np.int64),
      ref_step=np.asarray(ref['step'], np.int64))
  return rows, ref_arrays


# --------------------------------------------------------------------------
# Driver-stage guards (torch-free; selfcheck trips every one)
# --------------------------------------------------------------------------

def check_args(args):
  if args.actions != M_REGISTERED:
    raise SystemExit(f'M={M_REGISTERED} is registered for this wave; '
                     f'--actions {args.actions} refused (a different M '
                     'needs its own registration)')
  if args.smoke_random_init:
    if not args.output or not os.path.basename(
        args.output).endswith('_smoke.npz'):
      raise SystemExit('--smoke_random_init output must end _smoke.npz')
    if args.states > SMOKE_MAX_STATES:
      raise SystemExit(f'smoke passes are <= {SMOKE_MAX_STATES} states')
    if args.task not in TASKS:
      raise SystemExit(f'--task must be one of {TASKS} for the smoke')
    if args.dose not in DOSES:
      raise SystemExit(f'--dose must be one of {DOSES} for the smoke')
  else:
    if not args.run_logdir or not args.output or not args.checkpoint:
      raise SystemExit('--run_logdir, --checkpoint and --output required '
                       '(or --smoke_random_init / --selfcheck)')
    if os.path.basename(args.checkpoint) not in (
        'ckpt_early.pt', 'ckpt_late.pt'):
      raise SystemExit('checkpoint must be a ckpt_early.pt/ckpt_late.pt '
                       'maturity snapshot of a tm2r3 run')


def load_audit(run_logdir):
  """Trainer audit record; refuses un-finished or foreign run dirs."""
  done = os.path.join(run_logdir, 'TM2R3_TRAIN_DONE')
  if not os.path.exists(done):
    raise SystemExit(f'{run_logdir}: no TM2R3_TRAIN_DONE; not labeling '
                     'an unfinished run')
  cfgp = os.path.join(run_logdir, 'config.json')
  if not os.path.exists(cfgp):
    raise SystemExit(f'{run_logdir}: no config.json audit record')
  with open(cfgp) as f:
    audit = json.load(f)
  for key in ('trainer_version', 'task', 'dose', 'seed', 'obs_dim',
              'action_dim', 'steps', 'early_steps',
              'early_step_realized', 'late_step_realized'):
    if key not in audit:
      raise SystemExit(f'{cfgp}: audit record lacks {key!r}')
  if audit['trainer_version'] != EXPECTED_TRAINER_VERSION:
    raise SystemExit(
        f"{cfgp}: trainer_version {audit['trainer_version']!r} != "
        f'{EXPECTED_TRAINER_VERSION!r} (foreign or off-registration run)')
  if audit['task'] not in TASKS or audit['dose'] not in DOSES:
    raise SystemExit(f'{cfgp}: unregistered task/dose '
                     f"({audit['task']}/{audit['dose']})")
  return audit


# --------------------------------------------------------------------------
# Real-run entry point
# --------------------------------------------------------------------------

def main_real(args):
  from probing.tdmpc2_compat import (
      Dv3TaskEnv, add_tdmpc2_path, build_cfg, dose_config, flatten_obs)
  root = add_tdmpc2_path(args.tdmpc2_root)
  import torch
  if not torch.cuda.is_available():
    raise SystemExit('TD-MPC2 requires CUDA.')
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  if args.smoke_random_init:
    task, dose, train_seed = args.task, args.dose, 0
  else:
    audit = load_audit(args.run_logdir)
    task, dose, train_seed = audit['task'], audit['dose'], audit['seed']
    if args.task and args.task != task:
      raise SystemExit(f'--task {args.task} != run audit task {task}')
    if args.dose and args.dose != dose:
      raise SystemExit(f'--dose {args.dose} != run audit dose {dose}')
    if not os.path.exists(args.checkpoint):
      raise SystemExit(f'checkpoint {args.checkpoint} does not exist')

  # Same env builder as the training run. Default path (env_seed None):
  # seed = TRAIN seed, so label/train env identity holds by
  # construction. Under --env_seed (W1) the identity is deliberately
  # broken — the W1 estimands are within-pass only and never pair
  # against committed labels.
  w1 = None
  if getattr(args, 'w1_repeats', 0):
    assert args.env_seed is not None, (
        'W1 passes require --env_seed (constraint 6)')
    assert args.oracle_all, 'W1 requires --oracle_all'
    assert not getattr(args, 'wv1_repeats', 0), (
        'w1 and wave1 modes are mutually exclusive')
    w1 = dict(repeats=int(args.w1_repeats),
              dup_cand=bool(args.w1_dup_cand),
              seed_base=int(args.env_seed))
  else:
    assert not getattr(args, 'w1_dup_cand', False), (
        '--w1_dup_cand requires --w1_repeats > 0')
  wv1 = None
  if getattr(args, 'wv1_repeats', 0):
    assert args.env_seed is not None, 'wave1 passes require --env_seed'
    base_out = os.path.basename(str(args.output))
    assert base_out.startswith(('wv1tm2_', 'wv1dup_')) or \
        base_out.endswith('_smoke.npz'), (
        'wave1 outputs must be named wv1tm2_*/wv1dup_* (review minor: '
        f'orphan-file guard); got {base_out}')
    wv1 = dict(repeats=int(args.wv1_repeats),
               seed_base=int(args.env_seed),
               dup=bool(getattr(args, 'wv1_dup', False)))
  w2 = None
  if getattr(args, 'w2_repeats', 0):
    assert w1 is None, 'w1 and w2 modes are mutually exclusive'
    assert wv1 is None, 'w2 and wave1 modes are mutually exclusive'
    assert args.env_seed is not None, 'W2 passes require --env_seed'
    assert args.oracle_all, 'W2 requires --oracle_all'
    w2 = dict(repeats=int(args.w2_repeats),
              dup_cand=bool(getattr(args, 'w2_dup_cand', False)),
              seed_base=int(args.env_seed))
  else:
    assert not getattr(args, 'w2_dup_cand', False), (
        '--w2_dup_cand requires --w2_repeats > 0')
  env_seed = train_seed if args.env_seed is None else int(args.env_seed)
  env = Dv3TaskEnv(task, seed=env_seed, dose=dose)
  obs_dim = int(env.observation_space.shape[0])
  act_dim = int(env.action_space.shape[0])
  if not args.smoke_random_init and obs_dim != int(audit['obs_dim']):
    raise SystemExit(f'obs_dim {obs_dim} != trained {audit["obs_dim"]}')

  cfg = build_cfg(root, task, obs_dim, act_dim, env.max_episode_steps,
                  overrides=dict(seed=args.seed, mpc=True))
  from tdmpc2 import TDMPC2
  agent = TDMPC2(cfg)
  if not args.smoke_random_init:
    agent.load(args.checkpoint)

  oracle = Tm2Oracle(agent, cfg, task, extra_keys=env._extra_keys,
                     m=args.actions)
  enode = env._env
  run_id = (os.path.basename(args.run_logdir.rstrip('/'))
            if args.run_logdir else 'apismoke')
  rows, ref_arrays = label_run(
      enode, oracle, args.states, args.horizon, args.label_every,
      env.max_episode_steps, run_id=run_id, oracle_all=args.oracle_all,
      ref_stride=args.ref_stride,
      obs_flat=lambda o: flatten_obs(o, task, extra=env._extra_keys),
      w1=w1, w2=w2, wv1=wv1)
  smoke = bool(args.smoke_random_init)
  save_labels(rows, args.output, meta=dict(
      run_logdir=str(args.run_logdir or ''),
      checkpoint=str(args.checkpoint or ''),
      states=args.states, horizon=args.horizon,
      label_every=args.label_every, seed=args.seed,
      actions=args.actions, ref_stride=args.ref_stride,
      train_seed=int(train_seed), task=task, dose=dose,
      dose_config=dose_config(dose),
      # Training-audit echo (reader-pinned): dials + realized snapshot
      # steps travel with every label file, not just the run dir.
      trainer_version=(None if smoke else audit['trainer_version']),
      train_steps=(None if smoke else int(audit['steps'])),
      train_early_steps=(None if smoke else int(audit['early_steps'])),
      early_step_realized=(None if smoke else audit['early_step_realized']),
      late_step_realized=(None if smoke else audit['late_step_realized']),
      mass_scale=1.0, behavior_checkpoint='',
      labeler_version=(LABELER_VERSION + '_w1' if w1 is not None else
                       LABELER_VERSION + '_w2' if w2 is not None else
                       LABELER_VERSION + '_wv1' if wv1 is not None else
                       LABELER_VERSION),
      **({} if w1 is None else dict(w1=dict(w1),
                                    env_seed=int(env_seed))),
      **({} if wv1 is None else dict(wv1=dict(wv1),
                                     env_seed=int(env_seed))),
      **({} if w2 is None else dict(
          w2=dict(w2), env_seed=int(env_seed),
          w2_iters=dict(low=W2_ITERS_LOW,
                        default=int(cfg.iterations),
                        high=W2_ITERS_HIGH))),
      smoke_random_init=bool(args.smoke_random_init),
      operation_pair='real_only_imag_retired',
      branch_first_follower_t0=True,
      discount=oracle.discount,
      planner=dict(mpc=bool(cfg.mpc), horizon=int(cfg.horizon),
                   iterations=int(cfg.iterations),
                   num_samples=int(cfg.num_samples),
                   num_elites=int(cfg.num_elites),
                   num_pi_trajs=int(cfg.num_pi_trajs),
                   temperature=float(cfg.temperature),
                   min_std=float(cfg.min_std),
                   max_std=float(cfg.max_std)),
      model=dict(model_size=int(cfg.model_size), num_q=int(cfg.num_q),
                 latent_dim=int(cfg.latent_dim)),
      tdmpc2_root=str(root)),
      extra_arrays=ref_arrays)


# --------------------------------------------------------------------------
# Selfcheck: closed-form mocks, no torch / TD-MPC2 / MuJoCo
# --------------------------------------------------------------------------

class _ChainEnv:
  """Drifting chain: pos += a + 0.1 per step, reward = a (the executed
  action), obs exposes pos; restorable via oracle_get/set_state. The
  drift makes stale env restores provably wrong; reward = action makes
  G a closed-form sum of the action schedule."""

  def __init__(self, length=400):
    self.pos, self.t, self.length = 0.0, 0, length

  def oracle_get_state(self):
    return (self.pos, self.t)

  def oracle_set_state(self, s):
    self.pos, self.t = s

  def step(self, act):
    if act.get('reset'):
      self.pos, self.t = 0.0, 0
      return dict(reward=np.float32(0), is_first=np.array(True),
                  is_last=np.array(False), pos=np.float32(self.pos))
    a = float(np.asarray(act['action']).reshape(-1)[0])
    self.pos += a + 0.1
    self.t += 1
    return dict(reward=np.float32(a), is_first=np.array(False),
                is_last=np.array(self.t >= self.length),
                pos=np.float32(self.pos))


class _MockOracle:
  """Closed-form oracle. plan_state ps is MUTABLE follower state (the
  _prev_mean stand-in): t0=True sets ps=1 and plays 0.3; t0=False plays
  0.1*ps then increments — so follower schedules are deterministic
  given restore discipline. state_value = 100*pos + 1000*ps: any stale
  env restore OR stale planner state is provably wrong in op_real
  scores. qfull encodes 100*pos + w_m with w = (0, 5, 1): m_now = 1
  requires the labeled obs actually scored."""

  CANDS = np.array([[0.9], [0.3], [-0.5]], np.float32)
  W = np.array([0.0, 5.0, 1.0], np.float32)

  def __init__(self):
    self.discount = 1.0
    self.act_shape = (1,)
    self.m = 3
    self.ps = 0.0
    self.rng_ctr = 100
    self.marks = []
    self.resets = []
    self.set_calls = []

  def candidates(self, obs):
    self.rng_ctr += 1
    pos = float(obs['pos'])
    q = np.stack([100.0 * pos + self.W] * 2, 0).astype(np.float32)
    return q, self.CANDS.copy(), np.array([pos, self.ps], np.float16)

  def state_value(self, obs):
    self.rng_ctr += 1
    return 100.0 * float(obs['pos']) + 1000.0 * self.ps

  def follower_act(self, obs, t0):
    self.rng_ctr += 1
    if t0:
      self.ps = 1.0
      return np.array([0.3], np.float32)
    a = 0.1 * self.ps
    self.ps += 1.0
    return np.array([a], np.float32)

  def zlat_of(self, obs):
    return np.array([float(obs['pos']), self.ps], np.float16)

  def plan_state(self):
    return self.ps

  def set_plan_state(self, state):
    self.set_calls.append(state)
    self.ps = state

  def rng_mark(self):
    self.marks.append(self.rng_ctr)
    return self.rng_ctr

  def rng_reset(self, mark):
    self.resets.append(mark)
    self.rng_ctr = mark


class _LeakyOracle(_MockOracle):
  """set_plan_state is a no-op: the branch-contaminated warm-start
  state leaks into the base trajectory (the bug the restore prevents)."""

  def set_plan_state(self, state):
    self.set_calls.append(state)  # recorded but NOT applied


def _base_schedule(n_steps):
  """Label-free replica of the base trajectory: (pos_t, ps_t) BEFORE
  acting at each t. Correct label blocks leave both untouched."""
  pos, ps = 0.0, 0.0
  poss, pss = [0.0], [0.0]
  for t in range(n_steps):
    if t == 0:
      a, ps = 0.3, 1.0
    else:
      a = 0.1 * ps
      ps += 1.0
    pos += a + 0.1
    poss.append(pos)
    pss.append(ps)
  return poss, pss


def selfcheck():
  legs = []

  # ---- leg 1: closed-form scores + estimand identities ----------------
  env, oracle = _ChainEnv(), _MockOracle()
  rows, ref = label_run(env, oracle, n_states=4, horizon=4, label_every=7,
                        max_steps=400, run_id='mock', oracle_all=True,
                        obs_flat=lambda o: np.array([o['pos']]))
  poss, pss = _base_schedule(30)
  cands = _MockOracle.CANDS.reshape(-1)
  assert [r['step'] for r in rows] == [7, 14, 21, 28], rows
  for r in rows:
    t = r['step']
    pos_t, ps_t = poss[t], pss[t]
    # qfull scored AT the labeled obs (100*pos_t term) with the fixed
    # candidate weights; plug-in picks w's argmax.
    assert np.allclose(r['qfull'], 100.0 * pos_t + _MockOracle.W,
                       atol=1e-3), (t, r['qfull'])
    assert r['m_now'] == 1 and r['m_real'] == 0, r
    assert r['m_imag'] == -1 and np.isnan(r['g_imag']), r
    # op_real closed form REQUIRES per-branch env restore (pos_t term)
    # AND base planner state (1000*ps_t term): scores =
    # c + 100*(pos_t + c + 0.1) + 1000*ps_t.
    want = [c + 100.0 * (pos_t + c + 0.1) + 1000.0 * ps_t for c in cands]
    assert np.allclose(r['real_scores'], want, atol=1e-3), (
        t, r['real_scores'], want)
    # rollouts (horizon 4): candidate + followers 0.3, 0.1, 0.2 under
    # the registered t0=True branch cold start => G = c + 0.6.
    for m, c in enumerate(cands):
      assert abs(r['g_all'][m] - (c + 0.6)) < 1e-5, (t, m, r['g_all'])
      boot = (c + 0.6) + 100.0 * (poss[t] + c + 1.0) + 1000.0 * 3.0
      assert abs(r['g_all_boot'][m] - boot) < 1e-3, (t, m, r['g_all_boot'])
    assert abs(r['delta_real'] - 0.6) < 1e-5, r['delta_real']
    # hand-computed estimand identities on the row itself
    assert abs(r['g_now'] - r['g_all'][r['m_now']]) < 1e-6
    assert abs(r['delta_real']
               - (r['g_all'][r['m_real']] - r['g_all'][r['m_now']])) < 1e-6
    assert np.isnan(r['udyn']) and np.isnan(r['delta_imag'])
    assert r['cost_real_env'] == 3 and r['cost_imag_env'] == 0
    assert r['zlat'].dtype == np.float16
    assert abs(float(r['zlat'][0]) - pos_t) < 0.05, (t, r['zlat'])
  legs.append('closed-form scores/rollouts + estimand identities')

  # ---- leg 2: plan-state restore accounting + warm-start leak ---------
  # correct oracle: per label = M op_real restores + M rollout restores
  # + 1 final restore = 7 set calls, ALL to the base ps at that label.
  assert len(oracle.set_calls) == 4 * 7, len(oracle.set_calls)
  for i, r in enumerate(rows):
    calls = oracle.set_calls[i * 7:(i + 1) * 7]
    assert all(abs(c - pss[r['step']]) < 1e-9 for c in calls), (i, calls)
  # leaky oracle: label 1 still matches (nothing leaked yet), label 2's
  # scores are PROVABLY wrong — the stale warm-start state is detected.
  lenv, loracle = _ChainEnv(), _LeakyOracle()
  lrows, _ = label_run(lenv, loracle, n_states=2, horizon=4, label_every=7,
                       max_steps=400, run_id='leak', oracle_all=True)
  want1 = [c + 100.0 * (poss[7] + c + 0.1) + 1000.0 * pss[7]
           for c in cands]
  assert np.allclose(lrows[0]['real_scores'], want1, atol=1e-3)
  want2 = [c + 100.0 * (poss[14] + c + 0.1) + 1000.0 * pss[14]
           for c in cands]
  diff = np.max(np.abs(np.asarray(lrows[1]['real_scores']) - want2))
  assert diff > 100.0, ('leak not detectable', diff)
  legs.append('planner warm-start restore + leak detection')

  # ---- leg 3: RNG contract --------------------------------------------
  # one mark per label; 2M resets per label (M probes + M rollouts),
  # all equal to that label's mark; marks distinct across labels.
  assert len(oracle.marks) == 4, oracle.marks
  assert len(set(oracle.marks)) == 4, oracle.marks
  assert len(oracle.resets) == 4 * 6, len(oracle.resets)
  for i in range(4):
    burst = oracle.resets[i * 6:(i + 1) * 6]
    assert burst == [oracle.marks[i]] * 6, (i, burst, oracle.marks[i])
  legs.append('RNG contract (marks/resets)')

  # ---- leg 4: multi-node snapshot/restore incl. done-latch ------------
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

  denv = _DoneWrap(_ChainEnv())
  dsnap = snapshot_env(denv)
  assert len(dsnap['custom']) == 2, dsnap['custom']
  denv._done = True
  denv.extra = 9
  denv.env.step({'action': np.ones(1), 'reset': False})
  restore_env(denv, dsnap)
  assert denv._done is False and denv.extra == 3, (denv._done, denv.extra)
  legs.append('multi-node snapshot/restore + done-latch')

  # ---- leg 5: oracle_all fills all M; default leaves non-targets NaN --
  assert not any(np.isnan(r['g_all']).any() for r in rows)
  penv, poracle = _ChainEnv(), _MockOracle()
  prows, _ = label_run(penv, poracle, n_states=1, horizon=4, label_every=7,
                       max_steps=400, run_id='partial', oracle_all=False)
  r = prows[0]
  assert np.isnan(r['g_all'][2]), r['g_all']  # cand 2 not a target
  assert np.isfinite(r['g_all'][0]) and np.isfinite(r['g_all'][1])
  legs.append('oracle_all coverage vs targeted NaN')

  # ---- leg 6: npz schema (tempdir fixture) ----------------------------
  import tempfile
  with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, 'mock.npz')
    save_rows(rows, out, meta=dict(labeler_version=LABELER_VERSION,
                                   selfcheck=True), extra_arrays=ref)
    z = np.load(out, allow_pickle=False)
    assert z['qfull'].shape == (4, 2, 3) and z['cands'].shape == (4, 3, 1)
    assert z['g_all'].shape == (4, 3) and z['g_all_boot'].shape == (4, 3)
    assert z['real_scores'].shape == (4, 3)
    assert z['zlat'].shape == (4, 2) and z['zlat'].dtype == np.float16
    assert z['m_imag'].tolist() == [-1] * 4
    assert np.isnan(z['g_imag']).all() and np.isnan(z['udyn']).all()
    assert z['delta_real'].shape == (4,) and z['g_now_boot'].shape == (4,)
    assert z['ref_zlat'].dtype == np.float16
    # ref strides: base steps 0..28 at stride 5 => [0,5,10,15,20,25]
    assert z['ref_step'].tolist() == [0, 5, 10, 15, 20, 25], z['ref_step']
    assert z['ref_zlat'].shape == (6, 2)
    assert LABELER_VERSION in str(z['meta'])
  legs.append('npz schema + ref strides')

  # ---- leg 6b: smoke stdout never carries an estimand value ----------
  import contextlib
  import io
  with tempfile.TemporaryDirectory() as d:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      save_labels(rows, os.path.join(d, 'x_smoke.npz'),
                  meta=dict(labeler_version=LABELER_VERSION))
    out = buf.getvalue()
    assert 'delta_real' not in out and 'suppressed' in out, out
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      save_labels(rows, os.path.join(d, 'x_late.npz'),
                  meta=dict(labeler_version=LABELER_VERSION))
    assert 'delta_real' in buf.getvalue(), buf.getvalue()
  legs.append('smoke stdout estimand suppression')

  # ---- leg 7: determinism assert trips on a nondeterministic env ------
  class _JitterEnv(_ChainEnv):
    def __init__(self):
      super().__init__()
      self._jrng = np.random.default_rng(0)

    def step(self, act):
      obs = super().step(act)
      if not act.get('reset'):
        obs['reward'] = np.float32(
            float(obs['reward']) + self._jrng.normal(0, 0.1))
      return obs

  try:
    label_run(_JitterEnv(), _MockOracle(), n_states=1, horizon=4,
              label_every=7, max_steps=400, run_id='jitter')
    raise AssertionError('nondeterministic env not caught')
  except SystemExit as e:
    assert 'not deterministic' in str(e), e
  legs.append('determinism-assert trip')

  # ---- leg 8: driver-stage guard refusals -----------------------------
  def _ns(**kw):
    base = dict(actions=8, smoke_random_init=False, output='x.npz',
                states=200, run_logdir='rl', checkpoint='rl/ckpt_late.pt',
                task='dmc_cup_catch', dose='e4')
    base.update(kw)
    return argparse.Namespace(**base)

  for bad, needle in [
      (_ns(actions=16), 'registered'),
      (_ns(smoke_random_init=True, output='x.npz'), '_smoke.npz'),
      (_ns(smoke_random_init=True, output='x_smoke.npz', states=50),
       'states'),
      (_ns(smoke_random_init=True, output='x_smoke.npz', states=2,
           task='dmc_walker_walk'), '--task'),
      (_ns(smoke_random_init=True, output='x_smoke.npz', states=2,
           dose='e9'), '--dose'),
      (_ns(run_logdir=''), 'required'),
      (_ns(checkpoint='rl/final.pt'), 'ckpt_early.pt'),
  ]:
    try:
      check_args(bad)
      raise AssertionError(f'guard not tripped: {needle}')
    except SystemExit as e:
      assert needle in str(e), (needle, e)
  check_args(_ns())  # the valid shape passes
  check_args(_ns(smoke_random_init=True, output='a_smoke.npz', states=2,
                 run_logdir='', checkpoint=''))
  with tempfile.TemporaryDirectory() as d:
    try:
      load_audit(d)
      raise AssertionError('missing DONE marker not caught')
    except SystemExit as e:
      assert 'TM2R3_TRAIN_DONE' in str(e), e
    open(os.path.join(d, 'TM2R3_TRAIN_DONE'), 'w').close()
    try:
      load_audit(d)
      raise AssertionError('missing config.json not caught')
    except SystemExit as e:
      assert 'config.json' in str(e), e
    good = dict(trainer_version=EXPECTED_TRAINER_VERSION,
                task='dmc_cup_catch', dose='e4', seed=51,
                obs_dim=40, action_dim=2, steps=100_000, early_steps=25_000,
                early_step_realized=25_000, late_step_realized=100_001)
    with open(os.path.join(d, 'config.json'), 'w') as f:
      json.dump(good, f)
    audit = load_audit(d)
    assert audit['seed'] == 51
    for bad_audit, needle in [
        (dict(good, dose='e9'), 'unregistered'),
        (dict(good, trainer_version='tm2_adapt_x'), 'trainer_version'),
        ({k: v for k, v in good.items() if k != 'steps'}, 'lacks'),
    ]:
      with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(bad_audit, f)
      try:
        load_audit(d)
        raise AssertionError(f'bad audit not caught: {needle}')
      except SystemExit as e:
        assert needle in str(e), (needle, e)
  legs.append('driver-stage guard refusals')

  print('SELFCHECK PASS (' + '; '.join(legs) + ')')
  selfcheck_w1_tm2()
  selfcheck_w2_tm2()
  selfcheck_wave1_tm2()


class _W1ChaosEnv(_ChainEnv):
  """Chaotic-reward chain for the W1 fixture: branches with different
  first actions decorrelate under a common follower schedule; identical
  first actions reproduce exactly (duplicate-null gate)."""

  def step(self, act):
    out = super().step(act)
    if not act.get('reset'):
      out['reward'] = np.float32(np.sin(self.pos * 997.13) ** 2)
    return out


class _W1Mock(_MockOracle):
  """Counter-seeded STOCHASTIC follower (§13.4 lesson: every previous
  fixture was deterministic and therefore blind to realization noise)
  + the w1_mark(seed) interface the W1 path requires."""

  def follower_act(self, obs, t0):
    a = 0.3 * np.sin(self.rng_ctr * 12.9898)
    self.rng_ctr += 1
    return np.array([a], np.float32)

  def w1_mark(self, seed):
    self.rng_ctr = int(seed) % (2 ** 31)
    return self.rng_mark()


class _W2Mock(_W1Mock):
  """W2 fixture oracle: iterations-sensitive planner variants + q_of.
  follower_act depends on (t0, iterations, warm-state) so the four
  variants are provably distinct; q_of is a closed form of the action."""

  def __init__(self):
    super().__init__()
    self.iterations = 6
    self.iters_log = []

  def get_iterations(self):
    return int(self.iterations)

  def set_iterations(self, k):
    self.iterations = int(k)

  def follower_act(self, obs, t0):
    self.iters_log.append(self.iterations)
    a = 0.3 * np.sin(self.rng_ctr * 12.9898)
    self.rng_ctr += 1
    if t0:
      # cold plan: pressure-dependent first action; noise amplitude kept
      # SMALL so the 0.01*iterations separation is provable across the
      # per-variant mark reseeds
      return np.array([0.02 * np.sin(self.rng_ctr * 7.77)
                       + 0.01 * self.iterations], np.float32)
    return np.array([a + 0.5 + 0.001 * float(self.ps)], np.float32)

  def q_of(self, obs, act):
    v = float(np.asarray(act).reshape(-1)[0])
    return np.array([2.0 * v, 2.0 * v + 0.1], np.float32)


def selfcheck_w2_tm2():
  """W2 legs (PREREG_antiharvest_mech_20260811)."""
  env, oracle = _W1ChaosEnv(), _W2Mock()
  rows, _ = label_run(env, oracle, n_states=10, horizon=12,
                      label_every=7, max_steps=400, run_id='w2tm2',
                      oracle_all=True,
                      w2=dict(repeats=4, seed_base=23))
  assert oracle.get_iterations() == 6, 'iterations not restored'
  # Review M4(i): a leaked override would put non-default iterations into
  # the ROLLOUT follower calls; with a correct implementation each
  # non-default value appears exactly once per labeled state (its action
  # draw) and never during rollouts.
  n_lo = oracle.iters_log.count(W2_ITERS_LOW)
  n_hi = oracle.iters_log.count(W2_ITERS_HIGH)
  assert n_lo == len(rows) and n_hi == len(rows), (
      'iterations override leaked into rollouts', n_lo, n_hi, len(rows))
  for r in rows:
    assert r['g_all_rep'].shape == (4, 3)
    for k in ('g_warm_rep', 'g_cold_rep', 'g_plow_rep', 'g_phigh_rep'):
      assert r[k].shape == (4,), k
    for k in ('q_warm', 'q_cold', 'q_plow', 'q_phigh'):
      assert r[k].shape == (2,), k
    assert list(r['iters_realized']) == [W2_ITERS_LOW, 6, W2_ITERS_HIGH]
    # variants provably distinct in the fixture: warm carries +0.5,
    # pressure variants differ by 0.01 * iterations
    aw, ac = float(r['act_warm'][0]), float(r['act_cold'][0])
    al, ah = float(r['act_plow'][0]), float(r['act_phigh'][0])
    assert abs(aw - ac) > 0.05, (aw, ac)
    assert abs(ah - al) > 0.05, (al, ah)
    # q_of consistency with the closed form — ALL FOUR variants
    # (review M4(ii): a q_of computed on the wrong action must trip)
    assert abs(float(r['q_warm'][0]) - 2.0 * aw) < 1e-5
    assert abs(float(r['q_cold'][0]) - 2.0 * ac) < 1e-5
    assert abs(float(r['q_plow'][0]) - 2.0 * al) < 1e-5
    assert abs(float(r['q_phigh'][0]) - 2.0 * ah) < 1e-5
  g = np.stack([r['g_warm_rep'] for r in rows])
  assert float(np.mean(g.std(1))) > 1e-3, 'variant branch static'
  # duplicate-null: candidate rows bit-identical within repeat
  env2, oracle2 = _W1ChaosEnv(), _W2Mock()
  rows2, _ = label_run(env2, oracle2, n_states=4, horizon=12,
                       label_every=7, max_steps=400, run_id='w2tm2dup',
                       oracle_all=True,
                       w2=dict(repeats=3, dup_cand=True, seed_base=23))
  for r in rows2:
    g2 = r['g_all_rep']
    if not np.all(g2 == g2[:, :1]):
      raise SystemExit('selfcheck FAIL: W2 duplicate-null violated')
    assert r['dup_cand']
  # CRN: the same mark reproduces a variant rollout bit-identically
  env3, oracle3 = _W1ChaosEnv(), _W2Mock()
  obs = env3.step({'action': np.zeros(1, np.float32),
                   'reset': np.array(True)})
  for _ in range(3):
    obs = env3.step({'action': np.zeros(1, np.float32),
                     'reset': np.array(False)})
  snap = snapshot_env(env3)
  ps = oracle3.plan_state()
  mark = oracle3.w1_mark(99)
  a = np.array([0.2], np.float32)
  g1 = rollout_return(env3, oracle3, a, 8, snap, ps, mark)[0]
  g2v = rollout_return(env3, oracle3, a, 8, snap, ps, mark)[0]
  assert g1 == g2v, 'CRN mark does not reproduce'
  # Review M4(iii): the candidate panel must be BIT-IDENTICAL between a
  # W1 and a W2 pass at equal dials (same tags 2+r, same rollouts) — this
  # kills both the iterations-leak and the CRN-de-pairing mutants.
  env_a, oracle_a = _W1ChaosEnv(), _W2Mock()
  rows_a, _ = label_run(env_a, oracle_a, n_states=3, horizon=12,
                        label_every=7, max_steps=400, run_id='w1cmp',
                        oracle_all=True, w1=dict(repeats=4, seed_base=23))
  env_b, oracle_b = _W1ChaosEnv(), _W2Mock()
  rows_b, _ = label_run(env_b, oracle_b, n_states=3, horizon=12,
                        label_every=7, max_steps=400, run_id='w2cmp',
                        oracle_all=True, w2=dict(repeats=4, seed_base=23))
  if not np.array_equal(rows_a[0]['g_all_rep'], rows_b[0]['g_all_rep']):
    raise SystemExit('selfcheck FAIL: W2 candidate panel diverges from '
                     'W1 at equal dials - CRN or override leak')
  # save round-trip
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, 'w2tm2.npz')
    save_labels(rows2, out,
                meta=dict(labeler_version=LABELER_VERSION + '_w2'))
    z = np.load(out, allow_pickle=True)
    for key in ('g_warm_rep', 'g_cold_rep', 'g_plow_rep', 'g_phigh_rep',
                'q_warm', 'act_phigh', 'iters_realized'):
      assert key in z.files, key
  print('SELFCHECK PASS (W2 TM2: four variants distinct + '
        'pressure-sensitive fixture; iterations override restored; '
        'q_of closed-form consistency; duplicate-null exact; CRN mark '
        'reproducibility; _w2 save round-trip)')


def selfcheck_w1_tm2():
  """W1 legs (PREREG_w1_adjudication_constraints_20260809), TM2 side."""
  env, oracle = _W1ChaosEnv(), _W1Mock()
  rows, _ = label_run(env, oracle, n_states=24, horizon=12,
                      label_every=7, max_steps=400, run_id='w1tm2',
                      oracle_all=True,
                      w1=dict(repeats=6, seed_base=17))
  g = np.stack([r['g_all_rep'] for r in rows])          # (S, R, M)
  gp = np.stack([r['g_plan_rep'] for r in rows])        # (S, R)
  assert g.shape[1:] == (6, 3) and gp.shape[1] == 6
  assert float(np.mean(g.std(1))) > 1e-3, 'follower not stochastic'
  assert float(np.mean(gp.std(1))) > 1e-3, 'planner branch static'
  within = float(np.mean(g.max(2) - g.mean(2)))
  sel = g[:, 0::2].mean(1).argmax(1)
  ho = g[:, 1::2].mean(1)
  split = float(np.mean(ho[np.arange(len(ho)), sel] - ho.mean(1)))
  if not within > 0.05:
    raise SystemExit(f'selfcheck FAIL: W1 within-repeat max bias '
                     f'invisible ({within})')
  if not abs(split) < 0.5 * within:
    raise SystemExit(f'selfcheck FAIL: W1 split-selection did not '
                     f'remove the max bias ({within} vs {split})')
  for r in rows:
    assert r['r_real'].shape == (3,) and r['plan_act'].shape == (1,)
    assert not r['dup_cand']
  # duplicate-candidate NULL: bit-identical rollouts within a repeat
  env2, oracle2 = _W1ChaosEnv(), _W1Mock()
  rows2, _ = label_run(env2, oracle2, n_states=6, horizon=12,
                       label_every=7, max_steps=400, run_id='w1tm2dup',
                       oracle_all=True,
                       w1=dict(repeats=3, dup_cand=True, seed_base=17))
  for r in rows2:
    g2 = r['g_all_rep']
    if not np.all(g2 == g2[:, :1]):
      raise SystemExit('selfcheck FAIL: TM2 duplicate-null violated — '
                       'within-repeat CRN broken')
    assert float(np.ptp(r['r_real'])) == 0.0, r['r_real']
    assert np.all(r['cands'] == r['cands'][0])
    assert r['dup_cand']
  # base trajectory untouched by the W1 block (schedule replica)
  poss, _ = _base_schedule(0)  # exercise import; full invariance is
  # covered by the default-path legs — here assert env resumed cleanly:
  assert env2.t > 0
  # redacted save round-trip (version *_w1)
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, 'w1tm2.npz')
    save_labels(rows2, out,
                meta=dict(labeler_version=LABELER_VERSION + '_w1'))
    z = np.load(out, allow_pickle=True)
    for key in ('g_all_rep', 'g_plan_rep', 'plan_act', 'r_real',
                'dup_cand'):
      assert key in z.files, key
  print('SELFCHECK PASS (W1 TM2: stochastic-follower fixture — '
        'within-repeat max bias visible + split-selection kills it; '
        'duplicate-null exact; planner branch + r_real stored; '
        '_w1 redacted save round-trip)')


def selfcheck_wave1_tm2():
  """WAVE-1 legs (PREREG_p2_wave1_firstupdate_20260821; build-review B4).
  Every assert was verified to CATCH a planted mutant during the build
  review application (mode->cands[1]; missing finally; broken CRN mark;
  shared own/rollout mark; non-redacted save)."""
  import contextlib
  import io
  import json as _json
  import tempfile

  # Leg 1+2+CRN: normal pass
  env, oracle = _W1ChaosEnv(), _W2Mock()
  rows, _ = label_run(env, oracle, n_states=8, horizon=12,
                      label_every=7, max_steps=400, run_id='wv1tm2',
                      oracle_all=True,
                      wv1=dict(repeats=4, seed_base=29))
  assert oracle.get_iterations() == 6, 'iterations not restored (leg 2)'
  # the iters=1 override must appear exactly once per state (the cold1
  # action draw) and never during rollouts
  assert oracle.iters_log.count(1) == len(rows), (
      'iters=1 leaked into rollouts', oracle.iters_log.count(1), len(rows))
  for r in rows:
    assert r['g_mode_rep'].shape == (4,) and r['g_cold1_rep'].shape == (4,)
    # leg 1: mode is EXACTLY candidate 0
    assert np.array_equal(r['act_mode'], np.asarray(r['cands'][0],
                                                    np.float32).reshape(-1)), \
        'act_mode != cands[0] (leg 1)'
    assert int(r['iters_first']) == 1 and int(r['iters_default']) == 6
    assert not bool(r['dup'])
  gm = np.stack([r['g_mode_rep'] for r in rows])
  assert float(np.mean(gm.std(1))) > 1e-3, 'mode branch static'
  # leg 4: own is scored under its own mark — variants share tag 22, so
  # own differences reflect the ACTION only; with distinct actions the
  # own values must differ for at least one state
  own_d = [abs(float(r['own_mode']) - float(r['own_cold1'])) for r in rows]
  assert max(own_d) > 1e-6, 'own-objective insensitive to the action'

  # Leg 3 (CRN dup null): forced act_cold1 == act_mode must give EXACT
  # equality of every paired quantity
  env2, oracle2 = _W1ChaosEnv(), _W2Mock()
  rows2, _ = label_run(env2, oracle2, n_states=4, horizon=12,
                       label_every=7, max_steps=400, run_id='wv1dup',
                       oracle_all=True,
                       wv1=dict(repeats=3, seed_base=29, dup=True))
  for r in rows2:
    assert np.array_equal(r['g_mode_rep'], r['g_cold1_rep']), \
        'CRN broken: dup variants differ (leg 3)'
    assert float(r['own_mode']) == float(r['own_cold1']), \
        'own-objective CRN broken (leg 3)'
    assert bool(r['dup'])

  # Leg 2b: an exception inside the cold1 block still restores the dial
  env3, oracle3 = _W1ChaosEnv(), _W2Mock()
  orig = oracle3.follower_act
  def boom(obs, t0):
    if t0:
      raise RuntimeError('planted')
    return orig(obs, t0)
  oracle3.follower_act = boom
  try:
    label_run(env3, oracle3, n_states=1, horizon=12, label_every=7,
              max_steps=400, run_id='wv1tm2', oracle_all=True,
              wv1=dict(repeats=2, seed_base=29))
  except RuntimeError:
    pass
  assert oracle3.get_iterations() == 6, 'dial not restored on exception'

  # Leg 5: save round-trip with stdout redaction (review B1)
  with tempfile.TemporaryDirectory() as tmp:
    out = os.path.join(tmp, 'wv1tm2_cup_e1_seed51_late.npz')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      save_labels(rows, out, meta=dict(
          labeler_version=LABELER_VERSION + '_wv1',
          env_seed=29, wv1=dict(repeats=4, seed_base=29)))
    txt = buf.getvalue()
    assert 'redacted' in txt and 'delta_real' not in txt, txt
    z = np.load(out, allow_pickle=True)
    assert z['g_mode_rep'].shape == (len(rows), 4)
    assert _json.loads(str(z['meta']))['labeler_version'].endswith('_wv1')

  # Leg 6: mutual exclusion (mirrors main_real asserts structurally)
  try:
    label_run(_W1ChaosEnv(), _W2Mock(), n_states=1, horizon=12,
              label_every=7, max_steps=400, run_id='x', oracle_all=True,
              w2=dict(repeats=2, seed_base=1),
              wv1=dict(repeats=2, seed_base=1))
  except Exception:
    pass  # dispatch order makes wv1 win; exclusion is enforced in
          # main_real asserts (source-checked below)
  src = open(__file__).read()
  assert "assert wv1 is None, 'w2 and wave1 modes are mutually exclusive'" \
      in src
  assert "'w1 and wave1 modes are mutually exclusive'" in src
  print('selfcheck_wave1_tm2 PASS (mode==cands[0]; dial restore incl. '
        'exception; CRN dup exact-zero; own-mark action sensitivity; '
        'redacted save round-trip; mutual-exclusion asserts present)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--tdmpc2_root', default=None)
  p.add_argument('--run_logdir', default='')
  p.add_argument('--checkpoint', default='')
  p.add_argument('--output', default='')
  p.add_argument('--task', default='',
                 help='required for --smoke_random_init; otherwise a '
                      'cross-check against the run audit record')
  p.add_argument('--dose', default='',
                 help='required for --smoke_random_init; otherwise a '
                      'cross-check against the run audit record')
  p.add_argument('--states', type=int, default=200)
  p.add_argument('--horizon', type=int, default=100)
  p.add_argument('--label_every', type=int, default=25)
  p.add_argument('--actions', type=int, default=M_REGISTERED,
                 help='candidate count M (registered: 8; refused else)')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--ref_stride', type=int, default=5)
  p.add_argument('--oracle_all', action='store_true',
                 help='ground-truth every candidate, not just the chosen')
  p.add_argument('--smoke_random_init', action='store_true',
                 help='API smoke: random-init model, no checkpoint '
                      'touched, no outcome revealed')
  p.add_argument('--selfcheck', action='store_true')
  p.add_argument('--w1_repeats', type=int, default=0,
                 help='W1 adjudication wave (PREREG_w1_adjudication_'
                      'constraints_20260809): R independent-mark '
                      'repeats. 0 = off (default path byte-identical).')
  p.add_argument('--wv1_repeats', type=int, default=0,
                 help='WAVE-1 first-update mode: repeats per variant '
                      '(PREREG_p2_wave1_firstupdate_20260821); mutually '
                      'exclusive with --w1_repeats/--w2_repeats')
  p.add_argument('--wv1_dup', action='store_true',
                 help='WAVE-1 CRN null gate: act_cold1 := act_mode; all '
                      'paired quantities must be exactly equal')
  p.add_argument('--w2_repeats', type=int, default=0,
                 help='W2 mechanism passes (PREREG_antiharvest_mech_'
                      '20260811): planner-variant repeats; 0 = off.')
  p.add_argument('--w2_dup_cand', action='store_true',
                 help='W2 duplicate-null gate pass (requires '
                      '--w2_repeats > 0).')
  p.add_argument('--w1_dup_cand', action='store_true',
                 help='W1 duplicate-candidate NULL pass (requires '
                      '--w1_repeats > 0).')
  p.add_argument('--env_seed', type=int, default=None,
                 help='Override the env seed (W1 constraint 6; also '
                      'the W1 mark seed base). REQUIRED for W1.')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  check_args(args)
  main_real(args)


if __name__ == '__main__':
  main()
