"""Goodhart's-Simulator harness: a world model as a policy EVALUATOR.

Sprint design (innovation audit 13 Jul, "Evaluator-Goodhart sprint"):
score a pool of existing policy checkpoints by their IMAGINED return
inside one learned world model M (WM + reward/continue heads), compare
against their real environment returns (already measured in each adapt
run's scores.jsonl), and ask whether ordinary rank correlation survives
*selection pressure*: top-k regret after choosing policies by M-score
(adaptivity level 2) versus the pool-wide correlation (level 1).

Mechanics of one (M, pi) evaluation: both models warm up on the same
real observation prefix (their own filtering); then for H steps pi acts
from its own recurrent state on the OBSERVATIONS DECODED BY M, and M
advances its latent with pi's action, emitting reward and continue
probabilities. M is a neural simulator interacting with pi through the
observation interface only; pi and M may come from entirely different
runs (separate agents, separate parameter dicts).

Usage
-----
Score one policy under one evaluator (GPU or CPU)::

    python -m probing.wm_evaluator score \
        --evaluator_run $RUNROOT/adapt_ax1q1s1_finger_seed1_ckpt500000 \
        --policy_run    $RUNROOT/adapt_ax1q1s0_finger_seed3_ckpt500000 \
        --init_replay   $RUNROOT/pilot_goal_finger/replay \
        --output $RUNROOT/goodhart_finger/eval_seed3_s0.json

Pool metrics (rank correlation, top-k regret, inversions)::

    python -m probing.wm_evaluator collate \
        --evals '$RUNROOT/goodhart_finger/eval_*.json' \
        --runroot $RUNROOT --real_window 10 \
        --output $RUNROOT/goodhart_finger/pool_metrics.json
"""

import argparse
import glob as globlib
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np

from probing import probeset as probeset_mod


def episode_start_windows(replay_dir, burn_in, n_starts, seed):
  """(B, burn_in) arrays per key, windows anchored at episode starts."""
  starts = []
  for chain in probeset_mod.chain_streams(replay_dir):
    stream = probeset_mod.load_stream(chain)
    first = np.asarray(stream['is_first'], bool)
    idxs = np.flatnonzero(first)
    for i in idxs:
      if i + burn_in <= len(first) and not first[i + 1:i + burn_in].any():
        starts.append((stream, i))
  if len(starts) < n_starts:
    raise SystemExit(f'Only {len(starts)} clean episode-start windows of '
                     f'length {burn_in} in {replay_dir} (< {n_starts}).')
  rng = np.random.default_rng(seed)
  pick = rng.choice(len(starts), size=n_starts, replace=False)
  keys = sorted(starts[0][0].keys())
  out = {}
  for k in keys:
    out[k] = np.stack([starts[i][0][k][starts[i][1]:starts[i][1] + burn_in]
                       for i in pick], 0)
  return out


def build_agent(run_logdir, platform, scratch):
  from dreamerv3.main import make_agent
  from probing.collect import load_run_config, load_frozen_agent
  config = load_run_config(run_logdir, platform, scratch, False)
  config = config.update({'jax': {
      'precompile': False, 'enable_policy': False, 'prealloc': False}})
  agent = make_agent(config)
  load_frozen_agent(agent, os.path.join(run_logdir, 'ckpt'))
  return agent, config


def cmd_score(args):
  import jax
  import jax.numpy as jnp
  import ninjax as nj

  f32 = jnp.float32
  scratch = args.scratch or os.path.join(
      os.path.dirname(os.path.abspath(args.output)) or '.', '_wm_eval_tmp')
  ev_agent, ev_cfg = build_agent(args.evaluator_run, args.platform, scratch)
  pi_agent, pi_cfg = build_agent(args.policy_run, args.platform, scratch)
  jax.config.update('jax_transfer_guard', 'allow')
  M, PI = ev_agent.model, pi_agent.model

  assert ev_cfg.agent.expl.mode == 'task' or not args.require_task, (
      f'evaluator {args.evaluator_run} is not task-mode; its reward head is '
      f'untrained (pass --require_task False to override).')

  exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
  obs_keys = sorted(k for k, v in ev_agent.obs_space.items()
                    if k not in exclude and len(v.shape) <= 1)
  act_keys = sorted(ev_agent.act_space.keys())
  windows = episode_start_windows(
      args.init_replay, args.burn_in, args.n_starts, args.seed)
  B, P = windows['is_first'].shape[0], args.burn_in

  def warm(model, obs, action_dict, reset):
    Bn = reset.shape[0]
    enc_carry = model.enc.initial(Bn)
    dyn_carry = model.dyn.initial(Bn)
    enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)
    prevact = {k: jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], 1)
               for k, v in action_dict.items()}
    dyn_carry, _, post = model.dyn.observe(
        dyn_carry, tokens, prevact, reset, training=False)
    latent = {'deter': post['deter'][:, -1], 'stoch': post['stoch'][:, -1]}
    return enc_carry, dyn_carry, latent

  def policy_action(latent):
    dists = PI.pol(PI.feat2tensor(latent), 1)
    return jax.tree.map(lambda d: f32(d.pred()), dists)

  def pi_warm(obs, action_dict, reset):
    enc_carry, dyn_carry, latent = warm(PI, obs, action_dict, reset)
    return enc_carry, dyn_carry, policy_action(latent)

  def pi_step(enc_carry, dyn_carry, obs, prevact):
    reset = jnp.zeros(obs[obs_keys[0]].shape[:2], bool)
    enc_carry, _, tokens = PI.enc(enc_carry, obs, reset, training=False)
    dyn_carry, _, post = PI.dyn.observe(
        dyn_carry, tokens, prevact, reset, training=False)
    latent = {'deter': post['deter'][:, -1], 'stoch': post['stoch'][:, -1]}
    return enc_carry, dyn_carry, policy_action(latent)

  def m_warm(obs, action_dict, reset):
    _, _, latent = warm(M, obs, action_dict, reset)
    return latent

  def m_step(latent, action):
    acts = {k: v[:, None] for k, v in action.items()}
    _, feat, _ = M.dyn.imagine(latent, acts, 1, training=False)
    nxt = {'deter': feat['deter'][:, -1], 'stoch': feat['stoch'][:, -1]}
    seq = {'deter': nxt['deter'][:, None], 'stoch': nxt['stoch'][:, None]}
    reset = jnp.zeros((nxt['deter'].shape[0], 1), bool)
    _, _, rec = M.dec(M.dec.initial(nxt['deter'].shape[0]), seq, reset,
                      training=False)
    obs = {k: f32(rec[k].pred()) for k in obs_keys}
    inp = M.feat2tensor(seq)
    rew = f32(M.rew(inp, 2).pred())[:, 0]
    con = f32(M.con(inp, 2).prob(1))[:, 0]
    return nxt, obs, rew, con

  jit = lambda fn: (lambda pure: jax.jit(
      lambda p, *a: pure(p, *a, seed=jnp.array([args.seed, 1], np.uint32))
      ))(nj.pure(fn))
  j_pi_warm, j_pi_step = jit(pi_warm), jit(pi_step)
  j_m_warm, j_m_step = jit(m_warm), jit(m_step)

  ev_params = jax.tree.map(np.asarray, ev_agent.params)
  pi_params = jax.tree.map(np.asarray, pi_agent.params)

  obs0 = {k: jnp.asarray(windows[k], np.float32) for k in obs_keys}
  acts0 = {k: jnp.asarray(windows[k], np.float32) for k in act_keys}
  reset0 = jnp.asarray(np.asarray(windows['is_first'], bool))

  _, (enc_c, dyn_c, act) = j_pi_warm(pi_params, obs0, acts0, reset0)
  _, latent = j_m_warm(ev_params, obs0, acts0, reset0)

  gamma = 1.0 - 1.0 / float(ev_cfg.agent.horizon)
  ret = np.zeros(B)
  ret_disc = np.zeros(B)
  alive = np.ones(B)
  for t in range(args.horizon):
    action = {k: np.asarray(v) for k, v in
              (act.items() if isinstance(act, dict) else [('action', act)])}
    _, (latent, obs, rew, con) = j_m_step(ev_params, latent, {
        k: jnp.asarray(v) for k, v in action.items()})
    rew, con = np.asarray(rew), np.asarray(con)
    ret += np.asarray(rew) * alive
    ret_disc += (gamma ** t) * np.asarray(rew) * alive
    alive = alive * (np.asarray(con) > 0.5)
    obs_in = {k: jnp.asarray(np.asarray(v))[:, None] for k, v in obs.items()}
    prevact = {k: jnp.asarray(v)[:, None] for k, v in action.items()}
    _, (enc_c, dyn_c, act) = j_pi_step(pi_params, enc_c, dyn_c, obs_in,
                                       prevact)

  result = dict(
      evaluator_run=os.path.abspath(args.evaluator_run),
      evaluator_expl_mode=str(ev_cfg.agent.expl.mode),
      policy_run=os.path.abspath(args.policy_run),
      policy_id=os.path.basename(args.policy_run.rstrip('/')),
      init_replay=os.path.abspath(args.init_replay),
      n_starts=B, burn_in=P, horizon=args.horizon, seed=args.seed,
      gamma=gamma,
      m_return_mean=float(ret.mean()), m_return_median=float(np.median(ret)),
      m_return_disc_mean=float(ret_disc.mean()),
      m_returns=[round(float(x), 4) for x in ret],
      survived_frac=float(alive.mean()))
  os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.',
              exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(result, f, indent=2)
  print(f'{result["policy_id"]}: M-return {ret.mean():.2f} '
        f'(median {np.median(ret):.2f}, survived {alive.mean():.2f}) '
        f'-> {args.output}')


def real_return(run_dir, window):
  path = os.path.join(run_dir, 'scores.jsonl')
  if not os.path.exists(path):
    return None
  scores = []
  with open(path) as f:
    for line in f:
      scores.append(json.loads(line)['episode/score'])
  if not scores:
    return None
  return float(np.mean(scores[-window:]))


def cmd_collate(args):
  from scipy import stats
  rows = []
  for path in sorted(globlib.glob(args.evals)):
    with open(path) as f:
      e = json.load(f)
    run_dir = os.path.join(args.runroot, e['policy_id']) if args.runroot \
        else e['policy_run']
    real = real_return(run_dir, args.real_window)
    if real is None:
      print(f'SKIP {e["policy_id"]}: no scores.jsonl')
      continue
    rows.append(dict(policy_id=e['policy_id'], m_return=e['m_return_mean'],
                     m_return_disc=e['m_return_disc_mean'], real=real))
  assert len(rows) >= 5, f'pool too small ({len(rows)})'
  m = np.asarray([r['m_return'] for r in rows])
  real = np.asarray([r['real'] for r in rows])
  rho = stats.spearmanr(m, real)
  # Top-k regret: real return foregone by trusting the evaluator's top-k.
  order = np.argsort(-m)
  best_real = real.max()
  topk = {}
  for k in (1, 3, 5):
    if k <= len(rows):
      topk[f'top{k}_regret'] = float(best_real - real[order[:k]].max())
      topk[f'top{k}_regret_frac'] = float(
          (best_real - real[order[:k]].max()) / (abs(best_real) + 1e-9))
  # Pairwise inversion rate.
  n = len(rows)
  inv = total = 0
  for i in range(n):
    for j in range(i + 1, n):
      if real[i] == real[j]:
        continue
      total += 1
      inv += int((m[i] - m[j]) * (real[i] - real[j]) < 0)
  out = dict(
      n_policies=n, spearman=float(rho.statistic), spearman_p=float(rho.pvalue),
      inversion_rate=float(inv / max(total, 1)), **topk,
      best_real=float(best_real),
      selected_real_top1=float(real[order[0]]),
      rows=rows, evals_glob=args.evals, real_window=args.real_window)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'{n} policies: spearman={rho.statistic:+.3f} '
        f'inversions={out["inversion_rate"]:.3f} '
        f'top1_regret={topk.get("top1_regret", float("nan")):.2f} '
        f'-> {args.output}')


def main():
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)

  s = sub.add_parser('score')
  s.add_argument('--evaluator_run', required=True,
                 help='Run dir providing the world model + reward head.')
  s.add_argument('--policy_run', required=True,
                 help='Run dir providing the policy (actor + its own WM '
                      'for filtering).')
  s.add_argument('--init_replay', required=True,
                 help='Replay dir supplying real warm-up prefixes.')
  s.add_argument('--n_starts', type=int, default=20)
  s.add_argument('--burn_in', type=int, default=16)
  s.add_argument('--horizon', type=int, default=64)
  s.add_argument('--seed', type=int, default=0)
  s.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  s.add_argument('--require_task', type=lambda x: x != 'False', default=True)
  s.add_argument('--scratch', default='')
  s.add_argument('--output', required=True)
  s.set_defaults(fn=cmd_score)

  c = sub.add_parser('collate')
  c.add_argument('--evals', required=True, help='Glob of score JSONs.')
  c.add_argument('--runroot', default='',
                 help='Resolve policy_id under this root for scores.jsonl '
                      '(default: the recorded absolute policy_run path).')
  c.add_argument('--real_window', type=int, default=10,
                 help='Real return = mean of the last N eval episodes.')
  c.add_argument('--output', required=True)
  c.set_defaults(fn=cmd_collate)

  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
