"""E4: regime- and reward-stratified world-model prediction error.

Role (post-5a addendum E4; plan v4; sharpened after the 13-Jul W0/battery
read): a DESCRIPTIVE mechanism panel over already-fitted world models --
"capacity-allocation signature", never mediation, no confirmatory
contrasts.  For every WM checkpoint it scores per-frame prediction error
on one fixed, frozen probe set per domain and stratifies the error by
properties of the TARGET frame:

  * in-regime vs out-of-regime (mechanism-derived, probing/regimes.py);
  * reward-bearing vs non-reward-bearing (reward > 0);
  * their 2x2 cross, and per probe source.

The two stratifications coincide in finger (13-Jul battery: episode-level
occ<->reward correlation = +1.0) and dissociate in cup (-0.5/-0.7) -- which
is exactly why both are computed.  Signatures being looked for
(descriptive): occ of the fit buffer up => (err_out - err_in) up; Q1
high-occ side lower err_in / higher err_out; task-arm (reward-aware) fit
lower err on reward-dense frames than the apt-arm fit of the SAME buffer.

Error definitions (per frame, target-aligned):
  * horizon 0 ("post"): decoder NLL of the filtering posterior at t
    against obs at t (features.py `wm_post_nll` convention);
  * horizon k: decoder NLL of the k-step open-loop prior anchored at
    t - k (rolled with the logged actions, stochastic states sampled by
    the model itself, as in imagination) against obs at t.  The first k
    frames of each episode have no valid anchor and are NaN.
  * reward-head NLL per horizon, only for reward-aware fits (the head
    exists but is untrained under reward_free -- recorded as null there).

Probe sets are built from held-out sources (the Gate-0 pilot replays --
never the Phase-4 replays the Axis-1 buffers were composed from), whole
episodes of the modal length, deterministically sampled, then frozen with
a sha256 manifest exactly like probing/probeset.py.

Usage
-----
Build the per-domain probe set once (login node, CPU)::

    python -m probing.stratified_error build-probeset \
        --task dmc_finger_turn_hard \
        --source goal=$RUNROOT/pilot_goal_finger/replay \
                 random=$RUNROOT/pilot_random_finger/replay \
                 p2e=$RUNROOT/pilot_p2e_finger/replay \
        --n_episodes 20 --seed 0 \
        --output $RUNROOT/e4_probesets/finger_v1

Measure one WM run (GPU; ~minutes per checkpoint)::

    python -m probing.stratified_error measure \
        --probeset $RUNROOT/e4_probesets/finger_v1 \
        --run_logdir $RUNROOT/ax1wm_finger_fq1s1_seed3

Collate summaries across runs::

    python -m probing.stratified_error collate \
        --runroot $RUNROOT --glob 'ax1wm_finger_*' \
        --probeset_id finger_v1 --output e4_finger.csv

Validate the numpy logic (alignment, stratification, probe-set build)::

    python -m probing.stratified_error selfcheck
"""

import argparse
import collections
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
from probing import regimes

PROBESET_NPZ = 'probeset_e4.npz'


# ---------------------------------------------------------------------------
# Pure helpers (exercised by selfcheck; keep them jax-free).
# ---------------------------------------------------------------------------

def to_target_alignment(nll_anchor, k):
  """(B,T) anchor-aligned errors -> target-aligned, NaN where no anchor.

  nll_anchor[:, t] scores the prediction anchored at t of the frame t+k.
  Returns arr with arr[:, t+k] = nll_anchor[:, t]; columns 0..k-1 are NaN
  (no anchor k steps before them), and the last k anchor columns (which
  would target beyond the episode) are dropped.
  """
  if k == 0:
    return np.asarray(nll_anchor, np.float64)
  B, T = nll_anchor.shape
  out = np.full((B, T), np.nan)
  out[:, k:] = nll_anchor[:, :T - k]
  return out


def stratum_means(nll_target, masks):
  """NaN-aware per-stratum means and frame counts.

  nll_target: (B,T) target-aligned errors (NaN = invalid frame).
  masks: {name: (B,T) bool} strata over target frames.
  """
  valid = ~np.isnan(nll_target)
  out = {}
  for name, m in masks.items():
    sel = np.asarray(m, bool) & valid
    n = int(sel.sum())
    out[name] = dict(
        mean=float(nll_target[sel].mean()) if n else None, n=n)
  return out


def split_episodes(stream, keys):
  """{key: (N, ...)} stream -> list of per-episode dicts (is_first splits)."""
  first = np.asarray(stream['is_first'], bool)
  starts = np.flatnonzero(first)
  if len(starts) == 0 or starts[0] != 0:
    starts = np.concatenate([[0], starts]) if len(starts) else np.array([0])
  bounds = list(starts) + [len(first)]
  episodes = []
  for a, b in zip(bounds[:-1], bounds[1:]):
    if b > a:
      episodes.append({k: np.asarray(stream[k][a:b]) for k in keys})
  return episodes


def standard_masks(in_regime, rewarded):
  """The registered strata over target frames."""
  in_r, rew = np.asarray(in_regime, bool), np.asarray(rewarded, bool)
  return {
      'all': np.ones_like(in_r),
      'in_regime': in_r, 'out_regime': ~in_r,
      'rewarded': rew, 'unrewarded': ~rew,
      'in_rew': in_r & rew, 'in_norew': in_r & ~rew,
      'out_rew': ~in_r & rew, 'out_norew': ~in_r & ~rew,
  }


# ---------------------------------------------------------------------------
# build-probeset
# ---------------------------------------------------------------------------

def cmd_build(args):
  spec = regimes.spec(args.task)
  frozen_marker = os.path.join(args.output, 'FROZEN')
  if os.path.exists(frozen_marker):
    raise SystemExit(f'{args.output} is FROZEN; refusing to overwrite. '
                     f'Build a new version into a fresh directory.')
  rng = np.random.default_rng(args.seed)
  per_source, sources_meta = {}, {}
  for item in args.source:
    label, _, replay_dir = item.partition('=')
    episodes = []
    for chain in probeset_mod.chain_streams(replay_dir):
      stream = probeset_mod.load_stream(chain)
      keys = sorted(stream.keys())
      episodes.extend(split_episodes(stream, keys))
    lengths = [len(e['is_first']) for e in episodes]
    modal = collections.Counter(lengths).most_common(1)[0][0]
    complete = [e for e, n in zip(episodes, lengths) if n == modal]
    if len(complete) < args.n_episodes:
      raise SystemExit(f'{label}: only {len(complete)} complete episodes of '
                       f'modal length {modal} (< {args.n_episodes}).')
    pick = rng.choice(len(complete), size=args.n_episodes, replace=False)
    per_source[label] = ([complete[i] for i in sorted(pick)], modal)
    sources_meta[label] = dict(
        replay_dir=os.path.abspath(replay_dir), n_available=len(complete),
        modal_length=int(modal), picked=sorted(int(i) for i in pick))
  modal_lengths = {m for _, m in per_source.values()}
  assert len(modal_lengths) == 1, (
      f'Sources disagree on episode length: '
      f'{ {l: m for l, (_, m) in per_source.items()} }')
  T = modal_lengths.pop()

  keys = sorted(next(iter(per_source.values()))[0][0].keys())
  arrays, source_labels = {}, []
  for label, (eps, _) in per_source.items():
    source_labels.extend([label] * len(eps))
    for k in keys:
      arrays.setdefault(k, []).extend([e[k] for e in eps])
  arrays = {k: np.stack(v, 0) for k, v in arrays.items()}
  N = len(source_labels)

  flat = {k: v.reshape((N * T,) + v.shape[2:]) for k, v in arrays.items()}
  threshold = args.threshold if args.threshold is not None else (
      spec['threshold'])
  vals = regimes.regime_values(args.task, flat).reshape(N, T)
  below = spec['direction'] == 'below'
  in_r = (vals < threshold) if below else (vals > threshold)
  rewarded = np.asarray(arrays.get('reward', np.zeros((N, T))) > 0)

  os.makedirs(args.output, exist_ok=True)
  npz_path = os.path.join(args.output, PROBESET_NPZ)
  np.savez_compressed(
      npz_path, **arrays, regime_value=vals.astype(np.float32),
      in_regime=in_r, rewarded=rewarded,
      source=np.array(source_labels))
  digest = probeset_mod.sha256_file(npz_path)
  manifest = dict(
      probeset_id=os.path.basename(args.output.rstrip('/')),
      kind='e4_stratified_error', task=args.task,
      regime=dict(name=spec['name'], threshold=float(threshold),
                  direction=spec['direction']),
      sources=sources_meta, n_episodes=int(N), length=int(T),
      seed=args.seed, sha256=digest,
      git_commit=probeset_mod.git_commit(),
      occ_frame=round(float(in_r.mean()), 6),
      reward_frame_frac=round(float(rewarded.mean()), 6),
      occ_by_source={label: round(float(in_r[
          np.array(source_labels) == label].mean()), 6)
          for label in per_source},
      frozen=not args.no_freeze)
  with open(os.path.join(args.output, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
  if not args.no_freeze:
    with open(frozen_marker, 'w') as f:
      f.write(digest + '\n')
  print(f'E4 probe set {manifest["probeset_id"]}: {N} episodes x {T} steps, '
        f'occ={manifest["occ_frame"]:.4f} rew={manifest["reward_frame_frac"]:.4f}')
  print(f'  per-source occ: {manifest["occ_by_source"]}')
  print(f'  sha256={digest[:16]}...  '
        f'{"FROZEN" if not args.no_freeze else "UNFROZEN"}\n  -> {args.output}')


def load_e4(probeset_dir, require_frozen=True):
  npz_path = os.path.join(probeset_dir, PROBESET_NPZ)
  with open(os.path.join(probeset_dir, 'manifest.json')) as f:
    manifest = json.load(f)
  digest = probeset_mod.sha256_file(npz_path)
  if digest != manifest['sha256']:
    raise SystemExit(f'{probeset_dir}: {PROBESET_NPZ} sha256 mismatch.')
  frozen = os.path.exists(os.path.join(probeset_dir, 'FROZEN'))
  if require_frozen and not frozen:
    raise SystemExit(f'{probeset_dir} is not FROZEN; pass --allow_unfrozen '
                     f'only for iteration.')
  if frozen:
    with open(os.path.join(probeset_dir, 'FROZEN')) as f:
      if f.read().strip() != digest:
        raise SystemExit(f'{probeset_dir}: FROZEN marker does not match file.')
  arrays = {k: np.asarray(v) for k, v in np.load(npz_path).items()}
  return arrays, manifest


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def cmd_measure(args):
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from dreamerv3.main import make_agent
  from probing.collect import load_run_config, load_frozen_agent

  f32 = jnp.float32
  arrays, manifest = load_e4(args.probeset, not args.allow_unfrozen)
  N, T = arrays['is_first'].shape
  horizons = tuple(args.horizons)
  out_dir = args.output or os.path.join(
      args.run_logdir, f'e4_{manifest["probeset_id"]}')
  os.makedirs(out_dir, exist_ok=True)

  config = load_run_config(args.run_logdir, args.platform, out_dir, False)
  config = config.update({'jax': {
      'precompile': False, 'enable_policy': False, 'prealloc': False}})
  expl_mode = config.agent.expl.mode
  reward_aware = (expl_mode == 'task')
  agent = make_agent(config)
  ckpt = args.checkpoint or os.path.join(args.run_logdir, 'ckpt')
  load_frozen_agent(agent, ckpt)
  model = agent.model
  jax.config.update('jax_transfer_guard', 'allow')

  exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
  obs_keys = sorted(k for k, v in agent.obs_space.items()
                    if k not in exclude and len(v.shape) <= 1)
  act_keys = sorted(agent.act_space.keys())
  missing = [k for k in obs_keys + act_keys if k not in arrays]
  assert not missing, f'E4 probe set lacks keys {missing}.'
  print(f'{os.path.basename(args.run_logdir.rstrip("/"))}: expl.mode='
        f'{expl_mode} reward_aware={reward_aware} horizons={horizons} '
        f'probes {N}x{T}')

  def shift(v, k):
    return jnp.concatenate([v[:, k:], jnp.zeros_like(v[:, :k])], 1)

  def fn(obs, action_dict, reward, reset):
    B = reset.shape[0]
    enc_carry = model.enc.initial(B)
    dyn_carry = model.dyn.initial(B)
    enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)
    prevact = {k: jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], 1)
               for k, v in action_dict.items()}
    dyn_carry, _, post = model.dyn.observe(
        dyn_carry, tokens, prevact, reset, training=False)
    out = {}

    def decode_nll(feat, k):
      _, _, rec = model.dec(model.dec.initial(B), feat, reset, training=False)
      for key in obs_keys:
        out[f'h{k}/{key}'] = f32(rec[key].loss(shift(f32(obs[key]), k)))
      if reward_aware:
        inp = model.feat2tensor({'deter': feat['deter'],
                                 'stoch': feat['stoch']})
        out[f'h{k}/_rew'] = f32(model.rew(inp, 2).loss(shift(reward, k)))

    decode_nll({'deter': post['deter'], 'stoch': post['stoch']}, 0)
    for k in horizons:
      carry = {
          'deter': post['deter'].reshape((B * T,) + post['deter'].shape[2:]),
          'stoch': post['stoch'].reshape((B * T,) + post['stoch'].shape[2:]),
      }
      windows = {}
      for ak, av in action_dict.items():
        A = av.shape[-1]
        pad = jnp.concatenate([av, jnp.zeros((B, k, A), av.dtype)], 1)
        win = jnp.stack([pad[:, i:i + T] for i in range(k)], 2)
        windows[ak] = win.reshape((B * T, k, A))
      _, feat, _ = model.dyn.imagine(carry, windows, k, training=False)
      deter_k = feat['deter'][:, -1].reshape((B, T) + feat['deter'].shape[2:])
      stoch_k = feat['stoch'][:, -1].reshape((B, T) + feat['stoch'].shape[2:])
      decode_nll({'deter': deter_k, 'stoch': stoch_k}, k)
    return out

  pure = nj.pure(fn)
  jit = jax.jit(lambda p, o, a, r, re, s: pure(p, o, a, r, re, seed=s))
  params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), agent.params)

  reset_np = np.asarray(arrays['is_first'], bool)
  acc = {}
  for lo in range(0, N, args.ep_batch):
    hi = min(lo + args.ep_batch, N)
    obs = {k: jnp.asarray(arrays[k][lo:hi], np.float32) for k in obs_keys}
    action_dict = {k: jnp.asarray(arrays[k][lo:hi], np.float32)
                   for k in act_keys}
    reward = jnp.asarray(arrays['reward'][lo:hi], np.float32)
    reset = jnp.asarray(reset_np[lo:hi])
    seed = jnp.array([args.seed, lo + 1], np.uint32)
    _, out = jit(params, obs, action_dict, reward, reset, seed)
    for k, v in out.items():
      acc.setdefault(k, []).append(np.asarray(v, np.float64))
    print(f'  episodes {lo}-{hi - 1}')
  acc = {k: np.concatenate(v, 0) for k, v in acc.items()}

  masks = standard_masks(arrays['in_regime'], arrays['rewarded'])
  sources = np.asarray(arrays['source'])
  summary = dict(
      run_logdir=os.path.abspath(args.run_logdir),
      run_id=os.path.basename(args.run_logdir.rstrip('/')),
      checkpoint=os.path.abspath(ckpt), expl_mode=str(expl_mode),
      reward_aware=reward_aware,
      probeset_id=manifest['probeset_id'], probeset_sha256=manifest['sha256'],
      horizons=[0] + list(horizons), horizon_stats={})
  save = {}
  for k in [0] + list(horizons):
    total = sum(to_target_alignment(acc[f'h{k}/{key}'], k)
                for key in obs_keys)
    save[f'nll_h{k}'] = total.astype(np.float32)
    strata = stratum_means(total, masks)
    stats = dict(
        strata=strata,
        err_diff_out_minus_in=(
            round(strata['out_regime']['mean'] - strata['in_regime']['mean'], 6)
            if strata['in_regime']['n'] and strata['out_regime']['n'] else None),
        per_key={key: stratum_means(
            to_target_alignment(acc[f'h{k}/{key}'], k),
            {s: masks[s] for s in ('in_regime', 'out_regime',
                                   'rewarded', 'unrewarded')})
            for key in obs_keys},
        per_source={label: stratum_means(
            total[sources == label],
            {s: masks[s][sources == label]
             for s in ('all', 'in_regime', 'out_regime')})
            for label in sorted(set(sources.tolist()))})
    if reward_aware:
      rew_nll = to_target_alignment(acc[f'h{k}/_rew'], k)
      save[f'rew_nll_h{k}'] = rew_nll.astype(np.float32)
      stats['reward_head'] = stratum_means(rew_nll, masks)
    summary['horizon_stats'][str(k)] = stats

  np.savez_compressed(os.path.join(out_dir, 'errors.npz'), **save)
  with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
  for k in [0] + list(horizons):
    s = summary['horizon_stats'][str(k)]['strata']
    print(f'  h{k}: all={s["all"]["mean"]:.4f} in={s["in_regime"]["mean"]:.4f} '
          f'out={s["out_regime"]["mean"]:.4f} '
          f'rew={s["rewarded"]["mean"]:.4f} norew={s["unrewarded"]["mean"]:.4f}')
  print(f'-> {out_dir}')


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------

def cmd_collate(args):
  rows = []
  strata = ('all', 'in_regime', 'out_regime', 'rewarded', 'unrewarded',
            'in_rew', 'in_norew', 'out_rew', 'out_norew')
  for d in sorted(globlib.glob(os.path.join(args.runroot, args.glob))):
    path = os.path.join(d, f'e4_{args.probeset_id}', 'summary.json')
    if not os.path.exists(path):
      continue
    with open(path) as f:
      s = json.load(f)
    for k, stats in s['horizon_stats'].items():
      row = dict(run_id=s['run_id'], expl_mode=s['expl_mode'],
                 reward_aware=int(s['reward_aware']), horizon=int(k),
                 err_diff=stats['err_diff_out_minus_in'])
      for name in strata:
        st = stats['strata'][name]
        row[f'nll_{name}'] = st['mean']
        row[f'n_{name}'] = st['n']
      if 'reward_head' in stats:
        row['rew_nll_all'] = stats['reward_head']['all']['mean']
        row['rew_nll_in'] = stats['reward_head']['in_regime']['mean']
        row['rew_nll_out'] = stats['reward_head']['out_regime']['mean']
      rows.append(row)
  if not rows:
    raise SystemExit(f'No e4_{args.probeset_id}/summary.json under '
                     f'{args.runroot}/{args.glob}')
  cols = sorted({c for r in rows for c in r}, key=lambda c: (c != 'run_id', c))
  import csv
  with open(args.output, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
  print(f'{len(rows)} rows ({len({r["run_id"] for r in rows})} runs) '
        f'-> {args.output}')


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------

def cmd_selfcheck(args):
  import tempfile
  from probing.build_controlled_replay import _synth_source

  # 1. Target alignment: anchor value == anchor index.
  B, T, k = 2, 10, 3
  anchor = np.tile(np.arange(T, dtype=np.float64), (B, 1))
  tgt = to_target_alignment(anchor, k)
  assert np.isnan(tgt[:, :k]).all()
  assert (tgt[:, k:] == anchor[:, :T - k]).all()
  assert (to_target_alignment(anchor, 0) == anchor).all()

  # 2. Stratification recovers an injected pattern exactly.
  rng = np.random.default_rng(0)
  in_r = rng.random((B, T)) < 0.3
  rew = rng.random((B, T)) < 0.2
  nll = 1.0 + 2.0 * in_r + 4.0 * rew
  masks = standard_masks(in_r, rew)
  means = stratum_means(nll, masks)
  assert abs(means['in_norew']['mean'] - 3.0) < 1e-12
  assert abs(means['out_rew']['mean'] - 5.0) < 1e-12
  assert abs(means['out_norew']['mean'] - 1.0) < 1e-12
  assert means['all']['n'] == B * T
  # NaN frames are excluded from both mean and count.
  nll2 = nll.copy()
  nll2[:, 0] = np.nan
  assert stratum_means(nll2, masks)['all']['n'] == B * (T - 1)

  # 3. build-probeset end-to-end on synthetic cup-like sources.
  with tempfile.TemporaryDirectory() as tmp:
    goal, p2e = f'{tmp}/goal/replay', f'{tmp}/p2e/replay'
    _synth_source(goal, 6000, 0.8, 0.3, seed=1)
    _synth_source(p2e, 6000, 0.05, 1.5, seed=2)
    out = f'{tmp}/e4_cup_v0'
    ns = argparse.Namespace(
        task='dmc_cup_catch', source=[f'goal={goal}', f'p2e={p2e}'],
        n_episodes=10, seed=0, threshold=None, output=out, no_freeze=False)
    cmd_build(ns)
    arrays, manifest = load_e4(out)
    assert arrays['is_first'].shape[0] == 20
    assert manifest['occ_by_source']['goal'] > manifest['occ_by_source']['p2e']
    assert arrays['in_regime'].shape == arrays['reward'].shape
    assert set(arrays['source'].tolist()) == {'goal', 'p2e'}
    # Determinism: same seed -> same sha (byte-stable content).
    ns2 = argparse.Namespace(**{**vars(ns), 'output': f'{tmp}/e4_cup_v0b'})
    cmd_build(ns2)
    _, manifest2 = load_e4(f'{tmp}/e4_cup_v0b')
    assert [s['picked'] for s in manifest['sources'].values()] == \
           [s['picked'] for s in manifest2['sources'].values()]
    # Frozen sets refuse overwrite.
    try:
      cmd_build(ns)
      raise AssertionError('frozen probe set was overwritten')
    except SystemExit:
      pass
  print('stratified_error selfcheck PASS')
  print('(measure path is jax; validate e2e on a debug run: '
        'python -m probing.stratified_error measure --probeset <set> '
        '--run_logdir <debug_run> --platform cpu --ep_batch 2)')


def main():
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)

  b = sub.add_parser('build-probeset')
  b.add_argument('--task', required=True)
  b.add_argument('--source', nargs='+', required=True,
                 help='label=replay_dir (held-out sources; use the Gate-0 '
                      'pilot replays, never Phase-4 replays).')
  b.add_argument('--n_episodes', type=int, default=20,
                 help='Episodes per source.')
  b.add_argument('--seed', type=int, default=0)
  b.add_argument('--threshold', type=float, default=None,
                 help='Regime threshold override (default: regimes.py spec).')
  b.add_argument('--no_freeze', action='store_true')
  b.add_argument('--output', required=True)
  b.set_defaults(fn=cmd_build)

  m = sub.add_parser('measure')
  m.add_argument('--probeset', required=True)
  m.add_argument('--run_logdir', required=True)
  m.add_argument('--checkpoint', default='',
                 help='Default: <run_logdir>/ckpt.')
  m.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 20])
  m.add_argument('--ep_batch', type=int, default=4)
  m.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  m.add_argument('--allow_unfrozen', action='store_true')
  m.add_argument('--seed', type=int, default=0)
  m.add_argument('--output', default='',
                 help='Default: <run_logdir>/e4_<probeset_id>.')
  m.set_defaults(fn=cmd_measure)

  c = sub.add_parser('collate')
  c.add_argument('--runroot', required=True)
  c.add_argument('--glob', required=True)
  c.add_argument('--probeset_id', required=True)
  c.add_argument('--output', required=True)
  c.set_defaults(fn=cmd_collate)

  sc = sub.add_parser('selfcheck')
  sc.set_defaults(fn=cmd_selfcheck)

  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
