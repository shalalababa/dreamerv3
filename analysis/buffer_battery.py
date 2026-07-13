"""Buffer analysis battery (plan v4 §5 — no new training).

Computes, from materialized buffer chunk dirs (Axis-1/E3 sides, dose
levels) or raw replay dirs, the pre-read confound/interpretation panel:

  (i)   reward-bearing-frame fraction per buffer + overlap with occ_phys;
  (ii)  regime-entry counts, dwell-length distribution, per-episode
        occupancy (not just per-frame);
  (iii) source-mixture entropy / effective number of sources (from an
        adjacent manifest.json when present);
  (iv)  state-transition diversity inside/outside the regime
        (mean std of per-step state deltas per stratum);
  (v)   action distributions: per-dim mean/std + per-dim Wasserstein-1
        between labeled buffers, full and regime-conditional;
  (vi)  independent coverage re-estimates: kNN entropy at two k values,
        two fresh subsample seeds each (search-noise check);
  (vii) `ceiling` subcommand: threshold-crossing times from adapt
        scores.jsonl dirs (cup saturation analysis).

Runs cluster-side where chunks live; `selfcheck` is synthetic and local.

Usage:
  python -m analysis.buffer_battery report \
      --buffer lo=$RUNROOT/axis1_finger/q1/side0 \
      --buffer hi=$RUNROOT/axis1_finger/q1/side1 \
      --task dmc_finger_turn_hard --output battery_q1_finger.json

  python -m analysis.buffer_battery ceiling \
      --runroot $RUNROOT --glob 'adapt_ax1*_cup_*' \
      --thresholds 200 400 600 800 --output ceiling_cup.json
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

import numpy as np

from probing import probeset
from probing import regimes
from probing.gate0_compose import knn_entropy
from probing.value_sensitive import state_matrix


def load_buffer(directory, spec):
  """Chunk dir -> list of per-episode dicts with frames + regime mask."""
  cov_keys = sorted(spec['coverage_keys'])
  episodes = []
  for chain in probeset.chain_streams(directory):
    data = probeset.load_stream(chain)
    data = {k: np.asarray(v) for k, v in data.items()}
    q = spec['fn'](data)
    below = spec['direction'] == 'below'
    mask = (q < spec['threshold']) if below else (q > spec['threshold'])
    episodes.append(dict(
        n=len(q), mask=mask,
        occ=float(mask.mean()),
        reward=np.asarray(data.get('reward', np.zeros(len(q)))),
        action=np.asarray(data['action']) if 'action' in data else None,
        states=state_matrix(data, cov_keys).astype(np.float32)))
  return episodes


def dwell_stats(mask):
  """Number of regime entries and dwell lengths for one episode."""
  m = np.asarray(mask, bool)
  starts = np.flatnonzero(m & ~np.concatenate(([False], m[:-1])))
  lengths = []
  for s in starts:
    e = s
    while e < len(m) and m[e]:
      e += 1
    lengths.append(e - s)
  return len(starts), lengths


def wasserstein1(a, b):
  """1-Wasserstein distance between two 1-D samples."""
  a, b = np.sort(np.asarray(a, np.float64)), np.sort(np.asarray(b, np.float64))
  n = max(len(a), len(b))
  if len(a) == 0 or len(b) == 0:
    return None
  qs = (np.arange(n) + 0.5) / n
  return float(np.mean(np.abs(
      np.quantile(a, qs) - np.quantile(b, qs))))


def buffer_report(episodes, max_frames, knns, seeds):
  occs = np.array([e['occ'] for e in episodes])
  rew_frac = [float((e['reward'] > 0).mean()) for e in episodes]
  entries, dwells = [], []
  for e in episodes:
    k, ls = dwell_stats(e['mask'])
    entries.append(k)
    dwells.extend(ls)
  # Transition diversity: per-step state deltas, split by regime stratum.
  din, dout = [], []
  for e in episodes:
    d = np.diff(e['states'], axis=0)
    m = e['mask'][:-1]
    if m.any():
      din.append(d[m])
    if (~m).any():
      dout.append(d[~m])
  def div(chunks):
    if not chunks:
      return None
    x = np.concatenate(chunks, 0)
    return round(float(np.linalg.norm(x.std(0))), 6)
  # Independent coverage re-estimates.
  frames = np.concatenate([e['states'] for e in episodes], 0)
  cov = {}
  for k in knns:
    for s in seeds:
      rng = np.random.default_rng(s)
      idx = rng.choice(len(frames), size=min(max_frames, len(frames)),
                       replace=False)
      cov[f'knn{k}_seed{s}'] = round(float(knn_entropy(frames[idx], k, 1.0)), 6)
  actions = (np.concatenate([e['action'] for e in episodes], 0)
             if episodes[0]['action'] is not None else None)
  act_in = act_out = None
  if actions is not None:
    mask = np.concatenate([e['mask'] for e in episodes], 0)
    act_in, act_out = actions[mask], actions[~mask]
  return dict(
      n_episodes=len(episodes),
      n_frames=int(sum(e['n'] for e in episodes)),
      occ_frame=round(float(np.average(occs, weights=[e['n'] for e in episodes])), 6),
      occ_episode=dict(mean=round(float(occs.mean()), 6),
                       median=round(float(np.median(occs)), 6),
                       p90=round(float(np.quantile(occs, 0.9)), 6),
                       frac_zero=round(float((occs <= 1e-6).mean()), 4)),
      reward=dict(frame_frac_mean=round(float(np.mean(rew_frac)), 6),
                  episodes_with_reward=int(sum(r > 0 for r in rew_frac)),
                  occ_reward_corr=round(float(np.corrcoef(occs, rew_frac)[0, 1]), 4)
                  if np.std(rew_frac) > 0 and np.std(occs) > 0 else None),
      regime_entries=dict(mean=round(float(np.mean(entries)), 3),
                          frac_episodes_entering=round(float(np.mean(
                              [e > 0 for e in entries])), 4)),
      dwell=dict(n=len(dwells),
                 mean=round(float(np.mean(dwells)), 2) if dwells else None,
                 median=float(np.median(dwells)) if dwells else None,
                 max=int(max(dwells)) if dwells else None),
      transition_diversity=dict(in_regime=div(din), out_regime=div(dout)),
      coverage=cov,
      _actions=actions, _act_in=act_in, _act_out=act_out)


def cmd_report(args):
  spec = regimes.spec(args.task)
  out = dict(task=args.task, buffers={}, pairwise={})
  reports = {}
  for item in args.buffer:
    label, _, path = item.partition('=')
    episodes = load_buffer(path, spec)
    rep = buffer_report(episodes, args.max_frames, args.knn, args.cov_seeds)
    reports[label] = rep
    manifest = os.path.join(os.path.dirname(path.rstrip('/')), 'manifest.json')
    if os.path.exists(manifest):
      with open(manifest) as f:
        m = json.load(f)
      for side in m.get('sides', []) + m.get('levels', []):
        if side.get('directory', '').rstrip('/').endswith(
            path.rstrip('/').split('/')[-1]) and side.get('mixture'):
          src = collections.Counter()
          for k, v in side['mixture'].items():
            src[k.split('/')[0]] += v
          p = np.array(list(src.values()), np.float64)
          p /= p.sum()
          ent = float(-(p * np.log(p)).sum())
          rep_srcs = dict(n_sources=len(src), entropy=round(ent, 4),
                          effective_n=round(float(np.exp(ent)), 2))
          rep['sources'] = rep_srcs
    out['buffers'][label] = {k: v for k, v in rep.items()
                             if not k.startswith('_')}
    print(f"{label}: occ={rep['occ_frame']:.4f} "
          f"rew_frac={rep['reward']['frame_frac_mean']:.4f} "
          f"entries/ep={rep['regime_entries']['mean']:.2f}")
  labels = [i.partition('=')[0] for i in args.buffer]
  for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
      a, b = reports[labels[i]], reports[labels[j]]
      pw = {}
      if a['_actions'] is not None and b['_actions'] is not None:
        dims = a['_actions'].shape[1]
        pw['action_w1'] = [
            round(wasserstein1(a['_actions'][:, d], b['_actions'][:, d]), 5)
            for d in range(dims)]
        pw['action_w1_in_regime'] = [
            round(w, 5) if (w := wasserstein1(
                a['_act_in'][:, d], b['_act_in'][:, d])) is not None else None
            for d in range(dims)] if len(a['_act_in']) and len(b['_act_in']) \
            else None
      pw['d_reward_frac'] = round(
          abs(a['reward']['frame_frac_mean'] - b['reward']['frame_frac_mean']), 6)
      pw['d_occ_frame'] = round(abs(a['occ_frame'] - b['occ_frame']), 6)
      out['pairwise'][f'{labels[i]}|{labels[j]}'] = pw
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'-> {args.output}')


def cmd_ceiling(args):
  rows = []
  for d in sorted(globlib.glob(os.path.join(args.runroot, args.glob))):
    path = os.path.join(d, 'scores.jsonl')
    if not os.path.exists(path):
      continue
    steps, scores = [], []
    with open(path) as f:
      for line in f:
        rec = json.loads(line)
        steps.append(rec['step'])
        scores.append(rec['episode/score'])
    steps, scores = np.asarray(steps), np.asarray(scores)
    # Bursted evals: bin at 16K (project logging cadence), running mean.
    bins = (steps // 16000).astype(int)
    bmeans = {b: scores[bins == b].mean() for b in np.unique(bins)}
    bsteps = sorted(bmeans)
    crossing = {}
    for thr in args.thresholds:
      hit = next((b for b in bsteps if bmeans[b] >= thr), None)
      crossing[str(thr)] = int((hit + 1) * 16000) if hit is not None else None
    rows.append(dict(run_id=os.path.basename(d), crossing=crossing,
                     final10=round(float(np.mean(scores[-10:])), 2),
                     n_eps=len(scores)))
  out = dict(runroot=args.runroot, glob=args.glob,
             thresholds=args.thresholds, runs=rows)
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'{len(rows)} runs -> {args.output}')


def cmd_selfcheck(args):
  import tempfile
  from probing.build_controlled_replay import _synth_source
  with tempfile.TemporaryDirectory() as tmp:
    goal, p2e = f'{tmp}/goal/replay', f'{tmp}/p2e/replay'
    _synth_source(goal, 6000, 0.8, 0.3, seed=1)
    _synth_source(p2e, 6000, 0.05, 1.5, seed=2)
    ns = argparse.Namespace(
        task='dmc_cup_catch', buffer=[f'goal={goal}', f'p2e={p2e}'],
        max_frames=2000, knn=[6, 12], cov_seeds=[0, 1],
        output=f'{tmp}/battery.json')
    cmd_report(ns)
    with open(f'{tmp}/battery.json') as f:
      b = json.load(f)
    g, p = b['buffers']['goal'], b['buffers']['p2e']
    assert g['reward']['frame_frac_mean'] > p['reward']['frame_frac_mean']
    assert g['occ_frame'] > p['occ_frame']
    assert g['dwell']['n'] > 0
    assert all(v is not None for v in g['coverage'].values())
    assert len(b['pairwise']['goal|p2e']['action_w1']) == 2
    # ceiling on synthetic scores
    run = f'{tmp}/adapt_fake_cup_seed1_ckpt1'
    os.makedirs(run)
    with open(f'{run}/scores.jsonl', 'w') as f:
      for i in range(60):
        f.write(json.dumps({'step': 16001 + (i // 12) * 16000 + i % 12,
                            'episode/score': 100.0 * (i // 12)}) + '\n')
    ns = argparse.Namespace(runroot=tmp, glob='adapt_fake_*',
                            thresholds=[200, 400],
                            output=f'{tmp}/ceiling.json')
    cmd_ceiling(ns)
    with open(f'{tmp}/ceiling.json') as f:
      c = json.load(f)
    assert c['runs'][0]['crossing']['200'] is not None
    assert c['runs'][0]['crossing']['200'] < c['runs'][0]['crossing']['400']
    print('buffer_battery selfcheck PASS')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  sub = p.add_subparsers(dest='cmd', required=True)

  rp = sub.add_parser('report')
  rp.add_argument('--buffer', nargs='+', required=True,
                  help='label=chunk_dir ...')
  rp.add_argument('--task', required=True)
  rp.add_argument('--max_frames', type=int, default=3000)
  rp.add_argument('--knn', type=int, nargs='+', default=[6, 12])
  rp.add_argument('--cov_seeds', type=int, nargs='+', default=[0, 1])
  rp.add_argument('--output', required=True)
  rp.set_defaults(fn=cmd_report)

  ce = sub.add_parser('ceiling')
  ce.add_argument('--runroot', required=True)
  ce.add_argument('--glob', required=True)
  ce.add_argument('--thresholds', type=float, nargs='+',
                  default=[200, 400, 600, 800])
  ce.add_argument('--output', required=True)
  ce.set_defaults(fn=cmd_ceiling)

  sc = sub.add_parser('selfcheck')
  sc.set_defaults(fn=cmd_selfcheck)

  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
