"""Build confound-controlled static replay buffers (runbook v2 Phase 6, item 1).

Axis-1 interventions need pairs of buffers that are (Q1) coverage-matched /
occupancy-different and (Q2) coverage-different / occupancy-matched. This tool
composes such buffers from Phase-4 pretraining replays (+ pilots) at
**whole-episode granularity** — never resampling frames — and writes valid
chunk directories that `probing/offline_fit.py` can ingest.

Temporal-structure contract (frozen design choice, recorded in the manifest):
each selected episode is written as ONE fresh chunk with successor UUID 0.
Within-episode contiguity is exact; sampled training windows never span
episode boundaries (the online replay's cross-episode windows are dropped on
BOTH sides of every pair, so the contrast stays matched). Episodes are
restricted to the modal episode length, so the episode-length and
segment-length distributions are identical across sides by construction.

Subcommands
-----------
index   Scan sources, find episodes + per-episode occupancy:
  python -m probing.build_controlled_replay index \
      --replay p2e_s1=$RUNROOT/pretrain_p2e_cup_seed1/replay ... \
      --task dmc_cup_catch --output $RUNROOT/axis1_cup/episodes.json

search  Episode-level matched-pair search (gate0_compose method):
  python -m probing.build_controlled_replay search \
      --index $RUNROOT/axis1_cup/episodes.json \
      --ref_replay $RUNROOT/pilot_goal_cup_seed1/replay \
      --n_episodes 200 --output $RUNROOT/axis1_cup/pairs.json

build   Materialize one pair (both sides) as chunk dirs + manifests:
  python -m probing.build_controlled_replay build \
      --index .../episodes.json --pairs .../pairs.json --which q1 \
      --output_root $RUNROOT/axis1_cup/q1

search-dose  (addendum E3) L pairwise coverage-matched buffers whose
  occupancies sit at the even quantiles of the Q1-feasible occupancy span:
  python -m probing.build_controlled_replay search-dose \
      --index .../episodes.json --ref_replay ... --levels 4 \
      --output $RUNROOT/axis1_cup/dose.json

search-rpair  (addendum E3) Q1-equivalent pair whose HIGH-occupancy side
  excludes named sources, docc as close as feasible to --target_docc;
  output is pairs.json-shaped (key 'q1', side0=low occ) so `build --which
  q1` materializes it unchanged:
  python -m probing.build_controlled_replay search-rpair \
      --index .../episodes.json --ref_replay ... \
      --exclude_sources random4 --target_docc 0.161 \
      --output $RUNROOT/axis1_cup/rpairs_r1.json

build-dose  Materialize all dose levels + one manifest:
  python -m probing.build_controlled_replay build-dose \
      --index .../episodes.json --dose .../dose.json \
      --output_root $RUNROOT/axis1_cup/dose

selfcheck     End-to-end on synthetic sources, incl. Replay ingestion.
selfcheck-e3  Same for search-dose/search-rpair/build-dose (3 sources).
"""

import argparse
import collections
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

import elements
from probing import probeset
from probing import regimes
from probing.gate0_compose import knn_entropy
from probing.value_sensitive import compute_scaler, state_matrix

ZERO_UUID = '0' * 22


# --------------------------------------------------------------------------
# Stream / episode plumbing.
# --------------------------------------------------------------------------

def stream_meta(replay_dir):
  """[{files: [...], lengths: [...]}] per successor-chained stream."""
  streams = []
  for chain in probeset.chain_streams(replay_dir):
    lengths = [int(pathlib.Path(f).stem.split('-')[3]) for f in chain]
    streams.append(dict(files=[str(f) for f in chain], lengths=lengths))
  return streams


def load_span(stream, start, end, keys=None):
  """Load frames [start, end) of one stream, touching only needed chunks."""
  bounds = np.cumsum([0] + stream['lengths'])
  first = int(np.searchsorted(bounds, start, side='right') - 1)
  last = int(np.searchsorted(bounds, end - 1, side='right') - 1)
  parts = collections.defaultdict(list)
  for ci in range(first, last + 1):
    with np.load(stream['files'][ci]) as data:
      use = keys if keys is not None else list(data.keys())
      lo = max(start - bounds[ci], 0)
      hi = min(end - bounds[ci], stream['lengths'][ci])
      for k in use:
        parts[k].append(data[k][lo:hi])
  return {k: np.concatenate(v, 0) for k, v in parts.items()}


def find_episodes(stream, spec):
  """[(start, end, occ)] episode spans + occupancy for one stream."""
  need = sorted(set(spec['needs']) | {'is_first'})
  total = int(sum(stream['lengths']))
  data = load_span(stream, 0, total, keys=need)
  is_first = np.asarray(data['is_first'], bool).reshape(-1)
  starts = list(np.flatnonzero(is_first))
  if not starts or starts[0] != 0:
    starts = [0] + starts
  spans = list(zip(starts, starts[1:] + [total]))
  quantity = spec['fn'](data)
  below = spec['direction'] == 'below'
  mask = (quantity < spec['threshold']) if below \
      else (quantity > spec['threshold'])
  return [(int(a), int(b), float(mask[a:b].mean())) for a, b in spans]


def occ_bin(occ):
  if occ <= 1e-6:
    return 'zero'
  return 'high' if occ >= 0.5 else 'mid'


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

def parse_replays(pairs):
  out = {}
  for item in pairs:
    label, _, path = item.partition('=')
    if not path:
      raise SystemExit(f'--replay wants label=dir, got {item!r}')
    out[label] = path
  return out


def cmd_index(args):
  spec = regimes.spec(args.task)
  sources = parse_replays(args.replay)
  index = dict(task=args.task, regime=spec['name'],
               threshold=spec['threshold'], sources={})
  for label, replay_dir in sources.items():
    streams = stream_meta(replay_dir)
    episodes = []
    for si, stream in enumerate(streams):
      for start, end, occ in find_episodes(stream, spec):
        episodes.append(dict(stream=si, start=start, end=end,
                             length=end - start, occ=round(occ, 6)))
    index['sources'][label] = dict(replay=replay_dir, streams=streams,
                                   episodes=episodes)
    print(f'{label}: {len(episodes)} episodes '
          f'({sum(e["length"] for e in episodes)} frames)')

  # Restrict to the modal episode length (identical length distribution on
  # both sides of every pair, by construction).
  lengths = [e['length'] for src in index['sources'].values()
             for e in src['episodes']]
  modal = collections.Counter(lengths).most_common(1)[0][0]
  n_all = len(lengths)
  for label, src in index['sources'].items():
    src['episodes'] = [e for e in src['episodes'] if e['length'] == modal]
  n_kept = sum(len(s['episodes']) for s in index['sources'].values())
  index['ep_len'] = int(modal)
  print(f'modal episode length {modal}: kept {n_kept}/{n_all} episodes')

  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(index, f)
  print(f'-> {args.output}')


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def allocate(weights, caps, total):
  """Proportional integer allocation capped by stratum size; None if short."""
  weights = np.asarray(weights, np.float64)
  if weights.sum() <= 0:
    return None
  want = weights / weights.sum() * total
  alloc = np.minimum(np.floor(want).astype(int), caps)
  while alloc.sum() < total:
    room = (alloc < caps) & (weights > 0)
    if not room.any():
      return None
    frac = np.where(room, want - alloc, -np.inf)
    alloc[int(np.argmax(frac))] += 1
  return alloc


def _overlap(a, b):
  inter = len(np.intersect1d(a['members'], b['members']))
  return inter / min(len(a['members']), len(b['members']))


def _source_l1(a, b, n_episodes):
  keys = set(a['mixture']) | set(b['mixture'])
  return sum(abs(a['mixture'].get(k, 0) - b['mixture'].get(k, 0))
             for k in keys) / (2 * n_episodes)


def _candidate_pool(args, index):
  """Candidate buffers + matching thresholds (shared by all search modes).

  The RNG call sequence is identical to the original `search`
  implementation, so for the same --index/--seed/knobs the pool reproduces
  the frozen 6-Jul searches bit-for-bit; search-dose / search-rpair then
  select from the same pool the original Q1/Q2 pairs came from.
  """
  task = index['task']
  spec = regimes.spec(task)
  cov_keys = sorted(spec['coverage_keys'])
  _, ref_mean, ref_std = compute_scaler(args.ref_replay, task)
  rng = np.random.default_rng(args.seed)

  # Per-episode coverage-key frames (standardized), gathered once.
  ep_frames, strata = [], collections.defaultdict(list)
  raw_cov_sources = {}
  for label, src in index['sources'].items():
    streams = src['streams']
    label_rows = []
    for e in src['episodes']:
      data = load_span(streams[e['stream']], e['start'], e['end'],
                       keys=cov_keys)
      x = ((state_matrix(data, cov_keys) - ref_mean) / ref_std).astype(
          np.float32)
      eid = len(ep_frames)
      ep_frames.append(x)
      strata[(label, occ_bin(e['occ']))].append(eid)
      label_rows.append((eid, e['occ']))
    # Raw source coverage at this estimator size -> cov_tol denominator.
    pool = np.concatenate([ep_frames[i] for i, _ in label_rows], 0)
    idx = rng.choice(len(pool), size=min(args.max_frames, len(pool)),
                     replace=False)
    raw_cov_sources[label] = knn_entropy(pool[idx], args.knn, args.logc)
  ep_occ = np.zeros(len(ep_frames))
  eid = 0
  for label, src in index['sources'].items():
    for e in src['episodes']:
      ep_occ[eid] = e['occ']
      eid += 1

  names = sorted(strata)
  caps = np.asarray([len(strata[s]) for s in names])
  cov_vals = list(raw_cov_sources.values())
  cov_range = max(cov_vals) - min(cov_vals)
  cov_tol = args.cov_match_frac * cov_range
  print(f'strata: { {f"{a}/{b}": int(c) for (a, b), c in zip(names, caps)} }')
  print(f'source coverages: {raw_cov_sources} -> cov_tol={cov_tol:.4f}')

  def measure(members):
    occ = float(ep_occ[members].mean())
    boots = [ep_occ[rng.choice(members, size=len(members))].mean()
             for _ in range(args.boot)]
    frames = np.concatenate([ep_frames[i] for i in members], 0)
    idx = rng.choice(len(frames), size=min(args.max_frames, len(frames)),
                     replace=False)
    cov = knn_entropy(frames[idx], args.knn, args.logc)
    return cov, occ, float(np.std(boots))

  # Candidates: corners (pure stratum / pure source) + sparse mixtures.
  weight_vecs = []
  for i in range(len(names)):
    w = np.zeros(len(names))
    w[i] = 1.0
    weight_vecs.append(w)
  labels = sorted(index['sources'])
  for label in labels:
    w = np.asarray([1.0 if s[0] == label else 0.0 for s in names])
    weight_vecs.append(w)
  while len(weight_vecs) < args.n_candidates:
    weight_vecs.append(rng.dirichlet(np.full(len(names), args.dirichlet)))

  candidates = []
  for w in weight_vecs:
    alloc = allocate(w, caps, args.n_episodes)
    if alloc is None:
      continue
    members = []
    for s, k in zip(names, alloc):
      if k:
        members.extend(rng.choice(strata[s], size=k, replace=False))
    members = np.asarray(sorted(members))
    cov, occ, noise = measure(members)
    mixture = {f'{a}/{b}': int(k) for (a, b), k in zip(names, alloc) if k}
    candidates.append(dict(members=members, cov=cov, occ=occ, noise=noise,
                           mixture=mixture))
  print(f'{len(candidates)} feasible candidates')

  occ_sep_min = args.occ_sep_mult * float(
      np.median([c['noise'] for c in candidates]))
  return dict(task=task, candidates=candidates, cov_tol=cov_tol,
              occ_sep_min=occ_sep_min, source_coverages=raw_cov_sources)


def cmd_search(args):
  with open(args.index) as f:
    index = json.load(f)
  pool = _candidate_pool(args, index)
  task, candidates = pool['task'], pool['candidates']
  cov_tol, occ_sep_min = pool['cov_tol'], pool['occ_sep_min']
  raw_cov_sources = pool['source_coverages']

  best = dict(q1=None, q2=None)
  n_pairs = dict(q1=0, q2=0)
  for i in range(len(candidates)):
    for j in range(i + 1, len(candidates)):
      a, b = candidates[i], candidates[j]
      dcov, docc = abs(a['cov'] - b['cov']), abs(a['occ'] - b['occ'])
      if _overlap(a, b) > args.max_overlap:
        continue
      if dcov <= cov_tol and docc >= occ_sep_min:
        n_pairs['q1'] += 1
        if best['q1'] is None or docc > best['q1']['docc']:
          best['q1'] = dict(i=i, j=j, dcov=dcov, docc=docc)
      if dcov > cov_tol and docc < occ_sep_min:
        n_pairs['q2'] += 1
        if best['q2'] is None or dcov > best['q2']['dcov']:
          best['q2'] = dict(i=i, j=j, dcov=dcov, docc=docc)

  out = dict(task=task, index=args.index, ref_replay=args.ref_replay,
             n_episodes=args.n_episodes, ep_len=index['ep_len'],
             criteria=dict(cov_match_frac=args.cov_match_frac,
                           occ_sep_mult=args.occ_sep_mult,
                           max_overlap=args.max_overlap, knn=args.knn,
                           logc=args.logc, max_frames=args.max_frames,
                           dirichlet=args.dirichlet, boot=args.boot,
                           seed=args.seed),
             source_coverages=raw_cov_sources, cov_tol=cov_tol,
             occ_sep_min=occ_sep_min, n_pairs=n_pairs, pairs={})
  for q in ('q1', 'q2'):
    if best[q] is None:
      out['pairs'][q] = None
      continue
    a, b = candidates[best[q]['i']], candidates[best[q]['j']]
    out['pairs'][q] = dict(
        dcov=round(best[q]['dcov'], 6), docc=round(best[q]['docc'], 6),
        overlap=round(_overlap(a, b), 4),
        source_l1=round(_source_l1(a, b, args.n_episodes), 4),
        sides=[dict(cov=round(c['cov'], 6), occ=round(c['occ'], 6),
                    occ_noise=round(c['noise'], 6), mixture=c['mixture'],
                    members=[int(m) for m in c['members']])
               for c in (a, b)])
    print(f"{q}: dcov={best[q]['dcov']:.4f} docc={best[q]['docc']:.4f} "
          f"({n_pairs[q]} valid pairs)")
  decision = 'OK' if (out['pairs']['q1'] and out['pairs']['q2']) else 'SHORT'
  out['decision'] = decision
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  print(f'{decision} -> {args.output}')


# --------------------------------------------------------------------------
# search-dose (addendum E3: occupancy dose-response levels)
# --------------------------------------------------------------------------

def cmd_search_dose(args):
  with open(args.index) as f:
    index = json.load(f)
  pool = _candidate_pool(args, index)
  candidates = pool['candidates']
  cov_tol, occ_sep_min = pool['cov_tol'], pool['occ_sep_min']

  # Q1-feasible candidates: members of at least one valid Q1 pair. The dose
  # span is the occupancy range reachable under coverage matching.
  feas = set()
  for i in range(len(candidates)):
    for j in range(i + 1, len(candidates)):
      a, b = candidates[i], candidates[j]
      if _overlap(a, b) > args.max_overlap:
        continue
      if abs(a['cov'] - b['cov']) <= cov_tol and \
          abs(a['occ'] - b['occ']) >= occ_sep_min:
        feas.update((i, j))
  out = dict(kind='dose', task=pool['task'], index=args.index,
             ref_replay=args.ref_replay, n_episodes=args.n_episodes,
             ep_len=index['ep_len'],
             criteria=dict(cov_match_frac=args.cov_match_frac,
                           occ_sep_mult=args.occ_sep_mult,
                           max_overlap=args.max_overlap, knn=args.knn,
                           logc=args.logc, max_frames=args.max_frames,
                           dirichlet=args.dirichlet, boot=args.boot,
                           seed=args.seed,
                           n_candidates=args.n_candidates,
                           levels=args.levels, beam=args.beam,
                           objective='min total |occ - target|, all pairs '
                                     'cov-matched and overlap-capped'),
             source_coverages=pool['source_coverages'], cov_tol=cov_tol,
             occ_sep_min=occ_sep_min)
  if not feas:
    out.update(decision='SHORT', levels=None,
               reason='no Q1-feasible candidates')
  else:
    lo = min(candidates[i]['occ'] for i in feas)
    hi = max(candidates[i]['occ'] for i in feas)
    L = args.levels
    targets = [lo + (hi - lo) * l / (L - 1) for l in range(L)]
    ranked = [sorted(feas, key=lambda i: abs(candidates[i]['occ'] - t))
              [:args.beam] for t in targets]

    def compatible(i, picks):
      for j in picks:
        if i == j:
          return False
        if abs(candidates[i]['cov'] - candidates[j]['cov']) > cov_tol:
          return False
        if _overlap(candidates[i], candidates[j]) > args.max_overlap:
          return False
      return True

    best = dict(dev=np.inf, picks=None)

    def dfs(level, picks, dev):
      if dev >= best['dev']:
        return
      if level == L:
        best.update(dev=dev, picks=list(picks))
        return
      for i in ranked[level]:
        if not compatible(i, picks):
          continue
        dfs(level + 1, picks + [i],
            dev + abs(candidates[i]['occ'] - targets[level]))

    dfs(0, [], 0.0)
    if best['picks'] is None:
      out.update(decision='SHORT', levels=None, occ_span=[lo, hi],
                 targets=targets,
                 reason='no pairwise-compatible level assignment in beam')
    else:
      picks = best['picks']
      dcovs = [abs(candidates[i]['cov'] - candidates[j]['cov'])
               for x, i in enumerate(picks) for j in picks[x + 1:]]
      overs = [_overlap(candidates[i], candidates[j])
               for x, i in enumerate(picks) for j in picks[x + 1:]]
      out.update(
          decision='OK', occ_span=[round(lo, 6), round(hi, 6)],
          targets=[round(t, 6) for t in targets],
          total_abs_dev=round(float(best['dev']), 6),
          max_pairwise_dcov=round(max(dcovs), 6),
          max_pairwise_overlap=round(max(overs), 4),
          levels=[dict(level=l, target=round(targets[l], 6),
                       cov=round(candidates[i]['cov'], 6),
                       occ=round(candidates[i]['occ'], 6),
                       occ_noise=round(candidates[i]['noise'], 6),
                       mixture=candidates[i]['mixture'],
                       members=[int(m) for m in candidates[i]['members']])
                  for l, i in enumerate(picks)])
      print('levels: ' + ' '.join(
          f"d{l}:occ={lv['occ']:.4f}(t={lv['target']:.4f})"
          for l, lv in enumerate(out['levels'])))
      print(f"max pairwise dcov={out['max_pairwise_dcov']:.4f} "
            f"(tol {cov_tol:.4f}), overlap={out['max_pairwise_overlap']}")
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  print(f"{out['decision']} -> {args.output}")


# --------------------------------------------------------------------------
# search-rpair (addendum E3: composition-robustness Q1-equivalent pairs)
# --------------------------------------------------------------------------

def cmd_search_rpair(args):
  with open(args.index) as f:
    index = json.load(f)
  pool = _candidate_pool(args, index)
  candidates = pool['candidates']
  cov_tol, occ_sep_min = pool['cov_tol'], pool['occ_sep_min']
  excl = set(args.exclude_sources)

  def uses_excluded(c):
    return any(k.split('/')[0] in excl for k in c['mixture'])

  # Valid Q1 pairs whose HIGH-occupancy side avoids the excluded sources;
  # selection is lexicographic (|docc - target_docc|, then dcov), i.e. as
  # close as feasible to the original pair's occupancy dose, breaking ties
  # toward tighter coverage match.
  best, best_key, n_valid = None, None, 0
  for i in range(len(candidates)):
    for j in range(i + 1, len(candidates)):
      a, b = candidates[i], candidates[j]
      lo, hi = (a, b) if a['occ'] <= b['occ'] else (b, a)
      dcov, docc = abs(a['cov'] - b['cov']), hi['occ'] - lo['occ']
      if _overlap(a, b) > args.max_overlap:
        continue
      if dcov > cov_tol or docc < occ_sep_min:
        continue
      if uses_excluded(hi):
        continue
      n_valid += 1
      key = (abs(docc - args.target_docc), dcov)
      if best_key is None or key < best_key:
        best, best_key = (lo, hi), key

  out = dict(kind='rpair', task=pool['task'], index=args.index,
             ref_replay=args.ref_replay, n_episodes=args.n_episodes,
             ep_len=index['ep_len'],
             criteria=dict(cov_match_frac=args.cov_match_frac,
                           occ_sep_mult=args.occ_sep_mult,
                           max_overlap=args.max_overlap, knn=args.knn,
                           logc=args.logc, max_frames=args.max_frames,
                           dirichlet=args.dirichlet, boot=args.boot,
                           seed=args.seed,
                           n_candidates=args.n_candidates,
                           exclude_sources=sorted(excl),
                           target_docc=args.target_docc,
                           objective='lexicographic (|docc-target|, dcov); '
                                     'exclusion applies to high-occ side'),
             source_coverages=pool['source_coverages'], cov_tol=cov_tol,
             occ_sep_min=occ_sep_min, n_pairs=dict(q1=n_valid), pairs={})
  if best is None:
    out['pairs']['q1'] = None
    out['decision'] = 'SHORT'
  else:
    lo, hi = best
    # side0 = low occupancy, side1 = high occupancy (Axis-1 Q1 convention).
    out['pairs']['q1'] = dict(
        dcov=round(abs(lo['cov'] - hi['cov']), 6),
        docc=round(hi['occ'] - lo['occ'], 6),
        overlap=round(_overlap(lo, hi), 4),
        source_l1=round(_source_l1(lo, hi, args.n_episodes), 4),
        sides=[dict(cov=round(c['cov'], 6), occ=round(c['occ'], 6),
                    occ_noise=round(c['noise'], 6), mixture=c['mixture'],
                    members=[int(m) for m in c['members']])
               for c in (lo, hi)])
    out['decision'] = 'OK'
    print(f"rpair: docc={out['pairs']['q1']['docc']:.4f} "
          f"(target {args.target_docc:.4f}) "
          f"dcov={out['pairs']['q1']['dcov']:.4f} ({n_valid} valid pairs; "
          f"excluded {sorted(excl)})")
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  print(f"{out['decision']} -> {args.output}")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def episode_by_eid(index):
  """eid -> (label, episode dict), matching cmd_search enumeration order."""
  table = {}
  eid = 0
  for label, src in index['sources'].items():
    for e in src['episodes']:
      table[eid] = (label, e)
      eid += 1
  return table


def write_episode_chunk(out_dir, frames):
  n = len(next(iter(frames.values())))
  uuid = elements.UUID()
  # Rewrite stepid to encode the NEW chunk uuid + index. Stale stepids would
  # make Replay.update() silently no-op (KeyError pass), so the offline WM
  # would train on the SOURCE agent's stored latent contexts forever instead
  # of refreshing its own (replay_context path in dreamerv3/agent.py).
  frames = dict(frames)
  frames['stepid'] = np.stack([
      np.frombuffer(bytes(uuid) + i.to_bytes(4, 'big'), np.uint8)
      for i in range(n)])
  name = f'{elements.timestamp(millis=True)}-{uuid}-{ZERO_UUID}-{n}.npz'
  path = os.path.join(out_dir, name)
  np.savez_compressed(path, **frames)
  return name


def action_stats(actions):
  a = np.asarray(actions, np.float64).reshape(len(actions), -1)
  return dict(mean=[round(float(x), 5) for x in a.mean(0)],
              std=[round(float(x), 5) for x in a.std(0)])


def _write_buffer(index, table, selection, out_dir):
  """Materialize one selected episode set as a chunk dir; return the report."""
  os.makedirs(out_dir, exist_ok=True)
  total, occ_frames, actions, terminals = 0, [], [], []
  for eid in selection['members']:
    label, e = table[eid]
    stream = index['sources'][label]['streams'][e['stream']]
    frames = load_span(stream, e['start'], e['end'])
    write_episode_chunk(out_dir, frames)
    total += e['length']
    occ_frames.append(e['occ'] * e['length'])
    if 'action' in frames:
      actions.append(np.asarray(frames['action']))
    for tkey in ('is_terminal', 'is_last'):
      if tkey in frames:
        terminals.append(float(np.asarray(frames[tkey], np.float32).mean()))
        break
  return dict(
      directory=out_dir, n_episodes=len(selection['members']),
      n_transitions=int(total),
      occ_recomputed=round(float(np.sum(occ_frames)) / total, 6),
      occ_search=selection['occ'], cov_search=selection['cov'],
      mixture=selection['mixture'],
      terminal_fraction=round(float(np.mean(terminals)), 6)
      if terminals else None,
      action_stats=action_stats(np.concatenate(actions, 0))
      if actions else None)


def _pairwise_deltas(a, b):
  """Confound checklist deltas between two side/level reports."""
  deltas = dict(n_transitions=abs(a['n_transitions'] - b['n_transitions']))
  if a['action_stats'] and b['action_stats']:
    m0, m1 = np.asarray(a['action_stats']['mean']), \
        np.asarray(b['action_stats']['mean'])
    d0, d1 = np.asarray(a['action_stats']['std']), \
        np.asarray(b['action_stats']['std'])
    deltas['action_mean_maxabs'] = round(float(np.abs(m0 - m1).max()), 5)
    deltas['action_std_maxabs'] = round(float(np.abs(d0 - d1).max()), 5)
  if a['terminal_fraction'] is not None and \
      b['terminal_fraction'] is not None:
    deltas['terminal_fraction'] = round(
        abs(a['terminal_fraction'] - b['terminal_fraction']), 6)
  deltas['episode_length'] = 0
  return deltas


def cmd_build(args):
  with open(args.index) as f:
    index = json.load(f)
  with open(args.pairs) as f:
    pairs = json.load(f)
  pair = pairs['pairs'][args.which]
  if pair is None:
    raise SystemExit(f'no {args.which} pair in {args.pairs}')
  table = episode_by_eid(index)

  manifest = dict(task=index['task'], which=args.which, index=args.index,
                  pairs=args.pairs, ep_len=index['ep_len'],
                  chunk_contract='one chunk per episode, succ=0; training '
                                 'windows never span episode boundaries '
                                 '(dropped on both sides of the pair)',
                  note='gradient-update count and model capacity are '
                       'equalized at offline_fit time (same config/steps '
                       'for both sides)',
                  sides=[])
  for si, side in enumerate(pair['sides']):
    out_dir = os.path.join(args.output_root, f'side{si}')
    side_report = _write_buffer(index, table, side, out_dir)
    manifest['sides'].append(side_report)
    print(f'side{si}: {side_report["n_episodes"]} episodes, '
          f'{side_report["n_transitions"]} frames -> {out_dir}')

  # Confound checklist deltas (runbook Phase 6 item 2).
  s0, s1 = manifest['sides']
  deltas = _pairwise_deltas(s0, s1)
  deltas['source_l1'] = pair['source_l1']
  manifest['confound_deltas'] = deltas

  path = os.path.join(args.output_root, 'manifest.json')
  with open(path, 'w') as f:
    json.dump(manifest, f, indent=2)
  print(f'confound deltas: {deltas}')
  print(f'-> {path}')


def cmd_build_dose(args):
  """Materialize all dose levels (addendum E3) under one output root."""
  with open(args.index) as f:
    index = json.load(f)
  with open(args.dose) as f:
    dose = json.load(f)
  if dose.get('decision') != 'OK' or not dose.get('levels'):
    raise SystemExit(f"dose search not OK in {args.dose}")
  table = episode_by_eid(index)

  manifest = dict(task=index['task'], kind='dose', index=args.index,
                  dose=args.dose, ep_len=index['ep_len'],
                  chunk_contract='one chunk per episode, succ=0; training '
                                 'windows never span episode boundaries '
                                 '(dropped identically at every level)',
                  note='gradient-update count and model capacity are '
                       'equalized at offline_fit time (same config/steps '
                       'for every level)',
                  levels=[])
  for lv in dose['levels']:
    out_dir = os.path.join(args.output_root, f"level{lv['level']}")
    report = _write_buffer(index, table, lv, out_dir)
    report.update(level=lv['level'], target_occ=lv['target'])
    manifest['levels'].append(report)
    print(f"level{lv['level']}: {report['n_episodes']} episodes, "
          f"{report['n_transitions']} frames, "
          f"occ={report['occ_recomputed']} -> {out_dir}")

  # Worst-case pairwise confound deltas across levels.
  worst = {}
  reports = manifest['levels']
  for x in range(len(reports)):
    for y in range(x + 1, len(reports)):
      for k, v in _pairwise_deltas(reports[x], reports[y]).items():
        worst[k] = max(worst.get(k, 0), v)
      l1 = _source_l1(reports[x], reports[y], reports[x]['n_episodes'])
      worst['source_l1'] = max(worst.get('source_l1', 0), round(l1, 4))
  manifest['confound_deltas_max'] = worst

  path = os.path.join(args.output_root, 'manifest.json')
  with open(path, 'w') as f:
    json.dump(manifest, f, indent=2)
  print(f'max pairwise confound deltas: {worst}')
  print(f'-> {path}')


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

def _synth_source(out_dir, n_steps, catch_frac, spread, seed, ep_len=200,
                  chunk_len=128):
  os.makedirs(out_dir, exist_ok=True)
  rng = np.random.default_rng(seed)
  pos = np.zeros((n_steps, 4), np.float32)
  rew = np.zeros(n_steps, np.float32)
  first = np.zeros(n_steps, bool)
  t = 0
  while t < n_steps:
    first[t] = True
    n = min(ep_len, n_steps - t)
    catch = rng.random() < catch_frac
    cup = rng.normal(0, spread, 2)
    for i in range(n):
      if catch and i / max(n - 1, 1) > 0.5:
        ball = cup + rng.normal(0, 0.01, 2)
        rew[t + i] = 1.0
      else:
        ball = cup + rng.normal(0, 0.5, 2) + spread * rng.random(2)
      pos[t + i, :2], pos[t + i, 2:] = cup, ball
    t += n
  vel = rng.normal(0, spread, (n_steps, 4)).astype(np.float32)
  act = rng.normal(0, 1, (n_steps, 2)).astype(np.float32)
  uuids = [elements.UUID() for _ in range(0, n_steps, chunk_len)]
  for ci, start in enumerate(range(0, n_steps, chunk_len)):
    end = min(start + chunk_len, n_steps)
    succ = uuids[ci + 1] if ci + 1 < len(uuids) else elements.UUID(0)
    name = f'{elements.timestamp(millis=True)}-{uuids[ci]}-{succ}-{end - start}.npz'
    np.savez_compressed(os.path.join(out_dir, name),
                        position=pos[start:end], velocity=vel[start:end],
                        reward=rew[start:end], is_first=first[start:end],
                        action=act[start:end])


def cmd_selfcheck(args):
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    goal, p2e = f'{tmp}/goal/replay', f'{tmp}/p2e/replay'
    _synth_source(goal, 8000, 0.8, 0.3, seed=1)
    _synth_source(p2e, 8000, 0.05, 1.5, seed=2)

    ns = argparse.Namespace(
        task='dmc_cup_catch', replay=[f'goal={goal}', f'p2e={p2e}'],
        output=f'{tmp}/episodes.json')
    cmd_index(ns)
    ns = argparse.Namespace(
        index=f'{tmp}/episodes.json', ref_replay=goal, n_episodes=20,
        n_candidates=120, dirichlet=0.3, max_overlap=0.2, max_frames=2000,
        knn=12, logc=1.0, cov_match_frac=0.1, occ_sep_mult=3.0, boot=50,
        seed=0, output=f'{tmp}/pairs.json')
    cmd_search(ns)
    with open(f'{tmp}/pairs.json') as f:
      pairs = json.load(f)
    assert pairs['pairs']['q1'] is not None, 'no q1 pair on synthetic data'
    ns = argparse.Namespace(
        index=f'{tmp}/episodes.json', pairs=f'{tmp}/pairs.json', which='q1',
        output_root=f'{tmp}/q1')
    cmd_build(ns)
    with open(f'{tmp}/q1/manifest.json') as f:
      manifest = json.load(f)

    # Rebuilt chunks: parse, chain, and match the manifest.
    for side in manifest['sides']:
      streams = probeset.chain_streams(side['directory'])
      assert len(streams) == side['n_episodes'], \
          (len(streams), side['n_episodes'])
      spec = regimes.spec('dmc_cup_catch')
      occs, total = [], 0
      for chain in streams:
        assert len(chain) == 1
        data = probeset.load_stream(chain)
        assert bool(np.asarray(data['is_first'])[0])
        q = spec['fn']({k: np.asarray(v) for k, v in data.items()})
        occs.append(float((q < spec['threshold']).mean()))
        total += len(q)
      assert total == side['n_transitions']
      recomputed = float(np.mean(occs))
      assert abs(recomputed - side['occ_recomputed']) < 1e-4, \
          (recomputed, side['occ_recomputed'])
      assert abs(recomputed - side['occ_search']) < 0.02

    # Ingestion: embodied Replay must load the chunks and hold items
    # (constructed with a directory, as dreamerv3.main.make_replay does).
    from embodied.core.replay import Replay
    rep = Replay(length=64, directory=f'{tmp}/replay_out')
    rep.load(directory=manifest['sides'][0]['directory'])
    n_items = len(rep.items)
    assert n_items > 0, 'Replay loaded no sampleable items'
    q1 = manifest['confound_deltas']
    assert q1['n_transitions'] == 0
    print(f'selfcheck PASS (q1 docc={pairs["pairs"]["q1"]["docc"]:.3f} at '
          f'dcov={pairs["pairs"]["q1"]["dcov"]:.3f}; replay items={n_items})')


def cmd_selfcheck_e3(args):
  """End-to-end check of search-dose / search-rpair / build-dose on
  synthetic sources (three collection policies so source exclusion keeps a
  feasible high-occ pool)."""
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    goal, apt, p2e = (f'{tmp}/goal/replay', f'{tmp}/apt/replay',
                      f'{tmp}/p2e/replay')
    _synth_source(goal, 8000, 0.8, 0.3, seed=1)
    _synth_source(apt, 8000, 0.4, 0.8, seed=3)
    _synth_source(p2e, 8000, 0.05, 1.5, seed=2)

    ns = argparse.Namespace(
        task='dmc_cup_catch',
        replay=[f'goal={goal}', f'apt={apt}', f'p2e={p2e}'],
        output=f'{tmp}/episodes.json')
    cmd_index(ns)
    common = dict(
        index=f'{tmp}/episodes.json', ref_replay=goal, n_episodes=20,
        n_candidates=150, dirichlet=0.3, max_overlap=0.2, max_frames=2000,
        knn=12, logc=1.0, cov_match_frac=0.15, occ_sep_mult=3.0, boot=50,
        seed=0)

    # Baseline pair search (for the rpair target + dominant source).
    ns = argparse.Namespace(**common, output=f'{tmp}/pairs.json')
    cmd_search(ns)
    with open(f'{tmp}/pairs.json') as f:
      pairs = json.load(f)
    q1 = pairs['pairs']['q1']
    assert q1 is not None, 'no q1 pair on synthetic data'
    hi_side = max(q1['sides'], key=lambda s: s['occ'])
    by_src = collections.Counter()
    for k, v in hi_side['mixture'].items():
      by_src[k.split('/')[0]] += v
    dominant = by_src.most_common(1)[0][0]

    # Dose levels: monotone occupancy, pairwise coverage-matched.
    ns = argparse.Namespace(**common, levels=3, beam=40,
                            output=f'{tmp}/dose.json')
    cmd_search_dose(ns)
    with open(f'{tmp}/dose.json') as f:
      dose = json.load(f)
    assert dose['decision'] == 'OK', dose.get('reason')
    occs = [lv['occ'] for lv in dose['levels']]
    assert occs == sorted(occs), f'dose levels not monotone: {occs}'
    assert dose['max_pairwise_dcov'] <= dose['cov_tol'] + 1e-9
    ns = argparse.Namespace(index=f'{tmp}/episodes.json',
                            dose=f'{tmp}/dose.json',
                            output_root=f'{tmp}/dose_buffers')
    cmd_build_dose(ns)
    with open(f'{tmp}/dose_buffers/manifest.json') as f:
      dm = json.load(f)
    assert len(dm['levels']) == 3
    for lv, srch in zip(dm['levels'], dose['levels']):
      assert lv['n_transitions'] > 0
      assert abs(lv['occ_recomputed'] - srch['occ']) < 0.02
    assert dm['confound_deltas_max']['n_transitions'] == 0

    # Composition-robustness pair: high-occ side excludes the dominant
    # source of the baseline q1 pair; sides ordered low->high occupancy.
    ns = argparse.Namespace(**common, exclude_sources=[dominant],
                            target_docc=q1['docc'],
                            output=f'{tmp}/rpairs.json')
    cmd_search_rpair(ns)
    with open(f'{tmp}/rpairs.json') as f:
      rp = json.load(f)
    assert rp['decision'] == 'OK', 'rpair SHORT on synthetic data'
    r1 = rp['pairs']['q1']
    assert r1['sides'][0]['occ'] <= r1['sides'][1]['occ']
    assert not any(k.split('/')[0] == dominant
                   for k in r1['sides'][1]['mixture'])
    assert r1['docc'] >= rp['occ_sep_min'] - 1e-9
    assert r1['dcov'] <= rp['cov_tol'] + 1e-9
    ns = argparse.Namespace(index=f'{tmp}/episodes.json',
                            pairs=f'{tmp}/rpairs.json', which='q1',
                            output_root=f'{tmp}/r1')
    cmd_build(ns)
    with open(f'{tmp}/r1/manifest.json') as f:
      rm = json.load(f)
    assert rm['confound_deltas']['n_transitions'] == 0
    assert rm['sides'][0]['occ_recomputed'] <= rm['sides'][1]['occ_recomputed']

    print(f"selfcheck-e3 PASS (dose occs={occs}, "
          f"rpair docc={r1['docc']:.3f} target={q1['docc']:.3f} "
          f"excluding {dominant!r})")


def main():
  p = argparse.ArgumentParser(description=__doc__)
  sub = p.add_subparsers(dest='cmd', required=True)

  ix = sub.add_parser('index')
  ix.add_argument('--replay', nargs='+', required=True,
                  help='label=replay_dir ...')
  ix.add_argument('--task', required=True)
  ix.add_argument('--output', required=True)
  ix.set_defaults(fn=cmd_index)

  def add_search_args(sp):
    sp.add_argument('--index', required=True)
    sp.add_argument('--ref_replay', required=True,
                    help='Fixed per-domain reference buffer for the scaler.')
    sp.add_argument('--n_episodes', type=int, default=200)
    sp.add_argument('--n_candidates', type=int, default=400)
    sp.add_argument('--dirichlet', type=float, default=0.3)
    sp.add_argument('--max_overlap', type=float, default=0.2)
    sp.add_argument('--max_frames', type=int, default=3000)
    sp.add_argument('--knn', type=int, default=12)
    sp.add_argument('--logc', type=float, default=1.0)
    sp.add_argument('--cov_match_frac', type=float, default=0.05)
    sp.add_argument('--occ_sep_mult', type=float, default=3.0)
    sp.add_argument('--boot', type=int, default=100)
    sp.add_argument('--seed', type=int, default=0)
    sp.add_argument('--output', required=True)

  se = sub.add_parser('search')
  add_search_args(se)
  se.set_defaults(fn=cmd_search)

  sd = sub.add_parser('search-dose')
  add_search_args(sd)
  sd.add_argument('--levels', type=int, default=4)
  sd.add_argument('--beam', type=int, default=40,
                  help='Per-level candidate shortlist for the assignment '
                       'search.')
  sd.set_defaults(fn=cmd_search_dose)

  sr = sub.add_parser('search-rpair')
  add_search_args(sr)
  sr.add_argument('--exclude_sources', nargs='+', required=True,
                  help='Source labels barred from the high-occupancy side.')
  sr.add_argument('--target_docc', type=float, required=True,
                  help="The original Q1 pair's docc to reproduce.")
  sr.set_defaults(fn=cmd_search_rpair)

  bu = sub.add_parser('build')
  bu.add_argument('--index', required=True)
  bu.add_argument('--pairs', required=True)
  bu.add_argument('--which', choices=['q1', 'q2'], required=True)
  bu.add_argument('--output_root', required=True)
  bu.set_defaults(fn=cmd_build)

  bd = sub.add_parser('build-dose')
  bd.add_argument('--index', required=True)
  bd.add_argument('--dose', required=True)
  bd.add_argument('--output_root', required=True)
  bd.set_defaults(fn=cmd_build_dose)

  sc = sub.add_parser('selfcheck')
  sc.set_defaults(fn=cmd_selfcheck)

  s3 = sub.add_parser('selfcheck-e3')
  s3.set_defaults(fn=cmd_selfcheck_e3)

  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
