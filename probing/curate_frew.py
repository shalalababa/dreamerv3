"""Curated high-f_R buffer search (PREREG_highfr_wave_20260802.md).

Selects two 200-episode draws from the frozen Axis-1 episode index whose
mean rewarded-regime occupancy sits at registered targets (0.60, 0.80) —
the manipulation of the falling-limb wave (PREREG_highfr_theory_20260730:
curated draws at matched 200-episode volume, same collector pool,
f_R in [0.60, 0.80]). Output is pairs-shaped (key 'q1', side0 = lower
target, side1 = higher target), so the FROZEN builder materializes it
unchanged (`build_controlled_replay build --which q1` — same chunk
contract, same stepid rewrite, same manifest/confound machinery as every
existing Axis-1 buffer).

Selection rule (deterministic, auditable): episodes sorted by occupancy
descending (ties by eid). Targets processed in DESCENDING order; for each
target, the CONTIGUOUS window of K episodes in the remaining sorted pool
whose mean occupancy is closest to the target is selected (ties -> the
earlier window); selected episodes are removed before the next target, so
the sides are disjoint by construction. A window missing its target by
more than --tol is infeasible: for the HIGHEST target only, the top-K
window is the registered fallback iff its mean >= FALLBACK_MIN; any other
infeasibility marks the cell SHORT and no pairs are emitted (the wave
halts pre-fit; a dated amendment is required). No coverage matching: the
manipulation is f_R itself; support breadth is MEASURED afterward
(spectral_v1_1 diversity_pr), not controlled.

Search touches only index occupancies — no frames, no fits, no transfer
outcome exists or is revealed at curation time.

Amendment 2 (PREREG_highfr_wave_amend2_20260802): --require_chunks
excludes from the SEARCH POOL every episode whose span covers a chunk
file absent on disk (scratch purge ate 26 donor chunks between indexing
and build). The covering-chunk rule is identical to the builder's
load_span, so a filtered selection is buildable by construction. The
index itself is NEVER rewritten — eid numbering (the builder contract)
is preserved — and excluded eids + missing files are recorded in the
output json. Availability is a filesystem fact, not an outcome; all
targets/tol/fallback/SHORT semantics re-arm unchanged on the filtered
pool.

Breadth-feasibility mode (PREREG_breadth_theory_20260803.md): find two
disjoint K-episode cells MATCHED on occupancy (both within --div_tol of
--div_target) and maximally SEPARATED in diversity_pr — the
participation ratio of the rewarded-frame symlog-proprio covariance,
computed EXACTLY as `spectral_measure` computes it (same obs_keys, same
symlog, same np.cov; PR is scale-invariant so the ddof convention is
immaterial). Two stages so the chunk pass runs once:

  scan-moments : one pass over the index's chunk files accumulating
      per-episode rewarded-frame sufficient statistics (count, sum,
      outer-product sum). Episodes touching a MISSING chunk file are
      recorded and excluded (the Amendment-2 availability rule — the
      pool has known purge gaps), so every scanned episode is buildable
      by construction. Value-blind: touches proprio frames + rewards
      only; no fit or transfer outcome exists or is revealed.
  search-div   : deterministic greedy-plus-repair over the cached
      moments. Candidates = scanned episodes with occupancy within
      --div_corridor of --div_target and >= --min_rew rewarded frames.
      side0 (lo-div) is grown FIRST by PR-minimizing greedy, side1
      (hi-div) from the remainder by PR-maximizing greedy; an exchange
      pass then repairs each cell's mean occupancy into --div_tol with
      minimal objective damage. Fully deterministic (ties by eid).
      Output is pairs-shaped (q1) so the FROZEN builder materializes it
      unchanged; realized diversity_pr is re-measured by the frozen
      spectral instrument on the BUILT buffers — the search is a
      heuristic, the instrument gates bind (the search-frew pattern).

Usage:
  python -m probing.curate_frew search-frew \
      --index $RUNROOT/axis1_finger/episodes.json \
      --n_episodes 200 --targets 0.41 0.80 --tol 0.05 \
      --require_chunks \
      --output $RUNROOT/axis1_finger/frew_pairs.json
  python -m probing.curate_frew scan-moments \
      --index $RUNROOT/axis1_finger/episodes.json \
      --output $RUNROOT/axis1_finger/div_moments.npz
  python -m probing.curate_frew search-div \
      --index $RUNROOT/axis1_finger/episodes.json \
      --moments $RUNROOT/axis1_finger/div_moments.npz \
      --n_episodes 200 --div_target 0.3232 \
      --output $RUNROOT/axis1_finger/div_pairs.json
  python -m probing.curate_frew --selfcheck
"""

import argparse
import collections
import json
import os

import numpy as np

from probing import build_controlled_replay as bcr

FALLBACK_MIN = 0.60

# search-div constants (feasibility defaults; the wave registration will
# freeze the binding gate constants — these steer the search only).
DIV_TOL = 0.02        # cell mean-occupancy tolerance around --div_target
DIV_CORRIDOR = 0.10   # candidate occupancy corridor around --div_target
MIN_REW_EPISODE = 10  # fewer rewarded frames -> diversity contribution
                      # ill-defined; episode dropped from the div pool
REPAIR_MAX_ITERS = 300
REPAIR_MEMBER_TOP = 30   # exchange pass scans the most-off-target members
REPAIR_CAND_TOP = 60     # against the most-correcting candidates


def _entries(index):
  """[(eid, label, occ)] in the frozen episode_by_eid enumeration order."""
  table = bcr.episode_by_eid(index)
  return [(eid, label, float(e['occ'])) for eid, (label, e)
          in sorted(table.items())]


def missing_chunk_eids(index):
  """eid -> [missing chunk files] for episodes touching absent chunks.

  Covering-chunk rule identical to the builder's load_span (searchsorted
  over cumulative stream lengths), so an episode is flagged iff
  materializing it would np.load a file that no longer exists.
  """
  exists = {}
  out = {}
  for eid, (label, e) in sorted(bcr.episode_by_eid(index).items()):
    stream = index['sources'][label]['streams'][e['stream']]
    bounds = np.cumsum([0] + stream['lengths'])
    first = int(np.searchsorted(bounds, e['start'], side='right') - 1)
    last = int(np.searchsorted(bounds, e['end'] - 1, side='right') - 1)
    gone = []
    for ci in range(first, last + 1):
      f = stream['files'][ci]
      if f not in exists:
        exists[f] = os.path.exists(f)
      if not exists[f]:
        gone.append(f)
    if gone:
      out[eid] = gone
  return out


def best_window(occs, k, target):
  """(start, mean, dev) of the k-window with mean closest to target.

  occs is sorted descending, so window means are nonincreasing in start;
  the scan keeps the earliest window at minimal deviation (tie-break).
  """
  n = len(occs)
  assert n >= k, (n, k)
  csum = np.concatenate([[0.0], np.cumsum(occs)])
  means = (csum[k:] - csum[:-k]) / k
  devs = np.abs(means - target)
  start = int(np.argmin(devs))          # argmin returns the FIRST minimum
  return start, float(means[start]), float(devs[start])


def search(index, targets, k, tol, seed=0, boot=200, exclude=frozenset()):
  assert 'ep_len' in index, 'index not restricted to modal episode length'
  assert len(targets) == 2 and targets[0] != targets[1], targets
  entries = sorted(_entries(index), key=lambda t: (-t[2], t[0]))
  if exclude:
    entries = [t for t in entries if t[0] not in exclude]
  occ_by_eid = {eid: occ for eid, _, occ in entries}
  label_by_eid = {eid: lab for eid, lab, _ in entries}
  rng = np.random.default_rng(seed)

  remaining = list(entries)
  cells = []
  for rank, target in enumerate(sorted(targets, reverse=True)):
    cell = dict(target=target, status='SHORT', fallback=False)
    if len(remaining) < k:
      cell['reason'] = f'pool {len(remaining)} < K {k}'
      cells.append(cell)
      continue
    occs = np.asarray([occ for _, _, occ in remaining])
    start, mean, dev = best_window(occs, k, target)
    if dev > tol:
      if rank == 0 and occs[:k].mean() >= FALLBACK_MIN:
        start, mean = 0, float(occs[:k].mean())
        cell['fallback'] = True
      else:
        cell['reason'] = (f'no K-window within tol {tol} of {target} '
                          f'(best mean {mean:.4f}); top-K mean '
                          f'{float(occs[:k].mean()):.4f}')
        cells.append(cell)
        continue
    picked = remaining[start:start + k]
    # Reported-mean/members invariant (reviewer M2): the mean the gates
    # and targets rest on must be the mean of exactly the emitted set.
    assert abs(float(np.mean([occ for _, _, occ in picked])) - mean) \
        < 1e-9, (mean, len(picked))
    cell.update(status='OK', mean=round(mean, 6),
                members=sorted(eid for eid, _, _ in picked))
    cells.append(cell)
    picked_ids = {eid for eid, _, _ in picked}
    remaining = [t for t in remaining if t[0] not in picked_ids]

  out = dict(kind='frew', task=index['task'], n_episodes=k,
             ep_len=index['ep_len'],
             criteria=dict(targets=sorted(targets), tol=tol,
                           fallback_min=FALLBACK_MIN, seed=seed, boot=boot,
                           n_excluded=len(exclude),
                           objective='contiguous K-window of the '
                                     'occupancy-desc pool closest to each '
                                     'target, targets descending, sides '
                                     'disjoint by removal; selection is '
                                     'fully deterministic — seed/boot feed '
                                     'only the descriptive occ_noise '
                                     'bootstrap'),
             cells=[{kk: v for kk, v in c.items() if kk != 'members'}
                    for c in cells])
  if not all(c['status'] == 'OK' for c in cells):
    out.update(decision='SHORT', pairs=dict(q1=None))
    return out

  def side(cell):
    members = cell['members']
    occs = np.asarray([occ_by_eid[m] for m in members])
    boots = [float(occs[rng.integers(0, k, k)].mean()) for _ in range(boot)]
    mix = collections.Counter(
        f'{label_by_eid[m]}/{bcr.occ_bin(occ_by_eid[m])}' for m in members)
    return dict(cov=None, occ=cell['mean'],
                occ_noise=round(float(np.std(boots)), 6),
                mixture=dict(sorted(mix.items())),
                members=[int(m) for m in members],
                target=cell['target'], fallback=cell['fallback'])

  lo, hi = sorted(cells, key=lambda c: c['mean'])
  s0, s1 = side(lo), side(hi)
  assert not set(s0['members']) & set(s1['members'])
  out.update(decision='OK', pairs=dict(q1=dict(
      dcov=None, docc=round(s1['occ'] - s0['occ'], 6), overlap=0.0,
      source_l1=bcr._source_l1(s0, s1, k), sides=[s0, s1])))
  return out


def cmd_search(args):
  with open(args.index) as f:
    index = json.load(f)
  exclude, missing = frozenset(), {}
  if args.require_chunks:
    missing = missing_chunk_eids(index)
    exclude = frozenset(missing)
    files = sorted({f for gone in missing.values() for f in gone})
    print(f'chunk check: {len(missing)} episode(s) excluded '
          f'({len(files)} missing chunk file(s))')
  out = search(index, tuple(args.targets), args.n_episodes, args.tol,
               seed=args.seed, boot=args.boot, exclude=exclude)
  out['index'] = args.index
  if args.require_chunks:
    out['availability'] = dict(
        require_chunks=True,
        excluded_eids={int(k): v for k, v in sorted(missing.items())},
        missing_files=sorted({f for g in missing.values() for f in g}))
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  for c in out['cells']:
    line = f"target {c['target']}: {c['status']}"
    if c['status'] == 'OK':
      line += f" mean={c['mean']:.4f} fallback={c['fallback']}"
    else:
      line += f" ({c['reason']})"
    print(line)
  print(f"{out['decision']} -> {args.output}")
  if out['decision'] != 'OK':
    raise SystemExit(1)


# --------------------------------------------------------------------------
# Breadth feasibility: per-episode rewarded-frame moments + diversity search
# --------------------------------------------------------------------------

def scan_moments(index):
  """One chunk pass -> per-eid rewarded-frame sufficient statistics.

  Returns (moments, meta): moments maps eid -> dict(n_rew, n_frames,
  s [D], S [D, D]) over that episode's reward>0 frames in the SAME
  feature space as spectral_measure's diversity_pr (obs_keys' proprio
  columns, symlog'd, float64). Episodes whose span touches a chunk file
  absent on disk are excluded entirely and recorded in meta (the
  availability rule) — every scanned episode is buildable.
  """
  from probing import spectral_measure as sm
  table = bcr.episode_by_eid(index)
  by_stream = collections.defaultdict(list)
  for eid, (label, e) in sorted(table.items()):
    by_stream[(label, e['stream'])].append((eid, e['start'], e['end']))
  keys = None
  moments = {}
  unavailable = {}
  for (label, si), eps in sorted(by_stream.items()):
    stream = index['sources'][label]['streams'][si]
    bounds = np.cumsum([0] + stream['lengths'])
    # availability first: an episode touching any absent chunk is out
    gone_files = [f for f in stream['files'] if not os.path.exists(f)]
    gone_ranges = [(bounds[ci], bounds[ci + 1])
                   for ci, f in enumerate(stream['files'])
                   if not os.path.exists(f)]
    live = []
    for eid, start, end in eps:
      hit = [f for (a, b), f in zip(gone_ranges, gone_files)
             if start < b and end > a]
      if hit:
        unavailable[eid] = hit
      else:
        live.append((eid, start, end))
    if not live:
      continue
    starts = np.asarray([s for _, s, _ in live])
    ends = np.asarray([e for _, _, e in live])
    for ci, path in enumerate(stream['files']):
      b0, b1 = int(bounds[ci]), int(bounds[ci + 1])
      touch = np.flatnonzero((starts < b1) & (ends > b0))
      if not len(touch):
        continue
      with np.load(path) as data:
        if keys is None:
          probe = {k: data[k] for k in data.keys()}
          keys = sm.obs_keys(probe)
        chunk = {k: data[k] for k in keys}
        rew = np.asarray(data['reward'], np.float64).reshape(-1)
      fr = sm.symlog(sm.frames_of(chunk, keys))
      for ti in touch:
        eid, start, end = live[ti]
        lo, hi = max(start - b0, 0), min(end - b0, b1 - b0)
        acc = moments.get(eid)
        if acc is None:
          d = fr.shape[1]
          acc = moments[eid] = dict(
              n_rew=0, n_frames=0, s=np.zeros(d), S=np.zeros((d, d)))
        seg, rseg = fr[lo:hi], rew[lo:hi]
        mask = rseg > 0
        acc['n_frames'] += int(hi - lo)
        if mask.any():
          x = seg[mask]
          acc['n_rew'] += int(mask.sum())
          acc['s'] += x.sum(0)
          acc['S'] += x.T @ x
  # frame-count integrity: every scanned episode saw its full span
  for eid, acc in moments.items():
    label, e = table[eid]
    assert acc['n_frames'] == e['end'] - e['start'], (
        eid, acc['n_frames'], e['end'] - e['start'])
  meta = dict(task=index['task'], ep_len=index['ep_len'],
              obs_keys=keys, dim=(len(next(iter(moments.values()))['s'])
                                  if moments else 0),
              n_scanned=len(moments), n_unavailable=len(unavailable),
              unavailable_eids={int(k): v
                                for k, v in sorted(unavailable.items())},
              missing_files=sorted({f for g in unavailable.values()
                                    for f in g}))
  return moments, meta


def save_moments(path, moments, meta):
  eids = np.asarray(sorted(moments), np.int64)
  d = meta['dim']
  np.savez_compressed(
      path, eids=eids,
      n_rew=np.asarray([moments[e]['n_rew'] for e in eids], np.int64),
      n_frames=np.asarray([moments[e]['n_frames'] for e in eids], np.int64),
      sums=np.stack([moments[e]['s'] for e in eids]) if len(eids)
      else np.zeros((0, d)),
      outers=np.stack([moments[e]['S'] for e in eids]) if len(eids)
      else np.zeros((0, d, d)))
  with open(_meta_path(path), 'w') as f:
    json.dump(meta, f, indent=2)


def load_moments(path):
  with np.load(path) as z:
    arrs = {k: z[k] for k in z.keys()}
  with open(_meta_path(path)) as f:
    meta = json.load(f)
  moments = {int(eid): dict(n_rew=int(arrs['n_rew'][i]),
                            n_frames=int(arrs['n_frames'][i]),
                            s=arrs['sums'][i], S=arrs['outers'][i])
             for i, eid in enumerate(arrs['eids'])}
  return moments, meta


def _meta_path(path):
  base = path[:-4] if path.endswith('.npz') else path
  return base + '.meta.json'


def _pr_from_ev(ev):
  ev = np.clip(ev, 0.0, None)
  s1 = ev.sum(-1)
  s2 = (ev ** 2).sum(-1)
  return np.where(s2 > 0, s1 ** 2 / np.where(s2 > 0, s2, 1.0), 0.0)


def cell_pr(n, s, m2):
  """participation_ratio of the pooled rewarded frames, from moments.

  Matches spectral_measure.participation_ratio(flat[rewarded]) exactly:
  np.cov (ddof=1) covariance of the pooled frames; PR is additionally
  scale-invariant, so the ddof convention cannot matter.
  """
  if n < 3:
    return 0.0
  c = (m2 - np.outer(s, s) / n) / (n - 1)
  return float(_pr_from_ev(np.linalg.eigvalsh(c)))


def _stacked_pr(n, s, m2):
  """Vectorized cell_pr over leading axis: n [m], s [m,D], m2 [m,D,D]."""
  n = np.asarray(n, np.float64)
  c = (m2 - s[:, :, None] * s[:, None, :] / n[:, None, None]) \
      / (n[:, None, None] - 1)
  pr = _pr_from_ev(np.linalg.eigvalsh(c))
  return np.where(n >= 3, pr, 0.0)


class _Cell:
  """Incremental moment accumulator for one candidate cell."""

  def __init__(self, dim):
    self.members = []
    self.n = 0
    self.s = np.zeros(dim)
    self.m2 = np.zeros((dim, dim))

  def add(self, eid, mom):
    self.members.append(eid)
    self.n += mom['n_rew']
    self.s = self.s + mom['s']
    self.m2 = self.m2 + mom['S']

  def remove(self, eid, mom):
    self.members.remove(eid)
    self.n -= mom['n_rew']
    self.s = self.s - mom['s']
    self.m2 = self.m2 - mom['S']

  def pr(self):
    return cell_pr(self.n, self.s, self.m2)


def _grow_cell(cand_eids, moments, k, maximize):
  """Deterministic greedy: seed = the candidate whose OWN single-episode
  PR is most extreme in the cell's objective direction (greedy never
  evicts its seed, so an objective-misaligned seed — e.g. largest-mass —
  would be locked into the cell); then repeatedly add the candidate
  whose inclusion optimizes the cell PR (ties resolved first-index on
  the eid-sorted candidate axis)."""
  cands = sorted(cand_eids)
  dim = len(moments[cands[0]]['s'])
  cell = _Cell(dim)
  own = {e: cell_pr(moments[e]['n_rew'], moments[e]['s'], moments[e]['S'])
         for e in cands}
  if maximize:
    seed = max(cands, key=lambda e: (own[e], -e))
  else:
    seed = min(cands, key=lambda e: (own[e], e))
  cell.add(seed, moments[seed])
  avail = [e for e in cands if e != seed]
  while len(cell.members) < k:
    assert avail, 'candidate pool exhausted mid-growth'
    ns = np.asarray([cell.n + moments[e]['n_rew'] for e in avail])
    ss = np.stack([cell.s + moments[e]['s'] for e in avail])
    m2s = np.stack([cell.m2 + moments[e]['S'] for e in avail])
    prs = _stacked_pr(ns, ss, m2s)
    pick = int(np.argmax(prs) if maximize else np.argmin(prs))
    eid = avail.pop(pick)
    cell.add(eid, moments[eid])
  return cell


def _repair_occ(cell, cand_eids, moments, occ, target, tol, maximize):
  """Exchange pass: swap members for outside candidates until the cell
  mean occupancy is within tol of target, choosing at each step the
  mean-correcting swap with the least objective damage (max PR-after
  for a maximize cell, min for a minimize cell; ties by (out, in) eid).
  Returns the number of swaps performed."""
  outside = sorted(set(cand_eids) - set(cell.members))
  swaps = 0
  for _ in range(REPAIR_MAX_ITERS):
    mean = float(np.mean([occ[e] for e in cell.members]))
    dev = mean - target
    if abs(dev) <= tol:
      break
    k = len(cell.members)
    # need mean to move by -dev: swap out a member on the dev side for a
    # candidate on the other side; scan the most-corrective pairs only
    members = sorted(cell.members,
                     key=lambda e: (-np.sign(dev) * occ[e], e))
    members = members[:REPAIR_MEMBER_TOP]
    cands = sorted(outside, key=lambda e: (np.sign(dev) * occ[e], e))
    cands = cands[:REPAIR_CAND_TOP]
    best = None
    for m in members:
      ins = [c for c in cands
             if abs(dev + (occ[c] - occ[m]) / k) < abs(dev)]
      if not ins:
        continue
      base_n = cell.n - moments[m]['n_rew']
      base_s = cell.s - moments[m]['s']
      base_m2 = cell.m2 - moments[m]['S']
      ns = np.asarray([base_n + moments[c]['n_rew'] for c in ins])
      ss = np.stack([base_s + moments[c]['s'] for c in ins])
      m2s = np.stack([base_m2 + moments[c]['S'] for c in ins])
      prs = _stacked_pr(ns, ss, m2s)
      j = int(np.argmax(prs) if maximize else np.argmin(prs))
      score = prs[j] if maximize else -prs[j]
      key = (score, -m, -ins[j])
      if best is None or key > best[0]:
        best = (key, m, ins[j])
    if best is None:
      break                       # no corrective swap exists
    _, m, c = best
    cell.remove(m, moments[m])
    cell.add(c, moments[c])
    outside.remove(c)
    outside.append(m)
    swaps += 1
  return swaps


def search_div(index, moments, meta, k, target, tol=DIV_TOL,
               corridor=DIV_CORRIDOR, min_rew=MIN_REW_EPISODE,
               seed=0, boot=200):
  assert 'ep_len' in index, 'index not restricted to modal episode length'
  assert meta['task'] == index['task'] and meta['ep_len'] == index['ep_len'], \
      'moments cache does not match this index'
  table = bcr.episode_by_eid(index)
  occ = {eid: float(e['occ']) for eid, (label, e) in table.items()}
  label_by_eid = {eid: label for eid, (label, e) in table.items()}
  scanned = set(moments)
  in_corr = {e for e in scanned if abs(occ[e] - target) <= corridor}
  eligible = sorted(e for e in in_corr
                    if moments[e]['n_rew'] >= min_rew)
  drops = dict(
      n_indexed=len(table), n_scanned=len(scanned),
      n_unavailable=meta['n_unavailable'],
      n_out_corridor=len(scanned) - len(in_corr),
      n_low_rew=len(in_corr) - len(eligible),
      n_eligible=len(eligible))
  criteria = dict(kind='div', target=target, tol=tol, corridor=corridor,
                  min_rew=min_rew, k=k, seed=seed, boot=boot,
                  order='lo-div grown first (PR-min greedy), hi-div from '
                        'the remainder (PR-max greedy); occupancy repaired '
                        'by least-damage exchange; deterministic, ties by '
                        'eid', drops=drops)
  out = dict(kind='div', task=index['task'], n_episodes=k,
             ep_len=index['ep_len'], criteria=criteria)
  if len(eligible) < 2 * k:
    out.update(decision='SHORT', pairs=dict(q1=None),
               reason=f'eligible pool {len(eligible)} < 2K {2 * k}')
    return out

  cells, swap_counts = [], []
  pool = list(eligible)
  for maximize in (False, True):
    cell = _grow_cell(pool, moments, k, maximize)
    swaps = _repair_occ(cell, pool, moments, occ, target, tol, maximize)
    cells.append(cell)
    swap_counts.append(swaps)
    pool = [e for e in pool if e not in set(cell.members)]

  rng = np.random.default_rng(seed)

  def side(cell, swaps):
    members = sorted(cell.members)
    occs = np.asarray([occ[m] for m in members])
    boots = [float(occs[rng.integers(0, k, k)].mean()) for _ in range(boot)]
    mix = collections.Counter(
        f'{label_by_eid[m]}/{bcr.occ_bin(occ[m])}' for m in members)
    return dict(cov=None, occ=round(float(occs.mean()), 6),
                occ_noise=round(float(np.std(boots)), 6),
                mixture=dict(sorted(mix.items())),
                members=[int(m) for m in members],
                target=target, fallback=False,
                div_pr=round(cell.pr(), 6), n_rew=int(cell.n),
                repair_swaps=swaps)

  s0, s1 = side(cells[0], swap_counts[0]), side(cells[1], swap_counts[1])
  assert not set(s0['members']) & set(s1['members'])
  # pool-level context (informational): PR of all eligible pooled + of a
  # deterministic random K-subset
  pn = sum(moments[e]['n_rew'] for e in eligible)
  ps = np.sum([moments[e]['s'] for e in eligible], 0)
  pm = np.sum([moments[e]['S'] for e in eligible], 0)
  rsub = list(np.asarray(eligible)[
      np.random.default_rng(seed).choice(len(eligible), k, replace=False)])
  rn = sum(moments[int(e)]['n_rew'] for e in rsub)
  rs = np.sum([moments[int(e)]['s'] for e in rsub], 0)
  rm = np.sum([moments[int(e)]['S'] for e in rsub], 0)
  ok = (abs(s0['occ'] - target) <= tol and abs(s1['occ'] - target) <= tol)
  out.update(
      decision='OK' if ok else 'OFF-TARGET',
      div=dict(pr_lo=s0['div_pr'], pr_hi=s1['div_pr'],
               separation=round(s1['div_pr'] - s0['div_pr'], 6),
               pr_pool=round(cell_pr(pn, ps, pm), 6),
               pr_random_k=round(cell_pr(rn, rs, rm), 6),
               docc=round(s1['occ'] - s0['occ'], 6)),
      pairs=dict(q1=dict(
          dcov=None, docc=round(s1['occ'] - s0['occ'], 6), overlap=0.0,
          source_l1=bcr._source_l1(s0, s1, k), sides=[s0, s1])))
  return out


def cmd_scan(args):
  with open(args.index) as f:
    index = json.load(f)
  moments, meta = scan_moments(index)
  meta['index'] = args.index
  save_moments(args.output, moments, meta)
  print(f"scanned {meta['n_scanned']} episodes "
        f"({meta['n_unavailable']} unavailable, "
        f"{len(meta['missing_files'])} missing chunk file(s)) "
        f"dim={meta['dim']} -> {args.output}")


def cmd_search_div(args):
  with open(args.index) as f:
    index = json.load(f)
  moments, meta = load_moments(args.moments)
  out = search_div(index, moments, meta, k=args.n_episodes,
                   target=args.div_target, tol=args.div_tol,
                   corridor=args.div_corridor, min_rew=args.min_rew,
                   seed=args.seed, boot=args.boot)
  out['index'] = args.index
  out['moments'] = args.moments
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  if out['decision'] == 'SHORT':
    print(f"SHORT ({out['reason']}) -> {args.output}")
    raise SystemExit(1)
  d = out['div']
  print(f"lo-div occ={out['pairs']['q1']['sides'][0]['occ']:.4f} "
        f"pr={d['pr_lo']:.3f} | hi-div "
        f"occ={out['pairs']['q1']['sides'][1]['occ']:.4f} "
        f"pr={d['pr_hi']:.3f} | separation={d['separation']:.3f} "
        f"(pool {d['pr_pool']:.3f}, random-K {d['pr_random_k']:.3f})")
  print(f"{out['decision']} -> {args.output}")
  if out['decision'] != 'OK':
    raise SystemExit(1)


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _hand_index(occ_lists, ep_len=100):
  sources = {}
  for label, occs in occ_lists.items():
    sources[label] = dict(replay='', streams=[], episodes=[
        dict(stream=0, start=i * ep_len, end=(i + 1) * ep_len,
             length=ep_len, occ=o) for i, o in enumerate(occs)])
  return dict(task='dmc_cup_catch', ep_len=ep_len, sources=sources)


def _brute_best_dev(occs, k, target):
  occs = np.asarray(occs)
  return min(abs(float(occs[i:i + k].mean()) - target)
             for i in range(len(occs) - k + 1))


def _write_div_stream(out_dir, episodes, chunk_len=64):
  """Write one uuid-chained chunk stream of hand-built cup episodes.

  episodes: list of (pos [T,4], vel [T,4], rew [T]) arrays. Regime for
  dmc_cup_catch is ||position[:2]-position[2:4]|| < 0.05, so callers
  control index occupancy via ball-cup distance and n_rew via reward.
  """
  os.makedirs(out_dir, exist_ok=True)
  elements = bcr.elements
  pos = np.concatenate([p for p, _, _ in episodes]).astype(np.float32)
  vel = np.concatenate([v for _, v, _ in episodes]).astype(np.float32)
  rew = np.concatenate([r for _, _, r in episodes]).astype(np.float32)
  first = np.zeros(len(pos), bool)
  t = 0
  for p, _, _ in episodes:
    first[t] = True
    t += len(p)
  act = np.zeros((len(pos), 2), np.float32)
  uuids = [elements.UUID() for _ in range(0, len(pos), chunk_len)]
  for ci, start in enumerate(range(0, len(pos), chunk_len)):
    end = min(start + chunk_len, len(pos))
    succ = uuids[ci + 1] if ci + 1 < len(uuids) else elements.UUID(0)
    name = (f'{elements.timestamp(millis=True)}-{uuids[ci]}-{succ}'
            f'-{end - start}.npz')
    np.savez_compressed(os.path.join(out_dir, name),
                        position=pos[start:end], velocity=vel[start:end],
                        reward=rew[start:end], is_first=first[start:end],
                        action=act[start:end])


def _div_episode(ep_len, n_regime, n_rew, cup_fn, vel_fn):
  """One cup episode: first n_regime frames in-regime (ball == cup), the
  first n_rew of those rewarded; the rest far off-regime."""
  pos = np.zeros((ep_len, 4))
  vel = np.zeros((ep_len, 4))
  rew = np.zeros(ep_len)
  for i in range(ep_len):
    cup = cup_fn(i)
    pos[i, :2] = cup
    pos[i, 2:] = cup if i < n_regime else cup + 1.0
    vel[i] = vel_fn(i)
  rew[:n_rew] = 1.0
  return pos, vel, rew


def selfcheck(args):
  # --- Part 1: selection-rule logic on hand-built indices -------------
  idx = _hand_index(dict(
      a=[0.95, 0.90, 0.85, 0.80],
      b=[round(0.75 - 0.05 * i, 4) for i in range(15)]))
  res = search(idx, (0.60, 0.85), k=4, tol=0.05)
  assert res['decision'] == 'OK', res
  s0, s1 = res['pairs']['q1']['sides']
  assert s0['occ'] < s1['occ'] and s0['target'] == 0.60
  assert not set(s0['members']) & set(s1['members'])
  assert abs(s1['occ'] - 0.85) <= 0.05 and abs(s0['occ'] - 0.60) <= 0.05
  assert not s0['fallback'] and not s1['fallback']
  # optimality of the hi window vs brute force over the full sorted pool
  all_occs = sorted([o for occs in (idx['sources']['a']['episodes'],
                                    idx['sources']['b']['episodes'])
                     for o in [e['occ'] for e in occs]], reverse=True)
  assert abs(abs(s1['occ'] - 0.85)
             - _brute_best_dev(all_occs, 4, 0.85)) < 1e-12
  # determinism: identical output on a second run
  assert search(idx, (0.60, 0.85), k=4, tol=0.05) == res

  # fallback: top target unreachable but top-K mean >= FALLBACK_MIN
  idx2 = _hand_index(dict(a=[0.70] * 6 + [0.40] * 10))
  res2 = search(idx2, (0.45, 0.99), k=4, tol=0.05)
  assert res2['decision'] == 'OK', res2
  hi2 = res2['pairs']['q1']['sides'][1]
  assert hi2['fallback'] and abs(hi2['occ'] - 0.70) < 1e-9

  # SHORT: top-K mean below FALLBACK_MIN kills the wave (no pairs)
  idx3 = _hand_index(dict(a=[0.30] * 12))
  res3 = search(idx3, (0.25, 0.80), k=4, tol=0.05)
  assert res3['decision'] == 'SHORT' and res3['pairs']['q1'] is None
  assert any(c['status'] == 'SHORT' for c in res3['cells'])

  # SHORT: lower target unreachable after hi-side removal
  idx4 = _hand_index(dict(a=[0.80] * 4 + [0.10] * 8))
  res4 = search(idx4, (0.60, 0.80), k=4, tol=0.05)
  assert res4['decision'] == 'SHORT', res4
  assert res4['cells'][0]['status'] == 'OK'      # hi cell fine
  assert res4['cells'][1]['status'] == 'SHORT'   # 0.60 gone with the top

  # pool too small for two disjoint sides
  idx5 = _hand_index(dict(a=[0.8, 0.7, 0.6, 0.5, 0.4]))
  res5 = search(idx5, (0.50, 0.70), k=4, tol=0.30)
  assert res5['decision'] == 'SHORT' and 'pool' in res5['cells'][1]['reason']

  # reviewer MIN11: the registered +/-0.05 tolerance is load-bearing —
  # a best window missing by 0.07 (< 2x tol) must go SHORT
  idx6 = _hand_index(dict(a=[0.90] * 4 + [0.38] * 8))
  res6 = search(idx6, (0.45, 0.90), k=4, tol=0.05)
  assert res6['decision'] == 'SHORT', res6
  assert res6['cells'][0]['status'] == 'OK'
  assert res6['cells'][1]['status'] == 'SHORT'
  assert 'tol' in res6['cells'][1]['reason']

  # reviewer MIN12: EXACT tie in window deviation -> the EARLIER
  # (higher-occupancy) window is selected. Dyadic occupancies make the
  # tie float-exact: windows (0.6875) and (0.5625) both deviate 0.0625
  # from target 0.625.
  idx7 = _hand_index(dict(a=[0.75, 0.625, 0.5] + [0.25] * 5))
  res7 = search(idx7, (0.25, 0.625), k=2, tol=0.07)
  assert res7['decision'] == 'OK', res7
  assert res7['pairs']['q1']['sides'][1]['occ'] == 0.6875, (
      res7['pairs']['q1']['sides'][1]['occ'])

  # --- Part 2: end-to-end with real chunks through the frozen builder --
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    goal, p2e = f'{tmp}/goal/replay', f'{tmp}/p2e/replay'
    bcr._synth_source(goal, 12000, 0.9, 0.3, seed=1)
    bcr._synth_source(p2e, 12000, 0.1, 1.5, seed=2)
    ns = argparse.Namespace(
        task='dmc_cup_catch', replay=[f'goal={goal}', f'p2e={p2e}'],
        output=f'{tmp}/episodes.json')
    bcr.cmd_index(ns)
    with open(f'{tmp}/episodes.json') as f:
      index = json.load(f)
    # adaptive reachable targets: t_hi = top-K mean; t_lo = the mean of a
    # concrete mid-pool window of the post-removal remainder (both are
    # exactly realizable contiguous windows, so tol always holds)
    occs = sorted((o for _, _, o in _entries(index)), reverse=True)
    k = 20
    t_hi = float(np.mean(occs[:k]))
    rem = occs[k:]
    mid = len(rem) // 2
    t_lo = float(np.mean(rem[mid:mid + k]))
    assert t_hi - t_lo > 0.1, (t_hi, t_lo)
    res = search(index, (round(t_lo, 4), round(t_hi, 4)), k=k, tol=0.05)
    assert res['decision'] == 'OK', res
    pairs_path = f'{tmp}/frew_pairs.json'
    res['index'] = f'{tmp}/episodes.json'
    with open(pairs_path, 'w') as f:
      json.dump(res, f)
    bcr.cmd_build(argparse.Namespace(
        index=f'{tmp}/episodes.json', pairs=pairs_path, which='q1',
        output_root=f'{tmp}/q1f'))
    with open(f'{tmp}/q1f/manifest.json') as f:
      manifest = json.load(f)
    assert manifest['confound_deltas']['n_transitions'] == 0
    for si, rep in enumerate(manifest['sides']):
      assert rep['n_episodes'] == k
      assert abs(rep['occ_recomputed']
                 - res['pairs']['q1']['sides'][si]['occ']) < 0.02, rep
    # ingestion (same contract the fits rely on)
    from embodied.core.replay import Replay
    rep = Replay(length=64, directory=f'{tmp}/replay_out')
    rep.load(directory=manifest['sides'][1]['directory'])
    assert len(rep.items) > 0
    # instrument linkage: rewarded-frame fraction orders with the draw
    from probing import spectral_measure as sm
    f0 = sm.measure(manifest['sides'][0]['directory'], 'cup', 'lo',
                    'q1f_s0')['f_rewarded']
    f1 = sm.measure(manifest['sides'][1]['directory'], 'cup', 'hi',
                    'q1f_s1')['f_rewarded']
    assert f1 > f0, (f0, f1)

    # --- Amendment 2: chunk-availability filter ----------------------
    # All files present -> nothing flagged.
    assert missing_chunk_eids(index) == {}
    # Delete the donor chunk covering a picked hi-side member's start:
    # the field failure (scratch purge between indexing and build).
    table = bcr.episode_by_eid(index)
    victim = res['pairs']['q1']['sides'][1]['members'][0]
    vlabel, ve = table[victim]
    vstream = index['sources'][vlabel]['streams'][ve['stream']]
    vbounds = np.cumsum([0] + vstream['lengths'])
    vci = int(np.searchsorted(vbounds, ve['start'], side='right') - 1)
    os.remove(vstream['files'][vci])
    missing = missing_chunk_eids(index)
    # Flagged set == interval-overlap oracle (independent restatement of
    # the covering rule): exactly the episodes whose span intersects the
    # deleted chunk's frame range, and only that file is reported.
    expected = {eid for eid, (lab, e) in table.items()
                if lab == vlabel and e['stream'] == ve['stream']
                and e['start'] < vbounds[vci + 1] and e['end'] > vbounds[vci]}
    assert victim in expected and set(missing) == expected, \
        (victim, sorted(missing), sorted(expected))
    assert {f for g in missing.values() for f in g} \
        == {vstream['files'][vci]}
    # The unfiltered pairs must reproduce the failure at build time.
    try:
      bcr.cmd_build(argparse.Namespace(
          index=f'{tmp}/episodes.json', pairs=pairs_path, which='q1',
          output_root=f'{tmp}/q1f_broken'))
      raise AssertionError('build must fail on the missing chunk')
    except FileNotFoundError:
      pass
    # Filtered search: picks avoid every flagged episode, is
    # deterministic, and eid numbering is preserved (the index is never
    # rewritten) — proven by building the filtered pair through the
    # frozen builder and matching recomputed occupancies.
    excl = frozenset(missing)
    res_f = search(index, (round(t_lo, 4), round(t_hi, 4)), k=k, tol=0.05,
                   exclude=excl)
    assert res_f['decision'] == 'OK', res_f
    assert res_f['criteria']['n_excluded'] == len(excl)
    for s in res_f['pairs']['q1']['sides']:
      assert not set(s['members']) & excl
    assert search(index, (round(t_lo, 4), round(t_hi, 4)), k=k, tol=0.05,
                  exclude=excl) == res_f
    res_f['index'] = f'{tmp}/episodes.json'
    pairs2 = f'{tmp}/frew_pairs2.json'
    with open(pairs2, 'w') as f:
      json.dump(res_f, f)
    bcr.cmd_build(argparse.Namespace(
        index=f'{tmp}/episodes.json', pairs=pairs2, which='q1',
        output_root=f'{tmp}/q1f2'))
    with open(f'{tmp}/q1f2/manifest.json') as f:
      man2 = json.load(f)
    for si, rep in enumerate(man2['sides']):
      assert abs(rep['occ_recomputed']
                 - res_f['pairs']['q1']['sides'][si]['occ']) < 0.02, rep
  # --- Part 3: breadth feasibility (scan-moments + search-div) --------
  from probing import spectral_measure as sm
  with tempfile.TemporaryDirectory() as tmp:
    ep_len, k = 50, 10
    rng = np.random.default_rng(7)
    u = np.asarray([1.0, 0.0])
    c0 = np.asarray([0.1, 0.1])

    def narrow_ep(gbase, n, jitter):
      def cup_fn(i):
        base = c0 + 0.003 * (gbase + i) * u
        if jitter:
          return base + np.asarray([0.0, 0.02]) * np.sin(gbase + i)
        return base
      return _div_episode(ep_len, n, n, cup_fn, lambda i: np.zeros(4))

    def broad_ep(n):
      cups = rng.normal(0, 0.5, (ep_len, 2))
      vels = rng.normal(0, 0.5, (ep_len, 4))
      return _div_episode(ep_len, n, n, lambda i: cups[i],
                          lambda i: vels[i])

    # narrow: 10 compact colinear eps at occ .56 (greedy magnets), then
    # 10 slightly-jittered colinear eps at occ .44 (repair material)
    nar = [narrow_ep(60 * i, 28, jitter=False) for i in range(10)] + \
          [narrow_ep(600 + 60 * i, 22, jitter=True) for i in range(10)]
    brd = [broad_ep(28 if i % 2 == 0 else 22) for i in range(20)]
    dis = [broad_ep(45) for _ in range(4)]              # occ .9: corridor
    lrw = [_div_episode(ep_len, 25, 2,                  # occ .5, n_rew 2
                        lambda i: c0 + rng.normal(0, 0.5, 2),
                        lambda i: rng.normal(0, 0.5, 4))
           for _ in range(2)]
    for label, eps in dict(nar=nar, brd=brd, dis=dis, lrw=lrw).items():
      _write_div_stream(f'{tmp}/{label}/replay', eps)
    bcr.cmd_index(argparse.Namespace(
        task='dmc_cup_catch',
        replay=[f'{lab}={tmp}/{lab}/replay' for lab in
                ('nar', 'brd', 'dis', 'lrw')],
        output=f'{tmp}/episodes.json'))
    with open(f'{tmp}/episodes.json') as f:
      index = json.load(f)
    table = bcr.episode_by_eid(index)
    assert len(table) == 46

    # scan: every available episode gets full-span moments; the low-rew
    # episodes are scanned (drop happens at search eligibility)
    moments, meta = scan_moments(index)
    assert meta['n_unavailable'] == 0 and meta['n_scanned'] == 46
    lab_of = {eid: lab for eid, (lab, e) in table.items()}
    occ_of = {eid: float(e['occ']) for eid, (lab, e) in table.items()}
    lrw_eids = [e for e, l in lab_of.items() if l == 'lrw']
    assert all(moments[e]['n_rew'] == 2 for e in lrw_eids)
    assert all(abs(occ_of[e] - 0.5) < 1e-6 for e in lrw_eids)

    # cache round-trip is exact
    save_moments(f'{tmp}/mom.npz', moments, meta)
    m2, meta2 = load_moments(f'{tmp}/mom.npz')
    assert set(m2) == set(moments) and meta2['dim'] == meta['dim']
    for e in moments:
      assert m2[e]['n_rew'] == moments[e]['n_rew']
      assert np.array_equal(m2[e]['s'], moments[e]['s'])
      assert np.array_equal(m2[e]['S'], moments[e]['S'])

    # exactness: pooled-moment PR == participation_ratio of the
    # materialized rewarded frames (same transform), any subset
    nar_ids = sorted(e for e, l in lab_of.items() if l == 'nar')[:6]
    keys = meta['obs_keys']
    mats = []
    for eid in nar_ids + sorted(e for e, l in lab_of.items()
                                if l == 'brd')[:4]:
      lab, e = table[eid]
      stream = index['sources'][lab]['streams'][e['stream']]
      d = bcr.load_span(stream, e['start'], e['end'],
                        keys=list(keys) + ['reward'])
      fr = sm.symlog(sm.frames_of({kk: d[kk] for kk in keys}, keys))
      mats.append(fr[np.asarray(d['reward']).reshape(-1) > 0])
    subset = nar_ids + sorted(e for e, l in lab_of.items()
                              if l == 'brd')[:4]
    n = sum(moments[e]['n_rew'] for e in subset)
    s = np.sum([moments[e]['s'] for e in subset], 0)
    m2s = np.sum([moments[e]['S'] for e in subset], 0)
    direct = sm.participation_ratio(np.concatenate(mats, 0))
    assert abs(cell_pr(n, s, m2s) - direct) < 1e-8, \
        (cell_pr(n, s, m2s), direct)

    # search: lo-div pure narrow, hi-div pure broad, both occupancy-
    # matched at 0.50 (forcing the repair pass on the lo cell: the ten
    # greedy-preferred compact episodes all sit at occ .56), distractor
    # (corridor) and low-rew (min_rew) episodes never picked
    res = search_div(index, moments, meta, k=k, target=0.50, tol=0.02,
                     corridor=0.15, min_rew=MIN_REW_EPISODE)
    assert res['decision'] == 'OK', res
    s0, s1 = res['pairs']['q1']['sides']
    assert all(lab_of[m] == 'nar' for m in s0['members']), s0['members']
    assert all(lab_of[m] == 'brd' for m in s1['members']), s1['members']
    assert res['div']['separation'] > 3.0, res['div']
    assert abs(s0['occ'] - 0.50) <= 0.02 and abs(s1['occ'] - 0.50) <= 0.02
    assert s0['repair_swaps'] >= 1, s0
    occs0 = {round(occ_of[m], 2) for m in s0['members']}
    assert occs0 == {0.56, 0.44}, occs0        # repair mixed occ classes
    assert not set(s0['members']) & set(s1['members'])
    for cell in (s0, s1):
      for m in cell['members']:
        assert lab_of[m] not in ('dis', 'lrw')
    # criteria record the drops
    drops = res['criteria']['drops']
    assert drops['n_low_rew'] == 2 and drops['n_out_corridor'] == 4, drops
    # determinism
    res_b = search_div(index, moments, meta, k=k, target=0.50, tol=0.02,
                       corridor=0.15, min_rew=MIN_REW_EPISODE)
    assert res_b == res
    # SHORT on an impossible K
    res_s = search_div(index, moments, meta, k=25, target=0.50, tol=0.02,
                       corridor=0.15, min_rew=MIN_REW_EPISODE)
    assert res_s['decision'] == 'SHORT' and res_s['pairs']['q1'] is None

    # builder e2e: the frozen builder materializes the div pair; the
    # frozen spectral instrument reproduces the searched diversity_pr on
    # the BUILT buffers (the binding measurement path of the wave)
    res['index'] = f'{tmp}/episodes.json'
    with open(f'{tmp}/div_pairs.json', 'w') as f:
      json.dump(res, f)
    bcr.cmd_build(argparse.Namespace(
        index=f'{tmp}/episodes.json', pairs=f'{tmp}/div_pairs.json',
        which='q1', output_root=f'{tmp}/q1d'))
    with open(f'{tmp}/q1d/manifest.json') as f:
      man = json.load(f)
    for si, rep in enumerate(man['sides']):
      assert rep['n_episodes'] == k
      assert abs(rep['occ_recomputed']
                 - res['pairs']['q1']['sides'][si]['occ']) < 0.02, rep
    d0 = sm.measure(man['sides'][0]['directory'], 'cup', 'lo', 'q1d_s0')
    d1 = sm.measure(man['sides'][1]['directory'], 'cup', 'hi', 'q1d_s1')
    assert abs(d0['diversity_pr'] - res['div']['pr_lo']) < 1e-6, \
        (d0['diversity_pr'], res['div']['pr_lo'])
    assert abs(d1['diversity_pr'] - res['div']['pr_hi']) < 1e-6, \
        (d1['diversity_pr'], res['div']['pr_hi'])
    assert d1['diversity_pr'] - d0['diversity_pr'] > 3.0

    # availability: delete one broad-stream chunk -> its episodes drop
    # from the scan (recorded), the search never touches them
    bstream = index['sources']['brd']['streams'][0]
    bbounds = np.cumsum([0] + bstream['lengths'])
    os.remove(bstream['files'][2])
    mom_a, meta_a = scan_moments(index)
    hit = {eid for eid, (lab, e) in table.items()
           if lab == 'brd' and e['start'] < bbounds[3]
           and e['end'] > bbounds[2]}
    assert set(meta_a['unavailable_eids']) == {int(h) for h in hit}
    assert meta_a['missing_files'] == [bstream['files'][2]]
    assert meta_a['n_scanned'] == 46 - len(hit)
    res_a = search_div(index, mom_a, meta_a, k=k, target=0.50, tol=0.02,
                       corridor=0.15, min_rew=MIN_REW_EPISODE)
    assert res_a['decision'] == 'OK'
    for side_ in res_a['pairs']['q1']['sides']:
      assert not set(side_['members']) & hit

  print('selfcheck PASS: window rule optimal + deterministic, sides '
        'disjoint and ordered, fallback/SHORT branches trip (incl. '
        'post-removal starvation and small pool), e2e: frozen builder '
        'materializes the pairs json, recomputed occupancies match the '
        'search, Replay ingests, instrument f_rewarded orders with the '
        'curated occupancy; availability filter: flagged set matches the '
        'interval-overlap oracle, unfiltered build reproduces '
        'FileNotFoundError, filtered pair avoids flagged eids, builds '
        'clean, and preserves eid numbering; div feasibility: moment '
        'scan exact vs materialized frames (1e-8), cache round-trip '
        'exact, lo/hi cells recover the planted narrow/broad families '
        'with occupancy repair forced and mixed classes, corridor + '
        'min_rew drops recorded and honored, deterministic, SHORT '
        'branch trips, frozen builder + frozen spectral instrument '
        'reproduce the searched diversity_pr on the built buffers '
        '(1e-6), chunk-availability drops recorded and avoided')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('cmd', nargs='?',
                  choices=('search-frew', 'scan-moments', 'search-div'))
  ap.add_argument('--index')
  ap.add_argument('--n_episodes', type=int, default=200)
  ap.add_argument('--targets', nargs=2, type=float, default=(0.60, 0.80))
  ap.add_argument('--tol', type=float, default=0.05)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--boot', type=int, default=200)
  ap.add_argument('--require_chunks', action='store_true')
  ap.add_argument('--moments')
  ap.add_argument('--div_target', type=float, default=0.3232)
  ap.add_argument('--div_tol', type=float, default=DIV_TOL)
  ap.add_argument('--div_corridor', type=float, default=DIV_CORRIDOR)
  ap.add_argument('--min_rew', type=int, default=MIN_REW_EPISODE)
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
    return
  assert args.cmd and args.index and args.output
  if args.cmd == 'search-frew':
    cmd_search(args)
  elif args.cmd == 'scan-moments':
    cmd_scan(args)
  else:
    cmd_search_div(args)


if __name__ == '__main__':
  main()
