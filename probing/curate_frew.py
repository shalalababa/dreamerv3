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

Usage:
  python -m probing.curate_frew search-frew \
      --index $RUNROOT/axis1_finger/episodes.json \
      --n_episodes 200 --targets 0.41 0.80 --tol 0.05 \
      --require_chunks \
      --output $RUNROOT/axis1_finger/frew_pairs.json
  python -m probing.curate_frew --selfcheck
"""

import argparse
import collections
import json
import os

import numpy as np

from probing import build_controlled_replay as bcr

FALLBACK_MIN = 0.60


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
  print('selfcheck PASS: window rule optimal + deterministic, sides '
        'disjoint and ordered, fallback/SHORT branches trip (incl. '
        'post-removal starvation and small pool), e2e: frozen builder '
        'materializes the pairs json, recomputed occupancies match the '
        'search, Replay ingests, instrument f_rewarded orders with the '
        'curated occupancy; availability filter: flagged set matches the '
        'interval-overlap oracle, unfiltered build reproduces '
        'FileNotFoundError, filtered pair avoids flagged eids, builds '
        'clean, and preserves eid numbering')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('cmd', nargs='?', choices=('search-frew',))
  ap.add_argument('--index')
  ap.add_argument('--n_episodes', type=int, default=200)
  ap.add_argument('--targets', nargs=2, type=float, default=(0.60, 0.80))
  ap.add_argument('--tol', type=float, default=0.05)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--boot', type=int, default=200)
  ap.add_argument('--require_chunks', action='store_true')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
    return
  assert args.cmd == 'search-frew' and args.index and args.output
  cmd_search(args)


if __name__ == '__main__':
  main()
