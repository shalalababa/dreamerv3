"""Frozen reader — anti-harvest mechanism wave (TM2 W2 passes).

Registration: PREREG_antiharvest_mech_20260811.md (rules verbatim).
ONE execution:

  python -m analysis.w2_mech_read --labels <dir> --output <dir>
Selfcheck: python -m analysis.w2_mech_read --selfcheck
"""

import argparse
import glob
import json
import math
import os
import re

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
EXPECT = dict(repeats=8, states=200, actions=8, env_seed=20260811)
ITERS = (1.0, 6.0, 12.0)
FILE_RE = re.compile(
    r'^w2tm2_(?P<dom>cup|finger)_(?P<cell>e1|e4)_seed(?P<seed>5[1-8])'
    r'_late\.npz$')
DUP_RE = re.compile(r'^w2tm2dup_(?P<dom>cup|finger)_')


def _erfinv(x):
  a = 0.147
  ln = math.log(1.0 - x * x)
  t1 = 2.0 / (math.pi * a) + ln / 2.0
  return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)


def _ndtri(p):
  return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def bca_cluster(vals, rng_seed=RNG_SEED, b=B_BOOT):
  vals = np.asarray(vals, float)
  n = len(vals)
  rng = np.random.default_rng(rng_seed)
  boots = vals[rng.integers(0, n, (b, n))].mean(1)
  theta = float(vals.mean())
  prop = float(np.mean(boots < theta))
  prop = min(max(prop, 1.0 / (b + 1)), 1.0 - 1.0 / (b + 1))
  z0 = _ndtri(prop)
  jack = np.asarray([np.delete(vals, i).mean() for i in range(n)])
  jm = jack.mean()
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.0
  def q(alpha_pt):
    z = _ndtri(alpha_pt)
    adj = z0 + (z0 + z) / (1.0 - acc * (z0 + z))
    p = 0.5 * (1.0 + math.erf(adj / math.sqrt(2.0)))
    return float(np.percentile(boots, 100.0 * min(max(p, 0.0), 1.0)))
  return theta, (q(0.025), q(0.975))


def signflip_p(vals, rng_seed=RNG_SEED, b=B_BOOT):
  vals = np.asarray(vals, float)
  rng = np.random.default_rng(rng_seed)
  obs = abs(vals.mean())
  flips = rng.choice([-1.0, 1.0], size=(b, len(vals)))
  null = np.abs((flips * vals[None]).mean(1))
  return float((np.sum(null >= obs - 1e-15) + 1) / (b + 1))


def _midranks(x):
  x = np.asarray(x, float)
  order = np.argsort(x, kind='mergesort')
  ranks = np.empty(len(x))
  i = 0
  xs = x[order]
  while i < len(x):
    j = i
    while j + 1 < len(x) and xs[j + 1] == xs[i]:
      j += 1
    ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
    i = j + 1
  return ranks


def spearman(a, b):
  ra, rb = _midranks(a), _midranks(b)
  sa, sb = ra.std(), rb.std()
  if sa < 1e-12 or sb < 1e-12:
    return float('nan')
  return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def _zs(x):
  x = np.asarray(x, float)
  sd = x.std()
  return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def cell_stats(d):
  """d: dict of arrays for one cell. Returns per-cell means."""
  g_cand = d['g_all_rep'].mean(1)                  # (S, M)
  gbar = g_cand.mean(1)
  g = {k: d[f'g_{k}_rep'].mean(1)
       for k in ('warm', 'cold', 'plow', 'phigh')}  # (S,)
  q_cand = d['qfull'].mean(1)                      # (S, M)
  q_warm = d['q_warm'].mean(1)                     # (S,)
  S = len(gbar)
  harv, curse, slope, rho = [], [], [], []
  lx = np.log(np.asarray(ITERS))
  lx = lx - lx.mean()
  for s in range(S):
    harv.append(g['warm'][s] - gbar[s])
    acts_q = np.concatenate([q_cand[s], [q_warm[s]]])
    acts_g = np.concatenate([g_cand[s], [g['warm'][s]]])
    # Review B2: a state whose 9 ground-truth returns are (near-)tied
    # carries no truth content — z_g collapses to zeros and the model-
    # favored warm action would contribute a SYSTEMATIC positive with
    # zero evidence. Registered convention (W1 batch-review B2): such
    # states drop out of the curse statistic.
    if acts_g.std() > 1e-12 and acts_q.std() > 1e-12:
      gap = _zs(acts_q) - _zs(acts_g)
      curse.append(gap[-1] - gap[:-1].mean())
    ys = np.asarray([g['plow'][s], g['cold'][s], g['phigh'][s]])
    slope.append(float(np.sum(lx * (ys - ys.mean())) / np.sum(lx * lx)))
    r = spearman(acts_q, acts_g)
    if not math.isnan(r):
      rho.append(r)
  return dict(
      harv=float(np.mean(harv)),
      curse=(float(np.mean(curse)) if curse else float('nan')),
      n_curse_states=len(curse), n_states=S,
      slope=float(np.mean(slope)),
      d_inertia=float(np.mean(g['warm'] - g['cold'])),
      rho_qg=float(np.mean(rho)) if rho else float('nan'),
      g_plow=float(np.mean(g['plow'])), g_cold=float(np.mean(g['cold'])),
      g_phigh=float(np.mean(g['phigh'])),
      g_mode=float(np.mean(g_cand[:, 0])))


def dup_gate(d):
  g = d['g_all_rep']
  return bool(np.all(g == g[:, :, :1]))


def analyse(cells, dup_ok):
  assert dup_ok, 'duplicate-null gate violated: INSTRUMENT-INVALID'
  res = {'n_cells': len(cells)}
  def pooled(key, subset=None):
    vals = [c[key] for c in (subset if subset is not None else cells)]
    mean, ci = bca_cluster(vals)
    return dict(mean=mean, ci=list(ci), perm_p=signflip_p(vals))
  g0 = pooled('harv')
  g0_pass = g0['ci'][1] < 0.0 and g0['perm_p'] < 0.05
  g0_reversed = g0['ci'][0] > 0.0 and g0['perm_p'] < 0.05
  res['g0_replication'] = dict(**g0, gate_pass=bool(g0_pass),
                               sign_reversed=bool(g0_reversed))
  # Review B2: P-N1 pools only cells with >= 1 retained state; registered
  # coverage floor = 25% of all states pooled, else INSTRUMENT-LIMITED.
  n1_cells = [c for c in cells if c['n_curse_states'] > 0]
  coverage = (sum(c['n_curse_states'] for c in cells)
              / max(1, sum(c['n_states'] for c in cells)))
  res['p_n1_coverage'] = dict(frac=float(coverage),
                              n_cells_retained=len(n1_cells))
  if coverage < 0.25 or not n1_cells:
    n1 = dict(mean=None, ci=[None, None], perm_p=None)
    n1_fire = False
    res['p_n1_curse'] = dict(**n1, fires=False,
                             status='INSTRUMENT-LIMITED: ground-truth '
                                    'spread below the registered '
                                    'coverage floor')
  else:
    n1 = pooled('curse', n1_cells)
    n1_fire = n1['ci'][0] > 0.0 and n1['perm_p'] < 0.05
    res['p_n1_curse'] = dict(**n1, fires=bool(n1_fire))
  n2 = pooled('slope')
  n2_fire = n2['ci'][1] < 0.0 and n2['perm_p'] < 0.05
  res['p_n2_pressure'] = dict(**n2, fires=bool(n2_fire))
  n3 = pooled('d_inertia')
  n3_fire = n3['ci'][1] < 0.0 and n3['perm_p'] < 0.05
  res['p_n3_inertia'] = dict(**n3, fires=bool(n3_fire))
  if g0_reversed:
    v = ('REPLICATION-SIGN-REVERSED: in-wave harvest significantly '
         'POSITIVE; mechanism legs reported with ZERO decision weight')
  elif not g0_pass:
    v = ('REPLICATION-FAILED: in-wave anti-harvest absent at this '
         'power; mechanism legs reported with ZERO decision weight')
  elif n1_fire and n2_fire:
    v = ("OPTIMIZERS-CURSE (pressure-causal)"
         + (" + inertia noted" if n3_fire else ""))
  elif n3_fire and not n1_fire and not n2_fire:
    v = 'INERTIA-DRIVEN'
  elif n3_fire and (n1_fire or n2_fire):
    v = 'MIXED'
  elif n1_fire or n2_fire:
    v = ('PARTIAL-CURSE: '
         + ('value-gap leg only' if n1_fire else 'pressure leg only'))
  else:
    v = ('UNRESOLVED: anti-harvest replicates but no mechanism leg '
         'fires - no mechanism wording licensed')
  res['verdict'] = v
  res['descriptives'] = {
      'rho_qg_pooled': float(np.nanmean([c['rho_qg'] for c in cells])),
      'pressure_means': {k: float(np.mean([c[k] for c in cells]))
                         for k in ('g_mode', 'g_plow', 'g_cold',
                                   'g_phigh')},
      'per_domain': {
          dom: {k: (float(np.mean(v_)) if (v_ := [c[k] for c in cells
                                                  if c['dom'] == dom])
                    else None)
                for k in ('harv', 'curse', 'slope', 'd_inertia')}
          for dom in ('cup', 'finger')},
  }
  return res


def load_cells(labels_dir):
  cells, dup_flags = [], []
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z['meta']))
    dm = DUP_RE.match(name)
    if dm:
      assert bool(np.asarray(z['dup_cand']).all()), name
      assert str(meta.get('labeler_version', '')).endswith('_w2'), name
      assert int(meta['env_seed']) == EXPECT['env_seed'], name
      dup_flags.append((dm.group('dom'),
                        dup_gate({k: np.asarray(z[k]) for k in
                                  ('g_all_rep',)})))
      continue
    m = FILE_RE.match(name)
    if not m:
      continue
    assert str(meta.get('labeler_version', '')).endswith('_w2'), name
    assert int(meta['w2']['repeats']) == EXPECT['repeats'], name
    assert int(meta['env_seed']) == EXPECT['env_seed'], name
    assert int(meta['states']) == EXPECT['states'], name
    assert meta['w2_iters'] == {'low': 1, 'default': 6, 'high': 12}, name
    # Review M5: filename <-> run_id identity + dial pins (W1 M4 pattern)
    rid = str(np.asarray(z['run_id']).reshape(-1)[0])
    assert (m.group('dom') in rid and m.group('cell') in rid
            and f"seed{m.group('seed')}" in rid), (name, rid)
    assert int(meta['horizon']) == 100 and int(meta['label_every']) == 25 \
        and int(meta['actions']) == 8, name
    # Review M6: realized-knob witness + planner pins
    itr = np.asarray(z['iters_realized'], int)
    assert (itr == np.asarray([1, 6, 12])).all(), (name, 'knob witness')
    assert bool(meta['planner']['mpc']) \
        and int(meta['planner']['iterations']) == 6, name
    d = {k: np.asarray(z[k], float) for k in
         ('g_all_rep', 'g_warm_rep', 'g_cold_rep', 'g_plow_rep',
          'g_phigh_rep', 'qfull', 'q_warm')}
    assert d['g_all_rep'].shape == (EXPECT['states'], EXPECT['repeats'],
                                    EXPECT['actions']), name
    assert not bool(np.asarray(z['dup_cand']).any()), name
    for k, v in d.items():
      assert np.isfinite(v).all(), (name, k)
    c = cell_stats(d)
    c['dom'] = m.group('dom')
    c['file'] = name
    cells.append(c)
  assert len(cells) == 32, (len(cells), 'expected 32 cells')
  assert len({c['file'] for c in cells}) == 32
  assert sorted(d for d, _ in dup_flags) == ['cup', 'finger'], dup_flags
  return cells, all(ok for _, ok in dup_flags)


# --------------------------------------------------------------------------

def _fake_cell(rng, harv=-0.3, curse=0.0, slope=0.0, inertia=0.0,
               dom='finger'):
  S, R, M, K = 40, 8, 8, 5
  true_g = rng.normal(1.0, 0.3, (S, M))
  d = {}
  d['g_all_rep'] = true_g[:, None, :] + rng.normal(0, 0.1, (S, R, M))
  gbar = true_g.mean(1)
  g_warm = gbar + harv + rng.normal(0, 0.05, S)
  g_cold = g_warm + inertia + rng.normal(0, 0.05, S)   # warm worse by `inertia`
  lx = np.log(np.asarray(ITERS))
  d['g_warm_rep'] = g_warm[:, None] + rng.normal(0, 0.05, (S, R))
  d['g_cold_rep'] = g_cold[:, None] + rng.normal(0, 0.05, (S, R))
  d['g_plow_rep'] = (g_cold + slope * (lx[0] - lx[1]))[:, None] \
      + rng.normal(0, 0.05, (S, R))
  d['g_phigh_rep'] = (g_cold + slope * (lx[2] - lx[1]))[:, None] \
      + rng.normal(0, 0.05, (S, R))
  # q: candidates track truth; warm action overvalued iff curse
  d['qfull'] = true_g[:, None, :] + rng.normal(0, 0.05, (S, K, M))
  q_warm = (true_g.max(1) + curse) if curse else \
      (g_warm + rng.normal(0, 0.05, S))
  d['q_warm'] = q_warm[:, None] + rng.normal(0, 0.02, (S, K))
  c = cell_stats(d)
  c['dom'] = dom
  return c


def selfcheck():
  mk = lambda i, **kw: _fake_cell(np.random.default_rng(i), **kw)
  # curse + pressure -> OPTIMIZERS-CURSE
  cells = [mk(i, curse=2.0, slope=-0.2) for i in range(32)]
  r = analyse(cells, True)
  assert r['g0_replication']['gate_pass']
  assert r['verdict'].startswith('OPTIMIZERS-CURSE'), r['verdict']
  # inertia only
  cells = [mk(100 + i, inertia=0.3) for i in range(32)]
  r = analyse(cells, True)
  assert r['verdict'] == 'INERTIA-DRIVEN', r['verdict']
  # mixed
  cells = [mk(200 + i, curse=2.0, slope=-0.2, inertia=0.3)
           for i in range(32)]
  r = analyse(cells, True)
  assert 'CURSE' in r['verdict'] and 'inertia noted' in r['verdict'], \
      r['verdict']
  # partial (pressure only)
  cells = [mk(300 + i, slope=-0.2) for i in range(32)]
  r = analyse(cells, True)
  assert r['verdict'].startswith('PARTIAL-CURSE: pressure'), r['verdict']
  # unresolved
  cells = [mk(400 + i) for i in range(32)]
  r = analyse(cells, True)
  assert r['verdict'].startswith('UNRESOLVED'), r['verdict']
  # replication fail
  cells = [mk(500 + i, harv=0.0, curse=2.0, slope=-0.2)
           for i in range(32)]
  r = analyse(cells, True)
  assert r['verdict'].startswith('REPLICATION-FAILED'), r['verdict']
  # dup gate refusal
  try:
    analyse([mk(600)], False)
    raise SystemExit('selfcheck FAIL: dup gate not enforced')
  except AssertionError:
    pass
  # B2 leg: tied ground truth everywhere -> INSTRUMENT-LIMITED, no fire
  flat_cells = []
  for i in range(32):
    c = mk(700 + i, harv=-0.3)
    c['curse'] = float('nan')
    c['n_curse_states'] = 0
    flat_cells.append(c)
  r = analyse(flat_cells, True)
  assert r['p_n1_curse']['fires'] is False
  assert 'INSTRUMENT-LIMITED' in r['p_n1_curse'].get('status', ''),       r['p_n1_curse']
  # M5 leg: load_cells round-trip via temp npz files + pin trips
  import tempfile
  def _write_cell(d_, name, rid, horizon=100, env_seed=20260811,
                  states=2, dup=False, iters=(1, 6, 12)):
    S, R, M, K = states, 8, 8, 5
    rng9 = np.random.default_rng(1)
    meta = dict(labeler_version='x_w2',
                w2=dict(repeats=8, dup_cand=dup, seed_base=env_seed),
                env_seed=env_seed, states=states, horizon=horizon,
                label_every=25, actions=8,
                w2_iters={'low': 1, 'default': 6, 'high': 12},
                planner={'mpc': True, 'iterations': 6})
    g_all = rng9.normal(1, .1, (S, R, M))
    if dup:
      g_all = np.repeat(g_all[:, :, :1], M, 2)
    np.savez(os.path.join(d_, name),
             meta=json.dumps(meta), run_id=np.array([rid] * S),
             g_all_rep=g_all,
             g_warm_rep=rng9.normal(0, .1, (S, R)),
             g_cold_rep=rng9.normal(0, .1, (S, R)),
             g_plow_rep=rng9.normal(0, .1, (S, R)),
             g_phigh_rep=rng9.normal(0, .1, (S, R)),
             qfull=rng9.normal(0, 1, (S, K, M)),
             q_warm=rng9.normal(0, 1, (S, K)),
             dup_cand=np.array([dup] * S),
             iters_realized=np.tile(np.asarray(iters, np.int64), (S, 1)))
  with tempfile.TemporaryDirectory() as td:
    for dom in ('cup', 'finger'):
      for cell in ('e1', 'e4'):
        for sd in range(51, 59):
          _write_cell(td, f'w2tm2_{dom}_{cell}_seed{sd}_late.npz',
                      f'tm2r3_{dom}_{cell}_seed{sd}')
      _write_cell(td, f'w2tm2dup_{dom}_e1_seed51_late.npz',
                  f'tm2r3_{dom}_e1_seed51', dup=True)
    old_states = EXPECT['states']
    EXPECT['states'] = 2
    try:
      cells_l, dup_ok = load_cells(td)
      assert len(cells_l) == 32 and dup_ok
      # run_id mismatch trips (M5)
      _write_cell(td, 'w2tm2_cup_e1_seed51_late.npz',
                  'tm2r3_finger_e4_seed58')
      try:
        load_cells(td)
        raise SystemExit('selfcheck FAIL: run_id pin not enforced')
      except AssertionError:
        pass
      _write_cell(td, 'w2tm2_cup_e1_seed51_late.npz',
                  'tm2r3_cup_e1_seed51')
      # knob-witness trip (M6)
      _write_cell(td, 'w2tm2_cup_e1_seed52_late.npz',
                  'tm2r3_cup_e1_seed52', iters=(1, 12, 12))
      try:
        load_cells(td)
        raise SystemExit('selfcheck FAIL: knob witness not enforced')
      except AssertionError:
        pass
    finally:
      EXPECT['states'] = old_states
  print('w2_mech_read selfcheck PASS (all verdict branches planted + '
        'recovered: curse/inertia/mixed/partial/unresolved/'
        'replication-failed; dup-gate refusal)')


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--labels')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.labels and args.output
  cells, dup_ok = load_cells(args.labels)
  res = analyse(cells, dup_ok)
  res['prereg'] = 'PREREG_antiharvest_mech_20260811.md'
  res['files'] = [c['file'] for c in cells]
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({k: res[k] for k in
                    ('verdict', 'g0_replication', 'p_n1_curse',
                     'p_n2_pressure', 'p_n3_inertia')}, indent=1))


if __name__ == '__main__':
  main()
