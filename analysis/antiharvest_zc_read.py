"""Frozen reader — anti-harvest zero-compute stage (TM2, W1 labels).

Registration: PREREG_antiharvest_zc_20260811.md (rules verbatim;
value-awareness disclosed there). ONE execution:

  python -m analysis.antiharvest_zc_read \
      --labels local_results/w1_labels_20260811_090727/w1_labels \
      --output artifacts/antiharvest_zc_read_<date>
Selfcheck: python -m analysis.antiharvest_zc_read --selfcheck
"""

import argparse
import glob
import json
import math
import os
import re
import tempfile

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
LAMBDA_PRIMARY = 1.0
LAMBDAS_DESC = (0.5, 2.0)
EXPECT = dict(repeats=8, states=200, actions=8, env_seed=20260809)
FILE_RE = re.compile(
    r'^w1tm2_(?P<dom>cup|finger)_(?P<cell>e1|e4)_seed(?P<seed>5[1-8])'
    r'_late\.npz$')
DUP_RE = re.compile(r'^w1tm2dup_')


def _erfinv(x):
  a = 0.147
  ln = math.log(1.0 - x * x)
  t1 = 2.0 / (math.pi * a) + ln / 2.0
  return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)


def _ndtri(p):
  return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def bca_cluster(vals, rng_seed=RNG_SEED, b=B_BOOT):
  """BCa 95% for the mean of per-cluster values (one value per cell)."""
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


def cell_stats(g_all_rep, g_plan_rep, qfull):
  """Per-cell means of the per-state estimands. Shapes: (S, R, M),
  (S, R), (S, K, M). Amendment 1: the selection layer (q, qstd, sel_q,
  sel_p) runs in qfull's OWN dtype (storage float32 at execution, the
  precision of the labeler's plugin_choice); g-side estimand arithmetic
  is float64 from load_cells."""
  g_cand = g_all_rep.mean(1)                       # (S, M)
  g_plan = g_plan_rep.mean(1)                      # (S,)
  gbar = g_cand.mean(1)                            # (S,)
  q = qfull.mean(1)                                # (S, M)
  qstd = qfull.std(1)          # (S, M); ddof=0 (registered)
  d_mode = g_plan - g_cand[:, 0]
  sel_q = q.argmax(1)
  s_idx = np.arange(len(g_plan))
  d_qsel = g_cand[s_idx, sel_q] - gbar
  out = dict(d_mode=float(d_mode.mean()), d_qsel=float(d_qsel.mean()))
  for lam in (LAMBDA_PRIMARY,) + LAMBDAS_DESC:
    sel_p = (q - lam * qstd).argmax(1)
    out[f'd_rep_lam{lam:g}'] = float(
        (g_cand[s_idx, sel_p] - g_cand[s_idx, sel_q]).mean())
    if lam == LAMBDA_PRIMARY:
      out['d_rep_vs_plan'] = float(
          (g_cand[s_idx, sel_p] - g_plan).mean())
  rhos = [spearman(q[s], g_cand[s]) for s in range(len(g_plan))]
  rhos = [r for r in rhos if not math.isnan(r)]
  out['rho_q_g'] = float(np.mean(rhos)) if rhos else float('nan')
  return out


def grade(name, mean, ci, p, fire_map):
  assert not (math.isnan(mean) or math.isnan(ci[0]) or math.isnan(ci[1])
              or math.isnan(p)), (name, 'non-finite estimand')
  fired = None
  if ci[1] < 0.0 and p < 0.05:
    fired = fire_map.get('neg')
  elif ci[0] > 0.0 and p < 0.05:
    fired = fire_map.get('pos')
  return fired or fire_map['null']


def analyse(cells):
  """cells: list of dicts from cell_stats + dom key."""
  res = {'n_cells': len(cells)}
  def pooled(key):
    vals = [c[key] for c in cells]
    mean, ci = bca_cluster(vals)
    p = signflip_p(vals)
    return dict(mean=mean, ci=list(ci), perm_p=p)
  z1 = pooled('d_mode')
  res['p_z1_prior_mode'] = z1
  res['p_z1_verdict'] = grade('z1', z1['mean'], z1['ci'], z1['perm_p'], {
      'neg': 'PLANNER-BELOW-MODE: optimization strictly harmful vs the '
             'un-optimized prior mode',
      'pos': 'MODE-BELOW-PLANNER: anti-harvest is sample-driven '
             '(stochastic-exploration account)',
      'null': 'INDETERMINATE'})
  z2 = pooled('d_qsel')
  res['p_z2_q_selection'] = z2
  res['p_z2_verdict'] = grade('z2', z2['mean'], z2['ci'], z2['perm_p'], {
      'neg': 'CURSE-AT-Q: the learned value misranks its own prior '
             'candidates - value-error suffices for anti-harvest',
      'pos': 'Q-HARVESTS: the value head ranks candidates usefully - '
             'the deficit is planner-search-specific',
      'null': 'NULL (uninformative)'})
  z3 = pooled(f'd_rep_lam{LAMBDA_PRIMARY:g}')
  res['p_z3_penalty_repair'] = z3
  res['p_z3_verdict'] = grade('z3', z3['mean'], z3['ci'], z3['perm_p'], {
      'pos': 'PENALTY-IMPROVES: head disagreement recovers real value',
      'neg': 'PENALTY-HURTS',
      'null': 'NULL'})
  res['descriptives'] = {
      'rho_q_g_pooled': float(np.nanmean([c['rho_q_g']
                                          for c in cells])),
      'rho_cells_finite': int(sum(not math.isnan(c['rho_q_g'])
                                  for c in cells)),
      'd_rep_vs_plan': pooled('d_rep_vs_plan'),
      **{f'd_rep_lam{lam:g}': pooled(f'd_rep_lam{lam:g}')
         for lam in LAMBDAS_DESC},
      'per_domain': {
          dom: {k: (float(np.mean(v)) if (v := [c[k] for c in cells
                                               if c['dom'] == dom])
                    else None)
                for k in ('d_mode', 'd_qsel',
                          f'd_rep_lam{LAMBDA_PRIMARY:g}')}
          for dom in ('cup', 'finger')},
  }
  return res


def load_cells(labels_dir):
  cells = []
  dup_seen = 0
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    if DUP_RE.match(name):
      dup_seen += 1
      continue
    m = FILE_RE.match(name)
    if not m:
      continue
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z['meta']))
    assert str(meta.get('labeler_version', '')).endswith('_w1'), name
    assert int(meta['w1']['repeats']) == EXPECT['repeats'], name
    assert int(meta['env_seed']) == EXPECT['env_seed'], name
    assert int(meta['states']) == EXPECT['states'], name
    g_all = np.asarray(z['g_all_rep'], float)
    g_plan = np.asarray(z['g_plan_rep'], float)
    # Amendment 1: qfull stays in STORAGE precision (float32) — the
    # labeler's plugin_choice ran on this array; a float64 upcast flips
    # the head-mean argmax on 61/6400 near-tie states and breaks the
    # identity gate below.
    qfull = z['qfull']
    assert g_all.shape == (EXPECT['states'], EXPECT['repeats'],
                           EXPECT['actions']), (name, g_all.shape)
    assert g_plan.shape == (EXPECT['states'], EXPECT['repeats']), name
    assert qfull.shape == (EXPECT['states'], 5, EXPECT['actions']), name
    assert not bool(np.asarray(z['dup_cand']).any()), (name, 'dup file?')
    # Review M8: NaN anywhere fails CLOSED, incl. g_plan
    assert np.isfinite(g_all).all() and np.isfinite(qfull).all() \
        and np.isfinite(g_plan).all(), name
    # Review B3 witness: the labeler's stored plug-in choice m_now IS the
    # Q-argmax (plugin_choice), so P-Z2's selected index must reproduce it
    m_now = np.asarray(z['m_now'], int)
    assert (qfull.mean(1).argmax(1) == m_now).all(), (
        name, 'sel_q != m_now - plugin_choice identity broken')
    # g_all_rep is stored (S, R, M) per the W1 writer
    c = cell_stats(g_all, g_plan, qfull)
    c['dom'] = m.group('dom')
    c['file'] = name
    cells.append(c)
  assert len(cells) == 32, (len(cells), 'expected exactly 32 cells')
  assert dup_seen == 2, (dup_seen, 'expected the 2 dup-gate files '
                         'present and untouched')
  return cells


# --------------------------------------------------------------------------

def _fake_cell(rng, mode_adv=0.0, q_curse=0.0, pen_gain=0.0, dom='finger',
               plan_off=-0.3):
  """Synthetic (S,R,M) arrays with plantable structure."""
  S, R, M, K = 40, 8, 8, 5
  base = rng.normal(1.0, 0.3, (S, M))
  base[:, 0] += mode_adv                      # prior-mode advantage
  true_g = base
  g_all = true_g[:, None, :] + rng.normal(0, 0.15, (S, R, M))
  g_plan = (true_g.mean(1) + plan_off
            + rng.normal(0, 0.15, (S, R)).mean(1))[:, None] \
      + rng.normal(0, 0.1, (S, R))
  # q: correlated with truth unless q_curse, disagreement structured
  noise = rng.normal(0, 0.3, (S, M))
  q_mean = (-true_g if q_curse else true_g) * (1 - 0.0) + noise
  qstd = rng.uniform(0.05, 0.15, (S, M))
  if pen_gain:
    # bad candidates get HIGH q but HIGH disagreement -> penalty fixes
    worst = true_g.argmin(1)
    s_idx = np.arange(S)
    q_mean[s_idx, worst] = q_mean.max(1) + 0.5
    qstd[s_idx, worst] = 3.0
  pattern = (np.arange(K) - (K - 1) / 2.0)
  pattern /= pattern.std()                    # zero-mean, unit-std heads
  qfull = q_mean[:, None, :] + rng.normal(0, 1e-3, (S, K, M)) \
      + pattern[None, :, None] * qstd[:, None, :]
  return dict(**cell_stats(g_all, g_plan, qfull), dom=dom)


def selfcheck():
  rng = np.random.default_rng(0)
  # planted: planner below mode (mode strong, plan offset negative)
  cells = [_fake_cell(np.random.default_rng(i), mode_adv=0.5,
                      plan_off=-0.5, dom='cup' if i % 2 else 'finger')
           for i in range(32)]
  r = analyse(cells)
  assert r['p_z1_verdict'].startswith('PLANNER-BELOW-MODE'), \
      r['p_z1_verdict']
  # planted curse: q anti-correlated with truth
  cells = [_fake_cell(np.random.default_rng(100 + i), q_curse=1.0)
           for i in range(32)]
  r = analyse(cells)
  assert r['p_z2_verdict'].startswith('CURSE-AT-Q'), r['p_z2_verdict']
  # planted harvest: q correlated with truth
  cells = [_fake_cell(np.random.default_rng(200 + i)) for i in range(32)]
  r = analyse(cells)
  assert r['p_z2_verdict'].startswith('Q-HARVESTS'), r['p_z2_verdict']
  # planted penalty repair
  cells = [_fake_cell(np.random.default_rng(300 + i), pen_gain=1.0)
           for i in range(32)]
  r = analyse(cells)
  assert r['p_z3_verdict'].startswith('PENALTY-IMPROVES'), \
      r['p_z3_verdict']
  # null: no structure anywhere -> z1/z3 do not fire spuriously
  cells = []
  for i in range(32):
    g = np.random.default_rng(400 + i)
    S, R, M, K = 40, 8, 8, 5
    g_all = g.normal(1.0, 0.3, (S, 1, M)) + g.normal(0, 0.15, (S, R, M))
    g_plan = g_all.mean(2).mean(1)[:, None] + g.normal(0, 0.1, (S, R))
    qfull = g.normal(0, 1, (S, K, M))
    cells.append(dict(**cell_stats(g_all, g_plan, qfull), dom='cup'))
  r = analyse(cells)
  assert r['p_z1_verdict'] == 'INDETERMINATE', r['p_z1_prior_mode']
  assert r['p_z3_verdict'] == 'NULL', r['p_z3_penalty_repair']
  # Amendment 1 defect-class leg: a float32 near-tie whose float64
  # upcast flips the head-mean argmax; the reader-path expression
  # (storage precision) must reproduce the labeler's m_now exactly,
  # through an npz round-trip.
  rng = np.random.default_rng(7)
  tie_q = None
  for _ in range(20000):
    cand = (60.0 + 1e-5 * rng.standard_normal((6, 5, 8))).astype(
        np.float32)
    if (cand.mean(1).argmax(1)
        != np.asarray(cand, float).mean(1).argmax(1)).any():
      tie_q = cand
      break
  assert tie_q is not None, 'could not construct a near-tie fixture'
  m_planted = tie_q.mean(1).argmax(1)     # storage-precision truth
  with tempfile.TemporaryDirectory() as td:
    tie_path = os.path.join(td, 'tie.npz')
    np.savez(tie_path, qfull=tie_q, m_now=m_planted)
    zz = np.load(tie_path)
    assert zz['qfull'].dtype == np.float32, zz['qfull'].dtype
    assert (zz['qfull'].mean(1).argmax(1) == zz['m_now']).all(), \
        'storage-precision identity must hold after round-trip'
    assert (np.asarray(zz['qfull'], float).mean(1).argmax(1)
            != zz['m_now']).any(), \
        'fixture must exhibit the float64 flip (defect class)'
  print('antiharvest_zc_read selfcheck PASS (planted below-mode / '
        'curse-at-q / q-harvests / penalty-repair recovered; null '
        'no-fires; conjunction CI-and-p enforced by grade(); '
        'float32 near-tie identity leg PASS)')


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
  cells = load_cells(args.labels)
  res = analyse(cells)
  res['prereg'] = 'PREREG_antiharvest_zc_20260811.md'
  res['files'] = [c['file'] for c in cells]
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({k: res[k] for k in
                    ('p_z1_verdict', 'p_z1_prior_mode',
                     'p_z2_verdict', 'p_z2_q_selection',
                     'p_z3_verdict', 'p_z3_penalty_repair')}, indent=1))


if __name__ == '__main__':
  main()
