"""Frozen reader — B1 capacity-descent λ trial.

Registration: PREREG_capdescent_20260811.md (rules verbatim; Part-B
predictions frozen in PREREG_theory_R1_gating_20260811). ONE execution:

  python -m analysis.capdescent_read --auc_csv <auc.csv> \
      --runroot <bundle>/runroot_light --ridge_glob '<...>/ridge_probe_finger_v1.json' \
      --fit_counters <json> --ckpt_steps <json> --output <dir>
Selfcheck: python -m analysis.capdescent_read --selfcheck
"""

import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
FLOOR = 113.682          # randominit executed band edge (rev-3 F15)
APT_LEG_FLOOR = 0.55     # P-B1c informativeness (rev-3 F7; apt@1m=0.401)
SIZES = ('sk3', 'sk1')   # 300k, 100k (descent order)
ARMS = ('task', 'apt')
SIDES = ('0', '1')
SEEDS = (1, 2, 3, 4)
DETER = {'sk3': 256, 'sk1': 128}
MODE_RE = re.compile(r'^ax1(?P<f>f?)(?P<size>sk[13])q1s(?P<side>[01])$')
ALPHA_KEY = 'alpha_0.001'


def wm_name(arm, size, side, seed):
  f = 'f' if arm == 'apt' else ''
  return f'ax1wm_finger_{f}{size}q1s{side}_seed{seed}'


def adapt_name(arm, size, side, seed):
  f = 'f' if arm == 'apt' else ''
  return f'adapt_ax1{f}{size}q1s{side}_finger_seed{seed}_ckpt500000'


def _erfinv(x):
  a = 0.147
  ln = math.log(1.0 - x * x)
  t1 = 2.0 / (math.pi * a) + ln / 2.0
  return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)


def _did_theta(groups):
  return ((np.mean(groups[('task', 'sk1')]) - np.mean(groups[('apt', 'sk1')]))
          - (np.mean(groups[('task', 'sk3')])
             - np.mean(groups[('apt', 'sk3')])))


def did_stats(groups, rng_seed=RNG_SEED, nboot=B_BOOT):
  """DID = [task-apt](sk1) - [task-apt](sk3). Group bootstrap BCa 95%
  (resample runs within each group) + permutation p (size labels
  permuted WITHIN arm). groups: {(arm, size): [values]}."""
  g = {k: np.asarray(v, float) for k, v in groups.items()}
  rng = np.random.default_rng(rng_seed)
  theta = float(_did_theta(g))
  boots = np.empty(nboot)
  for i in range(nboot):
    boots[i] = _did_theta({k: v[rng.integers(0, len(v), len(v))]
                           for k, v in g.items()})
  prop = float(np.mean(boots < theta))
  prop = min(max(prop, 1.0 / (nboot + 1)), 1.0 - 1.0 / (nboot + 1))
  z0 = math.sqrt(2) * _erfinv(2 * prop - 1)
  jack = []
  for k in g:
    for i in range(len(g[k])):
      gj = dict(g)
      gj[k] = np.delete(g[k], i)
      jack.append(_did_theta(gj))
  jack = np.asarray(jack)
  jm = jack.mean()
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.0
  def q(alpha):
    z = math.sqrt(2) * _erfinv(2 * alpha - 1)
    adj = z0 + (z0 + z) / (1.0 - acc * (z0 + z))
    p = 0.5 * (1.0 + math.erf(adj / math.sqrt(2)))
    return float(np.percentile(boots, 100.0 * min(max(p, 0.0), 1.0)))
  obs = abs(theta)
  # reviewer-3 F4: align the size main effect before permuting - the
  # raw within-arm pool is a location mixture under a main effect and
  # the permutation null variance inflates (self-defeating: power
  # 0.69 -> 0.04 at delta=250). theta is INVARIANT to a common sk1
  # shift (it cancels in the DID), so the observed statistic is
  # untouched; only the permutation pool is aligned. Post-fix
  # simulation: type-I 0.050 at every delta, power ~0.74 at DID=150.
  delta = 0.5 * ((np.mean(g[('task', 'sk1')])
                  - np.mean(g[('task', 'sk3')]))
                 + (np.mean(g[('apt', 'sk1')])
                    - np.mean(g[('apt', 'sk3')])))
  ga = {k: (v - delta if k[1] == 'sk1' else v) for k, v in g.items()}
  cnt = 0
  for _ in range(nboot):
    perm = {}
    for arm in ('task', 'apt'):
      pool = np.concatenate([ga[(arm, 'sk1')], ga[(arm, 'sk3')]])
      pool = rng.permutation(pool)
      n1 = len(ga[(arm, 'sk1')])
      perm[(arm, 'sk1')], perm[(arm, 'sk3')] = pool[:n1], pool[n1:]
    if abs(_did_theta(perm)) >= obs - 1e-12:
      cnt += 1
  p = float((cnt + 1) / (nboot + 1))
  return dict(did=theta, ci=[q(0.025), q(0.975)], perm_p=p,
              n={f'{a}_{s}': len(g[(a, s)]) for a in ARMS for s in SIZES})


def two_sample(a, b, rng_seed=RNG_SEED, nboot=B_BOOT):
  """House form: BCa 95% + two-sided permutation (reviewer-3 F8)."""
  a, b = np.asarray(a, float), np.asarray(b, float)
  rng = np.random.default_rng(rng_seed)
  boots = np.array([a[rng.integers(0, len(a), len(a))].mean()
                    - b[rng.integers(0, len(b), len(b))].mean()
                    for _ in range(nboot)])
  diff = float(a.mean() - b.mean())
  prop = float(np.mean(boots < diff))
  prop = min(max(prop, 1.0 / (nboot + 1)), 1.0 - 1.0 / (nboot + 1))
  z0 = math.sqrt(2) * _erfinv(2 * prop - 1)
  jack = [np.delete(a, i).mean() - b.mean() for i in range(len(a))]
  jack += [a.mean() - np.delete(b, j).mean() for j in range(len(b))]
  jack = np.asarray(jack)
  jm = jack.mean()
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.0
  def q(alpha):
    z = math.sqrt(2) * _erfinv(2 * alpha - 1)
    adj = z0 + (z0 + z) / (1.0 - acc * (z0 + z))
    p = 0.5 * (1.0 + math.erf(adj / math.sqrt(2)))
    return float(np.percentile(boots, 100.0 * min(max(p, 0.0), 1.0)))
  pool = np.concatenate([a, b])
  obs = abs(diff)
  cnt = 0
  for _ in range(nboot):
    perm = rng.permutation(pool)
    if abs(perm[:len(a)].mean() - perm[len(a):].mean()) >= obs - 1e-12:
      cnt += 1
  return dict(diff=diff, ci=[q(0.025), q(0.975)],
              perm_p=float((cnt + 1) / (nboot + 1)), n=[len(a), len(b)])


def _load_yaml(path):
  try:
    import yaml
    with open(path) as f:
      return yaml.safe_load(f)
  except ImportError:
    from ruamel.yaml import YAML
    with open(path) as f:
      return YAML(typ='safe').load(f)


def _find_rssm_deters(cfg):
  out = []
  def walk(d):
    if isinstance(d, dict):
      for k, v in d.items():
        if k == 'rssm' and isinstance(v, dict) and 'deter' in v:
          out.append(int(v['deter']))
        walk(v)
  walk(cfg)
  return out


def check_witness(fit_counters, ckpt_steps):
  for arm in ARMS:
    for size in SIZES:
      for side in SIDES:
        for seed in SEEDS:
          n = wm_name(arm, size, side, seed)
          assert n in fit_counters and \
              int(fit_counters[n]['update']) == \
              int(fit_counters[n]['total']), (n, fit_counters.get(n))
          assert n in ckpt_steps and int(ckpt_steps[n]) == 500000, \
              (n, ckpt_steps.get(n))


def check_configs(runroot):
  for arm in ARMS:
    for size in SIZES:
      for side in SIDES:
        for seed in SEEDS:
          wm = wm_name(arm, size, side, seed)
          cfg = _load_yaml(os.path.join(runroot, wm, 'config.yaml'))
          deters = _find_rssm_deters(cfg)
          assert deters and all(d == DETER[size] for d in deters), \
              (wm, 'rssm deter mismatch', deters, DETER[size])
          assert str(cfg['agent']['expl']['mode']) == arm, \
              (wm, cfg['agent']['expl']['mode'])
          ad = adapt_name(arm, size, side, seed)
          acfg = _load_yaml(os.path.join(runroot, ad, 'config.yaml'))
          fc = str(acfg['run']['from_checkpoint'])
          assert wm in fc, (ad, 'from_checkpoint does not name WM', fc)
          assert acfg['agent'].get('frozen_wm') is not False, \
              (ad, 'adapt not frozen')
          # reviewer-3 F5: the readout must be PINNED across sizes
          # (readout64) or the mediator confounds with head capacity
          assert int(acfg['agent']['policy']['units']) == 64, \
              (ad, 'adapt readout not pinned at 64 units',
               acfg['agent'].get('policy'))


def load_auc(path):
  rows, n_eps = {}, {}
  with open(path) as f:
    for r in csv.DictReader(f):
      m = MODE_RE.match(r['mode'])
      if not m:
        continue
      arm = 'apt' if m.group('f') else 'task'
      key = (arm, m.group('size'), m.group('side'), int(r['seed']))
      assert key not in rows, ('duplicate row', key)
      # reviewer-3 F6: the pre-registered adapt QC (standing rule)
      assert str(r['qc_pass']).lower() in ('1', 'true'), \
          (key, 'qc_pass failed')
      rows[key] = float(r['auc100k'])
      n_eps[key] = int(r['n_ep_100k'])
  for arm in ARMS:
    for size in SIZES:
      for side in SIDES:
        for seed in SEEDS:
          assert (arm, size, side, seed) in rows, \
              ('missing adapt row', arm, size, side, seed)
  assert len(rows) == 32, len(rows)
  vals, counts = np.unique(list(n_eps.values()), return_counts=True)
  modal = int(vals[counts.argmax()])
  sub = {k: v for k, v in n_eps.items() if v < modal}
  assert not sub, ('sub-modal n_ep_100k rows - REFUSE (realized-'
                   'training rule)', modal, sub)
  return rows


def load_ridge(pattern):
  out = {}
  for p in sorted(glob.glob(pattern)):
    with open(p) as f:
      d = json.load(f)
    rid = d['run_id']
    m = re.match(r'^ax1wm_finger_(f?)(sk[13])q1s([01])_seed([1-4])$', rid)
    assert m, (p, rid)
    # delta-rev F5: the bundle renames per-fit jsons by run_id
    assert os.path.basename(p).startswith(rid), (p, rid)
    assert d['probeset_id'].startswith('finger_v1'), (p, d['probeset_id'])
    assert d.get('reward_override') is None, (p, 'override-scored json')
    # reviewer-3 F15: a poisoned json (written before a witness raise)
    # must not be consumable
    assert d.get('witness_match') is not False, (p, 'witness mismatch')
    key = (('apt' if m.group(1) else 'task'), m.group(2), m.group(3),
           int(m.group(4)))
    assert key not in out, ('duplicate ridge json', key)
    v = d['probe'][ALPHA_KEY]['auroc']
    assert v is not None, (p, 'degenerate auroc (None)')
    out[key] = float(v)
  assert len(out) == 32, ('expected 32 ridge jsons', len(out))
  return out


def analyse(auc, ridge):
  res = {}
  groups = {(a, s): [auc[(a, s, side, seed)] for side in SIDES
                     for seed in SEEDS] for a in ARMS for s in SIZES}
  # P-B1c legibility leg FIRST - a fit-level quantity, informative even
  # when behavior is floor-censored (reviewer-3 F7); gated on apt
  # legibility headroom at the LARGER size
  rg = {(a, s): [ridge[(a, s, side, seed)] for side in SIDES
                 for seed in SEEDS] for a in ARMS for s in SIZES}
  res['legibility_means'] = {f'{a}_{s}': float(np.mean(rg[(a, s)]))
                             for a in ARMS for s in SIZES}
  apt_big = float(np.mean(rg[('apt', 'sk3')]))
  if apt_big < APT_LEG_FLOOR:
    res['p_b1c_legibility'] = dict(
        verdict=('APT-FLOOR-CENSORED: apt legibility has no headroom '
                 'at sk3 (mean %.3f < %.2f) - the legibility DID '
                 'cannot discriminate' % (apt_big, APT_LEG_FLOOR)))
  else:
    res['p_b1c_legibility'] = did_stats(rg)
  floor_ok = {}
  for s in SIZES:
    tm = float(np.mean(groups[('task', s)]))
    floor_ok[s] = tm > FLOOR
    res[f'floor_{s}'] = dict(task_mean=tm, floor=FLOOR,
                             passed=floor_ok[s])
  res['p_b1a'] = {s: two_sample(groups[('task', s)], groups[('apt', s)])
                  for s in SIZES}
  if not any(floor_ok.values()):
    res['verdict'] = ('INSTRUMENT-LIMITED: both sizes FLOOR-CENSORED - '
                      'the descent overshot; a shallower grid is a new '
                      'registration')
    return res
  if not all(floor_ok.values()):
    alive = [s for s in SIZES if floor_ok[s]][0]
    res['verdict'] = (f'INDETERMINATE-PARTIAL: {alive} alive only - '
                      'P-B1b unavailable; single-size interaction '
                      'reported')
    return res
  d = did_stats(groups)
  res['p_b1b'] = d
  if d['ci'][0] > 0 and d['perm_p'] < 0.05:
    verdict = ('LAMBDA-SUPPORTED: the interaction GROWS as capacity '
               'falls - competition binds at the boundary (two-regime '
               'adjudication per the frozen Part-B rule + Amendment 1 '
               'mapping, jointly with B2)')
  elif d['ci'][1] < 0 and d['perm_p'] < 0.05:
    verdict = ('INTERACTION-SHRINKS: reported; consistent with R1-plus-'
               'approaching-floor - interpret WITH the floor margins')
  else:
    verdict = 'INDETERMINATE (MDE note applies; licenses nothing)'
  res['verdict'] = verdict
  return res


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--auc_csv')
  ap.add_argument('--runroot')
  ap.add_argument('--ridge_glob')
  ap.add_argument('--fit_counters')
  ap.add_argument('--ckpt_steps')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  # reviewer-3 F15: the registered P-B1c cannot be skipped by an
  # omitted flag - ridge_glob is REQUIRED
  assert args.auc_csv and args.runroot and args.ridge_glob \
      and args.fit_counters and args.ckpt_steps and args.output
  check_witness(json.load(open(args.fit_counters)),
                json.load(open(args.ckpt_steps)))
  check_configs(args.runroot)
  res = analyse(load_auc(args.auc_csv), load_ridge(args.ridge_glob))
  res['prereg'] = 'PREREG_capdescent_20260811.md'
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({'verdict': res['verdict'],
                    'p_b1b': res.get('p_b1b'),
                    'floors': {s: res[f'floor_{s}'] for s in SIZES}},
                   indent=1))


def _mk_auc(rng, i_sk3, i_sk1, task_sk3=400.0, task_sk1=350.0, sd=40.0):
  auc = {}
  for size, tlev, ilev in (('sk3', task_sk3, i_sk3),
                           ('sk1', task_sk1, i_sk1)):
    for side in SIDES:
      for seed in SEEDS:
        auc[('task', size, side, seed)] = tlev + sd * rng.standard_normal()
        auc[('apt', size, side, seed)] = tlev - ilev \
            + sd * rng.standard_normal()
  return auc


def _mk_ridge(rng, apt_sk3=0.62, apt_sk1=0.30, task=0.85, sd=0.02):
  ridge = {}
  for s, av in (('sk3', apt_sk3), ('sk1', apt_sk1)):
    for side in SIDES:
      for seed in SEEDS:
        ridge[('task', s, side, seed)] = task + sd * rng.standard_normal()
        ridge[('apt', s, side, seed)] = av + sd * rng.standard_normal()
  return ridge


def selfcheck():
  import tempfile
  # reviewer-3 F12: gate constants asserted as LITERALS, independent of
  # the module values used to build fixtures
  assert DETER == {'sk3': 256, 'sk1': 128}, DETER
  assert FLOOR == 113.682, FLOOR
  assert APT_LEG_FLOOR == 0.55, APT_LEG_FLOOR
  rgv = _mk_ridge(np.random.default_rng(90))
  # lambda: interaction grows with descent (sk1 > sk3)
  r = analyse(_mk_auc(np.random.default_rng(0), i_sk3=100.0,
                      i_sk1=350.0), rgv)
  assert r['verdict'].startswith('LAMBDA-SUPPORTED'), \
      (r['verdict'], r['p_b1b'])
  # shrinks
  r = analyse(_mk_auc(np.random.default_rng(1), i_sk3=350.0,
                      i_sk1=100.0), rgv)
  assert r['verdict'].startswith('INTERACTION-SHRINKS'), r['verdict']
  # flat -> indeterminate, EVEN WITH a large size main effect (F4:
  # the aligned permutation must not lose its calibration)
  r = analyse(_mk_auc(np.random.default_rng(2), i_sk3=200.0,
                      i_sk1=200.0, task_sk3=500.0, task_sk1=250.0), rgv)
  assert r['verdict'].startswith('INDETERMINATE'), \
      (r['verdict'], r['p_b1b'])
  # lambda STILL detected under the same large main effect (F4 power)
  r = analyse(_mk_auc(np.random.default_rng(20), i_sk3=100.0,
                      i_sk1=400.0, task_sk3=500.0, task_sk1=250.0), rgv)
  assert r['verdict'].startswith('LAMBDA-SUPPORTED'), \
      (r['verdict'], r['p_b1b'])
  # one size floored (legibility leg still computed - F7)
  r = analyse(_mk_auc(np.random.default_rng(3), i_sk3=200.0, i_sk1=50.0,
                      task_sk1=90.0), rgv)
  assert r['verdict'].startswith('INDETERMINATE-PARTIAL'), r['verdict']
  assert 'ci' in r['p_b1c_legibility'], r['p_b1c_legibility']
  # both floored
  r = analyse(_mk_auc(np.random.default_rng(4), i_sk3=10.0, i_sk1=10.0,
                      task_sk3=95.0, task_sk1=90.0), rgv)
  assert r['verdict'].startswith('INSTRUMENT-LIMITED'), r['verdict']
  # apt legibility at chance -> P-B1c censored (F7 gate)
  r = analyse(_mk_auc(np.random.default_rng(5), i_sk3=100.0,
                      i_sk1=350.0),
              _mk_ridge(np.random.default_rng(91), apt_sk3=0.42,
                        apt_sk1=0.40))
  assert 'APT-FLOOR-CENSORED' in \
      r['p_b1c_legibility'].get('verdict', ''), r['p_b1c_legibility']
  # loader round-trip: csv(+qc cols) + configs(+readout pin) + ridge
  # via load_ridge (F12: apt LOWER so an arm inversion flips the sign)
  with tempfile.TemporaryDirectory() as td:
    auc = _mk_auc(np.random.default_rng(6), i_sk3=100.0, i_sk1=350.0)
    cp = os.path.join(td, 'auc.csv')
    with open(cp, 'w', newline='') as f:
      w = csv.DictWriter(f, ['mode', 'seed', 'auc100k', 'qc_pass',
                             'n_ep_100k'])
      w.writeheader()
      for (a, s, side, seed), v in sorted(auc.items()):
        w.writerow(dict(
            mode=f"ax1{'f' if a == 'apt' else ''}{s}q1s{side}",
            seed=seed, auc100k=v, qc_pass=1, n_ep_100k=96))
    loaded = load_auc(cp)
    assert loaded == {k: float(v) for k, v in auc.items()}
    counters, steps = {}, {}
    for arm in ARMS:
      for size in SIZES:
        for side in SIDES:
          for seed in SEEDS:
            wm = wm_name(arm, size, side, seed)
            counters[wm] = dict(update=500000, total=500000)
            steps[wm] = 500000
            os.makedirs(os.path.join(td, wm))
            with open(os.path.join(td, wm, 'config.yaml'), 'w') as f:
              json.dump({'agent': {'dyn': {'rssm':
                                           {'deter': DETER[size]}},
                                   'expl': {'mode': arm}}}, f)
            ad = adapt_name(arm, size, side, seed)
            os.makedirs(os.path.join(td, ad))
            with open(os.path.join(td, ad, 'config.yaml'), 'w') as f:
              json.dump({'run': {'from_checkpoint': f'/r/{wm}/ckpt'},
                         'agent': {'frozen_wm': True,
                                   'policy': {'units': 64}}}, f)
    check_witness(counters, steps)
    check_configs(td)
    rdir = os.path.join(td, 'ridge')
    os.makedirs(rdir)
    rg_true = _mk_ridge(np.random.default_rng(92))
    for (a, s, side, seed), v in rg_true.items():
      rid = wm_name(a, s, side, seed)
      with open(os.path.join(rdir, rid + '.json'), 'w') as f:
        json.dump(dict(run_id=rid, probeset_id='finger_v1',
                       reward_override=None, witness_match=True,
                       probe={ALPHA_KEY: {'auroc': v, 'r2': 0.1}}), f)
    ridge = load_ridge(os.path.join(rdir, '*.json'))
    assert len(ridge) == 32
    r = analyse(loaded, ridge)
    assert r['p_b1c_legibility']['did'] > 0 and \
        r['p_b1c_legibility']['ci'][0] > 0, r['p_b1c_legibility']
    # refusals: bad deter, unpinned readout, poisoned witness_match,
    # qc-fail row, sub-modal row, short counter
    wm0 = wm_name('task', 'sk3', '0', 1)
    with open(os.path.join(td, wm0, 'config.yaml'), 'w') as f:
      json.dump({'agent': {'dyn': {'rssm': {'deter': 512}},
                           'expl': {'mode': 'task'}}}, f)
    try:
      check_configs(td)
      raise SystemExit('expected deter-mismatch refusal')
    except AssertionError:
      pass
    with open(os.path.join(td, wm0, 'config.yaml'), 'w') as f:
      json.dump({'agent': {'dyn': {'rssm': {'deter': 256}},
                           'expl': {'mode': 'task'}}}, f)
    ad0 = adapt_name('task', 'sk3', '0', 1)
    with open(os.path.join(td, ad0, 'config.yaml'), 'w') as f:
      json.dump({'run': {'from_checkpoint': f'/r/{wm0}/ckpt'},
                 'agent': {'frozen_wm': True,
                           'policy': {'units': 16}}}, f)
    try:
      check_configs(td)
      raise SystemExit('expected readout-pin refusal')
    except AssertionError:
      pass
    with open(os.path.join(td, ad0, 'config.yaml'), 'w') as f:
      json.dump({'run': {'from_checkpoint': f'/r/{wm0}/ckpt'},
                 'agent': {'frozen_wm': True,
                           'policy': {'units': 64}}}, f)
    bad = os.path.join(rdir, wm0 + '.json')
    with open(bad, 'w') as f:
      json.dump(dict(run_id=wm0, probeset_id='finger_v1',
                     reward_override=None, witness_match=False,
                     probe={ALPHA_KEY: {'auroc': 0.8}}), f)
    try:
      load_ridge(os.path.join(rdir, '*.json'))
      raise SystemExit('expected witness_match refusal')
    except AssertionError:
      pass
    with open(bad, 'w') as f:
      json.dump(dict(run_id=wm0, probeset_id='finger_v1',
                     reward_override=None, witness_match=True,
                     probe={ALPHA_KEY: {'auroc': rg_true[('task', 'sk3',
                                                          '0', 1)]}}),
                f)
    rows = open(cp).read().splitlines()
    with open(cp, 'w') as f:
      f.write(rows[0] + '\n')
      f.write(rows[1].rsplit(',', 2)[0] + ',0,96\n')
      f.write('\n'.join(rows[2:]))
    try:
      load_auc(cp)
      raise SystemExit('expected qc refusal')
    except AssertionError:
      pass
    with open(cp, 'w') as f:
      f.write(rows[0] + '\n')
      f.write(rows[1].rsplit(',', 2)[0] + ',1,80\n')
      f.write('\n'.join(rows[2:]))
    try:
      load_auc(cp)
      raise SystemExit('expected sub-modal refusal')
    except AssertionError:
      pass
    counters[wm0] = dict(update=450000, total=500000)
    try:
      check_witness(counters, steps)
      raise SystemExit('expected counter refusal')
    except AssertionError:
      pass
  print('capdescent_read selfcheck PASS (lambda/shrinks/indeterminate/'
        'partial-floor/both-floor + F4 main-effect calibration legs; '
        'P-B1c computed under floor + censored under apt-chance; '
        'loader round-trips w/ deter/readout-pin/witness/qc/sub-modal/'
        'counter refusals; literal gate constants)')


if __name__ == '__main__':
  main()
