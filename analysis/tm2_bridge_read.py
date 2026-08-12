"""Frozen reader — TM2 bridge wave (P-F2 primary, powered P-F1).

Registration: PREREG_tm2_bridge_20260812.md (rules verbatim; P-F1
frozen in PREREG_theory_R1_gating_20260811 + amend2). ONE execution:

  python -m analysis.tm2_bridge_read --auc_csv <bundle>/auc.csv \
      --runroot <bundle>/runroot_light \
      --fit_counters <bundle>/tm2b_fit_counters.json --output <dir>
Selfcheck: python -m analysis.tm2_bridge_read --selfcheck
"""

import argparse
import csv
import json
import os

import numpy as np

from analysis.domains_read import one_sample

# Dreamer finger q1 task-apt LEVEL contrast, seeds 1-8 pooled sides,
# qc-passed auc100k (computed 2026-08-12 from the committed
# artifacts/p0_axis1_corrective_20260713 csvs) - pinned:
DV3_ANCHOR = 108.33852499999999
RECON_COEF = 20.0
ARMS = ('aware', 'free', 'rec')
SIDES = ('0', '1')
SEEDS = tuple(range(1, 9))
COEFS = {  # (reward_coef, value_coef, recon_coef)
    'aware': (0.1, 0.1, 0.0),
    'free': (0.0, 0.0, 0.0),
    'rec': (0.0, 0.0, RECON_COEF),
}
TASK = 'dmc_finger_turn_hard'


def wm_name(arm, side, seed):
  return f'tm2wm_finger_b{arm}q1s{side}_seed{seed}'


def adapt_name(arm, side, seed):
  return f'adapt_tm2b{arm}q1s{side}_finger_seed{seed}_ckpt500000'


def mode_name(arm, side):
  return f'tm2b{arm}q1s{side}'


def check_witness(fc):
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        n = wm_name(arm, side, seed)
        assert n in fc, ('missing fit witness', n)
        assert int(fc[n]['update']) == int(fc[n]['total']), (n, fc[n])
        assert fc[n].get('done') is True, (n, 'no TM2_FIT_DONE')


def check_configs(runroot):
  consistency = set()
  dec_params = set()
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        wm = wm_name(arm, side, seed)
        with open(os.path.join(runroot, wm, 'config.yaml')) as f:
          cfg = json.load(f)
        assert cfg['arm'] == arm, (wm, cfg['arm'])
        got = (float(cfg['reward_coef']), float(cfg['value_coef']),
               float(cfg['recon_coef']))
        assert got == COEFS[arm], (wm, got, COEFS[arm])
        consistency.add(float(cfg['consistency_coef']))
        dec_params.add(int(cfg['decoder_params']))
        assert cfg['task'] == TASK, (wm, cfg['task'])
        # review B2: the bridge manifest's key is source_dir (what
        # probing/tdmpc2_bridge.py actually writes) — hard-require it
        assert f'side{side}' in str(cfg['data']['source_dir']), \
            (wm, 'data side mismatch', cfg['data'])
        ad = adapt_name(arm, side, seed)
        with open(os.path.join(runroot, ad, 'config.yaml')) as f:
          acfg = json.load(f)
        assert wm in str(acfg['pretrained']), (ad, acfg['pretrained'])
        assert acfg.get('mpc') is True, (ad, 'mpc off')
        assert int(acfg['steps']) == 125000, (ad, acfg['steps'])
        assert list(acfg['frozen_prefixes']) == ['_encoder',
                                                 '_dynamics'], (ad, acfg)
        assert int(acfg['n_frozen_tensors']) > 0, (ad, 'nothing frozen')
        assert acfg['task'] == TASK, (ad, acfg['task'])
  assert len(consistency) == 1, ('consistency_coef differs across arms',
                                 consistency)
  assert len(dec_params) == 1 and dec_params.pop() > 0, \
      ('decoder parity broken', dec_params)


def load_auc(path):
  rows, n_eps = {}, {}
  with open(path) as f:
    for r in csv.DictReader(f):
      key = None
      for arm in ARMS:
        for side in SIDES:
          if r['mode'] == mode_name(arm, side) and r['domain'] == 'finger':
            key = (arm, side, int(r['seed']))
      if key is None or key[2] not in SEEDS:
        continue
      assert key not in rows, ('duplicate row', key)
      assert str(r['qc_pass']).lower() in ('1', 'true'), (key, 'qc fail')
      rows[key] = float(r['auc100k'])
      n_eps[key] = int(r['n_ep_100k'])
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        assert (arm, side, seed) in rows, ('missing row', arm, side, seed)
  assert len(rows) == 48, len(rows)
  vals, counts = np.unique(list(n_eps.values()), return_counts=True)
  modal = int(vals[counts.argmax()])
  # review X1/B4: strict equality (super-modal = the append-on-forced-
  # restart duplication signature; sub-modal = truncation)
  off = {k: v for k, v in n_eps.items() if v != modal}
  assert not off, ('non-modal n_ep_100k rows - REFUSE (strict)', modal, off)
  return rows


def paired(rows, a, b):
  return np.array([rows[(a, side, seed)] - rows[(b, side, seed)]
                   for side in SIDES for seed in SEEDS])


def analyse(rows):
  f2 = one_sample(paired(rows, 'free', 'rec'))
  f1 = one_sample(paired(rows, 'aware', 'free'))
  ar = one_sample(paired(rows, 'aware', 'rec'))

  if f2['ci'][0] > 0 and f2['perm_p'] < 0.05:
    v2 = 'RECON-REOPENS'
  elif f2['ci'][1] < 0 and f2['perm_p'] < 0.05:
    v2 = 'RECON-HELPS'
  else:
    v2 = 'INDETERMINATE'

  ratio = dict(point=f1['mean'] / DV3_ANCHOR,
               ci=[f1['ci'][0] / DV3_ANCHOR, f1['ci'][1] / DV3_ANCHOR])
  if f1['ci'][0] > 0 and f1['perm_p'] < 0.05:
    v1 = ('ATTENUATED-POSITIVE' if ratio['ci'][1] < 1
          else 'POSITIVE-NOT-RESOLVED')
  elif f1['ci'][1] < 0 and f1['perm_p'] < 0.05:
    v1 = 'INVERTED'
  else:
    v1 = 'NULL-CONTRAST'

  cell_means = {f'{arm}_s{side}': float(np.mean(
      [rows[(arm, side, s)] for s in SEEDS]))
      for arm in ARMS for side in SIDES}
  closure = (ar['mean'] - f1['mean']) / DV3_ANCHOR
  out = dict(
      p_f2=dict(stats=f2, verdict=v2),
      p_f1=dict(stats=f1, ratio=ratio, anchor=DV3_ANCHOR, verdict=v1),
      aware_minus_rec=ar, closure_fraction=float(closure),
      cell_means=cell_means)
  for tag, st in (('p_f2', f2), ('p_f1', f1)):
    if out[tag]['verdict'] in ('INDETERMINATE', 'NULL-CONTRAST'):
      out[tag]['mde_note'] = dict(
          mde80_score_units=float(0.74942 * st['sd']))
  return out


def run(args):
  with open(args.fit_counters) as f:
    fc = json.load(f)
  check_witness(fc)
  check_configs(args.runroot)
  rows = load_auc(args.auc_csv)
  out = analyse(rows)
  os.makedirs(args.output, exist_ok=True)
  path = os.path.join(args.output, 'tm2_bridge.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print(f'-> {path}')


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

def _mk_rows(rng, aware=140.0, free=110.0, rec=80.0, noise=20.0):
  rows = {}
  for side in SIDES:
    for seed in SEEDS:
      base = rng.normal(0, noise)
      rows[('aware', side, seed)] = aware + base + rng.normal(0, noise / 2)
      rows[('free', side, seed)] = free + base + rng.normal(0, noise / 2)
      rows[('rec', side, seed)] = rec + base + rng.normal(0, noise / 2)
  return rows


def _write_fixture(tmp, rows, *, coef_break=None, adapt_break=None,
                   dec_break=None, task_break=None, steps_break=None,
                   cons_break=None, nfrozen_break=None, side_break=None):
  os.makedirs(tmp, exist_ok=True)
  auc = os.path.join(tmp, 'auc.csv')
  with open(auc, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run_id', 'mode', 'domain', 'seed', 'milestone',
                'auc100k', 'n_ep_100k', 'qc_pass'])
    for (arm, side, seed), v in rows.items():
      w.writerow([adapt_name(arm, side, seed), mode_name(arm, side),
                  'finger', seed, 500000, v, 96, 1])
  fc = {}
  rr = os.path.join(tmp, 'runroot')
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        wm = wm_name(arm, side, seed)
        fc[wm] = dict(update=500000, total=500000, done=True)
        wd = os.path.join(rr, wm)
        os.makedirs(wd, exist_ok=True)
        r, v, rc = COEFS[arm]
        if coef_break == (arm, side, seed):
          rc = 5.0
        dp = 999 if dec_break == (arm, side, seed) else 12345
        cons = 5.0 if cons_break == (arm, side, seed) else 20.0
        ftask = 'dmc_cup_catch' if task_break == (arm, side, seed) else TASK
        dside = side
        if side_break == (arm, side, seed):
          dside = '1' if side == '0' else '0'
        with open(os.path.join(wd, 'config.yaml'), 'w') as f:
          json.dump(dict(arm=arm, task=ftask, reward_coef=r,
                         value_coef=v, recon_coef=rc,
                         consistency_coef=cons, decoder_params=dp,
                         # the key the PIPELINE writes (tdmpc2_bridge):
                         data=dict(source_dir=f'/x/q1/side{dside}',
                                   task=TASK)), f)
        ad = os.path.join(rr, adapt_name(arm, side, seed))
        os.makedirs(ad, exist_ok=True)
        pre = ('/x/other_wm' if adapt_break == (arm, side, seed)
               else f'/runs/{wm}')
        steps = 100000 if steps_break == (arm, side, seed) else 125000
        nfro = 0 if nfrozen_break == (arm, side, seed) else 8
        with open(os.path.join(ad, 'config.yaml'), 'w') as f:
          json.dump(dict(task=TASK, pretrained=pre, mpc=True,
                         steps=steps,
                         frozen_prefixes=['_encoder', '_dynamics'],
                         n_frozen_tensors=nfro), f)
  with open(os.path.join(tmp, 'fit_counters.json'), 'w') as f:
    json.dump(fc, f)
  return tmp


def _run_fixture(tmp):
  ns = argparse.Namespace(
      auc_csv=os.path.join(tmp, 'auc.csv'),
      runroot=os.path.join(tmp, 'runroot'),
      fit_counters=os.path.join(tmp, 'fit_counters.json'),
      output=os.path.join(tmp, 'out'))
  run(ns)
  with open(os.path.join(tmp, 'out', 'tm2_bridge.json')) as f:
    return json.load(f)


def selfcheck():
  import shutil
  import tempfile
  base = tempfile.mkdtemp(prefix='tm2bridge_selfcheck_')
  try:
    assert DV3_ANCHOR == 108.33852499999999
    assert RECON_COEF == 20.0
    # RECON-REOPENS + ATTENUATED-POSITIVE (aware-free=30 -> ratio<1)
    rng = np.random.default_rng(0)
    t = _write_fixture(os.path.join(base, 'a'), _mk_rows(rng))
    out = _run_fixture(t)
    assert out['p_f2']['verdict'] == 'RECON-REOPENS', out['p_f2']
    assert out['p_f1']['verdict'] == 'ATTENUATED-POSITIVE', out['p_f1']
    assert out['p_f1']['ratio']['ci'][1] < 1
    # RECON-HELPS (rec above free)
    rng = np.random.default_rng(1)
    t = _write_fixture(os.path.join(base, 'b'),
                       _mk_rows(rng, rec=150.0))
    out = _run_fixture(t)
    assert out['p_f2']['verdict'] == 'RECON-HELPS', out['p_f2']
    # INDETERMINATE (rec == free) carries MDE note
    rng = np.random.default_rng(2)
    t = _write_fixture(os.path.join(base, 'c'),
                       _mk_rows(rng, rec=110.0))
    out = _run_fixture(t)
    assert out['p_f2']['verdict'] == 'INDETERMINATE' and \
        'mde_note' in out['p_f2'], out['p_f2']
    # NULL-CONTRAST (aware == free) carries MDE note
    rng = np.random.default_rng(3)
    t = _write_fixture(os.path.join(base, 'd'),
                       _mk_rows(rng, aware=110.0, noise=30.0))
    out = _run_fixture(t)
    assert out['p_f1']['verdict'] == 'NULL-CONTRAST' and \
        'mde_note' in out['p_f1'], out['p_f1']
    # INVERTED (aware below free)
    rng = np.random.default_rng(4)
    t = _write_fixture(os.path.join(base, 'e'),
                       _mk_rows(rng, aware=60.0))
    out = _run_fixture(t)
    assert out['p_f1']['verdict'] == 'INVERTED', out['p_f1']
    # POSITIVE-NOT-RESOLVED (contrast ~ anchor scale)
    rng = np.random.default_rng(5)
    t = _write_fixture(os.path.join(base, 'f'),
                       _mk_rows(rng, aware=240.0, noise=15.0))
    out = _run_fixture(t)
    assert out['p_f1']['verdict'] == 'POSITIVE-NOT-RESOLVED', out['p_f1']
    # refusals
    def refused(build_kw, mutate=None):
      d = os.path.join(base, f'r{len(os.listdir(base))}')
      rng = np.random.default_rng(9)
      t = _write_fixture(d, _mk_rows(rng), **build_kw)
      if mutate:
        mutate(t)
      try:
        _run_fixture(t)
      except AssertionError:
        return True
      return False
    assert refused(dict(coef_break=('free', '0', 2))), 'coef gate dead'
    assert refused(dict(adapt_break=('rec', '1', 5))), 'pretrained gate dead'
    assert refused(dict(dec_break=('aware', '0', 1))), 'decoder-parity gate dead'
    assert refused(dict(task_break=('free', '1', 4))), 'task gate dead'
    assert refused(dict(steps_break=('aware', '0', 6))), 'steps gate dead'
    assert refused(dict(cons_break=('rec', '0', 7))), 'consistency gate dead'
    assert refused(dict(nfrozen_break=('free', '0', 8))), 'n_frozen gate dead'
    assert refused(dict(side_break=('aware', '1', 2))), 'data-side gate dead'
    def short_fit(t):
      p = os.path.join(t, 'fit_counters.json')
      fc = json.load(open(p))
      k = sorted(fc)[0]
      fc[k]['update'] = 400000
      json.dump(fc, open(p, 'w'))
    assert refused({}, short_fit), 'fit-counter gate dead'
    def not_done(t):
      p = os.path.join(t, 'fit_counters.json')
      fc = json.load(open(p))
      k = sorted(fc)[-1]
      fc[k]['done'] = False
      json.dump(fc, open(p, 'w'))
    assert refused({}, not_done), 'done gate dead'
    def drop_row(t):
      lines = open(os.path.join(t, 'auc.csv')).read().splitlines()
      with open(os.path.join(t, 'auc.csv'), 'w') as f:
        f.write('\n'.join(lines[:-1]) + '\n')
    assert refused({}, drop_row), 'missing-row gate dead'
    def supermodal(t):
      lines = open(os.path.join(t, 'auc.csv')).read().splitlines()
      parts = lines[-1].split(',')
      parts[-2] = '192'
      with open(os.path.join(t, 'auc.csv'), 'w') as f:
        f.write('\n'.join(lines[:-1] + [','.join(parts)]) + '\n')
    assert refused({}, supermodal), 'super-modal gate dead'
    def mpc_off(t):
      p = os.path.join(t, 'runroot', adapt_name('rec', '0', 3),
                       'config.yaml')
      cfg = json.load(open(p))
      cfg['mpc'] = False
      json.dump(cfg, open(p, 'w'))
    assert refused({}, mpc_off), 'mpc gate dead'
    print('SELFCHECK PASS')
  finally:
    shutil.rmtree(base, ignore_errors=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--selfcheck', action='store_true')
  ap.add_argument('--auc_csv')
  ap.add_argument('--runroot')
  ap.add_argument('--fit_counters')
  ap.add_argument('--output')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  for req in ('auc_csv', 'runroot', 'fit_counters', 'output'):
    assert getattr(args, req), f'--{req} required'
  run(args)


if __name__ == '__main__':
  main()
