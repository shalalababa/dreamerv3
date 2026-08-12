"""Frozen reader — third-domain wave (P-D1 reacher / P-D2 walker).

Registration: PREREG_domains_20260812.md (rules verbatim; predictions
frozen in PREREG_theory_R1_gating_20260811 Part A + amend2). TWO
registered executions, one per domain leg, each ONCE:

  python -m analysis.domains_read --domain reacher \
      --auc_csv <bundle>/auc.csv --runroot <bundle>/runroot_light \
      --fit_counters <bundle>/domains_fit_counters_<dom>.json \
      --ckpt_steps <bundle>/domains_ckpt_steps_<dom>.json \
      --manifest <bundle>/q1_manifest.json --output <dir>
(q1_manifest.json = the bundle's copy of axis1_<dom>/q1/manifest.json)
Selfcheck: python -m analysis.domains_read --selfcheck
"""

import argparse
import csv
import json
import math
import os

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
SEEDS = tuple(range(1, 17))
SIDES = ('0', '1')
ARMS = ('task', 'apt')
TASKS = {'reacher': 'dmc_reacher_hard', 'walker': 'dmc_walker_walk'}
# W0 executed finger anchor (artifacts/w0_n16_interaction_20260713,
# sensitivity.d_z) — pinned, zero recomputation:
FINGER_DZ = 0.8284213656547132
# paired n=16, alpha=.05 two-sided, power .80: (t_{.975,15}+t_{.80,15})/4
MDE80_DZ = 0.74942
# buffer quality-gate bars (prereg, decidable pre-outcome)
OCC_HI_MIN = 0.15
OCC_LO_MAX = 0.05
OCC_RATIO_MIN = 3.0


def mode_name(arm, side):
  return f"ax1{'f' if arm == 'apt' else ''}q1s{side}"


def wm_name(dom, arm, side, seed):
  return f"ax1wm_{dom}_{'f' if arm == 'apt' else ''}q1s{side}_seed{seed}"


def adapt_name(dom, arm, side, seed):
  return (f"adapt_ax1{'f' if arm == 'apt' else ''}q1s{side}_{dom}"
          f"_seed{seed}_ckpt500000")


def _erfinv(x):
  a = 0.147
  ln = math.log(1.0 - x * x)
  t1 = 2.0 / (math.pi * a) + ln / 2.0
  return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)


def one_sample(x, rng_seed=RNG_SEED, nboot=B_BOOT):
  """Mean of paired deltas: BCa 95% CI + EXACT sign-flip permutation p
  (all 2^n flips; n=16 -> 65536). Permutation primary (standing rule)."""
  x = np.asarray(x, float)
  n = len(x)
  rng = np.random.default_rng(rng_seed)
  obs = float(x.mean())
  boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nboot)])
  prop = float(np.mean(boots < obs))
  prop = min(max(prop, 1.0 / (nboot + 1)), 1.0 - 1.0 / (nboot + 1))
  z0 = math.sqrt(2) * _erfinv(2 * prop - 1)
  jack = np.array([np.delete(x, i).mean() for i in range(n)])
  jm = jack.mean()
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.0
  def q(alpha):
    z = math.sqrt(2) * _erfinv(2 * alpha - 1)
    adj = z0 + (z0 + z) / (1.0 - acc * (z0 + z))
    p = 0.5 * (1.0 + math.erf(adj / math.sqrt(2)))
    return float(np.percentile(boots, 100.0 * min(max(p, 0.0), 1.0)))
  signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
  flips = signs @ x / n
  perm_p = float(np.mean(np.abs(flips) >= abs(obs) - 1e-12))
  sd = float(x.std(ddof=1))
  d_z = obs / sd if sd > 0 else float('inf') * np.sign(obs)
  return dict(mean=obs, ci=[q(0.025), q(0.975)], perm_p=perm_p,
              d_z=float(d_z), sd=sd, n=n)


def _load_yaml(path):
  try:
    import yaml
    with open(path) as f:
      return yaml.safe_load(f)
  except ImportError:
    from ruamel.yaml import YAML
    with open(path) as f:
      return YAML(typ='safe').load(f)


def check_witness(dom, fit_counters, ckpt_steps):
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        n = wm_name(dom, arm, side, seed)
        assert n in fit_counters and \
            int(fit_counters[n]['update']) == \
            int(fit_counters[n]['total']), (n, fit_counters.get(n))
        assert n in ckpt_steps and ckpt_steps[n] is not None and \
            int(ckpt_steps[n]) == 500000, (n, ckpt_steps.get(n))


def check_configs(dom, runroot):
  task = TASKS[dom]
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        wm = wm_name(dom, arm, side, seed)
        cfg = _load_yaml(os.path.join(runroot, wm, 'config.yaml'))
        assert str(cfg['agent']['expl']['mode']) == arm, \
            (wm, cfg['agent']['expl']['mode'])
        assert str(cfg['task']) == task, (wm, cfg['task'])
        ad = adapt_name(dom, arm, side, seed)
        acfg = _load_yaml(os.path.join(runroot, ad, 'config.yaml'))
        fc = str(acfg['run']['from_checkpoint'])
        assert wm in fc, (ad, 'from_checkpoint does not name WM', fc)
        assert acfg['agent'].get('frozen_wm') is not False, \
            (ad, 'adapt not frozen')
        assert str(acfg['task']) == task, (ad, acfg['task'])


def check_manifest(manifest):
  s0 = manifest['sides'][0]['occ_recomputed']
  s1 = manifest['sides'][1]['occ_recomputed']
  assert s1 >= OCC_HI_MIN, ('side1 occ below bar', s1, OCC_HI_MIN)
  assert s0 <= OCC_LO_MAX, ('side0 occ above bar', s0, OCC_LO_MAX)
  assert s0 == 0 or s1 / s0 >= OCC_RATIO_MIN, ('occ ratio below bar', s1, s0)
  return dict(occ_s0=s0, occ_s1=s1)


def load_auc(path, dom):
  rows, n_eps = {}, {}
  with open(path) as f:
    for r in csv.DictReader(f):
      if r['domain'] != dom:
        continue
      key = None
      for arm in ARMS:
        for side in SIDES:
          if r['mode'] == mode_name(arm, side):
            key = (arm, side, int(r['seed']))
      if key is None:
        continue
      if key[2] not in SEEDS:
        continue
      assert key not in rows, ('duplicate row', key)
      assert str(r['qc_pass']).lower() in ('1', 'true'), \
          (key, 'qc_pass failed')
      rows[key] = float(r['auc100k'])
      n_eps[key] = int(r['n_ep_100k'])
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        assert (arm, side, seed) in rows, ('missing adapt row', arm,
                                           side, seed)
  assert len(rows) == 64, len(rows)
  vals, counts = np.unique(list(n_eps.values()), return_counts=True)
  modal = int(vals[counts.argmax()])
  # review X1/B4: STRICT equality — sub-modal = truncated training,
  # SUPER-modal = the scores.jsonl append-on-forced-restart signature
  # (duplicated eval episodes). Either REFUSES; benign variation goes
  # through a dated amendment, never a silent pass.
  off = {k: v for k, v in n_eps.items() if v != modal}
  assert not off, ('non-modal n_ep_100k rows - REFUSE (realized-'
                   'training rule, strict)', modal, off)
  return rows


def analyse(dom, rows):
  deltas = np.array(
      [(rows[('task', '1', s)] - rows[('task', '0', s)])
       - (rows[('apt', '1', s)] - rows[('apt', '0', s)]) for s in SEEDS])
  stats = one_sample(deltas)
  cell_means = {f'{arm}_s{side}': float(np.mean(
      [rows[(arm, side, s)] for s in SEEDS]))
      for arm in ARMS for side in SIDES}
  level_contrast = float(
      np.mean([rows[('task', side, s)] for side in SIDES for s in SEEDS])
      - np.mean([rows[('apt', side, s)] for side in SIDES for s in SEEDS]))
  fired_pos = stats['ci'][0] > 0 and stats['perm_p'] < 0.05
  fired_neg = stats['ci'][1] < 0 and stats['perm_p'] < 0.05
  out = dict(domain=dom, interaction=stats, cell_means=cell_means,
             level_contrast_task_minus_apt=level_contrast,
             per_seed_deltas=[float(d) for d in deltas],
             ceiling_note=float(max(cell_means.values()) / 1000.0))
  if dom == 'reacher':
    if fired_pos:
      out['verdict'] = 'PRESENT'
    elif fired_neg:
      out['verdict'] = 'REVERSED'
    else:
      out['verdict'] = 'ABSENT-OR-UNDERPOWERED'
      out['mde_note'] = dict(
          mde80_dz=MDE80_DZ,
          mde80_score_units=float(MDE80_DZ * stats['sd']),
          power_at_finger_dz=0.87)
  elif dom == 'walker':
    out['finger_dz_anchor'] = FINGER_DZ
    # review A3: a significantly NEGATIVE walker interaction is a
    # REVERSAL, not attenuation — it must never grade R1-consistent.
    if fired_neg:
      out['verdict'] = 'REVERSED'
    elif stats['d_z'] < FINGER_DZ:
      out['verdict'] = 'ATTENUATED-CONSISTENT'
    else:
      out['verdict'] = 'NOT-ATTENUATED'
    if fired_pos:
      out['breadth_present'] = True
  else:
    raise SystemExit(f'unknown domain {dom!r}')
  return out


def run(args):
  with open(args.fit_counters) as f:
    fc = json.load(f)
  with open(args.ckpt_steps) as f:
    cs = json.load(f)
  check_witness(args.domain, fc, cs)
  check_configs(args.domain, args.runroot)
  with open(args.manifest) as f:
    occ = check_manifest(json.load(f))
  rows = load_auc(args.auc_csv, args.domain)
  out = analyse(args.domain, rows)
  out['buffer_occ'] = occ
  os.makedirs(args.output, exist_ok=True)
  path = os.path.join(args.output, f'domains_{args.domain}.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print(f'-> {path}')


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

def _mk_rows(rng, interaction=0.0, base=100.0, noise=25.0):
  rows = {}
  for seed in SEEDS:
    eps = {(arm, side): rng.normal(0, noise) for arm in ARMS
           for side in SIDES}
    for arm in ARMS:
      for side in SIDES:
        v = base + eps[(arm, side)]
        if arm == 'task' and side == '1':
          v += interaction
        rows[(arm, side, seed)] = v
  return rows


def _gen_auc_via_pipeline(tmp, dom, rows):
  """Review X2: generate auc.csv through the REAL analysis.adaptation_auc
  pipeline (synthetic scores.jsonl per run dir) so column-name drift in
  the collate is caught here, not at the one read."""
  import sys
  from analysis import adaptation_auc as aa
  rr = os.path.join(tmp, 'scores_runroot')
  for (arm, side, seed), v in rows.items():
    d = os.path.join(rr, adapt_name(dom, arm, side, seed))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'scores.jsonl'), 'w') as f:
      for i in range(96):
        f.write(json.dumps({'step': (i + 1) * 1000,
                            'episode/score': float(v)}) + '\n')
  outdir = os.path.join(tmp, 'auc_out')
  argv = sys.argv
  try:
    sys.argv = ['adaptation_auc', '--runroot', rr, '--output', outdir]
    aa.main()
  finally:
    sys.argv = argv
  return os.path.join(outdir, 'auc.csv')


def _write_fixture(tmp, dom, rows, *, break_wm=None, frozen_false=None,
                   wrong_mode=None, wrong_task=None, occ=(0.03, 0.30)):
  os.makedirs(tmp, exist_ok=True)
  auc = os.path.join(tmp, 'auc.csv')
  with open(auc, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run_id', 'mode', 'domain', 'seed', 'milestone',
                'auc100k', 'n_ep_100k', 'qc_pass'])
    for (arm, side, seed), v in rows.items():
      w.writerow([adapt_name(dom, arm, side, seed), mode_name(arm, side),
                  dom, seed, 500000, v, 96, 1])
  fc, cs = {}, {}
  rr = os.path.join(tmp, 'runroot')
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        wm = wm_name(dom, arm, side, seed)
        fc[wm] = dict(update=500000, total=500000)
        cs[wm] = 500000
        wd = os.path.join(rr, wm)
        os.makedirs(wd, exist_ok=True)
        mode = arm
        if wrong_mode == (arm, side, seed):
          mode = 'apt' if arm == 'task' else 'task'
        fit_task = TASKS[dom]
        if wrong_task == (arm, side, seed):
          fit_task = 'dmc_cup_catch'
        with open(os.path.join(wd, 'config.yaml'), 'w') as f:
          json.dump(dict(task=fit_task,
                         agent=dict(expl=dict(mode=mode))), f)
        ad = os.path.join(rr, adapt_name(dom, arm, side, seed))
        os.makedirs(ad, exist_ok=True)
        fz = False if frozen_false == (arm, side, seed) else True
        fcp = ('/x/ckpt/none' if break_wm == (arm, side, seed)
               else f'/runs/{wm}/ckpt/x-000000500000')
        with open(os.path.join(ad, 'config.yaml'), 'w') as f:
          json.dump(dict(task=TASKS[dom],
                         run=dict(from_checkpoint=fcp),
                         agent=dict(frozen_wm=fz)), f)
  with open(os.path.join(tmp, 'fit_counters.json'), 'w') as f:
    json.dump(fc, f)
  with open(os.path.join(tmp, 'ckpt_steps.json'), 'w') as f:
    json.dump(cs, f)
  man = dict(sides=[dict(occ_recomputed=occ[0]),
                    dict(occ_recomputed=occ[1])])
  with open(os.path.join(tmp, 'manifest.json'), 'w') as f:
    json.dump(man, f)
  return tmp


def _run_fixture(tmp, dom):
  ns = argparse.Namespace(
      domain=dom, auc_csv=os.path.join(tmp, 'auc.csv'),
      runroot=os.path.join(tmp, 'runroot'),
      fit_counters=os.path.join(tmp, 'fit_counters.json'),
      ckpt_steps=os.path.join(tmp, 'ckpt_steps.json'),
      manifest=os.path.join(tmp, 'manifest.json'),
      output=os.path.join(tmp, 'out'))
  run(ns)
  with open(os.path.join(tmp, 'out', f'domains_{dom}.json')) as f:
    return json.load(f)


def selfcheck():
  import shutil
  import tempfile
  base = tempfile.mkdtemp(prefix='domains_selfcheck_')
  try:
    # pinned constants (mutation guard)
    assert FINGER_DZ == 0.8284213656547132
    assert MDE80_DZ == 0.74942
    # exact permutation calibration on a hand null / strong signal
    rng = np.random.default_rng(2)
    null = one_sample(rng.normal(0, 10, 16))
    assert null['perm_p'] > 0.05, null
    strong = one_sample(rng.normal(30, 10, 16))
    assert strong['perm_p'] < 0.01 and strong['ci'][0] > 0, strong
    # PRESENT — auc.csv generated through the REAL adaptation_auc
    # pipeline (review X2: catches collate column drift)
    rng = np.random.default_rng(0)
    rows_a = _mk_rows(rng, interaction=110.0)
    t = _write_fixture(os.path.join(base, 'a'), 'reacher', rows_a)
    real_csv = _gen_auc_via_pipeline(os.path.join(base, 'a'), 'reacher',
                                     rows_a)
    import shutil as _sh
    _sh.copy(real_csv, os.path.join(t, 'auc.csv'))
    out = _run_fixture(t, 'reacher')
    assert out['verdict'] == 'PRESENT', out['verdict']
    got = {round(v, 3) for v in out['per_seed_deltas']}
    want = {round((rows_a[('task', '1', s)] - rows_a[('task', '0', s)])
                  - (rows_a[('apt', '1', s)] - rows_a[('apt', '0', s)]), 3)
            for s in SEEDS}
    assert got == want, 'pipeline-generated auc does not round-trip'
    # ABSENT-OR-UNDERPOWERED carries the MDE note
    rng = np.random.default_rng(1)
    t = _write_fixture(os.path.join(base, 'b'), 'reacher',
                       _mk_rows(rng, interaction=0.0))
    out = _run_fixture(t, 'reacher')
    assert out['verdict'] == 'ABSENT-OR-UNDERPOWERED' and \
        'mde_note' in out, out
    # REVERSED
    rng = np.random.default_rng(3)
    t = _write_fixture(os.path.join(base, 'c'), 'reacher',
                       _mk_rows(rng, interaction=-120.0))
    out = _run_fixture(t, 'reacher')
    assert out['verdict'] == 'REVERSED', out['verdict']
    # walker ATTENUATED (small interaction -> d_z below anchor)
    rng = np.random.default_rng(4)
    t = _write_fixture(os.path.join(base, 'd'), 'walker',
                       _mk_rows(rng, interaction=5.0))
    out = _run_fixture(t, 'walker')
    assert out['verdict'] == 'ATTENUATED-CONSISTENT', out
    assert 'breadth_present' not in out, out
    # walker NOT-ATTENUATED (d_z above anchor) + breadth tag
    rng = np.random.default_rng(5)
    t = _write_fixture(os.path.join(base, 'e'), 'walker',
                       _mk_rows(rng, interaction=200.0, noise=15.0))
    out = _run_fixture(t, 'walker')
    assert out['verdict'] == 'NOT-ATTENUATED', out
    assert out.get('breadth_present') is True, out
    # walker REVERSED (review A3: a significant negative interaction
    # must NOT grade ATTENUATED-CONSISTENT)
    rng = np.random.default_rng(6)
    t = _write_fixture(os.path.join(base, 'e2'), 'walker',
                       _mk_rows(rng, interaction=-150.0))
    out = _run_fixture(t, 'walker')
    assert out['verdict'] == 'REVERSED', out['verdict']
    # refusals
    def refused(build_kw, mutate=None):
      d = os.path.join(base, f'r{len(os.listdir(base))}')
      rng = np.random.default_rng(9)
      t = _write_fixture(d, 'reacher', _mk_rows(rng, interaction=100.0),
                         **build_kw)
      if mutate:
        mutate(t)
      try:
        _run_fixture(t, 'reacher')
      except AssertionError:
        return True
      return False
    assert refused(dict(frozen_false=('task', '1', 3))), 'frozen gate dead'
    assert refused(dict(wrong_mode=('apt', '0', 5))), 'expl-mode gate dead'
    assert refused(dict(wrong_task=('task', '0', 2))), 'task gate dead'
    assert refused(dict(break_wm=('apt', '0', 7))), 'from_checkpoint gate dead'
    assert refused(dict(occ=(0.10, 0.30))), 'occ lo gate dead'
    assert refused(dict(occ=(0.03, 0.10))), 'occ hi gate dead'
    def drop_row(t):
      lines = open(os.path.join(t, 'auc.csv')).read().splitlines()
      with open(os.path.join(t, 'auc.csv'), 'w') as f:
        f.write('\n'.join(lines[:-1]) + '\n')
    assert refused({}, drop_row), 'missing-row gate dead'
    def dup_row(t):
      lines = open(os.path.join(t, 'auc.csv')).read().splitlines()
      with open(os.path.join(t, 'auc.csv'), 'w') as f:
        f.write('\n'.join(lines + [lines[-1]]) + '\n')
    assert refused({}, dup_row), 'duplicate-row gate dead'
    def submodal(t):
      lines = open(os.path.join(t, 'auc.csv')).read().splitlines()
      parts = lines[-1].split(',')
      parts[-2] = '40'
      with open(os.path.join(t, 'auc.csv'), 'w') as f:
        f.write('\n'.join(lines[:-1] + [','.join(parts)]) + '\n')
    assert refused({}, submodal), 'sub-modal gate dead'
    def supermodal(t):
      # the scores.jsonl append-on-restart signature (review B4/X1)
      lines = open(os.path.join(t, 'auc.csv')).read().splitlines()
      parts = lines[-1].split(',')
      parts[-2] = '192'
      with open(os.path.join(t, 'auc.csv'), 'w') as f:
        f.write('\n'.join(lines[:-1] + [','.join(parts)]) + '\n')
    assert refused({}, supermodal), 'super-modal gate dead'
    def qc_fail(t):
      lines = open(os.path.join(t, 'auc.csv')).read().splitlines()
      parts = lines[-1].split(',')
      parts[-1] = '0'
      with open(os.path.join(t, 'auc.csv'), 'w') as f:
        f.write('\n'.join(lines[:-1] + [','.join(parts)]) + '\n')
    assert refused({}, qc_fail), 'qc gate dead'
    def short_fit(t):
      p = os.path.join(t, 'fit_counters.json')
      fc = json.load(open(p))
      k = sorted(fc)[0]
      fc[k]['update'] = 312500
      json.dump(fc, open(p, 'w'))
    assert refused({}, short_fit), 'fit-counter gate dead'
    def short_ckpt(t):
      p = os.path.join(t, 'ckpt_steps.json')
      cs = json.load(open(p))
      k = sorted(cs)[0]
      cs[k] = 400000
      json.dump(cs, open(p, 'w'))
    assert refused({}, short_ckpt), 'ckpt-step gate dead'
    print('SELFCHECK PASS')
  finally:
    shutil.rmtree(base, ignore_errors=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--selfcheck', action='store_true')
  ap.add_argument('--domain', choices=('reacher', 'walker'))
  ap.add_argument('--auc_csv')
  ap.add_argument('--runroot')
  ap.add_argument('--fit_counters')
  ap.add_argument('--ckpt_steps')
  ap.add_argument('--manifest')
  ap.add_argument('--output')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  for req in ('domain', 'auc_csv', 'runroot', 'fit_counters',
              'ckpt_steps', 'manifest', 'output'):
    assert getattr(args, req), f'--{req} required'
  run(args)


if __name__ == '__main__':
  main()
