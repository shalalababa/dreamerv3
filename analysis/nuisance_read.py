"""Frozen reader — B2 nuisance-injection λ trial (crowding test).

Registration: PREREG_nuisance_20260811.md (rules verbatim; Part-B
predictions frozen in PREREG_theory_R1_gating_20260811). ONE execution:

  python -m analysis.nuisance_read \
      --ridge_glob '<bundle>/ridge/*.json' --runroot <bundle>/runroot_light \
      --fit_counters <json> --ckpt_steps <json> --output <dir>
Selfcheck: python -m analysis.nuisance_read --selfcheck
"""

import argparse
import glob
import json
import math
import os
import re

import numpy as np

from analysis.capdescent_read import did_stats, _load_yaml

APT_FLOOR = 0.55
LOADS = ('nz1', 'nz2')
ARMS = ('task', 'apt')
SIDES = ('0', '1')
SEEDS = (1, 2, 3, 4)
ALPHA_KEY = 'alpha_0.001'
PROBESET_OF = {'nz1': 'finger_nz1_v1', 'nz2': 'finger_nz2_v1'}
RID_RE = re.compile(r'^ax1wm_finger_(?P<f>f?)(?P<load>nz[12])q1s'
                    r'(?P<side>[01])_seed(?P<seed>\d)$')


def wm_name(arm, load, side, seed):
  f = 'f' if arm == 'apt' else ''
  return f'ax1wm_finger_{f}{load}q1s{side}_seed{seed}'


def check_manifests(nz1_manifest, nz2_manifest):
  """Delta-rev F2: the REPLAY-side sigma is the mediator; gate the
  buffer manifests, not just the env knob."""
  for load, path in (('nz1', nz1_manifest), ('nz2', nz2_manifest)):
    with open(path) as f:
      m = json.load(f)
    dz = m['distractor']
    assert float(dz['sigma']) == SIGMA_OF[load], (path, dz['sigma'])
    assert int(dz['dims']) == 16, (path, dz['dims'])
    assert int(dz['holdout_per_side']) >= 1, path


def check_witness(fit_counters, ckpt_steps):
  for arm in ARMS:
    for load in LOADS:
      for side in SIDES:
        for seed in SEEDS:
          n = wm_name(arm, load, side, seed)
          assert n in fit_counters and \
              int(fit_counters[n]['update']) == \
              int(fit_counters[n]['total']), (n, fit_counters.get(n))
          assert n in ckpt_steps and int(ckpt_steps[n]) == 500000, \
              (n, ckpt_steps.get(n))


SIGMA_OF = {'nz1': 1.0, 'nz2': 4.0}


def check_configs(runroot):
  """Reviewer-3 F2 form: gate on the RECORDED distractor knob (the dz
  base config writes config.distractor.{dim,basesd,scale}), not on
  obs-space regexes — real configs always carry model_obs '.*' and a
  regex gate is vacuous."""
  for arm in ARMS:
    for load in LOADS:
      for side in SIDES:
        for seed in SEEDS:
          wm = wm_name(arm, load, side, seed)
          cfg = _load_yaml(os.path.join(runroot, wm, 'config.yaml'))
          assert str(cfg['agent']['expl']['mode']) == arm, \
              (wm, cfg['agent']['expl']['mode'])
          dz = cfg.get('distractor') or {}
          assert int(dz.get('dim', 0)) == 16, \
              (wm, 'distractor.dim != 16 - the injection never reached '
               'the model', dz)
          assert float(dz.get('basesd', -1)) == SIGMA_OF[load], \
              (wm, 'distractor.basesd does not match the load', dz)


def load_ridge(pattern):
  out = {}
  for p in sorted(glob.glob(pattern)):
    with open(p) as f:
      d = json.load(f)
    m = RID_RE.match(d['run_id'])
    assert m, (p, d.get('run_id'))
    # delta-rev F5: ridge_probe writes identical basenames per fit dir;
    # the bundle renames by run_id and this is the check on the rename
    assert os.path.basename(p).startswith(d['run_id']), \
        (p, d['run_id'], 'filename does not carry run_id')
    load = m.group('load')
    assert str(d['probeset_id']).startswith(PROBESET_OF[load]), \
        (p, 'cross-load probeset', d['probeset_id'], load)
    assert d.get('reward_override') is None, (p, 'override-scored json')
    key = (('apt' if m.group('f') else 'task'), load, m.group('side'),
           int(m.group('seed')))
    assert key not in out, ('duplicate ridge json', key)
    out[key] = float(d['probe'][ALPHA_KEY]['auroc'])
  for arm in ARMS:
    for load in LOADS:
      for side in SIDES:
        for seed in SEEDS:
          assert (arm, load, side, seed) in out, \
              ('missing ridge json', arm, load, side, seed)
  assert len(out) == 32, len(out)
  return out


def analyse(ridge):
  groups = {(a, l): [ridge[(a, l, side, seed)] for side in SIDES
                     for seed in SEEDS] for a in ARMS for l in LOADS}
  res = {'panel_means': {f'{a}_{l}': float(np.mean(groups[(a, l)]))
                         for a in ARMS for l in LOADS}}
  apt_lo = float(np.mean(groups[('apt', 'nz1')]))
  res['informativeness_gate'] = dict(apt_nz1_mean=apt_lo,
                                     floor=APT_FLOOR,
                                     passed=apt_lo >= APT_FLOOR)
  res['task_load_slope'] = float(np.mean(groups[('task', 'nz2')])
                                 - np.mean(groups[('task', 'nz1')]))
  sds = [float(np.std(groups[(a, l)], ddof=1))
         for a in ARMS for l in LOADS]
  # se(DID) = sd*sqrt(4/8) = sd/sqrt(2) at n=8/group (delta-rev F4)
  se = float(np.mean(sds)) / 2 ** 0.5
  res['mde_note'] = (
      'realized per-group AUROC sd %.4f (pooled); DID se ~ %.4f at '
      'n=8/group; 80%%-power MDE ~ %.4f (aligned permutation)'
      % (float(np.mean(sds)), se, 2.8 * se))
  if apt_lo < APT_FLOOR:
    res['verdict'] = (
        'APT-FLOOR-CENSORED => INSTRUMENT-LIMITED: the reward-free arm '
        'has no legibility headroom at low load - the DID cannot '
        'discriminate; panels reported (itself a registered descriptive '
        'fact about reward-free legibility under nuisance)')
    return res
  # DID over LOADS: reuse the capdescent machinery by mapping
  # nz2 -> the "descent" slot (sk1) and nz1 -> sk3
  mapped = {('task', 'sk1'): groups[('task', 'nz2')],
            ('apt', 'sk1'): groups[('apt', 'nz2')],
            ('task', 'sk3'): groups[('task', 'nz1')],
            ('apt', 'sk3'): groups[('apt', 'nz1')]}
  d = did_stats(mapped)
  # delta-rev F6: relabel the slot names back to loads
  d['n'] = {k.replace('sk1', 'nz2').replace('sk3', 'nz1'): v
            for k, v in d['n'].items()}
  d['slot_mapping'] = 'nz2->sk1, nz1->sk3 (capdescent did_stats slots)'
  res['p_n2a'] = d
  if d['ci'][0] > 0 and d['perm_p'] < 0.05:
    res['verdict'] = ('LAMBDA-SUPPORTED: crowding hits the reward-free '
                      'arm harder - competition confirmed in the forced '
                      'regime (Part-B adjudication jointly with B1)')
  elif d['ci'][1] < 0 and d['perm_p'] < 0.05:
    res['verdict'] = ('NEGATIVE-DID: the task arm lost MORE - outside '
                      'both models\' registered predictions; fact '
                      'reported, no further wording')
  else:
    res['verdict'] = 'INDETERMINATE (MDE recorded; licenses nothing)'
  return res


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--ridge_glob')
  ap.add_argument('--runroot')
  ap.add_argument('--nz1_manifest')
  ap.add_argument('--nz2_manifest')
  ap.add_argument('--fit_counters')
  ap.add_argument('--ckpt_steps')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.ridge_glob and args.runroot and args.nz1_manifest \
      and args.nz2_manifest and args.fit_counters \
      and args.ckpt_steps and args.output
  check_manifests(args.nz1_manifest, args.nz2_manifest)
  check_witness(json.load(open(args.fit_counters)),
                json.load(open(args.ckpt_steps)))
  check_configs(args.runroot)
  res = analyse(load_ridge(args.ridge_glob))
  res['prereg'] = 'PREREG_nuisance_20260811.md'
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({'verdict': res['verdict'],
                    'p_n2a': res.get('p_n2a'),
                    'gate': res['informativeness_gate'],
                    'panel_means': res['panel_means']}, indent=1))


def _mk(rng, task_nz1, task_nz2, apt_nz1, apt_nz2, sd=0.02):
  ridge = {}
  for load, tl, al in (('nz1', task_nz1, apt_nz1),
                       ('nz2', task_nz2, apt_nz2)):
    for side in SIDES:
      for seed in SEEDS:
        ridge[('task', load, side, seed)] = tl + sd * rng.standard_normal()
        ridge[('apt', load, side, seed)] = al + sd * rng.standard_normal()
  return ridge


def selfcheck():
  import tempfile
  # delta-rev F7: gate constants asserted as literals
  assert SIGMA_OF == {'nz1': 1.0, 'nz2': 4.0}, SIGMA_OF
  assert APT_FLOOR == 0.55, APT_FLOOR
  # lambda: apt collapses at high load, task holds
  r = analyse(_mk(np.random.default_rng(0), 0.90, 0.88, 0.70, 0.52))
  assert r['verdict'].startswith('LAMBDA-SUPPORTED'), \
      (r['verdict'], r['p_n2a'])
  # negative DID: task loses more
  r = analyse(_mk(np.random.default_rng(1), 0.90, 0.70, 0.70, 0.68))
  assert r['verdict'].startswith('NEGATIVE-DID'), r['verdict']
  # flat -> indeterminate
  r = analyse(_mk(np.random.default_rng(2), 0.90, 0.85, 0.70, 0.65))
  assert r['verdict'].startswith('INDETERMINATE'), \
      (r['verdict'], r['p_n2a'])
  # apt at chance -> gate fires
  r = analyse(_mk(np.random.default_rng(3), 0.90, 0.88, 0.50, 0.48))
  assert r['verdict'].startswith('APT-FLOOR-CENSORED'), r['verdict']
  # loader round-trip incl. cross-load probeset refusal + config gates
  # (fixtures mirror REAL config shape: model_obs '.*' + a distractor
  # block - reviewer-3 F2: the old obs-space regex gate was vacuous on
  # real configs)
  def _cfg(arm, load, dz=True, basesd=None):
    c = {'agent': {'expl': {'mode': arm}, 'model_obs': '.*'}}
    if dz:
      c['distractor'] = {'dim': 16,
                         'basesd': (basesd if basesd is not None
                                    else SIGMA_OF[load]),
                         'scale': 1.0}
    return c
  with tempfile.TemporaryDirectory() as td:
    ridge = _mk(np.random.default_rng(4), 0.90, 0.88, 0.70, 0.52)
    rdir = os.path.join(td, 'ridge')
    os.makedirs(rdir)
    counters, steps = {}, {}
    for (a, l, side, seed), v in ridge.items():
      rid = wm_name(a, l, side, seed)
      with open(os.path.join(rdir, rid + '.json'), 'w') as f:
        json.dump(dict(run_id=rid, probeset_id=PROBESET_OF[l],
                       reward_override=None, witness_match=True,
                       probe={ALPHA_KEY: {'auroc': v}}), f)
      counters[rid] = dict(update=500000, total=500000)
      steps[rid] = 500000
      os.makedirs(os.path.join(td, rid))
      with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
        json.dump(_cfg(a, l), f)
    check_witness(counters, steps)
    check_configs(td)
    loaded = load_ridge(os.path.join(rdir, '*.json'))
    r = analyse(loaded)
    assert r['verdict'].startswith('LAMBDA-SUPPORTED'), r['verdict']
    # cross-load probeset -> refusal
    rid = wm_name('task', 'nz1', '0', 1)
    with open(os.path.join(rdir, rid + '.json'), 'w') as f:
      json.dump(dict(run_id=rid, probeset_id='finger_nz2_v1',
                     reward_override=None, witness_match=True,
                     probe={ALPHA_KEY: {'auroc': 0.9}}), f)
    try:
      load_ridge(os.path.join(rdir, '*.json'))
      raise SystemExit('expected cross-load probeset refusal')
    except AssertionError:
      pass
    with open(os.path.join(rdir, rid + '.json'), 'w') as f:
      json.dump(dict(run_id=rid, probeset_id='finger_nz1_v1',
                     reward_override=None, witness_match=True,
                     probe={ALPHA_KEY: {'auroc': 0.9}}), f)
    # a REAL-shaped config with NO distractor block (the finding-1
    # failure: injection never reached the model) -> refusal
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_cfg('task', 'nz1', dz=False), f)
    try:
      check_configs(td)
      raise SystemExit('expected missing-distractor refusal')
    except AssertionError:
      pass
    # wrong basesd for the load -> refusal
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_cfg('task', 'nz1', basesd=4.0), f)
    try:
      check_configs(td)
      raise SystemExit('expected basesd-mismatch refusal')
    except AssertionError:
      pass
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_cfg('task', 'nz1'), f)
    check_configs(td)
    # manifest gate (delta-rev F2): sigma mismatch refuses
    m1 = os.path.join(td, 'nz1_manifest.json')
    m2 = os.path.join(td, 'nz2_manifest.json')
    for path, sig in ((m1, 1.0), (m2, 4.0)):
      json.dump({'distractor': {'sigma': sig, 'dims': 16,
                                'holdout_per_side': 16}}, open(path, 'w'))
    check_manifests(m1, m2)
    json.dump({'distractor': {'sigma': 4.0, 'dims': 16,
                              'holdout_per_side': 16}}, open(m1, 'w'))
    try:
      check_manifests(m1, m2)
      raise SystemExit('expected manifest-sigma refusal')
    except AssertionError:
      pass
  print('nuisance_read selfcheck PASS (lambda/negative/indeterminate/'
        'floor-gate branches; loader round-trip w/ cross-load-probeset, '
        'missing-distractor-block, and basesd-mismatch refusals on '
        'real-shaped configs)')


if __name__ == '__main__':
  main()
