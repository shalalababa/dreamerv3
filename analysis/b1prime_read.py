"""Frozen reader — B1' 600k capacity point (capacity-side convergent
test of the sigma-ladder competition mechanism).

Registration: PREREG_b1prime_20260815.md. Authorized by the triggered
14-Aug conditional (sigma-ladder EXCESS-TASK-LOSS fire, user GO 15
Aug). Frozen prediction: as capacity falls (g_k rises) the
marginally-included RESCUED reward-relevant support erodes FIRST —
task-arm reward-legibility declines at 600k while behavior is alive.

Review-driven design facts (ONE-Opus review, all applied):
- The primary is SIDE-STRATIFIED (review B1): the executed decline is
  entirely a side-0 phenomenon (1m/300k/100k task side0 0.903/0.710/
  0.400) while side-1 legibility RISES as capacity falls (0.967/0.984/
  0.994). Each side tests against its OWN anchor distribution.
- The anchor is an ESTIMATE, not a constant (review B3): the primary
  is two_sample(new side fits, the 8 pinned per-run 1m anchor values
  of that side), so anchor uncertainty enters the null.
ONE execution:

  python -m analysis.b1prime_read --auc_csv <auc.csv> \
      --runroot <bundle>/runroot_light --ridge_glob '<bundle>/ridge/*.json' \
      --fit_counters <json> --ckpt_steps <json> --output <dir>
Selfcheck: python -m analysis.b1prime_read --selfcheck
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np

from analysis.capdescent_read import two_sample, _load_yaml, \
    _find_rssm_deters
from analysis.domains_read import one_sample

# ---- pinned executed anchors (value-aware, disclosed in the prereg).
# Per-run size1m task finger_v1 ridge AUROCs from the sha-verified
# bundle local_results/regen_ridge_e4_20260810_220849 (the
# finger_refit_read_20260810 substrate; run order = sorted run_id,
# seeds 1..8 per side). Side means 0.9027096945372575 /
# 0.9668266390482639; pooled 0.9347681667927606 (= the 0.93477 pin in
# analysis/tm2_diag_read.py, verified to the digit). ----
ANCHOR_S0 = (0.9455077864173864, 0.8644163417838757, 0.8829461405809119,
             0.7998172164096323, 0.9257064215058048, 0.9209553304111832,
             0.9431194974099123, 0.9392088217793525)
ANCHOR_S1 = (0.9604135558241971, 0.8934252293923794, 0.9789085822904775,
             0.9789873013009671, 0.9802234758330282, 0.9698871481034743,
             0.9833663533611907, 0.9894014662803972)
APT_LEG_ANCHOR = 0.40107      # pooled apt anchor (below chance;
                              # no-move CONTROL only, never a contrast)
FLOOR_AUC = 113.682           # capdescent behavioral floor constant
CONTROL_MARGIN = 0.10         # apt control band half-width
ALPHA = 0.05                  # step-up BH over the two side tests
ALPHA_KEY = 'alpha_0.001'
PROBESET = 'finger_v1'
DETER = 384                   # size600k rssm deter (config gate)
HEAD_UNITS = 64               # readout64_frozen pins ALL FOUR heads
HEADS = ('policy', 'value', 'rewhead', 'conhead')
ARMS = ('task', 'apt')
SIDES = ('0', '1')
SEEDS = (1, 2, 3, 4)
RID_RE = re.compile(r'^ax1wm_finger_(?P<f>f?)sk6q1s'
                    r'(?P<side>[01])_seed(?P<seed>\d)$')
ADAPT_RE = re.compile(r'^adapt_ax1(?P<f>f?)sk6q1s(?P<side>[01])_finger_'
                      r'seed(?P<seed>\d)_ckpt500000$')


def wm_name(arm, side, seed):
  f = 'f' if arm == 'apt' else ''
  return f'ax1wm_finger_{f}sk6q1s{side}_seed{seed}'


def adapt_name(arm, side, seed):
  f = 'f' if arm == 'apt' else ''
  return f'adapt_ax1{f}sk6q1s{side}_finger_seed{seed}_ckpt500000'


def check_witness(fit_counters, ckpt_steps):
  assert len(fit_counters) == 16 and len(ckpt_steps) == 16, \
      (len(fit_counters), len(ckpt_steps))
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        n = wm_name(arm, side, seed)
        assert n in fit_counters and \
            int(fit_counters[n]['update']) == \
            int(fit_counters[n]['total']), (n, fit_counters.get(n))
        assert n in ckpt_steps and int(ckpt_steps[n]) == 500000, \
            (n, ckpt_steps.get(n))


def check_configs(runroot):
  """Fit configs: EVERY rssm node at deter 384 (the capdescent walk)
  + expl mode by arm. Adapt configs: registered WM from_checkpoint +
  frozen + ALL FOUR readout64 head pins (the B1 mediator repair)."""
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        wm = wm_name(arm, side, seed)
        cfg = _load_yaml(os.path.join(runroot, wm, 'config.yaml'))
        assert str(cfg['agent']['expl']['mode']) == arm, \
            (wm, cfg['agent']['expl']['mode'])
        deters = _find_rssm_deters(cfg)
        assert deters and all(d == DETER for d in deters), \
            (wm, 'rssm deter != 384', deters)
        ad = adapt_name(arm, side, seed)
        acfg = _load_yaml(os.path.join(runroot, ad, 'config.yaml'))
        assert wm in str(acfg['run']['from_checkpoint']), \
            (ad, 'from_checkpoint does not name the registered WM',
             acfg['run']['from_checkpoint'])
        assert bool(acfg['agent']['frozen_wm']) is True, (ad, 'not frozen')
        for h in HEADS:
          assert int(acfg['agent'][h]['units']) == HEAD_UNITS, \
              (ad, 'readout64 pin violated', h, acfg['agent'][h])


def load_ridge(pattern):
  out = {}
  for p in sorted(glob.glob(pattern)):
    with open(p) as f:
      d = json.load(f)
    m = RID_RE.match(d['run_id'])
    assert m, (p, d.get('run_id'))
    assert os.path.basename(p).startswith(d['run_id']), \
        (p, d['run_id'], 'filename does not carry run_id')
    assert str(d['probeset_id']).startswith(PROBESET), \
        (p, 'wrong probeset', d['probeset_id'])
    assert d.get('reward_override') is None, (p, 'override-scored json')
    assert d.get('witness_match') is not False, (p, 'witness_match False')
    key = (('apt' if m.group('f') else 'task'), m.group('side'),
           int(m.group('seed')))
    assert key not in out, ('duplicate ridge json', key)
    v = d['probe'][ALPHA_KEY]['auroc']
    assert isinstance(v, (int, float)) and np.isfinite(v), \
        (p, 'degenerate/absent auroc', v)
    out[key] = float(v)
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        assert (arm, side, seed) in out, ('missing ridge json', arm,
                                          side, seed)
  assert len(out) == 16, len(out)
  return out


def load_auc(path):
  rows = {}
  with open(path) as fh:
    for r in csv.DictReader(fh):
      m = ADAPT_RE.match(r['run_id'])
      if not m:
        continue      # pooled collates carry other waves' rows; the
                      # exact-16 inventory below is the real guard
      assert r['domain'] == 'finger', r
      assert int(float(r['milestone'])) == 500000, r
      assert str(r.get('qc_pass', '1')) in ('True', '1', 'true'), \
          (r['run_id'], 'qc_pass fail')
      key = (('apt' if m.group('f') else 'task'), m.group('side'),
             int(m.group('seed')))
      assert key not in rows, ('duplicate auc row', key)
      rows[key] = dict(auc=float(r['auc100k']),
                       n_ep=int(float(r['n_ep_100k'])))
  for arm in ARMS:
    for side in SIDES:
      for seed in SEEDS:
        assert (arm, side, seed) in rows, ('missing auc row', arm,
                                           side, seed)
  n_eps = {k: v['n_ep'] for k, v in rows.items()}
  modal = int(np.bincount(list(n_eps.values())).argmax())
  off = {k: v for k, v in n_eps.items() if v != modal}
  assert not off, ('non-modal n_ep rows - REFUSE', modal, off)
  return {k: v['auc'] for k, v in rows.items()}, modal


def analyse(ridge, auc):
  leg = {(a, s): [ridge[(a, s, k)] for k in SEEDS]
         for a in ARMS for s in SIDES}
  beh = {a: [auc[(a, s, k)] for s in SIDES for k in SEEDS]
         for a in ARMS}
  anchors = {'0': ANCHOR_S0, '1': ANCHOR_S1}
  res = {'panel_means': {
      'task_leg_s0': float(np.mean(leg[('task', '0')])),
      'task_leg_s1': float(np.mean(leg[('task', '1')])),
      'apt_leg': float(np.mean(leg[('apt', '0')] + leg[('apt', '1')])),
      'task_auc': float(np.mean(beh['task'])),
      'apt_auc': float(np.mean(beh['apt']))},
      'pins': dict(anchor_s0_mean=float(np.mean(ANCHOR_S0)),
                   anchor_s1_mean=float(np.mean(ANCHOR_S1)),
                   apt_leg_anchor=APT_LEG_ANCHOR, floor_auc=FLOOR_AUC)}
  # P-C1 per side: two_sample(new side fits, that side's 8 pinned
  # anchor runs) — anchor uncertainty enters the null (review B3);
  # step-up BH over the two side p's
  side_stats = {}
  for s in SIDES:
    side_stats[s] = two_sample(leg[('task', s)], list(anchors[s]))
  res['p_c1_by_side'] = side_stats
  ranked = sorted(SIDES, key=lambda s: side_stats[s]['perm_p'])
  kmax = 0
  for rank, s in enumerate(ranked, start=1):
    if side_stats[s]['perm_p'] <= ALPHA * rank / 2:
      kmax = rank
  declines, rises = [], []
  for s in ranked[:kmax]:
    if side_stats[s]['ci'][1] < 0:
      declines.append(s)
    elif side_stats[s]['ci'][0] > 0:
      rises.append(s)
  res['fired'] = dict(declines=declines, rises=rises)
  # P-C2: behavioral floor (point rule, the B1 constant)
  c2_pass = float(np.mean(beh['task'])) > FLOOR_AUC
  res['p_c2_floor'] = dict(task_mean=float(np.mean(beh['task'])),
                           floor=FLOOR_AUC, passed=bool(c2_pass))
  # apt no-move control (direction-split, review M1): the anchor is
  # BELOW chance, so movement TOWARD 0.5 is mechanism-consistent
  # attenuation of anti-alignment, not instrument doubt
  apt_all = leg[('apt', '0')] + leg[('apt', '1')]
  ctl = one_sample([v - APT_LEG_ANCHOR for v in apt_all])
  if ctl['ci'][0] > -CONTROL_MARGIN and ctl['ci'][1] < CONTROL_MARGIN:
    ctl_label = 'CONTROL-OK'
  elif ctl['ci'][0] > CONTROL_MARGIN:
    ctl_label = 'CONTROL-DRIFT-TOWARD-CHANCE'   # descriptive, benign
  elif ctl['ci'][1] < -CONTROL_MARGIN:
    ctl_label = 'CONTROL-SUSPECT'               # more anti-predictive
  else:
    ctl_label = 'CONTROL-INDETERMINATE'         # wide CI; note only
  res['apt_control'] = dict(stat=ctl, label=ctl_label,
                            margin=CONTROL_MARGIN)
  # descriptives
  res['interaction_descriptive'] = dict(
      i600k=two_sample(beh['task'], beh['apt']),
      note='descriptive only; executed anchors (value-aware): 1m task '
           '~188.2 / apt ~79.9; 300k task 73.6 (floor-FAIL)')
  mde = {}
  for s in SIDES:
    sd_new = float(np.std(leg[('task', s)], ddof=1))
    sd_anc = float(np.std(anchors[s], ddof=1))
    se = (sd_new ** 2 / 4.0 + sd_anc ** 2 / 8.0) ** 0.5
    mde[s] = dict(sd_new=sd_new, sd_anchor=sd_anc, se=se,
                  mde80=2.8 * se)
  res['mde_note'] = dict(
      per_side=mde,
      note='includes anchor-sample uncertainty (review B3); executed '
           'per-fit sds ran 0.05 (1m) to 0.20 (300k) - under 300k-like '
           'dispersion this wave is unpowered for a sub-collapse '
           'decline (disclosed in the prereg honesty block)')
  # verdict map (registered order; review B1 side qualifiers)
  rise_s0 = '0' in rises
  if rise_s0:
    res['verdict'] = ('LEG-RISE-ANOMALY(side0): side-0 task legibility '
                      'significantly ABOVE its 1m anchor at 600k - '
                      'against the executed capacity series; '
                      'investigate before any wording (floor %s)'
                      % ('passed' if c2_pass else 'FAILED'))
  elif len(declines) == 2 and c2_pass:
    res['verdict'] = ('MECHANISM-CONVERGENT: task reward-legibility '
                      'declines on BOTH sides at 600k while behavior '
                      'is alive - the capacity-side knee-before-floor '
                      'ordering; the competition mechanism gains its '
                      'second independent g_k lever')
  elif len(declines) == 1 and c2_pass:
    s = declines[0]
    res['verdict'] = ('MECHANISM-CONVERGENT-SINGLE-SIDE(side%s): the '
                      'decline is demonstrated on side %s only; any '
                      'mechanism wording MUST carry the side qualifier '
                      '(the executed side-1 series RISES with falling '
                      'capacity - a single-side fire on side 0 is the '
                      'a-priori-likely form, disclosed)' % (s, s))
  elif not declines and c2_pass:
    res['verdict'] = ('KNEE-NOT-REACHED: no decline detected at 600k '
                      'with behavior alive; any legibility knee would '
                      'lie below 600k on the DESCRIPTIVE 300k value '
                      '(side0 0.710, never itself tested against the '
                      '1m anchor); convergence unadjudicated here')
  elif declines and not c2_pass:
    res['verdict'] = ('DECLINE-WITH-CLIFF: legibility declines '
                      '(side(s) %s) AND behavior floors at 600k - '
                      'ordering not establishable at this size; the '
                      'learnability cliff relocates to (600k, 1m]'
                      % ','.join(declines))
  else:
    res['verdict'] = ('DISSOCIATION-REPEATS: behavior floors while '
                      'legibility is not significantly declined - the '
                      'B1-300k shape recurs at 600k; cliff in '
                      '(600k, 1m]; no ordering claim')
  if '1' in rises:
    res['verdict'] += (' | note: side-1 legibility rose (the expected '
                       'extrapolation of the executed 1m/300k/100k '
                       'side-1 series - NOT an anomaly)')
  if ctl_label != 'CONTROL-OK':
    res['verdict'] += ' | ' + ctl_label
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
  assert args.auc_csv and args.runroot and args.ridge_glob \
      and args.fit_counters and args.ckpt_steps and args.output
  check_witness(json.load(open(args.fit_counters)),
                json.load(open(args.ckpt_steps)))
  check_configs(args.runroot)
  auc, modal = load_auc(args.auc_csv)
  res = analyse(load_ridge(args.ridge_glob), auc)
  res['modal_n_ep'] = modal
  res['prereg'] = 'PREREG_b1prime_20260815.md'
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({'verdict': res['verdict'],
                    'p_c1_by_side': res['p_c1_by_side'],
                    'p_c2': res['p_c2_floor'],
                    'apt_control': res['apt_control']['label'],
                    'panel_means': res['panel_means']}, indent=1))


def _mk(rng, t_s0, t_s1, apt_leg, task_auc, apt_auc, sd_leg=0.02,
        sd_auc=20.0):
  ridge, auc = {}, {}
  for s in SIDES:
    t = t_s0 if s == '0' else t_s1
    for k in SEEDS:
      ridge[('task', s, k)] = t + sd_leg * rng.standard_normal()
      ridge[('apt', s, k)] = apt_leg + sd_leg * rng.standard_normal()
      auc[('task', s, k)] = task_auc + sd_auc * rng.standard_normal()
      auc[('apt', s, k)] = apt_auc + sd_auc * rng.standard_normal()
  return ridge, auc


def selfcheck():
  import tempfile
  assert abs(float(np.mean(ANCHOR_S0)) - 0.9027096945372575) < 1e-12
  assert abs(float(np.mean(ANCHOR_S1)) - 0.9668266390482639) < 1e-12
  assert abs(float(np.mean(ANCHOR_S0 + ANCHOR_S1))
             - 0.9347681667927606) < 1e-12, 'pooled pin drifted'
  assert APT_LEG_ANCHOR == 0.40107 and FLOOR_AUC == 113.682
  def _expect_refusal(fn, what):
    try:
      fn()
      raise SystemExit('expected ' + what)
    except AssertionError:
      pass
  # MECHANISM-CONVERGENT: both sides down ~0.09, behavior alive
  r = analyse(*_mk(np.random.default_rng(0), 0.81, 0.875, 0.40,
                   160.0, 80.0))
  assert r['verdict'].startswith('MECHANISM-CONVERGENT:'), \
      (r['verdict'], r['p_c1_by_side'])
  assert r['apt_control']['label'] == 'CONTROL-OK', r['apt_control']
  # SINGLE-SIDE: side0 down, side1 at anchor
  r = analyse(*_mk(np.random.default_rng(1), 0.81, 0.967, 0.40,
                   160.0, 80.0))
  assert r['verdict'].startswith(
      'MECHANISM-CONVERGENT-SINGLE-SIDE(side0)'), r['verdict']
  # KNEE-NOT-REACHED: both at anchors, behavior alive
  r = analyse(*_mk(np.random.default_rng(2), 0.903, 0.967, 0.40,
                   160.0, 80.0))
  assert r['verdict'].startswith('KNEE-NOT-REACHED'), r['verdict']
  # DECLINE-WITH-CLIFF: side0 down, behavior floored
  r = analyse(*_mk(np.random.default_rng(3), 0.75, 0.967, 0.40,
                   90.0, 80.0))
  assert r['verdict'].startswith('DECLINE-WITH-CLIFF'), r['verdict']
  # DISSOCIATION-REPEATS: at anchors, behavior floored
  r = analyse(*_mk(np.random.default_rng(4), 0.903, 0.967, 0.40,
                   90.0, 80.0))
  assert r['verdict'].startswith('DISSOCIATION-REPEATS'), r['verdict']
  # side-0 rise = anomaly
  r = analyse(*_mk(np.random.default_rng(5), 0.98, 0.967, 0.40,
                   160.0, 80.0))
  assert r['verdict'].startswith('LEG-RISE-ANOMALY(side0)'), r['verdict']
  # side-1 rise = benign note, NOT anomaly (review B1.4; planted well
  # above the anchor spread - the fixture value is synthetic)
  r = analyse(*_mk(np.random.default_rng(6), 0.81, 1.06, 0.40,
                   160.0, 80.0, sd_leg=0.004))
  assert not r['verdict'].startswith('LEG-RISE'), r['verdict']
  assert 'NOT an anomaly' in r['verdict'], r['verdict']
  # control direction split (review M1): drift toward chance = benign
  r = analyse(*_mk(np.random.default_rng(7), 0.81, 0.875, 0.52,
                   160.0, 80.0, sd_leg=0.004))
  assert r['apt_control']['label'] == 'CONTROL-DRIFT-TOWARD-CHANCE', \
      r['apt_control']
  assert 'CONTROL-SUSPECT' not in r['verdict'], r['verdict']
  # away from chance = suspect
  r = analyse(*_mk(np.random.default_rng(8), 0.81, 0.875, 0.28,
                   160.0, 80.0, sd_leg=0.004))
  assert r['apt_control']['label'] == 'CONTROL-SUSPECT', r['apt_control']
  assert 'CONTROL-SUSPECT' in r['verdict'], r['verdict']
  # loader round-trip + refusals
  with tempfile.TemporaryDirectory() as td:
    ridge, auc = _mk(np.random.default_rng(9), 0.81, 0.875, 0.40,
                     160.0, 80.0)
    rdir = os.path.join(td, 'ridge')
    os.makedirs(rdir)
    counters, steps = {}, {}

    def _write_json(rid, fname=None, **over):
      rec = dict(run_id=rid, probeset_id=over.pop('probeset_id',
                                                  PROBESET),
                 reward_override=over.pop('reward_override', None),
                 witness_match=over.pop('witness_match', True),
                 probe={ALPHA_KEY: {'auroc': over.pop('auroc', 0.85)}})
      with open(os.path.join(rdir, (fname or rid) + '.json'), 'w') as f:
        json.dump(rec, f)

    def _fit_cfg(arm, deter=DETER):
      return {'agent': {'expl': {'mode': arm},
                        'dyn': {'rssm': {'deter': deter}}}}

    def _adapt_cfg(rid, frozen=True, units=None):
      u = dict.fromkeys(HEADS, HEAD_UNITS)
      if units:
        u.update(units)
      return {'run': {'from_checkpoint': f'/x/{rid}/ckpt/z'},
              'agent': dict({'frozen_wm': frozen},
                            **{h: {'units': u[h]} for h in HEADS})}
    for (a, s, k), v in ridge.items():
      rid = wm_name(a, s, k)
      _write_json(rid, auroc=v)
      counters[rid] = dict(update=500000, total=500000)
      steps[rid] = 500000
      os.makedirs(os.path.join(td, rid))
      with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
        json.dump(_fit_cfg(a), f)
      ad = adapt_name(a, s, k)
      os.makedirs(os.path.join(td, ad))
      with open(os.path.join(td, ad, 'config.yaml'), 'w') as f:
        json.dump(_adapt_cfg(rid), f)
    check_witness(counters, steps)
    check_configs(td)
    acsv = os.path.join(td, 'auc.csv')

    def _write_auc(nep_override=None, qc_override=None, drop=None,
                   dup=False):
      with open(acsv, 'w', newline='') as f:
        w = csv.DictWriter(f, ['run_id', 'domain', 'milestone',
                               'auc100k', 'n_ep_100k', 'qc_pass'])
        w.writeheader()
        for (a, s, k), v in auc.items():
          if drop == (a, s, k):
            continue
          nep = 90 if nep_override == (a, s, k) else 96
          qc = '0' if qc_override == (a, s, k) else '1'
          row = dict(run_id=adapt_name(a, s, k), domain='finger',
                     milestone='500000', auc100k=str(v),
                     n_ep_100k=str(nep), qc_pass=qc)
          w.writerow(row)
          if dup and (a, s, k) == ('task', '0', 1):
            w.writerow(row)
    _write_auc()
    loaded_auc, modal = load_auc(acsv)
    assert modal == 96
    r = analyse(load_ridge(os.path.join(rdir, '*.json')), loaded_auc)
    assert r['verdict'].startswith('MECHANISM-CONVERGENT'), r['verdict']
    _write_auc(nep_override=('task', '0', 1))
    _expect_refusal(lambda: load_auc(acsv), 'sub-modal refusal')
    _write_auc(qc_override=('task', '0', 1))
    _expect_refusal(lambda: load_auc(acsv), 'qc refusal')
    _write_auc(drop=('apt', '1', 4))
    _expect_refusal(lambda: load_auc(acsv), 'missing-row refusal')
    _write_auc(dup=True)
    _expect_refusal(lambda: load_auc(acsv), 'duplicate-row refusal')
    _write_auc()
    load_auc(acsv)
    rid = wm_name('task', '0', 1)
    pat = os.path.join(rdir, '*.json')
    good = ridge[('task', '0', 1)]
    # ridge refusals (review minor 9: full battery)
    _write_json(rid, probeset_id='finger_nzs2_v1')
    _expect_refusal(lambda: load_ridge(pat), 'probeset refusal')
    _write_json(rid, reward_override='dmc_finger_spin')
    _expect_refusal(lambda: load_ridge(pat), 'reward_override refusal')
    _write_json(rid, witness_match=False)
    _expect_refusal(lambda: load_ridge(pat), 'witness_match refusal')
    _write_json(rid, auroc=float('nan'))
    _expect_refusal(lambda: load_ridge(pat), 'nan-auroc refusal')
    _write_json(rid, auroc=good)
    _write_json(rid, fname='renamed_wrong')
    _expect_refusal(lambda: load_ridge(pat), 'rename refusal')
    os.remove(os.path.join(rdir, 'renamed_wrong.json'))
    _write_json(rid, fname=rid + '_copy')
    _expect_refusal(lambda: load_ridge(pat), 'duplicate-ridge refusal')
    os.remove(os.path.join(rdir, rid + '_copy.json'))
    os.rename(os.path.join(rdir, rid + '.json'),
              os.path.join(td, 'stash.json'))
    _expect_refusal(lambda: load_ridge(pat), 'missing-ridge refusal')
    os.rename(os.path.join(td, 'stash.json'),
              os.path.join(rdir, rid + '.json'))
    load_ridge(pat)
    # config refusals: deter walk, expl-mode, from_checkpoint,
    # frozen_wm, each of the four head pins
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_fit_cfg('task', deter=512), f)
    _expect_refusal(lambda: check_configs(td), 'deter refusal')
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_fit_cfg('apt'), f)
    _expect_refusal(lambda: check_configs(td), 'expl-mode refusal')
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_fit_cfg('task'), f)
    ad = adapt_name('task', '0', 1)
    with open(os.path.join(td, ad, 'config.yaml'), 'w') as f:
      json.dump(_adapt_cfg('ax1wm_finger_q1s0_seed1'), f)
    _expect_refusal(lambda: check_configs(td), 'from_checkpoint refusal')
    with open(os.path.join(td, ad, 'config.yaml'), 'w') as f:
      json.dump(_adapt_cfg(rid, frozen=False), f)
    _expect_refusal(lambda: check_configs(td), 'frozen refusal')
    for h in HEADS:
      with open(os.path.join(td, ad, 'config.yaml'), 'w') as f:
        json.dump(_adapt_cfg(rid, units={h: 48}), f)
      _expect_refusal(lambda: check_configs(td), f'{h}-units refusal')
    with open(os.path.join(td, ad, 'config.yaml'), 'w') as f:
      json.dump(_adapt_cfg(rid), f)
    check_configs(td)
    # witness refusals
    counters[rid] = dict(update=400000, total=500000)
    _expect_refusal(lambda: check_witness(counters, steps),
                    'counter refusal')
    counters[rid] = dict(update=500000, total=500000)
    steps[rid] = 450000
    _expect_refusal(lambda: check_witness(counters, steps),
                    'ckpt-step refusal')
    steps[rid] = 500000
    check_witness(counters, steps)
  print('b1prime_read selfcheck PASS (side-stratified two_sample '
        'primary vs pinned anchor runs; both-sides / single-side / '
        'knee-not-reached / decline-with-cliff / dissociation-repeats '
        '/ side0-rise-anomaly / side1-rise-benign-note; control '
        'direction split OK/toward-chance/suspect; refusals: '
        'sub-modal, qc, missing-row, duplicate-row, probeset, '
        'reward_override, witness_match, nan-auroc, rename, '
        'duplicate-ridge, missing-ridge, deter-walk, expl-mode, '
        'from_checkpoint, frozen, all-four-head-pins, counters, '
        'ckpt-step; anchor pins verified to the digit)')


if __name__ == '__main__':
  main()
