"""Frozen reader — sigma-ladder extension of the B2 crowding test.

Registration: PREREG_sigma_ladder_20260814.md. Discriminates the two
accounts of the executed B2 NEGATIVE-DID (`artifacts/
b2_nuisance_read_20260813/`): (a) proportional-headroom compression
(any monotone stressor shrinks each arm toward chance in proportion to
its headroom) vs (b) competition-with-swamped-rescue (the task arm's
reward-relevant support is what crowding excludes, so its NORMALIZED
loss exceeds the reward-free arm's, increasingly with sigma until
joint collapse). Estimand: per new load L in {nzs2 (sigma 2), nzs8
(sigma 8)}, the contrast of HEADROOM-NORMALIZED drops
  n_fit = (BASE_arm - auroc_fit) / (BASE_arm - 0.5)
with BASE_arm PINNED from the executed B2 read (nz1 arm means).
Under (a) the contrast is 0 at every load; under (b) it is positive
past the knee. ONE execution:

  python -m analysis.sigma_ladder_read \
      --ridge_glob '<bundle>/ridge/*.json' --runroot <bundle>/runroot_light \
      --nzs2_manifest <...>/q1_nzs2/manifest.json \
      --nzs8_manifest <...>/q1_nzs8/manifest.json \
      --fit_counters <json> --ckpt_steps <json> --output <dir>
Selfcheck: python -m analysis.sigma_ladder_read --selfcheck
"""

import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np

from analysis.capdescent_read import two_sample, _load_yaml

# ---- pinned constants (executed B2 read, artifacts/
# b2_nuisance_read_20260813/read.json panel_means; value-aware anchors
# disclosed in the prereg honesty block) ----
BASE_TASK = 0.9133080491668023     # task arm mean AUROC at nz1 (sigma 1)
BASE_APT = 0.7180130325674831      # apt arm mean AUROC at nz1
HR_TASK = BASE_TASK - 0.5          # 0.4133... headroom above chance
HR_APT = BASE_APT - 0.5            # 0.2180...
R_HEADROOM = HR_TASK / HR_APT      # 1.8958...
ANCHOR_TASK_NZ2 = 0.679408127270193    # sigma-4 anchors (descriptive
ANCHOR_APT_NZ2 = 0.6805027838421669    # curve only, never in a rule)

FLOOR = 0.55                       # same conventional bar as B2 (0.5
                                   # + 0.05 margin; provenance disclosed)
EQUIV_MARGIN = 0.25                # normalized units; ~ the 80%-power
                                   # MDE on the B2 sd basis (disclosed
                                   # weak at n=8; INDETERMINATE expected
                                   # when truth is null)
ALPHA = 0.05                       # BH over the informative new loads
ALPHA_KEY = 'alpha_0.001'
LOADS = ('nzs2', 'nzs8')
SIGMA_OF = {'nzs2': 2.0, 'nzs8': 8.0}
PROBESET_OF = {'nzs2': 'finger_nzs2_v1', 'nzs8': 'finger_nzs8_v1'}
ARMS = ('task', 'apt')
SIDES = ('0', '1')
SEEDS = (1, 2, 3, 4)
RID_RE = re.compile(r'^ax1wm_finger_(?P<f>f?)(?P<load>nzs[28])q1s'
                    r'(?P<side>[01])_seed(?P<seed>\d)$')


def wm_name(arm, load, side, seed):
  f = 'f' if arm == 'apt' else ''
  return f'ax1wm_finger_{f}{load}q1s{side}_seed{seed}'


# sha256 over the canonicalized holdout_files dict of BOTH executed B2
# buffer manifests (q1_nz1 and q1_nz2 agree; verified 2026-08-14 from
# local_results/b2_nuisance_20260813_213844). The pinned BASE_* means
# were probed on episodes drawn from exactly this holdout set, so the
# NEW loads' buffers must reproduce it — within-pair equality alone is
# nearly vacuous (the rng-0 draw depends only on the source LISTING at
# build time), the cross-era pin is the real invariant. Review B2 fix.
B2_HOLDOUT_SHA = ('0c76043733f0bd9bab13b94dc2d659c4b124b7322'
                  '0fe1f709012390026c0e97a')


def _holdout_sha(hf):
  canon = json.dumps({k: sorted(v) for k, v in sorted(hf.items())},
                     sort_keys=True)
  return hashlib.sha256(canon.encode()).hexdigest()


def check_manifests(nzs2_manifest, nzs8_manifest,
                    expected_sha=B2_HOLDOUT_SHA):
  """Replay-side sigma is the mediator (B2 delta-rev F2 lineage). The
  matched-probe-experience invariant the PRIMARY needs is that the new
  loads' holdout episodes are THE SAME set the pinned B2 baselines
  were probed on — gated by B2_HOLDOUT_SHA (cross-era), plus equality
  across the two new loads and the registered build parameters."""
  ms = {}
  for load, path in (('nzs2', nzs2_manifest), ('nzs8', nzs8_manifest)):
    with open(path) as f:
      m = json.load(f)
    dz = m['distractor']
    assert float(dz['sigma']) == SIGMA_OF[load], (path, dz['sigma'])
    assert int(dz['dims']) == 16, (path, dz['dims'])
    assert int(dz['holdout_per_side']) == 16, (path, dz['holdout_per_side'])
    assert dz.get('tool') == 'nuisance_replay_v1', (path, dz.get('tool'))
    ms[load] = dz
  assert ms['nzs2'].get('source') == ms['nzs8'].get('source'), \
      ('the two builds consumed different --input sources',
       ms['nzs2'].get('source'), ms['nzs8'].get('source'))
  h2, h8 = ms['nzs2'].get('holdout_files'), ms['nzs8'].get('holdout_files')
  assert h2 is not None and h2 == h8, \
      ('holdout filename sets differ across the two new loads', h2, h8)
  assert _holdout_sha(h2) == expected_sha, \
      ('holdout set differs from the B2 era the baselines are pinned '
       'to - the q1 source listing changed since 12 Aug; the '
       'normalized-drop estimand is contaminated - REFUSE',
       _holdout_sha(h2), expected_sha)


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


def check_configs(runroot):
  """Gate on the RECORDED knob (B2 reviewer-3 F2 form) — dim, the
  load's basesd, AND theta == 1.0 (the dzs configs emit i.i.d.
  channels; an AR(1) default would mismatch the replay side)."""
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
          assert float(dz.get('theta', -1)) == 1.0, \
              (wm, 'distractor.theta != 1.0 - env/replay temporal '
               'statistics mismatched', dz)


def load_ridge(pattern):
  out = {}
  for p in sorted(glob.glob(pattern)):
    with open(p) as f:
      d = json.load(f)
    m = RID_RE.match(d['run_id'])
    assert m, (p, d.get('run_id'))
    assert os.path.basename(p).startswith(d['run_id']), \
        (p, d['run_id'], 'filename does not carry run_id')
    load = m.group('load')
    assert str(d['probeset_id']).startswith(PROBESET_OF[load]), \
        (p, 'cross-load probeset', d['probeset_id'], load)
    assert d.get('reward_override') is None, (p, 'override-scored json')
    assert d.get('witness_match') is not False, (p, 'witness_match False')
    key = (('apt' if m.group('f') else 'task'), load, m.group('side'),
           int(m.group('seed')))
    assert key not in out, ('duplicate ridge json', key)
    v = d['probe'][ALPHA_KEY]['auroc']
    # review m8: a degenerate probe writes null (TypeError, unnamed)
    # and a literal NaN round-trips json - both refuse by name
    assert isinstance(v, (int, float)) and np.isfinite(v), \
        (p, 'degenerate/absent auroc', v)
    out[key] = float(v)
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
  raw = {f'{a}_{l}': float(np.mean(groups[(a, l)]))
         for a in ARMS for l in LOADS}
  res = {'panel_means': raw,
         'pins': dict(base_task=BASE_TASK, base_apt=BASE_APT,
                      hr_task=HR_TASK, hr_apt=HR_APT,
                      r_headroom=R_HEADROOM)}
  # descriptive 4-point curve (sigma 1 and 4 are PINNED executed
  # anchors, value-aware, never in a rule)
  curve = {'1.0': dict(task=BASE_TASK, apt=BASE_APT, s=0.0,
                       source='pinned (executed B2 nz1)'),
           '4.0': dict(task=ANCHOR_TASK_NZ2, apt=ANCHOR_APT_NZ2,
                       s=float((BASE_TASK - ANCHOR_TASK_NZ2)
                               - R_HEADROOM
                               * (BASE_APT - ANCHOR_APT_NZ2)),
                       source='pinned (executed B2 nz2)')}
  for load in LOADS:
    t, a = raw[f'task_{load}'], raw[f'apt_{load}']
    curve[str(SIGMA_OF[load])] = dict(
        task=t, apt=a,
        s=float((BASE_TASK - t) - R_HEADROOM * (BASE_APT - a)),
        source='this wave')
  res['p_s2_curve'] = curve
  # per-load floor classification + normalized contrast
  per_load, informative = {}, []
  for load in LOADS:
    t, a = raw[f'task_{load}'], raw[f'apt_{load}']
    if t < FLOOR and a < FLOOR:
      status = 'JOINT-COLLAPSE'
    elif t < FLOOR <= a:
      status = 'TASK-COLLAPSE-APT-RETAINED'
    elif a < FLOOR <= t:
      status = 'APT-COLLAPSE-TASK-RETAINED'
    else:
      status = 'INFORMATIVE'
      informative.append(load)
    nt = [(BASE_TASK - v) / HR_TASK for v in groups[('task', load)]]
    na = [(BASE_APT - v) / HR_APT for v in groups[('apt', load)]]
    per_load[load] = dict(status=status, task_mean=t, apt_mean=a,
                          norm_task_mean=float(np.mean(nt)),
                          norm_apt_mean=float(np.mean(na)),
                          contrast=two_sample(nt, na))
  res['per_load'] = per_load
  # per-load MDE note (review m11: RMS across the two heteroscedastic
  # arms, not the arithmetic mean of sds)
  mde = {}
  for load in LOADS:
    st = float(np.std([(BASE_TASK - v) / HR_TASK
                       for v in groups[('task', load)]], ddof=1))
    sa = float(np.std([(BASE_APT - v) / HR_APT
                       for v in groups[('apt', load)]], ddof=1))
    se = (st ** 2 / 8.0 + sa ** 2 / 8.0) ** 0.5
    mde[load] = dict(sd_task=st, sd_apt=sa, se=se, mde80=2.8 * se)
  res['mde_note'] = dict(
      per_load=mde,
      note='pinned-anchor se (~0.031/0.059 normalized) not propagated '
           '- disclosed')
  # step-up BH over the informative loads' permutation p's (m = count;
  # review M4: reject ranked[:kmax], kmax = max rank with
  # p_(r) <= alpha*r/m, then classify by CI sign)
  m = len(informative)
  fired_pos, fired_neg = [], []
  if m:
    ranked = sorted(informative,
                    key=lambda l: per_load[l]['contrast']['perm_p'])
    kmax = 0
    for rank, load in enumerate(ranked, start=1):
      per_load[load]['bh_threshold'] = ALPHA * rank / m
      if per_load[load]['contrast']['perm_p'] <= ALPHA * rank / m:
        kmax = rank
    for load in ranked[:kmax]:
      c = per_load[load]['contrast']
      if c['ci'][0] > 0:
        fired_pos.append(load)
      elif c['ci'][1] < 0:
        fired_neg.append(load)
  collapse_pos = [l for l in LOADS
                  if per_load[l]['status'] == 'TASK-COLLAPSE-APT-RETAINED']
  collapse_neg = [l for l in LOADS
                  if per_load[l]['status'] == 'APT-COLLAPSE-TASK-RETAINED']
  res['fired'] = dict(positive=fired_pos, negative=fired_neg,
                      collapse_positive=collapse_pos,
                      collapse_negative=collapse_neg)
  # verdict (registered order). Review B1: the task-side collapse
  # branch is (a)-IMPOSSIBLE under the pins (task<0.55<=apt forces
  # normalized task drop > 0.879 > 0.771 >= normalized apt drop) so it
  # fires; the apt-side collapse is the proportional null's own
  # mid-collapse signature (any common retention f in (.121, .229))
  # and is REPORTED ONLY - it triggers nothing.
  pos = bool(fired_pos or collapse_pos)
  neg = bool(fired_neg)
  if all(per_load[l]['status'] == 'JOINT-COLLAPSE' for l in LOADS):
    res['verdict'] = ('INSTRUMENT-LIMITED: both new loads JOINT-COLLAPSE '
                      '(both arms at/below the floor) - the proportional '
                      'null is trivially satisfied at total collapse; '
                      'no discrimination; panels reported')
  elif pos and neg:
    res['verdict'] = ('MIXED-DIRECTIONS: excess-task-loss and '
                      'excess-apt-loss each fire at some load - outside '
                      'both accounts as registered; facts reported, no '
                      'further wording')
  elif pos:
    res['verdict'] = ('EXCESS-TASK-LOSS: the task arm loses MORE than '
                      'headroom-proportional at ' +
                      ','.join(fired_pos + collapse_pos) +
                      ' - the competition-with-swamped-rescue account '
                      'is supported; proportional-compression is '
                      'rejected at those loads')
  elif neg:
    res['verdict'] = ('EXCESS-APT-LOSS: the reward-free arm loses more '
                      'than headroom-proportional at ' +
                      ','.join(fired_neg) +
                      ' - the originally-frozen lambda direction, at '
                      'sigma levels where B2 found the opposite; fact '
                      'reported, no further wording')
  elif m and all(
      per_load[l]['contrast']['ci'][0] > -EQUIV_MARGIN
      and per_load[l]['contrast']['ci'][1] < EQUIV_MARGIN
      for l in informative):
    res['verdict'] = ('PROPORTIONAL-CONSISTENT: the informative '
                      'load(s) [' + ','.join(informative) + '] are '
                      'null with the whole CI inside +-%.2f - the '
                      'compression account suffices at those loads'
                      % EQUIV_MARGIN)
  else:
    res['verdict'] = 'INDETERMINATE (MDE recorded; licenses nothing)'
  return res


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--ridge_glob')
  ap.add_argument('--runroot')
  ap.add_argument('--nzs2_manifest')
  ap.add_argument('--nzs8_manifest')
  ap.add_argument('--fit_counters')
  ap.add_argument('--ckpt_steps')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.ridge_glob and args.runroot and args.nzs2_manifest \
      and args.nzs8_manifest and args.fit_counters \
      and args.ckpt_steps and args.output
  check_manifests(args.nzs2_manifest, args.nzs8_manifest)
  check_witness(json.load(open(args.fit_counters)),
                json.load(open(args.ckpt_steps)))
  check_configs(args.runroot)
  res = analyse(load_ridge(args.ridge_glob))
  res['prereg'] = 'PREREG_sigma_ladder_20260814.md'
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({'verdict': res['verdict'],
                    'per_load': res['per_load'],
                    'panel_means': res['panel_means']}, indent=1))


def _mk(rng, means, sd=0.02):
  """means: {(arm, load): level}."""
  ridge = {}
  for (a, l), mu in means.items():
    for side in SIDES:
      for seed in SEEDS:
        ridge[(a, l, side, seed)] = mu + sd * rng.standard_normal()
  return ridge


def selfcheck():
  import tempfile
  # literal gate constants (B2 delta-rev F7 lineage)
  assert SIGMA_OF == {'nzs2': 2.0, 'nzs8': 8.0}, SIGMA_OF
  assert FLOOR == 0.55 and EQUIV_MARGIN == 0.25, (FLOOR, EQUIV_MARGIN)
  assert abs(R_HEADROOM - HR_TASK / HR_APT) < 1e-12
  # (b) signature: task loses far more than headroom-proportional at
  # sigma 8, proportional-neutral at sigma 2 (task 0.89 and apt 0.7057
  # are the SAME 0.0564 normalized drop) -> EXCESS-TASK-LOSS via the
  # contrast
  r = analyse(_mk(np.random.default_rng(1),
                  {('task', 'nzs2'): 0.89, ('apt', 'nzs2'): 0.7057,
                   ('task', 'nzs8'): 0.66, ('apt', 'nzs8'): 0.67},
                  sd=0.008))
  assert r['verdict'].startswith('EXCESS-TASK-LOSS'), \
      (r['verdict'], r['per_load'])
  assert 'nzs8' in r['fired']['positive'], r['fired']
  # collapse configuration: task at chance, apt retained -> fires via
  # the branch even with the contrast route unavailable at that load
  r = analyse(_mk(np.random.default_rng(101),
                  {('task', 'nzs2'): 0.89, ('apt', 'nzs2'): 0.7057,
                   ('task', 'nzs8'): 0.51, ('apt', 'nzs8'): 0.68},
                  sd=0.008))
  assert r['verdict'].startswith('EXCESS-TASK-LOSS'), r['verdict']
  assert r['fired']['collapse_positive'] == ['nzs8'], r['fired']
  # (a) signature: normalized drops equal at both loads, tiny sd ->
  # PROPORTIONAL-CONSISTENT (norm drop 0.2 both arms at nzs2, 0.5 at
  # nzs8: task 0.9133-0.2*0.4133=0.8306 / apt 0.7180-0.2*0.2180=0.6744;
  # task 0.7067 / apt 0.6090)
  r = analyse(_mk(np.random.default_rng(2),
                  {('task', 'nzs2'): 0.8306, ('apt', 'nzs2'): 0.6744,
                   ('task', 'nzs8'): 0.7067, ('apt', 'nzs8'): 0.6090},
                  sd=0.005))
  assert r['verdict'].startswith('PROPORTIONAL-CONSISTENT'), \
      (r['verdict'], r['per_load'])
  # joint collapse at both loads -> INSTRUMENT-LIMITED
  r = analyse(_mk(np.random.default_rng(3),
                  {('task', 'nzs2'): 0.52, ('apt', 'nzs2'): 0.51,
                   ('task', 'nzs8'): 0.50, ('apt', 'nzs8'): 0.50}))
  assert r['verdict'].startswith('INSTRUMENT-LIMITED'), r['verdict']
  # apt loses more -> EXCESS-APT-LOSS via the STATISTICAL route only
  # (both loads informative: apt 0.62/0.56 both above the floor)
  r = analyse(_mk(np.random.default_rng(4),
                  {('task', 'nzs2'): 0.89, ('apt', 'nzs2'): 0.62,
                   ('task', 'nzs8'): 0.87, ('apt', 'nzs8'): 0.56},
                  sd=0.008))
  assert r['verdict'].startswith('EXCESS-APT-LOSS'), \
      (r['verdict'], r['per_load'])
  # review B1: the apt-side collapse configuration is the proportional
  # null's own mid-collapse signature (common retention f in
  # (.121,.229) produces exactly this) - it must be REPORTED but must
  # NOT fire a directional verdict. Planted from the exact null at
  # f=0.20: task = 0.5+0.2*HR_T = 0.5827, apt = 0.5+0.2*HR_A = 0.5436.
  r = analyse(_mk(np.random.default_rng(7),
                  {('task', 'nzs2'): 0.8306, ('apt', 'nzs2'): 0.6744,
                   ('task', 'nzs8'): 0.5827, ('apt', 'nzs8'): 0.5436},
                  sd=0.008))
  assert not r['verdict'].startswith(('EXCESS-APT-LOSS', 'EXCESS-TASK',
                                      'MIXED')), \
      (r['verdict'], r['per_load'])
  assert r['fired']['collapse_negative'] == ['nzs8'], r['fired']
  assert r['fired']['negative'] == [], r['fired']
  # null with realistic noise -> INDETERMINATE (CI wider than the
  # equivalence margin at n=8, sd 0.05 raw)
  r = analyse(_mk(np.random.default_rng(5),
                  {('task', 'nzs2'): 0.8306, ('apt', 'nzs2'): 0.6744,
                   ('task', 'nzs8'): 0.7067, ('apt', 'nzs8'): 0.6090},
                  sd=0.05))
  assert r['verdict'].startswith(('INDETERMINATE',
                                  'PROPORTIONAL-CONSISTENT')), r['verdict']
  # the curve carries all four sigmas with the two pinned anchors
  assert set(r['p_s2_curve']) == {'1.0', '2.0', '4.0', '8.0'}
  assert r['p_s2_curve']['4.0']['s'] > 0.15  # the executed S(4) anchor
  # loader round-trip + refusal legs on REAL-shaped fixtures
  def _cfg(arm, load, dz=True, basesd=None, theta=1.0):
    c = {'agent': {'expl': {'mode': arm}, 'model_obs': '.*'}}
    if dz:
      c['distractor'] = {'dim': 16,
                         'basesd': (basesd if basesd is not None
                                    else SIGMA_OF[load]),
                         'scale': 1.0, 'theta': theta}
    return c
  def _expect_refusal(fn, what):
    try:
      fn()
      raise SystemExit('expected ' + what)
    except AssertionError:
      pass
  with tempfile.TemporaryDirectory() as td:
    # same distribution as the proven leg-1 fixture (fresh rng(1) ->
    # identical draws -> identical verdict, deterministic)
    ridge = _mk(np.random.default_rng(1),
                {('task', 'nzs2'): 0.89, ('apt', 'nzs2'): 0.7057,
                 ('task', 'nzs8'): 0.66, ('apt', 'nzs8'): 0.67},
                sd=0.008)
    rdir = os.path.join(td, 'ridge')
    os.makedirs(rdir)
    counters, steps = {}, {}

    def _write_json(rid, fname=None, **over):
      rec = dict(run_id=rid, probeset_id=over.pop(
          'probeset_id', PROBESET_OF[RID_RE.match(rid).group('load')]),
          reward_override=None, witness_match=True,
          probe={ALPHA_KEY: {'auroc': over.pop('auroc', 0.9)}})
      rec.update(over)
      with open(os.path.join(rdir, (fname or rid) + '.json'), 'w') as f:
        json.dump(rec, f)
    for (a, l, side, seed), v in ridge.items():
      rid = wm_name(a, l, side, seed)
      _write_json(rid, auroc=v)
      counters[rid] = dict(update=500000, total=500000)
      steps[rid] = 500000
      os.makedirs(os.path.join(td, rid))
      with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
        json.dump(_cfg(a, l), f)
    check_witness(counters, steps)
    check_configs(td)
    r = analyse(load_ridge(os.path.join(rdir, '*.json')))
    assert r['verdict'].startswith('EXCESS-TASK-LOSS'), r['verdict']
    rid = wm_name('task', 'nzs2', '0', 1)
    good_auroc = ridge[('task', 'nzs2', '0', 1)]
    pat = os.path.join(rdir, '*.json')
    # cross-load probeset -> refusal
    _write_json(rid, probeset_id='finger_nzs8_v1')
    _expect_refusal(lambda: load_ridge(pat), 'cross-load probeset refusal')
    # witness_match False -> refusal
    _write_json(rid, witness_match=False)
    _expect_refusal(lambda: load_ridge(pat), 'witness_match refusal')
    # reward_override present -> refusal (m10)
    _write_json(rid, reward_override='dmc_finger_spin')
    _expect_refusal(lambda: load_ridge(pat), 'reward_override refusal')
    # NaN auroc -> refusal (m8)
    _write_json(rid, auroc=float('nan'))
    _expect_refusal(lambda: load_ridge(pat), 'nan-auroc refusal')
    # unmatched run_id (the seed99 smoke shape) -> refusal (m10)
    _write_json(rid)  # restore good record first
    smoke = 'ax1wm_finger_nzs8q1s0_seed99'
    with open(os.path.join(rdir, smoke + '.json'), 'w') as f:
      json.dump(dict(run_id=smoke, probeset_id='finger_nzs8_v1',
                     reward_override=None, witness_match=True,
                     probe={ALPHA_KEY: {'auroc': 0.9}}), f)
    _expect_refusal(lambda: load_ridge(pat), 'seed99/unmatched refusal')
    os.remove(os.path.join(rdir, smoke + '.json'))
    # filename does not carry run_id -> refusal (m10)
    _write_json(rid, fname='renamed_wrong')
    _expect_refusal(lambda: load_ridge(pat), 'rename refusal')
    os.remove(os.path.join(rdir, 'renamed_wrong.json'))
    # duplicate run_id under a second filename -> refusal (m10)
    _write_json(rid, fname=rid + '_copy')
    _expect_refusal(lambda: load_ridge(pat), 'duplicate refusal')
    os.remove(os.path.join(rdir, rid + '_copy.json'))
    # missing json -> refusal (m10)
    os.rename(os.path.join(rdir, rid + '.json'),
              os.path.join(td, 'stash.json'))
    _expect_refusal(lambda: load_ridge(pat), 'missing-json refusal')
    os.rename(os.path.join(td, 'stash.json'),
              os.path.join(rdir, rid + '.json'))
    _write_json(rid, auroc=good_auroc)
    load_ridge(pat)
    # wrong theta (AR(1) default leaked in) -> refusal
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_cfg('task', 'nzs2', theta=0.1), f)
    _expect_refusal(lambda: check_configs(td), 'theta refusal')
    # wrong basesd -> refusal
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_cfg('task', 'nzs2', basesd=8.0), f)
    _expect_refusal(lambda: check_configs(td), 'basesd refusal')
    # missing distractor block entirely -> refusal (m10)
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_cfg('task', 'nzs2', dz=False), f)
    _expect_refusal(lambda: check_configs(td), 'missing-distractor refusal')
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_cfg('task', 'nzs2'), f)
    check_configs(td)
    # witness update != total -> refusal; wrong ckpt step -> refusal (m10)
    counters[rid] = dict(update=400000, total=500000)
    _expect_refusal(lambda: check_witness(counters, steps),
                    'counter refusal')
    counters[rid] = dict(update=500000, total=500000)
    steps[rid] = 450000
    _expect_refusal(lambda: check_witness(counters, steps),
                    'ckpt-step refusal')
    steps[rid] = 500000
    check_witness(counters, steps)
    # manifest gates at the REGISTERED shape (16/side, tool, source);
    # fixture sha passed explicitly - the real path uses the pinned
    # B2_HOLDOUT_SHA default
    m2 = os.path.join(td, 'nzs2_manifest.json')
    m8 = os.path.join(td, 'nzs8_manifest.json')
    hold = {'side0': ['h%02d.npz' % i for i in range(16)],
            'side1': ['g%02d.npz' % i for i in range(16)]}
    base = dict(dims=16, holdout_per_side=16, holdout_files=hold,
                tool='nuisance_replay_v1', source='/scratch/q1')
    for path, sig in ((m2, 2.0), (m8, 8.0)):
      json.dump({'distractor': dict(base, sigma=sig)}, open(path, 'w'))
    fx_sha = _holdout_sha(hold)
    check_manifests(m2, m8, expected_sha=fx_sha)
    # cross-era pin: toy set vs the pinned B2 sha -> refusal (review B2)
    _expect_refusal(lambda: check_manifests(m2, m8),
                    'B2-era holdout-sha refusal')
    # sigma mismatch -> refusal
    json.dump({'distractor': dict(base, sigma=4.0)}, open(m2, 'w'))
    _expect_refusal(lambda: check_manifests(m2, m8, expected_sha=fx_sha),
                    'manifest-sigma refusal')
    # holdout-set mismatch across the two new loads -> refusal
    json.dump({'distractor': dict(base, sigma=2.0,
                                  holdout_files={'side0': ['x.npz'],
                                                 'side1': ['y.npz']})},
              open(m2, 'w'))
    _expect_refusal(lambda: check_manifests(m2, m8, expected_sha=fx_sha),
                    'holdout-mismatch refusal')
    # wrong source -> refusal (m9)
    json.dump({'distractor': dict(base, sigma=2.0,
                                  source='/scratch/other')}, open(m2, 'w'))
    _expect_refusal(lambda: check_manifests(m2, m8, expected_sha=fx_sha),
                    'source-mismatch refusal')
  print('sigma_ladder_read selfcheck PASS (excess-task-loss via contrast '
        'AND via task-collapse branch; excess-apt-loss statistical route; '
        'apt-collapse REPORTED-ONLY on exact-null data (review B1); '
        'proportional-consistent; joint-collapse instrument-limited; '
        'indeterminate; 4-sigma curve w/ pinned anchors; step-up BH; '
        'refusals: cross-load probeset, witness_match, reward_override, '
        'nan-auroc, seed99/unmatched, rename, duplicate, missing-json, '
        'theta, basesd, missing-distractor, counters, ckpt-step, '
        'manifest sigma, B2-era holdout sha, holdout-mismatch, source; '
        'literal gate constants)')


if __name__ == '__main__':
  main()
