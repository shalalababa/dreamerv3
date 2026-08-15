"""Frozen reader — rescue-under-load wave (the a2-term boundary test).

Registration: PREREG_rescue_load_20260815.md. Discriminates the two
accounts of the s x w_r inert-a2 tension using the sigma-ladder's
knee: at load sigma=2 (just past the fitted rescued-support knee
~1.9), does scaling the reward-head loss weight w_r 1 -> 100 RESTORE
task-arm reward-legibility? PAIRED design: both cells fresh in-wave
with shared (side, seed) labels, so the per-pair delta kills the
side stratification. ONE execution:

  python -m analysis.rescue_load_read \
      --ridge_glob '<bundle>/ridge/*.json' --runroot <bundle>/runroot_light \
      --nzs2_manifest <...>/q1_nzs2/manifest.json \
      --linkage <bundle>/ruw_replay_linkage.json \
      --fit_counters <json> --ckpt_steps <json> --output <dir>
Selfcheck: python -m analysis.rescue_load_read --selfcheck
"""

import argparse
import glob
import json
import os
import re

import numpy as np

from analysis.capdescent_read import two_sample, _load_yaml
from analysis.domains_read import one_sample
from analysis.sigma_ladder_read import B2_HOLDOUT_SHA, _holdout_sha

# ---- pinned executed anchors (value-aware, disclosed in the prereg):
# the sigma-ladder's 8 task nzs2 per-fit AUROCs (run order = sorted
# run_id: s0 seeds 1-4, then s1 seeds 1-4). Used ONLY by the
# cross-hardware replication DESCRIPTIVE, never by the primary. ----
SGL_NZS2_TASK = (0.7410181706317724, 0.6660056558422437,
                 0.7569922006012847, 0.6725447532664371,
                 0.6942984066490587, 0.9541815810830789,
                 0.7531423351784428, 0.7953294076776419)
ALPHA_KEY = 'alpha_0.001'
PROBESET = 'finger_nzs2_v1'
SIGMA = 2.0
WR_OF = {'1': 1.0, '100': 100.0}
SIDES = ('0', '1')
SEEDS = (1, 2, 3, 4)
RID_RE = re.compile(r'^ax1wm_finger_ruw(?P<w>1|100)q1s'
                    r'(?P<side>[01])_seed(?P<seed>\d)$')


def wm_name(w, side, seed):
  return f'ax1wm_finger_ruw{w}q1s{side}_seed{seed}'


def check_manifest(path):
  with open(path) as f:
    m = json.load(f)
  assert 'distractor' in m, (path, 'manifest has no distractor block')
  dz = m['distractor']
  assert str(dz.get('source', '')).rstrip('/').endswith(
      'axis1_finger/q1'), (path, 'wrong --input source', dz.get('source'))
  assert float(dz['sigma']) == SIGMA, (path, dz['sigma'])
  assert int(dz['dims']) == 16, (path, dz['dims'])
  assert int(dz['holdout_per_side']) == 16, (path, dz['holdout_per_side'])
  assert dz.get('tool') == 'nuisance_replay_v1', (path, dz.get('tool'))
  hf = dz.get('holdout_files')
  assert hf is not None and _holdout_sha(hf) == B2_HOLDOUT_SHA, \
      ('holdout set differs from the pinned B2 era - the probeset '
       'lineage the executed anchors rest on is broken - REFUSE', path)


def check_linkage(path):
  """Review B1: the ONLY witness binding the fits to the sigma=2
  buffer - config.yaml records the env token, never --static_replay,
  so a _nzs8 mis-submission would pass every other gate. The read is
  REFUSED if this file is absent or any row points elsewhere."""
  with open(path) as f:
    lk = json.load(f)
  assert len(lk) == 16, ('linkage must carry exactly 16 rows', len(lk))
  for w in WR_OF:
    for side in SIDES:
      for seed in SEEDS:
        n = wm_name(w, side, seed)
        assert n in lk, ('linkage missing run', n)
        p = str(lk[n]).rstrip('/')
        assert p.endswith(f'q1_nzs2/side{side}'), \
            (n, 'fit consumed the WRONG buffer - REFUSE', p)


def check_witness(fit_counters, ckpt_steps):
  assert len(fit_counters) == 16 and len(ckpt_steps) == 16, \
      (len(fit_counters), len(ckpt_steps))
  for w in WR_OF:
    for side in SIDES:
      for seed in SEEDS:
        n = wm_name(w, side, seed)
        assert n in fit_counters and \
            int(fit_counters[n]['update']) == \
            int(fit_counters[n]['total']), (n, fit_counters.get(n))
        assert n in ckpt_steps and int(ckpt_steps[n]) == 500000, \
            (n, ckpt_steps.get(n))


def check_configs(runroot):
  """Every fit: expl task (AXIS1_WR is task-mode-only), the dzs2
  distractor block, and loss_scales.rew == the cell's w_r — the knob
  the whole wave is about, gated on the RECORDED value."""
  for w in WR_OF:
    for side in SIDES:
      for seed in SEEDS:
        wm = wm_name(w, side, seed)
        cfg = _load_yaml(os.path.join(runroot, wm, 'config.yaml'))
        assert str(cfg['agent']['expl']['mode']) == 'task', \
            (wm, cfg['agent']['expl']['mode'])
        dz = cfg.get('distractor') or {}
        assert int(dz.get('dim', 0)) == 16, (wm, 'distractor.dim', dz)
        assert float(dz.get('basesd', -1)) == SIGMA, (wm, 'basesd', dz)
        assert float(dz.get('theta', -1)) == 1.0, (wm, 'theta', dz)
        rew = float(cfg['agent']['loss_scales']['rew'])
        assert rew == WR_OF[w], (wm, 'loss_scales.rew != cell w_r', rew)


def load_ridge(pattern):
  out = {}
  for p in sorted(glob.glob(pattern)):
    with open(p) as f:
      d = json.load(f)
    m = RID_RE.match(d['run_id'])
    assert m, (p, d.get('run_id'))
    assert os.path.basename(p).startswith(d['run_id']), \
        (p, d['run_id'], 'filename does not carry run_id')
    assert str(d['probeset_id']) == PROBESET, \
        (p, 'wrong probeset', d['probeset_id'])
    assert d.get('reward_override') is None, (p, 'override-scored json')
    assert d.get('witness_match') is not False, (p, 'witness_match False')
    key = (m.group('w'), m.group('side'), int(m.group('seed')))
    assert key not in out, ('duplicate ridge json', key)
    v = (d.get('probe') or {}).get(ALPHA_KEY, {}).get('auroc')
    assert isinstance(v, (int, float)) and np.isfinite(v), \
        (p, 'degenerate/absent auroc', v)
    out[key] = float(v)
  for w in WR_OF:
    for side in SIDES:
      for seed in SEEDS:
        assert (w, side, seed) in out, ('missing ridge json', w, side,
                                        seed)
  assert len(out) == 16, len(out)
  return out


def analyse(ridge):
  cells = {w: [ridge[(w, s, k)] for s in SIDES for k in SEEDS]
           for w in WR_OF}
  deltas = [ridge[('100', s, k)] - ridge[('1', s, k)]
            for s in SIDES for k in SEEDS]
  res = {'panel_means': {
      'w1': float(np.mean(cells['1'])), 'w100': float(np.mean(cells['100'])),
      'w1_s0': float(np.mean([ridge[('1', '0', k)] for k in SEEDS])),
      'w1_s1': float(np.mean([ridge[('1', '1', k)] for k in SEEDS])),
      'w100_s0': float(np.mean([ridge[('100', '0', k)] for k in SEEDS])),
      'w100_s1': float(np.mean([ridge[('100', '1', k)] for k in SEEDS]))}}
  # P-R1 (PRIMARY): paired per-(side,seed) deltas, exact sign-flip
  st = one_sample(deltas)
  res['p_r1_paired'] = st
  sd = float(np.std(deltas, ddof=1))
  # review M3: 3.4 = the simulated multiplier for the registered
  # CONJUNCTION (BCa CI>0 AND exact sign-flip p<.05) at n=8 - the
  # normal-approx 2.8 understates it by ~19%
  mde80 = 3.4 * sd / 8 ** 0.5
  res['per_fit'] = {'|'.join(map(str, k)): v for k, v in ridge.items()}
  res['deltas'] = [float(d) for d in deltas]
  res['mde_note'] = ('paired-delta sd %.4f; se %.4f at n=8; 80%%-power '
                     'MDE %.4f for the registered conjunction (full-'
                     'rescue reference +0.16 detectable at ~87%%; half '
                     'rescue ~0.08 at ~35%% - NOT detectable; pairing '
                     'removes only the side fixed effect, ~8%% of '
                     'variance on the pinned basis)'
                     % (sd, sd / 8 ** 0.5, mde80))
  if st['ci'][0] > 0 and st['perm_p'] < 0.05:
    res['verdict'] = ('RESCUE-CONFIRMED: w_r=100 restores task '
                      'reward-legibility under load - the a2-rescue '
                      'magnitude binds AT the inclusion boundary; the '
                      's x w_r inertness is reconciled as '
                      'off-boundary saturation, the loss-weight lever '
                      'DOES reach the spectrum, and beta becomes an '
                      'experimentally manipulable parameter')
  elif st['ci'][1] < 0 and st['perm_p'] < 0.05:
    res['verdict'] = ('RESCUE-INVERTED: w_r=100 REDUCES legibility '
                      'under load - outside both registered accounts; '
                      'fact reported, no further wording')
  else:
    res['verdict'] = ('NO-RESCUE-DETECTED: the loss-weight lever does '
                      'not move the inclusion boundary at this dose - '
                      'favors the gradient-geometric reading of a2 '
                      'AGAINST FULL-MAGNITUDE rescues only (realized '
                      '80%%-power MDE %.3f AUROC; rescues below this '
                      'are NOT excluded); the s x w_r inertness '
                      'extends to the boundary regime at that '
                      'magnitude; NOT an equivalence claim' % mde80)
  # cross-hardware replication DESCRIPTIVE: in-wave w1 vs the pinned
  # executed sigma-ladder nzs2 task values (different hardware/era)
  res['w1_replication_descriptive'] = dict(
      stat=two_sample(cells['1'], list(SGL_NZS2_TASK)),
      note='descriptive only (cross-era, cross-hardware); a large '
           'discrepancy is an instrument note, never a verdict input')
  return res


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--ridge_glob')
  ap.add_argument('--runroot')
  ap.add_argument('--nzs2_manifest')
  ap.add_argument('--linkage')
  ap.add_argument('--fit_counters')
  ap.add_argument('--ckpt_steps')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.ridge_glob and args.runroot and args.nzs2_manifest \
      and args.linkage and args.fit_counters and args.ckpt_steps \
      and args.output
  check_manifest(args.nzs2_manifest)
  check_linkage(args.linkage)
  check_witness(json.load(open(args.fit_counters)),
                json.load(open(args.ckpt_steps)))
  check_configs(args.runroot)
  res = analyse(load_ridge(args.ridge_glob))
  res['prereg'] = 'PREREG_rescue_load_20260815.md'
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({'verdict': res['verdict'],
                    'p_r1_paired': res['p_r1_paired'],
                    'panel_means': res['panel_means']}, indent=1))


def _mk(rng, w1_s0, w1_s1, boost, sd=0.03):
  ridge = {}
  for s in SIDES:
    base = w1_s0 if s == '0' else w1_s1
    for k in SEEDS:
      noise = sd * rng.standard_normal()
      ridge[('1', s, k)] = base + noise
      # paired structure: w100 shares the (side,seed) base + its own
      # noise, plus the planted boost
      ridge[('100', s, k)] = base + boost + sd * rng.standard_normal()
  return ridge


def selfcheck():
  import tempfile
  assert abs(float(np.mean(SGL_NZS2_TASK)) - 0.754189063866245) < 1e-12, \
      'pinned sigma-ladder nzs2 task values drifted'
  def _expect_refusal(fn, what):
    try:
      fn()
      raise SystemExit('expected ' + what)
    except AssertionError:
      pass
  # RESCUE-CONFIRMED: +0.12 boost
  r = analyse(_mk(np.random.default_rng(0), 0.71, 0.80, 0.12))
  assert r['verdict'].startswith('RESCUE-CONFIRMED'), \
      (r['verdict'], r['p_r1_paired'])
  # NO-RESCUE: zero boost
  r = analyse(_mk(np.random.default_rng(1), 0.71, 0.80, 0.0))
  assert r['verdict'].startswith('NO-RESCUE-DETECTED'), r['verdict']
  # INVERTED: negative boost
  r = analyse(_mk(np.random.default_rng(2), 0.71, 0.80, -0.12))
  assert r['verdict'].startswith('RESCUE-INVERTED'), r['verdict']
  # pairing is DIAGNOSTIC (review M5): on the same big-side-gap
  # fixture the paired primary fires while the UNPAIRED comparator is
  # ns - the side fixed effect cancels in the delta and only there
  ridge_pair = _mk(np.random.default_rng(3), 0.60, 0.90, 0.05, sd=0.008)
  r = analyse(ridge_pair)
  assert r['verdict'].startswith('RESCUE-CONFIRMED'), \
      (r['verdict'], r['p_r1_paired'])
  up = two_sample([ridge_pair[('100', s, k)] for s in SIDES for k in SEEDS],
                  [ridge_pair[('1', s, k)] for s in SIDES for k in SEEDS])
  assert up['ci'][0] < 0 < up['ci'][1], \
      ('pairing leg not diagnostic: unpaired comparator also fires', up)
  # loader round-trip + refusals
  with tempfile.TemporaryDirectory() as td:
    ridge = _mk(np.random.default_rng(4), 0.71, 0.80, 0.12)
    rdir = os.path.join(td, 'ridge')
    os.makedirs(rdir)
    counters, steps = {}, {}

    def _write_json(rid, fname=None, **over):
      rec = dict(run_id=rid, probeset_id=over.pop('probeset_id',
                                                  PROBESET),
                 reward_override=over.pop('reward_override', None),
                 witness_match=over.pop('witness_match', True),
                 probe={ALPHA_KEY: {'auroc': over.pop('auroc', 0.8)}})
      with open(os.path.join(rdir, (fname or rid) + '.json'), 'w') as f:
        json.dump(rec, f)

    def _cfg(w, mode='task', basesd=SIGMA, theta=1.0, rew=None):
      return {'agent': {'expl': {'mode': mode},
                        'loss_scales': {'rew': (rew if rew is not None
                                                else WR_OF[w])}},
              'distractor': {'dim': 16, 'basesd': basesd,
                             'scale': 1.0, 'theta': theta}}
    for (w, s, k), v in ridge.items():
      rid = wm_name(w, s, k)
      _write_json(rid, auroc=v)
      counters[rid] = dict(update=500000, total=500000)
      steps[rid] = 500000
      os.makedirs(os.path.join(td, rid))
      with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
        json.dump(_cfg(w), f)
    check_witness(counters, steps)
    check_configs(td)
    r = analyse(load_ridge(os.path.join(rdir, '*.json')))
    assert r['verdict'].startswith('RESCUE-CONFIRMED'), r['verdict']
    rid = wm_name('100', '0', 1)
    pat = os.path.join(rdir, '*.json')
    good = ridge[('100', '0', 1)]
    # w-collision guard: 'ruw1' names must not swallow 'ruw100'
    assert RID_RE.match('ax1wm_finger_ruw100q1s0_seed1').group('w') == '100'
    assert RID_RE.match('ax1wm_finger_ruw1q1s0_seed1').group('w') == '1'
    assert RID_RE.match('ax1wm_finger_ruw10q1s0_seed1') is None
    # ridge refusals
    _write_json(rid, probeset_id='finger_v1')
    _expect_refusal(lambda: load_ridge(pat), 'probeset refusal')
    _write_json(rid, reward_override='x')
    _expect_refusal(lambda: load_ridge(pat), 'override refusal')
    _write_json(rid, witness_match=False)
    _expect_refusal(lambda: load_ridge(pat), 'witness_match refusal')
    _write_json(rid, auroc=float('nan'))
    _expect_refusal(lambda: load_ridge(pat), 'nan refusal')
    _write_json(rid, auroc=good)
    _write_json(rid, fname=rid + '_copy')
    _expect_refusal(lambda: load_ridge(pat), 'duplicate refusal')
    os.remove(os.path.join(rdir, rid + '_copy.json'))
    os.rename(os.path.join(rdir, rid + '.json'),
              os.path.join(td, 'stash.json'))
    _expect_refusal(lambda: load_ridge(pat), 'missing refusal')
    os.rename(os.path.join(td, 'stash.json'),
              os.path.join(rdir, rid + '.json'))
    load_ridge(pat)
    # config refusals: wrong rew for cell (BOTH directions), mode,
    # basesd, theta
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_cfg('100', rew=1.0), f)
    _expect_refusal(lambda: check_configs(td), 'w100-rew refusal')
    rid1 = wm_name('1', '0', 1)
    with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
      json.dump(_cfg('100'), f)
    with open(os.path.join(td, rid1, 'config.yaml'), 'w') as f:
      json.dump(_cfg('1', rew=100.0), f)
    _expect_refusal(lambda: check_configs(td), 'w1-rew refusal')
    with open(os.path.join(td, rid1, 'config.yaml'), 'w') as f:
      json.dump(_cfg('1', mode='apt'), f)
    _expect_refusal(lambda: check_configs(td), 'mode refusal')
    with open(os.path.join(td, rid1, 'config.yaml'), 'w') as f:
      json.dump(_cfg('1', basesd=4.0), f)
    _expect_refusal(lambda: check_configs(td), 'basesd refusal')
    with open(os.path.join(td, rid1, 'config.yaml'), 'w') as f:
      json.dump(_cfg('1', theta=0.1), f)
    _expect_refusal(lambda: check_configs(td), 'theta refusal')
    with open(os.path.join(td, rid1, 'config.yaml'), 'w') as f:
      json.dump(_cfg('1'), f)
    check_configs(td)
    # witness refusals
    counters[rid] = dict(update=400000, total=500000)
    _expect_refusal(lambda: check_witness(counters, steps),
                    'counter refusal')
    counters[rid] = dict(update=500000, total=500000)
    steps[rid] = 450000
    _expect_refusal(lambda: check_witness(counters, steps),
                    'ckpt refusal')
    steps[rid] = 500000
    check_witness(counters, steps)
    # manifest gate: sigma + holdout-sha refusals
    mpath = os.path.join(td, 'manifest.json')
    hold = {'side0': ['h%02d.npz' % i for i in range(16)],
            'side1': ['g%02d.npz' % i for i in range(16)]}
    json.dump({'distractor': dict(sigma=2.0, dims=16,
                                  holdout_per_side=16,
                                  tool='nuisance_replay_v1',
                                  holdout_files=hold)}, open(mpath, 'w'))
    _expect_refusal(lambda: check_manifest(mpath),
                    'holdout-sha refusal (toy set vs the B2 pin)')
    json.dump({'distractor': dict(sigma=4.0, dims=16,
                                  holdout_per_side=16,
                                  tool='nuisance_replay_v1',
                                  holdout_files=hold)}, open(mpath, 'w'))
    _expect_refusal(lambda: check_manifest(mpath), 'sigma refusal')
    json.dump({'foo': 1}, open(mpath, 'w'))
    _expect_refusal(lambda: check_manifest(mpath),
                    'missing-distractor refusal')
    # foreign run_id (a stale sigma-ladder json in the ridge dir)
    # -> REFUSE, not skip (review M6)
    stale = 'ax1wm_finger_nzs2q1s0_seed1'
    with open(os.path.join(rdir, stale + '.json'), 'w') as f:
      json.dump(dict(run_id=stale, probeset_id=PROBESET,
                     reward_override=None, witness_match=True,
                     probe={ALPHA_KEY: {'auroc': 0.75}}), f)
    _expect_refusal(lambda: load_ridge(pat), 'foreign run_id refusal')
    os.remove(os.path.join(rdir, stale + '.json'))
    # filename that does not carry the run_id -> rename refusal
    # (review minor 4: the _copy leg trips the duplicate assert, not
    # this one)
    _write_json(rid, fname='zz_wrong')
    _expect_refusal(lambda: load_ridge(pat), 'rename refusal')
    os.remove(os.path.join(rdir, 'zz_wrong.json'))
    load_ridge(pat)
    # linkage gates (review B1): pass leg, wrong-buffer leg, short leg
    lpath = os.path.join(td, 'ruw_replay_linkage.json')
    lk = {wm_name(w, s, k): f'/x/axis1_finger/q1_nzs2/side{s}'
          for w in WR_OF for s in SIDES for k in SEEDS}
    json.dump(lk, open(lpath, 'w'))
    check_linkage(lpath)
    bad = dict(lk)
    bad[wm_name('100', '0', 1)] = '/x/axis1_finger/q1_nzs8/side0'
    json.dump(bad, open(lpath, 'w'))
    _expect_refusal(lambda: check_linkage(lpath),
                    'wrong-buffer linkage refusal')
    short = dict(lk)
    del short[wm_name('1', '1', 4)]
    json.dump(short, open(lpath, 'w'))
    _expect_refusal(lambda: check_linkage(lpath),
                    'short-linkage refusal')
  print('rescue_load_read selfcheck PASS (rescue-confirmed / '
        'no-rescue-with-realized-MDE / inverted; DIAGNOSTIC pairing '
        'leg (paired fires, unpaired ns on the same side-gap data); '
        'ruw1-vs-ruw100 name disambiguation; refusals: probeset, '
        'override, witness_match, nan, duplicate, missing, foreign '
        'run_id, rename, per-cell rew BOTH directions, mode, basesd, '
        'theta, counters, ckpt-step, manifest sigma, B2-era holdout '
        'sha, missing-distractor, wrong-buffer linkage, short '
        'linkage; pinned sigma-ladder anchors verified)')


if __name__ == '__main__':
  main()
