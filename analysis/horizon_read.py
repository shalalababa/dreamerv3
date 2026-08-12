"""Frozen reader — adaptation-horizon wave (late-crossing test).

Registration: PREREG_horizon_20260811.md (REVISED post-review: bounded
branch via registered relative margin; per-run config.yaml provenance
gates; fail-closed run inventory). ONE execution:

  python -m analysis.horizon_read --runroot <bundle>/runroot_light \
      --output <dir>
Selfcheck: python -m analysis.horizon_read --selfcheck
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
LATE_LO, LATE_HI = 4.0e5, 5.0e5
EARLY_LO, EARLY_HI = 0.75e5, 1.25e5
MIN_STEP = 4.9e5
ARM_FLOOR = 6
BIN = 5e4
DELTA_FRAC = 0.15          # registered: delta = 0.15 x scratch late mean
ARMS = {
    'hzf': re.compile(r'^adapt_ax1hzfq1s(?P<side>[01])_finger_'
                      r'seed(?P<seed>[1-4])_ckpt500000$'),
    'hzt': re.compile(r'^adapt_ax1hztq1s(?P<side>[01])_finger_'
                      r'seed(?P<seed>[1-4])_ckpt500000$'),
    'scratch': re.compile(r'^adapt_hscratch_finger_seed(?P<seed>[1-8])'
                          r'_ckpt0$'),
}
REGISTERED_IDS = {
    'hzf': {f'adapt_ax1hzfq1s{s}_finger_seed{f}_ckpt500000'
            for s in '01' for f in range(1, 5)},
    'hzt': {f'adapt_ax1hztq1s{s}_finger_seed{f}_ckpt500000'
            for s in '01' for f in range(1, 5)},
    'scratch': {f'adapt_hscratch_finger_seed{k}_ckpt0'
                for k in range(1, 9)},
}
# unfrozen_readout signature keys (dreamerv3/configs.yaml:276-280).
# Reviewer-2 B1: NO expl-mode gate — the adapt stage never receives
# --agent.expl.mode (AXIS1_EXPL_MODE is a Stage-1 knob; real U1 apt
# adapts record mode 'task'); arm identity is carried by the
# from_checkpoint WM name alone.
WM_OF = {
    'hzf': lambda side, seed: f'ax1wm_finger_fq1s{side}_seed{seed}',
    'hzt': lambda side, seed: f'ax1wm_finger_q1s{side}_seed{seed}',
}


def _erfinv(x):
  a = 0.147
  ln = math.log(1.0 - x * x)
  t1 = 2.0 / (math.pi * a) + ln / 2.0
  return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)


def bca_two_sample(a, b, rng_seed=RNG_SEED, nboot=B_BOOT):
  """BCa 95% CI for mean(a) - mean(b), resampling runs within arms.
  (Jackknife concatenates leave-one-out influences of both arms without
  stratum weighting — mildly mis-weighted only under unequal n; noted.)"""
  a, b = np.asarray(a, float), np.asarray(b, float)
  rng = np.random.default_rng(rng_seed)
  boots = np.array([
      a[rng.integers(0, len(a), len(a))].mean()
      - b[rng.integers(0, len(b), len(b))].mean() for _ in range(nboot)])
  theta = float(a.mean() - b.mean())
  prop = float(np.mean(boots < theta))
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
  return theta, (q(0.025), q(0.975))


def perm_p_two_sample(a, b, rng_seed=RNG_SEED, nperm=B_BOOT):
  a, b = np.asarray(a, float), np.asarray(b, float)
  pool = np.concatenate([a, b])
  obs = abs(a.mean() - b.mean())
  rng = np.random.default_rng(rng_seed)
  cnt = 0
  for _ in range(nperm):
    perm = rng.permutation(pool)
    d = abs(perm[:len(a)].mean() - perm[len(a):].mean())
    if d >= obs - 1e-12:
      cnt += 1
  return float((cnt + 1) / (nperm + 1))


def _load_yaml(path):
  try:
    import yaml
    with open(path) as f:
      return yaml.safe_load(f)
  except ImportError:
    from ruamel.yaml import YAML
    with open(path) as f:
      return YAML(typ='safe').load(f)


def _check_provenance(rid, arm, m, cfg):
  """Per-run config.yaml gate (review #8). Fail-closed asserts."""
  if arm in ('hzf', 'hzt'):
    wm = WM_OF[arm](m.group('side'), m.group('seed'))
    fc = str(cfg['run']['from_checkpoint'])
    assert wm in fc, (rid, 'from_checkpoint does not name registered WM',
                      fc)
    assert str(cfg['run'].get('from_checkpoint_regex', '')) == \
        '^(enc|dyn|dec)/', (rid, 'unfrozen_readout regex missing')
    assert cfg['agent']['frozen_wm'] is False, (rid, 'frozen_wm not False')
  else:
    fc = str(cfg.get('run', {}).get('from_checkpoint', '') or '')
    # real scratch configs carry from_checkpoint: '' (reviewer-2 m4)
    assert fc == '', (rid, 'scratch run has a from_checkpoint', fc)


def load_runs(runroot):
  runs = {arm: {} for arm in ARMS}
  unmatched = []
  for d in sorted(glob.glob(os.path.join(runroot, 'adapt_*'))):
    rid = os.path.basename(d)
    hit = [(a, rx.match(rid)) for a, rx in ARMS.items() if rx.match(rid)]
    if not hit:
      unmatched.append(rid)
      continue
    arm, m = hit[0]
    cfgp = os.path.join(d, 'config.yaml')
    assert os.path.exists(cfgp), (rid, 'missing config.yaml')
    _check_provenance(rid, arm, m, _load_yaml(cfgp))
    sp = os.path.join(d, 'scores.jsonl')
    assert os.path.exists(sp), (rid, 'missing scores.jsonl')
    steps, scores = [], []
    with open(sp) as f:
      for line in f:
        if not line.strip():
          continue
        r = json.loads(line)
        steps.append(float(r['step']))
        scores.append(float(r['episode/score']))
    steps, scores = np.asarray(steps), np.asarray(scores)
    # review #9: appended/resumed scores.jsonl -> steps not increasing
    assert (np.diff(steps) > 0).all(), (rid, 'steps not strictly '
                                        'increasing - appended file?')
    runs[arm][rid] = (steps, scores)
  # review #9: refuse silently-dropped dirs
  assert not unmatched, ('unregistered adapt_* dirs present - refusing',
                         unmatched)
  for arm in ARMS:
    extra = set(runs[arm]) - REGISTERED_IDS[arm]
    assert not extra, (arm, 'ids outside the registered set', extra)
  return runs


def window_mean(steps, scores, lo, hi):
  m = (steps >= lo) & (steps < hi)      # half-open (review #23)
  return float(scores[m].mean()) if m.any() else float('nan')


def analyse(runs):
  res = {'excluded_short': [], 'n_per_arm': {}, 'run_stats': {}}
  gated = {}
  for arm in ARMS:
    vals, kept = {}, {}
    missing = REGISTERED_IDS[arm] - set(runs[arm])
    assert not missing, (arm, 'registered runs missing from bundle',
                         sorted(missing))
    for rid, (st, sc) in sorted(runs[arm].items()):
      res['run_stats'][rid] = dict(n_episodes=int(len(st)),
                                   max_step=float(st.max()))
      if st.max() < MIN_STEP:
        res['excluded_short'].append(
            (rid, float(st.max()), 'below completion gate 4.9e5'))
        continue
      v = window_mean(st, sc, LATE_LO, LATE_HI + 1)  # include 5e5 edge
      assert math.isfinite(v), (rid, 'no late-window episodes')
      vals[rid] = v
      kept[rid] = (st, sc)
    gated[arm] = kept
    res[f'late_{arm}'] = vals
    res['n_per_arm'][arm] = len(vals)
  late = {arm: list(res[f'late_{arm}'].values()) for arm in ARMS}
  if any(len(late[a]) < ARM_FLOOR for a in ARMS):
    res['verdict'] = ('INSTRUMENT-LIMITED: an arm fell below n=6 after '
                      'the completion gate - no branch read')
    return res

  hzf, scr, hzt = late['hzf'], late['scratch'], late['hzt']
  delta = DELTA_FRAC * float(np.mean(scr))
  diff, ci = bca_two_sample(hzf, scr)
  p = perm_p_two_sample(hzf, scr)
  res['p_h1'] = dict(diff=diff, ci=list(ci), perm_p=p, delta=delta,
                     n=[len(hzf), len(scr)])
  # review #6: margin-based branch map
  if ci[0] > 0 and p < 0.05:
    verdict = ('CROSSES-LATE: the reward-free null is horizon-bounded; '
               'headline wording must carry the budget qualifier')
  elif ci[1] < delta:
    if ci[1] < 0 and p < 0.05:
      verdict = ('APT-INIT-BELOW-SCRATCH-AT-4X (also bounded: no '
                 'meaningful late advantage, CI upper < delta)')
    else:
      verdict = ('BOUNDED-NULL-4X: no meaningful late advantage '
                 '(CI upper < delta); reward-free ~ untrained extends '
                 'to 4x budget')
  else:
    verdict = 'INDETERMINATE (MDE note applies; licenses nothing)'
  cdiff, cci = bca_two_sample(hzt, scr)
  cp = perm_p_two_sample(hzt, scr)
  res['p_h3_control'] = dict(diff=cdiff, ci=list(cci), perm_p=cp)
  if not (cci[0] > 0 and cp < 0.05):
    verdict = 'INSTRUMENT-SUSPECT (positive control failed) | ' + verdict
  res['verdict'] = verdict

  # P-H2 descriptives from the GATED set (review #11)
  bins = {}
  for arm in ARMS:
    curve = {}
    for lo in np.arange(0, LATE_HI, BIN):
      vs = [window_mean(st, sc, lo, lo + BIN)
            for st, sc in gated[arm].values()]
      vs = [v for v in vs if math.isfinite(v)]
      if vs:
        curve[int(lo)] = float(np.mean(vs))
    bins[arm] = curve
  common = sorted(set(bins['hzf']) & set(bins['scratch']))
  first_pos = None
  for lo in common:
    rest = [l for l in common if l >= lo]
    if all(bins['hzf'][l] > bins['scratch'][l] for l in rest):
      first_pos = int(lo)
      break
  early = {arm: [window_mean(st, sc, EARLY_LO, EARLY_HI)
                 for st, sc in gated[arm].values()] for arm in ARMS}
  early_m = {arm: (float(np.mean([v for v in early[arm]
                                  if math.isfinite(v)]))
                   if any(math.isfinite(v) for v in early[arm])
                   else None) for arm in ARMS}
  ratio = None
  if early_m['hzf'] is not None and early_m['scratch'] not in (None, 0.0):
    ratio = early_m['hzf'] / early_m['scratch']
  res['p_h2_descriptive'] = dict(
      bin_curves=bins, first_sustained_positive_bin=first_pos,
      early_ratio_hzf_over_scratch=ratio,
      note='descriptive only; early ratio is a DIRECTIONAL, non-matched '
           'comparison vs the U-wave AUC-based 0.5-0.6x anchors')
  return res


# ------------------------- selfcheck ---------------------------------

def _traj(rng, base, slope, n_ep=250, tmax=5.0e5, noise=30.0, run_sd=25.0,
          late_shift=0.0, shift_at=None):
  steps = np.sort(rng.uniform(2e3, tmax, n_ep))
  off = run_sd * rng.standard_normal()
  scores = base + off + slope * steps / tmax \
      + noise * rng.standard_normal(n_ep)
  if shift_at is not None:
    scores = scores + late_shift * (steps >= shift_at)
  return steps, scores


def _mk_runs(rng, hzf_late, scr_late, hzt_late=800.0, tmax=5.0e5,
             run_sd=25.0, hzf_shift=0.0, hzf_shift_at=None):
  runs = {'hzf': {}, 'hzt': {}, 'scratch': {}}
  for side in '01':
    for f in range(1, 5):
      runs['hzf'][f'adapt_ax1hzfq1s{side}_finger_seed{f}_ckpt500000'] = \
          _traj(rng, hzf_late * 0.5, hzf_late * 0.5, tmax=tmax,
                run_sd=run_sd, late_shift=hzf_shift,
                shift_at=hzf_shift_at)
      runs['hzt'][f'adapt_ax1hztq1s{side}_finger_seed{f}_ckpt500000'] = \
          _traj(rng, hzt_late * 0.5, hzt_late * 0.5, tmax=tmax,
                run_sd=run_sd)
  for k in range(1, 9):
    runs['scratch'][f'adapt_hscratch_finger_seed{k}_ckpt0'] = \
        _traj(rng, scr_late * 0.5, scr_late * 0.5, tmax=tmax,
              run_sd=run_sd)
  return runs


def _cfg_for(rid, frozen_wm=False):
  """Fixture configs mirror REAL executed config shape (reviewer-2 B1:
  expl.mode is 'task' on every adapt incl. apt-WM ones; scratch carries
  from_checkpoint: '')."""
  for arm, rx in ARMS.items():
    m = rx.match(rid)
    if not m:
      continue
    if arm == 'scratch':
      return {'run': {'from_checkpoint': ''},
              'agent': {'expl': {'mode': 'task'}}}
    return {'run': {'from_checkpoint':
                    f"/runs/{WM_OF[arm](m.group('side'), m.group('seed'))}/ckpt",
                    'from_checkpoint_regex': '^(enc|dyn|dec)/'},
            'agent': {'frozen_wm': frozen_wm,
                      'expl': {'mode': 'task'}}}
  raise AssertionError(rid)


def selfcheck():
  import tempfile
  r = analyse(_mk_runs(np.random.default_rng(0), 600.0, 250.0))
  assert r['verdict'].startswith('CROSSES-LATE'), r['verdict']
  # strictly below scratch -> the SPLIT branch, not "~ untrained"
  r = analyse(_mk_runs(np.random.default_rng(1), 120.0, 400.0))
  assert r['verdict'].startswith('APT-INIT-BELOW-SCRATCH-AT-4X'), \
      r['verdict']
  # near-equal, tight -> BOUNDED-NULL-4X via the delta margin
  r = analyse(_mk_runs(np.random.default_rng(2), 398.0, 400.0,
                       run_sd=20.0))
  assert r['verdict'].startswith('BOUNDED-NULL-4X'), \
      (r['verdict'], r['p_h1'])
  # near-equal but wide spread -> INDETERMINATE (CI upper >= delta)
  r = analyse(_mk_runs(np.random.default_rng(3), 400.0, 400.0,
                       run_sd=120.0))
  assert r['verdict'].startswith('INDETERMINATE'), \
      (r['verdict'], r['p_h1'])
  # positive control fails -> flagged
  r = analyse(_mk_runs(np.random.default_rng(4), 600.0, 250.0,
                       hzt_late=250.0))
  assert r['verdict'].startswith('INSTRUMENT-SUSPECT'), r['verdict']
  # TRUE CROSSING at 3.5e5: hzf below scratch early, above late
  # (review #7: kills wrong-window and wrong-bin mutants)
  r = analyse(_mk_runs(np.random.default_rng(5), 200.0, 400.0,
                       run_sd=10.0, hzf_shift=350.0, hzf_shift_at=3.5e5))
  assert r['verdict'].startswith('CROSSES-LATE'), (r['verdict'],
                                                   r['p_h1'])
  fp = r['p_h2_descriptive']['first_sustained_positive_bin']
  assert fp == 350000, fp
  # EARLY-ONLY advantage must NOT read CROSSES-LATE
  r = analyse(_mk_runs(np.random.default_rng(6), 700.0, 400.0,
                       run_sd=10.0, hzf_shift=-500.0, hzf_shift_at=2.0e5))
  assert not r['verdict'].startswith('CROSSES-LATE'), r['verdict']
  # short runs excluded; arm floor trips
  runs = _mk_runs(np.random.default_rng(7), 600.0, 250.0)
  short = {}
  for i, (rid, (st, sc)) in enumerate(sorted(runs['hzf'].items())):
    short[rid] = ((st * 0.5, sc) if i < 3 else (st, sc))
  runs['hzf'] = short
  r = analyse(runs)
  assert r['verdict'].startswith('INSTRUMENT-LIMITED'), r['verdict']
  assert len(r['excluded_short']) == 3, r['excluded_short']
  # missing registered run refuses
  runs = _mk_runs(np.random.default_rng(8), 600.0, 250.0)
  runs['hzt'].pop(sorted(runs['hzt'])[0])
  try:
    analyse(runs)
    raise SystemExit('expected missing-run refusal')
  except AssertionError:
    pass
  # temp round-trip incl. config.yaml provenance gates
  with tempfile.TemporaryDirectory() as td:
    runs = _mk_runs(np.random.default_rng(9), 600.0, 250.0)
    for arm in runs:
      for rid, (st, sc) in runs[arm].items():
        os.makedirs(os.path.join(td, rid))
        with open(os.path.join(td, rid, 'config.yaml'), 'w') as f:
          json.dump(_cfg_for(rid), f)   # json is valid yaml
        with open(os.path.join(td, rid, 'scores.jsonl'), 'w') as f:
          for s, v in zip(st, sc):
            f.write(json.dumps({'step': s, 'episode/score': v}) + '\n')
    loaded = load_runs(td)
    r = analyse(loaded)
    assert r['verdict'].startswith('CROSSES-LATE'), r['verdict']
    assert r['n_per_arm'] == {'hzf': 8, 'hzt': 8, 'scratch': 8}
    # wrong-WM provenance trips
    bad = sorted(loaded['hzf'])[0]
    cfgp = os.path.join(td, bad, 'config.yaml')
    cfg = _cfg_for(bad)
    cfg['run']['from_checkpoint'] = '/runs/ax1wm_finger_q1s0_seed1/ckpt'
    with open(cfgp, 'w') as f:
      json.dump(cfg, f)
    try:
      load_runs(td)
      raise SystemExit('expected wrong-WM provenance refusal')
    except AssertionError:
      pass
    with open(cfgp, 'w') as f:
      json.dump(_cfg_for(bad), f)
    # frozen adapt (default AXIS1_ADAPT_CONFIG) trips the gate (rev-2 M5)
    with open(cfgp, 'w') as f:
      json.dump(_cfg_for(bad, frozen_wm=True), f)
    try:
      load_runs(td)
      raise SystemExit('expected frozen-wm refusal')
    except AssertionError:
      pass
    with open(cfgp, 'w') as f:
      json.dump(_cfg_for(bad), f)
    # unregistered dir refuses
    os.makedirs(os.path.join(td, 'adapt_rogue_finger_seed1_ckpt0'))
    try:
      load_runs(td)
      raise SystemExit('expected unregistered-dir refusal')
    except AssertionError:
      pass
    os.rmdir(os.path.join(td, 'adapt_rogue_finger_seed1_ckpt0'))
    # appended (non-monotone) scores.jsonl refuses
    sp = os.path.join(td, bad, 'scores.jsonl')
    with open(sp) as f:
      lines = f.readlines()
    with open(sp, 'a') as f:
      f.writelines(lines[:5])
    try:
      load_runs(td)
      raise SystemExit('expected non-monotone-steps refusal')
    except AssertionError:
      pass
  print('horizon_read selfcheck PASS (crossing/below-scratch/bounded-'
        'delta/indeterminate/control-suspect branches; true-crossing '
        'bin located at 3.5e5; early-only-advantage rejected; short-'
        'excluded+floor; missing-run, wrong-WM, unregistered-dir, '
        'appended-file refusals; config-gate round-trip)')


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--runroot')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.runroot and args.output
  res = analyse(load_runs(args.runroot))
  res['prereg'] = 'PREREG_horizon_20260811.md'
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({'verdict': res['verdict'],
                    'p_h1': res.get('p_h1'),
                    'p_h3_control': res.get('p_h3_control')}, indent=1))


if __name__ == '__main__':
  main()
