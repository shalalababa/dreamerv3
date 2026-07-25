"""Frozen read for the pixel reconstruction-swamping diagnostic.

Registered in prereg/PREREG_pixel_swamping_20260724.md (+ Amendment 1, which
repaired the E4 measure path for pixel encoders BEFORE any pixel E4 number
existed). This script is committed before any pixel reward-NLL exists.

Decision statistic (frozen): task-arm in-regime reward-NLL LEVEL at h0,
pooled over sides, seed-clustered percentile bootstrap CI (B=10K, rng 0).
  - SWAMPING-CONSISTENT  iff CI entirely >= 2.0 nats
  - MEMBERSHIP-form      iff CI entirely <= 1.5 nats
  - otherwise UNRESOLVED
Guard: the probe set's trivial-predictor floor -- the cross-entropy of the
best CONSTANT reward-head prediction (the marginal two-hot distribution of
the probe set's own rewards on in-regime frames, under the agent's
symexp_twohot 255-bin convention). The probe set is common to both sides, so
the registered "per-side" floor is one number, disclosed as such. Floor >=
2.0 => the swamping threshold is unreachable by construction => read is
UNINFORMATIVE (reported, no fire).

Disclosed non-readouts recorded descriptively: the apt arm trains its reward
head on the APT intrinsic reward; the frozen E4 machinery does not score
intrinsic heads against true reward (reward_aware gate), so no apt reward
NLL exists -- recorded as absent-by-design. d_errin per arm x side is copied
through descriptively.

Usage:
  python -m analysis.pixel_swamping_read --e4_csv <e4_fingerpx_v1.csv> \
      --probeset <fingerpx_v1 dir> --output <dir>
  python -m analysis.pixel_swamping_read --selfcheck
"""

import argparse
import json
import os
import re

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
THRESH_SWAMP = 2.0
THRESH_MEMBER = 1.5
FLOOR_GUARD = 2.0
NBINS = 255
TASK_RE = re.compile(r'^ax1wm_finger_pxpxq1ms([01])_seed(\d+)$')
APT_RE = re.compile(r'^ax1wm_finger_fpxpxq1ms([01])_seed(\d+)$')
EXPECT_SEEDS = tuple(range(1, 9))


def symexp_bins(n=NBINS):
  assert n % 2 == 1
  half = np.linspace(-20, 0, (n - 1) // 2 + 1, dtype=np.float64)
  half = np.sign(half) * (np.exp(np.abs(half)) - 1)
  return np.concatenate([half, -half[:-1][::-1]], 0)


def twohot(target, bins):
  target = np.asarray(target, np.float64)
  below = (bins[None, :] <= target[:, None]).sum(-1) - 1
  above = len(bins) - (bins[None, :] > target[:, None]).sum(-1)
  below = np.clip(below, 0, len(bins) - 1)
  above = np.clip(above, 0, len(bins) - 1)
  equal = below == above
  d_lo = np.where(equal, 1, np.abs(bins[below] - target))
  d_hi = np.where(equal, 1, np.abs(bins[above] - target))
  total = d_lo + d_hi
  w_lo, w_hi = d_hi / total, d_lo / total
  t = np.zeros((len(target), len(bins)))
  t[np.arange(len(target)), below] += w_lo
  t[np.arange(len(target)), above] += w_hi
  return t


def trivial_floor(rewards_in_regime):
  """NLL of the marginal two-hot distribution on in-regime frames."""
  t = twohot(rewards_in_regime, symexp_bins())
  marginal = t.mean(0)
  logm = np.log(np.clip(marginal, 1e-30, None))
  return float(-(t * logm[None, :]).sum(-1).mean())


def seed_cluster_ci(values_by_seed, b=B_BOOT, seed=RNG_SEED):
  """Percentile bootstrap over seed clusters of the pooled mean."""
  seeds = sorted(values_by_seed)
  rng = np.random.default_rng(seed)
  n = len(seeds)
  point = float(np.mean([v for s in seeds for v in values_by_seed[s]]))
  stats = np.empty(b)
  for i in range(b):
    pick = rng.integers(0, n, n)
    vals = [v for j in pick for v in values_by_seed[seeds[j]]]
    stats[i] = np.mean(vals)
  lo, hi = np.percentile(stats, [2.5, 97.5])
  return point, float(lo), float(hi)


def load_rows(path):
  import csv
  with open(path) as f:
    return list(csv.DictReader(f))


def analyze(rows, probe_rewards, probe_in_regime):
  floor = trivial_floor(np.asarray(probe_rewards)[np.asarray(probe_in_regime)])
  task, apt_present = {}, set()
  d_errin = {}
  for r in rows:
    rid = r['run_id']
    h = int(r['horizon'])
    m = TASK_RE.match(rid)
    if m and h == 0:
      side, s = int(m.group(1)), int(m.group(2))
      assert r.get('rew_nll_in') not in (None, ''), (
          f'{rid}: task-arm row lacks rew_nll_in (reward head not scored?)')
      task.setdefault(s, []).append(float(r['rew_nll_in']))
    if APT_RE.match(rid):
      apt_present.add(rid)
      assert r.get('rew_nll_in') in (None, ''), (
          f'{rid}: apt arm unexpectedly has a true-reward NLL '
          '(reward_aware gate changed?)')
    if (m or APT_RE.match(rid)) and r.get('err_diff') not in (None, ''):
      d_errin.setdefault(rid, {})[h] = float(r['err_diff'])
  missing = [s for s in EXPECT_SEEDS if len(task.get(s, [])) != 2]
  assert not missing, (
      f'task arm incomplete: seeds {missing} lack both side rows at h0')
  point, lo, hi = seed_cluster_ci(task)
  if floor >= FLOOR_GUARD:
    verdict = ('UNINFORMATIVE: trivial-predictor floor '
               f'{floor:.3f} >= {FLOOR_GUARD} -- swamping threshold '
               'unreachable by construction (registered guard)')
    fired = None
  elif lo >= THRESH_SWAMP:
    verdict = ('P-SW1 SWAMPING-CONSISTENT: task-arm in-regime rew-NLL CI '
               'entirely >= 2.0 nats -- reward information absent from the '
               'pixel trunk; all-cells-floor = the g > G2 regime')
    fired = 'swamping'
  elif hi <= THRESH_MEMBER:
    verdict = ('MEMBERSHIP-form: CI entirely <= 1.5 nats -- pixel trunk '
               'carries reward information without behavioral transfer '
               '(membership->transfer boundary; theory problem)')
    fired = 'membership'
  else:
    verdict = 'UNRESOLVED: CI spans the registered bands (reported; no fire)'
    fired = None
  return dict(
      task_rew_nll_in_h0=dict(mean=point, ci=[lo, hi],
                              n_seeds=len(task), n_runs=sum(
                                  len(v) for v in task.values()),
                              per_seed={str(s): v for s, v in
                                        sorted(task.items())}),
      trivial_floor=floor,
      thresholds=dict(swamp=THRESH_SWAMP, member=THRESH_MEMBER,
                      floor_guard=FLOOR_GUARD),
      apt_true_reward_nll='absent-by-design (reward_aware gate; disclosed '
                          'non-readout in the prereg)',
      n_apt_runs_seen=len(apt_present),
      d_errin_descriptive=d_errin,
      fired=fired, verdict=verdict)


def read(args):
  rows = load_rows(args.e4_csv)
  z = np.load(os.path.join(args.probeset, 'probeset_e4.npz'))
  res = analyze(rows, z['reward'].reshape(-1), z['in_regime'].reshape(-1))
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'pixel_swamping.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  t = res['task_rew_nll_in_h0']
  print(f"task-arm rew_nll_in h0: {t['mean']:.4f} "
        f"[{t['ci'][0]:.4f}, {t['ci'][1]:.4f}] "
        f"(n_seeds={t['n_seeds']}, n_runs={t['n_runs']})")
  print(f"trivial floor: {res['trivial_floor']:.4f}")
  print(res['verdict'])
  print(f'-> {out}')


def _synth_rows(task_nll, seeds=EXPECT_SEEDS):
  rows = []
  for s in seeds:
    for side in (0, 1):
      rows.append(dict(run_id=f'ax1wm_finger_pxpxq1ms{side}_seed{s}',
                       horizon='0', rew_nll_in=str(task_nll(s, side)),
                       err_diff='0.01'))
      rows.append(dict(run_id=f'ax1wm_finger_fpxpxq1ms{side}_seed{s}',
                       horizon='0', rew_nll_in='', err_diff='0.02'))
  return rows


def selfcheck(args):
  rng = np.random.default_rng(1)
  # Sparse near-binary rewards: low-entropy marginal => floor far below 2.
  probe_r = (rng.random(5000) < 0.08).astype(np.float64)
  probe_in = np.ones(5000, bool)

  res = analyze(_synth_rows(lambda s, d: 2.4 + 0.05 * rng.standard_normal()),
                probe_r, probe_in)
  assert res['fired'] == 'swamping', res['verdict']

  res = analyze(_synth_rows(lambda s, d: 1.1 + 0.05 * rng.standard_normal()),
                probe_r, probe_in)
  assert res['fired'] == 'membership', res['verdict']

  res = analyze(_synth_rows(lambda s, d: 1.75 + 0.6 * (s % 2)),
                probe_r, probe_in)
  assert res['fired'] is None and res['verdict'].startswith('UNRESOLVED'), (
      res['verdict'])

  # High-entropy rewards spread across many symexp bins => floor >= 2 =>
  # the guard trips regardless of a swamping-consistent level.
  wide_r = np.sign(rng.standard_normal(5000)) * (
      np.exp(np.abs(rng.standard_normal(5000)) * 4) - 1)
  res = analyze(_synth_rows(lambda s, d: 2.4), wide_r, probe_in)
  assert res['fired'] is None and res['verdict'].startswith('UNINFORMATIVE'), (
      res['verdict'])

  # Missing side rows must trip.
  rows = _synth_rows(lambda s, d: 2.4)
  rows = [r for r in rows if not (r['run_id'].endswith('s1_seed3'))]
  try:
    analyze(rows, probe_r, probe_in)
    raise SystemExit('selfcheck FAIL: missing cell not caught')
  except AssertionError:
    pass

  # An apt row carrying a rew NLL (gate drift) must trip.
  rows = _synth_rows(lambda s, d: 2.4)
  for r in rows:
    if r['run_id'].startswith('ax1wm_finger_fpxpx'):
      r['rew_nll_in'] = '9.9'
      break
  try:
    analyze(rows, probe_r, probe_in)
    raise SystemExit('selfcheck FAIL: apt rew NLL not caught')
  except AssertionError:
    pass

  # Floor sanity: constant reward => floor ~ 0.
  assert trivial_floor(np.zeros(1000)) < 0.1
  print('selfcheck PASS: swamping/membership/unresolved verdicts, floor '
        'guard, missing-cell and apt-gate trips, constant-reward floor')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--e4_csv')
  ap.add_argument('--probeset')
  ap.add_argument('--output', default='analysis_out/pixel_swamping')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
  else:
    read(args)


if __name__ == '__main__':
  main()


