"""Frozen reader for the rl/sh own-label re-fit read (2026-08-10).

Registration: PREREG_rlsh_ownlabel_refit_20260808.md (P-RS1/P-RS2 paired
own-minus-true rew-NLL, S1 level bar, S2 d_errin fingerprint rider), as
amended by PREREG_refit_reads_amend1_20260810.md (checkpoint-step witness;
the per-run ridge jsons in the bundle are NOT consumed - unregistered).

Execution (ONE):
  python -m analysis.rlsh_ownlabel_read \
      --bundle local_results/rlsh_ownlabel_ridge_e4_20260810_220721 \
      --out artifacts/rlsh_ownlabel_read_20260810
Selfcheck: python -m analysis.rlsh_ownlabel_read selfcheck
"""

import argparse
import glob
import json
import math
import os

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
CKPT_STEP = 500_000
S1_ANCHOR = 0.827      # archived task-arm member band anchor (registered)
S1_BAR = 1.5           # registered claim bar (16-fit CI entirely <= 1.5)
S2_ARCHIVED = -0.018   # archived d_errin dist_to_target out-in prediction
KIND_BY_CODE = {'rl': 'relocate', 'sh': 'shuffle'}
ARCHIVED_TRUE = {      # rew_nll_out_panel (review_response_20260808), h0
    'rl': {'s0': {'in': 4.4104371233308, 'out': 0.2035808384117812},
           's1': {'in': 3.4315837561287976, 'out': 0.37633941187488007}},
    'sh': {'s0': {'in': 4.6202990894961395, 'out': 0.20833358189815693},
           's1': {'in': 2.0251103207083183, 'out': 0.19984001964861356}},
}


def _erfinv(x):
  a = 0.147
  ln = math.log(1.0 - x * x)
  t1 = 2.0 / (math.pi * a) + ln / 2.0
  return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)


def _ndtri(p):
  return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def bca(vals, clusters, rng_seed=RNG_SEED, b=B_BOOT):
  vals = np.asarray(vals, float)
  cl = np.asarray(clusters)
  uniq = sorted(set(cl.tolist()))
  idx = {c: np.where(cl == c)[0] for c in uniq}
  rng = np.random.default_rng(rng_seed)
  boots = np.empty(b)
  for i in range(b):
    pick = rng.choice(len(uniq), len(uniq), replace=True)
    sel = np.concatenate([idx[uniq[p]] for p in pick])
    boots[i] = np.nanmean(vals[sel])
  theta = float(np.nanmean(vals))
  prop = float(np.mean(boots < theta))
  prop = min(max(prop, 1.0 / (b + 1)), 1.0 - 1.0 / (b + 1))
  z0 = _ndtri(prop)
  jack = np.asarray([float(np.nanmean(vals[cl != c])) for c in uniq])
  jm = jack.mean()
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  a_acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.0
  def q(alpha_pt):
    z = _ndtri(alpha_pt)
    adj = z0 + (z0 + z) / (1.0 - a_acc * (z0 + z))
    p = 0.5 * (1.0 + math.erf(adj / math.sqrt(2.0)))
    return float(np.percentile(boots, 100.0 * min(max(p, 0.0), 1.0)))
  return theta, (q(0.025), q(0.975))


def signflip_p(vals, rng_seed=RNG_SEED, b=B_BOOT):
  """Per-fit sign-flip permutation for mean != 0 (two-sided), registered."""
  vals = np.asarray(vals, float)
  rng = np.random.default_rng(rng_seed)
  obs = abs(vals.mean())
  flips = rng.choice([-1.0, 1.0], size=(b, len(vals)))
  null = np.abs((flips * vals[None]).mean(1))
  return float((np.sum(null >= obs - 1e-15) + 1) / (b + 1))


def load_rows(bundle):
  rows = []
  for rd in sorted(glob.glob(os.path.join(bundle, 'runroot_light',
                                          'ax1wm_finger_*'))):
    run = os.path.basename(rd)
    code = 'rl' if '_rlq1' in run else ('sh' if '_shq1' in run else None)
    assert code, run
    kind = KIND_BY_CODE[code]
    tj = json.load(open(os.path.join(rd, 'e4_finger_v1refit',
                                     'summary.json')))
    oj = json.load(open(os.path.join(rd, f'e4_finger_v1refit_own_{kind}',
                                     'summary.json')))
    assert tj['run_id'] == run and oj['run_id'] == run, run
    for j in (tj, oj):
      assert int(os.path.basename(j['checkpoint']).split('-')[-1]) == \
          CKPT_STEP, (run, j['checkpoint'])
    assert os.path.basename(tj['checkpoint']).startswith(
        ('20260809', '20260810')), (run, 'stale checkpoint')
    assert tj['checkpoint'] == oj['checkpoint'], (run, 'ckpt mismatch')
    assert tj['reward_override'] is None, run
    ov = oj['reward_override']['sidecar']
    assert ov['kind'] == kind and ov['seed'] == 0, (run, ov)
    assert ov['probeset_sha256'] == tj['probeset_sha256'], run
    assert tj['expl_mode'] == 'task' and tj['reward_aware'] is True, run
    th = tj['horizon_stats']['0']
    oh = oj['horizon_stats']['0']
    side = 1 if 'q1s1_' in run else 0
    seed = int(run.rsplit('seed', 1)[1])
    dk = th['per_key']['dist_to_target']
    rows.append(dict(
        run=run, code=code, side=side, seed=seed,
        true_all=th['reward_head']['all']['mean'],
        true_in=th['reward_head']['in_regime']['mean'],
        true_out=th['reward_head']['out_regime']['mean'],
        own_all=oh['reward_head']['all']['mean'],
        own_in=oh['reward_head']['in_regime']['mean'],
        own_out=oh['reward_head']['out_regime']['mean'],
        d_errin_dist=(dk['out_regime']['mean'] - dk['in_regime']['mean']),
    ))
  assert len(rows) == 32, len(rows)
  for code in ('rl', 'sh'):
    assert sum(r['code'] == code for r in rows) == 16, code
  return rows


def code_analysis(rows, code):
  sub = [r for r in rows if r['code'] == code]
  diffs = [r['own_all'] - r['true_all'] for r in sub]
  seeds = [r['seed'] for r in sub]
  mean, ci = bca(diffs, seeds)
  p = signflip_p(diffs)
  if ci[1] < 0.0 and p < 0.05:
    verdict, why = 'LEARNED-BUT-UNALIGNED', (
        'own - true CI entirely < 0 and perm p < .05')
  elif ci[0] <= 0.0 <= ci[1]:
    verdict, why = 'FAILURE-TO-LEARN', (
        'own - true CI straddles 0 (own at the true-label level)')
  elif ci[0] > 0.0:
    verdict, why = 'REGISTERED-ANOMALY', (
        'own significantly ABOVE true - no wording licensed, disclosed')
  else:
    verdict, why = 'REGISTERED-ANOMALY', (
        'CI < 0 but perm p >= .05 - outside the registered branches')
  out = {'n': len(sub), 'verdict': verdict, 'why': why,
         'own_minus_true_all': dict(mean=mean, ci=list(ci), perm_p=p)}
  # per-side descriptive (registered n=8 descriptive)
  for s in (0, 1):
    d = [r['own_all'] - r['true_all'] for r in sub if r['side'] == s]
    out[f'side{s}_mean'] = float(np.mean(d))
  # S1: own-label in-regime level vs bar
  own_in = [r['own_in'] for r in sub]
  m1, ci1 = bca(own_in, seeds)
  out['s1'] = dict(own_in_mean=m1, ci=list(ci1), anchor=S1_ANCHOR,
                   bar=S1_BAR, claimable=bool(ci1[1] <= S1_BAR))
  # levels vs archived true-label panel (descriptive)
  lv = {}
  for s in (0, 1):
    ss = [r for r in sub if r['side'] == s]
    lv[f's{s}'] = dict(
        true_in=float(np.mean([r['true_in'] for r in ss])),
        true_out=float(np.mean([r['true_out'] for r in ss])),
        own_in=float(np.mean([r['own_in'] for r in ss])),
        own_out=float(np.mean([r['own_out'] for r in ss])),
        archived_true=ARCHIVED_TRUE[code][f's{s}'])
  out['levels'] = lv
  # S2 rider: dist_to_target out-in at h0, s0 side, TRUE pass (registered
  # exploratory, no verdict).
  d2 = [r['d_errin_dist'] for r in sub if r['side'] == 0]
  m2, ci2 = bca(d2, [r['seed'] for r in sub if r['side'] == 0])
  out['s2'] = dict(mean=m2, ci=list(ci2), archived=S2_ARCHIVED,
                   sign_replicates=bool(m2 < 0))
  return out


def analyse(rows):
  return {'P-RS1_rl': code_analysis(rows, 'rl'),
          'P-RS2_sh': code_analysis(rows, 'sh')}


# --------------------------------------------------------------------------

def _fake_rows(rl_gap, sh_gap, own_in_level=1.0, rng=None):
  rng = rng or np.random.default_rng(11)
  rows = []
  for code, gap in (('rl', rl_gap), ('sh', sh_gap)):
    for side in (0, 1):
      for seed in range(1, 9):
        true_all = 0.6 + rng.normal(0, 0.02)
        rows.append(dict(
            run=f'f_{code}_{side}_{seed}', code=code, side=side, seed=seed,
            true_all=true_all, true_in=3.5, true_out=0.3,
            own_all=true_all + gap + rng.normal(0, 0.02),
            own_in=own_in_level + rng.normal(0, 0.05), own_out=0.2,
            d_errin_dist=-0.018 + rng.normal(0, 0.004)))
  return rows


def selfcheck():
  r = analyse(_fake_rows(-0.3, -0.3))
  assert r['P-RS1_rl']['verdict'] == 'LEARNED-BUT-UNALIGNED', r['P-RS1_rl']
  assert r['P-RS2_sh']['verdict'] == 'LEARNED-BUT-UNALIGNED'
  assert r['P-RS1_rl']['s1']['claimable'] is True
  assert r['P-RS1_rl']['s2']['sign_replicates'] is True
  r = analyse(_fake_rows(0.0, 0.0, own_in_level=3.0))
  assert r['P-RS1_rl']['verdict'] == 'FAILURE-TO-LEARN', r['P-RS1_rl']
  assert r['P-RS1_rl']['s1']['claimable'] is False
  r = analyse(_fake_rows(0.3, -0.3))
  assert r['P-RS1_rl']['verdict'] == 'REGISTERED-ANOMALY'
  assert r['P-RS2_sh']['verdict'] == 'LEARNED-BUT-UNALIGNED'
  print('rlsh_ownlabel_read selfcheck PASS')


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd')
  sub.add_parser('selfcheck')
  ap.add_argument('--bundle')
  ap.add_argument('--out')
  args = ap.parse_args()
  if args.cmd == 'selfcheck':
    selfcheck()
    return
  assert args.bundle and args.out
  rows = load_rows(args.bundle)
  res = analyse(rows)
  res['bundle'] = args.bundle
  res['amendment'] = 'PREREG_refit_reads_amend1_20260810.md'
  res['unconsumed'] = ('per-run ridge_probe_finger_v1.json files present '
                       'in the bundle are NOT read (unregistered)')
  os.makedirs(args.out, exist_ok=True)
  with open(os.path.join(args.out, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(json.dumps({k: dict(verdict=res[k]['verdict'],
                            contrast=res[k]['own_minus_true_all'],
                            s1=res[k]['s1'], s2=res[k]['s2'])
                    for k in ('P-RS1_rl', 'P-RS2_sh')}, indent=1))


if __name__ == '__main__':
  main()
