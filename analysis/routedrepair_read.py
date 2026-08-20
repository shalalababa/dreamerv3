"""Frozen reader — routed repair kill (PREREG_routedrepair_20260820 +
Amendment 1). ONE execution.

Arms (finger q1 side1, seeds 1-8, regenerated fq1s1 donors):
  repair   mode 'ax1rrpq1s1'  — rgo-refit (500k) of the apt donor -> adapt
  aptctl   mode 'ax1raftq1s1' — plain full fine-tune of the SAME donor
                                (in-wave baseline, amend1 B13)
  snp      mode 'ax1rspq1s1'  — shrink-and-perturb (0.5, 0.01) -> adapt
                                (descriptive free arm)

D1 (amend1 B12): paired per-donor [repair_i - aptctl_i], exact 2^8
sign-flip + BCa; FIRE iff perm p < .05 AND CI lower > 0. The July pin
89.8984625 is descriptive context only (cross-era, demoted by amend1).
D2: repair one_sample vs the constant 147.6823; kill bar = repair mean >=
SCRATCH - 25.0 (computed here; = 122.6823). Neighborhood, not equivalence.
PROCEED iff D1 fires AND D2 bar met — and licenses only the follow-up
registrations (amend1 M14: this wave cannot separate repair from
overwrite). D1-only = DROP ("routing helps, does not reach scratch's
neighborhood"). D1 significant negative = REVERSED. Else DROP.

Gates: donor-sha pairing (repair_i and aptctl_i and snp_i must record the
SAME donor ckpt sha per seed); refit counters == 500000 x8; qc; STRICT
modal n_ep.

Usage: python -m analysis.routedrepair_read --auc <csv> --witness <json> \
    --output <dir>   (--selfcheck)
"""

import argparse
import json
import os

import numpy as np

from analysis.domains_read import one_sample
from analysis.rescue_slate_common import (
    Refusal, check_modal_nep, load_rows, refuse, side_auc)

PREREG = 'PREREG_routedrepair_20260820.md'
AMEND = 'PREREG_routedrepair_amend1_20260820.md'
ALPHA = 0.05
SEEDS = tuple(range(1, 9))
SCRATCH = 147.6823
BAR = SCRATCH - 25.0          # 122.6823, computed (amend1 m1)
JULY_PIN = 89.8984625         # descriptive context only (amend1 B12/B13)
REFIT_UPDATES = 500000
MODES = dict(repair='ax1rrpq1s1', aptctl='ax1raftq1s1', snp='ax1rspq1s1')
REFITS = [f'ax1wm_finger_rrpq1s1_seed{s}' for s in SEEDS]


def run(args):
  rows = load_rows(args.auc)
  w = json.load(open(args.witness))
  meta = w.get('_meta', {})
  if not str(meta.get('donor_generation', '')).startswith('REFIT_20260808'):
    refuse('witness _meta.donor_generation missing/wrong (amend1 donor '
           'declaration: August REFIT_20260808 generation)')
  if meta.get('latest_ckpt_500k_8of8') is not True:
    refuse('witness _meta.latest_ckpt_500k_8of8 is not true (amend1: '
           'latest_ckpt() must resolve 500000 for 8/8 donors)')
  for r in REFITS:
    if r not in w:
      refuse(f'witness missing refit {r}')
    if int(w[r]['counters']) != REFIT_UPDATES:
      refuse(f'{r}: refit counters {w[r]["counters"]} != {REFIT_UPDATES}')
  for s in SEEDS:
    shas = {w.get(f'donor_sha/{arm}/seed{s}') for arm in MODES}
    if len(shas) != 1 or None in shas:
      refuse(f'donor sha pairing broken at seed {s}: {shas}')
  modal = check_modal_nep(rows, set(MODES.values()), set(SEEDS), 'finger')

  auc = {arm: side_auc(rows, mode, set(SEEDS), 'finger')
         for arm, mode in MODES.items()}
  rep = np.array([auc['repair'][s] for s in SEEDS])
  ctl = np.array([auc['aptctl'][s] for s in SEEDS])
  snp = np.array([auc['snp'][s] for s in SEEDS])

  d1 = one_sample(rep - ctl)
  d1_fire = d1['perm_p'] < ALPHA and d1['ci'][0] > 0
  d1_rev = d1['perm_p'] < ALPHA and d1['ci'][1] < 0
  d2 = one_sample(rep - SCRATCH)
  d2_bar = float(rep.mean()) >= BAR

  if d1_rev:
    overall = ('ROUTED-REPAIR-REVERSED: the rgo-refit is significantly '
               'WORSE than plain fine-tuning of the same donor — reported '
               'verbatim; DROP')
  elif d1_fire and d2_bar:
    overall = ('PROCEED: routing beats full fine-tuning of the same sunk '
               'trunk AND reaches the scratch neighborhood. NOTE (amend1 '
               'M14): this wave cannot separate repair from overwrite; '
               'PROCEED licenses the short-refit, random-init-control and '
               'matched-budget follow-up registrations — never the '
               'deployable salvage claim by itself')
  elif d1_fire:
    overall = ('DROP (D1-only): routing helps but does not reach '
               "scratch's neighborhood — one sentence in the paper")
  else:
    overall = 'DROP: no separation from plain fine-tuning'

  out = dict(prereg=PREREG, amendment=AMEND, overall=overall,
             d1=dict(d1, alpha=ALPHA, fire=bool(d1_fire)),
             d2=dict(d2, bar=BAR, bar_met=bool(d2_bar),
                     note='vs constant; pin error not propagated — '
                          'anti-conservative, disclosed'),
             arms=dict(repair_mean=float(rep.mean()),
                       aptctl_mean=float(ctl.mean()),
                       snp_mean=float(snp.mean()),
                       july_pin_descriptive=JULY_PIN, scratch=SCRATCH),
             gates=dict(refits=len(REFITS), refit_updates=REFIT_UPDATES,
                        modal_n_ep=modal))
  os.makedirs(args.output, exist_ok=True)
  json.dump(out, open(os.path.join(args.output, 'read.json'), 'w'), indent=1)
  print(f"D1 repair-ctl {d1['mean']:+.2f} {d1['ci']} p {d1['perm_p']:.4f}; "
        f"repair mean {rep.mean():.1f} vs bar {BAR}")
  print(overall.split(':')[0])
  print(f"-> {args.output}/read.json")


def _fixture(base, d1_eff, repair_level, sd_noise=12.0, seed=0):
  import csv as csvlib
  rng = np.random.default_rng(seed)
  rows = []
  for arm, mode in MODES.items():
    for s in SEEDS:
      v = dict(repair=repair_level, aptctl=repair_level - d1_eff,
               snp=95.0)[arm]
      rows.append(dict(run_id=f'adapt_{mode}_finger_seed{s}', mode=mode,
                       domain='finger', seed=s, milestone=500000, auc50k=0,
                       n_ep_50k=48, auc100k=v + rng.normal(0, sd_noise),
                       n_ep_100k=96, auc125k=0, n_ep_125k=112, final10=0,
                       qc_pass=1))
  auc = os.path.join(base, 'auc.csv')
  with open(auc, 'w', newline='') as f:
    wcsv = csvlib.DictWriter(f, fieldnames=list(rows[0]))
    wcsv.writeheader()
    [wcsv.writerow(r) for r in rows]
  wit = {r: dict(counters=REFIT_UPDATES) for r in REFITS}
  for s in SEEDS:
    for arm in MODES:
      wit[f'donor_sha/{arm}/seed{s}'] = f'donor{s}'
  wit['_meta'] = dict(
      donor_generation='REFIT_20260808+seed5_completion_20260820',
      latest_ckpt_500k_8of8=True)
  wp = os.path.join(base, 'witness.json')
  json.dump(wit, open(wp, 'w'))
  return auc, wp


def selfcheck():
  import tempfile
  for d1_eff, level, expect in ((45.0, 150.0, 'PROCEED'),
                                (45.0, 100.0, 'DROP (D1-only)'),
                                (0.0, 100.0, 'DROP:'),
                                (-45.0, 80.0, 'ROUTED-REPAIR-REVERSED')):
    base = tempfile.mkdtemp(prefix='rrp_sc_')
    auc, wit = _fixture(base, d1_eff, level, seed=9)
    run(argparse.Namespace(auc=auc, witness=wit,
                           output=os.path.join(base, 'out')))
    r = json.load(open(os.path.join(base, 'out', 'read.json')))
    assert r['overall'].startswith(expect.rstrip(':')), \
        (d1_eff, level, r['overall'])
  n_ref = 0
  for mutate in ('short_refit', 'donor_split', 'missing_donor',
                 'wrong_generation', 'incomplete_donors'):
    base = tempfile.mkdtemp(prefix='rrp_sc_')
    auc, wit = _fixture(base, 40.0, 150.0, seed=10)
    w = json.load(open(wit))
    if mutate == 'short_refit':
      w[REFITS[0]]['counters'] = 100
    elif mutate == 'donor_split':
      w['donor_sha/repair/seed3'] = 'OTHER'
    elif mutate == 'missing_donor':
      del w['donor_sha/snp/seed5']
    elif mutate == 'wrong_generation':
      w['_meta']['donor_generation'] = 'JULY_ORIGINALS'
    elif mutate == 'incomplete_donors':
      w['_meta']['latest_ckpt_500k_8of8'] = False
    json.dump(w, open(wit, 'w'))
    try:
      run(argparse.Namespace(auc=auc, witness=wit,
                             output=os.path.join(base, 'out')))
    except Refusal as e:
      n_ref += 1
      print(f'  refusal OK [{mutate}]: {e}')
    else:
      raise AssertionError(mutate)
  assert n_ref == 5
  print('routedrepair_read selfcheck PASS (4 branches + 5 refusals; pins: '
        f'bar {BAR}, scratch {SCRATCH}, july pin {JULY_PIN} descriptive, '
        f'refit updates {REFIT_UPDATES})')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--auc')
  ap.add_argument('--witness')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  a = ap.parse_args()
  if a.selfcheck:
    selfcheck()
  else:
    assert a.auc and a.witness and a.output
    run(a)


if __name__ == '__main__':
  main()
