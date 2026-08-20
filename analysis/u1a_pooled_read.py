"""Frozen reader — U1a pooled re-test (PREREG_u1a_pooled_20260820 +
Amendment 1). ONE execution. Second and LAST look.

Estimand: per-seed unfrozen interaction
  I(seed) = [task_s1 - task_s0] - [apt_s1 - apt_s0], AUC100k.
Fresh seeds 9-16 modes: ax1uzq1s{side} (task) / ax1uzfq1s{side} (apt).
Look-1 (seeds 1-8): the sha-pinned per_seed array in unfrozen_u1.json
(order = U1 submit loop; unlabeled in the artifact — PARTIAL attestation,
disclosed per amend1 M8).

PRIMARY (amend1 M7 conjunction): pooled n=16 exact 2^16 sign-flip
(domains_read.one_sample): FIRE iff perm p < .0294 AND BCa CI lower > 0
AND fresh-only (9-16) point estimate > 0.
Branches: U1A-CONFIRMED / U1A-REVERSED (sig negative, verbatim) /
NO-CALL-UNDERPOWERED (no fire AND realized MDE80 > 145.007775 — amend1
B4; SUGGESTIVE persists on power grounds) / U1A-RETIRED (powered no-fire).
MDE80 = (z_{1-a/2} + z_{.80}) * sd / sqrt(n) — normal approximation,
disclosed.

Usage: python -m analysis.u1a_pooled_read --auc <csv> --witness <json> \
           --output <dir>       (selfcheck: --selfcheck)
"""

import argparse
import json
import os

import numpy as np

from analysis.domains_read import one_sample
from analysis.rescue_slate_common import (
    Refusal, _ndtri, check_config_identity, check_modal_nep, check_witness,
    load_rows, refuse, sha256_file, side_auc)

PREREG = 'PREREG_u1a_pooled_20260820.md'
AMEND = 'PREREG_u1a_pooled_amend1_20260820.md'
ALPHA = 0.0294
SEEDS_FRESH = tuple(range(9, 17))
UPDATES = 500000
LOOK1_POINT = 145.007775
U1_JSON = 'artifacts/unfrozen_u1_20260801/unfrozen_u1.json'
U1_SHA = '0b7dd65c01e115a0ff71d959e5e74f62371de3c28aabd23f9b28c0ddc7d0c288'
MODES = dict(task='ax1uzq1s{side}', apt='ax1uzfq1s{side}')
FITS = [f'ax1wm_finger_{p}q1s{sd}_seed{s}'
        for p in ('uz', 'uzf') for sd in ('0', '1') for s in SEEDS_FRESH]


def interaction(rows, seeds):
  cells = {}
  for arm, fmt in MODES.items():
    for side in ('0', '1'):
      cells[arm, side] = side_auc(rows, fmt.format(side=side),
                                  set(seeds), 'finger')
  return np.array([(cells['task', '1'][s] - cells['task', '0'][s])
                   - (cells['apt', '1'][s] - cells['apt', '0'][s])
                   for s in seeds])


def mde80(sd, n, alpha):
  return (_ndtri(1 - alpha / 2) + _ndtri(0.80)) * sd / np.sqrt(n)


def run(args):
  rows = load_rows(args.auc)
  w = check_witness(args.witness, FITS, UPDATES)
  meta = json.load(open(args.witness)).get('_meta', {})
  for field in ('config_ref_paths', 'config_ref_shas'):
    if not meta.get(field):
      refuse(f'witness _meta.{field} absent — the amend1 M9 preflight must '
             'record the finger_refit_20260810 reference configs')
  groups = {f'{p}s{sd}': [f'ax1wm_finger_{p}q1s{sd}_seed{s}'
                          for s in SEEDS_FRESH]
            for p in ('uz', 'uzf') for sd in ('0', '1')}
  check_config_identity(w, groups)
  modes = {fmt.format(side=sd) for fmt in MODES.values() for sd in ('0', '1')}
  modal = check_modal_nep(rows, modes, set(SEEDS_FRESH), 'finger')

  u1_path = args.u1 or U1_JSON
  if sha256_file(u1_path) != U1_SHA:
    refuse('U1 artifact sha mismatch')
  look1 = np.asarray(json.load(open(u1_path))['p_u1a_interaction']['per_seed'],
                     float)
  if len(look1) != 8:
    refuse('look-1 per_seed length != 8')

  fresh = interaction(rows, SEEDS_FRESH)
  pooled_x = np.concatenate([look1, fresh])
  pooled = one_sample(pooled_x)
  fresh_mean = float(fresh.mean())
  realized_mde = float(mde80(pooled['sd'], 16, ALPHA))

  fire = (pooled['perm_p'] < ALPHA and pooled['ci'][0] > 0
          and fresh_mean > 0)
  rev = pooled['perm_p'] < ALPHA and pooled['ci'][1] < 0
  if fire:
    overall = ('U1A-CONFIRMED: the headline interaction is licensed under '
               'unfrozen adaptation; the SUGGESTIVE label lifts')
  elif rev:
    overall = 'U1A-REVERSED: significant negative — reported verbatim'
  elif realized_mde > LOOK1_POINT:
    overall = ('NO-CALL-UNDERPOWERED: no fire and realized MDE80 '
               f'({realized_mde:.1f}) exceeds the look-1 point estimate '
               '(+145.0) — underpowered-non-replication; SUGGESTIVE '
               'persists on power grounds (amend1 B4)')
  else:
    overall = ('U1A-RETIRED: powered non-fire — SUGGESTIVE becomes the '
               'permanent label and the paper carries the unfrozen '
               'non-confirmation wherever the headline is stated')

  out = dict(prereg=PREREG, amendment=AMEND, overall=overall,
             primary_pooled=dict(pooled, alpha=ALPHA, fire=bool(fire)),
             fresh_only=dict(mean=fresh_mean, n=len(fresh),
                             note='non-decisional sign conjunct (M7)'),
             realized_mde80=realized_mde, look1_point=LOOK1_POINT,
             alpha_honesty='two-look union bound <= .0794 (amend1 B5)',
             gates=dict(fits=len(FITS), updates=UPDATES, modal_n_ep=modal))
  os.makedirs(args.output, exist_ok=True)
  json.dump(out, open(os.path.join(args.output, 'read.json'), 'w'), indent=1)
  print(f"pooled {pooled['mean']:+.2f} {pooled['ci']} p {pooled['perm_p']:.5f} "
        f"fresh {fresh_mean:+.2f} MDE80 {realized_mde:.1f}")
  print(overall.split(':')[0])
  print(f"-> {args.output}/read.json")


def _fixture(base, effect, sd_noise=60.0, seed=0):
  import csv as csvlib
  rng = np.random.default_rng(seed)
  rows = []
  for p, arm in (('uz', 'task'), ('uzf', 'apt')):
    for side in ('0', '1'):
      for s in SEEDS_FRESH:
        v = 150.0 + (effect if (arm == 'task' and side == '1') else 0.0)
        rows.append(dict(run_id=f'adapt_ax1{p}q1s{side}_finger_seed{s}',
                         mode=f'ax1{p}q1s{side}', domain='finger', seed=s,
                         milestone=500000, auc50k=0, n_ep_50k=48,
                         auc100k=v + rng.normal(0, sd_noise), n_ep_100k=96,
                         auc125k=0, n_ep_125k=112, final10=0, qc_pass=1))
  path = os.path.join(base, 'auc.csv')
  with open(path, 'w', newline='') as f:
    wcsv = csvlib.DictWriter(f, fieldnames=list(rows[0]))
    wcsv.writeheader()
    [wcsv.writerow(r) for r in rows]
  wit = {r: dict(counters=UPDATES, config_sha_excl_seed=r.split('_seed')[0])
         for r in FITS}
  wit['_meta'] = dict(config_ref_paths=['x'], config_ref_shas=['y'])
  wp = os.path.join(base, 'witness.json')
  json.dump(wit, open(wp, 'w'))
  return path, wp


def selfcheck():
  import tempfile
  # RETIRED is the else-branch (trivially reached); fixtures cover the
  # three decision-bearing branches. NO-CALL needs wide noise so the
  # realized MDE80 clears the 145.0 look-1 point (amend1 B4 semantics).
  for effect, noise, expect in ((350.0, 60.0, 'U1A-CONFIRMED'),
                                (0.0, 250.0, 'NO-CALL'),
                                (-1200.0, 10.0, 'U1A-REVERSED')):
    base = tempfile.mkdtemp(prefix='u1a_sc_')
    auc, wit = _fixture(base, effect, sd_noise=noise, seed=3)
    run(argparse.Namespace(auc=auc, witness=wit, u1=None,
                           output=os.path.join(base, 'out')))
    r = json.load(open(os.path.join(base, 'out', 'read.json')))
    assert r['overall'].startswith(expect), (effect, r['overall'])
  n_ref = 0
  for mutate in ('no_ref', 'short', 'missing_mode'):
    base = tempfile.mkdtemp(prefix='u1a_sc_')
    auc, wit = _fixture(base, 100.0, seed=4)
    w = json.load(open(wit))
    if mutate == 'no_ref':
      del w['_meta']['config_ref_paths']
    elif mutate == 'short':
      w[FITS[0]]['counters'] = 1
    elif mutate == 'missing_mode':
      s = open(auc).read().replace('ax1uzfq1s1', 'ax1WRONG')
      open(auc, 'w').write(s)
    json.dump(w, open(wit, 'w'))
    try:
      run(argparse.Namespace(auc=auc, witness=wit, u1=None,
                             output=os.path.join(base, 'out')))
    except Refusal as e:
      n_ref += 1
      print(f'  refusal OK [{mutate}]: {e}')
    else:
      raise AssertionError(mutate)
  assert n_ref == 3
  print('u1a_pooled_read selfcheck PASS (3 branches + 3 refusals; pins: '
        f'alpha {ALPHA}, U1 sha {U1_SHA[:12]}..., look1 point {LOOK1_POINT})')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--auc')
  ap.add_argument('--witness')
  ap.add_argument('--u1', default=None)
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
