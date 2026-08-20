"""Frozen reader — carrier fresh replication (PREREG_carrier_freshrep_20260820).

ONE execution. LAST look on the interaction-carrier estimand.

Estimand (Amendment-1 form, unchanged): per-seed
  D(seed) = B_rgo(seed) - B_sgb(seed),  B_arm = AUC100k(s1) - AUC100k(s0).

  PRIMARY   fresh-only seeds 17-56 (n=40): one_sample_auto(D_fresh),
            FIRE iff perm_p < .05 (two-sided) — BCa CI reported.
  SECONDARY pooled seeds 1-56 (n=56): RE-ENTRY iff perm_p <= .016667.
            Look-1 rows (seeds 1-16) come ONLY from the sha-pinned
            Amendment-1 artifact csv — never re-measured.

Branches:
  CARRIER-REPLICATES-AND-REENTERS   fresh fires AND pooled <= .016667
  CARRIER-REPLICATES-NO-REENTRY     fresh fires, pooled misses
  CARRIER-RETIRED                   fresh does not fire — the estimand is
                                    permanently retired (pre-stated)
Scaling12 re-entry is a FAMILY fact: reported true iff pooled re-entry
lands (step-up through rank 7 re-admits rank 6); no new Scaling12 data.

Gates (refusal = read NOT consumed): witnessed fit counters == 500000
x160; per-(arm,side) config sha identity; ops preflight flag
amend1_config_match; STRICT modal n_ep_100k over fresh rows; qc_pass;
look-1 csv sha pin.

Usage:
  python -m analysis.carrier_freshrep_read --auc <bundle>/adaptation_auc.csv \
      --witness <bundle>/carrier_witness.json --output <dir>
Selfcheck: python -m analysis.carrier_freshrep_read --selfcheck
"""

import argparse
import json
import os

import numpy as np

from analysis.rescue_slate_common import (
    Refusal, check_config_identity, check_modal_nep, check_witness,
    load_rows, one_sample_auto, refuse, sha256_file, side_auc)

PREREG = 'PREREG_carrier_freshrep_20260820.md'
ARMS = ('rgo', 'sgb')
SIDES = ('0', '1')
SEEDS_FRESH = tuple(range(17, 57))     # 40 fresh seeds
SEEDS_LOOK1 = tuple(range(1, 17))
ALPHA_FRESH = 0.05
ALPHA_POOLED = 7 * 0.05 / 21           # BH re-entry target (amend1 m2)
UPDATES = 500000
SD_PLAN = 91.26                        # planning constant, reported only
MODAL_NEP = 96                         # look-1 modal, pinned (amend1 B3)
DOMAIN = 'finger'
AMEND1_CSV = 'artifacts/p3_amend1_20260717/auc_pooled_1_16.csv'
AMEND1_SHA = ('664fd549636e7d74ef1aa97457d32054'
              'eef0a06434efb27b7bd55761744edd7b')


def d_contrast(rows, seeds, domain=DOMAIN):
  b = {}
  for arm in ARMS:
    s0 = side_auc(rows, f'ax1{arm}q1s0', set(seeds), domain)
    s1 = side_auc(rows, f'ax1{arm}q1s1', set(seeds), domain)
    b[arm] = {s: s1[s] - s0[s] for s in seeds}
  return np.array([b['rgo'][s] - b['sgb'][s] for s in seeds])


def check_linkage(linkage_path, fits):
  """amend1 B2: every fit's replay source must resolve to the registered
  side path axis1_finger/q1/side{side} (rescue_load precedent)."""
  with open(linkage_path) as f:
    lk = json.load(f)
  for run in fits:
    if run not in lk:
      refuse(f'linkage missing {run}')
    side = run.split('q1s')[1][0]
    if f'axis1_finger/q1/side{side}' not in lk[run]:
      refuse(f'{run}: replay source {lk[run]} not the registered side path')


def check_invocation_pairing(w):
  """amend1 B2: rgo and sgb members of each (seed, side) pair must share
  an invocation_id (no cross-invocation pairing)."""
  for sd in SIDES:
    for s in SEEDS_FRESH:
      a = w[f'ax1wm_finger_rgoq1s{sd}_seed{s}'].get('invocation_id')
      b = w[f'ax1wm_finger_sgbq1s{sd}_seed{s}'].get('invocation_id')
      if a is None or b is None or a != b:
        refuse(f'invocation pairing broken at s{sd} seed {s}: {a} vs {b}')


def run(args):
  rows = load_rows(args.auc)
  fits = [f'ax1wm_finger_{a}q1s{sd}_seed{s}'
          for a in ARMS for sd in SIDES for s in SEEDS_FRESH]
  w = check_witness(args.witness, fits, UPDATES)
  meta = json.load(open(args.witness)).get('_meta', {})
  if meta.get('amend1_config_match') is not True:
    refuse('witness _meta.amend1_config_match is not true (ops preflight '
           'must byte-compare arm configs vs Amendment 1, seed excepted)')
  groups = {f'{a}s{sd}': [f'ax1wm_finger_{a}q1s{sd}_seed{s}'
                          for s in SEEDS_FRESH]
            for a in ARMS for sd in SIDES}
  check_config_identity(w, groups)
  check_linkage(args.linkage, fits)
  check_invocation_pairing(w)
  modes = {f'ax1{a}q1s{sd}' for a in ARMS for sd in SIDES}
  modal = check_modal_nep(rows, modes, set(SEEDS_FRESH), DOMAIN,
                        expected=MODAL_NEP)

  look1_csv = args.look1 or AMEND1_CSV
  sha = sha256_file(look1_csv)
  if sha != AMEND1_SHA:
    refuse(f'look-1 csv sha {sha[:16]} != pinned {AMEND1_SHA[:16]}')

  d_fresh = d_contrast(rows, SEEDS_FRESH)
  d_look1 = d_contrast(load_rows(look1_csv), SEEDS_LOOK1)
  fresh = one_sample_auto(d_fresh)
  pooled = one_sample_auto(np.concatenate([d_look1, d_fresh]))

  # amend1 B1: FIRE is a conjunction (perm p AND CI lower > 0); a
  # significant negative is CARRIER-REVERSED, never a replication.
  fresh_fire = fresh['perm_p'] < ALPHA_FRESH and fresh['ci'][0] > 0
  reversed_ = fresh['perm_p'] < ALPHA_FRESH and fresh['ci'][1] < 0
  reentry = pooled['perm_p'] <= ALPHA_POOLED and pooled['ci'][0] > 0
  if reversed_:
    overall = ('CARRIER-REVERSED: the fresh primary is significant with a '
               'NEGATIVE sign — reported verbatim; the estimand is retired; '
               'no BH re-entry and no Scaling12 re-admission is claimed')
  elif fresh_fire and reentry:
    overall = ('CARRIER-REPLICATES-AND-REENTERS: fresh-only replication '
               'fires and the pooled estimate clears the BH re-entry '
               'target — the carrier re-enters the 21-primary family and '
               'Scaling12 re-admits by step-up arithmetic')
  elif fresh_fire:
    overall = ('CARRIER-REPLICATES-NO-REENTRY: the fresh replication '
               'fires at alpha=.05 but the pooled estimate misses the '
               '.016667 re-entry target — the carrier is real-but-demoted; '
               'family status unchanged; NO further look exists')
  else:
    overall = ('CARRIER-RETIRED: the fresh-only primary does not fire — '
               'the interaction-carrier estimand is PERMANENTLY RETIRED '
               '(pre-stated consequence); the LEVEL claim (8-Aug estimand '
               'split) is unaffected')

  out = dict(prereg=PREREG, amendment='PREREG_carrier_freshrep_amend1_20260820.md',
             overall=overall,
             primary_fresh=dict(fresh, alpha=ALPHA_FRESH, fire=bool(fresh_fire)),
             secondary_pooled=dict(pooled, alpha=ALPHA_POOLED,
                                   reentry=bool(reentry)),
             scaling12_readmitted=bool(fresh_fire and reentry),
             look1=dict(csv=look1_csv, sha256=sha,
                        mean=float(d_look1.mean()), n=len(d_look1)),
             gates=dict(fits_witnessed=len(fits), updates=UPDATES,
                        modal_n_ep=modal, sd_plan=SD_PLAN))
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(out, f, indent=1)
  print(f"fresh {fresh['mean']:+.2f} {fresh['ci']} perm_p {fresh['perm_p']:.5f}"
        f" -> fire={fresh_fire}")
  print(f"pooled {pooled['mean']:+.2f} {pooled['ci']} perm_p "
        f"{pooled['perm_p']:.6f} -> reentry={reentry}")
  print(overall.split(':')[0])
  print(f"-> {args.output}/read.json")


def _fixture(base, effect, sd=90.0, seed=0):
  import csv as csvlib
  rng = np.random.default_rng(seed)
  rows = []
  for a in ARMS:
    for sdd in SIDES:
      for s in SEEDS_FRESH:
        base_v = 150.0 + (effect if (a == 'rgo' and sdd == '1') else 0.0)
        rows.append(dict(run_id=f'adapt_ax1{a}q1s{sdd}_finger_seed{s}_ckpt500000',
                         mode=f'ax1{a}q1s{sdd}', domain=DOMAIN, seed=s,
                         milestone=500000, auc50k=0,
                         n_ep_50k=48, auc100k=base_v + rng.normal(0, sd),
                         n_ep_100k=96, auc125k=0, n_ep_125k=112,
                         final10=0, qc_pass=1))
  path = os.path.join(base, 'auc.csv')
  with open(path, 'w', newline='') as f:
    wcsv = csvlib.DictWriter(f, fieldnames=list(rows[0]))
    wcsv.writeheader()
    for r in rows:
      wcsv.writerow(r)
  wit = {f'ax1wm_finger_{a}q1s{sdd}_seed{s}':
         dict(counters=UPDATES, config_sha_excl_seed=f'{a}{sdd}sha',
              invocation_id=f'inv{sdd}{s}')
         for a in ARMS for sdd in SIDES for s in SEEDS_FRESH}
  wit['_meta'] = dict(amend1_config_match=True)
  wpath = os.path.join(base, 'witness.json')
  json.dump(wit, open(wpath, 'w'))
  lk = {f'ax1wm_finger_{a}q1s{sdd}_seed{s}':
        f'/runroot/axis1_finger/q1/side{sdd}'
        for a in ARMS for sdd in SIDES for s in SEEDS_FRESH}
  lpath = os.path.join(base, 'linkage.json')
  json.dump(lk, open(lpath, 'w'))
  return path, wpath, lpath


def selfcheck():
  import tempfile
  ok_refusals = 0
  # branch fixtures (look-1 rows come from the REAL pinned csv)
  for effect, expect in ((120.0, 'CARRIER-REPLICATES'),
                         (0.0, 'CARRIER-RETIRED'),
                         (-120.0, 'CARRIER-REVERSED')):
    base = tempfile.mkdtemp(prefix='cfr_sc_')
    auc, wit, lk = _fixture(base, effect, seed=1)
    args = argparse.Namespace(auc=auc, witness=wit, linkage=lk, look1=None,
                              output=os.path.join(base, 'out'))
    run(args)
    r = json.load(open(os.path.join(base, 'out', 'read.json')))
    assert r['overall'].startswith(expect), (effect, r['overall'])
    if expect == 'CARRIER-REVERSED':
      assert r['scaling12_readmitted'] is False and not r['primary_fresh']['fire']
  # refusal legs
  for mutate in ('short_counters', 'missing_fit', 'sha_split', 'no_meta',
                 'qc_fail', 'nep_drift', 'bad_look1', 'bad_linkage',
                 'wrong_side_link', 'invocation_split', 'nep_wrong_modal'):
    base = tempfile.mkdtemp(prefix='cfr_sc_')
    auc, wit, lk = _fixture(base, 100.0, seed=2)
    w = json.load(open(wit))
    if mutate == 'short_counters':
      w['ax1wm_finger_rgoq1s0_seed17']['counters'] = 200000
    elif mutate == 'missing_fit':
      del w['ax1wm_finger_sgbq1s1_seed56']
    elif mutate == 'sha_split':
      w['ax1wm_finger_rgoq1s1_seed20']['config_sha_excl_seed'] = 'other'
    elif mutate == 'no_meta':
      del w['_meta']
    elif mutate == 'invocation_split':
      w['ax1wm_finger_rgoq1s0_seed30']['invocation_id'] = 'other_inv'
    json.dump(w, open(wit, 'w'))
    if mutate == 'bad_linkage':
      l = json.load(open(lk)); del l['ax1wm_finger_rgoq1s0_seed17']
      json.dump(l, open(lk, 'w'))
    if mutate == 'wrong_side_link':
      l = json.load(open(lk))
      l['ax1wm_finger_sgbq1s1_seed20'] = '/runroot/axis1_finger/q1/side0'
      json.dump(l, open(lk, 'w'))
    if mutate == 'nep_wrong_modal':
      import csv as csvlib
      rws = list(csvlib.DictReader(open(auc)))
      for r0 in rws:
        r0['n_ep_100k'] = 95
      with open(auc, 'w', newline='') as f:
        wcsv = csvlib.DictWriter(f, fieldnames=list(rws[0]))
        wcsv.writeheader()
        for r0 in rws:
          wcsv.writerow(r0)
    look1 = None
    if mutate == 'qc_fail':
      import csv as csvlib
      rows = list(csvlib.DictReader(open(auc)))
      rows[0]['qc_pass'] = 0
      with open(auc, 'w', newline='') as f:
        wcsv = csvlib.DictWriter(f, fieldnames=list(rows[0]))
        wcsv.writeheader()
        for r in rows:
          wcsv.writerow(r)
    if mutate == 'nep_drift':
      s = open(auc).read().replace(',96,', ',95,', 1)
      open(auc, 'w').write(s)
    if mutate == 'bad_look1':
      look1 = auc  # wrong file -> sha mismatch
    try:
      run(argparse.Namespace(auc=auc, witness=wit, linkage=lk, look1=look1,
                             output=os.path.join(base, 'out')))
    except Refusal as e:
      ok_refusals += 1
      print(f'  refusal OK [{mutate}]: {e}')
    else:
      raise AssertionError(f'{mutate}: expected refusal')
  assert ok_refusals == 11
  print('carrier_freshrep_read selfcheck PASS (3 branches + 11 refusals; '
        'literal pins: alpha .05/.016667, updates 500000, seeds 17-56, '
        f'look-1 sha {AMEND1_SHA[:12]}...)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--auc')
  ap.add_argument('--witness')
  ap.add_argument('--linkage')
  ap.add_argument('--look1', default=None)
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
  else:
    assert args.auc and args.witness and args.linkage and args.output
    run(args)


if __name__ == '__main__':
  main()
