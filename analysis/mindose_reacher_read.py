"""Frozen reader — MIN-DOSE reacher kill (PREREG_mindose_reacher_20260820
+ Amendment 1). ONE execution.

Arms (reacher domain): dose = mode 'ax1mdd1' (level-1 curated buffer,
task-mode fits, UNFROZEN adapts, seeds 1-8); scratch = mode 'scratch'
(same adaptation protocol from random init, seeds 1-8).

PRIMARY (amend1 B11): two_sample [dose1 - scratch] (capdescent lineage:
label-permutation p + BCa). PROCEED iff perm p < .05 AND CI lower > 0
(branch string carries the amend1 M13 budget-asymmetry sentence: PROCEED
licenses only a matched-budget follow-up, never a recipe claim).
Significant negative = DROP-WITH-REVERSAL-RECORDED. Else DROP.

Gates: buffer manifest witness (occupancy in [0.08, 0.20], holdout flag
true, first-draw attestation — amend1 B10); fit counters == 500000 x8;
adapt witnesses x16; qc; STRICT modal n_ep (both arms share the adapt
protocol -> one modal). Realized MDE80 printed with any null.

Usage: python -m analysis.mindose_reacher_read --auc <csv> \
    --witness <json> --buffer_manifest <json> --output <dir> (--selfcheck)
"""

import argparse
import json
import os

import numpy as np

from analysis.capdescent_read import two_sample
from analysis.rescue_slate_common import (
    Refusal, _ndtri, check_modal_nep, check_witness, load_rows, refuse,
    side_auc)

PREREG = 'PREREG_mindose_reacher_20260820.md'
AMEND = 'PREREG_mindose_reacher_amend1_20260820.md'
ALPHA = 0.05
SEEDS = tuple(range(1, 9))
UPDATES = 500000
OCC_WINDOW = (0.08, 0.20)
MODE_DOSE, MODE_SCRATCH = 'ax1mdd1', 'scratch'
FITS = [f'ax1wm_reacher_mdd1_seed{s}' for s in SEEDS]


def run(args):
  rows = load_rows(args.auc)
  bm = json.load(open(args.buffer_manifest))
  occ = float(bm['occupancy'])
  if not OCC_WINDOW[0] <= occ <= OCC_WINDOW[1]:
    refuse(f'buffer occupancy {occ} outside registered window {OCC_WINDOW}')
  if not (bm.get('holdout_applied') is True
          or bm.get('holdout_disposition') == 'no-probeset-exists'):
    refuse('buffer manifest lacks the amend1 B10 holdout attestation '
           '(holdout_applied=true OR holdout_disposition='
           '"no-probeset-exists" — no reacher probeset existed at freeze)')
  if bm.get('first_draw') is not True:
    refuse('buffer manifest lacks first_draw=true (amend1 B10 — the FIRST '
           'invocation of the pinned command is the registered draw)')
  w = check_witness(args.witness, FITS, UPDATES)
  del w  # presence + counters is the gate; no pairing across arms (m6)
  modal = check_modal_nep(rows, {MODE_DOSE, MODE_SCRATCH}, set(SEEDS),
                          'reacher')
  dose = side_auc(rows, MODE_DOSE, set(SEEDS), 'reacher')
  scr = side_auc(rows, MODE_SCRATCH, set(SEEDS), 'reacher')
  a = np.array([dose[s] for s in SEEDS])
  b = np.array([scr[s] for s in SEEDS])
  st = two_sample(a, b)
  mde = float((_ndtri(1 - ALPHA / 2) + _ndtri(0.80))
              * np.sqrt(a.var(ddof=1) / 8 + b.var(ddof=1) / 8))
  fire = st['perm_p'] < ALPHA and st['ci'][0] > 0
  rev = st['perm_p'] < ALPHA and st['ci'][1] < 0
  if fire:
    overall = ('PROCEED: dose-1 unfrozen clears scratch on reacher — the '
               'recipe has legs outside finger. NOTE (amend1 M13): the '
               'comparison is unmatched-budget by design; this PROCEED '
               'licenses only a matched-budget follow-up registration, '
               'never a recipe claim by itself')
  elif rev:
    overall = ('DROP-WITH-REVERSAL-RECORDED: dose-1 is significantly '
               'BELOW scratch on reacher — reported verbatim')
  else:
    overall = ('DROP: no separation — the recipe has no demonstrated legs '
               'outside finger; no seed extension, no re-curation, no '
               'second look (frozen kill criterion)')
  out = dict(prereg=PREREG, amendment=AMEND, overall=overall,
             primary=dict(st, alpha=ALPHA, fire=bool(fire)),
             realized_mde80=mde,
             arms=dict(dose_mean=float(a.mean()), scratch_mean=float(b.mean())),
             buffer=dict(occupancy=occ, window=list(OCC_WINDOW)),
             gates=dict(fits=len(FITS), updates=UPDATES, modal_n_ep=modal),
             secondary_note=('dose vs W0 reacher task cells is cross-protocol '
                             '(frozen vs unfrozen) — context only (amend1 M12)'))
  os.makedirs(args.output, exist_ok=True)
  json.dump(out, open(os.path.join(args.output, 'read.json'), 'w'), indent=1)
  print(f"dose-scratch {st['diff']:+.2f} {st['ci']} p {st['perm_p']:.4f} "
        f"MDE80 {mde:.1f}")
  print(overall.split(':')[0])
  print(f"-> {args.output}/read.json")


def _fixture(base, effect, sd_noise=25.0, seed=0):
  import csv as csvlib
  rng = np.random.default_rng(seed)
  rows = []
  for mode, eff in ((MODE_DOSE, effect), (MODE_SCRATCH, 0.0)):
    for s in SEEDS:
      rows.append(dict(run_id=f'adapt_{mode}_reacher_seed{s}', mode=mode,
                       domain='reacher', seed=s, milestone=500000, auc50k=0,
                       n_ep_50k=48, auc100k=150.0 + eff
                       + rng.normal(0, sd_noise), n_ep_100k=96, auc125k=0,
                       n_ep_125k=112, final10=0, qc_pass=1))
  auc = os.path.join(base, 'auc.csv')
  with open(auc, 'w', newline='') as f:
    wcsv = csvlib.DictWriter(f, fieldnames=list(rows[0]))
    wcsv.writeheader()
    [wcsv.writerow(r) for r in rows]
  wit = {r: dict(counters=UPDATES) for r in FITS}
  wp = os.path.join(base, 'witness.json')
  json.dump(wit, open(wp, 'w'))
  bm = os.path.join(base, 'buffer.json')
  json.dump(dict(occupancy=0.13,
               holdout_disposition='no-probeset-exists',
               first_draw=True),
            open(bm, 'w'))
  return auc, wp, bm


def selfcheck():
  import tempfile
  for effect, expect in ((90.0, 'PROCEED'), (0.0, 'DROP:'),
                         (-90.0, 'DROP-WITH-REVERSAL')):
    base = tempfile.mkdtemp(prefix='mdr_sc_')
    auc, wit, bm = _fixture(base, effect, seed=7)
    run(argparse.Namespace(auc=auc, witness=wit, buffer_manifest=bm,
                           output=os.path.join(base, 'out')))
    r = json.load(open(os.path.join(base, 'out', 'read.json')))
    assert r['overall'].startswith(expect.rstrip(':')), (effect, r['overall'])
  n_ref = 0
  for mutate in ('occ_out', 'no_holdout', 'not_first', 'short'):
    base = tempfile.mkdtemp(prefix='mdr_sc_')
    auc, wit, bm = _fixture(base, 50.0, seed=8)
    b = json.load(open(bm))
    if mutate == 'occ_out':
      b['occupancy'] = 0.28
    elif mutate == 'no_holdout':
      b.pop('holdout_disposition', None)
    elif mutate == 'not_first':
      b['first_draw'] = False
    json.dump(b, open(bm, 'w'))
    if mutate == 'short':
      w = json.load(open(wit))
      w[FITS[0]]['counters'] = 9
      json.dump(w, open(wit, 'w'))
    try:
      run(argparse.Namespace(auc=auc, witness=wit, buffer_manifest=bm,
                             output=os.path.join(base, 'out')))
    except Refusal as e:
      n_ref += 1
      print(f'  refusal OK [{mutate}]: {e}')
    else:
      raise AssertionError(mutate)
  assert n_ref == 4
  print('mindose_reacher_read selfcheck PASS (3 branches + 4 refusals; '
        f'pins: alpha {ALPHA}, occ window {OCC_WINDOW}, updates {UPDATES})')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--auc')
  ap.add_argument('--witness')
  ap.add_argument('--buffer_manifest')
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  a = ap.parse_args()
  if a.selfcheck:
    selfcheck()
  else:
    assert a.auc and a.witness and a.buffer_manifest and a.output
    run(a)


if __name__ == '__main__':
  main()
