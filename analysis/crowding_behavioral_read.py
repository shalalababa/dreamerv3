"""Frozen reader — crowding/rescue behavioral leg
(PREREG_crowding_behavioral_20260820 + Amendment 1). ONE execution.

Design (amend1 B6): w_r {1,100} x sides {0,1} x seeds 1-4; pairing unit
(side, seed), n = 8 pairs. Fresh UNFROZEN adapts in the registered
namespace ax1uzruw{w}q1s{side} (amend1 B8); the read refuses unless the
preflight attests no target dir pre-existed.

PRIMARY: paired [AUC100k(w100) - AUC100k(w1)] per (side, seed), exact 2^8
sign-flip + BCa. FIRE iff perm p < .05 AND CI lower > 0 (positive =
rescue has behavioral consequence). Significant negative = REVERSAL
branch. NO-CALL-UNDERPOWERED iff no fire AND realized MDE80 > 33.22 AUC
(amend1 B9; the scratch-anchor sd). Otherwise powered null: mechanism
stays probe-level, downstream algorithm candidates gate CLOSED.

SECONDARY (labeled, look-0 descriptive, amend1 B7): the same paired delta
on the FROZEN-protocol rescue adapts (modes ax1ruw{w}q1s{side}, executed
16 Aug, never inspected before this read) — reported once, inside this
read, no alpha. Context: both unfrozen arms vs scratch 147.6823.

Usage: python -m analysis.crowding_behavioral_read --auc <csv> \
    --frozen_auc <csv> --witness <json> --output <dir>   (--selfcheck)
"""

import argparse
import json
import os

import numpy as np

from analysis.domains_read import one_sample
from analysis.rescue_slate_common import (
    Refusal, _ndtri, check_modal_nep, check_witness, load_rows, refuse,
    side_auc)

PREREG = 'PREREG_crowding_behavioral_20260820.md'
AMEND = 'PREREG_crowding_behavioral_amend1_20260820.md'
ALPHA = 0.05
SEEDS = (1, 2, 3, 4)
SIDES = ('0', '1')
SCRATCH = 147.6823
NOCALL_MDE = 33.22            # scratch-anchor sd (amend1 B9)
MODE_FRESH = 'ax1uzruw{w}q1s{side}'
MODE_FROZEN = 'ax1ruw{w}q1s{side}'
ADAPTS = [f'adapt_ax1uzruw{w}q1s{sd}_finger_seed{s}_ckpt500000'
          for w in (1, 100) for sd in SIDES for s in SEEDS]


def paired_delta(rows, fmt):
  cells = {}
  for w in (1, 100):
    for sd in SIDES:
      cells[w, sd] = side_auc(rows, fmt.format(w=w, side=sd), set(SEEDS),
                              'finger')
  return np.array([cells[100, sd][s] - cells[1, sd][s]
                   for sd in SIDES for s in SEEDS])


def run(args):
  rows = load_rows(args.auc)
  w = check_witness(args.witness, ADAPTS, args.adapt_steps)
  meta = json.load(open(args.witness)).get('_meta', {})
  if meta.get('no_preexisting_dirs') is not True:
    refuse('witness _meta.no_preexisting_dirs is not true (amend1 B8: the '
           'namespace must have been empty at submission)')
  modes = {MODE_FRESH.format(w=w_, side=sd) for w_ in (1, 100) for sd in SIDES}
  modal = check_modal_nep(rows, modes, set(SEEDS), 'finger')

  d = paired_delta(rows, MODE_FRESH)
  prim = one_sample(d)
  mde = float((_ndtri(1 - ALPHA / 2) + _ndtri(0.80)) * prim['sd']
              / np.sqrt(len(d)))
  fire = prim['perm_p'] < ALPHA and prim['ci'][0] > 0
  rev = prim['perm_p'] < ALPHA and prim['ci'][1] < 0
  if fire:
    overall = ('RESCUE-BEHAVIORAL-CONFIRMED: the beta-side rescue has a '
               'behavioral consequence — the crowding chain gains its '
               'transfer leg; downstream candidates gate OPEN')
  elif rev:
    overall = ('RESCUE-BEHAVIORAL-REVERSED: significant negative — '
               'registered surprising reversal, reported verbatim; '
               'candidates gate CLOSED')
  elif mde > NOCALL_MDE:
    overall = (f'NO-CALL-UNDERPOWERED: no fire and realized MDE80 '
               f'({mde:.1f}) > {NOCALL_MDE} AUC — not evidence of '
               'absence; candidates gate stays CLOSED pending power')
  else:
    overall = ('RESCUE-BEHAVIORAL-NULL (powered): the mechanism claim '
               'stays probe-level; the three downstream algorithm '
               'candidates gate CLOSED')

  frozen = None
  if args.frozen_auc:
    fd = paired_delta(load_rows(args.frozen_auc), MODE_FROZEN)
    frozen = dict(mean=float(fd.mean()), n=len(fd),
                  note='look-0 FROZEN-protocol descriptive (amend1 B7); '
                       'no alpha, first inspection is this read')
  ctx = {}
  for w_ in (1, 100):
    vals = [side_auc(rows, MODE_FRESH.format(w=w_, side=sd), set(SEEDS),
                     'finger')[s] for sd in SIDES for s in SEEDS]
    ctx[f'w{w_}_mean'] = float(np.mean(vals))
  out = dict(prereg=PREREG, amendment=AMEND, overall=overall,
             primary=dict(prim, alpha=ALPHA, fire=bool(fire)),
             realized_mde80=mde, nocall_bar=NOCALL_MDE,
             frozen_secondary=frozen,
             context=dict(ctx, scratch=SCRATCH),
             gates=dict(adapts=len(ADAPTS), adapt_steps=args.adapt_steps,
                        modal_n_ep=modal))
  os.makedirs(args.output, exist_ok=True)
  json.dump(out, open(os.path.join(args.output, 'read.json'), 'w'), indent=1)
  print(f"paired w100-w1 {prim['mean']:+.2f} {prim['ci']} p "
        f"{prim['perm_p']:.4f} MDE80 {mde:.1f}")
  print(overall.split(':')[0])
  print(f"-> {args.output}/read.json")


def _fixture(base, effect, sd_noise=15.0, seed=0, frozen=True):
  import csv as csvlib
  rng = np.random.default_rng(seed)

  def rows_for(fmt, eff):
    out = []
    for w_ in (1, 100):
      for sd in SIDES:
        for s in SEEDS:
          v = 160.0 + (eff if w_ == 100 else 0.0)
          out.append(dict(
              run_id=f'adapt_{fmt.format(w=w_, side=sd)}_finger_seed{s}',
              mode=fmt.format(w=w_, side=sd), domain='finger', seed=s,
              milestone=500000, auc50k=0, n_ep_50k=48,
              auc100k=v + rng.normal(0, sd_noise), n_ep_100k=96,
              auc125k=0, n_ep_125k=112, final10=0, qc_pass=1))
    return out

  def write(path, rws):
    with open(path, 'w', newline='') as f:
      wcsv = csvlib.DictWriter(f, fieldnames=list(rws[0]))
      wcsv.writeheader()
      [wcsv.writerow(r) for r in rws]

  auc = os.path.join(base, 'auc.csv')
  write(auc, rows_for(MODE_FRESH, effect))
  fauc = None
  if frozen:
    fauc = os.path.join(base, 'frozen.csv')
    write(fauc, rows_for(MODE_FROZEN, effect / 2))
  wit = {r: dict(counters=125000) for r in ADAPTS}
  wit['_meta'] = dict(no_preexisting_dirs=True)
  wp = os.path.join(base, 'witness.json')
  json.dump(wit, open(wp, 'w'))
  return auc, fauc, wp


def selfcheck():
  import tempfile
  for effect, noise, expect in ((40.0, 10.0, 'RESCUE-BEHAVIORAL-CONFIRMED'),
                                (0.0, 8.0, 'RESCUE-BEHAVIORAL-NULL'),
                                (0.0, 60.0, 'NO-CALL'),
                                (-40.0, 10.0, 'RESCUE-BEHAVIORAL-REVERSED')):
    base = tempfile.mkdtemp(prefix='cbh_sc_')
    auc, fauc, wit = _fixture(base, effect, sd_noise=noise, seed=5)
    run(argparse.Namespace(auc=auc, frozen_auc=fauc, witness=wit,
                           adapt_steps=125000,
                           output=os.path.join(base, 'out')))
    r = json.load(open(os.path.join(base, 'out', 'read.json')))
    assert r['overall'].startswith(expect), (effect, noise, r['overall'])
    assert r['frozen_secondary'] is not None
  n_ref = 0
  for mutate in ('preexist', 'missing_adapt', 'wrong_steps'):
    base = tempfile.mkdtemp(prefix='cbh_sc_')
    auc, fauc, wit = _fixture(base, 20.0, seed=6)
    w = json.load(open(wit))
    if mutate == 'preexist':
      w['_meta']['no_preexisting_dirs'] = False
    elif mutate == 'missing_adapt':
      del w[ADAPTS[3]]
    elif mutate == 'wrong_steps':
      w[ADAPTS[0]]['counters'] = 1000
    json.dump(w, open(wit, 'w'))
    try:
      run(argparse.Namespace(auc=auc, frozen_auc=fauc, witness=wit,
                             adapt_steps=125000,
                             output=os.path.join(base, 'out')))
    except Refusal as e:
      n_ref += 1
      print(f'  refusal OK [{mutate}]: {e}')
    else:
      raise AssertionError(mutate)
  assert n_ref == 3
  print('crowding_behavioral_read selfcheck PASS (4 branches + 3 refusals; '
        f'pins: alpha {ALPHA}, nocall bar {NOCALL_MDE}, scratch {SCRATCH})')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--auc')
  ap.add_argument('--frozen_auc', default=None)
  ap.add_argument('--witness')
  ap.add_argument('--adapt_steps', type=int, default=125000,
                  help='expected adapt counters (recorded by the preflight)')
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
