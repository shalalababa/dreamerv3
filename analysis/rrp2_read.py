"""Frozen reader — routed-repair PAIRED certification wave
(PREREG_rrp2_20260823). ONE execution.

Arms (finger q1 side1, seeds 1-8, all three arms of a seed produced by
ONE job on ONE GPU — the invocation-pairing gate is the point):
  repair'  mode 'ax1rp2rq1s1' — fresh re-adapts of the EXISTING rrp
           refits ax1wm_finger_rrpq1s1_seed{1..8}
  rca'     mode 'ax1rp2aq1s1' — adapts of the rgo-from-random-init
           source fits (prefix from _meta.rca_source_prefix: surviving
           ax1wm_finger_rgoq1s1 or fresh ax1wm_finger_rp2gq1s1)
  rcb'     mode 'ax1rp2bq1s1' — adapts of the NEW matched-budget
           refits ax1wm_finger_rp2bq1s1_seed{1..8}

E1' PRIMARY: paired per-seed [repair' - rca'], exact 2^8 sign-flip +
BCa. Fire+ -> INIT-ADVANTAGE (E2' adjudicated). Fire- ->
ANTI-SALVAGE-500K. Bounded null (CI inside +/-BOUND) ->
OVERWRITE-SUFFICIENT-BOUNDED (the refit-don't-finetune certificate).
Else NO-CALL. E2' CO-PRIMARY (verdict weight only on E1' fire+):
paired [repair' - rcb'] -> SALVAGE-CONFIRMED / BUDGET-EQUIVALENT /
ANTI-SALVAGE-AT-BUDGET / NO-CALL. E3' descriptive paired
[rcb' - rca']. S1 descriptive: per-seed [repair' - REPAIR_PINNED] on
the SAME refits — the direct invocation-shift instrument.
BOUND = G_PIN/2 (immovable look-0 constants, rrpctl justification).

Gates: 24 fit witnesses counters == 500000; rgo-arm cfg pins on the
rca'-source AND rp2b fits; rp2b init-sha EQUALITY vs the same-seed
rca'-source ckpt_sha + recipe pins + realized replay; 24 adapt
witnesses counters == 125000 + linkage; INVOCATION-PAIRING: per seed
all three adapts' job_log_file identical; _meta.rca_source_prefix in
the two registered literals; STRICT modal n_ep == 96; _meta.problems
empty; ONE-read guard.

Usage: python -m analysis.rrp2_read --auc <csv> --witness <json> \
    --output <dir>   (--selfcheck)
"""

import argparse
import json
import math
import os

import numpy as np

from analysis.domains_read import one_sample
from analysis.rescue_slate_common import (
    Refusal, _ndtri, check_modal_nep, load_rows, refuse, side_auc)

PREREG = 'PREREG_rrp2_20260823.md'
ALPHA = 0.05
SEEDS = tuple(range(1, 9))
REPAIR = {1: 360.9792, 2: 429.8125, 3: 257.6875, 4: 266.6667,
          5: 198.3542, 6: 263.6042, 7: 214.6146, 8: 339.8229}
G_PIN = 201.0885625
BOUND = G_PIN / 2.0
SCRATCH = 147.6823            # context only
FIT_UPDATES = 500000
ADAPT_STEPS = 125000
MODE_R = 'ax1rp2rq1s1'
MODE_A = 'ax1rp2aq1s1'
MODE_B = 'ax1rp2bq1s1'
RRP_FITS = [f'ax1wm_finger_rrpq1s1_seed{s}' for s in SEEDS]
RP2B_FITS = [f'ax1wm_finger_rp2bq1s1_seed{s}' for s in SEEDS]
RCA_SOURCE_PREFIXES = ('ax1wm_finger_rgoq1s1', 'ax1wm_finger_rp2gq1s1')
ADAPT_OF = {(m, s): f'adapt_{m}_finger_seed{s}_ckpt500000'
            for m in (MODE_R, MODE_A, MODE_B) for s in SEEDS}
ARM_PINS = {'task': 'dmc_finger_turn_hard',
            'agent.reward_grad': True, 'agent.repval_grad': False}
RP2B_PINS = dict(ARM_PINS,
                 **{'run.from_checkpoint_regex': '^(enc|dyn|dec)/',
                    'agent.frozen_enc': False})
REPLAY_SUFFIX = 'axis1_finger/q1/side1'
MODAL_NEP_PIN = 96


def _pins_ok(got, want):
  for k, v in want.items():
    g = got.get(k) if isinstance(got, dict) else None
    if isinstance(v, bool):
      if not isinstance(g, bool) or g is not v:
        return False, k
    else:
      if g != v:
        return False, k
  return True, None


def _mde80_paired(d):
  sd = float(np.std(d, ddof=1))
  return float((_ndtri(1 - ALPHA / 2) + _ndtri(0.80)) * sd
               / math.sqrt(len(d)))


def _fit_gate(w, fit):
  if fit not in w:
    refuse(f'witness missing fit {fit}')
  if int(w[fit]['counters']) != FIT_UPDATES:
    refuse(f'{fit}: fit counters {w[fit]["counters"]} != {FIT_UPDATES}')


def _adapt_gate(w, adapt, source_dir):
  if adapt not in w:
    refuse(f'witness missing adapt {adapt}')
  if int(w[adapt]['counters']) != ADAPT_STEPS:
    refuse(f'{adapt}: adapt counters {w[adapt]["counters"]} != '
           f'{ADAPT_STEPS}')
  fc = str(w[adapt].get('from_checkpoint') or '')
  if f'/{source_dir}/' not in fc and not fc.rstrip('/').endswith(
      source_dir):
    refuse(f'{adapt}: from_checkpoint {fc!r} does not reference its '
           f'source {source_dir} — linkage broken')


def _branchmap(e1, e2):
  e1_fire = e1['perm_p'] < ALPHA and e1['ci'][0] > 0
  e1_rev = e1['perm_p'] < ALPHA and e1['ci'][1] < 0
  e1_bounded = (not e1_fire and not e1_rev and e1['perm_p'] >= ALPHA
                and e1['ci'][0] > -BOUND and e1['ci'][1] < BOUND)
  e2_fire = e2['perm_p'] < ALPHA and e2['ci'][0] > 0
  e2_rev = e2['perm_p'] < ALPHA and e2['ci'][1] < 0
  e2_bounded = (not e2_fire and not e2_rev and e2['perm_p'] >= ALPHA
                and e2['ci'][0] > -BOUND and e2['ci'][1] < BOUND)
  if e1_rev:
    overall, e2_role = (
        'ANTI-SALVAGE-500K: within one invocation, a fresh 500k rgo '
        'fit BEATS the repair arm — the apt init is a liability even '
        'under routing; E2 descriptive'), 'descriptive'
  elif e1_fire:
    e2_role = 'adjudicated'
    if e2_fire:
      overall = ('SALVAGE-CONFIRMED: the apt phase beats an '
                 'equal-budget rgo phase, invocation-paired. Scope: '
                 'finger, q1s1, 500k+500k, reconstruction-based WMs')
    elif e2_rev:
      overall = ('ANTI-SALVAGE-AT-BUDGET: INIT-ADVANTAGE at 500k but '
                 'the equal-budget rgo->rgo pipeline beats repair')
    elif e2_bounded:
      overall = ('BUDGET-EQUIVALENT: INIT-ADVANTAGE at 500k; at '
                 'matched budget repair is bounded-equivalent to '
                 'rgo->rgo')
    else:
      overall = ('INIT-ADVANTAGE, E2 NO-CALL: matched-budget question '
                 'open (realized MDE80 reported)')
  elif e1_bounded:
    overall, e2_role = (
        'OVERWRITE-SUFFICIENT-BOUNDED: invocation-paired, repair is '
        'bounded within half the routed effect of a fresh 500k rgo '
        'fit — the refit-don\'t-finetune certificate ISSUES; E2 '
        'descriptive'), 'descriptive'
  else:
    overall, e2_role = (
        'NO-CALL-UNDERPOWERED at E1: no fire and the CI does not fit '
        'inside the +/-BOUND band even paired; realized MDE80 '
        'reported; E2 descriptive'), 'descriptive'
  return overall, e2_role, dict(
      e1_fire=e1_fire, e1_rev=e1_rev, e1_bounded=e1_bounded,
      e2_fire=e2_fire, e2_rev=e2_rev, e2_bounded=e2_bounded)


def run(args):
  outp = os.path.join(args.output, 'read.json')
  if os.path.exists(outp):
    refuse(f'{outp} exists — ONE read execution is registered')
  rows = load_rows(args.auc)
  w = json.load(open(args.witness))
  meta = w.get('_meta', {})
  probs = meta.get('problems')
  if probs:
    refuse(f'witness _meta.problems non-empty ({probs[:2]}...) — an '
           '--allow_problems build never reaches a read')
  src_prefix = meta.get('rca_source_prefix')
  if src_prefix not in RCA_SOURCE_PREFIXES:
    refuse(f'_meta.rca_source_prefix {src_prefix!r} not one of the '
           f'two registered literals {RCA_SOURCE_PREFIXES}')
  rca_fits = [f'{src_prefix}_seed{s}' for s in SEEDS]

  for fit in RRP_FITS + rca_fits + RP2B_FITS:
    _fit_gate(w, fit)
  for s, fit in zip(SEEDS, rca_fits):
    ok, key = _pins_ok(w[fit].get('cfg_pins'), ARM_PINS)
    if not ok:
      refuse(f'{fit}: rca-source cfg pin {key!r} wrong or missing '
             f'(got {w[fit].get("cfg_pins")!r})')
    # review B1: the arm is DEFINED as rgo-from-RANDOM-INIT — the
    # saved run.from_checkpoint must be empty in BOTH source paths.
    fc0 = (w[fit].get('cfg_pins') or {}).get('run.from_checkpoint')
    if fc0 not in (None, ''):
      refuse(f'{fit}: rca-source run.from_checkpoint {fc0!r} not '
             'empty — the random-init property fails')
    if src_prefix == 'ax1wm_finger_rp2gq1s1':
      # review B1: fresh path-B fits carry their own realized-replay
      # witness (path A inherits provenance from the carrier
      # registration, disclosed).
      rp0 = str(w[fit].get('replay_realized') or '')
      if not rp0.rstrip('/').endswith(REPLAY_SUFFIX):
        refuse(f'{fit}: path-B realized replay {rp0!r} is not the '
               f'registered buffer (...{REPLAY_SUFFIX})')
    if not w[fit].get('ckpt_sha'):
      refuse(f'{fit}: ckpt_sha missing/None — init-pairing target '
             'unrecordable')
  for s, fit in zip(SEEDS, RP2B_FITS):
    e = w[fit]
    src = f'{src_prefix}_seed{s}'
    fc = str(e.get('init_from_checkpoint') or '')
    if f'/{src}/' not in fc and not fc.rstrip('/').endswith(src):
      refuse(f'{fit}: init_from_checkpoint {fc!r} not inside its '
             f'same-seed source {src}')
    if not e.get('init_sha') or e['init_sha'] != w[src].get('ckpt_sha'):
      refuse(f'{fit}: init_sha missing or != the same-seed source '
             'ckpt_sha — rp2b was NOT initialized from the witnessed '
             'source checkpoint')
    ok, key = _pins_ok(e.get('cfg_pins'), RP2B_PINS)
    if not ok:
      refuse(f'{fit}: cfg pin {key!r} wrong or missing '
             f'(got {e.get("cfg_pins")!r})')
    rp = str(e.get('replay_realized') or '')
    if not rp.rstrip('/').endswith(REPLAY_SUFFIX):
      refuse(f'{fit}: realized replay {rp!r} is not the registered '
             f'buffer (...{REPLAY_SUFFIX})')
  src_of = {MODE_R: RRP_FITS, MODE_A: rca_fits, MODE_B: RP2B_FITS}
  for m, fits in src_of.items():
    for s, fit in zip(SEEDS, fits):
      _adapt_gate(w, ADAPT_OF[m, s], fit)
  # INVOCATION-PAIRING GATE — the point of this wave.
  for s in SEEDS:
    jobs = {str(w[ADAPT_OF[m, s]].get('job_log_file') or '')
            for m in (MODE_R, MODE_A, MODE_B)}
    if len(jobs) != 1 or '' in jobs:
      refuse(f'seed {s}: the three arms were NOT produced by one job '
             f'(job logs {sorted(jobs)!r}) — invocation pairing '
             'broken; the paired estimand is not licensed')

  modal = check_modal_nep(rows, {MODE_R, MODE_A, MODE_B}, set(SEEDS),
                          'finger', expected=MODAL_NEP_PIN)
  auc = {m: side_auc(rows, m, set(SEEDS), 'finger')
         for m in (MODE_R, MODE_A, MODE_B)}
  rep = np.array([auc[MODE_R][s] for s in SEEDS])
  rca = np.array([auc[MODE_A][s] for s in SEEDS])
  rcb = np.array([auc[MODE_B][s] for s in SEEDS])
  pin = np.array([REPAIR[s] for s in SEEDS])

  e1 = one_sample(rep - rca)
  e2 = one_sample(rep - rcb)
  e3 = one_sample(rcb - rca)
  shift = rep - pin
  overall, e2_role, flags = _branchmap(e1, e2)

  out = dict(
      prereg=PREREG, overall=overall,
      e1=dict(e1, alpha=ALPHA, bound=BOUND,
              realized_mde80=_mde80_paired(rep - rca), **{
                  k: bool(v) for k, v in flags.items()
                  if k.startswith('e1')}),
      e2=dict(e2, alpha=ALPHA, bound=BOUND, role=e2_role,
              realized_mde80=_mde80_paired(rep - rcb), **{
                  k: bool(v) for k, v in flags.items()
                  if k.startswith('e2')}),
      e3_descriptive=dict(e3, note='paired rcb\' - rca\'; no alpha'),
      s1_invocation_shift=dict(
          per_seed=[float(x) for x in shift],
          mean=float(shift.mean()), sd=float(shift.std(ddof=1)),
          note='repair\' - REPAIR_PINNED on the SAME refits '
               're-adapted; the direct adapt-level invocation-shift '
               'instrument; descriptive, no alpha'),
      arms=dict(repair_prime_mean=float(rep.mean()),
                rca_prime_mean=float(rca.mean()),
                rcb_prime_mean=float(rcb.mean()),
                repair_pinned_mean=float(pin.mean()), scratch=SCRATCH),
      rca_source_prefix=src_prefix,
      gates=dict(fits=24, fit_updates=FIT_UPDATES, adapts=24,
                 adapt_steps=ADAPT_STEPS, modal_n_ep=modal,
                 invocation_paired=True))
  os.makedirs(args.output, exist_ok=True)
  json.dump(out, open(outp, 'w'), indent=1)
  print(f"E1' {e1['mean']:+.2f} {e1['ci']} p {e1['perm_p']:.4f} | "
        f"E2' {e2['mean']:+.2f} p {e2['perm_p']:.4f} | "
        f"S1 shift {shift.mean():+.1f} sd {shift.std(ddof=1):.1f}")
  print(overall.split(':')[0])
  print(f'-> {outp}')


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _fixture(base, d_rca, d_rcb, j_a=3.0, j_b=3.0, seed=0,
             src_prefix='ax1wm_finger_rgoq1s1'):
  """Deterministic PAIRED structure: repair' rows carry base noise;
  rca'/rcb' rows equal the SAME seed's repair' value minus d_arm plus
  an alternating +/-j jitter (mean exactly zero over 8 seeds) — so
  E1'/E2' paired means equal d_arm EXACTLY and their sds equal ~1.07j,
  making every branch fixture land for a structural reason, never a
  draw (the rrpctl noise-draw fragility, fixed here)."""
  import csv as csvlib
  rng = np.random.default_rng(seed)
  rep_vals = {s: 291.0 + rng.normal(0, 25.0) for s in SEEDS}
  rows = []
  for mode in (MODE_R, MODE_A, MODE_B):
    for i, s in enumerate(SEEDS):
      if mode == MODE_R:
        v = rep_vals[s]
      elif mode == MODE_A:
        v = rep_vals[s] - d_rca + (j_a if i % 2 == 0 else -j_a)
      else:
        v = rep_vals[s] - d_rcb + (j_b if i % 2 == 0 else -j_b)
      rows.append(dict(
          run_id=f'adapt_{mode}_finger_seed{s}_ckpt500000', mode=mode,
          domain='finger', seed=s, milestone=500000, auc50k=0,
          n_ep_50k=48, auc100k=v,
          n_ep_100k=96, auc125k=0, n_ep_125k=112, final10=0, qc_pass=1))
  auc = os.path.join(base, 'auc.csv')
  with open(auc, 'w', newline='') as f:
    wcsv = csvlib.DictWriter(f, fieldnames=list(rows[0]))
    wcsv.writeheader()
    [wcsv.writerow(r) for r in rows]
  wit = {'_meta': dict(problems=[], rca_source_prefix=src_prefix)}
  for s in SEEDS:
    rrp, rca_f, rp2b = (f'ax1wm_finger_rrpq1s1_seed{s}',
                        f'{src_prefix}_seed{s}',
                        f'ax1wm_finger_rp2bq1s1_seed{s}')
    wit[rrp] = dict(counters=FIT_UPDATES)
    wit[rca_f] = dict(counters=FIT_UPDATES, cfg_pins=dict(ARM_PINS),
                      ckpt_sha=f'sha{s}',
                      replay_realized='/rr/axis1_finger/q1/side1')
    wit[rp2b] = dict(
        counters=FIT_UPDATES,
        init_from_checkpoint=f'/rr/{rca_f}/ckpt/x-000000500000',
        init_sha=f'sha{s}', cfg_pins=dict(RP2B_PINS),
        replay_realized='/rr/axis1_finger/q1/side1')
    for m, fit in ((MODE_R, rrp), (MODE_A, rca_f), (MODE_B, rp2b)):
      wit[ADAPT_OF[m, s]] = dict(
          counters=ADAPT_STEPS,
          from_checkpoint=f'/rr/{fit}/ckpt/x-000000500000',
          job_log_file=f'rrp2_bundle_{s}.out')
  wp = os.path.join(base, 'witness.json')
  json.dump(wit, open(wp, 'w'))
  return auc, wp


def selfcheck():
  import tempfile
  cases = (
      # (d_rca, d_rcb, j_a, j_b, expect) — paired gaps EXACT by
      # construction (see _fixture)
      (0.0, 0.0, 3.0, 3.0, 'OVERWRITE-SUFFICIENT-BOUNDED'),
      (121.0, 0.0, 3.0, 3.0, 'BUDGET-EQUIVALENT'),
      (141.0, 123.0, 3.0, 3.0, 'SALVAGE-CONFIRMED'),
      (141.0, -139.0, 3.0, 3.0, 'ANTI-SALVAGE-AT-BUDGET'),
      (-139.0, -139.0, 3.0, 3.0, 'ANTI-SALVAGE-500K'),
      (41.0, 41.0, 150.0, 150.0, 'NO-CALL-UNDERPOWERED'),
      (141.0, 41.0, 3.0, 130.0, 'INIT-ADVANTAGE, E2 NO-CALL'),
  )
  for d_a, d_b, j_a, j_b, expect in cases:
    base = tempfile.mkdtemp(prefix='rrp2_sc_')
    auc, wit = _fixture(base, d_a, d_b, j_a=j_a, j_b=j_b, seed=7)
    run(argparse.Namespace(auc=auc, witness=wit,
                           output=os.path.join(base, 'out')))
    r = json.load(open(os.path.join(base, 'out', 'read.json')))
    assert r['overall'].startswith(expect), (expect, r['overall'])
    if expect.startswith(('OVERWRITE', 'NO-CALL', 'ANTI-SALVAGE-500K')):
      assert r['e2']['role'] == 'descriptive'
  # alt source prefix accepted (incl. its replay witness)...
  base = tempfile.mkdtemp(prefix='rrp2_sc_')
  auc, wit = _fixture(base, 0.0, 0.0, seed=8,
                      src_prefix='ax1wm_finger_rp2gq1s1')
  run(argparse.Namespace(auc=auc, witness=wit,
                         output=os.path.join(base, 'out')))
  # ...and refused when a path-B source lacks its realized replay
  base = tempfile.mkdtemp(prefix='rrp2_sc_')
  auc, wit = _fixture(base, 0.0, 0.0, seed=8,
                      src_prefix='ax1wm_finger_rp2gq1s1')
  wb = json.load(open(wit))
  del wb['ax1wm_finger_rp2gq1s1_seed3']['replay_realized']
  json.dump(wb, open(wit, 'w'))
  try:
    run(argparse.Namespace(auc=auc, witness=wit,
                           output=os.path.join(base, 'out2')))
  except Refusal as e:
    print(f'  refusal OK [srcB_replay]: {e}')
  else:
    raise AssertionError('srcB_replay')
  n_ref = 0
  for mutate in ('short_fit', 'missing_fit', 'bad_prefix',
                 'sha_mismatch', 'pins_wrong', 'replay_foreign',
                 'src_init_nonempty', 'adapt_counters', 'adapt_link',
                 'job_split', 'job_missing', 'meta_problems',
                 'modal_pin', 'one_read'):
    base = tempfile.mkdtemp(prefix='rrp2_sc_')
    auc, wit = _fixture(base, 141.0, 123.0, seed=9)
    w = json.load(open(wit))
    outdir = os.path.join(base, 'out')
    if mutate == 'short_fit':
      w['ax1wm_finger_rp2bq1s1_seed1']['counters'] = 100
    elif mutate == 'missing_fit':
      del w['ax1wm_finger_rrpq1s1_seed4']
    elif mutate == 'bad_prefix':
      w['_meta']['rca_source_prefix'] = 'ax1wm_finger_fq1s1'
    elif mutate == 'sha_mismatch':
      w['ax1wm_finger_rp2bq1s1_seed3']['init_sha'] = 'OTHER'
    elif mutate == 'pins_wrong':
      w['ax1wm_finger_rgoq1s1_seed5']['cfg_pins'][
          'agent.repval_grad'] = True
    elif mutate == 'replay_foreign':
      w['ax1wm_finger_rp2bq1s1_seed6']['replay_realized'] = \
          '/rr/axis1_finger/q1/side0'
    elif mutate == 'src_init_nonempty':
      w['ax1wm_finger_rgoq1s1_seed4']['cfg_pins'][
          'run.from_checkpoint'] = '/rr/ax1wm_finger_fq1s1_seed4/ckpt/x'
    elif mutate == 'adapt_counters':
      w[ADAPT_OF[MODE_A, 2]]['counters'] = 123648
    elif mutate == 'adapt_link':
      w[ADAPT_OF[MODE_B, 7]]['from_checkpoint'] = \
          '/rr/ax1wm_finger_rgoq1s1_seed7/ckpt/x'
    elif mutate == 'job_split':
      w[ADAPT_OF[MODE_B, 5]]['job_log_file'] = 'rrp2_bundle_99.out'
    elif mutate == 'job_missing':
      del w[ADAPT_OF[MODE_R, 6]]['job_log_file']
    elif mutate == 'meta_problems':
      w['_meta']['problems'] = ['fixture problem']
    elif mutate == 'modal_pin':
      import csv as csvlib
      rd = list(csvlib.DictReader(open(auc)))
      for r_ in rd:
        r_['n_ep_100k'] = 95
      with open(auc, 'w', newline='') as f:
        wcsv = csvlib.DictWriter(f, fieldnames=list(rd[0]))
        wcsv.writeheader()
        [wcsv.writerow(r_) for r_ in rd]
    elif mutate == 'one_read':
      os.makedirs(outdir, exist_ok=True)
      open(os.path.join(outdir, 'read.json'), 'w').write('{}')
    json.dump(w, open(wit, 'w'))
    try:
      run(argparse.Namespace(auc=auc, witness=wit, output=outdir))
    except Refusal as e:
      n_ref += 1
      print(f'  refusal OK [{mutate}]: {e}')
    else:
      raise AssertionError(mutate)
  assert n_ref == 14
  print('rrp2_read selfcheck PASS (7 branches + alt source prefix + '
        f'14 refusals (incl. random-init + path-B replay provenance); pins: G_PIN {G_PIN}, BOUND {BOUND:.4f}, repair '
        f'pinned mean {np.mean(list(REPAIR.values())):.4f}, fit '
        f'{FIT_UPDATES}, adapt {ADAPT_STEPS}, modal {MODAL_NEP_PIN})')


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
