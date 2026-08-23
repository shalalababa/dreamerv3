"""Frozen reader — routed-repair control wave (PREREG_rrpctl_20260822).
ONE execution.

Arms (finger q1 side1):
  repair  PINNED per-seed constants (seeds 1-8) from the executed
          routed-repair read (routedrepair_20260822_180310 collate,
          mode ax1rrpq1s1) — cross-cohort, disclosed.
  rca     mode 'ax1rcaq1s1' — unfrozen adapts of the EXISTING carrier
          rgo fits ax1wm_finger_rgoq1s1_seed{17..24} (rgo from random
          init, 500k).
  rcb     mode 'ax1rcbq1s1' — unfrozen adapts of the NEW matched-budget
          refits ax1wm_finger_rcbq1s1_seed{17..24} ([rgo 500k -> rgo
          refit 500k]; phase-2 identical to the repair recipe, only
          phase-1 objective differs).

E1 (PRIMARY): two_sample[repair - rca] (capdescent house form). Fire
positive -> INIT-ADVANTAGE (E2 adjudicated). Fire negative ->
ANTI-SALVAGE-500K. No fire w/ CI inside (-BOUND, +BOUND) ->
OVERWRITE-SUFFICIENT-BOUNDED ("refit-don't-finetune"). Else
NO-CALL-UNDERPOWERED (realized MDE80 reported).
E2 (CO-PRIMARY, fixed-sequence — verdict weight ONLY on E1
INIT-ADVANTAGE, otherwise descriptive): two_sample[repair - rcb].
Fire positive -> SALVAGE-CONFIRMED. Bounded null -> BUDGET-EQUIVALENT.
Fire negative -> ANTI-SALVAGE-AT-BUDGET. Else NO-CALL at E2.
E3 (descriptive, no alpha): paired within-seed [rcb - rca].
BOUND = G_PIN/2 where G_PIN = the executed D1 repair effect
(201.0885625) — a constant containing neither control arm.

Gates (review B1/M1-M4/m2 applied pre-freeze): 16 fit witnesses (8
carrier rgo + 8 rcb) counters == 500000; rgo-arm identity pins on
BOTH fit sets (M1 — the carrier fits ARE the rca arm and phase-1 of
rcb); rcb init-sha EQUALITY pairing (M2 — run.from_checkpoint inside
the SAME-seed carrier fit dir AND its sha EQUAL to that fit's own
resolved done-ckpt sha); rcb phase-2 recipe pins (M3 —
from_checkpoint_regex '^(enc|dyn|dec)/' + frozen_enc False + realized
replay endswith axis1_finger/q1/side1); 16 adapt witnesses counters
== 125000 (four-signal builder evidence); adapt->source linkage;
STRICT modal n_ep PINNED == 96 (M4); _meta.problems must be empty
(m2); ONE-read guard (output must not pre-exist).

Usage: python -m analysis.rrpctl_read --auc <csv> --witness <json> \
    --output <dir>   (--selfcheck)
"""

import argparse
import json
import math
import os

import numpy as np

from analysis.capdescent_read import two_sample
from analysis.domains_read import one_sample
from analysis.rescue_slate_common import (
    Refusal, _ndtri, check_modal_nep, load_rows, refuse, side_auc)

PREREG = 'PREREG_rrpctl_20260822.md'
ALPHA = 0.05
SEEDS_CTL = tuple(range(17, 25))
REPAIR = {1: 360.9792, 2: 429.8125, 3: 257.6875, 4: 266.6667,
          5: 198.3542, 6: 263.6042, 7: 214.6146, 8: 339.8229}
G_PIN = 201.0885625
BOUND = G_PIN / 2.0           # 100.5443, computed
SCRATCH = 147.6823            # context only
FIT_UPDATES = 500000
ADAPT_STEPS = 125000
MODE_A = 'ax1rcaq1s1'
MODE_B = 'ax1rcbq1s1'
RGO_FITS = [f'ax1wm_finger_rgoq1s1_seed{s}' for s in SEEDS_CTL]
RCB_FITS = [f'ax1wm_finger_rcbq1s1_seed{s}' for s in SEEDS_CTL]
ADAPTS_A = [f'adapt_ax1rcaq1s1_finger_seed{s}_ckpt500000'
            for s in SEEDS_CTL]
ADAPTS_B = [f'adapt_ax1rcbq1s1_finger_seed{s}_ckpt500000'
            for s in SEEDS_CTL]
# review M1: BOTH fit sets carry the rgo-arm identity pins — the
# carrier fits ARE the rca arm and phase-1 of rcb.
ARM_PINS = {'task': 'dmc_finger_turn_hard',
            'agent.reward_grad': True, 'agent.repval_grad': False}
# review M3: the two dials that make phase-2 identical to the rrp
# recipe, pinned on the rcb refits.
RCB_PINS = dict(ARM_PINS, **{'run.from_checkpoint_regex': '^(enc|dyn|dec)/',
                             'agent.frozen_enc': False})
REPLAY_SUFFIX = 'axis1_finger/q1/side1'
MODAL_NEP_PIN = 96            # the pinned repair cohort's modal (M4)

TS = two_sample               # selfcheck rebinds with a smaller nboot


def _pins_ok(got, want):
  """Refusal-safe compare (valuefree lesson): bools before numerics."""
  for k, v in want.items():
    g = got.get(k) if isinstance(got, dict) else None
    if isinstance(v, bool):
      if not isinstance(g, bool) or g is not v:
        return False, k
    else:
      if g != v:
        return False, k
  return True, None


def _mde80_two_sample(a, b):
  sp = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
  return float((_ndtri(1 - ALPHA / 2) + _ndtri(0.80)) * sp
               * math.sqrt(1.0 / len(a) + 1.0 / len(b)))


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


def run(args):
  outp = os.path.join(args.output, 'read.json')
  if os.path.exists(outp):
    refuse(f'{outp} exists — ONE read execution is registered')
  rows = load_rows(args.auc)
  w = json.load(open(args.witness))
  probs = w.get('_meta', {}).get('problems')
  if probs:
    refuse(f'witness _meta.problems non-empty ({probs[:2]}...) — an '
           '--allow_problems build never reaches a read')

  for fit in RGO_FITS + RCB_FITS:
    _fit_gate(w, fit)
  for s, fit in zip(SEEDS_CTL, RGO_FITS):
    ok, key = _pins_ok(w[fit].get('cfg_pins'), ARM_PINS)
    if not ok:
      refuse(f'{fit}: carrier-fit cfg pin {key!r} wrong or missing '
             f'(got {w[fit].get("cfg_pins")!r}) — the rca arm/phase-1 '
             'identity witness fails')
    if not w[fit].get('ckpt_sha'):
      refuse(f'{fit}: ckpt_sha missing/None — the init-pairing '
             'equality target is unrecordable')
  for s, fit in zip(SEEDS_CTL, RCB_FITS):
    e = w[fit]
    fc = str(e.get('init_from_checkpoint') or '')
    src = f'ax1wm_finger_rgoq1s1_seed{s}'
    if f'/{src}/' not in fc and not fc.rstrip('/').endswith(src):
      refuse(f'{fit}: init_from_checkpoint {fc!r} not inside its '
             f'same-seed carrier fit {src}')
    if not e.get('init_sha'):
      refuse(f'{fit}: init_sha missing/None — pairing unverifiable')
    if e['init_sha'] != w[src].get('ckpt_sha'):
      refuse(f'{fit}: init_sha != the same-seed carrier fit ckpt_sha '
             '— rcb was NOT initialized from the witnessed carrier '
             'checkpoint (review M2)')
    ok, key = _pins_ok(e.get('cfg_pins'), RCB_PINS)
    if not ok:
      refuse(f'{fit}: cfg pin {key!r} wrong or missing '
             f'(got {e.get("cfg_pins")!r}) — the rgo-arm/phase-2 '
             'witness fails')
    rp = str(e.get('replay_realized') or '')
    if not rp.rstrip('/').endswith(REPLAY_SUFFIX):
      refuse(f'{fit}: realized replay {rp!r} is not the registered '
             f'buffer (...{REPLAY_SUFFIX}) — phase-2 ran on the wrong '
             'substrate (review M3)')
  for s, adapt in zip(SEEDS_CTL, ADAPTS_A):
    _adapt_gate(w, adapt, f'ax1wm_finger_rgoq1s1_seed{s}')
  for s, adapt in zip(SEEDS_CTL, ADAPTS_B):
    _adapt_gate(w, adapt, f'ax1wm_finger_rcbq1s1_seed{s}')

  modal = check_modal_nep(rows, {MODE_A, MODE_B}, set(SEEDS_CTL),
                          'finger', expected=MODAL_NEP_PIN)
  auc_a = side_auc(rows, MODE_A, set(SEEDS_CTL), 'finger')
  auc_b = side_auc(rows, MODE_B, set(SEEDS_CTL), 'finger')
  rca = np.array([auc_a[s] for s in SEEDS_CTL])
  rcb = np.array([auc_b[s] for s in SEEDS_CTL])
  rep = np.array([REPAIR[s] for s in sorted(REPAIR)])

  e1 = TS(rep, rca)
  e1_fire = e1['perm_p'] < ALPHA and e1['ci'][0] > 0
  e1_rev = e1['perm_p'] < ALPHA and e1['ci'][1] < 0
  e1_bounded = (not e1_fire and not e1_rev
                and e1['ci'][0] > -BOUND and e1['ci'][1] < BOUND)
  e2 = TS(rep, rcb)
  e2_fire = e2['perm_p'] < ALPHA and e2['ci'][0] > 0
  e2_rev = e2['perm_p'] < ALPHA and e2['ci'][1] < 0
  e2_bounded = (not e2_fire and not e2_rev
                and e2['ci'][0] > -BOUND and e2['ci'][1] < BOUND)
  e3 = one_sample(rcb - rca)

  if e1_rev:
    overall = ('ANTI-SALVAGE-500K: a fresh 500k rgo fit BEATS the '
               'repair arm — the apt init is a liability even under '
               'routing; reported verbatim, E2 descriptive')
    e2_role = 'descriptive'
  elif e1_fire:
    if e2_fire:
      overall = ('SALVAGE-CONFIRMED: the apt phase beats an '
                 'equal-budget rgo phase at matched two-phase '
                 'structure — the sunk trunk has value beyond generic '
                 'training. Scope: finger, q1s1, 500k+500k, '
                 'reconstruction-based WMs')
    elif e2_rev:
      overall = ('ANTI-SALVAGE-AT-BUDGET: INIT-ADVANTAGE at 500k, but '
                 'an equal-budget rgo->rgo pipeline beats repair — '
                 'the apt phase is worth LESS than its updates')
    elif e2_bounded:
      overall = ('BUDGET-EQUIVALENT: INIT-ADVANTAGE at 500k, and at '
                 'matched budget repair is bounded-equivalent to '
                 'rgo->rgo — a sunk apt trunk is worth its update '
                 'budget under routing, no more ("you do not lose the '
                 'compute", never "you would buy it")')
    else:
      overall = ('INIT-ADVANTAGE, E2 NO-CALL: the apt init adds value '
                 'over a fresh 500k rgo fit; the matched-budget '
                 'question stays open (realized MDE80 reported)')
    e2_role = 'adjudicated'
  elif e1_bounded:
    overall = ('OVERWRITE-SUFFICIENT-BOUNDED: repair is bounded within '
               'half the routed effect of a fresh 500k rgo fit — the '
               'claim demotes to "refit-don\'t-finetune" in every '
               'use; E2 descriptive')
    e2_role = 'descriptive'
  else:
    overall = ('NO-CALL-UNDERPOWERED at E1: no fire and the CI does '
               'not fit inside the +/-BOUND band; this wave decides '
               'large separations only (disclosed); E2 descriptive')
    e2_role = 'descriptive'

  out = dict(
      prereg=PREREG, overall=overall,
      e1=dict(e1, alpha=ALPHA, fire=bool(e1_fire), reversed=bool(e1_rev),
              bounded=bool(e1_bounded), bound=BOUND,
              realized_mde80=_mde80_two_sample(rep, rca)),
      e2=dict(e2, alpha=ALPHA, fire=bool(e2_fire), reversed=bool(e2_rev),
              bounded=bool(e2_bounded), bound=BOUND, role=e2_role,
              realized_mde80=_mde80_two_sample(rep, rcb)),
      e3_descriptive=dict(e3, note='paired within-seed rcb - rca; the '
                          'phase-2 dose value; no alpha'),
      arms=dict(repair_mean_pinned=float(rep.mean()),
                rca_mean=float(rca.mean()), rcb_mean=float(rcb.mean()),
                scratch=SCRATCH),
      disclosures=[
          'repair = pinned constants seeds 1-8 (cross-cohort vs '
          'controls at 17-24; two_sample, not paired; venue/era '
          'differences disclosed with aptctl-vs-July-pin 90.35/89.90 '
          'as the cross-era stability anchor)',
          'rca duplicates the registered valuefree q1uzs1 arm '
          '(different run_ids); all thresholds frozen pre-outcome',
          'crashed cells may resume into their logdir; a from-zero '
          'restart doubles n_ep and refuses at STRICT modal n_ep '
          '(pinned 96)'],
      gates=dict(fits=len(RGO_FITS) + len(RCB_FITS),
                 fit_updates=FIT_UPDATES, adapts=len(ADAPTS_A) +
                 len(ADAPTS_B), adapt_steps=ADAPT_STEPS,
                 modal_n_ep=modal))
  os.makedirs(args.output, exist_ok=True)
  json.dump(out, open(outp, 'w'), indent=1)
  print(f"E1 repair-rca {e1['diff']:+.2f} {e1['ci']} p {e1['perm_p']:.4f}"
        f" | E2 repair-rcb {e2['diff']:+.2f} {e2['ci']} "
        f"p {e2['perm_p']:.4f}")
  print(overall.split(':')[0])
  print(f'-> {outp}')


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _fixture(base, rca_level, rcb_level, sd_noise=10.0, seed=0):
  import csv as csvlib
  rng = np.random.default_rng(seed)
  rows = []
  for mode, level in ((MODE_A, rca_level), (MODE_B, rcb_level)):
    for s in SEEDS_CTL:
      rows.append(dict(
          run_id=f'adapt_{mode}_finger_seed{s}_ckpt500000', mode=mode,
          domain='finger', seed=s, milestone=500000, auc50k=0,
          n_ep_50k=48, auc100k=level + rng.normal(0, sd_noise),
          n_ep_100k=96, auc125k=0, n_ep_125k=112, final10=0, qc_pass=1))
  auc = os.path.join(base, 'auc.csv')
  with open(auc, 'w', newline='') as f:
    wcsv = csvlib.DictWriter(f, fieldnames=list(rows[0]))
    wcsv.writeheader()
    [wcsv.writerow(r) for r in rows]
  wit = {}
  for s in SEEDS_CTL:
    wit[f'ax1wm_finger_rgoq1s1_seed{s}'] = dict(
        counters=FIT_UPDATES, cfg_pins=dict(ARM_PINS),
        ckpt_sha=f'sha{s}')
    wit[f'ax1wm_finger_rcbq1s1_seed{s}'] = dict(
        counters=FIT_UPDATES,
        init_from_checkpoint=(f'/rr/ax1wm_finger_rgoq1s1_seed{s}/ckpt/'
                              '20260101T000000-000000500000'),
        init_sha=f'sha{s}', cfg_pins=dict(RCB_PINS),
        replay_realized='/rr/axis1_finger/q1/side1')
    wit[f'adapt_ax1rcaq1s1_finger_seed{s}_ckpt500000'] = dict(
        counters=ADAPT_STEPS,
        from_checkpoint=(f'/rr/ax1wm_finger_rgoq1s1_seed{s}/ckpt/'
                         '20260101T000000-000000500000'))
    wit[f'adapt_ax1rcbq1s1_finger_seed{s}_ckpt500000'] = dict(
        counters=ADAPT_STEPS,
        from_checkpoint=(f'/rr/ax1wm_finger_rcbq1s1_seed{s}/ckpt/'
                         '20260101T000000-000000500000'))
  wit['_meta'] = dict(problems=[])
  wp = os.path.join(base, 'witness.json')
  json.dump(wit, open(wp, 'w'))
  return auc, wp


def selfcheck():
  import tempfile
  global TS
  orig = TS
  TS = lambda a, b: two_sample(a, b, nboot=2000)
  try:
    cases = (
        # (rca, rcb, sd, expect)
        (291.0, 291.0, 10.0, 'OVERWRITE-SUFFICIENT-BOUNDED'),
        (170.0, 288.0, 10.0, 'BUDGET-EQUIVALENT'),
        (150.0, 168.0, 10.0, 'SALVAGE-CONFIRMED'),
        (150.0, 430.0, 10.0, 'ANTI-SALVAGE-AT-BUDGET'),
        (430.0, 430.0, 10.0, 'ANTI-SALVAGE-500K'),
        (250.0, 250.0, 300.0, 'NO-CALL-UNDERPOWERED'),
        (150.0, 250.0, 95.0, 'INIT-ADVANTAGE, E2 NO-CALL'),
    )
    for rca_l, rcb_l, sd, expect in cases:
      base = tempfile.mkdtemp(prefix='rrpctl_sc_')
      auc, wit = _fixture(base, rca_l, rcb_l, sd_noise=sd, seed=11)
      run(argparse.Namespace(auc=auc, witness=wit,
                             output=os.path.join(base, 'out')))
      r = json.load(open(os.path.join(base, 'out', 'read.json')))
      assert r['overall'].startswith(expect), \
          (rca_l, rcb_l, sd, r['overall'])
      if expect.startswith(('OVERWRITE', 'NO-CALL', 'ANTI-SALVAGE-500K')):
        assert r['e2']['role'] == 'descriptive', r['e2']['role']
    n_ref = 0
    for mutate in ('short_fit', 'missing_fit', 'init_sha_none',
                   'init_path_foreign', 'init_sha_mismatch',
                   'pins_repval_true', 'rgo_pins_missing',
                   'rgo_ckpt_sha_none', 'replay_foreign',
                   'meta_problems', 'modal_pin', 'missing_adapt',
                   'adapt_counters', 'adapt_link', 'one_read'):
      base = tempfile.mkdtemp(prefix='rrpctl_sc_')
      auc, wit = _fixture(base, 150.0, 168.0, seed=12)
      w = json.load(open(wit))
      outdir = os.path.join(base, 'out')
      if mutate == 'short_fit':
        w[RCB_FITS[0]]['counters'] = 100
      elif mutate == 'missing_fit':
        del w[RGO_FITS[3]]
      elif mutate == 'init_sha_none':
        w[RCB_FITS[2]]['init_sha'] = None
      elif mutate == 'init_path_foreign':
        w[RCB_FITS[1]]['init_from_checkpoint'] = \
            '/rr/ax1wm_finger_fq1s1_seed18/ckpt/x'
      elif mutate == 'init_sha_mismatch':
        w[RCB_FITS[3]]['init_sha'] = 'OTHER'
      elif mutate == 'pins_repval_true':
        w[RCB_FITS[4]]['cfg_pins']['agent.repval_grad'] = True
      elif mutate == 'rgo_pins_missing':
        del w[RGO_FITS[2]]['cfg_pins']
      elif mutate == 'rgo_ckpt_sha_none':
        w[RGO_FITS[5]]['ckpt_sha'] = None
      elif mutate == 'replay_foreign':
        w[RCB_FITS[6]]['replay_realized'] = '/rr/axis1_finger/q1/side0'
      elif mutate == 'meta_problems':
        w['_meta']['problems'] = ['fixture problem']
      elif mutate == 'modal_pin':
        import csv as csvlib
        rd = list(csvlib.DictReader(open(auc)))
        for r in rd:
          r['n_ep_100k'] = 95
        with open(auc, 'w', newline='') as f:
          wcsv = csvlib.DictWriter(f, fieldnames=list(rd[0]))
          wcsv.writeheader()
          [wcsv.writerow(r) for r in rd]
      elif mutate == 'missing_adapt':
        del w[ADAPTS_B[2]]
      elif mutate == 'adapt_counters':
        w[ADAPTS_A[0]]['counters'] = 123648
      elif mutate == 'adapt_link':
        w[ADAPTS_B[5]]['from_checkpoint'] = \
            '/rr/ax1wm_finger_rgoq1s1_seed22/ckpt/x'
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
    assert n_ref == 15
  finally:
    TS = orig
  print('rrpctl_read selfcheck PASS (7 branches incl. both E2-gated '
        'and E2-descriptive roles + 15 refusals; pins: '
        f'G_PIN {G_PIN}, BOUND {BOUND:.4f}, repair mean '
        f'{np.mean(list(REPAIR.values())):.4f}, fit {FIT_UPDATES}, '
        f'adapt {ADAPT_STEPS})')


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
