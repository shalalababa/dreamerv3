"""Frozen reader — value-free pretraining vs online task pretraining, v3
(PREREG_valuefree_online_20260821 v3: finger domain, RGO recipe arm,
all-unfrozen protocol, window ladder, joint-bootstrap equivalence).
ONE execution.

The question: does the VALUE-FREE recipe — reward-free explorer
collection + offline reward-grounded WM fit, no critic and no online
task interaction at pretraining — match a STANDARD online,
critic-driven, task-trained Dreamer trunk, at matched gradient-update
budget (500000 both arms)?

Arms (finger_turn_hard; ALL adapts unfrozen, ALL run fresh in ONE
batch — the initialization is the only registered difference):
  online   mode 'ontask'    8 fresh 5e5-step online task runs
  q1 s1    mode 'q1uzs1'    the RGO fits ax1wm_finger_rgoq1s1 (value-
                            gradient-blocked trunks, carrier seeds
                            17-24; house headline side), FRESH
                            unfrozen adapts — never run before
  q1 s0    mode 'q1uzs0'    secondary arm, descriptive only
  scratch  mode 'scratchvf' fresh random-init adapts (anchor +
                            equivalence yardstick)

WINDOW LADDER (the "late value-head effect" answer):
  PRIMARY  auc100k (the house window; disclosed planning: G ~ +341)
  SECONDARY auc50k, auc125k — if ONLINE-SUPERIOR fires at ANY
  secondary window, a primary MATCHED/NON-INFERIOR license is
  DOWNGRADED to window-scoped with the divergence recorded
  (pre-stated override); RECIPE-SUPERIOR at a secondary is recorded as
  strengthening.

PRIMARY ESTIMANDS at auc100k (d = mean(online) - mean(q1s1)):
  fires: two_sample (label-permutation + BCa, capdescent house form),
  ALPHA .05 with CI conjunction.
  ANCHOR: pooled(online, q1s1) - scratch must fire positive, else
  NO-CALL-ANCHOR (point estimates only are reported — review F14).
  MATCHED: joint seed-bootstrap (B=200000, pinned): fraction of
  resamples with G* > 0 AND |d*| <= 0.5 G* is >= 0.95 (the band's own
  sampling error is inside the event — review F10).
  RECIPE-NON-INFERIOR: fraction with G* > 0 AND d* <= 0.5 G* >= 0.95
  (one-sided; online may still be better by less than the margin).
  Else INDISTINGUISHABLE-UNDERPOWERED (realized MDE80 printed).

DISCLOSED PLANNING QUANTITIES (visible pre-registration, from the
already-read U1 bundle; the FRESH adapt draws are unseen): finger
unfrozen task s1 489.2 (sd 124.9), s0 333.5 (sd 110.7), scratch 147.7
(sd 33.2) at auc100k. Side s1 primary is the study's signal-side
semantics (a-priori), disclosed alongside these values.

Gates: witness — 8 online fits with counters from realized
metrics-updates within [485000, 515000] (online counters are realized
optimizer updates, never a ckpt-suffix) + 16 RGO fits with counters ==
500000 exact + cfg_pins (repval_grad False = the value-free witness);
config identity per group; env_steps in [480000, 520000] per online
fit (two-sided); adapt->checkpoint linkage gated; milestone column:
arms ckpt500000,
scratch ckpt0 (review F18); STRICT modal n_ep at every ladder window;
qc; ONE-read guard.

Usage: python -m analysis.valuefree_online_read --auc <csv> \
    --witness <json> --output <dir>     (--selfcheck)
"""

import argparse
import json
import os

import numpy as np

from analysis.capdescent_read import two_sample
from analysis.rescue_slate_common import (
    Refusal, _ndtri, check_config_identity, check_modal_nep, load_rows,
    refuse, side_auc)

PREREG = 'PREREG_valuefree_online_20260821.md'
ALPHA = 0.05
DOMAIN = 'finger'
SEEDS = tuple(range(17, 25))
MODE_ONLINE = 'ontask'
MODE_Q1S1 = 'q1uzs1'
MODE_Q1S0 = 'q1uzs0'
MODE_SCRATCH = 'scratchvf'
UPDATES_OFFLINE = 500000
ONLINE_UPDATES_BAND = (485000, 515000)
ENV_STEPS_MIN = 480000
FITS_ONLINE = tuple(f'ontask_finger_seed{s}' for s in SEEDS)
# D1: the recipe arm is the RGO fits — value-gradient-blocked trunks
# (repval_grad False), fresh carrier-wave batch, amend2 config-verified
FITS_Q1S1 = tuple(f'ax1wm_finger_rgoq1s1_seed{s}' for s in SEEDS)
FITS_Q1S0 = tuple(f'ax1wm_finger_rgoq1s0_seed{s}' for s in SEEDS)
# D10: config correctness pins (witness cfg_pins per fit)
ONLINE_CFG = {'task': 'dmc_finger_turn_hard',
              'run.train_ratio': 1024, 'run.steps': 500000}
Q1_CFG = {'task': 'dmc_finger_turn_hard',
          'agent.reward_grad': True, 'agent.repval_grad': False}
ENV_STEPS_MAX = 520000
PRIMARY_W = 'auc100k'
LADDER_W = ('auc50k', 'auc125k')
NEP_OF = dict(auc50k='n_ep_50k', auc100k='n_ep_100k',
              auc125k='n_ep_125k')
EQUIV_FRAC = 0.5
JOINT_B = 200000
JOINT_SEED = 20260821
JOINT_THRESH = 0.95


def mde80(sd_pooled, n_per_arm):
  return float((_ndtri(1 - ALPHA / 2) + _ndtri(0.80)) * sd_pooled
               * np.sqrt(2.0 / n_per_arm))


def joint_fracs(a, b, c, frac=EQUIV_FRAC, nboot=None,
                seed=JOINT_SEED):
  """Joint seed-bootstrap: resample each arm independently. D2: the
  yardstick G* = mean(b*) - mean(c*) is the RECIPE arm's value over
  scratch — it does NOT contain the online arm, so a stronger online
  arm can never widen its own tolerance. Returns
  (P[G*>0 and |d*|<=frac G*], P[G*>0 and d*<=frac G*])."""
  if nboot is None:
    nboot = JOINT_B
  rng = np.random.default_rng(seed)
  n = len(a)
  am = a[rng.integers(0, n, (nboot, n))].mean(1)
  bm = b[rng.integers(0, len(b), (nboot, len(b)))].mean(1)
  cm = c[rng.integers(0, len(c), (nboot, len(c)))].mean(1)
  d_ = am - bm
  g_ = bm - cm
  band = frac * g_
  ok = g_ > 0
  n_match = int(np.sum(ok & (np.abs(d_) <= band)))
  n_noninf = int(np.sum(ok & (d_ <= band)))
  return n_match / nboot, n_noninf / nboot


def arm_values(rows, mode, field):
  vals = side_auc(rows, mode, set(SEEDS), DOMAIN, field=field)
  return np.array([vals[s] for s in SEEDS])


def check_milestones(rows):
  for r in rows:
    if r['domain'] != DOMAIN or int(r['seed']) not in SEEDS:
      continue
    if r['mode'] in (MODE_ONLINE, MODE_Q1S1, MODE_Q1S0):
      if int(r['milestone']) != UPDATES_OFFLINE:
        refuse(f"{r['run_id']}: milestone {r['milestone']} != "
               f'{UPDATES_OFFLINE} (arm adapts load the 500000 ckpt)')
    elif r['mode'] == MODE_SCRATCH:
      if int(r['milestone']) != 0:
        refuse(f"{r['run_id']}: scratch milestone "
               f"{r['milestone']} != 0")


def select_branch(p):
  """Pure branch selection over the primary-window facts — unit-tested
  deterministically for every branch (the RECIPE-NON-INFERIOR crescent
  is too narrow to certify end-to-end at fixture scale)."""
  if not p['anchor']:
    return 'NO-CALL-ANCHOR'
  if p['pos']:
    return 'ONLINE-SUPERIOR'
  if p['neg']:
    return 'RECIPE-SUPERIOR'
  if p['p_match'] >= JOINT_THRESH:
    return 'MATCHED'
  if p['p_noninf'] >= JOINT_THRESH:
    return 'RECIPE-NON-INFERIOR'
  return 'INDISTINGUISHABLE-UNDERPOWERED'


BRANCH_TEXT = {
    'NO-CALL-ANCHOR': (
        'NO-CALL-ANCHOR: the recipe arm does not beat scratch at the '
        'primary window — no yardstick; point estimates only'),
    'ONLINE-SUPERIOR': (
        'ONLINE-SUPERIOR: online task pretraining beats the value-free '
        'recipe at matched updates. Registered asymmetric consequence: '
        'attribution ambiguous among onlineness / task-directed data / '
        '2.5x unique frames — only "the recipe does not match at these '
        'budgets" is licensed'),
    'RECIPE-SUPERIOR': (
        'RECIPE-SUPERIOR: the value-free recipe beats online task '
        'pretraining — a-fortiori strong'),
    'MATCHED': (
        'MATCHED: joint P(|d*| <= 0.5 G*) = {p_match:.3f} >= 0.95 — '
        'the value-free recipe matches standard online task '
        'pretraining at matched updates, within half the '
        'value-over-scratch'),
    'RECIPE-NON-INFERIOR': (
        'RECIPE-NON-INFERIOR: joint P(d* <= 0.5 G*) = {p_noninf:.3f} '
        '>= 0.95 — the recipe is not worse than online by more than '
        'half the value-over-scratch (two-sided match not certified)'),
    'INDISTINGUISHABLE-UNDERPOWERED': (
        'INDISTINGUISHABLE-UNDERPOWERED: no fire, no equivalence '
        'certificate; realized MDE80 {mde:.1f}; no claim'),
}


def window_fires(a, b):
  d = two_sample(a, b)
  pos = d['perm_p'] < ALPHA and d['ci'][0] > 0
  neg = d['perm_p'] < ALPHA and d['ci'][1] < 0
  return d, pos, neg


def run(args):
  rows = load_rows(args.auc)
  all_fits = list(FITS_ONLINE + FITS_Q1S1 + FITS_Q1S0)
  with open(args.witness) as f:
    w = json.load(f)
  missing = [r for r in all_fits if r not in w]
  if missing:
    refuse(f'witness missing {len(missing)} fits (first {missing[:3]})')
  for fit in FITS_Q1S1 + FITS_Q1S0:
    if int(w[fit]['counters']) != UPDATES_OFFLINE:
      refuse(f'{fit}: counters {w[fit]["counters"]} != '
             f'{UPDATES_OFFLINE}')
  for fit in FITS_ONLINE:
    c = int(w[fit]['counters'])
    if not ONLINE_UPDATES_BAND[0] <= c <= ONLINE_UPDATES_BAND[1]:
      refuse(f'{fit}: realized updates {c} outside the registered band '
             f'{ONLINE_UPDATES_BAND} (review F1: online counters are '
             'realized optimizer updates, never exact)')
    if w[fit].get('counters_source') not in ('metrics_updates',):
      refuse(f'{fit}: counters_source '
             f'{w[fit].get("counters_source")!r} != metrics_updates — '
             'a ckpt-suffix scan cannot witness an online run')
    es = int(w[fit].get('env_steps') or -1)
    if not ENV_STEPS_MIN <= es <= ENV_STEPS_MAX:
      refuse(f'{fit}: env_steps {es} outside '
             f'[{ENV_STEPS_MIN}, {ENV_STEPS_MAX}] (D6: the budget-parity '
             'gate is two-sided)')
  # D10: config CORRECTNESS, not just within-group identity
  for fit, pins in ([(f, ONLINE_CFG) for f in FITS_ONLINE]
                    + [(f, Q1_CFG) for f in FITS_Q1S1 + FITS_Q1S0]):
    got = w[fit].get('cfg_pins') or {}
    for key, want in pins.items():
      gv = got.get(key)
      try:
        if isinstance(want, bool):
          bad = not isinstance(gv, bool) or gv != want
        elif isinstance(want, (int, float)):
          bad = gv is None or float(gv) != float(want)
        else:
          bad = gv != want
      except (TypeError, ValueError):
        bad = True
      if bad:
        refuse(f'{fit}: cfg_pins[{key}] = {gv!r} != registered '
               f'{want!r}')
  # D5: adapt->checkpoint linkage witnessed by the builder
  if w.get('_meta', {}).get('adapt_ckpt_ok') is not True:
    refuse('witness _meta.adapt_ckpt_ok is not true — the adapt runs '
           'are not linked to their registered checkpoints (D5)')
  check_config_identity(w, dict(online=list(FITS_ONLINE),
                                q1s1=list(FITS_Q1S1),
                                q1s0=list(FITS_Q1S0)))
  check_milestones(rows)
  modes = {MODE_ONLINE, MODE_Q1S1, MODE_Q1S0, MODE_SCRATCH}
  modal = {wname: check_modal_nep(rows, modes, set(SEEDS), DOMAIN,
                                  field=NEP_OF[wname])
           for wname in (PRIMARY_W,) + LADDER_W}

  out = dict(prereg=PREREG, modal_n_ep=modal, windows={})
  per_w = {}
  for wname in (PRIMARY_W,) + LADDER_W:
    a = arm_values(rows, MODE_ONLINE, wname)
    b = arm_values(rows, MODE_Q1S1, wname)
    b0 = arm_values(rows, MODE_Q1S0, wname)
    c = arm_values(rows, MODE_SCRATCH, wname)
    d, pos, neg = window_fires(a, b)
    # D2: the anchor and the yardstick are the RECIPE arm vs scratch
    anchor = two_sample(b, c)
    anchor_fires = anchor['perm_p'] < ALPHA and anchor['ci'][0] > 0
    g = float(b.mean() - c.mean())
    p_match, p_noninf = joint_fracs(a, b, c)
    d0 = two_sample(a, b0)
    sd_pooled = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    per_w[wname] = dict(pos=pos, neg=neg, anchor=anchor_fires,
                        p_match=p_match, p_noninf=p_noninf, g=g,
                        mde=mde80(sd_pooled, len(SEEDS)))
    blk = dict(
        primary_online_minus_q1s1=(
            d if anchor_fires else dict(point_only=d['diff'])),
        anchor_q1s1_minus_scratch=anchor,
        value_over_scratch=g, p_match=p_match, p_noninf=p_noninf,
        secondary_online_minus_q1s0=dict(point_only=d0['diff']),
        realized_mde80=per_w[wname]['mde'],
        means=dict(online=float(a.mean()), q1s1=float(b.mean()),
                   q1s0=float(b0.mean()), scratch=float(c.mean())))
    out['windows'][wname] = blk

  p = per_w[PRIMARY_W]
  branch = select_branch(p)
  if not p['anchor']:
    # D4: under a failed primary anchor, suppress EVERYTHING beyond
    # point estimates — no CIs anywhere, no equivalence certificates,
    # no ladder fire pattern
    for wname, blk in out['windows'].items():
      blk['primary_online_minus_q1s1'] = dict(
          point_only=blk['means']['online'] - blk['means']['q1s1'])
      blk.pop('p_match', None)
      blk.pop('p_noninf', None)
      blk.pop('realized_mde80', None)
      blk['anchor_q1s1_minus_scratch'] = dict(
          point_only=blk['anchor_q1s1_minus_scratch'].get(
              'diff', blk['anchor_q1s1_minus_scratch'].get(
                  'point_only')))
  verdict = BRANCH_TEXT[branch].format(**p)

  # window ladder override (pre-stated): a late/early ONLINE-SUPERIOR
  # downgrades any match-grade license to window-scoped. D11: the
  # strengthening note attaches ONLY to a match-grade/recipe primary
  # (never decorates an online win), and any cross-window sign
  # disagreement is recorded symmetrically as DIVERGENT.
  overrides = ([wn for wn in LADDER_W if per_w[wn]['pos']]
               if p['anchor'] else [])
  strengthens = ([wn for wn in LADDER_W if per_w[wn]['neg']]
                 if p['anchor'] else [])
  match_grade = branch in ('MATCHED', 'RECIPE-NON-INFERIOR')
  if overrides and match_grade:
    verdict += (f' — OVERRIDE: ONLINE-SUPERIOR fires at {overrides}; '
                'the license is DOWNGRADED to the primary window only, '
                'with the divergence recorded (registered ladder '
                'clause)')
  if strengthens and (match_grade or p['neg']):
    verdict += (f' [recipe-superior also fires at {strengthens} — '
                'recorded as strengthening]')
  divergent = bool((overrides and (p['neg'] or match_grade))
                   or (strengthens and p['pos']))
  if divergent:
    verdict += ' [DIVERGENT: window fire pattern disagrees in sign]'
  out['overall'] = verdict
  if p['anchor']:
    out['ladder'] = dict(online_superior_at=overrides,
                         recipe_superior_at=strengthens,
                         divergent=divergent)

  os.makedirs(args.output, exist_ok=True)
  path = os.path.join(args.output, 'read.json')
  if os.path.exists(path):
    refuse(f'{path} exists — ONE read execution is registered')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  pw = out['windows'][PRIMARY_W]
  certs = ('' if 'p_match' not in pw else
           f" | p_match {pw['p_match']:.3f} p_noninf "
           f"{pw['p_noninf']:.3f}")
  mde_s = ('' if 'realized_mde80' not in pw
           else f" | MDE80 {pw['realized_mde80']:.1f}")
  print(f"[{PRIMARY_W}] means o/q1s1/scr "
        f"{pw['means']['online']:.1f}/{pw['means']['q1s1']:.1f}/"
        f"{pw['means']['scratch']:.1f} | G "
        f"{pw['value_over_scratch']:+.1f}{certs}{mde_s}")
  print(out['overall'].split(':')[0])
  print('->', path)
  return out


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _rows(mus, sds, rng, per_window_shift=None):
  """mus/sds keyed by mode; per_window_shift optionally maps
  (mode, window) -> extra shift for ladder fixtures."""
  shift = per_window_shift or {}
  rows = []
  for mode in (MODE_ONLINE, MODE_Q1S1, MODE_Q1S0, MODE_SCRATCH):
    for s in SEEDS:
      base = {w: mus[mode] + shift.get((mode, w), 0.0)
              + rng.normal(0, sds[mode])
              for w in ('auc50k', 'auc100k', 'auc125k')}
      ms = 0 if mode == MODE_SCRATCH else UPDATES_OFFLINE
      rows.append(dict(
          run_id=f'adapt_{mode}_finger_seed{s}_ckpt{ms}',
          mode=mode, domain=DOMAIN, seed=str(s), milestone=str(ms),
          auc50k=str(base['auc50k']), n_ep_50k='48',
          auc100k=str(base['auc100k']), n_ep_100k='96',
          auc125k=str(base['auc125k']), n_ep_125k='112',
          final10='0', qc_pass='1'))
  return rows


def _witness(tmp, online_counters=494368, break_source=False,
             env_steps=496496, adapt_ckpt_ok=True, cfg_break=None):
  w = {}
  for f in FITS_ONLINE:
    w[f] = dict(counters=online_counters,
                counters_source=('ckpt_scan' if break_source
                                 else 'metrics_updates'),
                config_sha_excl_seed='o', env_steps=env_steps,
                replay_ratio_realized=1040.0,
                cfg_pins=dict(ONLINE_CFG))
  for f in FITS_Q1S1:
    w[f] = dict(counters=UPDATES_OFFLINE, config_sha_excl_seed='p',
                cfg_pins=dict(Q1_CFG))
  for f in FITS_Q1S0:
    w[f] = dict(counters=UPDATES_OFFLINE, config_sha_excl_seed='q',
                cfg_pins=dict(Q1_CFG))
  if cfg_break:
    fit, key, val = cfg_break
    w[fit]['cfg_pins'][key] = val
  w['_meta'] = dict(adapt_ckpt_ok=adapt_ckpt_ok)
  path = os.path.join(tmp, 'w.json')
  with open(path, 'w') as fh:
    json.dump(w, fh)
  return path


def _write_case(tmp, rows):
  import csv
  auc = os.path.join(tmp, 'auc.csv')
  with open(auc, 'w', newline='') as fh:
    wtr = csv.DictWriter(fh, fieldnames=rows[0].keys())
    wtr.writeheader()
    wtr.writerows(rows)
  return auc


def selfcheck():
  import functools
  import tempfile
  import zlib
  # fast statistics for fixture volume (the registered read keeps the
  # full B; the selfcheck certifies BRANCH LOGIC, not inference grade)
  global two_sample, JOINT_B
  orig_ts, orig_jb = two_sample, JOINT_B
  two_sample = functools.partial(orig_ts, nboot=800)
  JOINT_B = 20000
  base_sds = {MODE_ONLINE: 60., MODE_Q1S1: 60., MODE_Q1S0: 60.,
              MODE_SCRATCH: 30.}

  def case(mus, want, shift=None, witness_kw=None, sds=None,
           n_streams=12, need=9):
    # D9: MAJORITY VOTE across independently-salted pinned streams —
    # a single stream carries irreducible ~5% type-I at n=8, so branch
    # logic is certified by >= need/n_streams agreement, and no shared
    # rng ordering can silently rewrite a fixture
    eq = {(MODE_Q1S1, wn): mus[MODE_ONLINE] - mus[MODE_Q1S1]
          for wn in LADDER_W}
    eq.update(shift or {})
    hits, first = 0, None
    for k in range(n_streams):
      rng = np.random.default_rng(zlib.crc32(want.encode()) + k)
      with tempfile.TemporaryDirectory() as tmp:
        auc = _write_case(tmp, _rows(mus, sds or base_sds, rng, eq))
        out = run(argparse.Namespace(
            auc=auc, witness=_witness(tmp, **(witness_kw or {})),
            output=os.path.join(tmp, 'o')))
        if out['overall'].startswith(want):
          hits += 1
          if first is None:
            first = out
    assert hits >= need, (want, f'{hits}/{n_streams}')
    return first

  case(dict(ontask=490., q1uzs1=490., q1uzs0=330., scratchvf=150.),
       'MATCHED')
  # RECIPE-NON-INFERIOR is an intrinsically NARROW crescent between
  # RECIPE-SUPERIOR and MATCHED (stream majority tops out ~4-6/12 by
  # geometry) — the branch SELECTION is certified deterministically on
  # select_branch for all six branches instead:
  base = dict(anchor=True, pos=False, neg=False, p_match=0.0,
              p_noninf=0.0, mde=100.0, g=300.0)
  assert select_branch(dict(base, anchor=False)) == 'NO-CALL-ANCHOR'
  assert select_branch(dict(base, pos=True)) == 'ONLINE-SUPERIOR'
  assert select_branch(dict(base, neg=True)) == 'RECIPE-SUPERIOR'
  assert select_branch(dict(base, p_match=0.96,
                            p_noninf=0.99)) == 'MATCHED'
  assert select_branch(dict(base, p_match=0.80,
                            p_noninf=0.96)) == 'RECIPE-NON-INFERIOR'
  assert select_branch(dict(base, p_match=0.80, p_noninf=0.80)) == \
      'INDISTINGUISHABLE-UNDERPOWERED'
  # precedence: a fire beats any certificate
  assert select_branch(dict(base, pos=True,
                            p_match=0.99)) == 'ONLINE-SUPERIOR'
  # ...and joint_fracs lands in the crescent on a constructed draw
  # (deterministic arrays: d = -180 vs band ~ 0.5*300)
  a0 = np.array([300., 310., 290., 305., 295., 300., 302., 298.]) - 90
  b0 = np.array([390., 400., 380., 395., 385., 390., 392., 388.])
  c0 = np.full(8, 90.)
  pm0, pn0 = joint_fracs(a0, b0, c0, nboot=20000)
  assert pn0 >= 0.95 and pm0 < 0.95, (pm0, pn0)
  case(dict(ontask=800., q1uzs1=490., q1uzs0=330., scratchvf=150.),
       'ONLINE-SUPERIOR')
  case(dict(ontask=300., q1uzs1=600., q1uzs0=330., scratchvf=150.),
       'RECIPE-SUPERIOR')
  # underpowered: sizeable positive d that neither fires nor certifies
  case(dict(ontask=550., q1uzs1=470., q1uzs0=330., scratchvf=150.),
       'INDISTINGUISHABLE-UNDERPOWERED',
       sds={MODE_ONLINE: 260., MODE_Q1S1: 100., MODE_Q1S0: 60.,
            MODE_SCRATCH: 30.})
  out = case(dict(ontask=490., q1uzs1=490., q1uzs0=330.,
                  scratchvf=560.), 'NO-CALL-ANCHOR')
  # review-3 MAJOR-1: recipe-superior secondaries under a FAILED anchor
  # must leak nothing into the verdict string
  outn = case(dict(ontask=200., q1uzs1=490., q1uzs0=330.,
                   scratchvf=560.), 'NO-CALL-ANCHOR')
  assert 'strengthening' not in outn['overall'], outn['overall']
  assert 'DIVERGENT' not in outn['overall'], outn['overall']
  assert 'ladder' not in outn
  for wn, blk in out['windows'].items():
    assert 'point_only' in blk['primary_online_minus_q1s1'], (wn, blk)
    assert 'p_match' not in blk and 'p_noninf' not in blk, (wn, blk)
  assert 'ladder' not in out, 'D4: ladder must be suppressed'
  out = case(dict(ontask=490., q1uzs1=490., q1uzs0=330.,
                  scratchvf=150.), 'MATCHED',
             shift={(MODE_ONLINE, 'auc125k'): 400.})
  assert 'OVERRIDE' in out['overall'], out['overall']
  assert out['ladder']['online_superior_at'] == ['auc125k']

  def expect_refusal(marker, witness_kw=None, mutate=None):
    rng = np.random.default_rng(zlib.crc32(marker.encode()))
    with tempfile.TemporaryDirectory() as tmp:
      rows = _rows(dict(ontask=490., q1uzs1=490., q1uzs0=330.,
                        scratchvf=150.), base_sds, rng)
      if mutate:
        mutate(rows)
      auc = _write_case(tmp, rows)
      try:
        run(argparse.Namespace(
            auc=auc, witness=_witness(tmp, **(witness_kw or {})),
            output=os.path.join(tmp, 'o')))
        raise AssertionError(f'refusal not tripped: {marker}')
      except (SystemExit, Refusal) as e:
        assert marker in str(e), (marker, str(e))

  expect_refusal('outside the registered band',
                 witness_kw=dict(online_counters=470000))
  expect_refusal('counters_source', witness_kw=dict(break_source=True))
  expect_refusal('env_steps', witness_kw=dict(env_steps=560000))
  expect_refusal('cfg_pins', witness_kw=dict(
      cfg_break=(FITS_Q1S1[0], 'agent.repval_grad', True)))
  expect_refusal('adapt_ckpt_ok', witness_kw=dict(adapt_ckpt_ok=False))

  def bad_milestone(rows):
    rows[0]['milestone'] = '250000'
  expect_refusal('milestone', mutate=bad_milestone)

  def bad_scratch_ms(rows):
    for r in rows:
      if r['mode'] == MODE_SCRATCH:
        r['milestone'] = '500000'
        break
  expect_refusal('scratch milestone', mutate=bad_scratch_ms)

  def bad_nep(rows):
    rows[3]['n_ep_100k'] = '80'
  expect_refusal('modal n_ep', mutate=bad_nep)

  with tempfile.TemporaryDirectory() as tmp:
    auc = _write_case(tmp, _rows(dict(ontask=490., q1uzs1=490.,
                                      q1uzs0=330., scratchvf=150.),
                                 base_sds, np.random.default_rng(101)))
    wpath = _witness(tmp)
    w = json.load(open(wpath))
    w[FITS_ONLINE[0]]['config_sha_excl_seed'] = 'DIFFERENT'
    json.dump(w, open(wpath, 'w'))
    try:
      run(argparse.Namespace(auc=auc, witness=wpath,
                             output=os.path.join(tmp, 'o')))
      raise AssertionError('config identity refusal failed')
    except (SystemExit, Refusal) as e:
      assert 'config identity' in str(e), e
  with tempfile.TemporaryDirectory() as tmp:
    auc = _write_case(tmp, _rows(dict(ontask=490., q1uzs1=490.,
                                      q1uzs0=330., scratchvf=150.),
                                 base_sds, np.random.default_rng(102)))
    outdir = os.path.join(tmp, 'o')
    os.makedirs(outdir)
    open(os.path.join(outdir, 'read.json'), 'w').close()
    try:
      run(argparse.Namespace(auc=auc, witness=_witness(tmp),
                             output=outdir))
      raise AssertionError('ONE-read guard failed to trip')
    except (SystemExit, Refusal) as e:
      assert 'ONE read execution' in str(e), e

  two_sample, JOINT_B = orig_ts, orig_jb
  print('SELFCHECK PASS (valuefree_online_read v3: MATCHED / '
        'NON-INFERIOR / ONLINE-SUPERIOR / RECIPE-SUPERIOR / '
        'UNDERPOWERED / NO-CALL-ANCHOR(point-only) branches; ladder '
        'override at auc125k; refusals: online counters band + '
        'counters_source + arm/scratch milestones + modal n_ep + '
        'config identity + ONE-read guard)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--auc')
  p.add_argument('--witness')
  p.add_argument('--output')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not (args.auc and args.witness and args.output):
    p.error('--auc, --witness, --output required (or --selfcheck)')
  run(args)


if __name__ == '__main__':
  main()
