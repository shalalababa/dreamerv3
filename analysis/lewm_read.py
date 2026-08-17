"""Frozen reader — LeWM / JEPA-family arm (P-J1).

Registration: PREREG_lewm_20260812.md (rules verbatim; P-J1 frozen in
PREREG_theory_R1_gating_20260811 + amend2). ONE execution:

  python -m analysis.lewm_read --e4_jp <bundle>/e4_fingerpx_v1_jp.csv \
      --e4_pe <bundle>/e4_fingerpx_v1_pe.csv --auc_csv <bundle>/auc.csv \
      --fidelity_glob '<bundle>/lewm_distill_finger_s*_seed*/fidelity.json' \
      --enc_checks <bundle>/jp_enc_checks.json \
      --fit_counters <bundle>/jp_fit_counters.json \
      --ckpt_steps <bundle>/jp_ckpt_steps.json \
      --lewm_probe_glob '<bundle>/lewm_probe/lewm_finger_s*_seed*.json' \
      --fpx_probe_glob '<bundle>/lewm_probe/fpx_s*_seed*.json' \
      --embed_control <bundle>/embed_control.json --output <dir>
Selfcheck: python -m analysis.lewm_read --selfcheck
"""

import argparse
import csv
import glob as globlib
import hashlib
import json
import os
import re

import numpy as np

from analysis.domains_read import one_sample
from analysis.pixel_swamping_read import seed_cluster_ci
from analysis.unfrozen_stress_read import load_cells

SEEDS = tuple(range(1, 9))
SIDES = ('0', '1')
# pe per-cell e4 csv, read 2026-08-02 (value-aware, disclosed) — pinned:
PE_E4_SHA256 = ('8bd7a68389a1f05cc78fc4b6f3b4be41fb1cdbfa348f305c8e8e1a86'
                'd7f8c112')  # Amendment 2 (PREREG_lewm_amend2_20260816):
# same-device (rtx6000) pe anchor panel replacing the historical csv
# (9907d733..., device unrecoverable; full-panel diff mean -0.172 sd 0.32)
FIDELITY_FLOOR = 0.5
THRESH_SWAMP = 2.0        # PREREG_pixel_swamping_20260724
SW_BASE_LO = 19.41        # committed swamping band lower edge
PE_ANCHOR = dict(point=23.032351416154537,
                 ci=[20.142859778988182, 25.676290912420605])
PE_BEH_ANCHOR = dict(point=-8.541818750000004,
                     ci=[-25.96698234375, 9.620574843749994])
JP_FIT_RE = re.compile(r'^ax1wm_finger_jppxq1ms([01])_seed([1-8])$')
PE_FIT_RE = re.compile(r'^ax1wm_finger_pepxq1ms([01])_seed([1-8])$')
JP_MODES = dict(s0='ax1jppxq1ms0', s1='ax1jppxq1ms1')
X2_BASE_MODES = dict(s0='ax1pxpxq1ms0', s1='ax1pxpxq1ms1')
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X2_BASELINE = os.path.join(_REPO, 'artifacts/pixel_x2_20260724/auc.csv')
ALPHA_KEY = 'alpha_0.001'
LEWM_ID_RE = re.compile(r'^lewm_finger_s([01])_seed([1-8])$')
FPX_ID_RE = re.compile(r'^ax1wm_finger_fpxpxq1ms([01])_seed([1-8])$')


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for blk in iter(lambda: f.read(1 << 20), b''):
      h.update(blk)
  return h.hexdigest()


def load_e4(path, fit_re):
  """h=0 rows -> {seed: {side: rew_nll_in}} with integrity guards."""
  out = {}
  seen = set()
  with open(path) as f:
    for r in csv.DictReader(f):
      m = fit_re.match(r['run_id'])
      if not m or int(r['horizon']) != 0:
        continue
      side, seed = m.group(1), int(m.group(2))
      key = (side, seed)
      assert key not in seen, ('duplicate e4 row', key)
      seen.add(key)
      assert r.get('reward_aware') in ('1', 'True', 'true'), \
          (r['run_id'], 'not task-mode')
      v = float(r['rew_nll_in'])
      assert np.isfinite(v), (r['run_id'], 'non-finite rew_nll_in')
      out.setdefault(seed, {})[side] = v
  for seed in SEEDS:
    assert seed in out and set(out[seed]) == set(SIDES), \
        ('incomplete e4 grid', seed, sorted(out.get(seed, {})))
  return out


def load_fidelity(pattern):
  out = {}
  for p in sorted(globlib.glob(pattern)):
    with open(p) as f:
      d = json.load(f)
    m = re.search(r'lewm_distill_finger_s([01])_seed([1-8])', p)
    assert m, p
    out[(m.group(1), int(m.group(2)))] = float(d['holdout_r2'])
  for side in SIDES:
    for seed in SEEDS:
      assert (side, seed) in out, ('missing fidelity', side, seed)
  assert len(out) == 16, len(out)
  return out


def load_probes(pattern, id_re, require_witness, expect_preproc=None):
  out, alphas, preprocs = {}, {}, {}
  for p in sorted(globlib.glob(pattern)):
    with open(p) as f:
      d = json.load(f)
    m = id_re.match(d['run_id'])
    assert m, (p, d['run_id'])
    assert d['probeset_id'].startswith('fingerpx_v1'), \
        (p, d['probeset_id'])
    if require_witness:
      assert d.get('witness_match') is not False, (p, 'witness mismatch')
      assert d.get('reward_override') is None, (p, 'override-scored')
    v = d['probe'][ALPHA_KEY]['auroc']
    assert v is not None, (p, 'degenerate auroc — leg REFUSES')
    key = (m.group(1), int(m.group(2)))
    assert key not in out, ('duplicate probe json', key)
    out[key] = float(v)
    alphas[key] = {a: d['probe'][a].get('auroc')
                   for a in d['probe']}
    if expect_preproc is not None:
      # review C3: every LeWM embedding must have been produced under
      # the verified preprocessing pin
      got = d.get('embed_manifest', {}).get('preproc')
      assert got == expect_preproc, (p, 'preproc mismatch', got,
                                     expect_preproc)
  for side in SIDES:
    for seed in SEEDS:
      assert (side, seed) in out, ('missing probe json', side, seed)
  assert len(out) == 16, len(out)
  return out, alphas


def check_witness(enc_checks, fit_counters, ckpt_steps):
  for side in SIDES:
    for seed in SEEDS:
      wm = f'ax1wm_finger_jppxq1ms{side}_seed{seed}'
      assert enc_checks.get(wm) is True, (wm, 'enc integrity not passed')
      assert wm in fit_counters and \
          int(fit_counters[wm]['update']) == \
          int(fit_counters[wm]['total']), (wm, fit_counters.get(wm))
      assert wm in ckpt_steps and int(ckpt_steps[wm]) == 500000, \
          (wm, ckpt_steps.get(wm))


def analyse(jp_nll, pe_nll, fidelity, lewm_probe, fpx_probe,
            auc_csv, alpha_panels=None):
  fid_ok = all(v >= FIDELITY_FLOOR for v in fidelity.values())

  # P-J1-rep: per-seed paired [pe - jp] on side-mean rew-NLL (n=8)
  rep_deltas = np.array([
      np.mean([pe_nll[s][side] for side in SIDES])
      - np.mean([jp_nll[s][side] for side in SIDES]) for s in SEEDS])
  rep = one_sample(rep_deltas)
  if rep['ci'][0] > 0 and rep['perm_p'] < 0.05:
    rep_v = 'JEPA-MORE-LEGIBLE-THAN-RECON'
  elif rep['ci'][1] < 0 and rep['perm_p'] < 0.05:
    rep_v = 'RECON-MORE-LEGIBLE'
  else:
    rep_v = 'NO-SEPARATION'
  jp_by_seed = {s: [jp_nll[s][side] for side in SIDES] for s in SEEDS}
  jp_point, jp_lo, jp_hi = seed_cluster_ci(jp_by_seed)
  bands = dict(inclusion_restored=bool(jp_hi < THRESH_SWAMP),
               below_swamping_band=bool(jp_hi < SW_BASE_LO))

  # P-J1-emb: paired [lewm - fpx] AUROC per (side, seed), n=16
  emb_deltas = np.array([lewm_probe[(side, s)] - fpx_probe[(side, s)]
                         for side in SIDES for s in SEEDS])
  emb = one_sample(emb_deltas)
  if emb['ci'][0] > 0 and emb['perm_p'] < 0.05:
    emb_v = 'EMB-MORE-LEGIBLE'
  elif emb['ci'][1] < 0 and emb['perm_p'] < 0.05:
    emb_v = 'EMB-LESS-LEGIBLE'
  else:
    emb_v = 'EMB-NO-SEPARATION'

  # P-J1-beh: pe P-PE2 estimator with jp cells
  cells = load_cells(auc_csv, JP_MODES)
  base = load_cells(X2_BASELINE, X2_BASE_MODES)
  beh_deltas = np.mean([cells[k] - base[k] for k in ('s0', 's1')], axis=0)
  beh = one_sample(np.asarray(beh_deltas))
  if beh['ci'][0] > 0 and beh['perm_p'] < 0.05:
    beh_v = 'LIFTS'
  elif beh['ci'][1] < 0 and beh['perm_p'] < 0.05:
    beh_v = 'BELOW-X2'
  else:
    beh_v = 'FLOOR-CONSISTENT'

  if not fid_ok:
    rep_v = f'INSTRUMENT-LIMITED (fidelity floor; computed: {rep_v})'
    beh_v = f'INSTRUMENT-LIMITED (fidelity floor; computed: {beh_v})'
    overall = ('INSTRUMENT-LIMITED: distillation fidelity below the '
               'registered floor — graft legs uninterpretable; emb leg '
               f'stands alone: {emb_v}')
  else:
    rep_fired = rep_v == 'JEPA-MORE-LEGIBLE-THAN-RECON'
    emb_fired = emb_v == 'EMB-MORE-LEGIBLE'
    if rep_fired and emb_fired:
      overall = 'P-J1 CONFIRMED (strict: both representation legs fire)'
    elif rep_fired or emb_fired:
      overall = ('P-J1 PARTIAL (exactly one representation leg fires: '
                 + ('graft' if rep_fired else 'emb') + ')')
    else:
      overall = ('P-J1 NOT CONFIRMED — the reward-free null is '
                 'objective-general across recon and latent-prediction '
                 'pretraining at this substrate (registered '
                 'strengthening of the negative result)')

  out = dict(
      fidelity=dict(values={f's{k[0]}_seed{k[1]}': v
                            for k, v in fidelity.items()},
                    floor=FIDELITY_FLOOR, all_pass=bool(fid_ok)),
      p_j1_rep=dict(stats=rep, verdict=rep_v, pe_anchor=PE_ANCHOR,
                    jp_seed_cluster=dict(point=jp_point,
                                         ci=[jp_lo, jp_hi]),
                    bands=bands),
      p_j1_emb=dict(stats=emb, verdict=emb_v,
                    lewm_mean=float(np.mean(list(lewm_probe.values()))),
                    fpx_mean=float(np.mean(list(fpx_probe.values()))),
                    # review C12: capacity-confound sensitivity — the
                    # full alpha curve, reported never decisional
                    alpha_sensitivity=alpha_panels),
      p_j1_beh=dict(stats=beh, verdict=beh_v,
                    pe_beh_anchor=PE_BEH_ANCHOR,
                    cell_means={k: float(np.mean(v))
                                for k, v in cells.items()}),
      overall=overall)
  for leg, st in (('p_j1_rep', rep), ('p_j1_emb', emb)):
    if 'NO-SEPARATION' in out[leg]['verdict']:
      # review C10: n=8 paired MDE80 d_z = 1.1560 (exact noncentral-t;
      # t-based note — the decision rule itself is permutation+BCa)
      out[leg]['mde_note'] = dict(
          mde80_units=float((1.1560 if leg == 'p_j1_rep' else 0.74942)
                            * st['sd']),
          basis='noncentral-t approximation; decision rule is '
                'permutation+BCa')
  return out


def run(args):
  got = sha256(args.e4_pe)
  assert got == PE_E4_SHA256, ('pe e4 csv sha mismatch — REFUSE', got)
  # review C3/C8: the embedding-pipeline positive control must have
  # PASSED at the smoke, else the whole read refuses (an instrument
  # false-negative must never become a registered claim)
  with open(args.embed_control) as f:
    ctrl = json.load(f)
  assert ctrl.get('pass') is True, ('embed_control did not pass — '
                                    'REFUSE', ctrl)
  preproc = ctrl['preproc']
  with open(args.enc_checks) as f:
    enc_checks = json.load(f)
  with open(args.fit_counters) as f:
    fc = json.load(f)
  with open(args.ckpt_steps) as f:
    cs = json.load(f)
  check_witness(enc_checks, fc, cs)
  jp_nll = load_e4(args.e4_jp, JP_FIT_RE)
  pe_nll = load_e4(args.e4_pe, PE_FIT_RE)
  fidelity = load_fidelity(args.fidelity_glob)
  lewm_probe, lewm_alphas = load_probes(
      args.lewm_probe_glob, LEWM_ID_RE, False, expect_preproc=preproc)
  fpx_probe, fpx_alphas = load_probes(args.fpx_probe_glob, FPX_ID_RE,
                                      True)
  def _alpha_means(panel):
    keys = sorted({a for d in panel.values() for a in d})
    return {a: (float(np.mean([d[a] for d in panel.values()
                               if d.get(a) is not None]))
                if any(d.get(a) is not None for d in panel.values())
                else None) for a in keys}
  alpha_panels = dict(lewm=_alpha_means(lewm_alphas),
                      fpx=_alpha_means(fpx_alphas))
  out = analyse(jp_nll, pe_nll, fidelity, lewm_probe, fpx_probe,
                args.auc_csv, alpha_panels=alpha_panels)
  out['embed_control'] = ctrl
  os.makedirs(args.output, exist_ok=True)
  path = os.path.join(args.output, 'lewm_read.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print(f'-> {path}')


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------

def _fixture(tmp, rng, *, jp_shift=-3.0, emb_shift=0.15, fid=0.8,
             beh_shift=0.0, enc_fail=None, pe_sha_ok=True):
  os.makedirs(tmp, exist_ok=True)

  def e4(path, fit_re_prefix, base):
    with open(path, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['run_id', 'horizon', 'rew_nll_in', 'reward_aware'])
      for side in SIDES:
        for seed in SEEDS:
          v = base + rng.normal(0, 0.5)
          w.writerow([f'{fit_re_prefix}{side}_seed{seed}', 0, v, 1])
  e4(os.path.join(tmp, 'jp.csv'), 'ax1wm_finger_jppxq1ms',
     23.0 + jp_shift)
  e4(os.path.join(tmp, 'pe.csv'), 'ax1wm_finger_pepxq1ms', 23.0)

  # X2-baseline-shaped auc fixture consumed via load_cells' contract:
  # both the jp csv and a local baseline csv (patch X2_BASELINE in
  # selfcheck to point here)
  def auc(path, modes, base):
    with open(path, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['run_id', 'mode', 'domain', 'seed', 'milestone',
                  'auc100k', 'n_ep_100k', 'qc_pass'])
      for k, mode in modes.items():
        for seed in SEEDS:
          w.writerow([f'adapt_{mode}_finger_seed{seed}_ckpt500000',
                      mode, 'finger', seed, '500000',
                      base + rng.normal(0, 6), 96, 1])
  auc(os.path.join(tmp, 'auc.csv'), JP_MODES, 78.0 + beh_shift)
  auc(os.path.join(tmp, 'x2.csv'), X2_BASE_MODES, 78.0)

  for side in SIDES:
    for seed in list(SEEDS) :
      d = os.path.join(tmp, f'lewm_distill_finger_s{side}_seed{seed}')
      os.makedirs(d, exist_ok=True)
      with open(os.path.join(d, 'fidelity.json'), 'w') as f:
        json.dump(dict(holdout_r2=fid), f)
  pb = os.path.join(tmp, 'probe')
  os.makedirs(pb, exist_ok=True)
  for side in SIDES:
    for seed in SEEDS:
      base_a = 0.42 + rng.normal(0, 0.02)
      for rid, val, name, wit, eman in (
          (f'lewm_finger_s{side}_seed{seed}', base_a + emb_shift,
           f'lewm_finger_s{side}_seed{seed}.json', None,
           dict(preproc='unit')),
          (f'ax1wm_finger_fpxpxq1ms{side}_seed{seed}', base_a,
           f'fpx_s{side}_seed{seed}.json', True, None)):
        d = dict(run_id=rid, probeset_id='fingerpx_v1',
                 witness_match=wit, reward_override=None,
                 probe={'alpha_0.0001': dict(auroc=val - 0.01, r2=0.0),
                        ALPHA_KEY: dict(auroc=val, r2=0.0),
                        'alpha_0.01': dict(auroc=val + 0.01, r2=0.0)})
        if eman is not None:
          d['embed_manifest'] = eman
        with open(os.path.join(pb, name), 'w') as f:
          json.dump(d, f)
  with open(os.path.join(tmp, 'embed_control.json'), 'w') as f:
    json.dump({'preproc': 'unit', 'max_abs_diff': 0.0, 'pass': True}, f)
  enc = {f'ax1wm_finger_jppxq1ms{side}_seed{seed}': True
         for side in SIDES for seed in SEEDS}
  if enc_fail:
    enc[enc_fail] = False
  fc = {k: dict(update=500000, total=500000) for k in enc}
  cs = {k: 500000 for k in enc}
  for name, obj in (('enc.json', enc), ('fc.json', fc), ('cs.json', cs)):
    with open(os.path.join(tmp, name), 'w') as f:
      json.dump(obj, f)
  return argparse.Namespace(
      e4_jp=os.path.join(tmp, 'jp.csv'), e4_pe=os.path.join(tmp, 'pe.csv'),
      auc_csv=os.path.join(tmp, 'auc.csv'),
      fidelity_glob=os.path.join(tmp, 'lewm_distill_finger_s*_seed*',
                                 'fidelity.json'),
      enc_checks=os.path.join(tmp, 'enc.json'),
      fit_counters=os.path.join(tmp, 'fc.json'),
      ckpt_steps=os.path.join(tmp, 'cs.json'),
      lewm_probe_glob=os.path.join(pb, 'lewm_finger_s*_seed*.json'),
      fpx_probe_glob=os.path.join(pb, 'fpx_s*_seed*.json'),
      embed_control=os.path.join(tmp, 'embed_control.json'),
      output=os.path.join(tmp, 'out'))


def selfcheck():
  global PE_E4_SHA256, X2_BASELINE
  import shutil
  import tempfile
  base = tempfile.mkdtemp(prefix='lewm_selfcheck_')
  keep_sha, keep_x2 = PE_E4_SHA256, X2_BASELINE
  try:
    assert PE_E4_SHA256.startswith('8bd7a683') and FIDELITY_FLOOR == 0.5

    def run_fx(name, **kw):
      tmp = os.path.join(base, name)
      rng = np.random.default_rng(sum(ord(c) for c in name))
      ns = _fixture(tmp, rng, **kw)
      global PE_E4_SHA256, X2_BASELINE
      PE_E4_SHA256 = sha256(ns.e4_pe)
      X2_BASELINE = os.path.join(tmp, 'x2.csv')
      run(ns)
      with open(os.path.join(ns.output, 'lewm_read.json')) as f:
        return json.load(f)

    out = run_fx('a')  # jp below pe, emb above fpx -> CONFIRMED
    assert out['p_j1_rep']['verdict'] == 'JEPA-MORE-LEGIBLE-THAN-RECON'
    assert out['p_j1_emb']['verdict'] == 'EMB-MORE-LEGIBLE'
    assert out['overall'].startswith('P-J1 CONFIRMED'), out['overall']
    assert out['p_j1_beh']['verdict'] == 'FLOOR-CONSISTENT'

    out = run_fx('b', jp_shift=0.0, emb_shift=0.0)
    assert out['p_j1_rep']['verdict'] == 'NO-SEPARATION'
    assert 'mde_note' in out['p_j1_rep']
    assert out['overall'].startswith('P-J1 NOT CONFIRMED'), out['overall']

    out = run_fx('c', jp_shift=3.0, emb_shift=0.0)  # jp WORSE than pe
    assert out['p_j1_rep']['verdict'] == 'RECON-MORE-LEGIBLE'
    assert out['overall'].startswith('P-J1 NOT CONFIRMED')

    out = run_fx('d', jp_shift=0.0)  # only emb fires -> PARTIAL
    assert out['overall'].startswith('P-J1 PARTIAL'), out['overall']

    out = run_fx('e', fid=0.3)  # fidelity floor -> INSTRUMENT-LIMITED
    assert out['overall'].startswith('INSTRUMENT-LIMITED'), out['overall']
    assert 'INSTRUMENT-LIMITED' in out['p_j1_rep']['verdict']
    assert out['p_j1_emb']['verdict'] == 'EMB-MORE-LEGIBLE'

    out = run_fx('f', beh_shift=40.0)  # behavioral lift branch
    assert out['p_j1_beh']['verdict'] == 'LIFTS', out['p_j1_beh']

    # refusals
    def refused(name, **kw):
      try:
        run_fx(name, **kw)
      except AssertionError:
        return True
      return False
    assert refused('r1', enc_fail='ax1wm_finger_jppxq1ms0_seed3')
    # pe sha mismatch: restore the REAL pin so the fixture csv fails it
    tmp = os.path.join(base, 'r2')
    rng = np.random.default_rng(1)
    ns = _fixture(tmp, rng)
    PE_E4_SHA256 = keep_sha
    X2_BASELINE = os.path.join(tmp, 'x2.csv')
    try:
      run(ns)
      raise SystemExit('pe-sha gate dead')
    except AssertionError:
      pass
    # degenerate auroc refusal
    tmp = os.path.join(base, 'r3')
    rng = np.random.default_rng(2)
    ns = _fixture(tmp, rng)
    p = globlib.glob(ns.lewm_probe_glob)[0]
    d = json.load(open(p))
    d['probe'][ALPHA_KEY]['auroc'] = None
    json.dump(d, open(p, 'w'))
    PE_E4_SHA256 = sha256(ns.e4_pe)
    X2_BASELINE = os.path.join(tmp, 'x2.csv')
    try:
      run(ns)
      raise SystemExit('degenerate-auroc gate dead')
    except AssertionError:
      pass
    # mutation-refusal battery (reviews C6/C3): each mutant must REFUSE
    def mutant(name, mutate):
      tmp = os.path.join(base, name)
      rng = np.random.default_rng(sum(ord(c) for c in name))
      ns = _fixture(tmp, rng)
      mutate(tmp, ns)
      global PE_E4_SHA256, X2_BASELINE
      PE_E4_SHA256 = sha256(ns.e4_pe)
      X2_BASELINE = os.path.join(tmp, 'x2.csv')
      try:
        run(ns)
      except AssertionError:
        return True
      return False

    def short_fc(tmp, ns):
      fc = json.load(open(ns.fit_counters))
      fc[sorted(fc)[0]]['update'] = 250000
      json.dump(fc, open(ns.fit_counters, 'w'))
    assert mutant('r4', short_fc), 'fit-counter gate dead'

    def short_ckpt(tmp, ns):
      cs = json.load(open(ns.ckpt_steps))
      cs[sorted(cs)[0]] = 400000
      json.dump(cs, open(ns.ckpt_steps, 'w'))
    assert mutant('r5', short_ckpt), 'ckpt-step gate dead'

    def bad_reward_aware(tmp, ns):
      lines = open(ns.e4_jp).read().splitlines()
      parts = lines[-1].split(',')
      parts[-1] = '0'
      open(ns.e4_jp, 'w').write('\n'.join(lines[:-1] + [','.join(parts)])
                                + '\n')
    assert mutant('r6', bad_reward_aware), 'reward_aware gate dead'

    def nan_nll(tmp, ns):
      lines = open(ns.e4_jp).read().splitlines()
      parts = lines[-1].split(',')
      parts[2] = 'nan'
      open(ns.e4_jp, 'w').write('\n'.join(lines[:-1] + [','.join(parts)])
                                + '\n')
    assert mutant('r7', nan_nll), 'nan-NLL gate dead'

    def _mutate_probe(pattern, key, val):
      def fn(tmp, ns):
        p = sorted(globlib.glob(getattr(ns, pattern)))[0]
        d = json.load(open(p))
        d[key] = val
        json.dump(d, open(p, 'w'))
      return fn
    assert mutant('r8', _mutate_probe('fpx_probe_glob',
                                      'witness_match', False)), \
        'witness gate dead'
    assert mutant('r9', _mutate_probe('fpx_probe_glob',
                                      'reward_override', '/x/o.npz')), \
        'override gate dead'
    assert mutant('rA', _mutate_probe('lewm_probe_glob',
                                      'probeset_id', 'cup_v1')), \
        'probeset-id gate dead'
    assert mutant('rB', _mutate_probe('lewm_probe_glob',
                                      'embed_manifest',
                                      dict(preproc='raw'))), \
        'preproc-pin gate dead'

    def ctrl_fail(tmp, ns):
      json.dump({'preproc': 'unit', 'max_abs_diff': 0.5, 'pass': False},
                open(ns.embed_control, 'w'))
    assert mutant('rC', ctrl_fail), 'embed-control gate dead'
    print('SELFCHECK PASS')
  finally:
    PE_E4_SHA256, X2_BASELINE = keep_sha, keep_x2
    shutil.rmtree(base, ignore_errors=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--selfcheck', action='store_true')
  ap.add_argument('--e4_jp')
  ap.add_argument('--e4_pe')
  ap.add_argument('--auc_csv')
  ap.add_argument('--fidelity_glob')
  ap.add_argument('--enc_checks')
  ap.add_argument('--fit_counters')
  ap.add_argument('--ckpt_steps')
  ap.add_argument('--lewm_probe_glob')
  ap.add_argument('--fpx_probe_glob')
  ap.add_argument('--embed_control')
  ap.add_argument('--output')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  for req in ('e4_jp', 'e4_pe', 'auc_csv', 'fidelity_glob', 'enc_checks',
              'fit_counters', 'ckpt_steps', 'lewm_probe_glob',
              'fpx_probe_glob', 'embed_control', 'output'):
    assert getattr(args, req), f'--{req} required'
  run(args)


if __name__ == '__main__':
  main()
