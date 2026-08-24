"""Derived reader — WCEM fresh-cohort anti-harvest test
(PREREG_wcem2_20260823). ONE execution. LAST look on the anti-harvest
question for this substrate.

Derived from the frozen analysis/wcem_read.py (executed 23 Aug; NEVER
edited): same load machinery, CEM pins, CRN witness, dup-null and
competence gates — parameterized for the fresh cohort and given the
pre-registered ONE-SIDED primary this registration exists for.

Motivation (registered honestly): the executed look-1 landed
CEM-ACTOR-INDISTINGUISHABLE with point -0.4803 and BCa CI
[-1.155, +0.001] — the upper bound grazing zero from below. This wave
asks the anti-harvest question properly, once, on 24 FRESH cells
(seeds 33-38, the unused R3 LATE checkpoints), and retires it either
way. Winner's-curse caveat carried: the look-1 point may be inflated.

Files (fresh env_seed 20260825):
  wcem_{cup|finger}_{e1|e4}_seed{33..38}_late.npz          (24 cells)
  wcemdup_finger_e1_seed33_late.npz, wcemdup_cup_e4_seed38_late.npz

PRIMARY (one-sided, alpha=.05): d_cell = mean[g_cem - g_actor] over
the 24 fresh cells; FIRES iff one-sided MC sign-flip p < .05 AND the
95% BCa CI upper bound < 0 (conservative conjunction, disclosed).
Branch map (competence gate first, as in look-1):
  ANTI-HARVEST-CONFIRMED    fires -> planning through the model's
                            value surface HURTS beyond the actor's
                            operating point (Goodhart-at-deployment);
                            item-3 (reward-only CEM) trigger MET
  CEM-OUTPERFORMS-FRESH     two-sided positive fire (symmetric
                            surprise, reported)
  ANTI-HARVEST-RETIRED      no fire -> the look-1 lean is retired as
                            noise PERMANENTLY (no third cohort);
                            '-QUALIFIED' appended iff realized
                            one-sided MDE80 > |LOOK1_POINT| (0.4803)
  NO-CALL-PLANNER-INCOMPETENT  competence gate failed (fresh cells)
POOLED (descriptive, NO decision weight): point + BCa CI over the 32
combined cells (both cohorts; different env_seeds so marks are
independent) — pre-named here, mirrors the w1_read_pooled precedent.

Gates: everything the frozen reader gates (CEM pins, env_seed 20260825
on fresh / 20260824 on look-1, repeats/states/horizon, version _wcem,
run_id identity, CRN witness, finiteness, action box, dup-null exact,
cell counts 24/8), PLUS: --look1_read must point at the executed
look-1 json and its cem_minus_actor.point must equal LOOK1_POINT
(pin-validation); ONE-read guard.

Usage:
  python -m analysis.wcem2_read --labels <fresh dir> \
      --look1_labels <look-1 bundle labels dir> \
      --look1_read artifacts/wcem_read_20260823/wcem_read.json \
      --output <dir>       (--selfcheck)
"""

import argparse
import glob
import json
import os
import re
import zlib

import numpy as np

from analysis import w1_read

ALPHA = 0.05
EXPECT_REPEATS = 8
EXPECT_STATES = 200
EXPECT_HORIZON = 100
CEM_PINS = dict(iters=4, samples=64, horizon=6, elites=8, std=0.5)
VERSION_SUFFIX = '_wcem'
LOOK1_POINT = -0.4803125
MC_N = 200000
MC_SEED = 20260820
# review B1: LAST-look permanence is ENFORCED by a path-pinned ONE-read
# guard (Tier-1 _guard_output precedent) — labels are not reproducible
# job-to-job, so a directory-scoped guard would permit a second draw.
REGISTERED_OUTPUT = 'artifacts/wcem2_read_20260823'

FRESH = dict(
    env_seed=20260825, cells=24,
    file_re=re.compile(r'^wcem_(cup|finger)_(e1|e4)_seed(3[3-8])'
                       r'_late\.npz$'),
    dup_re=re.compile(r'^wcemdup_(cup|finger)_(e1|e4)_seed(3[3-8])'
                      r'_late\.npz$'),
    dup_cells=('wcemdup_finger_e1_seed33_late.npz',
               'wcemdup_cup_e4_seed38_late.npz'))
LOOK1 = dict(
    env_seed=20260824, cells=8,
    file_re=re.compile(r'^wcem_(cup|finger)_(e1|e4)_seed(3[12])'
                       r'_late\.npz$'),
    dup_re=re.compile(r'^wcemdup_(cup|finger)_(e1|e4)_seed(3[12])'
                      r'_late\.npz$'),
    dup_cells=('wcemdup_finger_e1_seed31_late.npz',
               'wcemdup_cup_e4_seed32_late.npz'))


def refuse(msg):
  raise SystemExit(f'READ REFUSED: {msg}')


def load_cell(path, spec, dup=False):
  name = os.path.basename(path)
  z = np.load(path, allow_pickle=True)
  meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
  version = str(meta.get('labeler_version', ''))
  if not version.endswith(VERSION_SUFFIX):
    refuse(f'{name}: labeler_version {version!r} != *{VERSION_SUFFIX}')
  cem = meta.get('cem', {})
  for k, want in CEM_PINS.items():
    got = cem.get(k, None)
    if got is None or float(got) != float(want):
      refuse(f'{name}: cem.{k} {got} != registered {want}')
  w1 = meta.get('w1', {})
  for got, want, label in (
      (int(meta.get('env_seed', -1)), spec['env_seed'], 'env_seed'),
      (int(w1.get('repeats', -1)), EXPECT_REPEATS, 'repeats'),
      (int(meta.get('states', -1)), EXPECT_STATES, 'states'),
      (int(meta.get('horizon', -1)), EXPECT_HORIZON, 'horizon')):
    if got != want:
      refuse(f'{name}: {label} {got} != registered {want}')
  m = (spec['dup_re'] if dup else spec['file_re']).match(name)
  rid = str(np.asarray(z['run_id']).reshape(-1)[0])
  dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
  if not (dom in rid and dose in rid and f'seed{seed}' in rid):
    refuse(f'{name}: filename ({dom},{dose},{seed}) not in run_id {rid!r}')
  if not w1.get('cem'):
    refuse(f'{name}: meta.w1.cem not set — not a WCEM pass')
  e = dict(
      g=np.asarray(z['g_all_rep'], float),
      ga=np.asarray(z['g_actor_rep'], float),
      gc=np.asarray(z['g_cem_rep'], float),
      gw=np.asarray(z['g_wit_rep'], float),
      cem_act=np.asarray(z['cem_act'], float),
      cem_score=np.asarray(z['cem_score'], float),
      q_best=np.asarray(z['q_best'], float),
      rr=np.asarray(z['r_real'], float),
      dup=np.asarray(z['dup_cand'], bool))
  S, R, M = e['g'].shape
  if (S, R) != (EXPECT_STATES, EXPECT_REPEATS):
    refuse(f'{name}: g_all_rep shape {e["g"].shape}')
  if e['gc'].shape != (S, R) or e['ga'].shape != (S, R):
    refuse(f'{name}: consumer branch shapes {e["gc"].shape} '
           f'{e["ga"].shape}')
  for key in ('g', 'ga', 'gc', 'gw', 'cem_act', 'cem_score', 'q_best',
              'rr'):
    if not np.isfinite(e[key]).all():
      refuse(f'{name}: non-finite {key}')
  if float(np.max(np.abs(e['cem_act']))) > 1.0 + 1e-5:
    refuse(f'{name}: cem_act outside the action box')
  if not np.array_equal(e['gw'], e['g'][:, :, 0]):
    refuse(f'{name}: CRN witness g_wit_rep != g_all_rep[:,:,0] — '
           'within-repeat CRN broken (INSTRUMENT-INVALID)')
  if dup and not e['dup'].all():
    refuse(f'{name}: dup file without dup rows')
  if not dup and e['dup'].any():
    refuse(f'{name}: dup rows in a main file')
  return (dom, dose, seed), e


def load_all(labels_dir, spec):
  cells, dups = {}, []
  for path in sorted(glob.glob(os.path.join(labels_dir, 'wcem*.npz'))):
    name = os.path.basename(path)
    if spec['file_re'].match(name):
      key, e = load_cell(path, spec)
      if key in cells:
        refuse(f'duplicate cell {key}')
      cells[key] = e
    elif spec['dup_re'].match(name):
      if name not in spec['dup_cells']:
        refuse(f'{name}: dup pass not one of the registered '
               f'{spec["dup_cells"]}')
      dups.append((name, load_cell(path, spec, dup=True)[1]))
    else:
      refuse(f'unregistered wcem file {name}')
  if len(cells) != spec['cells']:
    refuse(f'{len(cells)} cells != registered {spec["cells"]}')
  if len(dups) != len(spec['dup_cells']):
    refuse(f'{len(dups)} dup passes != registered '
           f'{len(spec["dup_cells"])}')
  for name, e in dups:
    if not np.all(e['g'] == e['g'][:, :, :1]):
      refuse(f'DUPLICATE-NULL VIOLATION in {name}: candidate branches '
             'of identical actions differ')
    if float(np.max(np.ptp(e['rr'], axis=1))) != 0.0:
      refuse(f'DUPLICATE-NULL VIOLATION in {name}: r_real varies '
             'across identical candidates')
  return cells


def _stat(vals, cl, b=None):
  b = b or w1_read.B_BOOT
  point, ci = w1_read.bca(np.asarray(vals, float), cl, b=b)
  return dict(point=point, ci=list(ci),
              perm_p=w1_read.perm_p(vals, b=b), n_cells=len(vals))


def one_sided_p(vals, n_mc=MC_N):
  """P(sign-flip null mean <= observed mean); pinned MC."""
  vals = np.asarray(vals, float)
  rng = np.random.default_rng(MC_SEED)
  flips = rng.integers(0, 2, (n_mc, len(vals))) * 2 - 1
  null = (flips * vals).mean(1)
  return float((np.sum(null <= vals.mean() + 1e-15) + 1) / (n_mc + 1))


def run_read(labels_dir, look1_labels, look1_read, output, b=None,
             n_mc=None, _selfcheck=False):
  if not _selfcheck and os.path.abspath(output) != os.path.abspath(
      REGISTERED_OUTPUT):
    refuse(f'output {output!r} != registered {REGISTERED_OUTPUT!r} — '
           'the ONE-read guard is path-pinned (review B1: LAST-look '
           'permanence)')
  if not (look1_read and os.path.isfile(look1_read)):
    refuse('pass --look1_read <the executed wcem_read.json> — this '
           'registration is sequenced behind the look-1 read')
  with open(look1_read) as fh:
    look1_json = json.load(fh)
  got = float(look1_json.get('cem_minus_actor', {}).get('point', 9e9))
  if abs(got - LOOK1_POINT) > 1e-9:
    refuse(f'look-1 read point {got} != pinned {LOOK1_POINT} — wrong '
           'file or tampered pin')
  # review m2: a one-field stub must not satisfy the sequencing proof
  if int(look1_json.get('n_cells', -1)) != 8 or not str(
      look1_json.get('verdict', '')).startswith(
          'CEM-ACTOR-INDISTINGUISHABLE'):
    refuse('look-1 json lacks the executed read\'s n_cells=8 / '
           'verdict — not the executed artifact')
  os.makedirs(output, exist_ok=True)
  path = os.path.join(output, 'wcem2_read.json')
  if os.path.exists(path):
    refuse(f'{path} exists — ONE read execution is registered')
  cells = load_all(labels_dir, FRESH)
  cl = ['%s_%s_%d' % k for k in cells]
  d = [float(np.mean(e['gc'] - e['ga'])) for e in cells.values()]
  comp = [float(np.mean(e['cem_score'] - e['q_best']))
          for e in cells.values()]
  blk = _stat(d, cl, b=b)
  p_one = one_sided_p(d, n_mc=n_mc or MC_N)
  sd = float(np.std(d, ddof=1))
  mde80_one = float((1.6449 + 0.8416) * sd / np.sqrt(len(d)))
  fires_anti = p_one < ALPHA and blk['ci'][1] < 0
  fires_pos = blk['perm_p'] < ALPHA and blk['ci'][0] > 0
  competent = float(np.mean(comp)) >= 0.0
  if not competent:
    verdict = 'NO-CALL-PLANNER-INCOMPETENT'
  elif fires_anti:
    verdict = ('ANTI-HARVEST-CONFIRMED: planning through the model\'s '
               'value surface HURTS beyond the actor\'s operating '
               'point (Goodhart-at-deployment); item-3 trigger MET')
  elif fires_pos:
    verdict = 'CEM-OUTPERFORMS-FRESH (symmetric surprise, reported)'
  else:
    verdict = ('ANTI-HARVEST-RETIRED: the look-1 lean is retired as '
               'noise PERMANENTLY (LAST look, no third cohort)')
    if mde80_one > abs(LOOK1_POINT):
      verdict += ('-QUALIFIED: realized one-sided MDE80 '
                  f'{mde80_one:.3f} > |look-1 point| '
                  f'{abs(LOOK1_POINT):.3f} — power shortfall disclosed')
  pooled = None
  if look1_labels:
    # review m1: the pooled leg is descriptive and must not VETO a
    # fully-determined primary — a look-1 load failure is recorded,
    # never raised (every statistic is seeded, so re-running with a
    # repaired look-1 bundle reproduces the verdict bit-for-bit).
    try:
      l1 = load_all(look1_labels, LOOK1)
      d1 = [float(np.mean(e['gc'] - e['ga'])) for e in l1.values()]
      allv = d + d1
      allc = cl + ['%s_%s_%d' % k for k in l1]
      pooled = dict(_stat(allv, allc, b=b),
                    note='descriptive, NO decision weight; cohorts '
                         'have independent env_seed marks')
    except SystemExit as exc:
      pooled = dict(error=str(exc),
                    note='look-1 load failed; primary unaffected')
  out = dict(
      prereg='PREREG_wcem2_20260823.md',
      n_cells=len(cells),
      cem_minus_actor_fresh=dict(blk, p_one_sided=p_one,
                                 alpha=ALPHA,
                                 fires_anti=bool(fires_anti),
                                 fires_pos=bool(fires_pos)),
      planner_competence=dict(per_cell=comp,
                              pooled=float(np.mean(comp)),
                              gate='PASS' if competent else 'FAIL'),
      realized_mde80_one_sided=mde80_one,
      look1_point_pinned=LOOK1_POINT,
      pooled_descriptive=pooled,
      verdict=verdict)
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(f"fresh d {blk['point']:+.4f} {blk['ci']} p_one {p_one:.5f} | "
        f"MDE80(one) {mde80_one:.3f} | comp "
        f"{float(np.mean(comp)):+.2f}")
  print(verdict.split(':')[0])
  print('->', path)
  return out


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _write_cell(tmp, name, spec, edge=0.0, comp=0.5, noise=0.3,
                dup=False, run_id=None, meta_over=None, cem_over=None,
                break_witness=False):
  S, R, M, A = EXPECT_STATES, EXPECT_REPEATS, 8, 2
  m = (spec['dup_re'] if dup else spec['file_re']).match(name)
  dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
  rng = np.random.default_rng(zlib.crc32(name.encode()))
  g = rng.normal(0, 1.0, (S, R, M)).astype(np.float32)
  if dup:
    g = np.repeat(g[:, :, :1], M, 2)
  ga = rng.normal(0, 1.0, (S, R)).astype(np.float32)
  gc = (ga + edge + rng.normal(0, noise, (S, R))).astype(np.float32)
  gw = g[:, :, 0].copy()
  if break_witness:
    gw = gw + 1.0
  q_best = rng.normal(3.0, 0.2, S).astype(np.float32)
  cem_score = (q_best + comp + rng.normal(0, 0.1, S)).astype(np.float32)
  cem = dict(CEM_PINS)
  cem.update(cem_over or {})
  meta = dict(labeler_version='d1fix_20260724_wcem',
              env_seed=spec['env_seed'], states=S,
              horizon=EXPECT_HORIZON,
              w1=dict(repeats=R, cem=True), cem=cem)
  meta.update(meta_over or {})
  np.savez_compressed(
      os.path.join(tmp, name), meta=json.dumps(meta),
      g_all_rep=g, g_actor_rep=ga, g_cem_rep=gc, g_wit_rep=gw,
      cem_act=rng.uniform(-1, 1, (S, A)).astype(np.float32),
      cem_score=cem_score, q_best=q_best,
      r_real=np.zeros((S, M), np.float32),
      m_now=np.zeros(S, np.int64), dup_cand=np.full(S, dup),
      run_id=np.array([run_id or f'r3_{dom}_{dose}_seed{seed}'] * S))


def _write_suite(tmp, spec, seeds, edge=0.0, comp=0.5, noise=0.3,
                 cell_spread=0.0):
  i = 0
  for dom in ('cup', 'finger'):
    for dose in ('e1', 'e4'):
      for seed in seeds:
        # cell_spread: alternating per-CELL offset (mean exactly edge)
        # — per-sample noise averages out over S*R, so UNDERPOWERED
        # fixtures need cluster-level dispersion.
        e_i = edge + (cell_spread if i % 2 == 0 else -cell_spread)
        _write_cell(tmp, f'wcem_{dom}_{dose}_seed{seed}_late.npz',
                    spec, edge=e_i, comp=comp, noise=noise)
        i += 1
  for name in spec['dup_cells']:
    _write_cell(tmp, name, spec, dup=True)


def selfcheck():
  import tempfile
  B, NMC = 500, 4000

  def _look1(tmp):
    d1 = os.path.join(tmp, 'l1')
    os.makedirs(d1)
    _write_suite(d1, LOOK1, (31, 32), edge=-0.48)
    p = os.path.join(tmp, 'wcem_read.json')
    with open(p, 'w') as fh:
      json.dump(dict(cem_minus_actor=dict(point=LOOK1_POINT),
                     n_cells=8,
                     verdict='CEM-ACTOR-INDISTINGUISHABLE'), fh)
    return d1, p

  cases = ((-0.5, 0.0, 'ANTI-HARVEST-CONFIRMED'),
           (0.0, 0.0, 'ANTI-HARVEST-RETIRED:'),
           (0.0, 1.0, 'ANTI-HARVEST-RETIRED'),   # -QUALIFIED variant
           (0.5, 0.0, 'CEM-OUTPERFORMS-FRESH'))
  for edge, cs, want in cases:
    with tempfile.TemporaryDirectory() as tmp:
      fresh = os.path.join(tmp, 'fresh')
      os.makedirs(fresh)
      _write_suite(fresh, FRESH, range(33, 39), edge=edge,
                   cell_spread=cs)
      l1d, l1p = _look1(tmp)
      out = run_read(fresh, l1d, l1p, os.path.join(tmp, 'out'),
                     b=B, n_mc=NMC, _selfcheck=True)
      assert out['verdict'].startswith(want.rstrip(':')), \
          (edge, cs, out['verdict'])
      if cs == 1.0:
        assert '-QUALIFIED' in out['verdict'], out['verdict']
      if cs == 0.0 and want.startswith('ANTI-HARVEST-RETIRED'):
        assert '-QUALIFIED' not in out['verdict'], out['verdict']
      assert out['pooled_descriptive'] is not None
      assert out['pooled_descriptive']['n_cells'] == 32
  with tempfile.TemporaryDirectory() as tmp:
    fresh = os.path.join(tmp, 'fresh')
    os.makedirs(fresh)
    _write_suite(fresh, FRESH, range(33, 39), edge=-0.5, comp=-0.5)
    l1d, l1p = _look1(tmp)
    out = run_read(fresh, l1d, l1p, os.path.join(tmp, 'out'),
                   b=B, n_mc=NMC, _selfcheck=True)
    assert out['verdict'] == 'NO-CALL-PLANNER-INCOMPETENT'
  n_ref = 0
  for mutate in ('no_look1', 'look1_pin', 'env_seed', 'count',
                 'dup_name', 'one_read', 'cem_pin', 'version',
                 'run_id', 'crn_witness', 'w1cem', 'dup_violation'):
    with tempfile.TemporaryDirectory() as tmp:
      fresh = os.path.join(tmp, 'fresh')
      os.makedirs(fresh)
      _write_suite(fresh, FRESH, range(33, 39), edge=-0.5)
      l1d, l1p = _look1(tmp)
      outdir = os.path.join(tmp, 'out')
      if mutate == 'no_look1':
        l1p = None
      elif mutate == 'look1_pin':
        with open(l1p, 'w') as fh:
          json.dump(dict(cem_minus_actor=dict(point=-0.1), n_cells=8,
                         verdict='CEM-ACTOR-INDISTINGUISHABLE'), fh)
      elif mutate == 'env_seed':
        _write_cell(fresh, 'wcem_cup_e1_seed33_late.npz', FRESH,
                    meta_over=dict(env_seed=20260824))
      elif mutate == 'count':
        os.remove(os.path.join(fresh, 'wcem_cup_e1_seed33_late.npz'))
      elif mutate == 'dup_name':
        _write_cell(fresh, 'wcemdup_cup_e1_seed34_late.npz', FRESH,
                    dup=True)
      elif mutate == 'one_read':
        os.makedirs(outdir)
        open(os.path.join(outdir, 'wcem2_read.json'), 'w').close()
      elif mutate == 'cem_pin':
        _write_cell(fresh, 'wcem_cup_e1_seed33_late.npz', FRESH,
                    cem_over=dict(iters=1))
      elif mutate == 'version':
        _write_cell(fresh, 'wcem_cup_e1_seed33_late.npz', FRESH,
                    meta_over=dict(labeler_version='d1fix_20260724_w1'))
      elif mutate == 'run_id':
        _write_cell(fresh, 'wcem_cup_e1_seed33_late.npz', FRESH,
                    run_id='r3_finger_e1_seed33')
      elif mutate == 'crn_witness':
        _write_cell(fresh, 'wcem_cup_e1_seed33_late.npz', FRESH,
                    break_witness=True)
      elif mutate == 'w1cem':
        _write_cell(fresh, 'wcem_cup_e1_seed33_late.npz', FRESH,
                    meta_over=dict(w1=dict(repeats=EXPECT_REPEATS)))
      elif mutate == 'dup_violation':
        fix = os.path.join(fresh, FRESH['dup_cells'][0])
        z = dict(np.load(fix, allow_pickle=True))
        g2 = np.array(z['g_all_rep'])
        g2[:, :, 1] += 1.0
        z['g_all_rep'] = g2
        np.savez_compressed(fix, **z)
      try:
        run_read(fresh, l1d, l1p, outdir, b=B, n_mc=NMC,
                 _selfcheck=True)
        raise AssertionError(f'refusal not tripped: {mutate}')
      except SystemExit as e:
        n_ref += 1
        print(f'  refusal OK [{mutate}]: {str(e)[:90]}')
  assert n_ref == 12
  print('SELFCHECK PASS (wcem2_read: ANTI-HARVEST-CONFIRMED / RETIRED '
        '/ RETIRED-QUALIFIED / OUTPERFORMS-FRESH / INCOMPETENT; pooled '
        '32-cell descriptive; refusals: look1 sequencing + strengthened pin, '
        'env_seed, count, dup name, path-pinned ONE-read + 6 '
        'inherited gate legs (cem pin/version/run_id/CRN witness/'
        'w1.cem/dup-null); pins: look1 point '
        f'{LOOK1_POINT}, fresh env_seed {FRESH["env_seed"]}, cells '
        f'{FRESH["cells"]})')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--labels')
  p.add_argument('--look1_labels')
  p.add_argument('--look1_read')
  p.add_argument('--output')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not (args.labels and args.look1_labels and args.look1_read
          and args.output):
    p.error('--labels, --look1_labels, --look1_read, --output required')
  run_read(args.labels, args.look1_labels, args.look1_read, args.output)


if __name__ == '__main__':
  main()
