"""Frozen reader — Paper-2 Wave 1: the first-update cell + C5 dual-objective
(PREREG_p2_wave1_firstupdate_20260821). ONE execution (--pilot excepted:
a dispersion-only look defined below, masks every mean).

Estimand orientation is DAMAGE-POSITIVE and fixed here:
  per state:  d_raw = mean_r g(mode) - mean_r g(cold, iters=1)
  cluster (= run) value = mean over its states; n = 32 clusters.
  d_raw > 0  ==  the FIRST MPPI update damages the raw 100-step return
  relative to the un-optimized policy-prior mode.
C5 rides the same passes:
  d_own = own(mode) - own(cold1), own(a) = r + discount*V(s') — a
  discounted, bootstrapped ONE-step real-env score (the op_real scoring
  convention). It is NOT the planner's H-step latent-model objective:
  C5 closes the discounting/horizon-length half of channel D, not the
  model-vs-environment half (review M4 — stated wherever C5 is cited).
  d_own < 0 == the update HELPS its own one-step objective.

PRIMARY (raw): one_sample_auto over 32 cluster d_raw values (MC
sign-flip pinned at n>20); DAMAGE iff perm p < .05 AND BCa CI lower > 0.
C5 branch map (symmetric, all pre-named):
  SEARCH-PATHOLOGY            raw fires AND own fires (CI>0): the update
                              damages even on the planner's own terms
  OBJECTIVE-MISMATCH          raw fires AND own significantly NEGATIVE:
                              the update helps its own objective while
                              hurting raw G — S3/S4 re-scope to
                              objective mismatch
  DAMAGE-ATTRIBUTION-OPEN     raw fires, own indeterminate
  FIRST-UPDATE-HELPS          raw significantly negative (verbatim)
  MECHANISM-CLOSED-NEGATIVE   raw no-fire — closed negatively at the
                              printed realized MDE80 (registered: this
                              is strictly more than the prior
                              CLOSED-UNATTRIBUTED status)

Pilot mode (--pilot): reads at most 4 cells and prints ONLY n_clusters,
per-cluster SD, and the projected SE at n=32 — every mean is masked.
The wave proceeds regardless of the pilot unless projected SE > 2x the
borrowed 0.0436 (then a dated re-power FILL, before the full read).

Gates (refusal = read NOT consumed; review M1 brought this set to the
w1_read standard): exactly the 32 registered cells + the 2 registered
CRN dup gates (review M3); meta pins env_seed == 20260821, repeats == 4,
horizon == 100, label_every == 25, actions == 8, states == 400,
trainer_version == 'tm2r3_train_20260730', late_step_realized present;
labeler_version endswith '_wv1'; filename <-> run_id identity;
iters_first == 1 and iters_default recorded per state (the dial
witness); act_mode == cands[:, 0] (the mode==candidate-0 witness);
finite arrays; no unregistered wv1* files in the directory; dup gates
EXACTLY zero (g_mode == g_cold1, own_mode == own_cold1, dup flag set).

Usage: python -m analysis.p2_wave1_read --labels '<dir with 32 npz>' \
    --output <dir>   [--pilot]   (--selfcheck)
"""

import argparse
import glob as globlib
import json
import os

import numpy as np

from analysis.rescue_slate_common import Refusal, one_sample_auto, refuse, \
    _ndtri

PREREG = 'PREREG_p2_wave1_firstupdate_20260821.md'
ALPHA = 0.05
ENV_SEED = 20260821
REPEATS = 4
N_STATES = 400
SE_BORROWED = 0.0436
DOMS = ('cup', 'finger')
DOSES = ('e1', 'e4')
SEEDS = tuple(range(51, 59))
HORIZON = 100
LABEL_EVERY = 25
ACTIONS = 8
TRAINER_VERSION = 'tm2r3_train_20260730'
CELLS = [f'wv1tm2_{d}_{e}_seed{s}_late.npz'
         for d in DOMS for e in DOSES for s in SEEDS]
DUP_CELLS = ('wv1dup_finger_e1_seed51_late.npz',
             'wv1dup_cup_e4_seed58_late.npz')


def load_cell(path, expect_dup=False):
  name = os.path.basename(path)
  z = np.load(path, allow_pickle=True)
  meta = json.loads(str(z['meta'])) if 'meta' in z else {}
  if isinstance(meta, str):
    meta = json.loads(meta)
  ver = str(meta.get('labeler_version', ''))
  if not ver.endswith('_wv1'):
    refuse(f'{name}: labeler_version {ver!r} is not a wave-1 label file')
  pins = (('env_seed', ENV_SEED), ('horizon', HORIZON),
          ('label_every', LABEL_EVERY), ('actions', ACTIONS),
          ('states', N_STATES))
  for key, want in pins:
    if int(meta.get(key, -1)) != want:
      refuse(f'{name}: meta.{key} {meta.get(key)} != pinned {want}')
  if str(meta.get('trainer_version', '')) != TRAINER_VERSION:
    refuse(f'{name}: trainer_version {meta.get("trainer_version")!r} != '
           f'{TRAINER_VERSION!r}')
  if meta.get('late_step_realized') in (None, ''):
    refuse(f'{name}: late_step_realized missing (training-audit echo)')
  if int(meta.get('wv1', {}).get('repeats', -1)) != REPEATS:
    refuse(f'{name}: repeats != {REPEATS}')
  # filename <-> run_id identity (M1: file names alone are not trusted)
  stem = name.replace('wv1dup_', '').replace('wv1tm2_', '') \
             .replace('_late.npz', '')
  rid = str(np.asarray(z['run_id']).reshape(-1)[0])
  if stem not in rid:
    refuse(f'{name}: run_id {rid!r} does not carry cell identity {stem!r}')
  gm, gc = np.asarray(z['g_mode_rep']), np.asarray(z['g_cold1_rep'])
  om, oc = np.asarray(z['own_mode']), np.asarray(z['own_cold1'])
  if gm.shape[0] != N_STATES:
    refuse(f'{name}: {gm.shape[0]} states != {N_STATES}')
  if not (np.isfinite(gm).all() and np.isfinite(gc).all()
          and np.isfinite(om).all() and np.isfinite(oc).all()):
    refuse(f'{name}: non-finite label values')
  if not (np.asarray(z['iters_first']) == 1).all():
    refuse(f'{name}: iters_first != 1 somewhere (dial witness)')
  if 'iters_default' not in z:
    refuse(f'{name}: iters_default missing (dial-context witness)')
  cands = np.asarray(z['cands'])
  if not np.allclose(np.asarray(z['act_mode']), cands[:, 0], atol=1e-6):
    refuse(f'{name}: act_mode != cands[:, 0] (mode witness)')
  dup = bool(np.asarray(z['dup']).reshape(-1)[0])
  if dup != expect_dup:
    refuse(f'{name}: dup flag {dup} != expected {expect_dup}')
  if expect_dup:
    if not (np.array_equal(gm, gc) and np.array_equal(om, oc)):
      refuse(f'{name}: CRN dup gate NOT exactly zero — CRN broken')
  d_raw = float((gm.mean(1) - gc.mean(1)).mean())
  d_own = float((om - oc).mean())
  return d_raw, d_own


def run(args):
  paths = {os.path.basename(p): p
           for p in globlib.glob(os.path.join(args.labels, '*.npz'))}
  if args.pilot:
    have = [paths[c] for c in CELLS if c in paths][:4]
    if not have:
      refuse('pilot: no registered cells present')
    d = np.array([load_cell(p)[0] for p in have])
    sd = float(d.std(ddof=1)) if len(d) > 1 else float('nan')
    se32 = sd / np.sqrt(32)
    print(f'PILOT (dispersion-only, means masked): n_clusters={len(d)} '
          f'cluster_sd={sd:.4f} projected_SE_n32={se32:.4f} '
          f'borrowed_SE={SE_BORROWED} '
          f'{"RE-POWER REQUIRED" if se32 > 2 * SE_BORROWED else "OK"}')
    return
  missing = [c for c in CELLS if c not in paths]
  if missing:
    refuse(f'missing {len(missing)} cells (first: {missing[:3]})')
  extra = [n for n in paths if n.startswith(('wv1tm2_', 'wv1dup_'))
           and n not in CELLS and n not in DUP_CELLS]
  if extra:
    refuse(f'unregistered wv1 files present: {extra[:3]}')
  missing_dup = [c for c in DUP_CELLS if c not in paths]
  if missing_dup:
    refuse(f'missing CRN dup gates: {missing_dup}')
  for c in DUP_CELLS:
    load_cell(paths[c], expect_dup=True)
  d_raw, d_own = zip(*[load_cell(paths[c]) for c in CELLS])
  raw = one_sample_auto(np.asarray(d_raw))
  own = one_sample_auto(np.asarray(d_own))
  mde = float((_ndtri(1 - ALPHA / 2) + _ndtri(0.80)) * raw['sd']
              / np.sqrt(len(d_raw)))

  raw_fire = raw['perm_p'] < ALPHA and raw['ci'][0] > 0
  raw_rev = raw['perm_p'] < ALPHA and raw['ci'][1] < 0
  own_fire = own['perm_p'] < ALPHA and own['ci'][0] > 0
  own_neg = own['perm_p'] < ALPHA and own['ci'][1] < 0
  if raw_rev:
    overall = ('FIRST-UPDATE-HELPS: the first MPPI update significantly '
               'IMPROVES raw return over the prior mode — reported '
               'verbatim; the anti-harvest damage account does not hold '
               'at the first update')
  elif raw_fire and own_fire:
    overall = ('SEARCH-PATHOLOGY: the first MPPI update damages raw '
               'return AND the planner\'s own objective — attributed on '
               'the planner\'s own terms; the negative-VoC/planner-repair '
               'line is licensed')
  elif raw_fire and own_neg:
    overall = ('OBJECTIVE-MISMATCH: the first update helps the planner\'s '
               'own objective while damaging raw return — S3/S4 re-scope '
               'to objective mismatch; anti-harvest is NOT a search '
               'pathology claim')
  elif raw_fire:
    overall = ('DAMAGE-ATTRIBUTION-OPEN: raw damage confirmed; the own-'
               'objective leg is indeterminate — the mechanism stays '
               'unattributed and the paper says so')
  else:
    overall = (f'MECHANISM-CLOSED-NEGATIVE: no first-update damage at '
               f'realized MDE80 {mde:.4f} — the mechanism question closes '
               'negatively (registered as strictly more informative than '
               'CLOSED-UNATTRIBUTED)')

  out = dict(prereg=PREREG, overall=overall,
             primary_raw=dict(raw, alpha=ALPHA, fire=bool(raw_fire)),
             c5_own=dict(own, fire=bool(own_fire),
                         significantly_negative=bool(own_neg)),
             realized_mde80=mde, se_borrowed=SE_BORROWED,
             n_clusters=len(d_raw),
             per_cluster=dict(d_raw=list(map(float, d_raw)),
                              d_own=list(map(float, d_own))))
  os.makedirs(args.output, exist_ok=True)
  json.dump(out, open(os.path.join(args.output, 'read.json'), 'w'), indent=1)
  print(f"raw {raw['mean']:+.4f} {raw['ci']} p {raw['perm_p']:.5f}; "
        f"own {own['mean']:+.4f} {own['ci']} p {own['perm_p']:.5f}")
  print(overall.split(':')[0])
  print(f"-> {args.output}/read.json")


def _fixture(base, raw_eff, own_eff, sd=0.05, seed=0):
  rng = np.random.default_rng(seed)
  os.makedirs(base, exist_ok=True)
  S, R = N_STATES, REPEATS

  def write(name, dup):
    stem = name.replace('wv1dup_', '').replace('wv1tm2_', '') \
               .replace('_late.npz', '')
    gm = rng.normal(1.0, 0.3, (S, R)).astype(np.float32)
    if dup:
      gc, oc = gm.copy(), None
      om = rng.normal(0.5, 0.1, S).astype(np.float32)
      oc = om.copy()
    else:
      gc = (gm - raw_eff + rng.normal(0, sd, (S, R))).astype(np.float32)
      om = rng.normal(0.5, 0.1, S).astype(np.float32)
      oc = (om - own_eff + rng.normal(0, sd, S)).astype(np.float32)
    cands = rng.normal(0, 1, (S, 8, 2)).astype(np.float32)
    meta = dict(labeler_version='tm2_oracle_20260730_wv1',
                env_seed=ENV_SEED, horizon=HORIZON,
                label_every=LABEL_EVERY, actions=ACTIONS, states=S,
                trainer_version=TRAINER_VERSION, late_step_realized=100000,
                wv1=dict(repeats=R, seed_base=ENV_SEED, dup=dup))
    np.savez(os.path.join(base, name), g_mode_rep=gm, g_cold1_rep=gc,
             own_mode=om, own_cold1=oc, cands=cands,
             act_mode=cands[:, 0].copy(),
             iters_first=np.ones(S, np.int64),
             iters_default=np.full(S, 6, np.int64),
             dup=np.full(S, dup),
             run_id=np.array([f'tm2r3_{stem}'] * S),
             meta=json.dumps(meta))

  for c in CELLS:
    write(c, dup=False)
  for c in DUP_CELLS:
    write(c, dup=True)
  return base


def selfcheck():
  import tempfile
  cases = ((0.15, 0.15, 'SEARCH-PATHOLOGY'),
           (0.15, -0.15, 'OBJECTIVE-MISMATCH'),
           (0.15, 0.0, 'DAMAGE-ATTRIBUTION-OPEN'),
           (-0.15, 0.0, 'FIRST-UPDATE-HELPS'),
           (0.0, 0.0, 'MECHANISM-CLOSED-NEGATIVE'))
  for raw_eff, own_eff, expect in cases:
    base = tempfile.mkdtemp(prefix='pw1_sc_')
    _fixture(base, raw_eff, own_eff, seed=11)
    run(argparse.Namespace(labels=base, output=os.path.join(base, 'out'),
                           pilot=False))
    r = json.load(open(os.path.join(base, 'out', 'read.json')))
    assert r['overall'].startswith(expect), (raw_eff, own_eff, r['overall'])
  # pilot mode masks means
  base = tempfile.mkdtemp(prefix='pw1_sc_')
  _fixture(base, 0.2, 0.2, seed=12)
  import io, contextlib
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    run(argparse.Namespace(labels=base, output=base, pilot=True))
  txt = buf.getvalue()
  assert 'PILOT' in txt and 'cluster_sd' in txt
  assert '+0.2' not in txt and '0.20' not in txt.replace('0.2026', '')
  n_ref = 0
  for mutate in ('missing', 'seed', 'repeats', 'version', 'horizon',
                 'trainer', 'runid', 'mode_witness', 'dial', 'dup_broken',
                 'dup_missing', 'extra_file'):
    base = tempfile.mkdtemp(prefix='pw1_sc_')
    _fixture(base, 0.1, 0.1, seed=13)
    if mutate == 'missing':
      os.remove(os.path.join(base, CELLS[5]))
    elif mutate == 'dup_missing':
      os.remove(os.path.join(base, DUP_CELLS[0]))
    elif mutate == 'extra_file':
      import shutil
      shutil.copy(os.path.join(base, CELLS[0]),
                  os.path.join(base, 'wv1tm2_cup_e1_seed99_late.npz'))
    else:
      target = DUP_CELLS[1] if mutate == 'dup_broken' else CELLS[0]
      c = os.path.join(base, target)
      z = dict(np.load(c, allow_pickle=True))
      meta = json.loads(str(z['meta']))
      if mutate == 'seed':
        meta['env_seed'] = 20260809
      elif mutate == 'repeats':
        meta['wv1']['repeats'] = 8
      elif mutate == 'version':
        meta['labeler_version'] = 'tm2_oracle_20260730_w2'
      elif mutate == 'horizon':
        meta['horizon'] = 50
      elif mutate == 'trainer':
        meta['trainer_version'] = 'other'
      elif mutate == 'runid':
        z['run_id'] = np.array(['tm2r3_finger_e4_seed58'] * N_STATES)
      elif mutate == 'mode_witness':
        z['act_mode'] = np.asarray(z['act_mode']) + 0.5
      elif mutate == 'dial':
        z['iters_first'] = np.full(N_STATES, 6, np.int64)
      elif mutate == 'dup_broken':
        z['g_cold1_rep'] = np.asarray(z['g_cold1_rep']) + 0.01
      z['meta'] = json.dumps(meta)
      np.savez(c, **z)
    try:
      run(argparse.Namespace(labels=base, output=os.path.join(base, 'out'),
                             pilot=False))
    except Refusal as e:
      n_ref += 1
      print(f'  refusal OK [{mutate}]: {e}')
    else:
      raise AssertionError(mutate)
  assert n_ref == 12
  print('p2_wave1_read selfcheck PASS (5 branches + pilot-masking + 12 '
        f'refusals; pins: alpha {ALPHA}, env_seed {ENV_SEED}, repeats '
        f'{REPEATS}, states {N_STATES}, horizon {HORIZON}, borrowed SE '
        f'{SE_BORROWED})')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--labels')
  ap.add_argument('--output', default='.')
  ap.add_argument('--pilot', action='store_true')
  ap.add_argument('--selfcheck', action='store_true')
  a = ap.parse_args()
  if a.selfcheck:
    selfcheck()
  else:
    assert a.labels
    run(a)


if __name__ == '__main__':
  main()
