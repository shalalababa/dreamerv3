"""Frozen reader — Paper-2 FB-consumer wave
(PREREG_p2_fbconsumer_20260829). ONE execution.

Can a FOREIGN value surface see the replicated opportunity that the
native one definitionally cannot? Review B1 identity, registered:
m_now IS plugin_choice(qfull) = argmax of the Q-ensemble head MEAN,
so value(native-mean-selector) === 0 by construction — the deployed
plug-in is already the native surface's best pick, which is exactly
why the S2 opportunity is "unharvested". The wave therefore has ONE
primary:
  P-FB1: value(fb1) = mean_s mean_r [g(argmax fb_q_s1) - g(m_now)]
(FB picks are G-independent — fb_q never sees the rollouts — so the
all-repeat mean is unbiased; the split-selected ceiling uses the S2
even-select/odd-evaluate form.)
Branch map: FB-SEES-OPPORTUNITY (fires +) / FB-ANTI-SELECTS
(fires −, surprising) / FB-BLIND (no fire; MDE80 + capture ratio vs
the replicated ceiling). Secondaries (desc, no alpha): the s0
selector + the registered s1>s0 cross-paper prediction; the
pessimistic min-over-heads native selector (ensemble-disagreement
value vs the deployed mean-argmax); split ceiling + capture ratio;
the level-confounded FB ACTION contrasts (disclosed, never
verdict-bearing).

Gates: 16 registered cells + 2 dups only; meta pins (env_seed
20260830, repeats 8, states 200, horizon 100, label_every 25,
actions 8, task/dose finger/e1, version _wfb, fb.k==2, obs_keys_ok);
fb record pins (BOTH ckpt shas EXACT, z_seeds {8,1}, n_infer 5120,
frac_pos_reward EQUAL to the published witnesses 0.3263671875 /
0.0541015625 — review m4: equality proves z reproduction); in-file
CRN witness g_fb_rep[:,:,2] == g_all_rep[:,:,0] BITWISE; dup-null
exact (g branches, r_real, AND fb_q constancy); finiteness; run_id
identity; ONE-read guard PATH-PINNED.

Usage:
  python -m analysis.fbcons_read --labels <dir> \
      --output artifacts/fbcons_read_20260829     (--selfcheck)
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
EXPECT_ENV_SEED = 20260830
EXPECT_REPEATS = 8
EXPECT_STATES = 200
EXPECT_HORIZON = 100
EXPECT_LABEL_EVERY = 25
EXPECT_ACTIONS = 8
EXPECT_TASK = 'dmc_finger_turn_hard'
EXPECT_DOSE = 'e1'
VERSION_SUFFIX = '_wfb'
# review m4: the published z-inference witnesses, pinned EXACTLY —
# equality proves z was reproduced from the same export bytes + seed.
FRAC_POS_PINS = (0.3263671875, 0.0541015625)
EXPECT_CELLS = 16
K_FB = 2
S1_SHA = ('8a5409df637bd177b86c6addca2681b0a0df57a8ddc0adca369d9e335c'
          '179285')
S0_SHA = ('183bf521969719f3efa92a88a0c718ed8f0ab15769eb8f550fe6e22cb'
          '5947ff7')
Z_SEEDS = (8, 1)
N_INFER = 5120
FILE_RE = re.compile(
    r'^wfb_finger_e1_seed(5[1-8]|59|6[0-6])_late\.npz$')
DUP_RE = re.compile(
    r'^wfbdup_finger_e1_seed(5[1-8]|59|6[0-6])_late\.npz$')
DUP_CELLS = ('wfbdup_finger_e1_seed51_late.npz',
             'wfbdup_finger_e1_seed59_late.npz')
REGISTERED_OUTPUT = 'artifacts/fbcons_read_20260829'


def refuse(msg):
  raise SystemExit(f'READ REFUSED: {msg}')


def load_cell(path, dup=False):
  name = os.path.basename(path)
  z = np.load(path, allow_pickle=True)
  meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
  version = str(meta.get('labeler_version', ''))
  if not version.endswith(VERSION_SUFFIX):
    refuse(f'{name}: labeler_version {version!r} != *{VERSION_SUFFIX}')
  fbm = meta.get('fb', {})
  if int(fbm.get('k', -1)) != K_FB:
    refuse(f'{name}: fb.k {fbm.get("k")} != registered {K_FB}')
  if fbm.get('obs_keys_ok') is not True:
    refuse(f'{name}: fb.obs_keys_ok not true — obs-order gate '
           'unattested')
  recs = fbm.get('records', [])
  if len(recs) != K_FB:
    refuse(f'{name}: fb.records has {len(recs)} entries != {K_FB}')
  for i, (sha, zs) in enumerate(zip((S1_SHA, S0_SHA), Z_SEEDS)):
    r = recs[i]
    if r.get('ckpt_sha256') != sha:
      refuse(f'{name}: fb record {i} ckpt sha {r.get("ckpt_sha256")!r}'
             f' != registered {sha[:16]}…')
    if int(r.get('z_seed', -1)) != zs or int(
        r.get('n_infer', -1)) != N_INFER:
      refuse(f'{name}: fb record {i} z protocol '
             f'({r.get("z_seed")}, {r.get("n_infer")}) != ({zs}, '
             f'{N_INFER})')
    if abs(float(r.get('frac_pos_reward', -1))
           - FRAC_POS_PINS[i]) > 1e-9:
      refuse(f'{name}: fb record {i} frac_pos_reward '
             f'{r.get("frac_pos_reward")} != published pin '
             f'{FRAC_POS_PINS[i]} — z not reproduced from the '
             'registered export+seed (review m4)')
  w1 = meta.get('w1', {})
  for got, want, label in (
      (int(meta.get('env_seed', -1)), EXPECT_ENV_SEED, 'env_seed'),
      (int(w1.get('repeats', -1)), EXPECT_REPEATS, 'repeats'),
      (int(meta.get('states', -1)), EXPECT_STATES, 'states'),
      (int(meta.get('horizon', -1)), EXPECT_HORIZON, 'horizon'),
      (int(meta.get('label_every', -1)), EXPECT_LABEL_EVERY,
       'label_every'),
      (int(meta.get('actions', -1)), EXPECT_ACTIONS, 'actions')):
    if got != want:
      refuse(f'{name}: {label} {got} != registered {want}')
  for got, want, label in (
      (str(meta.get('task', '')), EXPECT_TASK, 'task'),
      (str(meta.get('dose', '')), EXPECT_DOSE, 'dose')):
    if got != want:
      refuse(f'{name}: {label} {got!r} != registered {want!r}')
  m = (DUP_RE if dup else FILE_RE).match(name)
  rid = str(np.asarray(z['run_id']).reshape(-1)[0])
  seed = int(m.group(1))
  if not ('finger' in rid and 'e1' in rid and f'seed{seed}' in rid):
    refuse(f'{name}: filename (finger,e1,{seed}) not in run_id {rid!r}')
  e = dict(
      g=np.asarray(z['g_all_rep'], float),        # (S, R, M)
      gfb=np.asarray(z['g_fb_rep'], float),       # (S, R, K+1)
      fbq=np.asarray(z['fb_q'], float),           # (S, K, M)
      qfull=np.asarray(z['qfull'], float),        # (S, H, M)
      mn=np.asarray(z['m_now'], int),
      rr=np.asarray(z['r_real'], float),
      obs=np.asarray(z['obs_now'], float),
      dup=np.asarray(z['dup_cand'], bool))
  S, R, M = e['g'].shape
  if (S, R) != (EXPECT_STATES, EXPECT_REPEATS):
    refuse(f'{name}: g_all_rep shape {e["g"].shape}')
  if e['gfb'].shape != (S, R, K_FB + 1):
    refuse(f'{name}: g_fb_rep shape {e["gfb"].shape} != '
           f'{(S, R, K_FB + 1)}')
  if e['fbq'].shape != (S, K_FB, M):
    refuse(f'{name}: fb_q shape {e["fbq"].shape}')
  for key in ('g', 'gfb', 'fbq', 'qfull', 'rr', 'obs'):
    if not np.isfinite(e[key]).all():
      refuse(f'{name}: non-finite {key}')
  if not np.array_equal(e['gfb'][:, :, K_FB], e['g'][:, :, 0]):
    refuse(f'{name}: in-file CRN witness g_fb_rep[:,:,{K_FB}] != '
           'g_all_rep[:,:,0] — pairing broken (INSTRUMENT-INVALID)')
  if dup and not e['dup'].all():
    refuse(f'{name}: dup file without dup rows')
  if not dup and e['dup'].any():
    refuse(f'{name}: dup rows in a main file')
  return seed, e


def load_all(labels_dir):
  cells, dups = {}, []
  for path in sorted(glob.glob(os.path.join(labels_dir, 'wfb*.npz'))):
    name = os.path.basename(path)
    if FILE_RE.match(name):
      seed, e = load_cell(path)
      if seed in cells:
        refuse(f'duplicate cell seed {seed}')
      cells[seed] = e
    elif DUP_RE.match(name):
      if name not in DUP_CELLS:
        refuse(f'{name}: dup pass not one of the registered '
               f'{DUP_CELLS}')
      dups.append((name, load_cell(path, dup=True)[1]))
    else:
      refuse(f'unregistered wfb file {name}')
  if len(cells) != EXPECT_CELLS:
    refuse(f'{len(cells)} cells != registered {EXPECT_CELLS}')
  if len(dups) != len(DUP_CELLS):
    refuse(f'{len(dups)} dup passes != registered {len(DUP_CELLS)}')
  for name, e in dups:
    if not np.all(e['g'] == e['g'][:, :, :1]):
      refuse(f'DUPLICATE-NULL VIOLATION in {name}: candidate branches '
             'differ')
    if float(np.max(np.ptp(e['rr'], axis=1))) != 0.0:
      refuse(f'DUPLICATE-NULL VIOLATION in {name}: r_real varies')
    if float(np.max(np.ptp(e['fbq'], axis=2))) != 0.0:
      refuse(f'DUPLICATE-NULL VIOLATION in {name}: fb_q varies across '
             'identical candidates — the FB scorer is not a function '
             'of (state, action)')
  return cells


def _sel_value(e, scores):
  """mean_s mean_r [g(argmax scores) - g(m_now)] — G-independent
  selector, all repeats."""
  pick = np.argmax(scores, axis=-1)
  i = np.arange(len(pick))
  return float(np.mean(e['g'][i, :, pick] - e['g'][i, :, e['mn']]))


def _split_ceiling(e):
  even = e['g'][:, 0::2].mean(1)
  odd = e['g'][:, 1::2].mean(1)
  sel = even.argmax(1)
  i = np.arange(len(sel))
  return float(np.mean(odd[i, sel] - odd[i, e['mn']]))


def _stat(vals, cl, b=None):
  b = b or w1_read.B_BOOT
  point, ci = w1_read.bca(np.asarray(vals, float), cl, b=b)
  return dict(point=point, ci=list(ci),
              perm_p=w1_read.perm_p(vals, b=b), n_cells=len(vals))


def run_read(labels_dir, output, b=None, _selfcheck=False):
  if not _selfcheck and os.path.abspath(output) != os.path.abspath(
      REGISTERED_OUTPUT):
    refuse(f'output {output!r} != registered {REGISTERED_OUTPUT!r} — '
           'the ONE-read guard is path-pinned')
  os.makedirs(output, exist_ok=True)
  path = os.path.join(output, 'fbcons_read.json')
  if os.path.exists(path):
    refuse(f'{path} exists — ONE read execution is registered')
  cells = load_all(labels_dir)
  cl = [str(s) for s in cells]
  fb1 = [_sel_value(e, e['fbq'][:, 0, :]) for e in cells.values()]
  fb0 = [_sel_value(e, e['fbq'][:, 1, :]) for e in cells.values()]
  # review B1: value(nativeQ-mean) === 0 BY CONSTRUCTION — m_now IS
  # plugin_choice(qfull) = argmax of the head mean, i.e. the deployed
  # plug-in is already the native mean-Q selector's pick. That
  # identity is the registered framing fact (it is WHY the S2
  # opportunity is unharvested), not an estimand. The pessimistic
  # min-over-heads selector is reported as a DESCRIPTIVE
  # disagreement-value secondary (no alpha).
  natq_min = [_sel_value(e, e['qfull'].min(1)) for e in cells.values()]
  ceil = [_split_ceiling(e) for e in cells.values()]
  act1 = [float(np.mean(e['gfb'][:, :, 0]
                        - e['g'].mean(-1))) for e in cells.values()]
  act0 = [float(np.mean(e['gfb'][:, :, 1]
                        - e['g'].mean(-1))) for e in cells.values()]
  b1 = _stat(fb1, cl, b=b)
  fire1 = b1['perm_p'] < ALPHA and b1['ci'][0] > 0
  rev1 = b1['perm_p'] < ALPHA and b1['ci'][1] < 0
  sd1 = float(np.std(fb1, ddof=1))
  if fire1:
    verdict = ('FB-SEES-OPPORTUNITY: a foreign, '
               'visitation-independent value surface selects real '
               'value where the native surface definitionally cannot '
               '(its best pick IS the baseline) — existence proof + '
               'critic-pathway localization')
  elif rev1:
    verdict = ('FB-ANTI-SELECTS: the FB ranking anti-correlates with '
               'real value — surprising, reported symmetrically, '
               'licenses nothing')
  else:
    verdict = ('FB-BLIND: the foreign surface does not see the '
               'opportunity either — the flagship generalizes across '
               'surface classes; realized MDE80 and the capture '
               'ratio against the replicated ceiling reported')
  out = dict(
      prereg='PREREG_p2_fbconsumer_20260829.md',
      n_cells=len(cells),
      p_fb1_fb_selector=dict(b1, alpha=ALPHA, fire=bool(fire1),
                             reversed=bool(rev1),
                             realized_mde80=float(
                                 2.8 * sd1 / np.sqrt(len(fb1)))),
      native_identity_note=('value(nativeQ-mean) === 0 by '
                            'construction: m_now = plugin_choice = '
                            'argmax head-mean (review B1)'),
      secondaries=dict(
          natq_min_disagreement=dict(
              _stat(natq_min, cl, b=b),
              note='pessimistic min-over-heads selector vs the '
                   'deployed mean-argmax baseline: the '
                   'ensemble-disagreement value; descriptive, no '
                   'alpha'),
          fb0_selector=_stat(fb0, cl, b=b),
          s1_gt_s0_prediction=dict(
              point=float(np.mean(fb1) - np.mean(fb0)),
              note='registered cross-paper prediction (FB support '
                   'gate): descriptive, no alpha'),
          split_ceiling=_stat(ceil, cl, b=b),
          capture_ratio_fb1=float(np.mean(fb1) / np.mean(ceil))
          if np.mean(ceil) > 0 else None,
          action_level_fb1=dict(_stat(act1, cl, b=b),
                                note='level-confounded consumer form '
                                     '(F1/F5 lesson): disclosed, '
                                     'never verdict-bearing'),
          action_level_fb0=_stat(act0, cl, b=b)),
      verdict=verdict)
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(f"FB1 {b1['point']:+.4f} p {b1['perm_p']:.4f} | ceiling "
        f"{np.mean(ceil):+.4f} | natq-min(desc) "
        f"{np.mean(natq_min):+.4f}")
  print(verdict.split(':')[0])
  print('->', path)
  return out


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _mk_cell(tmp, name, mode_fb='sees', mode_nq='blind', dup=False,
             seed_val=0):
  """Planted structure: candidate true values v_m per state; a 'sees'
  scorer equals v + tiny noise, a 'blind' scorer is independent
  noise. g = v + repeat noise. m_now = a middling fixed candidate."""
  S, R, M, H, K = EXPECT_STATES, EXPECT_REPEATS, 8, 5, K_FB
  rng = np.random.default_rng(zlib.crc32(name.encode()) + seed_val)
  v = rng.normal(0, 1.0, (S, M)).astype(np.float32)
  if dup:
    v = np.repeat(v[:, :1], M, 1)
  g = (v[:, None, :] + rng.normal(0, 0.4, (S, R, M))).astype(np.float32)
  if dup:
    g = np.repeat(g[:, :, :1], M, 2)

  def scorer(mode):
    if mode == 'sees':
      sc = v + rng.normal(0, 0.05, (S, M))
    elif mode == 'anti':
      sc = -v + rng.normal(0, 0.05, (S, M))
    else:
      sc = rng.normal(0, 1.0, (S, M))
    if dup:
      sc = np.repeat(sc[:, :1], M, 1)
    return sc.astype(np.float32)

  fbq = np.stack([scorer(mode_fb), scorer('blind')], 1)
  # per-head ensemble around a base scorer; qfull's HEAD MEAN defines
  # m_now (producer-faithful, review B1: d0.plugin_choice = argmax of
  # the head mean — the fixture must honor the producer invariant or
  # the selfcheck is blind to baseline-identity errors)
  q_base = scorer(mode_nq)
  qfull = (q_base[:, None, :]
           + rng.normal(0, 0.02, (S, H, M))).astype(np.float32)
  m_now = np.argmax(qfull.mean(1), axis=1).astype(np.int64)
  gfb = np.zeros((S, R, K + 1), np.float32)
  gfb[:, :, K] = g[:, :, 0]
  gfb[:, :, 0] = g.mean(-1) + 0.1
  gfb[:, :, 1] = g.mean(-1) - 0.1
  meta = dict(
      labeler_version='tm2v1_wfb', env_seed=EXPECT_ENV_SEED,
      states=S, horizon=EXPECT_HORIZON,
      label_every=EXPECT_LABEL_EVERY, actions=EXPECT_ACTIONS,
      task=EXPECT_TASK, dose=EXPECT_DOSE,
      w1=dict(repeats=R, fb=True),
      fb=dict(k=K, obs_keys_ok=True, records=[
          dict(ckpt_sha256=S1_SHA, z_seed=8, n_infer=N_INFER,
               frac_pos_reward=FRAC_POS_PINS[0]),
          dict(ckpt_sha256=S0_SHA, z_seed=1, n_infer=N_INFER,
               frac_pos_reward=FRAC_POS_PINS[1])]))
  m = (DUP_RE if dup else FILE_RE).match(name)
  np.savez_compressed(
      os.path.join(tmp, name), meta=json.dumps(meta),
      run_id=np.array([f'tm2r3_finger_e1_seed{m.group(1)}'] * S),
      g_all_rep=g, g_fb_rep=gfb, fb_q=fbq, qfull=qfull,
      m_now=m_now, r_real=np.zeros((S, M), np.float32),
      obs_now=np.zeros((S, 12), np.float32),
      dup_cand=np.full(S, dup))


def _mk_suite(tmp, mode_fb='sees', mode_nq='blind'):
  for s in list(range(51, 59)) + list(range(59, 67)):
    _mk_cell(tmp, f'wfb_finger_e1_seed{s}_late.npz', mode_fb, mode_nq)
  for name in DUP_CELLS:
    _mk_cell(tmp, name, dup=True)


def selfcheck():
  import tempfile
  B = 500
  for mfb, mnq, want in (('sees', 'blind', 'FB-SEES-OPPORTUNITY'),
                         ('blind', 'blind', 'FB-BLIND'),
                         ('anti', 'blind', 'FB-ANTI-SELECTS'),
                         ('blind', 'sees', 'FB-ANTI-SELECTS')):
    # 4th case: a PERFECT native ensemble makes m_now the TRUE argmax
    # — a blind FB selector is then systematically NEGATIVE against
    # it (any pick != the best pick loses value), so the correct
    # verdict is a negative fire; the fixture certifies decidability
    # under the B1 identity (no branch consults value(native-mean)).
    with tempfile.TemporaryDirectory() as tmp:
      _mk_suite(tmp, mfb, mnq)
      out = run_read(tmp, os.path.join(tmp, 'o'), b=B, _selfcheck=True)
      assert out['verdict'].startswith(want), (mfb, mnq, out['verdict'])
      if mnq == 'blind':
        assert out['secondaries']['split_ceiling']['point'] > 0.3
  n_ref = 0

  def mutate(tag, fn):
    nonlocal n_ref
    with tempfile.TemporaryDirectory() as tmp:
      _mk_suite(tmp)
      fn(tmp)
      try:
        run_read(tmp, os.path.join(tmp, 'o'), b=B, _selfcheck=True)
        raise AssertionError(f'refusal not tripped: {tag}')
      except SystemExit as e:
        n_ref += 1
        print(f'  refusal OK [{tag}]: {str(e)[:90]}')

  def _rewrite(tmp, name, **kw):
    _mk_cell(tmp, name, **kw)

  import tempfile

  def _meta_patch(tmp, name, patch):
    p = os.path.join(tmp, name)
    z = dict(np.load(p, allow_pickle=True))
    meta = json.loads(str(z['meta']))
    meta.update(patch)
    z['meta'] = json.dumps(meta)
    np.savez_compressed(p, **z)

  mutate('sha_pin', lambda t: _meta_patch(
      t, 'wfb_finger_e1_seed51_late.npz',
      dict(fb=dict(k=2, obs_keys_ok=True, records=[
          dict(ckpt_sha256='WRONG', z_seed=8, n_infer=N_INFER,
               frac_pos_reward=0.3),
          dict(ckpt_sha256=S0_SHA, z_seed=1, n_infer=N_INFER,
               frac_pos_reward=0.2)]))))
  mutate('env_seed', lambda t: _meta_patch(
      t, 'wfb_finger_e1_seed52_late.npz', dict(env_seed=1)))
  mutate('obs_keys', lambda t: _meta_patch(
      t, 'wfb_finger_e1_seed53_late.npz',
      dict(fb=dict(k=2, obs_keys_ok=False, records=[
          dict(ckpt_sha256=S1_SHA, z_seed=8, n_infer=N_INFER,
               frac_pos_reward=0.3),
          dict(ckpt_sha256=S0_SHA, z_seed=1, n_infer=N_INFER,
               frac_pos_reward=0.2)]))))

  def _break_witness(tmp):
    p = os.path.join(tmp, 'wfb_finger_e1_seed54_late.npz')
    z = dict(np.load(p, allow_pickle=True))
    g = np.array(z['g_fb_rep'])
    g[:, :, K_FB] += 1.0
    z['g_fb_rep'] = g
    np.savez_compressed(p, **z)

  mutate('crn_witness', _break_witness)

  def _dup_fbq(tmp):
    p = os.path.join(tmp, DUP_CELLS[0])
    z = dict(np.load(p, allow_pickle=True))
    q = np.array(z['fb_q'])
    q[:, 0, 1] += 1.0
    z['fb_q'] = q
    np.savez_compressed(p, **z)

  mutate('dup_fbq', _dup_fbq)
  mutate('count', lambda t: os.remove(
      os.path.join(t, 'wfb_finger_e1_seed55_late.npz')))
  mutate('unregistered', lambda t: _mk_cell(
      t, 'wfbdup_finger_e1_seed60_late.npz', dup=True))

  def _one_read(tmp):
    od = os.path.join(tmp, 'o')
    os.makedirs(od)
    open(os.path.join(od, 'fbcons_read.json'), 'w').close()

  mutate('one_read', _one_read)
  assert n_ref == 8
  print('fbcons_read selfcheck PASS (3-branch map over 4 '
        'producer-faithful fixtures incl. perfect-native baseline + '
        '8 refusals: sha pin, env_seed, obs_keys_ok, CRN witness, '
        'dup fb_q constancy, cell count, unregistered dup, '
        'path/one-read; pins: shas '
        f'{S1_SHA[:8]}…/{S0_SHA[:8]}…, env_seed {EXPECT_ENV_SEED}, '
        f'cells {EXPECT_CELLS})')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--labels')
  p.add_argument('--output')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not (args.labels and args.output):
    p.error('--labels and --output required (or --selfcheck)')
  run_read(args.labels, args.output)


if __name__ == '__main__':
  main()
