"""Frozen read for the competence-repair intervention (Paper 2).

Registered in prereg/PREREG_competence_repair_20260730.md and committed
BEFORE the consumer-model trainer has ever run on real labels and
before any repaired label exists — --consumer_model has never been
executed anywhere. The repaired side comes from the cm-extended labeler
(version d1fix_20260724_cm1, EXACT pin) deploying the LORO ridge model
of d0/train_consumer_model.py; the control side is the COMMITTED R3
late label set itself: repaired passes reuse the committed wave's
labeler seed 0 and dials, and the external chooser influences nothing
upstream of the saved row, so both passes walk bit-identical base
trajectories (DETERMINISM GATE below).

Files:
  repaired  <labels_dir>/rep_{dom}_{dose}_seed{s}_late.npz   (32 cells)
  committed <committed_dir>/r3_{dom}_{dose}_seed{s}_late.npz (32 cells)
  model     the single trainer output npz (sha-pinned three ways)

Determinism gate (registered): per cell, episode/step must match
EXACTLY and g_all must match exactly (registered fallback: atol
PAIR_ATOL = 1e-5, disclosed as pair_tolerance_used); ANY violation
QUARANTINES the read — env/library drift or a chooser side-channel
would surface here, and the smoke stage checks one cell via --detgate
before the wave.

Estimand per state: achieved = g_all[arange, m_real] - g_now, on each
side; the primary target is the per-state PAIRED difference
[achieved_rep - achieved_orig], per-run mean, run-clustered percentile
bootstrap (32 clusters, B=10K, rng 0).
  P-REP1  CI entirely > 0  (the repair helps at all)
  P-REP2  point >= MATERIALITY = 0.6465 (adjudicated only when P-REP1
          fires): 50% of the committed R3 pooled gap point 1.29304688
          (artifacts/r3_competence_20260729/r3.json) — the SAME
          registered constant as PREREG_r3_doubling_20260729.md (see
          the prereg for the truncation-direction disclosure).

Verdicts: REPAIRED (both) / PARTIAL (P-REP1 only) /
NOT-REPAIRABLE-FROM-OBSERVABLES (CI straddles 0; scope-limited: a
LINEAR probe on stored observables — NOT "no repair possible") /
HARMFUL (CI entirely < 0) / QUARANTINE (determinism gate) /
SUBSTRATE-GONE (existence gate marker; no adjudication).

Usage:
  python -m analysis.repair_read --labels <rep dir> \
      --committed <committed late-label dir> --model <model.npz> \
      --output <dir>
  python -m analysis.repair_read --detgate <rep dir> --committed <dir>
      (value-blind smoke gate on the ONE registered cell: touches ONLY
       meta version + episode/step/g_all; no estimand is ever computed)
  python -m analysis.repair_read --selfcheck
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re

import numpy as np

from analysis.r3_read import (
    B_BOOT, RNG_SEED, LABELER_VERSION, EXPECT_SEEDS, DOSES, MATS,
    validate_arrays, _cluster_boot)
from d0.train_consumer_model import (
    FEATURE_MAP_VERSION, TRAIN_FILES, load_consumer_model, run_key,
    sha256_file, sibling_files)

# Pin the shared-frozen import surface: an amendment to a sibling module
# must not silently change this reader's registered machinery.
assert EXPECT_SEEDS == tuple(range(31, 39)), EXPECT_SEEDS
assert B_BOOT == 10_000 and RNG_SEED == 0, (B_BOOT, RNG_SEED)
assert LABELER_VERSION == 'd1fix_20260724', LABELER_VERSION
assert MATS == ('early', 'late') and DOSES == ('e1', 'e4'), (MATS, DOSES)
assert FEATURE_MAP_VERSION == 'cmfeat1', FEATURE_MAP_VERSION
assert len(TRAIN_FILES) == 64, len(TRAIN_FILES)

CM_VERSION = 'd1fix_20260724_cm1'
assert CM_VERSION == LABELER_VERSION + '_cm1', CM_VERSION
CORE_DOMAINS = ('cup', 'finger')
N_STATES = 200
N_CANDS = 8
PAIR_ATOL = 1e-5
# 50% of the committed R3 pooled gap point 1.29304688
# (artifacts/r3_competence_20260729/r3.json p_r3b_gap.point), truncated
# to 4 dp — the SAME constant registered in PREREG_r3_doubling_20260729
# (truncation-direction note in the prereg: here the 2.3e-5 truncation
# is anti-conservative and is kept for cross-registration consistency).
MATERIALITY = 0.6465
# Frozen byte-pin of the committed R3 label bundle (the paired control
# AND the training data): sha256 over 'name:sha256(file)\n' of the 64
# registered files in sorted order, computed at freeze from
# local_results/r3_competence_20260729_105146/labels (byte-pinned in
# git at the read commit 4f74b187). check_model recomputes it from
# --committed at read time — "trained on the committed artifact" is a
# machine claim against THIS constant, not against whatever dir is
# passed.
COMMITTED_LABELS_DIGEST = (
    '02efe6c5bae641c1f24055ffb4791e37d1b636bb7304f833777c5bf017f2798d')
FILE_REP = re.compile(r'^rep_(cup|finger)_(e1|e4)_seed(\d+)_late\.npz$')
FILE_ORIG = re.compile(r'^r3_(cup|finger)_(e1|e4)_seed(\d+)_late\.npz$')
# Registered smoke cell: ONE full-dial repaired pass, determinism-gated
# via --detgate BEFORE the remaining 31 passes.
DETGATE_CELL = ('cup', 'e1', 31)
DETGATE_MARKER = 'REP_DETGATE_OK'
SUBSTRATE_MARKER = 'SUBSTRATE_GONE'
SHA_RECORD = 'CONSUMER_MODEL_SHA256'
# Registered dial identity (from every file's meta on BOTH sides): the
# repaired wave reuses the committed R3 dials INCLUDING labeler seed 0
# — that identity is what makes the committed labels the paired control.
DIALS_ORIG = dict(states=200, horizon=100, label_every=25, seed=0,
                  actions=8, rollouts=16, ref_stride=5, mass_scale=1.0,
                  behavior_checkpoint='')
DIALS_REP = dict(DIALS_ORIG, consumer_feature_map=FEATURE_MAP_VERSION)


def per_state_achieved(g_all, g_now, m_real):
  """achieved = G[m_real] - G[m_now], per labeled state."""
  g_all = np.asarray(g_all, float)
  return g_all[np.arange(len(m_real)), np.asarray(m_real, int)] - np.asarray(
      g_now, float)


def _load_common(path, name, md, dom, dose, seed):
  z = np.load(path, allow_pickle=True)
  for k, v in (DIALS_REP if md.get('labeler_version') == CM_VERSION
               else DIALS_ORIG).items():
    assert md.get(k) == v, (
        f'{name}: dial {k}={md.get(k)!r} != registered {v!r}')
  assert md.get('train_seed') == seed, (
      f"{name}: meta train_seed={md.get('train_seed')} != filename {seed}")
  want = f'r3_{dom}_{dose}_seed{seed}/ckpt'
  got = str(md.get('checkpoint', '')).rstrip('/')
  assert got.endswith(want), (
      f'{name}: checkpoint {got!r} is not the late ckpt ({want!r})')
  g_all = np.asarray(z['g_all'], float)
  g_now = np.asarray(z['g_now'], float)
  m_real = np.asarray(z['m_real'], int)
  m_now = np.asarray(z['m_now'], int)
  validate_arrays(name, g_all, g_now, m_real)
  assert g_all.shape == (N_STATES, N_CANDS), (
      f'{name}: g_all shape {g_all.shape} != registered '
      f'({N_STATES}, {N_CANDS})')
  assert m_now.min() >= 0 and m_now.max() < N_CANDS, f'{name}: m_now range'
  assert np.allclose(g_now, g_all[np.arange(N_STATES), m_now], atol=1e-5), (
      f'{name}: g_now != g_all[m_now] (oracle_all identity violated)')
  return z, dict(
      g_all=g_all, m_real=m_real, m_now=m_now,
      episode=np.asarray(z['episode'], int),
      step=np.asarray(z['step'], int),
      ach=per_state_achieved(g_all, g_now, m_real),
      opp=np.max(g_all, 1) - g_now)


def load_side(labels_dir, repaired):
  file_re = FILE_REP if repaired else FILE_ORIG
  version = CM_VERSION if repaired else LABELER_VERSION
  cells = {}
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    if name.endswith('_smoke.npz'):
      continue
    m = file_re.match(name)
    if not m:
      continue
    dom, dose, seed = m.group(1), m.group(2), int(m.group(3))
    z0 = np.load(path, allow_pickle=True)
    md = json.loads(str(z0['meta']))
    assert md.get('labeler_version') == version, (
        f'{name}: labeler_version {md.get("labeler_version")!r} != '
        f'{version!r} (exact pin; a base or extension pass from another '
        'wave must trip, never blend)')
    z, blk = _load_common(path, name, md, dom, dose, seed)
    if repaired:
      run = run_key(dom, dose, seed)
      assert md.get('consumer_model_run') == run, (
          f'{name}: consumer_model_run {md.get("consumer_model_run")!r} '
          f'!= {run!r} (LORO deployment: each pass must use its OWN '
          'held-out model)')
      sha = str(md.get('consumer_model_sha256', ''))
      assert len(sha) == 64, f'{name}: consumer_model_sha256 missing'
      ghat = np.asarray(z['ghat'], float)
      probe = np.asarray(z['m_real_probe'], int)
      assert ghat.shape == (N_STATES, N_CANDS), f'{name}: ghat shape'
      assert probe.min() >= 0 and probe.max() < N_CANDS, (
          f'{name}: m_real_probe range')
      # Tie-tolerant chooser identity: m_real must BE a maximizer of the
      # stored f32 ghat (first-index argmax equality would crash the ONE
      # read on an honest pass whenever f32 rounding creates a tie).
      picked = ghat[np.arange(N_STATES), blk['m_real']]
      assert np.array_equal(picked, ghat.max(1)), (
          f'{name}: m_real is not a maximizer of ghat (chooser identity '
          'violated — the pass did not deploy the external chooser it '
          'stamps)')
      blk.update(model_sha=sha, m_real_probe=probe)
    key = (dom, dose, seed)
    assert key not in cells, f'duplicate cell from {name}'
    cells[key] = blk
  if repaired:
    # Trip-don't-ignore: any rep_*.npz outside the registered grid
    # (stray domain, early maturity, misnamed pass) must surface, not
    # vanish — an off-registration --consumer_model execution would
    # otherwise leave no trace in the one read's output.
    stray = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(labels_dir, 'rep_*.npz'))
        if not os.path.basename(p).endswith('_smoke.npz')
        and not FILE_REP.match(os.path.basename(p)))
    assert not stray, f'unregistered rep_ label files present: {stray}'
  return cells


def check_grid(rep, orig):
  for tag, cells in (('repaired', rep), ('committed', orig)):
    assert cells, f'no {tag} late label files found'
    bad = sorted({k[2] for k in cells} - set(EXPECT_SEEDS))
    assert not bad, f'{tag}: unregistered seeds present: {bad}'
    for dom in CORE_DOMAINS:
      for dose in DOSES:
        for seed in EXPECT_SEEDS:
          assert (dom, dose, seed) in cells, (
              f'{tag}: cell missing: {dom}/{dose}/seed{seed} '
              '(all-or-nothing)')
    assert len(cells) == 32, f'{tag}: unexpected extra cells ({len(cells)})'
  return sorted(rep)


def pair_violations(rep, orig):
  """Registered determinism gate, value-blind by construction: only
  episode/step/m_now/g_all are compared (all index/oracle arrays); no
  achieved value is computed here."""
  viols, tol_used = [], []
  for r in sorted(rep):
    a, b = rep[r], orig[r]
    tag = '/'.join(map(str, r))
    if not (np.array_equal(a['episode'], b['episode'])
            and np.array_equal(a['step'], b['step'])):
      viols.append(tag + ': episode/step mismatch')
      continue
    # m_now enters the primary through g_now = g_all[m_now]; drift
    # confined to the value-head path could flip a near-tie plugin
    # argmax while g_all stays bitwise — the gate must see it (m_now is
    # an index array, as value-blind as episode/step).
    if not np.array_equal(a['m_now'], b['m_now']):
      viols.append(tag + ': m_now mismatch (plugin choice drifted)')
      continue
    if np.array_equal(a['g_all'], b['g_all']):
      continue
    if np.allclose(a['g_all'], b['g_all'], rtol=0, atol=PAIR_ATOL):
      tol_used.append(tag)
    else:
      viols.append(tag + f': g_all mismatch beyond PAIR_ATOL={PAIR_ATOL}')
  return viols, tol_used


def check_model(model_path, labels_dir, committed_dir, rep,
                frozen_digest=COMMITTED_LABELS_DIGEST):
  """Weights-provenance protocol (registered): ONE model file, recorded
  before labeling, trained by the frozen trainer on exactly the
  committed artifact, deployed LORO. Value-blind: no weight statistic
  or label statistic is computed here."""
  msha = sha256_file(model_path)
  bad = sorted('/'.join(map(str, k)) for k, v in rep.items()
               if v['model_sha'] != msha)
  assert not bad, (
      f'{len(bad)} repaired passes stamp a consumer_model_sha256 that '
      f'differs from --model {model_path}: ' + ', '.join(bad[:4]))
  rec_path = os.path.join(labels_dir, SHA_RECORD)
  assert os.path.exists(rec_path), (
      f'{SHA_RECORD} missing in {labels_dir}: the registered protocol '
      'records the model sha BEFORE any labeling (driver train stage)')
  recorded = open(rec_path).read().split()[0].strip()
  assert recorded == msha, (
      f'{SHA_RECORD} ({recorded[:12]}...) != sha256 of --model '
      f'({msha[:12]}...): the deployed model is not the recorded one')
  model = load_consumer_model(model_path)  # exact feature-map pin inside
  man = model['manifest']
  files = man.get('files', {})
  assert sorted(files) == list(TRAIN_FILES), (
      'model manifest file list != the 64 registered committed label '
      'files')
  digest = hashlib.sha256()
  for name in sorted(files):
    sha = files[name]
    path = os.path.join(committed_dir, name)
    assert os.path.exists(path), (
        f'manifest names {name} but it is absent from {committed_dir}')
    assert sha256_file(path) == sha, (
        f'{name}: committed file sha256 != manifest — the model was NOT '
        'trained on the committed artifact')
    digest.update(f'{name}:{sha}\n'.encode())
  # Tie the whole chain to THE committed bundle, not just to whatever
  # dir was passed: the frozen digest pins the exact 64 files
  # (selfcheck fixtures pass their own digest; the read uses the
  # frozen default).
  assert digest.hexdigest() == frozen_digest, (
      'committed-labels digest mismatch: --committed does not contain '
      'the byte-exact committed R3 label bundle this registration froze')
  # Code provenance: the model must have been produced by the frozen
  # trainer source, not a locally edited same-version-string variant.
  src_sha = man.get('trainer_source_sha256', '')
  trainer_path = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
      'd0', 'train_consumer_model.py')
  assert src_sha == sha256_file(trainer_path), (
      'trainer_source_sha256 in the model manifest != sha256 of the '
      'checked-out d0/train_consumer_model.py — the model was not '
      'produced by the frozen trainer')
  per_model = man.get('per_model', {})
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        run = run_key(dom, dose, seed)
        assert run in model['models'], f'model missing weights for {run}'
        want = sibling_files(dom, dose, seed)
        assert sorted(per_model.get(run, [])) == want, (
            f'{run}: LORO training list != the 14 registered sibling '
            'files (own-run leakage or scope drift)')
  return msha


def analyze(rep, orig):
  runs = check_grid(rep, orig)
  viols, tol_used = pair_violations(rep, orig)
  if viols:
    return dict(
        primaries=None,
        fires=dict(p_rep1=None, p_rep2_material=None, harmful=None),
        pair_violations=viols, pair_tolerance_used=tol_used,
        n_runs=len(runs),
        verdict=('QUARANTINE: determinism gate violated — repaired '
                 'passes are not state-identical to the committed '
                 'labels for: ' + '; '.join(viols[:6])
                 + (' ...' if len(viols) > 6 else '')
                 + '. No adjudication; audit the labeler/env stack '
                 '(library drift, chooser side-channel) before ANY use.'))

  diffs = {r: [float(np.mean(rep[r]['ach'] - orig[r]['ach']))]
           for r in runs}
  p1 = _cluster_boot(diffs)
  rep1 = bool(p1['ci'][0] > 0)
  harmful = bool(p1['ci'][1] < 0)
  material = bool(rep1 and p1['point'] >= MATERIALITY)

  mean_opp = float(np.mean([np.mean(orig[r]['opp']) for r in runs]))
  mean_ach_rep = float(np.mean([np.mean(rep[r]['ach']) for r in runs]))
  secondary = dict(
      per_domain={dom: _cluster_boot(
          {r: v for r, v in diffs.items() if r[0] == dom})
          for dom in CORE_DOMAINS},
      per_dose={dose: _cluster_boot(
          {r: v for r, v in diffs.items() if r[1] == dose})
          for dose in DOSES},
      achieved_rep=_cluster_boot(
          {r: [float(np.mean(rep[r]['ach']))] for r in runs}),
      achieved_orig=_cluster_boot(
          {r: [float(np.mean(orig[r]['ach']))] for r in runs}),
      oracle_agreement_rep=float(np.mean(
          [np.mean(rep[r]['m_real'] == np.argmax(rep[r]['g_all'], 1))
           for r in runs])),
      oracle_agreement_orig=float(np.mean(
          [np.mean(orig[r]['m_real'] == np.argmax(orig[r]['g_all'], 1))
           for r in runs])),
      chooser_change_rate=float(np.mean(
          [np.mean(rep[r]['m_real'] != orig[r]['m_real'])
           for r in runs])),
      # redundant instrument sanity: on identical trajectories the
      # repaired pass's own op_real argmax must reproduce the committed
      # realized choice (descriptive, never a gate)
      probe_agreement=float(np.mean(
          [np.mean(rep[r]['m_real_probe'] == orig[r]['m_real'])
           for r in runs])),
      residual_gap_fraction=(
          float(1.0 - mean_ach_rep / mean_opp) if mean_opp > 0 else None),
      opportunity_sanity=dict(
          pooled_late_opp=mean_opp,
          max_cell_absdiff=float(max(
              np.max(np.abs(rep[r]['opp'] - orig[r]['opp']))
              for r in runs)),
          note=('pairing sanity ONLY: states are identical by the '
                'determinism gate, so per-state opportunity IS the '
                'committed quantity; recomputed solely to confirm the '
                'pairing, never as a new estimand')))

  if harmful:
    verdict = ('HARMFUL: the paired achieved contrast is entirely '
               'negative — the external linear chooser picks WORSE '
               'candidates than the probe-informed realized choice it '
               'replaces (the real probe carries one-step information '
               'the ridge lacks). Reported; no headline claim.')
  elif rep1 and material:
    verdict = ('REPAIRED: P-REP1 fires and the point estimate clears '
               'the materiality bar (>= half the committed pooled gap) '
               '— the implementation gap is value-knowledge, '
               'harvestable from stored observables by a LINEAR probe; '
               'the competence failure is chooser calibration, not '
               'representation.')
  elif rep1:
    verdict = ('PARTIAL: P-REP1 fires below the materiality bar — a '
               'slice of the gap is linearly harvestable '
               'value-knowledge; most of the gap stands. Both numbers '
               'go to the paper.')
  else:
    verdict = ('NOT-REPAIRABLE-FROM-OBSERVABLES: the paired achieved '
               'contrast straddles 0 — a LINEAR probe on stored '
               'observables (deter, candidate action, qfull summaries, '
               'udyn), trained on oracle returns from sibling runs, '
               'does not detectably repair the consumer. Scope-limited '
               'by registration: NOT a claim that no repair is '
               'possible; the representational account is strengthened '
               '(support must be legible cross-link).')
  return dict(
      primaries=dict(p_rep1_paired_achieved=p1),
      fires=dict(p_rep1=rep1, p_rep2_material=material, harmful=harmful),
      thresholds=dict(materiality=MATERIALITY, pair_atol=PAIR_ATOL),
      pair_violations=[], pair_tolerance_used=tol_used,
      secondary=secondary, n_runs=len(runs), verdict=verdict)


def read(args):
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'repair.json')
  marker = os.path.join(args.labels, SUBSTRATE_MARKER)
  if os.path.exists(marker):
    res = dict(
        primaries=None,
        fires=dict(p_rep1=None, p_rep2_material=None, harmful=None),
        substrate_marker=os.path.abspath(marker),
        verdict=('SUBSTRATE-GONE: the registered existence gate failed '
                 '— run dirs or committed labels are not all present, '
                 'and the wave is all-or-nothing. Recorded as the '
                 'registered outcome; no labels, no adjudication.'))
    with open(out, 'w') as f:
      json.dump(res, f, indent=2)
    print(res['verdict'])
    print(f'-> {out}')
    return
  assert os.path.exists(os.path.join(args.labels, DETGATE_MARKER)), (
      f'{DETGATE_MARKER} missing: the registered smoke stage (one '
      'full-dial pass + --detgate) must run and pass BEFORE the read')
  # Machine-tie MATERIALITY to its committed source (Amendment-1 lesson:
  # constants without a leg tying them to their derivation are
  # structurally unprotected). Skipped only when the artifact is absent
  # (selfcheck tempdirs); the real read runs in-repo where it exists.
  src = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
      'artifacts', 'r3_competence_20260729', 'r3.json')
  if os.path.exists(src):
    with open(src) as f:
      point = json.load(f)['primaries']['p_r3b_gap']['point']
    assert abs(math.floor(point / 2 * 1e4) / 1e4 - MATERIALITY) < 1e-12, (
        f'MATERIALITY {MATERIALITY} != floor(p_r3b_gap.point/2, 4dp) '
        f'from {src} — constant/source drift')
  rep = load_side(args.labels, repaired=True)
  orig = load_side(args.committed, repaired=False)
  assert rep, f'no rep_* labels under {args.labels}'
  assert orig, f'no committed r3_*_late labels under {args.committed}'
  model_sha = check_model(
      args.model, args.labels, args.committed, rep,
      frozen_digest=(getattr(args, 'frozen_digest', None)
                     or COMMITTED_LABELS_DIGEST))
  res = analyze(rep, orig)
  res['labels_dir'] = os.path.abspath(args.labels)
  res['committed_dir'] = os.path.abspath(args.committed)
  res['model'] = dict(path=os.path.abspath(args.model), sha256=model_sha)
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  if res['primaries']:
    for name, blk in res['primaries'].items():
      print(f"{name}: {blk['point']:+.3f} [{blk['ci'][0]:+.3f},"
            f"{blk['ci'][1]:+.3f}] n={blk['n_clusters']}")
  print('fires:', res['fires'])
  print(res['verdict'])
  print(f'-> {out}')


def detgate(labels_dir, committed_dir):
  """Registered smoke gate on the ONE registered cell (DETGATE_CELL).
  VALUE-BLIND by construction: only the meta version and the
  episode/step/g_all arrays are ever read — the realized-choice
  columns, ghat, and g_now are never loaded and no achieved value,
  mean, or any other estimand is computed here."""
  dom, dose, seed = DETGATE_CELL
  pairs = [(os.path.join(labels_dir, f'rep_{dom}_{dose}_seed{seed}_late.npz'),
            CM_VERSION),
           (os.path.join(committed_dir,
                         f'r3_{dom}_{dose}_seed{seed}_late.npz'),
            LABELER_VERSION)]
  arrs = []
  for path, version in pairs:
    assert os.path.exists(path), f'detgate file missing: {path}'
    z = np.load(path, allow_pickle=True)
    md = json.loads(str(z['meta']))
    assert md.get('labeler_version') == version, (
        f'{os.path.basename(path)}: labeler_version '
        f'{md.get("labeler_version")!r} != {version!r}')
    arrs.append((np.asarray(z['episode']), np.asarray(z['step']),
                 np.asarray(z['m_now'], int),
                 np.asarray(z['g_all'], float)))
  (e1, s1, n1, g1), (e2, s2, n2, g2) = arrs
  assert np.isfinite(g1).all() and np.isfinite(g2).all(), (
      'detgate: non-finite g_all (oracle_all pass incomplete?)')
  ok_idx = np.array_equal(e1, e2) and np.array_equal(s1, s2)
  # m_now is an index array (as value-blind as episode/step); it enters
  # the primary through g_now = g_all[m_now], so plugin-choice drift
  # must gate here too.
  ok_now = np.array_equal(n1, n2)
  exact = np.array_equal(g1, g2)
  ok_g = exact or np.allclose(g1, g2, rtol=0, atol=PAIR_ATOL)
  if not (ok_idx and ok_now and ok_g):
    print('DETGATE QUARANTINE: the repaired pass of '
          f'{DETGATE_CELL} is not state-identical to the committed '
          f'label (episode/step ok={ok_idx}, m_now ok={ok_now}, '
          f'g_all ok={ok_g}). Do NOT run the wave; audit the '
          'labeler/env stack first.')
    raise SystemExit(1)
  print('DETGATE OK: episode/step exact, m_now exact, g_all '
        + ('exact' if exact else f'within registered atol {PAIR_ATOL}')
        + ' (value-blind: only episode/step/m_now/g_all touched, no '
        'estimand computed)')


# --------------------------------------------------------------------------
# selfcheck (synthetic fixtures only; no real label path is ever named)
# --------------------------------------------------------------------------

G_ROW = np.round(np.arange(N_CANDS) * 0.1, 6)  # G[m] = 0.1*m


def _mk_cell(m_real_val, m_now_val=0, g_bump=0.0, episode=None, step=None,
             probe=None):
  g_all = np.tile(G_ROW, (N_STATES, 1)) + g_bump
  m_real = np.full(N_STATES, m_real_val, int)
  m_now = np.full(N_STATES, m_now_val, int)
  g_now = g_all[np.arange(N_STATES), m_now]
  episode = np.zeros(N_STATES, int) if episode is None else episode
  step = (np.arange(N_STATES) * 25 + 25) if step is None else step
  blk = dict(g_all=g_all, m_real=m_real, m_now=m_now,
             episode=episode, step=step,
             ach=per_state_achieved(g_all, g_now, m_real),
             opp=np.max(g_all, 1) - g_now)
  if probe is not None:
    blk.update(model_sha='f' * 64,
               m_real_probe=np.full(N_STATES, probe, int))
  return blk


def _synth_sides(rep_m=3, orig_m=0):
  """achieved = 0.1*m (m_now=0), so the paired per-state diff is
  0.1*(rep_m - orig_m) exactly; the repaired probe pick reproduces the
  committed realized choice (probe_agreement = 1)."""
  rep, orig = {}, {}
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        rep[(dom, dose, seed)] = _mk_cell(rep_m, probe=orig_m)
        orig[(dom, dose, seed)] = _mk_cell(orig_m)
  return rep, orig


def _write_npz(path, blk, version, run, fseed, repaired, **meta_over):
  # `fseed` is the filename/train seed; the labeler-dial key 'seed'
  # stays reachable through meta_over (dial-trip legs).
  md = dict((DIALS_REP if repaired else DIALS_ORIG),
            labeler_version=version, train_seed=fseed,
            checkpoint=f'/x/{run}/ckpt')
  if repaired:
    md.update(consumer_model='/x/model.npz',
              consumer_model_sha256=blk.get('model_sha', 'f' * 64),
              consumer_model_run=run)
  md.update(meta_over)
  g_now = blk['g_all'][np.arange(len(blk['m_now'])), blk['m_now']]
  arrays = dict(meta=json.dumps(md),
                g_all=blk['g_all'].astype(np.float32), g_now=g_now,
                m_real=blk['m_real'], m_now=blk['m_now'],
                episode=blk['episode'], step=blk['step'])
  if repaired:
    ghat = np.zeros((len(blk['m_real']), N_CANDS), np.float32)
    ghat[np.arange(len(blk['m_real'])), blk['m_real']] = 1.0
    arrays.update(ghat=blk.get('ghat', ghat),
                  m_real_probe=blk['m_real_probe'])
  np.savez(path, **arrays)


def _write_model(model_path, committed_dir):
  files = {n: sha256_file(os.path.join(committed_dir, n))
           for n in TRAIN_FILES}
  per_model, arrays, runs = {}, {}, []
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        run = run_key(dom, dose, seed)
        runs.append(run)
        per_model[run] = sibling_files(dom, dose, seed)
        arrays.update({f'w_{run}': np.zeros(3),
                       f'b_{run}': np.float64(0.0),
                       f'mu_{run}': np.zeros(3),
                       f'sd_{run}': np.ones(3),
                       f'lambda_{run}': np.float64(1.0)})
  trainer_path = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
      'd0', 'train_consumer_model.py')
  manifest = dict(trainer_version='cm1',
                  feature_map_version=FEATURE_MAP_VERSION,
                  lambda_grid=[1.0], n_features=3,
                  trainer_source_sha256=sha256_file(trainer_path),
                  files=files, per_model=per_model)
  np.savez(model_path, feature_map_version=FEATURE_MAP_VERSION,
           trainer_version='cm1', lambda_grid=np.array([1.0]),
           runs=np.asarray(runs), manifest=json.dumps(manifest),
           **arrays)
  return sha256_file(model_path)


def _fill_wave(d_rep, d_com, rep_m=7, orig_m=0):
  """Write a complete synthetic wave: 32 rep + 64 committed files (the
  early files are placeholders that only feed the sha manifest), the
  model npz, the sha record, and the detgate marker. Returns model sha."""
  os.makedirs(d_rep, exist_ok=True)
  os.makedirs(d_com, exist_ok=True)
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        run = run_key(dom, dose, seed)
        _write_npz(os.path.join(d_rep, f'rep_{dom}_{dose}_seed{seed}'
                                       '_late.npz'),
                   _mk_cell(rep_m, probe=orig_m), CM_VERSION, run, seed,
                   repaired=True)
        _write_npz(os.path.join(d_com, f'r3_{dom}_{dose}_seed{seed}'
                                       '_late.npz'),
                   _mk_cell(orig_m), LABELER_VERSION, run, seed,
                   repaired=False)
        np.savez(os.path.join(d_com, f'r3_{dom}_{dose}_seed{seed}'
                                     '_early.npz'),
                 placeholder=np.zeros(1))
  model_path = os.path.join(d_rep, 'model.npz')
  msha = _write_model(model_path, d_com)
  # repaired metas must stamp the REAL model sha: rewrite them
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        run = run_key(dom, dose, seed)
        blk = _mk_cell(rep_m, probe=orig_m)
        blk['model_sha'] = msha
        _write_npz(os.path.join(d_rep, f'rep_{dom}_{dose}_seed{seed}'
                                       '_late.npz'),
                   blk, CM_VERSION, run, seed, repaired=True)
  with open(os.path.join(d_rep, SHA_RECORD), 'w') as f:
    f.write(msha + '\n')
  open(os.path.join(d_rep, DETGATE_MARKER), 'w').close()
  return model_path, msha


def selfcheck(args):
  import tempfile
  ns = argparse.Namespace

  # Estimand identity on a hand case: G=[[0,2,1],[1,1,1]], m_now=[0,2],
  # m_real=[2,0] => g_now=[0,1], achieved=[1-0, 1-1]=[1, 0].
  g = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
  ach = per_state_achieved(g, np.array([0.0, 1.0]), np.array([2, 0]))
  assert np.allclose(ach, [1.0, 0.0]), ach

  # Branch 1: REPAIRED (diff = 0.7 >= 0.6465, CI > 0).
  res = analyze(*_synth_sides(rep_m=7, orig_m=0))
  assert res['fires'] == dict(p_rep1=True, p_rep2_material=True,
                              harmful=False), res['fires']
  assert res['verdict'].startswith('REPAIRED'), res['verdict']
  blk = res['primaries']['p_rep1_paired_achieved']
  assert abs(blk['point'] - 0.7) < 1e-9 and blk['n_clusters'] == 32
  assert res['secondary']['probe_agreement'] == 1.0
  assert res['secondary']['oracle_agreement_rep'] == 1.0  # m=7 is argmax
  assert res['secondary']['chooser_change_rate'] == 1.0
  assert abs(res['secondary']['residual_gap_fraction'] - 0.0) < 1e-9
  assert res['secondary']['opportunity_sanity']['max_cell_absdiff'] == 0.0

  # Branch 2: PARTIAL (diff = 0.3 fires but below materiality).
  res = analyze(*_synth_sides(rep_m=3, orig_m=0))
  assert res['fires'] == dict(p_rep1=True, p_rep2_material=False,
                              harmful=False)
  assert res['verdict'].startswith('PARTIAL'), res['verdict']

  # Branch 3: NOT-REPAIRABLE (diff = 0 exactly, CI = [0,0]).
  res = analyze(*_synth_sides(rep_m=2, orig_m=2))
  assert res['fires'] == dict(p_rep1=False, p_rep2_material=False,
                              harmful=False)
  assert res['verdict'].startswith('NOT-REPAIRABLE-FROM-OBSERVABLES'), (
      res['verdict'])
  assert 'NOT a claim that no repair is possible' in res['verdict']

  # Branch 4: HARMFUL (diff = -0.2).
  res = analyze(*_synth_sides(rep_m=1, orig_m=3))
  assert res['fires'] == dict(p_rep1=False, p_rep2_material=False,
                              harmful=True)
  assert res['verdict'].startswith('HARMFUL'), res['verdict']

  # Branch 5: QUARANTINE on g_all mismatch beyond tolerance ...
  rep, orig = _synth_sides()
  rep[('cup', 'e1', 31)] = _mk_cell(3, g_bump=1.0, probe=0)
  res = analyze(rep, orig)
  assert res['verdict'].startswith('QUARANTINE'), res['verdict']
  assert res['fires'] == dict(p_rep1=None, p_rep2_material=None,
                              harmful=None)
  assert any('cup/e1/31' in v for v in res['pair_violations'])
  # ... and on episode/step mismatch ...
  rep, orig = _synth_sides()
  rep[('cup', 'e1', 31)] = _mk_cell(3, step=np.arange(N_STATES) * 25 + 26,
                                    probe=0)
  assert analyze(rep, orig)['verdict'].startswith('QUARANTINE')
  # ... and on m_now drift (plugin choice enters via g_now even when
  # g_all is bitwise identical) ...
  rep, orig = _synth_sides()
  rep[('cup', 'e1', 31)] = _mk_cell(3, m_now_val=1, probe=0)
  res = analyze(rep, orig)
  assert res['verdict'].startswith('QUARANTINE'), res['verdict']
  assert any('m_now mismatch' in v for v in res['pair_violations'])
  # ... while sub-atol wobble adjudicates and is disclosed.
  rep, orig = _synth_sides(rep_m=3, orig_m=0)
  rep[('cup', 'e1', 31)] = _mk_cell(3, g_bump=1e-6, probe=0)
  res = analyze(rep, orig)
  assert res['verdict'].startswith('PARTIAL')
  assert res['pair_tolerance_used'] == ['cup/e1/31'], (
      res['pair_tolerance_used'])

  # Grid trips: missing cell each side, unregistered seed.
  rep, orig = _synth_sides()
  del rep[('cup', 'e1', 31)]
  try:
    analyze(rep, orig)
    raise SystemExit('selfcheck FAIL: missing repaired cell not caught')
  except AssertionError as e:
    assert 'all-or-nothing' in str(e), e
  rep, orig = _synth_sides()
  del orig[('finger', 'e4', 38)]
  try:
    analyze(rep, orig)
    raise SystemExit('selfcheck FAIL: missing committed cell not caught')
  except AssertionError:
    pass
  rep, orig = _synth_sides()
  rep[('cup', 'e1', 99)] = _mk_cell(3, probe=0)
  try:
    analyze(rep, orig)
    raise SystemExit('selfcheck FAIL: unregistered seed not caught')
  except AssertionError as e:
    assert 'unregistered seeds' in str(e), e

  def _dir_digest(committed_dir):
    dg = hashlib.sha256()
    for name in sorted(TRAIN_FILES):
      dg.update(f'{name}:'
                f'{sha256_file(os.path.join(committed_dir, name))}\n'
                .encode())
    return dg.hexdigest()

  # End-to-end tempdir wave: full read (REPAIRED), then the registered
  # marker/gate mechanics (detgate marker, substrate marker).
  with tempfile.TemporaryDirectory() as d:
    d_rep, d_com = os.path.join(d, 'rep'), os.path.join(d, 'com')
    outdir = os.path.join(d, 'out')
    model_path, msha = _fill_wave(d_rep, d_com)
    detgate(d_rep, d_com)  # smoke-gate leg on the same fixtures
    read(ns(labels=d_rep, committed=d_com, model=model_path,
            output=outdir, frozen_digest=_dir_digest(d_com)))
    with open(os.path.join(outdir, 'repair.json')) as f:
      res = json.load(f)
    assert res['verdict'].startswith('REPAIRED'), res['verdict']
    assert res['fires'] == dict(p_rep1=True, p_rep2_material=True,
                                harmful=False)
    assert res['model']['sha256'] == msha
    # detgate marker is a registered precondition of the read
    os.remove(os.path.join(d_rep, DETGATE_MARKER))
    try:
      read(ns(labels=d_rep, committed=d_com, model=model_path,
              output=outdir))
      raise SystemExit('selfcheck FAIL: missing detgate marker not caught')
    except AssertionError as e:
      assert 'REP_DETGATE_OK' in str(e), e
    open(os.path.join(d_rep, DETGATE_MARKER), 'w').close()
    # SUBSTRATE-GONE marker short-circuits everything
    open(os.path.join(d_rep, SUBSTRATE_MARKER), 'w').close()
    read(ns(labels=d_rep, committed=d_com, model=model_path,
            output=outdir))
    with open(os.path.join(outdir, 'repair.json')) as f:
      assert json.load(f)['verdict'].startswith('SUBSTRATE-GONE')
    os.remove(os.path.join(d_rep, SUBSTRATE_MARKER))

    # Model-protocol trips on the same wave fixtures.
    # sha record mismatch
    with open(os.path.join(d_rep, SHA_RECORD), 'w') as f:
      f.write('0' * 64 + '\n')
    rep_cells = load_side(d_rep, repaired=True)
    try:
      check_model(model_path, d_rep, d_com, rep_cells)
      raise SystemExit('selfcheck FAIL: sha-record mismatch not caught')
    except AssertionError as e:
      assert 'recorded' in str(e), e
    with open(os.path.join(d_rep, SHA_RECORD), 'w') as f:
      f.write(msha + '\n')
    # missing sha record
    os.remove(os.path.join(d_rep, SHA_RECORD))
    try:
      check_model(model_path, d_rep, d_com, rep_cells)
      raise SystemExit('selfcheck FAIL: missing sha record not caught')
    except AssertionError as e:
      assert 'BEFORE any labeling' in str(e), e
    with open(os.path.join(d_rep, SHA_RECORD), 'w') as f:
      f.write(msha + '\n')
    # a pass stamping a different model sha
    bad = _mk_cell(7, probe=0)
    bad['model_sha'] = '0' * 64
    _write_npz(os.path.join(d_rep, 'rep_cup_e1_seed31_late.npz'), bad,
               CM_VERSION, run_key('cup', 'e1', 31), 31, repaired=True)
    rep_cells = load_side(d_rep, repaired=True)
    try:
      check_model(model_path, d_rep, d_com, rep_cells)
      raise SystemExit('selfcheck FAIL: per-pass sha mismatch not caught')
    except AssertionError as e:
      assert 'differs from --model' in str(e), e
    blk = _mk_cell(7, probe=0)
    blk['model_sha'] = msha
    _write_npz(os.path.join(d_rep, 'rep_cup_e1_seed31_late.npz'), blk,
               CM_VERSION, run_key('cup', 'e1', 31), 31, repaired=True)
    # committed file drift breaks the manifest tie
    tampered = os.path.join(d_com, 'r3_cup_e1_seed31_early.npz')
    np.savez(tampered, placeholder=np.ones(2))
    rep_cells = load_side(d_rep, repaired=True)
    try:
      check_model(model_path, d_rep, d_com, rep_cells)
      raise SystemExit('selfcheck FAIL: manifest sha drift not caught')
    except AssertionError as e:
      assert 'NOT' in str(e) and 'trained on the committed' in str(e), e
    _write_model(model_path, d_com)  # re-manifest: back to consistent
    with open(os.path.join(d_rep, SHA_RECORD), 'w') as f:
      f.write(sha256_file(model_path) + '\n')
    # LORO leakage: per_model containing the run's own file must trip
    z = dict(np.load(model_path, allow_pickle=False))
    man = json.loads(str(z['manifest']))
    man['per_model'][run_key('cup', 'e1', 31)][0] = (
        'r3_cup_e1_seed31_early.npz')
    z['manifest'] = json.dumps(man)
    np.savez(model_path, **z)
    with open(os.path.join(d_rep, SHA_RECORD), 'w') as f:
      f.write(sha256_file(model_path) + '\n')
    # rep metas still stamp the old sha -> refresh them via _fill_wave?
    # No: check the leakage trip directly with matching stamps bypassed
    # by rebuilding the wave against the tampered model.
    for dom in CORE_DOMAINS:
      for dose in DOSES:
        for seed in EXPECT_SEEDS:
          blk = _mk_cell(7, probe=0)
          blk['model_sha'] = sha256_file(model_path)
          _write_npz(os.path.join(d_rep, f'rep_{dom}_{dose}_seed{seed}'
                                         '_late.npz'),
                     blk, CM_VERSION, run_key(dom, dose, seed), seed,
                     repaired=True)
    rep_cells = load_side(d_rep, repaired=True)
    try:
      check_model(model_path, d_rep, d_com, rep_cells,
                  frozen_digest=_dir_digest(d_com))
      raise SystemExit('selfcheck FAIL: LORO leakage not caught')
    except AssertionError as e:
      assert 'LORO training list' in str(e), e

    # frozen-digest trip: a byte-consistent but WRONG committed bundle
    # (manifest self-consistent, digest != the frozen pin) must trip.
    _write_model(model_path, d_com)
    with open(os.path.join(d_rep, SHA_RECORD), 'w') as f:
      f.write(sha256_file(model_path) + '\n')
    for dom in CORE_DOMAINS:
      for dose in DOSES:
        for seed in EXPECT_SEEDS:
          blk = _mk_cell(7, probe=0)
          blk['model_sha'] = sha256_file(model_path)
          _write_npz(os.path.join(d_rep, f'rep_{dom}_{dose}_seed{seed}'
                                         '_late.npz'),
                     blk, CM_VERSION, run_key(dom, dose, seed), seed,
                     repaired=True)
    rep_cells = load_side(d_rep, repaired=True)
    try:
      check_model(model_path, d_rep, d_com, rep_cells,
                  frozen_digest='0' * 64)
      raise SystemExit('selfcheck FAIL: frozen-digest mismatch not caught')
    except AssertionError as e:
      assert 'digest mismatch' in str(e), e
    # trainer-source trip: a model whose manifest stamps a different
    # trainer source sha (post-freeze locally edited trainer) must trip.
    z = dict(np.load(model_path, allow_pickle=False))
    man = json.loads(str(z['manifest']))
    man['trainer_source_sha256'] = 'f' * 64
    z['manifest'] = json.dumps(man)
    np.savez(model_path, **z)
    with open(os.path.join(d_rep, SHA_RECORD), 'w') as f:
      f.write(sha256_file(model_path) + '\n')
    for dom in CORE_DOMAINS:
      for dose in DOSES:
        for seed in EXPECT_SEEDS:
          blk = _mk_cell(7, probe=0)
          blk['model_sha'] = sha256_file(model_path)
          _write_npz(os.path.join(d_rep, f'rep_{dom}_{dose}_seed{seed}'
                                         '_late.npz'),
                     blk, CM_VERSION, run_key(dom, dose, seed), seed,
                     repaired=True)
    rep_cells = load_side(d_rep, repaired=True)
    try:
      check_model(model_path, d_rep, d_com, rep_cells,
                  frozen_digest=_dir_digest(d_com))
      raise SystemExit('selfcheck FAIL: trainer-source drift not caught')
    except AssertionError as e:
      assert 'frozen trainer' in str(e), e
    # stray rep_ file trip (trip-don't-ignore)
    np.savez(os.path.join(d_rep, 'rep_cup_e1_seed31_early.npz'),
             placeholder=np.zeros(1))
    try:
      load_side(d_rep, repaired=True)
      raise SystemExit('selfcheck FAIL: stray rep_ file not caught')
    except AssertionError as e:
      assert 'unregistered rep_' in str(e), e
    os.remove(os.path.join(d_rep, 'rep_cup_e1_seed31_early.npz'))

  # Load-time guard trips (fresh minimal tempdirs).
  def _expect_load_trip(d, repaired, needle):
    try:
      load_side(d, repaired=repaired)
      raise SystemExit(f'selfcheck FAIL: {needle} not caught')
    except AssertionError as e:
      assert needle in str(e), e

  run31 = run_key('cup', 'e1', 31)
  with tempfile.TemporaryDirectory() as d:
    # a valid repaired file loads and keys correctly
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'),
               _mk_cell(3, probe=0), CM_VERSION, run31, 31, repaired=True)
    got = load_side(d, repaired=True)
    assert list(got) == [('cup', 'e1', 31)]
    assert abs(got[('cup', 'e1', 31)]['ach'][0] - 0.3) < 1e-6
  with tempfile.TemporaryDirectory() as d:
    # base version on the repaired side must trip (exact pin)
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'),
               _mk_cell(3, probe=0), LABELER_VERSION, run31, 31,
               repaired=True)
    _expect_load_trip(d, True, 'exact pin')
  with tempfile.TemporaryDirectory() as d:
    # cm version on the committed side must trip (exact pin)
    _write_npz(os.path.join(d, 'r3_cup_e1_seed31_late.npz'),
               _mk_cell(0), CM_VERSION, run31, 31, repaired=False)
    _expect_load_trip(d, False, 'exact pin')
  with tempfile.TemporaryDirectory() as d:
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'),
               _mk_cell(3, probe=0), CM_VERSION, run31, 31, repaired=True,
               seed=1)
    _expect_load_trip(d, True, 'dial seed')
  with tempfile.TemporaryDirectory() as d:
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'),
               _mk_cell(3, probe=0), CM_VERSION, run31, 31, repaired=True,
               consumer_feature_map='cmfeat0')
    _expect_load_trip(d, True, 'dial consumer_feature_map')
  with tempfile.TemporaryDirectory() as d:
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'),
               _mk_cell(3, probe=0), CM_VERSION, run31, 31, repaired=True,
               checkpoint=f'/x/{run31}/ckpt_early')
    _expect_load_trip(d, True, 'late ckpt')
  with tempfile.TemporaryDirectory() as d:
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'),
               _mk_cell(3, probe=0), CM_VERSION, run31, 31, repaired=True,
               train_seed=32)
    _expect_load_trip(d, True, 'train_seed')
  with tempfile.TemporaryDirectory() as d:
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'),
               _mk_cell(3, probe=0), CM_VERSION, run31, 31, repaired=True,
               consumer_model_run='r3_cup_e1_seed32')
    _expect_load_trip(d, True, 'LORO deployment')
  with tempfile.TemporaryDirectory() as d:
    # chooser identity: stored ghat must argmax to m_real
    blk = _mk_cell(3, probe=0)
    ghat = np.zeros((N_STATES, N_CANDS), np.float32)
    ghat[:, 5] = 1.0  # argmax 5 != m_real 3
    blk['ghat'] = ghat
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'), blk,
               CM_VERSION, run31, 31, repaired=True)
    _expect_load_trip(d, True, 'chooser identity')
  with tempfile.TemporaryDirectory() as d:
    blk = _mk_cell(3, probe=0)
    blk = dict(blk, g_all=blk['g_all'].copy())
    blk['g_all'][0, 0] = np.nan
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'), blk,
               CM_VERSION, run31, 31, repaired=True)
    _expect_load_trip(d, True, 'non-finite')
  with tempfile.TemporaryDirectory() as d:
    # g_now != g_all[m_now] (oracle_all identity) trips
    _write_npz(os.path.join(d, 'r3_cup_e1_seed31_late.npz'),
               _mk_cell(0), LABELER_VERSION, run31, 31, repaired=False)
    path = os.path.join(d, 'r3_cup_e1_seed31_late.npz')
    z = dict(np.load(path, allow_pickle=True))
    z['g_now'] = np.asarray(z['g_now']) + 0.5
    np.savez(path, **z)
    _expect_load_trip(d, False, 'oracle_all identity')

  # detgate trips: missing file, then beyond-tolerance mismatch.
  with tempfile.TemporaryDirectory() as d:
    d_rep, d_com = os.path.join(d, 'rep'), os.path.join(d, 'com')
    os.makedirs(d_rep)
    os.makedirs(d_com)
    try:
      detgate(d_rep, d_com)
      raise SystemExit('selfcheck FAIL: detgate missing file not caught')
    except AssertionError as e:
      assert 'detgate file missing' in str(e), e
    _write_npz(os.path.join(d_rep, 'rep_cup_e1_seed31_late.npz'),
               _mk_cell(3, probe=0), CM_VERSION, run31, 31, repaired=True)
    _write_npz(os.path.join(d_com, 'r3_cup_e1_seed31_late.npz'),
               _mk_cell(0, g_bump=1.0), LABELER_VERSION, run31, 31,
               repaired=False)
    try:
      detgate(d_rep, d_com)
      raise SystemExit('selfcheck FAIL: detgate mismatch not caught')
    except SystemExit as e:
      assert e.code == 1, e

  print('selfcheck PASS: estimand identity, six verdict branches '
        '(repaired, partial, not-repairable, harmful, quarantine x3 '
        'incl. m_now drift + tolerance fallback, substrate-gone), grid '
        'trips (missing cell both sides, unregistered seed, stray rep_ '
        'file), end-to-end tempdir wave read + detgate-marker gate, '
        'model-protocol trips (sha record missing/mismatch, per-pass '
        'sha, manifest drift, LORO leakage, frozen committed-bundle '
        'digest, trainer-source sha), load guards (exact version pins '
        'both directions, dials incl. seed-0 + feature map, late-ckpt '
        'suffix, train_seed, LORO deployment, tie-tolerant chooser '
        'identity, finiteness, oracle_all identity), detgate '
        'ok/missing/mismatch')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--labels')
  ap.add_argument('--committed')
  ap.add_argument('--model')
  ap.add_argument('--output', default='analysis_out/repair')
  ap.add_argument('--detgate', metavar='REP_LABELS_DIR')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
  elif args.detgate:
    if not args.committed:
      ap.error('--detgate requires --committed')
    detgate(args.detgate, args.committed)
  elif args.labels:
    # SUBSTRATE-GONE recording needs neither the committed dir nor the
    # model (no labels exist in that branch) — do not force dummy paths.
    if not os.path.exists(os.path.join(args.labels, SUBSTRATE_MARKER)):
      if not (args.committed and args.model):
        ap.error('--labels requires --committed and --model (unless the '
                 'SUBSTRATE_GONE marker is present)')
    read(args)
  else:
    ap.error('--labels, --detgate, or --selfcheck required')


if __name__ == '__main__':
  main()
