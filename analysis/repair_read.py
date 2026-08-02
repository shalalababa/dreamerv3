"""Frozen read for the competence-repair intervention (Paper 2).

Registered in prereg/PREREG_competence_repair_20260730.md, AMENDED by
prereg/PREREG_competence_repair_amend1_20260801.md after the parent's
QUARANTINE branch fired at smoke and the mandated instrument audit
(artifacts/r3_repair_quarantine_20260801/AUDIT.md) found the labeling
stack is not job-to-job deterministic on the cluster — the parent's
pair-against-committed estimand is RETIRED as unmeasurable-on-substrate
(an instrument fact, not an outcome). The repaired side comes from the
cm-extended labeler (version d1fix_20260724_cm1, EXACT pin) deploying
the LORO ridge model of d0/train_consumer_model.py; the CONTROL is the
IN-PASS shadow chooser: every repaired pass stores both m_real
(consumer argmax of ghat) and m_real_probe + real_scores (the unchanged
op_real probe) evaluated on the same states, trajectories, and g_all.

Files:
  repaired  <labels_dir>/rep_{dom}_{dose}_seed{s}_late.npz   (32 cells)
  committed <committed_dir>/r3_{dom}_{dose}_seed{s}_late.npz (32 cells;
            state-identity reference + provenance digest + descriptive
            cross-era context — NEVER a paired side)
  model     the single trainer output npz (sha-pinned four ways)

Gates (amended): per cell, episode/step must match the committed npz
EXACTLY (state identity — the component that held in the audit); ANY
violation QUARANTINES the read. Within each repaired pass:
g_now == g_all[m_now] (oracle_all identity), m_real is a maximizer of
ghat, and m_real_probe is a maximizer of real_scores (tie-tolerant) —
a coherent shadow pair. The parent's cross-pass m_now/g_all bitwise
comparisons are REMOVED (registered-unpassable per the audit): no
cross-job float comparison GATES anything — the one cross-era float
quantity, the descriptive opportunity_max_cell_absdiff secondary, is
registered as drift context and gates nothing.

Estimand per state, WITHIN each repaired pass:
  achieved_rep   = g_all[arange, m_real] - g_now
  achieved_probe = g_all[arange, m_real_probe] - g_now
  primary target = paired [achieved_rep - achieved_probe]
                 = g_all[arange, m_real] - g_all[arange, m_real_probe]
(g_now cancels), per-run mean, run-clustered percentile bootstrap
(32 clusters, B=10K, rng 0).
  P-REP1  CI entirely > 0  (the repair helps at all)
  P-REP2  point >= MATERIALITY = 0.6465 (adjudicated only when P-REP1
          fires): 50% of the committed R3 pooled gap point 1.29304688
          (artifacts/r3_competence_20260729/r3.json) — the SAME
          registered constant as PREREG_r3_doubling_20260729.md
          (truncation note in the parent; baseline-bridge disclosure
          in the amendment).
Sensitivity leg (disclosure-only): the primary excluding the smoke
cell (cup/e1/31, whose label file predates the amendment);
adjudication is on all 32; any fire-status difference appends a
registered FRAGILITY DISCLOSURE to the verdict, never changes it.

Verdicts: REPAIRED (both) / PARTIAL (P-REP1 only) /
NOT-REPAIRABLE-FROM-OBSERVABLES (CI straddles 0; scope-limited: a
LINEAR probe on stored observables — NOT "no repair possible") /
HARMFUL (CI entirely < 0) / QUARANTINE (state-identity gate) /
SUBSTRATE-GONE (existence gate marker; no adjudication).

Usage:
  python -m analysis.repair_read --labels <rep dir> \
      --committed <committed late-label dir> --model <model.npz> \
      --output <dir>
  python -m analysis.repair_read --detgate <rep dir> --committed <dir>
      (value-blind smoke gate on the ONE registered cell: meta version
       pins, episode/step identity vs committed, g_all finiteness, and
       the within-pass identity gates; value arrays are touched only
       inside identity/argmax assertions; no estimand is ever computed)
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
# Registered smoke cell: ONE full-dial repaired pass, gated via
# --detgate BEFORE the remaining 31 passes. Its label file predates
# Amendment 1, so the read reports a sensitivity leg excluding it.
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
      scores = np.asarray(z['real_scores'], float)
      assert ghat.shape == (N_STATES, N_CANDS), f'{name}: ghat shape'
      assert scores.shape == (N_STATES, N_CANDS), (
          f'{name}: real_scores shape')
      assert probe.min() >= 0 and probe.max() < N_CANDS, (
          f'{name}: m_real_probe range')
      # Finiteness before the maximizer identities: a NaN would make
      # np.argmax/array_equal fail with a misleading identity message.
      assert np.isfinite(ghat).all() and np.isfinite(scores).all(), (
          f'{name}: non-finite ghat/real_scores')
      # Tie-tolerant chooser identity: m_real must BE a maximizer of the
      # stored f32 ghat (first-index argmax equality would crash the ONE
      # read on an honest pass whenever f32 rounding creates a tie).
      picked = ghat[np.arange(N_STATES), blk['m_real']]
      assert np.array_equal(picked, ghat.max(1)), (
          f'{name}: m_real is not a maximizer of ghat (chooser identity '
          'violated — the pass did not deploy the external chooser it '
          'stamps)')
      # Amendment-1 shadow-pair identity (same tie-tolerant form):
      # m_real_probe must BE a maximizer of the stored op_real scores —
      # the pass provably contains a coherent in-pass control chooser.
      picked_p = scores[np.arange(N_STATES), probe]
      assert np.array_equal(picked_p, scores.max(1)), (
          f'{name}: m_real_probe is not a maximizer of real_scores '
          '(shadow-pair identity violated — the in-pass control chooser '
          'is incoherent)')
      # Amended primary lives entirely within this pass: g_now cancels.
      d_pair = (blk['g_all'][np.arange(N_STATES), blk['m_real']]
                - blk['g_all'][np.arange(N_STATES), probe])
      blk.update(model_sha=sha, m_real_probe=probe, d_pair=d_pair,
                 ach_probe=blk['ach'] - d_pair)
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


def state_identity_violations(rep, orig):
  """Amendment-1 state-identity gate, value-blind by construction: only
  episode/step are compared (index arrays). The parent's cross-pass
  m_now/g_all comparisons are REMOVED — the quarantine audit showed the
  stack is not job-to-job deterministic, so no cross-job float
  comparison GATES anything in this read (the descriptive
  opportunity_max_cell_absdiff secondary gates nothing)."""
  viols = []
  for r in sorted(rep):
    a, b = rep[r], orig[r]
    if not (np.array_equal(a['episode'], b['episode'])
            and np.array_equal(a['step'], b['step'])):
      viols.append('/'.join(map(str, r)) + ': episode/step mismatch')
  return viols


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


def _fires(boot):
  """Registered fire statuses from one bootstrap block (shared by the
  32-cell primary and the 31-cell sensitivity leg)."""
  rep1 = bool(boot['ci'][0] > 0)
  return dict(p_rep1=rep1,
              p_rep2_material=bool(rep1 and boot['point'] >= MATERIALITY),
              harmful=bool(boot['ci'][1] < 0))


def analyze(rep, orig):
  runs = check_grid(rep, orig)
  viols = state_identity_violations(rep, orig)
  if viols:
    return dict(
        primaries=None,
        fires=dict(p_rep1=None, p_rep2_material=None, harmful=None),
        state_identity_violations=viols,
        n_runs=len(runs),
        verdict=('QUARANTINE: state-identity gate violated — repaired '
                 'passes do not label the registered committed states '
                 'for: ' + '; '.join(viols[:6])
                 + (' ...' if len(viols) > 6 else '')
                 + '. No adjudication; audit the labeler/env stack '
                 'before ANY use.'))

  # Amended primary: per-state paired [achieved_rep - achieved_probe],
  # entirely within each repaired pass (Amendment 1).
  diffs = {r: [float(np.mean(rep[r]['d_pair']))] for r in runs}
  p1 = _cluster_boot(diffs)
  fires = _fires(p1)
  rep1, material = fires['p_rep1'], fires['p_rep2_material']
  harmful = fires['harmful']
  # Sensitivity leg (disclosure-only): exclude the smoke cell, whose
  # label file predates Amendment 1. Adjudication stays on all 32.
  sens_boot = _cluster_boot(
      {r: v for r, v in diffs.items() if r != DETGATE_CELL})
  sens_fires = _fires(sens_boot)
  fragile = sens_fires != fires

  mean_opp_rep = float(np.mean([np.mean(rep[r]['opp']) for r in runs]))
  mean_opp_orig = float(np.mean([np.mean(orig[r]['opp']) for r in runs]))
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
      achieved_probe=_cluster_boot(
          {r: [float(np.mean(rep[r]['ach_probe']))] for r in runs}),
      # cross-era descriptive context ONLY — never a paired side
      achieved_orig=_cluster_boot(
          {r: [float(np.mean(orig[r]['ach']))] for r in runs}),
      oracle_agreement_rep=float(np.mean(
          [np.mean(rep[r]['m_real'] == np.argmax(rep[r]['g_all'], 1))
           for r in runs])),
      oracle_agreement_probe=float(np.mean(
          [np.mean(rep[r]['m_real_probe']
                   == np.argmax(rep[r]['g_all'], 1)) for r in runs])),
      oracle_agreement_orig=float(np.mean(
          [np.mean(orig[r]['m_real'] == np.argmax(orig[r]['g_all'], 1))
           for r in runs])),
      # in-pass chooser change: consumer pick vs shadow probe pick
      chooser_change_rate=float(np.mean(
          [np.mean(rep[r]['m_real'] != rep[r]['m_real_probe'])
           for r in runs])),
      residual_gap_fraction=(
          float(1.0 - mean_ach_rep / mean_opp_rep)
          if mean_opp_rep > 0 else None),
      cross_era_drift=dict(
          probe_vs_committed_realized_agreement=float(np.mean(
              [np.mean(rep[r]['m_real_probe'] == orig[r]['m_real'])
               for r in runs])),
          opportunity_max_cell_absdiff=float(max(
              np.max(np.abs(rep[r]['opp'] - orig[r]['opp']))
              for r in runs)),
          pooled_late_opp_inpass=mean_opp_rep,
          pooled_late_opp_committed=mean_opp_orig,
          note=('descriptive drift context ONLY (Amendment 1): the '
                'quarantine audit showed job-to-job nondeterminism, so '
                'cross-era per-state quantities are expected to drift '
                '(~50% probe-choice flips — m_real 101/200 '
                'committed<->audit_nocm; plugin m_now flips ~25%; O(1) '
                'float drift); these are NOT sanity expectations and '
                'NOT paired estimands')))

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
  if fragile:
    verdict += (
        ' FRAGILITY DISCLOSURE (registered, Amendment 1): the '
        'sensitivity leg excluding the pre-amendment smoke cell '
        'changes at least one fire status (see '
        'sensitivity_excl_smoke); adjudication remains on all 32 by '
        'registration and this disclosure is mandatory in any '
        'reporting of the verdict.')
  return dict(
      primaries=dict(p_rep1_paired_achieved=p1),
      fires=fires,
      sensitivity_excl_smoke=dict(
          boot=sens_boot, fires=sens_fires, fragile=fragile,
          excluded_cell='/'.join(map(str, DETGATE_CELL))),
      thresholds=dict(materiality=MATERIALITY),
      state_identity_violations=[],
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
  """Amended smoke gate (PREREG_competence_repair_amend1_20260801) on
  the ONE registered cell (DETGATE_CELL): meta version pins both sides,
  episode/step identity vs the committed label, g_all finiteness, and
  the within-pass shadow-pair identities. VALUE-BLIND: value arrays
  (g_all/g_now/ghat/real_scores) are touched only inside
  identity/argmax assertions — no achieved value, mean, or any other
  estimand is computed. The parent's cross-pass m_now/g_all comparison
  is REMOVED (registered-unpassable: the quarantine audit showed the
  stack is not job-to-job deterministic)."""
  dom, dose, seed = DETGATE_CELL
  rep_path = os.path.join(labels_dir,
                          f'rep_{dom}_{dose}_seed{seed}_late.npz')
  com_path = os.path.join(committed_dir,
                          f'r3_{dom}_{dose}_seed{seed}_late.npz')
  for path, version in ((rep_path, CM_VERSION),
                        (com_path, LABELER_VERSION)):
    assert os.path.exists(path), f'detgate file missing: {path}'
  zr = np.load(rep_path, allow_pickle=True)
  mdr = json.loads(str(zr['meta']))
  assert mdr.get('labeler_version') == CM_VERSION, (
      f'{os.path.basename(rep_path)}: labeler_version '
      f'{mdr.get("labeler_version")!r} != {CM_VERSION!r}')
  zc = np.load(com_path, allow_pickle=True)
  mdc = json.loads(str(zc['meta']))
  assert mdc.get('labeler_version') == LABELER_VERSION, (
      f'{os.path.basename(com_path)}: labeler_version '
      f'{mdc.get("labeler_version")!r} != {LABELER_VERSION!r}')
  # committed side: episode/step ONLY (state identity); no committed
  # value array is loaded here.
  ok_idx = (np.array_equal(np.asarray(zr['episode']),
                           np.asarray(zc['episode']))
            and np.array_equal(np.asarray(zr['step']),
                               np.asarray(zc['step'])))
  g_all = np.asarray(zr['g_all'], float)
  g_now = np.asarray(zr['g_now'], float)
  m_now = np.asarray(zr['m_now'], int)
  m_real = np.asarray(zr['m_real'], int)
  probe = np.asarray(zr['m_real_probe'], int)
  ghat = np.asarray(zr['ghat'], float)
  scores = np.asarray(zr['real_scores'], float)
  n = len(m_real)
  ok_fin = bool(np.isfinite(g_all).all() and np.isfinite(ghat).all()
                and np.isfinite(scores).all())
  ok_now = ok_fin and bool(
      np.allclose(g_now, g_all[np.arange(n), m_now], atol=1e-5))
  ok_cm = ok_fin and bool(
      np.array_equal(ghat[np.arange(n), m_real], ghat.max(1)))
  ok_probe = ok_fin and bool(
      np.array_equal(scores[np.arange(n), probe], scores.max(1)))
  if not (ok_idx and ok_fin and ok_now and ok_cm and ok_probe):
    print('DETGATE QUARANTINE: the repaired pass of '
          f'{DETGATE_CELL} fails the amended gate (episode/step '
          f'ok={ok_idx}, g_all/ghat/real_scores finite={ok_fin}, '
          f'g_now==g_all[m_now] ok={ok_now}, m_real maximizes ghat '
          f'ok={ok_cm}, m_real_probe maximizes real_scores '
          f'ok={ok_probe}). Do NOT run the wave; audit the '
          'labeler/env stack first.')
    raise SystemExit(1)
  print('DETGATE OK (Amendment 1): episode/step exact vs committed, '
        'g_all finite, within-pass identities hold (g_now==g_all[m_now]'
        ', m_real maximizes ghat, m_real_probe maximizes real_scores) '
        '(value-blind: value arrays touched only inside identity '
        'assertions, no estimand computed)')


# --------------------------------------------------------------------------
# selfcheck (synthetic fixtures only; no real label path is ever named)
# --------------------------------------------------------------------------

G_ROW = np.round(np.arange(N_CANDS) * 0.1, 6)  # G[m] = 0.1*m


def _mk_cell(m_real_val, m_now_val=0, g_bump=0.0, episode=None, step=None,
             probe=None, g_scale=1.0):
  g_all = np.tile(G_ROW, (N_STATES, 1)) * g_scale + g_bump
  m_real = np.full(N_STATES, m_real_val, int)
  m_now = np.full(N_STATES, m_now_val, int)
  idx = np.arange(N_STATES)
  g_now = g_all[idx, m_now]
  episode = np.zeros(N_STATES, int) if episode is None else episode
  step = (np.arange(N_STATES) * 25 + 25) if step is None else step
  blk = dict(g_all=g_all, m_real=m_real, m_now=m_now,
             episode=episode, step=step,
             ach=per_state_achieved(g_all, g_now, m_real),
             opp=np.max(g_all, 1) - g_now)
  if probe is not None:
    m_probe = np.full(N_STATES, probe, int)
    scores = np.zeros((N_STATES, N_CANDS), np.float32)
    scores[idx, m_probe] = 1.0
    d_pair = g_all[idx, m_real] - g_all[idx, m_probe]
    blk.update(model_sha='f' * 64, m_real_probe=m_probe,
               real_scores=scores, d_pair=d_pair,
               ach_probe=blk['ach'] - d_pair)
  return blk


def _synth_sides(rep_m=3, orig_m=0):
  """achieved = 0.1*m (m_now=0). The repaired cells carry the in-pass
  shadow pair (m_real=rep_m, m_real_probe=orig_m), so the amended
  within-pass paired diff is 0.1*(rep_m - orig_m) exactly; the
  committed cells realize orig_m, so the descriptive
  probe-vs-committed agreement is 1."""
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
    n = len(blk['m_real'])
    ghat = np.zeros((n, N_CANDS), np.float32)
    ghat[np.arange(n), blk['m_real']] = 1.0
    scores = blk.get('real_scores')
    if scores is None:
      scores = np.zeros((n, N_CANDS), np.float32)
      scores[np.arange(n), blk['m_real_probe']] = 1.0
    arrays.update(ghat=blk.get('ghat', ghat),
                  m_real_probe=blk['m_real_probe'],
                  real_scores=scores)
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

  # Branch 1: REPAIRED (in-pass paired diff = 0.7 >= 0.6465, CI > 0).
  res = analyze(*_synth_sides(rep_m=7, orig_m=0))
  assert res['fires'] == dict(p_rep1=True, p_rep2_material=True,
                              harmful=False), res['fires']
  assert res['verdict'].startswith('REPAIRED'), res['verdict']
  assert 'FRAGILITY' not in res['verdict']
  blk = res['primaries']['p_rep1_paired_achieved']
  assert abs(blk['point'] - 0.7) < 1e-9 and blk['n_clusters'] == 32
  sec = res['secondary']
  assert sec['cross_era_drift'][
      'probe_vs_committed_realized_agreement'] == 1.0
  assert sec['cross_era_drift']['opportunity_max_cell_absdiff'] == 0.0
  assert sec['oracle_agreement_rep'] == 1.0   # m=7 is argmax
  assert sec['oracle_agreement_probe'] == 0.0  # probe=0 is not
  assert sec['chooser_change_rate'] == 1.0     # in-pass 7 vs 0
  assert abs(sec['residual_gap_fraction'] - 0.0) < 1e-9
  assert abs(sec['achieved_probe']['point'] - 0.0) < 1e-9
  sens = res['sensitivity_excl_smoke']
  assert sens['excluded_cell'] == 'cup/e1/31'
  assert sens['boot']['n_clusters'] == 31
  assert abs(sens['boot']['point'] - 0.7) < 1e-9
  assert sens['fragile'] is False

  # Amended decisive leg: cross-era drift on the committed side (float
  # drift AND plugin-choice drift on identical states) must ADJUDICATE
  # — the parent's cross-pass gate is removed — and must not move the
  # in-pass primary at all.
  rep, orig = _synth_sides(rep_m=7, orig_m=0)
  orig[('cup', 'e1', 31)] = _mk_cell(2, m_now_val=1, g_bump=1.0)
  res = analyze(rep, orig)
  assert res['verdict'].startswith('REPAIRED'), res['verdict']
  blk = res['primaries']['p_rep1_paired_achieved']
  assert abs(blk['point'] - 0.7) < 1e-9
  assert res['secondary']['cross_era_drift'][
      'opportunity_max_cell_absdiff'] > 0.0
  assert res['state_identity_violations'] == []

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

  # Branch 5: QUARANTINE on episode/step mismatch (the amended
  # state-identity gate — the only remaining cross-file gate). Both
  # halves are exercised: step-only drift AND episode-only drift
  # (mutation review: every other fixture has constant episode=0, so
  # a reader that silently dropped the episode comparison would
  # otherwise pass the whole selfcheck).
  rep, orig = _synth_sides()
  rep[('cup', 'e1', 31)] = _mk_cell(3, step=np.arange(N_STATES) * 25 + 26,
                                    probe=0)
  res = analyze(rep, orig)
  assert res['verdict'].startswith('QUARANTINE'), res['verdict']
  assert res['fires'] == dict(p_rep1=None, p_rep2_material=None,
                              harmful=None)
  assert any('cup/e1/31' in v and 'episode/step' in v
             for v in res['state_identity_violations'])
  rep, orig = _synth_sides()
  rep[('cup', 'e1', 31)] = _mk_cell(3, probe=0,
                                    episode=np.ones(N_STATES, int))
  res = analyze(rep, orig)
  assert res['verdict'].startswith('QUARANTINE'), res['verdict']
  assert any('cup/e1/31' in v for v in res['state_identity_violations'])

  # Sensitivity-leg numerics: a smoke cell that differs from the rest
  # shifts the pooled point but (here) flips no fire status.
  rep, orig = _synth_sides(rep_m=3, orig_m=0)
  rep[('cup', 'e1', 31)] = _mk_cell(7, probe=0)
  orig[('cup', 'e1', 31)] = _mk_cell(0)
  res = analyze(rep, orig)
  blk = res['primaries']['p_rep1_paired_achieved']
  assert abs(blk['point'] - (31 * 0.3 + 0.7) / 32) < 1e-9
  sens = res['sensitivity_excl_smoke']
  assert abs(sens['boot']['point'] - 0.3) < 1e-9
  assert sens['fragile'] is False and 'FRAGILITY' not in res['verdict']

  # Genuine FRAGILITY: the materiality fire is a point threshold, so a
  # single high-magnitude smoke cell flips it deterministically —
  # 31 cells at d=0.6 (below bar) + smoke at d=7.0 => pooled 0.8 (>=
  # bar, material on 32) while the sensitivity leg sits at 0.6 (not
  # material). Every cluster value is positive, so P-REP1 fires on
  # both sides regardless of resampling.
  rep, orig = _synth_sides(rep_m=6, orig_m=0)
  rep[('cup', 'e1', 31)] = _mk_cell(7, probe=0, g_scale=10.0)
  res = analyze(rep, orig)
  blk = res['primaries']['p_rep1_paired_achieved']
  assert abs(blk['point'] - (31 * 0.6 + 7.0) / 32) < 1e-9
  assert res['fires'] == dict(p_rep1=True, p_rep2_material=True,
                              harmful=False), res['fires']
  sens = res['sensitivity_excl_smoke']
  assert abs(sens['boot']['point'] - 0.6) < 1e-9
  assert sens['fires']['p_rep2_material'] is False
  assert sens['fragile'] is True
  assert res['verdict'].startswith('REPAIRED'), res['verdict']
  assert 'FRAGILITY DISCLOSURE' in res['verdict']

  # Fire-status extractor combinatorics.
  f_a = _fires(dict(ci=(0.1, 0.5), point=0.7))
  assert f_a == dict(p_rep1=True, p_rep2_material=True, harmful=False)
  f_b = _fires(dict(ci=(-0.1, 0.5), point=0.7))
  assert f_b == dict(p_rep1=False, p_rep2_material=False, harmful=False)
  f_c = _fires(dict(ci=(-0.5, -0.1), point=-0.3))
  assert f_c == dict(p_rep1=False, p_rep2_material=False, harmful=True)
  assert (f_a != f_b) and (f_b != f_c)

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
    assert res['sensitivity_excl_smoke']['fragile'] is False
    assert res['sensitivity_excl_smoke']['boot']['n_clusters'] == 31
    assert res['model']['sha256'] == msha
    # Source-pinning numerics THROUGH load_side (mutation review): the
    # wave fixtures store one-hot ghat (values 0/1), so a d_pair
    # wrongly computed from ghat would give 1.0 here, not 0.7, and a
    # sign-flipped ach_probe would give 1.4, not 0.0 — the primary
    # must come from g_all. Tolerance 1e-6: g_all round-trips float32.
    blk_p = res['primaries']['p_rep1_paired_achieved']
    assert abs(blk_p['point'] - 0.7) < 1e-6, blk_p['point']
    assert abs(res['secondary']['achieved_rep']['point'] - 0.7) < 1e-6
    assert abs(res['secondary']['achieved_probe']['point'] - 0.0) < 1e-6
    assert abs(res['secondary']['achieved_orig']['point'] - 0.0) < 1e-6
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
    # shadow-pair identity (Amendment 1): stored real_scores must
    # argmax to m_real_probe
    blk = _mk_cell(3, probe=0)
    scores = np.zeros((N_STATES, N_CANDS), np.float32)
    scores[:, 5] = 1.0  # argmax 5 != m_real_probe 0
    blk['real_scores'] = scores
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'), blk,
               CM_VERSION, run31, 31, repaired=True)
    _expect_load_trip(d, True, 'shadow-pair identity')
  with tempfile.TemporaryDirectory() as d:
    # non-finite real_scores must trip the dedicated finiteness guard,
    # not fail the maximizer identity with a misleading message
    blk = _mk_cell(3, probe=0)
    scores = blk['real_scores'].copy()
    scores[0, 0] = np.nan
    blk['real_scores'] = scores
    _write_npz(os.path.join(d, 'rep_cup_e1_seed31_late.npz'), blk,
               CM_VERSION, run31, 31, repaired=True)
    _expect_load_trip(d, True, 'non-finite ghat/real_scores')
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

  # detgate trips (amended): missing file; cross-era VALUE drift on the
  # committed side must now PASS (no cross-job float comparison);
  # episode/step mismatch fails; broken shadow-pair identity fails.
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
    # committed side with float drift AND a different plugin choice on
    # identical states: the amended gate must pass (decisive leg — the
    # parent gate quarantined exactly this).
    _write_npz(os.path.join(d_com, 'r3_cup_e1_seed31_late.npz'),
               _mk_cell(0, m_now_val=1, g_bump=1.0), LABELER_VERSION,
               run31, 31, repaired=False)
    detgate(d_rep, d_com)
    # episode/step mismatch still fails — both halves separately
    _write_npz(os.path.join(d_com, 'r3_cup_e1_seed31_late.npz'),
               _mk_cell(0, step=np.arange(N_STATES) * 25 + 26),
               LABELER_VERSION, run31, 31, repaired=False)
    try:
      detgate(d_rep, d_com)
      raise SystemExit('selfcheck FAIL: detgate step mismatch '
                       'not caught')
    except SystemExit as e:
      assert e.code == 1, e
    _write_npz(os.path.join(d_com, 'r3_cup_e1_seed31_late.npz'),
               _mk_cell(0, episode=np.ones(N_STATES, int)),
               LABELER_VERSION, run31, 31, repaired=False)
    try:
      detgate(d_rep, d_com)
      raise SystemExit('selfcheck FAIL: detgate episode mismatch '
                       'not caught')
    except SystemExit as e:
      assert e.code == 1, e
    _write_npz(os.path.join(d_com, 'r3_cup_e1_seed31_late.npz'),
               _mk_cell(0), LABELER_VERSION, run31, 31, repaired=False)
    detgate(d_rep, d_com)  # aligned again: passes
    # broken in-pass shadow-pair identity fails the amended gate
    blk = _mk_cell(3, probe=0)
    scores = np.zeros((N_STATES, N_CANDS), np.float32)
    scores[:, 5] = 1.0
    blk['real_scores'] = scores
    _write_npz(os.path.join(d_rep, 'rep_cup_e1_seed31_late.npz'), blk,
               CM_VERSION, run31, 31, repaired=True)
    try:
      detgate(d_rep, d_com)
      raise SystemExit('selfcheck FAIL: detgate shadow-pair identity '
                       'not caught')
    except SystemExit as e:
      assert e.code == 1, e

  print('selfcheck PASS (Amendment 1): estimand identity, six verdict '
        'branches (repaired incl. cross-era-drift-adjudicates decisive '
        'leg, partial, not-repairable, harmful, quarantine on step AND '
        'on episode separately, substrate-gone), within-pass primary '
        'numerics + e2e source-pinning through load_side (g_all-vs-'
        'ghat discriminating fixtures, ach_probe sign), '
        'sensitivity leg (n=31, point, genuine materiality-fragility '
        'flip + FRAGILITY DISCLOSURE append, fire-extractor '
        'combinatorics), grid trips (missing cell both sides, '
        'unregistered seed, stray rep_ file), end-to-end tempdir wave '
        'read + detgate-marker gate, model-protocol trips (sha record '
        'missing/mismatch, per-pass sha, manifest drift, LORO leakage, '
        'frozen committed-bundle digest, trainer-source sha), load '
        'guards (exact version pins both directions, dials incl. '
        'seed-0 + feature map, late-ckpt suffix, train_seed, LORO '
        'deployment, tie-tolerant chooser + shadow-pair identities, '
        'finiteness incl. dedicated ghat/real_scores guard, oracle_all '
        'identity), amended detgate (ok, missing file, cross-era value '
        'drift PASSES, step mismatch fails, episode mismatch fails, '
        'broken shadow pair fails)')


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
