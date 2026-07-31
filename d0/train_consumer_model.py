"""LORO consumer-model trainer for the competence-repair intervention.

Registered in prereg/PREREG_competence_repair_20260730.md and committed
BEFORE the trainer is ever run on real labels. Pure numpy (no
sklearn/jax/torch): per-run ridge regression from a FROZEN per-candidate
feature map ('cmfeat1', below) onto the committed R3 oracle returns
g_all[state, candidate], with strict leave-one-run-out (LORO) scope —
the model deployed on run X is trained ONLY on the OTHER 7 seeds of the
same (domain, dose) group, both maturities (14 source cells), so the
deployed chooser has never seen a single state of its own run.

Value-blindness (enforced by code, checked by the selfcheck):
- the ONLY npz fields ever read are the whitelist ALLOWED_KEYS below;
  the realized-choice and plug-in-choice columns and every per-state
  delta column of the label files are never touched (a source-scan
  selfcheck leg asserts their field names do not appear in this module);
- the output npz contains ONLY weights, intercepts, standardization
  constants, lambdas, version strings, and a training-data manifest —
  no statistic of the targets beyond the registered intercepts (each
  intercept is the fitting split's pooled target mean, a 14-cell
  cross-run aggregate, never a per-state or per-run estimand);
- stdout reports ONLY the chosen lambda per model (plus dimensions);
  the internal cross-validation rank scores are computed by the
  machinery and never printed, stored, or returned to the caller.

Feature map 'cmfeat1' (frozen; identical online and offline): see
consumer_features. Every input is a quantity d0_eval already returns at
a labeled state AND a stored field of the committed npz, so the
labeler-side chooser (d0/oracle_labels.py --consumer_model) computes
bitwise-identical features online.

Lambda selection (frozen): inner leave-one-training-run-out CV; the
score of a lambda for a held-out run is the mean over the run's two
maturity files of the per-file mean per-state Spearman rank correlation
between prediction and target across the candidate set (states where
either vector is constant are skipped; a file with no scorable state
contributes 0.0). Ties break to the LARGEST lambda (most regularized). Standardization constants come from the
fitting split only. The trainer is fully deterministic (no RNG): the
output npz is a pure function of this frozen code and the label files.

Usage:
  python -m d0.train_consumer_model --labels <committed labels dir> \
      --output <model.npz>
  python -m d0.train_consumer_model --selfcheck   (synthetic, no jax)
"""

import argparse
import glob
import hashlib
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

FEATURE_MAP_VERSION = 'cmfeat1'
TRAINER_VERSION = 'cm1'
LABELER_VERSION = 'd1fix_20260724'
LAMBDA_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1)
DOMAINS = ('cup', 'finger')
DOSES = ('e1', 'e4')
SEEDS = tuple(range(31, 39))
MATS = ('early', 'late')
N_STATES = 200
N_CANDS = 8
# Value-blind whitelist: the ONLY label-npz fields this module reads.
ALLOWED_KEYS = ('meta', 'qfull', 'cands', 'deter', 'udyn', 'g_all')
# The registered training set: the 64 committed R3 label files.
TRAIN_FILES = tuple(sorted(
    f'r3_{dom}_{dose}_seed{seed}_{mat}.npz'
    for dom in DOMAINS for dose in DOSES for seed in SEEDS for mat in MATS))


def run_key(dom, dose, seed):
  return f'r3_{dom}_{dose}_seed{seed}'


def sibling_files(dom, dose, seed):
  """The 14 registered LORO training files for one deployed run."""
  return sorted(f'r3_{dom}_{dose}_seed{s}_{mat}.npz'
                for s in SEEDS if s != seed for mat in MATS)


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


# --------------------------------------------------------------------------
# Frozen feature map (shared online/offline)
# --------------------------------------------------------------------------

def consumer_features(qfull, cands, deter, udyn):
  """Per-candidate features of ONE labeled state, frozen map 'cmfeat1'.

  Inputs exactly as d0_eval returns them online and as the committed
  npz stores them: qfull (K, M) f32, cands (M, A) f32, deter (D,)
  float16 (the stored belief latent; any input is routed through f16 so
  online and offline are bitwise identical), udyn scalar f32.

  Returns (M, F) float64 with F = 8 + A + D; columns:
    0  q_mean[m]                own-candidate head-mean Q
    1  q_std[m]                 own-candidate head-std (ddof=0)
    2  q_mean[m] - set_mean     centered head-mean
    3  q_mean[m] - set_max      margin to the set's best head-mean
    4  set_mean                 candidate-set stats (constant in m)
    5  set_std   (std over candidates of q_mean, ddof=0)
    6  set_max - set_min
    7  udyn
    8 .. 8+A-1                  the candidate action vector
    8+A .. 8+A+D-1              deter (f16 -> f32 -> f64)
  """
  qfull = np.asarray(qfull, np.float64)
  cands = np.asarray(cands, np.float64)
  det = np.asarray(deter).astype(np.float16).astype(np.float32)
  det = det.astype(np.float64)
  assert qfull.ndim == 2 and cands.ndim == 2 and det.ndim == 1, (
      qfull.shape, cands.shape, det.shape)
  m_count = cands.shape[0]
  assert qfull.shape[1] == m_count, (qfull.shape, cands.shape)
  qm = qfull.mean(0)
  qs = qfull.std(0)
  smean, sstd = float(qm.mean()), float(qm.std())
  smax, smin = float(qm.max()), float(qm.min())
  cols = [qm, qs, qm - smean, qm - smax,
          np.full(m_count, smean), np.full(m_count, sstd),
          np.full(m_count, smax - smin), np.full(m_count, float(udyn))]
  return np.column_stack(cols + [cands, np.tile(det, (m_count, 1))])


def predict(feats, w, b, mu, sd):
  """Ridge prediction on raw features (any leading shape ... x F)."""
  return ((feats - mu) / sd) @ w + b


# --------------------------------------------------------------------------
# Ridge + rank machinery (deterministic, numpy only)
# --------------------------------------------------------------------------

def _fit_ridge(x_flat, y_flat, lam):
  """Standardize on the fitting split, unpenalized intercept = target
  mean, w = solve(Z'Z/n + lam I, Z'(y - b)/n). Returns (w, b, mu, sd)."""
  mu = x_flat.mean(0)
  sd = x_flat.std(0)
  sd = np.where(sd == 0, 1.0, sd)
  z = (x_flat - mu) / sd
  n, n_feat = z.shape
  b = float(y_flat.mean())
  a = z.T @ z / n + lam * np.eye(n_feat)
  rhs = z.T @ (y_flat - b) / n
  return np.linalg.solve(a, rhs), b, mu, sd


def _rankdata(a):
  """Average-tie ranks (1-based), deterministic."""
  a = np.asarray(a, float)
  order = np.argsort(a, kind='stable')
  sa = a[order]
  r = np.empty(len(a))
  i = 0
  while i < len(a):
    j = i
    while j + 1 < len(a) and sa[j + 1] == sa[i]:
      j += 1
    r[i:j + 1] = (i + j) / 2.0 + 1.0
    i = j + 1
  ranks = np.empty(len(a))
  ranks[order] = r
  return ranks


def _spearman(a, b):
  """Spearman rho, or None if either vector is rank-constant."""
  ra, rb = _rankdata(a), _rankdata(b)
  if ra.std() == 0 or rb.std() == 0:
    return None
  ra = ra - ra.mean()
  rb = rb - rb.mean()
  return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def _per_state_rank_score(pred_2d, target_2d):
  """Mean per-state Spearman across the candidate set; skipped states
  (constant vectors) are dropped; no scorable state -> 0.0. Internal
  CV machinery only — never printed or stored."""
  scores = [s for s in (_spearman(pred_2d[i], target_2d[i])
                        for i in range(pred_2d.shape[0])) if s is not None]
  return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------
# Label loading (whitelist-enforced)
# --------------------------------------------------------------------------

class _WhitelistNpz:
  """Live enforcement of ALLOWED_KEYS: every npz access outside the
  whitelist raises, so the field whitelist is code, not documentation
  (the source-scan selfcheck leg catches literal tokens; this catches
  key-iteration or new-code regressions too)."""

  def __init__(self, z):
    self._z = z

  def __getitem__(self, key):
    assert key in ALLOWED_KEYS, (
        f'trainer read of npz field {key!r} outside ALLOWED_KEYS '
        f'{ALLOWED_KEYS} — value-blindness whitelist violated')
    return self._z[key]


def load_cell(path, n_states=N_STATES, n_cands=N_CANDS):
  """One label file -> (X (S, M, F), Y (S, M)). Reads ONLY ALLOWED_KEYS
  (enforced by _WhitelistNpz, not just documented)."""
  z = _WhitelistNpz(np.load(path, allow_pickle=True))
  name = os.path.basename(path)
  md = json.loads(str(z['meta']))
  assert md.get('labeler_version') == LABELER_VERSION, (
      f'{name}: labeler_version {md.get("labeler_version")!r} != '
      f'{LABELER_VERSION!r} (exact pin: training data must be the '
      'committed base-instrument labels, never an extension pass)')
  assert md.get('states') == n_states and md.get('actions') == n_cands, (
      f'{name}: dials states={md.get("states")}/actions={md.get("actions")} '
      f'!= registered {n_states}/{n_cands}')
  assert md.get('seed') == 0, (
      f'{name}: labeler seed {md.get("seed")} != registered 0')
  qfull = np.asarray(z['qfull'], np.float32)
  cands = np.asarray(z['cands'], np.float32)
  deter = np.asarray(z['deter'])
  udyn = np.asarray(z['udyn'], np.float32)
  y = np.asarray(z['g_all'], np.float64)
  assert qfull.shape[0] == n_states and qfull.shape[2] == n_cands, (
      f'{name}: qfull shape {qfull.shape}')
  assert cands.shape[:2] == (n_states, n_cands), (
      f'{name}: cands shape {cands.shape}')
  assert deter.shape[0] == n_states and udyn.shape == (n_states,), (
      f'{name}: deter/udyn shapes {deter.shape}/{udyn.shape}')
  assert y.shape == (n_states, n_cands) and np.isfinite(y).all(), (
      f'{name}: g_all shape {y.shape} or non-finite (oracle_all pass '
      'required)')
  x = np.stack([consumer_features(qfull[i], cands[i], deter[i],
                                  float(udyn[i]))
                for i in range(n_states)], 0)
  return x, y


# --------------------------------------------------------------------------
# LORO training
# --------------------------------------------------------------------------

def _flat(cells, seeds_used):
  xs = [cells[(s, mat)][0].reshape(-1, cells[(s, mat)][0].shape[-1])
        for s in seeds_used for mat in MATS]
  ys = [cells[(s, mat)][1].reshape(-1) for s in seeds_used for mat in MATS]
  return np.concatenate(xs, 0), np.concatenate(ys, 0)


def train_target(cells, target_seed, seeds=SEEDS, grid=LAMBDA_GRID):
  """One deployed model: LORO over `seeds` excluding target_seed, with
  inner leave-one-training-run-out lambda CV. Returns dict with
  w/b/mu/sd/lam. The CV scores stay internal."""
  train_seeds = [s for s in seeds if s != target_seed]
  fold_cache = {}
  for held in train_seeds:
    fit_seeds = [s for s in train_seeds if s != held]
    x_flat, y_flat = _flat(cells, fit_seeds)
    # Gram/moments do not depend on lambda: factor them per fold.
    mu = x_flat.mean(0)
    sd = np.where(x_flat.std(0) == 0, 1.0, x_flat.std(0))
    z = (x_flat - mu) / sd
    n = z.shape[0]
    b = float(y_flat.mean())
    fold_cache[held] = (z.T @ z / n, z.T @ (y_flat - b) / n, b, mu, sd)
  best = None
  n_feat = cells[(train_seeds[0], MATS[0])][0].shape[-1]
  for lam in grid:
    scores = []
    for held in train_seeds:
      gram, rhs, b, mu, sd = fold_cache[held]
      w = np.linalg.solve(gram + lam * np.eye(n_feat), rhs)
      per_mat = []
      for mat in MATS:
        x3, y2 = cells[(held, mat)]
        pred = predict(x3, w, b, mu, sd)
        per_mat.append(_per_state_rank_score(pred, y2))
      scores.append(float(np.mean(per_mat)))
    mean_score = float(np.mean(scores))
    if (best is None or mean_score > best[0]
        or (mean_score == best[0] and lam > best[1])):
      best = (mean_score, lam)
  lam = best[1]
  x_flat, y_flat = _flat(cells, train_seeds)
  w, b, mu, sd = _fit_ridge(x_flat, y_flat, lam)
  return dict(w=w, b=b, mu=mu, sd=sd, lam=float(lam))


def train_all(labels_dir, output, domains=DOMAINS, doses=DOSES,
              seeds=SEEDS, n_states=N_STATES, n_cands=N_CANDS):
  """Train the full registered model set (one LORO model per run) and
  write the single output npz with the training-data manifest."""
  missing = [n for n in TRAIN_FILES
             if not os.path.exists(os.path.join(labels_dir, n))]
  assert not missing, (
      f'{len(missing)} registered training files missing under '
      f'{labels_dir}, e.g. ' + ', '.join(missing[:4]))
  manifest_files = {n: sha256_file(os.path.join(labels_dir, n))
                    for n in TRAIN_FILES}
  arrays = {}
  per_model = {}
  runs = []
  n_feat = None
  for dom in domains:
    for dose in doses:
      cells = {}
      for seed in seeds:
        for mat in MATS:
          name = f'r3_{dom}_{dose}_seed{seed}_{mat}.npz'
          cells[(seed, mat)] = load_cell(
              os.path.join(labels_dir, name), n_states, n_cands)
      n_feat = cells[(seeds[0], MATS[0])][0].shape[-1]
      for seed in seeds:
        run = run_key(dom, dose, seed)
        model = train_target(cells, seed, seeds=seeds)
        arrays[f'w_{run}'] = model['w']
        arrays[f'b_{run}'] = np.float64(model['b'])
        arrays[f'mu_{run}'] = model['mu']
        arrays[f'sd_{run}'] = model['sd']
        arrays[f'lambda_{run}'] = np.float64(model['lam'])
        per_model[run] = sibling_files(dom, dose, seed)
        runs.append(run)
        print(f'{run}: lambda={model["lam"]:g} '
              f'(F={n_feat}, 14 training cells; standardization '
              'constants stored in the output npz)')
      del cells
  manifest = dict(
      trainer_version=TRAINER_VERSION,
      feature_map_version=FEATURE_MAP_VERSION,
      lambda_grid=list(LAMBDA_GRID),
      n_features=int(n_feat),
      # sha256 of THIS module's source at training time: the reader
      # asserts it equals the checked-out file, so a post-freeze locally
      # edited trainer (same version string) cannot produce an accepted
      # model — code provenance, not just data provenance.
      trainer_source_sha256=sha256_file(os.path.abspath(__file__)),
      files=manifest_files,
      per_model=per_model)
  os.makedirs(os.path.dirname(os.path.abspath(output)) or '.',
              exist_ok=True)
  np.savez_compressed(
      output,
      feature_map_version=FEATURE_MAP_VERSION,
      trainer_version=TRAINER_VERSION,
      lambda_grid=np.asarray(LAMBDA_GRID, np.float64),
      runs=np.asarray(runs),
      manifest=json.dumps(manifest),
      **arrays)
  print(f'wrote {output}: {len(runs)} LORO models, F={n_feat} '
        f'(feature map {FEATURE_MAP_VERSION}, trainer {TRAINER_VERSION})')
  return output


# --------------------------------------------------------------------------
# Deployment side (imported by d0/oracle_labels.py)
# --------------------------------------------------------------------------

def load_consumer_model(path):
  z = np.load(path, allow_pickle=False)
  fmv = str(z['feature_map_version'])
  assert fmv == FEATURE_MAP_VERSION, (
      f'consumer model {path}: feature_map_version {fmv!r} != '
      f'{FEATURE_MAP_VERSION!r}')
  runs = [str(r) for r in z['runs']]
  models = {}
  for r in runs:
    models[r] = dict(
        w=np.asarray(z[f'w_{r}'], np.float64),
        b=float(z[f'b_{r}']),
        mu=np.asarray(z[f'mu_{r}'], np.float64),
        sd=np.asarray(z[f'sd_{r}'], np.float64),
        lam=float(z[f'lambda_{r}']))
  return dict(feature_map_version=fmv,
              trainer_version=str(z['trainer_version']),
              runs=runs, models=models,
              manifest=json.loads(str(z['manifest'])))


def make_chooser(path, run):
  """(chooser, info) for deploying the LORO model of `run`. The chooser
  maps one labeled state's (qfull, cands, deter, udyn) to
  (argmax ghat, ghat f32) — pure numpy, no env/policy/RNG use."""
  model = load_consumer_model(path)
  assert run in model['models'], (
      f'consumer model {path} has no weights for run {run!r} '
      f'(runs: {model["runs"][:4]}...)')
  params = model['models'][run]
  w, b, mu, sd = params['w'], params['b'], params['mu'], params['sd']

  def chooser(qfull, cands, deter, udyn):
    feats = consumer_features(qfull, cands, deter, udyn)
    assert feats.shape[1] == w.shape[0], (
        f'feature width {feats.shape[1]} != model width {w.shape[0]} '
        f'(feature map {FEATURE_MAP_VERSION})')
    ghat = predict(feats, w, b, mu, sd)
    # Cast FIRST, argmax the stored array: the row stores ghat as f32,
    # and the reader adjudicates chooser identity against that array —
    # an f64 argmax could disagree after f32 rounding collapses a
    # near-tie, crashing the ONE read on an honest pass.
    ghat32 = np.asarray(ghat, np.float32)
    return int(np.argmax(ghat32)), ghat32

  info = dict(sha256=sha256_file(path), run=run,
              feature_map_version=model['feature_map_version'],
              trainer_version=model['trainer_version'])
  return chooser, info


# --------------------------------------------------------------------------
# Selfcheck (synthetic fixtures only; no real label path is ever named)
# --------------------------------------------------------------------------

def _synth_labels(d, w_true, b_true, noise=0.05, k_heads=3, a_dim=2,
                  d_dim=4, corrupt=()):
  """Write the full registered 64-file grid with a planted linear
  target y = feats @ w_true + b_true + noise; per-cell rng keyed by the
  file name so regeneration is deterministic. Cells in `corrupt` get a
  scrambled target (LORO-integrity leg)."""
  for dom in DOMAINS:
    for dose in DOSES:
      for seed in SEEDS:
        for mat in MATS:
          name = f'r3_{dom}_{dose}_seed{seed}_{mat}.npz'
          # hashlib, NOT hash(): str hash is salted per process, and a
          # process-dependent fixture makes the frozen "selfcheck PASS"
          # claim irreproducible across shells (verified failing under
          # PYTHONHASHSEED=99 with the old keying).
          rng = np.random.default_rng(int.from_bytes(
              hashlib.sha256(name.encode()).digest()[:4], 'little'))
          qfull = rng.standard_normal(
              (N_STATES, k_heads, N_CANDS)).astype(np.float32)
          cands = rng.standard_normal(
              (N_STATES, N_CANDS, a_dim)).astype(np.float32)
          deter = rng.standard_normal(
              (N_STATES, d_dim)).astype(np.float16)
          udyn = rng.random(N_STATES).astype(np.float32)
          y = np.empty((N_STATES, N_CANDS))
          for i in range(N_STATES):
            feats = consumer_features(qfull[i], cands[i], deter[i],
                                      float(udyn[i]))
            y[i] = feats @ w_true + b_true
          y += noise * rng.standard_normal(y.shape)
          if (dom, dose, seed, mat) in corrupt:
            y = -5.0 * y[:, ::-1] + 7.0
          meta = json.dumps(dict(labeler_version=LABELER_VERSION,
                                 states=N_STATES, actions=N_CANDS,
                                 seed=0))
          np.savez(os.path.join(d, name), meta=meta, qfull=qfull,
                   cands=cands, deter=deter, udyn=udyn,
                   g_all=y.astype(np.float32))


def selfcheck():
  import tempfile

  # Leg 1: feature-map hand case (layout frozen by this assert).
  qf = np.array([[1.0, 3.0], [3.0, 1.0]], np.float32)      # K=2, M=2
  ca = np.array([[0.5], [-0.5]], np.float32)               # A=1
  de = np.array([2.0, -1.0], np.float16)                   # D=2
  feats = consumer_features(qf, ca, de, 0.25)
  # q_mean = [2, 2], q_std = [1, 1]; set: mean 2, std 0, max-min 0
  want0 = [2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.25, 0.5, 2.0, -1.0]
  want1 = [2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.25, -0.5, 2.0, -1.0]
  assert feats.shape == (2, 11), feats.shape
  assert np.allclose(feats[0], want0) and np.allclose(feats[1], want1), feats
  # f16 routing: f32 input goes through f16, matching the stored field
  f32in = consumer_features(qf, ca, de.astype(np.float32), 0.25)
  assert np.array_equal(f32in, feats)

  # Leg 2: rank machinery (ties, constant vectors).
  assert np.allclose(_rankdata([3.0, 1.0, 2.0]), [3, 1, 2])
  assert np.allclose(_rankdata([1.0, 1.0, 2.0]), [1.5, 1.5, 3])
  assert abs(_spearman([1, 2, 3], [10, 20, 30]) - 1.0) < 1e-12
  assert abs(_spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12
  assert _spearman([1, 1, 1], [1, 2, 3]) is None
  assert _per_state_rank_score(np.ones((2, 3)), np.ones((2, 3))) == 0.0

  # Leg 3: ridge identity on a hand case (exact fit at lam -> 0).
  rng = np.random.default_rng(0)
  x = rng.standard_normal((500, 4))
  y = x @ np.array([1.0, -2.0, 0.5, 0.0]) + 3.0
  w, b, mu, sd = _fit_ridge(x, y, 1e-10)
  assert np.allclose(predict(x, w, b, mu, sd), y, atol=1e-6)

  # Plant on columns OUTSIDE the map's designed exact collinearity
  # (col0 = col2 + col4 by construction, so weights inside that group
  # are individually unidentifiable even though prediction is exact).
  w_true = np.zeros(11)
  w_true[1] = 2.0    # q_std
  w_true[8] = 1.5    # candidate action dim 0
  w_true[10] = -3.0  # deter dim 1
  b_true = 0.3

  with tempfile.TemporaryDirectory() as d:
    _synth_labels(d, w_true, b_true, a_dim=1, d_dim=2)
    out = os.path.join(d, 'model.npz')
    train_all(d, out)
    model = load_consumer_model(out)

    # Leg 4: planted linear signal recovered through LORO. The deployed
    # run's model never saw that run; per-state argmax of ghat must
    # still recover the planted argmax almost everywhere.
    run = run_key('cup', 'e1', 31)
    chooser, info = make_chooser(out, run)
    assert info['run'] == run and len(info['sha256']) == 64
    assert info['sha256'] == sha256_file(out)
    assert info['feature_map_version'] == FEATURE_MAP_VERSION
    z = np.load(os.path.join(d, 'r3_cup_e1_seed31_late.npz'),
                allow_pickle=True)
    hits, total = 0, 0
    for i in range(0, N_STATES, 5):
      feats = consumer_features(z['qfull'][i], z['cands'][i],
                                z['deter'][i], float(z['udyn'][i]))
      truth = int(np.argmax(feats @ w_true + b_true))
      m_hat, ghat = chooser(z['qfull'][i], z['cands'][i], z['deter'][i],
                            float(z['udyn'][i]))
      assert ghat.shape == (N_CANDS,) and ghat.dtype == np.float32
      assert m_hat == int(np.argmax(ghat))
      hits += int(m_hat == truth)
      total += 1
    assert hits / total >= 0.9, f'LORO recovery too weak: {hits}/{total}'
    # planted coordinates dominate the recovered z-space weights
    params = model['models'][run]
    w_raw = params['w'] / params['sd']  # back to raw-feature scale
    top = set(np.argsort(-np.abs(w_raw))[:3])
    assert top == {1, 8, 10}, (top, w_raw)
    # Ridge shrinks uniformly (the CV score is rank-based, so a
    # shrinking lambda can win); assert the planted RATIOS, plus a
    # bounded common shrink factor and near-zero off-plant weights.
    factor = w_raw[1] / 2.0
    assert 0.8 <= factor <= 1.02, (factor, w_raw)
    assert abs(w_raw[8] / 1.5 - factor) < 0.03, (factor, w_raw)
    assert abs(w_raw[10] / -3.0 - factor) < 0.03, (factor, w_raw)
    others = [i for i in range(11) if i not in (1, 8, 10)]
    assert np.max(np.abs(w_raw[others])) < 0.1, w_raw

    # Leg 5: LORO integrity — corrupting run X's OWN files must leave
    # X's deployed model bit-identical (X is excluded from its own
    # training data), while sibling models change.
    before = {r: {k: np.copy(v) if isinstance(v, np.ndarray) else v
                  for k, v in model['models'][r].items()}
              for r in (run_key('cup', 'e1', 33), run_key('cup', 'e1', 31))}
    corrupt = {('cup', 'e1', 33, mat) for mat in MATS}
    _synth_labels(d, w_true, b_true, a_dim=1, d_dim=2, corrupt=corrupt)
    out2 = os.path.join(d, 'model2.npz')
    train_all(d, out2)
    model2 = load_consumer_model(out2)
    r33 = run_key('cup', 'e1', 33)
    for k in ('w', 'mu', 'sd'):
      assert np.array_equal(model2['models'][r33][k], before[r33][k]), k
    assert model2['models'][r33]['b'] == before[r33]['b']
    assert model2['models'][r33]['lam'] == before[r33]['lam']
    r31 = run_key('cup', 'e1', 31)
    assert not np.allclose(model2['models'][r31]['w'], before[r31]['w']), (
        'corruption of seed 33 did not reach sibling models — LORO '
        'fold structure broken')

    # Leg 6: no-estimand — the output contains nothing but weights,
    # standardization, lambdas, versions, and the manifest; no key
    # names any decision or value summary field.
    zm = np.load(out, allow_pickle=False)
    fixed = {'feature_map_version', 'trainer_version', 'lambda_grid',
             'runs', 'manifest'}
    for key in zm.files:
      if key in fixed:
        continue
      assert key.split('_')[0] in ('w', 'b', 'mu', 'sd', 'lambda'), key
      for tok in ('ach', 'gap', 'opp', 'delta', 'now', 'real', 'imag'):
        assert tok not in key, (key, tok)
    # ... and this module's source never touches the label files'
    # choice/delta columns (tokens built by concat to avoid self-match).
    src = open(os.path.abspath(__file__)).read()
    for tok in ('g_' + 'now', 'm_' + 'real', 'm_' + 'now', 'm_' + 'imag',
                'g_' + 'real', 'g_' + 'imag', 'delta' + '_real',
                'delta' + '_imag', 'real_' + 'scores'):
      assert tok not in src, f'forbidden label field in trainer source: {tok}'

    # Leg 7: manifest integrity — 64 files with real sha256, per-model
    # lists exclude the deployed run's own two files.
    man = model['manifest']
    assert sorted(man['files']) == list(TRAIN_FILES)
    assert man['files']['r3_cup_e1_seed31_late.npz'] == sha256_file(
        os.path.join(d, 'r3_cup_e1_seed31_late.npz'))
    assert len(man['per_model'][run]) == 14
    assert not any('seed31' in n for n in man['per_model'][run])
    assert man['feature_map_version'] == FEATURE_MAP_VERSION
    assert man['n_features'] == 11

    # Leg 8: guards — unknown run, wrong feature width, missing file,
    # wrong labeler version.
    try:
      make_chooser(out, 'r3_cup_e1_seed99')
      raise SystemExit('selfcheck FAIL: unknown run not caught')
    except AssertionError as e:
      assert 'no weights' in str(e), e
    np.savez(os.path.join(d, 'narrow.npz'),
             feature_map_version=FEATURE_MAP_VERSION,
             trainer_version=TRAINER_VERSION,
             lambda_grid=np.asarray(LAMBDA_GRID),
             runs=np.asarray([run]), manifest=json.dumps({}),
             **{f'w_{run}': np.zeros(5), f'b_{run}': 0.0,
                f'mu_{run}': np.zeros(5), f'sd_{run}': np.ones(5),
                f'lambda_{run}': 1.0})
    ch_bad, _ = make_chooser(os.path.join(d, 'narrow.npz'), run)
    try:
      ch_bad(qf, ca, de, 0.25)
      raise SystemExit('selfcheck FAIL: feature-width mismatch not caught')
    except AssertionError as e:
      assert 'feature width' in str(e), e
    os.remove(os.path.join(d, 'r3_finger_e4_seed38_late.npz'))
    try:
      train_all(d, os.path.join(d, 'model3.npz'))
      raise SystemExit('selfcheck FAIL: missing training file not caught')
    except AssertionError as e:
      assert 'missing' in str(e), e

  # Leg 9: wrong labeler version trips load_cell; version pin is exact.
  with tempfile.TemporaryDirectory() as d:
    bad = os.path.join(d, 'bad.npz')
    np.savez(bad, meta=json.dumps(dict(labeler_version='d1fix_20260724_xc1',
                                       states=N_STATES, actions=N_CANDS,
                                       seed=0)),
             qfull=np.zeros((N_STATES, 2, N_CANDS), np.float32),
             cands=np.zeros((N_STATES, N_CANDS, 1), np.float32),
             deter=np.zeros((N_STATES, 2), np.float16),
             udyn=np.zeros(N_STATES, np.float32),
             g_all=np.zeros((N_STATES, N_CANDS), np.float32))
    try:
      load_cell(bad)
      raise SystemExit('selfcheck FAIL: extension-pass labels not caught')
    except AssertionError as e:
      assert 'exact pin' in str(e), e

  # Leg 10: lambda tie-break — constant target scores 0.0 everywhere,
  # so the LARGEST grid lambda must be selected.
  cells = {}
  rng = np.random.default_rng(1)
  for s in SEEDS:
    for mat in MATS:
      x3 = rng.standard_normal((6, N_CANDS, 11))
      cells[(s, mat)] = (x3, np.full((6, N_CANDS), 2.5))
  model = train_target(cells, SEEDS[0])
  assert model['lam'] == LAMBDA_GRID[-1], model['lam']

  print('trainer selfcheck PASS: feature-map hand case + f16 routing, '
        'rank machinery (ties/constant), ridge identity, planted-signal '
        'LORO recovery + top-weight identification, LORO-integrity '
        '(own-run corruption leaves deployed model bit-identical, '
        'reaches siblings), no-estimand (output keys + source scan), '
        'manifest integrity (64 shas, per-model exclusion), guards '
        '(unknown run, feature width, missing file, extension-version '
        'labels), lambda tie-break to largest')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--labels', help='committed R3 label dir (the 64 '
                                  'registered r3_*_{early,late}.npz)')
  p.add_argument('--output', help='output model npz path')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not args.labels or not args.output:
    p.error('--labels and --output required (or --selfcheck)')
  train_all(args.labels, args.output)


if __name__ == '__main__':
  main()
