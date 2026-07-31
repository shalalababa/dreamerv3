"""Frozen read for the R3 candidate-aware actionability ladder (Paper 2).

Registered in prereg/PREREG_r3_ladder_20260730.md and committed BEFORE
any per-state feature->target statistic on the corrected R3 labels
exists anywhere; metric amended pre-read by
prereg/PREREG_r3_ladder_amend1_20260731.md (within-run Spearman — see
the metric section below), also committed before the read. Successor of the instrument-invalidated D1 ladder
(PREREG_d1_ladder_20260723.md): that invalidation was entirely in the
LABELS (stale-carry + policy-RNG labeler defects, repaired
d1fix_20260724), not in the ladder machinery, so the machinery carries
over and is re-registered here on the committed corrected R3 label
campaign — plus one NEW rung (L4, candidate-aware). Answers the residue
of 11-Jul editorial objection 4: which observable features predict the
per-state implementation gap?

Registered-descriptive, NON-decisional. Targets per labeled state (the
r3_read.cell_stats estimands taken per state instead of as cell means;
identity asserted in the selfcheck):

  opportunity = max_m G[m] - G[m_now]                 (from g_all, g_now)
  gap         = opportunity - (G[m_real] - G[m_now])  (achieved deficit)

Cells: {early, late} x {e1, e4}, domains POOLED — 16 run clusters per
cell (cluster = one training run at one maturity = one label file).
Rungs per cell x target:

  L0  constants: always (pooled mean, cluster CI) vs never (0)
  L1  LORO OLS on F0 (the 5 d0.signals scalars)
  L2  LORO OLS on F1 = F0 + udyn + logdens + udyn_resid
  L3  nested-LORO ridge on [deter (512, standardized) + F1]
  L4  nested-LORO ridge on [deter + F1 + G], G = the 7 frozen
      candidate-geometry / Q-dispersion features (the NEW rung)

Metric per rung (AMENDED 31 Jul by PREREG_r3_ladder_amend1_20260731,
committed before the read): mean per-run held-out Spearman +
run-resampled percentile bootstrap CI (B=10K, default_rng(0)). Ranking
WITHIN runs removes run-/domain-level target structure and the LORO
intercept artifact, which the independent post-freeze review showed
make the parent's pooled Spearman a biased estimate of per-state
legibility under the between-run heterogeneity committed as true of
this data; the pooled statistic is carried point-only as a
non-evidential pooled_diagnostic. Runs with rank-constant targets
(all-floor label files) are excluded from the mean and counted in
n_nan_runs. The [L4 - L3] contrast is bootstrap-PAIRED: one joint pass
resamples clusters once per replicate and scores both rungs' per-run
rho means on the same draw.

Registered NON-decisional heuristics (resource consequence only: they
inform TARGETING of the competence-repair intervention, sibling
same-day registration PREREG_competence_repair_20260730.md; no outcome
unfreezes the 24+24, revives Route A, or changes any registered
verdict):

  COMPRESSION SIGNATURE  in (cell, target) iff L3 Spearman >= 0.2 with
                         cluster CI > 0 while L2 Spearman < 0.1.
  CANDIDATE SIGNATURE    in (cell, target) iff the paired [L4 - L3]
                         contrast >= 0.1 with paired cluster CI > 0
                         AND L4 Spearman >= 0.2 with cluster CI > 0.

Usage:
  python -m analysis.r3_ladder_read --labels <dir> --output <dir>
  python -m analysis.r3_ladder_read --selfcheck
      (five full-grid legs at the untrimmed B=10K code path plus the
      amend1 p=512 ridge leg, nothing patched; the amended bootstrap is
      far cheaper than the parent's pooled one, so this stays ~5-15 min)
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np
from scipy import stats

from analysis.r3_read import (
    B_BOOT, RNG_SEED, LABELER_VERSION, FILE_RE, EXPECT_SEEDS, DOSES, MATS,
    cell_stats, validate_arrays, _cluster_boot)
from d0 import signals as d0_signals

# Pin the shared-frozen-reader import surface: a future amendment to
# r3_read must not silently change this reader's registered grid,
# bootstrap, labeler pin, or filename contract.
assert B_BOOT == 10_000 and RNG_SEED == 0, (B_BOOT, RNG_SEED)
assert LABELER_VERSION == 'd1fix_20260724', LABELER_VERSION
assert EXPECT_SEEDS == tuple(range(31, 39)), EXPECT_SEEDS
assert DOSES == ('e1', 'e4') and MATS == ('early', 'late'), (DOSES, MATS)
assert FILE_RE.pattern == (
    r'^r3_(cup|finger|reacher)_(e1|e4)_seed(\d+)_(early|late)\.npz$'), (
    FILE_RE.pattern)

CORE_DOMAINS = ('cup', 'finger')
TARGETS = ('opp', 'gap')
N_STATES = 200
K_HEADS = 5
DETER_DIM = 512
LAMBDAS = (1e1, 1e2, 1e3, 1e4)
KNN_K = 10
KNN_EXCLUDE = 10
SIG_L3_MIN = 0.2   # compression: L3 rho at/above this with CI > 0 ...
SIG_L2_MAX = 0.1   # ... while L2 rho below this (PREREG_d1_ladder values)
CAND_DELTA_MIN = 0.1   # candidate: paired [L4-L3] at/above this, CI > 0 ...
CAND_L4_MIN = 0.2      # ... and L4 itself at/above this with CI > 0
# Registered dial identity, enforced from every file's meta — a
# mis-dialed manual retry of one pass trips instead of silently biasing.
DIALS = dict(states=200, horizon=100, label_every=25, seed=0, rollouts=16,
             ref_stride=5, mass_scale=1.0, behavior_checkpoint='', actions=8)

# Frozen byte-pin of the committed R3 label bundle (sha256 over
# 'name:sha256(file)\n' of the 64 registered files in sorted order,
# computed at freeze from local_results/r3_competence_20260729_105146/
# labels; byte-pinned in git at the read commit 4f74b187). The SAME
# constant is frozen in analysis/repair_read.py — the two reads consume
# the same substrate. read() recomputes it from --labels: "the
# committed labels" is a machine claim, not operator trust.
COMMITTED_LABELS_DIGEST = (
    '02efe6c5bae641c1f24055ffb4791e37d1b636bb7304f833777c5bf017f2798d')

# 31-Jul amendment, registered BEFORE the one read (trigger: independent
# post-freeze adversarial review; see the amendment file). The rung
# metric is the mean per-run held-out Spearman; the parent's pooled
# Spearman is carried point-only as a non-evidential diagnostic.
AMENDMENT_ID = 'PREREG_r3_ladder_amend1_20260731'


def labels_digest(labels_dir):
  # Same rule as the freeze-time computation (and repair_read's): ALL
  # non-smoke npz in sorted name order — a stray extra file changes the
  # digest and trips, which is the point.
  files = sorted(
      os.path.basename(p)
      for p in glob.glob(os.path.join(labels_dir, '*.npz'))
      if not os.path.basename(p).endswith('_smoke.npz'))
  dg = hashlib.sha256()
  for name in files:
    with open(os.path.join(labels_dir, name), 'rb') as f:
      sha = hashlib.sha256(f.read()).hexdigest()
    dg.update(f'{name}:{sha}\n'.encode())
  return dg.hexdigest()

# F0 = the five Gate-D1 signal scalars. Restated locally (not imported
# from analysis.gate_d1_read / gate_d1_r2_read: those modules carry the
# invalidated program's loaders and decision thresholds — see prereg
# "reuse decisions") and pinned against d0.signals.compute's actual
# output keys at import time.
FEATURES_F0 = ('uq', 'aflip', 'evpi_plugin', 'evpi_split', 'gap')
assert set(d0_signals.compute(np.zeros((2, 3, 4)))) == set(FEATURES_F0)
FEATURES_F1 = FEATURES_F0 + ('udyn', 'logdens', 'udyn_resid')
# The 7 frozen candidate-geometry / Q-dispersion features (rung L4).
# The head-mean top-2 margin is NOT duplicated here: it is already in
# the L4 stack as F0's 'gap' (d0.signals.advantage_gap).
FEATURES_G = ('q_cand_std', 'q_cand_range', 'hspread_argmax',
              'hspread_mean', 'act_pd_mean', 'act_pd_max',
              'act_greedy_dist')


def per_state_targets(g_all, g_now, m_real):
  """The two registered target families, per labeled state. Definitions
  match analysis.r3_read.cell_stats exactly (means asserted equal in the
  selfcheck)."""
  g_all = np.asarray(g_all, float)
  g_now = np.asarray(g_now, float)
  m_real = np.asarray(m_real, int)
  opp = np.nanmax(g_all, 1) - g_now
  ach = g_all[np.arange(len(m_real)), m_real] - g_now
  return dict(opp=opp, gap=opp - ach)


def knn_logdens(run, k=KNN_K, exclude=KNN_EXCLUDE):
  """-ln(mean distance to k nearest base-trajectory latents), per labeled
  state; reference standardized per run; same-episode points within
  +-exclude steps removed (the labeled state's own base twin). Local
  reimplementation of the gate_d1_r2_read definition — that module's
  loader pins the DEFECTIVE labeler and must not be imported here."""
  ref = np.asarray(run['ref_deter'], np.float64)
  mu, sd = ref.mean(0), ref.std(0)
  sd[sd == 0] = 1.0
  refz = (ref - mu) / sd
  lab = (np.asarray(run['deter'], np.float64) - mu) / sd
  out = np.empty(len(lab))
  for i in range(len(lab)):
    d = np.linalg.norm(refz - lab[i], axis=1)
    mask = ((run['ref_episode'] == run['episode'][i])
            & (np.abs(run['ref_step'] - run['step'][i]) <= exclude))
    d = d[~mask]
    kk = min(k, len(d))
    out[i] = -np.log(np.sort(d)[:kk].mean() + 1e-8)
  return out


def cand_features(qfull, cands, m_now):
  """The frozen G set. Inputs: qfull (N,K,M), cands (N,M,A), m_now (N,).
  Only critic/policy quantities enter — never g_all/g_now/m_real (no
  target leakage; m_real is part of the gap estimand and is excluded)."""
  qfull = np.asarray(qfull, np.float64)
  cands = np.asarray(cands, np.float64)
  m_now = np.asarray(m_now, int)
  qmean = qfull.mean(1)              # (N, M) head-mean Q per candidate
  hstd = qfull.std(1, ddof=1)        # (N, M) across-head spread
  amax = qmean.argmax(1)             # greedy candidate index
  idx = np.arange(len(qmean))
  iu = np.triu_indices(cands.shape[1], 1)
  pd = np.linalg.norm(
      cands[:, :, None, :] - cands[:, None, :, :], axis=3)[:, iu[0], iu[1]]
  return dict(
      q_cand_std=qmean.std(1, ddof=1),
      q_cand_range=qmean.max(1) - qmean.min(1),
      hspread_argmax=hstd[idx, amax],
      hspread_mean=hstd.mean(1),
      act_pd_mean=pd.mean(1),
      act_pd_max=pd.max(1),
      act_greedy_dist=np.linalg.norm(
          cands[idx, amax] - cands[idx, m_now], axis=1))


def features_all(run):
  f = d0_signals.compute(np.asarray(run['qfull'], np.float64))
  ld = knn_logdens(run)
  A = np.column_stack([np.ones(len(ld)), ld])
  w, *_ = np.linalg.lstsq(A, np.asarray(run['udyn'], np.float64),
                          rcond=None)
  f.update(udyn=np.asarray(run['udyn'], np.float64), logdens=ld,
           udyn_resid=np.asarray(run['udyn'], np.float64) - A @ w)
  f.update(cand_features(run['qfull'], run['cands'], run['m_now']))
  return f


# --------------------------------------------------------------------------
# cross-fitting (LORO OLS; nested-LORO ridge — d1_ladder machinery)
# --------------------------------------------------------------------------

def crossfit_ols(X, y):
  preds = {}
  keys = sorted(X)
  for held in keys:
    tr = [k for k in keys if k != held]
    Xtr = np.concatenate([X[k] for k in tr], 0)
    ytr = np.concatenate([y[k] for k in tr], 0)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    Ah = np.column_stack([np.ones(len(X[held])), (X[held] - mu) / sd])
    preds[held] = Ah @ w
  return preds


def _ridge_fit(Xtr, ytr, lam):
  mu, sd = Xtr.mean(0), Xtr.std(0)
  sd[sd == 0] = 1.0
  Z = (Xtr - mu) / sd
  ym = ytr.mean()
  p = Z.shape[1]
  w = np.linalg.solve(Z.T @ Z + lam * np.eye(p), Z.T @ (ytr - ym))
  return mu, sd, ym, w


def _ridge_pred(X, fit):
  mu, sd, ym, w = fit
  return ((X - mu) / sd) @ w + ym


def crossfit_ridge(X, y):
  """Nested LORO ridge: lambda chosen per outer fold by inner
  leave-one-training-run-out mean Spearman (never sees the held run)."""
  preds, chosen = {}, {}
  keys = sorted(X)
  for held in keys:
    tr = [k for k in keys if k != held]
    best_lam, best_score = None, -np.inf
    for lam in LAMBDAS:
      scores = []
      for inner in tr:
        itr = [k for k in tr if k != inner]
        fit = _ridge_fit(np.concatenate([X[k] for k in itr], 0),
                         np.concatenate([y[k] for k in itr], 0), lam)
        rho = _rho(_ridge_pred(X[inner], fit), y[inner])
        scores.append(0.0 if np.isnan(rho) else rho)
      score = float(np.mean(scores))
      if score > best_score:
        best_lam, best_score = lam, score
    fit = _ridge_fit(np.concatenate([X[k] for k in tr], 0),
                     np.concatenate([y[k] for k in tr], 0), best_lam)
    preds[held] = _ridge_pred(X[held], fit)
    chosen[held] = best_lam
  return preds, chosen


# --------------------------------------------------------------------------
# metrics (Spearman + run-clustered bootstrap; minimal reimplementation
# of gate_d1_read.criteria's spearman leg — see prereg "reuse decisions")
# --------------------------------------------------------------------------

def _rho(p, y):
  """Spearman rho via ranks (identical to scipy.stats.spearmanr —
  asserted in the selfcheck; nan when either side is constant)."""
  rp = stats.rankdata(p)
  ry = stats.rankdata(y)
  rp = rp - rp.mean()
  ry = ry - ry.mean()
  den = np.sqrt((rp @ rp) * (ry @ ry))
  if den == 0:
    return float('nan')
  return float((rp @ ry) / den)


def _run_rhos(preds, y):
  """Held-out Spearman computed WITHIN each run (nan when a run's
  targets are rank-constant, i.e. an all-floor label file)."""
  return {k: _rho(np.asarray(preds[k], np.float64),
                  np.asarray(y[k], np.float64)) for k in sorted(y)}


def _nanmean(a):
  a = a[~np.isnan(a)]
  return float(a.mean()) if len(a) else float('nan')


def _pooled_rho(preds, y):
  keys = sorted(y)
  return _rho(
      np.concatenate([np.asarray(preds[k], np.float64) for k in keys]),
      np.concatenate([np.asarray(y[k], np.float64) for k in keys]))


def rho_boot(preds, y, b=B_BOOT, seed=RNG_SEED):
  """AMENDED metric (PREREG_r3_ladder_amend1_20260731): mean per-run
  held-out Spearman with run-resampled percentile CI. Ranking WITHIN
  runs removes run-/domain-level target structure and the LORO
  intercept artifact (each fold's intercept is the training runs' mean,
  so POOLED ranks anti-order with held-run means; with pooled domains a
  domain-separable deter plus a domain-level target offset yields
  positive pooled rho with zero per-state signal — both demonstrated in
  the 31-Jul independent review). The parent registration's pooled
  Spearman is carried point-only as pooled_diagnostic, non-evidential
  (its cluster CI has no nominal meaning under run heterogeneity, so no
  CI is reported for it). nan runs are excluded from the mean and
  counted in n_nan_runs; a draw with no live run gives a nan replicate,
  dropped by nanpercentile (fail-safe: a nan CI never passes ci > 0)."""
  keys = sorted(y)
  rr = _run_rhos(preds, y)
  v = np.array([rr[k] for k in keys], np.float64)
  rng = np.random.default_rng(seed)
  n = len(keys)
  vals = np.empty(b)
  for i in range(b):
    vals[i] = _nanmean(v[rng.integers(0, n, n)])
  lo, hi = np.nanpercentile(vals, [2.5, 97.5])
  return dict(spearman=_nanmean(v), ci=[float(lo), float(hi)],
              n_clusters=n, n_nan_runs=int(np.isnan(v).sum()),
              pooled_diagnostic=_pooled_rho(preds, y))


def rho_pair_boot(preds_lo, preds_hi, y, b=B_BOOT, seed=RNG_SEED):
  """Joint bootstrap of two rungs on the SAME cluster draws (amended
  within-run metric): returns (blk_lo, blk_hi, blk_diff) where blk_diff
  bootstraps the run-paired mean of [rho_hi_r - rho_lo_r]. With fresh
  default_rng(seed) and one integers(0, n, n) draw per replicate,
  blk_lo/blk_hi are identical to what rho_boot would return; the
  degenerate same-predictions case gives an exactly-zero contrast with
  zero-width CI (selfcheck leg)."""
  keys = sorted(y)
  vlo = np.array([_run_rhos(preds_lo, y)[k] for k in keys], np.float64)
  vhi = np.array([_run_rhos(preds_hi, y)[k] for k in keys], np.float64)
  vd = vhi - vlo
  rng = np.random.default_rng(seed)
  n = len(keys)
  blo, bhi, bd = np.empty(b), np.empty(b), np.empty(b)
  for i in range(b):
    pick = rng.integers(0, n, n)
    blo[i] = _nanmean(vlo[pick])
    bhi[i] = _nanmean(vhi[pick])
    bd[i] = _nanmean(vd[pick])
  plo = _pooled_rho(preds_lo, y)
  phi = _pooled_rho(preds_hi, y)

  def _blk(point, vals, pooled, n_nan):
    lo_, hi_ = np.nanpercentile(vals, [2.5, 97.5])
    return dict(spearman=point, ci=[float(lo_), float(hi_)],
                n_clusters=n, n_nan_runs=n_nan, pooled_diagnostic=pooled)

  diff = _blk(_nanmean(vd), bd, phi - plo, int(np.isnan(vd).sum()))
  diff['spearman_diff'] = diff.pop('spearman')
  diff['pooled_diagnostic_diff'] = diff.pop('pooled_diagnostic')
  return (_blk(_nanmean(vlo), blo, plo, int(np.isnan(vlo).sum())),
          _blk(_nanmean(vhi), bhi, phi, int(np.isnan(vhi).sum())),
          diff)


def l0_constants(y):
  blk = _cluster_boot({k: [float(v) for v in y[k]] for k in y})
  return dict(always_mean=blk['point'], always_ci=blk['ci'],
              n_clusters=blk['n_clusters'], never=0.0,
              best_constant=max(blk['point'], 0.0))


# --------------------------------------------------------------------------
# loading + guards
# --------------------------------------------------------------------------

def load_labels(labels_dir):
  runs = {}
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    if name.endswith('_smoke.npz'):
      continue
    m = FILE_RE.match(name)
    if not m or m.group(1) not in CORE_DOMAINS:
      continue  # registered scope: the committed cup/finger campaign only
    dom, dose, seed, mat = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    z = np.load(path, allow_pickle=True)
    md = json.loads(str(z['meta']))
    # EXACT equality, never substring: extension passes stamp
    # 'd1fix_20260724_xc1'/'_cm1' which CONTAIN the base string — a
    # misnamed consumer-arm file must trip, not blend.
    assert md.get('labeler_version') == LABELER_VERSION, (
        f'{name}: labeler_version {md.get("labeler_version")!r} != '
        f'{LABELER_VERSION!r} (exact pin)')
    for k, v in DIALS.items():
      assert md.get(k) == v, (
          f'{name}: dial {k}={md.get(k)!r} != registered {v!r}')
    assert md.get('train_seed') == seed, (
        f"{name}: meta train_seed={md.get('train_seed')} != filename {seed}")
    want = (f'r3_{dom}_{dose}_seed{seed}/'
            + ('ckpt_early' if mat == 'early' else 'ckpt'))
    assert str(md.get('checkpoint', '')).endswith(want), (
        f"{name}: checkpoint {md.get('checkpoint')!r} is not the {mat} "
        f'snapshot of r3_{dom}_{dose}_seed{seed}')
    dd = md.get('dose', {})
    if dose == 'e1':
      assert int(dd.get('dim', -1)) == 0, f'{name}: e1 but dose {dd}'
    else:
      assert (int(dd.get('dim', -1)) == 32
              and float(dd.get('scale', 0.0)) == 3.0), (
          f'{name}: e4 but dose {dd}')
    g_all = np.asarray(z['g_all'], float)
    g_now = np.asarray(z['g_now'], float)
    m_real = np.asarray(z['m_real'], int)
    validate_arrays(name, g_all, g_now, m_real)
    n = len(g_now)
    qfull = np.asarray(z['qfull'], np.float64)
    cands = np.asarray(z['cands'], np.float64)
    deter = np.asarray(z['deter'], np.float64)
    ref_deter = np.asarray(z['ref_deter'], np.float64)
    udyn = np.asarray(z['udyn'], np.float64)
    m_now = np.asarray(z['m_now'], int)
    assert qfull.shape == (n, K_HEADS, DIALS['actions']), (
        f'{name}: qfull shape {qfull.shape} != registered '
        f'{(n, K_HEADS, DIALS["actions"])}')
    assert cands.ndim == 3 and cands.shape[:2] == (n, DIALS['actions']), (
        f'{name}: cands shape {cands.shape} != (n, M, A)')
    assert deter.shape == (n, DETER_DIM), (
        f'{name}: deter shape {deter.shape} != {(n, DETER_DIM)}')
    assert ref_deter.ndim == 2 and ref_deter.shape[1] == DETER_DIM, (
        f'{name}: ref_deter shape {ref_deter.shape}')
    for arr, tag in ((qfull, 'qfull'), (cands, 'cands'), (deter, 'deter'),
                     (ref_deter, 'ref_deter'), (udyn, 'udyn')):
      assert np.isfinite(arr).all(), f'{name}: non-finite {tag}'
    assert m_now.min() >= 0 and m_now.max() < DIALS['actions'], (
        f"{name}: m_now outside candidate range [0,{DIALS['actions']})")
    key = (dom, dose, seed, mat)
    assert key not in runs, f'duplicate cell from {name}'
    runs[key] = dict(
        g_all=g_all, g_now=g_now, m_real=m_real, m_now=m_now, qfull=qfull,
        cands=cands, deter=deter, udyn=udyn, ref_deter=ref_deter,
        ref_episode=np.asarray(z['ref_episode']),
        ref_step=np.asarray(z['ref_step']),
        episode=np.asarray(z['episode']), step=np.asarray(z['step']))
  return runs


def check_grid(runs):
  assert runs, 'no r3 label files found'
  bad = sorted({k[2] for k in runs} - set(EXPECT_SEEDS))
  assert not bad, f'unregistered seeds present: {bad}'
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for seed in EXPECT_SEEDS:
        for mat in MATS:
          assert (dom, dose, seed, mat) in runs, (
              f'grid incomplete: {dom}/{dose}/seed{seed}/{mat} missing '
              '(the committed 64-file campaign is all-or-nothing)')
  assert len(runs) == 64, f'unexpected extra cells: {len(runs)}'
  ns = sorted({len(v['g_now']) for v in runs.values()})
  assert ns == [N_STATES], (
      f'n_states differs from the registered {N_STATES}: {ns}')


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def _verdict(comp, cand):
  if cand:
    v = ('CANDIDATE SIGNATURE (non-decisional heuristic) in '
         f'{cand}: candidate-geometry/Q-dispersion features predict the '
         'per-state target beyond the belief state => competence-repair '
         'targeting points at the CONSUMER side (candidate handling), '
         'resource consequence only (sibling registration '
         'PREREG_competence_repair_20260730.md).')
    if comp:
      v += (' COMPRESSION SIGNATURE also present in '
            f'{comp}: the belief state beats scalar summaries there '
            '(read-out repair relevant too).')
  elif comp:
    v = ('COMPRESSION SIGNATURE (non-decisional heuristic) in '
         f'{comp}: the belief state predicts the per-state target where '
         'scalar summaries do not => repair targeting points at the '
         'READ-OUT (representation-side decoder), not the candidate set '
         '(resource consequence only).')
  else:
    v = ('NO PER-STATE SIGNATURE: no rung shows registered per-state '
         'predictability => the implementation gap is not state-legible '
         'at this information level; repair targeting cannot be '
         'feature-guided from these labels (resource consequence only; '
         'the registered default targeting stands).')
  return v


def analyze(runs):
  check_grid(runs)
  feats = {k: features_all(v) for k, v in runs.items()}
  tgts = {k: per_state_targets(v['g_all'], v['g_now'], v['m_real'])
          for k, v in runs.items()}
  floor = {k: cell_stats(v['g_all'], v['g_now'], v['m_real'])['floor']
           for k, v in runs.items()}
  result = dict(
      cells={}, compression_signature=[], candidate_signature=[],
      amendment=AMENDMENT_ID,
      # Registered look count (disclosed in the prereg + amendment);
      # pooled_diagnostics = the 32 rung + 8 contrast pooled points,
      # carried non-evidential (no CI — see amendment):
      looks=dict(rung_spearman=32, paired_contrasts=8, l0_means=8,
                 pooled_diagnostics=40),
      thresholds=dict(sig_l3_min=SIG_L3_MIN, sig_l2_max=SIG_L2_MAX,
                      cand_delta_min=CAND_DELTA_MIN,
                      cand_l4_min=CAND_L4_MIN))
  for mat in MATS:
    for dose in DOSES:
      ckeys = sorted(k for k in runs if k[3] == mat and k[1] == dose)
      assert len(ckeys) == len(CORE_DOMAINS) * len(EXPECT_SEEDS)
      cname = f'{mat}_{dose}'
      cellblk = dict(
          floor_frac=float(np.mean([floor[k] for k in ckeys])))
      for tgt in TARGETS:
        y = {k: tgts[k][tgt] for k in ckeys}
        X1 = {k: np.stack([feats[k][c] for c in FEATURES_F0], 1)
              for k in ckeys}
        X2 = {k: np.stack([feats[k][c] for c in FEATURES_F1], 1)
              for k in ckeys}
        X3 = {k: np.column_stack(
            [np.asarray(runs[k]['deter'], np.float64), X2[k]])
            for k in ckeys}
        X4 = {k: np.column_stack(
            [X3[k]] + [feats[k][c][:, None] for c in FEATURES_G])
            for k in ckeys}
        cell = dict(L0=l0_constants(y))
        cell['L1'] = rho_boot(crossfit_ols(X1, y), y)
        cell['L2'] = rho_boot(crossfit_ols(X2, y), y)
        p3, lam3 = crossfit_ridge(X3, y)
        p4, lam4 = crossfit_ridge(X4, y)
        blk3, blk4, diff = rho_pair_boot(p3, p4, y)
        blk3['lambda_per_fold'] = {
            '_'.join(map(str, k)): v for k, v in lam3.items()}
        blk4['lambda_per_fold'] = {
            '_'.join(map(str, k)): v for k, v in lam4.items()}
        cell['L3'], cell['L4'] = blk3, blk4
        cell['contrast_l4_minus_l3'] = diff
        sig_comp = bool(
            cell['L3']['spearman'] >= SIG_L3_MIN
            and cell['L3']['ci'][0] > 0
            and cell['L2']['spearman'] < SIG_L2_MAX)
        sig_cand = bool(
            diff['spearman_diff'] >= CAND_DELTA_MIN
            and diff['ci'][0] > 0
            and cell['L4']['spearman'] >= CAND_L4_MIN
            and cell['L4']['ci'][0] > 0)
        cell['compression_signature'] = sig_comp
        cell['candidate_signature'] = sig_cand
        tag = f'{cname}:{tgt}'
        if sig_comp:
          result['compression_signature'].append(tag)
        if sig_cand:
          result['candidate_signature'].append(tag)
        cellblk[tgt] = cell
      result['cells'][cname] = cellblk
  result['verdict'] = _verdict(result['compression_signature'],
                               result['candidate_signature'])
  return result


def read(args):
  # Substrate gate: --labels must be byte-exactly the committed bundle
  # (selfcheck fixtures never call read(); there is no skip flag).
  got = labels_digest(args.labels)
  assert got == COMMITTED_LABELS_DIGEST, (
      f'committed-labels digest mismatch under {args.labels}: {got[:12]}… '
      f'!= frozen {COMMITTED_LABELS_DIGEST[:12]}… — this read only ever '
      'runs on the byte-exact committed R3 label bundle')
  runs = load_labels(args.labels)
  res = analyze(runs)
  res['labels_dir'] = os.path.abspath(args.labels)
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'r3_ladder.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=1, default=float)
  print(f'metric: mean per-run held-out Spearman ({AMENDMENT_ID}); '
        'pooled points carried non-evidential')
  for cname, cblk in res['cells'].items():
    for tgt in TARGETS:
      cell = cblk[tgt]
      row = '  '.join(
          f"{lv} rho={cell[lv]['spearman']:+.3f}"
          f"[{cell[lv]['ci'][0]:+.3f},{cell[lv]['ci'][1]:+.3f}]"
          for lv in ('L1', 'L2', 'L3', 'L4'))
      d = cell['contrast_l4_minus_l3']
      print(f"{cname:9s} {tgt:4s} alw={cell['L0']['always_mean']:+.3f}  "
            f"{row}  d43={d['spearman_diff']:+.3f}"
            f"[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]  "
            f"nanruns={cell['L3']['n_nan_runs']}  "
            f"comp={cell['compression_signature']} "
            f"cand={cell['candidate_signature']}")
  print()
  print(res['verdict'])
  print(f'-> {out}')


# --------------------------------------------------------------------------
# selfcheck (synthetic fixtures only; no jax/mujoco/torch anywhere)
# --------------------------------------------------------------------------

def _synth_run(rng, mode, floor=False, n=N_STATES, m=8, k=K_HEADS,
               adim=2, dd=16, refn=300):
  """One synthetic label file. Targets are planted through g_all so the
  registered per_state_targets path is exercised end-to-end: with
  g_now = 0, m_real = 0 and the signal in column 1, opp = gap = t."""
  qfull = rng.normal(0, 1, (n, k, m))
  cands = rng.normal(0, 1, (n, m, adim))
  deter = rng.normal(0, 1, (n, dd))
  run = dict(
      qfull=qfull, cands=cands, deter=deter,
      udyn=rng.normal(0, 1, n), m_now=rng.integers(0, m, n),
      ref_deter=rng.normal(0, 1, (refn, dd)),
      ref_episode=np.zeros(refn, int),
      ref_step=np.arange(refn) * 1000,  # never inside the exclusion window
      episode=np.full(n, 7, int), step=np.arange(n))
  if floor:
    run.update(g_all=np.zeros((n, m)), g_now=np.zeros(n),
               m_real=np.zeros(n, int))
    return run
  if mode == 'deter':
    z = deter[:, :4].sum(1)          # belief-only signal
  elif mode == 'scalar':
    z = d0_signals.u_q(qfull)        # F0-scalar signal
  elif mode == 'cand':
    z = cand_features(qfull, cands, run['m_now'])['act_pd_mean']
  else:
    z = np.zeros(n)                  # null
  z = (z - z.mean()) / (z.std() + 1e-12)
  t = 8.0 + 2.0 * np.tanh(z) + 0.05 * rng.normal(size=n)  # always > 0
  g_all = np.zeros((n, m))
  g_all[:, 1] = t
  run.update(g_all=g_all, g_now=np.zeros(n), m_real=np.zeros(n, int))
  return run


def _synth_grid(mode_by_cell, floor_files=(), seed=3):
  rng = np.random.default_rng(seed)
  runs = {}
  for dom in CORE_DOMAINS:
    for dose in DOSES:
      for sd in EXPECT_SEEDS:
        for mat in MATS:
          key = (dom, dose, sd, mat)
          runs[key] = _synth_run(rng, mode_by_cell.get((mat, dose), 'null'),
                                 floor=key in floor_files)
  return runs


def _fixture_npz(d, name, meta_over=(), **arr_over):
  n, m, adim = N_STATES, DIALS['actions'], 2
  md = dict(DIALS, train_seed=31, labeler_version=LABELER_VERSION,
            checkpoint='/x/r3_cup_e1_seed31/ckpt_early',
            dose=dict(dim=0, scale=1.0))
  md.update(dict(meta_over))
  arrays = dict(
      meta=json.dumps(md),
      qfull=np.zeros((n, K_HEADS, m), np.float32),
      cands=np.zeros((n, m, adim), np.float32),
      g_all=np.ones((n, m), np.float32), g_now=np.ones(n),
      m_real=np.zeros(n, int), m_now=np.zeros(n, int),
      udyn=np.zeros(n, np.float32),
      deter=np.zeros((n, DETER_DIM), np.float16),
      ref_deter=np.zeros((40, DETER_DIM), np.float16),
      ref_episode=np.zeros(40, int), ref_step=np.arange(40) * 1000,
      episode=np.full(n, 7, int), step=np.arange(n))
  arrays.update(arr_over)
  np.savez(os.path.join(d, name), **arrays)


def selfcheck(args):
  import tempfile

  # --- metric/estimand identities (leg 1) --------------------------------
  rng = np.random.default_rng(0)
  p = np.round(rng.normal(size=500), 1)   # heavy ties
  q = np.round(rng.normal(size=500), 1)
  assert abs(_rho(p, q) - stats.spearmanr(p, q).statistic) < 1e-12
  assert np.isnan(_rho(p, np.ones(500)))
  # hand-computed estimand identity vs r3_read.cell_stats (leg 2)
  g_all = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
  g_now = np.array([0.5, 1.0])
  m_real = np.array([2, 0])
  t = per_state_targets(g_all, g_now, m_real)
  assert np.allclose(t['opp'], [1.5, 0.0])
  assert np.allclose(t['gap'], [1.0, 0.0])
  st = cell_stats(g_all, g_now, m_real)
  assert abs(np.mean(t['opp']) - st['opp']) < 1e-12
  assert abs(np.mean(t['gap']) - st['gap']) < 1e-12
  # knn density ordering: dense-cluster point beats an outlier (leg 3)
  r = _synth_run(np.random.default_rng(1), 'null', dd=2)
  r['deter'] = np.array([[0.0, 0.0], [30.0, 30.0]])
  r['episode'] = np.array([7, 7])
  r['step'] = np.array([0, 1])
  ld = knn_logdens(r)
  assert ld[0] > ld[1], ld
  # paired-bootstrap identity: same preds on both sides => exact-zero
  # contrast with exact-zero CI (leg 4)
  yp = {a: np.random.default_rng(a).normal(size=30) for a in (0, 1, 2)}
  pp = {a: np.random.default_rng(a + 9).normal(size=30) for a in (0, 1, 2)}
  _, _, dz = rho_pair_boot(pp, pp, yp)
  assert dz['spearman_diff'] == 0.0 and dz['ci'] == [0.0, 0.0], dz

  # --- planted-signal grid (legs 5-8): deter fires L3+L4 not L1/L2;
  # scalar caught by L1/L2 without the compression signature; candidate-
  # geometry fires L4 while L3 does not; null cell + floor accounting ---
  grid = _synth_grid({('early', 'e1'): 'deter', ('late', 'e4'): 'scalar',
                      ('early', 'e4'): 'cand'},
                     floor_files={('cup', 'e1', 31, 'late')})
  res = analyze(grid)
  for tgt in TARGETS:
    c = res['cells']['early_e1'][tgt]
    assert c['compression_signature'] and not c['candidate_signature'], c
    assert c['L2']['spearman'] < SIG_L2_MAX, c['L2']
    assert c['L3']['spearman'] >= SIG_L3_MIN and c['L3']['ci'][0] > 0
    # amend1: deter is in L4's stack too — the parent asserted only L3
    assert c['L4']['spearman'] >= CAND_L4_MIN and c['L4']['ci'][0] > 0, (
        c['L4'])
    s = res['cells']['late_e4'][tgt]
    assert s['L1']['spearman'] >= 0.2 and s['L2']['spearman'] >= 0.2, s
    assert not s['compression_signature'] and not s['candidate_signature']
    g = res['cells']['early_e4'][tgt]
    assert g['candidate_signature'] and not g['compression_signature'], g
    assert g['L3']['spearman'] < SIG_L3_MIN, g['L3']
    assert g['L4']['spearman'] >= CAND_L4_MIN and g['L4']['ci'][0] > 0
    assert g['contrast_l4_minus_l3']['spearman_diff'] >= CAND_DELTA_MIN
    nn = res['cells']['late_e1'][tgt]
    assert not nn['compression_signature'] and not nn['candidate_signature']
    # amend1: null cell bounded at EVERY rung (the parent skipped L1/L2)
    assert abs(nn['L1']['spearman']) < SIG_L2_MAX, nn['L1']
    assert abs(nn['L2']['spearman']) < SIG_L2_MAX, nn['L2']
    assert abs(nn['L3']['spearman']) < SIG_L3_MIN
    assert abs(nn['L4']['spearman']) < SIG_L3_MIN
    # amend1 nan policy: the all-floor file is rank-constant -> nan run,
    # excluded and counted, at every rung and in the paired contrast
    for lv in ('L1', 'L2', 'L3', 'L4'):
      assert nn[lv]['n_nan_runs'] == 1, (lv, nn[lv])
    assert nn['contrast_l4_minus_l3']['n_nan_runs'] == 1
    assert c['L3']['n_nan_runs'] == 0, c['L3']
  assert abs(res['cells']['late_e1']['floor_frac'] - 1 / 16) < 1e-12
  assert res['verdict'].startswith('CANDIDATE SIGNATURE'), res['verdict']
  assert 'COMPRESSION SIGNATURE also present' in res['verdict']

  # --- compression-only verdict branch (leg 9) ---------------------------
  res_c = analyze(_synth_grid({('late', 'e4'): 'deter'}, seed=4))
  assert res_c['verdict'].startswith('COMPRESSION SIGNATURE'), (
      res_c['verdict'])
  assert not res_c['candidate_signature']
  assert set(res_c['compression_signature']) == {'late_e4:opp',
                                                 'late_e4:gap'}

  # --- all-null grid stays clean (leg 10; amend1: every rung bounded) ----
  null_grid = _synth_grid({}, seed=5)
  res_n = analyze(null_grid)
  assert not res_n['compression_signature']
  assert not res_n['candidate_signature']
  assert res_n['verdict'].startswith('NO PER-STATE SIGNATURE'), (
      res_n['verdict'])
  for cname, cblk in res_n['cells'].items():
    for tgt in TARGETS:
      c = cblk[tgt]
      assert abs(c['L1']['spearman']) < SIG_L2_MAX, (cname, tgt, c['L1'])
      assert abs(c['L2']['spearman']) < SIG_L2_MAX, (cname, tgt, c['L2'])
      assert abs(c['L3']['spearman']) < SIG_L3_MIN, (cname, tgt, c['L3'])
      assert abs(c['L4']['spearman']) < SIG_L3_MIN, (cname, tgt, c['L4'])

  # --- amend1 heterogeneity legs: the within-run metric stays null under
  # run-/domain-level target structure; the pooled diagnostics document
  # the parent metric's bias on the SAME grids (the 31-Jul review's
  # blocking finding, machine-checked here) ------------------------------
  het = _synth_grid({}, seed=6)
  hrng = np.random.default_rng(7)
  for key, run in het.items():
    off = hrng.normal(0.0, 1.5 * 0.05)   # run effect, 1.5x within-run sd
    g = np.zeros_like(run['g_all'])
    g[:, 1] = run['g_all'][:, 1] + off
    run['g_all'] = g
  res_h = analyze(het)
  assert not res_h['compression_signature'], res_h['compression_signature']
  assert not res_h['candidate_signature'], res_h['candidate_signature']
  for cname, cblk in res_h['cells'].items():
    for tgt in TARGETS:
      c = cblk[tgt]
      assert abs(c['L1']['spearman']) < SIG_L2_MAX, (cname, tgt, c['L1'])
      assert abs(c['L2']['spearman']) < SIG_L2_MAX, (cname, tgt, c['L2'])
      assert abs(c['L3']['spearman']) < SIG_L3_MIN, (cname, tgt, c['L3'])
      assert abs(c['L4']['spearman']) < SIG_L3_MIN, (cname, tgt, c['L4'])
  pooled_h = [cblk[tgt][lv]['pooled_diagnostic']
              for cblk in res_h['cells'].values() for tgt in TARGETS
              for lv in ('L2', 'L3')]
  assert min(pooled_h) <= -0.15, min(pooled_h)  # mean-reversal bias

  dom = _synth_grid({}, seed=8)
  for key, run in dom.items():
    if key[0] == 'finger':
      g = np.zeros_like(run['g_all'])
      g[:, 1] = run['g_all'][:, 1] + 0.9 * 0.05  # domain-level offset
      run['g_all'] = g
      run['deter'] = run['deter'].copy()
      run['deter'][:, 0] += 3.0                  # domain-separable belief
  res_d = analyze(dom)
  assert not res_d['compression_signature'], res_d['compression_signature']
  assert not res_d['candidate_signature'], res_d['candidate_signature']
  for cname, cblk in res_d['cells'].items():
    for tgt in TARGETS:
      c = cblk[tgt]
      assert abs(c['L3']['spearman']) < SIG_L3_MIN, (cname, tgt, c['L3'])
      assert abs(c['L4']['spearman']) < SIG_L3_MIN, (cname, tgt, c['L4'])
  pooled_d = [cblk[tgt]['L3']['pooled_diagnostic']
              for cblk in res_d['cells'].values() for tgt in TARGETS]
  assert max(pooled_d) >= 0.15, max(pooled_d)  # domain-shortcut bias

  # --- amend1 registered-dimension leg: nested ridge at p = DETER_DIM on
  # a 16-run run-heterogeneous null (single rung-cell scale) -------------
  drng = np.random.default_rng(9)
  X512 = {i: drng.normal(0, 1, (N_STATES, DETER_DIM)) for i in range(16)}
  y512 = {i: 8.0 + 0.05 * drng.normal(size=N_STATES)
          + drng.normal(0, 1.5 * 0.05) for i in range(16)}
  p512, _ = crossfit_ridge(X512, y512)
  blk512 = rho_boot(p512, y512)
  assert abs(blk512['spearman']) < SIG_L3_MIN, blk512
  assert blk512['n_nan_runs'] == 0, blk512
  # measured -0.16 at build; bound left slack for cross-BLAS drift
  assert blk512['pooled_diagnostic'] <= -0.10, blk512  # bias at full width

  # --- grid guards tripped (leg 11) --------------------------------------
  bad = dict(null_grid)
  bad[('cup', 'e1', 99, 'early')] = _synth_run(
      np.random.default_rng(6), 'null')
  try:
    analyze(bad)
    raise SystemExit('selfcheck FAIL: unregistered seed not caught')
  except AssertionError as e:
    assert 'unregistered seeds' in str(e), e
  bad = dict(null_grid)
  del bad[('cup', 'e1', 31, 'late')]
  try:
    analyze(bad)
    raise SystemExit('selfcheck FAIL: missing file not caught')
  except AssertionError as e:
    assert 'grid incomplete' in str(e), e
  bad = dict(null_grid)
  bad[('cup', 'e1', 31, 'early')] = dict(
      bad[('cup', 'e1', 31, 'early')],
      g_now=bad[('cup', 'e1', 31, 'early')]['g_now'][:120])
  try:
    analyze(bad)
    raise SystemExit('selfcheck FAIL: off-registration n_states not caught')
  except AssertionError as e:
    assert 'n_states' in str(e), e

  # --- file-level tempdir-npz fixture legs (leg 12) ----------------------
  def _expect_trip(d, needle):
    try:
      load_labels(d)
      raise SystemExit(f'selfcheck FAIL: {needle} not caught')
    except AssertionError as e:
      assert needle in str(e), e

  with tempfile.TemporaryDirectory() as d:
    _fixture_npz(d, 'r3_cup_e1_seed31_early.npz')
    _fixture_npz(d, 'r3_reacher_e1_seed31_early_smoke.npz')  # ignored
    _fixture_npz(d, 'r3_reacher_e1_seed31_early.npz',
                 meta_over=dict(checkpoint=(
                     '/x/r3_reacher_e1_seed31/ckpt_early')))  # out of scope
    got = load_labels(d)
    assert list(got) == [('cup', 'e1', 31, 'early')]
    assert got[('cup', 'e1', 31, 'early')]['deter'].dtype == np.float64
    assert got[('cup', 'e1', 31, 'early')]['deter'].shape == (
        N_STATES, DETER_DIM)
    _fixture_npz(d, 'r3_cup_e1_seed031_early.npz')  # zero-padded duplicate
    _expect_trip(d, 'duplicate cell')
  # labels_digest hand identity + sensitivity (the read()'s substrate
  # gate compares against the frozen constant; here we check the
  # digest FUNCTION: deterministic, order-canonical, content-sensitive,
  # smoke-exempt).
  with tempfile.TemporaryDirectory() as d:
    for nm, payload in (('b.npz', 1), ('a.npz', 2), ('c_smoke.npz', 3)):
      np.savez(os.path.join(d, nm), x=np.array([payload]))
    d1 = labels_digest(d)
    assert d1 == labels_digest(d), 'digest not deterministic'
    exp = hashlib.sha256()
    for nm in ('a.npz', 'b.npz'):
      with open(os.path.join(d, nm), 'rb') as f:
        exp.update(
            f'{nm}:{hashlib.sha256(f.read()).hexdigest()}\n'.encode())
    assert d1 == exp.hexdigest(), 'digest != hand computation'
    np.savez(os.path.join(d, 'a.npz'), x=np.array([99]))
    assert labels_digest(d) != d1, 'digest insensitive to content change'
  for name, needle, meta_over, arr_over in (
      ('r3_cup_e1_seed31_early.npz', 'labeler_version',
       dict(labeler_version='wrong'), {}),
      # base+suffix contains the base string — the exact pin must trip
      # where a substring check would blend a misnamed consumer-arm file
      ('r3_cup_e1_seed31_early.npz', 'exact pin',
       dict(labeler_version='d1fix_20260724_cm1'), {}),
      ('r3_cup_e1_seed31_early.npz', 'dial horizon',
       dict(horizon=50), {}),
      ('r3_cup_e1_seed31_early.npz', 'dial actions',
       dict(actions=16), {}),
      ('r3_cup_e1_seed31_early.npz', 'train_seed',
       dict(train_seed=32), {}),
      ('r3_cup_e1_seed31_late.npz', 'is not the late',
       dict(), {}),  # fixture checkpoint stays .../ckpt_early
      ('r3_cup_e4_seed31_early.npz', 'e4 but dose',
       dict(checkpoint='/x/r3_cup_e4_seed31/ckpt_early'), {}),
      ('r3_cup_e1_seed31_early.npz', 'qfull shape', dict(),
       dict(qfull=np.zeros((N_STATES, 4, 8), np.float32))),
      ('r3_cup_e1_seed31_early.npz', 'non-finite deter', dict(),
       dict(deter=np.full((N_STATES, DETER_DIM), np.inf, np.float16))),
      ('r3_cup_e1_seed31_early.npz', 'non-finite G', dict(),
       dict(g_all=np.full((N_STATES, 8), np.nan, np.float32))),
      ('r3_cup_e1_seed31_early.npz', 'm_now outside', dict(),
       dict(m_now=np.full(N_STATES, 8, int))),
  ):
    with tempfile.TemporaryDirectory() as d:
      _fixture_npz(d, name, meta_over=meta_over, **arr_over)
      _expect_trip(d, needle)

  print('selfcheck PASS: rho/spearmanr identity + nan, estimand identity '
        'vs cell_stats, knn density ordering, paired-bootstrap zero '
        'contrast, deter-planted fires L3+L4 (compression signature) not '
        'L1/L2, scalar-planted caught by L1/L2 without signatures, '
        'candidate-planted fires L4 not L3 (candidate signature), null '
        'cells bounded at every rung + floor nan accounting, '
        'compression-only and no-signature verdict branches, amend1 '
        'heterogeneity legs (run-effect null + domain-offset null: '
        'within-run metric bounded, no signatures, pooled diagnostics '
        'reproduce the mean-reversal and domain-shortcut biases), amend1 '
        'p=512 ridge leg, grid guards (unregistered seed, grid '
        'incomplete, n_states), file fixture guards (duplicate, '
        'labeler_version wrong + base+suffix exact-pin trip, dials, '
        'train_seed, checkpoint maturity, dose identity, qfull shape, '
        'finiteness, m_now bounds, reacher out-of-scope skip), '
        'labels_digest hand identity + content sensitivity')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--labels')
  ap.add_argument('--output', default='analysis_out/r3_ladder')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if not args.selfcheck and not args.labels:
    ap.error('--labels required (or --selfcheck)')
  if args.selfcheck:
    selfcheck(args)
  else:
    read(args)


if __name__ == '__main__':
  main()
