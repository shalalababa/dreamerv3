"""Frozen read for the W1 adjudication wave (Paper 2).

Registered under PREREG_w1_adjudication_constraints_20260809.md (design
constraints, committed first) + PREREG_w1_wave_20260809.md (decision
rules, this reader frozen alongside). ONE execution per registration.

Label files (from the W1-modified labelers, labeler_version *_w1):
  dv3:  w1_{cup|finger}_{e1|e4}_seed{31..38}_late.npz        (32 cells)
        w1dup_{cup|finger}_e1_seed31_late.npz                (2 gates)
  tm2:  w1tm2_{cup|finger}_{e1|e4}_seed{51..58}_late.npz     (32 cells)
        w1tm2dup_{cup|finger}_e1_seed51_late.npz             (2 gates)

Arrays per state: g_all_rep (R, M) with CRN across branches within a
repeat and independent follower marks across repeats; g_actor_rep /
g_plan_rep (R,); real_scores, r_real (M,); m_now; dup_cand flag.

Estimators (per cell = per-state means; cluster = run; pooled over
cells; permutation-primary + BCa 95% per the 08-08 standing rule):

  SPLIT-SELECTED OPPORTUNITY (P-W1a):
    sel(s)   = argmax_m mean_{r even} g_all_rep[s, r, m]
    ho(s, m) = mean_{r odd} g_all_rep[s, r, m]
    opp_split(s) = ho(s, sel(s)) - ho(s, m_now(s))
  CONSUMER HARVEST (P-W1b, actor/planner):
    harv(s) = mean_{r odd} g_cons_rep(s, r) - mean_m ho(s, m)
  LEAK DECOMPOSITION (P-W1c, non-degenerate held-out states):
    rho_full  = Spearman(real_scores,          ho(s, :))
    rho_rew   = Spearman(r_real,               ho(s, :))
    rho_value = Spearman(real_scores - r_real, ho(s, :))

  DUPLICATE-NULL GATE: on dup passes every estimand must be exactly 0
  (within-repeat CRN makes identical actions bit-identical); any
  violation -> the family's read refuses (INSTRUMENT-INVALID).

Decision rules are frozen in PREREG_w1_wave_20260809.md and asserted
here verbatim; the selfcheck plants fixtures for every branch.

Usage:
  python -m analysis.w1_read --family dv3 --labels <dir> --output <dir>
  python -m analysis.w1_read --family tm2 --labels <dir> --output <dir>
  python -m analysis.w1_read --selfcheck
"""

import argparse
import glob
import json
import os
import re

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
FILE_RES = dict(
    dv3=re.compile(r'^w1_(cup|finger)_(e1|e4)_seed(3[1-8])_late\.npz$'),
    tm2=re.compile(r'^w1tm2_(cup|finger)_(e1|e4)_seed(5[1-8])_late\.npz$'))
DUP_RES = dict(
    dv3=re.compile(r'^w1dup_(cup|finger)_(e1|e4)_seed(3[1-8])_late\.npz$'),
    tm2=re.compile(r'^w1tm2dup_(cup|finger)_(e1|e4)_seed(5[1-8])_late\.npz$'))
CONS_KEY = dict(dv3='g_actor_rep', tm2='g_plan_rep')
VERSION_SUFFIX = '_w1'
EXPECT_CELLS = 32
MIN_DUP_PASSES = 2
DUP_TOL = 0.0            # exact: CRN makes duplicate branches identical
# Registered dial pins (batch-review M4; PREREG_w1_wave_20260809 —
# the 08-07 standing rule: every read verifies realized dials):
EXPECT_ENV_SEED = 20260809
EXPECT_REPEATS = 8
EXPECT_STATES = 200
EXPECT_HORIZON = 100


def _midranks(x):
  """Tie-averaged (mid)ranks — batch-review B2: argsort-of-argsort
  fabricates distinct ranks on ties, which lets index order masquerade
  as signal on constant inputs (sparse-reward r_real is frequently
  all-zero across candidates)."""
  x = np.asarray(x, float)
  order = np.argsort(x, kind='mergesort')
  ranks = np.empty(len(x), float)
  ranks[order] = np.arange(len(x), dtype=float)
  sx = x[order]
  i = 0
  while i < len(sx):
    j = i
    while j + 1 < len(sx) and sx[j + 1] == sx[i]:
      j += 1
    if j > i:
      ranks[order[i:j + 1]] = 0.5 * (i + j)
    i = j + 1
  return ranks


def spearman(a, b):
  a = np.asarray(a, float)
  b = np.asarray(b, float)
  # NaN on (near-)constant RAW inputs: a constant vector has no ranking
  # to correlate (batch-review B2 — the old rank-std guard was dead
  # code because fabricated ranks always vary).
  if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
    return np.nan
  ra, rb = _midranks(a), _midranks(b)
  sa, sb = ra.std(), rb.std()
  if sa < 1e-12 or sb < 1e-12:
    return np.nan
  return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def perm_p(vals, rng_seed=RNG_SEED, b=B_BOOT):
  """Sign-flip permutation p for mean > 0 (two-sided)."""
  vals = np.asarray(vals, float)
  vals = vals[np.isfinite(vals)]
  rng = np.random.default_rng(rng_seed)
  obs = abs(vals.mean())
  flips = rng.choice([-1.0, 1.0], size=(b, len(vals)))
  null = np.abs((flips * vals[None]).mean(1))
  return float((np.sum(null >= obs - 1e-15) + 1) / (b + 1))


def bca(vals, clusters, rng_seed=RNG_SEED, b=B_BOOT):
  """Cluster bootstrap + BCa correction (no scipy; normal quantiles via
  erfinv)."""
  vals = np.asarray(vals, float)
  cl = np.asarray(clusters)
  uniq = sorted(set(cl.tolist()))
  idx = {c: np.where(cl == c)[0] for c in uniq}
  rng = np.random.default_rng(rng_seed)
  boots = np.empty(b)
  for i in range(b):
    pick = rng.choice(len(uniq), len(uniq), replace=True)
    sel = np.concatenate([idx[uniq[p]] for p in pick])
    boots[i] = np.nanmean(vals[sel])
  theta = float(np.nanmean(vals))
  # bias correction
  prop = float(np.mean(boots < theta))
  prop = min(max(prop, 1.0 / (b + 1)), 1.0 - 1.0 / (b + 1))
  from math import sqrt
  def ndtri(p):
    # Acklam-style rational approximation via erfinv identity
    return sqrt(2.0) * _erfinv(2.0 * p - 1.0)
  def _erfinv(x):
    # Winitzki approximation (sufficient at B=10K resolution)
    import math
    a = 0.147
    ln = math.log(1.0 - x * x)
    t1 = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)
  z0 = ndtri(prop)
  # acceleration via leave-one-cluster-out jackknife
  jack = []
  for c in uniq:
    keep = cl != c
    jack.append(float(np.nanmean(vals[keep])))
  jack = np.asarray(jack)
  jm = jack.mean()
  num = np.sum((jm - jack) ** 3)
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  a_acc = float(num / den) if den > 0 else 0.0
  def q(alpha_pt):
    z = ndtri(alpha_pt)
    adj = z0 + (z0 + z) / (1.0 - a_acc * (z0 + z))
    from math import erf, sqrt as _s
    p = 0.5 * (1.0 + erf(adj / _s(2.0)))
    return float(np.percentile(boots, 100.0 * min(max(p, 0.0), 1.0)))
  return theta, (q(0.025), q(0.975))


def load_family(labels_dir, family):
  cells, dups = {}, []
  rex, dre = FILE_RES[family], DUP_RES[family]
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    name = os.path.basename(path)
    m, d = rex.match(name), dre.match(name)
    if not (m or d):
      continue
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
    version = str(meta.get('labeler_version', ''))
    assert version.endswith(VERSION_SUFFIX), (name, version)
    # M4 pins: registered dials + filename<->run identity, fail-closed.
    assert int(meta.get('env_seed', -1)) == EXPECT_ENV_SEED, (
        name, meta.get('env_seed'))
    assert int(meta.get('w1', {}).get('repeats', -1)) == EXPECT_REPEATS, (
        name, meta.get('w1'))
    assert int(meta.get('states', -1)) == EXPECT_STATES, (
        name, meta.get('states'))
    assert int(meta.get('horizon', -1)) == EXPECT_HORIZON, (
        name, meta.get('horizon'))
    grp = (m or d)
    rid = str(np.asarray(z['run_id']).reshape(-1)[0])
    dom, dose, seed = grp.group(1), grp.group(2), int(grp.group(3))
    assert (dom in rid and dose in rid and f'seed{seed}' in rid), (
        f'{name}: filename ({dom},{dose},{seed}) not found in run_id '
        f'{rid!r} — mislabeled cell, read refuses')
    entry = dict(
        g=np.asarray(z['g_all_rep'], float),
        gc=np.asarray(z[CONS_KEY[family]], float),
        rs=np.asarray(z['real_scores'], float),
        rr=np.asarray(z['r_real'], float),
        mn=np.asarray(z['m_now'], int),
        dup=np.asarray(z['dup_cand'], bool))
    assert len(entry['g']) == int(meta.get('states', -1)), (
        name, len(entry['g']), meta.get('states'))
    if d is not None:
      assert entry['dup'].all(), f'{name}: dup file without dup rows'
      dups.append((name, entry))
    else:
      assert not entry['dup'].any(), f'{name}: dup rows in a main file'
      key = (d or m).group(1), m.group(2), int(m.group(3))
      assert key not in cells, f'duplicate cell {key}'
      cells[key] = entry
  return cells, dups


def dup_gate(dups):
  """Every estimand must be EXACTLY zero on duplicate passes."""
  assert len(dups) >= MIN_DUP_PASSES, (
      f'{len(dups)} duplicate-null passes < {MIN_DUP_PASSES} — the gate '
      'is registered as mandatory')
  for name, e in dups:
    g = e['g']
    if not np.all(g == g[:, :, :1]):
      raise SystemExit(f'DUPLICATE-NULL VIOLATION in {name}: branches '
                       'of identical actions differ — within-repeat CRN '
                       'broken; read refuses (INSTRUMENT-INVALID)')
    opp = _cell_split_opp(e)
    if abs(opp) > DUP_TOL:
      raise SystemExit(f'DUPLICATE-NULL VIOLATION in {name}: split '
                       f'opportunity {opp} != 0; read refuses')
    # batch-review M3: the PROBE side must also be exactly constant on
    # duplicates — the historical leak (duplicate-witness rho=0.47) was
    # probe-side, and a probe-CRN breakage would otherwise pass this
    # gate while P-W1c reads through it.
    if float(np.max(np.ptp(e['rr'], axis=1))) > DUP_TOL:
      raise SystemExit(f'DUPLICATE-NULL VIOLATION in {name}: r_real '
                       'varies across identical candidates — probe CRN '
                       'broken; read refuses')
    if float(np.max(np.ptp(e['rs'], axis=1))) > DUP_TOL:
      raise SystemExit(f'DUPLICATE-NULL VIOLATION in {name}: '
                       'real_scores varies across identical candidates '
                       '— probe CRN broken; read refuses')
  return len(dups)


def _cell_split_opp(e):
  g, mn = e['g'], e['mn']
  even = g[:, 0::2].mean(1)
  odd = g[:, 1::2].mean(1)
  sel = even.argmax(1)
  i = np.arange(len(sel))
  return float(np.mean(odd[i, sel] - odd[i, mn]))


def _cell_harvest(e):
  g, gc = e['g'], e['gc']
  odd = g[:, 1::2].mean(1)
  return float(np.mean(gc[:, 1::2].mean(1) - odd.mean(1)))


def _cell_leak_rhos(e):
  """Held-out rank decomposition. Batch-review M2: r_real is bitwise
  the FIRST-STEP component of every repeat's G (same restored
  transition), so Spearman(r_real, G_odd) fires tautologically whenever
  r_real varies. The registered reward-only test therefore correlates
  r_real against the TAIL (G_odd − r_real): ground-truth first-step
  reward predicting the rest of the return. rho_value (= disc·Vhat, a
  model output, never a component of G) is unaffected."""
  g = e['g']
  odd = g[:, 1::2].mean(1)                      # (S, M) held-out means
  full, rew, val = [], [], []
  for s in range(len(g)):
    if odd[s].std() < 1e-12:
      continue
    tail = odd[s] - e['rr'][s]
    a = spearman(e['rs'][s], odd[s])
    bmark = spearman(e['rr'][s], tail)
    c = spearman(e['rs'][s] - e['rr'][s], odd[s])
    if np.isfinite(a):
      full.append(a)
    if np.isfinite(bmark):
      rew.append(bmark)
    if np.isfinite(c):
      val.append(c)
  return (float(np.mean(full)) if full else np.nan,
          float(np.mean(rew)) if rew else np.nan,
          float(np.mean(val)) if val else np.nan)


def stat_block(cell_vals, clusters):
  vals = np.asarray(cell_vals, float)
  point, ci = bca(vals, clusters)
  return dict(point=point, ci=list(ci), perm_p=perm_p(vals),
              n_cells=int(np.isfinite(vals).sum()),
              n_clusters=len(set(clusters)))


def run_read(labels_dir, family, output):
  cells, dups = load_family(labels_dir, family)
  assert len(cells) == EXPECT_CELLS, (
      f'{len(cells)} cells != registered {EXPECT_CELLS}')
  n_dup = dup_gate(dups)
  cl = ['%s_%s_%d' % k for k in cells]
  out = dict(family=family, n_cells=len(cells), n_dup_passes=n_dup)

  opp = [_cell_split_opp(e) for e in cells.values()]
  out['p_w1a_split_opportunity'] = stat_block(opp, cl)
  out['p_w1a_fires'] = bool(
      out['p_w1a_split_opportunity']['ci'][0] > 0
      and out['p_w1a_split_opportunity']['perm_p'] < 0.05)

  harv = [_cell_harvest(e) for e in cells.values()]
  out['p_w1b_consumer_harvest'] = stat_block(harv, cl)
  out['p_w1b_fires'] = bool(
      out['p_w1b_consumer_harvest']['ci'][0] > 0
      and out['p_w1b_consumer_harvest']['perm_p'] < 0.05)

  rhos = [_cell_leak_rhos(e) for e in cells.values()]
  full = [r[0] for r in rhos]
  rew = [r[1] for r in rhos]
  val = [r[2] for r in rhos]
  out['p_w1c_rho_full'] = stat_block(full, cl)
  out['p_w1c_rho_reward_only'] = stat_block(rew, cl)
  out['p_w1c_rho_value_component'] = stat_block(val, cl)
  v_fires = (out['p_w1c_rho_value_component']['ci'][0] > 0
             and out['p_w1c_rho_value_component']['perm_p'] < 0.05)
  r_fires = (out['p_w1c_rho_reward_only']['ci'][0] > 0
             and out['p_w1c_rho_reward_only']['perm_p'] < 0.05)
  out['p_w1c_verdict'] = (
      'DISSOCIATION-SURVIVES' if v_fires else
      'GROUND-TRUTH-REWARD-ONLY' if r_fires else
      'LEAK-WAS-ALL')

  os.makedirs(output, exist_ok=True)
  path = os.path.join(output, f'w1_{family}.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)
  return out


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _mk_cell(rng, S=40, R=8, M=6, signal=0.0, leak=0.0, value_sig=0.0,
             cons_edge=0.0, rew_sig=0.0):
  """Synthetic cell mirroring the REAL bitwise structure: G's first
  step IS r_real (same restored transition — review M2), i.e.
  g = rr + mu + repeat-noise. rr is zero (the sparse-reward common
  case) unless `leak` (rr correlated with the SELECTION half's
  realization — must classify LEAK-WAS-ALL) or `rew_sig` (rr predicts
  the TAIL — must classify GROUND-TRUTH-REWARD-ONLY) is set. signal:
  true per-candidate tail spread; value_sig: real_scores' learned-value
  component correlated with the true means (DISSOCIATION-SURVIVES);
  cons_edge: consumer branch advantage."""
  mu = rng.normal(0, signal, (S, M)) if signal else np.zeros((S, M))
  noise = rng.normal(0, 1.0, (S, R, M))
  rr = np.zeros((S, M))
  if leak:
    rr = leak * noise[:, 0]
  if rew_sig:
    rr = rng.normal(0, 1.0, (S, M))
    mu = mu + rew_sig * rr
  g = rr[:, None] + mu[:, None] + noise
  rs = rr + value_sig * mu + rng.normal(0, 0.1, (S, M))
  gc = (mu.mean(1)[:, None] + cons_edge
        + rng.normal(0, 1.0, (S, R)))
  return dict(g=g, gc=gc, rs=rs, rr=rr,
              mn=rng.integers(0, M, S),
              dup=np.zeros(S, bool))


def selfcheck():
  rng = np.random.default_rng(0)
  # 1) pure-noise cells: split opportunity must NOT fire
  cells = {(d, e, s): _mk_cell(rng)
           for d in ('cup', 'finger') for e in ('e1', 'e4')
           for s in range(31, 39)}
  cl = ['%s_%s_%d' % k for k in cells]
  opp = [_cell_split_opp(e) for e in cells.values()]
  blk = stat_block(opp, cl)
  assert not (blk['ci'][0] > 0 and blk['perm_p'] < 0.05), blk
  # ...and the WITHIN-pass estimator on the same cells DOES look
  # positive (the defect the wave exists to kill) — sanity, not a rule
  within = np.mean([np.mean(e['g'][:, 0::2].mean(1).max(1)
                            - e['g'][:, 0::2].mean(1).mean(1))
                    for e in cells.values()])
  assert within > 0.3, within
  # 2) true-signal cells: split opportunity fires
  cells2 = {k: _mk_cell(rng, signal=1.0) for k in cells}
  opp2 = [_cell_split_opp(e) for e in cells2.values()]
  blk2 = stat_block(opp2, cl)
  assert blk2['ci'][0] > 0 and blk2['perm_p'] < 0.05, blk2
  # 3) leak-only (real bitwise structure: rr ∈ G, correlated with the
  # SELECTION half): decomposition must classify LEAK-WAS-ALL — the
  # reward-only test correlates rr against the TAIL (M2), killing the
  # first-step tautology
  cells3 = {k: _mk_cell(rng, leak=2.0) for k in cells}
  rhos3 = [_cell_leak_rhos(e) for e in cells3.values()]
  v3 = stat_block([r[2] for r in rhos3], cl)
  r3 = stat_block([r[1] for r in rhos3], cl)
  assert not (v3['ci'][0] > 0 and v3['perm_p'] < 0.05), v3
  assert not (r3['ci'][0] > 0 and r3['perm_p'] < 0.05), r3
  # 3b) MUTANT (M2 regression): the tautological form
  # Spearman(rr, G_odd) WOULD have fired on the leak-only cells —
  # proving the registered tail form is the load-bearing choice
  taut = []
  for e in cells3.values():
    odd = e['g'][:, 1::2].mean(1)
    vals = [spearman(e['rr'][s], odd[s]) for s in range(len(odd))
            if odd[s].std() > 1e-12]
    vals = [v for v in vals if np.isfinite(v)]
    taut.append(float(np.mean(vals)))
  tblk = stat_block(taut, cl)
  assert tblk['ci'][0] > 0 and tblk['perm_p'] < 0.05, tblk
  # 3c) ground-truth-reward-only branch: rr predicts the TAIL
  cells3c = {k: _mk_cell(rng, rew_sig=1.5) for k in cells}
  rhos3c = [_cell_leak_rhos(e) for e in cells3c.values()]
  r3c = stat_block([r[1] for r in rhos3c], cl)
  v3c = stat_block([r[2] for r in rhos3c], cl)
  assert r3c['ci'][0] > 0 and r3c['perm_p'] < 0.05, r3c
  assert not (v3c['ci'][0] > 0 and v3c['perm_p'] < 0.05), v3c
  # 4) value-signal: DISSOCIATION-SURVIVES branch
  cells4 = {k: _mk_cell(rng, signal=1.0, value_sig=2.0) for k in cells}
  rhos4 = [_cell_leak_rhos(e) for e in cells4.values()]
  v4 = stat_block([r[2] for r in rhos4], cl)
  assert v4['ci'][0] > 0 and v4['perm_p'] < 0.05, v4
  # 4b) tie handling (B2): constant raw input -> NaN, never a rank
  # fabrication; midranks on ties exact
  assert np.isnan(spearman(np.zeros(8), np.arange(8.0)))
  assert abs(spearman([1., 1., 2., 2.], [1., 2., 3., 4.])
             - 0.894427) < 1e-4
  # 5) consumer edge fires P-W1b; absence does not
  h5 = stat_block([_cell_harvest(_mk_cell(rng, cons_edge=0.8))
                   for _ in range(32)], cl)
  assert h5['ci'][0] > 0 and h5['perm_p'] < 0.05, h5
  h6 = stat_block([_cell_harvest(_mk_cell(rng)) for _ in range(32)], cl)
  assert not (h6['ci'][0] > 0 and h6['perm_p'] < 0.05), h6
  # 6) duplicate gate trips on a violated fixture and passes on a clean
  clean = _mk_cell(rng)
  clean['g'] = np.repeat(clean['g'][:, :, :1], clean['g'].shape[2], 2)
  clean['rr'] = np.repeat(clean['rr'][:, :1], clean['rr'].shape[1], 1)
  clean['rs'] = np.repeat(clean['rs'][:, :1], clean['rs'].shape[1], 1)
  clean['dup'] = np.ones(len(clean['g']), bool)
  dup_gate([('clean.npz', clean)] * MIN_DUP_PASSES)
  bad = _mk_cell(rng)
  bad['dup'] = np.ones(len(bad['g']), bool)
  try:
    dup_gate([('bad.npz', bad)] * MIN_DUP_PASSES)
    raise AssertionError('dup gate failed to trip')
  except SystemExit as e:
    assert 'DUPLICATE-NULL VIOLATION' in str(e), e
  # 6b) PROBE-side violation (M3): rollouts identical but r_real varies
  # — the historical leak channel — must also refuse
  probe_bad = {k: (v.copy() if hasattr(v, 'copy') else v)
               for k, v in clean.items()}
  probe_bad['rr'] = rng.normal(0, 1, probe_bad['rr'].shape)
  try:
    dup_gate([('probebad.npz', probe_bad)] * MIN_DUP_PASSES)
    raise AssertionError('probe-side dup gate failed to trip')
  except SystemExit as e:
    assert 'probe CRN broken' in str(e), e
  # 7) too-few dup passes refused
  try:
    dup_gate([('clean.npz', clean)])
    raise AssertionError('dup count gate failed to trip')
  except AssertionError as e:
    if 'failed to trip' in str(e):
      raise
  # 8) load_family end-to-end: compliant files parse; dial-pin and
  # identity violations refuse (M4/m2)
  import json as _json
  import tempfile

  def _write(tmp, name, meta_over=None, run_id='r3_cup_e1_seed31',
             dup=False):
    S, R, M = EXPECT_STATES, EXPECT_REPEATS, 8
    rng2 = np.random.default_rng(1)
    g = rng2.normal(0, 1, (S, R, M))
    if dup:
      g = np.repeat(g[:, :, :1], M, 2)
    meta = dict(labeler_version='d1fix_20260724_w1',
                env_seed=EXPECT_ENV_SEED, states=S,
                horizon=EXPECT_HORIZON, w1=dict(repeats=R))
    meta.update(meta_over or {})
    np.savez_compressed(
        os.path.join(tmp, name), meta=_json.dumps(meta),
        g_all_rep=g.astype(np.float32),
        g_actor_rep=rng2.normal(0, 1, (S, R)).astype(np.float32),
        real_scores=np.zeros((S, M), np.float32),
        r_real=np.zeros((S, M), np.float32),
        m_now=np.zeros(S, np.int64),
        dup_cand=np.full(S, dup),
        run_id=np.array([run_id] * S))

  with tempfile.TemporaryDirectory() as tmp:
    _write(tmp, 'w1_cup_e1_seed31_late.npz')
    _write(tmp, 'w1dup_cup_e1_seed31_late.npz', dup=True)
    cells8, dups8 = load_family(tmp, 'dv3')
    assert len(cells8) == 1 and len(dups8) == 1
    dup_gate(dups8 * MIN_DUP_PASSES)
  for bad_kw, bad_name in (
      (dict(meta_over=dict(env_seed=1)), 'w1_cup_e1_seed32_late.npz'),
      (dict(meta_over=dict(w1=dict(repeats=4))),
       'w1_cup_e1_seed33_late.npz'),
      (dict(run_id='r3_finger_e1_seed31'),
       'w1_cup_e1_seed34_late.npz')):
    with tempfile.TemporaryDirectory() as tmp:
      _write(tmp, bad_name, **bad_kw)
      try:
        load_family(tmp, 'dv3')
        raise SystemExit(f'selfcheck FAIL: pin violation not refused: '
                         f'{bad_kw}')
      except AssertionError:
        pass
  print('SELFCHECK PASS (w1_read: noise cells no-fire + within-pass '
        'bias visible; true-signal fires; leak-only -> LEAK-WAS-ALL '
        '+ tautological-form mutant CAUGHT; reward-predicts-tail -> '
        'GROUND-TRUTH-REWARD-ONLY; value-signal -> DISSOCIATION-'
        'SURVIVES; tie/constant-input spearman guards; consumer-edge '
        'fire/no-fire; duplicate gate exact-zero + probe-side ptp trip '
        '+ clean pass + count guard; load_family dial pins + '
        'filename-identity refusals)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--family', choices=['dv3', 'tm2'])
  p.add_argument('--labels')
  p.add_argument('--output')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not (args.family and args.labels and args.output):
    p.error('--family, --labels, --output required (or --selfcheck)')
  run_read(args.labels, args.family, args.output)


if __name__ == '__main__':
  main()
