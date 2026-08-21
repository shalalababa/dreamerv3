"""Frozen read for the Paper-2 Tier-1 CALIBRATION BUNDLE (21 Aug 2026).

Registered under PREREG_p2_tier1_calibration_20260821.md — the ThreeAxes
board's items 2-5 plus the run-level A/B admissibility gate, registered
as ONE bundle BEFORE any computation (ThreeAxes §1.2: these are new
estimands over the archived W1 label arrays; computing first would burn
them the way the substrate screen burned itself). ONE execution of
--component main (local label arrays) + ONE of --component ab
(cluster-side eval scores), reported together.

Components (all on local_results/w1_labels_20260811_090727/w1_labels/):

  CCD  — consumer-competence dial: synthetic choosers sel_sigma(s) =
         argmax_m [even-half mean + sigma*eps], evaluated on the odd
         half against the mean-candidate baseline. sigma=0 is the
         perfect selector-on-available-information; 'rand' is the
         uniform chooser (truth 0 by exchangeability of the baseline).
         Buys the positive control at zero compute and places the
         deployed consumer (published P-W1b) on a measured competence
         axis.
  C7   — identification-gap reader: probe-rule EVSI
         evsi(s) = ho(s, m_real) - ho(s, m_now) on the odd half
         (m_real's selection mark is disjoint from every evaluation
         mark — channel-A/B immune, verified ThreeAxes §2.3), plus the
         gap opp_split - evsi (descriptive; opp_split is published).
  SA   — certified-injection Stage A: planted per-candidate values
         mu ~ N(0, S^2) added to the ARCHIVED arrays (real noise, not
         the sim suite's Gaussian iid). Review B1: the additive plant
         makes (opp' - truth) an algebraic identity in the archived
         noise, so there is NO recovery-bias estimand; the registered
         outputs are the DETECTION curve + MDS80 (reported in sd units
         AND opportunity units ~1.427*S, review B2 — every ceiling
         comparison is in opportunity units), the zero-plant identity
         gate, and the descriptive selector-efficiency curve
         (dv3-primary — dv3 is the null family the "calibrated
         instrument found nothing" sentence is about).
  CE   — harvest-side power calibration: realized-dispersion resampling
         of the per-cell harvest values (centered residuals + shift),
         giving the measured MDE80 for P-W1b-style fires at n=32; the
         fixture-based cons_edge sweep runs as a machinery-validation
         secondary.
  AB   — run-level purchase A/B admissibility gate (dispersion-only):
         MDE80 for a two-arm run-level contrast from realized eval-
         return spread, computed PER DOMAIN (review M12: a mixed-domain
         glob would make run_sd a between-task sd and guarantee DOA);
         t-based power factors; DEAD-ON-ARRIVAL if no domain's n=16/arm
         can see a 5%-of-mean effect.

Governance hardening (review B3/B4/M11/M13): refuses under python -O
(the inherited w1_read gates are asserts); the ONE-read guard is
path-pinned to artifacts/p2_tier1_20260821; anchors are checked both
relatively (recomputed == published bit-exact) and absolutely (published
== the disclosed constants); the output carries full substrate
provenance (per-file shas + reader/module shas).

Anchors/identity gates (recomputed-from-arrays quantities must equal the
published w1_read outputs BIT-EXACTLY, else the wiring is wrong and the
read refuses): pooled P-W1a point per family; Stage A at S=0.

Usage:
  python -m analysis.p2_tier1_bundle_read --component main \
      --labels <dir> --published artifacts/w1_read_20260811 --output <dir>
  python -m analysis.p2_tier1_bundle_read --component ab \
      --scores '<glob of scores.jsonl>' --output <dir>
  python -m analysis.p2_tier1_bundle_read --selfcheck
"""

import argparse
import glob as globmod
import json
import os
import re

import numpy as np

from analysis import w1_read

ALPHA = 0.05
FAMILIES = ('dv3', 'tm2')
# Review B4: the ONE-read guard is PATH-PINNED — a fresh --output would
# otherwise defeat it. Selfcheck fixtures bypass via _selfcheck only.
REGISTERED_OUTPUT = 'artifacts/p2_tier1_20260821'
# Review M11: absolute anchors (the disclosed published points) — a
# doctored (labels, published) pair that is merely self-consistent
# refuses against these.
DISCLOSED_ANCHORS = dict(dv3=0.004805, tm2=0.061719)
ANCHOR_ABS_TOL = 5e-5
EXPECT_LABELS_BASENAME = 'w1_labels'
# CCD pins
CCD_SEED = 20260821
CCD_SIGMAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)   # raw G units
CCD_DRAWS = 64
# Stage A pins. Review B1: the additive plant makes (opp' - truth) an
# algebraic IDENTITY equal to the archived noise-selection term, so no
# "recovery bias" estimand exists — Stage A's registered outputs are
# the DETECTION curve (+ MDS80), the zero-plant identity gate, and the
# descriptive selector-efficiency curve recovered/ceiling. Review B2:
# the sd grid is converted to OPPORTUNITY units (mean[mu.max - mu.mn],
# ~1.427*S at M=8) and every ceiling comparison is made in those units.
SA_SEED = 20260822
SA_PERM_SEED = (SA_SEED, 777)   # review m24: pinned, registered
SA_GRID = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)      # planted sd, G units
SA_PLANTS = 50
SA_PERM_B = 4096
CEILING_BAR = 0.2026            # archived screen, opportunity units
# CE pins
CE_SHIFTS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8)  # G units
CE_MC = 500
CE_FAST_B = 2048
CE_FIXTURE_GRID = (0.0, 0.1, 0.2, 0.4, 0.8)
CE_FIXTURE_MC = 200
# AB pins
AB_TAIL_EPISODES = 20
AB_ARMS_N = (8, 16, 32)
AB_MIN_RUNS = 8
AB_EFFECT_FRAC = 0.05    # policy-relevant minimum effect: 5% of mean
# review m21: t-based 80%-power factors (t_{.975,df} + t_{.80,df} at
# df = 2n-2), not the normal 2.8 which understates ~7% at n=8
AB_FACTORS = {8: 3.013, 16: 2.896, 32: 2.848}
AB_DOMAINS = ('cup', 'finger', 'reacher', 'walker', 'cheetah')
ANCHOR_ATOL = 1e-9       # identity gates: recomputed == published


def _stat(vals, clusters, b=None):
  b = b or w1_read.B_BOOT
  vals = np.asarray(vals, float)
  point, ci = w1_read.bca(vals, clusters, b=b)
  return dict(point=point, ci=list(ci), perm_p=w1_read.perm_p(vals, b=b),
              n_cells=int(np.isfinite(vals).sum()))


def _fires(blk):
  return bool(blk['ci'][0] > 0 and blk['perm_p'] < ALPHA)


def refuse(msg):
  raise SystemExit(f'READ REFUSED: {msg}')


# --------------------------------------------------------------------------
# Loading — w1_read.load_family for every inherited gate, plus a second
# aligned pass for the arrays it does not surface (m_real).
# --------------------------------------------------------------------------

def load_bundle(labels_dir, family):
  # review M11: substrate identity — the committed W1 label directory
  if os.path.basename(os.path.normpath(labels_dir)) != (
      EXPECT_LABELS_BASENAME):
    refuse(f'labels dir basename {labels_dir!r} != registered '
           f'{EXPECT_LABELS_BASENAME!r}')
  cells, dups = w1_read.load_family(labels_dir, family)
  if len(cells) != w1_read.EXPECT_CELLS:
    refuse(f'{family}: {len(cells)} cells != {w1_read.EXPECT_CELLS}')
  w1_read.dup_gate(dups)
  rex = w1_read.FILE_RES[family]
  extra = {}
  for path in sorted(globmod.glob(os.path.join(labels_dir, '*.npz'))):
    m = rex.match(os.path.basename(path))
    if not m:
      continue
    z = np.load(path, allow_pickle=True)
    if 'm_real' not in z.files:
      refuse(f'{os.path.basename(path)}: m_real missing (review m14)')
    key = (m.group(1), m.group(2), int(m.group(3)))
    extra[key] = dict(m_real=np.asarray(z['m_real'], int))
  for key, e in cells.items():
    if key not in extra:
      refuse(f'{family}: aligned pass missing {key}')
    if len(extra[key]['m_real']) != len(e['mn']):
      refuse(f'{family}: m_real length mismatch at {key}')
    e['m_real'] = extra[key]['m_real']
  return cells


def _published_anchor(published_dir, family):
  path = os.path.join(published_dir, family, f'w1_{family}.json')
  if not os.path.exists(path):
    refuse(f'published anchor missing: {path}')
  with open(path) as f:
    pub = json.load(f)
  return pub


def anchor_gate(cells, pub, family, disclosed=None):
  """Recomputed pooled P-W1a must equal the published point bit-exactly
  (same arrays, same code path) AND the published point must match the
  DISCLOSED constant (review M11: self-consistency alone admits a
  doctored labels+published pair) — the wiring witness for every
  downstream component."""
  opp = [w1_read._cell_split_opp(e) for e in cells.values()]
  recomputed = float(np.mean(opp))
  published = float(pub['p_w1a_split_opportunity']['point'])
  if abs(recomputed - published) > ANCHOR_ATOL:
    refuse(f'{family}: recomputed P-W1a {recomputed!r} != published '
           f'{published!r} — wiring/substrate mismatch')
  want = (DISCLOSED_ANCHORS if disclosed is None else disclosed)[family]
  if abs(published - want) > ANCHOR_ABS_TOL:
    refuse(f'{family}: published anchor {published!r} != disclosed '
           f'{want!r} — doctored or wrong published artifact')
  return recomputed


# --------------------------------------------------------------------------
# CCD — consumer-competence dial
# --------------------------------------------------------------------------

def _halves(e):
  g = e['g']
  return g[:, 0::2].mean(1), g[:, 1::2].mean(1)   # even (select), odd (eval)


def ccd_component(cells, pub, family, b=None):
  rng = np.random.default_rng([CCD_SEED, FAMILIES.index(family)])
  cl = ['%s_%s_%d' % k for k in cells]
  curve = {}
  entries = list(cells.values())
  halves = [_halves(e) for e in entries]
  for sigma in CCD_SIGMAS:
    vals = []
    for (even, odd) in halves:
      s_idx = np.arange(len(even))
      hs = []
      for _ in range(CCD_DRAWS):
        eps = rng.standard_normal(even.shape)
        sel = (even + sigma * eps).argmax(1)
        hs.append(np.mean(odd[s_idx, sel] - odd.mean(1)))
      vals.append(float(np.mean(hs)))
    curve[str(sigma)] = _stat(vals, cl, b=b)
  # uniform-random chooser (sigma = inf endpoint; truth 0)
  vals = []
  for (even, odd) in halves:
    s_idx = np.arange(len(even))
    hs = []
    for _ in range(CCD_DRAWS):
      sel = rng.integers(0, odd.shape[1], len(odd))
      hs.append(np.mean(odd[s_idx, sel] - odd.mean(1)))
    vals.append(float(np.mean(hs)))
  curve['rand'] = _stat(vals, cl, b=b)
  pc = _fires(curve['0.0'])
  # placement of the deployed consumer (published P-W1b) on the curve.
  # Review M6: the rand endpoint (sigma = inf) is part of the
  # interpolation domain; a harvest landing between the last grid
  # point and rand is ON-CURVE-TAIL (sigma unidentifiable there —
  # the grid's upper points collapse toward rand at the realized
  # noise scale, disclosed).
  harvest = float(pub['p_w1b_consumer_harvest']['point'])
  pts = [(s, curve[str(s)]['point']) for s in CCD_SIGMAS]
  rand_pt = curve['rand']['point']
  grid_vals = [p for _, p in pts]
  monotone = all(a >= b for a, b in zip(grid_vals[:-1], grid_vals[1:]))
  lo = min(grid_vals + [rand_pt])
  hi = max(grid_vals + [rand_pt])
  if harvest < rand_pt and harvest < lo:
    placement, sigma_equiv = 'BELOW-RANDOM', None
  elif harvest > hi:
    placement, sigma_equiv = 'ABOVE-PERFECT', None
  else:
    sigma_equiv = _interp_sigma(pts, harvest)
    if sigma_equiv is not None:
      placement = 'ON-CURVE'
    elif min(grid_vals[-1], rand_pt) <= harvest <= max(
        grid_vals[-1], rand_pt):
      placement = 'ON-CURVE-TAIL'
    else:
      placement = 'BELOW-RANDOM' if harvest < rand_pt else 'ON-CURVE'
  return dict(curve=curve, pc_fires=pc,
              curve_monotone=bool(monotone),
              verdict=('CCD-PC-FIRES' if pc else 'CCD-PC-FAILS'),
              consumer_harvest_published=harvest,
              placement=placement, sigma_equiv=sigma_equiv)


def _interp_sigma(pts, target):
  """First sigma (linear interpolation on the monotone-sorted curve)
  whose value crosses `target`. Descriptive only."""
  for (s0, p0), (s1, p1) in zip(pts[:-1], pts[1:]):
    lo, hi = min(p0, p1), max(p0, p1)
    if lo <= target <= hi and p0 != p1:
      return float(s0 + (s1 - s0) * (target - p0) / (p1 - p0))
  return None


# --------------------------------------------------------------------------
# C7 — identification-gap reader
# --------------------------------------------------------------------------

def c7_component(cells, family, b=None):
  cl = ['%s_%s_%d' % k for k in cells]
  evsi_cells, gap_cells = [], []
  for e in cells.values():
    _, odd = _halves(e)
    i = np.arange(len(odd))
    evsi = np.mean(odd[i, e['m_real']] - odd[i, e['mn']])
    evsi_cells.append(float(evsi))
    gap_cells.append(float(w1_read._cell_split_opp(e) - evsi))
  blk = _stat(evsi_cells, cl, b=b)
  neg = bool(blk['ci'][1] < 0 and blk['perm_p'] < ALPHA)
  verdict = ('PROBE-IDENTIFIES' if _fires(blk) else
             'PROBE-MISLEADS' if neg else 'PROBE-BLIND')
  return dict(probe_evsi=blk, verdict=verdict,
              identification_gap=_stat(gap_cells, cl, b=b))


# --------------------------------------------------------------------------
# Stage A — certified injection on real noise
# --------------------------------------------------------------------------

def _fast_perm_p(cell_vals, rng, b=SA_PERM_B):
  vals = np.asarray(cell_vals, float)
  vals = vals[np.isfinite(vals)]              # review m19: match perm_p
  if len(vals) == 0:
    return 1.0
  obs = abs(vals.mean())
  flips = rng.choice([-1.0, 1.0], size=(b, len(vals)))
  null = np.abs(flips @ vals) / len(vals)
  return float((np.sum(null >= obs - 1e-15) + 1) / (b + 1))


def stagea_component(cells, family, plants=SA_PLANTS, grid=SA_GRID,
                     _break_identity=False):
  """Review B1: the additive plant makes (opp' - truth) an algebraic
  identity in the archived noise, so there is NO recovery-bias
  estimand. Registered outputs: the DETECTION curve + MDS80 (in sd
  units AND opportunity units, review B2), the zero-plant identity
  gate, and the descriptive selector-efficiency curve
  recovered / planted-ceiling."""
  rng = np.random.default_rng([SA_SEED, FAMILIES.index(family)])
  entries = list(cells.values())
  base_opp = np.array([w1_read._cell_split_opp(e) for e in entries])
  out = dict(detect={}, ceiling_units={}, recovered={}, efficiency={},
             zero_identity=None)
  perm_rng = np.random.default_rng(list(SA_PERM_SEED))
  for S in grid:
    fires, recs, ceils = [], [], []
    n_p = 1 if S == 0.0 else plants
    for _ in range(n_p):
      opp_cells, ceil_cells = [], []
      for e in entries:
        g = e['g']
        Sn, R, M = g.shape
        mu = (S * rng.standard_normal((Sn, M))) if S else np.zeros((Sn, M))
        gp = g + mu[:, None, :]
        even = gp[:, 0::2].mean(1)
        odd = gp[:, 1::2].mean(1)
        sel = even.argmax(1)
        i = np.arange(Sn)
        opp_cells.append(float(np.mean(odd[i, sel] - odd[i, e['mn']])))
        # planted ceiling in OPPORTUNITY units (review B2): what a
        # perfect selector of the plant would harvest over m_now
        ceil_cells.append(float(np.mean(mu.max(1) - mu[i, e['mn']])))
      opp_cells = np.asarray(opp_cells)
      fires.append(_fast_perm_p(opp_cells, perm_rng) < ALPHA)
      recs.append(float(np.mean(opp_cells)))
      ceils.append(float(np.mean(ceil_cells)))
      if S == 0.0:
        if _break_identity:
          opp_cells = opp_cells + 1.0
        ident = bool(np.array_equal(opp_cells, base_opp))
        out['zero_identity'] = 'PASS' if ident else 'FAIL'
        if not ident:
          refuse(f'{family}: Stage A zero-plant does not reproduce the '
                 'archived estimator bit-exactly')
    out['detect'][str(S)] = float(np.mean(fires))
    out['ceiling_units'][str(S)] = float(np.mean(ceils))
    out['recovered'][str(S)] = float(np.mean(recs))
    out['efficiency'][str(S)] = (
        float((np.mean(recs) - base_opp.mean()) / np.mean(ceils))
        if S > 0 else None)
  mds = next((S for S in grid if S > 0 and out['detect'][str(S)] >= 0.8),
             None)
  out['mds80_sd_units'] = mds
  # review B2: the registered ceiling comparison is in OPPORTUNITY
  # units (~1.427*S at M=8); None => DETECTION-FLOOR-ABOVE-GRID and
  # the ceiling sentence is NOT licensed (review m17)
  out['mds80_ceiling_units'] = (
      out['ceiling_units'][str(mds)] if mds is not None else None)
  out['ceiling_bar'] = CEILING_BAR
  out['floor_verdict'] = (
      'DETECTION-FLOOR-ABOVE-GRID' if mds is None else
      'FLOOR-BELOW-BAR' if out['mds80_ceiling_units'] <= CEILING_BAR
      else 'FLOOR-ABOVE-BAR')
  return out


# --------------------------------------------------------------------------
# CE — harvest-side power calibration
# --------------------------------------------------------------------------

def _fast_fire_rate(resampled, rng, b=CE_FAST_B, chunk=64):
  """resampled: (T, n) replicate cell-value matrices. Fire = vectorized
  sign-flip perm p < ALPHA AND percentile bootstrap CI lower > 0.
  Registered approximations (disclosed): percentile CI, not BCa, and
  perm B=2048 — power-curve resolution, not read-grade inference.
  Review m20: flip/bootstrap matrices are drawn FRESH per chunk of
  replicates so the per-point MC error is the nominal sqrt(p(1-p)/T)."""
  T, n = resampled.shape
  fires = []
  for start in range(0, T, chunk):
    block = resampled[start:start + chunk]
    means = block.mean(1)
    flips = rng.choice([-1.0, 1.0], size=(b, n))
    null = np.abs(block @ flips.T) / n              # (chunk, b)
    perm_p = (np.sum(null >= np.abs(means)[:, None] - 1e-15, 1) + 1) / (
        b + 1)
    idx = rng.integers(0, n, size=(b, n))
    boots = block[:, idx].mean(2)                   # (chunk, b)
    lo = np.percentile(boots, 100 * ALPHA / 2, axis=1)
    fires.append((perm_p < ALPHA) & (lo > 0))
  return float(np.mean(np.concatenate(fires)))


def ce_component(cells, family):
  rng = np.random.default_rng([20260824, FAMILIES.index(family)])
  harv = np.array([w1_read._cell_harvest(e) for e in cells.values()])
  resid = harv - harv.mean()                         # centered residuals
  n = len(resid)
  power = {}
  for shift in CE_SHIFTS:
    idx = rng.integers(0, n, size=(CE_MC, n))
    resampled = resid[idx] + shift
    power[str(shift)] = _fast_fire_rate(resampled, rng)
  mde = next((s for s in CE_SHIFTS if power[str(s)] >= 0.8), None)
  # review M10: the power curve conditions on ONE realized sd whose own
  # relative s.e. is ~1/sqrt(2(n-1)) ~ 13% at n=32 — report the sd's
  # bootstrap band so no draft quotes MDE80 as a point fact
  sd_boots = np.array([
      resid[rng.integers(0, n, n)].std(ddof=1) for _ in range(2000)])
  sd_ci = [float(np.percentile(sd_boots, 2.5)),
           float(np.percentile(sd_boots, 97.5))]
  # fixture-based sweep: machinery validation on the frozen generator;
  # fixture power at shift 0 is the machinery SIZE witness (review m18)
  fix_rng = np.random.default_rng(20260825)
  fixture = {}
  for ce in CE_FIXTURE_GRID:
    vals = np.array([
        [w1_read._cell_harvest(w1_read._mk_cell(
            fix_rng, S=200, R=8, M=8, cons_edge=ce)) for _ in range(n)]
        for _ in range(CE_FIXTURE_MC)])
    fixture[str(ce)] = _fast_fire_rate(vals, fix_rng)
  return dict(cell_sd=float(resid.std(ddof=1)), cell_sd_ci=sd_ci,
              power=power, mde80_real=mde, fixture_power=fixture,
              n_cells=n)


# --------------------------------------------------------------------------
# AB — run-level purchase A/B admissibility gate (dispersion-only)
# --------------------------------------------------------------------------

def _ab_one_domain(paths, label):
  run_means = []
  for path in paths:
    scores = []
    with open(path) as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        row = json.loads(line)
        for key in ('episode/score', 'epstats/score', 'score'):
          if key in row:
            scores.append(float(row[key]))
            break
    if len(scores) < AB_TAIL_EPISODES:
      refuse(f'AB gate [{label}]: {path} has {len(scores)} episodes < '
             f'{AB_TAIL_EPISODES}')
    run_means.append(float(np.mean(scores[-AB_TAIL_EPISODES:])))
  run_means = np.asarray(run_means)
  sd = float(run_means.std(ddof=1))
  mean = float(run_means.mean())
  mde = {str(n): float(AB_FACTORS[n] * sd * np.sqrt(2.0 / n))
         for n in AB_ARMS_N}
  bound = AB_EFFECT_FRAC * abs(mean)
  admissible = mde['16'] <= bound
  return dict(n_runs=len(run_means), run_mean=mean, run_sd=sd,
              run_sd_rel_se=float(1.0 / np.sqrt(
                  2.0 * (len(run_means) - 1))),
              mde80_per_arm_n=mde, effect_bound=bound,
              verdict=('AB-ADMISSIBLE' if admissible
                       else 'AB-DEAD-ON-ARRIVAL'))


def _path_domain(path):
  found = [d for d in AB_DOMAINS if d in os.path.basename(path).lower()
           or d in path.lower()]
  return found[0] if len(found) == 1 else None


def ab_component(scores_globs):
  """Review M12: one glob PER DOMAIN (comma-separated); a glob mixing
  domains would turn run_sd into a between-task sd, silently
  guaranteeing DEAD-ON-ARRIVAL. Overall verdict: ADMISSIBLE iff any
  domain is admissible."""
  out = dict(domains={})
  for glob_str in scores_globs.split(','):
    glob_str = glob_str.strip()
    paths = sorted(globmod.glob(glob_str))
    if len(paths) < AB_MIN_RUNS:
      refuse(f'AB gate: {len(paths)} score files < {AB_MIN_RUNS} runs '
             f'for glob {glob_str!r}')
    doms = {_path_domain(p) for p in paths}
    if len(doms) != 1:
      refuse(f'AB gate: glob {glob_str!r} spans domains {doms} — '
             'run-level sd must be within-domain (review M12)')
    label = doms.pop() or 'unspecified'
    if label in out['domains']:
      refuse(f'AB gate: domain {label!r} appears in two globs')
    out['domains'][label] = _ab_one_domain(paths, label)
  out['verdict'] = ('AB-ADMISSIBLE' if any(
      d['verdict'] == 'AB-ADMISSIBLE' for d in out['domains'].values())
      else 'AB-DEAD-ON-ARRIVAL')
  return out


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------

def _guard_output(output, name, _selfcheck):
  if not _selfcheck and os.path.abspath(output) != os.path.abspath(
      REGISTERED_OUTPUT):
    refuse(f'output {output!r} != registered {REGISTERED_OUTPUT!r} — '
           'the ONE-read guard is path-pinned (review B4)')
  os.makedirs(output, exist_ok=True)
  path = os.path.join(output, name)
  if os.path.exists(path):
    refuse(f'{path} exists — ONE read execution is registered')
  return path


def _provenance(labels_dir, published_dir):
  """Review M13: tie the read to its substrate."""
  files = sorted(globmod.glob(os.path.join(labels_dir, '*.npz')))
  return dict(
      labels_dir=os.path.abspath(labels_dir),
      published_dir=os.path.abspath(published_dir),
      n_label_files=len(files),
      label_shas={os.path.basename(p): sha256_file(p) for p in files},
      reader_sha=sha256_file(__file__),
      w1_read_sha=sha256_file(w1_read.__file__))


def sha256_file(path):
  import hashlib
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def run_main(labels_dir, published_dir, output, _selfcheck=False):
  path = _guard_output(output, 'p2t1_main.json', _selfcheck)
  out = {}
  for family in FAMILIES:
    cells = load_bundle(labels_dir, family)
    pub = _published_anchor(published_dir, family)
    anchor = anchor_gate(cells, pub, family)
    stagea = stagea_component(cells, family)
    ccd = ccd_component(cells, pub, family)
    # review M7: PC-FAILS licenses the substrate-empty reading only if
    # the instrument's measured floor could have seen ceiling-level
    # signal; otherwise the finding is about power, not the substrate
    if ccd['verdict'] == 'CCD-PC-FAILS':
      floor = stagea.get('mds80_ceiling_units')
      if floor is None or floor > CEILING_BAR:
        ccd['verdict'] = 'CCD-PC-UNDERPOWERED'
    out[family] = dict(
        anchor_p_w1a=anchor,
        ccd=ccd,
        c7=c7_component(cells, family),
        stagea=stagea,
        ce=ce_component(cells, family))
  out['provenance'] = _provenance(labels_dir, published_dir)
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)
  return out


def run_ab(scores_globs, output, _selfcheck=False):
  path = _guard_output(output, 'p2t1_ab.json', _selfcheck)
  out = ab_component(scores_globs)
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)
  return out


# --------------------------------------------------------------------------
# Selfcheck — fixtures for every branch + refusal legs
# --------------------------------------------------------------------------

def _fixture_cells(rng, signal=0.0, probe_mode='blind', n=32, S=60, R=8,
                   M=8, cons_edge=0.0):
  """Cells with the real key layout. signal: true per-candidate spread.
  probe_mode: 'oracle' (m_real = argmax truth), 'blind' (uniform),
  'anti' (argmin truth)."""
  cells = {}
  keys = [(d, e, s) for d in ('cup', 'finger') for e in ('e1', 'e4')
          for s in range(31, 39)][:n]
  for k in keys:
    mu = rng.normal(0, signal, (S, M)) if signal else np.zeros((S, M))
    g = mu[:, None] + rng.normal(0, 1.0, (S, R, M))
    gc = mu.mean(1)[:, None] + cons_edge + rng.normal(0, 1.0, (S, R))
    m_real = (mu.argmax(1) if probe_mode == 'oracle' else
              mu.argmin(1) if probe_mode == 'anti' else
              rng.integers(0, M, S))
    cells[k] = dict(g=g, gc=gc, rs=np.zeros((S, M)), rr=np.zeros((S, M)),
                    mn=rng.integers(0, M, S), m_real=np.asarray(m_real),
                    dup=np.zeros(S, bool))
  return cells


def selfcheck():
  rng = np.random.default_rng(0)
  B = 500

  # --- CCD: planted signal -> sigma=0 fires, curve decreasing, rand ~ 0
  cells = _fixture_cells(rng, signal=1.0)
  pub = dict(p_w1b_consumer_harvest=dict(point=-0.3),
             p_w1a_split_opportunity=dict(point=0.0))
  ccd = ccd_component(cells, pub, 'dv3', b=B)
  assert ccd['pc_fires'], ccd['curve']['0.0']
  assert ccd['curve']['0.0']['point'] > ccd['curve']['8.0']['point']
  assert abs(ccd['curve']['rand']['point']) < 0.1, ccd['curve']['rand']
  assert ccd['placement'] == 'BELOW-RANDOM', ccd['placement']
  assert ccd['curve_monotone'] in (True, False)
  # placement legs (review M8): ON-CURVE with identifiable sigma;
  # ON-CURVE-TAIL between last grid point and rand; ABOVE-PERFECT
  mid = 0.5 * (ccd['curve']['0.0']['point']
               + ccd['curve']['1.0']['point'])
  ccd_on = ccd_component(cells, dict(
      p_w1b_consumer_harvest=dict(point=mid),
      p_w1a_split_opportunity=dict(point=0.0)), 'dv3', b=B)
  assert ccd_on['placement'] == 'ON-CURVE', ccd_on['placement']
  assert ccd_on['sigma_equiv'] is not None
  tail = 0.5 * (ccd['curve']['8.0']['point']
                + ccd['curve']['rand']['point'])
  ccd_tail = ccd_component(cells, dict(
      p_w1b_consumer_harvest=dict(point=tail),
      p_w1a_split_opportunity=dict(point=0.0)), 'dv3', b=B)
  assert ccd_tail['placement'] in ('ON-CURVE-TAIL', 'ON-CURVE'), (
      ccd_tail['placement'])
  ccd_hi = ccd_component(cells, dict(
      p_w1b_consumer_harvest=dict(point=99.0),
      p_w1a_split_opportunity=dict(point=0.0)), 'dv3', b=B)
  assert ccd_hi['placement'] == 'ABOVE-PERFECT'
  # noise-only: PC must fail
  cells0 = _fixture_cells(rng)
  ccd0 = ccd_component(cells0, dict(
      p_w1b_consumer_harvest=dict(point=0.0),
      p_w1a_split_opportunity=dict(point=0.0)), 'dv3', b=B)
  assert not ccd0['pc_fires'], ccd0['curve']['0.0']

  # --- C7: oracle probe -> IDENTIFIES; blind -> BLIND; anti -> MISLEADS
  for mode, want in (('oracle', 'PROBE-IDENTIFIES'),
                     ('blind', 'PROBE-BLIND'),
                     ('anti', 'PROBE-MISLEADS')):
    c7 = c7_component(_fixture_cells(rng, signal=1.0, probe_mode=mode),
                      'dv3', b=B)
    assert c7['verdict'] == want, (mode, c7)

  # --- Stage A (review B1/B2 redesign): zero-plant identity; rising
  # detection; opportunity-units conversion ~1.427*S at M=8;
  # efficiency in a sane band
  cells_sa = _fixture_cells(rng)
  sa = stagea_component(cells_sa, 'dv3', plants=12)
  assert sa['zero_identity'] == 'PASS'
  assert sa['detect']['2.0'] > sa['detect']['0.05'], sa['detect']
  assert sa['detect']['2.0'] >= 0.8, sa['detect']
  assert sa['mds80_sd_units'] is not None
  assert sa['mds80_ceiling_units'] is not None
  ratio = sa['ceiling_units']['0.5'] / 0.5
  assert 1.2 < ratio < 1.65, ratio        # E[max of 8] ~ 1.4236
  assert -0.2 < sa['efficiency']['2.0'] < 1.2, sa['efficiency']
  assert sa['floor_verdict'] in ('FLOOR-BELOW-BAR', 'FLOOR-ABOVE-BAR')
  # grid too low -> mds None + DETECTION-FLOOR-ABOVE-GRID (review m17)
  sa_low = stagea_component(cells_sa, 'dv3', plants=6,
                            grid=(0.0, 0.01))
  assert sa_low['mds80_sd_units'] is None
  assert sa_low['floor_verdict'] == 'DETECTION-FLOOR-ABOVE-GRID'
  # zero-identity refusal (review M8)
  try:
    stagea_component(cells_sa, 'dv3', plants=2, _break_identity=True)
    raise AssertionError('zero-identity refusal failed to trip')
  except SystemExit as e:
    assert 'bit-exactly' in str(e), e

  # --- CE: power monotone in shift; power(0) ~ size; sd band present
  ce = ce_component(cells0, 'dv3')
  assert ce['power']['0.8'] > ce['power']['0.0'], ce['power']
  assert ce['power']['0.0'] < 0.15, ce['power']
  assert ce['fixture_power']['0.8'] > ce['fixture_power']['0.0']
  assert ce['fixture_power']['0.0'] < 0.15, ce['fixture_power']
  assert ce['cell_sd_ci'][0] < ce['cell_sd'] < ce['cell_sd_ci'][1]

  # --- AB gate: per-domain formula + branches + mixed-domain refusal
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    for tight, dom in ((0.5, 'cup'), (200.0, 'finger')):
      for i in range(10):
        with open(os.path.join(tmp, f'{dom}_run{i}.jsonl'), 'w') as f:
          for ep in range(AB_TAIL_EPISODES + 5):
            f.write(json.dumps({'episode/score':
                                800.0 + tight * ((i % 5) - 2)}) + '\n')
    ok = ab_component(os.path.join(tmp, 'cup_*.jsonl'))
    assert ok['verdict'] == 'AB-ADMISSIBLE', ok
    assert ok['domains']['cup']['mde80_per_arm_n']['8'] > (
        ok['domains']['cup']['mde80_per_arm_n']['32'])
    doa = ab_component(os.path.join(tmp, 'finger_*.jsonl'))
    assert doa['verdict'] == 'AB-DEAD-ON-ARRIVAL', doa
    both = ab_component(os.path.join(tmp, 'cup_*.jsonl') + ',' +
                        os.path.join(tmp, 'finger_*.jsonl'))
    assert both['verdict'] == 'AB-ADMISSIBLE' and len(
        both['domains']) == 2
    try:
      ab_component(os.path.join(tmp, '*run*.jsonl'))
      raise AssertionError('mixed-domain refusal failed to trip')
    except SystemExit as e:
      assert 'spans domains' in str(e), e
    try:
      ab_component(os.path.join(tmp, 'nomatch_*.jsonl'))
      raise AssertionError('AB run-count refusal failed to trip')
    except SystemExit as e:
      assert 'READ REFUSED' in str(e), e

  # --- anchor gates: relative trip, absolute trip, clean pass
  true_opp = float(np.mean([w1_read._cell_split_opp(e)
                            for e in cells0.values()]))
  try:
    anchor_gate(cells0, dict(p_w1a_split_opportunity=dict(point=99.0)),
                'dv3', disclosed=dict(dv3=99.0))
    raise AssertionError('relative anchor gate failed to trip')
  except SystemExit as e:
    assert 'wiring/substrate mismatch' in str(e), e
  try:
    anchor_gate(cells0,
                dict(p_w1a_split_opportunity=dict(point=true_opp)),
                'dv3', disclosed=dict(dv3=true_opp + 1.0))
    raise AssertionError('absolute anchor gate failed to trip')
  except SystemExit as e:
    assert 'doctored' in str(e), e
  anchor_gate(cells0, dict(p_w1a_split_opportunity=dict(point=true_opp)),
              'dv3', disclosed=dict(dv3=true_opp))
  # published-artifact-missing refusal
  try:
    _published_anchor('/nonexistent', 'dv3')
    raise AssertionError('published-missing refusal failed to trip')
  except SystemExit as e:
    assert 'published anchor missing' in str(e), e

  # --- ONE-read guard + output path pin (review B4)
  with tempfile.TemporaryDirectory() as tmp:
    open(os.path.join(tmp, 'p2t1_main.json'), 'w').close()
    try:
      run_main('/nonexistent', '/nonexistent', tmp, _selfcheck=True)
      raise AssertionError('ONE-read guard failed to trip')
    except SystemExit as e:
      assert 'ONE read execution' in str(e), e
    try:
      run_main('/nonexistent', '/nonexistent',
               os.path.join(tmp, 'other'))
      raise AssertionError('output path pin failed to trip')
    except SystemExit as e:
      assert 'path-pinned' in str(e), e
    open(os.path.join(tmp, 'p2t1_ab.json'), 'w').close()
    try:
      run_ab('x', tmp, _selfcheck=True)
      raise AssertionError('AB ONE-read guard failed to trip')
    except SystemExit as e:
      assert 'ONE read execution' in str(e), e
  # labels-dir basename pin (review M11)
  try:
    load_bundle('/tmp/not_the_registered_dir', 'dv3')
    raise AssertionError('labels basename pin failed to trip')
  except SystemExit as e:
    assert 'basename' in str(e), e

  print('SELFCHECK PASS (p2_tier1_bundle_read: CCD signal-fires + '
        'monotone curve + rand~0 + placements BELOW-RANDOM/ON-CURVE/'
        'TAIL/ABOVE-PERFECT + noise no-fire; C7 oracle/blind/anti; '
        'Stage A zero-identity + refusal + rising detection + '
        'opportunity-units ~1.43x + efficiency band + '
        'floor-above-grid; CE power monotone + size witness + sd band; '
        'AB per-domain admissible/DOA/mixed-domain/run-count; anchor '
        'relative+absolute trips + pass + published-missing; ONE-read '
        'guards + output path pin + labels basename pin)')


def main():
  if not __debug__:
    refuse('reader must run with assertions enabled — python -O / '
           'PYTHONOPTIMIZE strips every inherited w1_read gate '
           '(review B3)')
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--component', choices=['main', 'ab'])
  p.add_argument('--labels')
  p.add_argument('--published')
  p.add_argument('--scores')
  p.add_argument('--output')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if args.component == 'main':
    if not (args.labels and args.published and args.output):
      p.error('--labels, --published, --output required for main')
    run_main(args.labels, args.published, args.output)
  elif args.component == 'ab':
    if not (args.scores and args.output):
      p.error('--scores and --output required for ab')
    run_ab(args.scores, args.output)
  else:
    p.error('--component required (or --selfcheck)')


if __name__ == '__main__':
  main()
