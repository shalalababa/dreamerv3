"""Gate D0 aggregation discipline and pre-registered P1-P4 evaluation.

All confirmatory reads are aggregates (quadrant occupancy, correlations,
reversal rates) with episode-clustered uncertainty; never per-state reads
(per-state estimator SNR is ~1.5 at K=5; note App. A.5). Quadrant
thresholds come from the *matching dose-zero (E1) run* -- same task, same
pipeline, same seed index (note App. A.5).

Provisional gate constants below mirror note App. A.6 [prov.] and must be
confirmed at the 1 Oct 2026 freeze.
"""

import dataclasses

import numpy as np
from scipy import stats

# --- Pre-registered [prov.] constants (note App. A.6); confirm at freeze. ---
P1_CORR_DROP = 0.3      # required Spearman drop E1 -> E2 at the top dose
P1_QI_TOP = 0.05        # required Q-I occupancy at the highest dose
P1_QI_E1 = 0.01         # required Q-I occupancy ceiling in E1
NEAR_TIE_DECILE = 0.1   # P2/P4 near-tie definition: bottom decile of gap


def quadrant_thresholds(e1_signals, keys=('udyn', 'aflip', 'evpi_split')):
  """High/low cutoffs = 75th/25th percentiles of the matching E1 run."""
  return {k: (np.percentile(e1_signals[k], 25),
              np.percentile(e1_signals[k], 75)) for k in keys}


def quadrant_occupancy(x, y, xcuts, ycuts):
  """2x2 occupancy fractions with percentile cutoffs (xcuts = (lo, hi)).

  Q-I: x high, y low. Q-II: x low, y high. Q-III: both low. Q-IV: both
  high. Middle-band states fall in no quadrant, so fractions need not sum
  to one; each is a fraction of all N states.
  """
  xlo, xhi = xcuts
  ylo, yhi = ycuts
  return dict(
      QI=float(((x >= xhi) & (y <= ylo)).mean()),
      QII=float(((x <= xlo) & (y >= yhi)).mean()),
      QIII=float(((x <= xlo) & (y <= ylo)).mean()),
      QIV=float(((x >= xhi) & (y >= yhi)).mean()),
  )


def cluster_bootstrap(fn, episodes, n_boot=500, seed=0):
  """SE of a statistic under resampling of whole episodes (clustered SE)."""
  episodes = np.asarray(episodes)
  uniq = np.unique(episodes)
  index = {e: np.flatnonzero(episodes == e) for e in uniq}
  rng = np.random.default_rng(seed)
  vals = []
  for _ in range(n_boot):
    chosen = rng.choice(uniq, len(uniq), replace=True)
    idx = np.concatenate([index[e] for e in chosen])
    vals.append(fn(idx))
  vals = np.asarray(vals, np.float64)
  return float(np.nanstd(vals, ddof=1))


def spearman(x, y, episodes=None, n_boot=500, seed=0):
  """Spearman rho with an episode-clustered bootstrap SE."""
  rho = float(stats.spearmanr(x, y).statistic)
  if episodes is None:
    return rho, float('nan')
  se = cluster_bootstrap(
      lambda idx: stats.spearmanr(x[idx], y[idx]).statistic,
      episodes, n_boot, seed)
  return rho, se


def spearman_matrix(sig, keys, episodes=None, n_boot=200, seed=0):
  """Four-signal Spearman matrix (point estimates + clustered SEs)."""
  n = len(keys)
  rho = np.eye(n)
  se = np.zeros((n, n))
  for i in range(n):
    for j in range(i + 1, n):
      r, s = spearman(sig[keys[i]], sig[keys[j]], episodes, n_boot,
                      seed + 97 * i + j)
      rho[i, j] = rho[j, i] = r
      se[i, j] = se[j, i] = s
  return rho, se


def reversal_rate(udyn, evpi_plugin, evpi_split, max_side=2000, seed=0):
  """P3: fraction of cross-quartile U_dyn pairs with opposite EVPI order.

  Pairs are (i in top U_dyn quartile, j in bottom quartile); a reversal is
  EVPI-hat_i < EVPI-hat_j. To tame estimator noise only pairs whose plug-in
  and split estimates agree in ordering are counted (note App. A.6 P3).
  """
  rng = np.random.default_rng(seed)
  top = np.flatnonzero(udyn >= np.percentile(udyn, 75))
  bot = np.flatnonzero(udyn <= np.percentile(udyn, 25))
  if len(top) > max_side:
    top = rng.choice(top, max_side, replace=False)
  if len(bot) > max_side:
    bot = rng.choice(bot, max_side, replace=False)
  dp = evpi_plugin[top][:, None] - evpi_plugin[bot][None, :]
  ds = evpi_split[top][:, None] - evpi_split[bot][None, :]
  agree = (np.sign(dp) == np.sign(ds)) & (dp != 0)
  if not agree.any():
    return float('nan'), 0
  return float((dp[agree] < 0).mean()), int(agree.sum())


def partial_spearman(x, y, z):
  """Rank-based partial correlation of x and y given z (calibration read)."""
  rx, ry, rz = (stats.rankdata(v).astype(np.float64) for v in (x, y, z))
  resid = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
  return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


@dataclasses.dataclass
class CellResult:
  """Aggregates for one (task, dose) cell, read against E1 thresholds."""
  dose: str
  corr_udyn_aflip: float
  corr_udyn_aflip_se: float
  corr_udyn_evpi: float
  corr_udyn_evpi_se: float
  occupancy_aflip: dict     # quadrants on (udyn, aflip)
  occupancy_evpi: dict      # quadrants on (udyn, evpi_split)
  reversal: float
  reversal_pairs: int
  bracket_width_full: float
  bracket_width_half: float
  plugin_shift_half: float  # mean(EVPI plug-in at R/2) - mean(at R): + expected
  near_tie_qii_rate: float  # P2: near-tie states landing in Q-II
  near_tie_aflip: float     # P4: mean flip rate among near-ties
  near_tie_evpi: float      # P4: mean split-EVPI among near-ties


def analyze_cell(sig, thresholds, episodes=None, dose='0', n_boot=200,
                 seed=0):
  """All pre-registered aggregates for one cell's signal dict."""
  udyn, aflip, evpis = sig['udyn'], sig['aflip'], sig['evpi_split']
  r1, s1 = spearman(udyn, aflip, episodes, n_boot, seed)
  r2, s2 = spearman(udyn, evpis, episodes, n_boot, seed + 1)
  occ_a = quadrant_occupancy(
      udyn, aflip, thresholds['udyn'], thresholds['aflip'])
  occ_e = quadrant_occupancy(
      udyn, evpis, thresholds['udyn'], thresholds['evpi_split'])
  rev, npairs = reversal_rate(
      udyn, sig['evpi_plugin'], evpis, seed=seed)
  ties = sig['gap'] <= np.quantile(sig['gap'], NEAR_TIE_DECILE)
  xlo, xhi = thresholds['udyn']
  ylo, yhi = thresholds['aflip']
  qii_tie = float(((udyn[ties] <= xlo) & (aflip[ties] >= yhi)).mean())
  width_full = float(np.mean(evpis - sig['evpi_plugin']))
  width_half = float(np.mean(
      sig['evpi_split_half'] - sig['evpi_plugin_half'])) if (
          'evpi_split_half' in sig) else float('nan')
  shift = float(np.mean(
      sig['evpi_plugin_half'] - sig['evpi_plugin'])) if (
          'evpi_plugin_half' in sig) else float('nan')
  return CellResult(
      dose=dose,
      corr_udyn_aflip=r1, corr_udyn_aflip_se=s1,
      corr_udyn_evpi=r2, corr_udyn_evpi_se=s2,
      occupancy_aflip=occ_a, occupancy_evpi=occ_e,
      reversal=rev, reversal_pairs=npairs,
      bracket_width_full=width_full, bracket_width_half=width_half,
      plugin_shift_half=shift,
      near_tie_qii_rate=qii_tie,
      near_tie_aflip=float(aflip[ties].mean()),
      near_tie_evpi=float(evpis[ties].mean()),
  )


def evaluate_p1(e1, doses):
  """P1 gate criterion from analyzed cells (e1: CellResult, doses: ordered
  list of E2 CellResults, lowest to highest dose)."""
  top = doses[-1]
  drop_a = e1.corr_udyn_aflip - top.corr_udyn_aflip
  drop_e = e1.corr_udyn_evpi - top.corr_udyn_evpi
  qi = [c.occupancy_aflip['QI'] for c in doses]
  monotone = all(a <= b + 1e-9 for a, b in zip(qi, qi[1:]))
  checks = dict(
      corr_drop_aflip=(drop_a, drop_a >= P1_CORR_DROP),
      corr_drop_evpi=(drop_e, drop_e >= P1_CORR_DROP),
      qi_top=(qi[-1], qi[-1] >= P1_QI_TOP),
      qi_e1=(e1.occupancy_aflip['QI'], e1.occupancy_aflip['QI'] < P1_QI_E1),
      qi_monotone=(qi, monotone),
  )
  ok = all(passed for _, passed in checks.values())
  return ok, checks


def render_report(task, e1, doses, p1):
  """Markdown report block for one task (decision-memo input format)."""
  ok, checks = p1
  lines = [f'### Task: {task}', '']
  lines.append('| dose | rho(Udyn,A) | rho(Udyn,EVPI) | Q-I | Q-II | '
               'reversal (pairs) | bracket R | bracket R/2 | plugin shift |')
  lines.append('|---|---|---|---|---|---|---|---|---|')
  for c in [e1] + list(doses):
    occ = c.occupancy_aflip
    lines.append(
        f"| {c.dose} | {c.corr_udyn_aflip:+.3f} ({c.corr_udyn_aflip_se:.3f})"
        f" | {c.corr_udyn_evpi:+.3f} ({c.corr_udyn_evpi_se:.3f})"
        f" | {occ['QI']:.3f} | {occ['QII']:.3f}"
        f" | {c.reversal:.3f} ({c.reversal_pairs})"
        f" | {c.bracket_width_full:.3f} | {c.bracket_width_half:.3f}"
        f" | {c.plugin_shift_half:+.3f} |")
  lines.append('')
  lines.append(f'P1 gate: **{"PASS" if ok else "FAIL"}**')
  for name, (value, passed) in checks.items():
    val = (f'{value:+.3f}' if isinstance(value, float)
           else '[' + ', '.join(f'{v:.3f}' for v in value) + ']')
    lines.append(f'- {name}: {val} -> {"ok" if passed else "FAIL"}')
  lines.append('')
  lines.append(
      f'P2 (near-tie Q-II rate): E1 {e1.near_tie_qii_rate:.3f}, top dose '
      f'{doses[-1].near_tie_qii_rate:.3f} (strengthening only, never gates)')
  lines.append(
      f'P4 (near-tie decoupling): flip rate {doses[-1].near_tie_aflip:.3f} '
      f'vs EVPI {doses[-1].near_tie_evpi:.3f} at top dose')
  lines.append('')
  return '\n'.join(lines)
