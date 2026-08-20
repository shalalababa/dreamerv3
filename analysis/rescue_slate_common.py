"""Shared machinery for the 20-Aug rescue-slate readers.

Frozen alongside the five PREREG_*_20260820 readers. House conventions:
permutation-primary + BCa (standing rule), STRICT modal n_ep, witnessed
counters (07-Aug rule), refusal-on-any-gate-failure (the read is not
consumed by a refusal).

one_sample_auto: exact sign-flip enumeration for n <= 20 (house
domains_read.one_sample); Monte-Carlo sign-flip with FIXED seed and
N_MC=200000 for larger n (exact enumeration is infeasible at n=40/56) —
the MC substitution is deterministic and pinned here.
"""

import csv as _csv
import hashlib
import json
import math
import os

import numpy as np

from analysis.domains_read import one_sample as _exact_one_sample

N_MC = 200000
MC_SEED = 20260820
B_BOOT = 20000


class Refusal(SystemExit):
  pass


def refuse(msg):
  raise Refusal(f'REFUSE: {msg}')


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def _ndtri(p):
  # Acklam rational approximation (|rel err| < 1.15e-9) — dependency-free.
  a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
       1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
  b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
       6.680131188771972e+01, -1.328068155288572e+01]
  c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
       -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
  d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
       3.754408661907416e+00]
  plow, phigh = 0.02425, 1 - 0.02425
  if not 0.0 < p < 1.0:
    refuse(f'ndtri domain: {p}')
  if p < plow:
    q = math.sqrt(-2 * math.log(p))
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
            + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  if p > phigh:
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  q = p - 0.5
  r = q * q
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
          + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                          + b[4]) * r + 1)


def one_sample_auto(x):
  """Exact house one_sample for n<=20; MC sign-flip + BCa for larger n."""
  x = np.asarray(x, float)
  n = len(x)
  if n <= 20:
    return dict(_exact_one_sample(x), method='exact_signflip')
  rng = np.random.default_rng(MC_SEED)
  obs = float(x.mean())
  boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B_BOOT)])
  prop = float(np.mean(boots < obs))
  prop = min(max(prop, 1.0 / (B_BOOT + 1)), 1.0 - 1.0 / (B_BOOT + 1))
  z0 = _ndtri(prop)
  jack = np.array([np.delete(x, i).mean() for i in range(n)])
  jm = jack.mean()
  den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
  acc = float(np.sum((jm - jack) ** 3) / den) if den > 0 else 0.0

  def q(alpha):
    z = _ndtri(alpha)
    adj = z0 + (z0 + z) / (1.0 - acc * (z0 + z))
    p = 0.5 * (1.0 + math.erf(adj / math.sqrt(2)))
    p = min(max(p, 1e-9), 1 - 1e-9)
    return float(np.percentile(boots, 100.0 * p))

  signs = rng.integers(0, 2, (N_MC, n)) * 2 - 1
  flips = signs @ x / n
  perm_p = float((np.sum(np.abs(flips) >= abs(obs) - 1e-12) + 1)
                 / (N_MC + 1))
  sd = float(x.std(ddof=1))
  return dict(mean=obs, ci=[q(0.025), q(0.975)], perm_p=perm_p, n=n, sd=sd,
              d_z=obs / sd if sd > 0 else float('nan'),
              method=f'mc_signflip_{N_MC}_seed{MC_SEED}')


def load_rows(auc_csv):
  with open(auc_csv) as f:
    return list(_csv.DictReader(f))


def side_auc(rows, mode, seeds, domain, field='auc100k'):
  """seed -> AUC for one mode; refuses on missing/dup/qc-fail."""
  out = {}
  for r in rows:
    if r['mode'] != mode or r['domain'] != domain:
      continue
    s = int(r['seed'])
    if s not in seeds:
      continue
    if int(r['qc_pass']) != 1:
      refuse(f'{mode} seed {s}: qc_pass != 1')
    if s in out:
      refuse(f'{mode} seed {s}: duplicate row')
    out[s] = float(r[field])
  missing = [s for s in seeds if s not in out]
  if missing:
    refuse(f'{mode}: missing seeds {missing}')
  return out


def check_modal_nep(rows, modes, seeds, domain, field='n_ep_100k',
                    expected=None):
  """STRICT modal n_ep: every counted row must equal the modal value; if
  `expected` is given (a pinned look-1 modal), the modal must equal it —
  otherwise pooling would silently mix estimand windows (carrier B3)."""
  vals = [int(r[field]) for r in rows
          if r['mode'] in modes and r['domain'] == domain
          and int(r['seed']) in seeds]
  if not vals:
    refuse('modal n_ep: no rows')
  modal = max(set(vals), key=vals.count)
  bad = [v for v in vals if v != modal]
  if bad:
    refuse(f'modal n_ep violation: modal {modal}, deviants {sorted(set(bad))}')
  if expected is not None and modal != expected:
    refuse(f'modal n_ep {modal} != pinned look-1 modal {expected}')
  return modal


def check_witness(witness_path, expected_fits, expected_updates):
  """Witness json: {run_id: {counters: int, config_sha_excl_seed: str}}.
  Refuses on missing run, short counters. Returns per-run dict."""
  with open(witness_path) as f:
    w = json.load(f)
  missing = [r for r in expected_fits if r not in w]
  if missing:
    refuse(f'witness missing {len(missing)} fits (first: {missing[:3]})')
  for run in expected_fits:
    c = int(w[run]['counters'])
    if c != expected_updates:
      refuse(f'{run}: counters {c} != {expected_updates}')
  return w


def check_config_identity(witness, groups):
  """groups: {label: [run_ids]} — each group must share ONE
  config_sha_excl_seed."""
  for label, runs in groups.items():
    shas = {witness[r]['config_sha_excl_seed'] for r in runs}
    if len(shas) != 1:
      refuse(f'config identity broken in {label}: {len(shas)} distinct shas')
