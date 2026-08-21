"""Null fixture — the R3-era pipeline on provably-zero true opportunity.

NON-REGISTERED ARCHIVED COMPUTE (Gate 0 of Rescue_Synthesis_20260819
§"THE DECISION CRITERION": the Paper-2 draft's contribution list cites
"a null fixture ... on which the unmodified pipeline returns +5.6" and
NO artifact exists for that number; user decision 21 Aug: rebuild).

This is the REBUILT fixture, not a reproduction of "+5.6": the
original fixture's parameters were never recorded, so the draft quotes
THIS artifact's numbers instead (the claim is qualitative — the
unmodified single-draw estimator returns a large positive opportunity
on a substrate with exactly zero true candidate value — and the
rebuilt version is stronger: its noise is calibrated per-state from
the ARCHIVED arrays, so the zero-truth fixture reproduces the archived
headline SCALE, the route1-sims E4 exhibit on real noise calibration).

Construction, per family:
  - sigma(s) = the archived per-state pure-noise sd: within-candidate
    across-repeat sd (ddof=1) of g_all_rep, averaged over candidates —
    an estimate of single-draw evaluation noise that contains NO
    between-candidate signal;
  - fixture: g(s, m) = sigma(s) * z(s, m), z iid N(0,1), pinned rng —
    true per-candidate value IDENTICALLY ZERO by construction;
  - the unmodified R3-era estimand on single draws:
      opp_chosen(s) = max_m g(s, m) - g(s, m_now),  m_now uniform
      (the archived A2 finding: choosers at chance)
      opp_mean(s)   = max_m g(s, m) - mean_m g(s, m)
      (the draft's "opportunity == max - mean to 1.7%" identity form)
  - a x2-noise variant (the draft's "adding evaluation noise doubles
    the effect" signature must reproduce on the fixture).

Everything is deterministic (seed 20260821); adjudicates nothing.

Usage:
  python -m analysis.null_fixture --labels <dir> --output <dir>
  python -m analysis.null_fixture --selfcheck
"""

import argparse
import glob
import json
import os

import numpy as np

from analysis.substrate_screen import FILE_RES

SEED = 20260821
M = 8
N_DRAWS = 50          # fixture replications (MC over the pinned stream)


def noise_sd_per_state(g):
  """(S, R, M) -> per-state single-draw noise sd, signal-free
  (within-candidate across repeats, ddof=1, averaged over candidates)."""
  return g.std(1, ddof=1).mean(1)


def fixture_opportunity(sigma, rng, m_count=M, noise_mult=1.0):
  """One fixture draw over all states: (opp_chosen, opp_mean)."""
  S = len(sigma)
  g = (noise_mult * sigma)[:, None] * rng.standard_normal((S, m_count))
  m_now = rng.integers(0, m_count, S)
  opp_chosen = g.max(1) - g[np.arange(S), m_now]
  opp_mean = g.max(1) - g.mean(1)
  return float(opp_chosen.mean()), float(opp_mean.mean())


def run_family(labels_dir, family):
  sigmas = []
  n_files = 0
  for path in sorted(glob.glob(os.path.join(labels_dir, '*.npz'))):
    if not FILE_RES[family].match(os.path.basename(path)):
      continue
    g = np.asarray(np.load(path, allow_pickle=True)['g_all_rep'], float)
    sigmas.append(noise_sd_per_state(g))
    n_files += 1
  sigma = np.concatenate(sigmas)
  rng = np.random.default_rng([SEED, 0 if family == 'dv3' else 1])
  base_c, base_m, noisy_c = [], [], []
  for _ in range(N_DRAWS):
    c, m = fixture_opportunity(sigma, rng)
    base_c.append(c)
    base_m.append(m)
    noisy_c.append(fixture_opportunity(sigma, rng, noise_mult=2.0)[0])
  return dict(
      n_files=n_files, n_states=int(len(sigma)),
      noise_sd_mean=float(sigma.mean()),
      noise_sd_p90=float(np.percentile(sigma, 90)),
      opp_chosen=float(np.mean(base_c)),
      opp_chosen_sd_over_draws=float(np.std(base_c, ddof=1)),
      opp_max_minus_mean=float(np.mean(base_m)),
      opp_chosen_2x_noise=float(np.mean(noisy_c)),
      true_opportunity=0.0)


def run(labels_dir, output):
  os.makedirs(output, exist_ok=True)
  out = {fam: run_family(labels_dir, fam) for fam in ('dv3', 'tm2')}
  out['status'] = ('NON-REGISTERED ARCHIVED COMPUTE (Gate 0 rebuild); '
                   'true opportunity is IDENTICALLY ZERO by '
                   'construction; the original "+5.6" figure had no '
                   'artifact and is SUPERSEDED by these numbers')
  path = os.path.join(output, 'null_fixture.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)
  return out


def selfcheck():
  rng = np.random.default_rng(0)
  # 1) closed form: with sigma = 1, E[opp_mean] = E[max_M z] (~1.4236
  # at M=8), E[opp_chosen] = same (chosen is mean-zero) — both large
  # positives from EXACTLY zero truth
  sigma = np.ones(20000)
  c, m = fixture_opportunity(sigma, np.random.default_rng(1))
  assert abs(m - 1.4236) < 0.03, m
  assert abs(c - 1.4236) < 0.06, c
  # 2) linear in noise: x2 noise doubles it
  c2, _ = fixture_opportunity(sigma, np.random.default_rng(2),
                              noise_mult=2.0)
  assert abs(c2 / c - 2.0) < 0.06, (c, c2)
  # 3) sigma estimator is signal-free: planted between-candidate spread
  # does NOT inflate it (within-candidate across repeats only)
  mu = rng.normal(0, 5.0, (300, 8))
  g = mu[:, None, :] + rng.normal(0, 1.0, (300, 8, 8))
  sd = noise_sd_per_state(g)
  assert abs(float(sd.mean()) - 1.0) < 0.05, sd.mean()
  # 4) zero-noise states contribute exactly zero
  c0, m0 = fixture_opportunity(np.zeros(100), np.random.default_rng(3))
  assert c0 == 0.0 and m0 == 0.0
  # 5) determinism: pinned stream reproduces bitwise
  s = np.abs(rng.standard_normal(500))
  a = fixture_opportunity(s, np.random.default_rng([SEED, 7]))
  b = fixture_opportunity(s, np.random.default_rng([SEED, 7]))
  assert a == b
  print('null_fixture selfcheck PASS (closed form E[max_8 z] ~ 1.4236 '
        'both estimand forms; x2-noise doubling; sigma estimator '
        'signal-free under planted spread; zero-noise -> exactly zero; '
        'pinned-stream determinism)')


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
  run(args.labels, args.output)


if __name__ == '__main__':
  main()
