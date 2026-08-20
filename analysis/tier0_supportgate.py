"""Tier-0 EXPLORATORY kill: support-gate leave-one-out on the 16 FB fits.

Algorithm candidate (Algorithm_Ideation_20260819 §3.2): before deploying an
FB zero-shot policy, compute the buffer's reward-positive fraction at
z-inference (frac_pos — a pure dataset statistic) and refuse zero-shot below
a threshold, falling back to fine-tuning.

This script is POST-READ EXPLORATORY analysis on the executed FB read's own
per-fit table (artifacts/fb_read_20260817/read.json). It consumes no
registered estimand and registers nothing. Zero compute.

Design (frozen in this file before looking at the LOO outcome):
- Deployment value of a policy: gated-in cells contribute their realized
  zero-shot return; gated-out cells contribute the fallback value F.
- Primary F = 113.682 (the house behavior-alive floor: the conservative
  stand-in for "fine-tuning at least reaches behavior-alive"). Fallback is
  NOT measured in this wave; the break-even sweep below is therefore the
  more decision-relevant output.
- Policies: always-trust (deploy all), never-trust (F everywhere),
  frac_pos-gate with LOO threshold selection (16 folds; on each fold the
  threshold maximizing mean deployment value over the 15 in-fold cells,
  candidates = midpoints of adjacent sorted frac_pos values plus -inf/+inf).
- Oracle = per-cell max(return, F): the regret reference.
- Break-even sweep: for F in a grid, best fixed gate vs always-trust —
  reports the F above which gating starts to pay.
- Catastrophe check: does the gate variable rank the worst realized cell
  (the failure a selective-prediction wrapper exists to catch) below
  deployable cells?

Usage: python -m analysis.tier0_supportgate --read artifacts/fb_read_20260817/read.json \
           --output artifacts/algo_tier0_20260820/supportgate
"""

import argparse
import json
import os

import numpy as np

FLOOR = 113.682


def gate_value(frac, ret, tau, F):
  dep = frac >= tau
  return float(np.mean(np.where(dep, ret, F)))


def candidate_taus(frac):
  s = np.sort(np.unique(frac))
  mids = (s[1:] + s[:-1]) / 2.0
  return np.concatenate(([-np.inf], mids, [np.inf]))


def best_tau(frac, ret, F):
  taus = candidate_taus(frac)
  vals = [gate_value(frac, ret, t, F) for t in taus]
  i = int(np.argmax(vals))  # ties -> most permissive (first) tau
  return float(taus[i]), float(vals[i])


def loo(frac, ret, F):
  n = len(frac)
  realized = np.zeros(n)
  taus = np.zeros(n)
  for i in range(n):
    m = np.ones(n, bool)
    m[i] = False
    tau, _ = best_tau(frac[m], ret[m], F)
    taus[i] = tau
    realized[i] = ret[i] if frac[i] >= tau else F
  return float(np.mean(realized)), taus


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--read', required=True)
  ap.add_argument('--output', required=True)
  args = ap.parse_args()
  r = json.load(open(args.read))
  pf = r['per_fit']
  keys = sorted(pf['zeroshot'].keys())
  ret = np.array([pf['zeroshot'][k] for k in keys], float)
  frac = np.array([pf['z_inference'][k]['frac_pos'] for k in keys], float)
  assert len(keys) == 16

  F = FLOOR
  always = float(np.mean(ret))
  never = F
  oracle = float(np.mean(np.maximum(ret, F)))
  tau_full, gate_full = best_tau(frac, ret, F)
  loo_mean, loo_taus = loo(frac, ret, F)

  # break-even sweep: at which fallback value does a fixed gate beat
  # always-trust?
  sweep = []
  for Fx in np.arange(80.0, 260.0, 2.5):
    t, v = best_tau(frac, ret, Fx)
    sweep.append(dict(F=float(Fx), tau=t, gate=v,
                      always=always * 0 + float(np.mean(np.where(
                          np.isfinite([1] * 16), ret, ret))),
                      wins=bool(v > np.mean(ret) + 1e-9)))
  # simplify: always-trust value is F-independent
  for row in sweep:
    row['always'] = always
  breakeven = next((row['F'] for row in sweep if row['wins']), None)

  # catastrophe check
  worst = int(np.argmin(ret))
  n_below = int(np.sum(frac < frac[worst]))
  catastrophe = dict(
      cell=keys[worst], ret=float(ret[worst]), frac_pos=float(frac[worst]),
      n_cells_gated_below_it=n_below,
      caught_by_any_frac_pos_gate=bool(n_below == 16 - 1 and False) or
      bool(frac[worst] <= float(np.min(frac))))

  out = dict(
      label='EXPLORATORY (post-read; no registered estimand consumed)',
      source=os.path.abspath(args.read),
      fallback_F=F,
      policies=dict(always_trust=always, never_trust=never, oracle=oracle,
                    gate_insample=dict(tau=tau_full, value=gate_full),
                    gate_loo=dict(value=loo_mean,
                                  taus=[float(t) for t in loo_taus])),
      regret=dict(always=oracle - always, never=oracle - never,
                  gate_loo=oracle - loo_mean),
      breakeven_F=breakeven,
      breakeven_note=('lowest fallback value at which the best fixed '
                      'frac_pos gate strictly beats always-trust'),
      sweep=sweep,
      catastrophe_check=catastrophe,
      cells=[dict(cell=k, ret=float(z), frac_pos=float(f))
             for k, z, f in zip(keys, ret, frac)],
  )
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'supportgate_loo.json'), 'w') as f:
    json.dump(out, f, indent=1)
  print(f"always-trust {always:.2f}  never-trust {never:.2f}  "
        f"oracle {oracle:.2f}")
  print(f"gate in-sample tau={tau_full:.4f} -> {gate_full:.2f}; "
        f"LOO -> {loo_mean:.2f}")
  print(f"break-even fallback F = {breakeven}")
  print(f"worst cell {keys[worst]} ret {ret[worst]} frac_pos "
        f"{frac[worst]:.4f}; cells gated below it: {n_below}/15")
  print(f"-> {args.output}/supportgate_loo.json")


if __name__ == '__main__':
  main()
