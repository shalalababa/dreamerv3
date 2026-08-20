"""Tier-0 EXPLORATORY kill: misalignment-aware readout (probe family).

CPU stage. Consumes probing/probe_family_dump.py outputs (one per run) and
fits the pre-stated code-equivariant recoding family on the reward-head's
predicted mean, scoring held-out Gaussian NLL on in-regime frames.

Family (all fit on TRAIN episodes only; evaluation episodes held out):
  identity      r = rhat, sigma fit on train
  affine        r = a*rhat + b            (subsumes sign-flip)
  isotonic      r = g(rhat), g monotone (PAV), best of both directions
  regime_affine separate affine inside/outside regime (4 params)
  mlp           2-layer MLP on [rhat, in_regime] (sklearn; skipped with a
                note if sklearn is absent)

Prediction under test (Algorithm_Ideation §3.3, honest prior ~0.30;
REVISED 20 Aug pre-outcome — only q1 dumps existed when fq1 proved
structurally undumpable and shq1 replaced it): the `rl` arm
(LEARNED-BUT-UNALIGNED, polarity-flipped regime indicator) should move
toward the task anchor under recoding; the random-stamp arms (srd0/srd1)
and the shuffled-label arm (shq1 — trained head, binding destroyed)
should NOT: their true-reward information is absent or destroyed, so no
recoding can help. fq1 is structural-NA (no reward head exists — a
finding about the screen's premise, not a measurable control).

Scoring currency is the family's own (Gaussian, fitted sigma) — NOT the
house symlog-NLL band; comparisons are cross-arm within this currency, with
the task arm as anchor. Split: even episodes train, odd episodes eval.

Usage:
  python -m analysis.tier0_probe_family --dumps '<glob of dump dirs>' \
      --output artifacts/algo_tier0_20260820/probe_family
"""

import argparse
import glob as globlib
import json
import os
import re
from collections import defaultdict

import numpy as np

ARM_RE = re.compile(r'^ax1wm_finger_(?P<arm>q1|fq1|rlq1|shq1|srd0q1|srd1q1|'
                    r'sidq1|rgoq1|sgbq1)s(?P<side>[01])_seed(?P<seed>\d+)$')


def gauss_nll(resid, sigma2):
  return 0.5 * np.log(2 * np.pi * sigma2) + resid ** 2 / (2 * sigma2)


def fit_score(x_tr, y_tr, x_ev, y_ev, member, reg_tr=None, reg_ev=None):
  """Returns held-out mean Gaussian NLL for one family member."""
  def finalize(pred_tr, pred_ev):
    s2 = max(float(np.mean((y_tr - pred_tr) ** 2)), 1e-8)
    return float(np.mean(gauss_nll(y_ev - pred_ev, s2)))

  if member == 'constant':
    mu = float(np.mean(y_tr))
    return finalize(np.full_like(y_tr, mu), np.full_like(y_ev, mu))
  if member == 'identity':
    return finalize(x_tr, x_ev)
  if member == 'affine':
    A = np.stack([x_tr, np.ones_like(x_tr)], 1)
    w, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return finalize(A @ w, np.stack([x_ev, np.ones_like(x_ev)], 1) @ w)
  if member == 'isotonic':
    best = None
    for sign in (1.0, -1.0):
      xs = sign * x_tr
      order = np.argsort(xs, kind='stable')
      g = _pav(y_tr[order])
      xg = xs[order]
      pred_tr = np.interp(sign * x_tr, xg, g)
      pred_ev = np.interp(sign * x_ev, xg, g)
      v = finalize(pred_tr, pred_ev)
      best = v if best is None else min(best, v)
    return best
  if member == 'regime_affine':
    pred_tr = np.empty_like(y_tr)
    pred_ev = np.empty_like(y_ev)
    for flag in (False, True):
      mt, me = reg_tr == flag, reg_ev == flag
      if mt.sum() < 4:
        pred_tr[mt] = y_tr.mean()
        pred_ev[me] = y_tr.mean()
        continue
      A = np.stack([x_tr[mt], np.ones(mt.sum())], 1)
      w, *_ = np.linalg.lstsq(A, y_tr[mt], rcond=None)
      pred_tr[mt] = A @ w
      pred_ev[me] = np.stack([x_ev[me], np.ones(me.sum())], 1) @ w
    return finalize(pred_tr, pred_ev)
  if member == 'mlp':
    try:
      from sklearn.neural_network import MLPRegressor
    except ImportError:
      return None
    X_tr = np.stack([x_tr, reg_tr.astype(float)], 1)
    X_ev = np.stack([x_ev, reg_ev.astype(float)], 1)
    m = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=500,
                     random_state=0)
    m.fit(X_tr, y_tr)
    return finalize(m.predict(X_tr), m.predict(X_ev))
  raise ValueError(member)


def _pav(y):
  """Pool-adjacent-violators: nondecreasing fit of y against its order."""
  g = y.astype(float).copy()
  w = np.ones_like(g)
  i = 0
  # standard stack-based PAV
  vals, wts = [], []
  for v in g:
    vals.append(v)
    wts.append(1.0)
    while len(vals) > 1 and vals[-2] > vals[-1]:
      v2, w2 = vals.pop(), wts.pop()
      v1, w1 = vals.pop(), wts.pop()
      vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
      wts.append(w1 + w2)
  out = np.empty_like(g)
  i = 0
  for v, wt in zip(vals, wts):
    out[i:i + int(wt)] = v
    i += int(wt)
  return out


MEMBERS = ('constant', 'identity', 'affine', 'isotonic', 'regime_affine', 'mlp')
RECODERS = ('affine', 'isotonic', 'regime_affine', 'mlp')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dumps', required=True,
                  help='glob matching probe_family dump DIRS')
  ap.add_argument('--output', required=True)
  args = ap.parse_args()

  runs = []
  for d in sorted(globlib.glob(args.dumps)):
    npz = os.path.join(d, 'probe_family_dump.npz')
    man = os.path.join(d, 'probe_family_manifest.json')
    if not (os.path.exists(npz) and os.path.exists(man)):
      continue
    m = json.load(open(man))
    a = ARM_RE.match(m['run_id'])
    if not a:
      print(f'skip (unrecognized run_id): {m["run_id"]}')
      continue
    z = np.load(npz)
    N = z['rhat'].shape[0]
    tr = np.arange(N) % 2 == 0
    ev = ~tr
    inr = z['in_regime']
    res = {}
    for member in MEMBERS:
      # frames pooled within split, in-regime eval only (the screen's regime)
      x_tr = z['rhat'][tr][inr[tr]]
      y_tr = z['reward'][tr][inr[tr]]
      x_ev = z['rhat'][ev][inr[ev]]
      y_ev = z['reward'][ev][inr[ev]]
      r_tr = np.ones_like(x_tr, bool)
      r_ev = np.ones_like(x_ev, bool)
      if member in ('regime_affine', 'mlp'):
        # these two see all frames + the regime flag (disclosed: scored on
        # the same all-frames eval pool; the cross-arm comparison is what
        # matters and every arm is scored identically)
        x_tr = z['rhat'][tr].ravel(); y_tr = z['reward'][tr].ravel()
        x_ev = z['rhat'][ev].ravel(); y_ev = z['reward'][ev].ravel()
        r_tr = inr[tr].ravel(); r_ev = inr[ev].ravel()
      s = fit_score(x_tr, y_tr, x_ev, y_ev, member, r_tr, r_ev)
      if s is not None:
        res[member] = s
    rec = {k: v for k, v in res.items() if k in RECODERS}
    runs.append(dict(run_id=m['run_id'], arm=a.group('arm'),
                     side=a.group('side'), seed=int(a.group('seed')),
                     device=m.get('device_kind'), scores=res,
                     best=min(rec, key=rec.get), best_nll=min(rec.values()),
                     identity_nll=res.get('identity'),
                     constant_nll=res.get('constant')))

  by_arm = defaultdict(list)
  for r in runs:
    by_arm[r['arm'] + '_s' + r['side']].append(r)
  summary = {}
  for arm, rs in sorted(by_arm.items()):
    summary[arm] = dict(
        n=len(rs),
        constant_mean=float(np.mean([r['constant_nll'] for r in rs])),
        identity_mean=float(np.mean([r['identity_nll'] for r in rs])),
        best_mean=float(np.mean([r['best_nll'] for r in rs])),
        info_gain_mean=float(np.mean([r['constant_nll'] - r['best_nll']
                                      for r in rs])),
        best_members=sorted({r['best'] for r in rs}))
  devices = sorted({r['device'] for r in runs})
  out = dict(
      label='EXPLORATORY (post-read; family currency, not the house band)',
      n_runs=len(runs), devices=devices,
      single_device=len(devices) == 1,
      summary=summary, runs=runs)
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'probe_family.json'), 'w') as f:
    json.dump(out, f, indent=1)
  for arm, s in summary.items():
    print(f'{arm:12s} n={s["n"]:2d} const {s["constant_mean"]:7.3f} '
          f'identity {s["identity_mean"]:7.3f} best {s["best_mean"]:7.3f} '
          f'info_gain {s["info_gain_mean"]:+7.3f} {s["best_members"]}')
  print(f'devices: {devices} -> {args.output}/probe_family.json')


if __name__ == '__main__':
  main()
