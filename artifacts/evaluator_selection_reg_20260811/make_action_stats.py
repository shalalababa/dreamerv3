"""Registered producer (PREREG_evaluator_selection_20260811 leg 3b):
per-policy and per-evaluator replay action stats.

Usage (cluster):
  python make_action_stats.py --evaluator $RUNROOT/ax1wm_finger_q1s1_seed1 \
      --policies $RUNROOT/adapt_ax1sw*q1s0_* --output action_stats.json
Concatenates the per-dim action mean and std over every episode npz in
each run's replay/ dir; the evaluator entry is keyed 'evaluator'.
"""
import argparse, glob, json, os
import numpy as np

def stats_of(run_dir, cap=200):
    acts = []
    for p in sorted(glob.glob(os.path.join(run_dir, 'replay', '*.npz')))[:cap]:
        try:
            z = np.load(p)
            if 'action' in z.files:
                acts.append(np.asarray(z['action'], np.float32))
        except Exception:
            continue
    if not acts:
        return None
    a = np.concatenate(acts, 0)
    return np.concatenate([a.mean(0), a.std(0)]).astype(float).tolist()

ap = argparse.ArgumentParser()
ap.add_argument('--evaluator', required=True)
ap.add_argument('--policies', nargs='+', required=True)
ap.add_argument('--output', required=True)
args = ap.parse_args()
out = {}
ev = stats_of(args.evaluator)
assert ev is not None, 'evaluator replay unreadable'
out['evaluator'] = ev
missing = []
for pat in args.policies:
    for d in sorted(glob.glob(pat)):
        s = stats_of(d)
        if s is None:
            missing.append(os.path.basename(d))
        else:
            out[os.path.basename(d)] = s
json.dump(out, open(args.output, 'w'))
print(f'wrote {args.output}: {len(out)-1} policies, missing {missing}')
