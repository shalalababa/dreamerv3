"""EXPLORATORY target-angle audit of built finger buffers (2026-08-08,
review item #5). CPU-only; runs where the built buffer chunks live.

Question: `target_position` is a per-episode constant, blacklisted from
coverage by construction, and high occupancy is achievable only where the
target was reachable — so the Q1 occupancy contrast may be partly a
target-placement contrast that W1's within-collector design does NOT
remove (selection on occupancy is still selection on target placement).
This audit reads the per-episode target angle marginal off the built
sides and reports the side contrast.

Usage:
  python -m probing.target_angle_audit --pair $RUNROOT/axis1_finger/q1 \
      --output target_angle_q1.json
"""

import argparse
import json
import math
import os

import numpy as np

from probing.relabel_replay import load_episode_chunks


def side_stats(episodes):
  angles, occs, radii = [], [], []
  for f in episodes:
    tp = np.asarray(f['target_position'], np.float64)
    tp0 = tp[0]
    assert np.allclose(tp, tp0[None], atol=1e-6), \
        'target_position varies within an episode (not finger-like)'
    angles.append(math.atan2(tp0[1], tp0[0]))
    radii.append(float(np.linalg.norm(tp0)))
    # batch-review n18: reward fires at dist <= 0 (boundary inclusive);
    # occs stays index-aligned with angles (None kept in place, filtered
    # PAIRWISE by the caller).
    if 'dist_to_target' in f:
      occs.append(float((np.asarray(f['dist_to_target']) <= 0).mean()))
    else:
      occs.append(None)
  return angles, occs, radii


def circ_mean_r(angles):
  z = np.exp(1j * np.asarray(angles))
  m = z.mean()
  return float(np.angle(m)), float(np.abs(m))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--pair', required=True,
                 help='Built pair dir with side0/ side1/ episode chunks.')
  p.add_argument('--output', required=True)
  args = p.parse_args()

  out = dict(pair=os.path.abspath(args.pair), sides={})
  all_angles = {}
  for side in ('side0', 'side1'):
    eps = load_episode_chunks(os.path.join(args.pair, side))
    angles, occs, radii = side_stats(eps)
    mu, r = circ_mean_r(angles)
    all_angles[side] = np.asarray(angles)
    paired = [(a, o) for a, o in zip(angles, occs) if o is not None]
    pa = np.array([a for a, _ in paired])
    po = np.array([o for _, o in paired])
    out['sides'][side] = dict(
        n_episodes=len(eps),
        circ_mean=mu, circ_concentration=r,
        angle_deciles=[float(x) for x in np.percentile(
            angles, np.arange(0, 101, 10))],
        radius_mean=float(np.mean(radii)),
        occ_mean=(float(po.mean()) if len(po) else None),
        # circular corr(angle, occ) via sin/cos components, PAIRWISE
        corr_occ_sin=(float(np.corrcoef(np.sin(pa), po)[0, 1])
                      if len(po) > 1 else None),
        corr_occ_cos=(float(np.corrcoef(np.cos(pa), po)[0, 1])
                      if len(po) > 1 else None))
  # Two-sample comparison of the angle marginals: KS on wrapped angles and
  # on sin/cos components (rotation-safe).
  a0, a1 = all_angles['side0'], all_angles['side1']
  def ks(x, y):
    xs = np.sort(x); ys = np.sort(y)
    grid = np.concatenate([xs, ys])
    cx = np.searchsorted(xs, grid, 'right') / len(xs)
    cy = np.searchsorted(ys, grid, 'right') / len(ys)
    return float(np.max(np.abs(cx - cy)))
  out['contrast'] = dict(
      ks_angle=ks(a0, a1),
      ks_sin=ks(np.sin(a0), np.sin(a1)),
      ks_cos=ks(np.cos(a0), np.cos(a1)),
      note=('large KS => the occupancy contrast is also a target-placement '
            'contrast; the interaction inherits it as a shared-buffer fact '
            '(same buffers both arms), but any occupancy-mechanism wording '
            'must say "occupancy-and-placement".'))
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=2)
  print(json.dumps(out['contrast'], indent=1))
  print(f'-> {args.output}')


if __name__ == '__main__':
  main()
