"""Pre-named DESCRIPTIVE exhibit for the wfb wave
(PREREG_p2_fbconsumer_20260829). No alpha, no verdict weight, run
AFTER the ONE read; pre-named in the prereg so it is not a post-hoc
search.

Two quantities:
1. VALUE-RANGE TRACKING (labels only, no torch): per state, the
   spread (sd over candidates) of FB's Q scores vs the spread of the
   ground-truth candidate values mean_r g — Spearman per cell, pooled.
   "Does the FB surface's per-state RANGE track where true value
   spread lives?" — the WDC-flatness bridge in this wave's own cells.
2. B-STATE GEOMETRY (optional, --ckpt + torch): B(obs_now) embeddings
   of the label states; reports per-cell embedding dispersion vs the
   cell's split-selected opportunity — whether FB's state geometry is
   richer where opportunity is larger. Skipped with a note when torch
   or the ckpt is unavailable.

Usage:
  python -m analysis.fb_bdiv_exhibit --labels <dir> \
      --output artifacts/fbcons_read_20260829/bdiv_exhibit.json \
      [--ckpt <fbwm_finger_q1s1_seed8/fb_ckpt.pt>]
"""

import argparse
import glob
import json
import os

import numpy as np

from analysis.fbcons_read import EXPECT_CELLS, FILE_RE, S1_SHA, load_cell


def spearman(a, b):
  a, b = np.asarray(a, float), np.asarray(b, float)
  ra = np.argsort(np.argsort(a)).astype(float)
  rb = np.argsort(np.argsort(b)).astype(float)
  if ra.std() == 0 or rb.std() == 0:
    return float('nan')
  return float(np.corrcoef(ra, rb)[0, 1])


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--labels', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--ckpt', default=None)
  args = p.parse_args()
  cells = {}
  for path in sorted(glob.glob(os.path.join(args.labels, 'wfb_*.npz'))):
    if FILE_RE.match(os.path.basename(path)):
      seed, e = load_cell(path)
      cells[seed] = e
  assert len(cells) == EXPECT_CELLS, (
      f'{len(cells)} cells != registered {EXPECT_CELLS}')
  if args.ckpt:
    from probing.fb_embed import sha256_file
    got = sha256_file(args.ckpt)
    assert got == S1_SHA, ('--ckpt is not the pinned s1 fit', got)
  per_cell = {}
  pooled_q, pooled_g = [], []
  for seed, e in sorted(cells.items()):
    q_spread = e['fbq'][:, 0, :].std(-1)          # (S,) s1 scorer
    g_spread = e['g'].mean(1).std(-1)             # (S,) true spread
    per_cell[seed] = dict(rho=spearman(q_spread, g_spread))
    pooled_q.append(q_spread)
    pooled_g.append(g_spread)
  pooled_rho = spearman(np.concatenate(pooled_q),
                        np.concatenate(pooled_g))
  bgeo = None
  if args.ckpt:
    try:
      import torch
      from probing.fb_fit import load_agent
      agent, _ = load_agent(args.ckpt, device='cpu')
      bgeo = {}
      for seed, e in sorted(cells.items()):
        with torch.no_grad():
          emb = agent.backward_net(
              torch.as_tensor(e['obs'], dtype=torch.float32)).numpy()
        even = e['g'][:, 0::2].mean(1)
        odd = e['g'][:, 1::2].mean(1)
        sel = even.argmax(1)
        i = np.arange(len(sel))
        opp = float(np.mean(odd[i, sel] - odd[i, e['mn']]))
        disp = float(np.linalg.norm(
            emb - emb.mean(0, keepdims=True), axis=1).mean())
        bgeo[seed] = dict(b_dispersion=disp, split_opp=opp)
    except Exception as exc:                       # noqa: BLE001
      bgeo = dict(skipped=str(exc))
  out = dict(tool='fb_bdiv_exhibit_v1',
             note='DESCRIPTIVE, no alpha, no verdict weight '
                  '(pre-named in PREREG_p2_fbconsumer_20260829)',
             value_range_tracking=dict(per_cell=per_cell,
                                       pooled_rho=pooled_rho),
             b_state_geometry=bgeo)
  os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=1)
  print(f'pooled value-range-tracking rho = {pooled_rho:+.3f} '
        f'({len(cells)} cells) -> {args.output}')


if __name__ == '__main__':
  main()
