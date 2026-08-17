#!/usr/bin/env python3
"""Enumerate EVERY accepted Q1 pair the walker curation search considered.

Read-only diagnostic (2026-08-16, ops). Answers one question the frozen
artifacts cannot: among the 2715 pairs that already satisfy every registered
criterion, how low can the LOW side's occupancy go while the pair stays
coverage-matched and separated?

WHY THIS IS NOT AN INSTRUMENT CHANGE. It imports
`probing.build_controlled_replay` and calls its own `_candidate_pool` /
`_overlap`; nothing in that module is edited, monkeypatched or re-implemented.
The acceptance test below is copy-identical to `cmd_search` lines 356-361. The
only difference is that `cmd_search` keeps `argmax docc` and throws the rest
away, while this keeps all of them. Selection is NOT changed -- the argmax is
recomputed here purely as a self-check.

SELF-CHECK (printed first, and the script fails loudly if it misses): the
recomputed argmax must reproduce the frozen `pairs.json` q1 record --
side0 occ 0.0802, side1 0.154755, docc 0.074555, dcov 0.008456, n_pairs 2715.
`_candidate_pool`'s docstring guarantees the RNG call sequence matches the
original `search`, so a mismatch means the pool did not reproduce and every
number below is void.

Usage (RCC login node, CPU):
  python -m artifacts.walker_curation_scan_20260816.enumerate_q1_pairs \\
      --index   $RUNROOT/axis1_walker/episodes.json \\
      --pairs   $RUNROOT/axis1_walker/pairs.json \\
      --out     $RUNROOT/axis1_walker_enum/q1_pairs.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from probing import build_controlled_replay as bcr  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--index', required=True)
  ap.add_argument('--pairs', required=True, help='frozen pairs.json (criteria source + self-check)')
  ap.add_argument('--out', required=True)
  a = ap.parse_args()

  with open(a.pairs) as f:
    frozen = json.load(f)
  crit = frozen['criteria']

  # Every knob from the frozen record; n_candidates/n_episodes are not in the
  # criteria block, so the CLI defaults that produced it are used (400 / 200).
  args = argparse.Namespace(
      index=a.index, ref_replay=frozen['ref_replay'],
      n_episodes=frozen.get('n_episodes', 200), n_candidates=400,
      cov_match_frac=crit['cov_match_frac'], occ_sep_mult=crit['occ_sep_mult'],
      max_overlap=crit['max_overlap'], knn=crit['knn'], logc=crit['logc'],
      max_frames=crit['max_frames'], dirichlet=crit['dirichlet'],
      boot=crit['boot'], seed=crit['seed'], holdout=None)

  with open(a.index) as f:
    index = json.load(f)
  index = bcr.apply_holdout(index, None)
  pool = bcr._candidate_pool(args, index)
  cands = pool['candidates']
  cov_tol, occ_sep_min = pool['cov_tol'], pool['occ_sep_min']

  rows, best = [], None
  for i in range(len(cands)):
    for j in range(i + 1, len(cands)):
      x, y = cands[i], cands[j]
      if bcr._overlap(x, y) > args.max_overlap:
        continue
      dcov, docc = abs(x['cov'] - y['cov']), abs(x['occ'] - y['occ'])
      if not (dcov <= cov_tol and docc >= occ_sep_min):
        continue
      lo, hi = (x, y) if x['occ'] <= y['occ'] else (y, x)
      rows.append(dict(i=i, j=j, lo_occ=lo['occ'], hi_occ=hi['occ'],
                       lo_cov=lo['cov'], hi_cov=hi['cov'],
                       docc=docc, dcov=dcov,
                       overlap=bcr._overlap(x, y)))
      if best is None or docc > best['docc']:
        best = rows[-1]

  # --- self-check against the frozen record --------------------------------
  fq = frozen['pairs']['q1']
  fz_lo = min(s['occ'] for s in fq['sides'])
  fz_hi = max(s['occ'] for s in fq['sides'])
  ok = (len(rows) == frozen['n_pairs']['q1']
        and abs(best['lo_occ'] - fz_lo) < 1e-6
        and abs(best['hi_occ'] - fz_hi) < 1e-6
        and abs(best['docc'] - fq['docc']) < 1e-6)
  print('=== self-check: recomputed argmax vs frozen pairs.json ===')
  print(f"  n_pairs   frozen {frozen['n_pairs']['q1']:6d}   recomputed {len(rows):6d}")
  print(f"  side0 occ frozen {fz_lo:.6f}   recomputed {best['lo_occ']:.6f}")
  print(f"  side1 occ frozen {fz_hi:.6f}   recomputed {best['hi_occ']:.6f}")
  print(f"  docc      frozen {fq['docc']:.6f}   recomputed {best['docc']:.6f}")
  print(f"  VERDICT: {'REPRODUCED' if ok else 'MISMATCH -- results below are VOID'}")
  if not ok:
    sys.exit(2)

  rows.sort(key=lambda r: r['lo_occ'])
  bar = 0.05
  under = [r for r in rows if r['lo_occ'] <= bar]
  print(f"\n=== accepted Q1 pairs: {len(rows)} "
        f"(cov_tol {cov_tol:.5f}, occ_sep_min {occ_sep_min:.5f}) ===")
  print(f"  low-side occupancy range over accepted pairs: "
        f"{rows[0]['lo_occ']:.5f} .. {rows[-1]['lo_occ']:.5f}")
  print(f"  accepted pairs with low side <= {bar}: {len(under)}")
  if under:
    b = max(under, key=lambda r: r['docc'])
    m = under[0]
    print(f"  MINIMUM low side : lo={m['lo_occ']:.5f} hi={m['hi_occ']:.5f} "
          f"docc={m['docc']:.5f} dcov={m['dcov']:.5f}")
    print(f"  best separation among compliant: lo={b['lo_occ']:.5f} "
          f"hi={b['hi_occ']:.5f} docc={b['docc']:.5f} dcov={b['dcov']:.5f}")
  os.makedirs(os.path.dirname(a.out), exist_ok=True)
  with open(a.out, 'w') as f:
    json.dump(dict(n_pairs=len(rows), cov_tol=cov_tol,
                   occ_sep_min=occ_sep_min, bar=bar,
                   n_under_bar=len(under), argmax_docc=best,
                   pairs=rows), f, indent=1)
  print(f"\nwrote {a.out}")


if __name__ == '__main__':
  main()
