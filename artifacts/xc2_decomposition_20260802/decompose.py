"""EXPLORATORY post-read decomposition of the xc read's d_pair (2026-08-02).

Runs on the ALREADY-ADJUDICATED xc bundle (ONE read executed 2026-08-02,
artifacts/r3_xc_read_20260802/ — verdict MIXED, off-map XC2 negative
excursion). This analysis is EXPLORATORY: it cannot alter the verdict;
it decomposes the registered estimand to inform whether any follow-up
deserves registration. Quantities were fixed in chat BEFORE execution:

  d_pair = [g_all[m_real_x] - g_all[m_now_x]] - [g_all[m_real] - g_all[m_now]]
         = real_term - now_term
  real_term = g_all[m_real_x] - g_all[m_real]   (probe-pick value change
              under consumer heads: negative = harm story)
  now_term  = g_all[m_now_x]  - g_all[m_now]    (plug-in-pick value change:
              positive = self-consistency story)

Per side (eval=early is the XC2 excursion side): per-run means,
run-clustered percentile bootstrap (B=10K, default_rng(0)), per-domain
splits, chooser agreement rates, conditional terms on changed picks,
and opportunity concentration (per-run spearman of per-state d_pair vs
opportunity = max(g_all) - g_now; tercile means).

Anchor: the reconstructed pooled d_pair per side must match the read
json's primary points EXACTLY (same arrays, same functional).

REV 2 (same session, before any concentration number was trusted):
rev 1's per-run terciles collapsed (opportunity heavily tied within
runs -> empty bins -> NaN) and ordinal-rank spearman is unreliable
under ties. Concentration re-done as average-rank spearman (runs with
non-degenerate opp only) + POOLED opportunity-quartile split
(descriptive). Term/anchor machinery unchanged from rev 1 (rev 1
values reproduced identically).

REV 3 (same session): rev 2's pooled quartiles ALSO collapsed —
diagnostic in itself: >=75% of pooled states sit at exactly the
minimum opportunity (floor states; 17/32 early-side runs have constant
opp — matches the ladder read's floor concentration in early/e4).
On floor states d_pair == 0 by construction, so the quartile split is
replaced by a FLOOR SPLIT (opp <= 1e-9 vs above): counts + mean terms
per stratum. Terms/anchor unchanged again.
"""

import glob
import json
import os
import re

import numpy as np

BUNDLE = 'local_results/r3_xc_20260802_135102/r3_local/xc_labels'
READ_JSON = 'artifacts/r3_xc_read_20260802/r3_xconsumer.json'
OUT = 'artifacts/xc2_decomposition_20260802/decomposition.json'
FILE_RE = re.compile(
    r'^xc_(cup|finger)_(e1|e4)_seed(3[1-8])_(early|late)_dual\.npz$')
VERSION = 'd1fix_20260724_xc2'
B, RNG = 10000, 0


def boot_ci(per_run, rng):
  per_run = np.asarray(per_run, np.float64)
  n = len(per_run)
  idx = rng.integers(0, n, (B, n))
  means = per_run[idx].mean(1)
  return dict(point=float(per_run.mean()),
              ci=[float(np.percentile(means, 2.5)),
                  float(np.percentile(means, 97.5))], n_clusters=n)


def avgrank(x):
  x = np.asarray(x, np.float64)
  order = np.argsort(x, kind='mergesort')
  ranks = np.empty(len(x), np.float64)
  ranks[order] = np.arange(len(x))
  vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
  sums = np.zeros(len(vals), np.float64)
  np.add.at(sums, inv, ranks)
  return (sums / counts)[inv]


def spearman(a, b):
  ra, rb = avgrank(a), avgrank(b)
  ra -= ra.mean(); rb -= rb.mean()
  den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
  return float((ra * rb).sum() / den) if den else None


def main():
  files = sorted(glob.glob(os.path.join(BUNDLE, 'xc_*_dual.npz')))
  assert len(files) == 64, len(files)
  rows = []
  for path in files:
    m = FILE_RE.match(os.path.basename(path))
    assert m, path
    dom, dose, seed, emat = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d['meta']))
    assert meta['labeler_version'] == VERSION, (path, meta['labeler_version'])
    assert meta['dual_chooser'] is True and meta['seed'] == 1
    # consumer = OTHER maturity (early eval -> plain ckpt = late; late -> _early)
    suffix = meta['consumer_checkpoint'].rstrip('/').rsplit('/', 1)[-1]
    assert suffix == ('ckpt' if emat == 'early' else 'ckpt_early'), (path, suffix)
    g = np.asarray(d['g_all'], np.float64)
    ar = np.arange(len(g))
    gr, grx = g[ar, d['m_real']], g[ar, d['m_real_x']]
    gn, gnx = g[ar, d['m_now']], g[ar, d['m_now_x']]
    real_term, now_term = grx - gr, gnx - gn
    d_pair = real_term - now_term
    opp = g.max(1) - np.asarray(d['g_now'], np.float64)
    chg_real = d['m_real_x'] != d['m_real']
    chg_now = d['m_now_x'] != d['m_now']
    rows.append(dict(
        dom=dom, dose=dose, seed=seed, emat=emat,
        d_pair=float(d_pair.mean()),
        real_term=float(real_term.mean()), now_term=float(now_term.mean()),
        real_term_changed=float(real_term[chg_real].mean())
        if chg_real.any() else None,
        now_term_changed=float(now_term[chg_now].mean())
        if chg_now.any() else None,
        agree_real=float(1 - chg_real.mean()), agree_now=float(1 - chg_now.mean()),
        opp_distinct=int(len(np.unique(opp))),
        spearman_dpair_opp=spearman(d_pair, opp)
        if len(np.unique(opp)) > 1 else None,
        arrays=dict(d_pair=d_pair, real_term=real_term, now_term=now_term,
                    opp=opp)))

  out = dict(bundle=BUNDLE, version=VERSION, n_files=len(rows),
             note='EXPLORATORY post-read decomposition; cannot alter the '
                  'MIXED verdict of artifacts/r3_xc_read_20260802/',
             sides={})
  read = json.load(open(READ_JSON))
  anchors = dict(early='p_xc2_early_eval', late='p_xc1_late_eval')
  for emat in ('early', 'late'):
    side = [r for r in rows if r['emat'] == emat]
    assert len(side) == 32
    rng = np.random.default_rng(RNG)
    res = dict(
        d_pair=boot_ci([r['d_pair'] for r in side], rng),
        real_term=boot_ci([r['real_term'] for r in side], rng),
        now_term=boot_ci([r['now_term'] for r in side], rng))
    # anchor: reconstructed pooled d_pair == the read's primary point
    assert abs(res['d_pair']['point']
               - read['primaries'][anchors[emat]]['point']) < 1e-9, \
        (emat, res['d_pair']['point'], read['primaries'][anchors[emat]])
    for dom in ('cup', 'finger'):
      sub = [r for r in side if r['dom'] == dom]
      rngd = np.random.default_rng(RNG)
      res[f'{dom}_real_term'] = boot_ci([r['real_term'] for r in sub], rngd)
      res[f'{dom}_now_term'] = boot_ci([r['now_term'] for r in sub], rngd)
    res['agree_real'] = float(np.mean([r['agree_real'] for r in side]))
    res['agree_now'] = float(np.mean([r['agree_now'] for r in side]))
    res['real_term_on_changed'] = float(np.mean(
        [r['real_term_changed'] for r in side
         if r['real_term_changed'] is not None]))
    res['now_term_on_changed'] = float(np.mean(
        [r['now_term_changed'] for r in side
         if r['now_term_changed'] is not None]))
    sp = [r['spearman_dpair_opp'] for r in side
          if r['spearman_dpair_opp'] is not None]
    res['spearman_dpair_opp'] = dict(
        mean=float(np.mean(sp)), n_valid_runs=len(sp))
    # POOLED floor split (descriptive): opp <= 1e-9 = floor (no candidate
    # beats the current policy anywhere; d_pair == 0 by construction
    # whenever g_all is flat-zero), vs non-floor where choice matters.
    pool = {k: np.concatenate([r['arrays'][k] for r in side])
            for k in ('d_pair', 'real_term', 'now_term', 'opp')}
    floor = pool['opp'] <= 1e-9
    res['pooled_floor_split'] = dict(
        floor_frac=float(floor.mean()),
        counts=dict(floor=int(floor.sum()), nonfloor=int((~floor).sum())),
        floor={k: float(pool[k][floor].mean()) for k in
               ('d_pair', 'real_term', 'now_term')},
        nonfloor={k: float(pool[k][~floor].mean()) for k in
                  ('d_pair', 'real_term', 'now_term')},
        nonfloor_opp_mean=float(pool['opp'][~floor].mean()))
    out['sides'][emat] = res

  out['per_run'] = [{k: v for k, v in r.items() if k != 'arrays'}
                    for r in rows]
  with open(OUT, 'w') as f:
    json.dump(out, f, indent=2)
  for emat in ('early', 'late'):
    s = out['sides'][emat]
    print(f"--- eval={emat} (consumer = {'late' if emat=='early' else 'early'} "
          f"heads){'  [XC2 excursion side]' if emat == 'early' else ''}")
    for k in ('d_pair', 'real_term', 'now_term'):
      v = s[k]
      print(f"  {k:10s} {v['point']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")
    print(f"  per-domain real_term: cup {s['cup_real_term']['point']:+.4f} "
          f"[{s['cup_real_term']['ci'][0]:+.4f},{s['cup_real_term']['ci'][1]:+.4f}]"
          f"  finger {s['finger_real_term']['point']:+.4f} "
          f"[{s['finger_real_term']['ci'][0]:+.4f},{s['finger_real_term']['ci'][1]:+.4f}]")
    print(f"  per-domain now_term:  cup {s['cup_now_term']['point']:+.4f}  "
          f"finger {s['finger_now_term']['point']:+.4f}")
    print(f"  agree: m_real {s['agree_real']:.3f}  m_now {s['agree_now']:.3f}; "
          f"on-changed real_term {s['real_term_on_changed']:+.4f} "
          f"now_term {s['now_term_on_changed']:+.4f}")
    sp = s['spearman_dpair_opp']
    q = s['pooled_floor_split']
    print(f"  spearman(d_pair, opp) mean {sp['mean']:+.3f} "
          f"({sp['n_valid_runs']}/32 runs valid)")
    print(f"  floor split: {q['floor_frac']:.1%} floor "
          f"(d_pair {q['floor']['d_pair']:+.4f}); nonfloor n="
          f"{q['counts']['nonfloor']} opp {q['nonfloor_opp_mean']:.2f}: "
          f"d_pair {q['nonfloor']['d_pair']:+.4f} real "
          f"{q['nonfloor']['real_term']:+.4f} now "
          f"{q['nonfloor']['now_term']:+.4f}")
  print(f'-> {OUT}')


if __name__ == '__main__':
  main()
