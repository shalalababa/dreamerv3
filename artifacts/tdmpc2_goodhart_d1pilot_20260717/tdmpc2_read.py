"""Registered read for PREREG_tdmpc2_crossfamily_20260716.md.

B_arm(k) = AUC100k(s1) - AUC100k(s0) within seed k.
PRIMARY: mean over seeds 1-8 of [B_aware - B_free], cluster-bootstrap
percentile 95% CI, B=10,000, numpy default_rng seed 0, decision on CI.
Secondaries: B_aware, B_free simple effects.
Sensitivity suite (robustness only): paired t, exact sign-flip
permutation, Wilcoxon, d_z, LOO.
"""
import csv, itertools, math, sys
import numpy as np

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
auc = {}
for r in rows:
    if r['run_id'] == 'run_id':
        continue
    assert r['qc_pass'] == '1', r['run_id']
    auc[(r['mode'], int(r['seed']))] = float(r['auc100k'])

seeds = list(range(1, 9))
def B(arm):
    return np.array([auc[(f'tm2{arm}q1s1', k)] - auc[(f'tm2{arm}q1s0', k)]
                     for k in seeds])

b_aware, b_free = B('aware'), B('free')
diff = b_aware - b_free

def boot_ci(x, B_n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B_n, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])

def sensitivity(x, name):
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    t = m / (sd / math.sqrt(n))
    # exact sign-flip permutation, two-sided
    perms = np.array(list(itertools.product([1, -1], repeat=n)))
    pm = (perms * x).mean(axis=1)
    p_perm = (np.abs(pm) >= abs(m) - 1e-12).mean()
    dz = m / sd
    loo = [np.delete(x, i).mean() for i in range(n)]
    lo, hi = boot_ci(x)
    print(f'{name}: mean {m:+.1f}  CI95 [{lo:+.1f}, {hi:+.1f}]  '
          f'(t={t:+.2f}, perm p={p_perm:.4f}, d_z={dz:+.2f}, '
          f'LOO range [{min(loo):+.1f}, {max(loo):+.1f}])')
    return lo, hi

print('per-seed B_aware:', np.round(b_aware, 1).tolist())
print('per-seed B_free :', np.round(b_free, 1).tolist())
print('per-seed diff   :', np.round(diff, 1).tolist())
print()
lo, hi = sensitivity(diff, 'PRIMARY  [B_aware - B_free]')
sensitivity(b_aware, 'SEC-1    B_aware          ')
sensitivity(b_free, 'SEC-2    B_free           ')
print()
print('PRIMARY verdict:', 'CI > 0: REPLICATES cross-family' if lo > 0
      else 'CI spans/below 0: family-scope limitation (registered)')
print()
# descriptive cell means (never compared across families)
for arm in ('aware', 'free'):
    for side in ('0', '1'):
        v = [auc[(f'tm2{arm}q1s{side}', k)] for k in seeds]
        print(f'cell tm2{arm} s{side}: mean AUC100k {np.mean(v):7.1f} '
              f'(sd {np.std(v, ddof=1):5.1f})')
