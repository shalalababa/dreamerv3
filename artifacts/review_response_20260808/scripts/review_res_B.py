"""Paper-1 review resolution — Part B: statistics re-adjudication (Part III).

Labeled EXPLORATORY/post-read sensitivity; NO registered verdict is re-executed.
Deterministic: rng(0), B=10000 for CIs.
"""
import csv, glob, json, os, re, collections, math
import numpy as np
from scipy import stats as sps

R = '/home/rickybao/projects/dreamerv3'
B = 10000
OUT = {}
rng0 = lambda: np.random.default_rng(0)

# ---------------- test battery ----------------
def pct_ci(x, B=B):
    x = np.asarray(x, float); rng = rng0()
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def bca_ci(x, B=B):
    x = np.asarray(x, float); n = len(x); rng = rng0()
    boots = x[rng.integers(0, n, (B, n))].mean(1)
    theta = x.mean()
    z0 = sps.norm.ppf(max(1, (boots < theta).sum()) / (B + 1))
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    d = jack.mean() - jack
    a = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5 + 1e-300)
    lo, hi = [], []
    out = []
    for alpha in (0.025, 0.975):
        z = sps.norm.ppf(alpha)
        adj = sps.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(float(np.percentile(boots, 100 * np.clip(adj, 1e-6, 1 - 1e-6))))
    return out

def boot_t_ci(x, B=B, inner=200):
    x = np.asarray(x, float); n = len(x); rng = rng0()
    tstars = []
    se_hat = x.std(ddof=1) / math.sqrt(n)
    for _ in range(B // 2):
        xb = x[rng.integers(0, n, n)]
        se_b = xb.std(ddof=1) / math.sqrt(n)
        if se_b == 0: continue
        tstars.append((xb.mean() - x.mean()) / se_b)
    q_lo, q_hi = np.percentile(tstars, [2.5, 97.5])
    return [float(x.mean() - q_hi * se_hat), float(x.mean() - q_lo * se_hat)]

def perm_p(x):
    x = np.asarray(x, float); n = len(x)
    obs = abs(x.mean())
    if n <= 20:
        cnt = 0
        for m in range(2 ** n):
            s = np.array([1 if (m >> i) & 1 else -1 for i in range(n)])
            if abs((x * s).mean()) >= obs - 1e-12: cnt += 1
        return cnt / 2 ** n
    rng = rng0()
    s = rng.choice([-1, 1], (100000, n))
    return float((np.abs((x * s).mean(1)) >= obs - 1e-12).mean())

def sign_p(x):
    x = np.asarray(x, float)
    npos = int((x > 0).sum()); n = int((x != 0).sum())
    return float(sps.binomtest(npos, n, 0.5).pvalue)

def loo_fires(x, B=2000):
    """# of leave-one-out subsamples whose percentile CI excludes 0."""
    x = np.asarray(x, float); k = 0
    for i in range(len(x)):
        xi = np.delete(x, i); rng = rng0()
        m = xi[rng.integers(0, len(xi), (B, len(xi)))].mean(1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        if (lo > 0) == (hi > 0): k += 1
    return k

def tost_p(x, margin=24.7):
    """TOST equivalence vs +/-margin: p = max of the two one-sided t tests."""
    x = np.asarray(x, float); n = len(x)
    se = x.std(ddof=1) / math.sqrt(n)
    t1 = (x.mean() + margin) / se   # H0: mu <= -margin
    t2 = (x.mean() - margin) / se   # H0: mu >= +margin
    p1 = 1 - sps.t.cdf(t1, n - 1)
    p2 = sps.t.cdf(t2, n - 1)
    return float(max(p1, p2))

def battery(name, x, margin=24.7):
    x = np.asarray(x, float); n = len(x)
    t = sps.ttest_1samp(x, 0)
    tci = t.confidence_interval()
    try: wil = float(sps.wilcoxon(x).pvalue)
    except Exception: wil = None
    pct = pct_ci(x); bca = bca_ci(x); bt = boot_t_ci(x)
    return dict(
        name=name, n=n, mean=float(x.mean()), pos=int((x > 0).sum()),
        pct_ci=pct, pct_fires=bool((pct[0] > 0) == (pct[1] > 0)),
        bca_ci=bca, bca_fires=bool((bca[0] > 0) == (bca[1] > 0)),
        boot_t_ci=bt, boot_t_fires=bool((bt[0] > 0) == (bt[1] > 0)),
        t_ci=[float(tci.low), float(tci.high)], t_p=float(t.pvalue),
        perm_p=perm_p(x), wilcoxon_p=wil, sign_p=sign_p(x),
        loo_fires=f'{loo_fires(x)}/{n}', tost_p_pm24p7=tost_p(x, margin),
        d_z=float(x.mean() / (x.std(ddof=1) + 1e-300)),
        per_seed=[round(float(v), 4) for v in x])

# ---------------- AUC gathering (same as part A) ----------------
def gather_auc(paths):
    best = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            if r.get('milestone') != '500000' and r.get('mode') != 'scratch':
                continue
            if not r.get('auc100k'): continue
            key = (r['mode'], r['domain'], int(r['seed']))
            n = int(r['n_ep_100k'] or 0)
            if key not in best or n > best[key][1]:
                best[key] = (float(r['auc100k']), n)
    return {k: v[0] for k, v in best.items()}

CANON = [
    f'{R}/local_results/volume_repl_20260729_204854/analysis/auc/auc.csv',
    f'{R}/local_results/unfrozen_u1_20260801_212204/auc/auc.csv',
    f'{R}/local_results/synth_phasebpp_20260722_092523/analysis/auc/auc.csv',
    f'{R}/local_results/pixel_x2_20260724_213538/analysis/auc/auc.csv',
    f'{R}/local_results/unfrozen_calib_20260725_172314/analysis/auc/auc.csv',
    f'{R}/local_results/axis1_seed_extension_20260711_201510/analysis/auc/auc.csv',
    f'{R}/artifacts/p3_amend1_20260717/auc_pooled_1_16.csv',
    f'{R}/artifacts/stamping_20260718/auc.csv',
    f'{R}/artifacts/vgo_extended_20260726/auc.csv',
]
A = gather_auc(CANON)

def vec(mode, dom='finger', seeds=None):
    d = {s: v for (m, dm, s), v in A.items() if m == mode and dm == dom}
    ss = sorted(d) if seeds is None else sorted(set(d) & set(seeds))
    return np.array([d[s] for s in ss]), ss

def delta(m1, m0, dom='finger', seeds=None):
    a, sa = vec(m1, dom, seeds); b, sb = vec(m0, dom, seeds)
    common = sorted(set(sa) & set(sb))
    ai = {s: v for s, v in zip(sa, a)}; bi = {s: v for s, v in zip(sb, b)}
    return np.array([ai[s] - bi[s] for s in common]), common

# ---------------- 1. re-adjudication table ----------------
tab = []
u1 = json.load(open(f'{R}/artifacts/unfrozen_u1_20260801/unfrozen_u1.json'))
tab.append(battery('U1_P-U1a_interaction_unfrozen',
                   u1['p_u1a_interaction']['per_seed']))

S8 = set(range(1, 9)); S16 = set(range(1, 17))
b_rgo, _ = delta('ax1rgoq1s1', 'ax1rgoq1s0')
b_sgb, _ = delta('ax1sgbq1s1', 'ax1sgbq1s0')
tab.append(battery('P3_rgo_simple_n8_seeds1-8', b_rgo[:8]))
tab.append(battery('P3_rgo_simple_n16_pooled', b_rgo))
tab.append(battery('P3_carrier_rgo-sgb_n16_pooled', b_rgo - b_sgb))
tab.append(battery('P3_carrier_rgo-sgb_n8_seeds1-8', (b_rgo - b_sgb)[:8]))

tm_aw, _ = delta('tm2awareq1s1', 'tm2awareq1s0')
tm_fr, _ = delta('tm2freeq1s1', 'tm2freeq1s0')
tab.append(battery('TM2_aware_simple_n16', tm_aw))
tab.append(battery('TM2_aware_simple_n8_seeds1-8', tm_aw[:8]))
tab.append(battery('TM2_interaction_n16', tm_aw - tm_fr))

s12_t, _ = delta('ax1s12q1s1', 'ax1s12q1s0', seeds=S8)
s12_f, _ = delta('ax1fs12q1s1', 'ax1fs12q1s0', seeds=S8)
tab.append(battery('Scaling12m_interaction_n8', s12_t - s12_f))

w0_t, _ = delta('ax1q1s1', 'ax1q1s0')
w0_f, _ = delta('ax1fq1s1', 'ax1fq1s0')
tab.append(battery('W0_interaction_n16', w0_t - w0_f))
tab.append(battery('W0_aware_simple_n16', w0_t))
tab.append(battery('W0_apt_simple_n16', w0_f))
tab.append(battery('ThreeWay_12m_minus_1m_n8',
                   (s12_t - s12_f) - (w0_t - w0_f)[:8]))

sy_t, _ = delta('ax1q1s1', 'ax1q1s0', dom='synth')
sy_sh, _ = delta('ax1shq1s1', 'ax1shq1s0', dom='synth')
tab.append(battery('Synth_shuffle_collapse_n8', sy_sh - sy_t))

og_rgo, _ = delta('ax1ogrgoq1s1', 'ax1ogrgoq1s0')
og_sgb, _ = delta('ax1ogsgbq1s1', 'ax1ogsgbq1s0')
# registered primary = per-(seed,side) rgo-sgb LEVEL contrast on spin (n=16)
og_lvl = np.concatenate([
    delta('ax1ogrgoq1s0', 'ax1ogsgbq1s0')[0],
    delta('ax1ogrgoq1s1', 'ax1ogsgbq1s1')[0]])
tab.append(battery('Orthogonal_primary_level_n16', og_lvl))
tab.append(battery('Orthogonal_interaction_descr_n16', og_rgo - og_sgb))

st = json.load(open(f'{R}/artifacts/stamping_20260718/stamping_read.json'))
b_srd = np.array(st['b_srd_pooled']); b_sid = np.array(st['b_arm']['sid'])
tab.append(battery('Stamping_srd-sid_n8', b_srd - b_sid))
tab.append(battery('Stamping_S1_srd-apt_n8',
                   b_srd - np.array(st['b_apt_hist'])))

c2, _ = delta('ax1v4q1v400s1', 'ax1v2q1v200s1', seeds=set(range(1, 7)))
tab.append(battery('OptC_C2_task_v400-v200_s1_n6', c2))

OUT['readjudication'] = tab

# ---------------- 2. batch tests ----------------
def welch(a, b):
    t = sps.ttest_ind(a, b, equal_var=False)
    return dict(mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
                p=float(t.pvalue))
w0_int = w0_t - w0_f
OUT['batch_tests'] = dict(
    w0_interaction_b1_vs_b2=welch(w0_int[:8], w0_int[8:]),
    w0_aware_b1_vs_b2=welch(w0_t[:8], w0_t[8:]),
    w0_apt_b1_vs_b2=welch(w0_f[:8], w0_f[8:]),
    p3_carrier_b1_vs_b2=welch((b_rgo - b_sgb)[:8], (b_rgo - b_sgb)[8:]))
sgb0, s_ = vec('ax1sgbq1s0')
OUT['batch_tests']['sgb_s0_level_b1_vs_b2'] = welch(sgb0[:8], sgb0[8:])

# ---------------- 3. pairing correlations ----------------
t0v, _ = vec('ax1q1s0'); t1v, _ = vec('ax1q1s1')
f0v, _ = vec('ax1fq1s0'); f1v, _ = vec('ax1fq1s1')
r1v, _ = vec('ax1rgoq1s1'); g1v, _ = vec('ax1sgbq1s1')
cor = lambda a, b: float(np.corrcoef(a, b)[0, 1])
OUT['pairing'] = dict(
    corr_Btask_Bapt_seeds1_8=cor(w0_t[:8], w0_f[:8]),
    corr_Btask_Bapt_n16=cor(w0_t, w0_f),
    sd_diff=float((w0_t[:8] - w0_f[:8]).std(ddof=1)),
    sd_if_independent=float(math.sqrt(w0_t[:8].var(ddof=1) + w0_f[:8].var(ddof=1))),
    corr_task_s0_s1=cor(t0v, t1v), corr_apt_s0_s1=cor(f0v, f1v),
    corr_rgo_s1_sgb_s1=cor(r1v, g1v))

# ---------------- 4. floor censoring ----------------
px = np.concatenate([vec(m)[0] for m in
                     ['ax1pxpxq1ms0', 'ax1pxpxq1ms1', 'ax1fpxpxq1ms0', 'ax1fpxpxq1ms1']])
apt = np.concatenate([f0v, f1v])
uzf0, _ = vec('ax1uzfq1s0'); uzf1, _ = vec('ax1uzfq1s1')
uzt0, _ = vec('ax1uztq1s0'); uzt1, _ = vec('ax1uztq1s1')
px_lift = np.concatenate([delta('ax1uzpxpxq1ms0', 'ax1pxpxq1ms0')[0],
                          delta('ax1uzpxpxq1ms1', 'ax1pxpxq1ms1')[0]])
apt_lift = (np.concatenate([delta('ax1uzfq1s0', 'ax1fq1s0')[0],
                            delta('ax1uzfq1s1', 'ax1fq1s1')[0]]))
task_lift = (np.concatenate([delta('ax1uztq1s0', 'ax1q1s0', seeds=S8)[0],
                             delta('ax1uztq1s1', 'ax1q1s1', seeds=S8)[0]]))
def liftstat(x):
    x = np.asarray(x, float)
    # collapse to per-seed means over the two sides (8 clusters)
    n = len(x) // 2
    per_seed = (x[:n] + x[n:]) / 2
    return dict(mean=float(per_seed.mean()), ci=pct_ci(per_seed),
                pos=int((per_seed > 0).sum()), n=n)
OUT['floor_censoring'] = dict(
    pixel_x2_all=dict(n=len(px), mean=float(px.mean()), sd=float(px.std(ddof=1))),
    proprio_apt_all=dict(n=len(apt), mean=float(apt.mean()), sd=float(apt.std(ddof=1))),
    welch_p=welch(px, apt)['p'],
    lift_pixel=liftstat(px_lift), lift_proprio_apt=liftstat(apt_lift),
    lift_proprio_task=liftstat(task_lift))

# ---------------- 5. ladder adjacency ----------------
b_full = w0_t[:8]
b_rl, _ = delta('ax1rlq1s1', 'ax1rlq1s0')
b_sh, _ = delta('ax1shq1s1', 'ax1shq1s0')
b_vgo, _ = delta('ax1vgoq1s1', 'ax1vgoq1s0')
b_apt8 = w0_f[:8]
lad = dict(full=b_full, rgo=b_rgo[:8], rl=b_rl, sh=b_sh, sgb=b_sgb[:8],
           vgo=b_vgo, apt=b_apt8)
adj = {}
order = ['full', 'rgo', 'rl', 'sh', 'sgb', 'vgo', 'apt']
for a, b_ in zip(order[:-1], order[1:]):
    d = lad[a] - lad[b_]
    adj[f'{a}-{b_}'] = dict(mean=float(d.mean()), perm_p=perm_p(d))
OUT['ladder_adjacency'] = dict(
    arm_deltas={k: float(v.mean()) for k, v in lad.items()}, adjacent=adj)

# ---------------- 6. W1 ICC ----------------
rows = list(csv.DictReader(open(f'{R}/artifacts/w123_e3v2_20260716/auc_w123.csv')))
coll = collections.defaultdict(dict)
for r in rows:
    m = re.match(r'ax1(f?)w(\d+)s([01])$', r['mode'])
    if not m or r['domain'] != 'finger': continue
    arm = 'apt' if m.group(1) else 'task'
    coll[int(m.group(2))].setdefault((arm, m.group(3)), []).append(
        float(r['auc100k']))
ints = {}
for w, cells in sorted(coll.items()):
    if len(cells) < 4: continue
    seed_ints = []
    for i in range(2):
        try:
            ti = cells[('task', '1')][i] - cells[('task', '0')][i] \
                 - (cells[('apt', '1')][i] - cells[('apt', '0')][i])
            seed_ints.append(ti)
        except IndexError:
            pass
    ints[w] = seed_ints
per_coll_mean = np.array([np.mean(v) for v in ints.values()])
within_var = float(np.mean([np.var(v, ddof=1) for v in ints.values()
                            if len(v) == 2]))
between_var_raw = float(np.var(per_coll_mean, ddof=1))
sigma2_between = max(0.0, between_var_raw - within_var / 2)
OUT['w1_icc'] = dict(
    n_collectors=len(ints),
    grand_mean=float(per_coll_mean.mean()),
    within_var=within_var, between_var_of_means=between_var_raw,
    sigma2_between=sigma2_between,
    icc=float(sigma2_between / (sigma2_between + within_var)))

# ---------------- 7. Phase-5a refit ----------------
drv = {}
for r in csv.DictReader(open(f'{R}/artifacts/phase5a_20260707/drivers.csv')):
    if r['cov'] and r['occ_phys']:
        drv[(r['mode'], r['domain'], r['seed'], r['milestone'])] = (
            float(r['cov']), float(r['occ_phys']))
rows5, clus, doms = [], [], []
for r in csv.DictReader(open(f'{R}/artifacts/phase5a_20260707/auc.csv')):
    k = (r['mode'], r['domain'], r['seed'], r['milestone'])
    if k in drv and r['domain'] in ('cup', 'finger'):
        rows5.append((drv[k][0], drv[k][1], float(r['auc100k'])))
        clus.append((r['mode'], r['domain'], r['seed']))
        doms.append(r['domain'])
rows5 = np.array(rows5)
def cluster_ols(X, y, clusters):
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    bread = np.linalg.inv(Xd.T @ Xd)
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    uniq = sorted(set(clusters))
    for c in uniq:
        idx = [i for i, cc in enumerate(clusters) if cc == c]
        u = Xd[idx].T @ resid[idx]
        meat += np.outer(u, u)
    G = len(uniq)
    vcov = bread @ meat @ bread * G / (G - 1)
    se = np.sqrt(np.diag(vcov))
    return beta, se, G
def refit(mask, label):
    X = rows5[mask][:, :2].copy(); y = rows5[mask][:, 2].copy()
    Xz = (X - X.mean(0)) / X.std(0); yz = (y - y.mean()) / y.std()
    cl = [c for c, m in zip(clus, mask) if m]
    beta, se, G = cluster_ols(Xz, yz, cl)
    return {label: dict(n_rows=int(mask.sum()), n_clusters=G,
                        beta_cov=float(beta[1]), se_cov=float(se[1]),
                        beta_occ_phys=float(beta[2]), se_occ=float(se[2]),
                        r_cov_occ=float(np.corrcoef(Xz.T)[0, 1]))}
doms = np.array(doms)
OUT['phase5a_refit'] = {}
OUT['phase5a_refit'].update(refit(np.ones(len(rows5), bool), 'pooled'))
OUT['phase5a_refit'].update(refit(doms == 'cup', 'cup_only'))
OUT['phase5a_refit'].update(refit(doms == 'finger', 'finger_only'))
# outcome variance by domain
for dm in ('cup', 'finger'):
    runs = collections.defaultdict(list)
    for (c, v) in zip(clus, rows5[:, 2][np.ones(len(rows5), bool)]):
        pass
run_means = collections.defaultdict(list)
for c, row in zip(clus, rows5):
    run_means[c].append(row[2])
for dm in ('cup', 'finger'):
    ms = [np.mean(v) for c, v in run_means.items() if c[1] == dm]
    OUT['phase5a_refit'][f'{dm}_between_run_sd'] = float(np.std(ms, ddof=1))
    OUT['phase5a_refit'][f'{dm}_run_mean'] = float(np.mean(ms))

# ---------------- 8. coverage simulation ----------------
def coverage_sim(pop, n, sims=3000, Bb=2000):
    pop = np.asarray(pop, float) - np.mean(pop)  # H0 true
    rng = np.random.default_rng(1)
    hit_pct = hit_t = rej_perm = 0
    for _ in range(sims):
        x = rng.choice(pop, n, replace=True)
        bm = x[rng.integers(0, n, (Bb, n))].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        if lo <= 0 <= hi: hit_pct += 1
        t = sps.ttest_1samp(x, 0)
        ci = t.confidence_interval()
        if ci.low <= 0 <= ci.high: hit_t += 1
        # permutation via normal approx of sign-flip dist (exact enough here)
        s = rng.choice([-1, 1], (2000, n))
        p = (np.abs((x * s).mean(1)) >= abs(x.mean()) - 1e-12).mean()
        if p < 0.05: rej_perm += 1
    return dict(pct_coverage=hit_pct / sims, t_coverage=hit_t / sims,
                perm_type1=rej_perm / sims, sims=sims)
OUT['coverage_sim_w0_deltas'] = dict(
    n8=coverage_sim(w0_int, 8), n16=coverage_sim(w0_int, 16))
OUT['coverage_sim_u1_deltas'] = dict(
    n8=coverage_sim(np.array(u1['p_u1a_interaction']['per_seed']), 8))

# ---------------- 9. group-sequential sim ----------------
def groupseq_sim(sd=127.0, sims=20000, Bb=800):
    rng = np.random.default_rng(2)
    fire1 = fire2 = 0
    est_sel = []
    for _ in range(sims):
        x1 = rng.normal(0, sd, 8)
        bm = x1[rng.integers(0, 8, (Bb, 8))].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        if (lo > 0) == (hi > 0):
            fire1 += 1; continue
        x2 = rng.normal(0, sd, 8)
        x = np.concatenate([x1, x2])
        bm = x[rng.integers(0, 16, (Bb, 16))].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        if (lo > 0) == (hi > 0): fire2 += 1
    return dict(fire_look1=fire1 / sims, fire_look2_only=fire2 / sims,
                fire_either=(fire1 + fire2) / sims, sims=sims)
OUT['groupseq_sim'] = groupseq_sim()

# ---------------- 10. BH over registered primaries ----------------
# perm p-values: computed here where per-seed data is local; otherwise from
# the archived read jsons (marked source=json).
prim = {
    'W1_within_collector': (0.0001220703125, 'json:w123_read'),
    'W0_interaction': (perm_p(w0_int), 'computed'),
    'Phase5a_occ_phys': (0.0058, 'RESULTS(bootstrap present; adopted)'),
    'P0_2x2': (0.0078125, 'reviewer[A-adopted]'),
    'UnfrozenCalib_rgo-sgb_level': (0.0078125, 'reviewer[A-adopted]'),
    'P3_S2_shuffle': (None, 'compute below'),
    'OptC_C2': (perm_p(c2), 'computed'),
    'Synth_collapse': (perm_p(sy_sh - sy_t), 'computed'),
    'P3_carrier_pooled': (perm_p(b_rgo - b_sgb), 'computed'),
    'U1a': (perm_p(np.array(u1['p_u1a_interaction']['per_seed'])), 'computed'),
    'Scaling12_interaction': (perm_p(s12_t - s12_f), 'computed'),
    'TM2_F1_interaction': (perm_p(tm_aw - tm_fr), 'computed'),
    'Orthogonal_primary': (perm_p(og_rgo - og_sgb), 'computed'),
    'Stamping_primary': (perm_p(b_srd - b_sid), 'computed'),
    'VGO_dd': (0.9296875, 'json:vgo_extended'),
    'U1b_apt_occupancy': (perm_p(np.array(u1['p_u1b_apt_occupancy']['per_seed'])), 'computed'),
    'U2_srd-sid_unfrozen': (0.9375, 'json-ish; null'),
    'U3_orth_unfrozen': (0.65, 'json-ish; null'),
    'W2_cup_apt': (0.180633544921875, 'json:w123_read'),
    'PE_P-PE2': (0.20, 'null; approx'),
    'RDE_P-RD2': (0.15, 'null; approx'),
}
# P3 shuffle (S2): xsgb vs sgb? shuffle arm = ax1xsgbq1s1 (shuffled labels on
# sgb? actually S2 = shuffle collapse rgo vs shuffled-rgo); use reviewer p.
prim['P3_S2_shuffle'] = (0.0234375, 'reviewer[A-adopted]')
names = sorted(prim, key=lambda k: prim[k][0])
ps = np.array([prim[k][0] for k in names])
m = len(ps)
bh_thresh = 0.05 * (np.arange(1, m + 1)) / m
passed = ps <= bh_thresh
kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
OUT['bh'] = dict(
    family_note='Assembled family (21 registered primaries with local data or '
                'archived p; the reviewer used 30 - this is a subset, labeled).',
    rows=[dict(name=k, p=float(prim[k][0]), source=prim[k][1],
               bh_survives=bool(i < kmax))
          for i, k in enumerate(names)])

json.dump(OUT, open(os.path.dirname(os.path.abspath(__file__)) +
                    '/review_res_B.json', 'w'), indent=1, default=float)
print('WROTE review_res_B.json')
for row in OUT['readjudication']:
    print(f"{row['name']:38s} mean {row['mean']:+8.1f} pos {row['pos']:2d}/{row['n']:2d} "
          f"pct[{row['pct_ci'][0]:+7.1f},{row['pct_ci'][1]:+7.1f}]{'F' if row['pct_fires'] else ' '} "
          f"bca{'F' if row['bca_fires'] else '.'} bt{'F' if row['boot_t_fires'] else '.'} "
          f"perm {row['perm_p']:.4f} t {row['t_p']:.3f} sign {row['sign_p']:.3f} "
          f"LOO {row['loo_fires']} TOST {row['tost_p_pm24p7']:.3f}")
print(json.dumps({k: OUT[k] for k in ['batch_tests', 'pairing', 'floor_censoring',
                                      'ladder_adjacency', 'w1_icc', 'phase5a_refit',
                                      'coverage_sim_w0_deltas', 'coverage_sim_u1_deltas',
                                      'groupseq_sim']}, indent=1, default=float))
