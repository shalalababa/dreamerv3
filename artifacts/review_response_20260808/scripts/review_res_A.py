"""Paper-1 review resolution — Part A: zero-compute analyses (§3.1 + D1 sensitivity).

Labeled EXPLORATORY/post-read. Deterministic: rng(0), B=10000.
Outputs JSON to scratchpad; artifact copy made after inspection.
"""
import csv, glob, json, os, re, collections
import numpy as np

R = '/home/rickybao/projects/dreamerv3'
OUT = {}
rng0 = lambda: np.random.default_rng(0)
B = 10000

def boot_ci(x, B=B, stat=np.mean):
    x = np.asarray(x, float); rng = rng0()
    idx = rng.integers(0, len(x), (B, len(x)))
    means = stat(x[idx], axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

def boot_ci_unpaired(a, b, B=B):
    a, b = np.asarray(a, float), np.asarray(b, float); rng = rng0()
    da = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    d = da - db
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

def perm_p(x):
    """Exact sign-flip permutation p (two-sided) for mean of paired deltas."""
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

# ---------- AUC gathering ----------
def gather_auc(paths):
    """(mode, domain, seed) -> auc100k, preferring max n_ep_100k; ckpt500000 only."""
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
]
A = gather_auc(CANON)

def cell(mode, dom='finger', seeds=None):
    xs = sorted((s, v) for (m, d, s), v in A.items() if m == mode and d == dom
                and (seeds is None or s in seeds))
    return [v for _, v in xs], [s for s, _ in xs]

# ---------- 1. random-policy floor ----------
floor = []
for sd in range(1, 6):
    p = (f'{R}/local_results/runroot_snapshot_20260705_101044/runroot_light/'
         f'pretrain_random_finger_seed{sd}/scores.jsonl')
    eps = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            eps.append((int(rec['step']), float(rec['episode/score'])))
    eps.sort()
    scores = [s for st, s in eps if st <= 100_000]
    floor.append(dict(seed=sd, auc100k=float(np.mean(scores)), n_ep=len(scores)))
fv = [f['auc100k'] for f in floor]
rf_cells = {}
for m in ['ax1fq1s0', 'ax1fq1s1', 'ax1fq2s0', 'ax1fq2s1']:
    v, s = cell(m)
    rf_cells[m] = dict(mean=float(np.mean(v)), n=len(v))
OUT['random_policy_floor'] = dict(
    per_seed=floor, mean=float(np.mean(fv)), sd=float(np.std(fv, ddof=1)),
    band_2sd=[float(np.mean(fv) - 2 * np.std(fv, ddof=1)),
              float(np.mean(fv) + 2 * np.std(fv, ddof=1))],
    reward_free_cells=rf_cells,
    all_inside_1sd_band=all(
        abs(c['mean'] - np.mean(fv)) <= np.std(fv, ddof=1) * 1.0 + 5
        for c in rf_cells.values()))

# ---------- 2. worse than random init ----------
uzf0, _ = cell('ax1uzfq1s0'); uzf1, _ = cell('ax1uzfq1s1')
scr, _ = cell('scratch')
uzt0, _ = cell('ax1uztq1s0'); uzt1, _ = cell('ax1uztq1s1')
OUT['worse_than_random_init'] = dict(
    apt_unfrozen_s0=float(np.mean(uzf0)), apt_unfrozen_s1=float(np.mean(uzf1)),
    scratch=float(np.mean(scr)), scratch_sd=float(np.std(scr, ddof=1)),
    ratio_s0=float(np.mean(uzf0) / np.mean(scr)),
    ratio_s1=float(np.mean(uzf1) / np.mean(scr)),
    task_unfrozen_s0=float(np.mean(uzt0)), task_unfrozen_s1=float(np.mean(uzt1)),
    apt_s0_minus_scratch_ci=boot_ci_unpaired(uzf0, scr),
    apt_s1_minus_scratch_ci=boot_ci_unpaired(uzf1, scr))

# ---------- 3. log-scale interaction (W0 n=16 + pixel) ----------
def two_by_two_log(tm0, tm1, fm0, fm1, seeds=None):
    t0, s0 = cell(tm0, seeds=seeds); t1, s1 = cell(tm1, seeds=seeds)
    f0, s2 = cell(fm0, seeds=seeds); f1, s3 = cell(fm1, seeds=seeds)
    n = min(map(len, [t0, t1, f0, f1]))
    assert s0 == s1 == s2 == s3, (s0, s1, s2, s3)
    t0, t1, f0, f1 = map(lambda x: np.asarray(x[:n], float), [t0, t1, f0, f1])
    logdel = (np.log(t1) - np.log(t0)) - (np.log(f1) - np.log(f0))
    lin = (t1 - t0) - (f1 - f0)
    return dict(
        n=n, seeds=s0,
        linear_interaction=float(lin.mean()), linear_ci=boot_ci(lin),
        log_interaction=float(logdel.mean()), log_ci=boot_ci(logdel),
        ratio_of_ratios=float(np.exp(logdel.mean())),
        ratio_ci=[float(np.exp(c)) for c in boot_ci(logdel)],
        pos=int((logdel > 0).sum()), perm_p=perm_p(logdel),
        aware_occ_ratio=float(np.mean(t1) / np.mean(t0)),
        free_occ_ratio=float(np.mean(f1) / np.mean(f0)))
OUT['log_interaction_w0'] = two_by_two_log(
    'ax1q1s0', 'ax1q1s1', 'ax1fq1s0', 'ax1fq1s1')
OUT['log_interaction_pixel'] = two_by_two_log(
    'ax1pxpxq1ms0', 'ax1pxpxq1ms1', 'ax1fpxpxq1ms0', 'ax1fpxpxq1ms1')

# ---------- 7. volume 2x2 at n=12 ----------
cells = {}
for m, lab in [('ax1v2q1v200s0', 'v200_s0'), ('ax1v2q1v200s1', 'v200_s1'),
               ('ax1v4q1v400s0', 'v400_s0'), ('ax1v4q1v400s1', 'v400_s1')]:
    v, s = cell(m)
    cells[lab] = dict(mean=float(np.mean(v)), n=len(v), seeds=s)
def contrast_cells(ma, mb):
    a, sa = cell(ma); b, sb = cell(mb)
    common = sorted(set(sa) & set(sb))
    if len(common) >= 6:
        av = np.array([a[sa.index(s)] for s in common])
        bv = np.array([b[sb.index(s)] for s in common])
        d = av - bv
        return dict(kind='paired', n=len(common), mean=float(d.mean()),
                    ci=boot_ci(d), pos=int((d > 0).sum()), perm_p=perm_p(d))
    return dict(kind='unpaired', n=[len(a), len(b)],
                mean=float(np.mean(a) - np.mean(b)),
                ci=boot_ci_unpaired(a, b))
OUT['volume_2x2_n12'] = dict(
    cells=cells,
    orientation_note='v400 pair is SIDE-INVERTED (Option-C disclosure): v400s0 '
        'is the HIGH-occ (.290) side, v400s1 the LOW-occ (.094) side. '
        'Registered C1 = task [v400s0 - v400s1].',
    volume_v400s0_minus_v200s0=contrast_cells('ax1v4q1v400s0', 'ax1v2q1v200s0'),
    volume_v400s1_minus_v200s1=contrast_cells('ax1v4q1v400s1', 'ax1v2q1v200s1'),
    occ_hi_minus_lo_at_v200=contrast_cells('ax1v2q1v200s1', 'ax1v2q1v200s0'),
    registered_C1_v400s0_minus_v400s1_n12=contrast_cells(
        'ax1v4q1v400s0', 'ax1v4q1v400s1'),
    registered_C1_first6=None)
# registered C1 on the original n=6 seeds (1-6)
def contrast_seeds(ma, mb, seeds):
    a, sa = cell(ma, seeds=seeds); b, sb = cell(mb, seeds=seeds)
    d = np.array(a) - np.array(b)
    return dict(n=len(d), mean=float(d.mean()), ci=boot_ci(d),
                pos=int((d > 0).sum()), perm_p=perm_p(d))
OUT['volume_2x2_n12']['registered_C1_first6'] = contrast_seeds(
    'ax1v4q1v400s0', 'ax1v4q1v400s1', set(range(1, 7)))

# ---------- 8. rew_nll_out panel (h0, all arms, both sides) ----------
E4S = {
    'rgo': f'{R}/artifacts/e4_amend1_optionc_20260723/e4_finger_v1_p3amend1_rgo.csv',
    'sgb': f'{R}/artifacts/e4_amend1_optionc_20260723/e4_finger_v1_p3amend1_sgb.csv',
}
# p3 wave arms (vgo/rl/sh + seeds1-8 rgo/sgb) live in the p3 wave e4 csv
for extra in glob.glob(f'{R}/artifacts/p3_wave_20260717/*.csv'):
    E4S[os.path.basename(extra)] = extra
panel = collections.defaultdict(dict)
for tag, p in E4S.items():
    if not os.path.exists(p): continue
    for r in csv.DictReader(open(p)):
        if r.get('horizon') != '0': continue
        rid = r['run_id']
        m = re.search(r'ax1wm_finger_([a-z0-9]+?)q1(s\d)_seed(\d+)', rid)
        if not m: continue
        arm, side, seed = m.groups()
        panel[(arm, side)][int(seed)] = (
            float(r['rew_nll_in']), float(r['rew_nll_out']), float(r['rew_nll_all']))
OUT['rew_nll_out_panel'] = {
    f'{arm}_{side}': dict(
        n=len(d),
        rew_nll_in=float(np.mean([v[0] for v in d.values()])),
        rew_nll_out=float(np.mean([v[1] for v in d.values()])),
        rew_nll_all=float(np.mean([v[2] for v in d.values()])))
    for (arm, side), d in sorted(panel.items())}

# ---------- D1 pixel leak-excluded sensitivity ----------
ps = np.load(f'{R}/local_results/pixel_swamping_20260725_090958/runroot_light/'
             'e4_probesets/fingerpx_v1/probeset_e4.npz', allow_pickle=True)
in_r = np.asarray(ps['in_regime'], bool)          # (120, 1001)
man = json.load(open(f'{R}/local_results/pixel_swamping_20260725_090958/'
                     'runroot_light/e4_probesets/fingerpx_v1/manifest.json'))
order = ['random100', 'random104', 'random105', 'p2e99', 'p2e101', 'apt101']
# row index of (source, replay-ordinal) in the npz: sources are stacked in
# manifest-dict order with 20 rows each, rows sorted by picked ordinal
src_order = list(man['sources'].keys())
row_of = {}
r0 = 0
for s in src_order:
    for pos, j in enumerate(sorted(man['sources'][s]['picked'])):
        row_of[(s, j)] = r0 + pos
    r0 += len(man['sources'][s]['picked'])
leak = {
    's0': [('random100', 2), ('random100', 42), ('random104', 53), ('random104', 81),
           ('random105', 16), ('p2e101', 16), ('apt101', 10), ('apt101', 53)],
    's1': [('random100', 36), ('random105', 23), ('random105', 42),
           ('apt101', 32), ('apt101', 51)],
}
leak_rows = {side: sorted(row_of[k] for k in ks) for side, ks in leak.items()}

def rew_nll_in_from_errors(npz_path, drop_rows=()):
    z = np.load(npz_path)
    e = np.asarray(z['rew_nll_h0'], float)  # (120, 1001)
    keep = np.ones(e.shape[0], bool)
    for r_ in drop_rows: keep[r_] = False
    mask = in_r[keep]
    return float(e[keep][mask].mean())

leak_sens = {}
for tag, pat, runs_glob in [
    ('X2', 'pxpx', f'{R}/local_results/pixel_swamping_20260725_104532/runroot_light/'
     'ax1wm_finger_pxpxq1m*/e4_fingerpx_v1/errors.npz'),
    ('rde', 'rdepx', f'{R}/local_results/rde_pixel_20260804_180214/runs/'
     'ax1wm_finger_rdepxq1m*/e4_fingerpx_v1/errors.npz'),
    ('pe', 'pepx', f'{R}/local_results/pe_pixel_20260802_165601/runroot/'
     'ax1wm_finger_pepxq1m*/e4_fingerpx_v1/errors.npz'),
]:
    per_side = collections.defaultdict(lambda: ([], []))
    for p in sorted(glob.glob(runs_glob)):
        side = 's0' if f'{pat}q1ms0' in p else 's1'
        full = rew_nll_in_from_errors(p)
        excl = rew_nll_in_from_errors(p, leak_rows[side])
        per_side[side][0].append(full); per_side[side][1].append(excl)
    leak_sens[tag] = {
        side: dict(n=len(f), full=float(np.mean(f)), leak_excluded=float(np.mean(x)),
                   delta=float(np.mean(x) - np.mean(f)))
        for side, (f, x) in sorted(per_side.items())}
OUT['d1_pixel_leak_sensitivity'] = dict(
    leaked_rows=leak_rows, results=leak_sens,
    note='rew_nll_in recomputed from errors.npz over in-regime frames, all 120 '
         'probe episodes vs excluding the leaked episodes of the fit side\'s own '
         'buffer. Frame-weighted mean (pools frames across episodes).')

json.dump(OUT, open(os.path.dirname(os.path.abspath(__file__)) +
                    '/review_res_A.json', 'w'), indent=1)
print(json.dumps(OUT, indent=1))
