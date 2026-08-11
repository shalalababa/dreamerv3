"""D1 closure (2026-08-11): proprio leak-excluded E4 recompute + X2 anchor
recompute + canonical cross-check. Executes the INSTANCE items of review
finding D1 (Review_Resolution_20260808) under the registered
leak-exclusion rule. Labeled sensitivity: no verdict re-adjudicated.
"""
import collections, glob, json, os
import numpy as np

R = os.getcwd()
PAY = f'{R}/local_results/paper1_d1_closure_payload_20260810_232932/d1_closure'
OUT = {}

# ---------- leg A: proprio finger_v1, regen q1 TASK fits (original weights)
man = json.load(open(f'{R}/local_results/e4_goodhart_20260714_223140/probesets/finger_v1/manifest.json'))
ps = np.load(f'{PAY}/probeset_e4.npz', allow_pickle=True)
in_r = np.asarray(ps['in_regime'], bool)
src_arr = np.asarray(ps['source'])
assert man['n_episodes'] == 60 and in_r.shape == (60, 1001)
row_of, r0 = {}, 0
for s in man['sources']:
    picked = sorted(man['sources'][s]['picked'])
    for pos, j in enumerate(picked):
        row_of[(s, j)] = r0 + pos
    # mapping sanity: the npz per-row source labels must match the block
    assert all(src_arr[r0 + p] == s for p in range(len(picked))), s
    r0 += len(picked)
OUT['probeset_manifest_sha256'] = man['sha256']
# d1_overlap.json hits (committed): fit-buffer/probe overlap, side1 only
ov = json.load(open(f'{R}/artifacts/review_response_20260808/d1_overlap.json'))
hits1 = ov['finger']['q1_side1']['hits']
assert ov['finger']['q1_side0']['overlap'] == 0
name_map = {'pilot_goal': 'goal', 'pilot_p2e': 'p2e', 'pilot_random': 'random'}
leak_rows_s1 = sorted(row_of[(name_map[s], j)] for s, j in hits1)
OUT['proprio_leak_rows_s1'] = leak_rows_s1

def rew_nll_in(npz_path, inr, drop_rows=()):
    e = np.asarray(np.load(npz_path)['rew_nll_h0'], float)
    keep = np.ones(e.shape[0], bool)
    for r_ in drop_rows:
        keep[r_] = False
    m = inr[keep]
    return float(e[keep][m].mean())

prop = {}
for side in (0, 1):
    fulls, excls, repro = [], [], []
    for seed in range(1, 9):
        run = f'ax1wm_finger_q1s{side}_seed{seed}'
        p = f'{R}/local_results/regen_ridge_e4_20260810_220849/runs/{run}/e4_finger_v1refit/errors.npz'
        full = rew_nll_in(p, in_r)
        excl = rew_nll_in(p, in_r, leak_rows_s1 if side == 1 else ())
        # cross-check vs the summary's frame-weighted rew_nll_in
        s = json.load(open(p.replace('errors.npz', 'summary.json')))
        summ = s['horizon_stats']['0']['reward_head']['in_regime']['mean']
        repro.append(abs(full - summ))
        fulls.append(full); excls.append(excl)
    prop[f's{side}'] = dict(
        n=8, full=float(np.mean(fulls)), leak_excluded=float(np.mean(excls)),
        delta=float(np.mean(excls) - np.mean(fulls)),
        max_summary_repro_err=float(max(repro)))
OUT['proprio_q1_task'] = dict(
    results=prop, substrate='regen bundle q1 task fits = restored ORIGINAL '
    'July weights (amendment 2 census)',
    note='rew_nll_in from errors.npz, frame-weighted in-regime mean; s1 '
         'drops the 7 fit-buffer-overlap probe episodes; s0 overlap = 0 '
         'by the committed d1_overlap census (identity expected)')

# ---------- leg B: X2 anchor (pxpx) + leg C canonical rde/pe cross-check
psx = np.load(f'{R}/local_results/pixel_swamping_20260725_090958/runroot_light/e4_probesets/fingerpx_v1/probeset_e4.npz', allow_pickle=True)
in_rx = np.asarray(psx['in_regime'], bool)
manx = json.load(open(f'{R}/local_results/pixel_swamping_20260725_090958/runroot_light/e4_probesets/fingerpx_v1/manifest.json'))
row_of_x, r0 = {}, 0
for s in manx['sources']:
    for pos, j in enumerate(sorted(manx['sources'][s]['picked'])):
        row_of_x[(s, j)] = r0 + pos
    r0 += len(manx['sources'][s]['picked'])
ovx = ov['pixel_fingerpx']
leak_x = {f's{i}': sorted(row_of_x[(s, j)]
                          for s, j in ovx[f'q1m_side{i}']['hits'])
          for i in (0, 1)}
OUT['pixel_leak_rows'] = leak_x
pix = {}
for tag, pat in (('X2_pxpx', 'pxpx'), ('rde', 'rdepx'), ('pe', 'pepx')):
    per = {}
    for side in (0, 1):
        fulls, excls = [], []
        for seed in range(1, 9):
            p = f'{PAY}/errors_payload/ax1wm_finger_{pat}q1ms{side}_seed{seed}/e4_fingerpx_v1/errors.npz'
            fulls.append(rew_nll_in(p, in_rx))
            excls.append(rew_nll_in(p, in_rx, leak_x[f's{side}']))
        per[f's{side}'] = dict(
            n=8, full=float(np.mean(fulls)),
            leak_excluded=float(np.mean(excls)),
            delta=float(np.mean(excls) - np.mean(fulls)))
    pix[tag] = per
OUT['pixel'] = pix
# canonical cross-check vs the 8-Aug run (rde/pe legs)
prev = json.load(open(f'{R}/artifacts/review_response_20260808/zero_compute.json'))
prev_leak = prev['d1_pixel_leak_sensitivity']['results']
# mapping identity vs the archived leaked_rows lists (independent check)
assert leak_x == {k: sorted(v) for k, v in prev['d1_pixel_leak_sensitivity']['leaked_rows'].items()}, (leak_x, prev['d1_pixel_leak_sensitivity']['leaked_rows'])
cc = {}
for tag_new, tag_old in (('rde', 'rde'), ('pe', 'pe')):
    cc[tag_new] = {
        side: dict(delta_full=abs(pix[tag_new][side]['full']
                                  - prev_leak[tag_old][side]['full']),
                   delta_excl=abs(pix[tag_new][side]['leak_excluded']
                                  - prev_leak[tag_old][side]['leak_excluded']))
        for side in ('s0', 's1')}
OUT['canonical_crosscheck_vs_0808'] = cc
# X2 anchor pins consumed by the executed rde/pe reads (sensitivity only)
OUT['x2_anchor_pins'] = dict(
    baseline_pin=23.338872993169502,
    member_band=[19.409267325402837, 27.285688932142705],
    note='pins consumed by executed rde/pe reads; leak-excluded X2 values '
         'here are a DISCLOSED SENSITIVITY on those pins, never a '
         're-adjudication')

json.dump(OUT, open(f'{R}/artifacts/d1_closure_20260811/d1_closure.json', 'w'),
          indent=1, sort_keys=True)
print(json.dumps(OUT, indent=1, sort_keys=True))
