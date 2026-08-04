"""Frozen read for the breadth-causal wave (matched-f_R diversity pair).

Registered in prereg/PREREG_breadth_wave_20260803.md, adjudicating the
frozen theory predictions PREREG_breadth_theory_20260803 (P-BD1 causal
breadth effect, P-BD2 anomaly-attribution ordering, P-BD3 membership
dissociation). Committed BEFORE any q1d buffer, fit, or adaptation
exists (only the value-blind feasibility search output exists).

Cells (canonical AUC csv modes, task arm, finger, matched f_R ~ 0.30):
  lo-div side0  ax1bdq1ds0  seeds 1-8  (curated starved support,
                                        search diversity_pr 7.495)
  hi-div side1  ax1bdq1ds1  seeds 1-8  (curated broad support,
                                        search diversity_pr 12.336)
  comparator    ax1v2q1v200s1 seeds 1-12 (natural f=0.3232,
                                        diversity_pr 9.2300, matched
                                        volume; frozen rows from the
                                        volume_repl bundle — anchor for
                                        the P-BD2 ordering only, never
                                        part of the primary)

PRIMARY (sole confirmatory, P-BD1): mean[hi-div] - mean[lo-div],
AUC100k, two-sample seed-cluster bootstrap (B=10K, default_rng(0));
FIRES iff CI entirely > 0; REVERSED iff CI entirely < 0 (registered
refutation of the lambda-monotonicity form); a straddle is the
registered INFORMATIVE NULL (conditional on the gates below, which the
read enforces — the diversity_pr manipulation is instrument-verified).

P-BD2 (registered secondary, weak point-ordering, non-confirmatory):
mean[lo-div] < mean[v200s1] <= mean[hi-div]. Descriptive CIs for both
legs (rng 1, 2) are recorded; the ordering itself is a point statement
(8-vs-12 CIs on each leg are underpowered and NOT part of the rule).

P-BD3 (registered secondary): BOTH cells' task fits' in-regime
reward-NLL at horizon 0 (frozen E4 machinery, finger_v1 probe set),
8-seed cluster CIs (rng 3 = hi, rng 4 = lo): membership intact iff CI
entirely <= 1.5 nats; out iff CI entirely >= 2.0 (the registered
constants, same substrate/anchors; v200s1 anchor mean 0.827); else
indeterminate. INCLUSION-LEAK alternative: lo-div nonmember means the
manipulation leaked into legibility — a fired P-BD1 is then reported
under the inclusion reading and pure-breadth mediation is NOT licensed.

Validity gates (violation => the read REFUSES; the wave should have
halted pre-fit at the registered spectral gate):
  - both curated-side jsons: spectral_v1_1_20260724, 200 episodes,
    domain finger, sides lo/hi (swap-pinned);
  - BOTH sides' instrument f_rewarded in [0.27, 0.34] AND
    |f_hi - f_lo| <= 0.02  (the matched-legibility control);
  - instrument diversity_pr separation div_hi - div_lo >= 3.0
    (manipulation-validity: the mediator must actually move);
  - all-or-nothing qc-passing 8-seed cohorts on BOTH cells (both are
    arms of the primary — either failing is FATAL, unlike highfr's
    side0) and exactly seeds 1-12 on the comparator;
  - frozen reference json pinned bit-exact (f 0.32317182817182816,
    diversity_pr 9.230033291492585).

Usage:
  python -m analysis.breadth_read --auc <fresh auc.csv> \
      --auc_frozen <volume_repl bundle auc.csv> \
      --spectral_lo <q1d/spectral_side0.json> \
      --spectral_hi <q1d/spectral_side1.json> \
      --block_lo <q1d/block_side0.json> \
      --block_hi <q1d/block_side1.json> \
      --e4 <e4_finger_v1_bd.csv> --output <dir>
  python -m analysis.breadth_read --selfcheck
"""

import argparse
import csv
import json
import os

import numpy as np

B_BOOT = 10_000
# Dedicated rng seed per registered statistic (draw order independence).
RNG_PRIMARY, RNG_HI_V, RNG_LO_V, RNG_NLL_HI, RNG_NLL_LO = 0, 1, 2, 3, 4

MODE_LO = 'ax1bdq1ds0'
MODE_HI = 'ax1bdq1ds1'
MODE_V200S1 = 'ax1v2q1v200s1'
WM_LO = 'ax1wm_finger_bdq1ds0_seed'
WM_HI = 'ax1wm_finger_bdq1ds1_seed'
SEEDS_BD = tuple(range(1, 9))
SEEDS_V200S1 = tuple(range(1, 13))

MEASURE_VERSION = 'spectral_v1_1_20260724'
F_RANGE = (0.27, 0.34)      # both cells (search realized 0.3038/0.3037;
                            # search-vs-instrument occupancy drift
                            # precedent ~1e-3 on this regime)
F_MATCH_MAX = 0.02          # matched-legibility gate between the cells
MIN_DIV_SEP = 3.0           # manipulation-validity gate (search 4.841)
# Review B1: diversity_pr's 524-dim space is 97.7% dyn/deter collector
# latents the fitted WM never models. The OBSERVABLE-support gate below
# requires the proprio-block (12-dim) PR to separate too — otherwise
# the manipulation is collector-latent-only and neither the fire nor
# the null licenses any "support breadth" statement about the WM's
# training distribution. Block PRs come from curate_frew block-pr
# (replay mode) run on the BUILT buffers.
MIN_PROPRIO_SEP = 0.5
BLOCK_FULL_TOL = 1e-6       # block json must be THIS buffer: its full
                            # PR equals the spectral json's diversity_pr
REF_F = 0.32317182817182816       # v200s1, frozen instrument value
REF_DIV = 9.230033291492585       # v200s1 diversity_pr, frozen
NLL_MEMBER_MAX = 1.5              # registered membership constants
NLL_NONMEMBER_MIN = 2.0           # (v200s1 anchor mean 0.827; see the
                                  # prereg's P-BD3 scope disclosure —
                                  # a gross-collapse screen, not a
                                  # discriminating instrument, on this
                                  # regime where reward == the regime
                                  # indicator)

# Constant self-consistency (review M1: swapped-constant mutants).
assert MODE_LO.endswith('s0') and MODE_HI.endswith('s1')
assert 'bdq1ds0_' in WM_LO and 'bdq1ds1_' in WM_HI
assert len(SEEDS_BD) == 8 and SEEDS_BD[0] == 1
assert len(SEEDS_V200S1) == 12 and SEEDS_V200S1[0] == 1


def load_auc(path, modes):
  """{mode: {seed: auc}} over qc-passing registered rows, plus
  {mode: [seeds]} of qc failures. EVERY consumed cohort in this wave is
  load-bearing, so the caller treats any qc failure as fatal."""
  cells = {m: {} for m in modes}
  qc_fail = {m: [] for m in modes}
  with open(path) as f:
    for r in csv.DictReader(f):
      if (r['mode'] not in cells or r['domain'] != 'finger'
          or int(r['milestone']) != 500000):
        continue
      if r['qc_pass'] in ('1', 'True', 'true'):
        cells[r['mode']][int(r['seed'])] = float(r['auc100k'])
      else:
        qc_fail[r['mode']].append(int(r['seed']))
  return cells, qc_fail


def load_spectral(path, expected_side):
  with open(path) as f:
    m = json.load(f)
  assert m.get('version') == MEASURE_VERSION, (
      f"{path}: version {m.get('version')!r} != {MEASURE_VERSION}")
  assert m.get('n_episodes') == 200, m.get('n_episodes')
  assert m.get('side') == expected_side and m.get('domain') == 'finger', (
      f"{path}: side/domain {m.get('side')!r}/{m.get('domain')!r} != "
      f"{expected_side!r}/'finger' — swapped or foreign instrument json")
  return dict(f=float(m['f_rewarded']), div=float(m['diversity_pr']),
              spectrum_pr=(float(m['spectrum_pr'])
                           if m.get('spectrum_pr') is not None else None),
              tag=m.get('tag'))


def check_ref(path):
  with open(path) as f:
    m = json.load(f)
  assert m.get('version') == MEASURE_VERSION
  assert abs(float(m['f_rewarded']) - REF_F) < 1e-9, m['f_rewarded']
  assert abs(float(m['diversity_pr']) - REF_DIV) < 1e-9, m['diversity_pr']


def load_block(path, expected_side):
  """Instrument-grade block decomposition (curate_frew block-pr,
  replay mode, run on the BUILT buffer)."""
  with open(path) as f:
    b = json.load(f)
  assert b.get('mode') == 'replay', (
      f"{path}: mode {b.get('mode')!r} != 'replay' — the read wants the "
      'built-buffer decomposition, not the pre-build moments one')
  assert b.get('side') == expected_side, (
      f"{path}: side {b.get('side')!r} != {expected_side!r} — swapped "
      'block jsons')
  assert b.get('n_episodes') == 200, b.get('n_episodes')
  for k in ('pr_full', 'pr_proprio', 'trace_share_latent'):
    assert np.isfinite(b.get(k, np.nan)), (path, k, b.get(k))
  return b


def load_e4(path, prefix, seeds):
  vals_in, vals_out = {}, {}
  with open(path) as f:
    for r in csv.DictReader(f):
      if not r['run_id'].startswith(prefix):
        continue
      suffix = r['run_id'][len(prefix):]
      if not suffix.isdigit():
        continue                  # e.g. smoke/suffixed run ids: skip,
      seed = int(suffix)          # never crash the read on them
      if int(r['horizon']) != 0:
        continue
      assert r['reward_aware'] in ('1', 'True', 'true'), (
          f"{r['run_id']}: reward_aware={r['reward_aware']}")
      assert seed not in vals_in, f'{prefix}{seed}: duplicate E4 h0 row'
      vals_in[seed] = float(r['rew_nll_in'])
      if r.get('rew_nll_out') not in (None, ''):
        vals_out[seed] = float(r['rew_nll_out'])
  missing = [s for s in seeds if s not in vals_in]
  assert not missing, f'{prefix}*: missing E4 h0 rows for seeds {missing}'
  return vals_in, vals_out


def _two_sample(a, b, rng):
  a, b = np.asarray(a, float), np.asarray(b, float)
  point = float(a.mean() - b.mean())
  n, m = len(a), len(b)
  stats = np.empty(B_BOOT)
  for i in range(B_BOOT):
    stats[i] = (a[rng.integers(0, n, n)].mean()
                - b[rng.integers(0, m, m)].mean())
  lo, hi = np.percentile(stats, [2.5, 97.5])
  return dict(point=point, ci=[float(lo), float(hi)], n=[n, m])


def _one_sample(a, rng):
  a = np.asarray(a, float)
  n = len(a)
  stats = np.empty(B_BOOT)
  for i in range(B_BOOT):
    stats[i] = a[rng.integers(0, n, n)].mean()
  lo, hi = np.percentile(stats, [2.5, 97.5])
  return dict(point=float(a.mean()), ci=[float(lo), float(hi)], n=n)


def _membership(nll):
  return ('member' if nll['ci'][1] <= NLL_MEMBER_MAX else
          'nonmember' if nll['ci'][0] >= NLL_NONMEMBER_MIN else
          'indeterminate')


def analyze(cells, spec_lo, spec_hi, blk_lo, blk_hi, nll_hi_vals,
            nll_lo_vals, nll_out_hi=None, nll_out_lo=None, qc_fail=None):
  qc_fail = qc_fail or {}
  # ---- validity gates (all fatal — both cells carry the primary) -----
  for name, spec in (('side0 (lo-div)', spec_lo), ('side1 (hi-div)',
                                                   spec_hi)):
    assert F_RANGE[0] <= spec['f'] <= F_RANGE[1], (
        f"{name} f_rewarded {spec['f']:.4f} outside f-band {F_RANGE} — "
        'invalid wave, the read refuses (dated amendment required)')
  assert abs(spec_hi['f'] - spec_lo['f']) <= F_MATCH_MAX, (
      f"matched-f gate: |{spec_hi['f']:.4f} - {spec_lo['f']:.4f}| > "
      f'{F_MATCH_MAX} — legibility not controlled, the read refuses')
  div_sep = spec_hi['div'] - spec_lo['div']
  assert div_sep >= MIN_DIV_SEP, (
      f'manipulation-validity gate: diversity_pr separation {div_sep:.3f} '
      f'< {MIN_DIV_SEP} — the mediator did not move, the read refuses '
      '(dated amendment required)')
  # block jsons must describe the SAME buffers the spectral jsons do
  for name, blk, spec in (('lo', blk_lo, spec_lo), ('hi', blk_hi,
                                                    spec_hi)):
    assert abs(blk['pr_full'] - spec['div']) <= BLOCK_FULL_TOL, (
        f'block json ({name}) pr_full {blk["pr_full"]:.6f} != spectral '
        f'diversity_pr {spec["div"]:.6f} — different buffer or stale '
        'decomposition, the read refuses')
  proprio_sep = blk_hi['pr_proprio'] - blk_lo['pr_proprio']
  assert proprio_sep >= MIN_PROPRIO_SEP, (
      f'observable-support gate: proprio-block PR separation '
      f'{proprio_sep:.3f} < {MIN_PROPRIO_SEP} — the manipulation is '
      'collector-latent-only (dyn/deter dominates the full-space index '
      'and the fitted WM never models it); no support-breadth statement '
      'about the training distribution is licensed, the read refuses '
      '(dated amendment required)')
  gates = dict(f_lo=spec_lo['f'], f_hi=spec_hi['f'],
               f_match=round(abs(spec_hi['f'] - spec_lo['f']), 6),
               div_lo=spec_lo['div'], div_hi=spec_hi['div'],
               div_separation=round(div_sep, 6),
               pr_proprio_lo=blk_lo['pr_proprio'],
               pr_proprio_hi=blk_hi['pr_proprio'],
               proprio_separation=round(proprio_sep, 6),
               f_range=list(F_RANGE), f_match_max=F_MATCH_MAX,
               min_div_sep=MIN_DIV_SEP,
               min_proprio_sep=MIN_PROPRIO_SEP)

  # ---- cohorts (all-or-nothing; fatal everywhere) --------------------
  for mode, seeds in ((MODE_LO, SEEDS_BD), (MODE_HI, SEEDS_BD),
                      (MODE_V200S1, SEEDS_V200S1)):
    assert not qc_fail.get(mode), (
        f'{mode}: qc-failed rows for seeds {sorted(qc_fail[mode])}')
    missing = [s for s in seeds if s not in cells[mode]]
    assert not missing, f'{mode}: missing seeds {missing}'

  # ---- registered statistics -----------------------------------------
  sel = lambda m, seeds: [cells[m][s] for s in seeds]
  primary = _two_sample(sel(MODE_HI, SEEDS_BD), sel(MODE_LO, SEEDS_BD),
                        np.random.default_rng(RNG_PRIMARY))
  fires = primary['ci'][0] > 0
  reversed_ = primary['ci'][1] < 0

  mean = lambda m, seeds: float(np.mean(sel(m, seeds)))
  lo_mean, hi_mean = mean(MODE_LO, SEEDS_BD), mean(MODE_HI, SEEDS_BD)
  v_mean = mean(MODE_V200S1, SEEDS_V200S1)
  ordering_holds = bool(lo_mean < v_mean <= hi_mean)
  hi_vs_v = _two_sample(sel(MODE_HI, SEEDS_BD),
                        sel(MODE_V200S1, SEEDS_V200S1),
                        np.random.default_rng(RNG_HI_V))
  lo_vs_v = _two_sample(sel(MODE_LO, SEEDS_BD),
                        sel(MODE_V200S1, SEEDS_V200S1),
                        np.random.default_rng(RNG_LO_V))

  nll_hi = _one_sample([nll_hi_vals[s] for s in SEEDS_BD],
                       np.random.default_rng(RNG_NLL_HI))
  nll_lo = _one_sample([nll_lo_vals[s] for s in SEEDS_BD],
                       np.random.default_rng(RNG_NLL_LO))
  mem_hi, mem_lo = _membership(nll_hi), _membership(nll_lo)
  both_member = bool(mem_hi == 'member' and mem_lo == 'member')
  inclusion_leak = bool(mem_lo == 'nonmember')
  hi_anomaly = bool(mem_hi == 'nonmember')

  verdict = []
  if fires:
    if both_member:
      verdict.append(
          'P-BD1 FIRES + P-BD3 both cells member: BREADTH-CAUSAL '
          'CONFIRMED — at matched f_R and matched volume, broader '
          'observable support transfers better with reward membership '
          'intact on both sides; the support effect is licensed (P-BD3 '
          'is a gross-collapse screen only — see the registration '
          'disclosure; mediation attribution rests on the matched-f '
          'design).')
    elif inclusion_leak:
      verdict.append(
          'P-BD1 FIRES but the lo-div cell leaves the membership band: '
          'INCLUSION-LEAK (registered alternative) — the manipulation '
          'leaked into legibility; the effect is reported under the '
          'inclusion reading and pure-breadth mediation is NOT '
          'licensed.')
    elif hi_anomaly:
      verdict.append(
          'P-BD1 FIRES but the HI-div cell leaves the membership band: '
          'HI-CELL LEGIBILITY ANOMALY (registered disclosure clause — '
          'an unpredicted direction; mediation is NOT licensed and the '
          'anomaly is reported for follow-up).')
    else:
      verdict.append(
          'P-BD1 FIRES; P-BD3 indeterminate '
          f'(hi={mem_hi}, lo={mem_lo}) — the breadth effect is real but '
          'mediation is unadjudicated.')
  elif reversed_:
    verdict.append(
        'P-BD1 REVERSED: the lo-div cell transfers better — the '
        'lambda-monotonicity form of the breadth claim is REFUTED '
        '(frozen consequence: revision required).')
  else:
    verdict.append(
        'P-BD1 INFORMATIVE NULL: no breadth effect detected at an '
        f'instrument-verified separation of {div_sep:.2f} PR units — '
        'the lambda-input reading of the volume anomaly loses its '
        'principal support (frozen consequence: the anomaly requires a '
        'non-breadth account).')
  verdict.append(
      f'P-BD2 ordering lo < v200s1 <= hi '
      f'{"HOLDS" if ordering_holds else "FAILS"} '
      f'({lo_mean:.1f} / {v_mean:.1f} / {hi_mean:.1f}; weak point form, '
      'non-confirmatory).')

  return dict(
      gates=gates,
      primary_hi_minus_lo=primary, fires=bool(fires),
      reversed=bool(reversed_),
      pbd2=dict(lo_mean=round(lo_mean, 4), v200s1_mean=round(v_mean, 4),
                hi_mean=round(hi_mean, 4), ordering_holds=ordering_holds,
                hi_vs_v200s1_descriptive=hi_vs_v,
                lo_vs_v200s1_descriptive=lo_vs_v),
      pbd3=dict(nll_hi=nll_hi, nll_lo=nll_lo,
                membership_hi=mem_hi, membership_lo=mem_lo,
                both_member=both_member, inclusion_leak=inclusion_leak,
                hi_anomaly=hi_anomaly,
                member_max=NLL_MEMBER_MAX,
                nonmember_min=NLL_NONMEMBER_MIN,
                nll_out_hi_mean=(round(float(np.mean(
                    [nll_out_hi[s] for s in SEEDS_BD])), 6)
                    if nll_out_hi and all(s in nll_out_hi
                                          for s in SEEDS_BD) else None),
                nll_out_lo_mean=(round(float(np.mean(
                    [nll_out_lo[s] for s in SEEDS_BD])), 6)
                    if nll_out_lo and all(s in nll_out_lo
                                          for s in SEEDS_BD) else None)),
      descriptives=dict(
          spectrum_pr_lo=spec_lo.get('spectrum_pr'),
          spectrum_pr_hi=spec_hi.get('spectrum_pr'),
          diversity_ref=REF_DIV,
          straddles_ref=bool(spec_lo['div'] < REF_DIV < spec_hi['div']),
          pr_latent_lo=blk_lo.get('pr_latent'),
          pr_latent_hi=blk_hi.get('pr_latent'),
          trace_share_latent_lo=blk_lo.get('trace_share_latent'),
          trace_share_latent_hi=blk_hi.get('trace_share_latent')),
      cell_means={m: {str(s): v for s, v in sorted(cells[m].items())}
                  for m in cells},
      verdict=' '.join(verdict))


def read(args):
  cells_new, qf_new = load_auc(args.auc, (MODE_LO, MODE_HI))
  cells_frozen, qf_frozen = load_auc(args.auc_frozen, (MODE_V200S1,))
  cells = {**cells_new, **cells_frozen}
  qc_fail = {**qf_new, **qf_frozen}
  check_ref(args.spectral_ref)
  spec_lo = load_spectral(args.spectral_lo, 'lo')
  spec_hi = load_spectral(args.spectral_hi, 'hi')
  blk_lo = load_block(args.block_lo, 'lo')
  blk_hi = load_block(args.block_hi, 'hi')
  nll_hi, nll_out_hi = load_e4(args.e4, WM_HI, SEEDS_BD)
  nll_lo, nll_out_lo = load_e4(args.e4, WM_LO, SEEDS_BD)
  res = analyze(cells, spec_lo, spec_hi, blk_lo, blk_hi, nll_hi, nll_lo,
                nll_out_hi=nll_out_hi, nll_out_lo=nll_out_lo,
                qc_fail=qc_fail)
  res['inputs'] = dict(auc=os.path.abspath(args.auc),
                       auc_frozen=os.path.abspath(args.auc_frozen),
                       spectral_lo=os.path.abspath(args.spectral_lo),
                       spectral_hi=os.path.abspath(args.spectral_hi),
                       spectral_ref=os.path.abspath(args.spectral_ref),
                       block_lo=os.path.abspath(args.block_lo),
                       block_hi=os.path.abspath(args.block_hi),
                       e4=os.path.abspath(args.e4))
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, 'breadth.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  p = res['primary_hi_minus_lo']
  print(f"PRIMARY hi-div - lo-div: {p['point']:+.1f} "
        f"CI=[{p['ci'][0]:+.1f},{p['ci'][1]:+.1f}] fires={res['fires']}")
  print(res['verdict'])
  print(f'-> {out}')


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _mk(hi=280.0, lo=120.0, v1=188.0, sd=25.0, f_hi=0.3037, f_lo=0.3038,
        div_hi=12.336, div_lo=7.495, ppr_hi=6.8, ppr_lo=3.2,
        nll_hi=1.2, nll_lo=1.2, nll_sd=0.05, seed=5):
  rng = np.random.default_rng(seed)
  cells = {MODE_LO: {}, MODE_HI: {}, MODE_V200S1: {}}
  for s in SEEDS_BD:
    cells[MODE_HI][s] = hi + sd * rng.standard_normal()
    cells[MODE_LO][s] = lo + sd * rng.standard_normal()
  for s in SEEDS_V200S1:
    cells[MODE_V200S1][s] = v1 + sd * rng.standard_normal()
  spec_lo = dict(f=f_lo, div=div_lo, tag='q1d_side0')
  spec_hi = dict(f=f_hi, div=div_hi, tag='q1d_side1')
  blk_lo = dict(mode='replay', side='lo', n_episodes=200,
                pr_full=div_lo, pr_proprio=ppr_lo, pr_latent=div_lo,
                trace_share_latent=0.95, n_rew=60000)
  blk_hi = dict(mode='replay', side='hi', n_episodes=200,
                pr_full=div_hi, pr_proprio=ppr_hi, pr_latent=div_hi,
                trace_share_latent=0.95, n_rew=60000)
  nh = {s: nll_hi + nll_sd * rng.standard_normal() for s in SEEDS_BD}
  nl = {s: nll_lo + nll_sd * rng.standard_normal() for s in SEEDS_BD}
  return cells, spec_lo, spec_hi, blk_lo, blk_hi, nh, nl


def _write_fixture(tmp, cells, spec_lo, spec_hi, blk_lo, blk_hi,
                   nll_hi, nll_lo):
  rows = ['run_id,mode,domain,seed,milestone,auc100k,qc_pass']
  frozen = ['run_id,mode,domain,seed,milestone,auc100k,qc_pass']
  for m, seedmap in cells.items():
    dst = frozen if m == MODE_V200S1 else rows
    for s, v in seedmap.items():
      dst.append(f'r,{m},finger,{s},500000,{v},1')
  paths = {}
  for name, content in (('auc.csv', '\n'.join(rows)),
                        ('auc_frozen.csv', '\n'.join(frozen))):
    paths[name] = os.path.join(tmp, name)
    with open(paths[name], 'w') as f:
      f.write(content)
  for name, side, spec in (('lo.json', 'lo', spec_lo),
                           ('hi.json', 'hi', spec_hi)):
    paths[name] = os.path.join(tmp, name)
    with open(paths[name], 'w') as f:
      json.dump(dict(version=MEASURE_VERSION, n_episodes=200,
                     side=side, domain='finger',
                     f_rewarded=spec['f'], diversity_pr=spec['div'],
                     spectrum_pr=35.0, tag=spec['tag']), f)
  for name, blk in (('blk_lo.json', blk_lo), ('blk_hi.json', blk_hi)):
    paths[name] = os.path.join(tmp, name)
    with open(paths[name], 'w') as f:
      json.dump(blk, f)
  paths['ref.json'] = os.path.join(tmp, 'ref.json')
  with open(paths['ref.json'], 'w') as f:
    json.dump(dict(version=MEASURE_VERSION, n_episodes=200,
                   f_rewarded=REF_F, diversity_pr=REF_DIV,
                   tag='volume_q1v200_s1'), f)
  e4 = ['run_id,horizon,rew_nll_in,rew_nll_out,reward_aware']
  for prefix, vals in ((WM_HI, nll_hi), (WM_LO, nll_lo)):
    for s, v in vals.items():
      for h in (0, 1, 5):
        vv = v if h == 0 else v + 1.0
        e4.append(f'{prefix}{s},{h},{vv},{vv + 2.0},1')
  # suffixed smoke id: load_e4 must SKIP it, never crash (review NIT)
  e4.append(f'{WM_HI}99smoke,0,9.9,9.9,1')
  paths['e4.csv'] = os.path.join(tmp, 'e4.csv')
  with open(paths['e4.csv'], 'w') as f:
    f.write('\n'.join(e4))
  return paths


def _trip(fn):
  """Returns the AssertionError message (truthy) if fn trips, else
  None — so callers can assert on WHICH gate fired (review MAJOR 4:
  vacuously-confounded gate fixtures)."""
  try:
    fn()
  except AssertionError as e:
    return str(e) or 'assertion'
  return None


def selfcheck(args):
  # Bootstrap machinery pin (review M1): golden values freeze B_BOOT,
  # the percentile pair, and the resampling scheme.
  g = _two_sample(list(range(8)), list(range(12)),
                  np.random.default_rng(0))
  assert g['point'] == -2.0 and g['n'] == [8, 12]
  assert round(g['ci'][0], 6) == -4.501042, g
  assert round(g['ci'][1], 6) == 0.5, g

  # Branch 1: fires + both member => BREADTH-CAUSAL CONFIRMED
  cells, sl, sh, bl, bh, nh, nl = _mk()
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert res['fires'] and not res['reversed']
  assert 'BREADTH-CAUSAL CONFIRMED' in res['verdict']
  assert res['pbd3']['both_member'] and not res['pbd3']['inclusion_leak']
  assert res['pbd2']['ordering_holds'] and 'HOLDS' in res['verdict']
  assert res['descriptives']['straddles_ref']
  assert res['primary_hi_minus_lo']['n'] == [8, 8]   # cohort cardinality
  assert analyze(cells, sl, sh, bl, bh, nh, nl) == res   # determinism

  # Branch 2: reversed (CI entirely negative)
  cells, sl, sh, bl, bh, nh, nl = _mk(hi=120.0, lo=280.0)
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert res['reversed'] and not res['fires']
  assert 'REVERSED' in res['verdict']
  assert not res['pbd2']['ordering_holds']

  # Branch 3: straddle => informative null
  cells, sl, sh, bl, bh, nh, nl = _mk(hi=205.0, lo=200.0, sd=40.0)
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert not res['fires'] and not res['reversed']
  assert 'INFORMATIVE NULL' in res['verdict']

  # Branch 3b (boundary): positive POINT with straddling CI must not
  # fire — the fire rule is the whole CI, not the point (mutant killer).
  cells, sl, sh, bl, bh, nh, nl = _mk()
  base = np.linspace(-1.5, 1.5, 8)
  cells[MODE_HI] = {s: 215.0 + 60 * b for s, b in zip(SEEDS_BD, base)}
  cells[MODE_LO] = {s: 200.0 + 5 * b for s, b in zip(SEEDS_BD, base)}
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  p = res['primary_hi_minus_lo']
  assert p['point'] > 0 and p['ci'][0] < 0 < p['ci'][1], p
  assert not res['fires'] and 'INFORMATIVE NULL' in res['verdict']

  # Branch 3c (review M1 mirror): negative POINT with straddling CI
  # must not read REVERSED — reversal needs the whole CI negative.
  cells, sl, sh, bl, bh, nh, nl = _mk()
  cells[MODE_HI] = {s: 195.0 + 60 * b for s, b in zip(SEEDS_BD, base)}
  cells[MODE_LO] = {s: 210.0 + 5 * b for s, b in zip(SEEDS_BD, base)}
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  p = res['primary_hi_minus_lo']
  assert p['point'] < 0 and p['ci'][0] < 0 < p['ci'][1], p
  assert not res['reversed'] and 'INFORMATIVE NULL' in res['verdict']

  # Branch 4: fires + lo-div nonmember => INCLUSION-LEAK
  cells, sl, sh, bl, bh, nh, nl = _mk(nll_lo=2.4)
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert res['fires'] and res['pbd3']['inclusion_leak']
  assert 'INCLUSION-LEAK' in res['verdict']
  assert 'NOT licensed' in res['verdict']

  # Branch 4b (review M2): HI-cell nonmember is its own labeled
  # disclosure clause, not "indeterminate".
  cells, sl, sh, bl, bh, nh, nl = _mk(nll_hi=2.4)
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert res['fires'] and res['pbd3']['hi_anomaly']
  assert 'HI-CELL LEGIBILITY ANOMALY' in res['verdict']
  assert 'BREADTH-CAUSAL CONFIRMED' not in res['verdict']

  # Branch 5: fires + indeterminate lo => mediation unadjudicated
  cells, sl, sh, bl, bh, nh, nl = _mk(nll_lo=1.7, nll_sd=0.01)
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert res['fires'] and not res['pbd3']['both_member']
  assert not res['pbd3']['inclusion_leak']
  assert 'unadjudicated' in res['verdict']

  # Branch 5b: membership is a CI rule at BOTH thresholds — point on the
  # member side of 1.5 with CI crossing it, and point past 2.0 with CI
  # crossing back, must both read indeterminate (mutant killers).
  cells, sl, sh, bl, bh, nh, nl = _mk()
  spread = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8]      # mean 1.4
  nl = {s: v for s, v in zip(SEEDS_BD, spread)}
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert res['pbd3']['nll_lo']['point'] < NLL_MEMBER_MAX
  assert res['pbd3']['nll_lo']['ci'][1] > NLL_MEMBER_MAX
  assert res['pbd3']['membership_lo'] == 'indeterminate'
  nl = {s: v + 0.7 for s, v in zip(SEEDS_BD, spread)}     # mean 2.1
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert res['pbd3']['nll_lo']['point'] >= NLL_NONMEMBER_MIN
  assert res['pbd3']['nll_lo']['ci'][0] < NLL_NONMEMBER_MIN
  assert res['pbd3']['membership_lo'] == 'indeterminate'

  # Branch 6: P-BD2 boundary — equality on the right leg HOLDS ('<='),
  # equality on the left leg FAILS ('<'). Dyadic values, float-exact.
  cells, sl, sh, bl, bh, nh, nl = _mk(sd=0.0)
  cells[MODE_LO] = {s: 100.0 for s in SEEDS_BD}
  cells[MODE_HI] = {s: 188.0 for s in SEEDS_BD}
  cells[MODE_V200S1] = {s: 188.0 for s in SEEDS_V200S1}
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert res['pbd2']['ordering_holds']                # v == hi passes <=
  cells[MODE_LO] = {s: 188.0 for s in SEEDS_BD}
  cells[MODE_HI] = {s: 280.0 for s in SEEDS_BD}
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert not res['pbd2']['ordering_holds']            # lo == v fails <

  # Gate legs (review MAJOR 4: each gate must trip ON ITS OWN MESSAGE,
  # and the f-band needs both-sides-moved fixtures the match gate
  # cannot absorb).
  # f-band, single side out (the other in range; |df| also > 0.02 but
  # the band assert fires first and the message proves it):
  for kw in (dict(f_lo=0.26), dict(f_lo=0.35), dict(f_hi=0.26),
             dict(f_hi=0.35)):
    cells, sl, sh, bl, bh, nh, nl = _mk(**kw)
    msg = _trip(lambda: analyze(cells, sl, sh, bl, bh, nh, nl))
    assert msg and 'f-band' in msg, (kw, msg)
  # f-band, BOTH sides out with |df| <= 0.02 — kills band-bound mutants
  # that the match gate would otherwise mask:
  for kw in (dict(f_lo=0.26, f_hi=0.265), dict(f_lo=0.355, f_hi=0.35)):
    cells, sl, sh, bl, bh, nh, nl = _mk(**kw)
    msg = _trip(lambda: analyze(cells, sl, sh, bl, bh, nh, nl))
    assert msg and 'f-band' in msg, (kw, msg)
  # matched-f gate, both in band, both directions (kills abs-drop):
  for kw in (dict(f_lo=0.275, f_hi=0.30), dict(f_lo=0.30, f_hi=0.275)):
    cells, sl, sh, bl, bh, nh, nl = _mk(**kw)
    msg = _trip(lambda: analyze(cells, sl, sh, bl, bh, nh, nl))
    assert msg and 'matched-f' in msg, (kw, msg)
  # separation boundary: 2.99 refuses, 3.01 passes (straddle the gate);
  # swapped sides (negative separation) refuses — kills abs() mutants:
  cells, sl, sh, bl, bh, nh, nl = _mk(div_lo=9.0, div_hi=11.99,
                                      ppr_lo=3.0, ppr_hi=6.0)
  msg = _trip(lambda: analyze(cells, sl, sh, bl, bh, nh, nl))
  assert msg and 'manipulation-validity' in msg, msg
  cells, sl, sh, bl, bh, nh, nl = _mk(div_lo=9.0, div_hi=12.01,
                                      ppr_lo=3.0, ppr_hi=6.0)
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert res['gates']['div_separation'] >= 3.0
  assert res['descriptives']['straddles_ref']         # 9.0 < 9.23 < 12.01
  cells, sl, sh, bl, bh, nh, nl = _mk(div_lo=12.336, div_hi=7.495,
                                      ppr_lo=3.2, ppr_hi=6.8)
  msg = _trip(lambda: analyze(cells, sl, sh, bl, bh, nh, nl))
  assert msg and 'manipulation-validity' in msg, msg
  cells, sl, sh, bl, bh, nh, nl = _mk(div_lo=9.5, div_hi=12.51,
                                      ppr_lo=3.0, ppr_hi=6.0)
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert not res['descriptives']['straddles_ref']     # both above the ref
  # observable-support gate (review B1): 0.49 refuses, 0.51 passes
  # (boundary straddled); reversed proprio separation refuses:
  cells, sl, sh, bl, bh, nh, nl = _mk(ppr_lo=3.2, ppr_hi=3.69)
  msg = _trip(lambda: analyze(cells, sl, sh, bl, bh, nh, nl))
  assert msg and 'observable-support' in msg, msg
  cells, sl, sh, bl, bh, nh, nl = _mk(ppr_lo=3.2, ppr_hi=3.71)
  res = analyze(cells, sl, sh, bl, bh, nh, nl)
  assert res['gates']['proprio_separation'] >= MIN_PROPRIO_SEP
  cells, sl, sh, bl, bh, nh, nl = _mk(ppr_lo=6.8, ppr_hi=3.2)
  msg = _trip(lambda: analyze(cells, sl, sh, bl, bh, nh, nl))
  assert msg and 'observable-support' in msg, msg
  # block-json/spectral-json binding: pr_full must equal diversity_pr
  cells, sl, sh, bl, bh, nh, nl = _mk()
  bl2 = dict(bl, pr_full=bl['pr_full'] + 0.01)
  msg = _trip(lambda: analyze(cells, sl, sh, bl2, bh, nh, nl))
  assert msg and 'different buffer' in msg, msg

  # Cohorts: any missing seed or qc failure anywhere is fatal
  for mode in (MODE_LO, MODE_HI):
    cells, sl, sh, bl, bh, nh, nl = _mk()
    del cells[mode][3]
    assert _trip(lambda: analyze(cells, sl, sh, bl, bh, nh, nl)), mode
    cells, sl, sh, bl, bh, nh, nl = _mk()
    assert _trip(lambda: analyze(cells, sl, sh, bl, bh, nh, nl,
                                 qc_fail={mode: [2]})), mode
  cells, sl, sh, bl, bh, nh, nl = _mk()
  del cells[MODE_V200S1][12]
  assert _trip(lambda: analyze(cells, sl, sh, bl, bh, nh, nl))

  # Loader chain through files
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    cells, sl, sh, bl, bh, nh, nl = _mk()
    paths = _write_fixture(tmp, cells, sl, sh, bl, bh, nh, nl)
    new_cells, new_qf = load_auc(paths['auc.csv'], (MODE_LO, MODE_HI))
    frz_cells, frz_qf = load_auc(paths['auc_frozen.csv'], (MODE_V200S1,))
    loaded = {**new_cells, **frz_cells}
    qf = {**new_qf, **frz_qf}
    assert loaded[MODE_HI] == {s: cells[MODE_HI][s] for s in SEEDS_BD}
    assert not any(qf.values())
    check_ref(paths['ref.json'])
    e4h, e4h_out = load_e4(paths['e4.csv'], WM_HI, SEEDS_BD)
    e4l, e4l_out = load_e4(paths['e4.csv'], WM_LO, SEEDS_BD)
    assert all(abs(e4h[s] - nh[s]) < 1e-9 for s in SEEDS_BD)   # h0 only
    assert all(abs(e4h_out[s] - nh[s] - 2.0) < 1e-9 for s in SEEDS_BD)
    assert 99 not in e4h                     # smoke suffix row skipped
    res_files = analyze(loaded, load_spectral(paths['lo.json'], 'lo'),
                        load_spectral(paths['hi.json'], 'hi'),
                        load_block(paths['blk_lo.json'], 'lo'),
                        load_block(paths['blk_hi.json'], 'hi'),
                        e4h, e4l, nll_out_hi=e4h_out, nll_out_lo=e4l_out,
                        qc_fail=qf)
    assert res_files['fires']
    assert res_files['descriptives']['spectrum_pr_hi'] == 35.0
    assert abs(res_files['pbd3']['nll_out_hi_mean']
               - (np.mean([nh[s] for s in SEEDS_BD]) + 2.0)) < 1e-6
    # swapped --spectral_lo/--spectral_hi must trip; same for blocks
    assert _trip(lambda: load_spectral(paths['hi.json'], 'lo'))
    assert _trip(lambda: load_spectral(paths['lo.json'], 'hi'))
    assert _trip(lambda: load_block(paths['blk_hi.json'], 'lo'))
    assert _trip(lambda: load_block(paths['blk_lo.json'], 'hi'))
    # a moments-mode block json is refused (the read wants replay mode)
    with open(paths['blk_lo.json']) as f:
      mblk = json.load(f)
    mblk['mode'] = 'moments'
    with open(os.path.join(tmp, 'mblk.json'), 'w') as f:
      json.dump(mblk, f)
    assert _trip(lambda: load_block(os.path.join(tmp, 'mblk.json'),
                                    'lo'))
    # a non-finger row with an absurd value must be ignored (domain
    # filter mutant killer)
    with open(paths['auc.csv']) as f:
      plus = f.read() + f'\nr,{MODE_HI},cup,1,500000,99999.0,1'
    with open(os.path.join(tmp, 'plus.csv'), 'w') as f:
      f.write(plus)
    plus_cells, _ = load_auc(os.path.join(tmp, 'plus.csv'),
                             (MODE_LO, MODE_HI))
    assert plus_cells[MODE_HI] == loaded[MODE_HI]
    # qc_pass=0 on ANY new-cell row -> loader reports, analyze aborts
    for mode in (MODE_LO, MODE_HI):
      with open(paths['auc.csv']) as f:
        bad = '\n'.join(l[:-1] + '0' if f',{mode},finger,4,' in l else l
                        for l in f.read().splitlines())
      badp = os.path.join(tmp, f'bad_{mode}.csv')
      with open(badp, 'w') as f:
        f.write(bad)
      bad_cells, bad_qf = load_auc(badp, (MODE_LO, MODE_HI))
      assert bad_qf[mode] == [4] and 4 not in bad_cells[mode]
      assert _trip(lambda: analyze(
          {**bad_cells, **frz_cells}, sl, sh, bl, bh, nh, nl,
          qc_fail={**bad_qf, **frz_qf}))
    # tampered ref json must trip — on EITHER pinned value (review M1:
    # the f_rewarded pin was untested)
    for field, val in (('diversity_pr', 9.0), ('f_rewarded', 0.32)):
      with open(paths['ref.json']) as f:
        ref = json.load(f)
      ref[field] = val
      badref = os.path.join(tmp, f'badref_{field}.json')
      with open(badref, 'w') as f:
        json.dump(ref, f)
      assert _trip(lambda: check_ref(badref)), field
    # missing e4 seed trips; duplicate row trips; reward_aware=0 trips
    e4rows = open(paths['e4.csv']).read().splitlines()
    with open(os.path.join(tmp, 'bade4.csv'), 'w') as f:
      f.write('\n'.join(r for r in e4rows
                        if not r.startswith(f'{WM_HI}7,0,')))
    assert _trip(lambda: load_e4(os.path.join(tmp, 'bade4.csv'), WM_HI,
                                 SEEDS_BD))
    with open(os.path.join(tmp, 'bade4b.csv'), 'w') as f:
      f.write('\n'.join(e4rows
                        + [r for r in e4rows
                           if r.startswith(f'{WM_HI}2,0,')]))
    assert _trip(lambda: load_e4(os.path.join(tmp, 'bade4b.csv'), WM_HI,
                                 SEEDS_BD))
    e4rows2 = [r.rsplit(',', 1)[0] + ',0'
               if r.startswith(f'{WM_HI}2,0,') else r for r in e4rows]
    with open(os.path.join(tmp, 'bade4c.csv'), 'w') as f:
      f.write('\n'.join(e4rows2))
    assert _trip(lambda: load_e4(os.path.join(tmp, 'bade4c.csv'), WM_HI,
                                 SEEDS_BD))
    # wrong spectral version trips
    with open(paths['hi.json']) as f:
      spec = json.load(f)
    spec['version'] = 'spectral_v1_20260724'
    with open(os.path.join(tmp, 'badspec.json'), 'w') as f:
      json.dump(spec, f)
    assert _trip(lambda: load_spectral(os.path.join(tmp, 'badspec.json'),
                                       'hi'))
    # wrong episode count trips
    with open(paths['hi.json']) as f:
      spec = json.load(f)
    spec['n_episodes'] = 199
    with open(os.path.join(tmp, 'badspec2.json'), 'w') as f:
      json.dump(spec, f)
    assert _trip(lambda: load_spectral(os.path.join(tmp, 'badspec2.json'),
                                       'hi'))
  print('selfcheck PASS: bootstrap goldens pinned (B=10K, 2.5/97.5), '
        'confirmed/reversed/null branches, point-vs-CI boundaries BOTH '
        'directions, inclusion-leak + hi-cell anomaly + indeterminate '
        '(CI rule at both thresholds), P-BD2 <=-right/<-left equality '
        'edges, gate trips message-verified (f-band single- AND '
        'both-sides-moved, matched-f both directions, div separation '
        'straddled at 3.0 + swapped-sides, observable-support straddled '
        'at 0.5 + reversed, block/spectral binding), all-cohorts-fatal, '
        'loader chain (domain filter mutant killer, qc split, ref pin '
        'both fields, e4 h0 + reward_aware + missing + duplicate + '
        'smoke-suffix skip + nll_out, spectral + block side pins, '
        'moments-mode refusal), determinism')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--auc')
  ap.add_argument('--auc_frozen')
  ap.add_argument('--spectral_lo')
  ap.add_argument('--spectral_hi')
  ap.add_argument('--spectral_ref',
                  default='artifacts/volume_repl_20260729/'
                          'diversity_q1v200_s1.json')
  ap.add_argument('--block_lo')
  ap.add_argument('--block_hi')
  ap.add_argument('--e4')
  ap.add_argument('--output', default='analysis_out/breadth')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
  else:
    read(args)


if __name__ == '__main__':
  main()
