"""Frozen read for the Stage-0 second-k density-proxy robustness test (Paper 2).

Registered in prereg/PREREG_stage0_secondk_20260730.md and committed BEFORE
any k != 10 dump exists anywhere. Executes the Stage-0 memos' own registered
caveat ("kNN density in high-dim latent spaces is itself a proxy ...; a
TRACKS_DENSITY verdict should be sanity-checked against a second k before
externalizing") on the committed 10/10 TRACKS_DENSITY claim
(artifacts/latent_uq_stage0_20260710_221232/DECISION.md).

Substrate fact frozen here: a second k CANNOT be computed from the committed
dumps -- knn_distance keeps only the mean of the k smallest distances, and
neither anchors nor reference encodings are saved (probing/latent_uq.py).
A second-k pass is a re-run of the dumper per run with --knn K2 and EVERY
other dial pinned to the committed run's values; the ONLY changed variable
is k. This reader enforces that single-variable property machine-checkably:

  INTEGRITY GUARD  every k-INdependent array (disag_steps, err_*, pred_*,
                   true_*, source_label, stream_start) in each re-dump must
                   match the committed k=10 dump for the same run and
                   checkpoint. Two registered modes (the stack computes in
                   bfloat16 and the RSSM SAMPLES the categorical stoch at
                   every observe/imagine step, so cross-backend drift can
                   flip discrete draws and diverge whole windows O(1); the
                   re-dumps therefore pin --platform cuda like the committed
                   dumps, and the guard mode is selected by an explicit
                   calibration, never assumed):
                     tolerance -- exact for non-float arrays, floats within
                       rtol 1e-4 / atol 1e-6;
                     fallback  -- exact non-floats + identical key sets and
                       shapes + true_* within the numeric tolerance
                       (numpy-computed from the sha-pinned probeset,
                       backend-independent) + per-array Spearman rank
                       correlation >= FALLBACK_RANK_FLOOR for remaining
                       floats (sample-flip drift re-draws the same windows
                       and preserves cross-window order in aggregate; a
                       changed anchor/probe-window/reference set
                       decorrelates ranks toward 0).
                   ANY violation under the selected mode => QUARANTINE: no
                   verdict is computed.

  GUARD CALIBRATION (check_smoke; also standalone as --smoke_check, run
  between the smoke dumps and the full passes; value-blind -- only
  k-independent arrays and the k=10 calibration are touched, no k2
  estimand): two single-checkpoint dumps on the earliest committed
  cup-seed1 snapshot must precede the full passes:
    cup_v1_k10cal_smoke (--knn 10): compared to the committed dump with
      density_knn INCLUDED (same k -- this also exercises the density
      REFERENCE side, replay chunks -> ref windows, end to end).
      Within tolerance => guard mode 'tolerance'; fails tolerance but
      passes the fallback guard => guard mode 'fallback' (recorded);
      fails BOTH => 'invalid' => registered QUARANTINE (the backend
      cannot reproduce the committed substrate at the SAME k).
    cup_v1_k5_smoke (--knn 5): checked under the selected mode
      (density_knn exempt -- k differs). Any failure => QUARANTINE.

Cohort: the 10 held-out p2e cells (pretrain_p2e_{cup,finger}_seed{1..5}),
ALL-OR-NOTHING -- the committed claim is 10/10 and a partial cohort cannot
adjudicate its robustness. Both flanks are registered: k2 in {5, 20}.

Registered rule (verdicts computed by the SAME frozen machinery imported
from probing.latent_uq_analysis -- within_signals/row_stats/analyze_cell,
anchor row at horizon 5 of the FINAL checkpoint, stream-clustered bootstrap
n_boot=300, VERDICT_SIGMA=2.0):

  P-K1  SECOND-K ROBUST iff ALL 10 cells' final-checkpoint anchor-h5
        verdicts are TRACKS_DENSITY at BOTH k=5 and k=20 (20/20).
        Any other verdict (TRACKS_ERROR or AMBIGUOUS) anywhere =>
        K-SENSITIVE, flipped cells enumerated; the externalization
        qualifier becomes mandatory.

The committed k=10 side is NEVER recomputed here beyond the integrity
comparison; the reader asserts at read time that the committed artifact
(artifacts/latent_uq_stage0_20260710_221232/{cup,finger}_p2e/analysis.json)
still says TRACKS_DENSITY for all 10 pinned cells. The registered existence
gate's JSON record is consumed via --gate_json (must say SUBSTRATE OK; the
SUBSTRATE-GONE branch runs no passes and has no read). Secondary
(descriptive, never decisional): per-checkpoint verdict trajectories at
each k2 (the committed phase-change shape) and the h=1 addendum read.

Usage:
  python -m analysis.stage0_secondk_read --smoke_check \
      --dumps_root <dir> --committed_root <dir>
  python -m analysis.stage0_secondk_read --dumps_root <dir> \
      --committed_root <dir> --gate_json <secondk_substrate_gate.json> \
      --output <dir>
  python -m analysis.stage0_secondk_read --selfcheck
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from probing.latent_uq_analysis import (
    VERDICT_SIGMA, analyze_cell, clusters_of, load_dump, row_stats,
    verdict_of, within_signals)

# Import pin: the verdict machinery this read reuses must still be the
# committed one (a silent change would re-adjudicate the baseline).
assert VERDICT_SIGMA == 2.0, VERDICT_SIGMA

K_COMMITTED = 10
K2S = (5, 20)
DOMS = ('cup', 'finger')
SEEDS = (1, 2, 3, 4, 5)
CELLS = tuple((dom, seed) for dom in DOMS for seed in SEEDS)  # 10 cells
EXPECTED_COMMITTED_VERDICT = 'TRACKS_DENSITY'
HORIZONS = (1, 5)          # the committed analysis grid
PRIMARY_HORIZON = 5        # committed headline row: anchor, h=5, final ckpt
N_BOOT = 300               # committed n_boot
BOOT_SEED = 0              # committed bootstrap seed base (seed + 13*i + h)
DUMP_SEED = 0              # committed dumper --seed (meta.json 'seed')
N_CKPTS = 5                # committed retained-snapshot series length
BURN_IN = 32
EVAL_STEPS = 8
REF_WINDOWS = 2048
EP_BATCH = 64              # rng-mark batching dial; recorded by new dumps
PROBESET_ID = dict(cup='ball_in_cup_v1_9327e960', finger='finger_v1_7a32ff08')
PROBESET_SHA = dict(
    cup='9327e96044a6df2659a58e07833dc4f793a2a22bf263bd794a31b022200cb632',
    finger='7a32ff087dd4d8ee639bc1badc955a99039fe438d4c1f8c040eab77ded25663c')
FLOAT_RTOL = 1e-4
FLOAT_ATOL = 1e-6
FALLBACK_RANK_FLOOR = 0.9  # registered Spearman floor for the fallback mode
K_DEPENDENT = ('density_knn',)  # the ONLY array k may change
ARTIFACT_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'artifacts', 'latent_uq_stage0_20260710_221232')
SMOKE_K5_REL = os.path.join('pretrain_p2e_cup_seed1', 'latent_uq',
                            'cup_v1_k5_smoke')
SMOKE_K10CAL_REL = os.path.join('pretrain_p2e_cup_seed1', 'latent_uq',
                                'cup_v1_k10cal_smoke')


def cell_label(dom, seed):
  return f'p2e_{dom}_seed{seed}'


def run_name(dom, seed):
  return f'pretrain_p2e_{dom}_seed{seed}'


def _git_commit():
  try:
    import subprocess
    out = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, text=True, timeout=10)
    return out.stdout.strip() or None
  except Exception:
    return None


def _rank(x):
  flat = np.asarray(x, np.float64).ravel()
  order = np.argsort(flat, kind='mergesort')
  ranks = np.empty(flat.size, np.float64)
  ranks[order] = np.arange(flat.size)
  return ranks


def rank_corr(a, b):
  """Spearman rank correlation over flattened arrays (stable-sort ranks)."""
  ra, rb = _rank(a), _rank(b)
  ra -= ra.mean()
  rb -= rb.mean()
  na = float(np.sqrt((ra ** 2).sum()))
  nb = float(np.sqrt((rb ** 2).sum()))
  if na == 0.0 or nb == 0.0:  # degenerate (size<=1)
    return 1.0 if np.array_equal(_rank(a), _rank(b)) else 0.0
  return float((ra * rb).sum() / (na * nb))


def check_artifact(artifact_root):
  """The committed baseline must still read 10/10 TRACKS_DENSITY."""
  for dom in DOMS:
    path = os.path.join(artifact_root, f'{dom}_p2e', 'analysis.json')
    assert os.path.exists(path), f'committed artifact missing: {path}'
    with open(path) as f:
      r = json.load(f)
    assert r['primary_horizon'] == PRIMARY_HORIZON, (
        f'{dom}_p2e: committed primary_horizon {r["primary_horizon"]} != '
        f'{PRIMARY_HORIZON}')
    assert r['n_boot'] == N_BOOT, (
        f'{dom}_p2e: committed n_boot {r["n_boot"]} != {N_BOOT}')
    assert r['verdict_sigma'] == VERDICT_SIGMA, (
        f'{dom}_p2e: committed verdict_sigma {r["verdict_sigma"]} != '
        f'{VERDICT_SIGMA}')
    assert not r['cross'] and r['held_out'], (
        f'{dom}_p2e: committed cell battery is not the within/held-out one')
    labels = sorted(c['label'] for c in r['cells'])
    expect = sorted(cell_label(dom, s) for s in SEEDS)
    assert labels == expect, (
        f'{dom}_p2e: committed cells {labels} != pinned {expect}')
    for c in r['cells']:
      assert c['final_verdict'] == EXPECTED_COMMITTED_VERDICT, (
          f'committed baseline changed: {c["label"]} final_verdict is '
          f'{c["final_verdict"]}, not {EXPECTED_COMMITTED_VERDICT} -- this '
          f'read adjudicates robustness OF the committed claim and cannot '
          f'run against a different baseline')


def check_meta(dump, expected_knn, tag, new_dump=False):
  """Per-checkpoint dial identity from the dumper's own meta.json.

  new_dump=True for every re-dump this registration produces: those must
  carry the ep_batch key (value-relevant — batching sets the rng marks;
  the dumper records it as of the same freeze). The committed k=10 dumps
  predate the key and are exempt from its presence, never its value.
  """
  for entry in dump['checkpoints']:
    mpath = os.path.join(dump['dir'], entry['name'], 'meta.json')
    assert os.path.exists(mpath), f'{tag}/{entry["name"]}: meta.json missing'
    with open(mpath) as f:
      md = json.load(f)
    assert md.get('knn') == expected_knn, (
        f'{tag}/{entry["name"]}: meta knn={md.get("knn")} != {expected_knn}')
    assert md.get('seed') == DUMP_SEED, (
        f'{tag}/{entry["name"]}: dumper seed={md.get("seed")} != '
        f'registered {DUMP_SEED}')
    ep = md.get('ep_batch', None)
    assert ep is None or ep == EP_BATCH, (
        f'{tag}/{entry["name"]}: ep_batch={ep} != registered {EP_BATCH}')
    if new_dump:
      assert ep == EP_BATCH, (
          f'{tag}/{entry["name"]}: re-dump meta lacks ep_batch — dumper '
          'predates the registered patch; re-dump with the frozen code')


def check_index(dump, expected_knn, dom, run, committed_names, tag):
  """Registered dial pins on a dump's index.json."""
  idx = dump['index']
  assert idx['knn'] == expected_knn, (
      f'{tag}: index knn={idx["knn"]} != expected {expected_knn}')
  assert idx['probeset_id'] == PROBESET_ID[dom], (
      f'{tag}: probeset_id {idx["probeset_id"]} != pinned '
      f'{PROBESET_ID[dom]}')
  assert idx['probeset_sha256'] == PROBESET_SHA[dom], (
      f'{tag}: probeset sha {idx["probeset_sha256"]} != pinned sha')
  assert idx['burn_in'] == BURN_IN and idx['eval_steps'] == EVAL_STEPS, (
      f'{tag}: burn_in/eval_steps {idx["burn_in"]}/{idx["eval_steps"]} != '
      f'pinned {BURN_IN}/{EVAL_STEPS}')
  assert idx['ref_windows'] == REF_WINDOWS, (
      f'{tag}: ref_windows {idx["ref_windows"]} != pinned {REF_WINDOWS} '
      f'(density reference shrank or grew -- not a single-variable change)')
  ref = str(idx.get('ref_replay', '')).rstrip('/')
  assert ref.endswith(f'/{run}/replay'), (
      f'{tag}: ref_replay {idx.get("ref_replay")!r} is not this run\'s own '
      f'replay (.../{run}/replay, as in all 10 committed indices) -- a '
      f'mis-dialed density reference changes ONLY density_knn and would '
      f'evade the k-independent integrity guard')
  assert idx['dec_symlog'] and idx['has_disag'], (
      f'{tag}: dec_symlog/has_disag flags differ from the committed cells')
  assert idx.get('held_out', False), (
      f'{tag}: dump is not held-out; the registered cells are held-out')
  names = [c['name'] for c in dump['checkpoints']]
  assert len(names) == N_CKPTS, (
      f'{tag}: {len(names)} checkpoints != registered {N_CKPTS}')
  if committed_names is not None:
    assert names == committed_names, (
        f'{tag}: checkpoint names {names} != committed {committed_names}')


def integrity_failures(label, k, new, committed, mode, exempt=K_DEPENDENT):
  """Registered integrity guard: k-independent arrays must match.

  mode 'tolerance': float arrays within FLOAT_RTOL/FLOAT_ATOL.
  mode 'fallback' (guard calibration failed the tolerance but passed the
  rank guard): true_* still within the numeric tolerance (numpy-computed
  from the sha-pinned probeset, backend-independent); other float arrays
  must have Spearman rank correlation >= FALLBACK_RANK_FLOOR.
  Non-float arrays, key sets, and shapes are exact in both modes.
  """
  assert mode in ('tolerance', 'fallback'), mode
  fails = []
  for nc, cc in zip(new['checkpoints'], committed['checkpoints']):
    nkeys, ckeys = set(nc['npz']), set(cc['npz'])
    if nkeys != ckeys:
      fails.append(dict(cell=label, k=k, ckpt=nc['name'], key='<key set>',
                        detail=f'differing keys: {sorted(nkeys ^ ckeys)}'))
      continue
    for key in sorted(ckeys):
      if key in exempt:
        continue
      a, b = nc['npz'][key], cc['npz'][key]
      if a.shape != b.shape:
        fails.append(dict(cell=label, k=k, ckpt=nc['name'], key=key,
                          detail=f'shape {a.shape} != {b.shape}'))
      elif a.dtype.kind == 'f':
        if mode == 'tolerance' or key.startswith('true_'):
          if not np.allclose(a, b, rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
            fails.append(dict(
                cell=label, k=k, ckpt=nc['name'], key=key,
                detail='beyond tolerance',
                max_abs=float(np.max(np.abs(
                    np.asarray(a, np.float64) - np.asarray(b, np.float64))))))
        else:
          rc = rank_corr(a, b)
          if rc < FALLBACK_RANK_FLOOR:
            fails.append(dict(
                cell=label, k=k, ckpt=nc['name'], key=key,
                detail=f'rank correlation {rc:.4f} below registered '
                       f'fallback floor {FALLBACK_RANK_FLOOR}'))
      elif not np.array_equal(a, b):
        fails.append(dict(cell=label, k=k, ckpt=nc['name'], key=key,
                          detail='non-float arrays differ'))
  return fails


def _load_smoke(dumps_root, rel, expected_knn, committed_first, tag):
  path = os.path.join(dumps_root, rel)
  assert os.path.exists(os.path.join(path, 'index.json')), (
      f'registered second-k smoke gate not satisfied: {path}/index.json '
      f'missing ({tag}: one single-checkpoint dump on the earliest '
      f'committed cup-seed1 snapshot must precede the full passes)')
  dump = load_dump(path)
  idx = dump['index']
  assert idx.get('knn') == expected_knn, (
      f'smoke gate ({tag}): knn={idx.get("knn")} != {expected_knn}')
  # Full dial pins on the smoke dumps too: a mis-dialed calibration
  # (wrong seed/ref_replay/burn_in) could otherwise select guard mode
  # 'fallback' on a false premise and weaken the guard for all 20
  # passes. Same asserts as the full passes, minus the 5-ckpt count.
  assert idx.get('probeset_id') == PROBESET_ID['cup'], (
      f'smoke gate ({tag}): probeset_id {idx.get("probeset_id")} != '
      f'pinned {PROBESET_ID["cup"]}')
  assert idx.get('probeset_sha256') == PROBESET_SHA['cup'], (
      f'smoke gate ({tag}): probeset sha != pinned cup sha')
  assert (idx.get('burn_in') == BURN_IN
          and idx.get('eval_steps') == EVAL_STEPS), (
      f'smoke gate ({tag}): burn_in/eval_steps '
      f'{idx.get("burn_in")}/{idx.get("eval_steps")} != pinned '
      f'{BURN_IN}/{EVAL_STEPS}')
  assert idx.get('ref_windows') == REF_WINDOWS, (
      f'smoke gate ({tag}): ref_windows {idx.get("ref_windows")} != '
      f'pinned {REF_WINDOWS}')
  ref = str(idx.get('ref_replay', '')).rstrip('/')
  assert ref.endswith(f'/{run_name("cup", 1)}/replay'), (
      f'smoke gate ({tag}): ref_replay {idx.get("ref_replay")!r} is not '
      f'cup seed1\'s own replay')
  names = [c['name'] for c in dump['checkpoints']]
  assert names == [committed_first], (
      f'smoke gate ({tag}): checkpoints {names} != earliest committed '
      f'[{committed_first!r}]')
  check_meta(dump, expected_knn, f'smoke:{tag}', new_dump=True)
  return dump


def check_smoke(dumps_root, committed_root):
  """Registered smoke + guard-calibration gate. Returns (mode, record).

  Value-blind: only k-independent arrays and the k=10 calibration are
  touched; no k2 estimand. Registered ladder (mirrored in the prereg):
    k10cal within tolerance                  -> mode 'tolerance'
    k10cal fails tolerance, passes fallback  -> mode 'fallback' (recorded)
    k10cal fails BOTH                        -> mode 'invalid' (QUARANTINE)
  For the k=10 calibration density_knn is NOT exempt (same k as the
  committed dump; also exercises the density-reference side end to end).
  The k=5 smoke dump is then checked under the selected mode.
  """
  com = load_dump(os.path.join(committed_root, run_name('cup', 1),
                               'latent_uq', 'cup_v1'))
  first = com['checkpoints'][0]['name']
  com_first = dict(com, checkpoints=com['checkpoints'][:1])
  cal = _load_smoke(dumps_root, SMOKE_K10CAL_REL, K_COMMITTED, first,
                    'k10 calibration')
  cal_tol = integrity_failures('k10cal_smoke', K_COMMITTED, cal, com_first,
                               'tolerance', exempt=())
  cal_fb = None
  if not cal_tol:
    mode = 'tolerance'
  else:
    cal_fb = integrity_failures('k10cal_smoke', K_COMMITTED, cal, com_first,
                                'fallback', exempt=())
    mode = 'fallback' if not cal_fb else 'invalid'
  smoke_fails = None
  if mode != 'invalid':
    smoke = _load_smoke(dumps_root, SMOKE_K5_REL, 5, first, 'k5 smoke')
    smoke_fails = integrity_failures('k5_smoke', 5, smoke, com_first, mode)
  record = dict(
      guard_mode=mode, fallback_rank_floor=FALLBACK_RANK_FLOOR,
      calibration_failures_tolerance=cal_tol,
      calibration_failures_fallback=cal_fb,
      smoke_failures=smoke_fails)
  return mode, record


def load_cohort(dumps_root, committed_root):
  """All-or-nothing load of committed + both-k dumps for the 10 cells."""
  missing = []
  for dom, seed in CELLS:
    run = run_name(dom, seed)
    com = os.path.join(committed_root, run, 'latent_uq', f'{dom}_v1')
    if not os.path.isdir(com):
      missing.append(com)
    for k in K2S:
      new = os.path.join(dumps_root, run, 'latent_uq', f'{dom}_v1_k{k}')
      if not os.path.isdir(new):
        missing.append(new)
  assert not missing, (
      'cohort incomplete (all-or-nothing: the committed claim is 10/10 and '
      'a partial cohort cannot adjudicate its robustness); missing: '
      + ', '.join(missing))
  committed, new_dumps = {}, {}
  for dom, seed in CELLS:
    label = cell_label(dom, seed)
    run = run_name(dom, seed)
    com = load_dump(os.path.join(committed_root, run, 'latent_uq',
                                 f'{dom}_v1'))
    check_index(com, K_COMMITTED, dom, run, None, f'{label}@k{K_COMMITTED}')
    check_meta(com, K_COMMITTED, f'{label}@k{K_COMMITTED}')
    com_names = [c['name'] for c in com['checkpoints']]
    committed[label] = com
    for k in K2S:
      dump = load_dump(os.path.join(dumps_root, run, 'latent_uq',
                                    f'{dom}_v1_k{k}'))
      check_index(dump, k, dom, run, com_names, f'{label}@k{k}')
      check_meta(dump, k, f'{label}@k{k}', new_dump=True)
      new_dumps[(label, k)] = dump
  return committed, new_dumps


def cell_verdict(label, dump, n_boot):
  """Committed verdict machinery, unchanged, on one second-k dump."""
  ckpt_rows = []
  for i, entry in enumerate(dump['checkpoints']):
    npz = entry['npz']
    clusters = clusters_of(npz)
    rows = []
    for kind, h, D, E, rho in within_signals(npz, list(HORIZONS)):
      rows.append(dict(kind=kind, horizon=h, **row_stats(
          D, E, rho, clusters, n_boot, BOOT_SEED + 13 * i + h)))
    ckpt_rows.append((entry, rows))
  return analyze_cell(label, ckpt_rows, PRIMARY_HORIZON)


def trajectory(cell):
  """Descriptive per-checkpoint anchor-h5 verdicts (phase-change shape)."""
  out = []
  for ck in cell['checkpoints']:
    pick = [r for r in ck['rows']
            if r['kind'] == 'anchor' and r['horizon'] == PRIMARY_HORIZON]
    out.append(dict(name=ck['name'],
                    verdict=pick[0]['verdict'] if pick else 'NO_DATA'))
  return out


def adjudicate(cells):
  """P-K1 on the 20 final-checkpoint anchor-h5 verdicts."""
  flips = [dict(cell=label, k=k, verdict=c['final_verdict'])
           for (label, k), c in sorted(cells.items())
           if c['final_verdict'] != EXPECTED_COMMITTED_VERDICT]
  if not flips:
    verdict = (
        'SECOND-K ROBUST: all 10 held-out p2e cells re-verdict '
        'TRACKS_DENSITY at BOTH k=5 and k=20 (20/20) -- the committed '
        '10/10 density-meter claim is not an artifact of k=10; the '
        'single-k externalization qualifier is discharged.')
  else:
    enum = ', '.join(f'{f["cell"]}@k{f["k"]}={f["verdict"]}' for f in flips)
    verdict = (
        f'K-SENSITIVE: {len(flips)} of 20 cell x k verdicts differ from '
        f'TRACKS_DENSITY ({enum}) -- the committed claim is k-dependent; '
        f'every external TRACKS_DENSITY statement must carry the k=10 '
        f'qualifier and enumerate the flipped cells.')
  return flips, verdict


def analyze(committed, new_dumps, mode, n_boot=N_BOOT):
  """Integrity first; verdicts only on a clean cohort (else QUARANTINE)."""
  fails, n_checked = [], 0
  for (label, k), dump in sorted(new_dumps.items()):
    fails += integrity_failures(label, k, dump, committed[label], mode)
    n_checked += sum(len(c['npz']) - len(K_DEPENDENT)
                     for c in dump['checkpoints'])
  integrity = dict(guard_mode=mode, n_arrays_checked=n_checked,
                   n_failures=len(fails),
                   float_rtol=FLOAT_RTOL, float_atol=FLOAT_ATOL,
                   fallback_rank_floor=FALLBACK_RANK_FLOOR,
                   failures=fails)
  if fails:
    enum = '; '.join(
        f'{f["cell"]}@k{f["k"]}/{f["ckpt"]}/{f["key"]}' for f in fails[:20])
    verdict = (
        f'QUARANTINE: {len(fails)} k-independent array(s) differ from the '
        f'committed k=10 dumps under guard mode {mode!r} ({enum}) -- the '
        f're-dump is NOT a controlled single-variable change (anchors, '
        f'probe windows, reference set, or checkpoint changed, not just '
        f'k); no verdict; audit the pass before ANY use.')
    return dict(integrity=integrity, cells=None, flips=None,
                p_k1_robust=None, verdict=verdict)
  cells = {(label, k): cell_verdict(label, dump, n_boot)
           for (label, k), dump in sorted(new_dumps.items())}
  flips, verdict = adjudicate(cells)
  cell_blocks = {
      f'{label}@k{k}': dict(
          final_verdict=c['final_verdict'],
          addendum_pass_h1=c['addendum_pass'],
          trajectory_anchor_h5=trajectory(c),
          checkpoints=c['checkpoints'])
      for (label, k), c in sorted(cells.items())}
  return dict(integrity=integrity, cells=cell_blocks, flips=flips,
              p_k1_robust=not flips, verdict=verdict)


def run_read(dumps_root, committed_root, artifact_root, output,
             n_boot=N_BOOT, gate_json=None):
  mode, smoke_rec = check_smoke(dumps_root, committed_root)
  check_artifact(artifact_root)
  gate = None
  if gate_json is not None:
    assert os.path.exists(gate_json), (
        f'--gate_json {gate_json} missing (the registered existence gate '
        f'writes it before any pass; sync it back with the dumps)')
    with open(gate_json) as f:
      gate = json.load(f)
    assert gate.get('verdict') == 'SUBSTRATE OK', (
        f'gate record says {gate.get("verdict")!r}: the registered '
        f'SUBSTRATE-GONE branch runs NO passes and has no read')
  if mode == 'invalid':
    res = dict(
        integrity=None, cells=None, flips=None, p_k1_robust=None,
        verdict=(
            'QUARANTINE: registered guard calibration failed -- the k=10 '
            'calibration re-dump does not reproduce the committed '
            'cup-seed1 dump under the tolerance NOR the fallback rank '
            'guard; the backend cannot reproduce the committed substrate '
            'at the SAME k, so no k2 pass is interpretable as a '
            'single-variable change; no verdict; audit before ANY use.'))
  elif smoke_rec['smoke_failures']:
    res = dict(
        integrity=None, cells=None, flips=None, p_k1_robust=None,
        verdict=(
            'QUARANTINE: the k=5 smoke dump fails the integrity guard '
            'against the committed earliest cup-seed1 checkpoint under '
            f'guard mode {mode!r}; the full wave is not interpretable; '
            'audit before ANY use.'))
  else:
    committed, new_dumps = load_cohort(dumps_root, committed_root)
    res = analyze(committed, new_dumps, mode, n_boot)
  res['smoke_gate'] = smoke_rec
  res['substrate_gate'] = gate
  res['constants'] = dict(
      k_committed=K_COMMITTED, k2s=list(K2S), horizons=list(HORIZONS),
      primary_horizon=PRIMARY_HORIZON, n_boot=n_boot,
      verdict_sigma=VERDICT_SIGMA, guard_mode=mode,
      float_rtol=FLOAT_RTOL, float_atol=FLOAT_ATOL,
      fallback_rank_floor=FALLBACK_RANK_FLOOR,
      expected_committed_verdict=EXPECTED_COMMITTED_VERDICT)
  res['dumps_root'] = os.path.abspath(dumps_root)
  res['committed_root'] = os.path.abspath(committed_root)
  res['artifact_root'] = os.path.abspath(artifact_root)
  res['created'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
  res['command'] = ' '.join(sys.argv)
  res['git_commit'] = _git_commit()
  os.makedirs(output, exist_ok=True)
  out = os.path.join(output, 'stage0_secondk.json')
  with open(out, 'w') as f:
    json.dump(res, f, indent=2)
  print(f'guard mode: {mode}')
  if res.get('integrity') is not None:
    print(f'integrity: {res["integrity"]["n_failures"]} failures / '
          f'{res["integrity"]["n_arrays_checked"]} arrays checked')
  if res['cells'] is not None:
    for name, blk in res['cells'].items():
      print(f'  {name}: {blk["final_verdict"]}')
  print(res['verdict'])
  print(f'-> {out}')
  return res


# --------------------------------------------------------------------------
# selfcheck (synthetic tempdir fixtures through the real machinery)
# --------------------------------------------------------------------------

def _write_dump(dump_dir, dom, run, knn, names, feats_per_ckpt, rho,
                ep_batch=EP_BATCH):
  os.makedirs(dump_dir, exist_ok=True)
  index = dict(
      run_logdir=f'/fake/{run}', probeset='/fake/probeset',
      probeset_id=PROBESET_ID[dom], probeset_sha256=PROBESET_SHA[dom],
      burn_in=BURN_IN, eval_steps=EVAL_STEPS,
      ref_replay=f'/fake/{run}/replay',
      ref_windows=REF_WINDOWS, knn=knn, dec_symlog=True, has_disag=True,
      held_out=True, checkpoints=[])
  for name, feats in zip(names, feats_per_ckpt):
    step_dir = os.path.join(dump_dir, name)
    os.makedirs(step_dir, exist_ok=True)
    np.savez_compressed(os.path.join(step_dir, 'uq.npz'),
                        density_knn=rho, **feats)
    with open(os.path.join(step_dir, 'meta.json'), 'w') as f:
      json.dump(dict(knn=knn, seed=DUMP_SEED, ep_batch=ep_batch,
                     exact_step=int(name[4:])), f)
    index['checkpoints'].append(dict(
        name=name, exact_step=int(name[4:]), milestone=None,
        path=os.path.join(name, 'uq.npz')))
  with open(os.path.join(dump_dir, 'index.json'), 'w') as f:
    json.dump(index, f, indent=2)


def _write_cohort(tmp, rng, flip_cell=None, write_smoke=True, S=320,
                  n_streams=40):
  """Committed(k10) + k5/k20 fixtures in the registered layout.

  Disagreement is density-tied (the committed 'biased' regime), so every
  verdict is TRACKS_DENSITY. k2 densities are rank-preserving perturbations
  of the k10 density; for flip_cell=(dom, seed, k) that one dump's density
  is replaced by independent noise, so its verdict flips to TRACKS_ERROR
  through the ONLY k-dependent array -- integrity stays green. Both
  registered smoke dumps (k5 + the k10 guard-calibration) are written on
  cup seed1's earliest checkpoint; the calibration reuses the committed
  arrays exactly, so the untouched ladder selects guard mode 'tolerance'.
  """
  H = EVAL_STEPS
  dumps_root = os.path.join(tmp, 'new')
  committed_root = os.path.join(tmp, 'committed')
  names = [f'step{(i + 1) * 100000:012d}' for i in range(N_CKPTS)]
  std = lambda v: (v - v.mean()) / (v.std() + 1e-9)
  smoke_src = None
  for dom, seed in CELLS:
    run = run_name(dom, seed)
    z = rng.normal(0, 1, S)
    streams = rng.integers(0, n_streams, S)
    labels = np.asarray(['p2e'] * S)
    rho10 = np.exp(z + 0.05 * rng.normal(0, 1, S)).astype(np.float32)
    feats_per_ckpt = []
    for _ in range(N_CKPTS):
      e1 = 0.5 * z + rng.normal(0, 1, S)
      d1 = 1.2 * z + 0.1 * std(e1) + 0.3 * rng.normal(0, 1, S)
      err = (np.exp(e1)[:, None] * np.linspace(1, 2, H)[None]
             * np.exp(0.1 * rng.normal(0, 1, (S, H)))).astype(np.float32)
      disag = (np.exp(0.5 * d1)[:, None] * np.linspace(1, 1.5, H)[None]
               * np.exp(0.05 * rng.normal(0, 1, (S, H)))).astype(np.float32)
      feats_per_ckpt.append(dict(
          err_steps=err, err_obs=(2.0 * err).astype(np.float32),
          disag_steps=disag, source_label=labels,
          stream_start=np.stack(
              [streams, np.zeros(S, np.int64)], -1)))
    _write_dump(os.path.join(committed_root, run, 'latent_uq', f'{dom}_v1'),
                dom, run, K_COMMITTED, names, feats_per_ckpt, rho10)
    for k in K2S:
      if flip_cell == (dom, seed, k):
        rho_k = np.exp(rng.normal(0, 1, S)).astype(np.float32)
      else:
        rho_k = np.exp(
            np.log(rho10) + 0.02 * rng.normal(0, 1, S)).astype(np.float32)
      _write_dump(
          os.path.join(dumps_root, run, 'latent_uq', f'{dom}_v1_k{k}'),
          dom, run, k, names, feats_per_ckpt, rho_k)
      if (dom, seed, k) == ('cup', 1, 5):
        smoke_src = (names[:1], feats_per_ckpt[:1], rho10, rho_k)
  if write_smoke:
    names1, feats1, rho10_1, rho_k5_1 = smoke_src
    run1 = run_name('cup', 1)
    _write_dump(os.path.join(dumps_root, SMOKE_K5_REL), 'cup', run1, 5,
                names1, feats1, rho_k5_1)
    _write_dump(os.path.join(dumps_root, SMOKE_K10CAL_REL), 'cup', run1,
                K_COMMITTED, names1, feats1, rho10_1)
  artifact_root = os.path.join(tmp, 'artifact')
  for dom in DOMS:
    d = os.path.join(artifact_root, f'{dom}_p2e')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'analysis.json'), 'w') as f:
      json.dump(dict(
          primary_horizon=PRIMARY_HORIZON, n_boot=N_BOOT,
          verdict_sigma=VERDICT_SIGMA, cross=False, held_out=True,
          cells=[dict(label=cell_label(dom, s),
                      final_verdict=EXPECTED_COMMITTED_VERDICT)
                 for s in SEEDS]), f)
  return dumps_root, committed_root, artifact_root


def _edit_npz(path, fn):
  z = dict(np.load(path, allow_pickle=True))
  np.savez_compressed(path, **fn(z))


def _edit_json(path, fn):
  with open(path) as f:
    d = json.load(f)
  with open(path, 'w') as f:
    json.dump(fn(d), f)


def _expect_trip(fn, needle):
  try:
    fn()
    raise SystemExit(f'selfcheck FAIL: {needle!r} not caught')
  except AssertionError as e:
    assert needle in str(e), (needle, str(e))


def selfcheck(args):
  import shutil
  import tempfile
  n_boot = 40  # selfcheck speed only; the read path always uses N_BOOT=300

  # Hand-computed estimand identities: within_signals row slicing, the
  # verdict_of thresholds, the registered tolerance arithmetic, and the
  # fallback rank guard semantics.
  hz = dict(disag_steps=np.arange(24, dtype=np.float32).reshape(3, 8) + 1,
            err_steps=np.arange(24, dtype=np.float32).reshape(3, 8) * 2,
            density_knn=np.asarray([3.0, 1.0, 2.0], np.float32))
  rows = within_signals(hz, list(HORIZONS))
  assert [(r[0], r[1]) for r in rows] == [
      ('anchor', 1), ('path', 1), ('anchor', 5), ('path', 5)], rows
  kind, h, D, E, rho = rows[2]  # anchor, h=5
  assert np.array_equal(D, hz['disag_steps'][:, 0])
  assert np.array_equal(E, hz['err_steps'][:, 4])       # err at horizon 5
  assert np.array_equal(rho, hz['density_knn'])
  assert verdict_of(-1.0, 0.4) == 'TRACKS_DENSITY'      # -1 < -2*0.4
  assert verdict_of(+1.0, 0.4) == 'TRACKS_ERROR'
  assert verdict_of(+0.5, 0.4) == 'AMBIGUOUS'
  a = np.asarray([1.0], np.float32)
  assert np.allclose(a * (1 + 2e-5), a, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
  assert not np.allclose(a * 1.01, a, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
  v = np.asarray([1.0, 2.0, 3.0])
  # tolerance, not float equality: the sqrt-normalized quotient returns
  # 0.9999999999999998 on exact monotone inputs
  assert abs(rank_corr(v, 10 * v) - 1.0) < 1e-9
  assert abs(rank_corr(v, v[::-1].copy()) + 1.0) < 1e-9
  # Fallback semantics: rank-preserving float drift beyond the tolerance
  # passes for ordinary floats, but true_* stays under the tolerance and
  # rank destruction fails.
  base_np = dict(err_steps=np.linspace(1, 2, 8).astype(np.float32),
                 true_pos=np.linspace(1, 2, 8).astype(np.float32),
                 source_label=np.asarray(['p2e'] * 8))
  mk = lambda npz: dict(checkpoints=[dict(name='step0', npz=npz)])
  drifted = dict(base_np,
                 err_steps=(base_np['err_steps'] * 1.05).astype(np.float32))
  assert integrity_failures('u', 5, mk(drifted), mk(base_np), 'tolerance')
  assert not integrity_failures('u', 5, mk(drifted), mk(base_np), 'fallback')
  tdrift = dict(base_np,
                true_pos=(base_np['true_pos'] * 1.05).astype(np.float32))
  fb = integrity_failures('u', 5, mk(tdrift), mk(base_np), 'fallback')
  assert [f['key'] for f in fb] == ['true_pos'], fb
  shuf = dict(base_np, err_steps=base_np['err_steps'][::-1].copy())
  fb = integrity_failures('u', 5, mk(shuf), mk(base_np), 'fallback')
  assert [f['key'] for f in fb] == ['err_steps'], fb
  assert 'rank correlation' in fb[0]['detail'], fb

  tmp = tempfile.mkdtemp(prefix='stage0_secondk_selfcheck_')
  rng = np.random.default_rng(20260730)

  # Leg A: robust cohort, incl. a within-tolerance float perturbation on
  # one re-dump (cross-backend drift must NOT quarantine); guard mode
  # resolves to 'tolerance' (calibration reproduces the committed arrays).
  dumps_root, committed_root, artifact_root = _write_cohort(tmp, rng)
  drift = os.path.join(dumps_root, run_name('cup', 2), 'latent_uq',
                       'cup_v1_k5', 'step000000100000', 'uq.npz')
  _edit_npz(drift, lambda z: dict(
      z, err_steps=(z['err_steps'] * (1 + 2e-5)).astype(np.float32)))
  mode, rec = check_smoke(dumps_root, committed_root)
  assert mode == 'tolerance' and rec['smoke_failures'] == [], (mode, rec)
  outA = os.path.join(tmp, 'outA')
  res = run_read(dumps_root, committed_root, artifact_root, outA, n_boot)
  assert res['verdict'].startswith('SECOND-K ROBUST'), res['verdict']
  assert res['p_k1_robust'] is True and res['flips'] == []
  assert res['integrity']['n_failures'] == 0
  assert res['integrity']['guard_mode'] == 'tolerance'
  assert res['smoke_gate']['guard_mode'] == 'tolerance'
  assert res['substrate_gate'] is None
  assert res['created'] and res['command']
  assert len(res['cells']) == 20
  assert all(b['final_verdict'] == 'TRACKS_DENSITY'
             for b in res['cells'].values())
  assert all(len(b['trajectory_anchor_h5']) == N_CKPTS
             for b in res['cells'].values())
  with open(os.path.join(outA, 'stage0_secondk.json')) as f:
    assert json.load(f)['p_k1_robust'] is True

  # Gate-record consumption: OK embeds; GONE and missing-file trip.
  gate_ok = os.path.join(tmp, 'gate_ok.json')
  with open(gate_ok, 'w') as f:
    json.dump(dict(verdict='SUBSTRATE OK', missing=[], runs={}), f)
  res = run_read(dumps_root, committed_root, artifact_root,
                 os.path.join(tmp, 'outA2'), n_boot, gate_json=gate_ok)
  assert res['substrate_gate']['verdict'] == 'SUBSTRATE OK'
  gate_gone = os.path.join(tmp, 'gate_gone.json')
  with open(gate_gone, 'w') as f:
    json.dump(dict(verdict='SUBSTRATE-GONE', missing=['x'], runs={}), f)
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot,
                                gate_json=gate_gone),
               'SUBSTRATE-GONE')
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot,
                                gate_json=os.path.join(tmp, 'nope.json')),
               'gate_json')

  # Guard needles on leg A's tree (each restored after tripping).
  smoke_idx = os.path.join(dumps_root, SMOKE_K5_REL, 'index.json')
  os.rename(smoke_idx, smoke_idx + '.bak')
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'smoke gate')
  os.rename(smoke_idx + '.bak', smoke_idx)
  cal_idx = os.path.join(dumps_root, SMOKE_K10CAL_REL, 'index.json')
  os.rename(cal_idx, cal_idx + '.bak')
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'k10 calibration')
  os.rename(cal_idx + '.bak', cal_idx)
  art = os.path.join(artifact_root, 'cup_p2e', 'analysis.json')
  _edit_json(art, lambda d: dict(d, cells=[
      dict(c, final_verdict='AMBIGUOUS') if c['label'] == 'p2e_cup_seed1'
      else c for c in d['cells']]))
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'committed baseline changed')
  _edit_json(art, lambda d: dict(d, cells=[
      dict(c, final_verdict=EXPECTED_COMMITTED_VERDICT)
      for c in d['cells']]))
  idx_path = os.path.join(dumps_root, run_name('finger', 1), 'latent_uq',
                          'finger_v1_k5', 'index.json')
  _edit_json(idx_path, lambda d: dict(d, knn=10))
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'index knn')
  _edit_json(idx_path, lambda d: dict(d, knn=5))
  _edit_json(idx_path, lambda d: dict(d, probeset_sha256='deadbeef'))
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'probeset sha')
  _edit_json(idx_path, lambda d: dict(d, probeset_sha256=PROBESET_SHA[
      'finger']))
  _edit_json(idx_path, lambda d: dict(
      d, ref_replay='/fake/pretrain_p2e_cup_seed2/replay'))
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'ref_replay')
  _edit_json(idx_path, lambda d: dict(
      d, ref_replay=f'/fake/{run_name("finger", 1)}/replay'))
  _edit_json(idx_path, lambda d: dict(d, checkpoints=[
      dict(c, name='step999999999999') if i == 0 else c
      for i, c in enumerate(d['checkpoints'])]))
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'checkpoint names')
  _edit_json(idx_path, lambda d: dict(d, checkpoints=[
      dict(c, name=f'step{(i + 1) * 100000:012d}')
      for i, c in enumerate(d['checkpoints'])]))
  meta_path = os.path.join(dumps_root, run_name('cup', 1), 'latent_uq',
                           'cup_v1_k20', 'step000000100000', 'meta.json')
  _edit_json(meta_path, lambda d: dict(d, seed=7))
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'dumper seed')
  _edit_json(meta_path, lambda d: dict(d, seed=DUMP_SEED))
  _edit_json(meta_path, lambda d: dict(d, ep_batch=32))
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'ep_batch')
  _edit_json(meta_path, lambda d: {k: v for k, v in dict(
      d, ep_batch=EP_BATCH).items()})
  # a re-dump LACKING the key entirely must also trip (dumper predates
  # the registered patch)
  _edit_json(meta_path, lambda d: {k: v for k, v in d.items()
                                   if k != 'ep_batch'})
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'lacks ep_batch')
  _edit_json(meta_path, lambda d: dict(d, ep_batch=EP_BATCH))

  # Leg D: missing cell => all-or-nothing refusal (no partial cohort).
  gone = os.path.join(dumps_root, run_name('finger', 5), 'latent_uq',
                      'finger_v1_k20')
  shutil.move(gone, gone + '.bak')
  _expect_trip(lambda: run_read(dumps_root, committed_root, artifact_root,
                                os.path.join(tmp, 'x'), n_boot),
               'all-or-nothing')
  shutil.move(gone + '.bak', gone)

  # Leg C: beyond-tolerance float drift + a stray extra key => QUARANTINE
  # with both failures enumerated and NO verdicts computed.
  bad1 = os.path.join(dumps_root, run_name('cup', 3), 'latent_uq',
                      'cup_v1_k20', 'step000000200000', 'uq.npz')
  _edit_npz(bad1, lambda z: dict(
      z, err_steps=(z['err_steps'] * 1.01).astype(np.float32)))
  bad2 = os.path.join(dumps_root, run_name('finger', 2), 'latent_uq',
                      'finger_v1_k5', 'step000000300000', 'uq.npz')
  _edit_npz(bad2, lambda z: dict(z, extra=np.zeros(3, np.float32)))
  outC = os.path.join(tmp, 'outC')
  res = run_read(dumps_root, committed_root, artifact_root, outC, n_boot)
  assert res['verdict'].startswith('QUARANTINE'), res['verdict']
  assert res['cells'] is None and res['p_k1_robust'] is None
  got = {(f['cell'], f['k'], f['ckpt'], f['key'])
         for f in res['integrity']['failures']}
  assert ('p2e_cup_seed3', 20, 'step000000200000', 'err_steps') in got, got
  assert ('p2e_finger_seed2', 5, 'step000000300000', '<key set>') in got, got
  assert len(got) == 2, got

  # Smoke-failure quarantine: a k=5 smoke dump that fails the guard
  # preempts everything else (checked before the cohort is even loaded).
  smoke_npz = os.path.join(dumps_root, SMOKE_K5_REL, 'step000000100000',
                           'uq.npz')
  _edit_npz(smoke_npz, lambda z: dict(
      z, err_steps=(z['err_steps'] * 1.01).astype(np.float32)))
  res = run_read(dumps_root, committed_root, artifact_root,
                 os.path.join(tmp, 'outS'), n_boot)
  assert res['verdict'].startswith('QUARANTINE: the k=5 smoke dump'), (
      res['verdict'])
  assert res['cells'] is None and res['integrity'] is None
  assert res['smoke_gate']['smoke_failures'], res['smoke_gate']

  # Leg B: one cell's k=20 density decoupled => K-SENSITIVE, that cell
  # enumerated, all other 19 verdicts unchanged; integrity stays green.
  tmpB = os.path.join(tmp, 'B')
  os.makedirs(tmpB, exist_ok=True)
  dumps_root, committed_root, artifact_root = _write_cohort(
      tmpB, rng, flip_cell=('finger', 3, 20))
  outB = os.path.join(tmp, 'outB')
  res = run_read(dumps_root, committed_root, artifact_root, outB, n_boot)
  assert res['verdict'].startswith('K-SENSITIVE'), res['verdict']
  assert res['p_k1_robust'] is False
  assert res['integrity']['n_failures'] == 0
  assert [(f['cell'], f['k']) for f in res['flips']] == [
      ('p2e_finger_seed3', 20)], res['flips']
  assert 'p2e_finger_seed3@k20' in res['verdict']
  others = [b['final_verdict'] for name, b in res['cells'].items()
            if name != 'p2e_finger_seed3@k20']
  assert len(others) == 19 and all(v == 'TRACKS_DENSITY' for v in others)

  # Leg E: guard-calibration ladder -> fallback mode. The k=10 calibration
  # drifts beyond the tolerance but rank-preserving (bf16-style backend
  # drift); a full-pass dump carries the same kind of drift; the read
  # completes under guard mode 'fallback' and records it.
  tmpE = os.path.join(tmp, 'E')
  os.makedirs(tmpE, exist_ok=True)
  dumps_root, committed_root, artifact_root = _write_cohort(tmpE, rng)
  cal_npz = os.path.join(dumps_root, SMOKE_K10CAL_REL, 'step000000100000',
                         'uq.npz')
  _edit_npz(cal_npz, lambda z: dict(
      z, err_steps=(z['err_steps'] * 1.01).astype(np.float32)))
  full_npz = os.path.join(dumps_root, run_name('finger', 2), 'latent_uq',
                          'finger_v1_k20', 'step000000400000', 'uq.npz')
  _edit_npz(full_npz, lambda z: dict(
      z, err_obs=(z['err_obs'] * 1.02).astype(np.float32)))
  outE = os.path.join(tmp, 'outE')
  res = run_read(dumps_root, committed_root, artifact_root, outE, n_boot)
  assert res['verdict'].startswith('SECOND-K ROBUST'), res['verdict']
  assert res['integrity']['guard_mode'] == 'fallback'
  assert res['smoke_gate']['guard_mode'] == 'fallback'
  assert res['smoke_gate']['calibration_failures_tolerance'], res['smoke_gate']
  assert res['smoke_gate']['calibration_failures_fallback'] == []
  assert res['integrity']['n_failures'] == 0

  # Leg F: under fallback, rank destruction on a full-pass dump (changed
  # probe-window set) still quarantines.
  _edit_npz(full_npz, lambda z: dict(
      z, err_steps=z['err_steps'][::-1].copy()))
  outF = os.path.join(tmp, 'outF')
  res = run_read(dumps_root, committed_root, artifact_root, outF, n_boot)
  assert res['verdict'].startswith('QUARANTINE'), res['verdict']
  assert res['integrity']['guard_mode'] == 'fallback'
  keys = {f['key'] for f in res['integrity']['failures']}
  assert 'err_steps' in keys, res['integrity']['failures']
  assert any('rank correlation' in f['detail']
             for f in res['integrity']['failures'])

  # Leg G: calibration fails BOTH modes => guard invalid => QUARANTINE
  # before any cohort work.
  _edit_npz(cal_npz, lambda z: dict(
      z, err_steps=z['err_steps'][::-1].copy()))
  outG = os.path.join(tmp, 'outG')
  res = run_read(dumps_root, committed_root, artifact_root, outG, n_boot)
  assert res['verdict'].startswith(
      'QUARANTINE: registered guard calibration failed'), res['verdict']
  assert res['integrity'] is None and res['cells'] is None
  assert res['smoke_gate']['guard_mode'] == 'invalid'

  print('selfcheck PASS: robust leg (20/20 TRACKS_DENSITY incl. '
        'within-tolerance drift, guard mode tolerance), flip leg '
        '(K-SENSITIVE via the only k-dependent array, flipped cell '
        'enumerated, integrity green), quarantine leg (beyond-tolerance + '
        'key-set failures enumerated, no verdicts computed), smoke-failure '
        'quarantine (k=5 smoke guard preempts the read), calibration '
        'ladder (fallback mode robust read, fallback rank-destruction '
        'quarantine, calibration-invalid quarantine), gate-record legs '
        '(OK embedded, SUBSTRATE-GONE and missing file trip), '
        'all-or-nothing missing-cell trip, guard needles (smoke gate, k10 '
        'calibration, committed-baseline change, index knn, probeset sha, '
        'ref_replay, checkpoint names, dumper seed, ep_batch wrong/absent '
        'on new dumps, smoke full-dial pins), hand identities '
        '(anchor-row slicing, verdict_of thresholds, tolerance arithmetic, '
        'rank_corr, fallback true_*/rank semantics), import pin '
        'VERDICT_SIGMA=2.0')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dumps_root',
                  help='Root of the second-k dumps: '
                       '<root>/pretrain_p2e_{dom}_seed{s}/latent_uq/'
                       '{dom}_v1_k{5,20}')
  ap.add_argument('--committed_root',
                  help='Root of the committed k=10 dumps: '
                       '<root>/pretrain_p2e_{dom}_seed{s}/latent_uq/'
                       '{dom}_v1')
  ap.add_argument('--artifact_root', default=ARTIFACT_DEFAULT)
  ap.add_argument('--gate_json', default=None,
                  help='Existence-gate JSON record '
                       '(secondk_substrate_gate.json); must say '
                       'SUBSTRATE OK; embedded in the output.')
  ap.add_argument('--output', default='analysis_out/stage0_secondk')
  ap.add_argument('--smoke_check', action='store_true',
                  help='Run ONLY the registered smoke + guard-calibration '
                       'gate (value-blind); use between the smoke dumps '
                       'and the full passes.')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
    return
  if not (args.dumps_root and args.committed_root):
    ap.error('--dumps_root and --committed_root required (or --selfcheck)')
  if args.smoke_check:
    mode, rec = check_smoke(args.dumps_root, args.committed_root)
    n_tol = len(rec['calibration_failures_tolerance'])
    print(f'guard calibration: {n_tol} tolerance failure(s); '
          f'guard_mode={mode}')
    if mode == 'invalid':
      raise SystemExit(
          'SMOKE-CHECK QUARANTINE: the k=10 calibration fails the '
          'tolerance AND the fallback rank guard; the backend cannot '
          'reproduce the committed substrate at the same k. Do NOT run '
          'the full passes; audit.')
    if rec['smoke_failures']:
      raise SystemExit(
          f'SMOKE-CHECK QUARANTINE: the k=5 smoke dump fails the '
          f'integrity guard under guard mode {mode!r}. Do NOT run the '
          f'full passes; audit.')
    print(f'SMOKE-CHECK OK: guard_mode={mode}; proceed with the 20 full '
          f'passes.')
    return
  run_read(args.dumps_root, args.committed_root, args.artifact_root,
           args.output, gate_json=args.gate_json)


if __name__ == '__main__':
  main()
