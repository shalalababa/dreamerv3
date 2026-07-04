"""Latent-UQ Stage 0 analysis: does disagreement track error or density?

Consumes dumps written by ``probing/latent_uq.py`` and reports, per run
(cell) and per retained checkpoint, the pre-registered Stage-0 read
(Latent-UQ idea brief Sec. 6; EVPI theory note App. A.4 item 5 / A.7):

* Spearman correlations disagreement<->held-out rollout error and
  disagreement<->latent training density, with stream-clustered bootstrap
  SEs (windows from one replay stream are autocorrelated);
* rank-based *partial* correlations pcorr(D, E | rho) vs pcorr(D, rho | E)
  and their difference Delta (clustered SE) -- the calibration read;
* a replication verdict per checkpoint: TRACKS_DENSITY (the Biased-Dreams
  attractor bias replicates), TRACKS_ERROR, or AMBIGUOUS (|Delta| within
  ``VERDICT_SIGMA`` clustered SEs);
* the Stage-0 calibration-addendum line: PASS iff pcorr(D, E | rho) >=
  pcorr(D, rho | E) at the final checkpoint, evaluated on the
  pre-registered *one-step* read (anchor, h=1; App. A.4 item 5) regardless
  of ``--primary_horizon``.  This is the addendum-style proxy the memo
  reports; the full App. A.7 gate criterion additionally applies the dose
  adjustment, which lives in the D0 pipeline (``d0/``), not here.

Disagreement functionals: ``anchor`` = one-step disagreement at the segment
anchor (primary, the brief's functional); ``path`` = mean disagreement along
the open-loop path up to the horizon (secondary). With ``--cross``, the
given dumps are treated as members of a cross-seed/refit ensemble on the
*same* frozen probe set and disagreement is instead the per-dimension
normalized variance across members' decoded open-loop predictions
(observation space, where coordinates are identifiable across members);
error and density are member means.

Density sign convention: ``density_knn`` is a mean kNN *distance* (larger =
sparser). The attractor bias of Biased Dreams (arXiv 2604.25416) therefore
predicts a *positive* partial correlation of disagreement with the proxy.

Outputs ``analysis.json`` and the evidence memo ``memo.md`` (structure:
``artifacts/latent_uq_stage0/MEMO_TEMPLATE.md``) into ``--output``
(default ``artifacts/latent_uq_stage0/analysis_<date>``).

Usage
-----
Within-checkpoint (disag-head) cells, one dump per run::

    python -m probing.latent_uq_analysis \
        --dumps p2e_cup_seed1=$RUN/pretrain_p2e_cup_seed1/latent_uq/cup_v1 \
                p2e_cup_seed2=$RUN/pretrain_p2e_cup_seed2/latent_uq/cup_v1 \
        --horizons 1 5 15 --output artifacts/latent_uq_stage0/cup_p2e

Cross-seed ensemble (one cell from K member dumps on the same probe set)::

    python -m probing.latent_uq_analysis --cross --cell apt_cup \
        --dumps seed1=... seed2=... seed3=... seed4=... seed5=... \
        --horizons 1 5 15 --output artifacts/latent_uq_stage0/cup_apt_cross

Validation (synthetic end-to-end, no data or JAX needed)::

    python -m probing.latent_uq_analysis --selfcheck
"""

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from d0.analysis import cluster_bootstrap, partial_spearman, spearman

# Provisional verdict constant (mirror of the d0/analysis.py discipline):
# |Delta| must exceed this many clustered SEs to leave AMBIGUOUS.
VERDICT_SIGMA = 2.0


# --------------------------------------------------------------------------
# Shared numpy math (also imported by probing/latent_uq.py).
# --------------------------------------------------------------------------

def np_symlog(x):
  """Numpy mirror of embodied.jax.nets.symlog (decoder vec target space)."""
  x = np.asarray(x, np.float32)
  return np.sign(x) * np.log1p(np.abs(x))


def knn_distance(x, ref, k, chunk=512):
  """Mean distance of each row of x to its k nearest rows of ref.

  Dimensions are standardized by the reference sample's mean/std first
  (plain Euclidean kNN is fragile in anisotropic high-dim latent spaces).
  Larger values = sparser (lower training density).
  """
  x = np.asarray(x, np.float64)
  ref = np.asarray(ref, np.float64)
  k = min(k, ref.shape[0])
  mu, sd = ref.mean(0), ref.std(0) + 1e-6
  x = (x - mu) / sd
  ref = (ref - mu) / sd
  rr = (ref ** 2).sum(-1)
  out = np.empty(x.shape[0], np.float64)
  for lo in range(0, x.shape[0], chunk):
    hi = min(lo + chunk, x.shape[0])
    xx = x[lo:hi]
    d2 = (xx ** 2).sum(-1)[:, None] + rr[None, :] - 2 * xx @ ref.T
    d2 = np.maximum(d2, 0.0)
    part = np.partition(d2, k - 1, axis=1)[:, :k]
    out[lo:hi] = np.sqrt(part).mean(1)
  return out.astype(np.float32)


# --------------------------------------------------------------------------
# Dump loading and signal construction.
# --------------------------------------------------------------------------

def load_dump(dump_dir):
  """Load a latent_uq dump directory into {index, checkpoints:[{...}]}."""
  with open(os.path.join(dump_dir, 'index.json')) as f:
    index = json.load(f)
  ckpts = []
  for entry in index['checkpoints']:
    npz = {k: np.asarray(v)
           for k, v in np.load(os.path.join(dump_dir, entry['path'])).items()}
    ckpts.append(dict(entry, npz=npz))
  return dict(index=index, checkpoints=ckpts, dir=os.path.abspath(dump_dir))


def clusters_of(npz):
  """Stream-level cluster ids (source label + stream index)."""
  labels = npz['source_label'].astype(str)
  streams = npz['stream_start'][:, 0].astype(int)
  return np.asarray([f'{l}:{s}' for l, s in zip(labels, streams)])


def clamp_horizons(horizons, available):
  kept = sorted({h for h in horizons if 1 <= h <= available})
  dropped = sorted(set(horizons) - set(kept))
  if dropped:
    print(f'WARNING: horizons {dropped} exceed the dump\'s eval_steps '
          f'({available}); dropped.')
  if not kept:
    raise SystemExit(f'No usable horizons among {horizons} '
                     f'(eval_steps={available}).')
  return kept


def within_signals(npz, horizons):
  """(kind, horizon, D, E, rho) rows for one checkpoint's own disag head."""
  if 'disag_steps' not in npz:
    return []
  disag, err, rho = npz['disag_steps'], npz['err_steps'], npz['density_knn']
  rows = []
  for h in horizons:
    rows.append(('anchor', h, disag[:, 0], err[:, h - 1], rho))
    rows.append(('path', h, disag[:, :h].mean(1), err[:, h - 1], rho))
  return rows


def cross_disagreement(members):
  """Per-step cross-member disagreement from decoded predictions.

  Variance across members of pred_<key>, per dimension normalized by the
  (shared) true targets' per-dim std, averaged over dims and keys.
  Returns (S, H).
  """
  keys = sorted(k[5:] for k in members[0] if k.startswith('pred_'))
  assert keys, 'cross mode needs pred_* arrays (vector observation keys)'
  parts = []
  for k in keys:
    preds = np.stack([m[f'pred_{k}'] for m in members], 0)  # (K, S, H, D)
    true = members[0][f'true_{k}']
    sd = true.reshape(-1, true.shape[-1]).std(0) + 1e-6
    parts.append(preds.var(0) / (sd ** 2)[None, None])
  return np.concatenate(parts, -1).mean(-1)


def cross_signals(member_npzs, horizons):
  """(kind, horizon, D, E, rho) rows for a cross-seed/refit ensemble."""
  disag = cross_disagreement(member_npzs)
  err = np.stack([m['err_steps'] for m in member_npzs], 0).mean(0)
  rho = np.stack([m['density_knn'] for m in member_npzs], 0).mean(0)
  rows = []
  for h in horizons:
    rows.append(('anchor', h, disag[:, 0], err[:, h - 1], rho))
    rows.append(('path', h, disag[:, :h].mean(1), err[:, h - 1], rho))
  return rows


# --------------------------------------------------------------------------
# Statistics.
# --------------------------------------------------------------------------

def row_stats(D, E, rho, clusters, n_boot=300, seed=0):
  """All Stage-0 aggregates for one (D, E, rho) triple."""
  D, E, rho = (np.asarray(v, np.float64) for v in (D, E, rho))
  rho_de, se_de = spearman(D, E, clusters, n_boot, seed)
  rho_dp, se_dp = spearman(D, rho, clusters, n_boot, seed + 1)
  rho_ep, _ = spearman(E, rho, None)
  pcorr_de = partial_spearman(D, E, rho)
  pcorr_dp = partial_spearman(D, rho, E)
  delta = pcorr_de - pcorr_dp
  delta_se = cluster_bootstrap(
      lambda idx: (partial_spearman(D[idx], E[idx], rho[idx])
                   - partial_spearman(D[idx], rho[idx], E[idx])),
      clusters, n_boot, seed + 2)
  return dict(
      n=int(D.size),
      rho_de=rho_de, rho_de_se=se_de,
      rho_dp=rho_dp, rho_dp_se=se_dp,
      rho_ep=rho_ep,
      pcorr_de=pcorr_de, pcorr_dp=pcorr_dp,
      delta=delta, delta_se=delta_se,
      verdict=verdict_of(delta, delta_se))


def verdict_of(delta, delta_se):
  if not np.isfinite(delta) or not np.isfinite(delta_se):
    return 'AMBIGUOUS'
  if delta > VERDICT_SIGMA * delta_se:
    return 'TRACKS_ERROR'
  if delta < -VERDICT_SIGMA * delta_se:
    return 'TRACKS_DENSITY'
  return 'AMBIGUOUS'


def analyze_cell(label, ckpt_rows, primary_horizon):
  """Cell summary from per-checkpoint row lists (within or cross).

  The headline verdict reads the anchor row at ``primary_horizon``; the
  calibration addendum always reads the anchor h=1 row -- the
  pre-registered one-step quantity (App. A.4 item 5), not steerable from
  the command line.
  """
  checkpoints = []
  for entry, rows in ckpt_rows:
    checkpoints.append(dict(
        name=entry['name'], exact_step=entry['exact_step'],
        milestone=entry['milestone'], rows=rows))
  final = checkpoints[-1]['rows'] if checkpoints else []
  pick = lambda h: [r for r in final
                    if r['kind'] == 'anchor' and r['horizon'] == h]
  primary = pick(primary_horizon)
  final_verdict = primary[0]['verdict'] if primary else 'NO_DATA'
  onestep = pick(1)
  addendum = bool(onestep[0]['pcorr_de'] >= onestep[0]['pcorr_dp']) \
      if onestep else None
  return dict(label=label, checkpoints=checkpoints,
              final_verdict=final_verdict, addendum_pass=addendum,
              addendum_horizon=1)


# --------------------------------------------------------------------------
# Memo.
# --------------------------------------------------------------------------

def render_memo(result):
  r = result
  cells = r['cells']
  verdicts = [c['final_verdict'] for c in cells]
  if all(v == 'TRACKS_DENSITY' for v in verdicts):
    headline = ('REPLICATES: disagreement tracks training density, '
                'not held-out error (attractor bias present).')
  elif all(v == 'TRACKS_ERROR' for v in verdicts):
    headline = ('DOES NOT REPLICATE: disagreement tracks held-out error '
                'over training density on this stack.')
  else:
    headline = ('MIXED/AMBIGUOUS: see per-cell verdicts; do not aggregate '
                'into a single claim.')
  lines = [
      f'# Latent-UQ Stage 0 evidence memo — {r["created"][:10]}',
      '',
      f'**Replication verdict (Biased Dreams, arXiv 2604.25416): '
      f'{headline}**',
      '',
      '## Protocol',
      '',
      f'- Dumps: {", ".join(f"`{k}`" for k in r["dumps"])}'
      + (' (cross-ensemble members)' if r['cross'] else ''),
      f'- Probe set(s): {", ".join(sorted(r["probeset_ids"]))} '
      f'(frozen, hash-verified at dump time)'
      + ('' if r['held_out'] else
         ' — **NOT HELD-OUT for at least one probed run; error reads are '
         'in-distribution there and must not gate**'),
      f'- Horizons: {r["horizons"]} (open-loop, decoder target space); '
      f'headline read: anchor one-step disagreement vs horizon '
      f'{r["primary_horizon"]} error; addendum read: anchor one-step '
      f'disagreement vs one-step (h=1) error (pre-registered, App. A.4 '
      f'item 5)',
      f'- Density proxy: mean kNN distance of anchor posterior means '
      f'against training-buffer encodings (larger = sparser); '
      f'bias signature = positive partials with disagreement',
      f'- SEs: stream-clustered bootstrap, n_boot={r["n_boot"]}; verdict '
      f'threshold {VERDICT_SIGMA} SE on '
      f'Delta = pcorr(D,E|rho) − pcorr(D,rho|E)',
      '',
  ]
  for cell in cells:
    lines += [f'## Cell: {cell["label"]}', '']
    lines.append('| checkpoint | kind | h | rho(D,E) | rho(D,rho) | '
                 'pcorr(D,E|rho) | pcorr(D,rho|E) | Delta (SE) | verdict |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for ck in cell['checkpoints']:
      step = ck['exact_step'] if ck['exact_step'] is not None else ck['name']
      for row in ck['rows']:
        lines.append(
            f"| {step} | {row['kind']} | {row['horizon']}"
            f" | {row['rho_de']:+.3f} ({row['rho_de_se']:.3f})"
            f" | {row['rho_dp']:+.3f} ({row['rho_dp_se']:.3f})"
            f" | {row['pcorr_de']:+.3f} | {row['pcorr_dp']:+.3f}"
            f" | {row['delta']:+.3f} ({row['delta_se']:.3f})"
            f" | {row['verdict']} |")
    lines += [
        '',
        f'Final-checkpoint verdict (anchor, h={r["primary_horizon"]}): '
        f'**{cell["final_verdict"]}**',
        f'Stage-0 calibration addendum (one-step read per EVPI note App. '
        f'A.4 item 5 — pass iff pcorr(D,E|rho) >= pcorr(D,rho|E) at the '
        f'final checkpoint, anchor h=1; the App. A.7 gate criterion '
        f'additionally applies the dose adjustment, evaluated in the D0 '
        f'pipeline, not here): '
        f'**{"PASS" if cell["addendum_pass"] else "FAIL" if cell["addendum_pass"] is not None else "NO DATA"}**',
        '',
    ]
  lines += [
      '## Interpretation notes',
      '',
      '- `TRACKS_DENSITY` at a cell = the attractor bias replicates there: '
      'disagreement is explained by training density beyond what error '
      'explains. Route per the brief: Stage 1 fixes become the priority '
      'and the D0 ensemble read is suspect.',
      '- Per-window reads are never reported; all claims above are '
      'aggregates with stream-clustered uncertainty.',
      '- kNN density in high-dim latent spaces is itself a proxy '
      '(brief, open questions); a TRACKS_DENSITY verdict should be '
      'sanity-checked against a second k before externalizing.',
      '',
      '## Files',
      '',
      f'- analysis: `{r["output"]}/analysis.json`',
      f'- command: `{r["command"]}`',
      f'- git commit: `{r["git_commit"]}`',
      '',
  ]
  return '\n'.join(lines)


def git_commit():
  try:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
        stderr=subprocess.DEVNULL).decode().strip()
  except Exception:
    return None


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------

def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--dumps', nargs='+', default=[],
                 help='label=dump_dir entries (probing/latent_uq.py outputs).')
  p.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 15])
  p.add_argument('--primary_horizon', type=int, default=5,
                 help='Horizon for the headline verdict (clamped). The '
                      'calibration addendum always reads the pre-registered '
                      'one-step (h=1) anchor row, independent of this flag.')
  p.add_argument('--cross', action='store_true',
                 help='Treat all dumps as members of one cross-seed/refit '
                      'ensemble on the same probe set.')
  p.add_argument('--cell', default='cross',
                 help='Cell label for --cross mode.')
  p.add_argument('--n_boot', type=int, default=300)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--output', default='',
                 help='Default artifacts/latent_uq_stage0/analysis_<date>.')
  p.add_argument('--selfcheck', action='store_true',
                 help='Synthetic end-to-end self-check and exit.')
  return p.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  if args.selfcheck:
    selfcheck()
    return
  assert args.dumps, 'Need at least one label=dump_dir entry.'
  named = []
  for item in args.dumps:
    label, _, directory = item.rpartition('=')
    named.append((label or os.path.basename(os.path.normpath(directory)),
                  directory))
  dumps = {label: load_dump(directory) for label, directory in named}

  out_dir = args.output or os.path.join(
      str(REPO), 'artifacts', 'latent_uq_stage0',
      'analysis_' + datetime.date.today().strftime('%Y%m%d'))
  os.makedirs(out_dir, exist_ok=True)

  eval_steps = min(d['index']['eval_steps'] for d in dumps.values())
  horizons = clamp_horizons(args.horizons, eval_steps)
  if 1 not in horizons:
    print('NOTE: adding horizon 1 (required for the pre-registered one-step '
          'addendum read).')
    horizons = [1] + horizons
  primary = args.primary_horizon if args.primary_horizon in horizons else \
      horizons[-1]
  if primary != args.primary_horizon:
    print(f'WARNING: primary horizon {args.primary_horizon} unavailable; '
          f'using {primary}.')

  cells = []
  if args.cross:
    assert len(dumps) >= 2, '--cross needs >= 2 member dumps'
    shas = {d['index']['probeset_sha256'] for d in dumps.values()}
    assert len(shas) == 1, (
        f'--cross members must share one frozen probe set, got {shas}')
    counts = {len(d['checkpoints']) for d in dumps.values()}
    assert len(counts) == 1, (
        f'--cross members must have equal checkpoint counts, got {counts}')
    members = list(dumps.values())
    ckpt_rows = []
    for i, entry in enumerate(members[0]['checkpoints']):
      milestones = {m['checkpoints'][i]['milestone'] for m in members}
      if len(milestones) > 1:
        print(f'WARNING: cross members disagree on milestone at position '
              f'{i}: {milestones}; aligning positionally.')
      npzs = [m['checkpoints'][i]['npz'] for m in members]
      clusters = clusters_of(npzs[0])
      rows = []
      for kind, h, D, E, rho in cross_signals(npzs, horizons):
        rows.append(dict(kind=kind, horizon=h, **row_stats(
            D, E, rho, clusters, args.n_boot, args.seed + 13 * i + h)))
      ckpt_rows.append((entry, rows))
    cells.append(analyze_cell(args.cell, ckpt_rows, primary))
  else:
    for label, dump in dumps.items():
      if not dump['index'].get('has_disag', False):
        print(f'NOTE: {label}: no disag head in dump; skipped (use --cross '
              f'over seeds/refits for this cell).')
        continue
      ckpt_rows = []
      for i, entry in enumerate(dump['checkpoints']):
        npz = entry['npz']
        clusters = clusters_of(npz)
        rows = []
        for kind, h, D, E, rho in within_signals(npz, horizons):
          rows.append(dict(kind=kind, horizon=h, **row_stats(
              D, E, rho, clusters, args.n_boot, args.seed + 13 * i + h)))
        ckpt_rows.append((entry, rows))
      cells.append(analyze_cell(label, ckpt_rows, primary))
  assert cells, 'No analyzable cells (no disag heads and not --cross).'

  result = dict(
      created=datetime.datetime.now().isoformat(timespec='seconds'),
      command=' '.join(sys.argv), git_commit=git_commit(),
      dumps={label: d['dir'] for label, d in dumps.items()},
      probeset_ids=sorted({d['index']['probeset_id']
                           for d in dumps.values()}),
      held_out=all(d['index'].get('held_out', True) for d in dumps.values()),
      cross=args.cross, horizons=horizons, primary_horizon=primary,
      n_boot=args.n_boot, verdict_sigma=VERDICT_SIGMA,
      output=os.path.abspath(out_dir), cells=cells)
  with open(os.path.join(out_dir, 'analysis.json'), 'w') as f:
    json.dump(result, f, indent=2)
  memo = render_memo(result)
  with open(os.path.join(out_dir, 'memo.md'), 'w') as f:
    f.write(memo)
  print(f'\nWrote {out_dir}/analysis.json and {out_dir}/memo.md')
  for cell in cells:
    print(f'  {cell["label"]}: verdict={cell["final_verdict"]}, '
          f'addendum={"PASS" if cell["addendum_pass"] else "FAIL"}')


# --------------------------------------------------------------------------
# Selfcheck.
# --------------------------------------------------------------------------

def _write_fake_dump(root, scenario, rng, S=400, H=16, n_ckpt=3, members=1):
  """Synthetic dumps in the real layout; returns list of dump dirs.

  Latent anchors come from a 2-Gaussian mixture; density_knn uses the real
  kNN helper against a reference from the same mixture. Errors correlate
  moderately with sparsity. Scenario 'biased' ties disagreement to density,
  'calibrated' ties it to error (exp keeps signals positive without
  breaking ranks). Cross members' predictions spread more in sparse
  regions, so cross-disagreement is density-tied regardless of scenario.
  """
  def standardize(v):
    return (v - v.mean()) / (v.std() + 1e-9)

  anchors = np.concatenate([
      rng.normal(0, 1, (S // 2, 8)), rng.normal(3, 2, (S - S // 2, 8))], 0)
  ref = np.concatenate([
      rng.normal(0, 1, (2000, 8)), rng.normal(3, 2, (48, 8))], 0)
  rho = knn_distance(anchors, ref, k=10)
  z_rho = standardize(rho)
  streams = rng.integers(0, 40, S)
  labels = np.asarray(['p2e' if s % 2 else 'apt' for s in streams])

  per_ckpt = []
  for c in range(n_ckpt):
    err1 = 0.5 * z_rho + rng.normal(0, 1, S)
    z_err = standardize(err1)
    if scenario == 'biased':
      d1 = 1.2 * z_rho + 0.1 * z_err + 0.3 * rng.normal(0, 1, S)
    else:
      d1 = 1.2 * z_err + 0.1 * z_rho + 0.3 * rng.normal(0, 1, S)
    err_steps = np.exp(err1)[:, None] * np.linspace(1, 2, H)[None]
    err_steps = err_steps * np.exp(0.1 * rng.normal(0, 1, (S, H)))
    disag_steps = np.exp(0.5 * d1)[:, None] * np.linspace(1, 1.5, H)[None]
    disag_steps = disag_steps * np.exp(0.05 * rng.normal(0, 1, (S, H)))
    shared_pred = rng.normal(0, 1, (S, H, 4))
    per_ckpt.append((err_steps, disag_steps, shared_pred))

  dirs = []
  for m in range(members):
    d = os.path.join(root, f'member{m}')
    dirs.append(d)
    index = dict(
        run_logdir=f'/fake/run{m}', probeset='/fake/probeset',
        probeset_id='fake_v1_deadbeef', probeset_sha256='0' * 64,
        burn_in=8, eval_steps=H, ref_replay='/fake/replay',
        ref_windows=len(ref), knn=10, dec_symlog=True,
        has_disag=(members == 1), checkpoints=[])
    for c, (err_steps, disag_steps, shared_pred) in enumerate(per_ckpt):
      feats = dict(
          err_steps=err_steps.astype(np.float32),
          density_knn=rho.astype(np.float32),
          source_label=labels,
          stream_start=np.stack([streams, np.zeros(S, np.int64)], -1))
      if members == 1:
        feats['disag_steps'] = disag_steps.astype(np.float32)
      else:
        # Member noise scaled by sparsity: var-across-members tracks rho.
        noise = rng.normal(0, 1, (S, H, 4)) * (
            1 + 2 * np.maximum(z_rho, 0))[:, None, None]
        feats['pred_obs'] = (shared_pred + 0.3 * noise).astype(np.float32)
        feats['true_obs'] = shared_pred.astype(np.float32)
      name = f'step{(c + 1) * 1000:012d}'
      step_dir = os.path.join(d, name)
      os.makedirs(step_dir, exist_ok=True)
      np.savez_compressed(os.path.join(step_dir, 'uq.npz'), **feats)
      index['checkpoints'].append(dict(
          name=name, exact_step=(c + 1) * 1000, milestone=(c + 1) * 1000,
          path=os.path.join(name, 'uq.npz')))
    with open(os.path.join(d, 'index.json'), 'w') as f:
      json.dump(index, f, indent=2)
  return dirs


def selfcheck():
  """Synthetic end-to-end: verdicts, addendum, cross path, memo render."""
  import tempfile
  tmp = tempfile.mkdtemp(prefix='latent_uq_selfcheck_')
  rng = np.random.default_rng(20260703)

  # 1. Helper sanity: kNN distance is larger for points far from the
  #    reference mass, and symlog matches its definition.
  ref = rng.normal(0, 1, (500, 4))
  near, far = np.zeros((1, 4)), np.full((1, 4), 8.0)
  dn = knn_distance(near, ref, 5)[0]
  df = knn_distance(far, ref, 5)[0]
  assert df > 3 * dn, (dn, df)
  x = np.asarray([-3.0, 0.0, 5.0])
  assert np.allclose(np_symlog(x), np.sign(x) * np.log1p(np.abs(x)))

  # 2. Partial-correlation direction on constructed data.
  z = rng.normal(0, 1, 2000)
  e = 0.5 * z + rng.normal(0, 1, 2000)
  d_bias = 1.2 * z + 0.1 * e + 0.3 * rng.normal(0, 1, 2000)
  assert partial_spearman(d_bias, z, e) > partial_spearman(d_bias, e, z)

  # 3. Within-checkpoint path, biased scenario -> TRACKS_DENSITY + FAIL.
  biased = _write_fake_dump(
      os.path.join(tmp, 'biased'), 'biased', rng)[0]
  out1 = os.path.join(tmp, 'out_biased')
  main(['--dumps', f'cellA={biased}', '--horizons', '1', '5', '15',
        '--n_boot', '120', '--output', out1])
  with open(os.path.join(out1, 'analysis.json')) as f:
    res = json.load(f)
  cell = res['cells'][0]
  assert cell['final_verdict'] == 'TRACKS_DENSITY', cell['final_verdict']
  assert cell['addendum_pass'] is False
  assert res['horizons'] == [1, 5, 15]
  memo = open(os.path.join(out1, 'memo.md')).read()
  assert 'REPLICATES: disagreement tracks training density' in memo
  assert 'FAIL' in memo

  # 4. Calibrated scenario -> TRACKS_ERROR + PASS.
  calib = _write_fake_dump(
      os.path.join(tmp, 'calib'), 'calibrated', rng)[0]
  out2 = os.path.join(tmp, 'out_calib')
  main(['--dumps', f'cellB={calib}', '--horizons', '1', '5', '15',
        '--n_boot', '120', '--output', out2])
  with open(os.path.join(out2, 'analysis.json')) as f:
    res = json.load(f)
  assert res['cells'][0]['final_verdict'] == 'TRACKS_ERROR'
  assert res['cells'][0]['addendum_pass'] is True

  # 5. Cross-member path (members' spread scales with sparsity ->
  #    density-tied cross-disagreement) + horizon clamping.
  members = _write_fake_dump(
      os.path.join(tmp, 'cross'), 'biased', rng, members=3)
  out3 = os.path.join(tmp, 'out_cross')
  main(['--cross', '--cell', 'xseed',
        '--dumps'] + [f'm{i}={d}' for i, d in enumerate(members)] +
       ['--horizons', '1', '5', '15', '32', '--n_boot', '120',
        '--output', out3])
  with open(os.path.join(out3, 'analysis.json')) as f:
    res = json.load(f)
  assert res['horizons'] == [1, 5, 15], res['horizons']
  cell = res['cells'][0]
  anchor5 = [r for c in cell['checkpoints'] for r in c['rows']
             if r['kind'] == 'anchor' and r['horizon'] == 5]
  assert all(r['rho_dp'] > 0.3 for r in anchor5), anchor5
  assert cell['final_verdict'] == 'TRACKS_DENSITY', cell['final_verdict']

  print('latent_uq_analysis selfcheck: PASS')


if __name__ == '__main__':
  main()
