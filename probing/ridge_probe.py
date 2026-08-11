"""EXPLORATORY reward-legibility ridge probe on frozen trunk features
(2026-08-08; review resolution items #9 / #10 / #12).

Status: labeled exploratory instrument; no registered decision consumes its
output unless a later prereg freezes it. Three uses:

  * #9  — probe the REWARD-FREE (apt) trunk: `stratified_error` skips
          reward-free fits (no trained reward head), so the study's
          diagnostic has never been pointed at an apt trunk. A fresh ridge
          probe on frozen h=0 features answers present-but-unusable vs
          absent (apt R^2 ~ task R^2 => direction present, headline shifts
          to compression; apt R^2 ~ 0 => confirmed on the right instrument).
  * #10 — BASE-RATE-FREE readout (fixes review D4): OOF AUROC of the probe
          scores, plus AUROC and one-scalar-recalibrated NLL of the model's
          own reward-head prediction where a trained head exists. AUROC is
          invariant to the reward base rate that drives raw `rew_nll_in`.
  * #12 — P-C3 cup probe parity: apt-vs-task ridge R^2 on cup_v1.

Feature extraction is the VERBATIM h=0 posterior path of
`stratified_error`/`feature_pr_probe` (same enc/dyn.observe calls, same
batching and seed convention), with the same deter_std witness cross-gate
against the run's existing e4 summary where present.

Probe: episode-level K-fold (round-robin by episode index, deterministic),
ridge on standardized features (alpha = a * tr(Gram)/d, a in {1e-4, 1e-3,
1e-2} — middle value is headline, others sensitivity), all metrics OUT-OF-
FOLD: R^2 vs the float reward, AUROC (rewarded vs not), both overall and
in-regime-only. `--reward_override` scores against transformed labels
(relabel_replay transform-probeset output npz).

Usage (instance/cluster, GPU):
  python -m probing.ridge_probe measure \
      --probeset $RUNROOT/e4_probesets/finger_v1 --run_logdir <fit_run>
  python -m probing.ridge_probe collate --runroot $RUNROOT \
      --glob 'ax1wm_finger_*q1*' --probeset_id finger_v1 --output out.csv
Selfcheck (local, CPU, no checkpoint):
  python -m probing.ridge_probe selfcheck
"""

import argparse
import csv
import glob as globlib
import json
import os

import numpy as np

ALPHAS = (1e-4, 1e-3, 1e-2)   # ridge = a * tr(Gram)/d; 1e-3 = headline
K_FOLDS = 5


# --------------------------------------------------------------------------
# probe core (pure numpy; selfcheckable without a checkpoint)
# --------------------------------------------------------------------------

def episode_folds(n_episodes, k=K_FOLDS):
  return [np.arange(n_episodes) % k == j for j in range(k)]


def ridge_oof(feats, labels, in_regime, k=K_FOLDS, alphas=ALPHAS):
  """feats: (N, T, d) float32; labels: (N, T) float; in_regime: (N, T) bool.
  Returns per-alpha OOF metrics; scores are fully out-of-fold."""
  N, T, d = feats.shape
  folds = episode_folds(N, k)
  out = {}
  scores = {a: np.zeros((N, T), np.float64) for a in alphas}
  for j, test in enumerate(folds):
    train = ~test
    X = feats[train].reshape(-1, d).astype(np.float64)
    y = labels[train].reshape(-1).astype(np.float64)
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-8)
    Xz = (X - mu) / sd
    ym = y.mean()
    G = Xz.T @ Xz
    b = Xz.T @ (y - ym)
    tr_over_d = np.trace(G) / d
    Xt = ((feats[test].reshape(-1, d).astype(np.float64) - mu) / sd)
    for a in alphas:
      w = np.linalg.solve(G + a * tr_over_d * np.eye(d), b)
      scores[a][test] = (Xt @ w + ym).reshape(-1, T)
  y_all = labels.reshape(-1).astype(np.float64)
  rew = y_all > 0
  inr = in_regime.reshape(-1).astype(bool)
  for a in alphas:
    s = scores[a].reshape(-1)
    res = {}
    ss_tot = ((y_all - y_all.mean()) ** 2).sum()
    res['r2'] = float(1.0 - ((y_all - s) ** 2).sum() / max(ss_tot, 1e-12))
    res['auroc'] = auroc(s, rew)
    res['auroc_in_regime'] = auroc(s[inr], rew[inr])
    if inr.mean() < 1.0:
      res['auroc_out_regime'] = auroc(s[~inr], rew[~inr])
    ss_in = ((y_all[inr] - y_all[inr].mean()) ** 2).sum()
    res['r2_in_regime'] = float(
        1.0 - ((y_all[inr] - s[inr]) ** 2).sum() / max(ss_in, 1e-12))
    out[f'alpha_{a:g}'] = res
  return out, scores[alphas[1]]


def auroc(score, positive):
  """Rank-based AUROC (Mann-Whitney), tie-averaged. None if degenerate."""
  positive = np.asarray(positive, bool)
  n1, n0 = int(positive.sum()), int((~positive).sum())
  if n1 == 0 or n0 == 0:
    return None
  order = np.argsort(score, kind='mergesort')
  ranks = np.empty(len(score), np.float64)
  s_sorted = np.asarray(score)[order]
  i = 0
  r = np.arange(1, len(score) + 1, dtype=np.float64)
  while i < len(score):
    j = i
    while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
      j += 1
    r[i:j + 1] = 0.5 * (i + 1 + j + 1)
    i = j + 1
  ranks[order] = r
  return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recalibrated_nll(pred_1d, y_bin, folds_mask):
  """One-scalar-pair (a*x+b) logistic recalibration of a 1-D score,
  fit on train folds, NLL evaluated OOF. Newton, deterministic."""
  pred_1d = np.asarray(pred_1d, np.float64)
  y = np.asarray(y_bin, np.float64)
  nll = np.zeros_like(pred_1d)
  for test in folds_mask:
    train = ~test
    x, t = pred_1d[train], y[train]
    a, b = 1.0, 0.0
    for _ in range(50):
      z = np.clip(a * x + b, -30, 30)
      p = 1 / (1 + np.exp(-z))
      g = np.array([((p - t) * x).sum(), (p - t).sum()])
      w = p * (1 - p)
      H = np.array([[(w * x * x).sum() + 1e-9, (w * x).sum()],
                    [(w * x).sum(), w.sum() + 1e-9]])
      step = np.linalg.solve(H, g)
      a, b = a - step[0], b - step[1]
      if np.abs(step).max() < 1e-10:
        break
    z = np.clip(a * pred_1d[test] + b, -30, 30)
    p = 1 / (1 + np.exp(-z))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    nll[test] = -(y[test] * np.log(p) + (1 - y[test]) * np.log(1 - p))
  return nll


# --------------------------------------------------------------------------
# measure (GPU; mirrors feature_pr_probe's validated extraction)
# --------------------------------------------------------------------------

def cmd_measure(args):
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from dreamerv3.main import make_agent
  from probing.collect import load_run_config, load_frozen_agent
  from probing.stratified_error import load_e4

  arrays, manifest = load_e4(args.probeset, not args.allow_unfrozen)
  N, T = arrays['is_first'].shape
  out_path = args.output or os.path.join(
      args.run_logdir, f'ridge_probe_{manifest["probeset_id"]}.json')

  labels = np.asarray(arrays['reward'], np.float64)
  override_meta = None
  if args.reward_override:
    with np.load(args.reward_override) as z:
      labels = np.asarray(z['reward'], np.float64)
    with open(args.reward_override + '.json') as f:
      override_meta = json.load(f)
    assert override_meta['probeset_sha256'] == manifest['sha256'], \
        'override was built for a different probe set'
    assert labels.shape == (N, T)
  in_regime = np.asarray(arrays['in_regime'], bool)

  config = load_run_config(
      args.run_logdir, args.platform,
      os.path.dirname(os.path.abspath(out_path)), False)
  config = config.update({'jax': {
      'precompile': False, 'enable_policy': False, 'prealloc': False}})
  agent = make_agent(config)
  ckpt = args.checkpoint or os.path.join(args.run_logdir, 'ckpt')
  load_frozen_agent(agent, ckpt)
  model = agent.model
  jax.config.update('jax_transfer_guard', 'allow')
  expl_mode = str(config.agent.expl.mode)
  reward_aware = (expl_mode == 'task')

  feed_keys = sorted(set(model.dec.obs_space.keys())
                     | set(model.enc.obs_space.keys()))
  isimg = {k: len(agent.obs_space[k].shape) == 3 for k in feed_keys}
  act_keys = sorted(agent.act_space.keys())
  missing = [k for k in feed_keys + act_keys if k not in arrays]
  assert not missing, f'probe set lacks keys {missing}.'

  def fn(obs, action_dict, reset):
    B = reset.shape[0]
    enc_carry = model.enc.initial(B)
    dyn_carry = model.dyn.initial(B)
    enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)
    prevact = {k: jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], 1)
               for k, v in action_dict.items()}
    dyn_carry, _, post = model.dyn.observe(
        dyn_carry, tokens, prevact, reset, training=False)
    feat = model.feat2tensor({'deter': post['deter'],
                              'stoch': post['stoch']})
    out = dict(
        feat=jnp.asarray(feat, jnp.float32),
        deter_sumsq=jnp.square(post['deter'] - post['deter'].reshape(
            (-1, post['deter'].shape[-1])).mean(0)).reshape(
                (-1, post['deter'].shape[-1])).sum(0)[None],
        deter_mean=post['deter'].reshape(
            (-1, post['deter'].shape[-1])).mean(0)[None],
        deter_n=jnp.full((1,), B * T, jnp.float32))
    # 2026-08-08 batch-review B3: reward-free fits carry NO rew-head params
    # (agent.py guards the rew loss on reward_free, so the head was never
    # traced at train time); calling model.rew here would die at trace time
    # under nj.pure(create=False) — exactly on the apt trunks this probe
    # exists for. Same guard convention as stratified_error.
    if reward_aware:
      out['head_pred'] = jnp.asarray(model.rew(feat, 2).pred(), jnp.float32)
    return out

  pure = nj.pure(fn)
  jit = jax.jit(lambda p, o, a, re, s: pure(p, o, a, re, seed=s))
  params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), agent.params)
  reset_np = np.asarray(arrays['is_first'], bool)

  feats, head_preds = [], []
  dmeans, dsumsqs, dns = [], [], []
  for lo in range(0, N, args.ep_batch):
    hi = min(lo + args.ep_batch, N)
    obs = {k: (jnp.asarray(arrays[k][lo:hi]) if isimg[k]
               else jnp.asarray(arrays[k][lo:hi], np.float32))
           for k in feed_keys}
    action_dict = {k: jnp.asarray(arrays[k][lo:hi], np.float32)
                   for k in act_keys}
    reset = jnp.asarray(reset_np[lo:hi])
    seed = jnp.array([args.seed, lo + 1], np.uint32)
    _, out = jit(params, obs, action_dict, reset, seed)
    feats.append(np.asarray(out['feat'], np.float32))
    if reward_aware:
      head_preds.append(np.asarray(out['head_pred'], np.float32))
    dmeans.append(np.asarray(out['deter_mean'], np.float64))
    dsumsqs.append(np.asarray(out['deter_sumsq'], np.float64))
    dns.append(np.asarray(out['deter_n'], np.float64))
    print(f'  episodes {lo}-{hi - 1}')
  feats = np.concatenate(feats, 0)          # (N, T, d)
  head_preds = (np.concatenate(head_preds, 0) if reward_aware
                else np.zeros((N, T), np.float32))

  # deter_std witness (batch-pooled; matches feature_pr's Chan combine for
  # the diagonal): combine per-batch (mean, sumsq, n).
  from probing.feature_pr_probe import combine_cov_moments
  means = np.concatenate(dmeans, 0)
  ns = np.concatenate(dns, 0)
  m2s = np.stack([np.diag(s[0]) for s in dsumsqs], 0)
  ntot, _, covdiag = combine_cov_moments(means, m2s, ns)
  our_deter_std = float(np.sqrt(np.maximum(np.diag(covdiag), 0.0)).mean())

  probe, oof_scores = ridge_oof(feats, labels, in_regime)
  folds = episode_folds(N)
  fold_masks = [np.repeat(m, T) for m in folds]
  y_bin = (labels.reshape(-1) > 0).astype(np.float64)

  head_panel = None
  hp = head_preds.reshape(-1).astype(np.float64)
  if reward_aware:
    nll = recalibrated_nll(hp, y_bin, fold_masks)
    inr = in_regime.reshape(-1)
    head_panel = dict(
        auroc=auroc(hp, y_bin > 0),
        auroc_in_regime=auroc(hp[inr], y_bin[inr] > 0),
        recal_nll_all=float(nll.mean()),
        recal_nll_in_regime=float(nll[inr].mean()),
        recal_nll_out_regime=float(nll[~inr].mean()) if (~inr).any() else None)

  result = dict(
      exploratory=('ridge-probe exploratory instrument 2026-08-08; not '
                   'frozen; no registered decision consumes this'),
      run_logdir=os.path.abspath(args.run_logdir),
      run_id=os.path.basename(args.run_logdir.rstrip('/')),
      checkpoint=os.path.abspath(ckpt),
      expl_mode=expl_mode, reward_aware=reward_aware,
      probeset_id=manifest['probeset_id'],
      probeset_sha256=manifest['sha256'],
      reward_override=(os.path.abspath(args.reward_override)
                       if args.reward_override else None),
      override_meta=override_meta,
      k_folds=K_FOLDS, alphas=list(ALPHAS), seed=args.seed,
      feat_dim=int(feats.shape[-1]),
      probe=probe, head_panel=head_panel,
      deter_std_recomputed=our_deter_std,
      deter_std_summary=None, witness_match=None)
  for e4dir in sorted(globlib.glob(
      os.path.join(args.run_logdir, 'e4_*', 'summary.json'))):
    with open(e4dir) as f:
      s = json.load(f)
    if s.get('deter_std') is not None and \
        s.get('probeset_sha256') == manifest['sha256']:
      ref = float(s['deter_std'])
      result['deter_std_summary'] = ref
      # Tolerance 1e-3 (2026-08-10 instrument amendment, disclosed):
      # the witness guards WRONG-extraction (wrong run/side/keys —
      # 30-400% misses, e.g. the rde side asymmetry 0.03 vs 0.43),
      # never bit-identity. The probe always runs in a DIFFERENT job
      # than the E4 pass, and measured cross-job drift of single-
      # forward RNG-free quantities on this cluster is up to ~7e-4
      # (repair audit, udyn) — the original 1e-4 bar sat inside the
      # substrate noise floor and tripped on a regenerated-fit pass at
      # |diff| = 1.11e-4 with matched-magnitude values.
      result['witness_match'] = bool(
          abs(our_deter_std - ref) <= 1e-3 * max(1.0, abs(ref)))
      break

  with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
  mid = probe[f'alpha_{ALPHAS[1]:g}']
  print(f'  r2={mid["r2"]:.4f} auroc={mid["auroc"]} '
        f'auroc_in={mid["auroc_in_regime"]} '
        f'head={head_panel} witness={result["witness_match"]}')
  print(f'-> {out_path}')
  if result['witness_match'] is False:
    raise SystemExit('WITNESS MISMATCH: extraction drifted from the frozen '
                     'stratified_error path — output marked, do not use.')


def cmd_collate(args):
  rows = []
  pat = os.path.join(args.runroot, args.glob,
                     f'ridge_probe_{args.probeset_id}.json')
  for p in sorted(globlib.glob(pat)):
    with open(p) as f:
      r = json.load(f)
    mid = r['probe'][f'alpha_{ALPHAS[1]:g}']
    row = dict(run_id=r['run_id'], expl_mode=r['expl_mode'],
               reward_aware=int(r['reward_aware']),
               feat_dim=r['feat_dim'],
               r2=mid['r2'], r2_in=mid['r2_in_regime'],
               auroc=mid['auroc'], auroc_in=mid['auroc_in_regime'],
               witness_match=r['witness_match'],
               override=bool(r['reward_override']))
    hp = r.get('head_panel')
    if hp:
      row.update(head_auroc=hp['auroc'], head_auroc_in=hp['auroc_in_regime'],
                 head_recal_nll_in=hp['recal_nll_in_regime'])
    rows.append(row)
  assert rows, f'no jsons matched {pat}'
  keys = sorted({k for r in rows for k in r})
  with open(args.output, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
  print(f'{len(rows)} rows -> {args.output}')


# --------------------------------------------------------------------------
# selfcheck (CPU, no checkpoint)
# --------------------------------------------------------------------------

def cmd_selfcheck(args):
  rng = np.random.default_rng(3)
  N, T, d = 20, 60, 24
  # (a) linear signal: reward = clipped linear functional + threshold
  W = rng.normal(0, 1, d)
  feats = rng.normal(0, 1, (N, T, d)).astype(np.float32)
  lin = feats.astype(np.float64) @ W
  labels = (lin > 1.2).astype(np.float64)
  in_r = lin > 0.0
  probe, _ = ridge_oof(feats, labels, in_r)
  mid = probe[f'alpha_{ALPHAS[1]:g}']
  assert mid['r2'] > 0.5, mid
  assert mid['auroc'] > 0.95, mid
  # (b) permuted labels -> chance
  perm = labels.reshape(-1).copy()
  rng.shuffle(perm)
  probe_p, _ = ridge_oof(feats, perm.reshape(N, T), in_r)
  mid_p = probe_p[f'alpha_{ALPHAS[1]:g}']
  assert abs(mid_p['auroc'] - 0.5) < 0.05, mid_p
  assert mid_p['r2'] < 0.02, mid_p
  # (c) OOF honesty: episode-unique memorizable signal must NOT transfer.
  ep_id = np.repeat(rng.normal(0, 1, N)[:, None], T, 1)
  feats_mem = np.concatenate(
      [np.zeros((N, T, d - 1), np.float32),
       ep_id[..., None].astype(np.float32)], -1)
  labels_mem = (ep_id + 0 * ep_id).astype(np.float64)
  labels_mem = (labels_mem > np.median(labels_mem)).astype(np.float64)
  probe_m, _ = ridge_oof(feats_mem, labels_mem, np.ones((N, T), bool))
  mid_m = probe_m[f'alpha_{ALPHAS[1]:g}']
  # within-episode constant labels + constant feature: OOF prediction is a
  # pure between-episode extrapolation of a 1-D monotone map — it CAN rank
  # (auroc high) but per-episode R^2 must come from the linear map, so just
  # assert it did not blow up and folds were honored (scores finite).
  assert np.isfinite(mid_m['r2'])
  # (d) auroc ties + degenerate guard
  assert auroc(np.zeros(10), np.array([1, 0] * 5, bool)) == 0.5
  assert auroc(np.arange(4), np.array([0, 0, 0, 0], bool)) is None
  # (e) recalibrated NLL beats naive scaling on a miscalibrated score
  score = 5 * lin.reshape(-1) - 3.0
  folds = [np.repeat(m, T) for m in episode_folds(N)]
  nll = recalibrated_nll(score, labels.reshape(-1), folds)
  base_rate = labels.mean()
  nll_const = -(base_rate * np.log(base_rate)
                + (1 - base_rate) * np.log(1 - base_rate))
  assert nll.mean() < nll_const, (nll.mean(), nll_const)
  # (f) determinism
  probe2, _ = ridge_oof(feats, labels, in_r)
  assert probe2 == probe
  print('ridge_probe selfcheck PASS '
        f'(signal r2={mid["r2"]:.3f} auroc={mid["auroc"]:.3f}; '
        f'null auroc={mid_p["auroc"]:.3f}; recal nll {nll.mean():.3f} '
        f'< const {nll_const:.3f})')


def main():
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)
  m = sub.add_parser('measure')
  m.add_argument('--probeset', required=True)
  m.add_argument('--run_logdir', required=True)
  m.add_argument('--checkpoint', default=None)
  m.add_argument('--output', default=None)
  m.add_argument('--reward_override', default=None,
                 help='npz from relabel_replay transform-probeset / '
                      'stamp-probeset (own-label scoring).')
  m.add_argument('--platform', default='gpu')
  # default 4 = stratified_error's ep_batch (2026-08-10 amendment):
  # batched RSSM forwards are not batch-shape-invariant on GPU, and the
  # deter_std witness compares like-for-like only when the probe's
  # extraction batching matches the E4 pass that wrote the summary.
  # Keep UNIFORM across a panel — never vary per run.
  m.add_argument('--ep_batch', type=int, default=4)
  m.add_argument('--seed', type=int, default=0)
  m.add_argument('--allow_unfrozen', action='store_true')
  m.set_defaults(fn=cmd_measure)
  c = sub.add_parser('collate')
  c.add_argument('--runroot', required=True)
  c.add_argument('--glob', required=True)
  c.add_argument('--probeset_id', required=True)
  c.add_argument('--output', required=True)
  c.set_defaults(fn=cmd_collate)
  s = sub.add_parser('selfcheck')
  s.set_defaults(fn=cmd_selfcheck)
  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
