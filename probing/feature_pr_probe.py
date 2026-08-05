"""EXPLORATORY feature-space breadth probe (post-read; 2026-08-04).

Status: labeled exploratory analysis under the value-blindness policy of
PREREG_rde_pixel_20260802 (post-read analyses allowed, labeled,
verdict-untouchable). NOT a frozen instrument; no registered decision
consumes its output. Motivating question (rde ONE read, PARTIAL-RELIEF):
did removing reconstruction's gradient competition buy reward legibility
by NARROWING the trunk's usable support? Prediction (stated before any
value was computed, in-chat 2026-08-04): rde trunks — especially side1 —
show materially lower feature-space participation ratio than the X2
task arm at similar-or-better reward legibility.

What it measures, per run: the h=0 filtering-posterior features of the
frozen probeset (the EXACT extraction path of
probing/stratified_error.py cmd_measure — same enc/dyn.observe calls,
same batching, same seed convention), accumulated as per-batch CENTERED
second moments (the deter_std witness idiom, matrix form) and combined
in float64. Blocks:
  * deter        — the witness scope (per-dim recurrent state),
  * stoch        — flattened categorical sample,
  * head_input   — model.feat2tensor({'deter','stoch'}): the exact
                   space the reward/value heads consume.
For each block: covariance eigenspectrum, participation ratio
PR = (tr C)^2 / tr(C^2), trace, top-10 eigenvalue shares.

Built-in cross-instrument gate: deter_std is recomputed from this
script's own deter moments and compared (rtol 1e-4) against the run's
existing e4 summary deter_std where present (rde + fpxpx-calibration
runs carry it). A mismatch means the extraction drifted from the
frozen witness path — the run's output is marked UNVERIFIED and the
process exits nonzero at the end.

Usage (instance, GPU):
  python -m probing.feature_pr_probe measure \
      --probeset $RUNROOT/e4_probesets/fingerpx_v1 --run_logdir <run>
  python -m probing.feature_pr_probe collate \
      --runroot $RUNROOT --glob 'ax1wm_finger_*pxq1m*' \
      --probeset_id fingerpx_v1 --output $RUNROOT/feature_pr_fingerpx.csv
Selfcheck (local, CPU, no checkpoint needed):
  python -m probing.feature_pr_probe selfcheck
"""

import argparse
import glob as globlib
import json
import os

import numpy as np


def combine_cov_moments(means, m2s, ns):
  """Matrix Chan combination of per-batch (mean, centered scatter).

  means: (B, D) per-batch feature means; m2s: (B, D, D) per-batch
  centered scatter matrices sum_i (x-mu_b)(x-mu_b)^T; ns: (B,) counts.
  Returns (n_total, mean (D,), covariance C (D, D)) with C the
  population covariance (divide by n), float64 throughout. Per-batch
  centering keeps f32 device sums cancellation-free (the
  _combine_deter_moments rationale, matrix form).
  """
  means = np.asarray(means, np.float64)
  m2s = np.asarray(m2s, np.float64)
  ns = np.asarray(ns, np.float64).reshape(-1)
  assert means.ndim == 2 and m2s.ndim == 3 and len(ns) == len(means), (
      means.shape, m2s.shape, ns.shape)
  ntot = ns.sum()
  mean = (ns[:, None] * means).sum(0) / ntot
  dev = means - mean
  m2 = m2s.sum(0) + np.einsum('b,bi,bj->ij', ns, dev, dev)
  return float(ntot), mean, m2 / ntot


def spectrum_stats(cov):
  """Eigenspectrum descriptives of a covariance matrix (float64)."""
  ev = np.linalg.eigvalsh(np.asarray(cov, np.float64))
  ev = np.maximum(ev, 0.0)[::-1]
  tr = float(ev.sum())
  tr2 = float((ev ** 2).sum())
  pr = (tr * tr / tr2) if tr2 > 0 else 0.0
  shares = (ev[:10] / tr).tolist() if tr > 0 else [0.0] * min(10, len(ev))
  return dict(dim=int(len(ev)), trace=tr, pr=float(pr),
              top10_shares=[round(float(s), 6) for s in shares],
              spectrum=[float(v) for v in ev])


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
      args.run_logdir, f'feature_pr_{manifest["probeset_id"]}.json')

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

  feed_keys = sorted(set(model.dec.obs_space.keys())
                     | set(model.enc.obs_space.keys()))
  isimg = {k: len(agent.obs_space[k].shape) == 3 for k in feed_keys}
  act_keys = sorted(agent.act_space.keys())
  missing = [k for k in feed_keys + act_keys if k not in arrays]
  assert not missing, f'probe set lacks keys {missing}.'
  print(f'{os.path.basename(args.run_logdir.rstrip("/"))}: '
        f'feature-pr probe, {N}x{T} states')

  def fn(obs, action_dict, reset):
    # Verbatim h=0 posterior extraction (stratified_error cmd_measure).
    B = reset.shape[0]
    enc_carry = model.enc.initial(B)
    dyn_carry = model.dyn.initial(B)
    enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)
    prevact = {k: jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], 1)
               for k, v in action_dict.items()}
    dyn_carry, _, post = model.dyn.observe(
        dyn_carry, tokens, prevact, reset, training=False)
    blocks = {
        'deter': post['deter'].reshape((-1, post['deter'].shape[-1])),
        'stoch': post['stoch'].reshape(
            (-1, int(np.prod(post['stoch'].shape[2:])))),
        'head_input': model.feat2tensor(
            {'deter': post['deter'], 'stoch': post['stoch']}).reshape(
                (B * T, -1)),
    }
    out = {}
    for name, x in blocks.items():
      x = jnp.asarray(x, jnp.float32)
      mu = x.mean(0)
      d = x - mu
      out[f'{name}/mean'] = mu[None]
      out[f'{name}/m2'] = jnp.einsum('ni,nj->ij', d, d)[None]
      out[f'{name}/n'] = jnp.full((1,), x.shape[0], jnp.float32)
    return out

  pure = nj.pure(fn)
  jit = jax.jit(lambda p, o, a, re, s: pure(p, o, a, re, seed=s))
  params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), agent.params)
  reset_np = np.asarray(arrays['is_first'], bool)

  acc = {}
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
    for k, v in out.items():
      acc.setdefault(k, []).append(np.asarray(v, np.float64))
    print(f'  episodes {lo}-{hi - 1}')

  result = dict(
      exploratory=('post-read exploratory probe, 2026-08-04; not a frozen '
                   'instrument; no registered decision consumes this'),
      run_logdir=os.path.abspath(args.run_logdir),
      run_id=os.path.basename(args.run_logdir.rstrip('/')),
      checkpoint=os.path.abspath(ckpt),
      expl_mode=str(config.agent.expl.mode),
      # Old (pre-rde) saved configs lack recon_grad; load_run_config
      # overlays them onto current defaults, so the key always resolves
      # (True for every recon-trained arm — their training truth).
      recon_grad=bool(config.agent.recon_grad),
      probeset_id=manifest['probeset_id'],
      probeset_sha256=manifest['sha256'],
      seed=args.seed, blocks={})
  deter_stats = None
  for name in ('deter', 'stoch', 'head_input'):
    means = np.concatenate(acc[f'{name}/mean'], 0)
    m2s = np.concatenate(acc[f'{name}/m2'], 0).reshape(
        len(means), means.shape[1], means.shape[1])
    ns = np.concatenate(acc[f'{name}/n'], 0)
    ntot, _, cov = combine_cov_moments(means, m2s, ns)
    stats = spectrum_stats(cov)
    stats['n_states'] = int(ntot)
    result['blocks'][name] = stats
    if name == 'deter':
      deter_stats = (cov, ntot)

  # Cross-instrument gate: deter_std from our own moments must match the
  # frozen witness value in the run's e4 summary (where present).
  cov, ntot = deter_stats
  our_deter_std = float(np.sqrt(np.maximum(np.diag(cov), 0.0)).mean())
  result['deter_std_recomputed'] = our_deter_std
  result['deter_std_summary'] = None
  result['witness_match'] = None
  for e4dir in sorted(globlib.glob(
      os.path.join(args.run_logdir, 'e4_*', 'summary.json'))):
    with open(e4dir) as f:
      s = json.load(f)
    if s.get('deter_std') is not None and \
        s.get('probeset_sha256') == manifest['sha256']:
      ref = float(s['deter_std'])
      result['deter_std_summary'] = ref
      result['witness_match'] = bool(
          abs(our_deter_std - ref) <= 1e-4 * max(1.0, abs(ref)))
      break

  with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
  b = result['blocks']
  print(f'  deter pr={b["deter"]["pr"]:.2f} stoch pr={b["stoch"]["pr"]:.2f} '
        f'head_input pr={b["head_input"]["pr"]:.2f} '
        f'deter_std={our_deter_std:.6f} '
        f'witness_match={result["witness_match"]}')
  print(f'-> {out_path}')
  if result['witness_match'] is False:
    raise SystemExit('WITNESS MISMATCH: extraction drifted from the frozen '
                     'stratified_error path — output marked, do not use.')


def cmd_collate(args):
  rows = []
  for d in sorted(globlib.glob(os.path.join(args.runroot, args.glob))):
    path = os.path.join(d, f'feature_pr_{args.probeset_id}.json')
    if not os.path.exists(path):
      continue
    with open(path) as f:
      r = json.load(f)
    row = dict(run_id=r['run_id'], expl_mode=r['expl_mode'],
               recon_grad=r.get('recon_grad'),
               n_states=r['blocks']['deter']['n_states'],
               deter_std=r['deter_std_recomputed'],
               witness_match=r['witness_match'])
    for name, st in r['blocks'].items():
      row[f'pr_{name}'] = st['pr']
      row[f'trace_{name}'] = st['trace']
      row[f'top1_share_{name}'] = st['top10_shares'][0]
    rows.append(row)
  if not rows:
    raise SystemExit(f'no feature_pr_{args.probeset_id}.json under '
                     f'{args.runroot}/{args.glob}')
  import csv
  cols = sorted({c for r in rows for c in r}, key=lambda c: (c != 'run_id', c))
  with open(args.output, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
  print(f'{len(rows)} runs -> {args.output}')


def cmd_selfcheck(args):
  rng = np.random.default_rng(0)

  # 1. Matrix-Chan combination == direct covariance, uneven batches +
  #    large common offset (the cancellation regime), 1e-9 relative.
  x = rng.normal(size=(517, 24)) @ rng.normal(size=(24, 24)) + 1e3
  splits = np.split(x, [200, 310, 480])
  means = np.stack([b.mean(0) for b in splits])
  m2s = np.stack([np.einsum('ni,nj->ij', b - b.mean(0), b - b.mean(0))
                  for b in splits])
  ns = np.array([len(b) for b in splits], np.float64)
  ntot, mean, cov = combine_cov_moments(means, m2s, ns)
  ref_cov = np.cov(x, rowvar=False, bias=True)
  assert ntot == len(x)
  assert np.allclose(mean, x.mean(0), rtol=0, atol=1e-9 * 1e3)
  denom = max(1.0, float(np.abs(ref_cov).max()))
  assert np.abs(cov - ref_cov).max() <= 1e-9 * denom, \
      np.abs(cov - ref_cov).max()

  # 2. PR properties: scale invariance; planted-rank recovery (isotropic
  #    rank-k spectrum -> PR == k); single-direction -> PR == 1.
  st = spectrum_stats(cov)
  st_scaled = spectrum_stats(cov * 37.5)
  assert abs(st['pr'] - st_scaled['pr']) < 1e-9
  for k in (1, 5, 17):
    ev = np.zeros(24); ev[:k] = 2.7
    q, _ = np.linalg.qr(rng.normal(size=(24, 24)))
    planted = spectrum_stats((q * ev) @ q.T)
    assert abs(planted['pr'] - k) < 1e-8, (k, planted['pr'])
  # 3. Anisotropy lowers PR below dim; spectrum sorted descending.
  assert st['pr'] < st['dim']
  assert all(a >= b for a, b in zip(st['spectrum'], st['spectrum'][1:]))
  assert abs(sum(st['spectrum']) - st['trace']) < 1e-9 * max(1.0, st['trace'])

  # 4. deter_std reduction from the same moments matches the scalar
  #    witness combiner (stratified_error._combine_deter_moments).
  from probing.stratified_error import _combine_deter_moments
  m = np.stack([np.stack([b.mean(0), ((b - b.mean(0)) ** 2).sum(0)])
                for b in splits])
  ours = float(np.sqrt(np.maximum(np.diag(cov), 0.0)).mean())
  ref = _combine_deter_moments(m, ns.reshape(-1, 1))
  assert abs(ours - ref) <= 1e-9 * max(1.0, ref), (ours, ref)

  # 5. Determinism of the numpy path.
  ntot2, mean2, cov2 = combine_cov_moments(means, m2s, ns)
  assert ntot2 == ntot and np.array_equal(mean2, mean) \
      and np.array_equal(cov2, cov)

  print('feature_pr_probe selfcheck PASS (matrix-Chan 1e-9; PR scale-inv + '
        'planted-rank exact; deter_std reduction == witness combiner; '
        'deterministic). Measure path validated on-instance by the '
        'per-run witness_match gate.')


def main():
  p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  sub = p.add_subparsers(dest='cmd', required=True)
  m = sub.add_parser('measure')
  m.add_argument('--probeset', required=True)
  m.add_argument('--run_logdir', required=True)
  m.add_argument('--checkpoint', default='')
  m.add_argument('--ep_batch', type=int, default=4)
  m.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  m.add_argument('--allow_unfrozen', action='store_true')
  m.add_argument('--seed', type=int, default=0)
  m.add_argument('--output', default='')
  m.set_defaults(fn=cmd_measure)
  c = sub.add_parser('collate')
  c.add_argument('--runroot', required=True)
  c.add_argument('--glob', required=True)
  c.add_argument('--probeset_id', required=True)
  c.add_argument('--output', required=True)
  c.set_defaults(fn=cmd_collate)
  s = sub.add_parser('selfcheck')
  s.set_defaults(fn=cmd_selfcheck)
  a = p.parse_args()
  a.fn(a)


if __name__ == '__main__':
  main()
