"""Reward-legibility ridge probe on RAW embedding features
(PREREG_lewm_20260812 — the distill-free representation leg).

Mirrors probing/ridge_probe.py EXACTLY at the estimator level (same
ridge_oof: alpha grid, grouped episode folds, standardization, midrank
AUROC — imported, not re-implemented) but takes precomputed per-frame
embeddings (probing/lewm_embed.py probeset sidecar, [N, T, D]) instead
of a Dreamer checkpoint. Output json carries the same
probe['alpha_0.001'] schema so the frozen readers can consume both
sides of a comparison symmetrically.

Runs in the dreamer env (numpy only). Usage:
  python -m probing.embed_probe --emb <.../probeset_emb.npz> \
      --embed_manifest <.../embed_manifest.json> \
      --probeset $RUNROOT/e4_probesets/fingerpx_v1 \
      --run_id lewm_finger_s0_seed1 --output <dir-or-json>
Selfcheck: python -m probing.embed_probe --selfcheck
"""

import argparse
import json
import os

import numpy as np

from probing.ridge_probe import ALPHAS, K_FOLDS, ridge_oof


def run(args):
  from probing.stratified_error import load_e4
  arrays, manifest = load_e4(args.probeset, True)
  N, T = arrays['is_first'].shape
  with np.load(args.emb) as z:
    feats = np.asarray(z['emb'], np.float32)
  assert feats.shape[:2] == (N, T), (feats.shape, (N, T))
  with open(args.embed_manifest) as f:
    eman = json.load(f)
  assert eman['n_episodes'] == N, (eman['n_episodes'], N)
  labels = np.asarray(arrays['reward'], np.float64)
  in_regime = np.asarray(arrays['in_regime'], bool)
  probe, _ = ridge_oof(feats, labels, in_regime)
  result = dict(
      instrument='embed_probe (ridge_probe estimator on raw embeddings)',
      run_id=args.run_id,
      emb=os.path.abspath(args.emb),
      embed_manifest=eman,
      probeset_id=manifest['probeset_id'],
      probeset_sha256=manifest['sha256'],
      k_folds=K_FOLDS, alphas=list(ALPHAS),
      feat_dim=int(feats.shape[-1]),
      probe=probe)
  out = args.output
  if os.path.isdir(out) or not out.endswith('.json'):
    os.makedirs(out, exist_ok=True)
    out = os.path.join(out, f'{args.run_id}.json')
  with open(out, 'w') as f:
    json.dump(result, f, indent=2)
  mid = probe[f'alpha_{ALPHAS[1]:g}']
  print(f'  r2={mid["r2"]:.4f} auroc={mid["auroc"]} -> {out}')


def selfcheck():
  """The estimator itself is ridge_probe's (its own selfcheck covers it);
  here: (a) an embedding that linearly encodes reward scores AUROC ~1,
  (b) pure-noise embeddings score ~0.5, (c) shape/manifest refusals."""
  import shutil
  import tempfile
  rng = np.random.default_rng(0)
  N, T, D = 20, 30, 16
  labels = (rng.random((N, T)) < 0.15).astype(np.float64)
  inr = rng.random((N, T)) < 0.5
  good = np.concatenate(
      [labels[..., None] + 0.05 * rng.normal(size=(N, T, 1)),
       rng.normal(size=(N, T, D - 1))], -1).astype(np.float32)
  noise = rng.normal(size=(N, T, D)).astype(np.float32)
  pg, _ = ridge_oof(good, labels, inr)
  pn, _ = ridge_oof(noise, labels, inr)
  a = f'alpha_{ALPHAS[1]:g}'
  assert pg[a]['auroc'] > 0.95, pg[a]
  assert abs(pn[a]['auroc'] - 0.5) < 0.06, pn[a]

  # refusal legs via a fake probeset dir
  tmp = tempfile.mkdtemp(prefix='embed_probe_selfcheck_')
  try:
    import hashlib
    ps = os.path.join(tmp, 'ps')
    os.makedirs(ps)
    npz = os.path.join(ps, 'probeset_e4.npz')
    np.savez(npz, is_first=np.zeros((N, T), bool), reward=labels,
             in_regime=inr)
    sha = hashlib.sha256(open(npz, 'rb').read()).hexdigest()
    with open(os.path.join(ps, 'manifest.json'), 'w') as f:
      json.dump(dict(sha256=sha, probeset_id='fake_v1'), f)
    with open(os.path.join(ps, 'FROZEN'), 'w') as f:
      f.write(sha)
    emb_path = os.path.join(tmp, 'probeset_emb.npz')
    np.savez(emb_path, emb=good)
    eman = os.path.join(tmp, 'embed_manifest.json')
    with open(eman, 'w') as f:
      json.dump(dict(n_episodes=N, dim=D), f)
    ns = argparse.Namespace(emb=emb_path, embed_manifest=eman,
                            probeset=ps, run_id='fixture',
                            output=tmp)
    run(ns)
    with open(os.path.join(tmp, 'fixture.json')) as f:
      out = json.load(f)
    assert out['probe'][a]['auroc'] > 0.95
    assert out['probeset_sha256'] == sha
    # wrong-shape embedding refused
    np.savez(emb_path, emb=good[:-1])
    try:
      run(ns)
      raise SystemExit('shape mismatch not refused')
    except AssertionError:
      pass
    # manifest count mismatch refused
    np.savez(emb_path, emb=good)
    with open(eman, 'w') as f:
      json.dump(dict(n_episodes=N + 3, dim=D), f)
    try:
      run(ns)
      raise SystemExit('manifest mismatch not refused')
    except AssertionError:
      pass
    print('SELFCHECK PASS')
  finally:
    shutil.rmtree(tmp, ignore_errors=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--selfcheck', action='store_true')
  ap.add_argument('--emb')
  ap.add_argument('--embed_manifest')
  ap.add_argument('--probeset')
  ap.add_argument('--run_id')
  ap.add_argument('--output')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  for req in ('emb', 'embed_manifest', 'probeset', 'run_id', 'output'):
    assert getattr(args, req), f'--{req} required'
  run(args)


if __name__ == '__main__':
  main()
