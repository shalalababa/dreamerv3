"""Shrink-and-perturb donor transform (PREREG_routedrepair_20260820 +
Amendment 1's S&P free arm; descriptive-only by registration).

theta -> LAM * theta + eps, eps ~ N(0, SIGMA^2) per parameter, applied to
every float ndarray in the donor's agent payload (Ash & Adams 2020 form;
single pinned point, no tuning). Deterministic given --seed: keys are
processed in sorted order with one seeded generator.

Writes a RUN-DIR-shaped output the existing harness can consume
unchanged: <output>/{config.yaml (copied from the donor), ckpt/<folder>/
{agent.pkl, done}, ckpt/latest} plus snp_manifest.json (donor ckpt path +
sha256, lam, sigma, seed, per-key stats, output sha256). latest_ckpt()
and --run.from_checkpoint resolve it like any fit run.

No registered instrument is modified; this file is additive. Pure
pickle+numpy — no jax needed.

Usage:
  python -m probing.shrink_perturb --donor $RUNROOT/<donor_run> \
      --output $RUNROOT/<snp_run> [--lam 0.5] [--sigma 0.01] [--seed 0]
Selfcheck: python -m probing.shrink_perturb --selfcheck
"""

import argparse
import hashlib
import json
import os
import pickle
import shutil

import numpy as np

LAM = 0.5
SIGMA = 0.01


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def latest_done_ckpt(run_dir):
  ck = os.path.join(run_dir, 'ckpt')
  cands = sorted(d for d in os.listdir(ck)
                 if os.path.isdir(os.path.join(ck, d))
                 and os.path.exists(os.path.join(ck, d, 'done')))
  assert cands, f'no done ckpt under {ck}'
  return os.path.join(ck, cands[-1])


def transform(obj, rng, lam, sigma, stats, path='payload'):
  """Recursively shrink-perturb float ndarrays; everything else intact."""
  if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.floating):
    eps = rng.normal(0.0, sigma, obj.shape).astype(obj.dtype)
    stats.append(dict(key=path, shape=list(obj.shape),
                      dtype=str(obj.dtype)))
    return (lam * obj + eps).astype(obj.dtype)
  if isinstance(obj, dict):
    return {k: transform(obj[k], rng, lam, sigma, stats, f'{path}/{k}')
            for k in sorted(obj)}
  if isinstance(obj, (list, tuple)):
    out = [transform(v, rng, lam, sigma, stats, f'{path}[{i}]')
           for i, v in enumerate(obj)]
    return type(obj)(out)
  return obj


def run(args):
  src_ckpt = args.checkpoint or latest_done_ckpt(args.donor)
  folder = os.path.basename(src_ckpt.rstrip('/'))
  pkls = sorted(f for f in os.listdir(src_ckpt) if f.endswith('.pkl'))
  assert pkls, f'no .pkl in {src_ckpt}'
  os.makedirs(args.output, exist_ok=False)  # refuse to clobber
  shutil.copy2(os.path.join(args.donor, 'config.yaml'),
               os.path.join(args.output, 'config.yaml'))
  out_ck = os.path.join(args.output, 'ckpt', folder)
  os.makedirs(out_ck)
  rng = np.random.default_rng(args.seed)
  stats, in_shas, out_shas = [], {}, {}
  for name in pkls:
    src = os.path.join(src_ckpt, name)
    in_shas[name] = sha256_file(src)
    with open(src, 'rb') as f:
      payload = pickle.load(f)
    new = transform(payload, rng, args.lam, args.sigma, stats,
                    path=name)
    dst = os.path.join(out_ck, name)
    with open(dst, 'wb') as f:
      pickle.dump(new, f)
    out_shas[name] = sha256_file(dst)
  open(os.path.join(out_ck, 'done'), 'w').close()
  with open(os.path.join(args.output, 'ckpt', 'latest'), 'w') as f:
    f.write(folder)
  manifest = dict(
      tool='shrink_perturb_v1', donor=os.path.abspath(args.donor),
      donor_ckpt=os.path.abspath(src_ckpt), donor_shas=in_shas,
      lam=args.lam, sigma=args.sigma, seed=args.seed,
      n_arrays=len(stats), output_shas=out_shas, arrays=stats)
  with open(os.path.join(args.output, 'snp_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
  print(f'{len(stats)} float arrays transformed (lam={args.lam}, '
        f'sigma={args.sigma}, seed={args.seed}) -> {args.output}')


def selfcheck():
  import tempfile
  base = tempfile.mkdtemp(prefix='snp_sc_')
  donor = os.path.join(base, 'donor')
  ck = os.path.join(donor, 'ckpt', '20260820T000000F000-000000500000')
  os.makedirs(ck)
  open(os.path.join(donor, 'config.yaml'), 'w').write('agent: {}\n')
  rng0 = np.random.default_rng(42)
  payload = {'enc/w': rng0.standard_normal((64, 32)).astype(np.float32),
             'dyn/b': rng0.standard_normal(128).astype(np.float64),
             'meta/step': np.array(500000, np.int64),
             'nested': {'rew/k': rng0.standard_normal((4, 4)).astype(
                 np.float32)},
             'name': 'donor'}
  with open(os.path.join(ck, 'agent.pkl'), 'wb') as f:
    pickle.dump(payload, f)
  open(os.path.join(ck, 'done'), 'w').close()
  with open(os.path.join(donor, 'ckpt', 'latest'), 'w') as f:
    f.write(os.path.basename(ck))

  out1 = os.path.join(base, 'snp1')
  out2 = os.path.join(base, 'snp2')
  for out in (out1, out2):
    run(argparse.Namespace(donor=donor, checkpoint=None, output=out,
                           lam=LAM, sigma=SIGMA, seed=7))
  s1 = sha256_file(os.path.join(
      out1, 'ckpt', os.path.basename(ck), 'agent.pkl'))
  s2 = sha256_file(os.path.join(
      out2, 'ckpt', os.path.basename(ck), 'agent.pkl'))
  assert s1 == s2, 'same seed must be byte-deterministic'
  with open(os.path.join(out1, 'ckpt', os.path.basename(ck),
                         'agent.pkl'), 'rb') as f:
    new = pickle.load(f)
  # transform stats: mean(theta') ~ lam*mean(theta), added variance sigma^2
  d = new['enc/w'] - LAM * payload['enc/w']
  assert abs(float(d.mean())) < 5 * SIGMA / np.sqrt(d.size)
  assert abs(float(d.std()) - SIGMA) < 0.2 * SIGMA
  assert new['meta/step'] == payload['meta/step'], 'ints must be intact'
  assert new['name'] == 'donor'
  assert new['nested']['rew/k'].dtype == np.float32
  assert os.path.exists(os.path.join(out1, 'ckpt', 'latest'))
  assert os.path.exists(os.path.join(out1, 'config.yaml'))
  m = json.load(open(os.path.join(out1, 'snp_manifest.json')))
  assert m['n_arrays'] == 3 and m['lam'] == LAM and m['sigma'] == SIGMA
  try:
    run(argparse.Namespace(donor=donor, checkpoint=None, output=out1,
                           lam=LAM, sigma=SIGMA, seed=7))
  except (FileExistsError, OSError):
    pass
  else:
    raise AssertionError('must refuse to clobber an existing output')
  print('shrink_perturb selfcheck PASS (determinism, stats, int/str '
        'preservation, run-dir shape, no-clobber)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--donor')
  ap.add_argument('--checkpoint', default=None,
                  help='explicit ckpt folder (default: latest done)')
  ap.add_argument('--output')
  ap.add_argument('--lam', type=float, default=LAM)
  ap.add_argument('--sigma', type=float, default=SIGMA)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
  else:
    assert args.donor and args.output
    run(args)


if __name__ == '__main__':
  main()
