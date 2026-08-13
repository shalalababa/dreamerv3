"""Embed pixel episodes with a trained LeWM encoder (PREREG_lewm_20260812).

Loads a LeWM training run (the pinned le-wm checkout's outputs: run_dir
with config.yaml + <name>_weights.ckpt), reconstructs the model via the
run's own Hydra config, and writes one embedding sidecar npz per input
episode: {'emb': [T, D] float32} named <episode-stem>.npz under
--output, plus an embed_manifest.json (run dir, ckpt file, sha256 of
the ckpt, D, episode count, source, preproc).

PREPROCESSING PIN (review C3): --preproc is REQUIRED and recorded in
every manifest — 'raw' (float 0-255), 'unit' (/255), or 'imagenet'
(unit, then ImageNet mean/std). The value that matches the pinned
checkout's own training-time pixel path is established ONCE by the
registered `verify` subcommand at the smoke (below) and then used for
every embed pass; the frozen reader refuses mixed or unverified
preproc.

Inputs are either (a) --side_dir: a chunk dir (one npz per episode,
'image' [T,H,W,C] uint8 — the pxq1m contract) or (b) --probeset: a
stratified_error probeset dir (probeset_e4.npz with 'image'
[N,T,H,W,C]) — sidecar probeset_emb.npz with 'emb' [N, T, D].

`verify` (the registered embedding-pipeline positive control): compares
this script's preprocessed pixels for ONE episode against the pinned
checkout's OWN training-time pixel path — Amendment 1: the factory call
`utils.get_img_preprocessor(source='pixels', target='pixels',
img_size=--img_size)` fed a sample dict {'pixels': uint8 [T,C,H,W]
tensor}, exactly the stable_worldmodel HDF5Dataset yield the training
loop transforms — and writes embed_control.json {preproc, img_size,
spt_version, max_abs_diff, pass}. It FAILS LOUDLY if the checkout's
preprocessor cannot be located or the shapes cannot match — never a
silent pass; interface adjustments land as dated amendments before any
full training (FILL discipline).

Runs in the lewm conda env (torch + hydra + the pinned checkout on
LEWM_CHECKOUT). GPU optional.

Usage:
  python -m probing.lewm_embed embed --run_dir $RUNROOT/lewm_finger_s0_seed1 \
      --side_dir $RUNROOT/axis1_finger/pxq1m/side0 --preproc <pinned> \
      --output $RUNROOT/lewm_emb/s0_seed1
  python -m probing.lewm_embed verify --episode <one .npz chunk> \
      --preproc <candidate> --output <dir>
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _add_checkout_path():
  co = os.environ.get('LEWM_CHECKOUT')
  if co and co not in sys.path:
    sys.path.insert(0, co)
  return co


def preprocess(frames_uint8, preproc):
  """[T,H,W,C] uint8 -> [T,C,H,W] float32 under the pinned convention."""
  import torch
  t = torch.from_numpy(np.asarray(frames_uint8))
  assert t.dtype == torch.uint8 and t.ndim == 4, (t.dtype, t.shape)
  pix = t.permute(0, 3, 1, 2).float()
  if preproc == 'raw':
    return pix
  if preproc == 'unit':
    return pix / 255.0
  if preproc == 'imagenet':
    # verification N2: the (1,3,1,1) constants would silently
    # broadcast a 1-channel input
    assert pix.shape[1] == 3, ('imagenet preproc needs RGB', pix.shape)
    pix = pix / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (pix - mean) / std
  raise SystemExit(f'unknown preproc {preproc!r}')


def load_model(run_dir):
  import torch
  from omegaconf import OmegaConf
  import hydra
  _add_checkout_path()
  cfg = OmegaConf.load(os.path.join(run_dir, 'config.yaml'))
  model = hydra.utils.instantiate(cfg.model)
  ckpts = sorted(glob.glob(os.path.join(run_dir, '*_weights.ckpt')))
  assert len(ckpts) == 1, (run_dir, ckpts)
  sd = torch.load(ckpts[0], map_location='cpu', weights_only=False)
  if isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']
  sd = {(k[len('model.'):] if k.startswith('model.') else k): v
        for k, v in sd.items()}
  missing, unexpected = model.load_state_dict(sd, strict=False)
  # encoder weights must load COMPLETELY — anything else is a wrong ckpt
  enc_missing = [k for k in missing if k.startswith('encoder')]
  assert not enc_missing, f'encoder keys missing from ckpt: {enc_missing[:5]}'
  model.eval()
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.to(dev)
  return model, ckpts[0], dev


def embed_frames(model, frames_uint8, dev, preproc, batch=256):
  """frames [T,H,W,C] uint8 -> [T,D] float32 CLS embeddings (the audited
  le-wm call: encoder(pixels, interpolate_pos_encoding=True)
  .last_hidden_state[:, 0])."""
  import torch
  out = []
  pix = preprocess(frames_uint8, preproc)
  with torch.no_grad():
    for lo in range(0, len(pix), batch):
      chunk = pix[lo:lo + batch].to(dev)
      res = model.encoder(chunk, interpolate_pos_encoding=True)
      cls = res.last_hidden_state[:, 0]
      out.append(cls.float().cpu().numpy())
  return np.concatenate(out, 0)


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for blk in iter(lambda: f.read(1 << 20), b''):
      h.update(blk)
  return h.hexdigest()


def cmd_embed(args):
  model, ckpt, dev = load_model(args.run_dir)
  os.makedirs(args.output, exist_ok=True)
  dim = None
  n = 0
  if args.side_dir:
    files = sorted(glob.glob(os.path.join(args.side_dir, '*.npz')))
    assert files, args.side_dir
    for f in files:
      with np.load(f) as z:
        emb = embed_frames(model, z['image'], dev, args.preproc)
      dim = emb.shape[-1]
      np.savez(os.path.join(args.output, os.path.basename(f)),
               emb=emb.astype(np.float32))
      n += 1
    source = os.path.abspath(args.side_dir)
  else:
    arrs = np.load(os.path.join(args.probeset, 'probeset_e4.npz'))
    img = arrs['image']              # [N, T, H, W, C]
    N = img.shape[0]
    embs = np.stack([embed_frames(model, img[i], dev, args.preproc)
                     for i in range(N)], 0)
    dim = embs.shape[-1]
    np.savez(os.path.join(args.output, 'probeset_emb.npz'),
             emb=embs.astype(np.float32))
    n = N
    source = os.path.abspath(args.probeset)
  manifest = dict(run_dir=os.path.abspath(args.run_dir),
                  ckpt=os.path.abspath(ckpt), ckpt_sha256=sha256(ckpt),
                  dim=int(dim), n_episodes=int(n), source=source,
                  preproc=args.preproc, device=dev)
  with open(os.path.join(args.output, 'embed_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
  print(f'{n} episodes embedded (D={dim}, preproc={args.preproc}) '
        f'-> {args.output}')


def _checkout_preprocessor():
  """Locate the pinned checkout's own training-time image-preprocessor
  FACTORY. FAILS LOUDLY when not found (review C3: never a silent pass).

  Amendment 1 (interface pin, from the checkout at 8edfeb33):
  train.py:59 builds the training transform as
    get_img_preprocessor(source='pixels', target='pixels',
                         img_size=cfg.img_size)
  (utils.py:6), a stable_pretraining dict-transform Compose of
  ToImage(**ImageNet_stats) + Resize(img_size) keyed on 'pixels'."""
  co = _add_checkout_path()
  assert co, 'LEWM_CHECKOUT env var must point at the pinned checkout'
  try:
    import utils as lewm_utils
  except Exception as e:  # noqa: BLE001 — diagnostic path only
    raise SystemExit(
        f'could not import the checkout `utils` module ({e}) — pin the '
        'correct symbol via a dated amendment before any full training '
        '(FILL discipline).')
  factory = getattr(lewm_utils, 'get_img_preprocessor', None)
  if factory is None:
    raise SystemExit(
        'checkout utils has no get_img_preprocessor — pin the correct '
        'symbol via a dated amendment before any full training '
        '(FILL discipline).')
  return factory, 'utils.get_img_preprocessor'


def cmd_verify(args):
  import torch
  factory, symbol = _checkout_preprocessor()
  with np.load(args.episode) as z:
    img = np.asarray(z['image'])
  ours = preprocess(img, args.preproc)
  # Replicate the checkout's training-time pixel path EXACTLY:
  # stable_worldmodel HDF5Dataset._load_slice yields 'pixels' as a torch
  # uint8 tensor permuted [T, C, H, W]; the transform consumes/returns a
  # sample dict keyed 'pixels' (train.py:59 call signature).
  raw = torch.from_numpy(img).permute(0, 3, 1, 2)
  fn = factory(source='pixels', target='pixels', img_size=args.img_size)
  theirs = torch.as_tensor(fn({'pixels': raw})['pixels']).float()
  try:
    import importlib.metadata as md
    spt_version = md.version('stable-pretraining')
  except Exception:  # noqa: BLE001 — audit field only
    spt_version = 'unknown'
  os.makedirs(args.output, exist_ok=True)
  path = os.path.join(args.output, 'embed_control.json')
  if theirs.shape != ours.shape:
    # img_size != source resolution: the embed path has no resize, so no
    # --preproc candidate can match — recorded fail, loud exit
    rec = dict(preproc=args.preproc, preprocessor_symbol=symbol,
               episode=os.path.abspath(args.episode),
               img_size=args.img_size, spt_version=spt_version,
               their_shape=list(theirs.shape), our_shape=list(ours.shape),
               max_abs_diff=None, threshold=1e-5, **{'pass': False})
    with open(path, 'w') as f:
      json.dump(rec, f, indent=1)
    raise SystemExit(
        f'SHAPE MISMATCH: checkout path yields {tuple(theirs.shape)} vs '
        f'ours {tuple(ours.shape)} — the embed path has no resize; a '
        'dated amendment is required before any full training.')
  diff = float((theirs - ours).abs().max())
  ok = diff < 1e-5
  rec = dict(preproc=args.preproc, preprocessor_symbol=symbol,
             episode=os.path.abspath(args.episode),
             img_size=args.img_size, spt_version=spt_version,
             max_abs_diff=diff, threshold=1e-5, **{'pass': bool(ok)})
  with open(path, 'w') as f:
    json.dump(rec, f, indent=1)
  print(json.dumps(rec, indent=1))
  if not ok:
    raise SystemExit(
        f'PREPROC MISMATCH: {args.preproc} differs from {symbol} by '
        f'{diff} — try the other --preproc values; the passing one is '
        'the pin.')
  print(f'-> {path}')


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd', required=True)
  em = sub.add_parser('embed')
  em.add_argument('--run_dir', required=True)
  em.add_argument('--side_dir')
  em.add_argument('--probeset')
  em.add_argument('--preproc', required=True,
                  choices=('raw', 'unit', 'imagenet'))
  em.add_argument('--output', required=True)
  ve = sub.add_parser('verify')
  ve.add_argument('--episode', required=True)
  ve.add_argument('--preproc', required=True,
                  choices=('raw', 'unit', 'imagenet'))
  ve.add_argument('--img_size', type=int, default=64,
                  help='cfg.img_size of the registered training config '
                       '(prereg pin: 64)')
  ve.add_argument('--output', required=True)
  args = ap.parse_args()
  if args.cmd == 'verify':
    cmd_verify(args)
    return
  assert bool(args.side_dir) != bool(args.probeset), \
      'exactly one of --side_dir / --probeset'
  cmd_embed(args)


if __name__ == '__main__':
  main()
