"""B-embeddings of a frozen proprio probeset for embed_probe.

FB wave (FB_Wave_Plan_20260816; prereg PREREG_fb_20260817) — the P-FB2
representation leg's embedding producer. Two modes:

  fb   — load fb_ckpt.pt, embed every probeset frame through
         backward_net (B: 12-dim proprio -> z_dim), write
         probeset_emb.npz [N, T, z_dim] + embed_manifest.json in the
         probing/lewm_embed.py output format (probing/embed_probe.py
         consumes both unchanged).
  raw  — identity comparator: emb = the pinned 12-dim proprio concat
         itself (the paired null: does the FB objective structure
         reward-relevance beyond the raw observables?).

Probeset frames are concatenated in probing/fb_export.OBS_KEYS order —
the SAME pinned order the fits trained on.

Usage (torch env for fb mode; raw mode is numpy-only):
  CONTROLLABLE_AGENT_ROOT=... python -m probing.fb_embed fb \
      --ckpt <logdir>/fb_ckpt.pt --probeset $RUNROOT/e4_probesets/finger_v1 \
      --output <dir>
  python -m probing.fb_embed raw --probeset ... --output <dir>
Selfcheck (torch env): python -m probing.fb_embed --selfcheck
"""

import argparse
import hashlib
import json
import os

import numpy as np

from probing.fb_export import OBS_KEYS, OBS_DIMS, OBS_DIM


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def probeset_frames(probeset_dir):
  # review B4: the e4 probesets store probeset_e4.npz; go through
  # stratified_error.load_e4 so the FROZEN/sha verification applies and
  # the filename can never drift.
  from probing.stratified_error import load_e4
  z, ps_manifest = load_e4(probeset_dir, require_frozen=True)
  parts = []
  for k in OBS_KEYS:
    a = np.asarray(z[k], np.float32)
    if a.ndim == 2:
      a = a[..., None]
    assert a.shape[2] == OBS_DIMS[k], (k, a.shape)
    parts.append(a)
  frames = np.concatenate(parts, axis=2)  # [N, T, 12]
  assert frames.shape[2] == OBS_DIM
  return frames, ps_manifest


def gpu_name(device):
  """Physical device identity for the single-GPU panel gate (review-2
  B-C): the torch device STRING ('cuda') is not a GPU identity."""
  if str(device).startswith('cuda'):
    import torch
    return torch.cuda.get_device_name(0)
  return 'cpu'


def write_out(args, embs, ckpt_path, ckpt_sha, device):
  os.makedirs(args.output, exist_ok=True)
  np.savez(os.path.join(args.output, 'probeset_emb.npz'),
           emb=embs.astype(np.float32))
  run_dir = ('raw-identity' if ckpt_path == 'raw-identity'
             else os.path.dirname(os.path.abspath(ckpt_path)))
  manifest = dict(run_dir=run_dir,
                  ckpt=ckpt_path, ckpt_sha256=ckpt_sha,
                  dim=int(embs.shape[-1]), n_episodes=int(embs.shape[0]),
                  source=os.path.abspath(args.probeset),
                  preproc='fb_obs_keys_v1', device=device,
                  gpu_name=gpu_name(device))
  with open(os.path.join(args.output, 'embed_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
  print(f'{embs.shape[0]} eps embedded (D={embs.shape[-1]}) '
        f'-> {args.output}')


def run_fb(args):
  import torch
  from probing.fb_fit import load_agent
  agent, _ = load_agent(args.ckpt, device=args.device)
  frames, _ps = probeset_frames(args.probeset)
  N, T, D = frames.shape
  flat = torch.as_tensor(frames.reshape(-1, D), dtype=torch.float32,
                         device=args.device)
  outs = []
  with torch.no_grad():
    for i in range(0, len(flat), 8192):
      outs.append(agent.backward_net(flat[i:i + 8192]).cpu().numpy())
  embs = np.concatenate(outs).reshape(N, T, -1)
  assert np.isfinite(embs).all()
  write_out(args, embs, os.path.abspath(args.ckpt),
            sha256_file(args.ckpt), args.device)


def run_raw(args):
  frames, _ps = probeset_frames(args.probeset)
  write_out(args, frames, 'raw-identity', 'raw-identity', 'cpu')


def selfcheck():
  import tempfile
  base = tempfile.mkdtemp(prefix='fb_embed_sc_')
  ps = os.path.join(base, 'ps')
  os.makedirs(ps)
  rng = np.random.default_rng(0)
  N, T = 5, 7
  arrs = {k: rng.standard_normal((N, T, OBS_DIMS[k])).astype(np.float32)
          for k in OBS_KEYS}
  arrs['dist_to_target'] = arrs['dist_to_target'][..., 0]
  npz_path = os.path.join(ps, 'probeset_e4.npz')
  np.savez(npz_path, **arrs)
  digest = sha256_file(npz_path)
  json.dump(dict(sha256=digest, probeset_id='sc_v1'),
            open(os.path.join(ps, 'manifest.json'), 'w'))
  open(os.path.join(ps, 'FROZEN'), 'w').write(digest)
  args = argparse.Namespace(probeset=ps, output=os.path.join(base, 'raw'))
  run_raw(args)
  z = np.load(os.path.join(base, 'raw', 'probeset_emb.npz'))
  assert z['emb'].shape == (N, T, OBS_DIM)
  man = json.load(open(os.path.join(base, 'raw', 'embed_manifest.json')))
  assert man['dim'] == OBS_DIM and man['preproc'] == 'fb_obs_keys_v1'
  # fb mode against a tiny trained ckpt (torch env)
  try:
    import torch  # noqa: F401
    from probing import fb_fit
    E, TT = 3, 17
    data = os.path.join(base, 'data.npz')
    np.savez(data, obs=rng.standard_normal((E, TT, 12)).astype(np.float32),
             action=rng.uniform(-1, 1, (E, TT, 2)).astype(np.float32),
             reward=np.zeros((E, TT), np.float32))
    json.dump(dict(tool='fb_export_v1', obs_dim=12, action_dim=2,
                   n_episodes=E, ep_len=TT, obs_keys=[],
                   output_sha256=sha256_file(data)),
              open(data + '.manifest.json', 'w'))
    logdir = os.path.join(base, 'run')
    fb_fit.fit(argparse.Namespace(data=data, logdir=logdir, seed=1,
                                  updates=2, device='cpu'))
    run_fb(argparse.Namespace(ckpt=os.path.join(logdir, 'fb_ckpt.pt'),
                              probeset=ps, output=os.path.join(base, 'fb'),
                              device='cpu'))
    z2 = np.load(os.path.join(base, 'fb', 'probeset_emb.npz'))
    assert z2['emb'].shape == (N, T, 50)
    print('fb_embed selfcheck PASS (raw + fb modes, shapes + manifest)')
  except ImportError:
    print('fb_embed selfcheck PARTIAL PASS (raw mode only — no torch '
          'in this env; run the full selfcheck in the torch env)')


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd')
  fb = sub.add_parser('fb')
  fb.add_argument('--ckpt', required=True)
  fb.add_argument('--probeset', required=True)
  fb.add_argument('--output', required=True)
  fb.add_argument('--device', default='cuda')
  raw = sub.add_parser('raw')
  raw.add_argument('--probeset', required=True)
  raw.add_argument('--output', required=True)
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
  elif args.cmd == 'fb':
    run_fb(args)
  else:
    assert args.cmd == 'raw'
    run_raw(args)


if __name__ == '__main__':
  main()
