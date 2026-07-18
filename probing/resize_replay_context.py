"""Zero-initialize the stored replay-context latents of a built pair at
new model dims (scaling pilot; PREREG_scaling_pilot_20260717.md).

Why: built Axis-1 buffers byte-preserve the SOURCE runs' `dyn/deter` /
`dyn/stoch` replay-context entries (size1m: 512 / 32x4). With
`replay_context: 1` the agent rebuilds its RSSM carry from these
entries (dreamerv3/agent.py `_apply_replay_context`), so a size12m fit
(deter 2048, stoch 32x16) fails at the compiled-shape check. This tool
copies a pair with `dyn/*` replaced by ZEROS of the target dims; every
other key is byte-preserved and stepid is re-encoded
(`write_episode_chunk`), so Replay.update() refreshes the context with
the new agent's own latents after the first pass -- the same contract
the synthesized synth buffers use (validated end-to-end 2026-07-17).

Protocol note for the prereg: the historical 1m fits started from the
source agents' stored latents, the resized 12m fits start from zeros --
a stratum-level protocol difference that is internally consistent
within each size stratum (both 12m arms use the same resized pair) and
is disclosed in the registration.

Usage (login node, CPU)::

    python -m probing.resize_replay_context resize \
        --input $RUNROOT/axis1_finger/q1 --deter 2048 --stoch 32 16 \
        --output $RUNROOT/axis1_finger/q1_ctx12

    python -m probing.resize_replay_context selfcheck
"""

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from probing.build_controlled_replay import write_episode_chunk
from probing.relabel_replay import load_episode_chunks

LATENT_KEYS = ('dyn/deter', 'dyn/stoch')


def cmd_resize(args):
  in_manifest = os.path.join(args.input, 'manifest.json')
  if not os.path.exists(in_manifest):
    raise SystemExit(f'{args.input}: no manifest.json (not a built pair).')
  if os.path.exists(os.path.join(args.output, 'manifest.json')):
    raise SystemExit(f'{args.output} already materialized; refusing to '
                     f'overwrite (buffers are registered artifacts).')
  with open(in_manifest) as f:
    manifest = json.load(f)

  sides = sorted(d for d in os.listdir(args.input)
                 if d.startswith('side') and
                 os.path.isdir(os.path.join(args.input, d)))
  if not sides:
    raise SystemExit(f'{args.input}: no side*/ directories.')
  stoch = tuple(args.stoch)
  side_stats = {}
  for side in sides:
    episodes = load_episode_chunks(os.path.join(args.input, side))
    out_dir = os.path.join(args.output, side)
    os.makedirs(out_dir, exist_ok=True)
    src_shapes = None
    for frames in episodes:
      latents = [k for k in frames if k.startswith(('dyn/', 'enc/', 'dec/'))]
      unexpected = sorted(set(latents) - set(LATENT_KEYS))
      if unexpected:
        raise SystemExit(f'{side}: unexpected latent keys {unexpected}; '
                         f'this tool only handles {LATENT_KEYS}.')
      n = len(np.asarray(frames['reward']))
      if src_shapes is None and all(k in frames for k in LATENT_KEYS):
        src_shapes = {k: list(frames[k].shape[1:]) for k in LATENT_KEYS}
      out = {k: v for k, v in frames.items() if k not in LATENT_KEYS}
      out['dyn/deter'] = np.zeros((n, args.deter), np.float32)
      out['dyn/stoch'] = np.zeros((n,) + stoch, np.float32)
      write_episode_chunk(out_dir, out)
    side_stats[side] = dict(n_episodes=len(episodes),
                            source_latent_shapes=src_shapes)
    print(f'{side}: {side_stats[side]}')

  manifest = dict(manifest)
  manifest['context_resize'] = dict(
      source_pair=os.path.abspath(args.input),
      deter=args.deter, stoch=list(stoch),
      sides=side_stats,
      note='dyn/* replay-context entries zero-initialized at the target '
           'model dims; all other keys byte-preserved; stepid re-encoded '
           'so Replay.update() refreshes the context during the fit')
  with open(os.path.join(args.output, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
  print(f'-> {args.output} (deter {args.deter}, stoch {stoch})')


def cmd_selfcheck(args):
  import tempfile
  rng = np.random.default_rng(3)
  with tempfile.TemporaryDirectory() as tmp:
    src = os.path.join(tmp, 'q1')
    for side in ('side0', 'side1'):
      out = os.path.join(src, side)
      os.makedirs(out)
      for _ in range(2):
        n = 30
        frames = {
            'position': rng.normal(0, 1, (n, 4)).astype(np.float32),
            'reward': (rng.random(n) < 0.2).astype(np.float32),
            'is_first': np.eye(1, n, 0, dtype=bool)[0],
            'stepid': np.zeros((n, 20), np.uint8),
            'dyn/deter': rng.normal(0, 1, (n, 8)).astype(np.float32),
            'dyn/stoch': rng.normal(0, 1, (n, 4, 2)).astype(np.float32),
        }
        write_episode_chunk(out, frames)
    with open(os.path.join(src, 'manifest.json'), 'w') as f:
      json.dump(dict(task='dmc_cup_catch'), f)

    dst = os.path.join(tmp, 'q1_ctx')
    cmd_resize(argparse.Namespace(
        input=src, output=dst, deter=32, stoch=[4, 8]))
    for side in ('side0', 'side1'):
      orig = load_episode_chunks(os.path.join(src, side))
      new = load_episode_chunks(os.path.join(dst, side))
      assert len(orig) == len(new)
      for o, t in zip(orig, new):
        assert t['dyn/deter'].shape == (30, 32)
        assert t['dyn/stoch'].shape == (30, 4, 8)
        assert not t['dyn/deter'].any() and not t['dyn/stoch'].any()
        for key in ('position', 'reward', 'is_first'):
          assert np.array_equal(o[key], t[key]), key
        assert not np.array_equal(o['stepid'], t['stepid'])
    with open(os.path.join(dst, 'manifest.json')) as f:
      m = json.load(f)
    assert m['context_resize']['deter'] == 32
    assert m['context_resize']['sides']['side0']['source_latent_shapes'] == \
        {'dyn/deter': [8], 'dyn/stoch': [4, 2]}
    # refuses overwrite
    try:
      cmd_resize(argparse.Namespace(
          input=src, output=dst, deter=32, stoch=[4, 8]))
      raise AssertionError('overwrote an existing resized pair')
    except SystemExit:
      pass
  print('resize_replay_context selfcheck PASS')


def main():
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)

  r = sub.add_parser('resize')
  r.add_argument('--input', required=True,
                 help='Built pair dir (side0/side1 + manifest.json).')
  r.add_argument('--output', required=True)
  r.add_argument('--deter', type=int, required=True,
                 help='Target deter width (size12m: 2048).')
  r.add_argument('--stoch', type=int, nargs='+', required=True,
                 help='Target stoch shape (size12m: 32 16).')
  r.set_defaults(fn=cmd_resize)

  sc = sub.add_parser('selfcheck')
  sc.set_defaults(fn=cmd_selfcheck)

  args = p.parse_args()
  args.fn(args)


if __name__ == '__main__':
  main()
