"""Fit a fresh DreamerV3 agent offline on a static, externally-built replay.

Purpose
-------
Two uses share one code path:

* **Phase 0 mini-gate** -- prove the trainer can *ingest a static replay* (valid
  ``time-uuid-succ-length.npz`` chunks written by something other than this run)
  and that the world-model losses go down.  This de-risks Phase 6 before any
  expensive run depends on it.
* **Phase 6 offline WM fit** -- retrain a fresh world model on each
  confound-controlled buffer built by ``build_controlled_replay.py``, then hand
  the checkpoint to frozen-readout adaptation.

Unlike ``embodied.run.train`` (which only populates replay from live env steps
and only calls ``replay.load`` when *resuming* a checkpoint), this harness builds
a fresh agent, explicitly loads a static chunk directory via ``Replay.load`` --
which reconstructs sampling items across the successor-chained chunks -- and runs
pure gradient steps with no environment interaction.  It then writes a standard
``ckpt/`` so the result loads via ``--run.from_checkpoint`` and
``probing/collect.py`` exactly like an online run.

Example (mini-gate, debug sizes, CPU)::

    python -m probing.offline_fit \
        --logdir /tmp/offfit_cup --static_replay /tmp/static_cup_replay \
        --updates 200 --configs dmc_proprio debug --task dmc_cup_catch \
        --jax.platform cpu

The agent architecture must match the one that produced the chunks (same size
preset), because the replay stores latent-context keys (``dyn/deter`` etc.).
"""

import argparse
import functools
import pathlib
import sys
import time

folder = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(folder))

import elements
import numpy as np
import ruamel.yaml as yaml

import embodied
from dreamerv3 import main as dv3_main


# Loss keys that are NOT world-model reconstruction/dynamics (policy/critic side).
# Everything else under loss/* is a world-model term (decoder heads, dyn, rep,
# con, and per-observation reconstruction losses whose names vary by domain).
NON_WM_LOSS = {'policy', 'value', 'repval', 'actor', 'critic', 'ent', 'disag'}


def _losses(mets):
  """All loss/* terms in a metrics dict as {name: scalar float}."""
  out = {}
  for k, v in mets.items():
    if k.startswith('loss/'):
      out[k[len('loss/'):]] = float(np.asarray(v).mean())
  return out


def _write_progress(logdir, update, total_updates, wm_total):
  path = pathlib.Path(str(logdir)) / 'OFFLINE_FIT_PROGRESS'
  path.write_text(
      f'updated_at={time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n'
      f'update={update}\n'
      f'total_updates={total_updates}\n'
      f'wm_total={wm_total:.6f}\n')


def build_config(argv):
  """Reconstruct a config exactly like dreamerv3/main.main, minus the run."""
  cfg_path = pathlib.Path(dv3_main.__file__).parent / 'configs.yaml'
  configs = yaml.YAML(typ='safe').load(elements.Path(cfg_path).read())
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=config.logdir.format(
      timestamp=elements.timestamp()))
  return config


def parse_args(argv):
  p = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--static_replay', required=True,
                 help='Directory of .npz replay chunks to fit on (read-only).')
  p.add_argument('--updates', type=int, default=200,
                 help='Number of gradient steps.')
  p.add_argument('--log_every', type=int, default=25)
  p.add_argument('--save_every_updates', type=int, default=50000,
                 help='Save resumable checkpoints every N updates; <=0 disables intermediate saves.')
  p.add_argument('--keep', type=int, default=2,
                 help='Number of offline-fit checkpoints to retain.')
  p.add_argument('--resume', action='store_true', default=True,
                 help='Resume from logdir/ckpt/latest when present (on by default).')
  p.add_argument('--no_resume', dest='resume', action='store_false')
  p.add_argument('--save', action='store_true', default=True,
                 help='Write a ckpt/ at the end (on by default).')
  p.add_argument('--no_save', dest='save', action='store_false')
  return p.parse_known_args(argv)


def main(argv=None):
  argv = sys.argv[1:] if argv is None else argv
  args, passthrough = parse_args(argv)

  config = build_config(passthrough)
  logdir = elements.Path(config.logdir)
  pathlib.Path(str(logdir)).mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(f'Logdir: {logdir}', flush=True)
  print(f'Static replay: {args.static_replay}', flush=True)

  agent = dv3_main.make_agent(config)
  replay = dv3_main.make_replay(config, 'replay')
  step = elements.Counter()
  cp = None
  resumed = False
  if args.save:
    cp = elements.Checkpoint(logdir / 'ckpt', keep=args.keep, step=step)
    cp.step = step
    cp.agent = agent
    if args.resume and cp.exists():
      cp.load()
      resumed = True
      print(f'Resumed offline-fit checkpoint at update {int(agent.n_updates)}',
            flush=True)

  # Partial init from a donor checkpoint (mirrors embodied/run/train.py's
  # from_checkpoint handling for the adapt stage). Regex-loads only matching
  # params; everything else keeps its fresh init. agent.load restores the
  # donor's update counters even in regex mode, which would make the loop
  # below start at the donor's n_updates and silently skip the entire fit —
  # reset them to 0. Skipped on resume: a resumed partial-fit ckpt already
  # contains the donor params.
  if config.run.from_checkpoint and not resumed:
    elements.checkpoint.load(config.run.from_checkpoint, dict(
        agent=functools.partial(
            agent.load, regex=config.run.from_checkpoint_regex)))
    agent.n_updates.value = 0
    agent.n_batches.value = 0
    agent.n_actions.value = 0
    print(f'PARTIAL_INIT from={config.run.from_checkpoint} '
          f'regex={config.run.from_checkpoint_regex} counters_reset=0',
          flush=True)

  # The real Phase-6 ingestion path: pull in externally-authored chunks.
  replay.load(directory=args.static_replay)
  n_items = len(replay)
  print(f'Loaded static replay: {n_items} sampling items '
        f'(seq length {replay.length}).', flush=True)
  if n_items < config.batch_size:
    raise SystemExit(
        f'Only {n_items} items < batch_size {config.batch_size}; the static '
        f'buffer is too small or its chunks are shorter than the sample '
        f'length {replay.length} without successor chaining.')

  stream = iter(agent.stream(dv3_main.make_stream(config, replay, 'train')))
  carry = [agent.init_train(config.batch_size)]

  history = []
  start_update = int(agent.n_updates) if args.save else 0
  if start_update >= args.updates:
    print(f'Already reached requested updates: {start_update}/{args.updates}',
          flush=True)
  last_saved_update = start_update
  for i in range(start_update, args.updates):
    batch = next(stream)
    carry[0], outs, mets = agent.train(carry[0], batch)
    if 'replay' in outs:
      replay.update(outs['replay'])
    if i == 0 or (i + 1) % args.log_every == 0 or i == args.updates - 1:
      row = _losses(mets)
      history.append((i + 1, row))
      wm = sum(v for k, v in row.items() if k not in NON_WM_LOSS)
      shown = {k: row[k] for k in sorted(row) if k not in NON_WM_LOSS}
      msg = '  '.join(f'{k}={v:.3f}' for k, v in shown.items())
      _write_progress(logdir, i + 1, args.updates, wm)
      print(f'  update {i + 1:>5}/{args.updates}: wm_total={wm:.3f}  {msg}',
            flush=True)
    if (args.save and args.save_every_updates > 0 and
        (i + 1) % args.save_every_updates == 0 and
        (i + 1) < args.updates):
      step.value = i + 1
      cp.save()
      last_saved_update = i + 1
      print(f'Saved intermediate offline-fit checkpoint at update {i + 1}',
            flush=True)

  if args.save:
    step.value = max(args.updates, int(agent.n_updates))
    if last_saved_update != args.updates or not cp.latest():
      cp.save()
    print(f'Saved offline-fit checkpoint under {logdir / "ckpt"}', flush=True)

  # Sanity read-out: did the world model losses fall from first to last log?
  # (Use the second logged point as the baseline; step 1 metrics can be noisy.)
  if len(history) >= 2:
    base_i = 1 if len(history) > 2 else 0
    first, last = history[base_i][1], history[-1][1]
    keys = sorted(set(first) & set(last) - NON_WM_LOSS)
    fw = sum(first[k] for k in keys)
    lw = sum(last[k] for k in keys)
    print(f'WM-loss change (update {history[base_i][0]} -> {history[-1][0]}):',
          flush=True)
    for k in keys:
      arrow = 'down' if last[k] < first[k] - 1e-6 else (
          'up' if last[k] > first[k] + 1e-6 else 'flat')
      print(f'  {k:12s}: {first[k]:.4f} -> {last[k]:.4f} ({arrow})',
            flush=True)
    print(f'  {"wm_total":12s}: {fw:.4f} -> {lw:.4f} '
          f'({"DECREASED" if lw < fw else "did not decrease"})',
          flush=True)
  return history


if __name__ == '__main__':
  main()
