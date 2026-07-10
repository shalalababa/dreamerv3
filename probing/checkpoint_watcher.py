"""Retain time-stamped DreamerV3 checkpoints for the dose-response study.

Background
----------
`embodied/run/train.py` saves checkpoints on a *clock* schedule
(`run.save_every` seconds) into ``<logdir>/ckpt/`` using ``elements.Checkpoint``
with ``keep=1``.  The on-disk layout (verified) is::

    <logdir>/ckpt/
      latest                       # text file: name of newest complete save
      <timestamp>/                 # e.g. 20260701T145012F926412
        agent.pkl                  # (or agent-0000.pkl ... shards)
        step.pkl                   # pickled int: the exact env step
        replay.pkl                 # None for online replay (data lives in replay/)
        done                       # empty marker written last

Because ``keep=1``, every new save deletes the previous folder, and because the
schedule is clock-based the folder name is a bare timestamp with *no* step
suffix -- the exact step lives inside ``step.pkl``.  To make the Phase 5
dose-response possible we must copy checkpoints out before they are overwritten.

What this does
--------------
Poll ``<logdir>/ckpt/latest`` and, each time a *new* completed save appears,
copy its minimal loadable payload (everything except ``replay*``) into
``<snapshots_dir>/step<exact_step>/`` and append to ``manifest.json``.

Rather than gate on milestones at collection time (fragile: a clock-based save
can jump past several milestones at once, collapsing them onto one checkpoint),
we keep *every* save and let the analysis pick the snapshot nearest each desired
milestone via ``nearest_snapshots`` / ``--select``.  Agent-only snapshots are
small (~7 MB at size1m), so retaining all of them over a 500 K-step run is cheap.

A snapshot dir contains ``agent.pkl`` (+ ``step.pkl`` + ``done``) and therefore
loads directly both via ``probing/collect.py`` (``load_frozen_agent``) and via
``--run.from_checkpoint <snapshot>`` in Phase 5.

Usage
-----
Run alongside training (same node), e.g. inside the pretraining sbatch::

    python -m probing.checkpoint_watcher \
        --run_logdir $RUN/pretrain_p2e_cup_seed1 \
        --stop_step 500000 --poll_seconds 30 &

Then stops itself once the run reaches ``--stop_step`` (with a grace poll to
catch the final save).  Post-hoc milestone selection::

    python -m probing.checkpoint_watcher --select \
        --run_logdir $RUN/pretrain_p2e_cup_seed1 \
        --milestones 100000 200000 300000 400000 500000
"""

import argparse
import json
import os
import pickle
import shutil
import time


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--run_logdir', required=True,
                 help='Training run dir containing the ckpt/ subdir.')
  p.add_argument('--snapshots_dir', default='',
                 help='Where to write snapshots (default <run_logdir>/ckpt_snapshots).')
  p.add_argument('--poll_seconds', type=float, default=30.0)
  p.add_argument('--stop_step', type=int, default=0,
                 help='Exit once a save reaches this env step (0 = run until '
                      'killed or --stop_file appears).')
  p.add_argument('--stop_file', default='',
                 help='Optional path; if it appears, do a final sweep and exit.')
  p.add_argument('--milestones', type=int, nargs='+',
                 default=[100000, 200000, 300000, 400000, 500000])
  p.add_argument('--select', action='store_true',
                 help='One-shot: print the snapshot nearest each milestone and '
                      'write nearest.json, then exit (no watching).')
  return p.parse_args()


def _ckpt_dir(run_logdir):
  return os.path.join(run_logdir, 'ckpt')


def _read_latest(ckpt_dir):
  """Return (folder_abs_path, folder_name) of the newest complete save, or None."""
  latest = os.path.join(ckpt_dir, 'latest')
  if not os.path.exists(latest):
    return None
  with open(latest) as f:
    name = f.read().strip()
  if not name:
    return None
  folder = os.path.join(ckpt_dir, name)
  # `latest` is written only after the folder's `done` marker, so this is safe;
  # guard anyway in case of a torn read during cleanup.
  if not os.path.exists(os.path.join(folder, 'done')):
    return None
  return folder, name


def _read_step(folder):
  """Exact env step from a save folder's step.pkl (pickled int)."""
  try:
    with open(os.path.join(folder, 'step.pkl'), 'rb') as f:
      return int(pickle.load(f))
  except Exception:
    return None


def _copy_snapshot(src_folder, dst_dir):
  """Copy a save folder's loadable payload (all files except replay*) to dst_dir.

  Copies to a temp dir first and writes the `done` marker last so a snapshot is
  only ever observed complete.  Returns True on success.
  """
  tmp = dst_dir + '.tmp'
  shutil.rmtree(tmp, ignore_errors=True)
  os.makedirs(tmp, exist_ok=True)
  try:
    names = sorted(os.listdir(src_folder))
  except FileNotFoundError:
    return False  # source was cleaned up mid-copy; retry next poll
  payload = [n for n in names if not n.startswith('replay') and n != 'done']
  try:
    for n in payload:
      shutil.copy2(os.path.join(src_folder, n), os.path.join(tmp, n))
    # Marker last -> completeness is atomic from a reader's point of view.
    open(os.path.join(tmp, 'done'), 'wb').close()
  except FileNotFoundError:
    shutil.rmtree(tmp, ignore_errors=True)
    return False
  shutil.rmtree(dst_dir, ignore_errors=True)
  os.replace(tmp, dst_dir)
  return True


def _load_manifest(snapshots_dir):
  path = os.path.join(snapshots_dir, 'manifest.json')
  if os.path.exists(path):
    with open(path) as f:
      return json.load(f)
  return {'run_logdir': None, 'snapshots': []}


def _save_manifest(snapshots_dir, manifest):
  path = os.path.join(snapshots_dir, 'manifest.json')
  tmp = path + '.tmp'
  with open(tmp, 'w') as f:
    json.dump(manifest, f, indent=2)
  os.replace(tmp, path)


def nearest_snapshots(snapshots_dir, milestones):
  """Map each milestone step to the retained snapshot with the closest step.

  Returns a list of dicts {milestone, step, snapshot, abs_error}. Milestones with
  no snapshot within a full inter-milestone spacing are still reported (callers
  can flag large abs_error).
  """
  def step_from_name(path):
    name = os.path.basename(os.path.normpath(path))
    if name.startswith('step') and name[4:].isdigit():
      return int(name[4:])
    return None

  def localize(snapshot):
    if not snapshot:
      return snapshot
    if os.path.exists(os.path.join(snapshot, 'done')):
      return snapshot
    candidate = os.path.join(snapshots_dir, os.path.basename(snapshot))
    return candidate if os.path.exists(os.path.join(candidate, 'done')) else None

  manifest = _load_manifest(snapshots_dir)
  snaps = []
  for row in manifest.get('snapshots', []):
    snapshot = localize(row.get('snapshot'))
    if not snapshot:
      continue
    step = row.get('step')
    if step is None:
      step = step_from_name(snapshot)
    if step is None:
      continue
    row = dict(row, snapshot=snapshot, step=step)
    snaps.append(row)
  if not snaps:
    for path in sorted(glob.glob(os.path.join(snapshots_dir, 'step*'))):
      if not os.path.exists(os.path.join(path, 'done')):
        continue
      step = step_from_name(path)
      if step is not None:
        snaps.append(dict(step=step, snapshot=path, src_folder='', wall_time=0))
  snaps = sorted(snaps, key=lambda s: s['step'])
  out = []
  for m in sorted(milestones):
    if not snaps:
      out.append(dict(milestone=m, step=None, snapshot=None, abs_error=None))
      continue
    best = min(snaps, key=lambda s: abs(s['step'] - m))
    out.append(dict(milestone=m, step=best['step'],
                    snapshot=localize(best['snapshot']),
                    abs_error=abs(best['step'] - m)))
  return out


def resolve_checkpoints(run_logdir, milestones, snapshots_dir='',
                        checkpoints=()):
  """Return [(name, ckpt_dir, exact_step, milestone)] for offline dumps.

  Shared by the per-checkpoint inference scripts (probing/latents.py,
  probing/latent_uq.py): explicit checkpoint dirs win; otherwise milestones
  map to the nearest retained snapshots in <run_logdir>/ckpt_snapshots.
  Stdlib-only, so plan/dry-run paths need no JAX import.
  """
  out = []
  if checkpoints:
    for ckpt in checkpoints:
      live = _read_latest(ckpt)
      if live is not None:
        # A live ckpt/ dir: step.pkl lives in the save folder the `latest`
        # pointer names, so resolve it to keep the step<exact> output layout.
        ckpt = live[0]
      step = None
      step_pkl = os.path.join(ckpt, 'step.pkl')
      if os.path.exists(step_pkl):
        with open(step_pkl, 'rb') as f:
          step = int(pickle.load(f))
      name = f'step{step:012d}' if step is not None else \
          os.path.basename(os.path.normpath(ckpt))
      out.append((name, ckpt, step, None))
    return out
  snapshots_dir = snapshots_dir or os.path.join(run_logdir, 'ckpt_snapshots')
  rows = nearest_snapshots(snapshots_dir, milestones)
  gaps = [b - a for a, b in zip(sorted(milestones), sorted(milestones)[1:])]
  spacing = min(gaps) if gaps else max(milestones)
  seen = set()
  for r in rows:
    if r['snapshot'] is None:
      print(f'WARNING: no snapshot near milestone {r["milestone"]}; skipped.')
      continue
    if r['abs_error'] > spacing / 2:
      print(f'WARNING: milestone {r["milestone"]} maps to step {r["step"]} '
            f'(gap {r["abs_error"]} > half the milestone spacing); use the '
            f'recorded exact_step, not the milestone, on any x-axis.')
    if r['snapshot'] in seen:
      print(f'WARNING: milestone {r["milestone"]} maps to an already-selected '
            f'snapshot (step {r["step"]}); skipped duplicate.')
      continue
    seen.add(r['snapshot'])
    out.append((f'step{r["step"]:012d}', r['snapshot'], r['step'],
                r['milestone']))
  return out


def do_select(snapshots_dir, milestones):
  rows = nearest_snapshots(snapshots_dir, milestones)
  with open(os.path.join(snapshots_dir, 'nearest.json'), 'w') as f:
    json.dump(rows, f, indent=2)
  print(f'{"milestone":>10}  {"actual":>10}  {"|err|":>8}  snapshot')
  for r in rows:
    step = '-' if r['step'] is None else r['step']
    err = '-' if r['abs_error'] is None else r['abs_error']
    print(f'{r["milestone"]:>10}  {step:>10}  {err:>8}  {r["snapshot"]}')
  return rows


def main():
  args = parse_args()
  snapshots_dir = args.snapshots_dir or os.path.join(
      args.run_logdir, 'ckpt_snapshots')
  os.makedirs(snapshots_dir, exist_ok=True)

  if args.select:
    do_select(snapshots_dir, args.milestones)
    return

  ckpt_dir = _ckpt_dir(args.run_logdir)
  manifest = _load_manifest(snapshots_dir)
  manifest['run_logdir'] = args.run_logdir
  seen_folders = {s['src_folder'] for s in manifest['snapshots']}
  print(f'Watching {ckpt_dir}\n  -> {snapshots_dir}')
  print(f'  poll={args.poll_seconds}s  stop_step={args.stop_step or "never"}')

  last_step = -1
  stopping = False
  while True:
    res = _read_latest(ckpt_dir)
    if res is not None:
      folder, name = res
      if folder not in seen_folders:
        step = _read_step(folder)
        if step is not None:
          dst = os.path.join(snapshots_dir, f'step{step:012d}')
          if _copy_snapshot(folder, dst):
            seen_folders.add(folder)
            manifest['snapshots'].append(dict(
                step=step, src_folder=folder, snapshot=dst,
                wall_time=time.time()))
            _save_manifest(snapshots_dir, manifest)
            print(f'[{time.strftime("%H:%M:%S")}] snapshot step={step} -> {dst}')
            last_step = max(last_step, step)

    if stopping:
      break
    if args.stop_file and os.path.exists(args.stop_file):
      print('Stop file seen; final sweep then exit.')
      stopping = True
      continue
    if args.stop_step and last_step >= args.stop_step:
      print(f'Reached stop_step ({last_step} >= {args.stop_step}); '
            'one grace poll then exit.')
      stopping = True
      time.sleep(args.poll_seconds)  # grace: catch the final save
      continue
    time.sleep(args.poll_seconds)

  rows = do_select(snapshots_dir, args.milestones)
  worst = max((r['abs_error'] for r in rows if r['abs_error'] is not None),
              default=None)
  n = len(manifest['snapshots'])
  print(f'Done. {n} snapshots retained; worst milestone gap = {worst} steps.')


if __name__ == '__main__':
  main()
