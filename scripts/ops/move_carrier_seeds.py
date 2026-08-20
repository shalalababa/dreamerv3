#!/usr/bin/env python3
"""Move whole carrier seeds from running instances onto a fresh instance.

The carrier wave's primary is a per-seed double difference, so a SEED is the
unit that may move -- never a cell. This script therefore refuses to touch a
seed unless all of its still-pending cells can move together, and refuses to
move a seed that has already started anywhere.

Order of operations is the rule the SE-wave mover established: GATE THE
DESTINATION FIRST. Cells are queued on the target and counted there before a
single task is marked removed on the source. A crash in the middle leaves
work duplicated (recoverable: the loser is skipped by its run dir) rather
than lost.

Dry run by default; --commit performs it.

  usage: move_carrier_seeds.py --target 11 --moves 1:19,20 2:26,27 [--commit]
"""
import argparse, json, subprocess, sys, os

REPO = '/home/rickybao/projects/dreamerv3'
ENV = os.path.join(REPO, 'ops/state/instances.env')


def inst_addr(n):
  ip = port = None
  for line in open(ENV):
    if line.startswith(f'export IP{n}='):   ip = line.strip().split('=', 1)[1]
    if line.startswith(f'export PORT{n}='): port = line.strip().split('=', 1)[1]
  if not ip or not port:
    sys.exit(f'instance {n}: no address in {ENV}')
  return ip, port


def sh(n, cmd, check=True):
  ip, port = inst_addr(n)
  r = subprocess.run(['ssh', '-n', '-p', port, '-o', 'BatchMode=yes',
                      '-o', 'ConnectTimeout=25', f'root@{ip}', cmd],
                     capture_output=True, text=True)
  if check and r.returncode != 0:
    sys.exit(f'instance {n}: ssh failed: {r.stderr.strip()[:300]}')
  return r.stdout


def lane_state(n):
  """Per lane: active queue dir, next_index, running_index, task lines."""
  out = sh(n, r'''
    for l in 0 1 2 3 4 5 6 7; do
      qc=/workspace/dreamerv3_runs/_queue_control/lane_$l
      [ -f "$qc/active_queue" ] || continue
      qd="$qc/$(cat $qc/active_queue)"
      echo "LANE $l $(cat $qd/next_index 2>/dev/null || echo 1) $(cat $qd/running_index 2>/dev/null || echo 0)"
      nl -ba -w1 -s'@@@' "$qd/tasks.txt"
      echo "ENDLANE"
    done''')
  lanes, cur = {}, None
  for line in out.splitlines():
    if line.startswith('LANE '):
      _, l, nxt, run = line.split()
      cur = dict(lane=l, next=int(nxt), running=int(run), tasks=[])
      lanes[l] = cur
    elif line == 'ENDLANE':
      cur = None
    elif cur is not None and '@@@' in line:
      idx, text = line.split('@@@', 1)
      cur['tasks'].append((int(idx), text))
  return lanes


def seed_of(text):
  import re
  m = re.search(r'_seed(\d+)_ckpt', text)
  return int(m.group(1)) if m else None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--target', required=True)
  ap.add_argument('--target-lanes', type=int, default=8)
  ap.add_argument('--moves', nargs='+', required=True,
                  help='src:seed,seed,...')
  ap.add_argument('--commit', action='store_true')
  a = ap.parse_args()

  plan, moved = [], []
  for spec in a.moves:
    src, seeds = spec.split(':')
    seeds = {int(x) for x in seeds.split(',')}
    lanes = lane_state(src)
    if not lanes:
      sys.exit(f'instance {src}: no active queues')
    for l, st in sorted(lanes.items()):
      for idx, text in st['tasks']:
        if text.startswith('#removed#'):
          continue
        s = seed_of(text)
        if s not in seeds:
          continue
        if idx < st['next'] or idx == st['running']:
          sys.exit(f'REFUSE: instance {src} lane {l} task {idx} (seed {s}) is '
                   f'running or already dispatched -- seed {s} cannot move')
        plan.append(dict(src=src, lane=l, idx=idx, seed=s))
        moved.append(text)

  by_seed = {}
  for p in plan:
    by_seed.setdefault(p['seed'], []).append(p)
  print(f'moving {len(by_seed)} seed(s), {len(plan)} cell(s) -> instance {a.target}')
  for s in sorted(by_seed):
    n = len(by_seed[s])
    flag = '' if n == 4 else f'   <-- {n} cells, expected 4'
    print(f'  seed {s}: {n} cell(s) from inst '
          f'{",".join(sorted({p["src"] for p in by_seed[s]}))}{flag}')
  bad = [s for s, v in by_seed.items() if len(v) != 4]
  if bad:
    sys.exit(f'REFUSE: seeds {bad} would move partially; a seed moves whole or not at all')
  if not a.commit:
    print('\nDRY RUN -- nothing queued, nothing removed. Re-run with --commit.')
    return

  # --- 1. queue on the target, and COUNT there, before touching the source
  files = {l: [] for l in range(a.target_lanes)}
  for i, text in enumerate(moved):
    files[i % a.target_lanes].append(text)
  tip, tport = inst_addr(a.target)
  sh(a.target, 'mkdir -p /workspace/dreamerv3_runs/_waves/carrier_freshrep_moved')
  for l, lines in files.items():
    if not lines:
      continue
    local = f'/tmp/carrier_move_lane_{l}.cmds'
    with open(local, 'w') as f:
      f.write(f'# carrier_freshrep seeds moved to instance {a.target}, lane {l}\n')
      f.write('\n'.join(lines) + '\n')
    subprocess.run(['scp', '-q', '-P', tport, '-o', 'BatchMode=yes', local,
                    f'root@{tip}:/workspace/dreamerv3_runs/_waves/carrier_freshrep_moved/'],
                   check=True)
    out = sh(a.target,
             'source /root/dreamer_instance_helpers.sh; DV3_ABORT_ON_FAIL=0 '
             f'dv3_queue_or_add /workspace/dreamerv3_runs/_waves/carrier_freshrep_moved/'
             f'carrier_move_lane_{l}.cmds {l}')
    print(f'  target lane {l}: {len(lines)} cell(s) -> {out.strip().splitlines()[-1][:80]}')

  got = int(sh(a.target,
      'cat /workspace/dreamerv3_runs/_queue_control/lane_*/queue_*/tasks.txt '
      '/workspace/dreamerv3_runs/_queue_control/lane_*/queue_*/add_*.txt 2>/dev/null '
      '| grep -c "RUN_ID=" || true').strip() or 0)
  print(f'target now holds {got} queued cell(s)')
  if got < len(plan):
    sys.exit(f'REFUSE to remove from sources: target holds {got} < {len(plan)} moved cells')

  # --- 2. only now remove from the sources
  for src in sorted({p['src'] for p in plan}):
    for l in sorted({p['lane'] for p in plan if p['src'] == src}):
      idxs = sorted((p['idx'] for p in plan if p['src'] == src and p['lane'] == l),
                    reverse=True)
      out = sh(src, 'source /root/dreamer_instance_helpers.sh; '
                    f'dv3_remove_tasks {l} {" ".join(str(i) for i in idxs)}')
      ok = out.count('removed task')
      print(f'  inst {src} lane {l}: removed {ok}/{len(idxs)}')
      if ok != len(idxs):
        print(f'    WARNING: {out.strip()[:200]}')


if __name__ == '__main__':
  main()
