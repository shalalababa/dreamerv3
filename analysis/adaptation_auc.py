"""Extract early-adaptation AUC from adapt-run scores.jsonl (PREREG_phase5a §2).

Scans a runroot for ``adapt_<mode>_<dom>_seed<k>_ckpt<milestone>`` directories
and writes one CSV row per adapt run:

  run_id, mode, domain, seed, milestone, auc50k, auc100k, auc125k,
  n_ep_50k, n_ep_100k, n_ep_125k, final10, qc_pass

AUC_W = mean episode/score over episodes with end step <= W (fixed-length DMC
episodes, so this is the normalized area under the score-vs-step curve).
QC (pre-registered): >= 20 episodes with step <= 100000, else qc_pass=0 and
the row is also listed in exclusions.csv.

Usage:
  python -m analysis.adaptation_auc --runroot $RUNROOT --output analysis_out/
"""

import argparse
import collections
import csv
import glob
import json
import os
import re

RUN_RE = re.compile(
    r'^adapt_(?P<mode>[a-z0-9]+)_(?P<domain>[a-z]+)_seed(?P<seed>\d+)'
    r'_ckpt(?P<milestone>\d+)$')

WINDOWS = (50_000, 100_000, 125_000)
PRIMARY_W = 100_000
MIN_EPISODES = 20


def read_scores(path):
  episodes = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      episodes.append((int(rec['step']), float(rec['episode/score'])))
  episodes.sort(key=lambda x: x[0])
  return episodes


def auc_row(run_id, meta, episodes):
  row = dict(run_id=run_id, **meta)
  for w in WINDOWS:
    scores = [s for step, s in episodes if step <= w]
    row[f'auc{w // 1000}k'] = round(sum(scores) / len(scores), 4) \
        if scores else ''
    row[f'n_ep_{w // 1000}k'] = len(scores)
  tail = [s for _, s in episodes][-10:]
  row['final10'] = round(sum(tail) / len(tail), 4) if tail else ''
  row['qc_pass'] = int(row[f'n_ep_{PRIMARY_W // 1000}k'] >= MIN_EPISODES)
  return row


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--runroot', required=True,
                 help='Directory containing adapt_* run dirs.')
  p.add_argument('--output', required=True, help='Output directory.')
  args = p.parse_args()
  os.makedirs(args.output, exist_ok=True)

  rows, exclusions = [], []
  run_dirs = sorted(glob.glob(os.path.join(args.runroot, 'adapt_*')))
  for run_dir in run_dirs:
    run_id = os.path.basename(run_dir)
    m = RUN_RE.match(run_id)
    if not m:
      exclusions.append((run_id, 'unparseable run id'))
      continue
    meta = dict(mode=m['mode'], domain=m['domain'], seed=int(m['seed']),
                milestone=int(m['milestone']))
    scores_path = os.path.join(run_dir, 'scores.jsonl')
    if not os.path.exists(scores_path):
      exclusions.append((run_id, 'missing scores.jsonl'))
      continue
    row = auc_row(run_id, meta, read_scores(scores_path))
    rows.append(row)
    if not row['qc_pass']:
      exclusions.append(
          (run_id, f"only {row[f'n_ep_{PRIMARY_W // 1000}k']} episodes "
                   f'<= {PRIMARY_W} (need {MIN_EPISODES})'))

  if not rows:
    raise SystemExit(f'No adapt_* runs with scores under {args.runroot}')
  cols = list(rows[0].keys())
  out_csv = os.path.join(args.output, 'auc.csv')
  with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
  with open(os.path.join(args.output, 'exclusions.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run_id', 'reason'])
    w.writerows(exclusions)
  n_qc = sum(r['qc_pass'] for r in rows)
  print(f'{len(rows)} adapt runs -> {out_csv} '
        f'({n_qc} pass QC, {len(exclusions)} exclusion entries)')
  # 2026-08-08 (review D14): the registered QC bar (>=20 eps) admits
  # in-progress snapshots as qc_pass=1. Warn loudly on sub-modal windows so
  # incomplete rows are visible at collate time; readers must verify
  # n_ep_100k against the modal count before consuming (standing rule).
  n_by_count = collections.Counter(
      r[f'n_ep_{PRIMARY_W // 1000}k'] for r in rows)
  modal_ep = n_by_count.most_common(1)[0][0]
  short = [r for r in rows
           if r['qc_pass'] and r[f'n_ep_{PRIMARY_W // 1000}k'] < modal_ep]
  for r in short:
    print(f"WARNING sub-modal window: {r['run_id']} "
          f"n_ep_100k={r[f'n_ep_{PRIMARY_W // 1000}k']} < modal {modal_ep} "
          f"(qc_pass=1 under the registered >=20 bar)")


if __name__ == '__main__':
  main()
