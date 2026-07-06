"""Collate per-run measure/ outputs into drivers.csv (PREREG_phase5a §3).

Walks pretrain_* run dirs under --runroot, reads
  measure/drivers_buffer.json          -> cov, occ_phys, occ_rew
  measure/ms<K>/probe/probe_results.csv-> fwd (rssm/imag5/state/h5),
                                           geom (rssm/posterior/state/h0),
                                           retdec (rssm/posterior/return_to_go/h0)
  measure/ms<K>/vsa.json               -> vsa (vsa_5; only if critic usable)
and writes one row per (mode, domain, seed, milestone).

Usage:
  python -m analysis.collate_drivers --runroot $RUNROOT --output analysis_out/drivers.csv
"""

import argparse
import csv
import glob
import json
import os
import re

RUN_RE = re.compile(
    r'^pretrain_(?P<mode>[a-z0-9]+)_(?P<domain>[a-z]+)_seed(?P<seed>\d+)$')

PROBE_ROWS = dict(
    fwd=('rssm', 'imag5', 'state', '5', '1'),
    geom=('rssm', 'posterior', 'state', '0', '1'),
    retdec=('rssm', 'posterior', 'return_to_go', '0', '1'),
)


def probe_metrics(path):
  out = {}
  with open(path, newline='') as f:
    for row in csv.DictReader(f):
      key = (row['model'], row['site'], row['target'], row['horizon'],
             row['receptive_field'])
      for name, want in PROBE_ROWS.items():
        if key == want and row['r2_ridge'] not in ('', None):
          out[name] = float(row['r2_ridge'])
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--runroot', required=True)
  p.add_argument('--output', required=True)
  args = p.parse_args()

  rows, missing = [], []
  for run_dir in sorted(glob.glob(os.path.join(args.runroot, 'pretrain_*'))):
    run_id = os.path.basename(run_dir)
    m = RUN_RE.match(run_id)
    if not m:
      continue
    meas = os.path.join(run_dir, 'measure')
    buf_path = os.path.join(meas, 'drivers_buffer.json')
    buffer_by_ms = {}
    if os.path.exists(buf_path):
      with open(buf_path) as f:
        buffer_by_ms = {r['milestone']: r for r in json.load(f)['milestones']}
    else:
      missing.append((run_id, 'drivers_buffer.json'))

    for ms_dir in sorted(glob.glob(os.path.join(meas, 'ms*'))):
      ms = int(os.path.basename(ms_dir)[2:])
      row = dict(mode=m['mode'], domain=m['domain'], seed=int(m['seed']),
                 milestone=ms)
      buf = buffer_by_ms.get(ms, {})
      for k in ('cov', 'occ_phys', 'occ_rew'):
        row[k] = buf.get(k, '')
      probe_csv = os.path.join(ms_dir, 'probe', 'probe_results.csv')
      if os.path.exists(probe_csv):
        row.update(probe_metrics(probe_csv))
      else:
        missing.append((f'{run_id}/ms{ms}', 'probe_results.csv'))
      vsa_path = os.path.join(ms_dir, 'vsa.json')
      if os.path.exists(vsa_path):
        with open(vsa_path) as f:
          vsa = json.load(f)
        if vsa.get('critic_usable') and vsa.get('vsa_5') is not None:
          row['vsa'] = vsa['vsa_5']
      rows.append(row)

  if not rows:
    raise SystemExit(f'No measure outputs under {args.runroot}/pretrain_*')
  cols = ['mode', 'domain', 'seed', 'milestone', 'cov', 'occ_phys',
          'occ_rew', 'fwd', 'geom', 'vsa', 'retdec']
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
      w.writerow({c: r.get(c, '') for c in cols})
  print(f'{len(rows)} rows -> {args.output}')
  for what, why in missing:
    print(f'  MISSING {why}: {what}')


if __name__ == '__main__':
  main()
