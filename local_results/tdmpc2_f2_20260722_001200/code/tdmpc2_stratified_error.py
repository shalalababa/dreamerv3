"""E4-style stratified error measures over TD-MPC2 fit checkpoints (F2).

Cross-family analog of ``probing/stratified_error.py`` on the frozen E4
probe set (finger_v1), evaluated per fit checkpoint with NO training:

- **probe_mse** (the decisional quantity): closed-form ridge readout of
  the CURRENT-frame reward from the FROZEN latent z_t = encode(obs_t)
  (the h=0 analog of Dreamer's reward-head NLL). A fresh probe is
  required because the free arm trains with reward_coef=0 — its native
  reward head stays at init and cannot report what the trunk contains.
  Even episodes train the probe, odd episodes evaluate (registered
  split); per-frame squared error on eval frames, stratified.
- **cons_err**: one-step latent-consistency error
  ||next(z_t, a_t) − encode(obs_{t+1})||² / latent_dim, stratified at
  the TARGET frame t+1 (the d_errin analog; both arms train this term).
- **rew_head_mse** (descriptive): the native head's
  (two_hot_inv(reward(z_t, a_t)) − r_{t+1})², stratified at t+1.
  Meaningful for aware only; recorded for free with head_trained=False.

Pairing follows the dv3 episode convention: action row t = action taken
AT obs_t; reward row t = reward on arrival at obs_t. Hence the head
target for (z_t, a_t) is reward[t+1], and the probe target for z_t is
reward[t].

The torch/model path mirrors probing/tdmpc2_offline_fit.py (build_cfg +
TDMPC2 + load_state_dict) and is validated on the cluster via
``--smoke_random_init`` (random-init model, no checkpoint touched — API
validation without revealing any fit outcome). The numpy core (split,
ridge, strata, pairing) is covered by ``selfcheck`` locally.

Usage (cluster, tdmpc2 conda env, GPU):
  python -m probing.tdmpc2_stratified_error measure \
      --run $RUNROOT/tm2wm_finger_awareq1s1_seed1 \
      --probeset $RUNROOT/e4_probesets/finger_v1 \
      --task dmc_finger_turn_hard --output <dir>/tm2e4_<run>.json
  python -m probing.tdmpc2_stratified_error collate \
      --inputs '<dir>/tm2e4_*.json' --output <csv>
  python -m probing.tdmpc2_stratified_error selfcheck
"""

import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np

from probing.tdmpc2_compat import add_tdmpc2_path, build_cfg, obs_keys

PROBESET_NPZ = 'probeset.npz'
RUN_RE = re.compile(r'tm2wm_finger_(aware|free)q1s(\d)_seed(\d+)$')
RIDGE_SCALE = 1e-3  # lambda = RIDGE_SCALE * n_train on standardized feats
STRATA = ('all', 'in_regime', 'out_regime', 'rewarded', 'unrewarded')


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def load_probeset(probeset_dir):
  """Frozen E4 probe set -> (arrays, manifest); sha + FROZEN enforced."""
  npz_path = os.path.join(probeset_dir, PROBESET_NPZ)
  with open(os.path.join(probeset_dir, 'manifest.json')) as f:
    manifest = json.load(f)
  digest = sha256_file(npz_path)
  if digest != manifest['sha256']:
    raise SystemExit(f'{probeset_dir}: {PROBESET_NPZ} sha256 mismatch.')
  marker = os.path.join(probeset_dir, 'FROZEN')
  if not os.path.exists(marker):
    raise SystemExit(f'{probeset_dir} is not FROZEN.')
  with open(marker) as f:
    if f.read().strip() != digest:
      raise SystemExit(f'{probeset_dir}: FROZEN marker does not match file.')
  arrays = {k: np.asarray(v) for k, v in np.load(npz_path).items()}
  return arrays, manifest


def standard_masks(in_regime, rewarded):
  in_r, rew = np.asarray(in_regime, bool), np.asarray(rewarded, bool)
  return {'all': np.ones_like(in_r), 'in_regime': in_r,
          'out_regime': ~in_r, 'rewarded': rew, 'unrewarded': ~rew}


def stratum_means(err, masks):
  """err [M] with NaN allowed; masks {name: [M] bool} -> per-stratum
  mean and count over finite entries."""
  err = np.asarray(err, np.float64)
  out = {}
  for name, m in masks.items():
    sel = err[np.asarray(m, bool)]
    sel = sel[np.isfinite(sel)]
    out[name] = dict(mean=float(sel.mean()) if len(sel) else None,
                     n=int(len(sel)))
  return out


def ridge_probe(z_tr, y_tr, z_ev, y_ev):
  """Closed-form ridge z -> y. Standardize on train (std floor 1e-6),
  center y on train, lambda = RIDGE_SCALE * n_train. Returns per-frame
  squared error on eval and eval R^2. Fully deterministic."""
  z_tr = np.asarray(z_tr, np.float64)
  z_ev = np.asarray(z_ev, np.float64)
  y_tr = np.asarray(y_tr, np.float64)
  y_ev = np.asarray(y_ev, np.float64)
  mu, sd = z_tr.mean(0), z_tr.std(0)
  sd = np.maximum(sd, 1e-6)
  xt = (z_tr - mu) / sd
  xe = (z_ev - mu) / sd
  ym = y_tr.mean()
  lam = RIDGE_SCALE * len(xt)
  d = xt.shape[1]
  beta = np.linalg.solve(xt.T @ xt + lam * np.eye(d), xt.T @ (y_tr - ym))
  pred = xe @ beta + ym
  err = (pred - y_ev) ** 2
  denom = ((y_ev - y_ev.mean()) ** 2).sum()
  r2 = float(1.0 - err.sum() / denom) if denom > 0 else None
  return err, r2


def split_episodes(n_eps):
  """Registered split: even episode indices train, odd evaluate."""
  idx = np.arange(n_eps)
  return idx[idx % 2 == 0], idx[idx % 2 == 1]


def episode_obs_tensor(arrays, task):
  """Probe arrays {key: [N, T, ...]} -> obs [N, T, D] canonical concat."""
  cols = []
  for k in obs_keys(task):
    v = np.asarray(arrays[k], np.float32)
    cols.append(v.reshape(v.shape[0], v.shape[1], -1))
  return np.concatenate(cols, 2)


def measure_metrics(z, arrays):
  """Numpy side of the measure: probe + strata, given latents z [N,T,L]
  plus per-frame model outputs already computed by the torch stage
  (cons_err [N,T-1], rew_head_err [N,T-1] aligned to target t+1).
  Returns the metrics dict (probe only; callers merge model metrics)."""
  n, t, _ = z.shape
  rew = np.asarray(arrays['reward'], np.float64)
  in_r = np.asarray(arrays['in_regime'], bool)
  rewarded = np.asarray(arrays['rewarded'], bool)
  tr, ev = split_episodes(n)
  err, r2 = ridge_probe(
      z[tr].reshape(-1, z.shape[2]), rew[tr].reshape(-1),
      z[ev].reshape(-1, z.shape[2]), rew[ev].reshape(-1))
  masks = standard_masks(in_r[ev].reshape(-1), rewarded[ev].reshape(-1))
  return dict(probe_mse=stratum_means(err, masks), probe_r2_eval=r2,
              n_train_frames=int(len(tr) * t), n_eval_frames=int(len(ev) * t))


def target_strata(err_nt1, arrays):
  """Stratify [N, T-1] per-frame errors at the TARGET frame t+1."""
  in_r = np.asarray(arrays['in_regime'], bool)[:, 1:]
  rewarded = np.asarray(arrays['rewarded'], bool)[:, 1:]
  masks = standard_masks(in_r.reshape(-1), rewarded.reshape(-1))
  return stratum_means(np.asarray(err_nt1).reshape(-1), masks)


def cmd_measure(args):
  import torch
  arrays, manifest = load_probeset(args.probeset)
  if manifest['task'] != args.task:
    raise SystemExit(f"probeset task {manifest['task']} != {args.task}")
  run_dir = args.run.rstrip('/')
  m = RUN_RE.search(os.path.basename(run_dir))
  if not m and not args.smoke_random_init:
    raise SystemExit(f'run dir does not parse: {run_dir}')
  with open(os.path.join(run_dir, 'config.yaml')) as f:
    audit = json.load(f)

  obs = episode_obs_tensor(arrays, args.task)          # [N, T, D]
  act = np.asarray(arrays['action'], np.float32)       # [N, T, A] dv3 rows
  rew = np.asarray(arrays['reward'], np.float32)
  n, t, d = obs.shape

  add_tdmpc2_path(args.tdmpc2_root)
  cfg = build_cfg(args.tdmpc2_root or os.environ.get('TDMPC2_ROOT'),
                  args.task, d, act.shape[2], t - 1, dict(seed=0))
  for k in ('model_size', 'consistency_coef'):
    if k in audit:
      assert getattr(cfg, k) == audit[k] or args.smoke_random_init, \
          (k, getattr(cfg, k), audit[k])
  from tdmpc2 import TDMPC2
  from common import math as tm2_math
  agent = TDMPC2(cfg)
  smoke = bool(args.smoke_random_init)
  if not smoke:
    state = torch.load(os.path.join(run_dir, 'tm2_ckpt.pt'),
                       weights_only=False)
    agent.model.load_state_dict(state['model'])
  model = agent.model.eval()
  dev = next(model.parameters()).device

  with torch.no_grad():
    z_all = []
    flat = torch.from_numpy(obs.reshape(-1, d)).to(dev)
    for i in range(0, len(flat), args.batch):
      z_all.append(model.encode(flat[i:i + args.batch], None))
    z = torch.cat(z_all, 0).reshape(n, t, -1)
    a = torch.from_numpy(act).to(dev)
    zt, at = z[:, :-1].reshape(-1, z.shape[2]), a[:, :-1].reshape(-1, act.shape[2])
    z_next_pred, rhat, cons, rhead = [], [], [], []
    for i in range(0, len(zt), args.batch):
      zi, ai = zt[i:i + args.batch], at[i:i + args.batch]
      z_next_pred.append(model.next(zi, ai, None))
      rhat.append(tm2_math.two_hot_inv(model.reward(zi, ai, None), cfg))
    znp = torch.cat(z_next_pred, 0)
    ztgt = z[:, 1:].reshape(-1, z.shape[2])
    cons = ((znp - ztgt) ** 2).mean(1).reshape(n, t - 1).cpu().numpy()
    rhat = torch.cat(rhat, 0).reshape(n, t - 1).cpu().numpy()
    z_np = z.cpu().numpy()
  rhead = (rhat - rew[:, 1:]) ** 2

  metrics = measure_metrics(z_np, arrays)
  metrics['cons_err'] = target_strata(cons, arrays)
  metrics['rew_head_mse'] = target_strata(rhead, arrays)
  out = dict(
      run_id=os.path.basename(run_dir),
      arm=m.group(1) if m else audit.get('arm'),
      side=int(m.group(2)) if m else None,
      seed=int(m.group(3)) if m else None,
      head_trained=bool(float(audit.get('reward_coef', 0)) > 0),
      smoke=smoke, task=args.task,
      probeset=dict(id=manifest['probeset_id'], sha256=manifest['sha256']),
      cfg_echo=dict(model_size=audit.get('model_size'),
                    consistency_coef=audit.get('consistency_coef'),
                    reward_coef=audit.get('reward_coef'),
                    value_coef=audit.get('value_coef'),
                    latent_dim=int(z_np.shape[2])),
      metrics=metrics)
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, 'w') as f:
    json.dump(out, f, indent=1)
  pm = metrics['probe_mse']
  print(f"{out['run_id']}: probe_mse in={pm['in_regime']['mean']:.5f} "
        f"out={pm['out_regime']['mean']:.5f} r2={metrics['probe_r2_eval']} "
        f"cons_in={metrics['cons_err']['in_regime']['mean']:.5f} "
        f"{'SMOKE' if smoke else ''} -> {args.output}")


def cmd_collate(args):
  import csv
  rows = []
  for path in sorted(glob.glob(args.inputs)):
    with open(path) as f:
      d = json.load(f)
    row = dict(run_id=d['run_id'], arm=d['arm'], side=d['side'],
               seed=d['seed'], head_trained=int(d['head_trained']),
               smoke=int(d['smoke']), probeset_sha=d['probeset']['sha256'],
               probe_r2=d['metrics']['probe_r2_eval'])
    for metric in ('probe_mse', 'cons_err', 'rew_head_mse'):
      for st in STRATA:
        cell = d['metrics'][metric][st]
        row[f'{metric}_{st}'] = cell['mean']
        row[f'n_{metric}_{st}'] = cell['n']
    rows.append(row)
  assert rows, f'no measure jsons match {args.inputs}'
  with open(args.output, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  print(f'{len(rows)} rows -> {args.output}')


def selfcheck():
  rng = np.random.default_rng(0)
  n, t, d = 8, 60, 16
  z = rng.normal(0, 1, (n, t, d)).astype(np.float64)
  w = np.zeros(d)
  w[:2] = (1.0, -0.5)
  rew = z @ w + rng.normal(0, 0.05, (n, t))
  in_r = rew > 0.5
  arrays = dict(reward=rew, in_regime=in_r, rewarded=rew > 0.8)
  m = measure_metrics(z, arrays)
  assert m['probe_mse']['in_regime']['mean'] < 0.02, m['probe_mse']
  assert m['probe_r2_eval'] > 0.95, m['probe_r2_eval']
  # Permuted latents destroy decodability.
  zp = z.reshape(-1, d)[rng.permutation(n * t)].reshape(n, t, d)
  mp = measure_metrics(zp, arrays)
  assert mp['probe_mse']['all']['mean'] > 10 * m['probe_mse']['all']['mean']
  # Determinism.
  m2 = measure_metrics(z, arrays)
  assert m2['probe_mse']['all']['mean'] == m['probe_mse']['all']['mean']
  # Split: even train / odd eval, disjoint and exhaustive.
  tr, ev = split_episodes(9)
  assert list(tr) == [0, 2, 4, 6, 8] and list(ev) == [1, 3, 5, 7]
  # Target-frame stratification drops frame 0 and aligns to t+1.
  err = np.arange(n * (t - 1), dtype=np.float64).reshape(n, t - 1)
  ts = target_strata(err, arrays)
  assert ts['all']['n'] == n * (t - 1)
  masks = standard_masks(in_r[:, 1:].reshape(-1),
                         (rew > 0.8)[:, 1:].reshape(-1))
  assert ts['in_regime']['n'] == int(masks['in_regime'].sum())
  # Ridge ~ lstsq in the small-lambda limit.
  xt = rng.normal(0, 1, (500, 4))
  yt = xt @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(0, 0.01, 500)
  e, r2 = ridge_probe(xt, yt, xt, yt)
  assert r2 > 0.999
  print('selfcheck PASS: probe recovers planted direction, permutation '
        'destroys it, deterministic, split/strata/ridge exact')


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  sub = ap.add_subparsers(dest='cmd', required=True)
  me = sub.add_parser('measure')
  me.add_argument('--run', required=True)
  me.add_argument('--probeset', required=True)
  me.add_argument('--task', required=True)
  me.add_argument('--output', required=True)
  me.add_argument('--tdmpc2_root', default=None)
  me.add_argument('--batch', type=int, default=4096)
  me.add_argument('--smoke_random_init', action='store_true')
  co = sub.add_parser('collate')
  co.add_argument('--inputs', required=True, help='glob of measure jsons')
  co.add_argument('--output', required=True)
  sub.add_parser('selfcheck')
  args = ap.parse_args()
  if args.cmd == 'selfcheck':
    selfcheck()
  elif args.cmd == 'collate':
    cmd_collate(args)
  else:
    cmd_measure(args)


if __name__ == '__main__':
  main()
