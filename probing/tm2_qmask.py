"""B5 TD-MPC2 Q-ensemble interventional mask instrument
(PREREG_trackB_tm2_20260822.md; built + selfchecked (torch-free
core) BEFORE the B5 wave's compute; the torch path is gated by the
registered cluster smokes; revised pre-freeze per review R4).

The taxonomy question: does TD-MPC2's uncertainty proxy — the
Q-ensemble std (its only ensemble; no dynamics disagreement exists,
tdmpc2_oracle_labels.py:45) — price the SE planted channels?

Design (decoder-free; TM2's encoder is feedforward, so anchors are
SINGLE states — no windows, no burn-in):
  1. Collect S=512 anchor states (per-key dicts) by stepping
     Dv3TaskEnv(task, dose='se', planted=True) under the trained
     policy (eval mode), stride-subsampled. The anchor pairs
     (s_t, a_t) with a_t = pi(s_t) captured BEFORE the step
     (R4-B1: the state acted FROM, not the successor); the t=0
     post-reset state is excluded (its t0=True planner state is
     qualitatively different). Anchors are stride-subsampled from
     ~4 episodes of one trajectory and are NOT independent — the
     anchor-level p/BCa are DIAGNOSTIC for that reason (R4-M4).
     The statistic itself is RNG-free (encode/Q are sampling-free);
     torch/np seeds are set once, before env construction, for the
     one-shot collection only (R4-M3).
  2. Freeze the BASELINE action per anchor. The action is held
     fixed across every variant — the mask must not change the
     action, or the delta confounds policy shift with Q-std shift
     (the registered estimand is the accounting, not the policy).
  3. Variant suite (rev 2 per R-A1-M12: the distractor is a pure
     exogenous AR(1), so BATCH PERMUTATION is law-preserving and
     its population delta is ZERO BY EXCHANGEABILITY for any
     statistic — it cannot be a fire channel):
       distractor MEAN-SUBSTITUTION -> the FIRE channel
         (law-changing, level-sensitive: every anchor's distractor
         replaced by the anchor-population mean vector);
       velocity MEAN-SUBSTITUTION -> the form-matched specificity
         comparator;
       distractor batch permutation -> the built-in
         EXCHANGEABILITY-NULL calibration row (population delta 0
         by construction; materially nonzero = instrument defect);
       velocity batch permutation -> the coupling teeth (velocity
         is state-coupled, so this one is informative);
       dup0 substitution == bitwise no-op (hard-asserted);
       dup1/dup2 substitution (descriptive) + FRESH-RESAMPLE.
     House RNG offsets (seed+13 perm, seed+1300+idx resamples,
     seed+17 velocity — R4-m17).
  4. Statistic per anchor and variant: z = encode(flat(variant));
     q = two_hot_inv(Q(z, a_i, return_type='all')) over the K=5
     heads; qstd_i = std over heads. Delta per anchor = masked -
     base.

Run:  python -m probing.tm2_qmask --run_logdir <tm2 run dir>
          [--ckpt ckpt_late.pt] --tdmpc2_root ...
      python -m probing.tm2_qmask --selfcheck   (torch-free core)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zlib

import numpy as np

from uncfield.se_mask import bca_interval, signflip_p

TASK = 'dmc_cheetah_run'
CHANNELS_FIRE = ('distractor', 'planted_dup0', 'planted_dup1',
                 'planted_dup2')
S_PIN = 512
STRIDE = 7          # anchor subsample stride along the eval trajectory


def parse_args(argv=None):
  p = argparse.ArgumentParser()
  p.add_argument('--run_logdir', required=False, default=None)
  p.add_argument('--tdmpc2_root', default=None)
  p.add_argument('--ckpt', default='ckpt_late.pt')
  p.add_argument('--output', default='',
                 help='default: <run>/tm2_qmask')
  p.add_argument('--n_eval', type=int, default=S_PIN)
  p.add_argument('--n_perm', type=int, default=1000)
  p.add_argument('--n_boot', type=int, default=2000)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--selfcheck', action='store_true')
  return p.parse_args(argv)


def build_variants(arrays, source_key, basesd, eps_ladder, seed):
  """The house mask suite on SINGLE-state anchors: returns
  (variants, info). arrays: {key: (S, dim)}. RNG streams use the
  house fixed offsets (R4-m17): seed+13 distractor perm,
  seed+1300+idx resamples, seed+17 velocity perm."""
  S = arrays[source_key].shape[0]
  perm = np.random.default_rng(seed + 13).permutation(S)
  variants, info = {}, {}
  for ch in CHANNELS_FIRE:
    v = dict(arrays)
    if ch == 'distractor':
      v[ch] = arrays[ch][perm]
      info[ch] = dict(form='batch_permutation',
                      fixed_points=int((perm == np.arange(S)).sum()))
    else:
      sub = arrays[source_key].copy()
      info[ch] = dict(form='source_substitution',
                      bitwise_noop=bool(np.array_equal(sub,
                                                       arrays[ch])))
      v[ch] = sub
    variants[ch] = v
  for idx in (1, 2):
    ch = f'planted_dup{idx}'
    rr = np.random.default_rng(seed + 1300 + idx)
    v = dict(arrays)
    v[ch] = (arrays[source_key]
             + rr.normal(0.0, eps_ladder[idx] * basesd,
                         arrays[source_key].shape)
             ).astype(arrays[ch].dtype)
    variants[f'{ch}_resample'] = v
    info[f'{ch}_resample'] = dict(
        form='fresh_resample', eps=float(eps_ladder[idx] * basesd))
  vperm = np.random.default_rng(seed + 17).permutation(S)
  v = dict(arrays)
  v['velocity'] = arrays['velocity'][vperm]
  variants['velocity_control'] = v
  info['velocity_control'] = dict(
      form='batch_permutation_real_key_control',
      fixed_points=int((vperm == np.arange(S)).sum()))
  # rev 2 (R-A1-M12): law-CHANGING mean-substitution channels —
  # the fire channel and its form-matched specificity comparator
  for ch, tag in (('distractor', 'distractor_meansub'),
                  ('velocity', 'velocity_meansub')):
    m = arrays[ch].mean(0, keepdims=True)
    v = dict(arrays)
    v[ch] = np.broadcast_to(m, arrays[ch].shape).astype(
        arrays[ch].dtype).copy()
    variants[tag] = v
    info[tag] = dict(form='mean_substitution',
                     sub_mean_norm=float(np.linalg.norm(m)))
  return variants, info


def qmask_core(arrays, flat_fn, qstd_fn, source_key, basesd,
               eps_ladder, seed, n_perm, n_boot):
  """Torch-free core: arrays {key: (S, dim)} numpy; flat_fn(dict) ->
  (S, D) numpy; qstd_fn((S, D)) -> (S,) numpy Q-ensemble std at the
  FROZEN baseline actions. Returns (channels dict, deltas npz dict,
  base vector)."""
  assert np.array_equal(arrays['planted_dup0'], arrays[source_key]), (
      'planted_dup0 != source — construction violated')
  variants, info = build_variants(
      arrays, source_key, basesd, eps_ladder, seed)
  base = qstd_fn(flat_fn(arrays)).astype(np.float64)
  channels, dnpz = {}, {}
  for ch, v in variants.items():
    masked = qstd_fn(flat_fn(v)).astype(np.float64)
    d = masked - base
    rec = dict(info[ch])
    chkey = zlib.crc32(ch.encode()) % 2 ** 16
    obs_m, p_red, p_inf = signflip_p(
        d, n_perm, np.random.default_rng(seed + chkey + 101))
    _, lo, hi = bca_interval(
        d, np.random.default_rng(seed + chkey + 211), n_boot)
    rec.update(delta_qstd_mean=obs_m, delta_qstd_bca=[lo, hi],
               p_reduce=p_red, p_inflate=p_inf)
    channels[ch] = rec
    dnpz[f'delta_qstd_{ch}'] = d
  return channels, dnpz, base


def run(args):
  from probing.tdmpc2_compat import (Dv3TaskEnv, add_tdmpc2_path,
                                     build_cfg, obs_keys,
                                     PLANTED_CFG)
  from embodied.envs.planted import EPS_LADDER
  assert args.run_logdir, '--run_logdir is required'
  add_tdmpc2_path(args.tdmpc2_root)
  import torch
  out_dir = args.output or os.path.join(args.run_logdir, 'tm2_qmask')
  out_json = os.path.join(out_dir, 'tm2_qmask.json')
  # one-shot: a re-submitted pass must NEVER silently replace a
  # run's numbers with a different anchor sample (R4-M2)
  assert not os.path.exists(out_json), (
      f'{out_json} exists — refusing to overwrite a recorded pass '
      f'(quarantine it explicitly if a re-run is intended)')
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(args.run_logdir, 'config.json')) as f:
    audit = json.load(f)
  assert audit['task'] == TASK and audit['dose'] == 'se' and \
      audit.get('planted'), (
      f'{args.run_logdir}: not a B5 run ({audit.get("task")}, '
      f'{audit.get("dose")}, planted={audit.get("planted")})')
  seed = int(audit['seed'])
  # seeds BEFORE any env/agent construction (R4-M2); note dm_control
  # episode initial states are OS-entropy either way — the anchor
  # sample is one-shot by the overwrite guard above, and its sha is
  # recorded below so a swap is detectable
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  env = Dv3TaskEnv(TASK, seed=seed + 10_000, dose='se', planted=True)
  cfg = build_cfg(args.tdmpc2_root or os.environ.get('TDMPC2_ROOT'),
                  TASK, int(env.observation_space.shape[0]),
                  int(env.action_space.shape[0]),
                  env.max_episode_steps,
                  # the house precedent's overrides (R4-M1): never
                  # trust the checkout default for the planner switch
                  overrides=dict(seed=args.seed, mpc=True))
  from tdmpc2 import TDMPC2
  agent = TDMPC2(cfg)
  agent.load(os.path.join(args.run_logdir, args.ckpt))
  # Q heads carry dropout in train mode — the statistic must be a
  # paired deterministic contrast (R4-M9; Tm2Oracle does the same)
  agent.model.eval()
  assert not agent.model.training
  from common import math as tm2_math

  # 1. collect anchors under the trained policy: pair (s_t, a_t)
  # with a_t computed FROM s_t, captured before the step (R4-B1);
  # skip the t=0 post-reset state (t0=True planner state)
  keys = obs_keys(TASK) + env._extra_keys
  anchors = {k: [] for k in keys}
  actions = []
  obs_flat = env.reset()
  t = 0
  total_steps = 0
  while len(actions) < args.n_eval:
    total_steps += 1
    assert total_steps < 50 * args.n_eval, (
        'collection loop runaway — env wedged?')
    raw = env.last_obs_dict
    a = agent.act(obs_flat, t0=(t == 0), eval_mode=True)
    if t > 0 and t % STRIDE == 0:
      for k in keys:
        anchors[k].append(np.asarray(raw[k], np.float32).reshape(-1))
      actions.append(np.asarray(a.detach().cpu().numpy(),
                                np.float32))
    obs_flat, r, done, info = env.step(a)
    t += 1
    if done:
      obs_flat = env.reset()
      t = 0
  arrays = {k: np.stack(v[:args.n_eval], 0)
            for k, v in anchors.items()}
  acts = np.stack(actions[:args.n_eval], 0)
  S = arrays[keys[0]].shape[0]
  # the 512 pin is the READER's gate (house form: se_apt_mask);
  # the instrument itself accepts any n_eval so the reduced-S smoke
  # can exercise the full path (R4-B2)
  assert S == args.n_eval, (S, args.n_eval)

  device = next(agent.model.parameters()).device

  def flat_fn(arrs):
    rows = [np.concatenate([arrs[k][i] for k in keys])
            for i in range(S)]
    return np.stack(rows, 0).astype(np.float32)

  a_t = torch.from_numpy(acts).to(device)

  def qstd_fn(flat_np):
    with torch.no_grad():
      obs_t = torch.from_numpy(flat_np).to(device)
      z = agent.model.encode(obs_t, None)
      logits = agent.model.Q(z, a_t, None, return_type='all')
      q = tm2_math.two_hot_inv(logits, cfg).squeeze(-1)   # (K, S)
      return q.std(0).cpu().numpy()

  channels, dnpz, base = qmask_core(
      arrays, flat_fn, qstd_fn, 'position',
      float(PLANTED_CFG['basesd']), EPS_LADDER, args.seed,
      args.n_perm, args.n_boot)
  anchor_sha = hashlib.sha256(
      flat_fn(arrays).tobytes() + acts.tobytes()).hexdigest()
  result = dict(run_logdir=os.path.abspath(args.run_logdir),
                ckpt=args.ckpt, n_eval=S, seed=args.seed,
                train_seed=seed, task=TASK, dose='se', planted=True,
                num_q=int(cfg.num_q), mpc=bool(cfg.mpc),
                dropout=float(getattr(cfg, 'dropout', -1.0)),
                extra_keys=list(env._extra_keys),
                anchor_sha256=anchor_sha,
                device=(torch.cuda.get_device_name(device)
                        if device.type == 'cuda' else str(device)),
                torch_version=torch.__version__,
                base_qstd_mean=float(base.mean()),
                estimand='population-intervention deltas on the '
                         'Q-ensemble std at FROZEN baseline actions '
                         '(a_t = pi(s_t), pre-step capture); anchors '
                         'stride-subsampled from ~4 episodes, NOT '
                         'independent — anchor p/BCa diagnostic',
                channels=channels)
  with open(out_json, 'w') as f:
    json.dump(result, f, indent=1)
  np.savez(os.path.join(out_dir, 'tm2_qmask.npz'),
           base_qstd=base, **dnpz)
  print(json.dumps({ch: round(c['delta_qstd_mean'], 6)
                    for ch, c in channels.items()}, indent=1))
  return result


def selfcheck():
  """Torch-free core mechanics on synthetic anchors: a fake qstd
  that prices ONLY the distractor slice must produce a negative
  distractor delta, a bitwise-zero dup0 delta, ~zero resample
  deltas, and determinism."""
  from embodied.envs.planted import EPS_LADDER
  rng = np.random.default_rng(0)
  S = 96
  pos = rng.normal(0, 1, (S, 8)).astype(np.float32)
  arrays = dict(
      position=pos,
      velocity=rng.normal(0, 1, (S, 9)).astype(np.float32),
      # correlated with position, so batch-permutation destroys a
      # REAL per-anchor coupling (the fake statistic prices it)
      distractor=(pos + rng.normal(0, 0.5, (S, 8))).astype(
          np.float32),
      planted_dup1=None, planted_dup2=None, planted_const=np.zeros(
          (S, 4), np.float32))
  arrays['planted_dup0'] = arrays['position'].copy()
  arrays['planted_dup1'] = (arrays['position'] + rng.normal(
      0, EPS_LADDER[1] * 0.0976, (S, 8))).astype(np.float32)
  arrays['planted_dup2'] = (arrays['position'] + rng.normal(
      0, EPS_LADDER[2] * 0.0976, (S, 8))).astype(np.float32)
  keys = ('position', 'velocity', 'distractor', 'planted_dup0',
          'planted_dup1', 'planted_dup2', 'planted_const')

  def flat_fn(arrs):
    return np.concatenate([arrs[k] for k in keys], 1)

  # fake statistic: prices the coupling between position and the
  # distractor slice (permutation destroys it -> delta negative)
  def qstd_fn(flat):
    pos = flat[:, :8]
    dis = flat[:, 17:25]
    return 1.0 + np.abs((pos * dis).mean(1))

  ch, dnpz, base = qmask_core(arrays, flat_fn, qstd_fn, 'position',
                              0.0976, EPS_LADDER, 0, 300, 300)
  assert ch['planted_dup0']['bitwise_noop']
  assert ch['planted_dup0']['delta_qstd_mean'] == 0.0
  # the COUPLED fake: permutation destroys a real coupling (the
  # velocity-analog machinery check)
  assert ch['distractor']['delta_qstd_mean'] < 0, ch['distractor']
  for c in ('planted_dup1_resample', 'planted_dup2_resample'):
    assert abs(ch[c]['delta_qstd_mean']) < 0.02, (c, ch[c])
  assert 'velocity_control' in ch
  assert ch['distractor_meansub']['form'] == 'mean_substitution'
  ch2, _, _ = qmask_core(arrays, flat_fn, qstd_fn, 'position',
                         0.0976, EPS_LADDER, 0, 300, 300)
  for c in ch:
    assert ch[c]['delta_qstd_mean'] == ch2[c]['delta_qstd_mean'], c
  # rev-2 semantics on an EXOGENOUS channel (R-A1-M12): with a
  # level-sensitive statistic that reads only the distractor,
  # permutation is EXACTLY zero-mean (same multiset re-paired)
  # while MEAN-SUBSTITUTION fires negative (law-changing)
  arrays_ex = dict(arrays)
  arrays_ex['distractor'] = rng.normal(0, 1.2, (S, 8)).astype(
      np.float32)                       # exogenous: pos-independent

  def qstd_level(flat):
    dis = flat[:, 17:25]
    return 1.0 + (dis ** 2).mean(1)

  chL, _, _ = qmask_core(arrays_ex, flat_fn, qstd_level, 'position',
                         0.0976, EPS_LADDER, 0, 300, 300)
  assert abs(chL['distractor']['delta_qstd_mean']) < 1e-12, (
      "permutation must be exactly zero-mean on a channel-only "
      "statistic")
  assert chL['distractor_meansub']['delta_qstd_mean'] < -0.5, (
      chL['distractor_meansub'])
  assert abs(chL['velocity_meansub']['delta_qstd_mean']) < 1e-12
  # channel_slices layout matches flatten order
  from probing.tdmpc2_compat import channel_slices

  class _Sp:
    def __init__(self, shape):
      self.shape = shape
  space = {k: _Sp(arrays[k].shape[1:]) for k in keys}
  sl = channel_slices(TASK, keys[2:], space)
  assert sl['position'] == (0, 8) and sl['distractor'] == (17, 25), sl
  print('tm2_qmask selfcheck PASS (torch-free core: coupling-priced '
        'fake fires negative under permutation; dup0 bitwise-zero; '
        'resample nulls ~0; velocity control present; determinism; '
        'rev-2 exogenous case: permutation EXACTLY zero-mean while '
        'MEAN-SUBSTITUTION fires (the R-A1-M12 re-spec); '
        'channel_slices layout)')


def main():
  args = parse_args()
  if args.selfcheck:
    selfcheck()
  else:
    run(args)


if __name__ == '__main__':
  main()
