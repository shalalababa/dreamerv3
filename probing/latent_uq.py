"""Latent-UQ Stage 0 dump: disagreement / rollout error / latent density.

Background (Latent-UQ idea brief, 2 Jul 2026, Sec. 6 Stage 0; EVPI theory
note App. A.4 item 5 -- the Gate D0 calibration addendum)
--------------------------------------------------------------------------
*Biased Dreams* (arXiv 2604.25416) reports that RSSM latent-ensemble
disagreement tracks the *density* of the training distribution in latent
space rather than held-out prediction error.  Stage 0 replicates-or-refutes
that on our stack: pure inference over saved checkpoints, no GPU queue
dependence (pass ``--platform cpu`` to run anywhere).  Per checkpoint and
per held-out probe window this script dumps the three raw signals the
diagnosis needs:

(a) **Disagreement** ``disag_steps (S, H)``: the repo's own U_dyn signal
    (``explore.Disag.reward`` -- variance of the one-step latent-prediction
    ensemble), evaluated along the open-loop path at the window's logged
    actions.  Column 0 is the one-step disagreement at the anchor (the
    brief's primary functional).  Only present when the checkpoint trained
    a disag ensemble (p2e runs, or task runs with ``expl.disag_task``);
    checkpoints without one still dump (b) and (c), and cross-seed
    disagreement is then computed in the analysis from ``pred_*``.
(b) **Held-out open-loop rollout error** ``err_steps (S, H)`` plus per-key
    ``err_<key> (S, H)``: state is reset at the window start, burned in for
    the probe set's ``burn_in`` steps, rolled open-loop for the
    ``eval_steps`` tail with logged actions, decoded, and compared to the
    true observations in the decoder's target space (symlog for the default
    vector heads).  ``err_steps`` is per-dimension normalized by the probe
    set's own std so keys are comparable; horizons are selected later in
    the analysis (``err_steps[:, h-1]`` = error at horizon h).
    Decoded prediction means ``pred_<key> (S, H, D)`` and targets
    ``true_<key>`` are stored for vector keys so cross-seed (identifiable,
    observation-space) disagreement needs no second encode pass.
(c) **Latent density proxy** ``density_knn (S,)``: mean distance of the
    anchor posterior mean (deter + softmax probs, the disag target space)
    to its k nearest neighbors in a reference sample of *training-buffer*
    encodings (same checkpoint, same burn-in protocol; larger = sparser).

Output layout (one directory per run x probe set), consumed by
``probing/latent_uq_analysis.py``::

    <output>/
      index.json                # checkpoints processed + probeset identity
      step<exactstep>/uq.npz    # signals above + source_label, stream_start
      step<exactstep>/meta.json

Usage
-----
Over the retained snapshot series of one Phase-4 run (milestones map to the
nearest retained snapshots, as in probing/latents.py)::

    python -m probing.latent_uq \
        --probeset $SCRATCH/probesets/cup_v1 \
        --run_logdir $RUN/pretrain_p2e_cup_seed1 \
        --milestones 100000 200000 300000 400000 500000 \
        --output $RUN/pretrain_p2e_cup_seed1/latent_uq/cup_v1

Or on explicit checkpoints (e.g. a Gate 0 pilot's final ``ckpt``)::

    python -m probing.latent_uq --probeset ... --run_logdir <RUN> \
        --checkpoints <RUN>/ckpt --output ...

The density reference defaults to ``<run_logdir>/replay``; point
``--ref_replay`` at the actual training buffer when checkpoints were copied
without it.  For probe sets intended for Stage 0, build with
``--eval_steps 16`` so the brief's horizons 1/5/15 are all available.

Validation
----------
    python -m probing.latent_uq_analysis --selfcheck   # analysis math
    python -m probing.latent_uq --probeset <set> --run_logdir <debug_run> \
        --checkpoints <debug_run>/ckpt --output /tmp/uq_smoke --platform cpu
    python -m probing.latent_uq ... --dry_run   # plan only, no JAX/agent
"""

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np

from probing import probeset as probeset_mod
from probing.checkpoint_watcher import resolve_checkpoints
from probing.latent_uq_analysis import knn_distance, np_symlog


def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--probeset', required=True,
                 help='Frozen probe-set directory (probing/probeset.py).')
  p.add_argument('--run_logdir', required=True,
                 help='DreamerV3 run dir (config.yaml).')
  p.add_argument('--output', required=True, help='Output directory.')
  p.add_argument('--milestones', type=int, nargs='+',
                 default=[100000, 200000, 300000, 400000, 500000],
                 help='Mapped to nearest retained snapshots in '
                      '<run_logdir>/ckpt_snapshots (checkpoint_watcher).')
  p.add_argument('--snapshots_dir', default='',
                 help='Default <run_logdir>/ckpt_snapshots.')
  p.add_argument('--checkpoints', nargs='+', default=[],
                 help='Explicit checkpoint dirs; overrides --milestones.')
  p.add_argument('--ref_replay', default='',
                 help='Training replay dir for the density reference '
                      '(default <run_logdir>/replay).')
  p.add_argument('--ref_windows', type=int, default=2048,
                 help='Reference encodings for the kNN density proxy.')
  p.add_argument('--knn', type=int, default=10,
                 help='k for the kNN density proxy (mean distance).')
  p.add_argument('--ep_batch', type=int, default=64,
                 help='Probe windows per forward pass.')
  p.add_argument('--platform', default='', choices=['', 'cpu', 'cuda'])
  p.add_argument('--allow_unfrozen', action='store_true')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--dry_run', action='store_true',
                 help='Resolve probe set, checkpoints, and density reference '
                      'and print the plan without loading JAX or the agent.')
  return p.parse_args()


def batched(fn, arrays, keys, reset_np, batch, seed):
  """Run a jitted pure fn over probe windows in batches; concat outputs."""
  import jax.numpy as jnp
  S = reset_np.shape[0]
  feats = {}
  for lo in range(0, S, batch):
    hi = min(lo + batch, S)
    obs = {k: jnp.asarray(arrays[k][lo:hi], np.float32) for k in keys['obs']}
    act = {k: jnp.asarray(arrays[k][lo:hi], np.float32) for k in keys['act']}
    reset = jnp.asarray(reset_np[lo:hi])
    sd = jnp.array([seed, lo + 1], np.uint32)
    out = fn(obs, act, reset, sd)
    for k, v in out.items():
      feats.setdefault(k, []).append(np.asarray(v, np.float32))
  return {k: np.concatenate(v, 0) for k, v in feats.items()}


def main():
  args = parse_args()
  arrays, manifest = probeset_mod.load(
      args.probeset, require_frozen=not args.allow_unfrozen)
  burn_in, eval_steps = manifest['burn_in'], manifest['eval_steps']
  L = manifest['length']
  S = arrays['is_first'].shape[0]
  print(f'Probe set {manifest["probeset_id"]}: {S} windows x {L} steps '
        f'({burn_in} burn-in + {eval_steps} eval)')
  if eval_steps < 15:
    print(f'WARNING: eval_steps={eval_steps} < 15; the brief\'s horizon-15 '
          f'read will be unavailable (build Stage-0 sets with '
          f'--eval_steps 16).')

  ref_replay = args.ref_replay or os.path.join(args.run_logdir, 'replay')
  n_chunks = len(probeset_mod.replay_dataset.list_chunks(ref_replay)) if (
      os.path.isdir(ref_replay)) else 0
  if not n_chunks:
    raise SystemExit(
        f'Density reference replay {ref_replay} has no .npz chunks; pass '
        f'--ref_replay pointing at the run\'s training buffer.')
  print(f'Density reference: {ref_replay} ({n_chunks} chunks, '
        f'target {args.ref_windows} windows, k={args.knn})')

  # Held-out sanity: the probe set must not come from the probed run's own
  # training data -- neither <run_logdir>/replay nor the density reference,
  # which stands in for the training buffer when checkpoints were copied
  # without their replay/.
  held_out = True
  guards = {
      os.path.realpath(os.path.join(args.run_logdir, 'replay')):
          'the replay of the run being probed',
      os.path.realpath(ref_replay):
          'the density-reference (training) replay',
  }
  for label, src in manifest['sources'].items():
    src_dir = os.path.realpath(src['replay_dir'])
    for guard_dir, what in guards.items():
      if src_dir == guard_dir or src_dir.startswith(guard_dir + os.sep):
        held_out = False
        print(f'WARNING: probe-set source {label!r} is {what}; rollout '
              f'errors are NOT held-out for this run.')
        break

  todo = resolve_checkpoints(args.run_logdir, args.milestones,
                             args.snapshots_dir, args.checkpoints)
  if not todo:
    raise SystemExit('No checkpoints to process.')
  print(f'Checkpoints: {[name for name, *_ in todo]}')

  if args.dry_run:
    print('DRY RUN: plan resolved; skipping agent load and inference.')
    return

  # Heavy imports only past the dry-run gate, so plan checks run anywhere.
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  from dreamerv3.main import make_agent
  from probing.collect import load_run_config, load_frozen_agent

  f32 = jnp.float32
  os.makedirs(args.output, exist_ok=True)
  config = load_run_config(args.run_logdir, args.platform, args.output, False)
  agent = make_agent(config)
  jax.config.update('jax_transfer_guard', 'allow')
  model = agent.model
  has_disag = model.disag is not None
  if not has_disag:
    print('NOTE: checkpoint architecture has no disag ensemble (mode '
          f'{config.agent.expl.mode!r}, '
          f'disag_task={bool(config.agent.expl.disag_task)}); '
          'dumping error/density/predictions only -- use cross-seed '
          'disagreement in the analysis.')
  dec_symlog = bool(model.dec.symlog)
  assert not model.dec.imgkeys, (
      'latent_uq supports proprio (vector) observations only, like the '
      'probe-set stack; got image keys', model.dec.imgkeys)

  exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
  obs_keys = sorted(k for k, v in agent.obs_space.items()
                    if k not in exclude and len(v.shape) <= 1)
  vec_keys = list(obs_keys)
  act_keys = sorted(agent.act_space.keys())
  keys = dict(obs=obs_keys, act=act_keys)
  missing = [k for k in obs_keys + act_keys if k not in arrays]
  assert not missing, f'Probe set lacks keys {missing} required by the agent.'

  # Reference windows from the training buffer: burn-in only, anchor at the
  # last step. Same rng protocol as the probe-set builder.
  ref_arrays, _, ref_stats = probeset_mod.collect_windows(
      ref_replay, burn_in, args.ref_windows, np.random.default_rng(args.seed),
      allow_fewer=True)
  ref_arrays = {k: np.stack(v, 0).astype(np.float32)
                if np.asarray(v[0]).dtype != bool else np.stack(v, 0)
                for k, v in ref_arrays.items()}
  n_ref = ref_arrays['is_first'].shape[0]
  print(f'Reference sample: {n_ref} windows x {burn_in} steps '
        f'from {ref_stats["streams"]} streams')

  postfeat = lambda deter, logit: jnp.concatenate([
      f32(deter),
      jax.nn.softmax(f32(logit), -1).reshape((*deter.shape[:-1], -1))], -1)

  def encode(obs, action_dict, reset):
    """Posterior features over a window (mirrors probing/latents.py)."""
    B = reset.shape[0]
    enc_carry = model.enc.initial(B)
    dyn_carry = model.dyn.initial(B)
    enc_carry, _, tokens = model.enc(enc_carry, obs, reset, training=False)
    prevact = {k: jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], 1)
               for k, v in action_dict.items()}
    _, _, post = model.dyn.observe(
        dyn_carry, tokens, prevact, reset, training=False)
    return post

  def ref_fn(obs, action_dict, reset):
    post = encode(obs, action_dict, reset)
    return {'anchor': postfeat(post['deter'][:, -1], post['logit'][:, -1])}

  def probe_fn(obs, action_dict, reset):
    B = reset.shape[0]
    b, H = burn_in, eval_steps
    post = encode(obs, action_dict, reset)
    out = {'anchor': postfeat(post['deter'][:, b - 1], post['logit'][:, b - 1])}
    # Open-loop roll from the posterior at the end of burn-in, driven by the
    # window's logged actions; prior state j corresponds to window index
    # b-1+j+1, i.e. horizon h reads prior index h-1 (probing/latents.py).
    carry = {'deter': post['deter'][:, b - 1], 'stoch': post['stoch'][:, b - 1]}
    acts = {k: v[:, b - 1:b - 1 + H] for k, v in action_dict.items()}
    _, prifeat, _ = model.dyn.imagine(carry, acts, H, training=False)
    dec_carry = model.dec.initial(B)
    _, _, recons = model.dec(
        dec_carry, prifeat, jnp.zeros((B, H), bool), training=False)
    for k in vec_keys:
      out[f'pred_{k}'] = f32(recons[k].pred())
    if has_disag:
      # Disagreement along the same path: state before horizon step j is the
      # anchor (j=0) then prior j-1, paired with the action applied at it.
      states = {
          'deter': jnp.concatenate(
              [post['deter'][:, b - 1:b], prifeat['deter'][:, :H - 1]], 1),
          'stoch': jnp.concatenate(
              [post['stoch'][:, b - 1:b], prifeat['stoch'][:, :H - 1]], 1)}
      actvec = jnp.concatenate(
          [action_dict[k][:, b - 1:b - 1 + H].reshape((B, H, -1))
           for k in act_keys], -1)
      out['disag_steps'] = f32(
          model.disag.reward(model.feat2tensor(states), actvec))
    return out

  jit_probe = jax.jit(lambda p, o, a, r, s: nj.pure(probe_fn)(p, o, a, r, seed=s))
  jit_ref = jax.jit(lambda p, o, a, r, s: nj.pure(ref_fn)(p, o, a, r, seed=s))

  # Fixed per-dim normalizers from the probe set itself (decoder target
  # space), so err_steps is comparable across checkpoints and runs.
  true_tail, norms = {}, {}
  for k in vec_keys:
    tail = np.asarray(arrays[k][:, burn_in:], np.float32).reshape(S, eval_steps, -1)
    true_tail[k] = np_symlog(tail) if dec_symlog else tail
    norms[k] = true_tail[k].reshape(-1, true_tail[k].shape[-1]).std(0) + 1e-6

  reset_probe = np.zeros((S, L), bool)
  reset_probe[:, 0] = True
  reset_ref = np.zeros((n_ref, burn_in), bool)
  reset_ref[:, 0] = True

  index = dict(
      run_logdir=os.path.abspath(args.run_logdir),
      probeset=os.path.abspath(args.probeset),
      probeset_id=manifest['probeset_id'], probeset_sha256=manifest['sha256'],
      burn_in=burn_in, eval_steps=eval_steps,
      ref_replay=os.path.abspath(ref_replay), ref_windows=n_ref,
      knn=args.knn, dec_symlog=dec_symlog, has_disag=has_disag,
      held_out=held_out, checkpoints=[])

  for name, ckpt, exact_step, milestone in todo:
    load_frozen_agent(agent, ckpt)
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                          agent.params)
    probe_out = batched(lambda o, a, r, s: jit_probe(params, o, a, r, s)[1],
                        arrays, keys, reset_probe, args.ep_batch, args.seed)
    ref_out = batched(lambda o, a, r, s: jit_ref(params, o, a, r, s)[1],
                      ref_arrays, keys, reset_ref, args.ep_batch,
                      args.seed + 1)

    feats = {}
    err_norm = []
    for k in vec_keys:
      pred = probe_out[f'pred_{k}'].reshape(S, eval_steps, -1)
      feats[f'pred_{k}'] = pred
      feats[f'true_{k}'] = true_tail[k]
      sq = (pred - true_tail[k]) ** 2
      feats[f'err_{k}'] = sq.mean(-1)
      err_norm.append(sq / (norms[k] ** 2)[None, None])
    feats['err_steps'] = np.concatenate(err_norm, -1).mean(-1)
    if has_disag:
      feats['disag_steps'] = probe_out['disag_steps']
    feats['density_knn'] = knn_distance(
        probe_out['anchor'], ref_out['anchor'], args.knn)
    feats['source_label'] = arrays['source_label']
    feats['stream_start'] = arrays['stream_start']

    step_dir = os.path.join(args.output, name)
    os.makedirs(step_dir, exist_ok=True)
    np.savez_compressed(os.path.join(step_dir, 'uq.npz'), **feats)
    meta = dict(
        checkpoint=os.path.abspath(ckpt), exact_step=exact_step,
        milestone=milestone, probeset_id=manifest['probeset_id'],
        probeset_sha256=manifest['sha256'], burn_in=burn_in,
        eval_steps=eval_steps, ref_replay=os.path.abspath(ref_replay),
        ref_windows=n_ref, knn=args.knn, dec_symlog=dec_symlog,
        has_disag=has_disag, seed=args.seed,
        disag_ens=int(config.agent.expl.disag_ens) if has_disag else 0,
        shapes={k: list(v.shape) for k, v in feats.items()})
    with open(os.path.join(step_dir, 'meta.json'), 'w') as f:
      json.dump(meta, f, indent=2)
    index['checkpoints'].append(dict(
        name=name, exact_step=exact_step, milestone=milestone,
        path=os.path.join(name, 'uq.npz')))
    print(f'  {name} (milestone {milestone}): disag='
          f'{"yes" if has_disag else "no"}, err_steps '
          f'{feats["err_steps"].shape}, density_knn mean '
          f'{feats["density_knn"].mean():.4f} -> {step_dir}')

  with open(os.path.join(args.output, 'index.json'), 'w') as f:
    json.dump(index, f, indent=2)
  print(f'Done: {len(index["checkpoints"])} checkpoints -> {args.output}')


if __name__ == '__main__':
  main()
