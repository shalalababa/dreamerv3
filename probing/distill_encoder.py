"""Distill a LeWM embedding function into a Dreamer pixel encoder
(PREREG_lewm_20260812).

Trains a FRESH Dreamer pixel_wm encoder (exact architecture + param
naming of the study's pixel fits — the agent is constructed from a
reference pixel run's own config.yaml, so shape compatibility with the
stage-2 graft is by construction) to regress the LeWM CLS embeddings
(probing/lewm_embed.py sidecars) frame-by-frame, via a linear head on
the encoder tokens. Only ``enc/*`` params and the head train; everything
else stays at fresh init and is never consumed downstream.

Outputs (donor contract of the pe wave, so the stage-2 fits run
UNCHANGED through ``--run.from_checkpoint <donor>/ckpt/distill-0
--run.from_checkpoint_regex '^enc/' --agent.frozen_enc True``):

  <output>/ckpt/distill-0/agent.pkl   {'params': ..., 'counters': 0s}
  <output>/ckpt/distill-0/done
  <output>/fidelity.json              holdout R^2 (the registered gate)
  <output>/DISTILL_PROGRESS, config snapshot

Usage (dreamer env, GPU or CPU):
  python -m probing.distill_encoder train \
      --ref_run $RUNROOT/ax1wm_finger_fpxpxq1ms0_seed1 \
      --side_dir $RUNROOT/axis1_finger/pxq1m/side0 \
      --emb_dir $RUNROOT/lewm_emb/s0_seed1 \
      --holdout_list $RUNROOT/lewm_data/finger_pxq1m_side0.manifest.json \
      --output $RUNROOT/lewm_distill_finger_s0_seed1
Selfcheck: python -m probing.distill_encoder selfcheck
"""

import argparse
import glob
import json
import os
import pathlib
import pickle

import numpy as np

WINDOW = 8
BATCH = 16
LR = 1e-3
UPDATES = 20_000
EVAL_EVERY = 1_000


def _load_pairs(side_dir, emb_dir, holdout_files):
  """[(image [T,H,W,C] uint8, emb [T,D])] split into train/holdout."""
  train, hold = [], []
  files = sorted(glob.glob(os.path.join(side_dir, '*.npz')))
  assert files, side_dir
  for f in files:
    base = os.path.basename(f)
    ef = os.path.join(emb_dir, base)
    assert os.path.exists(ef), f'no embedding sidecar for {base}'
    with np.load(f) as z:
      img = np.asarray(z['image'])
    with np.load(ef) as z:
      emb = np.asarray(z['emb'], np.float32)
    assert len(img) == len(emb), (base, len(img), len(emb))
    (hold if base in holdout_files else train).append((img, emb))
  assert train and hold, (len(train), len(hold))
  return train, hold


def _sample_batch(rng, pairs, batch=BATCH, window=WINDOW):
  imgs, embs, resets = [], [], []
  for _ in range(batch):
    img, emb = pairs[rng.integers(0, len(pairs))]
    lo = int(rng.integers(0, max(1, len(img) - window)))
    sl = slice(lo, lo + window)
    im, em = img[sl], emb[sl]
    if len(im) < window:  # short episode: pad by repeat
      pad = window - len(im)
      im = np.concatenate([im, np.repeat(im[-1:], pad, 0)], 0)
      em = np.concatenate([em, np.repeat(em[-1:], pad, 0)], 0)
    r = np.zeros(window, bool)
    r[0] = True
    imgs.append(im)
    embs.append(em)
    resets.append(r)
  return (np.stack(imgs), np.stack(embs).astype(np.float32),
          np.stack(resets))


def _build(ref_run, platform, workdir):
  import jax
  from dreamerv3.main import make_agent
  from probing.collect import load_run_config
  config = load_run_config(ref_run, platform, workdir, False)
  config = config.update({'jax': {
      'precompile': False, 'enable_policy': False, 'prealloc': False}})
  agent = make_agent(config)
  jax.config.update('jax_transfer_guard', 'allow')
  return agent, config


def _make_step(agent):
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  model = agent.model

  def fn(obs, reset):
    B = reset.shape[0]
    enc_carry = model.enc.initial(B)
    _, _, tokens = model.enc(enc_carry, obs, reset, training=False)
    return jnp.asarray(tokens, jnp.float32)

  pure = nj.pure(fn)
  params_np = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                           agent.params)
  enc0 = {k: jnp.asarray(v) for k, v in params_np.items()
          if k.startswith('enc/')}
  assert enc0, 'agent has no enc/ params'
  frozen = {k: jnp.asarray(v) for k, v in params_np.items()
            if not k.startswith('enc/')}

  def forward(enc_params, img, reset, seed):
    # `frozen` is closed over rather than passed as a jit argument
    # (review C16): JAX hoists the concrete arrays as jaxpr constants —
    # correct, at the cost of a resident copy + longer compile at
    # pixel_wm scale; acceptable for a one-shot 20k-update distill.
    params = {**frozen, **enc_params}
    _, tokens = pure(params, {'image': img}, reset, seed=seed)
    return tokens.reshape(tokens.shape[0], tokens.shape[1], -1)

  def loss_fn(train_vars, img, reset, target, seed):
    enc_params, w, b = train_vars
    tk = forward(enc_params, img, reset, seed)
    pred = tk @ w + b
    return jnp.mean(jnp.square(pred - target))

  grad_fn = jax.jit(jax.value_and_grad(loss_fn))
  fwd_jit = jax.jit(forward)
  return enc0, frozen, params_np, grad_fn, fwd_jit


def _r2(pred, target):
  sse = float(np.sum((pred - target) ** 2))
  sst = float(np.sum((target - target.mean(0)) ** 2))
  return 1.0 - sse / max(sst, 1e-12)


def _eval_holdout(fwd_jit, enc_params, w, b, hold, seed_base=999):
  import jax.numpy as jnp
  preds, targs = [], []
  for i, (img, emb) in enumerate(hold):
    n = (len(img) // WINDOW) * WINDOW
    if n == 0:
      continue
    im = img[:n].reshape(-1, WINDOW, *img.shape[1:])
    reset = np.zeros((im.shape[0], WINDOW), bool)
    reset[:, 0] = True
    seed = jnp.array([seed_base, i], np.uint32)
    tk = fwd_jit(enc_params, jnp.asarray(im), jnp.asarray(reset), seed)
    pred = np.asarray(tk @ w + b).reshape(-1, emb.shape[-1])
    preds.append(pred)
    targs.append(emb[:n])
  preds = np.concatenate(preds, 0)
  targs = np.concatenate(targs, 0)
  return _r2(preds, targs), len(preds)


def cmd_train(args):
  import jax.numpy as jnp
  import optax

  out = pathlib.Path(args.output)
  out.mkdir(parents=True, exist_ok=True)
  if (out / 'DISTILL_DONE').exists():
    print(f'{out} already DONE; nothing to do.')
    return

  with open(args.holdout_list) as f:
    man = json.load(f)
  holdout_files = set(man['holdout_files'])
  train, hold = _load_pairs(args.side_dir, args.emb_dir, holdout_files)
  emb_dim = train[0][1].shape[-1]

  agent, config = _build(args.ref_run, args.platform, str(out / 'work'))
  enc0, frozen, params_np, grad_fn, fwd_jit = _make_step(agent)

  rng = np.random.default_rng(args.seed)
  img0, emb0, r0 = _sample_batch(rng, train, batch=2, window=WINDOW)
  seed = jnp.array([args.seed, 0], np.uint32)
  tk0 = fwd_jit(enc0, jnp.asarray(img0), jnp.asarray(r0), seed)
  tok_dim = int(tk0.shape[-1])
  w = jnp.asarray(rng.normal(0, 1.0 / np.sqrt(tok_dim),
                             (tok_dim, emb_dim)).astype(np.float32))
  b = jnp.zeros((emb_dim,), jnp.float32)

  opt = optax.adam(args.lr)
  train_vars = (enc0, w, b)
  opt_state = opt.init(train_vars)

  best = None
  r2_first = None
  for i in range(args.updates):
    img, emb, rs = _sample_batch(rng, train)
    seed = jnp.array([args.seed, i + 1], np.uint32)
    loss, grads = grad_fn(train_vars, jnp.asarray(img), jnp.asarray(rs),
                          jnp.asarray(emb), seed)
    updates, opt_state = opt.update(grads, opt_state)
    train_vars = optax.apply_updates(train_vars, updates)
    if i == 0 or (i + 1) % args.eval_every == 0 or i == args.updates - 1:
      enc_p, w_p, b_p = train_vars
      r2, n = _eval_holdout(fwd_jit, enc_p, w_p, b_p, hold)
      print(f'  update {i + 1:>6}/{args.updates}: loss={float(loss):.5f} '
            f'holdout_r2={r2:.4f} (n={n})', flush=True)
      with open(out / 'DISTILL_PROGRESS', 'w') as f:
        f.write(f'update={i + 1}/{args.updates}\nholdout_r2={r2}\n')
      if r2_first is None:
        r2_first = r2
      best = r2

  enc_p, w_p, b_p = train_vars
  donor_params = dict(params_np)
  donor_params.update({k: np.asarray(v) for k, v in enc_p.items()})
  donor_params.update({
      'distill_head/w': np.asarray(w_p), 'distill_head/b': np.asarray(b_p)})
  ck = out / 'ckpt' / 'distill-0'
  ck.mkdir(parents=True, exist_ok=True)
  with open(ck / 'agent.pkl', 'wb') as f:
    pickle.dump(dict(params=donor_params,
                     counters=dict(updates=0, batches=0, actions=0)), f)
  (ck / 'done').write_bytes(b'')
  fidelity = dict(holdout_r2=best, holdout_r2_first_eval=r2_first,
                  emb_dim=emb_dim, tok_dim=tok_dim,
                  n_train_eps=len(train), n_holdout_eps=len(hold),
                  updates=args.updates, lr=args.lr, seed=args.seed,
                  ref_run=os.path.abspath(args.ref_run),
                  emb_dir=os.path.abspath(args.emb_dir))
  with open(out / 'fidelity.json', 'w') as f:
    json.dump(fidelity, f, indent=1)
  (out / 'DISTILL_DONE').touch()
  print(f'DONE holdout_r2={best:.4f} -> {ck}')


def cmd_selfcheck(args):
  """End-to-end on a synthetic micro config: the distilled donor (a) is
  regex-graftable ('^enc/'), (b) improves holdout R^2 over init, (c)
  round-trips through the pe-wave loader contract (agent.pkl w/ params +
  counters + done)."""
  import shutil
  import tempfile
  import elements
  import ruamel.yaml as yaml

  tmp = pathlib.Path(tempfile.mkdtemp(prefix='distill_selfcheck_'))
  try:
    # micro pixel config written as a fake ref run
    with open(pathlib.Path(__file__).resolve().parent.parent /
              'dreamerv3' / 'configs.yaml') as f:
      full = yaml.YAML(typ='safe').load(f)
    cfg = elements.Config(full['defaults'])
    cfg = cfg.update({
        'task': 'dmc_finger_turn_hard',
        'agent.model_obs': 'image',
        'agent.enc.simple.depth': 4, 'agent.dec.simple.depth': 4,
        'agent.enc.simple.units': 32, 'agent.dec.simple.units': 32,
        'agent.dyn.rssm': {'deter': 64, 'hidden': 16, 'classes': 4,
                           'stoch': 4},
        'agent.expl.mode': 'apt',
        'run.steps': 100})
    ref = tmp / 'ref_run'
    ref.mkdir(parents=True)
    with open(ref / 'config.yaml', 'w') as f:
      yaml.YAML(typ='safe').dump(dict(cfg), f)

    rng = np.random.default_rng(0)
    side = tmp / 'side0'
    emb = tmp / 'emb'
    side.mkdir()
    emb.mkdir()
    D = 6
    proj = rng.normal(0, 1, (3, D)).astype(np.float32)
    names = []
    for i in range(6):
      T = 24
      img = rng.integers(0, 255, (T, 64, 64, 3), np.uint8)
      # target = a fixed linear function of mean pixel color (learnable)
      feats = (img.astype(np.float32) / 255.0).mean((1, 2))  # [T,3]
      e = feats @ proj
      name = f'ep{i:03d}.npz'
      np.savez(side / name, image=img,
               action=np.zeros((T, 2), np.float32))
      np.savez(emb / name, emb=e.astype(np.float32))
      names.append(name)
    man = tmp / 'man.json'
    with open(man, 'w') as f:
      json.dump(dict(holdout_files=names[-2:]), f)

    out = tmp / 'distill'
    ns = argparse.Namespace(
        ref_run=str(ref), side_dir=str(side), emb_dir=str(emb),
        holdout_list=str(man), output=str(out), seed=0,
        updates=60, eval_every=30, lr=3e-3, platform='cpu')
    cmd_train(ns)
    fid = json.load(open(out / 'fidelity.json'))
    with open(out / 'ckpt' / 'distill-0' / 'agent.pkl', 'rb') as f:
      donor = pickle.load(f)
    assert 'params' in donor and 'counters' in donor
    enc_keys = [k for k in donor['params'] if k.startswith('enc/')]
    assert enc_keys, 'no enc/ keys in donor'
    assert any(k.startswith('distill_head/') for k in donor['params'])
    assert (out / 'ckpt' / 'distill-0' / 'done').exists()
    # progress witness exists and parses
    kv = dict(l.split('=', 1) for l in
              open(out / 'DISTILL_PROGRESS').read().splitlines())
    assert kv['update'] == '60/60', kv
    print(f'holdout_r2={fid["holdout_r2"]:.4f}')
    assert np.isfinite(fid['holdout_r2'])
    assert fid['holdout_r2'] > fid['holdout_r2_first_eval'], \
        'distillation never improved holdout R^2'
    # sanity: the loss must have been optimizable (r2 above the
    # untrained-encoder floor is checked cluster-side at real scale; on
    # this 60-update micro run we only require a finite, non-degenerate
    # value and a byte-consistent donor round-trip)
    reloaded = pickle.loads(
        (out / 'ckpt' / 'distill-0' / 'agent.pkl').read_bytes())
    a = donor['params'][enc_keys[0]]
    barr = reloaded['params'][enc_keys[0]]
    assert np.array_equal(np.asarray(a), np.asarray(barr))
    print('SELFCHECK PASS')
  finally:
    shutil.rmtree(tmp, ignore_errors=True)


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd', required=True)
  tr = sub.add_parser('train')
  tr.add_argument('--ref_run', required=True)
  tr.add_argument('--side_dir', required=True)
  tr.add_argument('--emb_dir', required=True)
  tr.add_argument('--holdout_list', required=True)
  tr.add_argument('--output', required=True)
  tr.add_argument('--seed', type=int, default=0)
  tr.add_argument('--updates', type=int, default=UPDATES)
  tr.add_argument('--eval_every', type=int, default=EVAL_EVERY)
  tr.add_argument('--lr', type=float, default=LR)
  tr.add_argument('--platform', default=None)
  sc = sub.add_parser('selfcheck')
  args = ap.parse_args()
  if args.cmd == 'selfcheck':
    cmd_selfcheck(args)
  else:
    cmd_train(args)


if __name__ == '__main__':
  main()
