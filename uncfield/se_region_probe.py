"""SE region-resolved disagreement probe (mechanism instrument for the
avoidance program; PREREG_nfi_avoidance_20260821.md §4). Per-run,
built + selfchecked BEFORE the A1 wave's compute.

Extends the frozen se_probe's anchor pass by SAVING the per-state
per-key decoder-projected ensemble variance instead of aggregating it,
together with each anchor state's raw region coordinate
(position[region_index] at the window's last step). This lets the
frozen avoidance reader decompose disagreement by spatial region
(high- vs low-amplitude side of the registered threshold) for real
keys vs the planted channels — the data the avoidance-mechanism
question needs and se_probe discards.

Identical mechanics to se_probe where shared (same collect_windows rng
=> same window sample at the same seed; same postsplit/decode/
member-variance pipeline; same normalizer convention incl. the 0.05x
real-scale floor). No rollouts, no permutation tests — this is a
measurement dump; statistics live in the frozen reader.

Run:  python -m uncfield.se_region_probe --run_logdir <dir>
      python -m uncfield.se_region_probe --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from uncfield.se_m3_read import GATE_THRESHOLD
from uncfield.se_probe import (EXCLUDE, PLANTED_KEYS, REPO, SMOKE,
                               np_symlog, resolve_ckpt)

REGION_KEY = "position"
REGION_INDEX = 0
REGION_THRESHOLD = GATE_THRESHOLD    # single source of the pin (#27 m15)
MIN_REGION = 32                      # per-run mechanism adjudicability


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_logdir", default=str(SMOKE))
    p.add_argument("--output", default="",
                   help="default: <run>/se_region_probe")
    p.add_argument("--ckpt", default="")
    p.add_argument("--n_eval", type=int, default=512)
    p.add_argument("--burn_in", type=int, default=16)
    p.add_argument("--ep_batch", type=int, default=64)
    p.add_argument("--platform", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def run(args):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3.main import make_agent
    from probing.collect import load_run_config, load_frozen_agent
    from probing import probeset as probeset_mod

    f32 = jnp.float32
    out_dir = args.output or os.path.join(args.run_logdir,
                                          "se_region_probe")
    os.makedirs(out_dir, exist_ok=True)
    config = load_run_config(args.run_logdir, args.platform, out_dir,
                             False)
    agent = make_agent(config)
    jax.config.update("jax_transfer_guard", "allow")
    model = agent.model
    assert model.disag is not None, "region probe requires a disag ensemble"
    ens = int(config.agent.expl.disag_ens)
    dec_symlog = bool(model.dec.symlog)
    ckpt = resolve_ckpt(args.run_logdir, args.ckpt)
    load_frozen_agent(agent, ckpt)
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                          agent.params)

    obs_keys = sorted(k for k, v in agent.obs_space.items()
                      if k not in EXCLUDE and len(v.shape) <= 1)
    act_keys = sorted(agent.act_space.keys())
    assert REGION_KEY in obs_keys, obs_keys

    rng = np.random.default_rng(args.seed)
    arrays, _, stats = probeset_mod.collect_windows(
        os.path.join(args.run_logdir, "replay"), args.burn_in,
        args.n_eval, rng, allow_fewer=True)
    arrays = {k: np.stack(v, 0) for k, v in arrays.items()}
    S = arrays["is_first"].shape[0]
    L = args.burn_in
    reset = np.zeros((S, L), bool)
    reset[:, 0] = True
    print(f"region-probe windows: {S} x {L} from {stats['streams']} "
          f"streams (requested {args.n_eval})")

    # normalizers: the frozen se_probe convention incl. the floor
    raw_norms = {}
    for k in obs_keys:
        v = np.asarray(arrays[k], np.float32).reshape(S * L, -1)
        v = np_symlog(v) if dec_symlog else v
        raw_norms[k] = v.std(0)
    real = [k for k in obs_keys if k not in PLANTED_KEYS]
    real_scale = float(np.mean(np.concatenate([raw_norms[k]
                                               for k in real])))
    norms = {k: np.maximum(raw_norms[k], 0.05 * real_scale)
             for k in obs_keys}

    # --- model machinery (pinned to se_probe's frozen definitions)
    def postsplit(pred):
        ddim = model.dyn.deter
        det = pred[..., :ddim]
        probs = pred[..., ddim:]
        probs = probs.reshape((*probs.shape[:-1], model.dyn.stoch,
                               model.dyn.classes))
        probs = jnp.clip(probs, 1e-6, 1.0)
        probs = probs / probs.sum(-1, keepdims=True)
        return det, probs

    def decode_feats(featdict, B, T):
        dec_carry = model.dec.initial(B)
        _, _, recons = model.dec(dec_carry, featdict,
                                 jnp.zeros((B, T), bool), training=False)
        return {k: f32(recons[k].pred()) for k in obs_keys}

    def member_vars(feat_t, actvec, B, T):
        preds = f32(model.disag.predict(feat_t, actvec))
        means = {k: [] for k in obs_keys}
        for e in range(ens):
            det, probs = postsplit(preds[e])
            dec = decode_feats(dict(deter=det, stoch=probs), B, T)
            for k in obs_keys:
                means[k].append(dec[k])
        return {k: jnp.stack(means[k], 0).var(0) for k in obs_keys}

    def encode(obs, action_dict, reset_):
        B = reset_.shape[0]
        enc_carry = model.enc.initial(B)
        dyn_carry = model.dyn.initial(B)
        enc_carry, _, tokens = model.enc(enc_carry, obs, reset_,
                                         training=False)
        prevact = {k: jnp.concatenate(
            [jnp.zeros_like(v[:, :1]), v[:, :-1]], 1)
            for k, v in action_dict.items()}
        _, _, post = model.dyn.observe(dyn_carry, tokens, prevact,
                                       reset_, training=False)
        return post

    def anchor_fn(obs, action_dict, reset_):
        B = reset_.shape[0]
        post = encode(obs, action_dict, reset_)
        anchor = {"deter": post["deter"][:, -1:],
                  "stoch": post["stoch"][:, -1:]}
        feat_t = model.feat2tensor(anchor)
        actvec = jnp.concatenate(
            [action_dict[k][:, -1:].reshape((B, 1, -1))
             for k in act_keys], -1)
        out = {"intrinsic": f32(model.disag.reward(feat_t, actvec))[:, 0]}
        mv = member_vars(feat_t, actvec, B, 1)
        for k in obs_keys:
            out[f"var_{k}"] = mv[k][:, 0]
        return out

    jit_anchor = jax.jit(
        lambda p, o, a, r, s: nj.pure(anchor_fn)(p, o, a, r, seed=s))

    outs = []
    for i in range(0, S, args.ep_batch):
        sl = slice(i, min(i + args.ep_batch, S))
        o = {k: jnp.asarray(arrays[k][sl])
             for k in obs_keys + list(EXCLUDE) if k in arrays}
        a = {k: jnp.asarray(arrays[k][sl]) for k in act_keys}
        _, res = jit_anchor(params, o, a, jnp.asarray(reset[sl]),
                            args.seed + i)
        outs.append({k: np.asarray(v) for k, v in res.items()})
    anchor_out = {k: np.concatenate([o[k] for o in outs], 0)
                  for k in outs[0]}

    # per-state normalized per-key disagreement + region coordinate
    assert arrays[REGION_KEY].ndim == 3, (
        REGION_KEY, arrays[REGION_KEY].shape)        # #27 m20
    coord = np.asarray(arrays[REGION_KEY][:, -1, REGION_INDEX],
                       np.float64)
    region_high = coord > REGION_THRESHOLD
    per_state, dim_key = [], []
    for k in obs_keys:
        v = anchor_out[f"var_{k}"] / (norms[k] ** 2)[None]   # (S, dim)
        per_state.append(np.asarray(v, np.float32))
        dim_key.extend([k] * v.shape[1])
    per_state = np.concatenate(per_state, axis=1)            # (S, D)
    dim_key = np.asarray(dim_key)

    # summary json: per-region per-key means (reader recomputes from
    # the npz — these are the human-readable view + determinism keys)
    summary = {}
    for k in obs_keys:
        cols = dim_key == k
        summary[k] = dict(
            high=float(per_state[region_high][:, cols].mean())
            if region_high.any() else None,
            low=float(per_state[~region_high][:, cols].mean())
            if (~region_high).any() else None)
    n_high = int(region_high.sum())
    n_low = int((~region_high).sum())
    result = dict(run_logdir=os.path.abspath(args.run_logdir),
                  ckpt=os.path.abspath(ckpt), seed=args.seed,
                  n_eval=S, n_high=n_high, n_low=n_low,
                  region_adjudicable=bool(n_high >= MIN_REGION
                                          and n_low >= MIN_REGION),
                  region=dict(key=REGION_KEY, index=REGION_INDEX,
                              threshold=REGION_THRESHOLD),
                  per_region_means=summary,
                  anchor_intrinsic_mean=float(
                      anchor_out["intrinsic"].mean()))
    with open(os.path.join(out_dir, "se_region_probe.json"), "w") as f:
        json.dump(result, f, indent=1)
    # npz carries its own provenance (#27 M4): the reader binds it to
    # the run + checkpoint without trusting the directory layout
    np.savez(os.path.join(out_dir, "se_region_probe.npz"),
             per_state=per_state, dim_key=dim_key, coord=coord,
             region_high=region_high,
             anchor_intrinsic=np.asarray(anchor_out["intrinsic"],
                                         np.float32),
             raw_norms=np.concatenate([raw_norms[k] for k in obs_keys]),
             norm_floor=np.float32(0.05 * real_scale),
             ckpt_basename=np.str_(os.path.basename(
                 os.path.normpath(ckpt))),
             probe_seed=np.int64(args.seed),
             n_eval=np.int64(S))
    print(json.dumps(dict(n_high=result["n_high"], n_low=result["n_low"],
                          distractor=summary.get("distractor"),
                          position=summary.get("position")), indent=1))
    return result


def selfcheck():
    args = parse_args([
        "--run_logdir", str(SMOKE), "--n_eval", "48", "--burn_in", "12",
        "--ep_batch", "24", "--platform", "cpu",
        "--output", str(REPO / "local_results" / "uncfield"
                        / "se_region_probe_smoke"),
    ])
    r1 = run(args)
    assert r1["n_high"] + r1["n_low"] == r1["n_eval"]
    assert r1["n_high"] > 0 and r1["n_low"] > 0, (
        "smoke must populate both regions (threshold is the median)")
    assert "region_adjudicable" in r1               # #27 m21 surfaced
    z = np.load(os.path.join(args.output, "se_region_probe.npz"))
    # region labels match the stored coordinate exactly
    assert np.array_equal(np.asarray(z["region_high"]),
                          np.asarray(z["coord"]) > REGION_THRESHOLD)
    # json summary reproduces from the npz
    ps = np.asarray(z["per_state"], np.float64)
    dk = np.asarray(z["dim_key"]).astype(str)
    hi = np.asarray(z["region_high"])
    for k, v in r1["per_region_means"].items():
        cols = dk == k
        assert abs(ps[hi][:, cols].mean() - v["high"]) < 1e-6, k
        assert abs(ps[~hi][:, cols].mean() - v["low"]) < 1e-6, k
    assert np.isfinite(ps).all() and (ps >= 0).all()
    r2 = run(args)
    assert r2["per_region_means"] == r1["per_region_means"], \
        "region probe must be deterministic under fixed seed"
    assert r2["anchor_intrinsic_mean"] == r1["anchor_intrinsic_mean"]
    print("se_region_probe selfcheck PASS (both regions populated; "
          "labels==coord; json reproduces from npz; deterministic; "
          f"n_high/low {r1['n_high']}/{r1['n_low']})")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
