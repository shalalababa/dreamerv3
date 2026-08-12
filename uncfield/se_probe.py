"""SE-wave attribution probe: decoder-projected per-key disagreement
(PREREG_nfi_scale_exhibit_20260812.md §§4-5; 12 Aug 2026).

Reads a trained SE run (DreamerV3 + p2e disagreement, planted channels)
and computes, at replay anchor states and along policy-driven imagined
rollouts, the PRIMARY attribution: for each disag-ensemble member, its
predicted next latent (postfeat) is split into (deter-hat, probs-hat)
and decoded through the frozen decoder's per-key heads with SOFT stoch;
d_k = Var over members of the per-key decoded means, per-dim normalized
by the probe windows' variance in the decoder target space (symlog),
averaged over key dims. Planted share = sum_planted d_k / sum_all d_k.

Registered reads:
  P-SE1: per-channel share vs the dim-level permutation null (planted /
         real labels reassigned across DIMS, share recomputed; >=1000
         permutations) and vs theta_1 = the source key's own share.
  P-SE2: M imagined rollouts from anchor starts under the trained
         policy (dyn.imagine with policy callable, rssm.py:94 +
         agent.py:19 sample pattern), ranked by imagined intrinsic
         return (disag.reward along the path); statistic = mean planted
         share of the top-k vs the all-rollout mean, permutation test
         over rollout labels.

Calibration control (registered in the prereg §4): decoding the TRUE
soft postfeat of held states must track decoding their actual sampled
(deter, stoch) state — reported per key and asserted in selfcheck.

Descriptive per-run instrument; the registered cross-seed read pools
the per-run outputs (se_read, built at freeze). CPU-capable
(--platform cpu).

Run:  python -m uncfield.se_probe --run_logdir <dir> --output <dir>
      python -m uncfield.se_probe --selfcheck   (uses the smoke bundle)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import zlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
SMOKE = REPO / "local_results" / "uncfield_se_smoke_20260812_155552" / "se_smoke0"
PLANTED_KEYS = ("planted_dup0", "planted_dup1", "planted_dup2",
                "planted_const", "distractor")
# fire-eligible channels (review #21 B3): const is the decoder-projection
# floor DIAGNOSTIC only — it never enters pooled statistics, the P-SE2
# numerator, or any fire decision (prereg §§2,5).
FIRE_KEYS = ("planted_dup0", "planted_dup1", "planted_dup2", "distractor")
EXCLUDE = ("is_first", "is_last", "is_terminal", "reward")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_logdir", default=str(SMOKE))
    p.add_argument("--output", default="")
    p.add_argument("--ckpt", default="", help="default: <run>/ckpt/<latest>")
    p.add_argument("--n_eval", type=int, default=512)
    p.add_argument("--burn_in", type=int, default=16)
    p.add_argument("--rollouts", type=int, default=256)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--ep_batch", type=int, default=64)
    p.add_argument("--platform", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def resolve_ckpt(run_logdir, ckpt):
    if ckpt:
        return ckpt
    root = os.path.join(run_logdir, "ckpt")
    latest = os.path.join(root, "latest")
    if os.path.exists(latest):
        name = open(latest).read().strip()
        cand = os.path.join(root, name)
        if os.path.isdir(cand):
            return cand
    dirs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
    assert dirs, f"no checkpoint under {root}"
    return os.path.join(root, dirs[-1])


# ---------------------------------------------------------- numpy statistics

def np_symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def share_stats(dims_d, dims_label):
    """dims_d: (n_dims,) pooled per-dim attribution; dims_label: bool
    (True = planted). Returns planted share."""
    tot = dims_d.sum()
    return float(dims_d[dims_label].sum() / tot) if tot > 0 else 0.0


def perm_null(dims_d, dims_label, n_perm, rng):
    """Dim-level label permutation null for the planted share."""
    obs = share_stats(dims_d, dims_label)
    null = np.empty(n_perm)
    lab = dims_label.copy()
    for i in range(n_perm):
        rng.shuffle(lab)
        null[i] = share_stats(dims_d, lab)
    p = (1 + (null >= obs).sum()) / (1 + n_perm)
    return obs, float(p), float(np.quantile(null, 0.95))


def rollout_perm(shares, order, topk, n_perm, rng):
    """P-SE2: mean top-k share (by intrinsic order) vs permuted-rank null."""
    top = float(shares[order[:topk]].mean())
    base = float(shares.mean())
    null = np.empty(n_perm)
    idx = np.arange(len(shares))
    for i in range(n_perm):
        rng.shuffle(idx)
        null[i] = shares[idx[:topk]].mean()
    p = (1 + (null >= top).sum()) / (1 + n_perm)
    return top, base, float(p)


# ------------------------------------------------------------------- main

def run(args):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3.main import make_agent
    from probing.collect import load_run_config, load_frozen_agent
    from probing import probeset as probeset_mod

    f32 = jnp.float32
    out_dir = args.output or os.path.join(args.run_logdir, "se_probe")
    os.makedirs(out_dir, exist_ok=True)
    config = load_run_config(args.run_logdir, args.platform, out_dir, False)
    agent = make_agent(config)
    jax.config.update("jax_transfer_guard", "allow")
    model = agent.model
    assert model.disag is not None, "SE probe requires a disag ensemble"
    ens = int(config.agent.expl.disag_ens)
    dec_symlog = bool(model.dec.symlog)
    ckpt = resolve_ckpt(args.run_logdir, args.ckpt)
    load_frozen_agent(agent, ckpt)
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                          agent.params)

    obs_keys = sorted(k for k, v in agent.obs_space.items()
                      if k not in EXCLUDE and len(v.shape) <= 1)
    act_keys = sorted(agent.act_space.keys())
    planted = [k for k in obs_keys if k in PLANTED_KEYS]
    real = [k for k in obs_keys if k not in PLANTED_KEYS]
    assert planted, f"no planted keys in {obs_keys}"
    source_key = str(config.planted["source_key"])

    # --- replay windows (the run's own buffer = the deployed eval dist.)
    rng = np.random.default_rng(args.seed)
    arrays, _, stats = probeset_mod.collect_windows(
        os.path.join(args.run_logdir, "replay"), args.burn_in, args.n_eval,
        rng, allow_fewer=True)
    arrays = {k: np.stack(v, 0) for k, v in arrays.items()}
    S = arrays["is_first"].shape[0]
    L = args.burn_in
    reset = np.zeros((S, L), bool)
    reset[:, 0] = True
    print(f"probe windows: {S} x {L} from {stats['streams']} streams "
          f"(requested {args.n_eval})")

    # per-dim normalizers in the decoder target space. FLOOR (pinned
    # instrument constant): 0.05 x the mean per-dim std of the REAL keys
    # — a (near-)constant key has std ~0, and an unfloored normalizer
    # divides by ~1e-12 and manufactures a dominant share out of decoder
    # round-off (caught on the smoke run: planted_const share 0.9999).
    raw_norms = {}
    for k in obs_keys:
        v = np.asarray(arrays[k], np.float32).reshape(S * L, -1)
        v = np_symlog(v) if dec_symlog else v
        raw_norms[k] = v.std(0)
    real_scale = float(np.mean(np.concatenate(
        [raw_norms[k] for k in real])))
    norm_floor = 0.05 * real_scale
    norms = {k: np.maximum(raw_norms[k], norm_floor) for k in obs_keys}

    def postsplit(pred):
        """(..., target_dim) -> deter, probs(soft stoch), logit."""
        ddim = model.dyn.deter
        det = pred[..., :ddim]
        probs = pred[..., ddim:]
        probs = probs.reshape((*probs.shape[:-1], model.dyn.stoch,
                               model.dyn.classes))
        probs = jnp.clip(probs, 1e-6, 1.0)
        probs = probs / probs.sum(-1, keepdims=True)
        return det, probs, jnp.log(probs)

    def decode_feats(featdict, B, T):
        # decoder consumes deter/stoch only (rssm.py:288-296); no logit
        dec_carry = model.dec.initial(B)
        _, _, recons = model.dec(dec_carry, featdict,
                                 jnp.zeros((B, T), bool), training=False)
        return {k: f32(recons[k].pred()) for k in obs_keys}

    def member_vars(feat_t, actvec, B, T):
        """Decoder-projected per-key ensemble variance at (B, T) points."""
        preds = f32(model.disag.predict(feat_t, actvec))   # (E, B, T, tgt)
        means = {k: [] for k in obs_keys}
        for e in range(ens):
            det, probs, _ = postsplit(preds[e])
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
        _, _, post = model.dyn.observe(dyn_carry, tokens, prevact, reset_,
                                       training=False)
        return post

    def anchor_fn(obs, action_dict, reset_):
        B = reset_.shape[0]
        post = encode(obs, action_dict, reset_)
        anchor = {"deter": post["deter"][:, -1:], "stoch": post["stoch"][:, -1:]}
        feat_t = model.feat2tensor(anchor)
        actvec = jnp.concatenate(
            [action_dict[k][:, -1:].reshape((B, 1, -1)) for k in act_keys],
            -1)
        out = {}
        mv = member_vars(feat_t, actvec, B, 1)
        for k in obs_keys:
            out[f"var_{k}"] = mv[k][:, 0]
        # calibration: soft-postfeat decode vs actual-state decode
        soft_probs = jax.nn.softmax(f32(post["logit"][:, -1:]), -1)
        dec_soft = decode_feats(dict(
            deter=post["deter"][:, -1:], stoch=soft_probs), B, 1)
        dec_hard = decode_feats(dict(
            deter=post["deter"][:, -1:], stoch=post["stoch"][:, -1:]), B, 1)
        for k in obs_keys:
            out[f"calsoft_{k}"] = dec_soft[k][:, 0]
            out[f"calhard_{k}"] = dec_hard[k][:, 0]
        out["anchor_deter"] = post["deter"][:, -1]
        out["anchor_stoch"] = post["stoch"][:, -1]
        return out

    sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)

    def rollout_fn(deter, stoch):
        B = deter.shape[0]
        carry = {"deter": deter, "stoch": stoch}
        policyfn = lambda feat: sample(model.pol(model.feat2tensor(feat), 1))
        _, imgfeat, imgact = model.dyn.imagine(
            carry, policyfn, args.horizon, training=False)
        # PAIRING (review #21 B1): dyn.imagine returns at index i the
        # POST-action state s_{i+1} with the action a_i taken at s_i; the
        # deployed disag pairing is (s_i, a_i) (agent.py:289-291, and the
        # planner's own imagined reward, agent.py:311-321). Rebuild the
        # pre-action state sequence from the anchor carry, as
        # probing/latent_uq.py:281-293 does.
        states = {
            "deter": jnp.concatenate(
                [deter[:, None], imgfeat["deter"][:, :-1]], 1),
            "stoch": jnp.concatenate(
                [stoch[:, None], imgfeat["stoch"][:, :-1]], 1)}
        feat_t = model.feat2tensor(states)
        actvec = jnp.concatenate(
            [imgact[k].reshape((B, args.horizon, -1)) for k in act_keys], -1)
        out = {"intrinsic": f32(model.disag.reward(feat_t, actvec))}
        mv = member_vars(feat_t, actvec, B, args.horizon)
        for k in obs_keys:
            out[f"var_{k}"] = mv[k]
        return out

    jit_anchor = jax.jit(
        lambda p, o, a, r, s: nj.pure(anchor_fn)(p, o, a, r, seed=s))
    jit_roll = jax.jit(
        lambda p, d, st, s: nj.pure(rollout_fn)(p, d, st, seed=s))

    # ---- batched anchor pass
    outs = []
    for i in range(0, S, args.ep_batch):
        sl = slice(i, min(i + args.ep_batch, S))
        o = {k: jnp.asarray(arrays[k][sl]) for k in obs_keys + list(EXCLUDE)
             if k in arrays}
        a = {k: jnp.asarray(arrays[k][sl]) for k in act_keys}
        _, res = jit_anchor(params, o, a, jnp.asarray(reset[sl]),
                            args.seed + i)
        outs.append({k: np.asarray(v) for k, v in res.items()})
    anchor_out = {k: np.concatenate([o[k] for o in outs], 0)
                  for k in outs[0]}

    # ---- P-SE1: pooled per-dim attribution + permutation null
    dims_d, dims_label, dim_key = [], [], []
    per_key_d = {}
    for k in obs_keys:
        v = anchor_out[f"var_{k}"] / (norms[k] ** 2)[None]   # (S, dim)
        per_key_d[k] = float(v.mean())
        dims_d.append(v.mean(0))
        dims_label.extend([k in FIRE_KEYS] * v.shape[1])
        dim_key.extend([k] * v.shape[1])
    dims_d = np.concatenate(dims_d)
    dims_label = np.asarray(dims_label)
    dim_key_arr = np.asarray(dim_key)
    # pooled statistic (DESCRIPTIVE only, review #21 M2): fire+real dims,
    # const excluded entirely
    pool_mask = np.asarray([dk != "planted_const" for dk in dim_key_arr])
    prng = np.random.default_rng(args.seed + 999)
    obs_share, p_perm, null95 = perm_null(
        dims_d[pool_mask], dims_label[pool_mask], args.n_perm, prng)
    tot = sum(per_key_d.values())
    key_share = {k: per_key_d[k] / tot for k in obs_keys}
    channel = {k: dict(share=key_share[k],
                       vs_source=key_share[k] / max(key_share[source_key],
                                                    1e-12))
               for k in planted}

    # per-channel permutation p: channel dims vs REAL dims only; rng keyed
    # by crc32 (python hash() is process-salted — review #21 B2)
    for k in planted:
        mask = np.asarray([dk == k or dk in real for dk in dim_key_arr])
        sub_d = dims_d[mask]
        sub_lab = np.asarray([dk == k for dk in dim_key_arr[mask]])
        _, pk, _ = perm_null(
            sub_d, sub_lab, args.n_perm,
            np.random.default_rng(args.seed + zlib.crc32(k.encode()) % 2**16))
        channel[k]["p_perm"] = pk
        channel[k]["fire_eligible"] = k in FIRE_KEYS
        if k == "planted_const":
            channel[k]["diagnostic_only"] = True

    # ---- calibration control
    cal = {}
    for k in obs_keys:
        diff = np.abs(anchor_out[f"calsoft_{k}"] - anchor_out[f"calhard_{k}"])
        cal[k] = float(diff.mean() / (norms[k].mean()))

    # ---- P-SE2: policy rollouts from anchor starts
    K = min(args.rollouts, S)
    pick = np.random.default_rng(args.seed + 7).choice(S, K, replace=False)
    roll_outs = []
    for i in range(0, K, args.ep_batch):
        sl = pick[i:i + args.ep_batch]
        _, res = jit_roll(params,
                          jnp.asarray(anchor_out["anchor_deter"][sl]),
                          jnp.asarray(anchor_out["anchor_stoch"][sl]),
                          args.seed + 31 + i)
        roll_outs.append({k: np.asarray(v) for k, v in res.items()})
    roll = {k: np.concatenate([o[k] for o in roll_outs], 0)
            for k in roll_outs[0]}
    intr = roll["intrinsic"].sum(1)                       # (K,)
    rshare = np.zeros(K)
    num = np.zeros(K)
    den = np.zeros(K)
    for k in obs_keys:
        v = (roll[f"var_{k}"] / (norms[k] ** 2)[None, None]).mean(-1).sum(1)
        den += v
        if k in FIRE_KEYS:
            num += v
    rshare = num / np.maximum(den, 1e-12)
    order = np.argsort(-intr)
    top_share, base_share, p_rank = rollout_perm(
        rshare, order, args.topk, args.n_perm,
        np.random.default_rng(args.seed + 77))

    result = dict(
        run_logdir=os.path.abspath(args.run_logdir),
        ckpt=os.path.abspath(ckpt), n_eval=S, ens=ens,
        dec_symlog=dec_symlog, seed=args.seed,
        key_share=key_share, channels=channel,
        planted_share_total=obs_share, p_perm_total=p_perm,
        perm_null95=null95, source_key=source_key,
        theta1_source_share=key_share[source_key],
        calibration=cal,
        pse2=dict(K=K, topk=args.topk, top_share=top_share,
                  base_share=base_share, p_rank=p_rank,
                  intrinsic_mean=float(intr.mean())),
    )
    with open(os.path.join(out_dir, "se_probe.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez(os.path.join(out_dir, "se_probe_dims.npz"),
             dims_d=dims_d, dims_label=dims_label,
             dim_key=dim_key_arr, rollout_share=rshare,
             rollout_intrinsic=intr,
             raw_norms=np.concatenate([raw_norms[k] for k in obs_keys]),
             norm_floor=np.float32(norm_floor))
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("key_share", "planted_share_total",
                               "p_perm_total", "calibration", "pse2")},
                     indent=1))
    return result


def selfcheck():
    """End-to-end on the smoke checkpoint + statistics mutants."""
    # statistics mutants first (cheap, no jax)
    rng = np.random.default_rng(0)
    d = np.abs(rng.normal(1, 0.1, 60))
    lab = np.zeros(60, bool)
    lab[:20] = True
    d_inflated = d.copy()
    d_inflated[:20] *= 10
    _, p_hot, _ = perm_null(d_inflated, lab.copy(), 500,
                            np.random.default_rng(1))
    assert p_hot < 0.01, p_hot                      # inflation detected
    _, p_null, _ = perm_null(d, lab.copy(), 500, np.random.default_rng(2))
    assert p_null > 0.05, p_null                    # exchangeable -> null
    sh = np.linspace(0, 1, 40)
    top, base, p_r = rollout_perm(sh, np.argsort(-sh), 5, 500,
                                  np.random.default_rng(3))
    assert top > base and p_r < 0.01
    # under a rank-uninformative order the p-value is ~uniform: assert the
    # MEAN p over 20 random orders is near 0.5 (single-order asserts are
    # 5%-flaky by construction)
    ps = [rollout_perm(sh, np.random.default_rng(40 + i).permutation(40),
                       5, 300, np.random.default_rng(60 + i))[2]
          for i in range(20)]
    assert 0.2 < float(np.mean(ps)) < 0.8, np.mean(ps)
    # end-to-end on the smoke ckpt (small)
    args = parse_args([
        "--run_logdir", str(SMOKE), "--n_eval", "48", "--burn_in", "12",
        "--rollouts", "24", "--horizon", "8", "--topk", "5",
        "--n_perm", "300", "--ep_batch", "24", "--platform", "cpu",
        "--output", str(REPO / "local_results" / "uncfield" / "se_probe_smoke"),
    ])
    r1 = run(args)
    shares = r1["key_share"]
    assert abs(sum(shares.values()) - 1.0) < 1e-6
    assert all(np.isfinite(list(shares.values())))
    assert all(v < 1.5 for v in r1["calibration"].values()), r1["calibration"]
    r2 = run(args)
    for k in r1["channels"]:
        assert r2["channels"][k]["p_perm"] == r1["channels"][k]["p_perm"], k
        assert r2["key_share"][k] == r1["key_share"][k], k
    assert r2["pse2"]["p_rank"] == r1["pse2"]["p_rank"]
    assert abs(r2["planted_share_total"] - r1["planted_share_total"]) < 1e-9, \
        "probe must be deterministic under fixed seed (incl. per-channel)"
    print("se_probe selfcheck PASS (mutants killed; smoke end-to-end; "
          f"planted_share={r1['planted_share_total']:.3f} "
          f"p={r1['p_perm_total']:.4f}; cal max "
          f"{max(r1['calibration'].values()):.3f}; deterministic)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
