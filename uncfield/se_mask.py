"""SE-wave interventional mask instrument (SECONDARY, replay-only;
PREREG_nfi_scale_exhibit_20260812.md §4; built BEFORE THE READ as
registered — review #21 B5).

Causal complement to the decoder-projected primary (se_probe): for each
fire-eligible planted channel, delete exactly its fictitious content in
RAW obs space (before symlog — the encoder does its own preprocessing)
over the FULL probe window (same burn_in window and same pinned replay
slice as the primary: identical collect_windows rng), re-encode, and
measure the signed paired change in the DEPLOYED disagreement at the
anchor state.

Registered mask forms (pinned, prereg §4):
  N (distractor)  -> batch-permutation: the channel's whole window
                     trajectory is swapped across probe windows (pinned
                     rng), preserving marginals and within-window OU
                     structure while breaking alignment with the state.
  D-family        -> source-substitution: D := the recorded source-key
                     values, deleting exactly the eps-noise (dup1/dup2).
                     For dup0 (exact duplicate) substitution is a
                     BITWISE NO-OP by construction — reported as
                     delta == 0 with a noop flag (the value-level causal
                     answer; D0's redundancy content is carried by the
                     primary). Bitwise equality of the replayed dup0 and
                     source is ASSERTED (a §2 construction check at read
                     level); if replay munging ever broke it, the
                     substitution would be real and the flag drops.

Registered directional hypothesis: removal of a farmed key REDUCES
disag -> mean paired delta < 0, one-sided sign-flip permutation
(p_reduce); the inflation side (p_inflate) is reported SEPARATELY as a
shift-artifact diagnostic (review M-2). BCa CIs reporting-only
(standing rule). Deltas are reported for (a) the deployed intrinsic
reward at the anchor (disag.reward — the registered "disag") and
(b) the channel's own decoder-projected d_k (mechanism diagnostic).
Decision role: outcome-map cell 6 only (prereg §§4,6).

Run:  python -m uncfield.se_mask --run_logdir <dir> --output <dir>
      python -m uncfield.se_mask --selfcheck   (uses the smoke bundle)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import norm as _norm

from uncfield.se_probe import (EXCLUDE, FIRE_KEYS, PLANTED_KEYS, SMOKE,
                               REPO, np_symlog, resolve_ckpt)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_logdir", default=str(SMOKE))
    p.add_argument("--output", default="")
    p.add_argument("--ckpt", default="", help="default: <run>/ckpt/<latest>")
    p.add_argument("--n_eval", type=int, default=512)
    p.add_argument("--burn_in", type=int, default=16)
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--ep_batch", type=int, default=64)
    p.add_argument("--platform", default="cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="MUST match the se_probe seed so the pinned "
                        "replay slice (collect_windows rng) is identical")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------- statistics

def signflip_p(delta, n_perm, rng):
    """Paired sign-flip permutation on the mean. Returns (obs mean,
    p_reduce = P(null <= obs), p_inflate = P(null >= obs))."""
    delta = np.asarray(delta, np.float64)
    obs = float(delta.mean())
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = (delta * rng.choice([-1.0, 1.0], size=delta.size)).mean()
    p_reduce = float((1 + (null <= obs).sum()) / (1 + n_perm))
    p_inflate = float((1 + (null >= obs).sum()) / (1 + n_perm))
    return obs, p_reduce, p_inflate


def bca_interval(x, rng, n_boot=2000, alpha=0.05):
    """BCa bootstrap CI for the mean (reporting-only, standing rule)."""
    x = np.asarray(x, np.float64)
    n = x.size
    obs = float(x.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = x[rng.integers(0, n, n)].mean()
    frac = ((boots < obs).sum() + 0.5 * (boots == obs).sum()) / n_boot
    z0 = _norm.ppf(np.clip(frac, 1e-9, 1 - 1e-9))
    jack = np.array([(x.sum() - x[i]) / (n - 1) for i in range(n)])
    jm = jack.mean()
    den = 6.0 * (((jm - jack) ** 2).sum()) ** 1.5
    a = float(((jm - jack) ** 3).sum() / den) if den > 0 else 0.0
    lo_hi = []
    for z in (_norm.ppf(alpha / 2), _norm.ppf(1 - alpha / 2)):
        adj = _norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        lo_hi.append(float(np.quantile(boots, np.clip(adj, 0.0, 1.0))))
    return obs, lo_hi[0], lo_hi[1]


def build_masks(arrays, source_key, channels, rng):
    """Masked raw-obs variants, one per fire-eligible channel (prereg
    §4 pinned forms). Returns (variants dict, info dict)."""
    S = arrays["is_first"].shape[0]
    perm = rng.permutation(S)
    variants, info = {}, {}
    for ch in channels:
        v = dict(arrays)
        if ch == "distractor":
            v[ch] = arrays[ch][perm]
            info[ch] = dict(form="batch_permutation",
                            fixed_points=int((perm == np.arange(S)).sum()))
        else:
            sub = arrays[source_key].copy()
            noop = bool(np.array_equal(sub, arrays[ch]))
            v[ch] = sub
            info[ch] = dict(form="source_substitution", bitwise_noop=noop)
        variants[ch] = v
    return variants, info


# ---------------------------------------------------------------------- main

def run(args):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3.main import make_agent
    from probing.collect import load_run_config, load_frozen_agent
    from probing import probeset as probeset_mod

    f32 = jnp.float32
    out_dir = args.output or os.path.join(args.run_logdir, "se_mask")
    os.makedirs(out_dir, exist_ok=True)
    config = load_run_config(args.run_logdir, args.platform, out_dir, False)
    agent = make_agent(config)
    jax.config.update("jax_transfer_guard", "allow")
    model = agent.model
    assert model.disag is not None, "SE mask probe requires a disag ensemble"
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
    mask_channels = [k for k in obs_keys if k in FIRE_KEYS]

    # pinned replay slice: IDENTICAL rng consumption to se_probe
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

    # normalizers: BASELINE arrays, same convention + floor as se_probe
    raw_norms = {}
    for k in obs_keys:
        v = np.asarray(arrays[k], np.float32).reshape(S * L, -1)
        v = np_symlog(v) if dec_symlog else v
        raw_norms[k] = v.std(0)
    real_scale = float(np.mean(np.concatenate([raw_norms[k] for k in real])))
    norms = {k: np.maximum(raw_norms[k], 0.05 * real_scale)
             for k in obs_keys}

    # --- model machinery (pinned to se_probe's definitions verbatim)
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
        preds = f32(model.disag.predict(feat_t, actvec))   # (E, B, T, tgt)
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
        _, _, post = model.dyn.observe(dyn_carry, tokens, prevact, reset_,
                                       training=False)
        return post

    def anchor_fn(obs, action_dict, reset_):
        B = reset_.shape[0]
        post = encode(obs, action_dict, reset_)
        anchor = {"deter": post["deter"][:, -1:],
                  "stoch": post["stoch"][:, -1:]}
        feat_t = model.feat2tensor(anchor)
        actvec = jnp.concatenate(
            [action_dict[k][:, -1:].reshape((B, 1, -1)) for k in act_keys],
            -1)
        out = {"intrinsic": f32(model.disag.reward(feat_t, actvec))[:, 0]}
        mv = member_vars(feat_t, actvec, B, 1)
        for k in obs_keys:
            out[f"var_{k}"] = mv[k][:, 0]
        return out

    jit_anchor = jax.jit(
        lambda p, o, a, r, s: nj.pure(anchor_fn)(p, o, a, r, seed=s))

    def anchor_pass(obs_arrays):
        outs = []
        for i in range(0, S, args.ep_batch):
            sl = slice(i, min(i + args.ep_batch, S))
            o = {k: jnp.asarray(obs_arrays[k][sl])
                 for k in obs_keys + list(EXCLUDE) if k in obs_arrays}
            a = {k: jnp.asarray(obs_arrays[k][sl]) for k in act_keys}
            _, res = jit_anchor(params, o, a, jnp.asarray(reset[sl]),
                                args.seed + i)
            outs.append({k: np.asarray(v) for k, v in res.items()})
        return {k: np.concatenate([o[k] for o in outs], 0)
                for k in outs[0]}

    # construction check (§2 at read level): replayed dup0 == source
    dup0_equal = bool(np.array_equal(arrays.get("planted_dup0"),
                                     arrays[source_key]))
    assert dup0_equal, \
        "replayed planted_dup0 != source — construction violated"

    variants, mask_info = build_masks(
        arrays, source_key, mask_channels,
        np.random.default_rng(args.seed + 13))

    base = anchor_pass(arrays)
    channels = {}
    dnpz = {}
    for ch in mask_channels:
        masked = anchor_pass(variants[ch])
        d_int = (masked["intrinsic"] - base["intrinsic"]).astype(np.float64)
        own_b = (base[f"var_{ch}"] / (norms[ch] ** 2)[None]).mean(1)
        own_m = (masked[f"var_{ch}"] / (norms[ch] ** 2)[None]).mean(1)
        d_own = (own_m - own_b).astype(np.float64)
        rec = dict(mask_info[ch])
        for name, d in (("intrinsic", d_int), ("own_d", d_own)):
            obs_m, p_red, p_inf = signflip_p(
                d, args.n_perm,
                np.random.default_rng(args.seed + 101 + len(name)))
            _, lo, hi = bca_interval(
                d, np.random.default_rng(args.seed + 211 + len(name)),
                args.n_boot)
            rec[f"delta_{name}_mean"] = obs_m
            rec[f"delta_{name}_bca"] = [lo, hi]
            rec[f"p_reduce_{name}"] = p_red
            rec[f"p_inflate_{name}"] = p_inf
        channels[ch] = rec
        dnpz[f"delta_intrinsic_{ch}"] = d_int
        dnpz[f"delta_own_{ch}"] = d_own

    result = dict(
        run_logdir=os.path.abspath(args.run_logdir),
        ckpt=os.path.abspath(ckpt), n_eval=S, seed=args.seed,
        source_key=source_key, dup0_bitwise_equal_source=dup0_equal,
        base_intrinsic_mean=float(base["intrinsic"].mean()),
        channels=channels,
    )
    with open(os.path.join(out_dir, "se_mask.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez(os.path.join(out_dir, "se_mask_deltas.npz"),
             base_intrinsic=base["intrinsic"], **dnpz)
    print(json.dumps(result["channels"], indent=1))
    return result


def selfcheck():
    """Statistics mutants + mask construction checks + end-to-end smoke."""
    # sign-flip test: detects a planted negative shift, null on exchangeable
    rng = np.random.default_rng(0)
    d_neg = rng.normal(-1.0, 0.1, 100)
    _, p_red, p_inf = signflip_p(d_neg, 500, np.random.default_rng(1))
    assert p_red < 0.01 and p_inf > 0.99, (p_red, p_inf)
    d_null = rng.normal(0.0, 1.0, 100)
    _, p_red0, _ = signflip_p(d_null, 500, np.random.default_rng(2))
    assert p_red0 > 0.05, p_red0
    # BCa: brackets the mean, detects a clear shift
    m, lo, hi = bca_interval(rng.normal(2.0, 0.5, 60),
                             np.random.default_rng(3), 1000)
    assert lo < m < hi and lo > 1.0, (lo, m, hi)
    # mask construction on synthetic arrays
    syn = {"is_first": np.zeros((10, 4), bool),
           "position": rng.normal(0, 1, (10, 4, 8)).astype(np.float32),
           "distractor": rng.normal(0, 1, (10, 4, 8)).astype(np.float32)}
    syn["planted_dup0"] = syn["position"].copy()
    syn["planted_dup1"] = (syn["position"]
                           + rng.normal(0, .01, (10, 4, 8))).astype(np.float32)
    variants, info = build_masks(
        syn, "position",
        ["planted_dup0", "planted_dup1", "distractor"],
        np.random.default_rng(4))
    assert info["planted_dup0"]["bitwise_noop"] is True
    assert info["planted_dup1"]["bitwise_noop"] is False
    assert np.array_equal(variants["planted_dup1"]["planted_dup1"],
                          syn["position"])
    ms = variants["distractor"]["distractor"]
    assert not np.array_equal(ms, syn["distractor"])
    assert np.array_equal(np.sort(ms, axis=None),
                          np.sort(syn["distractor"], axis=None)), \
        "batch permutation must preserve marginals exactly"
    # end-to-end on the smoke ckpt (small) + determinism
    args = parse_args([
        "--run_logdir", str(SMOKE), "--n_eval", "48", "--burn_in", "12",
        "--n_perm", "300", "--n_boot", "300", "--ep_batch", "24",
        "--platform", "cpu",
        "--output", str(REPO / "local_results" / "uncfield" / "se_mask_smoke"),
    ])
    r1 = run(args)
    assert r1["dup0_bitwise_equal_source"]
    c0 = r1["channels"]["planted_dup0"]
    assert c0["bitwise_noop"] and c0["delta_intrinsic_mean"] == 0.0 \
        and c0["delta_own_d_mean"] == 0.0, c0
    for ch, rec in r1["channels"].items():
        assert np.isfinite(rec["delta_intrinsic_mean"])
        assert rec["delta_intrinsic_bca"][0] <= rec["delta_intrinsic_mean"] \
            <= rec["delta_intrinsic_bca"][1], (ch, rec)
    r2 = run(args)
    for ch in r1["channels"]:
        for f in ("delta_intrinsic_mean", "p_reduce_intrinsic",
                  "delta_own_d_mean", "p_reduce_own_d"):
            assert r1["channels"][ch][f] == r2["channels"][ch][f], (ch, f)
    ds = {ch: r1["channels"][ch]["delta_intrinsic_mean"]
          for ch in r1["channels"]}
    print("se_mask selfcheck PASS (mutants killed; dup0 noop delta==0; "
          "marginals preserved; deterministic; smoke deltas "
          + json.dumps({k: round(v, 5) for k, v in ds.items()}) + ")")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
