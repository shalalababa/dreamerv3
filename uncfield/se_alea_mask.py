"""B4 aleatoric-repair instrument
(PREREG_trackB_alea_20260822.md + Amendment 1; built + selfchecked
BEFORE the B4 wave's compute).

PRIMARY statistics (Amendment 1 — the SHARE form; the det se_mask
baseline showed the masked-delta form is blind to level-type
pricing, artifacts/det_semask_baseline_20260822):
  share_raw(key)  = key's share of the decoder-projected,
                    data-variance-normalized ensemble variance of
                    the RAW member predictions (the theta_1
                    machinery, se_probe idiom verbatim)
  share_norm(key) = the same projection of the ALEATORIC-WHITENED
                    members mu_bar + (mu_e - mu_bar)/sigma_bar —
                    the deployed reward's per-dim reallocation
                    pushed into channel space (nonlinear-decoder
                    pushforward; disclosed first-order
                    approximation)
Per-anchor per-key rows saved to the npz for the reader's paired
bootstrap.

DESCRIPTIVE (retained from the original registration): the masked
population-intervention deltas on both scalar rewards
  raw_rew  = var over members of mu           (the Stage-1 pricing)
  norm_rew = var(mu) / (ensemble-mean predicted aleatoric var)
under the house mask suite (distractor batch permutation, dup0
bitwise no-op anchor, dup substitution + fresh-resample, velocity
control) — expected ~0 per the det baseline; a per-run replication
of the level-vs-coupling dissociation on the gauss arm.

Run:  python -m uncfield.se_alea_mask --run_logdir <dir>
      python -m uncfield.se_alea_mask --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np

from uncfield.se_mask import bca_interval, build_masks, signflip_p
from uncfield.se_probe import (EXCLUDE, FIRE_KEYS, PLANTED_KEYS, REPO,
                               resolve_ckpt)


def whiten_members(mu, lv):
    """A1-B3.1/B4: aleatoric whitening with the divisor
    scale-normalized to unit geometric mean per anchor —
    sigma_tilde = sigma / gmean_dims(sigma). Carries ONLY the
    cross-dim reallocation; under uniform sigma, sigma_tilde == 1
    and the transform is the IDENTITY (white == mu exactly), so a
    constant rescaling (R3-B1) cannot move the shares. Returns
    (white, sigma_tilde, dev). Works on jax or numpy arrays."""
    import jax.numpy as jnp
    log_sigma = 0.5 * lv.mean(0)
    sigma_tilde = jnp.exp(
        log_sigma - log_sigma.mean(-1, keepdims=True))
    mu_bar = mu.mean(0)
    dev = mu - mu_bar[None]
    return mu_bar[None] + dev / sigma_tilde[None], sigma_tilde, dev


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_logdir", default="")
    p.add_argument("--output", default="",
                   help="default: <run>/se_alea_mask")
    p.add_argument("--ckpt", default="")
    p.add_argument("--n_eval", type=int, default=512)
    p.add_argument("--burn_in", type=int, default=16)
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--n_boot", type=int, default=2000)
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
                                          "se_alea_mask")
    os.makedirs(out_dir, exist_ok=True)
    config = load_run_config(args.run_logdir, args.platform, out_dir,
                             False)
    agent = make_agent(config)
    jax.config.update("jax_transfer_guard", "allow")
    model = agent.model
    assert model.disag is not None
    assert str(config.agent.expl["disag_head"]) == "gauss", (
        f"run is disag_head={config.agent.expl['disag_head']!r} — "
        f"the alea mask measures the gauss-head deployed reward")
    ckpt = resolve_ckpt(args.run_logdir, args.ckpt)
    load_frozen_agent(agent, ckpt)
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                          agent.params)

    obs_keys = sorted(k for k, v in agent.obs_space.items()
                      if k not in EXCLUDE and len(v.shape) <= 1)
    act_keys = sorted(agent.act_space.keys())
    source_key = str(config.planted["source_key"])
    mask_channels = [k for k in obs_keys if k in FIRE_KEYS]
    planted_basesd = float(config.planted["basesd"])

    rng = np.random.default_rng(args.seed)
    arrays, _, stats = probeset_mod.collect_windows(
        os.path.join(args.run_logdir, "replay"), args.burn_in,
        args.n_eval, rng, allow_fewer=True)
    arrays = {k: np.stack(v, 0) for k, v in arrays.items()}
    S = arrays["is_first"].shape[0]
    assert S == args.n_eval, (
        f"realized S {S} != pinned population {args.n_eval}")
    L = args.burn_in
    reset = np.zeros((S, L), bool)
    reset[:, 0] = True

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

    ddim = int(model.dyn.deter)
    lv_lo = float(model.disag.logvar_min)
    lv_hi = float(model.disag.logvar_max)
    ens = int(config.agent.expl["disag_ens"])
    dec_symlog = bool(model.dec.symlog)
    from uncfield.se_probe import np_symlog

    def postsplit(pred):
        det = pred[..., :ddim]
        probs = pred[..., ddim:]
        probs = probs.reshape((*probs.shape[:-1], model.dyn.stoch,
                               model.dyn.classes))
        probs = jnp.clip(probs, 1e-6, 1.0)
        return det, probs / probs.sum(-1, keepdims=True)

    def decode_feats(featdict, B, T):
        dec_carry = model.dec.initial(B)
        _, _, recons = model.dec(dec_carry, featdict,
                                 jnp.zeros((B, T), bool),
                                 training=False)
        return {k: f32(recons[k].pred()) for k in obs_keys}

    def _proj_vars(members, B):
        """(E, B, 1, tgt) member latents -> decoder-projected
        per-key ensemble variance at the anchors (se_probe idiom)."""
        means = {k: [] for k in obs_keys}
        for e in range(ens):
            det, probs = postsplit(members[e])
            dec = decode_feats(dict(deter=det, stoch=probs), B, 1)
            for k in obs_keys:
                means[k].append(dec[k])
        return {k: jnp.stack(means[k], 0).var(0)[:, 0]
                for k in obs_keys}                      # (B, dim)

    def anchor_fn(obs, action_dict, reset_):
        B = reset_.shape[0]
        post = encode(obs, action_dict, reset_)
        anchor = {"deter": post["deter"][:, -1:],
                  "stoch": post["stoch"][:, -1:]}
        feat = model.feat2tensor(anchor)
        actvec = jnp.concatenate(
            [action_dict[k][:, -1:].reshape((B, 1, -1))
             for k in act_keys], -1)
        mu = f32(model.disag.predict(feat, actvec))
        lv = f32(model.disag.predict_logvar(feat, actvec))
        alea = jnp.exp(lv).mean(0)
        ratio = mu.var(0) / (alea + 1e-8)
        raw = mu.var(0).mean(-1)[:, 0]
        norm = ratio.mean(-1)[:, 0]
        # rev R3-N3: interpretation diagnostics — logvar bound
        # fractions (the N2 clip ratchet made observable), aleatoric
        # level, and the deter/stoch block decomposition of the
        # deployed reward (a mean of per-dim ratios is dominated by
        # the lowest-alea dims, so the gauss head also RE-WEIGHTS
        # across blocks — distinct from de-pricing noisy dims)
        return {"raw": raw, "norm": norm,
                "alea_mean": alea.mean(-1)[:, 0],
                "lv_frac_lo": (lv <= lv_lo + 1e-6).astype(f32)
                              .mean((0, -1))[:, 0],
                "lv_frac_hi": (lv >= lv_hi - 1e-6).astype(f32)
                              .mean((0, -1))[:, 0],
                "norm_deter": ratio[..., :ddim].mean(-1)[:, 0],
                "norm_stoch": ratio[..., ddim:].mean(-1)[:, 0]}

    jit_anchor = jax.jit(
        lambda p, o, a, r, s: nj.pure(anchor_fn)(p, o, a, r, seed=s))

    def share_fn(obs, action_dict, reset_):
        """Amendment-1 PRIMARY: decoder-projected per-key ensemble
        variance of the RAW members and of the ALEATORIC-WHITENED
        members. Rev A1-B3.1: the divisor is SCALE-NORMALIZED to
        unit geometric mean per anchor, sigma_tilde = sigma /
        gmean_dims(sigma) — the whitening then carries ONLY the
        cross-dim reallocation (the estimand) with no net magnitude
        change: under uniform sigma it is the IDENTITY (white == mu
        bitwise), so the R3-B1 constant-rescaling pathology is
        excluded by construction, and the decode stays in the raw
        regime (no clip-saturation blowup). Saturation/extrapolation
        diagnostics recorded and reader-GATED (A1-B3.2)."""
        B = reset_.shape[0]
        post = encode(obs, action_dict, reset_)
        anchor = {"deter": post["deter"][:, -1:],
                  "stoch": post["stoch"][:, -1:]}
        feat = model.feat2tensor(anchor)
        actvec = jnp.concatenate(
            [action_dict[k][:, -1:].reshape((B, 1, -1))
             for k in act_keys], -1)
        mu = f32(model.disag.predict(feat, actvec))     # (E, B, 1, t)
        lv = f32(model.disag.predict_logvar(feat, actvec))
        white, sigma_tilde, dev = whiten_members(mu, lv)
        # A1-B3.2 diagnostics: whitened-prob clip saturation +
        # whitening gain per anchor
        wprobs = white[..., ddim:]
        clip_frac = (wprobs <= 1e-6).astype(f32).mean((0, -1))
        gain = (jnp.abs(dev / sigma_tilde[None]).mean((0, -1))
                / (jnp.abs(dev).mean((0, -1)) + 1e-12))
        out = {"share_clip_frac": clip_frac[:, 0],
               "share_gain": gain[:, 0]}
        vr = _proj_vars(mu, B)
        vw = _proj_vars(white, B)
        for k in obs_keys:
            out[f"pk_raw_{k}"] = vr[k]
            out[f"pk_norm_{k}"] = vw[k]
        return out

    jit_share = jax.jit(
        lambda p, o, a, r, s: nj.pure(share_fn)(p, o, a, r, seed=s))

    def rewards(obs_arrays):
        outs = []
        for i in range(0, S, args.ep_batch):
            sl = slice(i, min(i + args.ep_batch, S))
            o = {k: jnp.asarray(obs_arrays[k][sl])
                 for k in obs_keys + list(EXCLUDE)
                 if k in obs_arrays}
            a = {k: jnp.asarray(obs_arrays[k][sl]) for k in act_keys}
            _, res = jit_anchor(params, o, a, jnp.asarray(reset[sl]),
                                args.seed + i)
            outs.append({k: np.asarray(v, np.float64)
                         for k, v in res.items()})
        return {k: np.concatenate([o[k] for o in outs], 0)
                for k in outs[0]}

    dup0_equal = bool(np.array_equal(arrays.get("planted_dup0"),
                                     arrays[source_key]))
    assert dup0_equal, "planted_dup0 != source — construction violated"

    variants, mask_info = build_masks(
        arrays, source_key, mask_channels,
        np.random.default_rng(args.seed + 13))
    from embodied.envs.planted import EPS_LADDER
    for idx in (1, 2):
        ch = f"planted_dup{idx}"
        if ch not in arrays:
            continue
        rrng = np.random.default_rng(args.seed + 1300 + idx)
        v = dict(arrays)
        v[ch] = (arrays[source_key]
                 + rrng.normal(0.0, EPS_LADDER[idx] * planted_basesd,
                               arrays[source_key].shape)
                 ).astype(arrays[ch].dtype)
        variants[f"{ch}_resample"] = v
        mask_info[f"{ch}_resample"] = dict(
            form="fresh_resample",
            eps=float(EPS_LADDER[idx] * planted_basesd))
    if "velocity" in arrays:
        vrng = np.random.default_rng(args.seed + 17)
        perm = vrng.permutation(S)
        v = dict(arrays)
        v["velocity"] = arrays["velocity"][perm]
        variants["velocity_control"] = v
        mask_info["velocity_control"] = dict(
            form="batch_permutation_real_key_control",
            fixed_points=int((perm == np.arange(S)).sum()))
    all_channels = list(mask_channels) + [
        k for k in ("planted_dup1_resample", "planted_dup2_resample",
                    "velocity_control") if k in variants]

    # ---- Amendment-1 share pass (base anchors only; no variants)
    souts = []
    for i in range(0, S, args.ep_batch):
        sl = slice(i, min(i + args.ep_batch, S))
        o = {k: jnp.asarray(arrays[k][sl])
             for k in obs_keys + list(EXCLUDE) if k in arrays}
        a = {k: jnp.asarray(arrays[k][sl]) for k in act_keys}
        _, res = jit_share(params, o, a, jnp.asarray(reset[sl]),
                           args.seed + 7000 + i)
        souts.append({k: np.asarray(v, np.float64)
                      for k, v in res.items()})
    share_out = {k: np.concatenate([o[k] for o in souts], 0)
                 for k in souts[0]}
    # data-variance normalizers from the anchor windows (se_probe
    # idiom: per-dim std over S*L, symlog per decoder, house floor)
    raw_norms = {}
    for k in obs_keys:
        v = np.asarray(arrays[k], np.float32).reshape(S * L, -1)
        v = np_symlog(v) if dec_symlog else v
        raw_norms[k] = v.std(0).astype(np.float64)
    # A1-M6: real keys per the se_probe convention (PLANTED_KEYS
    # excluded — const is NOT a real key), so the floor is the
    # theta_1 machinery's floor verbatim
    real_keys = [k for k in obs_keys if k not in PLANTED_KEYS]
    real_scale = float(np.mean(np.concatenate(
        [raw_norms[k] for k in real_keys])))
    norm_floor = 0.05 * real_scale
    norms = {k: np.maximum(raw_norms[k], norm_floor)
             for k in obs_keys}
    # A1-B5: the REGISTERED share universe — planted_const is the
    # decoder-projection floor DIAGNOSTIC (se_probe review #21 B3)
    # and never enters a fire-bearing simplex; its rows are still
    # saved for the floor diagnostic
    share_universe = [k for k in obs_keys if k != "planted_const"]
    pk_raw = {k: (share_out[f"pk_raw_{k}"]
                  / (norms[k] ** 2)[None]).mean(-1)
              for k in obs_keys}                          # (S,)
    pk_norm = {k: (share_out[f"pk_norm_{k}"]
                   / (norms[k] ** 2)[None]).mean(-1)
               for k in obs_keys}
    tot_raw = float(np.sum([pk_raw[k].mean()
                            for k in share_universe]))
    tot_norm = float(np.sum([pk_norm[k].mean()
                             for k in share_universe]))
    shares = {k: dict(raw=float(pk_raw[k].mean() / tot_raw),
                      norm=float(pk_norm[k].mean() / tot_norm))
              for k in share_universe}

    base = rewards(arrays)
    channels, dnpz = {}, {}
    for ch in all_channels:
        masked = rewards(variants[ch])
        rec = dict(mask_info[ch])
        for stat in ("raw", "norm"):
            d = (masked[stat] - base[stat]).astype(np.float64)
            chkey = zlib.crc32(f"{ch}:{stat}".encode()) % 2 ** 16
            obs_m, p_red, p_inf = signflip_p(
                d, args.n_perm,
                np.random.default_rng(args.seed + chkey + 101))
            _, lo, hi = bca_interval(
                d, np.random.default_rng(args.seed + chkey + 211),
                args.n_boot)
            rec[f"delta_{stat}_mean"] = obs_m
            rec[f"delta_{stat}_bca"] = [lo, hi]
            rec[f"p_reduce_{stat}"] = p_red
            dnpz[f"delta_{stat}_{ch}"] = d
        channels[ch] = rec
    result = dict(run_logdir=os.path.abspath(args.run_logdir),
                  ckpt=os.path.abspath(ckpt), n_eval=S,
                  seed=args.seed, train_seed=int(config.seed),
                  task=str(config.task), source_key=source_key,
                  disag_head="gauss",
                  dup0_bitwise_equal_source=dup0_equal,
                  base_raw_mean=float(base["raw"].mean()),
                  base_norm_mean=float(base["norm"].mean()),
                  shares=shares,
                  share_universe=share_universe,
                  share_raw_distractor=shares["distractor"]["raw"],
                  share_norm_distractor=shares["distractor"]["norm"],
                  share_norm_floor=float(norm_floor),
                  share_floor_bound_keys=[
                      k for k in obs_keys
                      if bool((raw_norms[k] < norm_floor).any())],
                  share_clip_frac_mean=float(
                      share_out["share_clip_frac"].mean()),
                  share_gain_mean=float(
                      share_out["share_gain"].mean()),
                  share_gain_p95=float(np.percentile(
                      share_out["share_gain"], 95)),
                  gauss_diagnostics=dict(
                      logvar_min=lv_lo, logvar_max=lv_hi,
                      alea_mean=float(base["alea_mean"].mean()),
                      alea_std=float(base["alea_mean"].std()),
                      lv_frac_lo=float(base["lv_frac_lo"].mean()),
                      lv_frac_hi=float(base["lv_frac_hi"].mean()),
                      base_norm_deter_mean=float(
                          base["norm_deter"].mean()),
                      base_norm_stoch_mean=float(
                          base["norm_stoch"].mean())),
                  estimand="population-intervention deltas on BOTH "
                           "the raw (var-of-mu) and deployed "
                           "(aleatoric-normalized) rewards; anchor "
                           "p/BCa diagnostic-not-calibrated",
                  channels=channels)
    with open(os.path.join(out_dir, "se_alea_mask.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez(os.path.join(out_dir, "se_alea_mask.npz"),
             base_raw=base["raw"], base_norm=base["norm"],
             base_alea=base["alea_mean"],
             base_lv_frac_lo=base["lv_frac_lo"],
             base_lv_frac_hi=base["lv_frac_hi"],
             base_norm_deter=base["norm_deter"],
             base_norm_stoch=base["norm_stoch"],
             **{f"pkvar_raw_{k}": pk_raw[k] for k in obs_keys},
             **{f"pkvar_norm_{k}": pk_norm[k] for k in obs_keys},
             **{f"sharenorms_{k}": norms[k] for k in obs_keys},
             share_clip_frac=share_out["share_clip_frac"],
             share_gain=share_out["share_gain"],
             **dnpz)
    print(json.dumps({ch: dict(
        raw=round(c["delta_raw_mean"], 6),
        norm=round(c["delta_norm_mean"], 6))
        for ch, c in channels.items()}, indent=1))
    return result


def selfcheck():
    """Mechanics on the local gauss-head smoke checkpoint with a
    FABRICATED synthetic replay (the tiny smoke never flushes
    chunks; the real 5e5 wave runs always do): dup0 exact zero on
    BOTH statistics, determinism, finite, resample/velocity
    variants present. Exercises the REAL gauss model end-to-end on
    synthetic observations — a mechanics gate, never an outcome."""
    import tempfile
    smoke = os.environ.get(
        "ALEA_SMOKE_DIR",
        "/tmp/claude-1000/-home-rickybao-projects-dreamerv3/"
        "ea924efe-0382-4de8-b4d2-30244c36a0ee/scratchpad/b4_smoke3")
    assert os.path.isdir(os.path.join(smoke, "ckpt")), (
        f"gauss smoke ckpt not found at {smoke} (set ALEA_SMOKE_DIR)")
    tmp_root = tempfile.mkdtemp(prefix="alea_sc_")
    rd = os.path.join(tmp_root, "run")
    os.makedirs(os.path.join(rd, "replay"))
    os.symlink(os.path.join(smoke, "config.yaml"),
               os.path.join(rd, "config.yaml"))
    os.symlink(os.path.join(smoke, "ckpt"), os.path.join(rd, "ckpt"))
    # fabricate one replay chunk in the house naming
    # (time-uuid-succ-length.npz, ZERO_UUID terminator)
    rng = np.random.default_rng(4)
    T = 1000
    pos = rng.normal(0, 0.3, (T, 8)).astype(np.float32)
    from embodied.envs.planted import EPS_LADDER
    chunk = dict(
        position=pos,
        velocity=rng.normal(0, 1, (T, 9)).astype(np.float32),
        distractor=rng.normal(0, 1.2, (T, 8)).astype(np.float32),
        planted_dup0=pos.copy(),
        planted_dup1=(pos + rng.normal(0, EPS_LADDER[1] * 0.0976,
                                       (T, 8))).astype(np.float32),
        planted_dup2=(pos + rng.normal(0, EPS_LADDER[2] * 0.0976,
                                       (T, 8))).astype(np.float32),
        planted_const=np.zeros((T, 4), np.float32),
        action=rng.uniform(-1, 1, (T, 6)).astype(np.float32),
        reward=np.zeros(T, np.float32),
        is_first=(np.arange(T) % 250 == 0),
        is_last=np.zeros(T, bool),
        is_terminal=np.zeros(T, bool))
    np.savez(os.path.join(
        rd, "replay",
        f"20260822T000000-{'a' * 22}-{'0' * 22}-{T}.npz"), **chunk)
    args = parse_args([
        "--run_logdir", rd, "--n_eval", "48", "--burn_in", "12",
        "--n_perm", "200", "--n_boot", "200", "--ep_batch", "24",
        "--platform", "cpu",
        "--output", str(REPO / "local_results" / "uncfield"
                        / "se_alea_mask_smoke")])
    r1 = run(args)
    c0 = r1["channels"]["planted_dup0"]
    assert c0["delta_raw_mean"] == 0.0 and \
        c0["delta_norm_mean"] == 0.0, c0
    for ch, rec in r1["channels"].items():
        assert np.isfinite(rec["delta_raw_mean"])
        assert np.isfinite(rec["delta_norm_mean"])
    for ch in ("planted_dup1_resample", "velocity_control"):
        assert ch in r1["channels"], ch
    gd = r1["gauss_diagnostics"]
    assert gd["logvar_min"] == -8.0 and gd["logvar_max"] == 6.0
    assert np.isfinite(gd["alea_mean"]) and gd["alea_mean"] > 0
    for k in ("lv_frac_lo", "lv_frac_hi"):
        assert 0.0 <= gd[k] <= 1.0, (k, gd[k])
    assert np.isfinite(gd["base_norm_deter_mean"])
    assert np.isfinite(gd["base_norm_stoch_mean"])
    # A1-B4: the whitening identity property — uniform sigma is the
    # exact identity transform (unit test of the REAL formula)
    import jax.numpy as _jnp
    rngw = np.random.default_rng(5)
    # Uniform sigma: sigma_tilde is EXACTLY ones (pow-2 dims make
    # the mean reduction exact), and the transform is the identity
    # up to the mu_bar+dev reassembly roundoff (~2e-7 in f32 —
    # perturbs shares at O(1e-10), i.e. R3-B1 excluded)
    mu_t = _jnp.asarray(rngw.normal(0, 1, (4, 6, 1, 16)),
                        _jnp.float32)
    lv_c = _jnp.full((4, 6, 1, 16), -2.7, _jnp.float32)
    w_c, st_c, _ = whiten_members(mu_t, lv_c)
    assert np.array_equal(np.asarray(st_c),
                          np.ones_like(np.asarray(st_c))), (
        "uniform sigma must give sigma_tilde == 1 exactly")
    assert np.allclose(np.asarray(w_c), np.asarray(mu_t),
                       rtol=0, atol=1e-6)
    # non-power-of-two dims (the deployed 640 case)
    mu_o = _jnp.asarray(rngw.normal(0, 1, (4, 6, 1, 12)),
                        _jnp.float32)
    lv_o = _jnp.full((4, 6, 1, 12), -2.7, _jnp.float32)
    w_o, _, _ = whiten_members(mu_o, lv_o)
    assert np.allclose(np.asarray(w_o), np.asarray(mu_o),
                       rtol=0, atol=1e-6)
    lv_v = _jnp.asarray(rngw.normal(-1, 1, (4, 6, 1, 16)),
                        _jnp.float32)
    w_v, _, _ = whiten_members(mu_t, lv_v)
    assert np.abs(np.asarray(w_v)
                  - np.asarray(mu_t)).max() > 1e-3, (
        "non-uniform sigma must actually reallocate")
    # Amendment-1 shares: present, valid simplex over the PINNED
    # universe (const excluded), both statistics
    sh = r1["shares"]
    assert "planted_const" not in sh and \
        list(r1["share_universe"]) == sorted(
            r1["share_universe"]) and \
        "planted_const" not in r1["share_universe"], r1[
        "share_universe"]
    assert set(sh) == set(r1["share_universe"])
    for stat in ("raw", "norm"):
        tot = sum(v[stat] for v in sh.values())
        assert abs(tot - 1.0) < 1e-9, (stat, tot)
        assert all(0.0 <= v[stat] <= 1.0 for v in sh.values())
    assert r1["share_raw_distractor"] == sh["distractor"]["raw"]
    assert 0.0 <= r1["share_clip_frac_mean"] <= 1.0
    assert np.isfinite(r1["share_gain_mean"]) and \
        r1["share_gain_mean"] > 0
    npz1 = np.load(os.path.join(
        args.output or "", "se_alea_mask.npz"))
    for k in sh:
        for stat in ("raw", "norm"):
            v = np.asarray(npz1[f"pkvar_{stat}_{k}"])
            assert v.size == r1["n_eval"] and np.all(
                np.isfinite(v)) and np.all(v >= 0), (k, stat)
    # whitening actually changes the projection
    assert not np.array_equal(npz1["pkvar_raw_distractor"],
                              npz1["pkvar_norm_distractor"])
    r2 = run(args)
    for ch in r1["channels"]:
        assert r1["channels"][ch]["delta_norm_mean"] == \
            r2["channels"][ch]["delta_norm_mean"], ch
    assert r1["share_norm_distractor"] == r2["share_norm_distractor"]
    print("se_alea_mask selfcheck PASS (gauss ckpt; dup0 zero on "
          "both masked statistics; finite; resample+velocity "
          "variants; gauss diagnostics; Amendment-1 shares valid "
          "simplex on both statistics w/ per-anchor rows; "
          "deterministic incl. shares)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
