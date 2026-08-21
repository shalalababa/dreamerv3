"""APT-column interventional mask instrument (Track B2;
PREREG_nfi_apt_20260821.md). Per-run, built + selfchecked BEFORE the
B2 wave's compute.

APT (k-NN particle entropy in latent space) has no ensemble, so the
attribution is interventional. THE ESTIMAND IS A POPULATION-
INTERVENTION DELTA (#29 C-B1): every window is masked simultaneously,
so each anchor's k-NN distances change through its own feature AND
its neighbors' — per-state isolation is not claimed.

MASK SUITE (post-#29 redesign — source-substitution DELETES VARIANCE
into an entropy statistic, mechanically predetermining D-family
deltas; the suite below separates content from spread):
  distractor -> batch-permutation (marginal-PRESERVING; the fire
      channel);
  dup1/dup2  -> TWO masks each: (a) source-substitution
      (spread-inclusive, DESCRIPTIVE — carries the mechanical
      cloud-contraction component) and (b) FRESH-RESAMPLE
      (dup_k := source + newly drawn N(0,(eps_k*basesd)^2), pinned
      rng — variance-matched, content-only; for i.i.d. noise the
      content beyond spread is nil, so (b) ~ 0 is the null
      PREDICTION and (a)-(b) is the mechanical component);
  dup0       -> substitution = bitwise no-op (hard-asserted), the
      exact-zero anchor;
  velocity   -> batch-permutation of a REAL key (CALIBRATION
      CONTROL: expected NEGATIVE if APT prices real cross-window
      coupling; the distractor-vs-velocity asymmetry is the
      mechanism row).
Anchor-level sign-flip p and BCa are DIAGNOSTIC, NOT CALIBRATED
(#29 C-M4: one shared k-NN graph => anchors are dependent by
construction); the cross-run rule in the prereg is the inference.

Mechanics reused verbatim from the frozen reviewed se_mask
(build_masks, signflip_p, bca_interval, window sampling, encode);
the only new element is the APT reward head at the anchor.

Run:  python -m uncfield.se_apt_mask --run_logdir <dir>
      python -m uncfield.se_apt_mask --selfcheck
"""

from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np

from uncfield.se_mask import bca_interval, build_masks, signflip_p
from uncfield.se_probe import (EXCLUDE, FIRE_KEYS, PLANTED_KEYS, REPO,
                               SMOKE, resolve_ckpt)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_logdir", default=str(SMOKE))
    p.add_argument("--output", default="",
                   help="default: <run>/se_apt_mask")
    p.add_argument("--ckpt", default="")
    p.add_argument("--n_eval", type=int, default=512)
    p.add_argument("--burn_in", type=int, default=16)
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--ep_batch", type=int, default=64)
    p.add_argument("--apt_knn", type=int, default=12)
    p.add_argument("--apt_logc", type=float, default=1.0)
    p.add_argument("--platform", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_nonapt", action="store_true",
                   help="selfcheck only: run mechanics on a non-apt "
                        "checkpoint")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def run(args):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3 import explore
    from dreamerv3.main import make_agent
    from probing.collect import load_run_config, load_frozen_agent
    from probing import probeset as probeset_mod

    out_dir = args.output or os.path.join(args.run_logdir,
                                          "se_apt_mask")
    os.makedirs(out_dir, exist_ok=True)
    config = load_run_config(args.run_logdir, args.platform, out_dir,
                             False)
    agent = make_agent(config)
    jax.config.update("jax_transfer_guard", "allow")
    model = agent.model
    ckpt = resolve_ckpt(args.run_logdir, args.ckpt)
    load_frozen_agent(agent, ckpt)
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                          agent.params)

    obs_keys = sorted(k for k, v in agent.obs_space.items()
                      if k not in EXCLUDE and len(v.shape) <= 1)
    act_keys = sorted(agent.act_space.keys())
    source_key = str(config.planted["source_key"])
    mask_channels = [k for k in obs_keys if k in FIRE_KEYS]
    assert mask_channels, obs_keys
    # #29 C-m8/m9: objective + knob pins come from the RUN CONFIG
    expl_mode = str(config.agent.expl["mode"])
    assert expl_mode == "apt" or args.allow_nonapt, (
        f"run is expl.mode={expl_mode!r}, not apt — pass "
        f"--allow_nonapt only for mechanics selfchecks")
    cfg_knn = int(config.agent.expl["apt_knn"])
    cfg_logc = float(config.agent.expl["apt_logc"])
    assert cfg_knn == args.apt_knn and         abs(cfg_logc - args.apt_logc) < 1e-12, (
        f"config apt knobs ({cfg_knn}, {cfg_logc}) != instrument "
        f"({args.apt_knn}, {args.apt_logc})")
    planted_basesd = float(config.planted["basesd"])

    rng = np.random.default_rng(args.seed)
    arrays, _, stats = probeset_mod.collect_windows(
        os.path.join(args.run_logdir, "replay"), args.burn_in,
        args.n_eval, rng, allow_fewer=True)
    arrays = {k: np.stack(v, 0) for k, v in arrays.items()}
    S = arrays["is_first"].shape[0]
    # #29 C-M5: APT magnitudes are population-size dependent — S is a
    # hard validity pin, not a soft floor
    assert S == args.n_eval, (
        f"realized S {S} != pinned population {args.n_eval} — APT "
        f"deltas are not comparable across differing S")
    L = args.burn_in
    reset = np.zeros((S, L), bool)
    reset[:, 0] = True
    print(f"apt-mask windows: {S} x {L} from {stats['streams']} "
          f"streams (requested {args.n_eval})")

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

    def anchor_feat_fn(obs, action_dict, reset_):
        post = encode(obs, action_dict, reset_)
        anchor = {"deter": post["deter"][:, -1:],
                  "stoch": post["stoch"][:, -1:]}
        return {"feat": model.feat2tensor(anchor)}

    jit_anchor = jax.jit(
        lambda p, o, a, r, s: nj.pure(anchor_feat_fn)(p, o, a, r,
                                                      seed=s))

    def anchor_feats(obs_arrays):
        outs = []
        for i in range(0, S, args.ep_batch):
            sl = slice(i, min(i + args.ep_batch, S))
            o = {k: jnp.asarray(obs_arrays[k][sl])
                 for k in obs_keys + list(EXCLUDE) if k in obs_arrays}
            a = {k: jnp.asarray(obs_arrays[k][sl]) for k in act_keys}
            _, res = jit_anchor(params, o, a, jnp.asarray(reset[sl]),
                                args.seed + i)
            outs.append(np.asarray(res["feat"]))
        return np.concatenate(outs, 0)                # (S, 1, D)

    def apt_over(feats):
        import jax.numpy as jnp
        r = explore.apt_reward(jnp.asarray(feats), args.apt_knn,
                               args.apt_logc)
        return np.asarray(r)[:, 0]                    # (S,)

    dup0_equal = bool(np.array_equal(arrays.get("planted_dup0"),
                                     arrays[source_key]))
    assert dup0_equal, \
        "replayed planted_dup0 != source — construction violated"

    variants, mask_info = build_masks(
        arrays, source_key, mask_channels,
        np.random.default_rng(args.seed + 13))
    # #29 C-B1: variance-matched fresh-resample masks for dup1/dup2
    # (content-only removal) + the velocity calibration control
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

    base_feats = anchor_feats(arrays)
    base_apt = apt_over(base_feats)
    channels, dnpz = {}, {}
    for ch in all_channels:
        masked_apt = apt_over(anchor_feats(variants[ch]))
        d = (masked_apt - base_apt).astype(np.float64)
        rec = dict(mask_info[ch])
        chkey = zlib.crc32(ch.encode()) % 2 ** 16
        obs_m, p_red, p_inf = signflip_p(
            d, args.n_perm, np.random.default_rng(args.seed + chkey
                                                  + 101))
        _, lo, hi = bca_interval(
            d, np.random.default_rng(args.seed + chkey + 211),
            args.n_boot)
        rec.update(delta_apt_mean=obs_m, delta_apt_bca=[lo, hi],
                   p_reduce=p_red, p_inflate=p_inf)
        channels[ch] = rec
        dnpz[f"delta_apt_{ch}"] = d
    result = dict(run_logdir=os.path.abspath(args.run_logdir),
                  ckpt=os.path.abspath(ckpt), n_eval=S,
                  seed=args.seed,
                  train_seed=int(config.seed),
                  task=str(config.task),
                  source_key=source_key,
                  apt_knn=args.apt_knn, apt_logc=args.apt_logc,
                  dup0_bitwise_equal_source=dup0_equal,
                  base_apt_mean=float(base_apt.mean()),
                  expl_mode=expl_mode,
                  estimand="population-intervention delta; anchor-"
                           "level p/BCa diagnostic-not-calibrated "
                           "(#29 C-M4); training-time APT population "
                           "is imagined rollouts (B*K=1024, T=16) — "
                           "this instrument is a replay-anchor PROXY "
                           "for the trained functional (#29 C-m11)",
                  channels=channels)
    with open(os.path.join(out_dir, "se_apt_mask.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez(os.path.join(out_dir, "se_apt_mask.npz"),
             base_apt=base_apt, **dnpz)
    print(json.dumps({ch: dict(d=round(c["delta_apt_mean"], 6),
                               p_red=round(c["p_reduce"], 4))
                      for ch, c in channels.items()}, indent=1))
    return result


def selfcheck():
    """Mechanics on the p2e smoke ckpt (apt_reward is a pure function
    of features, so the instrument mechanics are objective-agnostic;
    the objective-matched validation is the registered B2 smoke)."""
    args = parse_args([
        "--run_logdir", str(SMOKE), "--n_eval", "48", "--burn_in",
        "12", "--n_perm", "300", "--n_boot", "300", "--ep_batch",
        "24", "--platform", "cpu", "--allow_nonapt",
        "--output", str(REPO / "local_results" / "uncfield"
                        / "se_apt_mask_smoke"),
    ])
    r1 = run(args)
    assert r1["dup0_bitwise_equal_source"]
    c0 = r1["channels"]["planted_dup0"]
    assert c0["bitwise_noop"] and c0["delta_apt_mean"] == 0.0, c0
    for ch, rec in r1["channels"].items():
        assert np.isfinite(rec["delta_apt_mean"])
        assert rec["delta_apt_bca"][0] <= rec["delta_apt_mean"] \
            <= rec["delta_apt_bca"][1], (ch, rec)
    # redesigned suite present (#29 C-B1)
    for ch in ("planted_dup1_resample", "planted_dup2_resample",
               "velocity_control"):
        assert ch in r1["channels"], ch
        assert np.isfinite(r1["channels"][ch]["delta_apt_mean"])
    assert r1["channels"]["planted_dup1_resample"]["form"] ==         "fresh_resample"
    assert "estimand" in r1 and "train_seed" in r1
    r2 = run(args)
    for ch in r1["channels"]:
        assert r1["channels"][ch]["delta_apt_mean"] == \
            r2["channels"][ch]["delta_apt_mean"], ch
        assert r1["channels"][ch]["p_reduce"] == \
            r2["channels"][ch]["p_reduce"], ch
    ds = {ch: round(r1["channels"][ch]["delta_apt_mean"], 5)
          for ch in r1["channels"]}
    print(f"se_apt_mask selfcheck PASS (dup0 noop delta==0; "
          f"deterministic; smoke deltas {json.dumps(ds)})")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
