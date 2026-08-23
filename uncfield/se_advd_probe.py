"""Wave-B adversarial-actor evaluation probe
(PREREG_waveb_advd_20260822; the transmission-rescue slate, part 3).
Per-checkpoint instrument, built BEFORE the wave's compute.

Loads TWO agents in ONE invocation (the 1-Aug rule — no
cross-invocation pairing):
  source  — the original exploration checkpoint (Stage-1 or A1)
  adapted — the advd actor adapted on the source's FROZEN WM

Arms (FIXED order, one CUDA device, fresh env per arm — CRN):
  random  — uniform actions (normalizer source, floor anchor)
  policy  — the SOURCE checkpoint's exploration policy
  advd    — the ADAPTED adversarial actor
BOTH agent arms SAMPLE from their action distributions (Agent.policy
ignores mode outside the cemplan branch — rev WB-n4): the
comparison is stochastic-policy vs stochastic-policy, never
mean-vs-sample.

Integrity gates computed HERE (the reader asserts them):
  - WM-IDENTITY: every param under ^(enc|dyn|dec|disag)/ in the
    adapted checkpoint is BITWISE equal to the source's — the
    post-hoc proof the freeze held.
  - ENV-IDENTITY: the env-defining config subtrees (task,
    distractor, planted) are equal between source and adapted runs.

Measurement identical to the frozen se_planner_probe (#32 form):
online posterior states under the SOURCE WM, pre-action pairing,
random-arm normalizers, occupancy vs the pinned gate, per-step npz
+ exact-float64 json means, reset_hits == 0 asserted.

Registered imagination rows (stride-subsampled starts):
  - from POLICY-arm visited states: H-step imagined NORMALIZED
    distractor attribution under BOTH policies -> the reader's
    IN-MODEL DOMINANCE gate (the adapted actor must beat the
    incumbent on its own objective in imagination, else the
    adaptation failed).
  - from ADVD-arm visited states: the same imagined attribution
    under the advd actor -> the imagined-vs-realized GAP row
    (comparable units: both are norms^2-normalized per-dim
    distractor variance — the Wave-A off-support-imagination
    mechanism, now on an amortized attacker).

Run:  python -m uncfield.se_advd_probe --run_logdir <adapted dir>
          --source_logdir <source dir> [--smoke]
      python -m uncfield.se_advd_probe --selfcheck   (pure helpers)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np

from uncfield.se_probe import (EXCLUDE, PLANTED_KEYS, SMOKE, np_symlog,
                               resolve_ckpt)
from uncfield.se_m3_read import (GATE_INDEX, GATE_KEY, GATE_THRESHOLD)

ARMS = ("random", "policy", "advd")
WM_REGEX = r"^(enc|dyn|dec|disag)/"
IMAG_STRIDE = 5


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_logdir", default="",
                   help="the ADAPTED (advd) run dir")
    p.add_argument("--source_logdir", default="",
                   help="the SOURCE (Stage-1/A1) run dir")
    p.add_argument("--output", default="",
                   help="default: <adapted run>/se_advd")
    p.add_argument("--ckpt", default="")
    p.add_argument("--source_ckpt", default="")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--ep_len", type=int, default=500)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--platform", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def wm_identity(source_params, adapted_params, regex=WM_REGEX):
    """Bitwise WM+disag identity between two param dicts (numpy).
    Returns (ok, n_checked, mismatched_keys). The FREEZE gate."""
    pat = re.compile(regex)
    src = {k: v for k, v in source_params.items() if pat.match(k)}
    adp = {k: v for k, v in adapted_params.items() if pat.match(k)}
    assert src, "no wm/disag params matched on the source side"
    assert set(src) == set(adp), (
        "wm/disag param key sets differ",
        sorted(set(src) ^ set(adp))[:5])
    bad = [k for k in sorted(src)
           if not np.array_equal(np.asarray(src[k]),
                                 np.asarray(adp[k]))]
    return (not bad), len(src), bad


_ENV_KEYS = ("task", "distractor", "planted")


def _load_env_defaults():
    import yaml
    p = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "dreamerv3", "configs.yaml")
    with open(p) as f:
        return yaml.safe_load(f)["defaults"]


def env_identity(source_cfg, adapted_cfg, defaults=None):
    """Env-defining subtrees must match AFTER normalizing both sides
    through today's config defaults (rev WB-B1: the Stage-1 sources
    predate optional keys like distractor.mod_*/gate_* — a raw
    comparison would abort every ambient probe on schema
    generation, not on any real env difference; the probing/collect
    load path uses the same defaults-overlay idiom). A key present
    on one side but absent from the defaults still surfaces as a
    diff — the overlay cannot mask a genuine drift."""
    defaults = defaults or _load_env_defaults()

    def view(cfg):
        out = {}
        for k in _ENV_KEYS:
            v, d = cfg.get(k), defaults.get(k)
            if isinstance(d, dict):
                m = dict(d)
                m.update(v or {})
                out[k] = m
            else:
                out[k] = v
        return out

    s, a = view(source_cfg), view(adapted_cfg)
    diffs = [k for k in _ENV_KEYS if s[k] != a[k]]
    return (not diffs), diffs


def run(args):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3.main import make_agent, make_env
    from probing.collect import load_run_config, load_frozen_agent

    if args.smoke:
        args.episodes, args.ep_len = 1, 40
        args.horizon, args.chunk = 6, 20

    f32 = jnp.float32
    out_dir = args.output or os.path.join(args.run_logdir, "se_advd")
    os.makedirs(out_dir, exist_ok=True)

    import yaml
    with open(os.path.join(args.run_logdir, "config.yaml")) as f:
        adapted_cfg_raw = yaml.safe_load(f)
    with open(os.path.join(args.source_logdir, "config.yaml")) as f:
        source_cfg_raw = yaml.safe_load(f)
    env_ok, env_diffs = env_identity(source_cfg_raw, adapted_cfg_raw)
    assert env_ok, f"env-defining config differs: {env_diffs}"
    assert str(adapted_cfg_raw["agent"]["expl"]["mode"]) == "advd", (
        adapted_cfg_raw["agent"]["expl"]["mode"])

    # rev WB-n3: source<->adapted ARCHITECTURE parity — the validity
    # condition of the params-substitution imagination rows (the
    # source-built module tree evaluated with the adapted params)
    def _arch_view(c):
        ag = c["agent"]
        typ = ag["dyn"]["typ"]
        return dict(
            dyn={k: ag["dyn"][typ][k]
                 for k in ("deter", "stoch", "classes")},
            policy_units=ag["policy"]["units"],
            value_units=ag["value"]["units"],
            disag=dict(
                ens=ag["expl"]["disag_ens"],
                head=ag["expl"].get("disag_head", "det"),
                target=ag["expl"]["disag_target"],
                units=ag["expl"]["disag_units"],
                layers=ag["expl"]["disag_layers"]))
    arch_s, arch_a = (_arch_view(source_cfg_raw),
                      _arch_view(adapted_cfg_raw))
    assert arch_s == arch_a, (
        "source/adapted architecture parity broken — the "
        "params-substitution imagination rows would be invalid",
        arch_s, arch_a)

    # SOURCE agent: the measurement WM + the incumbent policy
    src_out = os.path.join(out_dir, "_src_scope")
    config = load_run_config(args.source_logdir, args.platform,
                             src_out, False)
    agent = make_agent(config)
    jax.config.update("jax_transfer_guard", "allow")
    model = agent.model
    assert model.disag is not None
    ens = int(config.agent.expl.disag_ens)
    dec_symlog = bool(model.dec.symlog)
    src_ckpt = resolve_ckpt(args.source_logdir, args.source_ckpt)
    load_frozen_agent(agent, src_ckpt)
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                          agent.params)

    # ADAPTED agent: the adversarial actor (its own config/ckpt)
    adp_out = os.path.join(out_dir, "_adp_scope")
    config_a = load_run_config(args.run_logdir, args.platform,
                               adp_out, False)
    agent_a = make_agent(config_a)
    # make_agent re-arms the transfer guard; the probe moves numpy
    # arrays host<->device throughout
    jax.config.update("jax_transfer_guard", "allow")
    adp_ckpt = resolve_ckpt(args.run_logdir, args.ckpt)
    load_frozen_agent(agent_a, adp_ckpt)
    params_a = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                            agent_a.params)

    # THE FREEZE GATE
    wm_ok, wm_n, wm_bad = wm_identity(params, params_a)
    assert wm_ok, (
        f"WM/disag NOT frozen: {len(wm_bad)} of {wm_n} params "
        f"differ, e.g. {wm_bad[:3]}")

    obs_keys = sorted(k for k, v in agent.obs_space.items()
                      if k not in EXCLUDE and len(v.shape) <= 1)
    act_keys = sorted(agent.act_space.keys())
    assert act_keys == ["action"], act_keys
    adim = int(np.prod(agent.act_space["action"].shape))
    real_keys = [k for k in obs_keys if k not in PLANTED_KEYS]

    def postsplit(pred):
        ddim = model.dyn.deter
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

    def member_vars(feat_t, actvec, B, T):
        preds = f32(model.disag.predict(feat_t, actvec))
        means = {k: [] for k in obs_keys}
        for e in range(ens):
            det, probs = postsplit(preds[e])
            dec = decode_feats(dict(deter=det, stoch=probs), B, T)
            for k in obs_keys:
                means[k].append(dec[k])
        return {k: jnp.stack(means[k], 0).var(0) for k in obs_keys}

    def init_fn():
        return dict(enc=model.enc.initial(1), dyn=model.dyn.initial(1))

    def obs_step_fn(enc_carry, dyn_carry, obs, prevact, reset_):
        enc_carry, _, tokens = model.enc(enc_carry, obs, reset_,
                                         training=False)
        dyn_carry, _, feat = model.dyn.observe(
            dyn_carry, tokens, prevact, reset_, training=False)
        return dict(enc=enc_carry, dyn=dyn_carry,
                    deter=feat["deter"][:, -1],
                    stoch=feat["stoch"][:, -1])

    sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)

    def attr_fn(deter, stoch, actions):
        feat_t = model.feat2tensor({"deter": deter, "stoch": stoch})
        out = {"intrinsic": f32(model.disag.reward(feat_t, actions))}
        mv = member_vars(feat_t, actions, deter.shape[0],
                         deter.shape[1])
        for k in obs_keys:
            out[f"var_{k}"] = mv[k]
        return out

    jit_init = jax.jit(lambda p, s: nj.pure(init_fn)(p, seed=s))
    jit_obs_step = jax.jit(
        lambda p, ec, dc, o, pa, r, s: nj.pure(obs_step_fn)(
            p, ec, dc, o, pa, r, seed=s))
    jit_attr = jax.jit(
        lambda p, d, st, a, s: nj.pure(attr_fn)(p, d, st, a, seed=s))

    # imagination under EITHER agent's own policy: the wm inside
    # each params dict is bitwise identical (asserted), so the only
    # live difference is the policy head — exactly the estimand
    H = args.horizon

    def imag_fn(deter, stoch):
        B = deter.shape[0]
        carry = {"deter": deter, "stoch": stoch}
        policyfn = lambda feat: sample(model.pol(
            model.feat2tensor(feat), 1))
        _, imgfeat, imgact = model.dyn.imagine(carry, policyfn, H,
                                               training=False)
        states = {
            "deter": jnp.concatenate(
                [deter[:, None], imgfeat["deter"][:, :-1]], 1),
            "stoch": jnp.concatenate(
                [stoch[:, None], imgfeat["stoch"][:, :-1]], 1)}
        actvec = jnp.concatenate(
            [imgact[k].reshape((B, H, -1)) for k in act_keys], -1)
        feat_t = model.feat2tensor(states)
        mv = member_vars(feat_t, actvec, B, H)
        return {f"var_{k}": mv[k] for k in obs_keys}

    jit_imag = jax.jit(
        lambda p, d, st, s: nj.pure(imag_fn)(p, d, st, seed=s))

    def roll_arm(arm, seed_base):
        env = make_env(config, 0)                     # CRN per arm
        zero_act = {k: np.zeros(v.shape, v.dtype)
                    for k, v in env.act_space.items() if k != "reset"}
        E, L = args.episodes, args.ep_len
        obs_rec = {k: [] for k in obs_keys}
        act_rec, rew_rec = [], []
        det_rec, sto_rec = [], []
        reset_hits = 0
        pol_carry = agent.init_policy(batch_size=1)
        pol_carry_a = agent_a.init_policy(batch_size=1)
        t0 = time.time()
        for e in range(E):
            obs = env.step({**zero_act, "reset": np.array(True)})
            _, init = jit_init(params, 0)
            enc_carry, dyn_carry = init["enc"], init["dyn"]
            prev_act = np.zeros((1, 1, adim), np.float32)
            first = True
            for t in range(L):
                if bool(obs["is_last"]):
                    reset_hits += 1
                    obs = env.step({**zero_act,
                                    "reset": np.array(True)})
                    first = True
                step_seed = (seed_base + e * 100003 + t * 7)
                o = {k: jnp.asarray(
                    np.asarray(obs[k], np.float32)[None, None])
                    for k in obs_keys}
                for k in EXCLUDE:
                    if k in obs:
                        o[k] = jnp.asarray(
                            np.asarray(obs[k])[None, None])
                _, st = jit_obs_step(
                    params, enc_carry, dyn_carry, o,
                    {"action": jnp.asarray(prev_act)},
                    jnp.asarray([[first]]), step_seed)
                enc_carry, dyn_carry = st["enc"], st["dyn"]
                det_rec.append(np.asarray(st["deter"][0]))
                sto_rec.append(np.asarray(st["stoch"][0]))
                agent_obs = {k: np.asarray(v)[None]
                             for k, v in obs.items()
                             if not k.startswith("log/")}
                if arm == "random":
                    rng = np.random.default_rng(step_seed)
                    act = rng.uniform(-1.0, 1.0, adim).astype(
                        np.float32)
                elif arm == "policy":
                    pol_carry, acts, _ = agent.policy(
                        pol_carry, agent_obs, mode="eval")
                    act = np.asarray(acts["action"][0], np.float32)
                else:
                    pol_carry_a, acts, _ = agent_a.policy(
                        pol_carry_a, agent_obs, mode="eval")
                    act = np.asarray(acts["action"][0], np.float32)
                for k in obs_keys:
                    obs_rec[k].append(np.asarray(obs[k], np.float32))
                act_rec.append(act)
                prev_act = act[None, None]
                first = False
                obs = env.step({"action": act,
                                "reset": np.array(False)})
                rew_rec.append(float(np.asarray(obs["reward"])))
        env.close()
        assert reset_hits == 0, (
            f"{arm}: {reset_hits} mid-episode env resets")
        return dict(obs={k: np.stack(v, 0).reshape(E, L, -1)
                         for k, v in obs_rec.items()},
                    actions=np.stack(act_rec, 0).reshape(E, L, adim),
                    deter=np.stack(det_rec, 0).reshape(E, L, -1),
                    stoch=np.stack(sto_rec, 0).reshape(
                        E, L, *sto_rec[0].shape),
                    rewards=np.asarray(rew_rec, np.float64),
                    reset_hits=reset_hits,
                    elapsed=time.time() - t0)

    def compute_norms(random_obs):
        raw = {}
        for k in obs_keys:
            v = random_obs[k].reshape(-1, random_obs[k].shape[-1])
            v = np_symlog(v) if dec_symlog else v
            raw[k] = v.std(0).astype(np.float64)
        real_scale = float(np.mean(np.concatenate(
            [raw[k] for k in real_keys])))
        floor = 0.05 * real_scale
        return ({k: np.maximum(raw[k], floor) for k in obs_keys},
                raw, floor)

    def attribute(arm_data, norms):
        L = args.ep_len
        outs = []
        for i in range(0, L, args.chunk):
            sl = slice(i, min(i + args.chunk, L))
            _, res = jit_attr(
                params, jnp.asarray(arm_data["deter"][:, sl]),
                jnp.asarray(arm_data["stoch"][:, sl]),
                jnp.asarray(arm_data["actions"][:, sl]),
                args.seed + 91 + i)
            outs.append({k: np.asarray(v) for k, v in res.items()})
        res = {k: np.concatenate([o[k] for o in outs], 1)
               for k in outs[0]}
        intr = np.asarray(res["intrinsic"], np.float64).reshape(-1)
        attr = {}
        for k in obs_keys:
            v = np.asarray(res[f"var_{k}"], np.float64)
            attr[k] = ((v / (norms[k] ** 2)[None, None]).mean(-1)
                       .reshape(-1))
        pos = arm_data["obs"][GATE_KEY][..., GATE_INDEX].reshape(-1)
        occ = (pos > GATE_THRESHOLD).astype(np.float64)
        return attr, intr, occ

    def imag_attr(arm_data, norms, whose_params, tag):
        """Imagined NORMALIZED distractor attribution over H-step
        rollouts under `whose_params`' policy, from stride-sampled
        visited states of `arm_data`. Returns per-start means."""
        det = arm_data["deter"].reshape(
            -1, arm_data["deter"].shape[-1])[::IMAG_STRIDE]
        sto = arm_data["stoch"].reshape(
            -1, *arm_data["stoch"].shape[2:])[::IMAG_STRIDE]
        outs = []
        bs = max(1, args.chunk)
        for i in range(0, det.shape[0], bs):
            _, res = jit_imag(whose_params,
                              jnp.asarray(det[i:i + bs]),
                              jnp.asarray(sto[i:i + bs]),
                              args.seed + 7000 + i)
            v = np.asarray(res["var_distractor"], np.float64)
            outs.append(
                (v / (norms["distractor"] ** 2)[None, None])
                .mean(-1).mean(-1))          # mean dims, mean horizon
        return np.concatenate(outs, 0)       # (n_starts,)

    t_start = time.time()
    results, npz_out, arm_data = {}, {}, {}
    norms = norms_raw = floor = None
    for ai, arm in enumerate(ARMS):
        data = roll_arm(arm, args.seed + 1_000_003 * (ai + 1))
        arm_data[arm] = data
        if arm == "random":
            norms, norms_raw, floor = compute_norms(data["obs"])
        print(f"[{arm}] rolled in {data['elapsed']:.1f}s", flush=True)
    for arm in ARMS:
        attr, intr, occ = attribute(arm_data[arm], norms)
        rec = dict(
            n_steps=int(intr.size),
            attr_mean={k: float(attr[k].mean()) for k in obs_keys},
            intrinsic_mean=float(intr.mean()),
            occupancy=float(occ.mean()),
            return_mean=float(np.mean(arm_data[arm]["rewards"])),
            reset_hits=arm_data[arm]["reset_hits"],
            elapsed_s=arm_data[arm]["elapsed"])
        results[arm] = rec
        for k in obs_keys:
            npz_out[f"{arm}_attr_{k}"] = attr[k]
        npz_out[f"{arm}_intr"] = intr
        npz_out[f"{arm}_occ"] = occ

    # registered imagination rows
    imag = {}
    for state_arm, pol_tag, prm in (
            ("policy", "policy", params),
            ("policy", "advd", params_a),
            ("advd", "advd", params_a)):
        v = imag_attr(arm_data[state_arm], norms, prm, pol_tag)
        key = f"imag_{state_arm}states_{pol_tag}pol"
        npz_out[key] = v
        imag[key] = float(v.mean())

    import jax as _jax
    result = dict(
        run_logdir=os.path.abspath(args.run_logdir),
        source_logdir=os.path.abspath(args.source_logdir),
        ckpt=os.path.abspath(adp_ckpt),
        source_ckpt=os.path.abspath(src_ckpt),
        train_seed=int(config_a.seed),
        source_seed=int(config.seed),
        task=str(config.task),
        probe_seed=args.seed, arms=list(ARMS),
        episodes=args.episodes, ep_len=args.ep_len,
        horizon=args.horizon, imag_stride=IMAG_STRIDE,
        chunk=args.chunk,
        advd_scale=float(config_a.agent.expl.advd_scale),
        smoke=bool(args.smoke),
        wm_identity=dict(ok=bool(wm_ok), n_params=int(wm_n),
                         regex=WM_REGEX),
        env_identity=True,
        devices=str(_jax.devices()),
        backend=str(_jax.default_backend()),
        norm_floor=float(floor),
        gate=dict(key=GATE_KEY, index=GATE_INDEX,
                  threshold=GATE_THRESHOLD),
        imag=imag,
        pairing="pre-action (s_t, a_t) at ONLINE posterior states "
                "under the SOURCE WM (bitwise-identical to the "
                "adapted WM, asserted); fresh env per arm (CRN); "
                "imagination rows stride-subsampled, normalized "
                "attr units",
        per_arm=results,
        total_elapsed_s=time.time() - t_start)
    with open(os.path.join(out_dir, "se_advd.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "se_advd.npz"),
                        **npz_out,
                        **{f"norm_{k}": norms[k] for k in obs_keys},
                        **{f"rawnorm_{k}": norms_raw[k]
                           for k in obs_keys})
    print(json.dumps({a: dict(
        d=round(results[a]["attr_mean"]["distractor"], 5),
        intr=round(results[a]["intrinsic_mean"], 6)) for a in ARMS},
        indent=1))
    print("imag:", json.dumps({k: round(v, 5)
                               for k, v in imag.items()}, indent=1))
    print(f"TOTAL {result['total_elapsed_s']:.0f}s")
    return result


def selfcheck():
    """Pure-helper unit tests (the full path is exercised by the
    registered CPU smoke on the local wb_src/wb_adv pair)."""
    # wm_identity: equal dicts pass; a flipped bit is caught; a
    # policy-only diff is ignored
    rng = np.random.default_rng(0)
    src = {"enc/w": rng.normal(0, 1, (4, 4)),
           "dyn/w": rng.normal(0, 1, (8,)),
           "dec/w": rng.normal(0, 1, (3, 2)),
           "disag/m0out/kernel": rng.normal(0, 1, (5, 5)),
           "pol/w": rng.normal(0, 1, (6,))}
    adp = {k: v.copy() for k, v in src.items()}
    adp["pol/w"] = adp["pol/w"] + 1.0
    ok, n, bad = wm_identity(src, adp)
    assert ok and n == 4 and not bad, (ok, n, bad)
    adp2 = {k: v.copy() for k, v in adp.items()}
    adp2["dyn/w"][3] += 1e-7
    ok, n, bad = wm_identity(src, adp2)
    assert not ok and bad == ["dyn/w"], (ok, bad)
    # env_identity — rev WB-B1: the defaults overlay must make an
    # OLD-SCHEMA source (no mod_*/gate_* keys) equal to a new-schema
    # adapted run whose extra keys sit at their inert defaults...
    dfl = dict(task="dummy",
               distractor=dict(dim=8, scale=1.0, theta=0.1,
                               basesd=0.0, calib=1000, gate_key="",
                               gate_index=0, gate_threshold=0.0,
                               mod_key="", mod_index=0, mod_lo=0.0,
                               mod_hi=1.0),
               planted=dict(source_key="position", basesd=0.0))
    old_src = dict(task="dmc_cheetah_run",
                   distractor=dict(dim=8, scale=1.0, theta=0.1,
                                   basesd=1.215, calib=1000),
                   planted=dict(source_key="position",
                                basesd=0.0976), seed=1)
    new_adp = dict(task="dmc_cheetah_run",
                   distractor=dict(dim=8, scale=1.0, theta=0.1,
                                   basesd=1.215, calib=1000,
                                   gate_key="", gate_index=0,
                                   gate_threshold=0.0, mod_key="",
                                   mod_index=0, mod_lo=0.0,
                                   mod_hi=1.0),
                   planted=dict(source_key="position",
                                basesd=0.0976), seed=2)
    ok, diffs = env_identity(old_src, new_adp, dfl)
    assert ok and not diffs, diffs
    # ...but a GENUINE difference still fails (mod_key active)
    hot = dict(new_adp)
    hot["distractor"] = dict(new_adp["distractor"], mod_key="position",
                             mod_lo=-0.322, mod_hi=0.197)
    ok, diffs = env_identity(old_src, hot, dfl)
    assert not ok and diffs == ["distractor"]
    # ...and an unknown extra key on one side is NOT masked
    weird = dict(old_src)
    weird["distractor"] = dict(old_src["distractor"], surprise=3)
    ok, diffs = env_identity(weird, new_adp, dfl)
    assert not ok and diffs == ["distractor"]
    # ...and a scale diff fails as before
    ok, diffs = env_identity(old_src, dict(new_adp) | {
        "distractor": dict(new_adp["distractor"], scale=2.0)}, dfl)
    assert not ok and diffs == ["distractor"]
    # the REAL defaults + the REAL Stage-1 schema gap (regression
    # for the exact WB-B1 incident): a 5-key distractor source vs a
    # 12-key adapted block at inert values must PASS
    real_dfl = _load_env_defaults()
    src5 = dict(task="dmc_cheetah_run",
                distractor=dict(basesd=1.215, calib=1000, dim=8,
                                scale=1.0, theta=0.1),
                planted=dict(source_key="position", basesd=0.0976))
    adp12 = dict(task="dmc_cheetah_run",
                 distractor=dict(real_dfl["distractor"], basesd=1.215,
                                 dim=8, scale=1.0),
                 planted=dict(source_key="position", basesd=0.0976))
    ok, diffs = env_identity(src5, adp12, real_dfl)
    assert ok, diffs
    print("se_advd_probe selfcheck PASS (wm_identity: bitwise catch "
          "on a 1e-7 flip, policy diffs ignored, key-set guard; "
          "env_identity: WB-B1 defaults-overlay — old-schema source "
          "vs inert-default adapted PASSES incl. the real Stage-1 "
          "5-key regression, genuine mod drift FAILS, unknown extra "
          "key FAILS, scale drift FAILS)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
