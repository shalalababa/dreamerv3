"""Wave-A policy-seeded refinement probe
(PREREG_planner_refine_20260822.md; the transmission-rescue slate,
part 2 of 3). Per-checkpoint instrument, built + smoke-verified
BEFORE the wave's compute.

Question (Fable F2/F5 repair): is the trained policy's visitation a
LOCAL MAXIMUM of the absolute distractor-attribution field? The
ambient wave's greedy CEM planned from scratch and lost to the
amortized policy on every objective (a failed relative positive
control). Here the planner starts FROM the policy's own imagined
plan and refines:

Arms (FIXED order, one CUDA device, fresh env per arm — CRN):
  random        — uniform actions (normalizer source, floor anchor)
  policy        — the trained sampling policy (the incumbent)
  refine_attr   — CEM warm-started at the policy's H-step imagined
                  action plan (init std REFINE_STD around it),
                  objective = distractor attribution
  refine_disag  — the same warm-started CEM on the raw disag
                  objective = the BUILT-IN POSITIVE CONTROL
                  (refinement of the policy's own objective must not
                  destroy its harvest; the read gates on it)

Measurement identical to the frozen se_planner_probe (#32 form):
online posterior states, pre-action pairing, random-arm normalizers,
occupancy vs the pinned gate, per-step npz + exact-float64 json
means, CEM elite diagnostics, reset_hits == 0 asserted.

Run:  python -m uncfield.se_refine_probe --run_logdir <dir>
      python -m uncfield.se_refine_probe --run_logdir <dir> --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from uncfield.se_probe import (EXCLUDE, PLANTED_KEYS, SMOKE, np_symlog,
                               resolve_ckpt)
from uncfield.se_m3_read import (GATE_INDEX, GATE_KEY, GATE_THRESHOLD)
from uncfield.se_planner_probe import cem_elite, pre_action_states

ARMS = ("random", "policy", "refine_attr", "refine_disag")
PLANNER_ARMS = ("refine_attr", "refine_disag")
REFINE_STD = 0.2
STD_FLOOR = 0.02
CEM_KEYS = ("rand_mean", "elite_first", "elite_final", "best_first",
            "best_final", "warm_score", "clip_frac", "refine_disp")


def refine_loop(mean, std, score_of, N, K, I, seed_fn, floor):
    """The Wave-A CEM refinement loop, factored out so --selfcheck
    exercises the REAL code (rev A-M4). Candidate 0 of iteration 0
    IS the warm-start plan (eps[0]=0 — rev A-B1), so
    diags['warm_score'] is the warm plan's own objective value and
    best_first >= warm_score by construction. diags also record the
    first-action displacement of the refined plan from the warm
    start and the iteration-0 clip fraction (rev A-M8)."""
    import jax
    import jax.numpy as jnp
    warm_first_action = mean[0]
    diags = {}
    e_mean = best = None
    for it in range(I):
        eps = jax.random.normal(seed_fn(it), (N,) + mean.shape)
        if it == 0:
            eps = eps.at[0].set(0.0)
        cand = jnp.clip(mean[None] + std[None] * eps, -1.0, 1.0)
        score = score_of(cand)
        if it == 0:
            diags["warm_score"] = score[0]
            diags["rand_mean"] = score.mean()
            diags["clip_frac"] = (jnp.abs(cand)
                                  >= 1.0 - 1e-6).mean()
        mean, std, e_mean, best = cem_elite(cand, score, K, floor)
        if it == 0:
            diags["elite_first"] = e_mean
            diags["best_first"] = best
    diags["elite_final"] = e_mean
    diags["best_final"] = best
    diags["refine_disp"] = jnp.linalg.norm(
        mean[0] - warm_first_action)
    return mean, diags


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_logdir", default=str(SMOKE))
    p.add_argument("--output", default="",
                   help="default: <run>/se_refine")
    p.add_argument("--ckpt", default="")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--ep_len", type=int, default=500)
    p.add_argument("--n_cand", type=int, default=256)
    p.add_argument("--n_elite", type=int, default=32)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--platform", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def selfcheck():
    """Unit test of the REAL refine_loop (rev A-M4) on a smooth toy
    objective: candidate 0 of iteration 0 is bitwise the warm plan;
    best_first >= warm_score by construction; refinement moves the
    plan toward the optimum; diagnostics present and finite."""
    import jax
    import jax.numpy as jnp
    H, adim, N, K, I = 6, 3, 64, 8, 3
    rng = np.random.default_rng(0)
    target = jnp.asarray(rng.uniform(-0.8, 0.8, (H, adim)),
                         jnp.float32)
    warm = jnp.zeros((H, adim), jnp.float32)

    def score_of(cand):
        return -((cand - target[None]) ** 2).mean((1, 2))

    keys = [jax.random.PRNGKey(i) for i in range(I)]
    mean, diags = refine_loop(warm, 0.2 * jnp.ones((H, adim)),
                              score_of, N, K, I,
                              lambda it: keys[it], STD_FLOOR)
    assert mean.shape == (H, adim), mean.shape
    ws = float(diags["warm_score"])
    assert abs(ws - float(score_of(warm[None])[0])) < 1e-6, (
        "candidate 0 of iter 0 is not the warm plan")
    assert float(diags["best_first"]) >= ws - 1e-9
    assert float(((mean - target) ** 2).mean()) < \
        float(((warm - target) ** 2).mean()), "no refinement progress"
    for k in CEM_KEYS:
        assert k in diags and np.isfinite(float(diags[k])), k
    assert float(diags["refine_disp"]) > 0
    assert 0.0 <= float(diags["clip_frac"]) <= 1.0
    # determinism under the same keys
    mean2, diags2 = refine_loop(warm, 0.2 * jnp.ones((H, adim)),
                                score_of, N, K, I,
                                lambda it: keys[it], STD_FLOOR)
    assert np.array_equal(np.asarray(mean), np.asarray(mean2))
    assert float(diags2["warm_score"]) == ws
    print("se_refine_probe selfcheck PASS (refine_loop: warm plan "
          "is candidate 0 exactly; best_first >= warm_score; "
          "progress toward a smooth optimum; all CEM diagnostics "
          "finite; deterministic under fixed keys)")


def run(args):
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3.main import make_agent, make_env
    from probing.collect import load_run_config, load_frozen_agent

    if args.smoke:
        args.episodes, args.ep_len = 1, 40
        args.n_cand, args.n_elite, args.iters, args.horizon = 32, 8, 2, 6
        args.chunk = 20

    f32 = jnp.float32
    out_dir = args.output or os.path.join(args.run_logdir, "se_refine")
    os.makedirs(out_dir, exist_ok=True)
    config = load_run_config(args.run_logdir, args.platform, out_dir,
                             False)
    agent = make_agent(config)
    jax.config.update("jax_transfer_guard", "allow")
    model = agent.model
    assert model.disag is not None
    ens = int(config.agent.expl.disag_ens)
    dec_symlog = bool(model.dec.symlog)
    ckpt = resolve_ckpt(args.run_logdir, args.ckpt)
    load_frozen_agent(agent, ckpt)
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)),
                          agent.params)

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

    def policy_plan_fn(deter, stoch):
        """The policy's own H-step imagined action plan (B=1)."""
        carry = {"deter": deter, "stoch": stoch}
        policyfn = lambda feat: sample(model.pol(
            model.feat2tensor(feat), 1))
        _, _, imgact = model.dyn.imagine(carry, policyfn,
                                         args.horizon, training=False)
        return imgact["action"][0]                    # (H, adim)

    def make_plan_fn(objective):
        N, K, H, I = (args.n_cand, args.n_elite, args.horizon,
                      args.iters)

        def score_fn(feat, actvec, norms_sq):
            if objective == "disag":
                return f32(model.disag.reward(feat, actvec)).sum(1)
            mv = member_vars(feat, actvec, N, H)
            s = (mv["distractor"]
                 / norms_sq["distractor"][None, None]).mean(-1)
            return s.sum(1)

        def plan_fn(deter, stoch, norms_sq):
            # WARM START at the policy's own plan (the Wave-A device)
            mean = policy_plan_fn(deter, stoch)
            assert mean.shape == (H, adim), mean.shape   # rev A-M4
            std = REFINE_STD * jnp.ones((H, adim))
            carry = {"deter": jnp.repeat(deter, N, 0),
                     "stoch": jnp.repeat(stoch, N, 0)}

            def score_of(cand):
                _, imgfeat, _ = model.dyn.imagine(
                    carry, {"action": cand}, H, training=False)
                states = pre_action_states(
                    carry["deter"], carry["stoch"], imgfeat)
                return score_fn(model.feat2tensor(states),
                                f32(cand), norms_sq)

            mean, diags = refine_loop(mean, std, score_of, N, K, I,
                                      lambda it: nj.seed(),
                                      STD_FLOOR)
            return dict(action=mean[0],
                        **{k: f32(v) for k, v in diags.items()})

        return plan_fn

    jit_init = jax.jit(lambda p, s: nj.pure(init_fn)(p, seed=s))
    jit_obs_step = jax.jit(
        lambda p, ec, dc, o, pa, r, s: nj.pure(obs_step_fn)(
            p, ec, dc, o, pa, r, seed=s))
    jit_plans = {
        arm: jax.jit(lambda p, d, st, ns, s, _fn=make_plan_fn(obj):
                     nj.pure(_fn)(p, d, st, ns, seed=s))
        for arm, obj in (("refine_attr", "attr"),
                         ("refine_disag", "disag"))}

    def attr_fn(deter, stoch, actions):
        feat_t = model.feat2tensor({"deter": deter, "stoch": stoch})
        out = {"intrinsic": f32(model.disag.reward(feat_t, actions))}
        mv = member_vars(feat_t, actions, deter.shape[0],
                         deter.shape[1])
        for k in obs_keys:
            out[f"var_{k}"] = mv[k]
        return out

    jit_attr = jax.jit(
        lambda p, d, st, a, s: nj.pure(attr_fn)(p, d, st, a, seed=s))

    def roll_arm(arm, norms_sq_j, seed_base):
        env = make_env(config, 0)                      # CRN per arm
        zero_act = {k: np.zeros(v.shape, v.dtype)
                    for k, v in env.act_space.items() if k != "reset"}
        E, L = args.episodes, args.ep_len
        obs_rec = {k: [] for k in obs_keys}
        act_rec, rew_rec = [], []
        det_rec, sto_rec = [], []
        reset_hits = 0
        cem = {k: [] for k in CEM_KEYS}
        pol_carry = agent.init_policy(batch_size=1)
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
                if arm == "random":
                    rng = np.random.default_rng(step_seed)
                    act = rng.uniform(-1.0, 1.0, adim).astype(
                        np.float32)
                elif arm == "policy":
                    agent_obs = {k: np.asarray(v)[None]
                                 for k, v in obs.items()
                                 if not k.startswith("log/")}
                    pol_carry, acts, _ = agent.policy(
                        pol_carry, agent_obs, mode="eval")
                    act = np.asarray(acts["action"][0], np.float32)
                else:
                    _, plan = jit_plans[arm](
                        params, st["deter"], st["stoch"], norms_sq_j,
                        step_seed + 3)
                    act = np.asarray(plan["action"], np.float32)
                    for k in cem:
                        cem[k].append(float(plan[k]))
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
                    elapsed=time.time() - t0,
                    cem={k: np.asarray(v, np.float64)
                         for k, v in cem.items()})

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

    t_start = time.time()
    results, npz_out, arm_data = {}, {}, {}
    norms = norms_raw = floor = None
    norms_sq_j = None
    for ai, arm in enumerate(ARMS):
        data = roll_arm(arm, norms_sq_j,
                        args.seed + 1_000_003 * (ai + 1))
        arm_data[arm] = data
        if arm == "random":
            norms, norms_raw, floor = compute_norms(data["obs"])
            norms_sq_j = {k: jnp.asarray((norms[k] ** 2)
                                         .astype(np.float32))
                          for k in obs_keys}
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
        if arm in PLANNER_ARMS:
            c = arm_data[arm]["cem"]
            rec["cem_improvement_mean"] = float(
                (c["elite_final"] - c["elite_first"]).mean())
            rec["cem_improves"] = bool(
                (c["elite_final"] - c["elite_first"]).mean() > 0)
            # in-plan control (rev A-B1): model-predicted gain of the
            # refined plan over the policy's own warm-start plan —
            # horizon-fair, the hard validity gate in the reader
            rec["inplan_gain_mean"] = float(
                (c["best_final"] - c["warm_score"]).mean())
            rec["inplan_improves"] = bool(
                rec["inplan_gain_mean"] > 0)
            rec["refine_disp_mean"] = float(c["refine_disp"].mean())
            rec["clip_frac_mean"] = float(c["clip_frac"].mean())
            for k, v in c.items():
                npz_out[f"{arm}_cem_{k}"] = v
        results[arm] = rec
        for k in obs_keys:
            npz_out[f"{arm}_attr_{k}"] = attr[k]
        npz_out[f"{arm}_intr"] = intr
        npz_out[f"{arm}_occ"] = occ

    import jax as _jax
    result = dict(
        run_logdir=os.path.abspath(args.run_logdir),
        ckpt=os.path.abspath(ckpt),
        train_seed=int(config.seed), task=str(config.task),
        probe_seed=args.seed, arms=list(ARMS),
        episodes=args.episodes, ep_len=args.ep_len,
        n_cand=args.n_cand, n_elite=args.n_elite,
        iters=args.iters, horizon=args.horizon,
        refine_std=REFINE_STD, chunk=args.chunk,
        smoke=bool(args.smoke),
        devices=str(_jax.devices()),
        backend=str(_jax.default_backend()),
        norm_floor=float(floor),
        gate=dict(key=GATE_KEY, index=GATE_INDEX,
                  threshold=GATE_THRESHOLD),
        pairing="pre-action (s_t, a_t) at ONLINE posterior states; "
                "planner arms warm-start at ONE STOCHASTIC SAMPLE "
                "of the policy's imagined plan (disclosed; "
                "refine_std pinned); candidate 0 of CEM iter 0 IS "
                "the warm plan; fresh env per arm (CRN)",
        per_arm=results,
        total_elapsed_s=time.time() - t_start)
    with open(os.path.join(out_dir, "se_refine.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "se_refine.npz"),
                        **npz_out,
                        **{f"norm_{k}": norms[k] for k in obs_keys},
                        **{f"rawnorm_{k}": norms_raw[k]
                           for k in obs_keys})
    print(json.dumps({a: dict(
        d=round(results[a]["attr_mean"]["distractor"], 5),
        intr=round(results[a]["intrinsic_mean"], 6)) for a in ARMS},
        indent=1))
    print(f"TOTAL {result['total_elapsed_s']:.0f}s")
    return result


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
