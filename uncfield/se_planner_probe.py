"""Planner-in-the-loop deployment probe (transmission axis;
PREREG_nfi_planner_20260821.md; review #32 adjudicated 21 Aug:
2B/10M/12m ALL applied). Per-run instrument, built + smoke-verified
BEFORE the wave's compute.

Question: the SE program shows the misprice exists in the field
(Stage-1), survives ranking only anti-transmitted (rankx), and does
not move the TRAINED policy (M3, gradient wave). The remaining axis:
a DEPLOYMENT-TIME planner that greedily maximizes a finite-horizon
surrogate of the disagreement objective over the frozen WM (#32 m5:
undiscounted H-step sum, no terminal value/continue weighting — a
greedy surrogate of, not identical to, the deployed p2e return).

Arms per checkpoint (FIXED order, all on ONE device; #32 M7: each
arm gets a FRESH env via make_env(config, 0) — common random
numbers: identical wrapper seed streams and initial states across
arms; trajectories diverge only through the actions):
  random          — uniform actions (normalizer + floor anchor)
  policy          — the trained SAMPLING policy (#32 m4: Agent.policy
                    ignores mode; this is the stochastic actor)
  cem_disag       — CEM/MPC maximizing raw disag.reward over H
  cem_real        — CEM maximizing decoded-variance attribution on
                    REAL keys (position+velocity)
  cem_distractor  — CEM maximizing distractor attribution (the
                    steerability ceiling arm)

Measurement (#32 M9: attribution is computed at the ONLINE posterior
states — every arm, including random/policy, carries the same
jitted single-step observe and stores (deter, stoch) per realized
step; the post-pass runs member_vars directly on those stored states
with the review-#21 pre-action (s_t, a_t) pairing — no offline
re-encode, no posterior resampling). Normalizers from the RANDOM arm
only (se_probe symlog convention, 0.05x floor). Occupancy vs the
pinned se_m3_read gate constants. Mid-episode env resets are
FORBIDDEN (#32 M8): reset_hits must be 0 (asserted per arm) —
cheetah has no early termination at the registered ep_len.

CEM diagnostics (#32 M3): per planning step, elite-mean of the FIRST
and FINAL iterations and best-of-first/final; the run-validity gate
quantity is elite_mean(final) > elite_mean(first) (a like-for-like
optimization check, not max-vs-mean).

Run:  python -m uncfield.se_planner_probe --run_logdir <dir>
      python -m uncfield.se_planner_probe --run_logdir <dir> --smoke
      python -m uncfield.se_planner_probe --selfcheck
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from uncfield.se_probe import (EXCLUDE, PLANTED_KEYS, SMOKE, np_symlog,
                               resolve_ckpt)
from uncfield.se_m3_read import (GATE_INDEX, GATE_KEY, GATE_THRESHOLD)

ARMS = ("random", "policy", "cem_disag", "cem_real", "cem_distractor")
PLANNER_ARMS = ("cem_disag", "cem_real", "cem_distractor")
INIT_STD = 0.5
STD_FLOOR = 0.02


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run_logdir", default=str(SMOKE))
    p.add_argument("--output", default="",
                   help="default: <run>/se_planner")
    p.add_argument("--ckpt", default="")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--ep_len", type=int, default=500)
    p.add_argument("--n_cand", type=int, default=256)
    p.add_argument("--n_elite", type=int, default=32)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--chunk", type=int, default=100,
                   help="T-chunk for the attribution post-pass")
    p.add_argument("--platform", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end mechanics run (1 ep x 40 "
                        "steps, N=32, H=6, I=2) + runtime report")
    p.add_argument("--selfcheck", action="store_true",
                   help="CPU unit tests of the CEM update math and "
                        "the pre-action pairing (#32 M10)")
    return p.parse_args(argv)


# ------------------------------------------------ pure, unit-testable pieces

def cem_elite(cand, score, n_elite, std_floor):
    """One CEM elite update on numpy/jnp-agnostic arrays. Returns
    (mean, std, elite_mean_score, best_score). Pure; exercised by
    --selfcheck against a closed-form quadratic (#32 M10)."""
    import jax.numpy as jnp
    order = jnp.argsort(-score)
    elite = cand[order[:n_elite]]
    e_scores = score[order[:n_elite]]
    return (elite.mean(0), elite.std(0) + std_floor,
            e_scores.mean(), score[order[0]])


def pre_action_states(deter0, stoch0, imgfeat):
    """Rebuild the PRE-action state sequence (s_i paired with a_i)
    from the imagination output: [anchor] ++ imgfeat[:-1]
    (review-#21 convention; rssm.imagine returns post-action states).
    Pure; exercised by --selfcheck on a hand-built sequence."""
    import jax.numpy as jnp
    return {
        "deter": jnp.concatenate(
            [deter0[:, None], imgfeat["deter"][:, :-1]], 1),
        "stoch": jnp.concatenate(
            [stoch0[:, None], imgfeat["stoch"][:, :-1]], 1)}


# ------------------------------------------------------------------- runner

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
    out_dir = args.output or os.path.join(args.run_logdir, "se_planner")
    os.makedirs(out_dir, exist_ok=True)
    config = load_run_config(args.run_logdir, args.platform, out_dir,
                             False)
    agent = make_agent(config)
    jax.config.update("jax_transfer_guard", "allow")
    model = agent.model
    assert model.disag is not None, "planner probe requires disag"
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

    # ---------------------------------------------------- jitted pieces
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

    def make_plan_fn(objective):
        N, K, H, I = (args.n_cand, args.n_elite, args.horizon,
                      args.iters)

        def score_fn(feat, actvec, norms_sq):
            if objective == "disag":
                return f32(model.disag.reward(feat, actvec)).sum(1)
            target = (real_keys if objective == "real"
                      else ["distractor"])
            mv = member_vars(feat, actvec, N, H)
            s = 0.0
            for k in target:
                s = s + (mv[k] / norms_sq[k][None, None]).mean(-1)
            return s.sum(1)

        def plan_fn(deter, stoch, norms_sq):
            mean = jnp.zeros((H, adim))
            std = INIT_STD * jnp.ones((H, adim))
            carry = {"deter": jnp.repeat(deter, N, 0),
                     "stoch": jnp.repeat(stoch, N, 0)}
            diags = {}
            for it in range(I):
                eps = jax.random.normal(nj.seed(), (N, H, adim))
                cand = jnp.clip(mean[None] + std[None] * eps,
                                -1.0, 1.0)
                _, imgfeat, _ = model.dyn.imagine(
                    carry, {"action": cand}, H, training=False)
                states = pre_action_states(
                    carry["deter"], carry["stoch"], imgfeat)
                score = score_fn(model.feat2tensor(states),
                                 f32(cand), norms_sq)
                mean, std, e_mean, best = cem_elite(
                    cand, score, K, STD_FLOOR)
                if it == 0:
                    diags["rand_mean"] = f32(score.mean())
                    diags["elite_first"] = f32(e_mean)
                    diags["best_first"] = f32(best)
            diags["elite_final"] = f32(e_mean)
            diags["best_final"] = f32(best)
            return dict(action=mean[0], **diags)

        return plan_fn

    jit_init = jax.jit(lambda p, s: nj.pure(init_fn)(p, seed=s))
    jit_obs_step = jax.jit(
        lambda p, ec, dc, o, pa, r, s: nj.pure(obs_step_fn)(
            p, ec, dc, o, pa, r, seed=s))
    jit_plans = {
        arm: jax.jit(lambda p, d, st, ns, s, _fn=make_plan_fn(obj):
                     nj.pure(_fn)(p, d, st, ns, seed=s))
        for arm, obj in (("cem_disag", "disag"), ("cem_real", "real"),
                         ("cem_distractor", "distractor"))}

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

    # ------------------------------------------------------- env rollout
    def roll_arm(arm, norms_sq_j, seed_base):
        # #32 M7: FRESH env per arm with the SAME index -> identical
        # wrapper seed streams and initial states (CRN)
        env = make_env(config, 0)
        zero_act = {k: np.zeros(v.shape, v.dtype)
                    for k, v in env.act_space.items() if k != "reset"}
        E, L = args.episodes, args.ep_len
        obs_rec = {k: [] for k in obs_keys}
        act_rec, rew_rec = [], []
        det_rec, sto_rec = [], []
        reset_hits = 0
        cem = {k: [] for k in ("rand_mean", "elite_first",
                               "elite_final", "best_first",
                               "best_final")}
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
                # ONLINE posterior for every arm (#32 M9)
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
                # #32 m7: reward attributed to the action just taken
                rew_rec.append(float(np.asarray(obs["reward"])))
        env.close()
        # #32 M8: mid-episode resets forbidden at registered constants
        assert reset_hits == 0, (
            f"{arm}: {reset_hits} mid-episode env resets — the "
            f"attribution pairing would be corrupted; lower ep_len")
        elapsed = time.time() - t0
        arrs = {k: np.stack(v, 0).reshape(E, L, -1)
                for k, v in obs_rec.items()}
        return dict(obs=arrs,
                    actions=np.stack(act_rec, 0).reshape(E, L, adim),
                    deter=np.stack(det_rec, 0).reshape(E, L, -1),
                    stoch=np.stack(sto_rec, 0).reshape(
                        E, L, *sto_rec[0].shape),
                    rewards=np.asarray(rew_rec, np.float64),
                    reset_hits=reset_hits, elapsed=elapsed,
                    cem={k: np.asarray(v, np.float64)
                         for k, v in cem.items()})

    # ----------------------------------------------------- normalizers
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

    # ------------------------------- attribution at the ONLINE states
    def attribute(arm_data, norms):
        E, L = args.episodes, args.ep_len
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
            v = (v / (norms[k] ** 2)[None, None]).mean(-1)
            attr[k] = v.reshape(-1)
        pos = arm_data["obs"][GATE_KEY][..., GATE_INDEX].reshape(-1)
        occ = (pos > GATE_THRESHOLD).astype(np.float64)
        return attr, intr, occ

    # ----------------------------------------------------------- main
    t_start = time.time()
    results, npz_out = {}, {}
    arm_data = {}
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
        print(f"[{arm}] rolled {args.episodes}x{args.ep_len} in "
              f"{data['elapsed']:.1f}s", flush=True)
    for arm in ARMS:
        attr, intr, occ = attribute(arm_data[arm], norms)
        rec = dict(
            n_steps=int(intr.size),
            attr_mean={k: float(attr[k].mean()) for k in obs_keys},
            intrinsic_mean=float(intr.mean()),
            occupancy=float(occ.mean()),
            return_mean=float(arm_data[arm]["rewards"].sum()
                              / args.episodes),
            reset_hits=arm_data[arm]["reset_hits"],
            elapsed_s=arm_data[arm]["elapsed"])
        if arm in PLANNER_ARMS:
            c = arm_data[arm]["cem"]
            rec["cem_improvement_mean"] = float(
                (c["elite_final"] - c["elite_first"]).mean())
            rec["cem_improves"] = bool(
                (c["elite_final"] - c["elite_first"]).mean() > 0)
            rec["cem_best_gain_mean"] = float(
                (c["best_final"] - c["best_first"]).mean())
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
        probe_seed=args.seed,
        arms=list(ARMS),
        episodes=args.episodes, ep_len=args.ep_len,
        n_cand=args.n_cand, n_elite=args.n_elite,
        iters=args.iters, horizon=args.horizon, chunk=args.chunk,
        smoke=bool(args.smoke),
        devices=str(_jax.devices()),
        backend=str(_jax.default_backend()),
        norm_floor=float(floor),
        gate=dict(key=GATE_KEY, index=GATE_INDEX,
                  threshold=GATE_THRESHOLD),
        pairing="pre-action (s_t, a_t) at the ONLINE posterior "
                "states (#32 M9); normalizers from the RANDOM arm "
                "only (0.05x floor); fresh env per arm, same index "
                "(CRN, #32 M7)",
        per_arm=results,
        total_elapsed_s=time.time() - t_start)
    with open(os.path.join(out_dir, "se_planner.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "se_planner.npz"),
                        **npz_out,
                        **{f"norm_{k}": norms[k] for k in obs_keys},
                        **{f"rawnorm_{k}": norms_raw[k]
                           for k in obs_keys})
    print(json.dumps({a: dict(
        d=round(results[a]["attr_mean"]["distractor"], 5),
        occ=round(results[a]["occupancy"], 3)) for a in ARMS},
        indent=1))
    print(f"TOTAL {result['total_elapsed_s']:.0f}s")
    return result


# ---------------------------------------------------------------- selfcheck

def selfcheck():
    """CPU unit tests of the two novel components (#32 M10): the CEM
    elite-update math against a closed-form quadratic, and the
    pre-action pairing on a hand-built sequence."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    # (1) CEM on a quadratic: maximize -(x - x*)^2 summed; iterating
    # cem_elite must converge the mean to x* well within INIT_STD
    H, adim, N, K = 4, 3, 256, 32
    x_star = jnp.asarray(rng.uniform(-0.6, 0.6, (H, adim)))
    mean = jnp.zeros((H, adim))
    std = INIT_STD * jnp.ones((H, adim))
    first_em = None
    for it in range(6):
        eps = jnp.asarray(rng.standard_normal((N, H, adim)))
        cand = jnp.clip(mean[None] + std[None] * eps, -1.0, 1.0)
        score = -((cand - x_star[None]) ** 2).sum((1, 2))
        mean, std, e_mean, best = cem_elite(cand, score, K, STD_FLOOR)
        if it == 0:
            first_em = float(e_mean)
    assert float(jnp.abs(mean - x_star).max()) < 0.05, (
        float(jnp.abs(mean - x_star).max()))
    assert float(e_mean) > first_em          # the gate quantity moves
    # elite scores dominate the candidate mean by construction
    assert float(best) >= float(e_mean)
    # mutant check: a sign-flipped argsort would ANTI-converge
    mean2 = jnp.zeros((H, adim))
    std2 = INIT_STD * jnp.ones((H, adim))
    for it in range(6):
        eps = jnp.asarray(rng.standard_normal((N, H, adim)))
        cand = jnp.clip(mean2[None] + std2[None] * eps, -1.0, 1.0)
        score = ((cand - x_star[None]) ** 2).sum((1, 2))  # minimize!
        mean2, std2, _, _ = cem_elite(cand, score, K, STD_FLOOR)
    assert float(jnp.abs(mean2 - x_star).max()) > 0.2

    # (2) pre-action pairing: with imgfeat[t] = post-action state
    # s_{t+1}, the rebuilt sequence must be [s_0, s_1, ..., s_{H-1}]
    B, H, D = 2, 5, 3
    s = jnp.asarray(np.arange(B * (H + 1) * D, dtype=np.float32)
                    .reshape(B, H + 1, D))       # s_0..s_H
    imgfeat = {"deter": s[:, 1:], "stoch": s[:, 1:]}
    out = pre_action_states(s[:, 0], s[:, 0], imgfeat)
    assert np.array_equal(np.asarray(out["deter"]),
                          np.asarray(s[:, :H]))
    assert np.array_equal(np.asarray(out["stoch"]),
                          np.asarray(s[:, :H]))
    print("se_planner_probe selfcheck PASS (CEM quadratic "
          "convergence + gate-quantity movement + argsort-mutant "
          "anti-convergence; pre-action pairing exact on a "
          "hand-built sequence)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
