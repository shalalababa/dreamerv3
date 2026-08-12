"""Held-out predictive-quality vs coherence-audit table (12 Aug 2026;
assessment-response plan §C.2, Plan_AssessmentResponse_20260812.md).

Answers the reviewer question the untrained-filter control raises
("aren't arbitrary neural functions incoherent too?"): the trained
members have NORMAL held-out predictive quality — far from the untrained
ceiling, approaching the exact-filter floor — while still failing the
coherence audit. Ordinary training/evaluation does not surface the
defect.

Metrics per model, on FRESH episodes (env/policy seeds disjoint from
every training, warmup, and imagination seed in the study):
  obs-NLL  mean per SENSED step of the incoming-observation NLL under
           the prior belief (learned: obs head; exact: c'Pc + r).
  z-NLL    mean per step of the post-step latent NLL (learned: z head
           on the updated belief, diagonal by parameterization; exact:
           full-covariance N(mu, P) after update+predict — both target
           z_after under the same step convention as training).
Rows: exact filter (floor), pilot2 GRU m0-3, family-2 LG-LSTM m0-3
(trained; coherence verdicts from the stored summaries), untrained GRU
and LSTM at the same seeds (ceiling; §3b of the response record shows
an untrained exemplar still reaches CIG-ONLY).

Descriptive — no registered decision rules. Output:
artifacts/nfi_assessment_response_20260812/heldout.json.

Run:  python -m uncfield.heldout_eval            (or --selfcheck)
"""

from __future__ import annotations

import argparse
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from . import learnedwm as lw
from . import lgfield as lg

N_EP, T_STEPS = 50, 600
ENV_SEED, POL_SEED = 777123, 888777      # disjoint from all study seeds
ROOT = pathlib.Path(__file__).resolve().parent.parent
PILOT2 = ROOT / "local_results" / "uncfield" / "pilot2"
FAMILY2 = ROOT / "local_results" / "uncfield" / "family2"
OUTDIR = ROOT / "artifacts" / "nfi_assessment_response_20260812"


def collect(n_ep=N_EP, t_steps=T_STEPS):
    episodes = []
    for e in range(n_ep):
        env = lg.Env(seed=ENV_SEED + e)
        ref = lg.Referee()
        rng = np.random.default_rng(POL_SEED + e)
        episodes.append(lg.rollout_random(env, ref, t_steps, rng))
    return episodes


def eval_learned(params, data):
    """(mean obs-NLL per sensed step, mean z-NLL per step)."""
    def seq(xs, a1h, zt, ymask, ytrue):
        b0 = jnp.zeros(lw.belief_width(params))

        def step(b, inp):
            x, a, z, m, y = inp
            ymu, ylv = lw.obs_head(params, b, a)
            onll = m * lw._gauss_nll(y, ymu, ylv)
            b_next = lw.cell_step(params, b, x)
            mu, lv = lw.z_head(params, b_next)
            return b_next, (onll, jnp.sum(lw._gauss_nll(z, mu, lv)), m)

        _, (o, z_, m) = jax.lax.scan(step, b0, (xs, a1h, zt, ymask, ytrue))
        return o.sum(), z_.sum(), m.sum(), jnp.float32(len(z_))

    o, z_, m, n = jax.vmap(seq)(*data)
    return float(o.sum() / m.sum()), float(z_.sum() / n.sum())


def eval_exact(episodes):
    """Exact-filter floor under the identical step convention."""
    onll_sum = znll_sum = 0.0
    n_sense = n_step = 0
    for recs in episodes:
        ref = lg.Referee()
        for r in recs:
            if r["sensed"] is not None:
                s = lg.SENSORS[r["sensed"]]
                var = float(s.c @ ref.p @ s.c + s.r)
                mean = float(s.c @ ref.mu)
                onll_sum += 0.5 * (np.log(2 * np.pi * var)
                                   + (r["y"] - mean) ** 2 / var)
                n_sense += 1
                ref.update(s, r["y"])
            ref.predict()
            d = r["z_after"] - ref.mu
            p = ref.p + lg.JITTER * np.eye(lg.DZ)
            sign, logdet = np.linalg.slogdet(p)
            znll_sum += 0.5 * (lg.DZ * np.log(2 * np.pi) + logdet
                               + d @ np.linalg.solve(p, d))
            n_step += 1
    return onll_sum / n_sense, znll_sum / n_step


def _verdicts():
    v = {}
    s = json.load(open(PILOT2 / "summary.json"))
    for m in s["members"]:
        v[("pilot2_gru", m["member"])] = m["verdict"]
    s = json.load(open(FAMILY2 / "lg_lstm" / "summary.json"))
    for m in s["members"]:
        v[("lg_lstm", m["member"])] = m["verdict"]
    return v


def run():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    episodes = collect()
    data = lw.episodes_to_arrays(episodes)
    verdicts = _verdicts()
    rows = []
    o, z = eval_exact(episodes)
    rows.append(dict(model="exact_filter", member=-1, obs_nll=o, z_nll=z,
                     verdict="(Bayes floor)"))
    for label, path in (("pilot2_gru", PILOT2 / "ensemble.pkl"),
                        ("lg_lstm", FAMILY2 / "lg_lstm" / "ensemble.pkl")):
        for m, params in enumerate(lw.load_ensemble(path)):
            o, z = eval_learned(params, data)
            rows.append(dict(model=label, member=m, obs_nll=o, z_nll=z,
                             verdict=verdicts[(label, m)]))
    for m in range(4):
        o, z = eval_learned(jax.tree_util.tree_map(
            jnp.asarray, lw.init_params(m)), data)
        rows.append(dict(model="untrained_gru", member=m, obs_nll=o,
                         z_nll=z, verdict="(ceiling)"))
        o, z = eval_learned(jax.tree_util.tree_map(
            jnp.asarray, lw.init_params_lstm(m)), data)
        rows.append(dict(model="untrained_lstm", member=m, obs_nll=o,
                         z_nll=z, verdict="(ceiling)"))
    out = dict(n_ep=N_EP, t_steps=T_STEPS, env_seed=ENV_SEED,
               pol_seed=POL_SEED, rows=rows)
    with open(OUTDIR / "heldout.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"{'model':>14} m {'obs-NLL':>9} {'z-NLL':>9}  verdict")
    for r in rows:
        print(f"{r['model']:>14} {r['member']:>1} {r['obs_nll']:>9.3f} "
              f"{r['z_nll']:>9.3f}  {r['verdict']}")
    return out


def selfcheck():
    """Tiny-scale ordering + determinism gate."""
    global N_EP, T_STEPS
    n_ep, t = 4, 120
    episodes = []
    for e in range(n_ep):
        env = lg.Env(seed=ENV_SEED + e)
        ref = lg.Referee()
        rng = np.random.default_rng(POL_SEED + e)
        episodes.append(lg.rollout_random(env, ref, t, rng))
    data = lw.episodes_to_arrays(episodes)
    o_ex, z_ex = eval_exact(episodes)
    p_tr = lw.load_ensemble(PILOT2 / "ensemble.pkl")[0]
    o_tr, z_tr = eval_learned(p_tr, data)
    p_un = jax.tree_util.tree_map(jnp.asarray, lw.init_params(0))
    o_un, z_un = eval_learned(p_un, data)
    assert np.isfinite([o_ex, z_ex, o_tr, z_tr, o_un, z_un]).all()
    assert o_ex < o_tr < o_un, (o_ex, o_tr, o_un)   # floor < trained < ceiling
    assert z_ex < z_tr < z_un, (z_ex, z_tr, z_un)
    o2, z2 = eval_learned(p_tr, data)
    assert (o2, z2) == (o_tr, z_tr)                  # deterministic
    print(f"heldout_eval selfcheck PASS (obs {o_ex:.2f}<{o_tr:.2f}<{o_un:.2f}; "
          f"z {z_ex:.2f}<{z_tr:.2f}<{z_un:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    selfcheck() if args.selfcheck else run()


if __name__ == "__main__":
    main()
