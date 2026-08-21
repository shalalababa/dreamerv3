"""Planner-in-the-loop FROZEN reader
(PREREG_nfi_planner_20260821.md §2; review #32 adjudicated 21 Aug:
2B/10M/12m ALL applied; built + selfchecked BEFORE the wave's
compute; ONE execution).

Read order:
  0. Per-run validity: Stage-1 pin block (bootstrap ON wave), seeds
     10-17, REGISTERED planner constants, task pin, smoke flag off,
     single CUDA device (#32 m3: a CPU fallback is a different
     numeric regime and is rejected), gate-constants pin, final-ckpt
     gate, json==npz provenance (exact float64 means), n_steps ==
     episodes*ep_len, reset_hits == 0, CEM sanity (elite_mean(final)
     > elite_mean(first) on all three planner arms — #32 M3).
     A DEFECTIVE run is caught, recorded in excluded_runs, and
     counts AGAINST every count rule with the denominator FIXED at
     8 (#32 M2/M6 — house convention; the read persists on every
     path).
  1. STEERABILITY GATE (#32 M1): ceil = attr_d(cem_distractor) -
     max(attr_d(policy), attr_d(random)) > 0 in >= 7 of the 8
     REGISTERED runs AND pooled mean > 0. Fails -> UNSTEERABLE
     (registered informative null). attr_d(cem_distractor) -
     attr_d(cem_real) is reported as a secondary row only.
  2. PRIMARY (#32 B1, CONJUNCTIVE): TRANSMISSION fires iff BOTH
     legs hold —
       delta_C = attr_d(cem_disag) - attr_d(cem_real)  > 0 and
       delta_P = attr_d(cem_disag) - attr_d(policy)    > 0
     each in >= 7 of the 8 REGISTERED runs AND each with exact
     sign-flip p <= .05 (2^n enumeration, one-sided). A
     comparator-suppression pattern (delta_C > 0 with delta_P <= 0)
     therefore CANNOT fire. Gate passed + no fire ->
     NO-TRANSMISSION-AT-DEPLOYMENT.
  3. Adjudicability: fewer than 7 valid runs -> NOT-ADJUDICABLE
     (the denominator never shrinks; no n-1 relaxation).
  4. DESCRIPTIVE (unconditional, computed on every path with >= 1
     valid run — #32 M5): pooled placement index, per-run
     numerator/denominator pairs (#32 m9), disag-policy,
     distractor-real, per-arm occupancy / intrinsic, BCa on both
     delta legs.

Run:  python -m uncfield.se_planner_read --runs "<glob>" --output <d>
      python -m uncfield.se_planner_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os

import numpy as np

from uncfield.se_mask import bca_interval
from uncfield.se_probe import resolve_ckpt
from uncfield.se_read import fit_counters
from uncfield.se_m3_read import (GATE_INDEX, GATE_KEY, GATE_THRESHOLD)

STAGE1_SEEDS = set(range(10, 18))
N_REGISTERED = 8
NEED = 7                     # of 8 registered, every count rule
MIN_VALID = 7                # adjudicability floor (#32 M6)
ARMS = ("random", "policy", "cem_disag", "cem_real", "cem_distractor")
PLANNER_ARMS = ("cem_disag", "cem_real", "cem_distractor")
REG_PARAMS = dict(episodes=4, ep_len=500, n_cand=256, n_elite=32,
                  iters=3, horizon=12)
TASK_PIN = "dmc_cheetah_run"
STAT_KEYS = ("attr_mean", "intrinsic_mean", "occupancy")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="")
    p.add_argument("--expect_steps", type=float, default=5e5)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def signflip_exact_p(deltas):
    """One-sided (positive) EXACT sign-flip p on the mean; n <= 12."""
    d = np.asarray(deltas, np.float64)
    n = d.size
    assert n <= 12, n
    obs = d.mean()
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    null = (signs * d[None, :]).mean(1)
    return float((null >= obs).sum() / (2 ** n))


def read_run(run_dir):
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = cfg["distractor"]
    pl = cfg["planted"]
    ex = cfg["agent"]["expl"]
    # Stage-1 pin block (this wave probes the EXISTING Stage-1 runs)
    assert str(ex["mode"]) == "p2e" and int(ex["disag_ens"]) == 8 \
        and abs(float(ex["disag_scale"]) - 1000.0) < 1e-9, (
        f"{run_dir}: expl pins differ from Stage-1")
    assert ex["disag_bootstrap"] is True, (
        f"{run_dir}: Stage-1 is the bootstrap-ON wave")
    assert int(d["dim"]) == 8 and abs(float(d["basesd"]) - 1.215) < 1e-9
    assert abs(float(d["scale"]) - 1.0) < 1e-9
    assert str(d.get("gate_key", "")) == "" and \
        str(d.get("mod_key", "")) == "", (
        f"{run_dir}: gates/mod must be off")
    assert str(pl.get("gate_key", "")) == ""
    assert str(pl["source_key"]) == "position" and \
        abs(float(pl["basesd"]) - 0.0976) < 1e-9, (
        f"{run_dir}: planted pins differ from Stage-1")
    assert str(cfg["task"]) == TASK_PIN, (
        f"{run_dir}: task {cfg['task']!r} != pinned")
    train_seed = int(cfg["seed"])
    assert train_seed in STAGE1_SEEDS, (
        f"{run_dir}: seed {train_seed} outside Stage-1's 10-17")

    pdir = os.path.join(run_dir, "se_planner")
    with open(os.path.join(pdir, "se_planner.json")) as f:
        pj = json.load(f)
    npz = np.load(os.path.join(pdir, "se_planner.npz"))
    assert not pj["smoke"], f"{run_dir}: smoke output, not the wave"
    assert list(pj["arms"]) == list(ARMS), pj["arms"]
    assert int(pj["train_seed"]) == train_seed, run_dir
    assert str(pj["task"]) == TASK_PIN, run_dir
    assert int(pj["probe_seed"]) == 0, (
        f"{run_dir}: probe_seed {pj['probe_seed']} != registered 0")
    for k, v in REG_PARAMS.items():
        assert int(pj[k]) == v, (
            f"{run_dir}: {k}={pj[k]} != registered {v}")
    g = pj["gate"]
    assert str(g["key"]) == GATE_KEY and int(g["index"]) == GATE_INDEX \
        and abs(float(g["threshold"]) - GATE_THRESHOLD) < 1e-12, (
        f"{run_dir}: occupancy gate constants drifted")
    # device pins (16-Aug rule + #32 m3): one device, CUDA backend
    n_dev = str(pj["devices"]).count("id=")
    assert n_dev == 1, f"{run_dir}: {n_dev} devices — panel must be 1"
    assert str(pj.get("backend", "")).lower() in ("gpu", "cuda"), (
        f"{run_dir}: backend {pj.get('backend')!r} — a CPU fallback "
        f"is a different numeric regime (#32 m3)")

    n_expect = REG_PARAMS["episodes"] * REG_PARAMS["ep_len"]
    per_arm = {}
    for arm in ARMS:
        rec = pj["per_arm"][arm]
        assert int(rec["n_steps"]) == n_expect, (
            f"{run_dir}/{arm}: n_steps {rec['n_steps']} != "
            f"{n_expect} (#32 m8)")
        assert int(rec["reset_hits"]) == 0, (
            f"{run_dir}/{arm}: mid-episode env resets (#32 M8)")
        # provenance: every stored mean == exact float64 npz mean
        for key, val in rec["attr_mean"].items():
            rc = float(np.asarray(npz[f"{arm}_attr_{key}"],
                                  np.float64).mean())
            assert rc == float(val), (
                f"{run_dir}/{arm}/{key}: npz mean {rc!r} != stored "
                f"{val!r} — provenance mismatch")
        for jkey, nkey in (("intrinsic_mean", "intr"),
                           ("occupancy", "occ")):
            rc = float(np.asarray(npz[f"{arm}_{nkey}"],
                                  np.float64).mean())
            assert rc == float(rec[jkey]), (
                f"{run_dir}/{arm}/{jkey}: provenance mismatch")
        per_arm[arm] = rec
    cem_ok = all(pj["per_arm"][a].get("cem_improves") is True
                 for a in PLANNER_ARMS)

    # final-ckpt gate (house rule; unreachable -> UNVERIFIED)
    try:
        final = resolve_ckpt(run_dir, "")
        ckpt_final_ok = (os.path.basename(os.path.normpath(final))
                         == os.path.basename(os.path.normpath(
                             pj["ckpt"])))
    except (AssertionError, FileNotFoundError, OSError):
        final, ckpt_final_ok = None, None
    assert ckpt_final_ok is not False, (
        f"{run_dir}: planner probe ran on a non-final checkpoint")

    return dict(run_dir=os.path.abspath(run_dir),
                train_seed=train_seed, ckpt_final_ok=ckpt_final_ok,
                devices=str(pj["devices"]),
                backend=str(pj.get("backend", "")), cem_ok=cem_ok,
                per_arm=per_arm)


def _ad(rec, arm):
    return float(rec["per_arm"][arm]["attr_mean"]["distractor"])


def aggregate(recs, excluded):
    seeds = sorted(r["train_seed"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert len(recs) + len(excluded) <= N_REGISTERED, (
        f"{len(recs)}+{len(excluded)} runs > registered "
        f"{N_REGISTERED} (#32 M4)")
    invalid_cem = [r["run_dir"] for r in recs if not r["cem_ok"]]
    valid = [r for r in recs if r["cem_ok"]]
    n = len(valid)
    out = dict(n_registered=N_REGISTERED, n_loaded=len(recs),
               n_valid=n, excluded_runs=excluded,
               invalid_cem_runs=invalid_cem,
               partial_flag=("COMPLETE" if n == N_REGISTERED else
                             f"PARTIAL ({n}/{N_REGISTERED} valid — "
                             f"missing/defective/CEM-invalid runs "
                             f"count AGAINST every rule; the "
                             f"denominator never shrinks)"),
               probe_ckpt_flag=("OK" if all(r["ckpt_final_ok"]
                                            for r in recs)
                                else "UNVERIFIED on some runs"),
               devices=[r["devices"] for r in recs],
               backends=[r["backend"] for r in recs])

    # unconditional reporting before any branch
    out["per_run"] = [
        {"run_dir": r["run_dir"], "train_seed": r["train_seed"],
         "cem_ok": r["cem_ok"],
         "arms": {a: {k: r["per_arm"][a][k] for k in STAT_KEYS}
                  for a in ARMS}} for r in recs]

    # DESCRIPTIVE — computed whenever >= 1 valid run (#32 M5),
    # BEFORE any early return
    if valid:
        ceil2 = [_ad(r, "cem_distractor") - _ad(r, "cem_real")
                 for r in valid]
        num = [_ad(r, "cem_disag") - _ad(r, "cem_real")
               for r in valid]
        den = [_ad(r, "cem_distractor")
               - max(_ad(r, "policy"), _ad(r, "random"))
               for r in valid]
        pooled_den = float(np.mean(den))
        out["descriptive"] = dict(
            placement_pooled=((float(np.mean(num)) / pooled_den)
                              if pooled_den > 0 else None),
            placement_pairs=[dict(num=a, den=b)
                             for a, b in zip(num, den)],
            distractor_minus_real_ceiling=ceil2,
            disag_minus_policy=[_ad(r, "cem_disag")
                                - _ad(r, "policy") for r in valid],
            occupancy={a: [r["per_arm"][a]["occupancy"]
                           for r in valid] for a in ARMS},
            intrinsic={a: [r["per_arm"][a]["intrinsic_mean"]
                           for r in valid] for a in ARMS})

    if n < MIN_VALID:
        out["primary"] = (f"NOT-ADJUDICABLE (only {n} valid runs; "
                          f"registered minimum {MIN_VALID} of "
                          f"{N_REGISTERED})")
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        return out

    # 1) steerability gate (#32 M1: passive-anchor ceiling)
    ceil_d = [_ad(r, "cem_distractor")
              - max(_ad(r, "policy"), _ad(r, "random"))
              for r in valid]
    n_ceil_pos = sum(1 for x in ceil_d if x > 0)
    steerable = bool(n_ceil_pos >= NEED
                     and float(np.mean(ceil_d)) > 0)
    out["steerability"] = dict(
        per_run=ceil_d, n_pos=n_ceil_pos, need=NEED,
        of_registered=N_REGISTERED, passes=steerable,
        form="attr_d(cem_distractor) - max(attr_d(policy), "
             "attr_d(random)) (#32 M1; distractor-real is "
             "descriptive only)")

    # 2) CONJUNCTIVE primary (#32 B1)
    d_c = [_ad(r, "cem_disag") - _ad(r, "cem_real") for r in valid]
    d_p = [_ad(r, "cem_disag") - _ad(r, "policy") for r in valid]
    n_pos_c = sum(1 for x in d_c if x > 0)
    n_pos_p = sum(1 for x in d_p if x > 0)
    p_c = signflip_exact_p(d_c)
    p_p = signflip_exact_p(d_p)
    mc, loc, hic = bca_interval(np.asarray(d_c),
                                np.random.default_rng(821), 2000)
    mp, lop, hip = bca_interval(np.asarray(d_p),
                                np.random.default_rng(823), 2000)
    fires = bool(steerable and n_pos_c >= NEED and n_pos_p >= NEED
                 and p_c <= 0.05 and p_p <= 0.05)
    out["primary"] = dict(
        statement="deployment-time transmission, CONJUNCTIVE "
                  "(#32 B1): the deployed-objective planner must "
                  "exceed BOTH the matched real-objective planner "
                  "AND the trained policy on distractor attribution "
                  "(>= 7/8 each, exact sign-flip p <= .05 each, "
                  "gated on steerability) — comparator suppression "
                  "cannot fire it",
        delta_vs_real=dict(per_run=d_c, n_pos=n_pos_c,
                           p_signflip=p_c, mean=mc,
                           bca=[loc, hic]),
        delta_vs_policy=dict(per_run=d_p, n_pos=n_pos_p,
                             p_signflip=p_p, mean=mp,
                             bca=[lop, hip]),
        need=NEED, of_registered=N_REGISTERED, fires=fires)
    if not steerable:
        out["outcome_cell"] = "UNSTEERABLE"
        out["primary"]["note"] = (
            "steerability gate FAILED — no planner moves distractor "
            "attribution above the passive anchors; transmission "
            "vacuously impossible at deployment (registered "
            "informative null); the primary numbers are reported, "
            "not adjudicated")
        return out
    out["outcome_cell"] = ("TRANSMISSION" if fires
                           else "NO-TRANSMISSION-AT-DEPLOYMENT")
    return out


def run(args):
    dirs = (sorted(globmod.glob(args.runs)) if "," not in args.runs
            else [x.strip() for x in args.runs.split(",")
                  if x.strip()])
    assert dirs, f"no runs matched {args.runs!r}"
    assert args.output and args.output != "TBD", (
        "registered read must persist: pass an explicit --output")
    recs, excluded = [], []
    for d in dirs:
        try:
            recs.append(read_run(d))
        except Exception as e:              # noqa: BLE001 (#32 M2)
            excluded.append(dict(run_dir=os.path.abspath(d),
                                 error=f"{type(e).__name__}: {e}"))
    out = aggregate(recs, excluded)
    out["fit_counters"] = {r["run_dir"]: fit_counters(
        r["run_dir"], args.expect_steps) for r in recs}
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "se_planner_read.json"),
              "w") as f:
        json.dump(out, f, indent=1)
    if isinstance(out["primary"], dict):
        pr = out["primary"]
        print(f"PLANNER PRIMARY: vs-real "
              f"{pr['delta_vs_real']['n_pos']}/{N_REGISTERED} "
              f"p={pr['delta_vs_real']['p_signflip']:.4f}; "
              f"vs-policy {pr['delta_vs_policy']['n_pos']}/"
              f"{N_REGISTERED} "
              f"p={pr['delta_vs_policy']['p_signflip']:.4f} -> "
              f"{out['outcome_cell']}")
    else:
        print(f"PLANNER PRIMARY: {out['outcome_cell']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, ad=None, cem_ok=True, smoke=False,
             n_dev=1, backend="gpu", boot=True, params_ok=True,
             doctor=None, ckpt_name="checkpoint_final",
             json_ckpt=None, n_steps=2000, reset_hits=0,
             drop_npz=False):
    """Fake run dir: config.yaml + ckpt + se_planner json/npz pair
    whose stored means ARE the exact npz means. `ad` maps arm ->
    requested distractor attr mean."""
    rng = np.random.default_rng(seed * 31 + 5)
    rd = os.path.join(tmp, name)
    pdir = os.path.join(rd, "se_planner")
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(os.path.join(rd, "ckpt", ckpt_name), exist_ok=True)
    ad = dict(ad or {})
    keys = ("position", "velocity", "distractor", "planted_dup0")
    per_arm, npz_out = {}, {}
    for arm in ARMS:
        rec = dict(n_steps=n_steps, attr_mean={}, elapsed_s=1.0,
                   return_mean=100.0, reset_hits=reset_hits)
        for k in keys:
            want = ad.get(arm, 0.02) if k == "distractor" else 0.01
            v = rng.normal(want, abs(want) * 0.1 + 1e-4, n_steps)
            v = (v - v.mean() + want).astype(np.float64)
            npz_out[f"{arm}_attr_{k}"] = v
            rec["attr_mean"][k] = float(v.mean())
        for jkey, nkey, base in (("intrinsic_mean", "intr", 3e-4),
                                 ("occupancy", "occ", 0.5)):
            v = np.full(n_steps, base, np.float64)
            npz_out[f"{arm}_{nkey}"] = v
            rec[jkey] = float(v.mean())
        if arm in PLANNER_ARMS:
            rec["cem_improvement_mean"] = 0.5 if cem_ok else -0.1
            rec["cem_improves"] = bool(cem_ok)
            rec["cem_best_gain_mean"] = 0.2
        per_arm[arm] = rec
    pj = dict(run_logdir=rd,
              ckpt=os.path.join(rd, "ckpt", json_ckpt or ckpt_name),
              train_seed=seed, task=TASK_PIN, probe_seed=0,
              arms=list(ARMS), smoke=smoke,
              devices=("[CudaDevice(id=0)]" if n_dev == 1
                       else "[CudaDevice(id=0), CudaDevice(id=1)]"),
              backend=backend,
              gate=dict(key=GATE_KEY, index=GATE_INDEX,
                        threshold=GATE_THRESHOLD),
              per_arm=per_arm,
              **(REG_PARAMS if params_ok
                 else dict(REG_PARAMS, n_cand=64)))
    if doctor:
        doctor(pj)
    with open(os.path.join(pdir, "se_planner.json"), "w") as f:
        json.dump(pj, f)
    if not drop_npz:
        np.savez(os.path.join(pdir, "se_planner.npz"), **npz_out)
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(f"seed: {seed}\ntask: {TASK_PIN}\n"
                f"planted:\n  gate_key: ''\n  source_key: 'position'\n"
                f"  basesd: 0.0976\n"
                f"distractor:\n  gate_key: ''\n  mod_key: ''\n"
                f"  dim: 8\n  scale: 1.0\n  basesd: 1.215\n"
                f"agent:\n  expl:\n    mode: p2e\n    disag_ens: 8\n"
                f"    disag_scale: 1000.0\n    disag_bootstrap: "
                f"{'true' if boot else 'false'}\n"
                f"run:\n  steps: 500000.0\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 499968}) + "\n")
    return rd


def _expect_fail(fn, needle):
    try:
        fn()
    except AssertionError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise SystemExit(f"gate did not fire: {needle}")


def selfcheck():
    import tempfile
    assert signflip_exact_p([1.0] * 8) == 1 / 256
    assert signflip_exact_p([-1.0] * 8) == 1.0
    # arm layouts (distractor attr means)
    TRANS = dict(random=0.010, policy=0.012, cem_disag=0.035,
                 cem_real=0.015, cem_distractor=0.040)
    ANTI = dict(TRANS, cem_disag=0.013)
    # comparator-suppression (#32 B1): disag > real but < policy
    SUPPR = dict(random=0.010, policy=0.050, cem_disag=0.035,
                 cem_real=0.015, cem_distractor=0.055)
    # unsteerable vs passive anchor (#32 M1): ceiling below policy
    FLAT = dict(random=0.010, policy=0.050, cem_disag=0.048,
                cem_real=0.015, cem_distractor=0.045)
    with tempfile.TemporaryDirectory() as tmp:
        # TRANSMISSION fire (both legs positive)
        runs = [_fixture(tmp, f"a{i}", 10 + i, ad=TRANS)
                for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "TRANSMISSION", agg["primary"]
        assert agg["steerability"]["passes"]
        assert agg["descriptive"]["placement_pooled"] is not None
        # NO-TRANSMISSION (steerable, disag leg negative)
        runs = [_fixture(tmp, f"b{i}", 10 + i, ad=ANTI)
                for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "NO-TRANSMISSION-AT-DEPLOYMENT"
        # SUPPRESSION does NOT fire (#32 B1): delta_vs_real 8/8 pos
        # but delta_vs_policy negative
        runs = [_fixture(tmp, f"s{i}", 10 + i, ad=SUPPR)
                for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["primary"]["delta_vs_real"]["n_pos"] == 8
        assert agg["primary"]["delta_vs_policy"]["n_pos"] == 0
        assert agg["outcome_cell"] == "NO-TRANSMISSION-AT-DEPLOYMENT"
        # UNSTEERABLE vs the passive anchor (#32 M1), descriptive
        # still present (#32 M5)
        runs = [_fixture(tmp, f"c{i}", 10 + i, ad=FLAT)
                for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "UNSTEERABLE"
        assert "note" in agg["primary"]
        assert "descriptive" in agg and \
            agg["descriptive"]["disag_minus_policy"]
        # CEM-invalid runs count against: 2 invalid -> 6 valid ->
        # NOT-ADJUDICABLE (descriptive still computed, #32 M5)
        runs = [_fixture(tmp, f"d{i}", 10 + i, ad=TRANS,
                         cem_ok=(i >= 2)) for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE"
        assert len(agg["invalid_cem_runs"]) == 2
        assert "descriptive" in agg
        # 1 invalid + 7 valid all-positive -> fires at 7/8
        runs = [_fixture(tmp, f"e{i}", 10 + i, ad=TRANS,
                         cem_ok=(i >= 1)) for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "TRANSMISSION"
        # 6 loaded -> NOT-ADJUDICABLE (#32 m12)
        runs = [_fixture(tmp, f"f{i}", 10 + i, ad=TRANS)
                for i in range(6)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE"
        # 9 loaded -> hard abort (#32 M4)
        runs = [_fixture(tmp, f"g{i}", 10 + (i % 8), ad=TRANS)
                for i in range(8)]
        extra = _fixture(tmp, "g8", 17, ad=TRANS)
        _expect_fail(
            lambda: aggregate([read_run(d) for d in runs]
                              + [read_run(extra)], []),
            "duplicate seeds")
        _expect_fail(
            lambda: aggregate([read_run(runs[0])],
                              [dict(run_dir=f"x{i}", error="e")
                               for i in range(8)]),
            "> registered")

        # identity/provenance gates
        rd = _fixture(tmp, "h0", 10, ad=TRANS, boot=False)
        _expect_fail(lambda: read_run(rd), "bootstrap-ON")
        rd = _fixture(tmp, "h1", 10, ad=TRANS, smoke=True)
        _expect_fail(lambda: read_run(rd), "smoke")
        rd = _fixture(tmp, "h2", 10, ad=TRANS, params_ok=False)
        _expect_fail(lambda: read_run(rd), "registered")
        rd = _fixture(tmp, "h3", 10, ad=TRANS, n_dev=2)
        _expect_fail(lambda: read_run(rd), "panel must be 1")
        rd = _fixture(tmp, "h4", 10, ad=TRANS, backend="cpu")
        _expect_fail(lambda: read_run(rd), "numeric regime")
        rd = _fixture(tmp, "h5", 10, ad=TRANS, reset_hits=1)
        _expect_fail(lambda: read_run(rd), "mid-episode")
        rd = _fixture(tmp, "h6", 10, ad=TRANS, n_steps=1999)
        _expect_fail(lambda: read_run(rd), "n_steps")

        def doctor(pj):
            pj["per_arm"]["cem_disag"]["attr_mean"]["distractor"] = 9.9
        rd = _fixture(tmp, "h7", 10, ad=TRANS, doctor=doctor)
        _expect_fail(lambda: read_run(rd), "provenance mismatch")

        def gdoc(pj):
            pj["gate"]["threshold"] = 0.0
        rd = _fixture(tmp, "h8", 10, ad=TRANS, doctor=gdoc)
        _expect_fail(lambda: read_run(rd), "gate constants")
        rd = _fixture(tmp, "h9", 10, ad=TRANS,
                      json_ckpt="checkpoint_000200")
        _expect_fail(lambda: read_run(rd), "non-final checkpoint")
        rd = _fixture(tmp, "h10", 55, ad=TRANS)
        _expect_fail(lambda: read_run(rd), "outside Stage-1")

        # defective run -> excluded, read persists (#32 M2), counts
        # against: 8 dirs, one with no npz -> 7 valid, still fires
        for i in range(8):
            _fixture(tmp, f"k{i}", 10 + i, ad=TRANS,
                     drop_npz=(i == 0))
        outd = os.path.join(tmp, "outK")
        out = run(parse_args(["--runs", os.path.join(tmp, "k*"),
                              "--output", outd]))
        assert len(out["excluded_runs"]) == 1 and out["n_valid"] == 7
        assert out["outcome_cell"] == "TRANSMISSION"
        assert os.path.exists(os.path.join(outd,
                                           "se_planner_read.json"))
        # determinism + end-to-end + output guard
        r1, r2 = read_run(runs[0]), read_run(runs[0])
        assert r1["per_arm"] == r2["per_arm"]
        _expect_fail(lambda: run(parse_args(
            ["--runs", os.path.join(tmp, "a*")])),
            "explicit --output")
    print("se_planner_read selfcheck PASS (conjunctive primary incl. "
          "the comparator-suppression no-fire fixture; passive-"
          "anchor steerability; cells TRANSMISSION / NO-TRANSMISSION "
          "/ UNSTEERABLE / NOT-ADJUDICABLE; denominator-fixed "
          "invalid handling at 1-vs-2 invalid; 6-loaded and 9-run "
          "bounds; descriptive on every path; excluded-run persist "
          "path; bootstrap/smoke/params/device/backend/reset/"
          "n_steps/gate-constants/provenance/final-ckpt/seed gates; "
          "determinism; end-to-end)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
