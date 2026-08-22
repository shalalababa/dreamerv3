"""Wave-A (policy-seeded refinement) FROZEN reader
(PREREG_planner_refine_20260822.md; built + selfchecked BEFORE the
wave's compute; ONE execution; revised pre-freeze per review R2).

8 Stage-1 checkpoints (seeds 10-17), 4 arms (random / policy /
refine_attr / refine_disag), the warm-started refinement instrument
se_refine_probe (registered constants + refine_std 0.2; candidate 0
of CEM iteration 0 IS the warm-start plan, so warm_score is exact).

Read order:
  0. Validity: Stage-1 pins, seeds 10-17, registered constants incl.
     refine_std, gate constants, backend/device, final-ckpt +
     newest-dir cross-check, provenance (float64-exact incl.
     occupancy), n_steps/reset_hits, CEM sanity on both refine arms,
     per-run degeneracy floor intrinsic(policy) > intrinsic(random)
     (rev A-M1). Defective runs -> excluded, read persists;
     denominator fixed at 8.
  1. HARD INSTRUMENT GATE (rev A-B1/A-M2, horizon-fair): the
     refine_disag arm's IN-PLAN control — mean(best_final -
     warm_score) > 0 (model-predicted improvement of its own
     objective over the policy's own plan) in >= 6/8 valid runs.
     Failure => NOT-ADJUDICABLE-INSTRUMENT-WEAK (the search cannot
     even improve its own objective in-model — this gate CANNOT
     fail for horizon/discount reasons).
  2. REALIZED-STRENGTH SCOPE (rev A-M2, not a gate): realized
     intrinsic(refine_disag) >= 0.9 x intrinsic(policy) in >= 6/8 —
     pass => the null carries the full stationarity wording; fail
     => the null is scoped to "greedy H-step execution" (the
     deviation legitimately compounds over 2000 steps under a
     different controller).
  3. PRIMARY: Delta = attr_d(refine_attr) - attr_d(policy); FIRES
     (REFINEMENT-TRANSMISSION) iff Delta > 0 in >= 7/8 AND exact
     sign-flip p <= .05. The NULL is the registered Tier-1 claim,
     reworded (rev A-M3): "greedy H-step replanning seeded at the
     policy's own imagined plan does not increase realized
     distractor attribution" (a controller-class statement, NOT a
     neighborhood/'local maximum' statement).
  4. Registered descriptives (always emitted, every path): SHARE
     rows, refine_disag attr row, occupancy/intrinsic/return rows,
     BCa on Delta, in-plan gain rows for BOTH refine arms (the
     predicted-vs-realized dissociation reporting, rev A-B1),
     refinement displacement + clip fraction (rev A-M8), and the
     free reproduction row: attr_d(policy)/attr_d(random) vs the
     ambient wave's published per-seed values (rev A-M6; tolerance
     descriptive — cross-GPU shifts are a measured phenomenon).

Run:  python -m uncfield.se_refine_read --runs "<glob>" --output <d>
      python -m uncfield.se_refine_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import subprocess
import time

import numpy as np

from uncfield.se_grad_read import BASESD_N, DIM
from uncfield.se_m3_read import (GATE_INDEX, GATE_KEY, GATE_THRESHOLD)
from uncfield.se_mask import bca_interval
from uncfield.se_planner_read import signflip_exact_p
from uncfield.se_probe import resolve_ckpt
from uncfield.se_read import fit_counters

STAGE1_SEEDS = set(range(10, 18))
N_REGISTERED = 8
NEED = 7
MIN_VALID = 7
ARMS = ("random", "policy", "refine_attr", "refine_disag")
PLANNER_ARMS = ("refine_attr", "refine_disag")
REG_PARAMS = dict(episodes=4, ep_len=500, n_cand=256, n_elite=32,
                  iters=3, horizon=12)
REFINE_STD = 0.2
TASK_PIN = "dmc_cheetah_run"
CONTROL_RATIO = 0.9
CONTROL_NEED = 6
INPLAN_NEED = 6
# ambient wave's published per-seed attr_d (policy, random) —
# se_planner_read_20260822; report-only reproduction anchors (A-M6)
REPRO_ANCHORS = {
    10: (0.035693, 0.004302), 11: (0.012033, 0.008154),
    12: (0.014355, 0.004468), 13: (0.012896, 0.008231),
    14: (0.014285, 0.003754), 15: (0.016300, 0.009249),
    16: (0.012102, 0.003387), 17: (0.016404, 0.004251)}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="")
    p.add_argument("--expect_steps", type=float, default=5e5)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def _stamp():
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = None
    return dict(git=git, time=time.strftime("%Y-%m-%dT%H:%M:%S"))


def read_run(run_dir):
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = cfg["distractor"]
    pl = cfg["planted"]
    ex = cfg["agent"]["expl"]
    train_seed = int(cfg["seed"])
    assert train_seed in STAGE1_SEEDS, (
        f"{run_dir}: seed {train_seed} outside Stage-1's 10-17")
    assert str(cfg["task"]) == TASK_PIN, run_dir
    assert str(ex["mode"]) == "p2e" and int(ex["disag_ens"]) == 8 \
        and abs(float(ex["disag_scale"]) - 1000.0) < 1e-9 \
        and ex["disag_bootstrap"] is True, (
        f"{run_dir}: expl pins differ from Stage-1")
    assert int(d["dim"]) == DIM and \
        abs(float(d["basesd"]) - BASESD_N) < 1e-9
    assert abs(float(d["scale"]) - 1.0) < 1e-9
    assert str(d.get("gate_key", "")) == "" and \
        str(d.get("mod_key", "")) == "", run_dir
    assert str(pl.get("gate_key", "")) == ""
    assert str(pl["source_key"]) == "position" and \
        abs(float(pl["basesd"]) - 0.0976) < 1e-9, run_dir
    assert float(cfg.get("penalty", {}).get("scale", 0.0)) == 0.0 \
        and ex.get("penalty_mix", False) is False, run_dir

    pdir = os.path.join(run_dir, "se_refine")
    with open(os.path.join(pdir, "se_refine.json")) as f:
        pj = json.load(f)
    npz = np.load(os.path.join(pdir, "se_refine.npz"))
    assert not pj["smoke"], f"{run_dir}: smoke output"
    assert list(pj["arms"]) == list(ARMS), pj["arms"]
    assert int(pj["train_seed"]) == train_seed
    assert str(pj["task"]) == TASK_PIN and int(pj["probe_seed"]) == 0
    assert os.path.basename(os.path.normpath(pj["run_logdir"])) == \
        os.path.basename(os.path.normpath(run_dir)), (
        f"{run_dir}: probe json belongs to {pj['run_logdir']}")
    for k, v in REG_PARAMS.items():
        assert int(pj[k]) == v, (
            f"{run_dir}: {k}={pj[k]} != registered {v}")
    assert abs(float(pj["refine_std"]) - REFINE_STD) < 1e-12, (
        f"{run_dir}: refine_std {pj['refine_std']}")
    g = pj["gate"]
    assert str(g["key"]) == GATE_KEY and int(g["index"]) == GATE_INDEX \
        and abs(float(g["threshold"]) - GATE_THRESHOLD) < 1e-12, (
        f"{run_dir}: occupancy gate constants drifted (rev A-M5)")
    # final-ckpt + newest-dir cross-check (rev A-M5)
    ck_probe = os.path.basename(os.path.normpath(pj["ckpt"]))
    ckroot = os.path.join(run_dir, "ckpt")
    try:
        subs = sorted(x for x in os.listdir(ckroot)
                      if os.path.isdir(os.path.join(ckroot, x)))
    except OSError:
        subs = []
    ckpt_final_ok = None
    if subs:
        assert ck_probe == subs[-1], (
            f"{run_dir}: probed ckpt {ck_probe} != newest {subs[-1]} "
            f"— stale ckpt/latest pointer")
        try:
            final = resolve_ckpt(run_dir, "")
            ckpt_final_ok = (os.path.basename(
                os.path.normpath(final)) == ck_probe)
        except (AssertionError, FileNotFoundError, OSError):
            ckpt_final_ok = None
        assert ckpt_final_ok is not False, (
            f"{run_dir}: probe ran on a non-final checkpoint")
    n_dev = str(pj["devices"]).count("id=")
    assert n_dev == 1, f"{run_dir}: {n_dev} devices"
    assert str(pj.get("backend", "")).lower() in ("gpu", "cuda"), (
        f"{run_dir}: backend {pj.get('backend')!r} — CPU fallback "
        f"is a different numeric regime")
    n_expect = REG_PARAMS["episodes"] * REG_PARAMS["ep_len"]
    per_arm = {}
    for a in ARMS:
        rec = pj["per_arm"][a]
        assert int(rec["n_steps"]) == n_expect, (run_dir, a)
        assert int(rec["reset_hits"]) == 0, (run_dir, a)
        for key, val in rec["attr_mean"].items():
            rc = float(np.asarray(npz[f"{a}_attr_{key}"],
                                  np.float64).mean())
            assert rc == float(val), (
                f"{run_dir}/{a}/{key}: provenance mismatch")
        rc = float(np.asarray(npz[f"{a}_intr"], np.float64).mean())
        assert rc == float(rec["intrinsic_mean"]), (run_dir, a)
        rc = float(np.asarray(npz[f"{a}_occ"], np.float64).mean())
        assert rc == float(rec["occupancy"]), (
            f"{run_dir}/{a}: occupancy provenance mismatch")
        per_arm[a] = rec
    for a in PLANNER_ARMS:
        for fld in ("inplan_gain_mean", "inplan_improves",
                    "refine_disp_mean", "clip_frac_mean"):
            assert fld in pj["per_arm"][a], (
                f"{run_dir}/{a}: {fld} missing — pre-revision "
                f"instrument output")
    cem_ok = all(pj["per_arm"][a].get("cem_improves") is True
                 for a in PLANNER_ARMS)
    # per-run degeneracy floor (rev A-M1): a dead WM makes the ratio
    # control vacuous (0/0 ~ 1) — require the incumbent to beat the
    # uniform-random floor on its own objective
    degenerate = not (per_arm["policy"]["intrinsic_mean"]
                      > per_arm["random"]["intrinsic_mean"])
    return dict(run_dir=os.path.abspath(run_dir),
                train_seed=train_seed, cem_ok=cem_ok,
                degenerate=degenerate, ckpt=ck_probe,
                ckpt_final_ok=ckpt_final_ok, per_arm=per_arm)


def _ad(rec, arm):
    return float(rec["per_arm"][arm]["attr_mean"]["distractor"])


def _share(rec, arm):
    am = rec["per_arm"][arm]["attr_mean"]
    return float(am["distractor"] / sum(am.values()))


def aggregate(recs, excluded):
    seeds = sorted(r["train_seed"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert len(recs) + len(excluded) <= N_REGISTERED
    valid = [r for r in recs if r["cem_ok"] and not r["degenerate"]]
    n = len(valid)
    out = dict(stamp=_stamp(), n_loaded=len(recs), n_valid=n,
               excluded_runs=excluded,
               invalid_cem=[r["run_dir"] for r in recs
                            if not r["cem_ok"]],
               degenerate_runs=[r["run_dir"] for r in recs
                                if r["degenerate"]])
    out["per_run"] = [
        {"run_dir": r["run_dir"], "train_seed": r["train_seed"],
         "cem_ok": r["cem_ok"], "degenerate": r["degenerate"],
         "ckpt": r["ckpt"], "ckpt_final_ok": r["ckpt_final_ok"],
         "attr_d": {a: _ad(r, a) for a in ARMS},
         "share_d": {a: _share(r, a) for a in ARMS},
         "intrinsic": {a: r["per_arm"][a]["intrinsic_mean"]
                       for a in ARMS},
         "occupancy": {a: r["per_arm"][a]["occupancy"]
                       for a in ARMS},
         "return_mean": {a: r["per_arm"][a].get("return_mean")
                         for a in ARMS},
         "inplan": {a: dict(
             gain=r["per_arm"][a]["inplan_gain_mean"],
             improves=r["per_arm"][a]["inplan_improves"],
             refine_disp=r["per_arm"][a]["refine_disp_mean"],
             clip_frac=r["per_arm"][a]["clip_frac_mean"])
             for a in PLANNER_ARMS},
         "repro_vs_ambient": {
             "policy": [_ad(r, "policy"),
                        REPRO_ANCHORS[r["train_seed"]][0]],
             "random": [_ad(r, "random"),
                        REPRO_ANCHORS[r["train_seed"]][1]]}}
        for r in recs]
    # registered descriptives on EVERY path (rev A-m4)
    out["s_refine_disag_attr"] = [
        _ad(r, "refine_disag") - _ad(r, "policy") for r in valid]
    out["s_share"] = {a: [_share(r, a) for r in valid] for a in ARMS}

    if n < MIN_VALID:
        out["primary"] = (f"NOT-ADJUDICABLE ({n} valid < "
                          f"{MIN_VALID} of {N_REGISTERED})")
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        return out

    # 1) HARD INSTRUMENT GATE (rev A-B1): in-plan, horizon-fair
    n_inplan = sum(1 for r in valid
                   if r["per_arm"]["refine_disag"]["inplan_improves"])
    inplan_ok = bool(n_inplan >= INPLAN_NEED)
    out["inplan_gate"] = dict(
        n_pass=n_inplan, need=INPLAN_NEED, passes=inplan_ok,
        gains=[r["per_arm"]["refine_disag"]["inplan_gain_mean"]
               for r in valid],
        statement="model-predicted improvement of the refiner's own "
                  "objective over the policy's own plan — cannot "
                  "fail for horizon/discount reasons")
    if not inplan_ok:
        out["primary"] = ("NOT-ADJUDICABLE-INSTRUMENT-WEAK: the "
                          "refiner cannot improve its own objective "
                          "IN-MODEL over the warm start — a null on "
                          "the attr arm is uninformative")
        out["outcome_cell"] = "NOT-ADJUDICABLE-INSTRUMENT-WEAK"
        return out

    # 2) REALIZED-STRENGTH SCOPE (rev A-M2; determines wording only)
    ratios = [r["per_arm"]["refine_disag"]["intrinsic_mean"]
              / max(r["per_arm"]["policy"]["intrinsic_mean"], 1e-30)
              for r in valid]
    n_ctrl = sum(1 for x in ratios if x >= CONTROL_RATIO)
    control = bool(n_ctrl >= CONTROL_NEED)
    out["realized_strength_scope"] = dict(
        ratios=ratios, n_pass=n_ctrl, need=CONTROL_NEED,
        bar=CONTROL_RATIO, passes=control,
        statement="realized-intrinsic retention; fail scopes the "
                  "null to greedy H-step execution, never voids it")

    # 3) PRIMARY
    deltas = [_ad(r, "refine_attr") - _ad(r, "policy") for r in valid]
    n_pos = sum(1 for x in deltas if x > 0)
    p = signflip_exact_p(deltas)
    m, lo, hi = bca_interval(np.asarray(deltas),
                             np.random.default_rng(824), 2000)
    fires = bool(n_pos >= NEED and p <= 0.05)
    attr_gains = [r["per_arm"]["refine_attr"]["inplan_gain_mean"]
                  for r in valid]
    out["primary"] = dict(
        statement="greedy H-step replanning seeded at the policy's "
                  "own imagined plan increases realized distractor "
                  "attribution (one-sided; the NULL is the "
                  "registered controller-class stationarity claim, "
                  "NOT a neighborhood claim — rev A-M3)",
        per_run_delta=deltas, n_pos=n_pos, need=NEED,
        p_signflip=p, mean=m, bca=[lo, hi], fires=fires,
        attr_inplan_gains=attr_gains,
        predicted_vs_realized_note=(
            "attr in-plan gains are the MODEL-PREDICTED headroom at "
            "the warm start; a null with large predicted gains is a "
            "predicted-but-unrealized dissociation (off-support "
            "imagination), a different finding from model-predicted "
            "stationarity — reported, not adjudicated"))
    if fires:
        out["outcome_cell"] = "REFINEMENT-TRANSMISSION"
    elif control:
        out["outcome_cell"] = "POLICY-STATIONARY"
    else:
        out["outcome_cell"] = "POLICY-STATIONARY-GREEDY-EXEC-SCOPED"
    return out


def run(args):
    dirs = (sorted(globmod.glob(args.runs)) if "," not in args.runs
            else [x.strip() for x in args.runs.split(",")
                  if x.strip()])
    assert dirs, f"no runs matched {args.runs!r}"
    assert args.output and args.output != "TBD", (
        "registered read must persist: pass an explicit --output")
    out_json = os.path.join(args.output, "se_refine_read.json")
    assert not os.path.exists(out_json), (
        f"{out_json} exists — the read is ONE execution")
    recs, excluded = [], []
    for d in dirs:
        try:
            recs.append(read_run(d))
        except Exception as e:              # noqa: BLE001
            excluded.append(dict(run_dir=os.path.abspath(d),
                                 error=f"{type(e).__name__}: {e}"))
    out = aggregate(recs, excluded)
    out["fit_counters"] = {r["run_dir"]: fit_counters(
        r["run_dir"], args.expect_steps) for r in recs}
    os.makedirs(args.output, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    if isinstance(out["primary"], dict):
        pr = out["primary"]
        print(f"REFINE PRIMARY: {pr['n_pos']}/{N_REGISTERED} pos "
              f"p={pr['p_signflip']:.4f} mean={pr['mean']:+.5f} -> "
              f"{out['outcome_cell']}")
    else:
        print(f"REFINE PRIMARY: {out['outcome_cell']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, ad=None, intr=None, cem_ok=True,
             inplan_ok=True, smoke=False, doctor=None,
             drop_npz=False):
    rng = np.random.default_rng(seed * 37 + 3)
    rd = os.path.join(tmp, name)
    pdir = os.path.join(rd, "se_refine")
    os.makedirs(pdir, exist_ok=True)
    ck = os.path.join(rd, "ckpt")
    os.makedirs(os.path.join(ck, "0000497000"), exist_ok=True)
    with open(os.path.join(ck, "latest"), "w") as f:
        f.write("0000497000")
    ad = dict(ad or {})
    intr = dict(intr or {})
    keys = ("position", "velocity", "distractor", "planted_dup0")
    per_arm, npz_out = {}, {}
    n_steps = 2000
    for a in ARMS:
        rec = dict(n_steps=n_steps, attr_mean={}, elapsed_s=1.0,
                   reset_hits=0, return_mean=100.0)
        for k in keys:
            want = ad.get(a, 0.02) if k == "distractor" else 0.01
            v = rng.normal(want, abs(want) * 0.1 + 1e-4, n_steps)
            v = (v - v.mean() + want).astype(np.float64)
            npz_out[f"{a}_attr_{k}"] = v
            rec["attr_mean"][k] = float(v.mean())
        iv = np.full(n_steps, intr.get(a, 3e-4), np.float64)
        npz_out[f"{a}_intr"] = iv
        rec["intrinsic_mean"] = float(iv.mean())
        ov = np.full(n_steps, 0.5, np.float64)
        npz_out[f"{a}_occ"] = ov
        rec["occupancy"] = float(ov.mean())
        if a in PLANNER_ARMS:
            rec["cem_improves"] = bool(cem_ok)
            rec["cem_improvement_mean"] = 0.4 if cem_ok else -0.1
            rec["inplan_gain_mean"] = 0.3 if inplan_ok else -0.05
            rec["inplan_improves"] = bool(inplan_ok)
            rec["refine_disp_mean"] = 0.21
            rec["clip_frac_mean"] = 0.07
        per_arm[a] = rec
    pj = dict(run_logdir=rd, ckpt=os.path.join(ck, "0000497000"),
              train_seed=seed,
              task=TASK_PIN, probe_seed=0, arms=list(ARMS),
              smoke=smoke, devices="[CudaDevice(id=0)]",
              backend="gpu", refine_std=REFINE_STD, chunk=100,
              gate=dict(key=GATE_KEY, index=GATE_INDEX,
                        threshold=GATE_THRESHOLD),
              per_arm=per_arm, **REG_PARAMS)
    if doctor:
        doctor(pj)
    with open(os.path.join(pdir, "se_refine.json"), "w") as f:
        json.dump(pj, f)
    npz_path = os.path.join(pdir, "se_refine.npz")
    if drop_npz:
        if os.path.exists(npz_path):
            os.remove(npz_path)
    else:
        np.savez(npz_path, **npz_out)
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(
            f"seed: {seed}\ntask: {TASK_PIN}\n"
            f"planted:\n  gate_key: ''\n  source_key: 'position'\n"
            f"  basesd: 0.0976\n"
            f"distractor:\n  gate_key: ''\n  mod_key: ''\n"
            f"  dim: {DIM}\n  scale: 1.0\n  basesd: {BASESD_N}\n"
            f"agent:\n  expl:\n    mode: p2e\n    disag_ens: 8\n"
            f"    disag_scale: 1000.0\n    disag_bootstrap: true\n"
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
    STAT = dict(random=0.006, policy=0.017, refine_attr=0.016,
                refine_disag=0.016)
    FIRE = dict(STAT, refine_attr=0.024)
    GOODI = dict(random=8e-5, policy=1.9e-4, refine_attr=1.5e-4,
                 refine_disag=1.85e-4)
    WEAKI = dict(GOODI, refine_disag=1.0e-4)   # ratio 0.53
    DEADI = dict(GOODI, policy=5e-5)           # policy < random
    with tempfile.TemporaryDirectory() as tmp:
        # POLICY-STATIONARY (full wording: strength scope passes)
        runs = [_fixture(tmp, f"a{s}", s, ad=STAT, intr=GOODI)
                for s in range(10, 18)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "POLICY-STATIONARY", (
            agg["primary"])
        assert agg["inplan_gate"]["passes"]
        assert agg["realized_strength_scope"]["passes"]
        assert agg["s_share"]["policy"], "descriptives missing"
        assert agg["per_run"][0]["repro_vs_ambient"]["policy"][1] \
            == REPRO_ANCHORS[10][0]
        # REFINEMENT-TRANSMISSION at exactly the NEED boundary:
        # 7/8 positive fires...
        runs = [_fixture(tmp, f"b{s}", s,
                         ad=(FIRE if s != 17 else STAT), intr=GOODI)
                for s in range(10, 18)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["primary"]["n_pos"] == 7
        assert agg["outcome_cell"] == "REFINEMENT-TRANSMISSION"
        assert agg["primary"]["p_signflip"] <= 0.05
        # ...and 6/8 does NOT (rev A-m6)
        runs = [_fixture(tmp, f"n{s}", s,
                         ad=(FIRE if s < 16 else STAT), intr=GOODI)
                for s in range(10, 18)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["primary"]["n_pos"] == 6
        assert agg["outcome_cell"] == "POLICY-STATIONARY"
        # realized-strength scope failure -> SCOPED null, never
        # INSTRUMENT-WEAK (rev A-M2)
        runs = [_fixture(tmp, f"c{s}", s, ad=STAT, intr=WEAKI)
                for s in range(10, 18)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == \
            "POLICY-STATIONARY-GREEDY-EXEC-SCOPED", (
            agg["outcome_cell"])
        # in-plan gate failure -> INSTRUMENT-WEAK (rev A-B1)
        runs = [_fixture(tmp, f"w{s}", s, ad=STAT, intr=GOODI,
                         inplan_ok=False) for s in range(10, 18)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == \
            "NOT-ADJUDICABLE-INSTRUMENT-WEAK"
        assert agg["s_refine_disag_attr"], "descriptives missing"
        # degenerate runs (policy <= random intrinsic) drop validity
        # (rev A-M1)
        runs = [_fixture(tmp, f"d{s}", s, ad=STAT,
                         intr=(DEADI if s < 12 else GOODI))
                for s in range(10, 18)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert len(agg["degenerate_runs"]) == 2
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE"
        # gates
        rd = _fixture(tmp, "g0", 10, ad=STAT, intr=GOODI, smoke=True)
        _expect_fail(lambda: read_run(rd), "smoke")

        def doc(pj):
            pj["refine_std"] = 0.5
        rd = _fixture(tmp, "g1", 10, ad=STAT, intr=GOODI, doctor=doc)
        _expect_fail(lambda: read_run(rd), "refine_std")

        def doc2(pj):
            pj["per_arm"]["refine_attr"]["attr_mean"]["distractor"] \
                = 9.9
        rd = _fixture(tmp, "g2", 11, ad=STAT, intr=GOODI, doctor=doc2)
        _expect_fail(lambda: read_run(rd), "provenance")
        rd = _fixture(tmp, "g3", 55, ad=STAT, intr=GOODI)
        _expect_fail(lambda: read_run(rd), "outside Stage-1")

        def doc3(pj):
            pj["gate"]["threshold"] = 0.5
        rd = _fixture(tmp, "g4", 12, ad=STAT, intr=GOODI, doctor=doc3)
        _expect_fail(lambda: read_run(rd), "gate constants")

        def doc4(pj):
            del pj["per_arm"]["refine_disag"]["inplan_gain_mean"]
        rd = _fixture(tmp, "g5", 13, ad=STAT, intr=GOODI, doctor=doc4)
        _expect_fail(lambda: read_run(rd), "pre-revision")
        # excluded-run path through run() + one-execution guard
        outd = os.path.join(tmp, "out")
        for s in range(10, 18):
            _fixture(tmp, f"e{s}", s, ad=STAT, intr=GOODI,
                     drop_npz=(s == 10))
        out = run(parse_args(["--runs", os.path.join(tmp, "e1*"),
                              "--output", outd]))
        assert len(out["excluded_runs"]) == 1
        assert out["outcome_cell"] == "POLICY-STATIONARY"
        _expect_fail(lambda: run(parse_args(
            ["--runs", os.path.join(tmp, "e1*"), "--output", outd])),
            "ONE execution")
    print("se_refine_read selfcheck PASS (STATIONARY full/scoped + "
          "TRANSMISSION at the 7/8 boundary (6/8 rejected) + "
          "in-plan INSTRUMENT-WEAK gate + degeneracy floor + "
          "NOT-ADJUDICABLE; smoke/refine_std/provenance/seed/gate-"
          "constants/pre-revision-output gates; descriptives on "
          "every path; excluded path + one-execution guard)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
