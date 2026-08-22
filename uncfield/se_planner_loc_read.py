"""Wave-C (localized-channel planner) FROZEN reader
(PREREG_planner_localized_20260822.md §2; built + selfchecked BEFORE
the wave's compute; ONE execution).

8 runs = the A1 checkpoints: hetero (scarecrow field, seeds 44-47)
vs flat (amplitude-matched, seeds 48-51), probed by the FROZEN
se_planner_probe (unchanged, registered constants).

PRIMARY (conjunctive): per-run Delta = attr_d(cem_disag) -
attr_d(policy); fires iff (i) mean Delta(hetero) - mean Delta(flat)
> 0 at exact C(8,4) label-permutation p <= .05 AND (ii) Delta > 0 in
>= 3/4 hetero runs. FULL 4+4 ONLY (#34 B3). Secondaries
(registered, no fire): region entry occ(cem_disag)-occ(policy),
SHARE rows (pre-declared this wave), hetero steerability row,
per-arm levels.

Run:  python -m uncfield.se_planner_loc_read --runs "<glob>"
          --output <dir>
      python -m uncfield.se_planner_loc_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import os
import subprocess
import time

import numpy as np

from uncfield.se_grad_read import (BASESD_N, DIM, FLAT_SCALE, MOD_HI,
                                   MOD_KEY, MOD_INDEX, MOD_LO)
from uncfield.se_m3_read import (exact_perm_p, GATE_INDEX, GATE_KEY,
                                 GATE_THRESHOLD)
from uncfield.se_mask import bca_interval
from uncfield.se_probe import resolve_ckpt
from uncfield.se_read import fit_counters

HET_SEEDS = (44, 45, 46, 47)
FLAT_SEEDS = (48, 49, 50, 51)
SEED_ARM = {**{s: "hetero" for s in HET_SEEDS},
            **{s: "flat" for s in FLAT_SEEDS}}
ARMS = ("random", "policy", "cem_disag", "cem_real", "cem_distractor")
PLANNER_ARMS = ("cem_disag", "cem_real", "cem_distractor")
REG_PARAMS = dict(episodes=4, ep_len=500, n_cand=256, n_elite=32,
                  iters=3, horizon=12)
TASK_PIN = "dmc_cheetah_run"
HET_NEED = 3          # of 4 hetero runs with Delta > 0
CKPT_STEP_FLOOR = 490_000   # A1 disclosed band 494,432-498,896
CTRL_RATIO = 0.9      # matched-strength control (the Fable-F2 bar)
CTRL_NEED = 3         # of 4 hetero runs


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
    assert train_seed in SEED_ARM, (
        f"{run_dir}: seed {train_seed} outside the registered 44-51")
    arm = SEED_ARM[train_seed]
    assert str(cfg["task"]) == TASK_PIN, run_dir
    assert str(ex["mode"]) == "p2e" and int(ex["disag_ens"]) == 8 \
        and abs(float(ex["disag_scale"]) - 1000.0) < 1e-9 \
        and ex["disag_bootstrap"] is True, (
        f"{run_dir}: expl pins differ from Stage-1")
    assert int(d["dim"]) == DIM and \
        abs(float(d["basesd"]) - BASESD_N) < 1e-9
    assert str(d.get("gate_key", "")) == "" and \
        str(pl.get("gate_key", "")) == "", run_dir
    assert str(pl["source_key"]) == "position" and \
        abs(float(pl["basesd"]) - 0.0976) < 1e-9, run_dir
    assert float(cfg.get("penalty", {}).get("scale", 0.0)) == 0.0 \
        and ex.get("penalty_mix", False) is False, (
        f"{run_dir}: penalty machinery must be inert")
    if arm == "hetero":
        assert abs(float(d["scale"]) - 1.0) < 1e-9, (
            f"{run_dir}: hetero scale {d['scale']}")
        assert str(d["mod_key"]) == MOD_KEY and \
            int(d["mod_index"]) == MOD_INDEX and \
            abs(float(d["mod_lo"]) - MOD_LO) < 1e-9 and \
            abs(float(d["mod_hi"]) - MOD_HI) < 1e-9, (
            f"{run_dir}: hetero mod quadruple differs from A1")
    else:
        assert abs(float(d["scale"]) - FLAT_SCALE) < 1e-9, (
            f"{run_dir}: flat scale {d['scale']}")
        assert str(d.get("mod_key", "")) == "", run_dir

    pdir = os.path.join(run_dir, "se_planner")
    with open(os.path.join(pdir, "se_planner.json")) as f:
        pj = json.load(f)
    npz = np.load(os.path.join(pdir, "se_planner.npz"))
    assert not pj["smoke"], f"{run_dir}: smoke output"
    assert list(pj["arms"]) == list(ARMS), pj["arms"]
    assert int(pj["train_seed"]) == train_seed, run_dir
    assert str(pj["task"]) == TASK_PIN and int(pj["probe_seed"]) == 0
    assert os.path.basename(os.path.normpath(pj["run_logdir"])) == \
        os.path.basename(os.path.normpath(run_dir)), (
        f"{run_dir}: probe json belongs to {pj['run_logdir']} "
        f"(rev C-m4)")
    for k, v in REG_PARAMS.items():
        assert int(pj[k]) == v, (
            f"{run_dir}: {k}={pj[k]} != registered {v}")
    g = pj["gate"]
    assert str(g["key"]) == GATE_KEY and int(g["index"]) == GATE_INDEX \
        and abs(float(g["threshold"]) - GATE_THRESHOLD) < 1e-12, (
        f"{run_dir}: occupancy gate constants drifted (rev C-M2)")
    # final-ckpt + newest-DIR cross-check + registered step floor
    # (rev C-B2: these 8 run dirs are where the stale ckpt/latest
    # incident actually happened — 4/8 first probed at 105k-133k)
    ck_probe = os.path.basename(os.path.normpath(pj["ckpt"]))
    ckroot = os.path.join(run_dir, "ckpt")
    try:
        subs = sorted(x for x in os.listdir(ckroot)
                      if os.path.isdir(os.path.join(ckroot, x)))
    except OSError:
        subs = []
    ckpt_final_ok = None            # ckpt/ unbundled -> UNVERIFIED
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
    ck_digits = "".join(c for c in ck_probe if c.isdigit())
    assert ck_digits and int(ck_digits) >= CKPT_STEP_FLOOR, (
        f"{run_dir}: probed ckpt {ck_probe} below the registered "
        f"step floor {CKPT_STEP_FLOOR}")
    n_dev = str(pj["devices"]).count("id=")
    assert n_dev == 1, f"{run_dir}: {n_dev} devices — panel must be 1"
    assert str(pj.get("backend", "")).lower() in ("gpu", "cuda"), (
        f"{run_dir}: backend {pj.get('backend')!r} — CPU fallback is "
        f"a different numeric regime")
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
        for jkey, nkey in (("intrinsic_mean", "intr"),
                           ("occupancy", "occ")):
            rc = float(np.asarray(npz[f"{a}_{nkey}"],
                                  np.float64).mean())
            assert rc == float(rec[jkey]), (run_dir, a, jkey)
        per_arm[a] = rec
    cem_ok = all(pj["per_arm"][a].get("cem_improves") is True
                 for a in PLANNER_ARMS)
    # value-aware normalizer accounting (rev C-M1/C-m7): hetero and
    # flat normalizers come from each run's OWN realized distractor
    # (hetero is heteroscedastic; Jensen makes its global std larger)
    norm_d = np.asarray(npz["norm_distractor"], np.float64)
    rawnorm_d = np.asarray(npz["rawnorm_distractor"], np.float64)
    return dict(run_dir=os.path.abspath(run_dir),
                train_seed=train_seed, arm=arm, cem_ok=cem_ok,
                ckpt=ck_probe, ckpt_final_ok=ckpt_final_ok,
                devices=str(pj["devices"]),
                backend=str(pj.get("backend", "")),
                norm_d_mean=float(norm_d.mean()),
                norm_sq_mean=float((norm_d ** 2).mean()),
                norm_floor_bound=bool(
                    not np.array_equal(norm_d, rawnorm_d)),
                per_arm=per_arm)


def _ad(rec, arm):
    return float(rec["per_arm"][arm]["attr_mean"]["distractor"])


def _share(rec, arm):
    am = rec["per_arm"][arm]["attr_mean"]
    return float(am["distractor"] / sum(am.values()))


def aggregate(recs, excluded):
    seeds = sorted(r["train_seed"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert len(recs) + len(excluded) <= 8, "runs > registered 8"
    het = [r for r in recs if r["arm"] == "hetero" and r["cem_ok"]]
    fla = [r for r in recs if r["arm"] == "flat" and r["cem_ok"]]
    out = dict(stamp=_stamp(), n_loaded=len(recs),
               excluded_runs=excluded,
               invalid_cem=[r["run_dir"] for r in recs
                            if not r["cem_ok"]],
               n_hetero=len(het), n_flat=len(fla))

    out["per_run"] = [
        {"run_dir": r["run_dir"], "arm": r["arm"],
         "train_seed": r["train_seed"], "cem_ok": r["cem_ok"],
         "ckpt": r["ckpt"], "ckpt_final_ok": r["ckpt_final_ok"],
         "devices": r["devices"], "backend": r["backend"],
         "delta_disag_minus_policy":
             _ad(r, "cem_disag") - _ad(r, "policy"),
         "delta_raw_approx":
             (_ad(r, "cem_disag") - _ad(r, "policy"))
             * r["norm_sq_mean"],
         "norm_d_mean": r["norm_d_mean"],
         "norm_sq_mean": r["norm_sq_mean"],
         "norm_floor_bound": r["norm_floor_bound"],
         "attr_d": {a: _ad(r, a) for a in ARMS},
         "share_d": {a: _share(r, a) for a in ARMS},
         "occupancy": {a: r["per_arm"][a]["occupancy"]
                       for a in ARMS},
         "intrinsic": {a: r["per_arm"][a]["intrinsic_mean"]
                       for a in ARMS}} for r in recs]

    # registered secondaries (rows always)
    out["s1_region_entry"] = {
        "hetero": [dict(seed=r["train_seed"],
                        occ_disag=r["per_arm"]["cem_disag"]
                        ["occupancy"],
                        occ_policy=r["per_arm"]["policy"]
                        ["occupancy"],
                        delta=r["per_arm"]["cem_disag"]["occupancy"]
                        - r["per_arm"]["policy"]["occupancy"])
                   for r in het],
        "flat": [dict(seed=r["train_seed"],
                      delta=r["per_arm"]["cem_disag"]["occupancy"]
                      - r["per_arm"]["policy"]["occupancy"])
                 for r in fla]}
    out["s2_share"] = {
        wing: [dict(seed=r["train_seed"],
                    disag=_share(r, "cem_disag"),
                    distractor_planner=_share(r, "cem_distractor"),
                    policy=_share(r, "policy"))
               for r in grp]
        for wing, grp in (("hetero", het), ("flat", fla))}
    out["s3_steerability_hetero"] = [
        dict(seed=r["train_seed"],
             ceil=_ad(r, "cem_distractor")
             - max(_ad(r, "policy"), _ad(r, "random")))
        for r in het]

    if len(het) < 4 or len(fla) < 4:
        out["primary"] = (f"NOT-ADJUDICABLE (hetero {len(het)}, "
                          f"flat {len(fla)}; full 4+4 required — "
                          f"#34 B3)")
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        return out

    d_het = [_ad(r, "cem_disag") - _ad(r, "policy") for r in het]
    d_fla = [_ad(r, "cem_disag") - _ad(r, "policy") for r in fla]
    vals = np.asarray(d_het + d_fla, np.float64)
    labels = np.array([True] * 4 + [False] * 4)
    obs, p, total = exact_perm_p(vals, labels, one_sided=True)
    assert total == math.comb(8, 4)
    n_het_pos = sum(1 for x in d_het if x > 0)
    fires = bool(obs > 0 and p <= 0.05 and n_het_pos >= HET_NEED)
    # matched-strength control (rev C-B1, Fable F2): the incumbent is
    # a trained optimizer of the measured field; a null from a CEM
    # that cannot even match the policy's realized intrinsic is
    # scoped to THIS optimizer at THIS strength, never upgraded
    ctrl = [dict(seed=r["train_seed"],
                 ratio=(r["per_arm"]["cem_disag"]["intrinsic_mean"]
                        / max(r["per_arm"]["policy"]
                              ["intrinsic_mean"], 1e-30)),
                 policy_gt_random=bool(
                     r["per_arm"]["policy"]["intrinsic_mean"]
                     > r["per_arm"]["random"]["intrinsic_mean"]))
            for r in het]
    ctrl_pass = (sum(1 for c in ctrl if c["ratio"] >= CTRL_RATIO)
                 >= CTRL_NEED
                 and all(c["policy_gt_random"] for c in ctrl))
    out["matched_strength_control"] = dict(
        per_run=ctrl, ratio_bar=CTRL_RATIO, need=CTRL_NEED,
        passes=bool(ctrl_pass))
    # de-normalized robustness row (rev C-M1) + indicative intervals
    dr_het = [pr["delta_raw_approx"] for pr in out["per_run"]
              if pr["arm"] == "hetero" and pr["cem_ok"]]
    dr_fla = [pr["delta_raw_approx"] for pr in out["per_run"]
              if pr["arm"] == "flat" and pr["cem_ok"]]
    out["denorm_robustness"] = dict(
        interaction_raw_approx=float(np.mean(dr_het)
                                     - np.mean(dr_fla)),
        note="Delta * mean(norm_d^2); wing-asymmetric normalizers "
             "(Jensen) shrink hetero Delta — this row is the "
             "unshrunk comparator, descriptive")
    rngb = np.random.default_rng(2026_08_22)
    out["wing_bca_indicative"] = {
        w: list(map(float, bca_interval(np.asarray(v), rngb, 2000)))
        for w, v in (("hetero", d_het), ("flat", d_fla))}
    out["primary"] = dict(
        statement="localized transmission: the deployed-objective "
                  "planner's attribution gain over the policy is "
                  "LARGER on the scarecrow field than on the flat "
                  "comparator (exact C(8,4), one-sided) AND positive "
                  "in >= 3/4 hetero runs; power note: fire requires "
                  "near-complete separation (floor 1/70), no power "
                  "against moderate effects",
        delta_hetero=d_het, delta_flat=d_fla,
        interaction=float(obs), p=float(p),
        min_attainable_p=1 / 70, n_het_pos=n_het_pos,
        need=HET_NEED, fires=fires)
    if fires:
        out["outcome_cell"] = "LOCALIZED-TRANSMISSION"
    elif ctrl_pass:
        out["outcome_cell"] = "NO-LOCALIZED-TRANSMISSION-MATCHED"
    else:
        out["outcome_cell"] = (
            "NO-LOCALIZED-TRANSMISSION-WEAK-OPTIMIZER")
    return out


def run(args):
    dirs = (sorted(globmod.glob(args.runs)) if "," not in args.runs
            else [x.strip() for x in args.runs.split(",")
                  if x.strip()])
    assert dirs, f"no runs matched {args.runs!r}"
    assert args.output and args.output != "TBD", (
        "registered read must persist: pass an explicit --output")
    out_json = os.path.join(args.output, "se_planner_loc_read.json")
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
        print(f"LOC PRIMARY: interaction={pr['interaction']:+.5f} "
              f"p={pr['p']:.4f} het_pos={pr['n_het_pos']}/4 "
              f"fires={pr['fires']} -> {out['outcome_cell']}")
    else:
        print(f"LOC PRIMARY: {out['outcome_cell']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, ad=None, cem_ok=True, smoke=False,
             doctor=None, drop_npz=False, intr=None,
             ckpt_name="0000497000", stale_extra_ckpt=False):
    """Fake A1-checkpoint planner-probe run for the seed's arm."""
    rng = np.random.default_rng(seed * 31 + 5)
    arm = SEED_ARM.get(seed, "hetero")   # off-family seeds still
    rd = os.path.join(tmp, name)         # build (the gate is tested)
    pdir = os.path.join(rd, "se_planner")
    os.makedirs(pdir, exist_ok=True)
    ck = os.path.join(rd, "ckpt")
    os.makedirs(os.path.join(ck, ckpt_name), exist_ok=True)
    if stale_extra_ckpt:
        os.makedirs(os.path.join(ck, "0000499999"), exist_ok=True)
    with open(os.path.join(ck, "latest"), "w") as f:
        f.write(ckpt_name)
    ad = dict(ad or {})
    intr = dict(intr or {})
    keys = ("position", "velocity", "distractor", "planted_dup0")
    per_arm, npz_out = {}, {}
    n_steps = 2000
    for a in ARMS:
        rec = dict(n_steps=n_steps, attr_mean={}, elapsed_s=1.0,
                   return_mean=0.0, reset_hits=0)
        for k in keys:
            want = ad.get(a, 0.02) if k == "distractor" else 0.01
            v = rng.normal(want, abs(want) * 0.1 + 1e-4, n_steps)
            v = (v - v.mean() + want).astype(np.float64)
            npz_out[f"{a}_attr_{k}"] = v
            rec["attr_mean"][k] = float(v.mean())
        intr_base = intr.get(a, 2e-4 if a == "random" else 3e-4)
        for jkey, nkey, base in (("intrinsic_mean", "intr",
                                  intr_base),
                                 ("occupancy", "occ",
                                  0.46 if a == "policy" else 0.6)):
            v = np.full(n_steps, base, np.float64)
            npz_out[f"{a}_{nkey}"] = v
            rec[jkey] = float(v.mean())
        if a in PLANNER_ARMS:
            rec["cem_improvement_mean"] = 0.5 if cem_ok else -0.1
            rec["cem_improves"] = bool(cem_ok)
        per_arm[a] = rec
    npz_out["norm_distractor"] = np.full(8, 0.435, np.float64)
    npz_out["rawnorm_distractor"] = np.full(8, 0.435, np.float64)
    pj = dict(run_logdir=rd, ckpt=os.path.join(ck, ckpt_name),
              train_seed=seed,
              task=TASK_PIN, probe_seed=0, arms=list(ARMS),
              smoke=smoke, devices="[CudaDevice(id=0)]",
              backend="gpu",
              gate=dict(key=GATE_KEY, index=GATE_INDEX,
                        threshold=GATE_THRESHOLD),
              per_arm=per_arm, **REG_PARAMS)
    if doctor:
        doctor(pj)
    with open(os.path.join(pdir, "se_planner.json"), "w") as f:
        json.dump(pj, f)
    npz_path = os.path.join(pdir, "se_planner.npz")
    if drop_npz:
        if os.path.exists(npz_path):
            os.remove(npz_path)          # re-fixture must not keep it
    else:
        np.savez(npz_path, **npz_out)
    het = arm == "hetero"
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(
            f"seed: {seed}\ntask: {TASK_PIN}\n"
            f"planted:\n  gate_key: ''\n  source_key: 'position'\n"
            f"  basesd: 0.0976\n"
            f"distractor:\n  gate_key: ''\n"
            f"  mod_key: '{MOD_KEY if het else ''}'\n"
            f"  mod_index: {MOD_INDEX}\n  mod_lo: {MOD_LO}\n"
            f"  mod_hi: {MOD_HI}\n  dim: {DIM}\n"
            f"  scale: {1.0 if het else FLAT_SCALE}\n"
            f"  basesd: {BASESD_N}\n"
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
    # arm attr layouts: FIRE = hetero planner beats policy, flat not
    FIRE_HET = dict(random=0.010, policy=0.020, cem_disag=0.045,
                    cem_real=0.015, cem_distractor=0.050)
    FIRE_FLA = dict(random=0.006, policy=0.017, cem_disag=0.009,
                    cem_real=0.008, cem_distractor=0.012)
    NULL_HET = dict(FIRE_HET, cem_disag=0.012)
    with tempfile.TemporaryDirectory() as tmp:
        def wave(sub, het_ad, fla_ad, **kw):
            runs = []
            for s in HET_SEEDS:
                runs.append(_fixture(tmp, f"{sub}{s}", s, ad=het_ad,
                                     **kw))
            for s in FLAT_SEEDS:
                runs.append(_fixture(tmp, f"{sub}{s}", s, ad=fla_ad,
                                     **kw))
            return runs

        runs = wave("a", FIRE_HET, FIRE_FLA)
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "LOCALIZED-TRANSMISSION", (
            agg["primary"])
        assert agg["primary"]["p"] == 1 / 70
        assert agg["s1_region_entry"]["hetero"][0]["delta"] > 0
        assert agg["s2_share"]["hetero"][0]["disag"] > 0
        assert agg["matched_strength_control"]["passes"]
        assert "denorm_robustness" in agg
        assert agg["wing_bca_indicative"]["hetero"][0] > 0

        # null with a MATCHED-strength attacker (control passes)
        runs = wave("b", NULL_HET, FIRE_FLA)
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == \
            "NO-LOCALIZED-TRANSMISSION-MATCHED", agg["outcome_cell"]

        # null with a WEAK optimizer (cem_disag realized intrinsic
        # far below the policy's — the Fable-F2 scope, rev C-B1)
        runs = wave("w", NULL_HET, FIRE_FLA,
                    intr=dict(cem_disag=1.2e-4, policy=3e-4))
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == \
            "NO-LOCALIZED-TRANSMISSION-WEAK-OPTIMIZER", (
            agg["outcome_cell"])

        # count guard (rev C-M6): 2/4 hetero positive can reach the
        # permutation floor yet MUST NOT fire
        runs = []
        for s, cd in zip(HET_SEEDS, (0.030, 0.030, 0.019, 0.019)):
            runs.append(_fixture(tmp, f"k{s}", s,
                                 ad=dict(FIRE_HET, cem_disag=cd)))
        for s in FLAT_SEEDS:
            runs.append(_fixture(tmp, f"k{s}", s, ad=FIRE_FLA))
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["primary"]["p"] == 1 / 70, agg["primary"]
        assert agg["primary"]["n_het_pos"] == 2
        assert not agg["primary"]["fires"]
        assert agg["outcome_cell"].startswith(
            "NO-LOCALIZED-TRANSMISSION")

        # full-4+4-only: drop one hetero -> NOT-ADJUDICABLE
        recs = [read_run(d) for d in wave("c", FIRE_HET, FIRE_FLA)]
        agg = aggregate([r for r in recs if r["train_seed"] != 44],
                        [dict(run_dir="x", error="e")])
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE"
        # cem-invalid hetero run -> NOT-ADJUDICABLE
        runs = wave("d", FIRE_HET, FIRE_FLA)
        runs[0] = _fixture(tmp, "d44x", 44, ad=FIRE_HET,
                           cem_ok=False)
        agg = aggregate([read_run(d) for d in runs
                         if not d.endswith("d44")], [])
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE"

        # gates
        rd = _fixture(tmp, "g0", 44, ad=FIRE_HET)
        cfgp = os.path.join(rd, "config.yaml")
        cfg = open(cfgp).read()
        open(cfgp, "w").write(cfg.replace(f"mod_key: '{MOD_KEY}'",
                                          "mod_key: ''"))
        _expect_fail(lambda: read_run(rd), "mod quadruple")
        rd = _fixture(tmp, "g1", 48, ad=FIRE_FLA)
        cfg = open(os.path.join(rd, "config.yaml")).read()
        open(os.path.join(rd, "config.yaml"), "w").write(
            cfg.replace(f"scale: {FLAT_SCALE}", "scale: 1.0"))
        _expect_fail(lambda: read_run(rd), "flat scale")
        rd = _fixture(tmp, "g2", 60, ad=FIRE_HET)
        _expect_fail(lambda: read_run(rd), "registered 44-51")
        rd = _fixture(tmp, "g3", 45, ad=FIRE_HET, smoke=True)
        _expect_fail(lambda: read_run(rd), "smoke")

        def doctor(pj):
            pj["per_arm"]["cem_disag"]["attr_mean"]["distractor"] = 9.
        rd = _fixture(tmp, "g4", 46, ad=FIRE_HET, doctor=doctor)
        _expect_fail(lambda: read_run(rd), "provenance")
        # ckpt gates (rev C-B2): stale pointer / low-step / gate drift
        rd = _fixture(tmp, "g5", 47, ad=FIRE_HET,
                      stale_extra_ckpt=True)
        _expect_fail(lambda: read_run(rd), "stale ckpt/latest")
        rd = _fixture(tmp, "g6", 47, ad=FIRE_HET,
                      ckpt_name="0000132000")
        _expect_fail(lambda: read_run(rd), "step floor")

        def doctor_gate(pj):
            pj["gate"]["threshold"] = 0.5
        rd = _fixture(tmp, "g7", 47, ad=FIRE_HET, doctor=doctor_gate)
        _expect_fail(lambda: read_run(rd), "gate constants")

        # excluded-run persistence via run()
        outd = os.path.join(tmp, "out")
        wave("e", FIRE_HET, FIRE_FLA)
        _fixture(tmp, "e44", 44, ad=FIRE_HET, drop_npz=True)
        out = run(parse_args(["--runs", os.path.join(tmp, "e[45]*"),
                              "--output", outd]))
        assert len(out["excluded_runs"]) == 1
        assert out["outcome_cell"] == "NOT-ADJUDICABLE"
        _expect_fail(lambda: run(parse_args(
            ["--runs", os.path.join(tmp, "e[45]*"),
             "--output", outd])), "ONE execution")
    print("se_planner_loc_read selfcheck PASS (fire at the exact "
          "floor + MATCHED/WEAK-OPTIMIZER null cells + count guard "
          "at the floor + full-4+4-only paths (missing + "
          "cem-invalid) + mod/flat/seed/smoke/provenance/stale-ckpt/"
          "step-floor/gate-constant gates + excluded-run "
          "persistence + one-execution guard)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
