"""Track-B ONSET-ladder FROZEN reader
(PREREG_trackB_onset_20260822.md §2; built + selfchecked BEFORE the
wave's compute; ONE execution). Revised pre-freeze per review R1
(22 Aug): grid moved onto the transition window {0.25, 0.30, 0.34,
0.38} (the registered [0.30,0.45] sampled the plateau against the
prereg's own anchors); identity gates FATAL (B1 pattern, not
silent-drop); missing-seed accounting + flagged-subset statement;
per-scale coverage/score cost curve delivered; logistic midpoint
with boundary/window flags + run-resample percentile interval;
ckpt-provenance recorded per run and aggregated.

PRIMARY: Spearman rho(theta_1, scale) > 0, one-sided POSITIVE label
permutation (100k draws, seed 2026_08_22); fires iff p <= .05 AND
rho > 0. Floor accounting FLIPPED from B1 (a binding floor deflates
low-scale theta_1, ANTI-conservative for a positive primary): a
FIRE with any floored loaded run is NOT-ADJUDICABLE; a
NON-CONFIRMED outcome stands. The fire licenses ONLY "the rising
limb replicates on fresh data" — the SHAPE (threshold vs smooth)
stays descriptive (crossing fractions + logistic midpoint, never a
fire).

Machinery reused verbatim from the FROZEN, executed B1 reader
(se_dose_read: spearman_perm_null, coverage_proxy, score_tail,
stage1_reference_gated) and the house per-run guts (se_read.read_run,
se_noboot_read._levels).

Run:  python -m uncfield.se_onset_read --runs "<glob>"
          --stage1_runs "<glob>" --output <dir>
      python -m uncfield.se_onset_read --selfcheck
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
from uncfield.se_mask import bca_interval
from uncfield.se_noboot_read import COLLAPSE_FACTOR, _levels
from uncfield.se_dose_read import (coverage_proxy, score_tail,
                                   spearman_perm_null,
                                   stage1_reference_gated)
from uncfield.se_read import fit_counters
from uncfield import se_read

SCALES = (0.25, 0.30, 0.34, 0.38)
SCALE_SEEDS = {0.25: set(range(96, 100)), 0.30: set(range(100, 104)),
               0.34: set(range(104, 108)), 0.38: set(range(108, 112))}
ALL_SEEDS = set(range(96, 112))
N_PERM_LABELS = 100_000
PERM_SEED = 2026_08_22
ALPHA = 0.05
MIN_USABLE = 14
MIN_PER_SCALE = 3
TASK = "dmc_cheetah_run"
ANCHORS = {                       # CROSS-WAVE DESCRIPTIVE only
    "b1@0.10": 0.11, "b1@0.25_bimodal_mean": 0.83,
    "b1@0.25_values": [0.09, 0.13, 1.52, 1.55],
    "b1@0.50": 8.41, "b1@0.75": 6.56, "flat@0.359": 7.10,
    "stage1@1.0": 5.26}
# logistic descriptive fit: m grid brackets the data window
# [0.25, 0.38] with margin; k covers BOTH signs (a reversed crossing
# pattern must fit as falling, not pin to a boundary — review M7)
LOGISTIC_M = np.linspace(0.20, 0.43, 93)
LOGISTIC_K = np.concatenate([np.linspace(-200.0, -5.0, 40),
                             np.linspace(5.0, 200.0, 40)])
DATA_WINDOW = (0.25, 0.38)
N_BOOT_MID = 200


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="")
    p.add_argument("--stage1_runs", default="")
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--n_perm_labels", type=int, default=N_PERM_LABELS)
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


def read_run(run_dir, n_perm):
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = cfg["distractor"]
    pl_ = cfg["planted"]
    ex = cfg["agent"]["expl"]
    scale = float(d["scale"])
    grid = min(SCALES, key=lambda x: abs(x - scale))
    assert abs(scale - grid) < 1e-9, (
        f"{run_dir}: scale {scale} is not on the registered grid "
        f"{SCALES}")
    # B1 pin block verbatim
    assert str(cfg.get("task", TASK)) == TASK, (
        f"{run_dir}: task {cfg.get('task')} != {TASK}")
    assert str(ex["mode"]) == "p2e" and \
        ex["disag_bootstrap"] is True and \
        int(ex["disag_ens"]) == 8 and \
        abs(float(ex["disag_scale"]) - 1000.0) < 1e-9 and \
        abs(float(ex.get("disag_bootstrap_prob", 0.8)) - 0.8) < 1e-9, (
        f"{run_dir}: expl pins differ from Stage-1")
    assert str(pl_["source_key"]) == "position" and \
        abs(float(pl_["basesd"]) - 0.0976) < 1e-9, (
        f"{run_dir}: planted pins differ from Stage-1")
    assert int(d["dim"]) == DIM and \
        abs(float(d["basesd"]) - BASESD_N) < 1e-9
    assert abs(float(d.get("theta", 0.1)) - 0.1) < 1e-9, (
        f"{run_dir}: distractor.theta must stay 0.1")
    assert str(d.get("gate_key", "")) == "" and \
        str(d.get("mod_key", "")) == "", (
        f"{run_dir}: gates/mod must be off")
    assert str(pl_.get("gate_key", "")) == ""
    assert float(cfg.get("penalty", {}).get("scale", 0.0)) == 0.0 \
        and ex.get("penalty_mix", False) is False, (
        f"{run_dir}: penalty machinery must be inert")
    rec = se_read.read_run(run_dir, "se_probe", n_perm)
    rec["train_seed_cfg"] = int(cfg["seed"])
    assert rec["train_seed_cfg"] in SCALE_SEEDS[grid], (
        f"{run_dir}: seed {rec['train_seed_cfg']} not registered for "
        f"scale {grid}")
    rec["scale"] = grid
    # newest-ckpt cross-check (the A1 append-verify lesson) — record
    # whether the gate was LIVE, not just whether it passed (rev M9)
    ckroot = os.path.join(run_dir, "ckpt")
    try:
        subs = sorted(x for x in os.listdir(ckroot)
                      if os.path.isdir(os.path.join(ckroot, x)))
    except OSError:
        subs = []
    rec["newest_ckpt_checked"] = bool(subs)
    if subs:
        with open(os.path.join(run_dir, "se_probe",
                               "se_probe.json")) as f:
            probed = os.path.basename(
                os.path.normpath(json.load(f)["ckpt"]))
        assert probed == subs[-1], (
            f"{run_dir}: probed ckpt {probed} != newest {subs[-1]} — "
            f"stale ckpt/latest pointer")
    lv = _levels(run_dir)
    stored = rec["channels"]["distractor"]["vs_source"]
    rc = lv["theta_recomputed"].get("distractor")
    assert rc is not None and abs(rc - stored) <= 1e-4 * max(
        1.0, abs(stored)), (
        f"{run_dir}: theta_1 provenance mismatch")
    rec["levels"] = lv
    npz = np.load(os.path.join(run_dir, "se_probe",
                               "se_probe_dims.npz"))
    dk = np.asarray(npz["dim_key"]).astype(str)
    raw_norms = np.asarray(npz["raw_norms"], np.float64)
    floor = float(npz["norm_floor"])
    dis = dk == "distractor"
    dims_d = np.asarray(npz["dims_d"], np.float64)
    norms_eff = np.maximum(raw_norms, floor)
    rec["floor"] = dict(
        distractor_raw_norm_mean=float(raw_norms[dis].mean()),
        norm_floor=floor,
        floor_binds=bool((raw_norms[dis] < floor).any()),
        floor_margin=float(raw_norms[dis].min() / floor))
    rec["unnorm_distractor_var"] = float(
        (dims_d[dis] * norms_eff[dis] ** 2).mean())
    cov, cov_steps, cov_flag = coverage_proxy(run_dir)
    rec["coverage"], rec["coverage_steps"], rec["coverage_flag"] = \
        cov, cov_steps, cov_flag
    rec["score"], rec["n_score_entries"] = score_tail(run_dir)
    return rec


def _logistic_fit(xs, ys):
    """Descriptive grid MLE of P(theta1>1) vs scale, both slope signs
    (rev M7): boundary/window flags + run-resample percentile interval
    for the midpoint (rev M6). Never a fire."""
    def fit(x, y):
        logits = LOGISTIC_K[:, None, None] * (
            x[None, None, :] - LOGISTIC_M[None, :, None])
        p = np.clip(1 / (1 + np.exp(-logits)), 1e-9, 1 - 1e-9)
        nll = -(y[None, None, :] * np.log(p)
                + (1 - y[None, None, :]) * np.log(1 - p)).sum(-1)
        ki, mi = np.unravel_index(np.argmin(nll), nll.shape)
        return float(LOGISTIC_M[mi]), float(LOGISTIC_K[ki])
    m, k = fit(xs, ys)
    rng = np.random.default_rng(20260822)
    boots = []
    for _ in range(N_BOOT_MID):
        idx = rng.integers(0, xs.size, xs.size)
        yb = ys[idx]
        if 0 < yb.sum() < yb.size:
            boots.append(fit(xs[idx], yb)[0])
    ci = ([float(np.percentile(boots, 2.5)),
           float(np.percentile(boots, 97.5))] if len(boots) >= 50
          else None)
    return dict(
        midpoint=m, slope=k,
        monotone_direction="rising" if k > 0 else "falling",
        boundary_pinned=bool(
            abs(m - LOGISTIC_M[0]) < 1e-12
            or abs(m - LOGISTIC_M[-1]) < 1e-12
            or abs(abs(k) - 200.0) < 1e-9 or abs(abs(k) - 5.0) < 1e-9),
        in_data_window=bool(
            DATA_WINDOW[0] - 1e-9 <= m <= DATA_WINDOW[1] + 1e-9),
        midpoint_pctl95=ci, n_boot_kept=len(boots),
        note="descriptive grid MLE + run-resample percentile "
             "interval; no fire")


def aggregate(recs, ref, n_perm_labels=N_PERM_LABELS):
    seeds = sorted(r["train_seed_cfg"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert set(seeds) <= ALL_SEEDS, f"seeds {seeds} outside 96-111"
    missing = sorted(ALL_SEEDS - set(seeds))
    out = dict(stamp=_stamp(), n_loaded=len(recs),
               missing_seeds=missing,
               stage1_reference=ref, anchors_descriptive=ANCHORS)

    collapsed = [r["run_dir"] for r in recs
                 if r["levels"]["real_mean"]
                 < ref["real_mean_min"] / COLLAPSE_FACTOR
                 or r["levels"]["intrinsic_mean"]
                 < ref["intrinsic_min"] / COLLAPSE_FACTOR]
    out["collapsed_runs"] = collapsed
    usable = [r for r in recs
              if r["valid"] and r["run_dir"] not in collapsed]
    per_scale_n = {s: sum(1 for r in usable if r["scale"] == s)
                   for s in SCALES}
    flagged = len(usable) < len(ALL_SEEDS)
    out["usable"] = dict(n=len(usable), per_scale=per_scale_n,
                         flagged=flagged)
    unverified = [r["run_dir"] for r in recs
                  if not r["newest_ckpt_checked"]
                  or r.get("ckpt_final_ok") is not True]
    out["probe_ckpt_flag"] = ("OK" if not unverified
                              else f"UNVERIFIED ({len(unverified)} runs)")

    # unconditional reporting
    out["per_run"] = [
        {"run_dir": r["run_dir"], "scale": r["scale"],
         "train_seed": r["train_seed_cfg"], "valid": r["valid"],
         "theta1": r["channels"]["distractor"]["vs_source"],
         "p_perm": r["channels"]["distractor"]["p_perm"],
         "floor": r["floor"],
         "unnorm_distractor_var": r["unnorm_distractor_var"],
         "coverage": r["coverage"], "coverage_flag": r["coverage_flag"],
         "score_tail": r["score"],
         "ckpt_final_ok": r.get("ckpt_final_ok"),
         "newest_ckpt_checked": r["newest_ckpt_checked"],
         "real_mean_level": r["levels"]["real_mean"],
         "intrinsic_mean": r["levels"]["intrinsic_mean"]}
        for r in recs]
    per_scale = {}
    for s in SCALES:
        rows = [r for r in usable if r["scale"] == s]
        th = [r["channels"]["distractor"]["vs_source"] for r in rows]
        row = dict(theta1=th,
                   crossing_frac=(sum(1 for x in th if x > 1.0)
                                  / len(th)) if th else None,
                   coverage=[r["coverage"] for r in rows],
                   score_tail=[r["score"] for r in rows])
        if len(th) >= 3:
            m, lo, hi = bca_interval(
                np.asarray(th),
                np.random.default_rng(int(round(s * 1000))), 2000)
            row["bca_indicative"] = [m, lo, hi]
        per_scale[str(s)] = row
    out["per_scale"] = per_scale
    # descriptive logistic midpoint of theta1>1 vs scale (never fires)
    xs = np.asarray([r["scale"] for r in usable], np.float64)
    ys = np.asarray([1.0 if r["channels"]["distractor"]["vs_source"]
                     > 1.0 else 0.0 for r in usable])
    out["logistic_midpoint"] = (_logistic_fit(xs, ys)
                                if 0 < ys.sum() < ys.size else None)
    # descriptive cost curve (rev M8): coverage/score live in
    # per_scale rows; Spearman(coverage, scale) reported with the
    # helper's native LEFT-tail p (B1's registered coverage trend was
    # negative; descriptive either way, never a fire)
    cov_pairs = [(r["scale"], r["coverage"]) for r in usable
                 if r["coverage"] is not None]
    cost = dict(n=len(cov_pairs))
    if len(cov_pairs) >= 8:
        cx = np.asarray([c for _, c in cov_pairs], np.float64)
        sx = np.asarray([s for s, _ in cov_pairs], np.float64)
        rho_c, p_left = spearman_perm_null(cx, sx, n_perm_labels,
                                           PERM_SEED)
        cost["spearman_coverage_scale"] = float(rho_c)
        cost["p_left_tail_descriptive"] = float(p_left)
    cost["note"] = ("descriptive; coverage proxy per B1 convention; "
                    "left-tail p from the shared helper")
    out["cost_curve"] = cost

    # degeneracy / partial adjudication
    if len(collapsed) >= 2:
        out["primary"] = ("GLOBAL-COLLAPSE: >= 2 collapsed runs — "
                          "trend not adjudicable")
        out["outcome_cell"] = "GLOBAL-COLLAPSE"
        return out
    if len(usable) < MIN_USABLE or \
            any(v < MIN_PER_SCALE for v in per_scale_n.values()):
        out["primary"] = (f"NOT-ADJUDICABLE (usable {len(usable)} < "
                          f"{MIN_USABLE} or a scale below "
                          f"{MIN_PER_SCALE})")
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        return out

    theta = np.asarray([r["channels"]["distractor"]["vs_source"]
                        for r in usable])
    scal = np.asarray([r["scale"] for r in usable])
    rho, p = spearman_perm_null(theta, scal, n_perm_labels, PERM_SEED)
    if not np.isfinite(rho):
        out["primary"] = ("NOT-ADJUDICABLE: degenerate Spearman "
                          "(constant theta_1 panel)")
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        return out
    # se_dose's spearman_perm_null tests the NEGATIVE direction
    # (p = P(null <= rho)); the onset primary is POSITIVE, so the
    # one-sided p is taken on the mirrored statistic
    rho2, p_pos = spearman_perm_null(-theta, scal, n_perm_labels,
                                     PERM_SEED)
    assert abs(rho2 + rho) < 1e-12, (
        f"mirror identity violated: rho={rho} rho2={rho2}")
    fires = bool(p_pos <= ALPHA and rho > 0)
    # floor scope = ALL LOADED runs (prereg: "any floored run" —
    # rev m12)
    floored = [r["run_dir"] for r in recs if r["floor"]["floor_binds"]]
    out["floored_runs"] = floored
    if fires and floored:
        # FLIPPED floor rule (prereg §2): a binding floor deflates
        # low-scale theta_1 — anti-conservative for a POSITIVE trend
        out["primary"] = (f"NOT-ADJUDICABLE: FIRE with binding "
                          f"normalizer floor on {len(floored)} runs "
                          f"(anti-conservative direction)")
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        return out
    statement = ("theta_1(distractor) INCREASES in scale across the "
                 "fresh onset window (RELATIVE estimand; rising limb "
                 "only — the SHAPE stays descriptive; Spearman > 0, "
                 "one-sided positive label permutation, "
                 f"{n_perm_labels} draws, seed {PERM_SEED})")
    if flagged:
        statement += (f"; FLAGGED SUBSET (n={len(usable)}, missing "
                      f"seeds {missing})")
    out["primary"] = dict(statement=statement, rho=float(rho),
                          p=float(p_pos), n=len(usable), fires=fires)
    out["outcome_cell"] = ("ONSET-CONFIRMED" if fires
                           else "ONSET-NOT-CONFIRMED")
    return out


def run(args):
    def expand(pat):
        return (sorted(globmod.glob(pat)) if "," not in pat
                else [x.strip() for x in pat.split(",") if x.strip()])

    dirs = expand(args.runs)
    s1 = expand(args.stage1_runs)
    assert dirs, f"no runs matched {args.runs!r}"
    assert s1, "degeneracy gate requires --stage1_runs"
    assert args.output and args.output != "TBD", (
        "registered read must persist: pass an explicit --output")
    out_json = os.path.join(args.output, "se_onset_read.json")
    assert not os.path.exists(out_json), (
        f"{out_json} exists — the read is ONE execution")
    ref = stage1_reference_gated(s1)
    # identity gates are FATAL (B1 pattern, rev B3): a gate failure
    # refuses the read; the operator repairs the run (or re-probes)
    # and re-executes — the one-execution guard survives because the
    # refusal happens before out_json is written
    recs = [read_run(d, args.n_perm) for d in dirs]
    out = aggregate(recs, ref, args.n_perm_labels)
    out["args"] = {k: v for k, v in vars(args).items()
                   if k != "selfcheck"}
    out["fit_counters"] = {r["run_dir"]: fit_counters(
        r["run_dir"], args.expect_steps) for r in recs}
    bad_fit = [rd for rd, fc in out["fit_counters"].items()
               if fc.get("flag") != "OK"]
    out["fit_flag"] = ("OK" if not bad_fit
                       else f"SUSPECT ({len(bad_fit)} runs)")
    if bad_fit and isinstance(out["primary"], dict):
        # registered disclosure-carried rule (rev m21): truncation
        # suspicion annotates the primary, never silently
        out["primary"]["statement"] += (
            f"; FIT-SUSPECT on {len(bad_fit)} runs (disclosure "
            f"carried, 7-Aug standing rule)")
    os.makedirs(args.output, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    if isinstance(out["primary"], dict):
        pr = out["primary"]
        print(f"ONSET PRIMARY: rho={pr['rho']:+.3f} p={pr['p']:.5f} "
              f"fires={pr['fires']} -> {out['outcome_cell']}; "
              f"fit {out['fit_flag']}; ckpt {out['probe_ckpt_flag']}")
        for s in SCALES:
            row = out["per_scale"][str(s)]
            print(f"  scale {s}: theta1 "
                  f"{[round(x, 3) for x in row['theta1']]} "
                  f"crossing {row['crossing_frac']}")
    else:
        print(f"ONSET PRIMARY: {out['outcome_cell']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, scale, theta, level_scale=1.0,
             intrinsic=3.5e-4, floor_binds=False):
    """Onset fixture on the reviewed dose-fixture pattern; the dose
    fixture hardcodes a non-binding floor, so a binding floor is
    doctored into the npz here (raw distractor norms pushed below
    norm_floor — the floor fields are what the reader consumes; the
    theta provenance rides dims_d and is untouched)."""
    from uncfield import se_dose_read as dr
    rd = dr._fixture(tmp, name, seed, dose=scale, theta=theta,
                     level_scale=level_scale, intrinsic=intrinsic)
    if floor_binds:
        pth = os.path.join(rd, "se_probe", "se_probe_dims.npz")
        npz = dict(np.load(pth))
        dk = np.asarray(npz["dim_key"]).astype(str)
        raw = np.asarray(npz["raw_norms"], np.float64)
        raw[dk == "distractor"] = 0.01          # < floor 0.05
        npz["raw_norms"] = raw
        np.savez(pth, **npz)
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
    from uncfield import se_dose_read as dr
    NP = se_read.FIX_PERM
    NPL = 2000

    def theta_rising(scale, seed):
        return {0.25: 0.3, 0.30: 1.2, 0.34: 5.0,
                0.38: 7.0}[scale] * (1 + 0.05 * (seed % 4))

    def theta_falling(scale, seed):
        return {0.25: 7.0, 0.30: 5.0, 0.34: 1.2,
                0.38: 0.3}[scale] * (1 + 0.05 * (seed % 4))

    def theta_flat(scale, seed):
        return 3.0 * (1 + 0.05 * (seed % 4))

    with tempfile.TemporaryDirectory() as tmp:
        s1 = []
        for i in range(4):
            d = dr._fixture(tmp, f"st{i}", 10 + i, dose=1.0,
                            theta=5.0)
            s1.append(d)
        ref = stage1_reference_gated(s1)

        def wave(sub, theta_of, **kw):
            runs = []
            for s, seeds in SCALE_SEEDS.items():
                for sd in sorted(seeds):
                    runs.append(_fixture(tmp, f"{sub}{sd}", sd, s,
                                         theta_of(s, sd), **kw))
            return runs

        # ONSET-CONFIRMED
        runs = wave("a", theta_rising)
        recs = [read_run(d, NP) for d in runs]
        agg = aggregate(recs, ref, NPL)
        assert agg["outcome_cell"] == "ONSET-CONFIRMED", agg["primary"]
        assert agg["primary"]["rho"] > 0.8
        assert agg["missing_seeds"] == []
        assert "FLAGGED" not in agg["primary"]["statement"]
        assert agg["per_scale"]["0.25"]["crossing_frac"] == 0.0
        assert agg["per_scale"]["0.38"]["crossing_frac"] == 1.0
        assert len(agg["per_scale"]["0.3"]["coverage"]) == 4
        mid = agg["logistic_midpoint"]
        assert mid is not None
        assert mid["monotone_direction"] == "rising"
        assert mid["in_data_window"], mid
        assert 0.24 <= mid["midpoint"] <= 0.36, mid

        # REVERSED crossing pattern fits as FALLING, not a boundary
        # pin (rev M7)
        runs = wave("r", theta_falling)
        agg = aggregate([read_run(d, NP) for d in runs], ref, NPL)
        assert agg["outcome_cell"] == "ONSET-NOT-CONFIRMED"
        mid = agg["logistic_midpoint"]
        assert mid["monotone_direction"] == "falling", mid
        assert mid["in_data_window"], mid

        # flat -> NOT-CONFIRMED
        runs = wave("b", theta_flat)
        agg = aggregate([read_run(d, NP) for d in runs], ref, NPL)
        assert agg["outcome_cell"] == "ONSET-NOT-CONFIRMED"

        # FLIPPED floor rule: rising trend + a floored low-scale run
        # -> the would-be FIRE is NOT-ADJUDICABLE
        runs = []
        for s, seeds in SCALE_SEEDS.items():
            for sd in sorted(seeds):
                runs.append(_fixture(tmp, f"c{sd}", sd, s,
                                     theta_rising(s, sd),
                                     floor_binds=(sd == 96)))
        agg = aggregate([read_run(d, NP) for d in runs], ref, NPL)
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE", (
            agg["outcome_cell"])
        assert "anti-conservative" in agg["primary"]
        # ...but a FLAT (non-fire) outcome with the same floored run
        # stands as registered
        runs = []
        for s, seeds in SCALE_SEEDS.items():
            for sd in sorted(seeds):
                runs.append(_fixture(tmp, f"d{sd}", sd, s,
                                     theta_flat(s, sd),
                                     floor_binds=(sd == 96)))
        agg = aggregate([read_run(d, NP) for d in runs], ref, NPL)
        assert agg["outcome_cell"] == "ONSET-NOT-CONFIRMED"

        # degeneracy + partial paths
        runs = wave("e", theta_rising, level_scale=1e-6,
                    intrinsic=1e-9)
        agg = aggregate([read_run(d, NP) for d in runs], ref, NPL)
        assert agg["outcome_cell"] == "GLOBAL-COLLAPSE"
        runs = wave("f", theta_rising)
        agg = aggregate([read_run(d, NP) for d in runs][:13], ref, NPL)
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE"
        assert agg["missing_seeds"], agg["missing_seeds"]

        # missing-seed FLAG on a 15-run fire path (rev M5): drop one
        # 0.38 run — per-scale minimum still met, primary flagged
        runs = wave("h", theta_rising)
        recs15 = [read_run(d, NP) for d in runs
                  if not d.endswith("h111")]
        agg = aggregate(recs15, ref, NPL)
        assert agg["outcome_cell"] == "ONSET-CONFIRMED"
        assert agg["missing_seeds"] == [111]
        assert "FLAGGED SUBSET" in agg["primary"]["statement"]

        # identity gates
        rd = _fixture(tmp, "g1", 96, 0.25, 2.0)
        cfgp = os.path.join(rd, "config.yaml")
        cfg = open(cfgp).read()
        open(cfgp, "w").write(cfg.replace("scale: 0.25",
                                          "scale: 0.33"))
        _expect_fail(lambda: read_run(rd, NP), "registered grid")
        rd = _fixture(tmp, "g2", 100, 0.25, 2.0)   # seed of 0.30
        _expect_fail(lambda: read_run(rd, NP), "not registered for")
        rd = _fixture(tmp, "g3", 96, 0.25, 2.0)
        cfg = open(os.path.join(rd, "config.yaml")).read()
        open(os.path.join(rd, "config.yaml"), "w").write(
            cfg.replace("disag_bootstrap: true",
                        "disag_bootstrap: false"))
        _expect_fail(lambda: read_run(rd, NP), "expl pins")
        # penalty-inertness gate exercised POSITIVELY (rev m10a)
        rd = _fixture(tmp, "g4", 97, 0.25, 2.0)
        cfg = open(os.path.join(rd, "config.yaml")).read()
        assert "penalty_mix" not in cfg  # fixture default: absent
        open(os.path.join(rd, "config.yaml"), "w").write(
            cfg.replace("mode: p2e", "mode: p2e\n    penalty_mix: true"))
        _expect_fail(lambda: read_run(rd, NP), "penalty machinery")
        # stale ckpt/latest pointer (rev m10b): a second, newer ckpt
        # dir appears but the probe recorded the older one
        rd = _fixture(tmp, "g5", 98, 0.25, 2.0)
        ck = os.path.join(rd, "ckpt")
        subs = sorted(x for x in os.listdir(ck)
                      if os.path.isdir(os.path.join(ck, x)))
        assert subs, "fixture must carry a ckpt dir"
        os.makedirs(os.path.join(ck, "zz_newer"))
        _expect_fail(lambda: read_run(rd, NP), "stale ckpt/latest")

        # end-to-end + one-execution guard
        outd = os.path.join(tmp, "out")
        out = run(parse_args([
            "--runs", os.path.join(tmp, "a[0-9]*"),
            "--stage1_runs", os.path.join(tmp, "st*"),
            "--n_perm", str(NP), "--n_perm_labels", str(NPL),
            "--output", outd]))
        assert out["outcome_cell"] == "ONSET-CONFIRMED"
        assert "args" in out and out["args"]["n_perm_labels"] == NPL
        _expect_fail(lambda: run(parse_args([
            "--runs", os.path.join(tmp, "a[0-9]*"),
            "--stage1_runs", os.path.join(tmp, "st*"),
            "--n_perm", str(NP), "--n_perm_labels", str(NPL),
            "--output", outd])), "ONE execution")
        # FATAL gate path (rev B3): a defective run in the glob
        # refuses the whole read instead of silently dropping
        rd = _fixture(tmp, "a999", 99, 0.25, 2.0)
        cfgp = os.path.join(rd, "config.yaml")
        cfg = open(cfgp).read()
        open(cfgp, "w").write(cfg.replace("scale: 0.25",
                                          "scale: 0.33"))
        _expect_fail(lambda: run(parse_args([
            "--runs", os.path.join(tmp, "a[0-9]*"),
            "--stage1_runs", os.path.join(tmp, "st*"),
            "--n_perm", str(NP), "--n_perm_labels", str(NPL),
            "--output", os.path.join(tmp, "out2")])),
            "registered grid")
    print("se_onset_read selfcheck PASS (ONSET-CONFIRMED w/ correct "
          "crossing fractions + in-window rising midpoint; REVERSED "
          "pattern fits falling; flat NOT-CONFIRMED; FLIPPED floor "
          "rule both directions; GLOBAL-COLLAPSE; partial "
          "NOT-ADJUDICABLE; missing-seed FLAG; grid/seed-map/pin/"
          "penalty/stale-ckpt gates; FATAL gate path; end-to-end + "
          "one-execution guard)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
