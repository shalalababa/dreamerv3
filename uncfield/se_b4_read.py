"""Track-B4 (aleatoric repair) FROZEN reader
(PREREG_trackB_alea_20260822.md §2; built + selfchecked BEFORE the
wave's compute; ONE execution; revised pre-freeze per review R3 —
the first-draft ratio R = delta_norm/delta_raw was demonstrated to
fire REPAIRED on three non-repairs: a constant rescaling, a
sign-flipping over-correction, and a statistic-death flat reward).

4 runs, seeds 120-123, Stage-1 config with DISAG_HEAD=gauss (mu
trained by the SAME MSE with the logvar head off a STOP-GRADIENTED
trunk — det-identical loss AND gradient path; sigma by NLL on
stop-gradiented residuals; deployed reward = aleatoric-normalized
disagreement). Instruments per run: se_probe (theta_1 consistency
row + degeneracy levels) and se_alea_mask (BOTH statistics on
identical masked variants + gauss diagnostics).

Read order:
  0. Validity: pins (disag_head gauss + logvar bounds + Stage-1
     rest + penalty inert), seeds, json==npz float64 provenance,
     dup0 zero on both statistics, final-ckpt + newest-ckpt gates
     for BOTH instruments cross-checked (rev R3-M5), instrument
     run-dir identity (rev R3-N5), S pin; defective -> excluded,
     read persists. FULL WAVE FIRST (rev R3-D2): any outcome cell
     needs all 4 runs loaded+valid.
  1. DEGENERACY vs --stage1_runs: real leg = se_probe real level;
     intrinsic leg = the mask's base_raw (mean var-of-mu — the
     det-commensurable quantity; the pse2 intrinsic under gauss is
     the normalized RATIO, ~1000x off the det scale, rev R3-B4).
     Collapse -> REPAIR-BY-DEATH.
  2. MANIPULATION CHECK: delta_raw(distractor) < 0 with its BCa
     upper bound < 0 (rev R3-D1) in 4/4. Fail ->
     RAW-MISPRICE-ABSENT.
  3. ALIVENESS GATE (rev R3-B3): the normalized statistic must have
     teeth — per run: (velocity_control delta_norm BCa excludes 0
     AND sign-matches its raw delta when the raw delta is itself
     distinguishable) AND anchor-dispersion floor
     cv(base_norm) >= 0.25 x cv(base_raw). Fail in any run ->
     REPAIR-BY-FLATTENING (a constant/flat reward trivially zeroes
     every delta — that is not repair).
  4. PRIMARY (scale-free, sign-constrained — rev R3-B1/B2):
     r_raw = delta_raw/base_raw, r_norm = delta_norm/base_norm,
     R_rel = r_norm/r_raw. Any run with delta_norm > 0 ->
     OVER-CORRECTED (its own cell; sign inversion is not repair).
     REPAIRED iff 0 <= R_rel < 0.5 in 4/4 (under a pure constant
     rescaling R_rel == 1 -> correctly no fire). PARTIAL-REPAIR
     otherwise. Per-run paired anchor bootstrap of R_rel reported
     (rev R3-D1). Power note: a 4/4 unanimity rule has null rate
     ~1/16 and one discordant run demotes with no gradation (rev
     R3-N8); all comparisons are within-run on identical variants,
     so the 1-Aug cross-invocation rule does not bind.
  5. Secondaries (no fire): velocity control on both statistics,
     dup + resample rows, theta_1 rows, gauss diagnostics (logvar
     bound fractions, alea level, deter/stoch reward re-weighting),
     coverage + score tails, base levels. Fit counters reported;
     non-OK ANNOTATES the primary (disclosure-carried, rev R3-D3).

Run:  python -m uncfield.se_b4_read --runs "<glob>"
          --stage1_runs "<glob>" --output <dir>
      python -m uncfield.se_b4_read --selfcheck
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
from uncfield.se_noboot_read import COLLAPSE_FACTOR, _levels
from uncfield.se_dose_read import coverage_proxy, score_tail
from uncfield.se_read import fit_counters
from uncfield import se_read

B4_SEEDS = (120, 121, 122, 123)
TASK_PIN = "dmc_cheetah_run"
REPAIR_BAR = 0.5
S_PIN = 512
PROV_RTOL = 1e-12
LOGVAR_MIN_PIN = -8.0
LOGVAR_MAX_PIN = 6.0
CV_FLOOR_FRAC = 0.25
N_BOOT_RREL = 2000


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="")
    p.add_argument("--stage1_runs", default="")
    p.add_argument("--n_perm", type=int, default=1000)
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
    pl = cfg["planted"]
    ex = cfg["agent"]["expl"]
    train_seed = int(cfg["seed"])
    assert train_seed in B4_SEEDS, (
        f"{run_dir}: seed {train_seed} outside the registered "
        f"120-123")
    assert str(cfg["task"]) == TASK_PIN, run_dir
    assert str(ex.get("disag_head", "det")) == "gauss", (
        f"{run_dir}: disag_head={ex.get('disag_head')!r} — the "
        f"gauss head did not bind")
    assert abs(float(ex["disag_logvar_min"]) - LOGVAR_MIN_PIN) < 1e-9 \
        and abs(float(ex["disag_logvar_max"]) - LOGVAR_MAX_PIN) \
        < 1e-9, (
        f"{run_dir}: logvar bounds differ from the registered "
        f"[-8, 6] — the repair strength is set by these (rev R3-M3)")
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

    rec = se_read.read_run(run_dir, "se_probe", n_perm)
    rec["train_seed_cfg"] = train_seed
    rec["levels"] = _levels(run_dir)
    # newest-ckpt cross-check, BOTH instruments (rev R3-M5)
    ckroot = os.path.join(run_dir, "ckpt")
    try:
        subs = sorted(x for x in os.listdir(ckroot)
                      if os.path.isdir(os.path.join(ckroot, x)))
    except OSError:
        subs = []
    with open(os.path.join(run_dir, "se_probe",
                           "se_probe.json")) as f:
        probed = os.path.basename(
            os.path.normpath(json.load(f)["ckpt"]))

    mdir = os.path.join(run_dir, "se_alea_mask")
    with open(os.path.join(mdir, "se_alea_mask.json")) as f:
        mj = json.load(f)
    npz = np.load(os.path.join(mdir, "se_alea_mask.npz"))
    mask_ck = os.path.basename(os.path.normpath(mj["ckpt"]))
    assert mask_ck == probed, (
        f"{run_dir}: se_probe ckpt {probed} != se_alea_mask ckpt "
        f"{mask_ck} — the two instruments measured different "
        f"checkpoints (rev R3-M5)")
    if subs:
        assert probed == subs[-1], (
            f"{run_dir}: probed ckpt {probed} != newest {subs[-1]}")
    assert os.path.basename(os.path.normpath(mj["run_logdir"])) == \
        os.path.basename(os.path.normpath(run_dir)), (
        f"{run_dir}: mask json belongs to {mj['run_logdir']} "
        f"(rev R3-N5)")
    assert int(mj["n_eval"]) == S_PIN, (
        f"{run_dir}: mask S {mj['n_eval']} != {S_PIN}")
    assert str(mj["disag_head"]) == "gauss"
    assert int(mj["train_seed"]) == train_seed
    gd = mj["gauss_diagnostics"]
    assert abs(float(gd["logvar_min"]) - LOGVAR_MIN_PIN) < 1e-9 and \
        abs(float(gd["logvar_max"]) - LOGVAR_MAX_PIN) < 1e-9, (
        f"{run_dir}: instrument saw logvar bounds "
        f"[{gd['logvar_min']}, {gd['logvar_max']}]")
    assert mj["dup0_bitwise_equal_source"], (
        f"{run_dir}: dup0 not bitwise-equal to source")
    c0 = mj["channels"]["planted_dup0"]
    assert c0["delta_raw_mean"] == 0.0 and \
        c0["delta_norm_mean"] == 0.0, (
        f"{run_dir}: dup0 not the exact-zero anchor on both stats")
    for ch, crec in mj["channels"].items():
        for stat in ("raw", "norm"):
            dvec = np.asarray(npz[f"delta_{stat}_{ch}"], np.float64)
            assert dvec.size == S_PIN
            assert np.all(np.isfinite(dvec)), (run_dir, ch, stat)
            rc = float(dvec.mean())
            stored = float(crec[f"delta_{stat}_mean"])
            assert abs(rc - stored) <= PROV_RTOL * max(1.0, abs(rc)), (
                f"{run_dir}/{ch}/{stat}: provenance mismatch")
    assert "velocity_control" in mj["channels"], run_dir
    rec["mask"] = mj["channels"]
    rec["gauss_diag"] = gd
    rec["arrays"] = dict(
        dr=np.asarray(npz["delta_raw_distractor"], np.float64),
        dn=np.asarray(npz["delta_norm_distractor"], np.float64),
        br=np.asarray(npz["base_raw"], np.float64),
        bn=np.asarray(npz["base_norm"], np.float64))
    cov, cov_steps, cov_flag = coverage_proxy(run_dir)
    rec["coverage"], rec["coverage_flag"] = cov, cov_flag
    rec["score"], _ = score_tail(run_dir)
    rec["base_raw"] = float(mj["base_raw_mean"])
    rec["base_norm"] = float(mj["base_norm_mean"])
    return rec


def _cv(x):
    m = float(np.mean(x))
    return float(np.std(x) / m) if m > 0 else 0.0


def _rrel_boot(a, rng, n=N_BOOT_RREL):
    S = a["dr"].size
    vals = []
    for _ in range(n):
        idx = rng.integers(0, S, S)
        drm, dnm = a["dr"][idx].mean(), a["dn"][idx].mean()
        brm, bnm = a["br"][idx].mean(), a["bn"][idx].mean()
        if drm < 0 and brm > 0 and bnm > 0:
            vals.append((dnm / bnm) / (drm / brm))
    if len(vals) < n // 2:
        return None
    return [float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)), len(vals)]


def aggregate(recs, excluded, ref):
    seeds = sorted(r["train_seed_cfg"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert len(recs) + len(excluded) <= 4
    out = dict(stamp=_stamp(), n_loaded=len(recs),
               excluded_runs=excluded, stage1_reference=ref)
    out["per_run"] = [
        {"run_dir": r["run_dir"], "train_seed": r["train_seed_cfg"],
         "valid": r["valid"],
         "delta_raw_distractor":
             r["mask"]["distractor"]["delta_raw_mean"],
         "delta_norm_distractor":
             r["mask"]["distractor"]["delta_norm_mean"],
         "velocity_raw": r["mask"]["velocity_control"]
         ["delta_raw_mean"],
         "velocity_norm": r["mask"]["velocity_control"]
         ["delta_norm_mean"],
         "theta1_distractor":
             r["channels"]["distractor"]["vs_source"],
         "base_raw": r["base_raw"], "base_norm": r["base_norm"],
         "gauss_diag": r["gauss_diag"],
         "coverage": r["coverage"], "score_tail": r["score"],
         "real_mean_level": r["levels"]["real_mean"],
         "intrinsic_pse2_ratio_scale":
             r["levels"]["intrinsic_mean"]}
        for r in recs]

    # FULL WAVE FIRST (rev R3-D2): no cell from a partial wave
    valid = [r for r in recs if r["valid"]]
    if len(recs) < 4 or len(valid) < 4:
        out["primary"] = (f"NOT-ADJUDICABLE ({len(valid)}/4 valid "
                          f"of {len(recs)} loaded; the 4/4 rules "
                          f"need the full wave)")
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        return out

    # 1) degeneracy: real leg from se_probe levels; intrinsic leg
    # from base_raw (det-commensurable — rev R3-B4)
    collapsed = [r["run_dir"] for r in valid
                 if r["levels"]["real_mean"]
                 < ref["real_mean_min"] / COLLAPSE_FACTOR
                 or r["base_raw"]
                 < ref["intrinsic_min"] / COLLAPSE_FACTOR]
    out["collapsed_runs"] = collapsed
    if collapsed:
        out["primary"] = ("REPAIR-BY-DEATH: levels collapsed below "
                          "1/10 of the Stage-1 minima (real level "
                          "or var-of-mu) — the gauss arm did not "
                          "preserve the substrate; the repair ratio "
                          "is not adjudicable")
        out["outcome_cell"] = "REPAIR-BY-DEATH"
        return out

    d_raw = [r["mask"]["distractor"]["delta_raw_mean"]
             for r in valid]
    d_norm = [r["mask"]["distractor"]["delta_norm_mean"]
              for r in valid]
    # 2) manipulation check with interval teeth (rev R3-D1)
    raw_ok = all(x < 0 for x in d_raw) and all(
        r["mask"]["distractor"]["delta_raw_bca"][1] < 0
        for r in valid)
    if not raw_ok:
        out["primary"] = dict(
            statement="RAW-MISPRICE-ABSENT: the raw var-of-mu "
                      "statistic does not price the distractor "
                      "(point < 0 AND BCa hi < 0) in 4/4 — a "
                      "different physical result (identical mu "
                      "LOSS does not guarantee identical data "
                      "distribution — the actor followed a "
                      "different reward); the repair ratio is not "
                      "adjudicable",
            delta_raw=d_raw,
            delta_raw_bca=[r["mask"]["distractor"]["delta_raw_bca"]
                           for r in valid],
            delta_norm=d_norm)
        out["outcome_cell"] = "RAW-MISPRICE-ABSENT"
        return out

    # 3) aliveness gate (rev R3-B3): the norm statistic must have
    # teeth — else a flat reward trivially "repairs"
    alive_rows = []
    for r in valid:
        v = r["mask"]["velocity_control"]
        vb = v["delta_norm_bca"]
        v_norm_distinct = vb[0] > 0 or vb[1] < 0
        rb = v["delta_raw_bca"]
        v_raw_distinct = rb[0] > 0 or rb[1] < 0
        sign_ok = (not v_raw_distinct) or (
            np.sign(v["delta_norm_mean"])
            == np.sign(v["delta_raw_mean"]))
        cv_ok = _cv(r["arrays"]["bn"]) >= CV_FLOOR_FRAC * _cv(
            r["arrays"]["br"])
        alive_rows.append(dict(
            run_dir=r["run_dir"],
            velocity_norm_distinct=bool(v_norm_distinct),
            velocity_sign_ok=bool(sign_ok),
            cv_norm=_cv(r["arrays"]["bn"]),
            cv_raw=_cv(r["arrays"]["br"]), cv_ok=bool(cv_ok),
            alive=bool(v_norm_distinct and sign_ok and cv_ok)))
    out["aliveness"] = alive_rows
    if not all(a["alive"] for a in alive_rows):
        out["primary"] = (
            "REPAIR-BY-FLATTENING: the normalized statistic fails "
            "the aliveness gate (velocity control indistinct / "
            "sign-inverted, or anchor dispersion collapsed) — a "
            "flat reward zeroes every delta without repairing "
            "anything")
        out["outcome_cell"] = "REPAIR-BY-FLATTENING"
        return out

    # 4) PRIMARY: scale-free, sign-constrained relative retention
    r_raw = [dr / r["base_raw"] for dr, r in zip(d_raw, valid)]
    r_norm = [dn / r["base_norm"] for dn, r in zip(d_norm, valid)]
    over = [r["run_dir"] for dn, r in zip(d_norm, valid) if dn > 0]
    rrel = [rn / rr for rn, rr in zip(r_norm, r_raw)]
    boots = [_rrel_boot(r["arrays"],
                        np.random.default_rng(r["train_seed_cfg"]))
             for r in valid]
    out["primary"] = dict(
        statement="ALEATORIC REPAIR (scale-free): the deployed "
                  "normalized reward retains < 0.5x of the raw "
                  "statistic's RELATIVE priced distractor signal "
                  "(R_rel = (delta_norm/base_norm)/(delta_raw/"
                  "base_raw)) with no sign inversion, in 4/4 runs "
                  "(paired, same variants, one instrument pass). "
                  "Power note: 4/4 unanimity, null rate ~1/16, no "
                  "gradation",
        delta_raw=d_raw, delta_norm=d_norm,
        rel_raw=r_raw, rel_norm=r_norm, rrel=rrel,
        rrel_boot_pctl95=boots, bar=REPAIR_BAR,
        over_corrected_runs=over)
    if over:
        out["primary"]["fires"] = False
        out["outcome_cell"] = "OVER-CORRECTED"
    else:
        fires = bool(all(0.0 <= x < REPAIR_BAR for x in rrel))
        out["primary"]["fires"] = fires
        out["outcome_cell"] = ("REPAIRED" if fires
                               else "PARTIAL-REPAIR")
    out["velocity_teeth"] = dict(
        raw=[r["mask"]["velocity_control"]["delta_raw_mean"]
             for r in valid],
        norm=[r["mask"]["velocity_control"]["delta_norm_mean"]
              for r in valid],
        note="the real-coupling control should stay priced under "
             "BOTH statistics — enforced by the aliveness gate, "
             "reported here")
    out["theta1_consistency"] = [
        r["channels"]["distractor"]["vs_source"] for r in valid]
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
    out_json = os.path.join(args.output, "se_b4_read.json")
    assert not os.path.exists(out_json), (
        f"{out_json} exists — the read is ONE execution")
    from uncfield.se_noboot_read import stage1_reference
    ref = stage1_reference(s1)
    recs, excluded = [], []
    for d in dirs:
        try:
            recs.append(read_run(d, args.n_perm))
        except Exception as e:              # noqa: BLE001
            excluded.append(dict(run_dir=os.path.abspath(d),
                                 error=f"{type(e).__name__}: {e}"))
    out = aggregate(recs, excluded, ref)
    out["fit_counters"] = {r["run_dir"]: fit_counters(
        r["run_dir"], args.expect_steps) for r in recs}
    bad_fit = [rd for rd, fc in out["fit_counters"].items()
               if fc.get("flag") != "OK"]
    out["fit_flag"] = ("OK" if not bad_fit
                       else f"SUSPECT ({len(bad_fit)} runs)")
    if bad_fit and isinstance(out["primary"], dict) and \
            "statement" in out["primary"]:
        out["primary"]["statement"] += (
            f"; FIT-SUSPECT on {len(bad_fit)} runs (disclosure "
            f"carried, rev R3-D3)")
    os.makedirs(args.output, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    if isinstance(out["primary"], dict) and \
            "rrel" in out["primary"]:
        pr = out["primary"]
        print(f"B4 PRIMARY: R_rel "
              f"{[round(x, 3) for x in pr['rrel']]} "
              f"fires={pr['fires']} -> {out['outcome_cell']}; "
              f"fit {out['fit_flag']}")
    else:
        print(f"B4 PRIMARY: {out['outcome_cell']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, rel_raw=-0.01, rel_norm=-0.001,
             base_raw=3e-4, base_norm=0.25, hot=None,
             level_scale=1.0, base_raw_dead=None,
             vel_rel=(-0.02, -0.015), cv_norm_frac=1.0,
             head="gauss", dup0_zero=True, lv_bounds=(-8.0, 6.0),
             doctor=None, drop_npz=False):
    """Physically consistent fixture (rev R3-D6): deltas are given
    RELATIVE to their base (|delta| <= base always); npz base arrays
    carry controllable dispersion so the cv aliveness leg is
    exercisable."""
    rd = se_read._fixture_run(tmp, name, seed,
                              hot=(hot or {"distractor": 5.0}))
    pdir = os.path.join(rd, "se_probe")
    with open(os.path.join(pdir, "se_probe.json")) as f:
        pj = json.load(f)
    pj["pse2"]["intrinsic_mean"] = 0.25    # gauss ratio scale
    with open(os.path.join(pdir, "se_probe.json"), "w") as f:
        json.dump(pj, f)
    if level_scale != 1.0:
        npzp = os.path.join(pdir, "se_probe_dims.npz")
        z = dict(np.load(npzp))
        z["dims_d"] = z["dims_d"] * level_scale
        np.savez(npzp, **z)
    br = base_raw_dead if base_raw_dead is not None else base_raw
    mdir = os.path.join(rd, "se_alea_mask")
    os.makedirs(mdir, exist_ok=True)
    rng = np.random.default_rng(seed * 11 + 3)
    d_raw = rel_raw * br
    d_norm = rel_norm * base_norm
    chans, dnpz = {}, {}
    spec = {"distractor": (d_raw, d_norm),
            "planted_dup0": (0.0, 0.0),
            "planted_dup1": (-1e-4 * br / 3e-4, -1e-4),
            "planted_dup2": (1e-4 * br / 3e-4, 1e-4),
            "planted_dup1_resample": (1e-4 * br / 3e-4, 1e-4),
            "planted_dup2_resample": (-1e-4 * br / 3e-4, -1e-4),
            "velocity_control": (vel_rel[0] * br,
                                 vel_rel[1] * base_norm)}
    for ch, (wr, wn) in spec.items():
        rec = dict(form="x")
        for stat, want in (("raw", wr), ("norm", wn)):
            if ch == "planted_dup0" and dup0_zero:
                dv = np.zeros(S_PIN)
            else:
                dv = rng.normal(want, max(abs(want), 1e-6) * 0.1,
                                S_PIN)
                dv = dv - dv.mean() + want
            m = float(dv.mean())
            rec[f"delta_{stat}_mean"] = m
            hw = max(abs(m), 1e-6) * 0.05
            rec[f"delta_{stat}_bca"] = [m - hw, m + hw]
            rec[f"p_reduce_{stat}"] = 0.01
            dnpz[f"delta_{stat}_{ch}"] = dv.astype(np.float64)
        chans[ch] = rec
    base_raw_arr = np.maximum(
        rng.normal(br, 0.3 * br, S_PIN), br * 0.1)
    base_norm_arr = np.maximum(
        rng.normal(base_norm, 0.3 * base_norm * cv_norm_frac, S_PIN),
        base_norm * 0.01)
    with open(os.path.join(pdir, "se_probe.json")) as f:
        probe_ck = os.path.basename(
            os.path.normpath(json.load(f)["ckpt"]))
    mj = dict(run_logdir=rd, ckpt=os.path.join(rd, "ckpt", probe_ck),
              n_eval=S_PIN, seed=0, train_seed=seed, task=TASK_PIN,
              source_key="position", disag_head="gauss",
              dup0_bitwise_equal_source=dup0_zero,
              base_raw_mean=float(base_raw_arr.mean() * 0 + br),
              base_norm_mean=float(base_norm),
              gauss_diagnostics=dict(
                  logvar_min=lv_bounds[0], logvar_max=lv_bounds[1],
                  alea_mean=1.0, alea_std=0.2, lv_frac_lo=0.01,
                  lv_frac_hi=0.0, base_norm_deter_mean=0.3,
                  base_norm_stoch_mean=0.2),
              channels=chans)
    if doctor:
        doctor(mj)
    with open(os.path.join(mdir, "se_alea_mask.json"), "w") as f:
        json.dump(mj, f)
    if not drop_npz:
        np.savez(os.path.join(mdir, "se_alea_mask.npz"),
                 base_raw=base_raw_arr, base_norm=base_norm_arr,
                 **dnpz)
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(
            f"seed: {seed}\ntask: {TASK_PIN}\n"
            f"planted:\n  gate_key: ''\n  source_key: 'position'\n"
            f"  basesd: 0.0976\n"
            f"distractor:\n  gate_key: ''\n  mod_key: ''\n"
            f"  dim: {DIM}\n  scale: 1.0\n  basesd: {BASESD_N}\n"
            f"agent:\n  expl:\n    mode: p2e\n    disag_ens: 8\n"
            f"    disag_scale: 1000.0\n    disag_bootstrap: true\n"
            f"    disag_head: {head}\n"
            f"    disag_logvar_min: -8.0\n"
            f"    disag_logvar_max: 6.0\n"
            f"run:\n  steps: 500000.0\n")
    os.makedirs(os.path.join(rd, "replay"), exist_ok=True)
    r2 = np.random.default_rng(seed)
    np.savez(os.path.join(rd, "replay", "c0.npz"),
             position=r2.normal(0, 1, (120000, 8)).astype(np.float32),
             velocity=r2.normal(0, 1, (120000, 9)).astype(np.float32))
    with open(os.path.join(rd, "scores.jsonl"), "w") as f:
        for i in range(150):
            f.write(json.dumps({"step": i,
                                "episode/score": 10.0}) + "\n")
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
    from uncfield.se_noboot_read import stage1_reference
    NP = se_read.FIX_PERM
    with tempfile.TemporaryDirectory() as tmp:
        s1 = []
        for i in range(4):
            d = se_read._fixture_run(tmp, f"st{i}", 10 + i,
                                     hot={"distractor": 5.0})
            with open(os.path.join(d, "se_probe",
                                   "se_probe.json")) as f:
                pj = json.load(f)
            pj["pse2"]["intrinsic_mean"] = 3.5e-4
            with open(os.path.join(d, "se_probe",
                                   "se_probe.json"), "w") as f:
                json.dump(pj, f)
            s1.append(d)
        ref = stage1_reference(s1)

        # REPAIRED: relative retention 0.1, no inversion, alive
        runs = [_fixture(tmp, f"a{s}", s, rel_raw=-0.01,
                         rel_norm=-0.001) for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "REPAIRED", agg["primary"]
        assert all(0 <= x < 0.5 for x in agg["primary"]["rrel"])
        assert agg["primary"]["rrel_boot_pctl95"][0] is not None
        # PARTIAL-REPAIR: retention 0.8
        runs = [_fixture(tmp, f"b{s}", s, rel_raw=-0.01,
                         rel_norm=-0.008) for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "PARTIAL-REPAIR"
        # R3-B1 adversarial: CONSTANT RESCALING (same relative
        # misprice, bases differ 100x) must NOT fire — R_rel == 1
        runs = [_fixture(tmp, f"r{s}", s, rel_raw=-0.01,
                         rel_norm=-0.01) for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "PARTIAL-REPAIR", (
            agg["outcome_cell"])
        assert all(abs(x - 1.0) < 0.05 for x in agg["primary"]
                   ["rrel"])
        # R3-B2 adversarial: SIGN INVERSION -> OVER-CORRECTED
        runs = [_fixture(tmp, f"s{s}", s, rel_raw=-0.01,
                         rel_norm=+0.005) for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "OVER-CORRECTED", (
            agg["outcome_cell"])
        # R3-B3 adversarial: STATISTIC DEATH (velocity zeroed under
        # norm + collapsed dispersion) -> REPAIR-BY-FLATTENING
        runs = [_fixture(tmp, f"f{s}", s, rel_raw=-0.01,
                         rel_norm=-0.0001, vel_rel=(-0.02, 0.0),
                         cv_norm_frac=0.02) for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "REPAIR-BY-FLATTENING", (
            agg["outcome_cell"])
        # RAW-MISPRICE-ABSENT: raw positive
        runs = [_fixture(tmp, f"c{s}", s, rel_raw=0.004,
                         rel_norm=0.0) for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "RAW-MISPRICE-ABSENT"
        # REPAIR-BY-DEATH via the det-commensurable base_raw leg
        # (rev R3-B4)
        runs = [_fixture(tmp, f"d{s}", s, level_scale=1e-6,
                         base_raw_dead=1e-9) for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "REPAIR-BY-DEATH"
        # partial wave: 3 runs -> NOT-ADJUDICABLE (rev R3-D2), even
        # though one of them would collapse
        runs = [_fixture(tmp, f"p{s}", s, rel_raw=-0.01,
                         rel_norm=-0.001,
                         base_raw_dead=(1e-9 if s == 120 else None))
                for s in B4_SEEDS[:3]]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE"
        # gates
        rd = _fixture(tmp, "g0", 120, head="det")
        _expect_fail(lambda: read_run(rd, NP), "gauss head")
        rd = _fixture(tmp, "g1", 90)
        _expect_fail(lambda: read_run(rd, NP), "registered 120-123")
        rd = _fixture(tmp, "g2", 121, dup0_zero=False)
        _expect_fail(lambda: read_run(rd, NP), "bitwise-equal")
        rd = _fixture(tmp, "g3", 122, lv_bounds=(-10.0, 6.0))
        _expect_fail(lambda: read_run(rd, NP), "logvar bounds")

        def doc_ck(mj):
            mj["ckpt"] = os.path.join(mj["run_logdir"], "ckpt",
                                      "other_ckpt")
        rd = _fixture(tmp, "g4", 123, doctor=doc_ck)
        _expect_fail(lambda: read_run(rd, NP),
                     "different checkpoints")

        def _doct(path):
            with open(path) as f:
                mj = json.load(f)
            mj["channels"]["distractor"]["delta_norm_mean"] = -9.0
            with open(path, "w") as f:
                json.dump(mj, f)
        rd = _fixture(tmp, "g5", 122)
        _doct(os.path.join(rd, "se_alea_mask", "se_alea_mask.json"))
        _expect_fail(lambda: read_run(rd, NP), "provenance")
        # excluded + one-execution via run()
        outd = os.path.join(tmp, "out")
        for s in B4_SEEDS:
            _fixture(tmp, f"e{s}", s, rel_raw=-0.01, rel_norm=-0.001,
                     drop_npz=(s == 120))
        out = run(parse_args([
            "--runs", os.path.join(tmp, "e12*"),
            "--stage1_runs", os.path.join(tmp, "st*"),
            "--n_perm", str(NP), "--output", outd]))
        assert len(out["excluded_runs"]) == 1
        assert out["outcome_cell"] == "NOT-ADJUDICABLE"
        _expect_fail(lambda: run(parse_args([
            "--runs", os.path.join(tmp, "e12*"),
            "--stage1_runs", os.path.join(tmp, "st*"),
            "--n_perm", str(NP), "--output", outd])), "ONE execution")
    print("se_b4_read selfcheck PASS (REPAIRED / PARTIAL / "
          "constant-rescaling NOT-fired / OVER-CORRECTED / "
          "REPAIR-BY-FLATTENING / RAW-MISPRICE-ABSENT / "
          "REPAIR-BY-DEATH-via-base_raw / partial-wave-first "
          "cells; head/seed/dup0/logvar/ckpt-mismatch/provenance "
          "gates; excluded->NOT-ADJUDICABLE + one-execution guard)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
