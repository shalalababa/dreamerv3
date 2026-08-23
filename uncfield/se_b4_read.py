"""Track-B4 (aleatoric repair) FROZEN reader
(PREREG_trackB_alea_20260822.md §2 + AMENDMENT 1, revised per the
amendment review R-A1 on 22 Aug; built + selfchecked BEFORE the
wave's compute; ONE execution).

AMENDMENT-1 SHARE FORM (post det-baseline; the masked-delta form is
CONSTRUCTION-INERT for the exogenous distractor — batch permutation
is law-preserving, so its population delta is zero by
exchangeability REGARDLESS of pricing; R-A1-M12). The primary runs
on the decoder-projected share line over the REGISTERED UNIVERSE
{distractor, planted_dup0/1/2, position, velocity} (planted_const
excluded — the decoder-projection floor diagnostic, se_probe review
#21 B3; R-A1-B5).

Read order:
  0. Validity: pins (gauss head + logvar bounds + Stage-1 rest +
     penalty inert), seeds 120-123, S=512 + probe/mask seed==0 +
     probe n_eval==512 (same-anchors identity, R-A1-M8), BOTH
     instruments on the SAME checkpoint + newest-ckpt gate, run-dir
     identity, json==npz provenance on masked channels AND shares,
     cross-instrument share provenance (mask share ratios ==
     se_probe key_share ratios, R-A1-M7a), theta_1 recomputed
     (R-A1-M7b), no universe key floor-bound (R-A1-m16), dup0 zero.
     Defective -> excluded. FULL WAVE FIRST; a NOT-ADJUDICABLE
     partial wave writes a timestamped side file and does NOT
     consume the one execution (R-A1-M13).
  1. DEGENERACY (real level + base_raw vs Stage-1 minima) ->
     REPAIR-BY-DEATH.
  2. INSTRUMENT-OUT-OF-SUPPORT (R-A1-B3): whitened-prob clip
     saturation > 0.10 or logvar-floor fraction > 0.25 in any run
     -> validity cell, never a repair cell.
  3. MISPRICE cells (R-A1-M9): theta_1 <= 1 in any run (healthy
     levels — DEATH already returned) ->
     MISPRICE-DISSOLVED-IN-TRAINING (a positive, reportable
     finding: the actor's visitation drift dissolved the misprice);
     else theta_1 < 2.36 (= 0.5 x the Stage-1 minimum 4.72) in any
     run -> MISPRICE-ATTENUATED.
  4. ALIVENESS (R-A1-B1, renormalization-invariant): on CONDITIONAL
     shares q(k) = share(k) / (1 - share(distractor)) over the
     non-distractor universe: (a) q_norm(velocity) in [0.5x, 2x]
     q_raw(velocity) — exactly 1.0 under a distractor-only repair;
     (b) spread conservation sum|q_norm - 1/K| / sum|q_raw - 1/K|
     >= 0.5 — collapses to 0 under global flattening; (c) anchor-cv
     floor on the whitened universe sum (non-positive means ->
     explicit failure, R-A1-m22). Fail -> REPAIR-BY-FLATTENING.
  5. PRIMARY: ratio = share_norm(d)/share_raw(d);
     DE-REPAIR-INVERTED iff >= 1 run's anchor-bootstrap 2.5th
     percentile > 1.0 (R-A1-B2 — a bare ratio hair above 1 is
     PARTIAL, not INVERTED; percentile bootstrap disclosed,
     R-A1-m15); else REPAIRED iff ratio < 0.5 in 4/4; else
     PARTIAL-REPAIR. Precedence: INVERTED > REPAIRED/PARTIAL
     (R-A1-M14). Power note: 4/4 unanimity ~1/16 null rate.
  6. Descriptives (emitted BEFORE any branch, R-A1-m17): share
     table, theta_1 rows, masked rows (the distractor mask is a
     CONSTRUCTION-INERT exchangeability control — a materially
     nonzero value indicates an instrument defect; velocity/dup
     masks remain informative), gauss + whitening diagnostics,
     coverage/score, base levels. Fit counters annotate.

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
XPROV_RTOL = 5e-2          # cross-instrument (Amendment 2,
#   20260823: bf16 graph-order divergence between the two compiled
#   instruments measured at mixed-sign ~1-2% on independent
#   channels; the gate's purpose — M6-class >=50% form drift,
#   wrong-checkout probes — is intact at 5e-2; anchor identity is
#   the R-A1-M8 seed/n_eval pins' job)
LOGVAR_MIN_PIN = -8.0
LOGVAR_MAX_PIN = 6.0
CV_FLOOR_FRAC = 0.25
VEL_BAND = (0.5, 2.0)      # on CONDITIONAL shares (R-A1-B1)
SPREAD_FLOOR = 0.5
THETA1_FLOOR = 1.0
THETA1_BAR = 2.36          # 0.5 x min Stage-1 theta_1 (4.72)
CLIP_FRAC_MAX = 0.10
LV_FRAC_LO_MAX = 0.25
UNIVERSE = ("distractor", "planted_dup0", "planted_dup1",
            "planted_dup2", "position", "velocity")
N_BOOT_SHARE = 2000


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
        f"[-8, 6]")
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
    # theta_1 provenance (R-A1-M7b): stored == recomputed from npz
    th_stored = float(rec["channels"]["distractor"]["vs_source"])
    th_rc = rec["levels"]["theta_recomputed"].get("distractor")
    assert th_rc is not None and abs(th_rc - th_stored) <= 1e-4 * \
        max(1.0, abs(th_stored)), (
        f"{run_dir}: theta_1 provenance mismatch")
    ckroot = os.path.join(run_dir, "ckpt")
    try:
        subs = sorted(x for x in os.listdir(ckroot)
                      if os.path.isdir(os.path.join(ckroot, x)))
    except OSError:
        subs = []
    with open(os.path.join(run_dir, "se_probe",
                           "se_probe.json")) as f:
        probe_j = json.load(f)
    probed = os.path.basename(os.path.normpath(probe_j["ckpt"]))
    # same-anchors identity (R-A1-M8)
    assert int(probe_j["seed"]) == 0 and \
        int(probe_j["n_eval"]) == S_PIN, (
        f"{run_dir}: se_probe seed/n_eval "
        f"({probe_j['seed']}/{probe_j['n_eval']}) not the "
        f"registered (0/{S_PIN}) — anchor sets would differ")

    mdir = os.path.join(run_dir, "se_alea_mask")
    with open(os.path.join(mdir, "se_alea_mask.json")) as f:
        mj = json.load(f)
    npz = np.load(os.path.join(mdir, "se_alea_mask.npz"))
    mask_ck = os.path.basename(os.path.normpath(mj["ckpt"]))
    assert mask_ck == probed, (
        f"{run_dir}: se_probe ckpt {probed} != se_alea_mask ckpt "
        f"{mask_ck} — the two instruments measured different "
        f"checkpoints")
    if subs:
        assert probed == subs[-1], (
            f"{run_dir}: probed ckpt {probed} != newest {subs[-1]}")
    assert os.path.basename(os.path.normpath(mj["run_logdir"])) == \
        os.path.basename(os.path.normpath(run_dir)), (
        f"{run_dir}: mask json belongs to {mj['run_logdir']}")
    assert int(mj["n_eval"]) == S_PIN, (
        f"{run_dir}: mask S {mj['n_eval']} != {S_PIN}")
    assert int(mj["seed"]) == 0, (
        f"{run_dir}: mask seed {mj['seed']} != registered 0")
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
    # Amendment-1 share block: registered universe + simplex +
    # within-instrument provenance + floor hygiene
    assert "shares" in mj, (
        f"{run_dir}: no share block — pre-amendment instrument "
        f"output")
    assert sorted(mj.get("share_universe", [])) == sorted(UNIVERSE), (
        f"{run_dir}: share universe "
        f"{mj.get('share_universe')} != registered {UNIVERSE}")
    sh = mj["shares"]
    assert set(sh) == set(UNIVERSE), f"{run_dir}: share keys {set(sh)}"
    fb = mj.get("share_floor_bound_keys", [])
    assert not (set(fb) & set(UNIVERSE)), (
        f"{run_dir}: universe key(s) {set(fb) & set(UNIVERSE)} are "
        f"floor-bound (R-A1-m16)")
    pk = {}
    for stat in ("raw", "norm"):
        tot = sum(v[stat] for v in sh.values())
        assert abs(tot - 1.0) < 1e-9, (
            f"{run_dir}: {stat} shares sum {tot} != 1")
        means = {}
        for k in UNIVERSE:
            v = np.asarray(npz[f"pkvar_{stat}_{k}"], np.float64)
            assert v.size == S_PIN and np.all(np.isfinite(v)) and \
                np.all(v >= 0), (run_dir, k, stat)
            pk[f"{stat}_{k}"] = v
            means[k] = float(v.mean())
        totm = float(np.sum(list(means.values())))
        for k in UNIVERSE:
            assert abs(sh[k][stat] - means[k] / totm) <= 1e-12, (
                f"{run_dir}/{k}/{stat}: share provenance mismatch")
    # cross-instrument share provenance (R-A1-M7a): the mask's raw
    # per-key ratios must reproduce se_probe's key_share ratios
    # (same anchors, same machinery; denominators cancel)
    ks = probe_j["key_share"]
    for k in UNIVERSE:
        r_mask = sh[k]["raw"] / sh["position"]["raw"]
        r_probe = float(ks[k]) / float(ks["position"])
        assert abs(r_mask - r_probe) <= XPROV_RTOL * max(
            1.0, abs(r_probe)), (
            f"{run_dir}/{k}: cross-instrument share ratio "
            f"{r_mask} != se_probe {r_probe}")
    rec["mask"] = mj["channels"]
    rec["gauss_diag"] = gd
    rec["share_clip_frac"] = float(mj["share_clip_frac_mean"])
    rec["share_gain_mean"] = float(mj["share_gain_mean"])
    rec["share_gain_p95"] = float(mj["share_gain_p95"])
    rec["shares"] = sh
    rec["pk"] = pk
    cov, cov_steps, cov_flag = coverage_proxy(run_dir)
    rec["coverage"], rec["coverage_flag"] = cov, cov_flag
    rec["score"], _ = score_tail(run_dir)
    rec["base_raw"] = float(mj["base_raw_mean"])
    rec["base_norm"] = float(mj["base_norm_mean"])
    return rec


def _cv(x):
    m = float(np.mean(x))
    if m <= 0:
        return None                      # explicit failure (m22)
    return float(np.std(x) / m)


def _share_ratio_boot(pk, rng, n=N_BOOT_SHARE):
    """Anchor-resample PERCENTILE interval (disclosed, R-A1-m15) of
    share_norm(distractor) / share_raw(distractor)."""
    S = pk["raw_distractor"].size
    vals = []
    for _ in range(n):
        idx = rng.integers(0, S, S)
        sr = {k: pk[f"raw_{k}"][idx].mean() for k in UNIVERSE}
        sn = {k: pk[f"norm_{k}"][idx].mean() for k in UNIVERSE}
        tr, tn = sum(sr.values()), sum(sn.values())
        if tr > 0 and tn > 0 and sr["distractor"] > 0:
            vals.append((sn["distractor"] / tn)
                        / (sr["distractor"] / tr))
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
    # descriptives BEFORE any branch (R-A1-m17)
    out["per_run"] = [
        {"run_dir": r["run_dir"], "train_seed": r["train_seed_cfg"],
         "valid": r["valid"],
         "shares": r["shares"],
         "theta1_distractor":
             r["channels"]["distractor"]["vs_source"],
         "base_raw": r["base_raw"], "base_norm": r["base_norm"],
         "gauss_diag": r["gauss_diag"],
         "share_clip_frac": r["share_clip_frac"],
         "share_gain_mean": r["share_gain_mean"],
         "share_gain_p95": r["share_gain_p95"],
         "coverage": r["coverage"], "score_tail": r["score"],
         "real_mean_level": r["levels"]["real_mean"]}
        for r in recs]
    out["masked_descriptive"] = dict(
        distractor_raw=[r["mask"]["distractor"]["delta_raw_mean"]
                        for r in recs],
        distractor_norm=[r["mask"]["distractor"]["delta_norm_mean"]
                         for r in recs],
        velocity_raw=[r["mask"]["velocity_control"]["delta_raw_mean"]
                      for r in recs],
        velocity_norm=[r["mask"]["velocity_control"]
                       ["delta_norm_mean"] for r in recs],
        note="the distractor mask is a CONSTRUCTION-INERT "
             "exchangeability control (exogenous channel; "
             "population delta 0 for any statistic) — a materially "
             "nonzero value indicates an instrument defect; the "
             "velocity and dup masks are informative (R-A1-M12)")
    out["theta1"] = [r["channels"]["distractor"]["vs_source"]
                     for r in recs]

    valid = [r for r in recs if r["valid"]]
    if len(recs) < 4 or len(valid) < 4:
        out["primary"] = (f"NOT-ADJUDICABLE ({len(valid)}/4 valid "
                          f"of {len(recs)} loaded; the 4/4 rules "
                          f"need the full wave)")
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        return out

    collapsed = [r["run_dir"] for r in valid
                 if r["levels"]["real_mean"]
                 < ref["real_mean_min"] / COLLAPSE_FACTOR
                 or r["base_raw"]
                 < ref["intrinsic_min"] / COLLAPSE_FACTOR]
    out["collapsed_runs"] = collapsed
    if collapsed:
        out["primary"] = ("REPAIR-BY-DEATH: levels collapsed below "
                          "1/10 of the Stage-1 minima — the gauss "
                          "arm did not preserve the substrate")
        out["outcome_cell"] = "REPAIR-BY-DEATH"
        return out

    # 2) instrument support (R-A1-B3)
    oos = [dict(run_dir=r["run_dir"],
                clip_frac=r["share_clip_frac"],
                lv_frac_lo=r["gauss_diag"]["lv_frac_lo"])
           for r in valid
           if r["share_clip_frac"] > CLIP_FRAC_MAX
           or r["gauss_diag"]["lv_frac_lo"] > LV_FRAC_LO_MAX]
    if oos:
        out["oos_runs"] = oos
        out["primary"] = (
            "INSTRUMENT-OUT-OF-SUPPORT: whitened members leave the "
            "decoder's support (clip saturation or logvar-floor "
            "pileup) — the share statistic measures extrapolation, "
            "not reallocation; a validity cell, never a repair cell")
        out["outcome_cell"] = "INSTRUMENT-OUT-OF-SUPPORT"
        return out

    # 3) misprice cells (R-A1-M9)
    th = out["theta1"]
    if any(x <= THETA1_FLOOR for x in th):
        out["primary"] = dict(
            statement="MISPRICE-DISSOLVED-IN-TRAINING: theta_1 <= 1 "
                      "with healthy levels — the actor's visitation "
                      "under the normalized reward dissolved the "
                      "misprice in the LEARNED model (a positive, "
                      "reportable behavioural outcome; the "
                      "accounting repair is then untestable here)",
            theta1=th)
        out["outcome_cell"] = "MISPRICE-DISSOLVED-IN-TRAINING"
        return out
    if any(x < THETA1_BAR for x in th):
        out["primary"] = dict(
            statement="MISPRICE-ATTENUATED: theta_1 in (1, 2.36) — "
                      "below half the Stage-1 minimum; the raw "
                      "share is too small for the ratio to be "
                      "identified",
            theta1=th)
        out["outcome_cell"] = "MISPRICE-ATTENUATED"
        return out

    # 4) aliveness (R-A1-B1): conditional shares
    nond = [k for k in UNIVERSE if k != "distractor"]
    K = len(nond)
    alive_rows = []
    for r in valid:
        qr = {k: r["shares"][k]["raw"]
              / (1.0 - r["shares"]["distractor"]["raw"])
              for k in nond}
        qn = {k: r["shares"][k]["norm"]
              / (1.0 - r["shares"]["distractor"]["norm"])
              for k in nond}
        vel_ratio = qn["velocity"] / max(qr["velocity"], 1e-30)
        vel_ok = VEL_BAND[0] <= vel_ratio <= VEL_BAND[1]
        spread_raw = sum(abs(qr[k] - 1.0 / K) for k in nond)
        spread_norm = sum(abs(qn[k] - 1.0 / K) for k in nond)
        spread_ratio = (spread_norm / spread_raw
                        if spread_raw > 0 else 0.0)
        spread_ok = spread_ratio >= SPREAD_FLOOR
        tot_raw = np.sum([r["pk"][f"raw_{k}"] for k in UNIVERSE], 0)
        tot_norm = np.sum([r["pk"][f"norm_{k}"] for k in UNIVERSE], 0)
        cvr, cvn = _cv(tot_raw), _cv(tot_norm)
        cv_ok = (cvr is not None and cvn is not None
                 and cvn >= CV_FLOOR_FRAC * cvr)
        alive_rows.append(dict(
            run_dir=r["run_dir"], vel_ratio_conditional=vel_ratio,
            vel_ok=bool(vel_ok), spread_ratio=spread_ratio,
            spread_ok=bool(spread_ok), cv_raw=cvr, cv_norm=cvn,
            cv_ok=bool(cv_ok),
            alive=bool(vel_ok and spread_ok and cv_ok)))
    out["aliveness"] = alive_rows
    if not all(a["alive"] for a in alive_rows):
        out["primary"] = (
            "REPAIR-BY-FLATTENING: the whitened statistic fails the "
            "renormalization-invariant aliveness gate (conditional "
            "velocity share left its band, non-distractor spread "
            "collapsed, or anchor dispersion died) — a flat/global "
            "reallocation is not distractor-specific repair")
        out["outcome_cell"] = "REPAIR-BY-FLATTENING"
        return out

    # 5) PRIMARY
    sh_raw = [r["shares"]["distractor"]["raw"] for r in valid]
    sh_norm = [r["shares"]["distractor"]["norm"] for r in valid]
    ratios = [n / r for n, r in zip(sh_norm, sh_raw)]
    boots = [_share_ratio_boot(
        r["pk"], np.random.default_rng(r["train_seed_cfg"]))
        for r in valid]
    inverted = [r["run_dir"] for b, r in zip(boots, valid)
                if b is not None and b[0] > 1.0]
    out["primary"] = dict(
        statement="ALEATORIC REPAIR (share form, Amendment 1 rev "
                  "R-A1): the deployed whitening reduces the "
                  "distractor's share of decoder-projected ensemble "
                  "variance (registered 6-key universe) to < 0.5x "
                  "its raw share in 4/4 runs. DE-REPAIR-INVERTED "
                  "needs bootstrap separation above 1 and takes "
                  "precedence. Power note: 4/4 unanimity, null rate "
                  "~1/16, no gradation",
        share_raw=sh_raw, share_norm=sh_norm, ratio=ratios,
        ratio_boot_pctl95=boots, bar=REPAIR_BAR, theta1=th,
        inverted_runs=inverted)
    if inverted:
        out["primary"]["fires"] = False
        out["outcome_cell"] = "DE-REPAIR-INVERTED"
    else:
        fires = bool(all(x < REPAIR_BAR for x in ratios))
        out["primary"]["fires"] = fires
        out["outcome_cell"] = ("REPAIRED" if fires
                               else "PARTIAL-REPAIR")
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
            f"carried)")
    os.makedirs(args.output, exist_ok=True)
    # R-A1-M13: a NOT-ADJUDICABLE partial wave does NOT consume the
    # one execution — it persists to a timestamped side file, and
    # se_b4_read.json is claimed only by an adjudicated outcome
    if out["outcome_cell"] == "NOT-ADJUDICABLE":
        side = os.path.join(
            args.output,
            f"se_b4_read_NOTADJ_{time.strftime('%Y%m%dT%H%M%S')}"
            f".json")
        with open(side, "w") as f:
            json.dump(out, f, indent=1)
        print(f"B4 PRIMARY: NOT-ADJUDICABLE (side file {side}; the "
              f"one execution is NOT consumed)")
        return out
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    if isinstance(out["primary"], dict) and "ratio" in out["primary"]:
        pr = out["primary"]
        print(f"B4 PRIMARY (share form): ratio "
              f"{[round(x, 3) for x in pr['ratio']]} "
              f"fires={pr['fires']} -> {out['outcome_cell']}; "
              f"fit {out['fit_flag']}")
    else:
        print(f"B4 PRIMARY: {out['outcome_cell']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, sh_raw_d=0.46, sh_norm_d=0.05,
             sh_raw_v=0.184, vel_norm_frac=1.0, flatten=False,
             cv_norm_frac=1.0, clip_frac=0.01, lv_frac_lo=0.02,
             hot=None, level_scale=1.0, base_raw_dead=None,
             head="gauss", dup0_zero=True, lv_bounds=(-8.0, 6.0),
             doctor=None, doctor_probe=None, drop_npz=False,
             drop_shares=False, boot_sep_inverted=False):
    """R-A1-M10: physically coupled fixture on the measured Stage-1
    geometry (sh_raw_d ~0.46, sh_raw_v ~0.184). Non-distractor norm
    shares follow the MECHANICAL renormalization
    sh_norm(k) = sh_raw(k) * (1 - sh_norm_d)/(1 - sh_raw_d) unless
    perturbed (vel_norm_frac) or flattened (uniform). The probe
    json's key_share is doctored to be ratio-consistent with the
    generated raw shares (the M7a cross-check is live)."""
    rd = se_read._fixture_run(tmp, name, seed,
                              hot=(hot or {"distractor": 5.0}))
    pdir = os.path.join(rd, "se_probe")
    with open(os.path.join(pdir, "se_probe.json")) as f:
        pj = json.load(f)
    pj["pse2"]["intrinsic_mean"] = 0.25
    br = base_raw_dead if base_raw_dead is not None else 3e-4
    # raw share layout (dups ~0.086 each, position the remainder)
    raw = {"distractor": sh_raw_d, "velocity": sh_raw_v,
           "planted_dup0": 0.086, "planted_dup1": 0.086,
           "planted_dup2": 0.086}
    raw["position"] = 1.0 - sum(raw.values())
    if flatten:
        norm = {k: 1.0 / len(UNIVERSE) for k in UNIVERSE}
    else:
        infl = (1.0 - sh_norm_d) / (1.0 - sh_raw_d)
        norm = {k: raw[k] * infl for k in UNIVERSE
                if k != "distractor"}
        norm["distractor"] = sh_norm_d
        norm["velocity"] = norm["velocity"] * vel_norm_frac
        # re-close the simplex proportionally over the untouched
        # keys (keeps every share positive under large vel knobs)
        rest = [k for k in UNIVERSE
                if k not in ("distractor", "velocity")]
        deficit = sum(norm.values()) - 1.0
        rest_tot = sum(norm[k] for k in rest)
        assert rest_tot > deficit, "fixture knob too extreme"
        for k in rest:
            norm[k] *= (rest_tot - deficit) / rest_tot
    # doctor the probe key_share to match raw ratios (7 keys w/ a
    # tiny const entry, then renormalized — ratios preserved)
    ks = dict(raw)
    ks["planted_const"] = 1e-7
    tot = sum(ks.values())
    pj["key_share"] = {k: v / tot for k, v in ks.items()}
    if doctor_probe:
        doctor_probe(pj)
    with open(os.path.join(pdir, "se_probe.json"), "w") as f:
        json.dump(pj, f)
    if level_scale != 1.0:
        npzp = os.path.join(pdir, "se_probe_dims.npz")
        z = dict(np.load(npzp))
        z["dims_d"] = z["dims_d"] * level_scale
        np.savez(npzp, **z)
    mdir = os.path.join(rd, "se_alea_mask")
    os.makedirs(mdir, exist_ok=True)
    rng = np.random.default_rng(seed * 11 + 3)
    chans, dnpz = {}, {}
    for ch in ("distractor", "planted_dup0", "planted_dup1",
               "planted_dup2", "planted_dup1_resample",
               "planted_dup2_resample", "velocity_control"):
        rec = dict(form="x")
        for stat in ("raw", "norm"):
            if ch == "planted_dup0" and dup0_zero:
                dv = np.zeros(S_PIN)
            else:
                dv = rng.normal(0.0, 1e-6, S_PIN)
            m = float(dv.mean())
            rec[f"delta_{stat}_mean"] = m
            rec[f"delta_{stat}_bca"] = [m - 1e-6, m + 1e-6]
            rec[f"p_reduce_{stat}"] = 0.5
            dnpz[f"delta_{stat}_{ch}"] = dv.astype(np.float64)
        chans[ch] = rec
    # per-anchor rows: mean-match the target shares exactly; the
    # spread is the anchor-level noise (cv leg controllable)
    shares, pk_store = {}, {}
    for stat, km in (("raw", raw), ("norm", norm)):
        spread = 0.3 if stat == "raw" else 0.3 * cv_norm_frac
        if boot_sep_inverted and stat == "norm":
            spread = 0.1     # tight enough to separate a 30%
            #                  excess, loose enough for the cv leg
        arrs = {}
        for k, mval in km.items():
            v = np.maximum(rng.normal(mval, spread * max(mval, 1e-3),
                                      S_PIN), 1e-6)
            v = v - v.mean() + mval          # exact mean
            arrs[k] = np.maximum(v, 1e-9).astype(np.float64)
        totm = float(np.sum([a.mean() for a in arrs.values()]))
        for k, a in arrs.items():
            pk_store[f"pkvar_{stat}_{k}"] = a
            shares.setdefault(k, {})[stat] = float(a.mean() / totm)
    with open(os.path.join(pdir, "se_probe.json")) as f:
        probe_ck = os.path.basename(
            os.path.normpath(json.load(f)["ckpt"]))
    mj = dict(run_logdir=rd, ckpt=os.path.join(rd, "ckpt", probe_ck),
              n_eval=S_PIN, seed=0, train_seed=seed, task=TASK_PIN,
              source_key="position", disag_head="gauss",
              dup0_bitwise_equal_source=dup0_zero,
              base_raw_mean=br, base_norm_mean=0.25,
              shares=shares, share_universe=list(UNIVERSE),
              share_raw_distractor=shares["distractor"]["raw"],
              share_norm_distractor=shares["distractor"]["norm"],
              share_norm_floor=0.01, share_floor_bound_keys=[
                  "planted_const"],
              share_clip_frac_mean=clip_frac,
              share_gain_mean=1.1, share_gain_p95=1.4,
              gauss_diagnostics=dict(
                  logvar_min=lv_bounds[0], logvar_max=lv_bounds[1],
                  alea_mean=1.0, alea_std=0.2, lv_frac_lo=lv_frac_lo,
                  lv_frac_hi=0.0, base_norm_deter_mean=0.3,
                  base_norm_stoch_mean=0.2),
              channels=chans)
    if drop_shares:
        del mj["shares"]
    if doctor:
        doctor(mj)
    with open(os.path.join(mdir, "se_alea_mask.json"), "w") as f:
        json.dump(mj, f)
    if not drop_npz:
        np.savez(os.path.join(mdir, "se_alea_mask.npz"),
                 base_raw=np.full(S_PIN, br),
                 base_norm=np.full(S_PIN, 0.25),
                 share_clip_frac=np.full(S_PIN, clip_frac),
                 share_gain=np.full(S_PIN, 1.1),
                 **pk_store, **dnpz)
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

        # REPAIRED at the MEASURED Stage-1 geometry (0.46 -> 0.05):
        # the mechanical velocity inflation (x1.76) passes the
        # CONDITIONAL band (ratio 1.0) — the R-A1-B1 case
        runs = [_fixture(tmp, f"a{s}", s) for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "REPAIRED", (
            agg["outcome_cell"], agg.get("aliveness"))
        assert all(x < 0.5 for x in agg["primary"]["ratio"])
        assert all(abs(a["vel_ratio_conditional"] - 1.0) < 0.05
                   for a in agg["aliveness"])
        assert "masked_descriptive" in agg   # pre-branch (m17)
        # PARTIAL-REPAIR: 0.46 -> 0.30
        runs = [_fixture(tmp, f"b{s}", s, sh_norm_d=0.30)
                for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "PARTIAL-REPAIR"
        # R-A1-B2: ratio a hair above 1 (0.65%, inside the anchor
        # noise band) WITHOUT bootstrap separation stays PARTIAL,
        # never INVERTED
        runs = [_fixture(tmp, f"h{s}", s, sh_norm_d=0.463)
                for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "PARTIAL-REPAIR", (
            agg["outcome_cell"])
        assert not agg["primary"]["inverted_runs"]
        # ...but a SEPARATED increase is DE-REPAIR-INVERTED
        runs = [_fixture(tmp, f"i{s}", s, sh_norm_d=0.60,
                         boot_sep_inverted=True) for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "DE-REPAIR-INVERTED", (
            agg["outcome_cell"])
        # R-A1-B1 blindness case: GLOBAL FLATTENING must land in
        # FLATTENING (the spread leg), never REPAIRED
        runs = [_fixture(tmp, f"f{s}", s, flatten=True)
                for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "REPAIR-BY-FLATTENING", (
            agg["outcome_cell"])
        # velocity-specific blowup outside the conditional band
        # (x2.2 conditional ratio > the 2.0 ceiling)
        runs = [_fixture(tmp, f"v{s}", s, vel_norm_frac=2.2)
                for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "REPAIR-BY-FLATTENING"
        # anchor-dispersion collapse
        runs = [_fixture(tmp, f"g{s}", s, cv_norm_frac=0.02)
                for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "REPAIR-BY-FLATTENING"
        # R-A1-B3: out-of-support gate (clip saturation)
        runs = [_fixture(tmp, f"o{s}", s, clip_frac=0.2)
                for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "INSTRUMENT-OUT-OF-SUPPORT"
        # ...and via logvar-floor pileup
        runs = [_fixture(tmp, f"l{s}", s, lv_frac_lo=0.4)
                for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "INSTRUMENT-OUT-OF-SUPPORT"
        # R-A1-M9: theta_1 <= 1 -> DISSOLVED (positive cell)
        runs = [_fixture(tmp, f"c{s}", s, hot={"distractor": 0.3})
                for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "MISPRICE-DISSOLVED-IN-TRAINING"
        # ...and 1 < theta_1 < 2.36 -> ATTENUATED
        runs = [_fixture(tmp, f"t{s}", s, hot={"distractor": 1.8})
                for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "MISPRICE-ATTENUATED"
        # REPAIR-BY-DEATH via the base_raw leg
        runs = [_fixture(tmp, f"d{s}", s, level_scale=1e-6,
                         base_raw_dead=1e-9) for s in B4_SEEDS]
        agg = aggregate([read_run(d, NP) for d in runs], [], ref)
        assert agg["outcome_cell"] == "REPAIR-BY-DEATH"
        # partial wave first
        runs = [_fixture(tmp, f"p{s}", s) for s in B4_SEEDS[:3]]
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
        rd = _fixture(tmp, "g4", 123, drop_shares=True)
        _expect_fail(lambda: read_run(rd, NP), "pre-amendment")

        def doc_uni(mj):
            mj["share_universe"] = [k for k in UNIVERSE
                                    if k != "velocity"]
        rd = _fixture(tmp, "g5", 120, doctor=doc_uni)
        _expect_fail(lambda: read_run(rd, NP), "share universe")

        def doc_floor(mj):
            mj["share_floor_bound_keys"] = ["planted_const",
                                            "distractor"]
        rd = _fixture(tmp, "g6", 121, doctor=doc_floor)
        _expect_fail(lambda: read_run(rd, NP), "floor-bound")

        # R-A1-M7a: cross-instrument drift (doctored key_share)
        def doc_ks(pj):
            pj["key_share"]["distractor"] *= 1.5
        rd = _fixture(tmp, "g7", 122, doctor_probe=doc_ks)
        _expect_fail(lambda: read_run(rd, NP), "cross-instrument")

        # R-A1-M8: mask seed drift
        def doc_seed(mj):
            mj["seed"] = 7
        rd = _fixture(tmp, "g8", 123, doctor=doc_seed)
        _expect_fail(lambda: read_run(rd, NP), "mask seed")

        def doc_ck(mj):
            mj["ckpt"] = os.path.join(mj["run_logdir"], "ckpt",
                                      "other_ckpt")
        rd = _fixture(tmp, "g9", 123, doctor=doc_ck)
        _expect_fail(lambda: read_run(rd, NP),
                     "different checkpoints")

        def doc_sh(mj):
            d0 = mj["shares"]["distractor"]["norm"]
            mj["shares"]["distractor"]["norm"] = 0.001
            mj["shares"]["position"]["norm"] += d0 - 0.001
        rd = _fixture(tmp, "ga", 122, doctor=doc_sh)
        _expect_fail(lambda: read_run(rd, NP), "share provenance")
        # excluded path -> NOT-ADJ side file; the one execution is
        # NOT consumed (R-A1-M13); a later complete read still runs
        outd = os.path.join(tmp, "out")
        for s in B4_SEEDS:
            _fixture(tmp, f"e{s}", s, drop_npz=(s == 120))
        out = run(parse_args([
            "--runs", os.path.join(tmp, "e12*"),
            "--stage1_runs", os.path.join(tmp, "st*"),
            "--n_perm", str(NP), "--output", outd]))
        assert out["outcome_cell"] == "NOT-ADJUDICABLE"
        assert not os.path.exists(os.path.join(outd,
                                               "se_b4_read.json"))
        assert globmod.glob(os.path.join(outd,
                                         "se_b4_read_NOTADJ_*"))
        _fixture(tmp, "e120", 120)     # repair the broken run
        out = run(parse_args([
            "--runs", os.path.join(tmp, "e12*"),
            "--stage1_runs", os.path.join(tmp, "st*"),
            "--n_perm", str(NP), "--output", outd]))
        assert out["outcome_cell"] == "REPAIRED"
        _expect_fail(lambda: run(parse_args([
            "--runs", os.path.join(tmp, "e12*"),
            "--stage1_runs", os.path.join(tmp, "st*"),
            "--n_perm", str(NP), "--output", outd])), "ONE execution")
    print("se_b4_read selfcheck PASS (R-A1 cells: REPAIRED-at-"
          "measured-geometry (conditional vel band ~1.0) / PARTIAL "
          "/ hair-above-1 stays PARTIAL / separated INVERTED / "
          "FLATTENING via spread+vel-band+cv / OUT-OF-SUPPORT via "
          "clip+lv-floor / DISSOLVED / ATTENUATED / DEATH / "
          "partial-wave side-file (execution NOT consumed, then "
          "complete read succeeds); head/seed/dup0/logvar/universe/"
          "floor-keys/cross-instrument/mask-seed/ckpt/share-"
          "provenance gates; one-execution guard)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
