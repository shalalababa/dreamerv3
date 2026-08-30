"""PHI-SWEEP FROZEN reader — the decisive falsifier for
theta_1 = Lambda = 1/(1-phi^2)
(PREREG_phisweep_20260830.md §3-§5; built + selfchecked BEFORE the
wave's compute; ONE execution, explicit --output).

Derived from the FROZEN, executed Track-B ONSET reader
(se_onset_read) and through it from the B1 dose reader: identity gates
FATAL (a defective run refuses the whole read), missing-seed
accounting, per-run ckpt provenance, degeneracy against the Stage-1
level reference, and the FLIPPED normalizer-floor rule.

PRIMARY-1 (the LAW, fires). Per arm,
    L_hat = mean_i theta_1_i / 5.257875          (registered ref const)
    CI    = exp( log(L_hat) +- t_{.975,n-1} * sqrt(s_log^2/n + .0231^2) )
LAW-CONFIRMED iff L_raw(phi) in CI and BOTH rivals outside;
LAW-KILLED iff L_raw(phi) outside CI. Wave cell LAW-CONFIRMED needs
both arms; any KILL -> LAW-REFUTED.
The ratio estimand is used precisely because raw and symlog
conventions differ by a near-constant multiplicative offset (-4.1 to
-4.7% across the three arms), so L is convention-free to <=0.64% —
far inside its ~4% SE. T3 cannot contaminate the law test.

PRIMARY-2 (the T3 convention adjudication) is pre-declared
UNDER-POWERED (1.80 sigma) and does NOT gate PRIMARY-1.

SECONDARY onset is DEFERRED (prereg §4.3); only the weak
above-onset-consistency datum is reported, never a fire.

Run:  python -m uncfield.se_phisweep_read --runs "<glob>"
          --stage1_runs "<glob>" --output <dir>
      python -m uncfield.se_phisweep_read --selfcheck
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
from uncfield.se_dose_read import (coverage_proxy, score_tail,
                                   stage1_reference_gated)
from uncfield.se_read import fit_counters
from uncfield import se_read

TASK = "dmc_cheetah_run"

# ---------------------------------------------------------------- registered
# constants (PREREG_phisweep_20260830 §2.2, §3). Every number below is
# frozen at registration; nothing here is estimated from this wave.

PHIS = (0.80, 0.95)
ARM_SEEDS = {0.80: frozenset(range(140, 144)),
             0.95: frozenset(range(144, 148))}
ALL_SEEDS = frozenset().union(*ARM_SEEDS.values())
SEED_ARM = {sd: phi for phi, sds in ARM_SEEDS.items() for sd in sds}

# theta = 1 - phi (distractor.py:23,63 -> _ar = 1 - theta)
THETA_OU = {0.80: 0.20, 0.95: 0.05}
# s_arm = k * s0(phi), k = 1/s0(0.90); s0 ∝ sqrt(1-phi^2) so the
# convention constant cancels and s_arm = sqrt((1-phi^2)/0.19) exactly.
SCALES = {0.80: 1.3764944032233706, 0.95: 0.7163503994113789}
SCALE_TOL = 1e-6

# the REFERENCE arm: Stage-1 det, seeds 10-17, s=1.0, phi=0.9.
REF_PHI = 0.90
REF_THETA1 = 5.257875          # registered mean (n=8)
REF_RELSE = 0.02310            # registered relative SE of that mean
REF_N = 8
REF_PERRUN_CV = 0.06533        # registered per-run CV

def _lam(phi):
    return 1.0 / (1.0 - phi ** 2)

LAMBDA_RAW = {0.80: _lam(0.80), 0.90: _lam(0.90), 0.95: _lam(0.95)}

# H_LAW-symlog: MC (N=8e6, pair-difference estimator of
# E[Var(symlog(X_{t+1})|X_t)] against Var(symlog(X)), stationary
# N(0,(s*1.215)^2)), evaluated AT EACH ARM'S OWN CONTROLLED SCALE —
# the symlog leverage is scale-dependent, so the s=1.0 numbers
# (2.6905 / 9.6516) do not apply to this design. The estimator
# reproduces the rescue panel's independent MC at every published
# anchor (phi=0.9: 5.2539/5.0146/4.8088 at s=0.1/1.0/2.0).
SYMLOG_MC = {0.80: 2.6610, 0.90: 5.0146, 0.95: 9.8347}

# H_RIVAL-A: theta_1 follows the dose curve in s/s0 units. All arms are
# held at s/s0 = 3.6364 = the reference's own position, so the rival is
# a FLAT null at the reference value.
RIVAL_A = {0.80: 5.258, 0.95: 5.258}
# H_RIVAL-B: the dose curve in RAW s units (no phi dependence).
# phi=0.95 (s=0.7164) is a log-interpolation inside the measured window;
# phi=0.80 (s=1.3765) is an EXTRAPOLATION off the 0.75->1.00 log slope
# (-0.7675) and is flagged as the weakest of the four predictions.
RIVAL_B = {0.80: 4.115, 0.95: 6.744}
RIVAL_B_EXTRAPOLATED = {0.80: True, 0.95: False}

# convention offsets symlog/raw, per arm at its own controlled scale
CONV_RAW = 1.0000
CONV_SYMLOG = 0.9584           # mean of 0.95796 (phi=.80), 0.95888 (.95)
CONV_SYMLOG_STAGE1 = 0.95277   # the reference arm's own offset

# T6 normalization guard: registered d_pos CV is 8.8% over 24 runs and
# 5 scales; the annotation threshold is 2x that.
DPOS_SPREAD_MAX = 0.176
DPOS_REGISTERED = 0.002016

MIN_PER_ARM = 3
MIN_USABLE = 7

# two-sided 95% Student t critical values (df -> t); hardcoded so the
# reader carries no scipy dependency and the interval is reproducible.
T975 = {2: 4.302652729911275, 3: 3.182446305284263,
        4: 2.7764451051977987, 5: 2.5705818366147395,
        6: 2.4469118487916806, 7: 2.3646242510102993,
        8: 2.306004135204168, 9: 2.262157162798205,
        10: 2.2281388519649385, 11: 2.200985160082949,
        12: 2.178812829667228, 13: 2.1603686564610127,
        14: 2.1447866879169273, 15: 2.131449545559323}

# registration self-consistency (cheap, catches an edited constant)
for _p in PHIS:
    assert abs(THETA_OU[_p] - (1.0 - _p)) < 1e-9, _p
    assert abs(SCALES[_p] - np.sqrt((1.0 - _p ** 2) / 0.19)) < 1e-12, _p


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="")
    p.add_argument("--stage1_runs", default="",
                   help="Stage-1 dirs (seeds 10-17, s=1.0) — the "
                        "degeneracy reference AND the supplementary "
                        "16-run convention pool")
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


def _tcrit(df):
    if df in T975:
        return T975[df]
    assert df >= 2, df
    return 1.959963984540054            # large-df normal limit


def _log_ci(point, values, extra_var=0.0, n=None):
    """Registered interval: exp(log(point) +- t_{.975,n-1} *
    sqrt(sd(log values)^2 / n + extra_var))."""
    v = np.asarray(values, np.float64)
    n = int(v.size if n is None else n)
    assert n >= 2 and (v > 0).all(), (n, v.tolist())
    s_log = float(np.std(np.log(v), ddof=1))
    se = float(np.sqrt(s_log ** 2 / n + extra_var))
    half = _tcrit(n - 1) * se
    return (float(point * np.exp(-half)), float(point * np.exp(half)),
            s_log, se, n)


def _inside(lo, hi, x):
    return bool(lo <= x <= hi)


# ------------------------------------------------------------------- per-run

def read_run(run_dir, n_perm):
    """FATAL identity gates (prereg §5.1-§5.5) + the per-run record."""
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = cfg["distractor"]
    pl_ = cfg["planted"]
    ex = cfg["agent"]["expl"]

    # §5.2 seed<->arm map is the ARM AUTHORITY: the arm is decided by
    # the registered seed block, then theta/scale must AGREE with it
    # (never the other way round — a mis-set theta must not silently
    # relabel the run into the other arm)
    seed = int(cfg["seed"])
    assert seed in SEED_ARM, (
        f"{run_dir}: seed {seed} is not a registered phisweep seed "
        f"{sorted(ALL_SEEDS)}")
    phi = SEED_ARM[seed]

    # §5.1 arm pins
    theta = float(d["theta"])
    assert abs(theta - THETA_OU[phi]) < 1e-9, (
        f"{run_dir}: distractor.theta {theta} != {THETA_OU[phi]} — "
        f"seed {seed} is registered for phi={phi} (phi = 1 - theta); "
        f"the phi axis did not bind")
    scale = float(d["scale"])
    assert abs(scale - SCALES[phi]) < SCALE_TOL, (
        f"{run_dir}: distractor.scale {scale!r} != the s/s0-controlled "
        f"{SCALES[phi]!r} for phi={phi} — the dose position is not "
        f"matched to the reference")

    # §5.3 Stage-1 pin block (verbatim from the B1/onset readers)
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
        abs(float(d["basesd"]) - BASESD_N) < 1e-9, (
        f"{run_dir}: distractor dim/basesd differ from Stage-1 — the "
        f"sweep must move persistence only")
    assert str(d.get("gate_key", "")) == "" and \
        str(d.get("mod_key", "")) == "", (
        f"{run_dir}: gates/mod must be off")
    assert str(pl_.get("gate_key", "")) == ""
    assert float(cfg.get("penalty", {}).get("scale", 0.0)) == 0.0 \
        and ex.get("penalty_mix", False) is False, (
        f"{run_dir}: penalty machinery must be inert")

    rec = se_read.read_run(run_dir, "se_probe", n_perm)
    rec["train_seed_cfg"] = seed
    rec["phi"] = phi
    rec["theta_ou"] = theta
    rec["scale"] = scale

    # §5.4 newest-ckpt cross-check (the A1 append-verify lesson) —
    # record whether the gate was LIVE, not just whether it passed
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

    # §5.5 theta_1 provenance
    lv = _levels(run_dir)
    stored = rec["channels"]["distractor"]["vs_source"]
    rc = lv["theta_recomputed"].get("distractor")
    assert rc is not None and abs(rc - stored) <= 1e-4 * max(
        1.0, abs(stored)), (
        f"{run_dir}: theta_1 provenance mismatch")
    rec["levels"] = lv
    rec["theta1"] = float(stored)

    # normalizer floor + the T6 normalization guard inputs
    npz = np.load(os.path.join(run_dir, "se_probe",
                               "se_probe_dims.npz"))
    dk = np.asarray(npz["dim_key"]).astype(str)
    raw_norms = np.asarray(npz["raw_norms"], np.float64)
    floor = float(npz["norm_floor"])
    dis = dk == "distractor"
    pos = dk == "position"
    dims_d = np.asarray(npz["dims_d"], np.float64)
    norms_eff = np.maximum(raw_norms, floor)
    rec["floor"] = dict(
        distractor_raw_norm_mean=float(raw_norms[dis].mean()),
        norm_floor=floor,
        floor_binds=bool((raw_norms[dis] < floor).any()),
        floor_margin=float(raw_norms[dis].min() / floor))
    rec["unnorm_distractor_var"] = float(
        (dims_d[dis] * norms_eff[dis] ** 2).mean())
    d_pos = float(dims_d[pos].mean()) if pos.any() else None
    d_dist = float(dims_d[dis].mean())
    rec["norm_guard"] = dict(
        d_pos=d_pos, d_dist=d_dist,
        dist_raw_norm=float(raw_norms[dis].mean()),
        pos_raw_norm=(float(raw_norms[pos].mean()) if pos.any()
                      else None),
        unnorm_dist_over_pos=(
            float((dims_d[dis] * norms_eff[dis] ** 2).mean()
                  / (dims_d[pos] * norms_eff[pos] ** 2).mean())
            if pos.any() else None))
    # prereg §4.3: the weak above-onset consistency datum (never fires)
    rec["above_onset_consistent"] = bool(
        rec["theta1"] > 1.0 and not rec["floor"]["floor_binds"])

    cov, cov_steps, cov_flag = coverage_proxy(run_dir)
    rec["coverage"], rec["coverage_steps"], rec["coverage_flag"] = \
        cov, cov_steps, cov_flag
    rec["score"], rec["n_score_entries"] = score_tail(run_dir)
    return rec


def stage1_theta1(stage1_dirs):
    """theta_1 of the REFERENCE arm, recomputed from the passed
    Stage-1 bundle. Feeds the supplementary 16-run convention pool and
    a report-only provenance check against the registered constant."""
    import yaml
    vals, seeds = [], []
    for dd in stage1_dirs:
        with open(os.path.join(dd, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        seeds.append(int(cfg["seed"]))
        t = _levels(dd)["theta_recomputed"].get("distractor")
        assert t is not None, f"{dd}: no distractor channel"
        vals.append(float(t))
    out = dict(n=len(vals), seeds=sorted(seeds), theta1=vals,
               mean=float(np.mean(vals)) if vals else None,
               registered_mean=REF_THETA1, flag="OK")
    if len(vals) == REF_N and out["mean"] is not None:
        rel = abs(out["mean"] - REF_THETA1) / REF_THETA1
        out["rel_dev_vs_registered"] = float(rel)
        if rel > 0.01:
            out["flag"] = (
                f"REF-MISMATCH: the full 8-run Stage-1 bundle means "
                f"{out['mean']:.6f}, {100 * rel:.2f}% from the "
                f"registered {REF_THETA1} — check the bundle identity "
                f"(report-only; the reference stays the registered "
                f"constant)")
    else:
        out["flag"] = f"PARTIAL ({len(vals)}/{REF_N} reference runs)"
    return out


# ----------------------------------------------------------------- aggregate

def _arm_primary(phi, rows):
    th = np.asarray([r["theta1"] for r in rows], np.float64)
    n = int(th.size)
    mean = float(th.mean())
    L_hat = mean / REF_THETA1
    lo, hi, s_log, se, _ = _log_ci(L_hat, th, extra_var=REF_RELSE ** 2,
                                   n=n)
    pred = dict(
        law_raw=LAMBDA_RAW[phi] / LAMBDA_RAW[REF_PHI],
        law_symlog=SYMLOG_MC[phi] / SYMLOG_MC[REF_PHI],
        rival_a=RIVAL_A[phi] / REF_THETA1,
        rival_b=RIVAL_B[phi] / REF_THETA1)
    inside = {k: _inside(lo, hi, v) for k, v in pred.items()}
    if not inside["law_raw"]:
        verdict = "LAW-KILLED"
    elif not inside["rival_a"] and not inside["rival_b"]:
        verdict = "LAW-CONFIRMED"
    else:
        verdict = "AMBIGUOUS"
    return dict(
        phi=phi, n=n, theta1=[float(x) for x in th],
        theta1_mean=mean, theta1_sd=(float(th.std(ddof=1))
                                     if n >= 2 else None),
        L_hat=float(L_hat), L_ci95=[lo, hi], s_log=s_log, se_log=se,
        predictions=pred, predictions_inside_ci=inside,
        verdict=verdict,
        absolute=dict(
            lambda_raw=LAMBDA_RAW[phi], law_symlog=SYMLOG_MC[phi],
            rival_a=RIVAL_A[phi], rival_b=RIVAL_B[phi],
            rival_b_extrapolated=RIVAL_B_EXTRAPOLATED[phi],
            theta1_over_lambda_raw=float(mean / LAMBDA_RAW[phi])))


def _convention(recs, stage1):
    """PRIMARY-2, pre-declared UNDER-POWERED (1.80 sigma)."""
    d = np.asarray([r["theta1"] / LAMBDA_RAW[r["phi"]] for r in recs],
                   np.float64)
    D = float(d.mean())
    lo, hi, s_log, se, n = _log_ci(D, d)
    raw_in, sym_in = _inside(lo, hi, CONV_RAW), _inside(lo, hi,
                                                        CONV_SYMLOG)
    cell = ("CONVENTION-RAW" if raw_in and not sym_in else
            "CONVENTION-SYMLOG" if sym_in and not raw_in else
            "CONVENTION-NOT-ADJUDICATED")
    out = dict(
        D=D, ci95=[lo, hi], n=n, s_log=s_log, se_log=se,
        predicted_raw=CONV_RAW, predicted_symlog=CONV_SYMLOG,
        raw_in_ci=raw_in, symlog_in_ci=sym_in, cell=cell,
        power_note=("PRE-DECLARED UNDER-POWERED: separation 4.16% "
                    "against a 2.31% SE = 1.80 sigma at n=8. "
                    "CONVENTION-NOT-ADJUDICATED is the registered "
                    "expected outcome and is NOT evidence for either "
                    "convention."))
    # SUPPLEMENTARY 16-run pool — declared supplementary because it
    # re-uses the arm that generated the observation
    if stage1["mean"] is not None and stage1["n"] >= 4:
        s1 = np.asarray(stage1["theta1"], np.float64) / \
            LAMBDA_RAW[REF_PHI]
        pooled = np.concatenate([d, s1])
        Dp = float(pooled.mean())
        plo, phi_, ps, pse, pn = _log_ci(Dp, pooled)
        out["supplementary_pooled"] = dict(
            D=Dp, ci95=[plo, phi_], n=pn,
            predicted_symlog_stage1_arm=CONV_SYMLOG_STAGE1,
            raw_in_ci=_inside(plo, phi_, CONV_RAW),
            symlog_in_ci=_inside(plo, phi_, CONV_SYMLOG),
            caveat=("SUPPLEMENTARY, never a fire: pools the Stage-1 "
                    "det arm, i.e. the very arm that generated the "
                    "theta_1 = Lambda observation. Its own symlog "
                    "offset is 0.95277, not 0.9584."))
    return out


def aggregate(recs, ref, stage1):
    seeds = sorted(r["train_seed_cfg"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert set(seeds) <= ALL_SEEDS, (
        f"seeds {sorted(set(seeds) - ALL_SEEDS)} outside 140-147")
    missing = sorted(ALL_SEEDS - set(seeds))
    out = dict(stamp=_stamp(), n_loaded=len(recs),
               missing_seeds=missing, stage1_reference=ref,
               stage1_theta1=stage1,
               registered=dict(
                   ref_theta1=REF_THETA1, ref_relse=REF_RELSE,
                   ref_perrun_cv=REF_PERRUN_CV,
                   lambda_raw=LAMBDA_RAW, symlog_mc=SYMLOG_MC,
                   rival_a=RIVAL_A, rival_b=RIVAL_B,
                   scales=SCALES, theta_ou=THETA_OU))

    collapsed = [r["run_dir"] for r in recs
                 if r["levels"]["real_mean"]
                 < ref["real_mean_min"] / COLLAPSE_FACTOR
                 or r["levels"]["intrinsic_mean"]
                 < ref["intrinsic_min"] / COLLAPSE_FACTOR]
    out["collapsed_runs"] = collapsed
    usable = [r for r in recs
              if r["valid"] and r["run_dir"] not in collapsed]
    per_arm_n = {str(p): sum(1 for r in usable if r["phi"] == p)
                 for p in PHIS}
    flagged = len(usable) < len(ALL_SEEDS)
    out["usable"] = dict(n=len(usable), per_arm=per_arm_n,
                         flagged=flagged)
    unverified = [r["run_dir"] for r in recs
                  if not r["newest_ckpt_checked"]
                  or r.get("ckpt_final_ok") is not True]
    out["probe_ckpt_flag"] = ("OK" if not unverified
                              else f"UNVERIFIED ({len(unverified)} runs)")
    floored = [r["run_dir"] for r in recs if r["floor"]["floor_binds"]]
    out["floored_runs"] = floored
    floored_lo = [r["run_dir"] for r in recs
                  if r["floor"]["floor_binds"] and r["phi"] == 0.80]
    out["floored_runs_low_phi"] = floored_lo

    # unconditional reporting (never gated on the primary)
    out["per_run"] = [
        {"run_dir": r["run_dir"], "phi": r["phi"],
         "theta_ou": r["theta_ou"], "scale": r["scale"],
         "train_seed": r["train_seed_cfg"], "valid": r["valid"],
         "theta1": r["theta1"],
         "p_perm": r["channels"]["distractor"]["p_perm"],
         "floor": r["floor"], "norm_guard": r["norm_guard"],
         "above_onset_consistent": r["above_onset_consistent"],
         "unnorm_distractor_var": r["unnorm_distractor_var"],
         "coverage": r["coverage"], "coverage_flag": r["coverage_flag"],
         "score_tail": r["score"],
         "ckpt_final_ok": r.get("ckpt_final_ok"),
         "newest_ckpt_checked": r["newest_ckpt_checked"],
         "real_mean_level": r["levels"]["real_mean"],
         "intrinsic_mean": r["levels"]["intrinsic_mean"]}
        for r in recs]
    out["onset_consistency_descriptive"] = {
        str(p): dict(
            n=per_arm_n[str(p)],
            above_onset=sum(1 for r in usable
                            if r["phi"] == p
                            and r["above_onset_consistent"]),
            note="prereg §4.3: DEFERRED secondary; descriptive only")
        for p in PHIS}

    # T6 normalization guard (prereg §5.10) — computed unconditionally
    arm_dpos = {}
    for p in PHIS:
        vals = [r["norm_guard"]["d_pos"] for r in usable
                if r["phi"] == p and r["norm_guard"]["d_pos"] is not None]
        arm_dpos[str(p)] = float(np.mean(vals)) if vals else None
    have = [v for v in arm_dpos.values() if v is not None]
    spread = (float((max(have) - min(have)) / np.mean(have))
              if len(have) == len(PHIS) and np.mean(have) > 0 else None)
    out["norm_guard"] = dict(
        d_pos_per_arm=arm_dpos, d_pos_registered=DPOS_REGISTERED,
        d_pos_arm_spread=spread, threshold=DPOS_SPREAD_MAX,
        confounded=bool(spread is not None and spread > DPOS_SPREAD_MAX),
        d_dist_per_arm={
            str(p): (float(np.mean([r["norm_guard"]["d_dist"]
                                    for r in usable if r["phi"] == p]))
                     if per_arm_n[str(p)] else None) for p in PHIS},
        unnorm_dist_over_pos_per_arm={
            str(p): (float(np.mean(
                [r["norm_guard"]["unnorm_dist_over_pos"]
                 for r in usable if r["phi"] == p
                 and r["norm_guard"]["unnorm_dist_over_pos"] is not None]))
                     if per_arm_n[str(p)] else None) for p in PHIS},
        note=("T6: theta_1's arm-invariance across reward arms was a "
              "cancellation (absolute ratio +1.64x against a -1.56x "
              "normalizer). Same masking could operate across phi."))

    # degeneracy / partial adjudication
    if len(collapsed) >= 2:
        out["primary"] = ("GLOBAL-COLLAPSE: >= 2 collapsed runs — the "
                          "phi law is not adjudicable")
        out["outcome_cell"] = "GLOBAL-COLLAPSE"
        return out
    if len(usable) < MIN_USABLE or \
            any(v < MIN_PER_ARM for v in per_arm_n.values()):
        out["primary"] = (f"NOT-ADJUDICABLE (usable {len(usable)} < "
                          f"{MIN_USABLE} or an arm below "
                          f"{MIN_PER_ARM})")
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        return out

    arms = {str(p): _arm_primary(p, [r for r in usable if r["phi"] == p])
            for p in PHIS}
    out["arms"] = arms
    verdicts = {k: v["verdict"] for k, v in arms.items()}

    if any(v == "LAW-KILLED" for v in verdicts.values()):
        cell = "LAW-REFUTED"
    elif all(v == "LAW-CONFIRMED" for v in verdicts.values()):
        cell = "LAW-CONFIRMED"
    elif any(v == "LAW-CONFIRMED" for v in verdicts.values()):
        cell = "LAW-SPLIT"
    else:
        cell = "NOT-ADJUDICABLE"

    # FLIPPED floor rule (prereg §5.9): a binding floor DEFLATES
    # theta_1. At phi=0.80 the law predicts the LOWEST value, so
    # deflation is anti-conservative there and kills a confirm; at
    # phi=0.95 it is conservative and the confirm stands, disclosed.
    if cell in ("LAW-CONFIRMED", "LAW-SPLIT") and floored_lo:
        out["primary"] = (
            f"NOT-ADJUDICABLE: a LAW-supporting outcome with a binding "
            f"normalizer floor on {len(floored_lo)} phi=0.80 run(s) "
            f"(deflation is anti-conservative for the arm whose "
            f"prediction is the lowest)")
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        out["arm_verdicts"] = verdicts
        return out

    statement = (
        "theta_1(distractor) scales as Lambda = 1/(1-phi^2) across "
        "phi in {0.80, 0.90, 0.95} at MATCHED dose-curve position "
        "(s/s0 = 3.6364); estimand = L(phi) = theta_1(phi)/5.257875, "
        "which is convention-free to <=0.64% (raw vs symlog)"
        if cell == "LAW-CONFIRMED" else
        "theta_1(distractor) does NOT follow Lambda = 1/(1-phi^2): "
        "the registered prediction falls outside the 95% interval in "
        "at least one arm"
        if cell == "LAW-REFUTED" else
        "the phi law is not cleanly separated from the rivals in "
        "every arm")
    if flagged:
        statement += (f"; FLAGGED SUBSET (n={len(usable)}, missing "
                      f"seeds {missing})")
    if floored:
        statement += (f"; normalizer floor binds on {len(floored)} run(s) "
                      f"(conservative direction only — see §5.9)")
    if out["norm_guard"]["confounded"]:
        statement += ("; NORMALIZATION-CONFOUNDED: the phi-free "
                      "instrument floor d_pos moves across arms by "
                      f"{100 * out['norm_guard']['d_pos_arm_spread']:.1f}% "
                      f"> {100 * DPOS_SPREAD_MAX:.1f}% — print the "
                      "absolute decomposition beside the ratio (T6)")
    out["primary"] = dict(statement=statement, arm_verdicts=verdicts,
                          fires=bool(cell == "LAW-CONFIRMED"))
    out["outcome_cell"] = cell
    out["convention_primary2"] = _convention(usable, stage1)
    out["not_claimed"] = [
        "no mechanism: Lambda still enters only as a level multiplying "
        "the phi-free constants d_pos and N_dist",
        "no promotion of any Tier-3 refuted item (sqrt(Lambda)~rho, "
        "theta_1=rho^2, s^1.9, c-bar, the space-invariance claim, the "
        "EVA provenance claim)",
        "no onset claim (prereg §4.3, DEFERRED)",
        "no cross-task / cross-objective / cross-dose generality",
        "the variance-ledger flagship is untouched in either direction"]
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
    out_json = os.path.join(args.output, "se_phisweep_read.json")
    assert not os.path.exists(out_json), (
        f"{out_json} exists — the read is ONE execution")
    ref = stage1_reference_gated(s1)
    stage1 = stage1_theta1(s1)
    # identity gates are FATAL: a defective run refuses the WHOLE read
    # instead of being silently dropped; the refusal happens before
    # out_json is written, so the one-execution guard survives a
    # repair-and-rerun
    recs = [read_run(d, args.n_perm) for d in dirs]
    out = aggregate(recs, ref, stage1)
    out["args"] = {k: v for k, v in vars(args).items()
                   if k != "selfcheck"}
    out["fit_counters"] = {r["run_dir"]: fit_counters(
        r["run_dir"], args.expect_steps) for r in recs}
    bad_fit = [rd for rd, fc in out["fit_counters"].items()
               if fc.get("flag") != "OK"]
    out["fit_flag"] = ("OK" if not bad_fit
                       else f"SUSPECT ({len(bad_fit)} runs)")
    if bad_fit and isinstance(out["primary"], dict):
        out["primary"]["statement"] += (
            f"; FIT-SUSPECT on {len(bad_fit)} runs (disclosure "
            f"carried, 7-Aug standing rule)")
    os.makedirs(args.output, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    if isinstance(out["primary"], dict):
        print(f"PHISWEEP PRIMARY-1: {out['outcome_cell']} "
              f"(fires={out['primary']['fires']}); "
              f"fit {out['fit_flag']}; ckpt {out['probe_ckpt_flag']}")
        for p in PHIS:
            a = out["arms"][str(p)]
            print(f"  phi={p}: theta1 mean {a['theta1_mean']:.4f} "
                  f"(n={a['n']}) L={a['L_hat']:.4f} "
                  f"CI[{a['L_ci95'][0]:.4f},{a['L_ci95'][1]:.4f}] "
                  f"vs law {a['predictions']['law_raw']:.4f} / "
                  f"rivalA {a['predictions']['rival_a']:.4f} / "
                  f"rivalB {a['predictions']['rival_b']:.4f} "
                  f"-> {a['verdict']}")
        c = out["convention_primary2"]
        print(f"  PRIMARY-2 (T3, under-powered): D={c['D']:.4f} "
              f"CI[{c['ci95'][0]:.4f},{c['ci95'][1]:.4f}] -> {c['cell']}")
    else:
        print(f"PHISWEEP PRIMARY-1: {out['outcome_cell']} — "
              f"{out['primary']}")
    return out


# ---------------------------------------------------------------- selfcheck

# realized Stage-1 det-arm deviations (theta_1_i / 5.257875), used as
# the fixtures' seed jitter so the planted panels carry the registered
# 6.533% per-run CV rather than an implausibly clean signal
JITTER = (0.896807, 1.066982, 0.993081, 1.021894,
          1.045153, 1.072752, 0.918516, 0.984716)


def _mult(seed, damp=1.0):
    return 1.0 + damp * (JITTER[(seed - 140) % 8] - 1.0)


def _fixture(tmp, name, seed, theta1_target, level_scale=1.0,
             intrinsic=3.5e-4, floor_binds=False, damp=1.0,
             theta_ou=None, scale=None, doctor_seed=None):
    """Plant a run at a chosen theta_1 for the seed's registered arm.
    Built on the reviewed B1 dose fixture (which already writes the
    ckpt/probe/replay/config tree the house readers consume); a binding
    floor is doctored into the npz the way the onset reader does."""
    from uncfield import se_dose_read as dr
    phi = SEED_ARM[seed]
    rd = dr._fixture(
        tmp, name, seed if doctor_seed is None else doctor_seed,
        dose=SCALES[phi] if scale is None else scale,
        theta=theta1_target * _mult(seed, damp),
        level_scale=level_scale, intrinsic=intrinsic,
        theta_ou=THETA_OU[phi] if theta_ou is None else theta_ou)
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

    with tempfile.TemporaryDirectory() as tmp:
        s1 = [dr._fixture(tmp, f"st{i}", 10 + i, dose=1.0,
                          theta=REF_THETA1 * JITTER[i])
              for i in range(4)]
        ref = stage1_reference_gated(s1)
        stg = stage1_theta1(s1)
        assert stg["flag"].startswith("PARTIAL"), stg["flag"]

        def wave(sub, target_of, **kw):
            return [_fixture(tmp, f"{sub}{sd}", sd,
                             target_of(SEED_ARM[sd]), **kw)
                    for sd in sorted(ALL_SEEDS)]

        def agg(runs, ref_=ref, stg_=stg):
            return aggregate([read_run(d, NP) for d in runs], ref_, stg_)

        # ---- H_LAW-raw planted -> LAW-CONFIRMED in both arms
        a = agg(wave("law", lambda p: LAMBDA_RAW[p]))
        assert a["outcome_cell"] == "LAW-CONFIRMED", (
            a["outcome_cell"], a.get("arms"))
        assert a["primary"]["fires"] is True
        for p in PHIS:
            arm = a["arms"][str(p)]
            assert arm["verdict"] == "LAW-CONFIRMED", arm
            assert arm["predictions_inside_ci"]["law_raw"]
            assert not arm["predictions_inside_ci"]["rival_a"]
            assert not arm["predictions_inside_ci"]["rival_b"]
        assert a["missing_seeds"] == []
        assert a["norm_guard"]["confounded"] is False
        assert "FLAGGED" not in a["primary"]["statement"]
        assert a["convention_primary2"]["cell"] == \
            "CONVENTION-NOT-ADJUDICATED", a["convention_primary2"]
        assert a["convention_primary2"]["raw_in_ci"] is True
        assert "supplementary_pooled" in a["convention_primary2"]
        # the DEFERRED onset secondary is descriptive and present
        assert a["onset_consistency_descriptive"]["0.8"]["above_onset"] \
            == 4

        # ---- H_RIVAL-A planted (flat 5.258) -> LAW-REFUTED
        a = agg(wave("ra", lambda p: RIVAL_A[p]))
        assert a["outcome_cell"] == "LAW-REFUTED", a["outcome_cell"]
        assert a["arms"]["0.8"]["verdict"] == "LAW-KILLED"
        assert a["arms"]["0.95"]["verdict"] == "LAW-KILLED"
        assert a["primary"]["fires"] is False

        # ---- H_RIVAL-B planted -> LAW-REFUTED
        a = agg(wave("rb", lambda p: RIVAL_B[p]))
        assert a["outcome_cell"] == "LAW-REFUTED", a["outcome_cell"]
        for p in PHIS:
            assert a["arms"][str(p)]["verdict"] == "LAW-KILLED"

        # ---- H_LAW-symlog planted -> PRIMARY-1 still LAW-CONFIRMED
        # (the ratio estimand is convention-free), and PRIMARY-2 lands
        # NOT-ADJUDICATED exactly as pre-declared at 1.80 sigma
        a = agg(wave("sy", lambda p: SYMLOG_MC[p]))
        assert a["outcome_cell"] == "LAW-CONFIRMED", (
            a["outcome_cell"], a["arms"])
        c = a["convention_primary2"]
        assert c["cell"] == "CONVENTION-NOT-ADJUDICATED", c
        assert c["raw_in_ci"] and c["symlog_in_ci"], c
        assert 0.94 < c["D"] < 0.98, c["D"]

        # ---- PRIMARY-2 code paths, forced by shrinking the scatter 5x
        # (a path test, not a power claim)
        a = agg(wave("cs", lambda p: SYMLOG_MC[p], damp=0.2))
        c = a["convention_primary2"]
        assert c["cell"] == "CONVENTION-SYMLOG", c
        a = agg(wave("cr", lambda p: LAMBDA_RAW[p], damp=0.2))
        c = a["convention_primary2"]
        assert c["cell"] == "CONVENTION-RAW", c

        # ---- FLIPPED floor rule, both directions
        runs = [_fixture(tmp, f"fl{sd}", sd, LAMBDA_RAW[SEED_ARM[sd]],
                         floor_binds=(sd == 140))
                for sd in sorted(ALL_SEEDS)]
        a = agg(runs)
        assert a["outcome_cell"] == "NOT-ADJUDICABLE", a["outcome_cell"]
        assert "anti-conservative" in a["primary"]
        runs = [_fixture(tmp, f"fh{sd}", sd, LAMBDA_RAW[SEED_ARM[sd]],
                         floor_binds=(sd == 144))
                for sd in sorted(ALL_SEEDS)]
        a = agg(runs)
        assert a["outcome_cell"] == "LAW-CONFIRMED", a["outcome_cell"]
        assert "normalizer floor binds on 1" in a["primary"]["statement"]

        # ---- T6 normalization guard: d_pos moved 1.0 vs 1.5 across
        # arms -> 40% spread -> annotated, NOT vetoed
        runs = [_fixture(tmp, f"t6{sd}", sd, LAMBDA_RAW[SEED_ARM[sd]],
                         level_scale=(1.0 if SEED_ARM[sd] == 0.80
                                      else 1.5))
                for sd in sorted(ALL_SEEDS)]
        a = agg(runs)
        assert a["outcome_cell"] == "LAW-CONFIRMED", a["outcome_cell"]
        assert a["norm_guard"]["confounded"] is True
        assert a["norm_guard"]["d_pos_arm_spread"] > DPOS_SPREAD_MAX
        assert "NORMALIZATION-CONFOUNDED" in a["primary"]["statement"]

        # ---- degeneracy + partial panels
        a = agg(wave("cl", lambda p: LAMBDA_RAW[p], level_scale=1e-6,
                     intrinsic=1e-9))
        assert a["outcome_cell"] == "GLOBAL-COLLAPSE", a["outcome_cell"]
        runs = wave("pa", lambda p: LAMBDA_RAW[p])
        a = agg(runs[:5])                    # arm 0.95 down to 1 run
        assert a["outcome_cell"] == "NOT-ADJUDICABLE", a["outcome_cell"]
        assert a["missing_seeds"] == [145, 146, 147]
        # 7 runs, 3 in one arm -> fires but FLAGGED
        runs = wave("fs", lambda p: LAMBDA_RAW[p])
        a = agg(runs[:7])
        assert a["outcome_cell"] == "LAW-CONFIRMED", a["outcome_cell"]
        assert a["missing_seeds"] == [147]
        assert "FLAGGED SUBSET" in a["primary"]["statement"]
        assert a["arms"]["0.95"]["n"] == 3

        # ---- FATAL identity gates
        rd = _fixture(tmp, "g1", 140, 2.8, theta_ou=0.05)
        _expect_fail(lambda: read_run(rd, NP), "the phi axis did not bind")
        rd = _fixture(tmp, "g2", 144, 10.0, scale=1.0)
        _expect_fail(lambda: read_run(rd, NP),
                     "the dose position is not matched")
        # a seed outside the registered blocks
        rd = _fixture(tmp, "g3", 140, 2.8, doctor_seed=99)
        _expect_fail(lambda: read_run(rd, NP),
                     "not a registered phisweep seed")
        # a phi=0.95 run carrying a phi=0.80 seed: the SEED decides the
        # arm, so theta/scale must disagree and the gate must fire
        rd = _fixture(tmp, "g4", 144, 10.0, doctor_seed=140)
        _expect_fail(lambda: read_run(rd, NP), "the phi axis did not bind")
        rd = _fixture(tmp, "g5", 141, 2.8)
        cfgp = os.path.join(rd, "config.yaml")
        cfg = open(cfgp).read()
        open(cfgp, "w").write(cfg.replace("disag_bootstrap: true",
                                          "disag_bootstrap: false"))
        _expect_fail(lambda: read_run(rd, NP), "expl pins")
        rd = _fixture(tmp, "g6", 142, 2.8)
        cfgp = os.path.join(rd, "config.yaml")
        cfg = open(cfgp).read()
        assert "penalty_mix" not in cfg
        open(cfgp, "w").write(cfg.replace(
            "mode: p2e", "mode: p2e\n    penalty_mix: true"))
        _expect_fail(lambda: read_run(rd, NP), "penalty machinery")
        rd = _fixture(tmp, "g7", 143, 2.8)
        ck = os.path.join(rd, "ckpt")
        assert [x for x in os.listdir(ck)
                if os.path.isdir(os.path.join(ck, x))]
        os.makedirs(os.path.join(ck, "zz_newer"))
        _expect_fail(lambda: read_run(rd, NP), "stale ckpt/latest")

        # ---- REF-MISMATCH provenance check on a full 8-run reference
        s8_ok = [dr._fixture(tmp, f"r{i}", 10 + i, dose=1.0,
                             theta=REF_THETA1 * JITTER[i])
                 for i in range(8)]
        assert stage1_theta1(s8_ok)["flag"] == "OK", \
            stage1_theta1(s8_ok)["flag"]
        s8_bad = [dr._fixture(tmp, f"q{i}", 10 + i, dose=1.0,
                              theta=7.0 * JITTER[i]) for i in range(8)]
        assert "REF-MISMATCH" in stage1_theta1(s8_bad)["flag"]

        # ---- end-to-end + ONE-execution guard + FATAL glob path
        outd = os.path.join(tmp, "out")
        wave("e2e", lambda p: LAMBDA_RAW[p])
        argv = ["--runs", os.path.join(tmp, "e2e14[0-7]"),
                "--stage1_runs", os.path.join(tmp, "st[0-9]"),
                "--n_perm", str(NP), "--output", outd]
        out = run(parse_args(argv))
        assert out["outcome_cell"] == "LAW-CONFIRMED"
        assert out["args"]["n_perm"] == NP
        assert out["fit_flag"] == "OK", out["fit_flag"]
        _expect_fail(lambda: run(parse_args(argv)), "ONE execution")
        _fixture(tmp, "e2e999", 140, 2.8, theta_ou=0.05,
                 doctor_seed=140)
        os.rename(os.path.join(tmp, "e2e999"),
                  os.path.join(tmp, "e2e148"))
        _expect_fail(lambda: run(parse_args(
            ["--runs", os.path.join(tmp, "e2e14[0-8]"),
             "--stage1_runs", os.path.join(tmp, "st[0-9]"),
             "--n_perm", str(NP),
             "--output", os.path.join(tmp, "out2")])),
            "the phi axis did not bind")

    print("se_phisweep_read selfcheck PASS (LAW planted -> "
          "LAW-CONFIRMED both arms; RIVAL-A and RIVAL-B planted -> "
          "LAW-REFUTED; SYMLOG planted -> PRIMARY-1 still CONFIRMED "
          "(ratio is convention-free) with PRIMARY-2 "
          "NOT-ADJUDICATED as pre-declared; PRIMARY-2 RAW/SYMLOG "
          "code paths; FLIPPED floor rule both directions; T6 "
          "normalization guard annotates without vetoing; "
          "GLOBAL-COLLAPSE; partial NOT-ADJUDICABLE; missing-seed "
          "FLAG; theta/scale/seed-map/expl/penalty/stale-ckpt gates; "
          "REF-MISMATCH; end-to-end + one-execution + FATAL glob)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
