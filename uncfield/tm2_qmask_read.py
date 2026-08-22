"""B5 TD-MPC2-column FROZEN reader
(PREREG_trackB_tm2_20260822.md §3; built + selfchecked BEFORE the
wave's compute; ONE execution; revised pre-freeze per review R4).

8 TM2 trainings on the SE substrate (cheetah + dose 'se' + planted
suite), seeds 132-139; per run the tm2_qmask instrument (Q-ensemble
std at frozen baseline actions, house mask suite).

REV 2 (R-A1-M12): the distractor is a pure exogenous AR(1), so
batch permutation is law-preserving — its population delta is ZERO
BY EXCHANGEABILITY for any statistic and it cannot be a fire
channel. The FIRE channel is now the law-changing
distractor MEAN-SUBSTITUTION; the permutation row is retained as a
built-in EXCHANGEABILITY-NULL calibration (materially nonzero =
instrument defect, flagged); velocity MEAN-SUBSTITUTION is the
form-matched specificity comparator; velocity permutation stays as
the coupling teeth.

PRIMARY (single fire channel, no BH; APT-read form): fires iff
channels["distractor_meansub"]["delta_qstd_mean"] < 0 in >= 7 of
the 8 REGISTERED runs (binomial 9/256; missing/defective runs count
AGAINST, denominator never shrinks). Outcome cells:
  i-TM2-PRICES-DISTRACTOR  — fire AND channel-specific;
  ii-TM2-IMMUNE            — no fire, >= 6 loaded;
  iii-TM2-NONSPECIFIC      — fire, but velocity MEAN-SUBSTITUTION
      moves the same way at comparable magnitude in a majority of
      loaded runs ("any off-manifold substitution lowers this
      statistic" explains the evidence);
  iv-TM2-PRICES-INFLATING  — delta > 0 in >= 7/8 (the sign under
      the intervention is not theoretically pinned; systematic
      inflation is pricing, not immunity);
  NO-EVIDENCE              — fewer than 6 loaded runs and no fire.
D-family + resample nulls + the exchangeability-null row stay
DESCRIPTIVE.

Run:  python -m uncfield.tm2_qmask_read --runs "<glob>" --output <d>
      python -m uncfield.tm2_qmask_read --selfcheck
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

B5_SEEDS = set(range(132, 140))
N_REGISTERED = 8
FIRE_MIN_NEG = 7
MIN_LOADED_FOR_IMMUNE = 6
S_PIN = 512
TASK_PIN = "dmc_cheetah_run"
NUM_Q_PIN = 5
STEPS_PIN = 100_000
CKPT_PIN = "ckpt_late.pt"
FIRE_CHANNEL = "distractor_meansub"
SPECIFICITY_CHANNEL = "velocity_meansub"
EXCH_NULL_CHANNEL = "distractor"     # inert by construction (M12)
NONSPEC_RATIO = 0.5
REQUIRED = ("distractor", "distractor_meansub", "velocity_meansub",
            "planted_dup0", "planted_dup1",
            "planted_dup2", "planted_dup1_resample",
            "planted_dup2_resample", "velocity_control")
FORM_PIN = {"distractor": "batch_permutation",
            "distractor_meansub": "mean_substitution",
            "velocity_meansub": "mean_substitution",
            "planted_dup0": "source_substitution",
            "planted_dup1": "source_substitution",
            "planted_dup2": "source_substitution",
            "planted_dup1_resample": "fresh_resample",
            "planted_dup2_resample": "fresh_resample",
            "velocity_control": "batch_permutation_real_key_control"}
PROV_RTOL = 1e-12
EXTRA_PIN = ["distractor", "planted_dup0", "planted_dup1",
             "planted_dup2", "planted_const"]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="")
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
    with open(os.path.join(run_dir, "config.json")) as f:
        audit = json.load(f)
    assert str(audit["task"]) == TASK_PIN, (
        f"{run_dir}: task {audit['task']!r}")
    assert str(audit["dose"]) == "se" and audit.get("planted"), (
        f"{run_dir}: not the registered SE+planted configuration")
    assert list(audit.get("extra_keys", [])) == EXTRA_PIN, (
        f"{run_dir}: extra-key order {audit.get('extra_keys')} != "
        f"pinned {EXTRA_PIN} — the flat layout would differ")
    assert int(audit["num_q"]) == NUM_Q_PIN, (
        f"{run_dir}: checkpoint-side num_q {audit['num_q']}")
    # realized training (7-Aug standing rule): a short-stepped run
    # must not slip in as a full one (R4-M6)
    assert int(audit["steps"]) == STEPS_PIN and \
        int(audit["late_step_realized"] or 0) >= STEPS_PIN, (
        f"{run_dir}: realized training "
        f"{audit.get('late_step_realized')} < registered {STEPS_PIN}")
    train_seed = int(audit["seed"])
    assert train_seed in B5_SEEDS, (
        f"{run_dir}: seed {train_seed} outside the registered "
        f"132-139")
    assert os.path.exists(os.path.join(run_dir,
                                       "TM2R3_TRAIN_DONE")), (
        f"{run_dir}: training done-marker absent")
    mdir = os.path.join(run_dir, "tm2_qmask")
    with open(os.path.join(mdir, "tm2_qmask.json")) as f:
        mj = json.load(f)
    npz = np.load(os.path.join(mdir, "tm2_qmask.npz"))
    assert str(mj["ckpt"]) == CKPT_PIN, (
        f"{run_dir}: instrument ran on {mj['ckpt']!r}, registered "
        f"{CKPT_PIN}")
    assert str(mj["dose"]) == "se", f"{run_dir}: instrument dose"
    assert list(mj["extra_keys"]) == EXTRA_PIN, (
        f"{run_dir}: instrument extra-key order drifted from the "
        f"pinned layout")
    assert mj.get("mpc") is True, (
        f"{run_dir}: instrument mpc flag absent/false — planner "
        f"identity unverified")
    assert int(mj["n_eval"]) == S_PIN, (
        f"{run_dir}: S {mj['n_eval']} != {S_PIN}")
    assert int(mj["train_seed"]) == train_seed
    assert str(mj["task"]) == TASK_PIN and mj["planted"]
    assert int(mj["num_q"]) == NUM_Q_PIN, (
        f"{run_dir}: num_q {mj['num_q']} != pinned {NUM_Q_PIN}")
    c0 = mj["channels"]["planted_dup0"]
    assert c0.get("bitwise_noop") and c0["delta_qstd_mean"] == 0.0, (
        f"{run_dir}: dup0 not the exact-zero anchor")
    d0 = np.asarray(npz["delta_qstd_planted_dup0"])
    assert np.all(d0 == 0.0), f"{run_dir}: dup0 vector not zero"
    for ch in REQUIRED:
        assert ch in mj["channels"], f"{run_dir}: {ch} missing"
        assert mj["channels"][ch]["form"] == FORM_PIN[ch], (
            f"{run_dir}/{ch}: form {mj['channels'][ch]['form']!r} != "
            f"registered {FORM_PIN[ch]!r}")
        dvec = np.asarray(npz[f"delta_qstd_{ch}"], np.float64)
        assert dvec.size == S_PIN and np.all(np.isfinite(dvec)), (
            run_dir, ch)
        rc = float(dvec.mean())
        stored = float(mj["channels"][ch]["delta_qstd_mean"])
        assert abs(rc - stored) <= PROV_RTOL * max(1.0, abs(rc)), (
            f"{run_dir}/{ch}: provenance mismatch")
    return dict(run_dir=os.path.abspath(run_dir),
                train_seed=train_seed,
                base_qstd=float(mj["base_qstd_mean"]),
                channels={ch: dict(mj["channels"][ch])
                          for ch in REQUIRED})


def aggregate(recs, excluded):
    seeds = sorted(r["train_seed"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert set(seeds) <= B5_SEEDS
    assert len(recs) + len(excluded) <= N_REGISTERED
    n_loaded = len(recs)
    out = dict(stamp=_stamp(), n_registered=N_REGISTERED,
               n_loaded=n_loaded, excluded_runs=excluded,
               partial_flag=("COMPLETE" if n_loaded == N_REGISTERED
                             else f"PARTIAL ({n_loaded}/"
                                  f"{N_REGISTERED} — missing runs "
                                  f"count AGAINST the fire)"))
    out["per_run"] = [
        {"run_dir": r["run_dir"], "train_seed": r["train_seed"],
         "base_qstd": r["base_qstd"], "channels": r["channels"]}
        for r in recs]
    deltas = [r["channels"][FIRE_CHANNEL]["delta_qstd_mean"]
              for r in recs]
    n_neg = sum(1 for x in deltas if x < 0)
    n_pos = sum(1 for x in deltas if x > 0)
    fires = bool(n_neg >= FIRE_MIN_NEG)
    inflates = bool(n_pos >= FIRE_MIN_NEG)
    p_att = (sum(math.comb(n_loaded, k)
                 for k in range(n_neg, n_loaded + 1)) / 2 ** n_loaded
             if n_loaded else 1.0)
    out["primary"] = dict(
        statement="TD-MPC2's Q-ensemble std prices the planted "
                  "distractor (rev 2: MEAN-SUBSTITUTION fire "
                  "channel — permutation is inert by "
                  "exchangeability, R-A1-M12): delta < 0 in >= 7/8 "
                  "registered runs; single channel, no BH; missing "
                  "runs count against; specificity + inflation "
                  "cells per the registered outcome map",
        per_run_delta=deltas, n_neg=n_neg, n_pos=n_pos,
        n_registered=N_REGISTERED, n_loaded=n_loaded, fires=fires,
        binom_p_registered_7of8=9 / 256, binom_p_attained=p_att)
    vel = [r["channels"]["velocity_control"]["delta_qstd_mean"]
           for r in recs]
    # specificity condition (rev 2): the FORM-MATCHED velocity
    # mean-substitution "explains" the fire when it moves the same
    # way at >= half the magnitude
    nonspec_runs = sum(
        1 for r in recs
        if r["channels"][SPECIFICITY_CHANNEL]["delta_qstd_mean"] < 0
        and abs(r["channels"][SPECIFICITY_CHANNEL]
                ["delta_qstd_mean"])
        > NONSPEC_RATIO
        * abs(r["channels"][FIRE_CHANNEL]["delta_qstd_mean"]))
    nonspecific = bool(n_loaded and nonspec_runs > n_loaded / 2)
    out["specificity"] = dict(
        channel=SPECIFICITY_CHANNEL,
        nonspec_runs=nonspec_runs, n_loaded=n_loaded,
        ratio_threshold=NONSPEC_RATIO, nonspecific=nonspecific)
    # exchangeability-null calibration (R-A1-M12): the distractor
    # PERMUTATION row is inert by construction — flag material
    # deviations as instrument defects (report-only)
    exch = [r["channels"][EXCH_NULL_CHANNEL]["delta_qstd_mean"]
            for r in recs]
    exch_flags = [r["run_dir"] for r in recs
                  if abs(r["channels"][EXCH_NULL_CHANNEL]
                         ["delta_qstd_mean"])
                  > 0.5 * abs(r["channels"][FIRE_CHANNEL]
                              ["delta_qstd_mean"]) + 1e-12
                  and (r["channels"][EXCH_NULL_CHANNEL]
                       ["delta_qstd_bca"][0] > 0
                       or r["channels"][EXCH_NULL_CHANNEL]
                       ["delta_qstd_bca"][1] < 0)]
    out["exchangeability_null"] = dict(
        per_run=exch, defect_flags=exch_flags,
        note="population delta 0 by construction (exogenous "
             "channel); a BCa-separated, fire-comparable value "
             "here indicates an instrument defect")
    out["descriptive"] = dict(
        velocity_control=dict(per_run=vel,
                              mean=(float(np.mean(vel)) if vel
                                    else None),
                              n_neg=sum(1 for x in vel if x < 0),
                              note="the coupling teeth row "
                                   "(velocity is state-coupled)"),
        velocity_meansub=[
            r["channels"][SPECIFICITY_CHANNEL]["delta_qstd_mean"]
            for r in recs],
        dup_substitution={
            ch: [r["channels"][ch]["delta_qstd_mean"] for r in recs]
            for ch in ("planted_dup1", "planted_dup2")},
        dup_resample={
            ch: [r["channels"][f"{ch}_resample"]["delta_qstd_mean"]
                 for r in recs]
            for ch in ("planted_dup1", "planted_dup2")})
    if fires and nonspecific:
        cell = "iii-TM2-NONSPECIFIC"
    elif fires:
        cell = "i-TM2-PRICES-DISTRACTOR"
    elif inflates:
        cell = "iv-TM2-PRICES-INFLATING"
    elif n_loaded < MIN_LOADED_FOR_IMMUNE:
        cell = f"NO-EVIDENCE (n_loaded {n_loaded} < " \
               f"{MIN_LOADED_FOR_IMMUNE})"
    else:
        cell = "ii-TM2-IMMUNE"
    out["outcome_cell"] = cell
    return out


def run(args):
    dirs = (sorted(globmod.glob(args.runs)) if "," not in args.runs
            else [x.strip() for x in args.runs.split(",")
                  if x.strip()])
    assert dirs, f"no runs matched {args.runs!r}"
    assert args.output and args.output != "TBD", (
        "registered read must persist: pass an explicit --output")
    out_json = os.path.join(args.output, "tm2_qmask_read.json")
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
    os.makedirs(args.output, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    pr = out["primary"]
    print(f"TM2 PRIMARY: {pr['n_neg']}/{pr['n_registered']} negative "
          f"({pr['n_pos']} positive) -> fires={pr['fires']} "
          f"({out['outcome_cell']}); velocity mean "
          f"{out['descriptive']['velocity_control']['mean']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, d_dist=-0.02, d_vel=-0.005,
             d_exch=0.0, d_velperm=-0.03,
             dup0_zero=True, extra=None, num_q=NUM_Q_PIN,
             doctor=None, doctor_audit=None, drop_npz=False):
    """rev 2: d_dist = the MEAN-SUBSTITUTION fire channel; d_vel =
    the form-matched velocity_meansub comparator; d_exch = the
    exchangeability-null permutation row (default 0 — inert by
    construction); d_velperm = the coupling teeth."""
    rng = np.random.default_rng(seed * 5 + 1)
    rd = os.path.join(tmp, name)
    mdir = os.path.join(rd, "tm2_qmask")
    os.makedirs(mdir, exist_ok=True)
    audit = dict(task=TASK_PIN, dose="se", planted=True, seed=seed,
                 steps=STEPS_PIN, late_step_realized=STEPS_PIN + 371,
                 num_q=NUM_Q_PIN,
                 extra_keys=(extra or EXTRA_PIN))
    if doctor_audit:
        doctor_audit(audit)
    with open(os.path.join(rd, "config.json"), "w") as f:
        json.dump(audit, f)
    open(os.path.join(rd, "TM2R3_TRAIN_DONE"), "w").close()
    spec = {"distractor": d_exch,
            "distractor_meansub": d_dist,
            "velocity_meansub": d_vel,
            "planted_dup0": 0.0,
            "planted_dup1": -1e-4, "planted_dup2": 1e-4,
            "planted_dup1_resample": 1e-4,
            "planted_dup2_resample": -1e-4,
            "velocity_control": d_velperm}
    chans, dnpz = {}, {}
    for ch, want in spec.items():
        if ch == "planted_dup0" and dup0_zero:
            dv = np.zeros(S_PIN)
        else:
            dv = rng.normal(want, max(abs(want), 1e-4) * 0.1, S_PIN)
            dv = dv - dv.mean() + want
        rec = dict(form=FORM_PIN[ch],
                   delta_qstd_mean=float(dv.mean()),
                   delta_qstd_bca=[float(dv.mean()) - 1,
                                   float(dv.mean()) + 1],
                   p_reduce=0.01, p_inflate=0.99)
        if ch == "planted_dup0":
            rec["bitwise_noop"] = dup0_zero
        chans[ch] = rec
        dnpz[f"delta_qstd_{ch}"] = dv.astype(np.float64)
    mj = dict(run_logdir=rd, ckpt=CKPT_PIN, n_eval=S_PIN,
              seed=0, train_seed=seed, task=TASK_PIN, dose="se",
              planted=True, num_q=num_q, mpc=True, dropout=0.01,
              base_qstd_mean=0.4, extra_keys=EXTRA_PIN,
              anchor_sha256="f" * 64, channels=chans)
    if doctor:
        doctor(mj)
    with open(os.path.join(mdir, "tm2_qmask.json"), "w") as f:
        json.dump(mj, f)
    if not drop_npz:
        np.savez(os.path.join(mdir, "tm2_qmask.npz"),
                 base_qstd=np.full(S_PIN, 0.4), **dnpz)
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
    with tempfile.TemporaryDirectory() as tmp:
        # fire 8/8, specific (velocity_meansub small), the
        # exchangeability row inert and unflagged
        runs = [_fixture(tmp, f"a{s}", s) for s in range(132, 140)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "i-TM2-PRICES-DISTRACTOR"
        assert agg["primary"]["binom_p_attained"] == 1 / 256
        assert not agg["specificity"]["nonspecific"]
        assert not agg["exchangeability_null"]["defect_flags"]
        # ...and a BCa-separated, fire-comparable exchangeability
        # value flags an instrument defect (report-only)
        def doc_exch(mj):
            c = mj["channels"]["distractor"]
            c["delta_qstd_mean"] = -0.019
            c["delta_qstd_bca"] = [-0.021, -0.017]
        runs = [_fixture(tmp, f"x{s}", s,
                         doctor=(doc_exch if s == 132 else None))
                for s in range(132, 140)]
        # doctored mean breaks npz provenance -> rebuild npz row
        import numpy as _np
        xz = os.path.join(tmp, "x132", "tm2_qmask", "tm2_qmask.npz")
        z = dict(_np.load(xz))
        z["delta_qstd_distractor"] = _np.full(S_PIN, -0.019)
        _np.savez(xz, **z)
        agg = aggregate([read_run(d) for d in runs], [])
        assert len(agg["exchangeability_null"]["defect_flags"]) == 1
        # fire but NONSPECIFIC (velocity moves too, comparable size)
        runs = [_fixture(tmp, f"n{s}", s, d_vel=-0.03)
                for s in range(132, 140)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "iii-TM2-NONSPECIFIC", agg[
            "outcome_cell"]
        assert agg["specificity"]["nonspec_runs"] == 8
        # immune: mixed signs (4 neg / 4 pos — neither tail)
        runs = [_fixture(tmp, f"b{s}", s,
                         d_dist=(0.002 if s % 2 else -0.002))
                for s in range(132, 140)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "ii-TM2-IMMUNE", agg[
            "outcome_cell"]
        # INFLATING 8/8 positive
        runs = [_fixture(tmp, f"i{s}", s, d_dist=0.02)
                for s in range(132, 140)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == "iv-TM2-PRICES-INFLATING"
        # conservative missing: 7 loaded all-neg fires
        runs = [_fixture(tmp, f"c{s}", s) for s in range(132, 139)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["primary"]["fires"] and \
            "PARTIAL" in agg["partial_flag"]
        # NO-EVIDENCE floor: 5 loaded, no fire possible to claim
        # immunity from
        runs = [_fixture(tmp, f"v{s}", s, d_dist=0.001)
                for s in range(132, 137)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"].startswith("NO-EVIDENCE"), agg[
            "outcome_cell"]
        # ...and zero loaded
        agg = aggregate([], [dict(run_dir="x", error="y")] * 8)
        assert agg["outcome_cell"].startswith("NO-EVIDENCE")
        # gates
        rd = _fixture(tmp, "g0", 132,
                      extra=["distractor", "planted_dup0"])
        _expect_fail(lambda: read_run(rd), "extra-key order")
        rd = _fixture(tmp, "g1", 132, num_q=2)
        _expect_fail(lambda: read_run(rd), "num_q")
        rd = _fixture(tmp, "g2", 132, dup0_zero=False)
        _expect_fail(lambda: read_run(rd), "exact-zero")
        rd = _fixture(tmp, "g3", 99)
        _expect_fail(lambda: read_run(rd), "registered 132-139")

        def doc(mj):
            mj["channels"]["distractor"]["delta_qstd_mean"] = -9.9
        rd = _fixture(tmp, "g4", 133, doctor=doc)
        _expect_fail(lambda: read_run(rd), "provenance")
        # new pins (R4-M6): wrong ckpt / truncated training /
        # instrument-side key drift / mpc flag / form drift
        rd = _fixture(tmp, "g5", 133,
                      doctor=lambda mj: mj.update(
                          ckpt="ckpt_early.pt"))
        _expect_fail(lambda: read_run(rd), "registered ckpt_late")
        rd = _fixture(tmp, "g6", 133,
                      doctor_audit=lambda a: a.update(
                          late_step_realized=50_000))
        _expect_fail(lambda: read_run(rd), "realized training")
        rd = _fixture(tmp, "g7", 133,
                      doctor=lambda mj: mj.update(
                          extra_keys=EXTRA_PIN[::-1]))
        _expect_fail(lambda: read_run(rd), "instrument extra-key")
        rd = _fixture(tmp, "g8", 133,
                      doctor=lambda mj: mj.pop("mpc"))
        _expect_fail(lambda: read_run(rd), "mpc flag")

        def doc_form(mj):
            mj["channels"]["velocity_control"]["form"] = \
                "fresh_resample"
        rd = _fixture(tmp, "g9", 133, doctor=doc_form)
        _expect_fail(lambda: read_run(rd), "form")
        # defective run -> excluded via run(); one-execution guard
        outd = os.path.join(tmp, "out")
        for s in range(132, 140):
            _fixture(tmp, f"e{s}", s, drop_npz=(s == 132))
        out = run(parse_args(["--runs", os.path.join(tmp, "e13*"),
                              "--output", outd]))
        assert len(out["excluded_runs"]) == 1
        assert out["primary"]["fires"]        # 7 neg of 8 registered
        _expect_fail(lambda: run(parse_args(
            ["--runs", os.path.join(tmp, "e13*"),
             "--output", outd])), "ONE execution")
    print("tm2_qmask_read selfcheck PASS (rev 2 meansub fire "
          "channel: fire-specific/NONSPECIFIC/immune/INFLATING/"
          "conservative-missing/NO-EVIDENCE cells; exchangeability-"
          "null inert + defect flag; extra-key/num_q/dup0/seed/"
          "provenance/ckpt/realized-steps/instrument-drift/mpc/form "
          "gates; excluded persistence + one-execution guard)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
