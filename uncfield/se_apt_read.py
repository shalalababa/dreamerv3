"""APT-column (Track B2) FROZEN reader + smoke gate
(PREREG_nfi_apt_20260821.md §3; built + selfchecked BEFORE the B2
read — the registered build-before-read; review #30 adjudicated
21 Aug: 2B/9M/8m ALL applied).

Registered read (ONE execution, --runs mode):
  PER-RUN STATISTIC (pinned verbatim, #29 C-M2):
      channels["distractor"]["delta_apt_mean"] < 0
  read from <run>/se_apt_mask/se_apt_mask.json, provenance-checked
  against the paired npz (float64 mean, rel tol 1e-12, exact-match
  recorded; finiteness asserted separately — #30 M5). NaN and -0.0
  count as NON-negative (conservative, per the verbatim statistic).
  CROSS-RUN PRIMARY — SINGLE fire channel, no BH: fires iff negative
  in >= 7 of the 8 REGISTERED runs (exact binomial p = 9/256).
  A missing OR DEFECTIVE run counts AGAINST the fire and never
  shrinks the denominator (#30 B1): read_run failures are caught,
  recorded in invalid_runs, and the read persists on every path.
  D-family: DESCRIPTIVE ONLY (#29 C-B1). Anchor-level p/BCa are
  DIAGNOSTIC, NOT CALIBRATED (#29 C-M4).
  Outcome cells: i-DISTRACTOR-FIRES / ii-APT-IMMUNE, plus a
  "+SPREAD-RESCOPED" suffix for prereg cell (iii) — resample
  controls SYSTEMATICALLY NEGATIVE: per-run trigger (res mean < 0,
  |res| > 0.5x that run's |distractor delta|, |res| > 2 SE) in a
  majority of loaded runs (#30 M2/M3; per-run reference + SE floor
  kill the cross-run-mean cancellation artifact). Positive-signed
  resample anomalies are reported separately, never as cell (iii).
  Fire rules: --fire_rule sign (default, registered) or
  control_relative (the REGISTERED AMENDMENT comparator, #30 M8:
  per-run statistic becomes distractor delta MINUS the mean of the
  two fresh-resample deltas < 0) — control_relative may be used only
  after the smoke's amendment trigger, per prereg §3.

Smoke gate (--smoke_gate mode; NOT the registered read): the #29
C-M7 criteria incl. run-completion evidence (#30 B2: fit counters
must be OK — TRAINING_DONE is written unconditionally by the sbatch
and is NOT completion evidence), FAIL-verdict-not-crash on defective
runs (#30 M1), and the resample-sign amendment trigger (registered
form: ANY |resample mean| > 0.5x |distractor|, sign-agnostic —
deliberately more conservative than the cell-(iii) flag; the two
serve different registered purposes).

Run:  python -m uncfield.se_apt_read --runs "<glob>" --output <dir>
      python -m uncfield.se_apt_read --smoke_gate <run_dir>
      python -m uncfield.se_apt_read --selfcheck
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

from uncfield.se_probe import FIRE_KEYS, resolve_ckpt
from uncfield.se_read import fit_counters

APT_SEEDS = set(range(68, 76))     # se_apt_s68..75
N_REGISTERED = 8
S_PIN = 512                        # #29 C-M5 hard pin
APT_KNN = 12
APT_LOGC = 1.0
TASK_PIN = "dmc_cheetah_run"
FIRE_CHANNEL = "distractor"
FIRE_MIN_NEG = 7                   # of 8 registered; binomial 9/256
SPREAD_MATERIALITY = 0.5
DUP_CHANNELS = ("planted_dup1", "planted_dup2")
RESAMPLE_CHANNELS = ("planted_dup1_resample", "planted_dup2_resample")
LIVE_KEYS = tuple(FIRE_KEYS) + ("planted_const",)   # #30 m4
REQUIRED_CHANNELS = tuple(FIRE_KEYS) + RESAMPLE_CHANNELS + (
    "velocity_control",)
PROV_RTOL = 1e-12                  # #30 M5 cross-host ulp tolerance


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="")
    p.add_argument("--smoke_gate", default="",
                   help="mechanical smoke-gate mode on ONE run dir "
                        "(never the registered read)")
    p.add_argument("--fire_rule", default="sign",
                   choices=("sign", "control_relative"),
                   help="control_relative = the REGISTERED amendment "
                        "comparator; use only after the smoke's "
                        "amendment trigger (prereg §3)")
    p.add_argument("--expect_steps", type=float, default=5e5)
    p.add_argument("--smoke_steps", type=float, default=2e4)
    p.add_argument("--output", default="")
    p.add_argument("--allow_seed_any", action="store_true",
                   help="smoke/selfcheck only (the smoke run is seed "
                        "9 by registration); rejected in --runs mode")
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


def _load_mask(run_dir):
    mdir = os.path.join(run_dir, "se_apt_mask")
    with open(os.path.join(mdir, "se_apt_mask.json")) as f:
        mj = json.load(f)
    npz = np.load(os.path.join(mdir, "se_apt_mask.npz"))
    return mj, npz


def read_run(run_dir, allow_seed_any=False):
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = cfg["distractor"]
    pl = cfg["planted"]
    ex = cfg["agent"]["expl"]
    # objective + knob pins (#29 C-m8/m9). The saved config carries
    # inert disag_* keys with NO ensemble built — the reader must not
    # gate on them (#29 C-m12), so no disag pins here.
    assert str(ex["mode"]) == "apt", (
        f"{run_dir}: expl.mode={ex['mode']!r}, not apt")
    assert int(ex["apt_knn"]) == APT_KNN and \
        abs(float(ex["apt_logc"]) - APT_LOGC) < 1e-12, (
        f"{run_dir}: apt knobs differ from registration")
    assert str(cfg["task"]) == TASK_PIN, (
        f"{run_dir}: task {cfg['task']!r} != pinned {TASK_PIN!r}")
    # Stage-1 environment pins (planted channels + doses unchanged)
    assert int(d["dim"]) == 8 and abs(float(d["basesd"]) - 1.215) < 1e-9
    assert abs(float(d["scale"]) - 1.0) < 1e-9, (
        f"{run_dir}: distractor scale is not Stage-1's 1.0")
    assert str(d.get("gate_key", "")) == "" and \
        str(d.get("mod_key", "")) == "", (
        f"{run_dir}: gates/mod must be off")
    assert str(pl.get("gate_key", "")) == ""
    assert str(pl["source_key"]) == "position" and \
        abs(float(pl["basesd"]) - 0.0976) < 1e-9, (
        f"{run_dir}: planted pins differ from Stage-1")
    train_seed = int(cfg["seed"])
    if not allow_seed_any:
        assert train_seed in APT_SEEDS, (
            f"{run_dir}: train seed {train_seed} outside the "
            f"registered 68-75 family")

    mj, npz = _load_mask(run_dir)
    # identity pins (#30 M4): the mask output must belong to THIS run
    # and THIS task. Results-sync relocates trees, so the pin is on
    # BASENAMES, not abspaths.
    assert os.path.basename(os.path.normpath(mj["run_logdir"])) == \
        os.path.basename(os.path.normpath(run_dir)), (
        f"{run_dir}: mask run_logdir basename "
        f"{mj['run_logdir']!r} does not match — foreign mask output?")
    assert str(mj["task"]) == TASK_PIN, (
        f"{run_dir}: mask task {mj['task']!r} != pinned")
    # instrument-side pins persisted in the json
    assert int(mj["n_eval"]) == S_PIN, (
        f"{run_dir}: realized S {mj['n_eval']} != pinned {S_PIN} — "
        f"APT deltas not comparable (#29 C-M5)")
    assert str(mj["expl_mode"]) == "apt", run_dir
    assert int(mj["apt_knn"]) == APT_KNN and \
        abs(float(mj["apt_logc"]) - APT_LOGC) < 1e-12, run_dir
    assert int(mj["train_seed"]) == train_seed, (
        f"{run_dir}: mask train_seed {mj['train_seed']} != config "
        f"{train_seed}")
    assert mj["dup0_bitwise_equal_source"], (
        f"{run_dir}: dup0 not bitwise-equal to source")
    c0 = mj["channels"]["planted_dup0"]
    assert c0.get("bitwise_noop") and c0["delta_apt_mean"] == 0.0, (
        f"{run_dir}: dup0 is not the exact-zero anchor: {c0}")
    # dup0 exact zero at the VECTOR level (#30 m2)
    d0 = np.asarray(npz["delta_apt_planted_dup0"])
    assert d0.size == S_PIN and np.all(d0 == 0.0), (
        f"{run_dir}: dup0 delta vector is not identically zero")
    # base_apt cross-check (#30 m2): STORED dtype (house lesson)
    rc_base = float(np.asarray(npz["base_apt"]).mean())
    assert abs(rc_base - float(mj["base_apt_mean"])) <= \
        PROV_RTOL * max(1.0, abs(rc_base)), (
        f"{run_dir}: base_apt provenance mismatch")
    # fire-channel mask-form pins (#30 m3): #29's blocker was
    # precisely substitution-into-entropy — the fire channel MUST be
    # the marginal-preserving permutation, and permutation channels
    # must not be degenerate no-ops
    assert mj["channels"][FIRE_CHANNEL]["form"] == \
        "batch_permutation", (
        f"{run_dir}: distractor mask form is "
        f"{mj['channels'][FIRE_CHANNEL]['form']!r}, not the pinned "
        f"batch_permutation")
    for ch in REQUIRED_CHANNELS:
        assert ch in mj["channels"], f"{run_dir}: channel {ch} missing"
        dvec = np.asarray(npz[f"delta_apt_{ch}"], np.float64)
        assert dvec.size == S_PIN, (run_dir, ch, dvec.size)
        assert np.all(np.isfinite(dvec)), (
            f"{run_dir}/{ch}: non-finite deltas — numerics failure, "
            f"not tampering (#30 M5)")
        rc = float(dvec.mean())
        stored = float(mj["channels"][ch]["delta_apt_mean"])
        assert abs(rc - stored) <= PROV_RTOL * max(1.0, abs(rc)), (
            f"{run_dir}/{ch}: npz-recomputed mean {rc!r} != stored "
            f"{stored!r} — provenance mismatch (json+npz pair)")
        mj["channels"][ch]["prov_exact"] = bool(rc == stored)
        mj["channels"][ch]["delta_se"] = float(
            dvec.std(ddof=1) / math.sqrt(dvec.size))
    for ch in (FIRE_CHANNEL, "velocity_control"):
        assert not np.all(np.asarray(npz[f"delta_apt_{ch}"]) == 0), (
            f"{run_dir}/{ch}: identically-zero permutation delta — "
            f"degenerate mask (#30 m3)")
    assert mj["channels"]["planted_dup1_resample"]["form"] == \
        "fresh_resample" and \
        mj["channels"]["velocity_control"]["form"] == \
        "batch_permutation_real_key_control", run_dir

    # FINAL-CHECKPOINT GATE (house rule): reachable ckpt dir -> the
    # masked checkpoint MUST be the run's final one; unreachable ->
    # flagged UNVERIFIED, never silent
    try:
        final = resolve_ckpt(run_dir, "")
        ckpt_final_ok = (os.path.basename(os.path.normpath(final))
                         == os.path.basename(os.path.normpath(
                             mj["ckpt"])))
    except (AssertionError, FileNotFoundError, OSError):
        final, ckpt_final_ok = None, None
    assert ckpt_final_ok is not False, (
        f"{run_dir}: se_apt_mask ran on "
        f"{os.path.basename(os.path.normpath(mj['ckpt']))} but the "
        f"final checkpoint is "
        f"{os.path.basename(os.path.normpath(final))}")

    chans = {ch: {k: mj["channels"][ch][k]
                  for k in ("delta_apt_mean", "delta_apt_bca",
                            "p_reduce", "p_inflate", "form",
                            "prov_exact", "delta_se")}
             for ch in REQUIRED_CHANNELS}
    return dict(run_dir=os.path.abspath(run_dir),
                train_seed=train_seed,
                ckpt_final_ok=ckpt_final_ok, n_eval=int(mj["n_eval"]),
                base_apt_mean=float(mj["base_apt_mean"]),
                channels=chans)


def _run_spread_trigger(rec):
    """Per-run cell-(iii) trigger (#30 M2/M3): a resample control
    that is NEGATIVE, material vs THIS run's distractor delta, and
    significantly nonzero (2 SE). Returns (neg_trigger, pos_anomaly).
    """
    dist = rec["channels"][FIRE_CHANNEL]["delta_apt_mean"]
    neg, pos = False, False
    for ch in RESAMPLE_CHANNELS:
        m = rec["channels"][ch]["delta_apt_mean"]
        se = rec["channels"][ch]["delta_se"]
        material = (abs(m) > SPREAD_MATERIALITY * abs(dist)
                    and abs(m) > 2.0 * se)
        neg = neg or (m < 0 and material)
        pos = pos or (m > 0 and material)
    return neg, pos


def _per_run_stat(rec, fire_rule):
    d = rec["channels"][FIRE_CHANNEL]["delta_apt_mean"]
    if fire_rule == "sign":
        return d
    res = [rec["channels"][ch]["delta_apt_mean"]
           for ch in RESAMPLE_CHANNELS]
    return d - float(np.mean(res))


def aggregate(recs, invalid_runs, fire_rule="sign"):
    seeds = sorted(r["train_seed"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert set(seeds) <= APT_SEEDS, f"seeds {seeds} outside 68-75"
    n_loaded = len(recs)
    n_bad = len(invalid_runs)
    assert n_loaded + n_bad <= N_REGISTERED, (
        f"{n_loaded}+{n_bad} runs exceed the registered "
        f"{N_REGISTERED}")
    out = dict(n_registered=N_REGISTERED, n_loaded=n_loaded,
               n_invalid=n_bad, invalid_runs=invalid_runs,
               seeds=seeds, fire_rule=fire_rule, stamp=_stamp(),
               partial_flag=("COMPLETE"
                             if n_loaded == N_REGISTERED else
                             f"PARTIAL ({n_loaded}/{N_REGISTERED} "
                             f"loaded, {n_bad} invalid — missing/"
                             f"invalid count AGAINST the fire, "
                             f"conservative)"),
               probe_ckpt_flag=("OK" if all(r["ckpt_final_ok"]
                                            for r in recs)
                                else "UNVERIFIED (ckpt dirs "
                                     "unreachable on some runs)"))

    # unconditional per-run reporting, before any branch
    out["per_run"] = [
        {"run_dir": r["run_dir"], "train_seed": r["train_seed"],
         "n_eval": r["n_eval"], "base_apt_mean": r["base_apt_mean"],
         "channels": r["channels"]} for r in recs]

    # PRIMARY (pinned): per-run statistic < 0 in >= 7 of the 8
    # REGISTERED runs; missing/invalid runs are non-negative by
    # convention. NaN/-0.0 count as non-negative (conservative).
    stats = [_per_run_stat(r, fire_rule) for r in recs]
    n_neg = sum(1 for x in stats if x < 0)
    fires = bool(n_neg >= FIRE_MIN_NEG)
    p_att = (sum(math.comb(n_loaded, k)
                 for k in range(n_neg, n_loaded + 1)) / 2 ** n_loaded
             if n_loaded else 1.0)
    out["primary"] = dict(
        statement="APT prices the planted distractor: per-run "
                  f"statistic ({fire_rule}) < 0 in >= 7/8 REGISTERED "
                  "runs; single fire channel, no BH; sign-at-zero "
                  "coin null ASSERTED per prereg #29 C-M3; NaN/-0.0 "
                  "count as non-negative; missing/invalid runs count "
                  "against the fire and missingness may be "
                  "informative (disclosed, #30 M7)",
        channel=FIRE_CHANNEL, per_run_stat=stats, n_neg=n_neg,
        n_registered=N_REGISTERED, n_loaded=n_loaded,
        n_invalid=n_bad, partial_flag=out["partial_flag"],
        fires=fires, binom_p_registered_7of8=9 / 256,
        binom_p_attained=p_att)

    # D-family, DESCRIPTIVE (#29 C-B1): dual-mask decomposition
    dual = {}
    for ch in DUP_CHANNELS:
        sub = [r["channels"][ch]["delta_apt_mean"] for r in recs]
        res = [r["channels"][f"{ch}_resample"]["delta_apt_mean"]
               for r in recs]
        dual[ch] = dict(
            substitution_per_run=sub, resample_per_run=res,
            mechanical_component_per_run=[s - x
                                          for s, x in zip(sub, res)],
            substitution_mean=(float(np.mean(sub)) if sub else None),
            resample_mean=(float(np.mean(res)) if res else None))
    vel = [r["channels"]["velocity_control"]["delta_apt_mean"]
           for r in recs]
    out["descriptive"] = dict(
        dual_mask=dual,
        dup0_anchor="0.0 exactly on every run (vector-level "
                    "hard-assert)",
        velocity_control=dict(per_run=vel,
                              mean=(float(np.mean(vel)) if vel
                                    else None),
                              n_neg=sum(1 for x in vel if x < 0)),
        note="anchor-level p/BCa diagnostic-not-calibrated "
             "(#29 C-M4); population-intervention estimand")

    # spread-rescope flag (prereg cell iii; #30 M2/M3 form):
    # per-run NEGATIVE-systematic trigger in a MAJORITY of loaded runs
    trig = [_run_spread_trigger(r) for r in recs]
    n_neg_trig = sum(1 for t, _ in trig if t)
    n_pos_anom = sum(1 for _, a in trig if a)
    spread = bool(n_loaded > 0 and n_neg_trig * 2 >= n_loaded
                  and n_neg_trig > 0)
    out["spread_rescoped"] = spread
    out["spread_detail"] = dict(
        n_neg_trigger=n_neg_trig, n_loaded=n_loaded,
        positive_resample_anomaly_runs=n_pos_anom,
        note="positive-signed material resample means are an "
             "anomaly report, never cell (iii)")
    base = ("i-DISTRACTOR-FIRES" if fires else "ii-APT-IMMUNE")
    out["outcome_cell"] = base + ("+SPREAD-RESCOPED" if spread else "")
    return out


def run(args):
    assert not args.allow_seed_any, (
        "--allow_seed_any is smoke/selfcheck only (#30 m5)")
    dirs = (sorted(globmod.glob(args.runs)) if "," not in args.runs
            else [x.strip() for x in args.runs.split(",")
                  if x.strip()])
    assert dirs, f"no runs matched {args.runs!r}"
    assert args.output and args.output != "TBD", (
        "registered read must persist: pass an explicit --output")
    out_json = os.path.join(args.output, "se_apt_read.json")
    assert not os.path.exists(out_json), (
        f"{out_json} exists — the read is ONE execution (#30 m7)")
    recs, invalid = [], []
    for d in dirs:
        try:
            recs.append(read_run(d))
        except Exception as e:              # noqa: BLE001 (#30 B1)
            invalid.append(dict(run_dir=os.path.abspath(d),
                                error=f"{type(e).__name__}: {e}"))
    out = aggregate(recs, invalid, args.fire_rule)
    out["fit_counters"] = {r["run_dir"]: fit_counters(
        r["run_dir"], args.expect_steps) for r in recs}
    bad_fit = [rd for rd, fc in out["fit_counters"].items()
               if fc.get("flag") != "OK"]
    out["fit_flag"] = ("OK" if not bad_fit
                       else f"SUSPECT ({len(bad_fit)} runs: "
                            f"{[os.path.basename(b) for b in bad_fit]})")
    os.makedirs(args.output, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    pr = out["primary"]
    print(f"APT PRIMARY [{out['fire_rule']}]: {pr['n_neg']}/"
          f"{pr['n_registered']} negative (loaded {pr['n_loaded']}, "
          f"invalid {pr['n_invalid']}) -> fires={pr['fires']} "
          f"({out['outcome_cell']}); attained binom p "
          f"{pr['binom_p_attained']:.4f}; fit {out['fit_flag']}; "
          f"{out['partial_flag']}")
    return out


# --------------------------------------------------------------- smoke gate

def smoke_gate(args):
    """Mechanical acceptance gate on the registered smoke run
    (#29 C-M7; #30 B2/M1 form). NOT powered; NOT the registered
    read. FAIL verdicts are reported, never crashed; the report is
    always written when --output is given."""
    rd = args.smoke_gate
    report = dict(run_dir=os.path.abspath(rd), checks={},
                  verdict="FAIL", stamp=_stamp())

    def _finish():
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            with open(os.path.join(args.output,
                                   "se_apt_smoke_gate.json"),
                      "w") as f:
                json.dump(report, f, indent=1)
        print(json.dumps(report, indent=1, default=str))
        return report

    try:
        rec = read_run(rd, allow_seed_any=True)
    except Exception as e:                   # noqa: BLE001 (#30 M1)
        report["error"] = f"{type(e).__name__}: {e}"
        report["checks"]["instrument_gates"] = False
        return _finish()
    checks = report["checks"]
    checks["instrument_gates"] = True        # S pin, dup0, forms,
    #                                          knobs — read_run passed
    # planted keys + distractor live: per-key decoder losses present
    # + finite (#30 m4: all five, incl. planted_const), robust to a
    # truncated final metrics line (#30 M1)
    live = {}
    try:
        with open(os.path.join(rd, "metrics.jsonl"), "rb") as f:
            lines = f.read().splitlines()
    except OSError as e:
        report["error"] = f"metrics.jsonl: {e}"
        lines = []
    for k in LIVE_KEYS:
        mk = f"train/loss/{k}"
        v = None
        for line in reversed(lines):
            try:
                dd = json.loads(line)
            except ValueError:
                continue
            if mk in dd:
                v = float(dd[mk])
                break
        live[k] = bool(v is not None and np.isfinite(v))
    checks["planted_keys_live"] = all(live.values())
    # run-completion evidence (#30 B2): TRAINING_DONE is written
    # unconditionally by the producer sbatch and is NOT evidence;
    # the fit counters are — the smoke must have reached its steps
    fc = fit_counters(rd, args.smoke_steps)
    checks["run_completed"] = bool(fc.get("flag") == "OK")
    report["fit_counters"] = fc

    dist = rec["channels"][FIRE_CHANNEL]["delta_apt_mean"]
    res = {f"{ch}": rec["channels"][ch]["delta_apt_mean"]
           for ch in RESAMPLE_CHANNELS}
    trigger = bool(any(abs(m) > SPREAD_MATERIALITY * abs(dist)
                       for m in res.values()))
    report.update(
        per_key_loss_live=live,
        distractor_delta=dist,
        resample_deltas=res,                 # _resample keys (#30 m1)
        amendment_trigger=trigger,
        note="amendment_trigger=True => the fire rule must be "
             "amended to the registered control-relative comparator "
             "(--fire_rule control_relative) BEFORE the wave "
             "submits (prereg §3); the trigger is sign-agnostic by "
             "registration, unlike the cell-(iii) flag")
    report["verdict"] = ("PASS" if all(
        v is True for v in checks.values()) else "FAIL")
    return _finish()


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, deltas=None, n_eval=S_PIN, mode="apt",
             knn=APT_KNN, dup0_zero=True, doctor_json=None,
             ckpt_name="checkpoint_final", mask_ckpt=None,
             metrics_keys=True, steps=5e5, metrics_frac=0.999,
             dist_form="batch_permutation", spread_se=None,
             drop_channel=None, drop_npz=False, task=TASK_PIN):
    """Fake run dir with config.yaml + metrics.jsonl + se_apt_mask
    json/npz pair whose stored means are the exact npz means."""
    rng = np.random.default_rng(seed * 7 + 1)
    rd = os.path.join(tmp, name)
    mdir = os.path.join(rd, "se_apt_mask")
    os.makedirs(mdir, exist_ok=True)
    ck = os.path.join(rd, "ckpt")
    os.makedirs(os.path.join(ck, ckpt_name), exist_ok=True)
    deltas = dict(deltas or {})
    chans, dnpz = {}, {}
    forms = {"distractor": dist_form,
             "planted_dup0": "source_substitution",
             "planted_dup1": "source_substitution",
             "planted_dup2": "source_substitution",
             "planted_dup1_resample": "fresh_resample",
             "planted_dup2_resample": "fresh_resample",
             "velocity_control":
                 "batch_permutation_real_key_control"}
    for ch, form in forms.items():
        want = deltas.get(ch, 0.0 if ch == "planted_dup0" else -0.01)
        if ch == "planted_dup0" and dup0_zero:
            d = np.zeros(n_eval)
        else:
            sd = (spread_se * math.sqrt(n_eval)
                  if (spread_se is not None
                      and ch in RESAMPLE_CHANNELS)
                  else max(abs(want), 1e-3) * 0.2)
            d = rng.normal(want, sd, n_eval)
            d = d - d.mean() + want          # exact requested mean
        rec = dict(form=form, delta_apt_mean=float(d.mean()),
                   delta_apt_bca=[float(d.mean()) - 1.0,
                                  float(d.mean()) + 1.0],
                   p_reduce=0.01, p_inflate=0.99)
        if ch == "planted_dup0":
            rec["bitwise_noop"] = dup0_zero
        chans[ch] = rec
        dnpz[f"delta_apt_{ch}"] = d.astype(np.float64)
    if drop_channel:
        del chans[drop_channel]
        del dnpz[f"delta_apt_{drop_channel}"]
    mj = dict(run_logdir=rd,
              ckpt=os.path.join(ck, mask_ckpt or ckpt_name),
              n_eval=n_eval, seed=0, train_seed=seed, task=task,
              source_key="position", apt_knn=knn, apt_logc=APT_LOGC,
              dup0_bitwise_equal_source=dup0_zero,
              base_apt_mean=1.5, expl_mode=mode, channels=chans)
    if doctor_json:
        doctor_json(mj)
    with open(os.path.join(mdir, "se_apt_mask.json"), "w") as f:
        json.dump(mj, f)
    if not drop_npz:
        np.savez(os.path.join(mdir, "se_apt_mask.npz"),
                 base_apt=np.full(n_eval, 1.5), **dnpz)
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(f"seed: {seed}\ntask: {task}\n"
                f"planted:\n  gate_key: ''\n  source_key: 'position'\n"
                f"  basesd: 0.0976\n"
                f"distractor:\n  gate_key: ''\n  mod_key: ''\n"
                f"  dim: 8\n  scale: 1.0\n  basesd: 1.215\n"
                f"agent:\n  expl:\n    mode: {mode}\n"
                f"    apt_knn: {knn}\n    apt_logc: 1.0\n"
                f"    disag_ens: 8\n    disag_scale: 1000.0\n"
                f"run:\n  steps: {steps}\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        row = {"step": int(steps * metrics_frac)}
        if metrics_keys:
            row.update({f"train/loss/{k}": 0.5 for k in LIVE_KEYS})
        f.write(json.dumps(row) + "\n")
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
        neg = {"distractor": -0.05, "planted_dup1": -0.02,
               "planted_dup2": -0.02, "planted_dup1_resample": 1e-4,
               "planted_dup2_resample": -1e-4,
               "velocity_control": -0.03}
        pos = dict(neg, distractor=+0.05)
        # 8/8 fire
        runs = [_fixture(tmp, f"a{i}", 68 + i, deltas=neg)
                for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["primary"]["fires"] and \
            agg["outcome_cell"] == "i-DISTRACTOR-FIRES", agg["primary"]
        assert not agg["spread_rescoped"]
        assert agg["primary"]["binom_p_attained"] == 1 / 256
        # 7/8 fire boundary; 6/8 no fire
        runs = [_fixture(tmp, f"b{i}", 68 + i,
                         deltas=(pos if i == 0 else neg))
                for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["primary"]["fires"] and agg["primary"]["n_neg"] == 7
        assert agg["primary"]["binom_p_attained"] == 9 / 256
        runs = [_fixture(tmp, f"c{i}", 68 + i,
                         deltas=(pos if i < 2 else neg))
                for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert not agg["primary"]["fires"] and \
            agg["outcome_cell"] == "ii-APT-IMMUNE"
        # missing run counts AGAINST (7 all-neg fires; 7 w/ 1 pos no)
        runs = [_fixture(tmp, f"d{i}", 68 + i, deltas=neg)
                for i in range(7)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["primary"]["fires"] and \
            "PARTIAL" in agg["partial_flag"]
        assert agg["primary"]["binom_p_attained"] == 1 / 128  # #30 M7
        runs = [_fixture(tmp, f"e{i}", 68 + i,
                         deltas=(pos if i == 0 else neg))
                for i in range(7)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert not agg["primary"]["fires"]

        # #30 B1: a DEFECTIVE run among 8 -> recorded invalid, read
        # completes, counts against the fire (via run())
        _fixture(tmp, "f0", 68, deltas=neg, drop_npz=True)
        for i in range(1, 8):
            _fixture(tmp, f"f{i}", 68 + i, deltas=neg)
        outd = os.path.join(tmp, "outB1")
        out = run(parse_args(["--runs", os.path.join(tmp, "f*"),
                              "--output", outd]))
        assert out["n_invalid"] == 1 and out["n_loaded"] == 7
        assert out["primary"]["fires"]           # 7 neg of 8 reg
        assert os.path.exists(os.path.join(outd, "se_apt_read.json"))
        # one-execution guard (#30 m7)
        _expect_fail(lambda: run(parse_args(
            ["--runs", os.path.join(tmp, "f*"), "--output", outd])),
            "ONE execution")
        # missing channel -> invalid (not abort) via run()
        _fixture(tmp, "g_ch", 69, deltas=neg,
                 drop_channel="velocity_control")
        out = run(parse_args(["--runs",
                              os.path.join(tmp, "g_ch"),
                              "--output", os.path.join(tmp, "outCH")]))
        assert out["n_invalid"] == 1 and out["n_loaded"] == 0

        # spread-rescope: majority negative + significant (tight SE)
        spread = dict(neg, planted_dup1_resample=-0.04,
                      planted_dup2_resample=-0.04)
        runs = [_fixture(tmp, f"s{i}", 68 + i, deltas=spread,
                         spread_se=1e-4) for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["outcome_cell"] == \
            "i-DISTRACTOR-FIRES+SPREAD-RESCOPED"
        # POSITIVE resample anomaly is NOT cell (iii) (#30 M2)
        posr = dict(neg, planted_dup1_resample=+0.04,
                    planted_dup2_resample=+0.04)
        runs = [_fixture(tmp, f"t{i}", 68 + i, deltas=posr,
                         spread_se=1e-4) for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert not agg["spread_rescoped"]
        assert agg["spread_detail"][
            "positive_resample_anomaly_runs"] == 8
        # noisy (insignificant) resample means do NOT trigger (2 SE)
        runs = [_fixture(tmp, f"u{i}", 68 + i,
                         deltas=dict(neg,
                                     planted_dup1_resample=-0.04,
                                     planted_dup2_resample=-0.04),
                         spread_se=0.1) for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [])
        assert not agg["spread_rescoped"]

        # control-relative fire rule (#30 M8): dist -0.05 vs resample
        # mean -0.06 -> stat +0.01 -> NOT negative
        cr = dict(neg, planted_dup1_resample=-0.06,
                  planted_dup2_resample=-0.06)
        runs = [_fixture(tmp, f"v{i}", 68 + i, deltas=cr,
                         spread_se=1e-4) for i in range(8)]
        agg = aggregate([read_run(d) for d in runs], [],
                        fire_rule="control_relative")
        assert not agg["primary"]["fires"]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["primary"]["fires"]           # sign rule still does

        # identity/provenance gates (hard aborts inside read_run)
        rd = _fixture(tmp, "g0", 68, deltas=neg, mode="p2e")
        _expect_fail(lambda: read_run(rd), "not apt")
        rd = _fixture(tmp, "g1", 68, deltas=neg, n_eval=256)
        _expect_fail(lambda: read_run(rd), "not comparable")
        rd = _fixture(tmp, "g2", 68, deltas=neg, knn=8)
        _expect_fail(lambda: read_run(rd), "apt knobs")
        rd = _fixture(tmp, "g3", 68, deltas=neg, dup0_zero=False)
        _expect_fail(lambda: read_run(rd), "bitwise-equal")

        def z_doc(mj):
            mj["channels"]["planted_dup0"]["delta_apt_mean"] = 0.0
            mj["channels"]["planted_dup0"]["bitwise_noop"] = True
        rd = _fixture(tmp, "g3b", 68,
                      deltas=dict(neg, planted_dup0=0.0),
                      dup0_zero=False, doctor_json=z_doc)
        _expect_fail(lambda: read_run(rd), "bitwise-equal")

        def doctor(mj):
            mj["channels"]["distractor"]["delta_apt_mean"] = -9.0
        rd = _fixture(tmp, "g4", 68, deltas=neg, doctor_json=doctor)
        _expect_fail(lambda: read_run(rd), "provenance mismatch")
        rd = _fixture(tmp, "g5", 68, deltas=neg,
                      mask_ckpt="checkpoint_000200")
        _expect_fail(lambda: read_run(rd), "final checkpoint")
        rd = _fixture(tmp, "g6", 99, deltas=neg)
        _expect_fail(lambda: read_run(rd), "registered 68-75")
        rd = _fixture(tmp, "g7", 68, deltas=neg,
                      dist_form="source_substitution")
        _expect_fail(lambda: read_run(rd), "batch_permutation")
        rd = _fixture(tmp, "g8", 68, deltas=neg, task="dmc_walker")
        _expect_fail(lambda: read_run(rd), "pinned")

        def foreign(mj):
            mj["run_logdir"] = "/some/other/run"
        rd = _fixture(tmp, "g9", 68, deltas=neg, doctor_json=foreign)
        _expect_fail(lambda: read_run(rd), "foreign mask")
        runs = [_fixture(tmp, f"h{i}", 68, deltas=neg)
                for i in range(2)]
        _expect_fail(lambda: aggregate([read_run(d, True)
                                        for d in runs], []),
                     "duplicate seeds")

        # smoke-gate: PASS, amendment trigger, FAIL-not-crash paths
        rd = _fixture(tmp, "s_ok", 9, deltas=neg, steps=2e4)
        rep = smoke_gate(parse_args(["--smoke_gate", rd,
                                     "--output",
                                     os.path.join(tmp, "sgout")]))
        assert rep["verdict"] == "PASS" and not rep["amendment_trigger"]
        assert rep["checks"]["run_completed"]
        assert os.path.exists(os.path.join(tmp, "sgout",
                                           "se_apt_smoke_gate.json"))
        trig = dict(neg, planted_dup1_resample=-0.04)
        rd = _fixture(tmp, "s_tr", 9, deltas=trig, steps=2e4)
        rep = smoke_gate(parse_args(["--smoke_gate", rd]))
        assert rep["amendment_trigger"]
        assert "planted_dup1_resample" in rep["resample_deltas"]
        rd = _fixture(tmp, "s_bad", 9, deltas=neg, steps=2e4,
                      metrics_keys=False)
        rep = smoke_gate(parse_args(["--smoke_gate", rd]))
        assert rep["verdict"] == "FAIL" and \
            not rep["checks"]["planted_keys_live"]
        # incomplete smoke -> FAIL on run_completed (#30 B2)
        rd = _fixture(tmp, "s_inc", 9, deltas=neg, steps=2e4,
                      metrics_frac=0.3)
        rep = smoke_gate(parse_args(["--smoke_gate", rd]))
        assert rep["verdict"] == "FAIL" and \
            not rep["checks"]["run_completed"]
        # defective smoke -> FAIL report, not crash (#30 M1)
        rd = _fixture(tmp, "s_S", 9, deltas=neg, steps=2e4,
                      n_eval=300)
        rep = smoke_gate(parse_args(["--smoke_gate", rd,
                                     "--output",
                                     os.path.join(tmp, "sgS")]))
        assert rep["verdict"] == "FAIL" and "error" in rep
        assert os.path.exists(os.path.join(tmp, "sgS",
                                           "se_apt_smoke_gate.json"))

        # end-to-end fire + fit flag + output guards
        outd = os.path.join(tmp, "out")
        out = run(parse_args(["--runs", os.path.join(tmp, "a*"),
                              "--output", outd]))
        assert out["primary"]["fires"] and out["fit_flag"] == "OK"
        _expect_fail(lambda: run(parse_args(
            ["--runs", os.path.join(tmp, "a*")])), "explicit --output")
        _expect_fail(lambda: run(parse_args(
            ["--runs", os.path.join(tmp, "a*"), "--output",
             os.path.join(tmp, "out2"), "--allow_seed_any"])),
            "smoke/selfcheck only")
    print("se_apt_read selfcheck PASS (fire 8/8 + 7/8 boundary + 6/8 "
          "no-fire; DEFECTIVE-run conservative path w/ persisted "
          "output + one-execution guard; attained binomial p; "
          "spread cell iii negative-only + SE floor + positive-"
          "anomaly separation; control-relative amendment rule; "
          "expl-mode/S-pin/knob/dup0-vector/provenance/final-ckpt/"
          "seed/task/foreign-mask/mask-form gates; smoke-gate "
          "PASS/FAIL/trigger/incomplete/defective-not-crash; "
          "end-to-end)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    elif args.smoke_gate:
        smoke_gate(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
