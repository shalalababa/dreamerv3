"""Wave-B (adversarial amortized actor) FROZEN reader
(PREREG_waveb_advd_20260822; built + selfchecked BEFORE the wave's
compute; ONE execution).

TWO REGISTERED WINGS (separate questions, both at alpha=.05, no
cross-wing correction — disclosed; the Tier-2 gate reads ONLY the
ambient cell):
  AMBIENT   — 8 adaptations on the Stage-1 checkpoints (source
              seeds 10-17): the Fable-ladder Tier-2 carrier.
  LOCALIZED — 8 adaptations on the A1 checkpoints (hetero 44-47,
              flat 48-51): the matched-strength probe of the
              channel Wave C's weak CEM left open.

Read order per wing:
  0. Validity: adapted-config pins (mode advd, advd_scale 100,
     from_checkpoint + regex, disag pins, per-wing env pins,
     penalty inert), probe pins (arms/episodes/ep_len/horizon/
     stride/probe_seed 0/gate constants/one device/gpu),
     WM-IDENTITY ok (the bitwise freeze proof), ENV-IDENTITY,
     run-dir + source identity, newest-ckpt cross-check, float64
     provenance, n_steps/reset_hits, per-run degeneracy floor
     intrinsic(policy) > intrinsic(random). Defective -> excluded;
     a wing needs its FULL panel (8 or 4+4).
  1. IN-MODEL DOMINANCE GATE: on POLICY-arm states, the adapted
     actor's imagined normalized distractor attribution must
     exceed the incumbent policy's (it must beat the incumbent on
     its own objective in imagination) — ambient >= 6/8; localized
     >= 3/4 in EACH sub-wing. Fail -> that wing
     NOT-ADJUDICABLE-INSTRUMENT-WEAK.
  2. AMBIENT PRIMARY: Delta = attr_d(advd) - attr_d(policy) > 0 in
     >= 7/8 AND exact sign-flip p <= .05 ->
     TRANSMISSION-AMPLIFIED; else CONTAINED-AT-MATCHED-STRENGTH
     (the Tier-2 carrier cell — the attacker is matched by CLASS
     (amortized) and by the in-model gate).
  3. LOCALIZED PRIMARY (conjunctive, full 4+4): interaction
     mean Delta(hetero) - mean Delta(flat) > 0 at exact C(8,4)
     p <= .05 AND Delta > 0 in >= 3/4 hetero ->
     LOCALIZED-TRANSMISSION-MATCHED; else
     LOCALIZED-CONTAINED-MATCHED.
  4. Registered secondaries (no fire, every path): the
     imagined-vs-realized rows (imagined attr under the actor from
     its own visited states vs realized attr_d(advd) — the Wave-A
     off-support-imagination mechanism on an amortized attacker),
     share rows, occupancy rows (localized: does the actor ENTER
     the avoided region?), task-return rows, intrinsic rows.

A wing with an incomplete panel is NOT-ADJUDICABLE; if NO wing
adjudicates, the output goes to a timestamped side file and the one
execution is NOT consumed (the R-A1-M13 rule).

Run:  python -m uncfield.se_advd_read --runs "<glob>" --output <d>
      python -m uncfield.se_advd_read --selfcheck
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
from uncfield.se_planner_read import signflip_exact_p
from uncfield.se_read import fit_counters

AMB_SEEDS = tuple(range(10, 18))
HET_SEEDS = (44, 45, 46, 47)
FLAT_SEEDS = (48, 49, 50, 51)
SEED_WING = {**{s: "ambient" for s in AMB_SEEDS},
             **{s: "hetero" for s in HET_SEEDS},
             **{s: "flat" for s in FLAT_SEEDS}}
ARMS = ("random", "policy", "advd")
# rev WB-M4: chunk is pinned — imag_fn samples actions, so batching
# + per-chunk seeds change the gate values
REG_PARAMS = dict(episodes=4, ep_len=500, horizon=12, imag_stride=5,
                  chunk=100)
ADVD_SCALE_PIN = 100.0
FROM_REGEX_PIN = r"^(enc|dyn|dec|disag)/"
TASK_PIN = "dmc_cheetah_run"
N_IMAG = 400            # 2000 steps / stride 5 (rev WB-n2)
AMB_NEED = 7
# rev WB-B2: under H0 (actor indistinguishable) each run's gate is
# a coin flip; the bar must itself be an alpha<=.05 test —
# P(>=7/8)=9/256=.035; localized pooled P(>=7/8)=.035 with the
# per-sub-wing >=3/4 as the balance condition
AMB_GATE_NEED = 7
LOC_HET_NEED = 3
LOC_GATE_POOLED = 7
LOC_GATE_EACH = 3
ADAPT_STEPS = 1e5
# rev WB-M2: the source checkpoints are PINNED (extracted from the
# verified wavea/wavec bundles, latest==newest checked at pin time)
# — WM-IDENTITY alone cannot catch a JOINTLY stale source (the
# adaptation and the probe would load the same early WM and match
# bitwise); these dirs are where the 22-Aug stale-pointer incident
# lived
SOURCE_CKPT_PIN = {
    "se_cheetah_seed10": "20260813T091238F251815",
    "se_cheetah_seed11": "20260813T090740F437213",
    "se_cheetah_seed12": "20260813T153547F545326",
    "se_cheetah_seed13": "20260813T152751F704709",
    "se_cheetah_seed14": "20260813T215850F694009",
    "se_cheetah_seed15": "20260813T214754F587996",
    "se_cheetah_seed16": "20260813T213035F670985",
    "se_cheetah_seed17": "20260813T220036F793840",
    "se_avoid_het_s44": "20260821T215437F360981",
    "se_avoid_het_s45": "20260821T214749F303055",
    "se_avoid_het_s46": "20260821T214816F771484",
    "se_avoid_het_s47": "20260821T213323F038189",
    "se_avoid_flat_s48": "20260821T215000F696859",
    "se_avoid_flat_s49": "20260821T215131F797564",
    "se_avoid_flat_s50": "20260821T213318F824843",
    "se_avoid_flat_s51": "20260821T213825F211836"}
SOURCE_NAME = {**{s: f"se_cheetah_seed{s}" for s in AMB_SEEDS},
               **{s: f"se_avoid_het_s{s}" for s in HET_SEEDS},
               **{s: f"se_avoid_flat_s{s}" for s in FLAT_SEEDS}}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="")
    p.add_argument("--expect_steps", type=float, default=ADAPT_STEPS)
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
    assert train_seed in SEED_WING, (
        f"{run_dir}: seed {train_seed} outside the registered "
        f"panels")
    wing = SEED_WING[train_seed]
    assert str(cfg["task"]) == TASK_PIN, run_dir
    assert str(ex["mode"]) == "advd", (
        f"{run_dir}: expl.mode {ex['mode']!r} — advd did not bind")
    assert abs(float(ex["advd_scale"]) - ADVD_SCALE_PIN) < 1e-9, (
        f"{run_dir}: advd_scale {ex['advd_scale']}")
    assert int(ex["disag_ens"]) == 8 and \
        str(ex.get("disag_head", "det")) == "det", (
        f"{run_dir}: disag pins differ from the det Stage-1 family")
    run_cfg = cfg.get("run", {})
    assert str(run_cfg.get("from_checkpoint", "")), (
        f"{run_dir}: from_checkpoint missing — not an adaptation")
    assert str(run_cfg.get("from_checkpoint_regex", "")) == \
        FROM_REGEX_PIN, (
        f"{run_dir}: from_checkpoint_regex "
        f"{run_cfg.get('from_checkpoint_regex')!r} != the registered "
        f"{FROM_REGEX_PIN!r} — disag would be fresh, not frozen")
    assert str(pl["source_key"]) == "position" and \
        abs(float(pl["basesd"]) - 0.0976) < 1e-9, run_dir
    assert int(d["dim"]) == DIM and \
        abs(float(d["basesd"]) - BASESD_N) < 1e-9
    assert str(d.get("gate_key", "")) == "" and \
        str(pl.get("gate_key", "")) == "", run_dir
    assert float(cfg.get("penalty", {}).get("scale", 0.0)) == 0.0 \
        and ex.get("penalty_mix", False) is False, run_dir
    if wing == "hetero":
        assert abs(float(d["scale"]) - 1.0) < 1e-9
        assert str(d["mod_key"]) == MOD_KEY and \
            int(d["mod_index"]) == MOD_INDEX and \
            abs(float(d["mod_lo"]) - MOD_LO) < 1e-9 and \
            abs(float(d["mod_hi"]) - MOD_HI) < 1e-9, (
            f"{run_dir}: hetero mod quadruple differs from A1")
    elif wing == "flat":
        assert abs(float(d["scale"]) - FLAT_SCALE) < 1e-9, run_dir
        assert str(d.get("mod_key", "")) == "", run_dir
    else:
        assert abs(float(d["scale"]) - 1.0) < 1e-9, run_dir
        assert str(d.get("mod_key", "")) == "", run_dir

    pdir = os.path.join(run_dir, "se_advd")
    with open(os.path.join(pdir, "se_advd.json")) as f:
        pj = json.load(f)
    npz = np.load(os.path.join(pdir, "se_advd.npz"))
    assert not pj["smoke"], f"{run_dir}: smoke output"
    assert list(pj["arms"]) == list(ARMS), pj["arms"]
    assert int(pj["train_seed"]) == train_seed
    assert int(pj["source_seed"]) == train_seed, (
        f"{run_dir}: adaptation seed {train_seed} != source seed "
        f"{pj['source_seed']} — the identity convention broke")
    # rev WB-M2: source-checkpoint provenance pin — the probe AND
    # the adaptation must both sit on the pinned FINAL source ckpt
    src_name = os.path.basename(os.path.normpath(pj["source_logdir"]))
    assert src_name == SOURCE_NAME[train_seed], (
        f"{run_dir}: source {src_name} != expected "
        f"{SOURCE_NAME[train_seed]}")
    src_ck = os.path.basename(os.path.normpath(pj["source_ckpt"]))
    assert src_ck == SOURCE_CKPT_PIN[src_name], (
        f"{run_dir}: probe source ckpt {src_ck} != pinned "
        f"{SOURCE_CKPT_PIN[src_name]} — jointly stale source")
    from_ck = os.path.basename(os.path.normpath(
        str(run_cfg["from_checkpoint"])))
    assert from_ck == SOURCE_CKPT_PIN[src_name], (
        f"{run_dir}: adaptation from_checkpoint {from_ck} != pinned "
        f"{SOURCE_CKPT_PIN[src_name]}")
    assert str(ex["disag_target"]) == "postfeat", (
        f"{run_dir}: disag_target {ex.get('disag_target')!r} — the "
        f"objective's definition drifted (rev WB-n3)")
    assert str(pj["task"]) == TASK_PIN and int(pj["probe_seed"]) == 0
    assert os.path.basename(os.path.normpath(pj["run_logdir"])) == \
        os.path.basename(os.path.normpath(run_dir)), (
        f"{run_dir}: probe json belongs to {pj['run_logdir']}")
    for k, v in REG_PARAMS.items():
        assert int(pj[k]) == v, (
            f"{run_dir}: {k}={pj[k]} != registered {v}")
    assert abs(float(pj["advd_scale"]) - ADVD_SCALE_PIN) < 1e-9
    wmi = pj["wm_identity"]
    assert wmi["ok"] is True and int(wmi["n_params"]) > 0, (
        f"{run_dir}: WM-IDENTITY gate not green — the freeze did "
        f"not hold ({wmi})")
    assert pj["env_identity"] is True, run_dir
    g = pj["gate"]
    assert str(g["key"]) == GATE_KEY and int(g["index"]) == GATE_INDEX \
        and abs(float(g["threshold"]) - GATE_THRESHOLD) < 1e-12, (
        f"{run_dir}: occupancy gate constants drifted")
    n_dev = str(pj["devices"]).count("id=")
    assert n_dev == 1, f"{run_dir}: {n_dev} devices"
    assert str(pj.get("backend", "")).lower() in ("gpu", "cuda"), (
        f"{run_dir}: backend {pj.get('backend')!r}")
    # newest-ckpt cross-check on the ADAPTED run
    ck_probe = os.path.basename(os.path.normpath(pj["ckpt"]))
    ckroot = os.path.join(run_dir, "ckpt")
    try:
        subs = sorted(x for x in os.listdir(ckroot)
                      if os.path.isdir(os.path.join(ckroot, x)))
    except OSError:
        subs = []
    if subs:
        assert ck_probe == subs[-1], (
            f"{run_dir}: probed ckpt {ck_probe} != newest {subs[-1]}")
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
    # imagination rows: presence + pinned length + provenance +
    # per-episode block means (rev WB-B2 descriptive: the honest
    # inferential unit — 400 stride-5 starts decompose into 4x100
    # by episode; a paired test over starts would be
    # anticonservative)
    imag, imag_blocks = {}, {}
    for key in ("imag_policystates_policypol",
                "imag_policystates_advdpol",
                "imag_advdstates_advdpol"):
        v = np.asarray(npz[key], np.float64)
        assert v.size == N_IMAG and np.all(np.isfinite(v)), (
            run_dir, key, v.size)
        assert abs(float(v.mean()) - float(pj["imag"][key])) <= \
            1e-12 * max(1.0, abs(float(v.mean()))), (
            f"{run_dir}/{key}: imag provenance mismatch")
        imag[key] = float(v.mean())
        imag_blocks[key] = [float(x) for x in
                            v.reshape(4, -1).mean(1)]
    # rev WB-B2.4: adaptation plateau descriptive from the run's
    # own training telemetry (report-only; the direct evidence for
    # the 1e5-step sufficiency claim)
    plateau = None
    try:
        vals = []
        with open(os.path.join(run_dir, "metrics.jsonl")) as f:
            for line in f:
                row = json.loads(line)
                if "train/expl/advd_rew_raw" in row:
                    vals.append(float(row["train/expl/advd_rew_raw"]))
        if len(vals) >= 10:
            k = len(vals) // 10
            early = float(np.mean(vals[4 * k:5 * k]))
            late = float(np.mean(vals[-k:]))
            plateau = dict(n=len(vals), decile5_mean=early,
                           last_decile_mean=late,
                           ratio=(late / early if early > 0
                                  else None))
    except OSError:
        pass
    degenerate = not (per_arm["policy"]["intrinsic_mean"]
                      > per_arm["random"]["intrinsic_mean"])
    return dict(run_dir=os.path.abspath(run_dir),
                train_seed=train_seed, wing=wing,
                degenerate=degenerate, ckpt=ck_probe,
                source_logdir=pj["source_logdir"],
                imag=imag, imag_blocks=imag_blocks,
                plateau=plateau, per_arm=per_arm)


def _ad(rec, arm):
    return float(rec["per_arm"][arm]["attr_mean"]["distractor"])


def _share(rec, arm):
    am = rec["per_arm"][arm]["attr_mean"]
    return float(am["distractor"] / sum(am.values()))


def _gate_pass(r):
    return r["imag"]["imag_policystates_advdpol"] > \
        r["imag"]["imag_policystates_policypol"]


def aggregate(recs, excluded):
    seeds = sorted(r["train_seed"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert len(recs) + len(excluded) <= 16
    out = dict(stamp=_stamp(), n_loaded=len(recs),
               excluded_runs=excluded,
               degenerate_runs=[r["run_dir"] for r in recs
                                if r["degenerate"]])
    # descriptives on every path
    out["per_run"] = [
        {"run_dir": r["run_dir"], "wing": r["wing"],
         "train_seed": r["train_seed"],
         "degenerate": r["degenerate"],
         "delta_advd_minus_policy": _ad(r, "advd") - _ad(r, "policy"),
         "attr_d": {a: _ad(r, a) for a in ARMS},
         "share_d": {a: _share(r, a) for a in ARMS},
         "occupancy": {a: r["per_arm"][a]["occupancy"]
                       for a in ARMS},
         "intrinsic": {a: r["per_arm"][a]["intrinsic_mean"]
                       for a in ARMS},
         "return_mean": {a: r["per_arm"][a]["return_mean"]
                         for a in ARMS},
         "imag": r["imag"],
         "imag_blocks": r["imag_blocks"],
         "gate_margin":
             (r["imag"]["imag_policystates_advdpol"]
              / max(r["imag"]["imag_policystates_policypol"], 1e-30)
              - 1.0),
         "intrinsic_policy_over_random":
             (r["per_arm"]["policy"]["intrinsic_mean"]
              / max(r["per_arm"]["random"]["intrinsic_mean"],
                    1e-30)),
         "plateau": r["plateau"],
         "imag_vs_realized_ratio":
             (r["imag"]["imag_advdstates_advdpol"]
              / max(_ad(r, "advd"), 1e-30)),
         "gate_pass": bool(_gate_pass(r))} for r in recs]

    valid = [r for r in recs if not r["degenerate"]]
    amb = [r for r in valid if r["wing"] == "ambient"]
    het = [r for r in valid if r["wing"] == "hetero"]
    fla = [r for r in valid if r["wing"] == "flat"]
    out["n_ambient"], out["n_hetero"], out["n_flat"] = (
        len(amb), len(het), len(fla))

    # ---------------- AMBIENT wing (the Tier-2 carrier)
    ambient = {}
    if len(amb) < 8:
        ambient["cell"] = (f"NOT-ADJUDICABLE (ambient {len(amb)}/8 "
                           f"— full panel required)")
    else:
        n_gate = sum(1 for r in amb if _gate_pass(r))
        ambient["gate"] = dict(n_pass=n_gate, need=AMB_GATE_NEED)
        if n_gate < AMB_GATE_NEED:
            ambient["cell"] = "NOT-ADJUDICABLE-INSTRUMENT-WEAK"
        else:
            deltas = [_ad(r, "advd") - _ad(r, "policy") for r in amb]
            n_pos = sum(1 for x in deltas if x > 0)
            p = signflip_exact_p(deltas)
            m, lo, hi = bca_interval(np.asarray(deltas),
                                     np.random.default_rng(825), 2000)
            fires = bool(n_pos >= AMB_NEED and p <= 0.05)
            ambient["primary"] = dict(
                per_run_delta=deltas, n_pos=n_pos, need=AMB_NEED,
                p_signflip=p, mean=m, bca=[lo, hi], fires=fires)
            ambient["cell"] = ("TRANSMISSION-AMPLIFIED" if fires
                               else "CONTAINED-AT-MATCHED-STRENGTH")
    out["ambient"] = ambient

    # ---------------- LOCALIZED wing
    localized = {}
    if len(het) < 4 or len(fla) < 4:
        localized["cell"] = (f"NOT-ADJUDICABLE (hetero {len(het)}, "
                             f"flat {len(fla)}; full 4+4 required)")
    else:
        gh = sum(1 for r in het if _gate_pass(r))
        gf = sum(1 for r in fla if _gate_pass(r))
        localized["gate"] = dict(het_pass=gh, flat_pass=gf,
                                 pooled=gh + gf,
                                 pooled_need=LOC_GATE_POOLED,
                                 each_need=LOC_GATE_EACH)
        if (gh + gf) < LOC_GATE_POOLED or gh < LOC_GATE_EACH \
                or gf < LOC_GATE_EACH:
            localized["cell"] = "NOT-ADJUDICABLE-INSTRUMENT-WEAK"
        else:
            d_het = [_ad(r, "advd") - _ad(r, "policy") for r in het]
            d_fla = [_ad(r, "advd") - _ad(r, "policy") for r in fla]
            vals = np.asarray(d_het + d_fla, np.float64)
            labels = np.array([True] * 4 + [False] * 4)
            obs, p, total = exact_perm_p(vals, labels, one_sided=True)
            assert total == math.comb(8, 4)
            n_het_pos = sum(1 for x in d_het if x > 0)
            fires = bool(obs > 0 and p <= 0.05
                         and n_het_pos >= LOC_HET_NEED)
            localized["primary"] = dict(
                delta_hetero=d_het, delta_flat=d_fla,
                interaction=float(obs), p=float(p),
                min_attainable_p=1 / 70, n_het_pos=n_het_pos,
                need=LOC_HET_NEED, fires=fires)
            localized["region_entry_hetero"] = [
                dict(seed=r["train_seed"],
                     occ_advd=r["per_arm"]["advd"]["occupancy"],
                     occ_policy=r["per_arm"]["policy"]["occupancy"])
                for r in het]
            localized["cell"] = (
                "LOCALIZED-TRANSMISSION-MATCHED" if fires
                else "LOCALIZED-CONTAINED-MATCHED")
    out["localized"] = localized

    # rev WB-M3: the ONE execution is keyed to the CARRIER — a
    # localized-only adjudication must never spend it while the
    # Tier-2 wing goes unread
    out["carrier_adjudicated"] = bool(
        isinstance(ambient.get("cell"), str)
        and not ambient["cell"].startswith("NOT-ADJUDICABLE"))
    out["any_wing_adjudicated"] = bool(any(
        isinstance(w.get("cell"), str)
        and not w["cell"].startswith("NOT-ADJUDICABLE")
        for w in (ambient, localized)))
    return out


def run(args):
    dirs = (sorted(globmod.glob(args.runs)) if "," not in args.runs
            else [x.strip() for x in args.runs.split(",")
                  if x.strip()])
    assert dirs, f"no runs matched {args.runs!r}"
    assert args.output and args.output != "TBD", (
        "registered read must persist: pass an explicit --output")
    out_json = os.path.join(args.output, "se_advd_read.json")
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
    bad_fit = [rd for rd, fc in out["fit_counters"].items()
               if fc.get("flag") != "OK"]
    out["fit_flag"] = ("OK" if not bad_fit
                       else f"SUSPECT ({len(bad_fit)} runs)")
    os.makedirs(args.output, exist_ok=True)
    if not out["carrier_adjudicated"]:
        side = os.path.join(
            args.output,
            f"se_advd_read_NOTADJ_{time.strftime('%Y%m%dT%H%M%S')}"
            f".json")
        with open(side, "w") as f:
            json.dump(out, f, indent=1)
        print(f"ADVD: carrier (ambient) wing not adjudicated — side "
              f"file {side}; the one execution is NOT consumed "
              f"(localized-only results ride the side file, "
              f"rev WB-M3). AMBIENT: {out['ambient'].get('cell')}; "
              f"LOCALIZED: {out['localized'].get('cell')}")
        return out
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"ADVD AMBIENT: {out['ambient'].get('cell')}")
    print(f"ADVD LOCALIZED: {out['localized'].get('cell')}; "
          f"fit {out['fit_flag']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, d_advd=None, gate_margin=0.01,
             imag_own=0.05, wm_ok=True, degenerate=False,
             smoke=False, doctor=None, drop_npz=False):
    """Adapted-run fixture. d_advd = attr_d(advd) − attr_d(policy)
    (policy pinned at 0.02); gate_margin > 0 makes the in-model
    dominance gate pass."""
    rng = np.random.default_rng(seed * 41 + 7)
    wing = SEED_WING.get(seed, "ambient")   # off-panel seeds still
    #                                         build (the gate is tested)
    rd = os.path.join(tmp, name)
    pdir = os.path.join(rd, "se_advd")
    os.makedirs(pdir, exist_ok=True)
    ck = os.path.join(rd, "ckpt")
    os.makedirs(os.path.join(ck, "0000100000"), exist_ok=True)
    with open(os.path.join(ck, "latest"), "w") as f:
        f.write("0000100000")
    d_advd = 0.0 if d_advd is None else d_advd
    keys = ("position", "velocity", "distractor", "planted_dup0")
    per_arm, npz_out = {}, {}
    n_steps = 2000
    base_ad = {"random": 0.006, "policy": 0.020,
               "advd": 0.020 + d_advd}
    intr = {"random": 8e-5,
            "policy": 1.9e-4 if not degenerate else 5e-5,
            "advd": 1.8e-4}
    for a in ARMS:
        rec = dict(n_steps=n_steps, attr_mean={}, elapsed_s=1.0,
                   return_mean=50.0, reset_hits=0)
        for k in keys:
            want = base_ad[a] if k == "distractor" else 0.01
            v = rng.normal(want, abs(want) * 0.1 + 1e-4, n_steps)
            v = (v - v.mean() + want).astype(np.float64)
            npz_out[f"{a}_attr_{k}"] = v
            rec["attr_mean"][k] = float(v.mean())
        iv = np.full(n_steps, intr[a], np.float64)
        npz_out[f"{a}_intr"] = iv
        rec["intrinsic_mean"] = float(iv.mean())
        ov = np.full(n_steps, 0.5, np.float64)
        npz_out[f"{a}_occ"] = ov
        rec["occupancy"] = float(ov.mean())
        per_arm[a] = rec
    imag = {}
    for key, mval in (
            ("imag_policystates_policypol", 0.030),
            ("imag_policystates_advdpol", 0.030 + gate_margin),
            ("imag_advdstates_advdpol", imag_own)):
        v = np.full(400, mval, np.float64)
        npz_out[key] = v
        imag[key] = float(v.mean())
    src_name = SOURCE_NAME.get(seed, f"se_cheetah_seed{seed}")
    src_pin = SOURCE_CKPT_PIN.get(src_name, "20260813T000000F000000")
    pj = dict(run_logdir=rd, source_logdir=f"/src/{src_name}",
              ckpt=os.path.join(ck, "0000100000"),
              source_ckpt=f"/src/{src_name}/ckpt/{src_pin}",
              train_seed=seed, source_seed=seed, task=TASK_PIN,
              probe_seed=0, arms=list(ARMS),
              advd_scale=ADVD_SCALE_PIN, smoke=smoke,
              wm_identity=dict(ok=bool(wm_ok), n_params=42,
                               regex=r"^(enc|dyn|dec|disag)/"),
              env_identity=True,
              devices="[CudaDevice(id=0)]", backend="gpu",
              norm_floor=0.03,
              gate=dict(key=GATE_KEY, index=GATE_INDEX,
                        threshold=GATE_THRESHOLD),
              imag=imag, per_arm=per_arm, **REG_PARAMS)
    if doctor:
        doctor(pj)
    with open(os.path.join(pdir, "se_advd.json"), "w") as f:
        json.dump(pj, f)
    npz_path = os.path.join(pdir, "se_advd.npz")
    if drop_npz:
        if os.path.exists(npz_path):
            os.remove(npz_path)
    else:
        np.savez(npz_path, **npz_out)
    het = wing == "hetero"
    fla = wing == "flat"
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(
            f"seed: {seed}\ntask: {TASK_PIN}\n"
            f"run:\n  from_checkpoint: "
            f"'/src/{src_name}/ckpt/{src_pin}'\n"
            f"  from_checkpoint_regex: '^(enc|dyn|dec|disag)/'\n"
            f"planted:\n  gate_key: ''\n  source_key: 'position'\n"
            f"  basesd: 0.0976\n"
            f"distractor:\n  gate_key: ''\n"
            f"  mod_key: '{MOD_KEY if het else ''}'\n"
            f"  mod_index: {MOD_INDEX}\n  mod_lo: {MOD_LO}\n"
            f"  mod_hi: {MOD_HI}\n  dim: {DIM}\n"
            f"  scale: {FLAT_SCALE if fla else 1.0}\n"
            f"  basesd: {BASESD_N}\n"
            f"agent:\n  expl:\n    mode: advd\n    disag_ens: 8\n"
            f"    disag_head: det\n    disag_target: postfeat\n"
            f"    advd_scale: 100.0\n"
            f"run_steps_note: adaptation\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        # plateau telemetry: rising then flat advd_rew_raw
        for i in range(40):
            f.write(json.dumps({
                "step": 2500 * (i + 1),
                "train/expl/advd_rew_raw":
                    0.1 + 0.2 * min(i, 20) / 20}) + "\n")
        f.write(json.dumps({"step": 99968}) + "\n")
    with open(os.path.join(rd, "scores.jsonl"), "w") as f:
        for i in range(40):
            f.write(json.dumps({"step": i,
                                "episode/score": 10.0}) + "\n")
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
    ALL = list(AMB_SEEDS) + list(HET_SEEDS) + list(FLAT_SEEDS)
    with tempfile.TemporaryDirectory() as tmp:
        def wave(sub, amb_d=0.0, het_d=0.0, fla_d=0.0, **kw):
            runs = []
            for s in ALL:
                dv = (amb_d if s in AMB_SEEDS
                      else het_d if s in HET_SEEDS else fla_d)
                runs.append(_fixture(tmp, f"{sub}{s}", s, d_advd=dv,
                                     **kw))
            return runs

        # both wings CONTAINED (deltas ~0 / slightly negative)
        runs = wave("a", amb_d=-0.001, het_d=-0.001, fla_d=-0.001)
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["ambient"]["cell"] == \
            "CONTAINED-AT-MATCHED-STRENGTH", agg["ambient"]
        assert agg["localized"]["cell"] == \
            "LOCALIZED-CONTAINED-MATCHED", agg["localized"]
        assert agg["per_run"][0]["imag_vs_realized_ratio"] > 1
        # ambient fires
        runs = wave("b", amb_d=0.008, het_d=-0.001, fla_d=-0.001)
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["ambient"]["cell"] == "TRANSMISSION-AMPLIFIED"
        assert agg["ambient"]["primary"]["p_signflip"] <= 0.05
        # localized fires (hetero up, flat not)
        runs = wave("c", amb_d=-0.001, het_d=0.010, fla_d=-0.002)
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["localized"]["cell"] == \
            "LOCALIZED-TRANSMISSION-MATCHED", agg["localized"]
        assert agg["localized"]["primary"]["p"] == 1 / 70
        # in-model gate failure -> wing INSTRUMENT-WEAK
        runs = wave("d", amb_d=-0.001, het_d=-0.001, fla_d=-0.001,
                    gate_margin=-0.01)
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["ambient"]["cell"] == \
            "NOT-ADJUDICABLE-INSTRUMENT-WEAK"
        assert agg["localized"]["cell"] == \
            "NOT-ADJUDICABLE-INSTRUMENT-WEAK"
        # partial ambient wing -> that wing NOT-ADJ, localized stands
        runs = [d for d in wave("e", amb_d=-0.001, het_d=-0.001,
                                fla_d=-0.001)
                if not d.endswith("e10")]
        agg = aggregate([read_run(d) for d in runs], [])
        assert agg["ambient"]["cell"].startswith("NOT-ADJUDICABLE (")
        assert agg["localized"]["cell"] == \
            "LOCALIZED-CONTAINED-MATCHED"
        # rev WB-M3: a localized-only adjudication must NOT count as
        # the carrier — run() would side-file it
        assert not agg["carrier_adjudicated"]
        assert agg["any_wing_adjudicated"]
        # degenerate runs drop from validity
        runs = wave("f", amb_d=-0.001, het_d=-0.001, fla_d=-0.001,
                    degenerate=True)
        agg = aggregate([read_run(d) for d in runs], [])
        assert len(agg["degenerate_runs"]) == 16
        assert agg["ambient"]["cell"].startswith("NOT-ADJUDICABLE")
        # gates
        rd = _fixture(tmp, "g0", 10, wm_ok=False)
        _expect_fail(lambda: read_run(rd), "WM-IDENTITY")
        rd = _fixture(tmp, "g1", 10, smoke=True)
        _expect_fail(lambda: read_run(rd), "smoke")

        def doc_mode(pj):
            pass
        rd = _fixture(tmp, "g2", 10)
        cfgp = os.path.join(rd, "config.yaml")
        cfg = open(cfgp).read()
        open(cfgp, "w").write(cfg.replace("mode: advd", "mode: p2e"))
        _expect_fail(lambda: read_run(rd), "advd did not bind")
        rd = _fixture(tmp, "g3", 10)
        cfg = open(os.path.join(rd, "config.yaml")).read()
        open(os.path.join(rd, "config.yaml"), "w").write(
            cfg.replace("^(enc|dyn|dec|disag)/", "^(enc|dyn|dec)/"))
        _expect_fail(lambda: read_run(rd), "disag would be fresh")

        def doc_src(pj):
            pj["source_seed"] = 11
        rd = _fixture(tmp, "g4", 10, doctor=doc_src)
        _expect_fail(lambda: read_run(rd), "identity convention")

        def doc_prov(pj):
            pj["per_arm"]["advd"]["attr_mean"]["distractor"] = 9.9
        rd = _fixture(tmp, "g5", 11, doctor=doc_prov)
        _expect_fail(lambda: read_run(rd), "provenance")

        def doc_imag(pj):
            pj["imag"]["imag_policystates_advdpol"] = 9.9
        rd = _fixture(tmp, "g6", 12, doctor=doc_imag)
        _expect_fail(lambda: read_run(rd), "imag provenance")
        rd = _fixture(tmp, "g7", 60)
        _expect_fail(lambda: read_run(rd), "registered panels")
        # no wing adjudicated -> side file, execution NOT consumed;
        # then a complete read succeeds
        outd = os.path.join(tmp, "out")
        for s in AMB_SEEDS:
            _fixture(tmp, f"h{s}", s, d_advd=-0.001,
                     drop_npz=(s == 10))
        out = run(parse_args(["--runs", os.path.join(tmp, "h1*"),
                              "--output", outd]))
        assert not out["any_wing_adjudicated"]
        assert not os.path.exists(os.path.join(outd,
                                               "se_advd_read.json"))
        _fixture(tmp, "h10", 10, d_advd=-0.001)
        out = run(parse_args(["--runs", os.path.join(tmp, "h1*"),
                              "--output", outd]))
        assert out["ambient"]["cell"] == \
            "CONTAINED-AT-MATCHED-STRENGTH"
        _expect_fail(lambda: run(parse_args(
            ["--runs", os.path.join(tmp, "h1*"),
             "--output", outd])), "ONE execution")
    print("se_advd_read selfcheck PASS (CONTAINED both wings + "
          "ambient TRANSMISSION-AMPLIFIED + localized "
          "LOCALIZED-TRANSMISSION-MATCHED at the exact floor + "
          "alpha-bar in-model gates (7/8; pooled 7/8 + 3/4 each) + "
          "carrier-keyed one-execution (localized-only side-files) "
          "+ partial-wing isolation + degeneracy floor; "
          "WM-IDENTITY/smoke/mode/regex/source-ckpt-PIN/"
          "from_checkpoint-PIN/disag_target/provenance/"
          "imag-provenance(size-400)/seed gates; plateau + "
          "gate-margin + block-mean descriptives; one-execution "
          "guard)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
