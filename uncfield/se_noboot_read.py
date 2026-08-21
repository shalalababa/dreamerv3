"""SE NOBOOT (bootstrap-decomposition) FROZEN reader
(PREREG_nfi_scale_exhibit_amend4_20260820.md §2, post-review-#26 form;
built + selfchecked BEFORE the arm's compute; ONE execution).

Arm: Stage-1 config with `agent.expl.disag_bootstrap: False` (members
differ by initialization only — the published Plan2Explore
convention), seeds 40-43.

Read order (amendment §2):
  0. DEGENERACY GATE (review #26 B1): per noboot run, mean REAL-dim
     per-dim attribution AND pse2.intrinsic_mean must each be >= 1/10
     of the Stage-1 per-run MINIMUM (loaded live via --stage1_runs);
     any failure -> GLOBAL-DISAGREEMENT-COLLAPSE, bands not
     adjudicable (a whole-instrument collapse must never be read as a
     distractor-specific result).
  1. PRIMARY: distractor theta_1 seed-level BCa (n=4), theta_1
     recomputed from the npz and asserted equal to the stored value
     (#26 M4), banded:
       (a) MISPRICE-SURVIVES-LARGE: CI > 2.0 AND 4/4 per-run p < .05
           (final-checkpoint scope; compatible with a substantial
           bootstrap share — point estimate is the headline);
       (b) BOOTSTRAP-DRIVEN: CI <= 1.10, OR (n_p_lt_05 <= 1 AND
           CI <= 2.0) (#26 M3 — the count disjunct alone must not
           fire while a large misprice survives);
       (c) ATTENUATED/MIXED: else.
  2. SECONDARY: dup0 theta_1 vs the par band [0.90, 1.10].
Reporting is UNCONDITIONAL on every branch (#26 M6): per-run shares,
theta_1, calibration, per-key absolute levels, intrinsic_mean,
disag_replay_rew, fit counters.

Run:  python -m uncfield.se_noboot_read --runs "<glob>"
          --stage1_runs "<glob>" --output <dir>
      python -m uncfield.se_noboot_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os

import numpy as np

from uncfield.se_grad_read import BASESD_N, DIM
from uncfield.se_mask import bca_interval
from uncfield.se_probe import PLANTED_KEYS
from uncfield.se_read import FIRE_KEYS, fit_counters
from uncfield import se_read

NOBOOT_SEEDS = set(range(40, 44))
HIGH_BAND = 2.0                  # analyst choice, amendment §2
PAR_HI = 1.10                    # Amendment-2 par band upper edge
PAR_LO = 0.90
COLLAPSE_FACTOR = 10.0           # degeneracy gate generosity (#26 B1)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="")
    p.add_argument("--stage1_runs", default="",
                   help="Stage-1 run dirs (se_probe outputs) — the "
                        "degeneracy-gate level reference")
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--expect_steps", type=float, default=5e5)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def _levels(run_dir):
    """Absolute per-key attribution levels + intrinsic from the probe
    outputs (data-variance-normalized dims_d -> cross-run comparable)."""
    pdir = os.path.join(run_dir, "se_probe")
    npz = np.load(os.path.join(pdir, "se_probe_dims.npz"))
    dims_d = np.asarray(npz["dims_d"], np.float64)
    dim_key = np.asarray(npz["dim_key"]).astype(str)
    with open(os.path.join(pdir, "se_probe.json")) as f:
        pj = json.load(f)
    keys = sorted(set(dim_key.tolist()))
    per_key = {k: float(dims_d[dim_key == k].mean()) for k in keys}
    real_mean = float(np.mean([per_key[k] for k in keys
                               if k not in PLANTED_KEYS]))
    src = str(pj.get("source_key", "position"))
    theta = {ch: (per_key[ch] / max(per_key.get(src, 0.0), 1e-300))
             for ch in keys if ch in PLANTED_KEYS}
    return dict(per_key=per_key, real_mean=real_mean,
                intrinsic_mean=float(pj["pse2"]["intrinsic_mean"]),
                theta_recomputed=theta)


def _disag_replay_rew(run_dir):
    try:
        with open(os.path.join(run_dir, "metrics.jsonl"), "rb") as f:
            lines = f.read().splitlines()
        for line in reversed(lines):
            d = json.loads(line)
            if "train/expl/disag_replay_rew" in d:
                return float(d["train/expl/disag_replay_rew"])
    except (OSError, ValueError):
        pass
    return None


def read_run(run_dir, n_perm):
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = cfg["distractor"]
    pl = cfg["planted"]
    ex = cfg["agent"]["expl"]
    assert ex["disag_bootstrap"] is False, (
        f"{run_dir}: disag_bootstrap is not False — not a noboot run")
    # Stage-1-verbatim pins (#26 m11)
    assert str(ex["mode"]) == "p2e" and int(ex["disag_ens"]) == 8 \
        and abs(float(ex["disag_scale"]) - 1000.0) < 1e-9, (
        f"{run_dir}: expl pins differ from Stage-1")
    assert int(d["dim"]) == DIM and abs(float(d["basesd"]) - BASESD_N) < 1e-9
    assert abs(float(d["scale"]) - 1.0) < 1e-9
    assert str(d.get("gate_key", "")) == "" and \
        str(d.get("mod_key", "")) == "", f"{run_dir}: gates/mod must be off"
    assert str(pl.get("gate_key", "")) == ""
    assert str(pl["source_key"]) == "position" and \
        abs(float(pl["basesd"]) - 0.0976) < 1e-9, (
        f"{run_dir}: planted pins differ from Stage-1")
    rec = se_read.read_run(run_dir, "se_probe", n_perm)
    rec["train_seed_cfg"] = int(cfg["seed"])
    lv = _levels(run_dir)
    # theta_1 provenance gate (#26 M4): recomputed from npz == stored
    for ch in FIRE_KEYS:
        stored = rec["channels"][ch]["vs_source"]
        rc = lv["theta_recomputed"].get(ch)
        assert rc is not None and abs(rc - stored) <= 1e-4 * max(
            1.0, abs(stored)), (
            f"{run_dir}/{ch}: theta_1 recomputed {rc} != stored "
            f"{stored} — provenance mismatch")
    rec["levels"] = lv
    rec["disag_replay_rew"] = _disag_replay_rew(run_dir)
    return rec


def stage1_reference(stage1_dirs):
    """Per-run Stage-1 level minima for the degeneracy gate."""
    real_means, intr = [], []
    for d in stage1_dirs:
        lv = _levels(d)
        real_means.append(lv["real_mean"])
        intr.append(lv["intrinsic_mean"])
    assert len(real_means) >= 4, "need the Stage-1 bundle for the gate"
    return dict(real_mean_min=float(min(real_means)),
                intrinsic_min=float(min(intr)),
                n=len(real_means))


def aggregate(recs, ref):
    seeds = sorted(r["train_seed_cfg"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert set(seeds) == NOBOOT_SEEDS, f"seeds {seeds} != [40..43]"
    invalid = [r["run_dir"] for r in recs if not r["valid"]]
    out = dict(n_loaded=len(recs), invalid_runs=invalid,
               instrument_flag=("CLEAN" if not invalid and len(recs) == 4
                                else "VALIDITY-VIOLATIONS"),
               probe_ckpt_flag=("OK" if all(r["ckpt_final_ok"]
                                            for r in recs)
                                else "UNVERIFIED (ckpt dirs unreachable "
                                     "on some runs)"),
               stage1_reference=ref)

    # unconditional reporting (#26 M6) — before any branch
    out["per_run"] = [
        {"run_dir": r["run_dir"], "train_seed": r["train_seed_cfg"],
         "n_eval": r["n_eval"], "valid": r["valid"],
         "cal_real_max": r["cal_real_max"],
         "real_mean_level": r["levels"]["real_mean"],
         "intrinsic_mean": r["levels"]["intrinsic_mean"],
         "disag_replay_rew": r["disag_replay_rew"],
         "per_key_levels": r["levels"]["per_key"],
         "channels": {ch: {k: r["channels"][ch][k]
                           for k in ("share_vs_real", "p_perm",
                                     "vs_source", "indicator")}
                      for ch in FIRE_KEYS}}
        for r in recs]
    d_th = [r["channels"]["planted_dup0"]["vs_source"] for r in recs]
    if len(d_th) >= 3:
        m2, lo2, hi2 = bca_interval(np.asarray(d_th),
                                    np.random.default_rng(405), 2000)
        out["secondary_dup0"] = dict(
            theta1_mean=m2, theta1_bca=[lo2, hi2], theta1_per_run=d_th,
            par_band=[PAR_LO, PAR_HI],
            parity_bootstrap_independent=bool(PAR_LO <= lo2
                                              and hi2 <= PAR_HI))

    # 0) degeneracy gate (#26 B1) — before the bands
    collapsed = [r["run_dir"] for r in recs
                 if r["levels"]["real_mean"]
                 < ref["real_mean_min"] / COLLAPSE_FACTOR
                 or r["levels"]["intrinsic_mean"]
                 < ref["intrinsic_min"] / COLLAPSE_FACTOR]
    out["collapsed_runs"] = collapsed
    if collapsed:
        out["primary"] = ("GLOBAL-DISAGREEMENT-COLLAPSE: absolute "
                          "disagreement levels fell below 1/10 of the "
                          "Stage-1 per-run minimum — a different "
                          "physical result (init diversity alone does "
                          "not sustain the signal); the theta_1 bands "
                          "are NOT adjudicable (amendment §2 gate)")
        out["outcome_cell"] = "GLOBAL-COLLAPSE"
        return out
    if out["instrument_flag"] != "CLEAN":
        out["primary"] = "NOT-ADJUDICABLE (need 4 valid runs)"
        out["outcome_cell"] = "NOT-ADJUDICABLE"
        return out

    # 1) primary bands
    n_th = [r["channels"]["distractor"]["vs_source"] for r in recs]
    n_p = sum(1 for r in recs
              if r["channels"]["distractor"]["p_perm"] < 0.05)
    m, lo, hi = bca_interval(np.asarray(n_th),
                             np.random.default_rng(404), 2000)
    if lo > HIGH_BAND and n_p == 4:
        cell = "a-MISPRICE-SURVIVES-LARGE"
    elif hi <= PAR_HI or (n_p <= 1 and hi <= HIGH_BAND):
        cell = "b-BOOTSTRAP-DRIVEN"
    else:
        cell = "c-ATTENUATED-MIXED"
    out["primary"] = dict(
        statement="distractor theta_1 under init-only ensemble "
                  "(bootstrap off), banded per amendment 4 §2; "
                  "final-checkpoint scope; cell (a) compatible with a "
                  "substantial bootstrap share — the point estimate "
                  "is the headline",
        theta1_mean=m, theta1_bca=[lo, hi], theta1_per_run=n_th,
        n_p_lt_05=n_p, cell=cell,
        stage1_anchor="5.26 [5.02, 5.46] (bootstrap on, n=8, "
                      "DESCRIPTIVE cross-wave)")
    out["outcome_cell"] = cell
    return out


def run(args):
    def expand(pat):
        return (sorted(globmod.glob(pat)) if "," not in pat
                else [x.strip() for x in pat.split(",") if x.strip()])

    dirs = expand(args.runs)
    s1 = expand(args.stage1_runs)
    assert dirs, f"no runs matched {args.runs!r}"
    assert s1, "degeneracy gate requires --stage1_runs (amendment §2)"
    assert args.output, "registered read must persist: pass --output"
    ref = stage1_reference(s1)
    recs = [read_run(d, args.n_perm) for d in dirs]
    out = aggregate(recs, ref)
    out["fit_counters"] = {r["run_dir"]: fit_counters(r["run_dir"],
                                                      args.expect_steps)
                           for r in recs}
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "se_noboot_read.json"), "w") as f:
        json.dump(out, f, indent=1)
    if isinstance(out["primary"], dict):
        pr = out["primary"]
        print(f"NOBOOT PRIMARY: theta1 {pr['theta1_mean']:.3f} "
              f"{[round(x, 3) for x in pr['theta1_bca']]} "
              f"n_p_lt_05 {pr['n_p_lt_05']}/4 -> {pr['cell']}")
    else:
        print(f"NOBOOT PRIMARY: {out['outcome_cell']}")
    if "secondary_dup0" in out:
        s2 = out["secondary_dup0"]
        print(f"  dup0 parity: {s2['theta1_mean']:.4f} "
              f"{[round(x, 4) for x in s2['theta1_bca']]} "
              f"bootstrap-independent="
              f"{s2['parity_bootstrap_independent']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, boot=False, hot=None, level_scale=1.0,
             intrinsic=3.5e-4, n_eval=512):
    """noboot run fixture on se_read's reviewed probe generator, with
    levels controlled via a uniform dims_d scale (uniform scaling
    preserves all ratios/p's, so the probe p provenance holds) and the
    theta provenance derived from the npz itself."""
    from uncfield import se_read as sr
    rd = os.path.join(tmp, name)
    sr._fixture_run(tmp, name, seed,
                    hot=(hot if hot is not None
                         else {"distractor": 5.0}), n_eval=n_eval)
    pdir = os.path.join(rd, "se_probe")
    npz = dict(np.load(os.path.join(pdir, "se_probe_dims.npz")))
    npz["dims_d"] = npz["dims_d"] * level_scale     # ratio-preserving
    np.savez(os.path.join(pdir, "se_probe_dims.npz"), **npz)
    with open(os.path.join(pdir, "se_probe.json")) as f:
        pj = json.load(f)
    dims_d = np.asarray(npz["dims_d"], np.float64)
    dim_key = np.asarray(npz["dim_key"]).astype(str)
    per_key = {k: float(dims_d[dim_key == k].mean())
               for k in set(dim_key.tolist())}
    for ch in pj["channels"]:
        pj["channels"][ch]["vs_source"] = per_key[ch] / per_key["position"]
    pj["pse2"]["intrinsic_mean"] = intrinsic
    pj["source_key"] = "position"
    with open(os.path.join(pdir, "se_probe.json"), "w") as f:
        json.dump(pj, f)
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(f"seed: {seed}\n"
                f"planted:\n  gate_key: ''\n  source_key: 'position'\n"
                f"  basesd: 0.0976\n"
                f"distractor:\n  gate_key: ''\n  mod_key: ''\n"
                f"  dim: {DIM}\n  scale: 1.0\n  basesd: {BASESD_N}\n"
                f"agent:\n  expl:\n    mode: p2e\n    disag_ens: 8\n"
                f"    disag_scale: 1000.0\n    disag_bootstrap: "
                f"{'true' if boot else 'false'}\n"
                f"run:\n  steps: 500000.0\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 499968,
                            "train/expl/disag_replay_rew": 0.02}) + "\n")
    return rd


def selfcheck():
    import tempfile
    NP = se_read.FIX_PERM
    HOT = {"distractor": 5.0}
    with tempfile.TemporaryDirectory() as tmp:
        # stage-1 reference fixtures (levels ~1.0, intrinsic 3.5e-4)
        s1 = [_fixture(tmp, f"s{i}", 40 + i, hot=HOT) for i in range(4)]
        # NOTE: reference fixtures reuse the seed numbers only for the
        # generator; stage1_reference never checks seeds
        ref = stage1_reference(s1)
        assert 0.9 < ref["real_mean_min"] < 1.1

        # (a) misprice survives: hot probes at healthy levels
        runs = [_fixture(tmp, f"a{i}", 40 + i, hot=HOT)
                for i in range(4)]
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome_cell"] == "a-MISPRICE-SURVIVES-LARGE"
        assert agg["secondary_dup0"]["parity_bootstrap_independent"]
        assert agg["per_run"][0]["disag_replay_rew"] == 0.02

        # (b) collapsed to par at HEALTHY levels: parity probes
        runs = [_fixture(tmp, f"b{i}", 40 + i, hot={})
                for i in range(4)]
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome_cell"] == "b-BOOTSTRAP-DRIVEN"

        # (b-guard, #26 M3): large theta but few small p's -> (c), not (b)
        # parity dims give theta~1; instead inflate distractor share
        # mildly so p is not small but theta CI sits in (1.10, 2.0]:
        runs = [_fixture(tmp, f"m{i}", 40 + i,
                         hot={"planted_dup0": 1.0, "distractor": 1.5})
                for i in range(4)]
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome_cell"] == "c-ATTENUATED-MIXED", (
            agg["outcome_cell"], agg["primary"])

        # GLOBAL COLLAPSE (#26 B1): hot ratios but levels 1e-6 of ref
        runs = [_fixture(tmp, f"g{i}", 40 + i, hot=HOT,
                         level_scale=1e-6, intrinsic=1e-9)
                for i in range(4)]
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome_cell"] == "GLOBAL-COLLAPSE"
        assert "per_run" in agg and "secondary_dup0" in agg  # M6
        assert len(agg["collapsed_runs"]) == 4

        # NOT-ADJUDICABLE still reports (#26 M6): one run under S floor
        runs = [_fixture(tmp, f"n{i}", 40 + i, hot=HOT,
                         n_eval=(200 if i == 0 else 512))
                for i in range(4)]
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE"
        assert "per_run" in agg and "secondary_dup0" in agg

        # identity gates
        rd = _fixture(tmp, "x", 40, boot=True, hot=HOT)
        try:
            read_run(rd, NP)
            raise SystemExit("noboot config gate FAILED to fire")
        except AssertionError as e:
            assert "not a noboot run" in str(e)
        rd = _fixture(tmp, "y", 40, hot=HOT)
        cfg = open(os.path.join(rd, "config.yaml")).read()
        with open(os.path.join(rd, "config.yaml"), "w") as f:
            f.write(cfg.replace("disag_scale: 1000.0",
                                "disag_scale: 500.0"))
        try:
            read_run(rd, NP)
            raise SystemExit("expl-pin gate FAILED to fire")
        except AssertionError as e:
            assert "expl pins" in str(e)
        # theta provenance gate (#26 M4): doctor the stored vs_source
        rd = _fixture(tmp, "z", 40, hot=HOT)
        pth = os.path.join(rd, "se_probe", "se_probe.json")
        with open(pth) as f:
            pj = json.load(f)
        pj["channels"]["distractor"]["vs_source"] = 99.0
        with open(pth, "w") as f:
            json.dump(pj, f)
        try:
            read_run(rd, NP)
            raise SystemExit("theta provenance gate FAILED to fire")
        except AssertionError as e:
            assert "provenance mismatch" in str(e)
        # wrong seed family fatal
        runs = [_fixture(tmp, f"e{i}", 50 + i, hot=HOT)
                for i in range(4)]
        try:
            aggregate([read_run(d, NP) for d in runs], ref)
            raise SystemExit("seed gate FAILED to fire")
        except AssertionError as e:
            assert "seeds" in str(e)

        # end-to-end run()
        outd = os.path.join(tmp, "out")
        out = run(parse_args([
            "--runs", os.path.join(tmp, "a*"),
            "--stage1_runs", os.path.join(tmp, "s*"),
            "--n_perm", str(NP), "--output", outd]))
        assert out["outcome_cell"].startswith("a-")
        assert os.path.exists(os.path.join(outd, "se_noboot_read.json"))
    print("se_noboot_read selfcheck PASS (cells a/b/c incl. the M3 "
          "guard fixture, GLOBAL-COLLAPSE gate w/ unconditional "
          "reporting, NOT-ADJUDICABLE reporting, noboot/expl-pin/theta-"
          "provenance/seed gates, end-to-end)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
