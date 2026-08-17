"""SE gradient-arm (level-vs-gradient) FROZEN reader
(PREREG_nfi_scale_exhibit_amend3_20260816.md §4, post-review-#24 form;
built + selfchecked BEFORE the arm's compute, executes the read ONCE).

Arms (same wave, same site): hetero (seeds 28-31, distractor
amplitude m = clip((position[0] - mod_lo)/(mod_hi - mod_lo), 0, 1),
lo/hi = the pre-Stage-1 smoke range); flat (seeds 36-39,
scale = 0.35891393 = the smoke-weighted mean m — amplitude-matched,
NO gradient, incentive-immune); ungated2 (seeds 32-35, Stage-1
verbatim — the dose reference).

Registered read order:
  1. MANIPULATION-CHECK GATE: distractor p_perm < .05 (frozen
     se_probe outputs, provenance-gated via se_read.read_run) in
     >= 3 of the 4 hetero runs, counted among VALID probes. Removal
     wording (cell iii) requires >= 3 valid probes; fewer -> cell iv
     "NOT-ADJUDICABLE (probe instrument)", no removal claim.
  2. PRIMARY (directional): occupancy(hetero) - occupancy(FLAT) in
     the high-amplitude region (position[0] > -0.13009691,
     full-replay scan, >= 1e5-step floor), exact one-sided label
     permutation over C(8,4) = 70, FIRES iff p <= .05. The
     amplitude-matched comparator discharges the dose alternative
     (review #24 M2).
  3. Cells: (i) gate PASS + fire -> gradient-following; (ii) gate
     PASS + no fire -> dissociation robust at the registered
     delivered contrast (1.86x amplitude / 3.2x variance,
     smoke-weighted); (iii) gate FAIL w/ >= 3 valid probes ->
     modulation removed the misprice; (iv) validity cells.

SECONDARY (reporting): hetero-ungated2 and flat-ungated2 occupancy
deltas (dose axis, descriptive); score deltas two-sided; per-arm
occupancy BCa; hetero + flat probe shares/theta_1 (dose descriptive);
ungated2 vs the prior ungated arm's 0.560 (cross-wave descriptive).

Identity gates (review #24 M3): ALL arms pin distractor.dim == 8,
basesd == 1.215, gate keys EMPTY on both wrappers; hetero pins the
mod quadruple + scale 1.0; flat pins scale 0.35891393 + mod empty;
ungated2 pins scale 1.0 + mod empty. Train seeds exactly
{28..31}/{36..39}/{32..35}; duplicates fatal; occupancy floor; fit
counters; probe final-ckpt status surfaced in the aggregate.

Run:  python -m uncfield.se_grad_read --hetero "<glob>"
          --flat "<glob>" --ungated "<glob>" --output <dir>
      python -m uncfield.se_grad_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os

import numpy as np

from uncfield.se_mask import bca_interval
from uncfield.se_m3_read import (GATE_INDEX, GATE_KEY, GATE_THRESHOLD,
                                 MIN_STEPS, SCORE_TAIL, exact_perm_p,
                                 occupancy)
from uncfield.se_read import fit_counters
from uncfield import se_read

MOD_KEY = "position"
MOD_INDEX = 0
MOD_LO = -0.32203162          # amendment 3 §2 (smoke position[0] range)
MOD_HI = 0.19711795
FLAT_SCALE = 0.35891393       # smoke-weighted mean m (amendment 3 §3)
DIM = 8
BASESD_N = 1.215
ARM_SEEDS = {"hetero": set(range(28, 32)), "ungated2": set(range(32, 36)),
             "flat": set(range(36, 40))}
GATE_MIN = 3                  # manipulation check: >= 3/4 hetero runs
PRIOR_UNGATED_OCC = 0.560     # Amendment-2 M3 read (descriptive only)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hetero", default="", help="4 hetero run dirs (glob)")
    p.add_argument("--flat", default="", help="4 flat run dirs (glob)")
    p.add_argument("--ungated", default="", help="4 ungated2 run dirs")
    p.add_argument("--n_perm", type=int, default=1000,
                   help="MUST match the se_probe executions")
    p.add_argument("--expect_steps", type=float, default=5e5)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def read_run(run_dir, arm, n_perm):
    """One gradient-wave run: config-derived arm identity + occupancy
    (+ the provenance-gated probe record where required/present)."""
    assert arm in ARM_SEEDS, arm
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    train_seed = int(cfg["seed"])
    d = cfg["distractor"]
    pgate = str(cfg["planted"].get("gate_key", ""))
    dgate = str(d.get("gate_key", ""))
    assert pgate == "" and dgate == "", (
        f"{run_dir}: gate keys must be EMPTY in the gradient wave "
        f"(planted={pgate!r}, distractor={dgate!r})")
    # shared pins (review #24 M3): the wave's distractor must exist and
    # match Stage-1 dim/basesd on every arm
    assert int(d["dim"]) == DIM, (
        f"{run_dir}: distractor.dim {d['dim']} != {DIM} — mislaunched "
        f"(D-only override leaked?)")
    assert abs(float(d["basesd"]) - BASESD_N) < 1e-9, (
        f"{run_dir}: distractor.basesd {d['basesd']} != {BASESD_N}")
    mk = str(d.get("mod_key", ""))
    scale = float(d["scale"])
    if arm == "hetero":
        assert mk == MOD_KEY and int(d["mod_index"]) == MOD_INDEX, (
            f"{run_dir}: mod target ({mk!r}, {d.get('mod_index')}) != "
            f"registered ({MOD_KEY!r}, {MOD_INDEX})")
        assert (abs(float(d["mod_lo"]) - MOD_LO) < 1e-9
                and abs(float(d["mod_hi"]) - MOD_HI) < 1e-9), (
            f"{run_dir}: mod range ({d['mod_lo']}, {d['mod_hi']}) != "
            f"registered ({MOD_LO}, {MOD_HI})")
        assert abs(scale - 1.0) < 1e-9, (
            f"{run_dir}: hetero scale {scale} != 1.0")
    else:
        assert mk == "", (
            f"{run_dir}: config mod_key={mk!r} but this run was passed "
            f"as {arm} (mod must be empty)")
        want = FLAT_SCALE if arm == "flat" else 1.0
        assert abs(scale - want) < 1e-9, (
            f"{run_dir}: {arm} scale {scale} != registered {want}")
    occ, n_steps = occupancy(run_dir, GATE_INDEX, GATE_THRESHOLD)
    scores = []
    try:
        with open(os.path.join(run_dir, "scores.jsonl")) as f:
            for line in f:
                rec = json.loads(line)
                if "episode/score" in rec:
                    scores.append(float(rec["episode/score"]))
    except OSError:
        pass
    out = dict(run_dir=os.path.abspath(run_dir), train_seed=train_seed,
               arm=arm, occupancy=occ, n_steps=int(n_steps),
               valid=bool(n_steps >= MIN_STEPS),
               score=(float(np.mean(scores[-SCORE_TAIL:]))
                      if scores else None),
               n_score_entries=len(scores))
    probe_path = os.path.join(run_dir, "se_probe", "se_probe.json")
    if arm == "hetero":
        assert os.path.exists(probe_path), (
            f"{run_dir}: se_probe/ missing — the §5 bundle spec "
            f"requires the manipulation-check probe on every hetero run")
    if os.path.exists(probe_path):
        pr = se_read.read_run(run_dir, "se_probe", n_perm)
        ch = pr["channels"]["distractor"]
        out["probe"] = dict(p_perm=ch["p_perm"],
                            share_vs_real=ch["share_vs_real"],
                            theta1=ch["vs_source"],
                            probe_valid=pr["valid"],
                            gate_all_p_one=pr["gate_all_p_one"],
                            ckpt_final_ok=pr["ckpt_final_ok"])
    return out


def aggregate(recs):
    arms = {a: [r for r in recs if r["arm"] == a] for a in ARM_SEEDS}
    for a, rs in arms.items():
        ss = sorted(r["train_seed"] for r in rs)
        assert len(set(ss)) == len(ss), f"duplicate seeds in {a}: {ss}"
        assert set(ss) == ARM_SEEDS[a], (
            f"{a} seeds {ss} != {sorted(ARM_SEEDS[a])}")
    het, flat, ung = arms["hetero"], arms["flat"], arms["ungated2"]
    invalid = [r["run_dir"] for r in recs if not r["valid"]]
    out = dict(n_hetero=len(het), n_flat=len(flat), n_ungated2=len(ung),
               invalid_runs=invalid,
               instrument_flag=("CLEAN" if not invalid
                                else "VALIDITY-VIOLATIONS"))

    # probe final-ckpt status surfaced (review #24 M5) — False is
    # already fatal inside se_read.read_run; None = unverifiable
    probes = [r for r in het + flat if "probe" in r]
    out["probe_ckpt_flag"] = (
        "OK" if probes and all(r["probe"]["ckpt_final_ok"] for r in probes)
        else "UNVERIFIED (ckpt dirs unreachable on some probed runs)")

    # descriptives computed before any early return (review #24 m9)
    for name, rs in (("hetero", het), ("flat", flat), ("ungated2", ung)):
        if len(rs) >= 3:
            m, lo, hi = bca_interval(
                np.asarray([r["occupancy"] for r in rs]),
                np.random.default_rng({"hetero": 21, "flat": 23,
                                       "ungated2": 22}[name]), 2000)
            out[f"occupancy_{name}"] = dict(mean=m, bca=[lo, hi])
    if "occupancy_ungated2" in out:
        out["ungated2_vs_prior_ungated"] = dict(
            prior=PRIOR_UNGATED_OCC,
            this_wave=out["occupancy_ungated2"]["mean"],
            note="cross-wave DESCRIPTIVE replication reference only")

    # 1) manipulation-check gate (amendment 3 §4; review #24 M4 form)
    het_probes = [r["probe"] for r in het]
    n_valid_probes = sum(1 for p in het_probes if p["probe_valid"])
    gate_succ = sum(1 for p in het_probes
                    if p["probe_valid"] and p["p_perm"] < 0.05)
    mc = dict(rule="distractor p_perm < .05 in >= 3/4 hetero runs, "
                   "counted among VALID probes",
              successes=gate_succ, n_valid_probes=n_valid_probes,
              n=len(het_probes),
              per_run=[dict(seed=r["train_seed"], **r["probe"])
                       for r in het])
    if n_valid_probes < GATE_MIN and gate_succ < GATE_MIN:
        mc["passes"] = False
        out["manipulation_check"] = mc
        out["primary"] = ("MANIPULATION CHECK NOT-ADJUDICABLE (probe "
                          "instrument: fewer than 3 valid probes) — no "
                          "removal claim, no diversion claim")
        out["outcome_cell"] = "iv"
        return out
    mc["passes"] = bool(gate_succ >= GATE_MIN)
    out["manipulation_check"] = mc
    if not mc["passes"]:
        out["primary"] = ("DIVERSION-NOT-ADJUDICABLE (manipulation gate "
                          "FAILED on >= 3 valid probes: amplitude "
                          "modulation removed the misprice — registered "
                          "outcome iii)")
        out["outcome_cell"] = "iii"
        return out

    if invalid or len(het) != 4 or len(flat) != 4:
        out["primary"] = "NOT-ADJUDICABLE (need 4+4 valid hetero/flat " \
                         "runs — registered validity cell iv)"
        out["outcome_cell"] = "iv"
        return out

    # 2) primary: hetero vs FLAT (amplitude-matched), one-sided
    ordered = het + flat
    labels = [True] * 4 + [False] * 4
    occ = [r["occupancy"] for r in ordered]
    d_occ, p_occ, total = exact_perm_p(occ, labels, one_sided=True)
    fires = bool(p_occ <= 0.05)
    out["primary"] = dict(
        statement="gradient-following: occupancy(hetero) > "
                  "occupancy(flat amplitude-matched) in the "
                  "high-amplitude region, one-sided exact permutation",
        delta=d_occ, p=p_occ, n_assignments=total, fires=fires,
        min_attainable_p=1.0 / total)
    out["outcome_cell"] = "i" if fires else "ii"

    # dose secondaries (descriptive) + scores
    if len(ung) == 4:
        for name, rs in (("hetero", het), ("flat", flat)):
            d_d, p_d, _ = exact_perm_p(
                [r["occupancy"] for r in rs + ung],
                [True] * 4 + [False] * 4, one_sided=False)
            out[f"dose_{name}_vs_ungated2"] = dict(
                delta=d_d, p=p_d, note="DESCRIPTIVE (dose axis)")
    if all(r["score"] is not None for r in ordered):
        d_sc, p_sc, _ = exact_perm_p([r["score"] for r in ordered],
                                     labels, one_sided=False)
        out["secondary_score"] = dict(delta=d_sc, p=p_sc,
                                      note="reporting only, two-sided")
    else:
        out["secondary_score"] = "UNAVAILABLE (missing scores.jsonl)"
    return out


def run(args):
    def expand(pat):
        if "," in pat:
            return [x.strip() for x in pat.split(",") if x.strip()]
        return sorted(globmod.glob(pat))

    groups = {"hetero": expand(args.hetero), "flat": expand(args.flat),
              "ungated2": expand(args.ungated)}
    assert all(groups.values()), "need --hetero, --flat and --ungated"
    assert args.output, "registered read must persist: pass --output"
    recs = [read_run(d, arm, args.n_perm)
            for arm, dirs in groups.items() for d in dirs]
    out = aggregate(recs)
    out["gate_region"] = dict(key=GATE_KEY, index=GATE_INDEX,
                              threshold=GATE_THRESHOLD)
    out["mod_range"] = [MOD_LO, MOD_HI]
    out["flat_scale"] = FLAT_SCALE
    out["n_perm"] = args.n_perm
    out["fit_counters"] = {r["run_dir"]: fit_counters(r["run_dir"],
                                                      args.expect_steps)
                           for r in recs}
    out["per_run"] = [{k: r.get(k) for k in
                       ("run_dir", "train_seed", "arm", "occupancy",
                        "n_steps", "valid", "score", "n_score_entries",
                        "probe")}
                      for r in recs]
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "se_grad_read.json"), "w") as f:
        json.dump(out, f, indent=1)
    mc = out.get("manipulation_check", {})
    print(f"MANIPULATION CHECK: {mc.get('successes')}/"
          f"{mc.get('n_valid_probes')} valid -> "
          f"passes={mc.get('passes')} | probe ckpt: "
          f"{out['probe_ckpt_flag']}")
    if isinstance(out["primary"], dict):
        pr = out["primary"]
        print(f"GRAD PRIMARY (vs flat): delta={pr['delta']:+.4f} "
              f"p={pr['p']:.4f} fires={pr['fires']} -> cell "
              f"{out['outcome_cell']}")
    else:
        print(f"GRAD PRIMARY: {out['primary']} -> cell "
              f"{out['outcome_cell']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture_run(tmp, name, seed, arm, occ_frac, probe_hot=None,
                 probe_cal=0.1, score=1.0, n_steps=120000, chunk=5000,
                 with_probe=None):
    """Synthetic gradient-wave run for `arm`. Probe fixtures (hetero
    always, flat when with_probe) come from se_read's reviewed
    cross-implementation generator; the grad config then overwrites
    se_read's stub config."""
    rd = os.path.join(tmp, name)
    make_probe = (arm == "hetero") if with_probe is None else with_probe
    if make_probe:
        se_read._fixture_run(tmp, name, seed, hot=(probe_hot or {}),
                             cal=probe_cal)
    os.makedirs(os.path.join(rd, "replay"), exist_ok=True)
    rng = np.random.default_rng(seed)
    left, ci = n_steps, 0
    while left > 0:
        n = min(chunk, left)
        n_hi = int(round(occ_frac * n))
        col = np.concatenate([
            np.full(n_hi, GATE_THRESHOLD + 0.1),
            np.full(n - n_hi, GATE_THRESHOLD - 0.1)])
        rng.shuffle(col)
        arr = np.zeros((n, 8), np.float32)
        arr[:, GATE_INDEX] = col
        arr[:, 1:] = rng.normal(0, 1, (n, 7))
        np.savez(os.path.join(rd, "replay", f"chunk{ci:04d}.npz"),
                 position=arr, is_first=np.zeros(n, bool))
        left -= n
        ci += 1
    scale = FLAT_SCALE if arm == "flat" else 1.0
    mod = (f"  mod_key: '{MOD_KEY}'\n  mod_index: {MOD_INDEX}\n"
           f"  mod_lo: {MOD_LO}\n  mod_hi: {MOD_HI}\n"
           if arm == "hetero" else
           "  mod_key: ''\n  mod_index: 0\n"
           "  mod_lo: 0.0\n  mod_hi: 1.0\n")
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(f"seed: {seed}\n"
                f"planted:\n  gate_key: ''\n"
                f"distractor:\n  gate_key: ''\n  dim: {DIM}\n"
                f"  scale: {scale}\n  basesd: {BASESD_N}\n{mod}"
                f"run:\n  steps: 500000.0\n")
    with open(os.path.join(rd, "scores.jsonl"), "w") as f:
        for i in range(150):
            f.write(json.dumps({"step": i,
                                "episode/score": score + 0.01 * (i % 3)})
                    + "\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 499968}) + "\n")
    return rd


def _batch(tmp, sub, het_occ, flat_occ, ung_occ, probe_hot, **kw):
    het = [_fixture_run(os.path.join(tmp, sub), f"h{i}", 28 + i,
                        "hetero", het_occ(i) if callable(het_occ)
                        else het_occ,
                        probe_hot=probe_hot(i) if callable(probe_hot)
                        else probe_hot, **kw) for i in range(4)]
    flat = [_fixture_run(os.path.join(tmp, sub), f"f{i}", 36 + i,
                         "flat", flat_occ) for i in range(4)]
    ung = [_fixture_run(os.path.join(tmp, sub), f"u{i}", 32 + i,
                        "ungated2", ung_occ) for i in range(4)]
    return het, flat, ung


def _read_all(het, flat, ung, n_perm):
    return ([read_run(d, "hetero", n_perm) for d in het]
            + [read_run(d, "flat", n_perm) for d in flat]
            + [read_run(d, "ungated2", n_perm) for d in ung])


def selfcheck():
    import tempfile
    HOT = {"distractor": 5.0}
    NP = se_read.FIX_PERM
    with tempfile.TemporaryDirectory() as tmp:
        # A) cell i: probes hot, hetero above amplitude-matched flat
        het, flat, ung = _batch(tmp, "A", 0.65, 0.45, 0.44, HOT,
                                score=2.0)
        agg = aggregate(_read_all(het, flat, ung, NP))
        assert agg["manipulation_check"]["passes"]
        assert agg["primary"]["fires"] and agg["outcome_cell"] == "i"
        assert abs(agg["primary"]["p"] - 1 / 70) < 1e-12
        assert "dose_hetero_vs_ungated2" in agg
        assert agg["ungated2_vs_prior_ungated"]["prior"] == 0.560

        # B) cell ii: probes hot, hetero == flat (no gradient-following)
        het, flat, ung = _batch(tmp, "B", lambda i: 0.48 + 0.001 * i,
                                0.48, 0.56, HOT)
        agg = aggregate(_read_all(het, flat, ung, NP))
        assert agg["manipulation_check"]["passes"]
        assert not agg["primary"]["fires"] and agg["outcome_cell"] == "ii"

        # C) cell iii: valid probes at parity -> removal wording OK
        het, flat, ung = _batch(tmp, "C", 0.65, 0.45, 0.45, {})
        agg = aggregate(_read_all(het, flat, ung, NP))
        assert not agg["manipulation_check"]["passes"]
        assert agg["outcome_cell"] == "iii"
        assert "removed the misprice" in agg["primary"]
        # descriptives still present on early return (review #24 m9)
        assert "occupancy_hetero" in agg and "occupancy_ungated2" in agg

        # C2) cell iv: probes INVALID (calibration breach) -> no
        #     removal claim (review #24 M4)
        het, flat, ung = _batch(tmp, "C2", 0.65, 0.45, 0.45, {},
                                probe_cal=0.6)
        agg = aggregate(_read_all(het, flat, ung, NP))
        assert agg["outcome_cell"] == "iv"
        assert "probe instrument" in agg["primary"]
        assert "removed" not in agg["primary"]

        # C3) gate boundary (review #24 m7): 3/4 hot passes, 2/4 fails
        het, flat, ung = _batch(
            tmp, "C3", 0.5, 0.5, 0.5,
            lambda i: (HOT if i < 3 else {}))
        agg = aggregate(_read_all(het, flat, ung, NP))
        assert agg["manipulation_check"]["successes"] == 3
        assert agg["manipulation_check"]["passes"]
        het, flat, ung = _batch(
            tmp, "C4", 0.5, 0.5, 0.5,
            lambda i: (HOT if i < 2 else {}))
        agg = aggregate(_read_all(het, flat, ung, NP))
        assert agg["manipulation_check"]["successes"] == 2
        assert not agg["manipulation_check"]["passes"]
        assert agg["outcome_cell"] == "iii"

        # D) identity gates
        rd = _fixture_run(os.path.join(tmp, "D"), "x", 28, "ungated2",
                          0.5)
        try:
            read_run(rd, "hetero", NP)
            raise SystemExit("arm-identity gate FAILED to fire")
        except AssertionError as e:
            assert "mod target" in str(e) or "mod_key" in str(e)
        rd = _fixture_run(os.path.join(tmp, "D2"), "y", 36, "ungated2",
                          0.5)
        try:
            read_run(rd, "flat", NP)     # scale 1.0 passed as flat
            raise SystemExit("flat-scale gate FAILED to fire")
        except AssertionError as e:
            assert "scale" in str(e)
        rd = _fixture_run(os.path.join(tmp, "D3"), "z", 28, "hetero",
                          0.5, probe_hot=HOT)
        cfg = open(os.path.join(rd, "config.yaml")).read()
        with open(os.path.join(rd, "config.yaml"), "w") as f:
            f.write(cfg.replace(f"dim: {DIM}", "dim: 0"))
        try:
            read_run(rd, "hetero", NP)   # D-only leak (review #24 M3)
            raise SystemExit("dim gate FAILED to fire")
        except AssertionError as e:
            assert "mislaunched" in str(e)
        rd = _fixture_run(os.path.join(tmp, "D4"), "w", 28, "hetero",
                          0.5, probe_hot=HOT)
        os.remove(os.path.join(rd, "se_probe", "se_probe.json"))
        try:
            read_run(rd, "hetero", NP)   # missing probe (review #24 m13)
            raise SystemExit("probe-missing gate FAILED to fire")
        except AssertionError as e:
            assert "bundle spec" in str(e)
        # wrong seed family fatal
        het, flat, ung = _batch(tmp, "D5", 0.6, 0.4, 0.4, HOT)
        bad = [_fixture_run(os.path.join(tmp, "D5"), f"b{i}", 40 + i,
                            "hetero", 0.6, probe_hot=HOT)
               for i in range(4)]
        try:
            aggregate(_read_all(bad, flat, ung, NP))
            raise SystemExit("seed-family gate FAILED to fire")
        except AssertionError as e:
            assert "hetero seeds" in str(e)

        # E) end-to-end run() with globs + output json
        outd = os.path.join(tmp, "out")
        out = run(parse_args([
            "--hetero", os.path.join(tmp, "A", "h*"),
            "--flat", os.path.join(tmp, "A", "f*"),
            "--ungated", os.path.join(tmp, "A", "u*"),
            "--n_perm", str(NP), "--output", outd]))
        assert out["outcome_cell"] == "i"
        assert out["probe_ckpt_flag"].startswith("OK")
        assert out["gate_region"]["key"] == GATE_KEY
        assert os.path.exists(os.path.join(outd, "se_grad_read.json"))

    print("se_grad_read selfcheck PASS (A cell-i p=1/70 vs flat + dose "
          "descriptives, B cell-ii, C cell-iii removal-wording w/ valid "
          "probes + descriptives-on-early-return, C2 cell-iv no-removal-"
          "claim on invalid probes, C3/C4 gate boundary 3/4-vs-2/4, D "
          "identity gates x5 incl. dim/scale/probe-missing/seed-family, "
          "E end-to-end)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
