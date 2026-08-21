"""SE amplitude-DOSE ladder FROZEN reader (Track B1;
PREREG_nfi_dose_20260821.md, post-review-#28 form; built + selfchecked
BEFORE the wave's compute; ONE execution).

Wave: 4 doses x 4 seeds, Stage-1 config with DISTRACTOR_SCALE in
{0.10, 0.25, 0.50, 0.75} (bootstrap ON; the dose axis moves amplitude
and NOTHING else — enforced by the full pin block incl. theta and
bootstrap_prob, #28 M1).

  DEGENERACY GATE (NOBOOT pattern; Stage-1 reference dirs identity-
    gated, #28 m4): GLOBAL-COLLAPSE verdict iff >= 2 runs collapse; a
    single collapsed/invalid/missing run routes to the FLAGGED-SUBSET
    path (#28 M6): primary on the usable runs, requiring >= 14 usable
    AND >= 3 per dose.
  NORMALIZER-FLOOR accounting (#28 M2, quantitatively resolved by the
    review: the floor does NOT bind at dose 0.10 — 2.3x margin — and
    if it ever bound it would DEFLATE theta_1 at low doses, i.e. bias
    AGAINST the registered direction): per-run floor fields emitted;
    a FIRE is valid under a binding floor; a NULL with any floored
    run is NOT-ADJUDICABLE.
  PRIMARY (confirmatory): theta_1(distractor) DECREASES in scale —
    Spearman rho < 0, one-sided negative label permutation (100,000
    vectorized draws, fixed rng; ties handled by average ranks);
    FIRES iff p <= .05 and rho < 0.
  ESTIMAND SCOPE (#28 M5): theta_1 is the RELATIVE (per-unit-data-
    variance) misprice; the un-normalized decoded disagreement is
    emitted per run and FALLS with dose — "priced more favorably",
    never "attracts more disagreement".
  DESCRIPTIVE (registered, never fire): per-dose theta_1 / coverage /
    level / SCORE-TAIL curves (BCa at n=4 is indicative only);
    coverage trend emitted unconditionally; anchors descriptive.

Run:  python -m uncfield.se_dose_read --runs "<glob>"
          --stage1_runs "<glob>" --output <dir>
      python -m uncfield.se_dose_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os

import numpy as np
from scipy.stats import rankdata, spearmanr

from uncfield.se_grad_read import BASESD_N, DIM
from uncfield.se_mask import bca_interval
from uncfield.se_noboot_read import COLLAPSE_FACTOR, _levels
from uncfield.se_read import fit_counters
from uncfield import se_read

DOSES = (0.10, 0.25, 0.50, 0.75)
DOSE_SEEDS = {0.10: set(range(52, 56)), 0.25: set(range(56, 60)),
              0.50: set(range(60, 64)), 0.75: set(range(64, 68))}
ALL_SEEDS = set(range(52, 68))
N_PERM_LABELS = 100000
PERM_SEED = 2026_08_21
ALPHA = 0.05
MIN_USABLE = 14
MIN_PER_DOSE = 3
TASK = "dmc_cheetah_run"
SCORE_TAIL = 100
ANCHORS = {"stage1@1.0": 5.26, "flat@0.359": 7.10}   # descriptive only


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="", help="the 16 dose runs (glob)")
    p.add_argument("--stage1_runs", default="",
                   help="Stage-1 dirs — degeneracy-gate reference")
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--expect_steps", type=float, default=5e5)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def spearman_perm_null(theta, scale, n_draws, seed):
    """Vectorized one-sided-negative label-permutation null for
    Spearman rho with tied scale labels (average ranks)."""
    rt = rankdata(theta)
    rs = rankdata(scale)
    rt_c = rt - rt.mean()
    dt = np.sqrt((rt_c ** 2).sum())
    rs_c0 = rs - rs.mean()
    rho_obs = float((rs_c0 @ rt_c) / (dt * np.sqrt((rs_c0 ** 2).sum())))
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.tile(rs, (n_draws, 1)), axis=1)
    pc = perms - rs.mean()
    rho_null = (pc @ rt_c) / (dt * np.sqrt((pc ** 2).sum(1)))
    p = float((1 + (rho_null <= rho_obs).sum()) / (1 + n_draws))
    return rho_obs, p


def coverage_proxy(run_dir):
    """Pinned cost estimand: sum over position+velocity dims of
    log(std) over the ENTIRE replay. NEVER fatal (#28 B1): returns
    (None, 0, flag) when replay is unavailable."""
    rd = os.path.join(run_dir, "replay")
    try:
        files = sorted(f for f in os.listdir(rd) if f.endswith(".npz"))
        assert files
        acc = {}
        for f in files:
            with np.load(os.path.join(rd, f)) as z:
                for k in ("position", "velocity"):
                    if k in z:
                        v = np.asarray(z[k], np.float64)
                        acc.setdefault(k, []).append(
                            v.reshape(-1, v.shape[-1]))
        assert set(acc) == {"position", "velocity"}, sorted(acc)  # m6
        total, n_steps = 0.0, 0
        for k, chunks in acc.items():
            v = np.concatenate(chunks, 0)
            n_steps = max(n_steps, v.shape[0])
            total += float(np.log(np.maximum(v.std(0), 1e-12)).sum())
        return total, n_steps, "OK"
    except (OSError, AssertionError) as e:
        return None, 0, f"REPLAY-UNAVAILABLE ({type(e).__name__})"


def score_tail(run_dir):
    scores = []
    try:
        with open(os.path.join(run_dir, "scores.jsonl")) as f:
            for line in f:
                rec = json.loads(line)
                if "episode/score" in rec:
                    scores.append(float(rec["episode/score"]))
    except OSError:
        pass
    return (float(np.mean(scores[-SCORE_TAIL:])) if scores else None,
            len(scores))


def read_run(run_dir, n_perm):
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = cfg["distractor"]
    pl_ = cfg["planted"]
    ex = cfg["agent"]["expl"]
    scale = float(d["scale"])
    dose = min(DOSES, key=lambda x: abs(x - scale))
    assert abs(scale - dose) < 1e-9, (
        f"{run_dir}: scale {scale} is not a registered dose {DOSES}")
    # full pin block (#28 M1): NOBOOT m11 pins + dose-specific extras
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
    assert int(d["dim"]) == DIM and abs(float(d["basesd"]) - BASESD_N) < 1e-9
    assert abs(float(d.get("theta", 0.1)) - 0.1) < 1e-9, (
        f"{run_dir}: distractor.theta {d.get('theta')} != 0.1 — the "
        f"dose axis must move amplitude and nothing else")
    assert str(d.get("gate_key", "")) == "" and \
        str(d.get("mod_key", "")) == "", f"{run_dir}: gates/mod must be off"
    assert str(pl_.get("gate_key", "")) == ""
    rec = se_read.read_run(run_dir, "se_probe", n_perm)
    rec["train_seed_cfg"] = int(cfg["seed"])
    assert rec["train_seed_cfg"] in DOSE_SEEDS[dose], (
        f"{run_dir}: seed {rec['train_seed_cfg']} not registered for "
        f"dose {dose}")
    rec["dose"] = dose
    lv = _levels(run_dir)
    for ch in ("distractor",):
        stored = rec["channels"][ch]["vs_source"]
        rc = lv["theta_recomputed"].get(ch)
        assert rc is not None and abs(rc - stored) <= 1e-4 * max(
            1.0, abs(stored)), (
            f"{run_dir}/{ch}: theta_1 provenance mismatch")
    rec["levels"] = lv
    # normalizer-floor + un-normalized decomposition (#28 M2/M5)
    npz = np.load(os.path.join(run_dir, "se_probe",
                               "se_probe_dims.npz"))
    dk = np.asarray(npz["dim_key"]).astype(str)
    raw_norms = np.asarray(npz["raw_norms"], np.float64)
    floor = float(npz["norm_floor"])
    dis_cols = dk == "distractor"
    dis_raw = raw_norms[dis_cols]
    dims_d = np.asarray(npz["dims_d"], np.float64)
    norms_eff = np.maximum(raw_norms, floor)
    rec["floor"] = dict(
        distractor_raw_norm_mean=float(dis_raw.mean()),
        norm_floor=floor,
        floor_binds=bool((dis_raw < floor).any()),
        floor_margin=float(dis_raw.min() / floor))
    rec["unnorm_distractor_var"] = float(
        (dims_d[dis_cols] * norms_eff[dis_cols] ** 2).mean())
    cov, cov_steps, cov_flag = coverage_proxy(run_dir)
    rec["coverage"], rec["coverage_steps"], rec["coverage_flag"] = \
        cov, cov_steps, cov_flag
    rec["score"], rec["n_score_entries"] = score_tail(run_dir)
    return rec


def stage1_reference_gated(stage1_dirs):
    """Stage-1 level minima with identity gates (#28 m4)."""
    import yaml
    real_means, intr = [], []
    for dd in stage1_dirs:
        with open(os.path.join(dd, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        assert abs(float(cfg["distractor"]["scale"]) - 1.0) < 1e-9, (
            f"{dd}: reference dir is not a Stage-1 run (scale != 1.0)")
        assert int(cfg["seed"]) in set(range(10, 18)), (
            f"{dd}: reference seed {cfg['seed']} not in [10..17]")
        lv = _levels(dd)
        real_means.append(lv["real_mean"])
        intr.append(lv["intrinsic_mean"])
    assert len(real_means) >= 4, "need the Stage-1 bundle"
    return dict(real_mean_min=float(min(real_means)),
                intrinsic_min=float(min(intr)), n=len(real_means))


def aggregate(recs, ref):
    seeds = sorted(r["train_seed_cfg"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    assert set(seeds) <= ALL_SEEDS, (              # #28 B2: subset OK
        f"seeds {sorted(set(seeds) - ALL_SEEDS)} outside [52..67]")
    missing = sorted(ALL_SEEDS - set(seeds))
    invalid = [r["run_dir"] for r in recs if not r["valid"]]
    collapsed = [r["run_dir"] for r in recs
                 if r["levels"]["real_mean"]
                 < ref["real_mean_min"] / COLLAPSE_FACTOR
                 or r["levels"]["intrinsic_mean"]
                 < ref["intrinsic_min"] / COLLAPSE_FACTOR]
    floored = [r["run_dir"] for r in recs if r["floor"]["floor_binds"]]
    out = dict(n_loaded=len(recs), missing_seeds=missing,
               invalid_runs=invalid, collapsed_runs=collapsed,
               floored_runs=floored,
               instrument_flag=("CLEAN" if not invalid and not missing
                                and not collapsed
                                else "VALIDITY-VIOLATIONS"),
               probe_ckpt_flag=("OK" if all(r["ckpt_final_ok"]
                                            for r in recs)
                                else "UNVERIFIED"),
               stage1_reference=ref, anchors_descriptive=ANCHORS)

    # unconditional descriptives (#28 M3/M4/m7): per-dose curves incl.
    # score tails + per-key levels; coverage trend whenever computable
    curves = {}
    for dose in DOSES:
        rs = [r for r in recs if r["dose"] == dose and r["valid"]]
        row = dict(n=len(rs))
        for f, get in (
                ("theta1",
                 lambda r: r["channels"]["distractor"]["vs_source"]),
                ("coverage", lambda r: r["coverage"]),
                ("real_level", lambda r: r["levels"]["real_mean"]),
                ("position_level",
                 lambda r: r["levels"]["per_key"].get("position")),
                ("distractor_level",
                 lambda r: r["levels"]["per_key"].get("distractor")),
                ("unnorm_distractor_var",
                 lambda r: r["unnorm_distractor_var"]),
                ("score", lambda r: r["score"])):
            vals = [get(r) for r in rs if get(r) is not None]
            if len(vals) >= 3:
                m, lo, hi = bca_interval(
                    np.asarray(vals), np.random.default_rng(
                        int(dose * 1000)), 2000)
                row[f] = dict(mean=m, bca=[lo, hi],
                              bca_note="n=4 BCa is indicative only",
                              per_run=vals)
            else:
                row[f] = dict(mean=(float(np.mean(vals)) if vals
                                    else None), per_run=vals)
        curves[str(dose)] = row
    out["dose_curves"] = curves
    cov_pairs = [(r["coverage"], r["dose"]) for r in recs
                 if r["coverage"] is not None]
    if len(cov_pairs) >= 8:
        cv, sc_ = zip(*cov_pairs)
        out["coverage_trend_descriptive"] = dict(
            rho=float(spearmanr(np.asarray(cv),
                                np.asarray(sc_)).statistic),
            n=len(cov_pairs),
            note="DESCRIPTIVE only — no registered fire; a null is "
                 "publishable robustness")

    # gates with partial paths (#28 M6)
    if len(collapsed) >= 2:
        out["primary"] = (f"GLOBAL-DISAGREEMENT-COLLAPSE in "
                          f"{len(collapsed)} runs — trend not "
                          f"adjudicable")
        out["outcome"] = "GLOBAL-COLLAPSE"
        return out
    usable = [r for r in recs
              if r["valid"] and r["run_dir"] not in collapsed]
    per_dose = {dd: sum(1 for r in usable if r["dose"] == dd)
                for dd in DOSES}
    out["usable"] = dict(n=len(usable), per_dose=per_dose,
                         flagged=bool(len(usable) < 16))
    if len(usable) < MIN_USABLE or min(per_dose.values()) < MIN_PER_DOSE:
        out["primary"] = (f"NOT-ADJUDICABLE: {len(usable)} usable runs "
                          f"(need >= {MIN_USABLE} and >= "
                          f"{MIN_PER_DOSE}/dose)")
        out["outcome"] = "NOT-ADJUDICABLE"
        return out

    theta = np.asarray([r["channels"]["distractor"]["vs_source"]
                        for r in usable])
    scale = np.asarray([r["dose"] for r in usable])
    rho_obs, p = spearman_perm_null(theta, scale, N_PERM_LABELS,
                                    PERM_SEED)
    fires = bool(p <= ALPHA and rho_obs < 0)
    out["primary"] = dict(
        statement="theta_1(distractor) decreases in amplitude scale "
                  "(RELATIVE, per-unit-data-variance estimand — #28 "
                  "M5): Spearman < 0, one-sided label permutation "
                  f"({N_PERM_LABELS} draws, seed {PERM_SEED})"
                  + ("; FLAGGED SUBSET" if len(usable) < 16 else ""),
        rho=rho_obs, p=p, fires=fires, n=len(usable))
    if fires:
        out["outcome"] = "DOSE-LAW-CONFIRMED"
    elif floored:
        # #28 M2: a binding floor deflates low-dose theta_1 — it can
        # mask the registered effect, never manufacture it
        out["outcome"] = "NOT-ADJUDICABLE-FLOOR (null with a binding " \
                         "normalizer floor is uninterpretable)"
    else:
        out["outcome"] = "DOSE-LAW-NOT-CONFIRMED"
    return out


def run(args):
    def expand(pat):
        return (sorted(globmod.glob(pat)) if "," not in pat
                else [x.strip() for x in pat.split(",") if x.strip()])

    dirs = expand(args.runs)
    s1 = expand(args.stage1_runs)
    assert dirs and s1, "need --runs and --stage1_runs"
    assert args.output, "registered read must persist: pass --output"
    ref = stage1_reference_gated(s1)
    recs = [read_run(d, args.n_perm) for d in dirs]
    out = aggregate(recs, ref)
    out["fit_counters"] = {r["run_dir"]: fit_counters(
        r["run_dir"], args.expect_steps) for r in recs}
    out["per_run"] = [
        {"run_dir": r["run_dir"], "train_seed": r["train_seed_cfg"],
         "dose": r["dose"], "valid": r["valid"],
         "cal_real_max": r["cal_real_max"],
         "ckpt_final_ok": r["ckpt_final_ok"],
         "theta1": r["channels"]["distractor"]["vs_source"],
         "share": r["channels"]["distractor"]["share_vs_real"],
         "p_perm": r["channels"]["distractor"]["p_perm"],
         "per_key_levels": r["levels"]["per_key"],
         "intrinsic_mean": r["levels"]["intrinsic_mean"],
         "unnorm_distractor_var": r["unnorm_distractor_var"],
         "floor": r["floor"], "coverage": r["coverage"],
         "coverage_steps": r["coverage_steps"],
         "coverage_flag": r["coverage_flag"], "score": r["score"],
         "n_score_entries": r["n_score_entries"],
         "n_eval": r["n_eval"]} for r in recs]
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "se_dose_read.json"),
              "w") as f:
        json.dump(out, f, indent=1)
    if isinstance(out["primary"], dict):
        pr = out["primary"]
        print(f"DOSE PRIMARY: rho={pr['rho']:+.3f} p={pr['p']:.5f} "
              f"fires={pr['fires']} -> {out['outcome']}")
    else:
        print(f"DOSE PRIMARY: {out['primary']} -> {out['outcome']}")
    for dose in DOSES:
        t = out["dose_curves"][str(dose)]["theta1"]
        print(f"  scale {dose}: theta1 "
              f"{t['mean'] if t['mean'] is None else round(t['mean'], 3)}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, dose, theta, level_scale=1.0,
             intrinsic=3.5e-4, coverage_sd=1.0, boot=True,
             theta_ou=0.1, source_key="position", with_replay=True,
             doctor_vs_source=None):
    from uncfield import se_read as sr
    rd = os.path.join(tmp, name)
    sr._fixture_run(tmp, name, seed, hot={"distractor": 5.0})
    pdir = os.path.join(rd, "se_probe")
    npz = dict(np.load(os.path.join(pdir, "se_probe_dims.npz")))
    dims_d = np.asarray(npz["dims_d"])
    dk = np.asarray(npz["dim_key"]).astype(str)
    per_key = {k: float(dims_d[dk == k].mean())
               for k in set(dk.tolist())}
    want = theta * per_key["position"]
    cols = dk == "distractor"
    dims_d = dims_d.copy()
    dims_d[cols] = dims_d[cols] * (want / max(per_key["distractor"],
                                              1e-300))
    dims_d = dims_d * level_scale
    npz["dims_d"] = dims_d
    npz["raw_norms"] = np.full(dims_d.size, 1.0)
    npz["norm_floor"] = np.float32(0.05)
    np.savez(os.path.join(pdir, "se_probe_dims.npz"), **npz)
    import zlib
    from uncfield.se_probe import perm_null as probe_perm_null
    with open(os.path.join(pdir, "se_probe.json")) as f:
        pj = json.load(f)
    per_key = {k: float(dims_d[dk == k].mean())
               for k in set(dk.tolist())}
    real_keys = [k for k in set(dk.tolist())
                 if not k.startswith("planted") and k != "distractor"]
    for ch in pj["channels"]:
        mask = np.asarray([x == ch or x in real_keys for x in dk])
        _, p, _ = probe_perm_null(
            dims_d[mask],
            np.asarray([x == ch for x in dk[mask]]),
            sr.FIX_PERM,
            np.random.default_rng(0 + zlib.crc32(ch.encode()) % 2 ** 16))
        pj["channels"][ch]["p_perm"] = p
        pj["channels"][ch]["vs_source"] = (per_key[ch]
                                           / per_key["position"])
    if doctor_vs_source is not None:
        pj["channels"]["distractor"]["vs_source"] = doctor_vs_source
    pj["pse2"]["intrinsic_mean"] = intrinsic
    pj["source_key"] = "position"
    with open(os.path.join(pdir, "se_probe.json"), "w") as f:
        json.dump(pj, f)
    if with_replay:
        os.makedirs(os.path.join(rd, "replay"), exist_ok=True)
        rng = np.random.default_rng(seed)
        np.savez(os.path.join(rd, "replay", "chunk0000.npz"),
                 position=rng.normal(0, coverage_sd,
                                     (120000, 8)).astype(np.float32),
                 velocity=rng.normal(0, coverage_sd,
                                     (120000, 9)).astype(np.float32),
                 is_first=np.zeros(120000, bool))
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(f"seed: {seed}\ntask: dmc_cheetah_run\n"
                f"planted:\n  gate_key: ''\n"
                f"  source_key: '{source_key}'\n  basesd: 0.0976\n"
                f"distractor:\n  gate_key: ''\n  mod_key: ''\n"
                f"  dim: {DIM}\n  scale: {dose}\n"
                f"  basesd: {BASESD_N}\n  theta: {theta_ou}\n"
                f"agent:\n  expl:\n    mode: p2e\n    disag_ens: 8\n"
                f"    disag_scale: 1000.0\n"
                f"    disag_bootstrap_prob: 0.8\n"
                f"    disag_bootstrap: {'true' if boot else 'false'}\n"
                f"run:\n  steps: 500000.0\n")
    with open(os.path.join(rd, "scores.jsonl"), "w") as f:
        for i in range(150):
            f.write(json.dumps({"step": i, "episode/score":
                                10.0 + 0.1 * (i % 5)}) + "\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 499968}) + "\n")
    return rd


def _wave(tmp, sub, theta_of, **kw):
    runs, seed = [], 52
    for dose in DOSES:
        for j in range(4):
            runs.append(_fixture(os.path.join(tmp, sub),
                                 f"d{dose}_{j}", seed, dose,
                                 theta_of(dose, j), **kw))
            seed += 1
    return runs


def selfcheck():
    import tempfile
    NP = se_read.FIX_PERM
    with tempfile.TemporaryDirectory() as tmp:
        s1 = [_fixture(os.path.join(tmp, "s1"), f"s{i}", 10 + i, 1.0,
                       5.0) for i in range(4)]
        ref = stage1_reference_gated(
            [os.path.join(tmp, "s1", f"s{i}") for i in range(4)])
        # ref identity gate binds (#28 m4)
        try:
            stage1_reference_gated(
                [_fixture(os.path.join(tmp, "s1x"), "b", 10, 0.25,
                          5.0)])
            raise SystemExit("s1 identity gate FAILED to fire")
        except AssertionError as e:
            assert "not a Stage-1 run" in str(e)

        # vectorized null == spearmanr (sanity)
        th = np.array([8.0, 7.5, 7.9, 8.1, 6.0, 6.2, 6.1, 5.9,
                       5.0, 5.2, 5.1, 4.9, 4.0, 4.2, 4.1, 3.9])
        sc = np.repeat(DOSES, 4)
        rho, _ = spearman_perm_null(th, sc, 100, 1)
        assert abs(rho - spearmanr(th, sc).statistic) < 1e-12

        # A) dose law fires
        runs = _wave(tmp, "A", lambda d, j: 8.0 - 4.0 * d + 0.05 * j)
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome"] == "DOSE-LAW-CONFIRMED"
        assert agg["dose_curves"]["0.1"]["score"]["mean"] is not None
        assert "coverage_trend_descriptive" in agg

        # B) noisy null (per-run jitter, #28 m3) -> no fire
        rng0 = np.random.default_rng(7)
        jit = rng0.normal(0, 0.4, 16).tolist()
        runs = _wave(tmp, "B",
                     lambda d, j: 5.0 + jit[DOSES.index(d) * 4 + j])
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert not agg["primary"]["fires"]
        assert agg["outcome"] == "DOSE-LAW-NOT-CONFIRMED"

        # B2) weak negative trend must NOT fire (#28 m2)
        runs = _wave(tmp, "B2",
                     lambda d, j: 5.0 - 0.1 * d
                     + [0.3, -0.2, 0.25, -0.3][j])
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert not agg["primary"]["fires"], agg["primary"]

        # B3) reversed trend must not fire
        runs = _wave(tmp, "B3", lambda d, j: 3.0 + 4.0 * d + 0.05 * j)
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert not agg["primary"]["fires"] and agg["primary"]["rho"] > 0

        # C) collapse: >=2 runs -> GLOBAL-COLLAPSE; exactly 1 ->
        #    flagged-subset primary (#28 M6)
        runs = _wave(tmp, "C", lambda d, j: 8.0 - 4.0 * d,
                     level_scale=1e-6, intrinsic=1e-9)
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome"] == "GLOBAL-COLLAPSE"
        assert "dose_curves" in agg
        runs = _wave(tmp, "C1", lambda d, j: 8.0 - 4.0 * d + 0.05 * j)
        _fixture(os.path.join(tmp, "C1"), "d0.1_0", 52, 0.10, 8.0,
                 level_scale=1e-6, intrinsic=1e-9)
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome"] == "DOSE-LAW-CONFIRMED"
        assert agg["usable"]["flagged"] and agg["primary"]["n"] == 15

        # D) missing run -> reported NOT-ADJUDICABLE-free path (#28 B2)
        runs = _wave(tmp, "D", lambda d, j: 8.0 - 4.0 * d + 0.05 * j)
        agg = aggregate([read_run(d, NP) for d in runs[:13]], ref)
        assert agg["outcome"] == "NOT-ADJUDICABLE"
        assert agg["missing_seeds"] and "dose_curves" in agg

        # E) replay-unavailable is NOT fatal (#28 B1)
        runs = _wave(tmp, "E", lambda d, j: 8.0 - 4.0 * d + 0.05 * j,
                     with_replay=False)
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome"] == "DOSE-LAW-CONFIRMED"
        assert all(r["coverage"] is None
                   for r in [read_run(runs[0], NP)])
        assert "coverage_trend_descriptive" not in agg

        # F) floor accounting: floored run + NULL -> NOT-ADJUDICABLE;
        #    floored run + FIRE -> stays confirmed (#28 M2)
        runs = _wave(tmp, "F", lambda d, j: 5.0 + 0.02 * j)
        rd = _fixture(os.path.join(tmp, "F"), "d0.1_0", 52, 0.10, 5.0)
        pth = os.path.join(rd, "se_probe", "se_probe_dims.npz")
        z = dict(np.load(pth))
        z["norm_floor"] = np.float32(2.0)      # floor above raw norms
        np.savez(pth, **z)
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome"].startswith("NOT-ADJUDICABLE-FLOOR")
        runs = _wave(tmp, "F2", lambda d, j: 8.0 - 4.0 * d + 0.05 * j)
        rd = _fixture(os.path.join(tmp, "F2"), "d0.1_0", 52, 0.10, 8.0)
        z = dict(np.load(os.path.join(rd, "se_probe",
                                      "se_probe_dims.npz")))
        z["norm_floor"] = np.float32(2.0)
        np.savez(os.path.join(rd, "se_probe", "se_probe_dims.npz"), **z)
        agg = aggregate([read_run(d, NP) for d in runs], ref)
        assert agg["outcome"] == "DOSE-LAW-CONFIRMED"

        # G) pin gates (#28 M1/m2): theta, source_key, vs_source doctor
        for kw, msg in ((dict(theta_ou=1.0), "theta"),
                        (dict(boot=False), "expl pins"),
                        (dict(doctor_vs_source=99.0), "provenance")):
            rd = _fixture(os.path.join(tmp, "G" + msg[:3]), "x", 52,
                          0.10, 5.0, **kw)
            try:
                read_run(rd, NP)
                raise SystemExit(f"{msg} gate FAILED to fire")
            except AssertionError as e:
                assert msg.split()[0] in str(e), (msg, str(e))

        # H) end-to-end
        outd = os.path.join(tmp, "out")
        out = run(parse_args([
            "--runs", os.path.join(tmp, "A", "d*"),
            "--stage1_runs", os.path.join(tmp, "s1", "s*"),
            "--n_perm", str(NP), "--output", outd]))
        assert out["outcome"] == "DOSE-LAW-CONFIRMED"
        assert os.path.exists(os.path.join(outd, "se_dose_read.json"))
    print("se_dose_read selfcheck PASS (A fire + score/coverage "
          "curves, B noisy-null, B2 weak-trend no-fire, B3 reversed, "
          "C collapse>=2 + single-collapse flagged-subset, D missing-"
          "run reported, E replay-unavailable non-fatal, F floor "
          "null->NOT-ADJUDICABLE + fire-survives, G theta/bootstrap/"
          "provenance gates, H end-to-end)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
