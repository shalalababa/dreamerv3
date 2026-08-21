"""SE avoidance-confirmatory FROZEN reader
(PREREG_nfi_avoidance_20260821.md §§2-3, post-review-#27 form; built +
selfchecked BEFORE the A1 wave's compute; ONE execution).

THE DIRECTION IS NOW REGISTERED (avoidance), after three prior
unregistered-direction observations.

Arms: hetero2 (amendment-3 hetero config verbatim, seeds 44-47) vs
flat2 (amendment-3 flat config verbatim, seeds 48-51).

Registered adjudication order (#27):
  0. DEGENERACY GATE: per-run real-dim level + rollout intrinsic_mean
     >= 1/10 of the Stage-1 minima (--stage1_runs, NOBOOT pattern);
     failure -> GLOBAL-COLLAPSE, nothing downstream adjudicated.
  1. MISPRICE GATE (amendment-3 form, re-registered): distractor
     p_perm < .05 in >= 3/4 VALID hetero2 probes (full se_read gates:
     provenance p-equality, final-ckpt, S >= 256, calibration) —
     without a live misprice there is nothing to avoid.
  2. GRADIENT-DELIVERY GATE: among REGION-ADJUDICABLE hetero2 runs
     (>= 32 anchors in each region, finite means; >= 3 of 4 required
     adjudicable else REGION-UNDERPOWERED), distractor disagreement
     higher in the high region in ALL of them. Near-degenerate on the
     pass side by construction (E[m^2] contrast) — its role is to
     catch collapse/instrument faults, DISCLOSED as such.
  3. OCCUPANCY VALIDITY: 4+4 valid runs (>= 1e5-step replay floor).
  4. PRIMARY: occupancy(hetero2) - occupancy(flat2) < 0, one-sided
     NEGATIVE exact permutation over C(8,4)=70; FIRES iff p <= .05
     and delta < 0. Power note: p <= .05 over 70 assignments requires
     the observed split among the 3 most extreme — near-complete
     separation; no power against moderate effects (registered).
  5. SECONDARY-1 (registered direction, GATED on the primary — fires
     alone licenses NOTHING): ranking delta(hetero2) < delta(flat2),
     one-sided, pse2 taken from the PROVENANCE-GATED se_read record.
Mechanism decomposition: DESCRIPTIVE within-run normalizer-invariant
ratios (dis_high/dis_low, real_high/real_low) per arm + BCa; no
outcome licenses any claim or venue change. All descriptives + score
emitted on every branch.

Run:  python -m uncfield.se_avoid_read --hetero "<glob>" --flat
          "<glob>" --stage1_runs "<glob>" --output <dir>
      python -m uncfield.se_avoid_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os

import numpy as np

from uncfield.se_mask import bca_interval
from uncfield.se_m3_read import (GATE_INDEX, GATE_THRESHOLD, MIN_STEPS,
                                 SCORE_TAIL, exact_perm_p, occupancy)
from uncfield.se_grad_read import (BASESD_N, DIM, FLAT_SCALE, MOD_HI,
                                   MOD_INDEX, MOD_KEY, MOD_LO)
from uncfield.se_noboot_read import COLLAPSE_FACTOR, _levels, \
    stage1_reference
from uncfield.se_probe import PLANTED_KEYS
from uncfield.se_read import fit_counters
from uncfield import se_read

HET2_SEEDS = set(range(44, 48))
FLAT2_SEEDS = set(range(48, 52))
ALPHA = 0.05
MIN_REGION = 32
TASK = "dmc_cheetah_run"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hetero", default="", help="4 hetero2 run dirs")
    p.add_argument("--flat", default="", help="4 flat2 run dirs")
    p.add_argument("--stage1_runs", default="",
                   help="Stage-1 dirs — degeneracy-gate reference")
    p.add_argument("--region_subdir", default="se_region_probe")
    p.add_argument("--probe_subdir", default="se_probe")
    p.add_argument("--n_perm", type=int, default=1000,
                   help="MUST match the se_probe executions")
    p.add_argument("--expect_steps", type=float, default=5e5)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def read_run(run_dir, arm, region_subdir, probe_subdir, n_perm):
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg["seed"])
    d = cfg["distractor"]
    pl_ = cfg["planted"]
    ex = cfg["agent"]["expl"]
    # full Stage-1-verbatim pin block (#27 B3, mirrors se_noboot_read)
    assert str(cfg.get("task", TASK)) == TASK, (
        f"{run_dir}: task {cfg.get('task')} != {TASK}")
    assert str(ex["mode"]) == "p2e" and \
        ex["disag_bootstrap"] is True and \
        int(ex["disag_ens"]) == 8 and \
        abs(float(ex["disag_scale"]) - 1000.0) < 1e-9, (
        f"{run_dir}: expl pins differ from Stage-1 (EXPL_CONFIG or "
        f"DISAG_BOOTSTRAP override leak?)")
    assert str(pl_["source_key"]) == "position" and \
        abs(float(pl_["basesd"]) - 0.0976) < 1e-9, (
        f"{run_dir}: planted pins differ from Stage-1")
    assert str(pl_.get("gate_key", "")) == "" and \
        str(d.get("gate_key", "")) == "", f"{run_dir}: gates must be off"
    assert int(d["dim"]) == DIM and abs(float(d["basesd"]) - BASESD_N) < 1e-9
    mk = str(d.get("mod_key", ""))
    scale = float(d["scale"])
    if arm == "hetero2":
        assert (mk == MOD_KEY and int(d["mod_index"]) == MOD_INDEX
                and abs(float(d["mod_lo"]) - MOD_LO) < 1e-9
                and abs(float(d["mod_hi"]) - MOD_HI) < 1e-9
                and abs(scale - 1.0) < 1e-9), f"{run_dir}: hetero2 pins"
    else:
        assert mk == "" and abs(scale - FLAT_SCALE) < 1e-9, (
            f"{run_dir}: flat2 pins")
    occ, n_steps = occupancy(run_dir, GATE_INDEX, GATE_THRESHOLD)

    # provenance-gated probe record (#27 B2): full se_read gates
    probe_json = os.path.join(run_dir, probe_subdir, "se_probe.json")
    assert os.path.exists(probe_json), (
        f"{run_dir}: se_probe/ missing — the §4 bundle spec requires "
        f"the probe on every run")
    pr = se_read.read_run(run_dir, probe_subdir, n_perm)
    with open(probe_json) as f:
        ps = json.load(f)["pse2"]
    lv = _levels(run_dir)

    # region probe with provenance binding (#27 M4)
    rdir = os.path.join(run_dir, region_subdir)
    rp = os.path.join(rdir, "se_region_probe.npz")
    assert os.path.exists(rp), (
        f"{run_dir}: se_region_probe/ missing — the §4 bundle spec "
        f"requires the region probe on every run")
    z = np.load(rp)
    with open(os.path.join(rdir, "se_region_probe.json")) as f:
        rj = json.load(f)
    assert rj["region"] == dict(key="position", index=0,
                                threshold=GATE_THRESHOLD), rj["region"]
    assert int(rj["n_eval"]) >= 256, (
        f"{run_dir}: region probe S {rj['n_eval']} < 256")
    assert os.path.basename(os.path.normpath(rj["run_logdir"])) == \
        os.path.basename(os.path.normpath(run_dir)), (
        f"{run_dir}: region probe json belongs to "
        f"{rj['run_logdir']}")
    ck_np = str(np.asarray(z["ckpt_basename"]))
    assert ck_np == os.path.basename(os.path.normpath(rj["ckpt"])), (
        f"{run_dir}: region npz/json ckpt mismatch")
    try:
        final = se_read.resolve_ckpt(run_dir, "")
        assert ck_np == os.path.basename(os.path.normpath(final)), (
            f"{run_dir}: region probe ckpt {ck_np} is not the final "
            f"checkpoint")
    except (FileNotFoundError, OSError):
        pass                       # ckpt dir unbundled -> UNVERIFIED
    except AssertionError as e:
        if "is not the final" in str(e):
            raise
        pass                       # resolve_ckpt's own no-dirs assert
    per_state = np.asarray(z["per_state"], np.float64)
    dk = np.asarray(z["dim_key"]).astype(str)
    coord = np.asarray(z["coord"], np.float64)
    hi = coord > GATE_THRESHOLD                     # recomputed (#27 M4)
    assert np.array_equal(hi, np.asarray(z["region_high"])), (
        f"{run_dir}: stored region labels != recomputed from coord")
    # cross-instrument tie: region probe and se_probe must describe
    # the same run+ckpt (dims_d = per-state mean, tolerance for device)
    pdims = np.load(os.path.join(run_dir, probe_subdir,
                                 "se_probe_dims.npz"))
    assert np.allclose(per_state.mean(0),
                       np.asarray(pdims["dims_d"], np.float64),
                       rtol=1e-3, atol=1e-9), (
        f"{run_dir}: region probe does not match se_probe (different "
        f"run/ckpt/windows?)")
    all_keys = sorted(set(dk.tolist()))
    real_cols = np.isin(dk, [k for k in all_keys
                             if k not in PLANTED_KEYS])
    dis_cols = dk == "distractor"
    assert dis_cols.any() and real_cols.any(), (      # #27 M5
        f"{run_dir}: region probe key set broken")
    n_high, n_low = int(hi.sum()), int((~hi).sum())
    region_adjudicable = bool(n_high >= MIN_REGION
                              and n_low >= MIN_REGION)
    region = dict(n_high=n_high, n_low=n_low,
                  adjudicable=region_adjudicable)
    if region_adjudicable:
        vals = dict(
            real_high=float(per_state[hi][:, real_cols].mean()),
            real_low=float(per_state[~hi][:, real_cols].mean()),
            dis_high=float(per_state[hi][:, dis_cols].mean()),
            dis_low=float(per_state[~hi][:, dis_cols].mean()))
        assert all(np.isfinite(v) for v in vals.values()), (   # M5
            f"{run_dir}: non-finite region means")
        vals["dis_ratio"] = vals["dis_high"] / max(vals["dis_low"],
                                                   1e-300)
        vals["real_ratio"] = vals["real_high"] / max(vals["real_low"],
                                                     1e-300)
        region.update(vals)

    scores = []
    try:
        with open(os.path.join(run_dir, "scores.jsonl")) as f:
            for line in f:
                rec = json.loads(line)
                if "episode/score" in rec:
                    scores.append(float(rec["episode/score"]))
    except OSError:
        pass
    return dict(run_dir=os.path.abspath(run_dir), arm=arm,
                train_seed=seed, occupancy=occ, n_steps=int(n_steps),
                valid=bool(n_steps >= MIN_STEPS),
                rankx_delta=float(ps["top_share"] - ps["base_share"]),
                probe=dict(p_perm=pr["channels"]["distractor"]["p_perm"],
                           theta1=pr["channels"]["distractor"]
                           ["vs_source"],
                           probe_valid=pr["valid"],
                           ckpt_final_ok=pr["ckpt_final_ok"]),
                levels=dict(real_mean=lv["real_mean"],
                            intrinsic_mean=lv["intrinsic_mean"]),
                region=region,
                score=(float(np.mean(scores[-SCORE_TAIL:]))
                       if scores else None))


def aggregate(recs, ref):
    het = sorted((r for r in recs if r["arm"] == "hetero2"),
                 key=lambda r: r["train_seed"])
    fl = sorted((r for r in recs if r["arm"] == "flat2"),
                key=lambda r: r["train_seed"])
    hs = [r["train_seed"] for r in het]
    fs = [r["train_seed"] for r in fl]
    assert len(set(hs)) == len(hs) and len(set(fs)) == len(fs)
    assert set(hs) == HET2_SEEDS, f"hetero2 seeds {hs} != [44..47]"
    assert set(fs) == FLAT2_SEEDS, f"flat2 seeds {fs} != [48..51]"
    invalid = [r["run_dir"] for r in recs if not r["valid"]]
    out = dict(n_hetero2=len(het), n_flat2=len(fl),
               invalid_runs=invalid,
               instrument_flag=("CLEAN" if not invalid
                                else "VALIDITY-VIOLATIONS"),
               probe_ckpt_flag=("OK" if all(r["probe"]["ckpt_final_ok"]
                                            for r in recs)
                                else "UNVERIFIED"))

    # unconditional descriptives (#27 m14 incl. score)
    for name, rs in (("hetero2", het), ("flat2", fl)):
        if len(rs) >= 3:
            m, lo, hi_ = bca_interval(
                np.asarray([r["occupancy"] for r in rs]),
                np.random.default_rng(31 if name == "hetero2" else 32),
                2000)
            out[f"occupancy_{name}"] = dict(mean=m, bca=[lo, hi_])
        ratios = [r["region"].get("dis_ratio") for r in rs
                  if r["region"]["adjudicable"]]
        rratios = [r["region"].get("real_ratio") for r in rs
                   if r["region"]["adjudicable"]]
        out[f"mechanism_{name}"] = dict(
            dis_ratio_per_run=ratios, real_ratio_per_run=rratios,
            note="DESCRIPTIVE within-run normalizer-invariant ratios; "
                 "no outcome licenses any claim or venue change")
    ordered_all = het + fl
    if all(r["score"] is not None for r in ordered_all):
        d_sc, p_sc, _ = exact_perm_p(
            [r["score"] for r in ordered_all],
            [True] * len(het) + [False] * len(fl), one_sided=False)
        out["score_two_sided"] = dict(delta=d_sc, p=p_sc,
                                      note="reporting only")

    # 0) degeneracy gate
    collapsed = [r["run_dir"] for r in recs
                 if r["levels"]["real_mean"]
                 < ref["real_mean_min"] / COLLAPSE_FACTOR
                 or r["levels"]["intrinsic_mean"]
                 < ref["intrinsic_min"] / COLLAPSE_FACTOR]
    out["collapsed_runs"] = collapsed
    if collapsed:
        out["primary"] = "GLOBAL-DISAGREEMENT-COLLAPSE — nothing " \
                         "downstream adjudicated"
        out["outcome"] = "GLOBAL-COLLAPSE"
        return out

    # 1) misprice gate (amendment-3 form)
    mp_succ = sum(1 for r in het
                  if r["probe"]["probe_valid"]
                  and r["probe"]["p_perm"] < 0.05)
    n_valid_probes = sum(1 for r in het if r["probe"]["probe_valid"])
    out["misprice_gate"] = dict(successes=mp_succ,
                                n_valid_probes=n_valid_probes,
                                passes=bool(mp_succ >= 3))
    if not out["misprice_gate"]["passes"]:
        out["primary"] = ("NOT-ADJUDICABLE: the misprice is not live "
                          "in >=3/4 valid hetero2 probes — nothing to "
                          "avoid")
        out["outcome"] = "MISPRICE-ABSENT"
        return out

    # 2) gradient-delivery gate over region-adjudicable hetero2 runs
    adj = [r for r in het if r["region"]["adjudicable"]]
    deliv = sum(1 for r in adj
                if r["region"]["dis_high"] > r["region"]["dis_low"])
    out["gradient_delivery"] = dict(
        n_adjudicable=len(adj), successes=deliv,
        passes=bool(len(adj) >= 3 and deliv == len(adj)),
        note="near-degenerate on the pass side by construction "
             "(E[m^2] contrast); its role is catching collapse/"
             "instrument faults")
    if len(adj) < 3:
        out["primary"] = "NOT-ADJUDICABLE: <3 region-adjudicable " \
                         "hetero2 runs (REGION-UNDERPOWERED)"
        out["outcome"] = "REGION-UNDERPOWERED"
        return out
    if not out["gradient_delivery"]["passes"]:
        out["primary"] = ("NOT-ADJUDICABLE: the manipulation did not "
                          "deliver a spatial disagreement gradient in "
                          "every adjudicable hetero2 run")
        out["outcome"] = "DELIVERY-FAILED"
        return out

    # 3) occupancy validity
    if invalid or len(het) != 4 or len(fl) != 4:
        out["primary"] = "NOT-ADJUDICABLE (need 4+4 valid runs)"
        out["outcome"] = "NOT-ADJUDICABLE"
        return out

    # 4) primary (one-sided NEGATIVE)
    ordered = het + fl
    labels = [True] * 4 + [False] * 4
    occ = [r["occupancy"] for r in ordered]
    d_neg, p_occ, total = exact_perm_p([-x for x in occ], labels,
                                       one_sided=True)
    d_occ = -d_neg
    fires = bool(p_occ <= ALPHA and d_occ < 0)
    out["primary"] = dict(
        statement="REGISTERED avoidance: occupancy(hetero2) < "
                  "occupancy(flat2), one-sided negative exact "
                  "permutation; power note: firing requires the "
                  "observed split among the 3 most extreme of 70",
        delta=d_occ, p=p_occ, n_assignments=total, fires=fires,
        min_attainable_p=1.0 / total)

    # 5) secondary (gated on the primary — no independent license)
    rk = [r["rankx_delta"] for r in ordered]
    d_rk, p_rk, _ = exact_perm_p([-x for x in rk], labels,
                                 one_sided=True)
    out["secondary_ranking"] = dict(
        delta=-d_rk, p=p_rk,
        fires=bool(fires and p_rk <= ALPHA and -d_rk < 0),
        note="GATED on the primary (#27 M10): firing alone licenses "
             "nothing; reported unconditionally")
    out["outcome"] = ("AVOIDANCE-CONFIRMED" if fires else
                      "AVOIDANCE-NOT-CONFIRMED")
    return out


def run(args):
    def expand(pat):
        return (sorted(globmod.glob(pat)) if "," not in pat
                else [x.strip() for x in pat.split(",") if x.strip()])

    het = expand(args.hetero)
    fl = expand(args.flat)
    s1 = expand(args.stage1_runs)
    assert het and fl, "need --hetero and --flat"
    assert s1, "degeneracy gate requires --stage1_runs"
    assert args.output, "registered read must persist: pass --output"
    ref = stage1_reference(s1)
    recs = ([read_run(d, "hetero2", args.region_subdir,
                      args.probe_subdir, args.n_perm) for d in het]
            + [read_run(d, "flat2", args.region_subdir,
                        args.probe_subdir, args.n_perm) for d in fl])
    out = aggregate(recs, ref)
    out["fit_counters"] = {r["run_dir"]: fit_counters(
        r["run_dir"], args.expect_steps) for r in recs}
    out["per_run"] = [{k: r.get(k) for k in
                       ("run_dir", "train_seed", "arm", "occupancy",
                        "n_steps", "valid", "rankx_delta", "probe",
                        "levels", "region", "score")} for r in recs]
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "se_avoid_read.json"),
              "w") as f:
        json.dump(out, f, indent=1)
    print(f"OUTCOME: {out['outcome']}")
    if isinstance(out.get("primary"), dict):
        pr = out["primary"]
        print(f"AVOID PRIMARY: delta={pr['delta']:+.4f} "
              f"p={pr['p']:.4f} fires={pr['fires']}")
        sr = out["secondary_ranking"]
        print(f"  ranking secondary: delta={sr['delta']:+.4f} "
              f"p={sr['p']:.4f} fires={sr['fires']}")
    else:
        print(f"AVOID PRIMARY: {out.get('primary')}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, arm, occ_frac, rankx=-0.05,
             dis_hi=2.0, dis_lo=1.0, real_hi=1.0, real_lo=1.0,
             probe_hot=None, probe_cal=0.1, level_scale=1.0,
             intrinsic=3.5e-4, n_hi_states=256, score=1.0,
             n_steps=120000, chunk=5000):
    """A1 run fixture: replay + grad-style config + a REAL se_read
    probe fixture (provenance-consistent) + a provenance-bound region
    npz consistent with the probe's dims_d."""
    from uncfield import se_read as sr
    rd = os.path.join(tmp, name)
    sr._fixture_run(tmp, name, seed,
                    hot=(probe_hot if probe_hot is not None
                         else {"distractor": 5.0}), cal=probe_cal)
    # levels + intrinsic for the degeneracy gate
    pdir = os.path.join(rd, "se_probe")
    npz = dict(np.load(os.path.join(pdir, "se_probe_dims.npz")))
    npz["dims_d"] = npz["dims_d"] * level_scale
    np.savez(os.path.join(pdir, "se_probe_dims.npz"), **npz)
    with open(os.path.join(pdir, "se_probe.json")) as f:
        pj = json.load(f)
    # level_scale is ratio-preserving so stored p's remain valid; the
    # rankx delta + intrinsic are injected (pse2 is not p-gated)
    pj["pse2"] = dict(top_share=0.4 + rankx, base_share=0.4,
                      p_rank=0.5, intrinsic_mean=intrinsic,
                      families={})
    with open(os.path.join(pdir, "se_probe.json"), "w") as f:
        json.dump(pj, f)
    # replay realizing occ_frac
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
    scale = FLAT_SCALE if arm == "flat2" else 1.0
    mod = (f"  mod_key: '{MOD_KEY}'\n  mod_index: {MOD_INDEX}\n"
           f"  mod_lo: {MOD_LO}\n  mod_hi: {MOD_HI}\n"
           if arm == "hetero2" else
           "  mod_key: ''\n  mod_index: 0\n  mod_lo: 0.0\n"
           "  mod_hi: 1.0\n")
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(f"seed: {seed}\ntask: dmc_cheetah_run\n"
                f"planted:\n  gate_key: ''\n  source_key: 'position'\n"
                f"  basesd: 0.0976\n"
                f"distractor:\n  gate_key: ''\n  dim: {DIM}\n"
                f"  scale: {scale}\n  basesd: {BASESD_N}\n{mod}"
                f"agent:\n  expl:\n    mode: p2e\n    disag_ens: 8\n"
                f"    disag_scale: 1000.0\n    disag_bootstrap: true\n"
                f"run:\n  steps: 500000.0\n")
    # region probe consistent with the probe npz: per_state mean over
    # states must equal dims_d (the cross-instrument tie)
    dims_d = np.asarray(npz["dims_d"], np.float64)
    dk = np.asarray(npz["dim_key"]).astype(str)
    S = 512
    hi = np.zeros(S, bool)
    hi[:n_hi_states] = True
    D = dims_d.size
    per_state = np.tile(dims_d[None, :], (S, 1)).astype(np.float64)
    # inject the region contrast on distractor + real cols while
    # PRESERVING the per-column mean (tie holds exactly)
    n_hi_f, n_lo_f = hi.sum(), (~hi).sum()
    if n_hi_f and n_lo_f:
        for cols, chi, clo in ((dk == "distractor", dis_hi, dis_lo),
                               (np.isin(dk, ["position", "velocity"]),
                                real_hi, real_lo)):
            base = dims_d[cols][None, :]
            tot = chi * n_hi_f + clo * n_lo_f
            per_state[np.ix_(hi, np.where(cols)[0])] = \
                base * (chi * S / tot)
            per_state[np.ix_(~hi, np.where(cols)[0])] = \
                base * (clo * S / tot)
    rdir = os.path.join(rd, "se_region_probe")
    os.makedirs(rdir, exist_ok=True)
    ck = os.path.basename(os.path.normpath(pj["ckpt"]))
    np.savez(os.path.join(rdir, "se_region_probe.npz"),
             per_state=per_state.astype(np.float32), dim_key=dk,
             coord=np.where(hi, GATE_THRESHOLD + 0.1,
                            GATE_THRESHOLD - 0.1),
             region_high=hi, ckpt_basename=np.str_(ck),
             probe_seed=np.int64(0), n_eval=np.int64(S))
    with open(os.path.join(rdir, "se_region_probe.json"), "w") as f:
        json.dump(dict(run_logdir=rd, ckpt=pj["ckpt"], seed=0,
                       n_eval=S, n_high=int(hi.sum()),
                       n_low=int((~hi).sum()),
                       region_adjudicable=bool(hi.sum() >= 32
                                               and (~hi).sum() >= 32),
                       region=dict(key="position", index=0,
                                   threshold=GATE_THRESHOLD)), f)
    with open(os.path.join(rd, "scores.jsonl"), "w") as f:
        for i in range(150):
            f.write(json.dumps({"step": i, "episode/score":
                                score + 0.01 * (i % 3)}) + "\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 499968}) + "\n")
    return rd


def selfcheck():
    import tempfile
    NP = se_read.FIX_PERM
    with tempfile.TemporaryDirectory() as tmp:
        s1 = [_fixture(tmp, f"s{i}", 44 + i, "hetero2", 0.5)
              for i in range(4)]
        ref = stage1_reference([os.path.join(tmp, f"s{i}")
                                for i in range(4)])

        def batch(sub, h_occ, f_occ, h_rankx=-0.08, f_rankx=-0.01,
                  **kw):
            het = [_fixture(os.path.join(tmp, sub), f"h{i}", 44 + i,
                            "hetero2",
                            h_occ(i) if callable(h_occ) else h_occ,
                            rankx=h_rankx, **kw) for i in range(4)]
            fl = [_fixture(os.path.join(tmp, sub), f"f{i}", 48 + i,
                           "flat2",
                           f_occ(i) if callable(f_occ) else f_occ,
                           rankx=f_rankx) for i in range(4)]
            return het, fl

        def read_all(het, fl):
            return ([read_run(d, "hetero2", "se_region_probe",
                              "se_probe", NP) for d in het]
                    + [read_run(d, "flat2", "se_region_probe",
                                "se_probe", NP) for d in fl])

        # A) confirmed
        het, fl = batch("A", 0.45, 0.55)
        agg = aggregate(read_all(het, fl), ref)
        assert agg["outcome"] == "AVOIDANCE-CONFIRMED"
        assert abs(agg["primary"]["p"] - 1 / 70) < 1e-12
        assert agg["secondary_ranking"]["fires"]
        assert "mechanism_hetero2" in agg and "score_two_sided" in agg

        # B) null + B2) reversed direction must not fire
        het, fl = batch("B", 0.5, 0.5)
        agg = aggregate(read_all(het, fl), ref)
        assert agg["outcome"] == "AVOIDANCE-NOT-CONFIRMED"
        het, fl = batch("B2", 0.65, 0.45)
        agg = aggregate(read_all(het, fl), ref)
        assert not agg["primary"]["fires"] and \
            agg["primary"]["delta"] > 0
        # secondary cannot fire alone (gated)
        assert not agg["secondary_ranking"]["fires"]

        # C) delivery gate fail (flat gradient in hetero2)
        het, fl = batch("C", 0.45, 0.55, dis_hi=1.0, dis_lo=1.0)
        agg = aggregate(read_all(het, fl), ref)
        assert agg["outcome"] == "DELIVERY-FAILED"
        assert "occupancy_hetero2" in agg

        # C2) region-underpowered: <32 high anchors in 2 hetero2 runs
        #     -> REGION-UNDERPOWERED, primary NOT aborted by assert
        het, fl = batch("C2", 0.45, 0.55)
        for i in (0, 1, 2):
            _fixture(os.path.join(tmp, "C2"), f"h{i}", 44 + i,
                     "hetero2", 0.45, n_hi_states=10)
        agg = aggregate(read_all(het, fl), ref)
        assert agg["outcome"] == "REGION-UNDERPOWERED"

        # C3) misprice gate: parity probes -> MISPRICE-ABSENT
        het, fl = batch("C3", 0.45, 0.55, probe_hot={})
        agg = aggregate(read_all(het, fl), ref)
        assert agg["outcome"] == "MISPRICE-ABSENT"

        # C4) degeneracy gate
        het, fl = batch("C4", 0.45, 0.55, level_scale=1e-6,
                        intrinsic=1e-9)
        agg = aggregate(read_all(het, fl), ref)
        assert agg["outcome"] == "GLOBAL-COLLAPSE"

        # D) gates: fabricated probe json must now FAIL (provenance)
        rd = _fixture(os.path.join(tmp, "D"), "x", 44, "hetero2", 0.45)
        with open(os.path.join(rd, "se_probe", "se_probe.json"),
                  "w") as f:
            json.dump({"pse2": {"top_share": 0.1,
                                "base_share": 0.9}}, f)
        try:
            read_run(rd, "hetero2", "se_region_probe", "se_probe", NP)
            raise SystemExit("probe provenance gate FAILED to fire")
        except (AssertionError, KeyError):
            pass
        # stale region npz (labels != coord)
        rd = _fixture(os.path.join(tmp, "D2"), "y", 44, "hetero2",
                      0.45)
        rp = os.path.join(rd, "se_region_probe", "se_region_probe.npz")
        z = dict(np.load(rp))
        z["region_high"] = ~np.asarray(z["region_high"])
        np.savez(rp, **z)
        try:
            read_run(rd, "hetero2", "se_region_probe", "se_probe", NP)
            raise SystemExit("region label gate FAILED to fire")
        except AssertionError as e:
            assert "recomputed" in str(e)
        # cross-instrument tie: doctored per_state
        rd = _fixture(os.path.join(tmp, "D3"), "z", 44, "hetero2",
                      0.45)
        rp = os.path.join(rd, "se_region_probe", "se_region_probe.npz")
        z = dict(np.load(rp))
        z["per_state"] = np.asarray(z["per_state"]) * 3.0
        np.savez(rp, **z)
        try:
            read_run(rd, "hetero2", "se_region_probe", "se_probe", NP)
            raise SystemExit("cross-instrument tie FAILED to fire")
        except AssertionError as e:
            assert "does not match se_probe" in str(e)
        # config pin: bootstrap off
        rd = _fixture(os.path.join(tmp, "D4"), "w", 44, "hetero2",
                      0.45)
        cfg = open(os.path.join(rd, "config.yaml")).read()
        with open(os.path.join(rd, "config.yaml"), "w") as f:
            f.write(cfg.replace("disag_bootstrap: true",
                                "disag_bootstrap: false"))
        try:
            read_run(rd, "hetero2", "se_region_probe", "se_probe", NP)
            raise SystemExit("expl pin gate FAILED to fire")
        except AssertionError as e:
            assert "expl pins" in str(e)

        # E) end-to-end
        outd = os.path.join(tmp, "out")
        out = run(parse_args([
            "--hetero", os.path.join(tmp, "A", "h*"),
            "--flat", os.path.join(tmp, "A", "f*"),
            "--stage1_runs", ",".join(os.path.join(tmp, f"s{i}")
                                      for i in range(4)),
            "--n_perm", str(NP), "--output", outd]))
        assert out["outcome"] == "AVOIDANCE-CONFIRMED"
        assert os.path.exists(os.path.join(outd, "se_avoid_read.json"))
    print("se_avoid_read selfcheck PASS (A confirmed + gated "
          "secondary, B null, B2 reversed no-fire + secondary-gating, "
          "C delivery-fail, C2 region-underpowered NO-ABORT, C3 "
          "misprice-absent, C4 collapse, D fabricated-probe/stale-"
          "region/tie/pin gates, E end-to-end)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
