"""Track-A2 SCARECROW FROZEN reader
(PREREG_trackA_scarecrow_20260822.md §2; review #34 adjudicated
22 Aug: 3B/12M/15m ALL applied; built + selfchecked BEFORE the wave's
compute; ONE execution).

Arms (seeds pinned): sc_ctrl 76-79 (flat 0.35891393), sc_scare 80-83
(A1-hetero mod quadruple), sc_scare2 84-87 (quadruple at scale 2.0),
sc_pen1 88-91 (penalty_mix lambda 0.5), sc_pen2 92-95 (lambda 2.0).
Penalty arms carry the ctrl observation CONFIGURATION at matched
observation statistics — independent streams, different seeds
(#34 M9: no common random numbers anywhere in this design).

Read order:
  0. Per-run validity: Stage-1 pin block + PER-ARM pins with STRICT
     key presence (#34 M7 — a pre-freeze checkout cannot pass by
     omission); se_probe validity + final-ckpt gate PLUS the
     newest-ckpt-dir cross-check (#34 M15 — ckpt/latest itself can be
     a frozen stale pointer, the A1 append-verify lesson); ONE
     replay scan per run (#34 m9) computing occupancy (MIN_STEPS),
     coverage proxies, and the penalty-delivery gate (float32-exact
     -lambda in-region / 0 out; lambda grid {0.5, 2.0} is
     float32-exact by construction); penalty arms must show the
     reward head actually TRAINED (#34 M6/B1: last train/loss/rew <
     0.9*ln(255) AND below the first — presence alone is the
     signature of the frozen-head bug) + the realized
     penalty/intrinsic trajectory rows (#34 M5). A DEFECTIVE run is
     caught, recorded in excluded_runs, and the read persists.
  1. DEGENERACY, PER CONTRAST (#34 M4): per-run collapse flags vs
     the PINNED Stage-1 reference (#34 M14: the 8-run
     uncfield_se_stage1_20260813_193324 bundle; ref n == 8 hard-
     asserted); a collapsed run inside a contrast voids THAT
     contrast, never the whole read; per-arm collapse rows are
     always emitted.
  2. PRIMARY: occupancy(sc_scare) < occupancy(sc_ctrl), exact
     C(8,4)=70 one-sided permutation, FULL 4+4 ONLY (#34 B3: any
     missing/invalid/collapsed run in either primary arm =>
     NOT-ADJUDICABLE — exclusion must never be able to manufacture
     a fire; the partial contrast is reported descriptively).
  3. REGISTERED SECONDARIES (computed on adjudicable paths, raw
     per-arm rows always emitted): S1 scare2-vs-ctrl + delta to
     scare; S2 pen1/pen2-vs-ctrl; S3 scare-vs-pen2 DESCRIPTIVE (no
     fire); S4 off-region coverage EXCLUDING the gated position dim
     (#34 M10 — conditioning on a coordinate and measuring its own
     truncated spread is mechanically coupled to the fence; the
     gated dim gets its own row); S5 theta_1 rows.

Run:  python -m uncfield.se_scarecrow_read --runs "<glob>"
          --stage1_runs "<glob>" --output <dir>
      python -m uncfield.se_scarecrow_read --selfcheck
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
from uncfield.se_m3_read import (GATE_INDEX, GATE_KEY, GATE_THRESHOLD,
                                 MIN_STEPS, exact_perm_p)
from uncfield.se_noboot_read import (COLLAPSE_FACTOR, _levels,
                                     stage1_reference)
from uncfield.se_read import fit_counters
from uncfield import se_read

TASK_PIN = "dmc_cheetah_run"
ARM_SEEDS = {
    "sc_ctrl": (76, 77, 78, 79),
    "sc_scare": (80, 81, 82, 83),
    "sc_scare2": (84, 85, 86, 87),
    "sc_pen1": (88, 89, 90, 91),
    "sc_pen2": (92, 93, 94, 95),
}
SEED_ARM = {s: a for a, ss in ARM_SEEDS.items() for s in ss}
PEN_LAMBDA = {"sc_pen1": 0.5, "sc_pen2": 2.0}
SCARE_SCALE = {"sc_scare": 1.0, "sc_scare2": 2.0}
REW_LOSS_BAR = 0.9 * math.log(255)     # #34 M6: below = actually trained
STD_FLOOR = 1e-6                       # #34 m10
STAGE1_N = 8                           # #34 M14 pinned reference size


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


def perm_less(a_vals, b_vals):
    """One-sided exact test of mean(a) < mean(b) via the house
    C(8,4) machinery (negated values put the mass in the counted
    tail). Returns (delta = mean(a)-mean(b), p)."""
    vals = np.concatenate([np.asarray(a_vals, np.float64),
                           np.asarray(b_vals, np.float64)])
    labels = np.array([True] * len(a_vals) + [False] * len(b_vals))
    obs_neg, p, total = exact_perm_p(-vals, labels, one_sided=True)
    assert total == math.comb(len(vals), int(labels.sum()))
    return float(-obs_neg), float(p)


def _replay_scan(run_dir, lam):
    """ONE pass over the replay (#34 m9) computing:
    occupancy (pinned gate functional), coverage proxies (all /
    off-region, position dims EXCLUDING the gated dim #34 M10 +
    velocity; the gated dim's off-region std as its own row), and —
    when lam is not None — the penalty-delivery counters."""
    rd = os.path.join(run_dir, "replay")
    files = sorted(f for f in os.listdir(rd) if f.endswith(".npz"))
    assert files, f"{run_dir}: empty replay dir"
    pos_all, vel_all = [], []
    hits = total = 0
    bad_in = bad_out = n_in = 0
    want = None if lam is None else np.float32(-lam)
    for f in files:
        with np.load(os.path.join(rd, f)) as z:
            p = np.asarray(z["position"], np.float64)
            v = np.asarray(z["velocity"], np.float64)
            p = p.reshape(-1, p.shape[-1])
            v = v.reshape(-1, v.shape[-1])
            inr = p[:, GATE_INDEX] > GATE_THRESHOLD
            hits += int(inr.sum())
            total += p.shape[0]
            pos_all.append(p)
            vel_all.append(v)
            if want is not None:
                r = np.asarray(z["reward"], np.float32).reshape(-1)
                n_in += int(inr.sum())
                bad_in += int((r[inr] != want).sum())
                bad_out += int((r[~inr] != np.float32(0.0)).sum())
    pos = np.concatenate(pos_all, 0)
    vel = np.concatenate(vel_all, 0)
    assert total >= MIN_STEPS, (
        f"{run_dir}: {total} replay steps < {MIN_STEPS}")
    occ = hits / total

    def _proxy(pm, vm):
        # position dims EXCLUDING the gated dim (#34 M10) + velocity
        pdims = [i for i in range(pm.shape[1]) if i != GATE_INDEX]
        stds = np.concatenate([pm[:, pdims].std(0), vm.std(0)])
        return (float(np.sum(np.log(np.maximum(stds, STD_FLOOR)))),
                float(stds.min()), pm.shape[0])

    cov_all = _proxy(pos, vel)
    off = pos[:, GATE_INDEX] <= GATE_THRESHOLD
    cov_off = (_proxy(pos[off], vel[off]) if off.sum() >= 2
               else (None, None, int(off.sum())))
    gated_off_std = (float(pos[off, GATE_INDEX].std())
                     if off.sum() >= 2 else None)
    out = dict(occupancy=occ, occ_steps=total,
               coverage_all=dict(proxy=cov_all[0],
                                 min_dim_std=cov_all[1], n=cov_all[2]),
               coverage_off=dict(proxy=cov_off[0],
                                 min_dim_std=cov_off[1], n=cov_off[2]),
               gated_dim_off_std=gated_off_std)
    if want is not None:
        assert bad_in == 0 and bad_out == 0, (
            f"{run_dir}: penalty NOT delivered as registered "
            f"(bad_in {bad_in}, bad_out {bad_out} of {total}; "
            f"lambda {lam})")
        out["penalty_delivery"] = dict(n_steps=total,
                                       frac_in_region=n_in / total)
    return out


def _penalty_training(run_dir):
    """#34 M5/M6: the realized penalty-vs-intrinsic trajectory and
    the trained-head gate inputs from metrics.jsonl."""
    first = last = None
    pen_first = pen_last = intr_last = None
    with open(os.path.join(run_dir, "metrics.jsonl"), "rb") as f:
        for line in f.read().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "train/loss/rew" in d:
                v = float(d["train/loss/rew"])
                if first is None:
                    first = v
                    pen_first = d.get("train/expl/penalty_rew")
                last = v
                pen_last = d.get("train/expl/penalty_rew",
                                 pen_last)
                intr_last = d.get("train/expl/intr_rew_raw",
                                  intr_last)
    assert first is not None, (
        f"{run_dir}: train/loss/rew absent — the reward head never "
        f"trained; penalty_mix did not take effect")
    # #34 M6/B1: PRESENCE is the frozen-head bug's own signature
    # (loss pinned at ln(255)=5.5413); the gate is on the VALUE
    assert last < REW_LOSS_BAR and last < first, (
        f"{run_dir}: train/loss/rew {first:.4f} -> {last:.4f} never "
        f"left the untrained plateau (bar {REW_LOSS_BAR:.3f}) — the "
        f"reward head is frozen (#34 B1 class)")
    return dict(rew_loss_first=first, rew_loss_last=last,
                penalty_rew_first=pen_first, penalty_rew_last=pen_last,
                intr_rew_raw_last=intr_last)


def read_run(run_dir, n_perm):
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = cfg["distractor"]
    pl = cfg["planted"]
    ex = cfg["agent"]["expl"]
    # #34 M7: STRICT presence — a pre-penalty-freeze checkout has
    # neither key and must not pass by defaulting
    assert "penalty" in cfg, (
        f"{run_dir}: no penalty config block — pre-freeze checkout?")
    assert "penalty_mix" in ex, (
        f"{run_dir}: no expl.penalty_mix key — pre-freeze checkout?")
    pen = cfg["penalty"]
    train_seed = int(cfg["seed"])
    assert train_seed in SEED_ARM, (
        f"{run_dir}: seed {train_seed} outside the registered 76-95")
    arm = SEED_ARM[train_seed]
    assert str(cfg["task"]) == TASK_PIN, run_dir
    assert str(ex["mode"]) == "p2e" and int(ex["disag_ens"]) == 8 \
        and abs(float(ex["disag_scale"]) - 1000.0) < 1e-9 \
        and ex["disag_bootstrap"] is True, (
        f"{run_dir}: expl pins differ from Stage-1")
    assert int(d["dim"]) == DIM and \
        abs(float(d["basesd"]) - BASESD_N) < 1e-9
    assert str(d.get("gate_key", "")) == "" and \
        str(pl.get("gate_key", "")) == "", (
        f"{run_dir}: distractor/planted gates must be off")
    assert str(pl["source_key"]) == "position" and \
        abs(float(pl["basesd"]) - 0.0976) < 1e-9, run_dir
    # per-arm pins (strict indexing throughout, #34 M7)
    if arm in SCARE_SCALE:
        assert abs(float(d["scale"]) - SCARE_SCALE[arm]) < 1e-9, (
            f"{run_dir}: {arm} scale {d['scale']}")
        assert str(d["mod_key"]) == MOD_KEY and \
            int(d["mod_index"]) == MOD_INDEX and \
            abs(float(d["mod_lo"]) - MOD_LO) < 1e-9 and \
            abs(float(d["mod_hi"]) - MOD_HI) < 1e-9, (
            f"{run_dir}: {arm} mod quadruple differs from A1-hetero")
        assert float(pen["scale"]) == 0.0 and \
            ex["penalty_mix"] is False, (
            f"{run_dir}: {arm} must not carry a penalty")
    elif arm == "sc_ctrl":
        assert abs(float(d["scale"]) - FLAT_SCALE) < 1e-9, (
            f"{run_dir}: ctrl scale {d['scale']} != flat")
        assert str(d["mod_key"]) == ""
        assert float(pen["scale"]) == 0.0 and \
            ex["penalty_mix"] is False, (
            f"{run_dir}: ctrl must not carry a penalty")
    else:
        lam = PEN_LAMBDA[arm]
        assert abs(float(d["scale"]) - FLAT_SCALE) < 1e-9 and \
            str(d["mod_key"]) == "", (
            f"{run_dir}: penalty arms carry the ctrl obs "
            f"configuration")
        assert ex["penalty_mix"] is True, (
            f"{run_dir}: {arm} penalty_mix did not bind")
        assert abs(float(pen["scale"]) - lam) < 1e-9, (
            f"{run_dir}: {arm} lambda {pen['scale']} != {lam}")
        assert str(pen["gate_key"]) == GATE_KEY and \
            int(pen["gate_index"]) == GATE_INDEX and \
            abs(float(pen["gate_threshold"]) - GATE_THRESHOLD) < 1e-9, (
            f"{run_dir}: penalty gate triple != the pinned region")

    rec = se_read.read_run(run_dir, "se_probe", n_perm)
    # #34 M15: ckpt/latest can itself be a frozen stale pointer (the
    # A1 append-verify incident) — cross-check the probed checkpoint
    # against the NEWEST ckpt subdir when the dir is listable
    ckroot = os.path.join(run_dir, "ckpt")
    try:
        subs = sorted(x for x in os.listdir(ckroot)
                      if os.path.isdir(os.path.join(ckroot, x)))
    except OSError:
        subs = []
    if subs:
        with open(os.path.join(run_dir, "se_probe",
                               "se_probe.json")) as f:
            probed = os.path.basename(
                os.path.normpath(json.load(f)["ckpt"]))
        assert probed == subs[-1], (
            f"{run_dir}: probed ckpt {probed} != newest ckpt dir "
            f"{subs[-1]} — stale ckpt/latest pointer (#34 M15)")
        rec["newest_ckpt"] = subs[-1]
    rec["train_seed_cfg"] = train_seed
    rec["arm"] = arm
    rec["levels"] = _levels(run_dir)
    rec.update(_replay_scan(run_dir, PEN_LAMBDA.get(arm)))
    if arm in PEN_LAMBDA:
        rec["penalty_training"] = _penalty_training(run_dir)
    return rec


def aggregate(recs, excluded, ref):
    assert int(ref["n"]) == STAGE1_N, (
        f"stage-1 reference has {ref['n']} runs != pinned "
        f"{STAGE1_N} (#34 M14)")
    by_arm = {a: [] for a in ARM_SEEDS}
    seeds = sorted(r["train_seed_cfg"] for r in recs)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds {seeds}"
    for r in recs:
        by_arm[r["arm"]].append(r)
    for a, rs in by_arm.items():
        assert len(rs) <= 4, (a, len(rs))
    out = dict(stamp=_stamp(), n_loaded=len(recs),
               excluded_runs=excluded,
               missing_seeds=sorted(set(SEED_ARM) - set(seeds)),
               n_per_arm={a: len(v) for a, v in by_arm.items()},
               stage1_reference=ref)

    # per-run collapse flags (#34 M4: per-contrast, never global)
    def _collapsed(r):
        return (r["levels"]["real_mean"]
                < ref["real_mean_min"] / COLLAPSE_FACTOR
                or r["levels"]["intrinsic_mean"]
                < ref["intrinsic_min"] / COLLAPSE_FACTOR)
    for r in recs:
        r["collapsed"] = _collapsed(r)
    out["collapse_rows"] = {a: [r["collapsed"] for r in v]
                            for a, v in by_arm.items()}

    # unconditional per-run reporting
    out["per_run"] = [
        {"run_dir": r["run_dir"], "arm": r["arm"],
         "train_seed": r["train_seed_cfg"], "valid": r["valid"],
         "collapsed": r["collapsed"],
         "occupancy": r["occupancy"], "occ_steps": r["occ_steps"],
         "coverage_all": r["coverage_all"],
         "coverage_off": r["coverage_off"],
         "gated_dim_off_std": r["gated_dim_off_std"],
         "theta1_distractor": r["channels"]["distractor"]["vs_source"],
         "real_mean_level": r["levels"]["real_mean"],
         "intrinsic_mean": r["levels"]["intrinsic_mean"],
         "penalty_delivery": r.get("penalty_delivery"),
         "penalty_training": r.get("penalty_training")}
        for r in recs]
    out["occupancy_rows"] = {a: [r["occupancy"] for r in v]
                             for a, v in by_arm.items()}
    out["coverage_rows"] = {
        a: dict(off=[r["coverage_off"]["proxy"] for r in v],
                all=[r["coverage_all"]["proxy"] for r in v],
                gated_dim_off_std=[r["gated_dim_off_std"]
                                   for r in v])
        for a, v in by_arm.items()}
    out["theta1_rows"] = {
        a: [r["channels"]["distractor"]["vs_source"] for r in v]
        for a, v in by_arm.items()}

    def _usable(arm):
        return [r for r in by_arm[arm]
                if r["valid"] and not r["collapsed"]]

    ctrl = _usable("sc_ctrl")
    scare = _usable("sc_scare")
    o_c = [r["occupancy"] for r in ctrl]
    o_s = [r["occupancy"] for r in scare]

    # 2) PRIMARY — FULL 4+4 ONLY (#34 B3: exclusion can never
    # manufacture a fire; anything less is NOT-ADJUDICABLE with the
    # partial contrast reported descriptively)
    if len(o_c) == 4 and len(o_s) == 4:
        delta, p = perm_less(o_s, o_c)
        fires = bool(p <= 0.05)
        out["primary"] = dict(
            statement="SCARECROW fences: occupancy(sc_scare) < "
                      "occupancy(sc_ctrl), one-sided exact C(8,4) "
                      "permutation, full 4+4 only",
            occ_scare=o_s, occ_ctrl=o_c, delta=delta, p=p,
            min_attainable_p=1 / 70, fires=fires)
        out["outcome_cell"] = ("SCARECROW-FENCES" if fires
                               else "TOOL-NOT-CONFIRMED")
    else:
        out["primary"] = (
            f"NOT-ADJUDICABLE (usable ctrl {len(o_c)}, scare "
            f"{len(o_s)}; the primary requires the full 4+4 — "
            f"#34 B3). Partial contrast, DESCRIPTIVE only: "
            f"mean(scare)-mean(ctrl) = "
            f"{(float(np.mean(o_s) - np.mean(o_c)) if o_s and o_c else None)}")
        out["outcome_cell"] = "NOT-ADJUDICABLE"

    # 3) registered secondaries — each contrast voided ONLY by its
    # own arms' problems (#34 M4); raw rows above are always present
    def _sec(a_name):
        arm_u = _usable(a_name)
        occ = [r["occupancy"] for r in arm_u]
        if len(occ) == 4 and len(o_c) == 4:
            dlt, pp = perm_less(occ, o_c)
            return dict(occ=occ, delta=dlt, p=pp)
        return dict(occ=occ, delta=None, p=None,
                    note="contrast voided (partial/collapsed arm) — "
                         "rows reported")
    out["s1_scare2_vs_ctrl"] = _sec("sc_scare2")
    out["s1_scare2_minus_scare"] = (
        float(np.mean(out["s1_scare2_vs_ctrl"]["occ"]) - np.mean(o_s))
        if out["s1_scare2_vs_ctrl"]["occ"] and o_s else None)
    out["s2_pen1_vs_ctrl"] = _sec("sc_pen1")
    out["s2_pen2_vs_ctrl"] = _sec("sc_pen2")
    pen2 = [r["occupancy"] for r in _usable("sc_pen2")]
    out["s3_scare_vs_pen2"] = dict(
        occ_scare=o_s, occ_pen2=pen2,
        delta=(float(np.mean(o_s) - np.mean(pen2))
               if pen2 and o_s else None),
        note="REGISTERED DESCRIPTIVE (pre-declared P(scarecrow "
             "wins) ~ 0.10); no fire either way; scores are NOT "
             "cross-arm comparable (penalty arms' reward is the "
             "penalty, #34 m3); deployability asymmetry is prose")
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
    out_json = os.path.join(args.output, "se_scarecrow_read.json")
    assert not os.path.exists(out_json), (
        f"{out_json} exists — the read is ONE execution")
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
    os.makedirs(args.output, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    if isinstance(out["primary"], dict):
        pr = out["primary"]
        print(f"SCARECROW PRIMARY: delta={pr['delta']:+.4f} "
              f"p={pr['p']:.4f} fires={pr['fires']} -> "
              f"{out['outcome_cell']}; fit {out['fit_flag']}")
    else:
        print(f"SCARECROW PRIMARY: {out['outcome_cell']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, occ=0.5, hot=None, level_scale=1.0,
             intrinsic=3.5e-4, rew_first=5.5413, rew_last=0.4,
             wrong_lambda=None, n_replay=int(1.2e5),
             drop_penalty_block=False, extra_ckpt=None):
    """Fake run: config.yaml per the seed's arm, se_probe via the
    reviewed se_read fixture generator, replay chunks with the target
    occupancy and (penalty arms) exact rewards, metrics with a
    trained-head trajectory."""
    arm = SEED_ARM[seed]
    rd = os.path.join(tmp, name)
    se_read._fixture_run(tmp, name, seed,
                         hot=(hot if hot is not None
                              else {"distractor": 5.0}))
    if extra_ckpt:
        os.makedirs(os.path.join(rd, "ckpt", extra_ckpt),
                    exist_ok=True)
    pdir = os.path.join(rd, "se_probe")
    npz = dict(np.load(os.path.join(pdir, "se_probe_dims.npz")))
    npz["dims_d"] = npz["dims_d"] * level_scale
    np.savez(os.path.join(pdir, "se_probe_dims.npz"), **npz)
    with open(os.path.join(pdir, "se_probe.json")) as f:
        pj = json.load(f)
    pj["pse2"]["intrinsic_mean"] = intrinsic
    dims_d = np.asarray(npz["dims_d"], np.float64)
    dim_key = np.asarray(npz["dim_key"]).astype(str)
    per_key = {k: float(dims_d[dim_key == k].mean())
               for k in set(dim_key.tolist())}
    for ch in pj["channels"]:
        pj["channels"][ch]["vs_source"] = \
            per_key[ch] / per_key["position"]
    with open(os.path.join(pdir, "se_probe.json"), "w") as f:
        json.dump(pj, f)

    if arm in SCARE_SCALE:
        scale, mod = SCARE_SCALE[arm], MOD_KEY
    else:
        scale, mod = FLAT_SCALE, ""
    pmix = arm in PEN_LAMBDA
    cfg_lam = (wrong_lambda if wrong_lambda is not None
               else PEN_LAMBDA.get(arm, 0.0))
    pen_block = (
        "" if drop_penalty_block else
        f"penalty:\n  scale: {cfg_lam}\n"
        f"  gate_key: '{GATE_KEY if pmix else ''}'\n"
        f"  gate_index: {GATE_INDEX}\n"
        f"  gate_threshold: {GATE_THRESHOLD}\n")
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(
            f"seed: {seed}\ntask: {TASK_PIN}\n"
            f"planted:\n  gate_key: ''\n  source_key: 'position'\n"
            f"  basesd: 0.0976\n"
            f"distractor:\n  gate_key: ''\n  mod_key: '{mod}'\n"
            f"  mod_index: {MOD_INDEX}\n  mod_lo: {MOD_LO}\n"
            f"  mod_hi: {MOD_HI}\n"
            f"  dim: {DIM}\n  scale: {scale}\n  basesd: {BASESD_N}\n"
            f"{pen_block}"
            f"agent:\n  expl:\n    mode: p2e\n    disag_ens: 8\n"
            f"    disag_scale: 1000.0\n    disag_bootstrap: true\n"
            f"    penalty_mix: {'true' if pmix else 'false'}\n"
            f"run:\n  steps: 500000.0\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        if pmix:
            f.write(json.dumps({"step": 1024,
                                "train/loss/rew": rew_first,
                                "train/expl/penalty_rew": -0.01,
                                "train/expl/intr_rew_raw": 0.28})
                    + "\n")
            f.write(json.dumps({"step": 499968,
                                "train/loss/rew": rew_last,
                                "train/expl/penalty_rew": -0.31,
                                "train/expl/intr_rew_raw": 3.5e-4})
                    + "\n")
        else:
            f.write(json.dumps({"step": 499968}) + "\n")
    rng = np.random.default_rng(seed * 13 + 5)
    os.makedirs(os.path.join(rd, "replay"), exist_ok=True)
    n = n_replay
    inr = rng.random(n) < occ
    pos = np.where(inr, GATE_THRESHOLD + 0.5,
                   GATE_THRESHOLD - 0.5).astype(np.float32)
    posmat = np.stack([pos, rng.standard_normal(n).astype(np.float32)],
                      1)
    vel = rng.standard_normal((n, 3)).astype(np.float32)
    replay_lam = (wrong_lambda if wrong_lambda is not None
                  else PEN_LAMBDA.get(arm, 0.0))
    rew = np.where(inr,
                   np.float32(-PEN_LAMBDA.get(arm, -0.1)
                              if pmix else 0.1),
                   np.float32(0.0)).astype(np.float32)
    if pmix:
        rew = np.where(inr, np.float32(-replay_lam),
                       np.float32(0.0)).astype(np.float32)
    half = n // 2
    for i, sl in enumerate((slice(0, half), slice(half, n))):
        np.savez(os.path.join(rd, "replay", f"c{i}.npz"),
                 position=posmat[sl], velocity=vel[sl],
                 reward=rew[sl])
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
    NP = se_read.FIX_PERM
    OCC = {"sc_ctrl": 0.52, "sc_scare": 0.40, "sc_scare2": 0.34,
           "sc_pen1": 0.30, "sc_pen2": 0.12}
    with tempfile.TemporaryDirectory() as tmp:
        s1 = [se_read._fixture_run(tmp, f"st{i}", 10 + i,
                                   hot={"distractor": 5.0})
              for i in range(STAGE1_N)]
        for d in s1:
            with open(os.path.join(d, "config.yaml"), "w") as f:
                f.write("seed: 10\n")
            pth = os.path.join(d, "se_probe", "se_probe.json")
            with open(pth) as f:
                pj1 = json.load(f)
            pj1["pse2"]["intrinsic_mean"] = 3.5e-4
            with open(pth, "w") as f:
                json.dump(pj1, f)
        ref = stage1_reference(s1)
        assert ref["n"] == STAGE1_N
        _expect_fail(lambda: aggregate([], [],
                                       dict(ref, n=4)), "pinned")

        # FENCES: all 20 arms at their designed occupancies
        runs = []
        for seed in range(76, 96):
            occ = OCC[SEED_ARM[seed]] + 0.01 * (seed % 4)
            runs.append(_fixture(tmp, f"a{seed}", seed, occ=occ))
        recs = [read_run(d, NP) for d in runs]
        agg = aggregate(recs, [], ref)
        assert agg["outcome_cell"] == "SCARECROW-FENCES", agg["primary"]
        assert agg["primary"]["p"] == 1 / 70
        assert agg["s2_pen2_vs_ctrl"]["p"] == 1 / 70
        assert agg["s3_scare_vs_pen2"]["delta"] > 0
        assert agg["s1_scare2_minus_scare"] < 0
        assert agg["missing_seeds"] == []
        for r in agg["per_run"]:
            if r["arm"] in PEN_LAMBDA:
                assert r["penalty_delivery"] and r["penalty_training"]
                assert r["penalty_training"]["rew_loss_last"] < \
                    REW_LOSS_BAR

        # TOOL-NOT-CONFIRMED
        runs2 = []
        for seed in range(76, 96):
            occ = (0.52 + 0.01 * (seed % 4)
                   if SEED_ARM[seed] in ("sc_ctrl", "sc_scare",
                                         "sc_scare2")
                   else OCC[SEED_ARM[seed]])
            runs2.append(_fixture(tmp, f"b{seed}", seed, occ=occ))
        agg = aggregate([read_run(d, NP) for d in runs2], [], ref)
        assert agg["outcome_cell"] == "TOOL-NOT-CONFIRMED"

        # #34 B3: ANY missing primary-arm run -> NOT-ADJUDICABLE
        # (specifically the anti-conservative case the review built:
        # dropping the LOWEST ctrl must not manufacture a fire)
        sub = [r for r in recs
               if not (r["arm"] == "sc_ctrl"
                       and r["train_seed_cfg"] == 76)]
        agg = aggregate(sub, [dict(run_dir="x", error="e")], ref)
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE"
        assert agg["missing_seeds"] == [76]
        assert "DESCRIPTIVE" in agg["primary"]
        # secondaries survive a primary-arm exclusion? ctrl is in
        # every secondary contrast -> voided notes, rows present
        assert agg["s2_pen2_vs_ctrl"]["p"] is None
        assert agg["s2_pen2_vs_ctrl"]["occ"]

        # #34 M4: a collapsed PENALTY run voids only its contrast
        runs3 = list(runs)
        runs3[16] = _fixture(tmp, "c92", 92, occ=0.12,
                             level_scale=1e-6, intrinsic=1e-9)
        recs3 = [read_run(d, NP) for d in runs3]
        agg = aggregate(recs3, [], ref)
        assert agg["outcome_cell"] == "SCARECROW-FENCES"   # primary OK
        assert agg["s2_pen2_vs_ctrl"]["p"] is None          # voided
        assert agg["collapse_rows"]["sc_pen2"].count(True) == 1
        # ... and a collapsed SCARE run voids the primary
        runs4 = list(runs)
        runs4[4] = _fixture(tmp, "c80", 80, occ=0.40,
                            level_scale=1e-6, intrinsic=1e-9)
        agg = aggregate([read_run(d, NP) for d in runs4], [], ref)
        assert agg["outcome_cell"] == "NOT-ADJUDICABLE"

        # gates
        rd = _fixture(tmp, "p_bad", 88, occ=0.3, wrong_lambda=1.7)
        _expect_fail(lambda: read_run(rd, NP), "lambda")
        rd = _fixture(tmp, "p_frozen", 92, occ=0.2, rew_last=5.5413)
        _expect_fail(lambda: read_run(rd, NP), "frozen")
        rd = _fixture(tmp, "p_noblock", 89, occ=0.3,
                      drop_penalty_block=True)
        _expect_fail(lambda: read_run(rd, NP), "pre-freeze")
        rd = _fixture(tmp, "p_short", 90, occ=0.3,
                      n_replay=50_000)
        _expect_fail(lambda: read_run(rd, NP), "replay steps")
        rd = _fixture(tmp, "s_stale", 82, occ=0.4,
                      extra_ckpt="zzz_newer_ckpt")
        _expect_fail(lambda: read_run(rd, NP), "stale ckpt/latest")
        rd = _fixture(tmp, "s_bad", 80, occ=0.4)
        cfgp = os.path.join(rd, "config.yaml")
        cfg = open(cfgp).read()
        open(cfgp, "w").write(cfg.replace(f"mod_key: '{MOD_KEY}'",
                                          "mod_key: ''"))
        _expect_fail(lambda: read_run(rd, NP), "mod quadruple")
        rd = _fixture(tmp, "s_pen", 81, occ=0.4)
        cfg = open(os.path.join(rd, "config.yaml")).read()
        open(os.path.join(rd, "config.yaml"), "w").write(
            cfg.replace("penalty:\n  scale: 0.0",
                        "penalty:\n  scale: 0.5"))
        _expect_fail(lambda: read_run(rd, NP), "must not carry")
        rd = _fixture(tmp, "s_seed", 80, occ=0.4)
        cfg = open(os.path.join(rd, "config.yaml")).read()
        open(os.path.join(rd, "config.yaml"), "w").write(
            cfg.replace("seed: 80", "seed: 42"))
        _expect_fail(lambda: read_run(rd, NP), "registered 76-95")

        # excluded-run persistence + fit SUSPECT via run()
        outd = os.path.join(tmp, "out")
        with open(os.path.join(runs[0], "config.yaml"), "w") as f:
            f.write("seed: 76\n")            # corrupt one ctrl run
        with open(os.path.join(runs[5], "metrics.jsonl"), "w") as f:
            f.write(json.dumps({"step": 200000}) + "\n")  # truncated
        out = run(parse_args([
            "--runs", os.path.join(tmp, "a*"),
            "--stage1_runs", os.path.join(tmp, "st*"),
            "--n_perm", str(NP), "--output", outd]))
        assert len(out["excluded_runs"]) == 1
        assert out["outcome_cell"] == "NOT-ADJUDICABLE"    # B3
        assert out["fit_flag"].startswith("SUSPECT")
        assert os.path.exists(os.path.join(outd,
                                           "se_scarecrow_read.json"))
        _expect_fail(lambda: run(parse_args([
            "--runs", os.path.join(tmp, "a*"),
            "--stage1_runs", os.path.join(tmp, "st*"),
            "--n_perm", str(NP), "--output", outd])), "ONE execution")
    print("se_scarecrow_read selfcheck PASS (FENCES at the exact "
          "floor + TOOL-NOT-CONFIRMED; #34 B3 full-4+4-only primary "
          "incl. the anti-conservative-exclusion fixture; #34 M4 "
          "per-contrast collapse (penalty collapse keeps the "
          "primary, scare collapse voids it); pinned stage-1 n==8; "
          "frozen-head/lambda/penalty-block/MIN_STEPS/stale-newest-"
          "ckpt/mod-quadruple/penalty-leak/seed gates; excluded-run "
          "persistence + fit-SUSPECT + one-execution guard)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
