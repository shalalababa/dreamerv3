"""SE-wave M4 persistence collator (SECONDARY; prereg v2.2 §5 M4,
operationalized in PREREG_nfi_scale_exhibit_amend2_20260814.md §3).
Built + selfchecked BEFORE the M4 read and executes it ONCE.

Registered content:
  WEAK FORM (v2.2 §5, the only decision content): fire-channel share
  remains above the per-run permutation-null median at 100% — restated
  from these inputs (the 100% files are byte-checked against the
  Stage-1 primary read's inputs when --stage1_runs is given).
  TRAJECTORY (reporting only): per-channel share, null median,
  indicator, and theta_1 at 25% / 50% / 100%; cross-run indicator
  counts per fraction; distractor non-decay descriptive
  (share_100 - share_25, seed-level BCa; labels RISING / FALLING /
  FLAT-OR-MIXED by CI vs 0 — DESCRIPTIVE, never a fire decision).

Gates per pass (same family as the frozen primary reader):
  - provenance: per-channel permutation p recomputed from the npz with
    se_read.channel_null (bit-identical arithmetic + crc32 rng) must
    EQUAL the stored p;
  - snapshot verification: the probed checkpoint must be step-named;
    realized fraction within +-0.05 of target; and when
    ckpt_snapshots/manifest.json is synced, the probed step must be
    the retained snapshot NEAREST the target (the Amendment-2
    selection rule);
  - validity: n_eval >= 256 and calibration < 0.5 on every REAL key,
    per pass; invalid cells are reported and excluded from BCa, never
    silently dropped.

Run:  python -m uncfield.se_m4_read --runs "<glob>" --output <dir>
          [--stage1_runs "<glob>"]
      python -m uncfield.se_m4_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import os
import zlib

import numpy as np

from uncfield.se_mask import bca_interval
from uncfield.se_probe import FIRE_KEYS, PLANTED_KEYS
from uncfield.se_read import CAL_MAX, S_MIN, channel_null

STEPS = 5e5                      # pinned run length (prereg §1)
FRACTIONS = (("se_probe_snap25", 0.25), ("se_probe_snap50", 0.50),
             ("se_probe", 1.00))
FRAC_TOL = 0.05                  # |realized - target| tolerance
N_REGISTERED = 8


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="",
                   help="comma-separated M4 run dirs, or a glob")
    p.add_argument("--stage1_runs", default="",
                   help="optional Stage-1 run dirs (same order semantics: "
                        "matched by basename) for 100%% byte-identity")
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--steps", type=float, default=STEPS)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def verify_ckpt(run_dir, subdir, pj_ckpt, frac, steps):
    """Snapshot-step verification (Amendment-2 selection rule)."""
    name = os.path.basename(os.path.normpath(pj_ckpt))
    if subdir == "se_probe":                       # 100%: timestamp ckpt
        return dict(ckpt=name, realized_step=None, realized_frac=1.0,
                    selection="FINAL (byte-identity vs Stage-1 when given)")
    assert name.startswith("step"), \
        f"{run_dir}/{subdir}: snapshot ckpt {name!r} is not step-named"
    step = int(name[4:])
    realized = step / steps
    assert abs(realized - frac) <= FRAC_TOL, (
        f"{run_dir}/{subdir}: realized fraction {realized:.4f} vs target "
        f"{frac} exceeds tolerance {FRAC_TOL}")
    man_path = os.path.join(run_dir, "ckpt_snapshots", "manifest.json")
    if os.path.exists(man_path):
        with open(man_path) as f:
            avail = [s["step"] for s in json.load(f)["snapshots"]]
        nearest = min(avail, key=lambda x: abs(x - frac * steps))
        assert step == nearest, (
            f"{run_dir}/{subdir}: probed step {step} is not the retained "
            f"snapshot nearest target {int(frac * steps)} (that is "
            f"{nearest}) — selection rule violated")
        sel = "VERIFIED-NEAREST"
    else:
        sel = "UNVERIFIED (no ckpt_snapshots/manifest.json synced)"
    return dict(ckpt=name, realized_step=step,
                realized_frac=float(realized), selection=sel)


def read_pass(run_dir, subdir, frac, n_perm, steps):
    """One probe output (run x fraction): provenance-gated per-channel
    stats. Mirrors the frozen se_read.read_run per-channel block with
    the final-ckpt gate replaced by snapshot verification."""
    pdir = os.path.join(run_dir, subdir)
    with open(os.path.join(pdir, "se_probe.json")) as f:
        pj = json.load(f)
    npz = np.load(os.path.join(pdir, "se_probe_dims.npz"))
    dims_d = np.asarray(npz["dims_d"])         # stored dtype (float32)
    dim_key = np.asarray(npz["dim_key"]).astype(str)
    seed = int(pj["seed"])
    real_keys = [k for k in sorted(set(dim_key.tolist()))
                 if k not in PLANTED_KEYS]
    rec = dict(run_dir=os.path.abspath(run_dir), subdir=subdir,
               n_eval=int(pj["n_eval"]), channels={},
               probe_seed=seed,
               probe_seed_flag=("OK" if seed == 0
                                else "NON-DEFAULT-PROBE-SEED"))
    # same-buffer-same-windows witness (review #23 M4): raw_norms +
    # norm_floor depend only on the replay windows, so bit-equality
    # across a run's three passes proves the registered fixed eval
    # distribution (parent §5 "FINAL replay buffer")
    if "raw_norms" in npz and "norm_floor" in npz:
        rec["norms_sha"] = hashlib.sha256(
            np.asarray(npz["raw_norms"]).tobytes()
            + np.asarray(npz["norm_floor"]).tobytes()).hexdigest()
    else:
        rec["norms_sha"] = None
    rec.update(verify_ckpt(run_dir, subdir, pj["ckpt"], frac, steps))
    for ch in FIRE_KEYS:
        rng = np.random.default_rng(seed + zlib.crc32(ch.encode()) % 2 ** 16)
        obs, p, med = channel_null(dims_d, dim_key, ch, real_keys,
                                   n_perm, rng)
        stored = float(pj["channels"][ch]["p_perm"])
        assert p == stored, (
            f"{run_dir}/{subdir}/{ch}: recomputed p {p} != stored {stored} "
            f"— provenance/n_perm mismatch")
        rec["channels"][ch] = dict(
            share_vs_real=obs, null_median=med, p_perm=p,
            indicator=bool(obs > med),
            vs_source=float(pj["channels"][ch]["vs_source"]))
    cal_real = {k: v for k, v in pj["calibration"].items()
                if k not in PLANTED_KEYS}
    rec["cal_real_max"] = max(cal_real.values())
    rec["valid"] = bool(rec["cal_real_max"] < CAL_MAX
                        and rec["n_eval"] >= S_MIN)
    return rec


def aggregate(cells, stage1_identity=None):
    """cells: {run_basename: {fraction_label: rec}}. Registered M4
    aggregation (weak form + trajectory reporting)."""
    runs = sorted(cells)
    n = len(runs)
    assert n <= N_REGISTERED, "more runs than registered"
    out = dict(n_runs=n, n_registered=N_REGISTERED,
               stage1_identity=stage1_identity)
    invalid = [(r, f) for r in runs for f, rec in cells[r].items()
               if not rec["valid"]]
    out["invalid_cells"] = [f"{r}/{f}" for r, f in invalid]
    # cross-fraction fixed-eval-distribution gate (review #23 M4)
    for r in runs:
        shas = [rec["norms_sha"] for rec in cells[r].values()
                if rec["norms_sha"] is not None]
        assert len(set(shas)) <= 1, (
            f"{r}: raw_norms/norm_floor differ across fractions — the "
            f"passes did not share the registered fixed eval "
            f"distribution (parent §5)")
    out["probe_seed_flags"] = sorted({
        rec["probe_seed_flag"] for r in runs
        for rec in cells[r].values()})

    traj = {}
    for ch in FIRE_KEYS:
        traj[ch] = {}
        for sub, frac in FRACTIONS:
            recs = [cells[r][sub] for r in runs if cells[r][sub]["valid"]]
            succ = sum(1 for rec in recs if rec["channels"][ch]["indicator"])

            def _mean(field):
                vals = [rec["channels"][ch][field] for rec in recs]
                return float(np.mean(vals)) if vals else None
            traj[ch][sub] = dict(
                frac=frac, n_valid=len(recs), successes=succ,
                share_mean=_mean("share_vs_real"),
                median_mean=_mean("null_median"),
                theta1_mean=_mean("vs_source"))
    out["trajectory"] = traj

    # WEAK FORM (v2.2 §5): fire-channel share above the null at 100%
    fin = traj["distractor"]["se_probe"]
    out["weak_form"] = dict(
        statement="fire-channel (distractor) share above per-run null "
                  "median at 100%",
        successes=fin["successes"], n_valid=fin["n_valid"],
        holds=bool(fin["successes"] >= 7))

    # non-decay descriptive: distractor share_100 - share_25 per run
    deltas = []
    for r in runs:
        a, b = cells[r]["se_probe_snap25"], cells[r]["se_probe"]
        if a["valid"] and b["valid"]:
            deltas.append(b["channels"]["distractor"]["share_vs_real"]
                          - a["channels"]["distractor"]["share_vs_real"])
    if len(deltas) >= 3:
        m, lo, hi = bca_interval(np.asarray(deltas),
                                 np.random.default_rng(777), 2000)
        label = ("RISING" if lo > 0 else
                 "FALLING" if hi < 0 else "FLAT-OR-MIXED")
        out["non_decay"] = dict(delta_mean=m, delta_bca=[lo, hi],
                                n=len(deltas), label=label,
                                note="DESCRIPTIVE only")
    return out


def run(args):
    if "," in args.runs:
        run_dirs = [d.strip() for d in args.runs.split(",") if d.strip()]
    else:
        run_dirs = sorted(globmod.glob(args.runs))
    assert run_dirs, f"no runs matched {args.runs!r}"
    assert args.output, "registered read must persist: pass --output"
    # review #23 M3: byte-identity vs Stage-1 is the collator's ONLY
    # final-checkpoint gate for the 100% pass (the m4 bundle carries no
    # ckpt/ dir) — it must never be skippable at the registered read
    assert args.stage1_runs, (
        "registered read requires --stage1_runs (100% final-ckpt gate "
        "by byte-identity to the Stage-1 read inputs)")

    stage1_identity = None
    if args.stage1_runs:
        s1 = {os.path.basename(os.path.normpath(d)): d
              for d in sorted(globmod.glob(args.stage1_runs))}
        stage1_identity = {}
        for d in run_dirs:
            base = os.path.basename(os.path.normpath(d))
            assert base in s1, f"no Stage-1 match for {base}"
            for f in ("se_probe.json", "se_probe_dims.npz"):
                a = _sha(os.path.join(d, "se_probe", f))
                b = _sha(os.path.join(s1[base], "se_probe", f))
                assert a == b, (
                    f"{base}/se_probe/{f}: 100% input differs from the "
                    f"Stage-1 primary read's input — refusing to restate "
                    f"the weak form from different bytes")
            stage1_identity[base] = "BYTE-IDENTICAL"

    cells = {}
    for d in run_dirs:
        base = os.path.basename(os.path.normpath(d))
        assert base not in cells, f"duplicate run basename {base}"
        cells[base] = {sub: read_pass(d, sub, frac, args.n_perm, args.steps)
                       for sub, frac in FRACTIONS}
    out = aggregate(cells, stage1_identity)
    out["cells"] = cells
    # standing fit-counter rule: the m4 bundle carries no metrics/config;
    # counters transfer from Stage-1 via the byte-identity gate above
    out["fit_counters"] = ("N/A in this bundle — carried from the "
                           "Stage-1 read (99.5-99.9% x8, flag OK) via "
                           "the 100% byte-identity gate")
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "se_m4_read.json"), "w") as f:
        json.dump(out, f, indent=1)
    wf = out["weak_form"]
    print(f"M4 WEAK FORM: {wf['successes']}/{wf['n_valid']} at 100% -> "
          f"holds={wf['holds']}")
    for ch in FIRE_KEYS:
        row = " ".join(
            f"{sub.split('_')[-1]}:{out['trajectory'][ch][sub]['share_mean']:.3f}"
            f"({out['trajectory'][ch][sub]['successes']}/"
            f"{out['trajectory'][ch][sub]['n_valid']})"
            for sub, _ in FRACTIONS)
        print(f"  {ch}: {row}")
    if "non_decay" in out:
        nd = out["non_decay"]
        print(f"  distractor non-decay: {nd['delta_mean']:+.4f} "
              f"{nd['delta_bca']} -> {nd['label']}")
    return out


# ---------------------------------------------------------------- selfcheck

KEY_DIMS = (("position", 8), ("velocity", 9), ("planted_dup0", 8),
            ("planted_dup1", 8), ("planted_dup2", 8), ("planted_const", 4),
            ("distractor", 8))
FIX_PERM = 200


def _fixture_pass(rd, subdir, hot, ckpt_name, gseed, cal=0.1, n_eval=512,
                  manifest_steps=None, norms_tag=1.0):
    """One synthetic probe output; stored p via se_probe's OWN perm_null
    (cross-implementation, as in the reviewed se_read fixture).
    norms_tag varies raw_norms bytes (the same-buffer witness)."""
    from uncfield.se_probe import perm_null as probe_perm_null
    pdir = os.path.join(rd, subdir)
    os.makedirs(pdir, exist_ok=True)
    rng = np.random.default_rng(gseed)
    dims_d, dim_key = [], []
    for k, dim in KEY_DIMS:
        level = hot.get(k)
        base = (np.full(dim, level) if level is not None
                else 1.0 + rng.normal(0, 0.01, dim))
        dims_d.append(np.abs(base + rng.normal(0, 0.005, dim)))
        dim_key.extend([k] * dim)
    dims_d = np.concatenate(dims_d)
    dim_key = np.asarray(dim_key)
    np.savez(os.path.join(pdir, "se_probe_dims.npz"), dims_d=dims_d,
             dims_label=np.asarray([k in FIRE_KEYS for k in dim_key]),
             dim_key=dim_key,
             raw_norms=np.full(dims_d.size, norms_tag, np.float32),
             norm_floor=np.float32(0.01))
    real_keys = [k for k, _ in KEY_DIMS if k not in PLANTED_KEYS]
    channels = {}
    for ch in [k for k, _ in KEY_DIMS if k in PLANTED_KEYS]:
        mask = np.asarray([dk == ch or dk in real_keys for dk in dim_key])
        sub_d = dims_d[mask]
        sub_lab = np.asarray([dk == ch for dk in dim_key[mask]])
        _, p, _ = probe_perm_null(
            sub_d, sub_lab, FIX_PERM,
            np.random.default_rng(0 + zlib.crc32(ch.encode()) % 2 ** 16))
        channels[ch] = dict(p_perm=p)
    tot = dims_d.sum()
    key_share = {k: float(dims_d[dim_key == k].sum() / tot)
                 for k, _ in KEY_DIMS}
    for ch in channels:
        channels[ch]["vs_source"] = key_share[ch] / max(
            key_share["position"], 1e-12)
    pj = dict(seed=0, n_eval=n_eval, ckpt=os.path.join(rd, "x", ckpt_name),
              channels=channels, key_share=key_share,
              calibration={k: (cal if k not in PLANTED_KEYS else 0.2)
                           for k, _ in KEY_DIMS},
              pse2=dict(top_share=0.4, base_share=0.4, p_rank=0.5,
                        families={}))
    with open(os.path.join(pdir, "se_probe.json"), "w") as f:
        json.dump(pj, f)
    if manifest_steps is not None:
        mdir = os.path.join(rd, "ckpt_snapshots")
        os.makedirs(mdir, exist_ok=True)
        with open(os.path.join(mdir, "manifest.json"), "w") as f:
            json.dump(dict(snapshots=[dict(step=s) for s in manifest_steps]),
                      f)


def _fixture_run_m4(tmp, name, hot25, hot50, hot100, gseed,
                    snap25="step000000126000", snap50="step000000250500",
                    manifest_steps=(0, 126000, 250500, 499000)):
    rd = os.path.join(tmp, name)
    _fixture_pass(rd, "se_probe_snap25", hot25, snap25, gseed,
                  manifest_steps=manifest_steps)
    _fixture_pass(rd, "se_probe_snap50", hot50, snap50, gseed + 1)
    _fixture_pass(rd, "se_probe", hot100, "20260814T000000F000000",
                  gseed + 2)
    return rd


def selfcheck():
    import tempfile
    HOT, COLD = {"distractor": 5.0}, {"distractor": 0.9}
    with tempfile.TemporaryDirectory() as tmp:
        # A) persistence: hot at all fractions -> weak form holds, RISING
        #    (100% hotter than 25%)
        runs = [_fixture_run_m4(os.path.join(tmp, "A"), f"r{i}",
                                {"distractor": 3.0}, HOT,
                                {"distractor": 6.0}, 100 + 10 * i)
                for i in range(8)]
        cells = {os.path.basename(d): {s: read_pass(d, s, f, FIX_PERM, STEPS)
                                       for s, f in FRACTIONS} for d in runs}
        agg = aggregate(cells)
        assert agg["weak_form"]["holds"] and \
            agg["weak_form"]["successes"] == 8
        assert agg["non_decay"]["label"] == "RISING", agg["non_decay"]
        assert agg["trajectory"]["distractor"]["se_probe_snap25"][
            "successes"] == 8
        assert agg["probe_seed_flags"] == ["OK"]

        # B) decay: hot at 25%, cold at 100% -> weak form FAILS, FALLING
        runs = [_fixture_run_m4(os.path.join(tmp, "B"), f"r{i}",
                                HOT, COLD, COLD, 200 + 10 * i)
                for i in range(8)]
        cells = {os.path.basename(d): {s: read_pass(d, s, f, FIX_PERM, STEPS)
                                       for s, f in FRACTIONS} for d in runs}
        agg = aggregate(cells)
        assert not agg["weak_form"]["holds"]
        assert agg["non_decay"]["label"] == "FALLING"

        # C) snapshot-name gate: non-step ckpt in a snap pass
        rd = _fixture_run_m4(os.path.join(tmp, "C"), "r0", HOT, HOT, HOT,
                             300, snap25="20260814T111111F000000")
        try:
            read_pass(rd, "se_probe_snap25", 0.25, FIX_PERM, STEPS)
            raise SystemExit("step-name gate FAILED to fire")
        except AssertionError as e:
            assert "step-named" in str(e)

        # D) fraction tolerance: snapshot too far from target
        rd = _fixture_run_m4(os.path.join(tmp, "D"), "r0", HOT, HOT, HOT,
                             400, snap25="step000000200000",
                             manifest_steps=(0, 200000, 250500))
        try:
            read_pass(rd, "se_probe_snap25", 0.25, FIX_PERM, STEPS)
            raise SystemExit("fraction-tolerance gate FAILED to fire")
        except AssertionError as e:
            assert "tolerance" in str(e)

        # E) selection rule: a nearer retained snapshot existed
        rd = _fixture_run_m4(os.path.join(tmp, "E"), "r0", HOT, HOT, HOT,
                             500, snap25="step000000130000",
                             manifest_steps=(0, 126000, 130000, 250500))
        try:
            read_pass(rd, "se_probe_snap25", 0.25, FIX_PERM, STEPS)
            raise SystemExit("selection-rule gate FAILED to fire")
        except AssertionError as e:
            assert "nearest" in str(e)

        # E2) manifest-absent branch: selection flagged UNVERIFIED,
        #     pass still loads (review #23 m13)
        rd = os.path.join(tmp, "E2", "r0")
        _fixture_pass(rd, "se_probe_snap25", HOT, "step000000126000",
                      950, manifest_steps=None)
        rec = read_pass(rd, "se_probe_snap25", 0.25, FIX_PERM, STEPS)
        assert rec["selection"].startswith("UNVERIFIED"), rec["selection"]

        # E3) cross-fraction norms witness: differing raw_norms across
        #     a run's passes must abort aggregation (review #23 M4)
        rd = _fixture_run_m4(os.path.join(tmp, "E3"), "r0", HOT, HOT,
                             HOT, 960)
        _fixture_pass(rd, "se_probe_snap50", HOT, "step000000250500",
                      961, norms_tag=2.0)
        cells_bad = {"r0": {s: read_pass(rd, s, f, FIX_PERM, STEPS)
                            for s, f in FRACTIONS}}
        try:
            aggregate(cells_bad)
            raise SystemExit("norms witness gate FAILED to fire")
        except AssertionError as e:
            assert "fixed eval distribution" in str(e)

        # F) provenance gate: doctored npz
        rd = _fixture_run_m4(os.path.join(tmp, "F"), "r0", HOT, HOT, HOT,
                             600)
        pth = os.path.join(rd, "se_probe_snap50", "se_probe_dims.npz")
        z = dict(np.load(pth))
        z["dims_d"] = z["dims_d"].copy()
        z["dims_d"][:4] = 99.0          # non-uniform: shares DO move
        np.savez(pth, **z)
        try:
            read_pass(rd, "se_probe_snap50", 0.50, FIX_PERM, STEPS)
            raise SystemExit("provenance gate FAILED to fire")
        except AssertionError as e:
            assert "provenance" in str(e)

        # G) end-to-end run() incl. stage1 byte-identity, both directions
        import shutil
        s1dir = os.path.join(tmp, "S1")
        runs = [_fixture_run_m4(os.path.join(tmp, "G"), f"r{i}",
                                HOT, HOT, HOT, 700 + 10 * i)
                for i in range(8)]
        for d in runs:
            dst = os.path.join(s1dir, os.path.basename(d))
            os.makedirs(dst, exist_ok=True)
            shutil.copytree(os.path.join(d, "se_probe"),
                            os.path.join(dst, "se_probe"))
        outd = os.path.join(tmp, "out")
        out = run(parse_args([
            "--runs", os.path.join(tmp, "G", "r*"),
            "--stage1_runs", os.path.join(s1dir, "r*"),
            "--n_perm", str(FIX_PERM), "--output", outd]))
        assert all(v == "BYTE-IDENTICAL"
                   for v in out["stage1_identity"].values())
        assert os.path.exists(os.path.join(outd, "se_m4_read.json"))
        with open(os.path.join(s1dir, "r0", "se_probe",
                               "se_probe.json")) as f:
            doc = json.load(f)
        doc["n_eval"] = 511
        with open(os.path.join(s1dir, "r0", "se_probe",
                               "se_probe.json"), "w") as f:
            json.dump(doc, f)
        try:
            run(parse_args([
                "--runs", os.path.join(tmp, "G", "r*"),
                "--stage1_runs", os.path.join(s1dir, "r*"),
                "--n_perm", str(FIX_PERM), "--output", outd]))
            raise SystemExit("byte-identity gate FAILED to fire")
        except AssertionError as e:
            assert "differs from the Stage-1" in str(e)

    print("se_m4_read selfcheck PASS (A persistence/RISING + weak-form, "
          "B decay/FALLING + weak-form-fails, C step-name gate, "
          "D fraction tolerance, E nearest-selection rule, F provenance, "
          "G end-to-end + byte-identity both directions)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
