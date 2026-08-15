"""SE-wave D-only interference-arm FROZEN reader
(PREREG_nfi_scale_exhibit_amend2_20260814.md §1; built + selfchecked
BEFORE the read, executes it ONCE).

The arm: Stage-1 config WITHOUT the distractor (D0/D1/D2 + C only),
seeds 10-13, 4 runs. Registered question: did N's presence suppress
D attribution in Stage 1 (cross-channel interference)?

Registered outcome map (amendment §1, theta_1 materiality band
[0.90, 1.10] — the raw CI-excludes-1.0 and >=3/4-indicator forms were
verified DEGENERATE on parity fixtures and replaced pre-freeze):
  (a) NO-INTERFERENCE: dup0 theta_1 seed-level BCa CI entirely within
      [0.90, 1.10] AND no D channel at 4/4 indicators AND all 4
      registered runs loaded+valid -> Stage-1 D-null stands
      unqualified.
  (b) INTERFERENCE: dup0 theta_1 CI entirely > 1.10 OR any D channel
      at 4/4 indicators -> Stage-1 D-null qualified as N-suppression.
  (c) MIXED: anything else -> descriptive only.
Power disclosure: n=4; 4/4 has exact binomial p=.0625 per channel
(familywise ~.18) — a registered ROBUSTNESS CHECK, not a confirmatory
test; the sensitive instrument is theta_1 (Stage-1 precision ±0.1%).
Invalid runs count as indicator FAILURES with the denominator pinned
at 4, and NO-INTERFERENCE is UNREACHABLE with a degraded instrument
(missing or invalid runs -> at best MIXED).

Gates per run (same family as the frozen primary reader): final-ckpt
gate; provenance p-equality (se_read.channel_null, bit-identical);
S >= 256; calibration < 0.5 on every real key; train seeds distinct
and within {10..13} (duplicates fatal; out-of-family flagged).

Run:  python -m uncfield.se_donly_read --runs "<glob>" --output <dir>
      python -m uncfield.se_donly_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import zlib

import numpy as np

from uncfield.se_mask import bca_interval
from uncfield.se_probe import PLANTED_KEYS, resolve_ckpt
from uncfield.se_read import CAL_MAX, S_MIN, binom_tail, channel_null

D_KEYS = ("planted_dup0", "planted_dup1", "planted_dup2")
N_ARM = 4                     # registered denominator (seeds 10-13)
SEED_FAMILY = set(range(10, 14))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="",
                   help="comma-separated run logdirs, or a glob")
    p.add_argument("--probe_subdir", default="se_probe")
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--expect_steps", type=float, default=5e5)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def read_run(run_dir, probe_subdir, n_perm):
    """One D-only run. Mirrors the frozen se_read.read_run gates with
    the fire-eligible set restricted to the D-family (no distractor
    key exists in this arm)."""
    pdir = os.path.join(run_dir, probe_subdir)
    with open(os.path.join(pdir, "se_probe.json")) as f:
        pj = json.load(f)
    npz = np.load(os.path.join(pdir, "se_probe_dims.npz"))
    dims_d = np.asarray(npz["dims_d"])          # stored dtype
    dim_key = np.asarray(npz["dim_key"]).astype(str)
    seed = int(pj["seed"])
    all_keys = sorted(set(dim_key.tolist()))
    assert "distractor" not in all_keys, (
        f"{run_dir}: distractor present — this is not a D-only run")
    real_keys = [k for k in all_keys if k not in PLANTED_KEYS]

    # final-checkpoint gate (as in se_read.read_run)
    try:
        final = resolve_ckpt(run_dir, "")
        ckpt_final_ok = (os.path.basename(os.path.normpath(final))
                         == os.path.basename(os.path.normpath(pj["ckpt"])))
    except (AssertionError, FileNotFoundError, OSError):
        final, ckpt_final_ok = None, None
    assert ckpt_final_ok is not False, (
        f"{run_dir}: probed ckpt is not the final checkpoint")

    train_seed = None
    try:
        import yaml
        with open(os.path.join(run_dir, "config.yaml")) as f:
            train_seed = int(yaml.safe_load(f)["seed"])
    except Exception:
        pass

    rec = dict(run_dir=os.path.abspath(run_dir), probe_seed=seed,
               train_seed=train_seed, ckpt_final_ok=ckpt_final_ok,
               n_eval=int(pj["n_eval"]), channels={})
    for ch in D_KEYS:
        assert ch in all_keys, f"{run_dir}: channel {ch} missing"
        rng = np.random.default_rng(seed + zlib.crc32(ch.encode()) % 2 ** 16)
        obs, p, med = channel_null(dims_d, dim_key, ch, real_keys,
                                   n_perm, rng)
        stored = float(pj["channels"][ch]["p_perm"])
        assert p == stored, (
            f"{run_dir}/{ch}: recomputed p {p} != stored {stored} — "
            f"provenance/n_perm mismatch")
        rec["channels"][ch] = dict(
            share_vs_real=obs, p_perm=p, null_median=med,
            indicator=bool(obs > med),
            vs_source=float(pj["channels"][ch]["vs_source"]))
    cal_real = {k: v for k, v in pj["calibration"].items()
                if k not in PLANTED_KEYS}
    rec["cal_real_max"] = max(cal_real.values())
    rec["valid"] = bool(rec["cal_real_max"] < CAL_MAX
                        and rec["n_eval"] >= S_MIN)
    return rec


def aggregate(recs):
    assert len(recs) <= N_ARM, "more runs than registered"
    tseeds = [r["train_seed"] for r in recs]
    known = [s for s in tseeds if s is not None]
    assert len(known) == len(set(known)), (
        f"duplicate training seeds {sorted(known)} — same run twice?")
    if None in tseeds:
        seed_check = "UNKNOWN-SEEDS"
    elif not all(s in SEED_FAMILY for s in known):
        seed_check = f"SEED-OUT-OF-FAMILY: {sorted(known)} != [10..13]"
    else:
        seed_check = "OK"
    invalid = [r["run_dir"] for r in recs if not r["valid"]]

    out = dict(n_registered=N_ARM, n_loaded=len(recs),
               invalid_runs=invalid, seed_check=seed_check,
               ckpt_flag=("OK" if all(r["ckpt_final_ok"] for r in recs)
                          else "UNVERIFIED (ckpt dir unreachable on "
                               "some runs)"),
               instrument_flag=("CLEAN" if not invalid
                                and len(recs) == N_ARM
                                else "VALIDITY-VIOLATIONS"))
    ch_out = {}
    for ch in D_KEYS:
        succ = sum(1 for r in recs
                   if r["valid"] and r["channels"][ch]["indicator"])
        theta = [r["channels"][ch]["vs_source"] for r in recs
                 if r["valid"]]
        rec = dict(successes=succ, binom_p=binom_tail(succ, N_ARM),
                   theta1_values=theta)
        if len(theta) >= 3:
            m, lo, hi = bca_interval(np.asarray(theta),
                                     np.random.default_rng(99), 2000)
            rec["theta1_mean"] = m
            rec["theta1_bca"] = [lo, hi]
            rec["theta1_minmax"] = [float(min(theta)), float(max(theta))]
        ch_out[ch] = rec
    out["channels"] = ch_out

    # amendment §1: theta_1 materiality band [0.90, 1.10] (raw
    # CI-excludes-1.0 degenerate at n=4 vs Stage-1's ±0.1% precision);
    # indicator arm is 4/4 only (>=3/4 fires at .3125/channel under the
    # parity null — verified degenerate on parity fixtures)
    d0 = ch_out["planted_dup0"]
    lo_gt_band = ("theta1_bca" in d0 and d0["theta1_bca"][0] > 1.10)
    ci_in_band = ("theta1_bca" in d0
                  and 0.90 <= d0["theta1_bca"][0]
                  and d0["theta1_bca"][1] <= 1.10)
    any_full = any(ch_out[ch]["successes"] == N_ARM for ch in D_KEYS)
    # review #23 M5: NO-INTERFERENCE requires the FULL clean instrument
    # (a missing/invalid run can only weaken the 4/4 route, so
    # degradation must never favor the claim-friendly verdict)
    clean = bool(len(recs) == N_ARM and not invalid)
    if lo_gt_band or any_full:
        verdict = "INTERFERENCE"
    elif ci_in_band and not any_full and clean:
        verdict = "NO-INTERFERENCE"
    else:
        verdict = "MIXED"
    out["verdict"] = verdict
    out["power_note"] = ("n=4 robustness check; 4/4 exact binomial "
                         "p=.0625 — not a confirmatory test")
    return out


def run(args):
    if "," in args.runs:
        run_dirs = [d.strip() for d in args.runs.split(",") if d.strip()]
    else:
        run_dirs = sorted(globmod.glob(args.runs))
    assert run_dirs, f"no runs matched {args.runs!r}"
    assert args.output, "registered read must persist: pass --output"
    recs = [read_run(d, args.probe_subdir, args.n_perm) for d in run_dirs]
    out = aggregate(recs)
    from uncfield.se_read import fit_counters
    out["fit_counters"] = {r["run_dir"]: fit_counters(r["run_dir"],
                                                      args.expect_steps)
                           for r in recs}
    out["per_run"] = [
        {f: r[f] for f in ("run_dir", "train_seed", "n_eval", "valid",
                           "cal_real_max", "ckpt_final_ok")}
        | {"indicators": {ch: r["channels"][ch]["indicator"]
                          for ch in D_KEYS},
           "theta1": {ch: r["channels"][ch]["vs_source"]
                      for ch in D_KEYS}}
        for r in recs]
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "se_donly_read.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"D-ONLY VERDICT: {out['verdict']} ({out['instrument_flag']}, "
          f"seeds {out['seed_check']})")
    for ch in D_KEYS:
        c = out["channels"][ch]
        print(f"  {ch}: {c['successes']}/4 theta1 "
              f"{c.get('theta1_mean', float('nan')):.4f} "
              f"{c.get('theta1_bca')}")
    return out


# ---------------------------------------------------------------- selfcheck

KEY_DIMS = (("position", 8), ("velocity", 9), ("planted_dup0", 8),
            ("planted_dup1", 8), ("planted_dup2", 8), ("planted_const", 4))
FIX_PERM = 200


def _fixture_run(tmp, name, seed, hot, theta_scale=None, n_eval=512,
                 cal=0.1):
    """D-only synthetic run (no distractor key). Stored p via
    se_probe's OWN perm_null (cross-implementation). theta_scale: if
    set, dup0's vs_source is scaled to that value regardless of dims
    (models decoder-share vs perm-share dissociation)."""
    from uncfield.se_probe import perm_null as probe_perm_null
    rd = os.path.join(tmp, name)
    pdir = os.path.join(rd, "se_probe")
    os.makedirs(pdir, exist_ok=True)
    rng = np.random.default_rng(1000 + seed)
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
             dims_label=np.asarray([k in D_KEYS for k in dim_key]),
             dim_key=dim_key)
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
    if theta_scale is not None:
        channels["planted_dup0"]["vs_source"] = float(theta_scale)
    ck = "20260815T000000F000000"
    os.makedirs(os.path.join(rd, "ckpt", ck), exist_ok=True)
    with open(os.path.join(rd, "ckpt", "latest"), "w") as f:
        f.write(ck)
    pj = dict(seed=0, n_eval=n_eval, ckpt=os.path.join(rd, "ckpt", ck),
              channels=channels, key_share=key_share,
              calibration={k: (cal if k not in PLANTED_KEYS else 0.2)
                           for k, _ in KEY_DIMS},
              pse2=dict(top_share=0.4, base_share=0.4, p_rank=0.5))
    with open(os.path.join(pdir, "se_probe.json"), "w") as f:
        json.dump(pj, f)
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(f"seed: {seed}\nrun:\n  steps: 500000.0\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 499968}) + "\n")
    return rd


def selfcheck():
    import tempfile
    HOT, PAR = 5.0, None
    with tempfile.TemporaryDirectory() as tmp:
        # (a) NO-INTERFERENCE: dups at parity with real dims
        runs = [_fixture_run(os.path.join(tmp, "A"), f"r{i}", 10 + i,
                             hot={}) for i in range(4)]
        agg = aggregate([read_run(d, "se_probe", FIX_PERM) for d in runs])
        assert agg["verdict"] == "NO-INTERFERENCE", agg["verdict"]
        assert agg["seed_check"] == "OK"
        assert agg["instrument_flag"] == "CLEAN"

        # (b) INTERFERENCE via indicators: dup0 hot in 4/4
        runs = [_fixture_run(os.path.join(tmp, "B"), f"r{i}", 10 + i,
                             hot={"planted_dup0": HOT}) for i in range(4)]
        agg = aggregate([read_run(d, "se_probe", FIX_PERM) for d in runs])
        assert agg["verdict"] == "INTERFERENCE"
        assert agg["channels"]["planted_dup0"]["successes"] == 4
        assert abs(agg["channels"]["planted_dup0"]["binom_p"]
                   - 1 / 16) < 1e-12

        # (b') INTERFERENCE via theta_1 alone (indicators low)
        runs = [_fixture_run(os.path.join(tmp, "B2"), f"r{i}", 10 + i,
                             hot={}, theta_scale=1.5 + 0.01 * i)
                for i in range(4)]
        agg = aggregate([read_run(d, "se_probe", FIX_PERM) for d in runs])
        assert agg["verdict"] == "INTERFERENCE"
        assert agg["channels"]["planted_dup0"]["theta1_bca"][0] > 1.0

        # (c) MIXED: dup0 mildly hot in 2/4
        runs = [_fixture_run(os.path.join(tmp, "C"), f"r{i}", 10 + i,
                             hot=({"planted_dup0": 1.3} if i < 2 else {}))
                for i in range(4)]
        agg = aggregate([read_run(d, "se_probe", FIX_PERM) for d in runs])
        assert agg["verdict"] == "MIXED", (
            agg["verdict"], agg["channels"]["planted_dup0"])

        # band boundaries (review #23 m14): theta ~1.05 tight CI is
        # inside the band -> NO-INTERFERENCE under the registered map
        # (would be INTERFERENCE under the superseded CI-excludes-1.0
        # rule — the mutant that separates the two rules)
        runs = [_fixture_run(os.path.join(tmp, "C3"), f"r{i}", 10 + i,
                             hot={}, theta_scale=1.05 + 0.001 * i)
                for i in range(4)]
        agg = aggregate([read_run(d, "se_probe", FIX_PERM) for d in runs])
        assert agg["verdict"] == "NO-INTERFERENCE", (
            agg["verdict"], agg["channels"]["planted_dup0"])
        # CI straddling the 1.10 edge -> MIXED
        runs = [_fixture_run(os.path.join(tmp, "C4"), f"r{i}", 10 + i,
                             hot={},
                             theta_scale=(1.05, 1.08, 1.12, 1.15)[i])
                for i in range(4)]
        agg = aggregate([read_run(d, "se_probe", FIX_PERM) for d in runs])
        assert agg["verdict"] == "MIXED", (
            agg["verdict"], agg["channels"]["planted_dup0"])

        # degraded instrument can never yield NO-INTERFERENCE (M5):
        # 3 parity runs (one registered run missing) -> MIXED
        runs = [_fixture_run(os.path.join(tmp, "C5"), f"r{i}", 10 + i,
                             hot={}) for i in range(3)]
        agg = aggregate([read_run(d, "se_probe", FIX_PERM) for d in runs])
        assert agg["verdict"] == "MIXED", agg["verdict"]
        assert agg["instrument_flag"] == "VALIDITY-VIOLATIONS"

        # distractor-key contamination must abort
        try:
            from uncfield import se_read as sr
            rd = sr._fixture_run(os.path.join(tmp, "D"), "r0", 10,
                                 hot={})
            read_run(rd, "se_probe", FIX_PERM)
            raise SystemExit("D-only arm gate FAILED to fire")
        except AssertionError as e:
            assert "not a D-only run" in str(e)

        # duplicate seeds fatal
        r1 = _fixture_run(os.path.join(tmp, "E"), "ra", 10, hot={})
        r2 = _fixture_run(os.path.join(tmp, "E"), "rb", 10, hot={})
        try:
            aggregate([read_run(d, "se_probe", FIX_PERM)
                       for d in (r1, r2)])
            raise SystemExit("duplicate-seed gate FAILED to fire")
        except AssertionError as e:
            assert "duplicate" in str(e)

        # out-of-family seed flagged (not fatal)
        r3 = _fixture_run(os.path.join(tmp, "F"), "r0", 15, hot={})
        agg = aggregate([read_run(r3, "se_probe", FIX_PERM)])
        assert agg["seed_check"].startswith("SEED-OUT-OF-FAMILY")

        # provenance gate
        rd = _fixture_run(os.path.join(tmp, "G"), "r0", 10, hot={})
        pth = os.path.join(rd, "se_probe", "se_probe_dims.npz")
        z = dict(np.load(pth))
        z["dims_d"] = z["dims_d"].copy()
        z["dims_d"][:4] = 99.0
        np.savez(pth, **z)
        try:
            read_run(rd, "se_probe", FIX_PERM)
            raise SystemExit("provenance gate FAILED to fire")
        except AssertionError as e:
            assert "provenance" in str(e)

        # end-to-end run() with output
        runs = [_fixture_run(os.path.join(tmp, "H"), f"r{i}", 10 + i,
                             hot={}) for i in range(4)]
        outd = os.path.join(tmp, "out")
        out = run(parse_args(["--runs", os.path.join(tmp, "H", "r*"),
                              "--n_perm", str(FIX_PERM),
                              "--output", outd]))
        assert out["verdict"] == "NO-INTERFERENCE"
        assert os.path.exists(os.path.join(outd, "se_donly_read.json"))

    print("se_donly_read selfcheck PASS (a no-interference, "
          "b indicator-interference 4/4 p=1/16, b' theta1-interference, "
          "c mixed, arm-contamination gate, duplicate-seed gate, "
          "out-of-family flag, provenance gate, end-to-end)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
