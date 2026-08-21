"""SE ranking-probe DESCRIPTIVE DISCLOSURE reader (rankx;
PREREG_nfi_rankx_disclosure_20260820.md). Built + selfchecked BEFORE
its ONE registered execution.

DISCLOSURE, NOT A READ: consumes the never-read `pse2` fields of the
gradient-wave probe outputs (hetero seeds 28-31, flat 36-39) plus the
Stage-1 probes (seeds 10-17) as the registered-baseline context row.
No fire rules, no outcome map, no verdicts — faithful numbers with
provenance, labeled DESCRIPTIVE throughout. The registered
cross-check: the one execution must reproduce review #25 addendum
A3's independently-derived values.

Per run: delta_overall = pse2.top_share - pse2.base_share;
delta_dfam / delta_noise from pse2.families; p_rank carried; arm
identity pinned from config (hetero mod quadruple / flat scale /
stage1 neither — the frozen readers' pin values reused).

Run:  python -m uncfield.se_rankx_read --hetero "<glob>" --flat
          "<glob>" --stage1 "<glob>" --output <dir>
      python -m uncfield.se_rankx_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os

import numpy as np

from uncfield.se_mask import bca_interval
from uncfield.se_grad_read import (BASESD_N, DIM, FLAT_SCALE, MOD_HI,
                                   MOD_INDEX, MOD_KEY, MOD_LO)

ARM_SEEDS = {"hetero": set(range(28, 32)), "flat": set(range(36, 40)),
             "stage1": set(range(10, 18))}
STAGE1_REGISTERED = dict(delta=-0.0170, bca=[-0.0277, -0.0064],
                         note="the registered Stage-1 P-SE2 read "
                              "(8 runs; anchor context)")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hetero", default="")
    p.add_argument("--flat", default="")
    p.add_argument("--stage1", default="")
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def read_run(run_dir, arm):
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg["seed"])
    d = cfg["distractor"]
    assert int(d["dim"]) == DIM and abs(float(d["basesd"]) - BASESD_N) < 1e-9
    mk = str(d.get("mod_key", ""))
    scale = float(d["scale"])
    if arm == "hetero":
        assert (mk == MOD_KEY and int(d["mod_index"]) == MOD_INDEX
                and abs(float(d["mod_lo"]) - MOD_LO) < 1e-9
                and abs(float(d["mod_hi"]) - MOD_HI) < 1e-9
                and abs(scale - 1.0) < 1e-9), (run_dir, "hetero pins")
    elif arm == "flat":
        assert mk == "" and abs(scale - FLAT_SCALE) < 1e-9, (
            run_dir, "flat pins")
    else:
        assert mk == "" and abs(scale - 1.0) < 1e-9 and \
            str(d.get("gate_key", "")) == "", (run_dir, "stage1 pins")
    with open(os.path.join(run_dir, "se_probe", "se_probe.json")) as f:
        pj = json.load(f)
    ps = pj["pse2"]
    fam = ps["families"]
    return dict(
        run_dir=os.path.abspath(run_dir), arm=arm, train_seed=seed,
        delta_overall=float(ps["top_share"] - ps["base_share"]),
        delta_dfam=float(fam["dfam"]["top_share"]
                         - fam["dfam"]["base_share"]),
        delta_noise=float(fam["noise"]["top_share"]
                          - fam["noise"]["base_share"]),
        p_rank=float(ps["p_rank"]), K=int(ps["K"]), topk=int(ps["topk"]))


def aggregate(recs):
    out = dict(status="DESCRIPTIVE DISCLOSURE — unregistered data, "
                      "no claims (PREREG_nfi_rankx_disclosure_20260820)",
               stage1_registered_anchor=STAGE1_REGISTERED)
    for arm in ("hetero", "flat", "stage1"):
        rs = sorted((r for r in recs if r["arm"] == arm),
                    key=lambda r: r["train_seed"])
        seeds = [r["train_seed"] for r in rs]
        assert len(set(seeds)) == len(seeds), (arm, seeds)
        assert set(seeds) == ARM_SEEDS[arm], (arm, seeds)
        a = dict(n=len(rs))
        for f in ("delta_overall", "delta_dfam", "delta_noise"):
            vals = np.asarray([r[f] for r in rs])
            # crc32, not hash() — python hash is process-salted
            import zlib
            m, lo, hi = bca_interval(vals, np.random.default_rng(
                zlib.crc32(f"{arm}/{f}".encode()) % 2 ** 16), 2000)
            a[f] = dict(mean=m, bca=[lo, hi],
                        per_run=[round(float(v), 6) for v in vals],
                        sign_agreement=f"{int((vals < 0).sum())}/"
                                       f"{len(vals)} negative")
        a["p_rank_per_run"] = [r["p_rank"] for r in rs]
        out[arm] = a
    return out


def run(args):
    def expand(pat):
        if "," in pat:
            return [x.strip() for x in pat.split(",") if x.strip()]
        return sorted(globmod.glob(pat))

    assert args.output, "disclosure must persist: pass --output"
    recs = []
    for arm, pat in (("hetero", args.hetero), ("flat", args.flat),
                     ("stage1", args.stage1)):
        dirs = expand(pat)
        assert dirs, f"no runs matched for {arm}"
        recs += [read_run(d, arm) for d in dirs]
    out = aggregate(recs)
    out["per_run"] = recs
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "se_rankx.json"), "w") as f:
        json.dump(out, f, indent=1)
    for arm in ("hetero", "flat", "stage1"):
        a = out[arm]
        print(f"{arm}: overall {a['delta_overall']['mean']:+.4f} "
              f"{[round(x, 4) for x in a['delta_overall']['bca']]} | "
              f"dfam {a['delta_dfam']['mean']:+.4f} | "
              f"noise {a['delta_noise']['mean']:+.4f}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture(tmp, name, seed, arm, d_over, d_dfam, d_noise):
    rd = os.path.join(tmp, name)
    os.makedirs(os.path.join(rd, "se_probe"), exist_ok=True)
    scale = FLAT_SCALE if arm == "flat" else 1.0
    mod = (f"  mod_key: '{MOD_KEY}'\n  mod_index: {MOD_INDEX}\n"
           f"  mod_lo: {MOD_LO}\n  mod_hi: {MOD_HI}\n"
           if arm == "hetero" else
           "  mod_key: ''\n  mod_index: 0\n  mod_lo: 0.0\n"
           "  mod_hi: 1.0\n")
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(f"seed: {seed}\nplanted:\n  gate_key: ''\n"
                f"distractor:\n  gate_key: ''\n  dim: {DIM}\n"
                f"  scale: {scale}\n  basesd: {BASESD_N}\n{mod}"
                f"run:\n  steps: 500000.0\n")
    base = 0.4
    pj = dict(pse2=dict(
        top_share=base + d_over, base_share=base, p_rank=0.9, K=256,
        topk=10,
        families=dict(
            dfam=dict(top_share=0.2 + d_dfam, base_share=0.2),
            noise=dict(top_share=0.2 + d_noise, base_share=0.2))))
    with open(os.path.join(rd, "se_probe", "se_probe.json"), "w") as f:
        json.dump(pj, f)
    return rd


def selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        het = [_fixture(tmp, f"h{i}", 28 + i, "hetero",
                        -0.08 - 0.001 * i, +0.055, -0.135)
               for i in range(4)]
        fl = [_fixture(tmp, f"f{i}", 36 + i, "flat",
                       -0.009, -0.01, +0.001) for i in range(4)]
        s1 = [_fixture(tmp, f"s{i}", 10 + i, "stage1",
                       -0.017, -0.011, -0.006) for i in range(8)]
        out = run(parse_args([
            "--hetero", os.path.join(tmp, "h*"),
            "--flat", os.path.join(tmp, "f*"),
            "--stage1", os.path.join(tmp, "s*"),
            "--output", os.path.join(tmp, "out")]))
        assert abs(out["hetero"]["delta_overall"]["mean"]
                   - (-0.0815)) < 1e-9
        assert out["hetero"]["delta_noise"]["sign_agreement"] == \
            "4/4 negative"
        assert out["hetero"]["delta_dfam"]["sign_agreement"] == \
            "0/4 negative"
        assert out["status"].startswith("DESCRIPTIVE")
        # identity gates: wrong-arm pins abort
        bad = _fixture(tmp, "bad", 28, "flat", 0, 0, 0)
        try:
            read_run(bad, "hetero")
            raise SystemExit("hetero pin gate FAILED to fire")
        except AssertionError as e:
            assert "hetero pins" in str(e)
        # wrong seed family fatal
        rogue = [_fixture(tmp, f"r{i}", 50 + i, "hetero",
                          -0.08, 0.05, -0.13) for i in range(4)]
        try:
            aggregate([read_run(d, "hetero") for d in rogue]
                      + [read_run(d, "flat") for d in fl]
                      + [read_run(d, "stage1") for d in s1])
            raise SystemExit("seed gate FAILED to fire")
        except AssertionError:
            pass
    print("se_rankx_read selfcheck PASS (means exact, sign tallies, "
          "hetero/flat/stage1 pin gates, seed-family gate, "
          "descriptive-status label)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
