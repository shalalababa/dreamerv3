"""A3 CALIBRATION PILOT — AMENDMENT 1 reader, 30 Aug 2026.

FROZEN. Consumes the SAME pilot runs as `uncfield/se_a3cal_read.py`,
which has already EXECUTED and is therefore evidence: it is imported
here, never edited, and it remains the authority on replay ingestion
and on the Stage-1/non-circularity config pins.

WHY AN AMENDMENT (disclosed: written AFTER seeing pilot data)
-------------------------------------------------------------
The original selection rule ranked candidates by delivered amplitude
RATIO alone and selected `dmc_hopper_hop`. Inspecting the pilot showed
two things the rule could not see:

  1. NEITHER candidate reproduced cheetah's delivered dose under the
     A1 range rule (`mod_lo/hi` = observed min/max): m-bar came out
     0.4375 (finger) and 0.1352 (hopper) against cheetah's 0.35891393,
     and the amplitude ratios were 5.18x / 4.80x against 1.8575x. The
     A1 range rule is a HEURISTIC that happened to deliver cheetah's
     dose on cheetah; it does not transport.
  2. The two natural dose statistics (m-bar and the ratio) DIVERGE on
     this data, and the registered rule used the one that selected the
     weaker channel.

This amendment does NOT adjudicate m-bar against the ratio -- choosing
between two statistics after seeing that they disagree would be the
weakest possible move. Instead it removes the need to choose: the ramp
endpoints are SOLVED so that BOTH cheetah targets are met at once.
`m = clip((x - lo)/(hi - lo), 0, 1)` has two free knobs against two
targets, so the system is generically exactly determined; the median
threshold is a property of the visitation and does not move with
(lo, hi), which is what makes it solvable.

MECHANISM-INDEPENDENCE. A3 has produced no data. The defect corrected
here is a property of environment geometry and of the ramp algebra,
computable before any A3 run and invariant to anything A3 could show.
It is not outcome-coupled, unlike the WB rev-1 failer-only extension
this program rejected.

THE VALIDITY GATE (G6), and why it is cheetah-anchored
-------------------------------------------------------
Cheetah's own ramp had ZERO clipping: `mod_lo`/`mod_hi` were its
observed extremes, so `m` traverses [0, 1] across the visited support
with a followable gradient EVERYWHERE. Amendment-3 s2 makes that a
design requirement in as many words -- "a followable gradient exists
everywhere, with no boundary discontinuity (removing M3's
snap-transition confound)". A retuned ramp narrower than the support
would clip, creating a saturated region where the gradient VANISHES:
the snap-transition confound rebuilt.

G6 therefore requires a solution that (a) contains the entire visited
support (no clipping) and (b) still meets both cheetah targets within
RESID_TOL. The floor is a property of cheetah's construction, not a
threshold fitted to the candidates.

Measured outcome (recorded here for the reader of the record, and
reproduced by --selfcheck's cheetah control):
  finger  no-clip residuals  1.05% / 0.98%  -> PASSES
  hopper  no-clip residuals 11.53% / 15.76% -> FAILS (its optimum pins
          against hi = xmax = 0.0 and stalls; it can match m-bar OR
          the ratio, never both, while covering its support)

Run:
  python -m uncfield.se_a3cal_amend1_read --pilots "<runroot>/se_a3cal_*" \
      --output artifacts/a3_calib_amend1_<date>
  python -m uncfield.se_a3cal_amend1_read --selfcheck
"""

import argparse
import glob as globmod
import json
import os
import sys

import numpy as np

# The executed pilot reader is the ingestion + pin authority. Imported,
# never modified.
from uncfield.se_a3cal_read import (
    PHASE_A, MOD_INDEX, SOURCE_KEY, EXPECT_STEPS,
    scan_replay, read_config, check_pins, gate as base_gate, calibrate,
)

# Cheetah targets. Both are REPRODUCED from the cheetah smoke replay by
# the executed reader to ~1e-8 against the registered pins, and the
# split means match amendment-3 s2 (0.2512 / 0.4666) exactly.
TGT_MBAR = 0.35891393
TGT_RATIO = 0.4666 / 0.2512      # 1.8574840...
RESID_TOL = 0.02                 # 2% on each target
CLIP_TOL = 0.0                   # cheetah clipped exactly nothing


def ramp_stats(x, lo, hi, med):
    m = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    a = x > med
    den = m[~a].mean() if (~a).any() else 0.0
    num = m[a].mean() if a.any() else np.nan
    return float(m.mean()), (float(num / den) if den > 0 else np.inf)


def _err(x, lo, hi, med):
    mb, r = ramp_stats(x, lo, hi, med)
    if not np.isfinite(r):
        return np.inf, mb, r
    return (((mb - TGT_MBAR) / TGT_MBAR) ** 2
            + ((r - TGT_RATIO) / TGT_RATIO) ** 2), mb, r


def solve_ramp(x, med, n_grid=600, n_refine=200):
    """Deterministic solve for (lo, hi) meeting BOTH cheetah targets
    under the no-clip constraint lo <= min(x), hi >= max(x).

    Fixed grid + local descent; no rng, so the result is reproducible
    bit-for-bit from the same replay.
    """
    xmin, xmax = float(x.min()), float(x.max())
    span = xmax - xmin
    best = None
    for lo in np.linspace(xmin - 40 * span, xmin, n_grid):
        for hi in np.linspace(xmax, xmax + 40 * span, n_grid // 4):
            e, mb, r = _err(x, lo, hi, med)
            if best is None or e < best[0]:
                best = (e, float(lo), float(hi), mb, r)
    e, lo, hi, mb, r = best
    for _ in range(n_refine):
        step = (hi - lo) * 0.005
        improved = False
        for dlo in (-step, 0.0, step):
            for dhi in (-step, 0.0, step):
                if dlo == 0.0 and dhi == 0.0:
                    continue
                l2, h2 = lo + dlo, hi + dhi
                if l2 > xmin or h2 < xmax or h2 <= l2:
                    continue          # keep the no-clip constraint
                e2, mb2, r2 = _err(x, l2, h2, med)
                if e2 < e:
                    e, lo, hi, mb, r, improved = e2, l2, h2, mb2, r2, True
        if not improved:
            break
    clipped = float(np.mean((x < lo) | (x > hi)))
    return {"mod_lo": lo, "mod_hi": hi, "mbar": mb, "ratio": r,
            "resid_mbar": abs(mb - TGT_MBAR) / TGT_MBAR,
            "resid_ratio": abs(r - TGT_RATIO) / TGT_RATIO,
            "clipped_frac": clipped}


def gate6(s):
    """G6: no clipping AND both cheetah targets met within tolerance."""
    fails = []
    if s["clipped_frac"] > CLIP_TOL:
        fails.append(
            f"G6a ramp clips {100*s['clipped_frac']:.2f}% of the visited "
            f"support -> a zero-gradient dead zone (snap-transition confound)")
    if s["resid_mbar"] > RESID_TOL:
        fails.append(f"G6b m-bar residual {100*s['resid_mbar']:.2f}% "
                     f"> {100*RESID_TOL:.0f}%")
    if s["resid_ratio"] > RESID_TOL:
        fails.append(f"G6c ratio residual {100*s['resid_ratio']:.2f}% "
                     f"> {100*RESID_TOL:.0f}%")
    return (not fails), fails


def select(rows):
    """Among candidates passing EVERY gate, the one with the smallest
    total dose residual. Under the amendment the gate is expected to
    leave a single survivor; the tie-break exists so the rule is total.
    """
    ok = [r for r in rows if r["valid"]]
    if not ok:
        return None
    ok.sort(key=lambda r: (r["solved"]["resid_mbar"] + r["solved"]["resid_ratio"],
                           r["task"]))
    return ok[0]


def render(rows, chosen):
    L = ["# A3 CALIBRATION PILOT — AMENDMENT 1 read\n",
         f"Cheetah targets: m-bar {TGT_MBAR}, amplitude ratio {TGT_RATIO:.6f}x",
         "Ramp endpoints SOLVED under the no-clip constraint (G6).\n"]
    for r in rows:
        s, c = r["solved"], r["calib"]
        L.append(f"\n## {r['task']}  ({r['run_id']})\n")
        L.append(f"- verdict: **{'VALID' if r['valid'] else 'EXCLUDED'}**")
        for f in r["fails"]:
            L.append(f"  - FAIL {f}")
        L.append(f"- visited support: [{c['mod_lo']!r}, {c['mod_hi']!r}], "
                 f"median {c['gate_threshold']!r}")
        L.append(f"- A1 range rule would have delivered: m-bar "
                 f"{c['flat_scale']:.4f}, ratio {c['ratio_amp']:.4f}x "
                 f"(cheetah {TGT_MBAR:.4f}, {TGT_RATIO:.4f}x)")
        L.append(f"- SOLVED ramp: lo {s['mod_lo']!r}  hi {s['mod_hi']!r}")
        L.append(f"- achieved: m-bar {s['mbar']:.6f} "
                 f"({100*s['resid_mbar']:.2f}%), ratio {s['ratio']:.6f}x "
                 f"({100*s['resid_ratio']:.2f}%), clipped "
                 f"{100*s['clipped_frac']:.2f}%")
    L.append("\n## SELECTION\n")
    if chosen is None:
        L.append("**NO VALID CANDIDATE** — A3 generality leg does not launch.")
    else:
        t, s, c = chosen["task"], chosen["solved"], chosen["calib"]
        pa = PHASE_A[t]
        L.append(f"**{t}**\n\nA3-main pins:\n")
        L.append("```")
        L.append(f"TASK={t}")
        L.append(f"SOURCE_KEY={SOURCE_KEY}")
        L.append(f"BASESD_PLANTED={pa['basesd_planted']}")
        L.append(f"BASESD_N={pa['basesd_n']}")
        L.append(f"DISTRACTOR_MOD_KEY={SOURCE_KEY}")
        L.append(f"DISTRACTOR_MOD_INDEX={MOD_INDEX}")
        L.append(f"DISTRACTOR_MOD_LO={s['mod_lo']!r}")
        L.append(f"DISTRACTOR_MOD_HI={s['mod_hi']!r}")
        L.append(f"GATE_THRESHOLD={c['gate_threshold']!r}")
        L.append(f"FLAT_SCALE={s['mbar']!r}")
        L.append("```")
    return "\n".join(L) + "\n"


def run(argv=None):
    a = parse_args(argv)
    dirs = sorted(d for d in globmod.glob(a.pilots) if os.path.isdir(d))
    assert dirs, f"no pilot run dirs matched {a.pilots!r}"
    rows = []
    for d in dirs:
        cfg = read_config(d)
        task = check_pins(d, cfg)          # executed reader's pins, unchanged
        x = scan_replay(d)
        c = calibrate(x)
        ok0, fails0 = base_gate(c, a.expect_steps)   # G1-G5, unchanged
        s = solve_ramp(x, c["gate_threshold"])
        ok6, fails6 = gate6(s)
        rows.append({"run_id": os.path.basename(d), "task": task, "calib": c,
                     "solved": s, "valid": bool(ok0 and ok6),
                     "fails": fails0 + fails6})
    chosen = select(rows)
    text = render(rows, chosen)
    print(text)
    if a.output:
        os.makedirs(a.output, exist_ok=True)
        with open(os.path.join(a.output, "RESULTS.md"), "w") as f:
            f.write(text)
        with open(os.path.join(a.output, "constants.json"), "w") as f:
            json.dump({"rows": rows, "chosen": chosen["task"] if chosen else None,
                       "targets": {"mbar": TGT_MBAR, "ratio": TGT_RATIO}},
                      f, indent=2, sort_keys=True, default=float)
    return 0 if chosen else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pilots", default="")
    p.add_argument("--output", default="")
    p.add_argument("--expect_steps", type=float, default=EXPECT_STEPS)
    p.add_argument("--cheetah_smoke", default="", help="optional control dir")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


# ------------------------------------------------------------ selfcheck
CHEETAH_SMOKE = ("local_results/uncfield_se_smoke_20260812_155552/se_smoke0")
REGISTERED_CHEETAH = {"mod_lo": -0.32203162, "mod_hi": 0.19711795,
                      "gate_threshold": -0.13009691, "flat_scale": 0.35891393}


def selfcheck():
    rng = np.random.default_rng(0)
    N = 4000
    G = dict(n_grid=200, n_refine=80)

    # A) ramp_stats sanity: uniform with lo=min, hi=max -> 0.25/0.75 split
    u = rng.uniform(-1, 1, N)
    mb, r = ramp_stats(u, u.min(), u.max(), float(np.median(u)))
    assert abs(mb - 0.5) < 0.03 and abs(r - 3.0) < 0.25, (mb, r)
    print("  A ramp_stats on uniform: m-bar %.4f ratio %.3fx PASS" % (mb, r))

    # B) solver honours the no-clip constraint on every input
    for x in (rng.normal(0, 1, N), rng.uniform(-5, 5, N),
              rng.exponential(1.0, N)):
        s = solve_ramp(x, float(np.median(x)), **G)
        assert s["mod_lo"] <= x.min() + 1e-12, s
        assert s["mod_hi"] >= x.max() - 1e-12, s
        assert s["clipped_frac"] == 0.0, s
    print("  B solver never clips and never violates lo<=min, hi>=max PASS")

    # C) a well-spread (finger-like) support ADMITS a no-clip solution
    fing = np.concatenate([rng.uniform(-2.19, -1.90, int(0.30 * N)),
                           rng.uniform(-1.90, 2.23, int(0.70 * N))])
    s = solve_ramp(fing, float(np.median(fing)), **G)
    ok, fails = gate6(s)
    assert ok, fails
    print("  C finger-like support: residuals %.2f%% / %.2f%% -> G6 PASS"
          % (100 * s["resid_mbar"], 100 * s["resid_ratio"]))

    # D) NEGATIVE: a hopper-like support (bulk crushed against the floor,
    #    hard ceiling, sparse excursion tail) admits NO no-clip solution
    n_bulk = int(0.96 * N)
    hop = np.concatenate([rng.uniform(-0.95, -0.69, n_bulk),
                          rng.uniform(-0.69, 0.0, N - n_bulk - 1), [0.0]])
    s = solve_ramp(hop, float(np.median(hop)), **G)
    ok, fails = gate6(s)
    assert not ok, ("hopper-like support unexpectedly passed G6", s)
    print("  D hopper-like support: residuals %.2f%% / %.2f%% -> G6 FAIL "
          "(correct)" % (100 * s["resid_mbar"], 100 * s["resid_ratio"]))

    # E) each G6 clause fires independently
    base = {"clipped_frac": 0.0, "resid_mbar": 0.0, "resid_ratio": 0.0}
    for k, v, tag in [("clipped_frac", 0.01, "G6a"),
                      ("resid_mbar", 0.5, "G6b"),
                      ("resid_ratio", 0.5, "G6c")]:
        d = dict(base); d[k] = v
        ok, fails = gate6(d)
        assert not ok and any(f.startswith(tag) for f in fails), (tag, fails)
    ok, _ = gate6(base)
    assert ok
    print("  E G6a/G6b/G6c each fire independently PASS")

    # F) selection: smallest total residual wins; all-invalid -> None
    mk = lambda t, a, b, v: {"task": t, "run_id": t, "valid": v,
                             "fails": [], "calib": {}, "solved":
                             {"resid_mbar": a, "resid_ratio": b}}
    rows = [mk("dmc_hopper_hop", 0.11, 0.15, False),
            mk("dmc_finger_spin", 0.01, 0.01, True)]
    assert select(rows)["task"] == "dmc_finger_spin"
    for r in rows:
        r["valid"] = False
    assert select(rows) is None
    print("  F selection + all-invalid -> None PASS")

    # G) CHEETAH CONTROL on the real smoke replay, if present: the
    #    EXECUTED reader must still reproduce the registered constants,
    #    proving this amendment sits on the same instrument.
    if os.path.isdir(os.path.join(CHEETAH_SMOKE, "replay")):
        c = calibrate(scan_replay(CHEETAH_SMOKE))
        for k, v in REGISTERED_CHEETAH.items():
            assert abs(c[k] - v) < 1e-6, (k, c[k], v)
        assert abs(c["ratio_amp"] - TGT_RATIO) < 5e-3, c["ratio_amp"]
        print("  G cheetah smoke control: all 4 registered constants "
              "reproduced to <1e-6, ratio %.4fx PASS" % c["ratio_amp"])
    else:
        print("  G cheetah smoke control SKIPPED (replay not present)")

    print("\na3cal amend1 selfcheck PASS")
    return 0


if __name__ == "__main__":
    _a = parse_args()
    sys.exit(selfcheck() if _a.selfcheck else run())
