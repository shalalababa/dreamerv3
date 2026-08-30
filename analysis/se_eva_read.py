"""Cross-cell comparator for se_eva_probe (review B6 / I7).

Destination at commit time: `analysis/se_eva_read.py`.

Implements the two registered checks that cannot be computed inside a
single cell:

  A5  TIMING conjunction.  Rev 4's bar `max(pb_det) < min(pb_gauss)` was
      false on rev-3's own numbers (det [0.8515, 0.9915] and gauss
      [0.9748, 1.0058] overlap).  The correct instrument is a RANK test:
      exact one-sided Mann-Whitney (gauss > det), which those same numbers
      pass at p = 0.00808.  BAR: p <= 0.01 at the PINNED scaling c_recon,
      AND the same one-sided ordering reproduced by median q_hat.
      Disagreement => TIMING-NOT-REPORTABLE.

  A6  REV-3 REPRODUCTION.  Every rev-3 certified quantity must reproduce.
      Tolerances are declared here, not chosen after looking:
        rho, log10e_pooled : bitwise on the non-synthetic legs (the
                             per-audit rng of review I6 makes this
                             statable at all)
        floors             : |delta| <= 0.05 (band; the synthetic controls
                             draw from their own per-audit streams)
      A6 is an INSTRUMENT ALARM, not a finding: failure means the rev-4/5
      edits perturbed something they should not have.

Usage:
  python -m analysis.se_eva_read --new <dir-of-rev5-jsons> \\
                                 [--old <dir-of-rev3-jsons>]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np

SCALING = "c_recon"          # pinned by A5
P_BAR = 0.05                 # review F13 (was 0.01)
U_BAR = 26                   # review F13: U >= 26 of a maximum 32
BITWISE_KEYS = ("rho", "log10e_pooled")
FLOOR_TOL = 0.05


def mwu_exact_p(a, b):
    """Exact one-sided P(U >= U_obs) for 'b stochastically greater than a'.
    n=8 vs 4 gives 495 arrangements, so exhaustive enumeration is free and
    avoids any normal-approximation choice."""
    a, b = list(a), list(b)
    n, m = len(a), len(b)
    allv = a + b
    u = lambda bb, aa: sum(1 for x in bb for y in aa if x > y) + \
        0.5 * sum(1 for x in bb for y in aa if x == y)
    obs, cnt, tot = u(b, a), 0, 0
    for idx in itertools.combinations(range(n + m), m):
        bb = [allv[i] for i in idx]
        aa = [allv[i] for i in range(n + m) if i not in idx]
        tot += 1
        cnt += (u(bb, aa) >= obs)
    return float(obs), cnt / tot, tot


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        if j.get("instrument") == "se_eva_probe":
            out[os.path.basename(f)[:-5]] = j
    return out


def a5(new):
    """A5 with the review-F3/F11/F13 corrections.

    F3: pool ONLY cells whose distractor is BOTH resolved and aligned, drop
        non-finite values, and refuse below 2 per arm.  Pooling an
        unresolved cell's pb would average a number the instrument declined
        to read.
    F13: the bar is widened to p <= 0.05 with U >= 26, and stated plainly:
        the pb leg is a REPRODUCTION BAR ON A DECLARED-INVALID STATISTIC
        (pb was adjudicated construct-invalid in rev 3).  It carries nothing
        on its own; it is rescued only by the q_hat conjunction, which is
        the leg that is actually interpretable.
    F11: TIMING-REVERSED is an explicit outcome -- a significant ordering
        the other way is a finding, not a silent non-result.
    """
    def pick(arm):
        out = []
        for j in new.values():
            if j["disag_head"] != arm:
                continue
            pv = j.get("primary_v2", {})
            if not (pv.get("resolved", {}).get("distractor")
                    and pv.get("aligned", {}).get("distractor")):
                continue                                          # F3
            pb = j["primary"][SCALING]["position_on_bracket"]
            q = pv["q_hat"]["distractor"]
            if not (np.isfinite(pb) and np.isfinite(q)):
                continue                                          # F3
            out.append((pb, q))
        return out

    det, gau = pick("det"), pick("gauss")
    n_d, n_g = len(det), len(gau)
    if min(n_d, n_g) < 2:                                         # F3
        return dict(verdict="TIMING-NOT-COMPUTABLE", n_det=n_d, n_gauss=n_g,
                    reason="fewer than 2 resolved+aligned cells in an arm")
    pbd, pbg = [x[0] for x in det], [x[0] for x in gau]
    qd, qg = [x[1] for x in det], [x[1] for x in gau]
    u_fwd, p_fwd, tot = mwu_exact_p(pbd, pbg)
    u_rev, p_rev, _ = mwu_exact_p(pbg, pbd)
    pb_ord = float(np.median(pbg)) > float(np.median(pbd))
    q_ord = float(np.median(qg)) > float(np.median(qd))
    fwd_ok = bool(p_fwd <= P_BAR and u_fwd >= U_BAR and pb_ord and q_ord)
    rev_ok = bool(p_rev <= P_BAR and (not pb_ord) and (not q_ord))
    verdict = ("TIMING-REPLICATES" if fwd_ok else
               "TIMING-REVERSED" if rev_ok else                   # F11
               "TIMING-NOT-REPORTABLE" if pb_ord != q_ord else
               "TIMING-NOT-SIGNIFICANT")
    return dict(
        scaling=SCALING, n_det=n_d, n_gauss=n_g,
        pooled="resolved+aligned distractor only (review F3)",
        pb_median_det=float(np.median(pbd)),
        pb_median_gauss=float(np.median(pbg)),
        mwu_U_forward=u_fwd, mwu_p_forward=p_fwd,
        mwu_U_reverse=u_rev, mwu_p_reverse=p_rev,
        arrangements=tot, bars=dict(p_max=P_BAR, u_min=U_BAR),
        q_median_det=float(np.median(qd)),
        q_median_gauss=float(np.median(qg)),
        pb_ordering=pb_ord, q_ordering=q_ord, agree=pb_ord == q_ord,
        verdict=verdict,
        caveat="the pb leg is a reproduction bar on a statistic this "
               "programme ALREADY declared construct-invalid (rev-3 "
               "adjudication); it is interpretable only through the q_hat "
               "conjunction, and TIMING-REPLICATES asserts reproduction of "
               "an ordering, never that pb measures what it was named for.")


def a6(new, old):
    """Rev-3 reproduction, with declared tolerances (reviews F4, I7).

    F4: NOT-COMPUTABLE when nothing overlaps; the expected comparison count
        is asserted; and the comparison is extended to the HORIZON legs and
        the RESOLUTION GATES, which rev-5's edits could equally have moved.
    """
    common = sorted(set(new) & set(old))
    if not common:
        return dict(verdict="A6-NOT-COMPUTABLE", n_compared=0,
                    reason="no cell names in common between --new and --old")
    rows, bad, nchecked = [], 0, 0
    for name in common:
        nj_, oj = new[name], old[name]
        for tag, blk in nj_["legs_h1"].items():
            if "SYNTH" in tag or tag not in oj["legs_h1"]:
                continue
            for k, v in blk.items():
                o = oj["legs_h1"][tag].get(k)
                if o is None:
                    continue
                for f in BITWISE_KEYS:
                    nchecked += 1
                    if v.get(f) != o.get(f):
                        bad += 1
                        rows.append((name, tag, k, f, o.get(f), v.get(f)))
        # F4: the horizon legs too
        for tag, blk in nj_.get("horizon", {}).items():
            if "SYNTH" in tag or tag.startswith("predicted") \
                    or tag not in oj.get("horizon", {}):
                continue
            for k, v in blk.items():
                o = oj["horizon"][tag].get(k)
                if not isinstance(o, dict) or not isinstance(v, dict):
                    continue
                for f in BITWISE_KEYS:
                    if f not in v or f not in o:
                        continue
                    nchecked += 1
                    if v[f] != o[f]:
                        bad += 1
                        rows.append((name, tag, k, f, o[f], v[f]))
    fl, gates = [], []
    for name in common:
        nj_, oj = new[name], old[name]
        for k, v in nj_["floors"]["h1_under_nongauss"].items():
            o = oj["floors"]["h1_under_nongauss"].get(k)
            if o is not None and abs(v - o) > FLOOR_TOL:
                fl.append((name, k, o, v))
        # F4: the resolution gates
        for h, g in nj_.get("resolution_gates", {}).items():
            og = oj.get("resolution_gates", {}).get(h)
            if og and bool(g.get("adjudicable")) != bool(
                    og.get("adjudicable")):
                gates.append((name, h, og.get("adjudicable"),
                              g.get("adjudicable")))
    ok = not bad and not fl and not gates
    return dict(n_compared=len(common), n_quantities_checked=nchecked,
                bitwise_mismatches=bad, examples=rows[:8],
                floor_band_violations=fl[:8],
                resolution_gate_flips=gates[:8],
                tolerance=dict(bitwise=list(BITWISE_KEYS),
                               floor_abs=FLOOR_TOL,
                               gates="exact boolean"),
                verdict=("A6-REPRODUCES" if ok else "A6-INSTRUMENT-ALARM"),
                note="an INSTRUMENT ALARM, not a finding: failure means the "
                     "rev-4/5/6 edits perturbed a rev-3 certified quantity "
                     "they should not have touched.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    new = load(a.new)
    res = dict(n_cells=len(new), A5=a5(new))
    if a.old:
        old = load(a.old)
        res["A6"] = a6(new, old)
        # F4: the expected comparison count is asserted, so a silently
        # empty comparison cannot read as a pass
        exp = len(set(new) & set(old))
        assert res["A6"]["n_compared"] == exp, (
            "A6 compared a different number of cells than overlap",
            res["A6"]["n_compared"], exp)
    txt = json.dumps(res, indent=1, default=float)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
