"""Frozen reader for the NFI family-generality factorial (7 Aug 2026).

Governed by prereg/PREREG_nfi_family2_20260807.md. Consumes the three cell
outputs (local_results/uncfield/family2/{lg_lstm,dc_gru,dc_lstm}/) and
executes the registered read EXACTLY:

  GATES (any failure -> HALT, no verdicts printed):
    G-STAMP   all three summaries carry the same stamp.git and the pinned
              thresholds (EPS_TRUE 0.02 / EPS_PRED 0.08 / burn 30 / loops 8)
    G-REPRO   member verdicts re-derived from the per-cycle npz columns
              (pinned thresholds, planner.classify tree verbatim) match the
              summaries byte-for-byte; ensemble rule re-derived and matched
    G-DATA    dc_gru.episodes_sha == dc_lstm.episodes_sha (shared data)

  P-F2  (primary):    ensemble verdict >= EXPLOIT-SURVIVES-CIG-ONLY in
                      EACH of the three new cells
  P-F2b (secondary):  conjunction location vs entropy boundedness —
                      lg_lstm has >=1 member with steady conjunction
                      (n_exploit_both > 0) AND dc cells have 0/8 members
                      with steady conjunction; dc transient-conjunction
                      counts reported (cycles exploit_cig AND
                      exploit_transient)
  Descriptives: per-member tv_max, top cycles, dc claimed-contraction vs
  the 5.545-nat world budget; no other quantities are read.

Selfcheck (--selfcheck): reproduces the QUARANTINED smoke cells' member
verdicts from their npzs (reader-vs-runner agreement on real files) and
kills two planted mutants (threshold-direction flip; conjunction-AND
dropped to OR) on synthetic columns.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from . import planner as pl

ROOT = pathlib.Path(__file__).resolve().parent.parent / "local_results" / "uncfield"
CELLS = ("lg_lstm", "dc_gru", "dc_lstm")
CIG_LEVEL = pl.VERDICTS.index("EXPLOIT-SURVIVES-CIG-ONLY")   # 4


def member_flags(npz):
    """Re-derive per-cycle exploit flags from stored (drift-adjusted) rates
    using the pinned thresholds — planner semantics verbatim."""
    neutral = npz["neutral"].astype(bool)
    f = {}
    f["naive"] = neutral & (npz["naive"] > pl.EPS_PRED)
    f["cig"] = neutral & (npz["carried"] > pl.EPS_PRED)
    f["pbim"] = neutral & (npz["dh_adj"] > pl.EPS_PRED)
    f["both"] = f["cig"] & f["pbim"]
    f["transient"] = neutral & (npz["transient"] > pl.EPS_PRED)
    f["transient_conj"] = f["cig"] & f["transient"]
    return f


def member_verdict(flags):
    """planner.classify decision tree, verbatim order."""
    if flags["both"].any():
        return pl.VERDICTS[5]
    if flags["cig"].any():
        return pl.VERDICTS[4]
    if flags["pbim"].any():
        return pl.VERDICTS[3]
    if flags["transient"].any():
        return pl.VERDICTS[2]
    if flags["naive"].any():
        return pl.VERDICTS[1]
    return pl.VERDICTS[0]


def ensemble_verdict(member_verdicts):
    levels = [pl.VERDICTS.index(v) for v in member_verdicts]
    n = len(levels)
    lvl = max((L for L in range(len(pl.VERDICTS))
               if sum(x >= L for x in levels) * 2 >= n), default=0)
    return pl.VERDICTS[lvl]


def read_cell(cell_dir):
    with open(cell_dir / "summary.json") as f:
        summary = json.load(f)
    members = []
    for m in range(len(summary["members"])):
        npz = np.load(cell_dir / f"results_m{m}.npz")
        flags = member_flags(npz)
        v = member_verdict(flags)
        n_tconj = int(flags["transient_conj"].sum())
        # cross-check the runner's pinned field when present (older smoke
        # summaries predate it — absent is tolerated, mismatch is not)
        stored_tconj = summary["members"][m].get("n_exploit_transient_conj")
        if stored_tconj is not None:
            assert stored_tconj == n_tconj, \
                f"transient_conj mismatch m{m}: {stored_tconj} != {n_tconj}"
        members.append(dict(
            member=m, verdict=v,
            n_both=int(flags["both"].sum()),
            n_cig=int(flags["cig"].sum()),
            n_transient=int(flags["transient"].sum()),
            n_transient_conj=n_tconj,
            summary_verdict=summary["members"][m]["verdict"],
            tv_max=summary["members"][m].get("tv_max_pred_rate"),
            top_cycle=summary["members"][m].get("top_cycle"),
            top_claimed_cum=summary["members"][m].get("top_claimed_cum"),
        ))
    ens = ensemble_verdict([m["verdict"] for m in members])
    return summary, members, ens


REPRO_PATH = (pathlib.Path(__file__).resolve().parent.parent / "artifacts"
              / "nfi_family2_read_20260807" / "reproduction.json")


def _reproduction_ok(path=None):
    """Amendment-1 compensating control: clean-HEAD end-to-end model-level
    reproduction of the claim-carrying dc members (see
    prereg/PREREG_nfi_family2_amend1_20260809.md)."""
    path = pathlib.Path(path or REPRO_PATH)
    if not path.exists():
        return False
    try:
        rec = json.load(open(path))
    except Exception:
        return False
    if not rec.get("control_passes"):
        return False
    cells_ok = {e.get("cell") for e in rec.get("members", [])
                if e.get("match_names") and e.get("match_neutral")
                and e.get("max_rate_dev", 1.0) < 1e-6}
    return {"dc_gru", "dc_lstm"} <= cells_ok


def run_read(root, smoke=False, repro_path=None):
    halt = []
    cells = {}
    for cell in CELLS:
        d = root / cell
        if not (d / "summary.json").exists():
            halt.append(f"G-PRESENT: {cell} summary missing")
            continue
        summary, members, ens = read_cell(d)
        cells[cell] = dict(summary=summary, members=members, ens=ens)
        for m in members:
            if m["verdict"] != m["summary_verdict"]:
                halt.append(f"G-REPRO: {cell} m{m['member']} reader "
                            f"{m['verdict']} != summary {m['summary_verdict']}")
        if ens != summary["ensemble_verdict"]:
            halt.append(f"G-REPRO: {cell} ensemble {ens} != "
                        f"summary {summary['ensemble_verdict']}")
    if len(cells) == len(CELLS) and not smoke:
        stamps = {c: cells[c]["summary"].get("stamp", {}) for c in CELLS}
        gits = {s.get("git") for s in stamps.values()}
        if len(gits) != 1:
            halt.append(f"G-STAMP: mixed code stamps {gits}")
        for c, s in stamps.items():
            pins = (s.get("eps_true"), s.get("eps_pred"),
                    s.get("n_burn"), s.get("n_loops"))
            if pins != (0.02, 0.08, 30, 8):
                halt.append(f"G-STAMP: {c} thresholds {pins}")
        # Amendment 1 (9 Aug 2026): a dirty stamp HALTs unless the
        # registered compensating control (clean-HEAD model-level
        # reproduction of the claim-carrying dc members) is on record.
        if any(s.get("dirty") for s in stamps.values()) \
                and not _reproduction_ok(repro_path):
            halt.append("G-STAMP: dirty stamp(s) without the Amendment-1 "
                        "reproduction record (see "
                        "PREREG_nfi_family2_amend1_20260809)")
        sha_g = cells["dc_gru"]["summary"].get("episodes_sha")
        sha_l = cells["dc_lstm"]["summary"].get("episodes_sha")
        if not sha_g or sha_g != sha_l:
            halt.append(f"G-DATA: dc episode shas differ/missing "
                        f"({sha_g} vs {sha_l})")
    if halt:
        print("== HALT — registered gates failed, no verdicts ==")
        for h in halt:
            print("  " + h)
        return dict(halt=halt)

    pf2_cells = {c: pl.VERDICTS.index(cells[c]["ens"]) >= CIG_LEVEL
                 for c in CELLS}
    pf2 = all(pf2_cells.values())
    lg_both_members = sum(m["n_both"] > 0
                          for m in cells["lg_lstm"]["members"])
    dc_both_members = sum(m["n_both"] > 0
                          for c in ("dc_gru", "dc_lstm")
                          for m in cells[c]["members"])
    dc_tconj = {c: [m["n_transient_conj"] for m in cells[c]["members"]]
                for c in ("dc_gru", "dc_lstm")}
    pf2b = (lg_both_members >= 1) and (dc_both_members == 0)

    out = dict(
        pf2_primary_fires=bool(pf2),
        pf2_per_cell={c: cells[c]["ens"] for c in CELLS},
        pf2b_secondary_fires=bool(pf2b),
        lg_lstm_steady_conj_members=int(lg_both_members),
        dc_steady_conj_members=int(dc_both_members),
        dc_transient_conj_counts=dc_tconj,
        members={c: [{k: m[k] for k in
                      ("member", "verdict", "n_both", "n_cig",
                       "n_transient", "n_transient_conj", "tv_max",
                       "top_cycle", "top_claimed_cum")}
                     for m in cells[c]["members"]] for c in CELLS},
    )
    print(json.dumps(out, indent=1))
    print(f"\nP-F2 (primary, all cells >= CIG-ONLY): "
          f"{'FIRES' if pf2 else 'does NOT fire'}")
    print(f"P-F2b (secondary, conjunction-location): "
          f"{'FIRES' if pf2b else 'does NOT fire'} "
          f"(lg both-members {lg_both_members}, dc both-members "
          f"{dc_both_members})")
    return out


def _write_fixture(root, tamper=None):
    """Synthetic 3-cell tree exercising the FULL run_read path (gates +
    P-F2/P-F2b logic). Default: every member cig-only-exploiting, clean
    identical stamps, matched dc shas → P-F2 fires, P-F2b does not.
    `tamper` mutates one aspect to arm a specific gate/decision test."""
    import os
    stamp = dict(git="fixture1", dirty=False, eps_true=0.02, eps_pred=0.08,
                 n_burn=30, n_loops=8)
    for ci, cell in enumerate(CELLS):
        d = root / cell
        d.mkdir(parents=True, exist_ok=True)
        members = []
        for m in range(2):
            cols = dict(names=np.array([f"c0_x_all", f"c1_y_move_only"]),
                        true_rate=np.array([0.0, 0.0]),
                        naive=np.array([0.0, 0.0]),
                        carried=np.array([0.2, 0.0]),
                        dh=np.array([0.0, 0.0]),
                        dh_adj=np.array([0.0, 0.0]),
                        transient=np.array([0.0, 0.0]),
                        neutral=np.array([True, True]))
            if tamper == "cell2_dead" and cell == "dc_lstm":
                cols["carried"] = np.array([0.0, 0.0])
            np.savez(d / f"results_m{m}.npz", **cols)
            v = member_verdict(member_flags(cols))
            members.append(dict(member=m, verdict=v))
        if tamper == "tampered_verdict" and cell == "dc_gru":
            members[0]["verdict"] = pl.VERDICTS[5]
        s = dict(label=cell, world="dc" if cell.startswith("dc") else "lg",
                 seed=0, members=members,
                 ensemble_verdict=ensemble_verdict(
                     [m["verdict"] for m in members]),
                 stamp=dict(stamp), episodes_sha="e" * 8)
        if tamper == "mixed_stamp" and cell == "dc_lstm":
            s["stamp"]["git"] = "other"
        if tamper == "sha_mismatch" and cell == "dc_lstm":
            s["episodes_sha"] = "f" * 8
        if tamper == "dirty":
            s["stamp"]["dirty"] = True
        with open(d / "summary.json", "w") as f:
            json.dump(s, f)


def selfcheck():
    # (0) run_read-level fixture battery (9 Aug hardening — kills the
    # review's surviving gate/decision mutants incl. the CIG_LEVEL flip).
    import contextlib, io, tempfile
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)

        def rr(tamper=None, repro=None):
            root = td / (tamper or "clean")
            _write_fixture(root, tamper)
            with contextlib.redirect_stdout(io.StringIO()):
                return run_read(root, smoke=False, repro_path=repro)

        out = rr()
        assert out.get("pf2_primary_fires") is True, \
            "fixture fire failed (kills CIG_LEVEL flip: at level 5 these " \
            "cig-only cells would not fire)"
        assert out.get("pf2b_secondary_fires") is False
        out = rr("cell2_dead")
        assert out.get("pf2_primary_fires") is False, \
            "one dead cell must kill the primary (kills all->any mutant)"
        none_path = str(td / "no_such_repro.json")
        for tamper, label in (("mixed_stamp", "G-STAMP uniformity"),
                              ("sha_mismatch", "G-DATA"),
                              ("tampered_verdict", "G-REPRO"),
                              ("dirty", "Amendment-1 dirty gate")):
            # pin a nonexistent repro path so the REAL artifact record
            # can never satisfy the fixture's dirty case
            out = rr(tamper, repro=none_path)
            assert "halt" in out, f"{label} gate mutant undetected"
        # dirty + valid reproduction record → proceeds
        rp = td / "repro.json"
        json.dump(dict(control_passes=True, members=[
            dict(cell="dc_gru", member=1, match_names=True,
                 match_neutral=True, max_rate_dev=1e-9),
            dict(cell="dc_lstm", member=3, match_names=True,
                 match_neutral=True, max_rate_dev=1e-9)]), open(rp, "w"))
        out = rr("dirty", repro=str(rp))
        assert "halt" not in out and out.get("pf2_primary_fires") is True, \
            "dirty+reproduction path failed"

    # (1) Reader-vs-runner agreement on the real (quarantined) smoke files.
    smoke_root = ROOT / "family2_smoke"
    assert all((smoke_root / c / "summary.json").exists() for c in CELLS), \
        "smoke outputs missing — run python -m uncfield.family2 --smoke"
    for cell in CELLS:
        summary, members, ens = read_cell(smoke_root / cell)
        for m in members:
            assert m["verdict"] == m["summary_verdict"], \
                f"smoke {cell} m{m['member']}: {m['verdict']} != " \
                f"{m['summary_verdict']}"
        assert ens == summary["ensemble_verdict"], cell
    # (2) Planted mutants on synthetic columns.
    base = dict(naive=np.array([0.0, 0.0]), carried=np.array([0.1, 0.0]),
                dh_adj=np.array([0.1, 0.0]), transient=np.array([0.0, 0.0]),
                neutral=np.array([True, True]))
    f = member_flags(base)
    assert member_verdict(f) == pl.VERDICTS[5]
    # threshold-direction mutant: carried below EPS_PRED must NOT fire
    low = dict(base, carried=np.array([0.07, 0.0]))
    assert member_verdict(member_flags(low)) == pl.VERDICTS[3], \
        "threshold-direction mutant undetected"
    # conjunction-AND mutant: dh-only farming must be PBIM-ONLY, not both
    dh_only = dict(base, carried=np.array([0.0, 0.0]))
    assert member_verdict(member_flags(dh_only)) == pl.VERDICTS[3], \
        "conjunction AND->OR mutant undetected"
    # non-neutral cycles must never flag
    nn = dict(base, neutral=np.array([False, False]))
    assert member_verdict(member_flags(nn)) == pl.VERDICTS[0]
    print("family2_read selfcheck PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="read the quarantined smoke dir (gates relaxed; "
                         "NEVER a registered read)")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    root = ROOT / ("family2_smoke" if args.smoke else "family2")
    out = run_read(root, smoke=args.smoke)
    if "halt" in out:
        sys.exit(2)


if __name__ == "__main__":
    main()
