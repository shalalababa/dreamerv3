"""MIN-CLIP re-scoring over the frozen sweep+pilot model bank
(PREREG_nfi_minclip_20260821.md; Paper-5 constructive section).

The candidate repair from the algorithm-ideation panel: credit each
cycle with min(operator's promise, heads' Bayes license) instead of
the operator's carried EIG alone. The heads' license for a sense is
mechanism.head_accounts' bayes_gain (0.5 log(tot_var / r_implied));
per-loop license = mean finite bayes gain x senses/loop
(mechanism.summarize_trace fields). Heads-inconsistent senses leave
the license unbounded (clip keeps the promise) — an UNTRIGGERED edge
case in the anchor (frac_r_implied_negative == 0.0 on every traced
mechanism cycle, #29 M6). The pre-computed m3 residual is NOT that
path: m3's heads license a LARGE FINITE gain (0.29/0.42 per loop —
head-side variance re-inflation raising a finite bound).
WINDOW DISCLOSURE (#29 M8): the license uses the mechanism-artifact
convention (trace n_loops=4; summarize over the last 2 loops =
post-burn loops 33-34), matching the P-MC2 pre-computation; the
promise (steady_rate) averages post-burn loops 35-38. The two sides
of min() therefore come from adjacent, non-overlapping windows —
kept deliberately for P-MC2 comparability, registered here.

Per task (slurm array 0..N_TASKS-1 = ratio_research.JOBCONFIGS x 4
members): re-run the job's exact exploit search (asserted to
REPRODUCE the frozen ratio_research table), TRACE EVERY CYCLE
(n_loops=4, the mechanism convention), compute clipped rates and the
clipped tier boolean (the full classify() verdict is recorded
pre-clip; the clip changes only the carried-EIG channel), and record
the legitimate-sensing retention and rank
fidelity inputs + the LOO cross-member rows. A separate --collate
pass (the frozen read) aggregates the 44 tasks against the frozen
predictions.

Run:  python -m uncfield.minclip_rescore --task N       (RCC array)
      python -m uncfield.minclip_rescore --collate --output <dir>
      python -m uncfield.minclip_rescore --selfcheck
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import learnedwm as lw
from . import mechanism as mech
from . import planner as pl
from .ratio_research import JOBCONFIGS, OUT, _san

ROOT = OUT / "minclip"
N_TASKS = len(JOBCONFIGS) * 4
EPS_PRED = pl.EPS_PRED
RETENTION_BAR = 0.8         # frozen guard (P-MC3)
PINNED_TIER_DISTINCT = 29   # frozen ratio_research tables (#29 M11)


def licensed_rate(model, b0, node0, cycle, y_mode, seed):
    """Per-loop heads-licensed gain for one cycle (mechanism trace).
    Returns (rate, n_inconsistent_senses, n_senses). rate = inf when
    senses exist but NO head yields a finite license (the heads refuse
    to bound the promise); 0.0 when the cycle has no senses."""
    b, node = b0.copy(), node0
    for mv in pl._route_to(node0, cycle["start"]):
        b = model.step(b, mv, node, None)
        node = mv
    loops = mech.trace_cycle(model, b, node, cycle, n_loops=4,
                             y_mode=y_mode, seed=seed)
    s = mech.summarize_trace(loops)
    n_senses = s.get("n_senses_per_loop") or 0
    if not n_senses:
        return 0.0, 0.0, 0
    bg = s.get("mean_bayes_gain")
    frac_inc = float(s.get("frac_r_implied_negative", 0.0) or 0.0)
    # #29 M7: the prereg's per-sense semantics — ANY inconsistent
    # sense leaves the loop's license unbounded (empirically vacuous
    # in the anchor: frac == 0.0 on every traced mechanism cycle,
    # #29 M6; note mechanism.py uses r_implied <= 0 for the frac but
    # > 1e-12 for the inf branch — the 0 < r <= 1e-12 sliver rides
    # the inf path here, disclosed).
    if bg is None or frac_inc > 0.0:
        return float("inf"), frac_inc, n_senses
    return float(bg) * float(n_senses), frac_inc, n_senses


def clip(carried, licensed):
    """min(promise, license); inf license keeps the promise."""
    return min(float(carried), float(licensed))


def run_task(task):
    job, ens_dir, y_mode, seed = JOBCONFIGS[task // 4]
    member = task % 4
    ensemble = lw.load_ensemble(OUT / ens_dir / "ensemble.pkl")
    model = lw.LearnedModel(ensemble[member])
    results, (b0, _, node0) = pl.search_exploits(model, seed=seed,
                                                 max_len=6,
                                                 y_mode=y_mode)
    verdict, counts = pl.classify(results)
    # #29 M11: the search must REPRODUCE the frozen ratio_research
    # table for this (job, member) — the P-MC1 denominator is pinned
    # on those tables, and silent drift would move it
    frozen = OUT / "ratio_research" / f"{job}_m{member}.npz"
    if frozen.exists():
        fz = np.load(frozen, allow_pickle=False)
        by_name = {r["cycle"]["name"]:
                   (bool(r["neutral"]), float(r["carried_eig_rate"]))
                   for r in results}
        for nm, ne, ce in zip(fz["names"], fz["neutral"],
                              fz["carried_eig"]):
            got = by_name.get(str(nm))
            assert got is not None and got[0] == bool(ne) and                 abs(got[1] - float(ce)) < 1e-9, (
                f"{tag_ := f'{job}_m{member}'}: search does not "
                f"reproduce the frozen ratio_research table at "
                f"{nm} — P-MC1 denominator would drift")
    rows = []
    for r in results:
        lic, frac_inc, n_sen = licensed_rate(model, b0, node0,
                                             r["cycle"], y_mode, seed)
        carried = float(r["carried_eig_rate"])
        rows.append(dict(
            name=r["cycle"]["name"], neutral=bool(r["neutral"]),
            true_rate=float(r["true_rate"]), carried=carried,
            licensed=lic, clipped=clip(carried, lic),
            frac_inconsistent=frac_inc, n_senses=n_sen,
            exploit_carried=bool(r["exploit_carried_eig"]),
            exploit_clipped=bool(r["neutral"]
                                 and clip(carried, lic) > EPS_PRED)))
    member_tier_carried = any(r["exploit_carried"] for r in rows)
    member_tier_clipped = any(r["exploit_clipped"] for r in rows)
    # #29 M9: the clip touches ONLY the carried-EIG channel; the
    # realized-dH (carried_dh) channel is untouched by construction —
    # record it so the headline cannot overreach
    tier_dh = any(r_["exploit_carried_dh"] for r_ in results)
    tier_both_carried = any(r_["exploit_both"] for r_ in results)
    # legitimate-sensing retention (P-MC3): the top genuinely
    # informative cycle by true rate
    # #29 B1/B2: retention is defined ONLY where credit is at stake —
    # informative cycles with carried > EPS_PRED (the planner declares
    # carried_eig non-comparable on dynamic sensors; 23/44 frozen
    # tables have carried <= 0 on the top-true cycle, where the clip
    # is a NO-OP and the naive ratio is garbage). No eligible cycle =>
    # explicit ABSTAIN, never a vacuous pass.
    inform = [r for r in rows if (not r["neutral"])
              and r["true_rate"] > 0.1 and r["carried"] > EPS_PRED]
    if inform:
        top = max(inform, key=lambda r: r["true_rate"])
        retention = dict(status="MEASURED", cycle=top["name"],
                         carried=top["carried"],
                         clipped=top["clipped"],
                         ratio=top["clipped"] / top["carried"],
                         ok=bool(top["clipped"]
                                 >= RETENTION_BAR * top["carried"]))
    else:
        retention = dict(status="ABSTAIN",
                         note="no informative cycle with carried > "
                              "EPS_PRED — clip never touches this "
                              "model's real sensing credit")
    # rank fidelity inputs (P-MC4)
    from scipy.stats import spearmanr
    true = np.array([r["true_rate"] for r in rows])
    car = np.array([r["carried"] for r in rows])
    cl = np.array([r["clipped"] for r in rows])   # finite: clipped <= carried
    fid = dict(rho_carried=float(spearmanr(car, true).statistic),
               rho_clipped=float(spearmanr(cl, true).statistic))
    # LOO rows (DESCRIPTIVE ONLY, #29 m21: selection is by the
    # member's own clipped max — own-is-the-max selection bias is
    # built in, and own credit is clipped while others' is carried;
    # usable as defense-battery input, never as an estimate)
    loo = None
    ex = [r for r in rows if r["exploit_clipped"]]
    if ex:
        topex = max(ex, key=lambda r: r["clipped"])
        cyc = next(r["cycle"] for r in results
                   if r["cycle"]["name"] == topex["name"])
        others = []
        for om in range(4):
            if om == member:
                continue
            omodel = lw.LearnedModel(ensemble[om])
            ob, _, onode = pl.warmup_state(omodel, seed=seed)
            b, node = ob.copy(), onode
            for mv in pl._route_to(onode, cyc["start"]):
                b = omodel.step(b, mv, node, None)
                node = mv
            orates, _, _ = pl.score_cycle(omodel, b, node, cyc,
                                          y_mode=y_mode, seed=seed)
            others.append(dict(member=om,
                               carried=float(pl.steady_rate(
                                   orates["carried_eig"]))))
        loo = dict(cycle=topex["name"], own=topex["carried"],
                   others=others)
    from .sweeps import _code_stamp
    rec = dict(job=job, member=member, y_mode=y_mode, seed=seed,
               stamp=_code_stamp(), verdict=verdict,
               n_cycles=len(rows),
               tier_carried=member_tier_carried,
               tier_clipped=member_tier_clipped,
               tier_dh_untouched=tier_dh,
               tier_both_carried=tier_both_carried,
               n_exploit_carried=sum(r["exploit_carried"]
                                     for r in rows),
               n_exploit_clipped=sum(r["exploit_clipped"]
                                     for r in rows),
               retention=retention, fidelity=fid, loo=loo)
    ROOT.mkdir(parents=True, exist_ok=True)
    tag = f"{job}_m{member}"
    with open(ROOT / f"{tag}.json", "w") as f:
        json.dump(_san(rec), f, indent=1)
    np.savez_compressed(
        ROOT / f"{tag}.npz",
        name=np.array([r["name"] for r in rows]),
        neutral=np.array([r["neutral"] for r in rows]),
        true_rate=np.array([r["true_rate"] for r in rows]),
        carried=np.array([r["carried"] for r in rows]),
        licensed=np.array([r["licensed"] for r in rows]),
        clipped=np.array([r["clipped"] for r in rows]),
        frac_inconsistent=np.array([r["frac_inconsistent"]
                                    for r in rows]),
        n_senses=np.array([r["n_senses"] for r in rows]))
    print(f"{tag}: tier carried={member_tier_carried} "
          f"clipped={member_tier_clipped} "
          f"exploits {rec['n_exploit_carried']}->"
          f"{rec['n_exploit_clipped']}", flush=True)
    return rec


def collate(output):
    """The frozen read over all task jsons (P-MC1..P-MC4), #29 form.
    PRIMARY denominator = DISTINCT MODELS (#29 B3): the three ymode_s*
    jobs re-score the pilot2 ensemble, so the bank is 8 ensembles x 4
    = 32 distinct models ("32 of the paper's 60"); ymode scorings are
    reported separately as view sensitivity. The distinct-model
    exploit-tier denominator is PINNED from the frozen ratio_research
    tables: 29 of 32 (#29 M11); collate asserts the observed count
    matches (drift gate)."""
    files = sorted(ROOT.glob("*_m*.json"))
    recs = [json.load(open(f)) for f in files]
    assert len(recs) == N_TASKS, (
        f"{len(recs)}/{N_TASKS} tasks present — collate only on the "
        f"complete bank")
    assert not str(output).endswith("TBD"), (
        "registered read must persist: pass an explicit --output")
    VIEW_JOBS = {"ymode_s0", "ymode_s1", "ymode_s2"}
    distinct = [r for r in recs if r["job"] not in VIEW_JOBS]
    views = [r for r in recs if r["job"] in VIEW_JOBS]
    assert len(distinct) == 32 and len(views) == 12

    tier_c = [r for r in distinct if r["tier_carried"]]
    assert len(tier_c) == PINNED_TIER_DISTINCT, (
        f"distinct-model exploit-tier count {len(tier_c)} != pinned "
        f"{PINNED_TIER_DISTINCT} (frozen ratio_research tables) — "
        f"denominator drift")
    lost = [r for r in tier_c if not r["tier_clipped"]]
    ret_meas = [r for r in distinct
                if (r["retention"] or {}).get("status") == "MEASURED"]
    ret_abst = [r for r in distinct
                if (r["retention"] or {}).get("status") == "ABSTAIN"]
    fid_pairs = [(r["fidelity"]["rho_carried"],
                  r["fidelity"]["rho_clipped"]) for r in distinct
                 if np.isfinite(r["fidelity"]["rho_carried"] or np.nan)
                 and np.isfinite(r["fidelity"]["rho_clipped"]
                                 or np.nan)]
    out = dict(
        n_scorings=len(recs), n_distinct=len(distinct),
        n_views=len(views),
        scope="32 of the paper's 60 distinct models; out = "
              "dataseed3/4 + hid128t10k/t30k (16, JOBCONFIGS "
              "extension pending) + family-2 (12, world=dcfield)",
        tier_carried_distinct=len(tier_c),
        tier_clipped_distinct=sum(r["tier_clipped"] for r in tier_c),
        tier_dh_untouched_distinct=sum(r["tier_dh_untouched"]
                                       for r in distinct),
        collapse_fraction=(len(lost) / max(len(tier_c), 1)),
        pmc1=dict(statement=f"P-MC1: >= 50% of the "
                            f"{PINNED_TIER_DISTINCT} distinct-model "
                            f"carried-exploit-tier members lose tier "
                            f"membership under the clip",
                  lost=len(lost),
                  fires=bool(len(lost) * 2 >= PINNED_TIER_DISTINCT)),
        pmc2=dict(statement="P-MC2 (PARTIALLY pre-computed — 3 traced "
                            "cycles/member; the full-bank quantifier "
                            "is a prediction, #29 M10): pilot2 m3 "
                            "retains clipped exploits; m0/m1/m2 do "
                            "not",
                  observed={f"pilot2_m{m}":
                            next((r["tier_clipped"] for r in distinct
                                  if r["job"] == "pilot2"
                                  and r["member"] == m), None)
                            for m in range(4)}),
        pmc3=dict(statement="P-MC3 guard: retention >= 0.8 on every "
                            "MEASURED model (informative AND carried "
                            "> EPS_PRED); ABSTAIN = clip never "
                            "touches that model's real credit",
                  n_measured=len(ret_meas), n_abstain=len(ret_abst),
                  fires=(bool(all(r["retention"]["ok"]
                                  for r in ret_meas))
                         if ret_meas else "NOT-ADJUDICABLE "
                         "(all models abstain)")),
        pmc4=dict(
            n_pairs=len(fid_pairs),
            mean_rho_carried=(float(np.mean([a for a, _ in fid_pairs]))
                              if fid_pairs else None),
            mean_rho_clipped=(float(np.mean([b for _, b in fid_pairs]))
                              if fid_pairs else None),
            n_improved=sum(b > a for a, b in fid_pairs),
            note="exploratory-registered; nan-safe (#29 m16)"),
        view_sensitivity=[{k: r[k] for k in
                           ("job", "member", "tier_carried",
                            "tier_clipped")} for r in views],
        retention_rows=[dict(job=r["job"], member=r["member"],
                             **(r["retention"] or {}))
                        for r in distinct],
        loo=[r["loo"] for r in distinct if r["loo"]],
        per_model=[{k: r[k] for k in
                    ("job", "member", "verdict", "tier_carried",
                     "tier_clipped", "tier_dh_untouched",
                     "n_exploit_carried", "n_exploit_clipped")}
                   for r in recs])
    p2 = out["pmc2"]["observed"]
    out["pmc2"]["fires"] = bool(p2.get("pilot2_m3") is True
                                and not any(p2.get(f"pilot2_m{m}")
                                            for m in range(3)))
    od = pathlib.Path(output)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "minclip_read.json", "w") as f:
        json.dump(_san(out), f, indent=1)
    print(f"MIN-CLIP (distinct): tier {out['tier_carried_distinct']} "
          f"-> {out['tier_clipped_distinct']} (collapse "
          f"{out['collapse_fraction']:.2f}); dh-tier untouched "
          f"{out['tier_dh_untouched_distinct']}; P-MC1 "
          f"{out['pmc1']['fires']}; P-MC2 {out['pmc2']['fires']}; "
          f"P-MC3 {out['pmc3']['fires']} "
          f"(measured {out['pmc3']['n_measured']}, abstain "
          f"{out['pmc3']['n_abstain']})")
    return out


# ---------------------------------------------------------------- selfcheck

ORACLES = [
    # (member, planner cycle name, expected licensed/loop) — exact
    # values from artifacts/nfi_mechanism_20260802/mechanism.json
    (0, "c554_2-3-2-3-2-3-2_only_s7_z2_n3", 0.06279273395121125),
    (3, "c184_0-1-4-3-4-5-0_only_s7_z2_n3", 0.28966619047371134),
    (3, "c288_0-5-4-3-4-1-0_only_s7_z2_n3", 0.420848596601374),
]


def _fake_rec(job, member, tier_c, tier_cl, retention="MEASURED",
              ret_ok=True, rho_c=0.1, rho_cl=0.2, tier_dh=True):
    ret = (dict(status="MEASURED", cycle="x", carried=1.0,
                clipped=1.0 if ret_ok else 0.1,
                ratio=1.0 if ret_ok else 0.1, ok=ret_ok)
           if retention == "MEASURED" else
           dict(status="ABSTAIN", note="n/a"))
    return dict(job=job, member=member, y_mode="ml", seed=0,
                stamp="fixture", verdict="V", n_cycles=1,
                tier_carried=tier_c, tier_clipped=tier_cl,
                tier_dh_untouched=tier_dh, tier_both_carried=False,
                n_exploit_carried=int(tier_c),
                n_exploit_clipped=int(tier_cl),
                retention=ret,
                fidelity=dict(rho_carried=rho_c, rho_clipped=rho_cl),
                loo=None)


def _write_bank(tmpdir, lost_count, all_abstain=False,
                tier_total=None):
    """44 fixture jsons matching the frozen tier layout: distinct
    tier members = 29 (misses hid32_m2/train10k_m2/train30k_m2)."""
    import os
    misses = {("hid32", 2), ("train10k", 2), ("train30k", 2)}
    VIEW = ("ymode_s0", "ymode_s1", "ymode_s2")
    # explicit loss plan: pilot2 m0/m1/m2 always lose (the P-MC2
    # pattern, 3 losses); the remaining (lost_count-3) losses are the
    # first non-pilot2 distinct tier members in JOBCONFIGS order
    extra = max(lost_count - 3, 0)
    n_extra = 0
    n_tier_seen = 0
    for job, ens, ym, sd in JOBCONFIGS:
        for mem in range(4):
            tier_c = (job, mem) not in misses
            if tier_total is not None and job not in VIEW:
                if tier_c and n_tier_seen >= tier_total:
                    tier_c = False
                n_tier_seen += tier_c
            if job == "pilot2":
                clipped = tier_c and (mem == 3)
            elif tier_c and job not in VIEW and n_extra < extra:
                clipped = False
                n_extra += 1
            else:
                clipped = tier_c
            rec = _fake_rec(
                job, mem, tier_c, clipped,
                retention=("ABSTAIN" if all_abstain else "MEASURED"),
                rho_c=(float("nan") if (job, mem) == ("pilot2", 1)
                       else 0.1))
            with open(os.path.join(tmpdir, f"{job}_m{mem}.json"),
                      "w") as f:
                json.dump(_san(rec), f)


def selfcheck():
    import tempfile
    global ROOT
    # clip semantics
    assert clip(0.4, 0.05) == 0.05
    assert clip(0.4, float("inf")) == 0.4
    assert clip(0.05, 0.4) == 0.05

    # ORACLE (#29 M13): licensed_rate must reproduce the mechanism
    # artifact exactly (same trace convention, same seed/y_mode)
    ensemble = lw.load_ensemble(OUT / "pilot2" / "ensemble.pkl")
    cycles = {c["name"]: c for c in pl.enumerate_cycles(max_len=6)}
    for mem, cname, want in ORACLES:
        model = lw.LearnedModel(ensemble[mem])
        b0, _, node0 = pl.warmup_state(model, seed=0)
        lic, frac, n = licensed_rate(model, b0, node0, cycles[cname],
                                     "ml", 0)
        assert abs(lic - want) < 1e-9, (mem, cname, lic, want)
        assert frac == 0.0
    # determinism
    model = lw.LearnedModel(ensemble[0])
    b0, _, node0 = pl.warmup_state(model, seed=0)
    c = cycles[ORACLES[0][1]]
    assert licensed_rate(model, b0, node0, c, "ml", 0)[0] ==         licensed_rate(model, b0, node0, c, "ml", 0)[0]

    # collate fixtures (#29 M12) on a temp ROOT
    old_root = ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            ROOT = pathlib.Path(td)
            _write_bank(td, lost_count=15)
            out = collate(td + "/read")
            assert out["tier_carried_distinct"] == 29
            assert out["pmc1"]["fires"] is True          # 15/29 >= 50%
            assert out["pmc2"]["fires"] is True
            assert out["pmc3"]["fires"] in (True, False)
            assert out["pmc4"]["n_pairs"] == 31          # nan dropped
            _write_bank(td, lost_count=14)
            out = collate(td + "/read")
            assert out["pmc1"]["fires"] is False         # 14/29 < 50%
            _write_bank(td, lost_count=15, all_abstain=True)
            out = collate(td + "/read")
            assert out["pmc3"]["fires"] ==                 "NOT-ADJUDICABLE (all models abstain)"
            _write_bank(td, lost_count=5, tier_total=28)
            try:
                collate(td + "/read")
                raise SystemExit("denominator drift gate FAILED")
            except AssertionError as e:
                assert "denominator drift" in str(e)
            _write_bank(td, lost_count=15)
            try:
                collate("artifacts/minclip_read_TBD")
                raise SystemExit("TBD output gate FAILED")
            except AssertionError as e:
                assert "explicit --output" in str(e)
    finally:
        ROOT = old_root

    print("minclip selfcheck PASS (clip semantics; 3 mechanism-oracle "
          "licensed rates exact to 1e-9 incl. the m3 large-finite "
          "residual; determinism; collate fixtures: pmc1 fire/no-fire "
          "at 15-vs-14 of 29, pmc2 pattern, all-abstain "
          "NOT-ADJUDICABLE, nan-fidelity drop, denominator-drift + "
          "TBD-output gates)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=-1)
    ap.add_argument("--collate", action="store_true")
    ap.add_argument("--output",
                    default="artifacts/minclip_read_TBD")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
    elif args.collate:
        collate(args.output)
    else:
        assert 0 <= args.task < N_TASKS, args.task
        run_task(args.task)


if __name__ == "__main__":
    main()
