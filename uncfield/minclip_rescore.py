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

# ---- registered EXTENSION bank (PREREG_nfi_minclip_ext_20260821.md):
# dataseed3/4 + the hid128 x t10k/t30k interaction arm — the 16
# distinct models the main prereg's scope accounting listed as
# "JOBCONFIGS extension pending". Scored with the IDENTICAL pipeline
# (licensed_rate/clip/retention/fidelity), persisted to a PARALLEL
# root, and read by a PARALLEL collate — the frozen 44-task bank, its
# 29-of-32 pinned denominator, and the registered P-MC1..P-MC4 are
# never touched. These jobs have no frozen ratio_research tables; the
# drift gate is the frozen sweeps/<job>/summary.json per-member counts
# (n_cycles, n_neutral, n_exploit_cig, verdict) instead.
JOBCONFIGS_EXT = [
    ("dataseed3",  "sweeps/dataseed3",  "ml", 3),
    ("dataseed4",  "sweeps/dataseed4",  "ml", 4),
    ("hid128t10k", "sweeps/hid128t10k", "ml", 0),
    ("hid128t30k", "sweeps/hid128t30k", "ml", 0),
]
N_TASKS_EXT = len(JOBCONFIGS_EXT) * 4
EXT_ROOT = OUT / "minclip_ext"
PINNED_TIER_DISTINCT_EXT = 15   # frozen sweeps summaries: tier iff
                                # n_exploit_cig >= 1 -> 15 of 16
                                # (miss: hid128t30k_m1)


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


# ---------------------------------------------------- EXTENSION bank

def run_task_ext(task):
    """One extension-bank member (PREREG_nfi_minclip_ext_20260821.md).
    Identical scoring pipeline to run_task; differs ONLY in the job
    table, the output root, and the drift gate (frozen sweeps
    summary.json counts — these jobs have no ratio_research tables).
    Deliberately NOT refactored into run_task: the main path is the
    reviewed frozen instrument and must not be perturbed."""
    job, ens_dir, y_mode, seed = JOBCONFIGS_EXT[task // 4]
    member = task % 4
    ensemble = lw.load_ensemble(OUT / ens_dir / "ensemble.pkl")
    model = lw.LearnedModel(ensemble[member])
    results, (b0, _, node0) = pl.search_exploits(model, seed=seed,
                                                 max_len=6,
                                                 y_mode=y_mode)
    verdict, counts = pl.classify(results)
    # drift gate (#31 B1 form): the search must REPRODUCE the frozen
    # sweep summary for this (job, member) across EVERY
    # model-dependent frozen field — the exploit-count family, the
    # verdict, the top-cycle identity, and the top-cycle/drift floats
    # to 1e-9. (n_cycles/n_neutral are model-independent constants of
    # the harness — kept as sanity, not billed as discrimination.)
    # top_self_consistency is NOT gated (it needs an extra
    # _self_consistency trace; disclosed in the prereg).
    with open(OUT / ens_dir / "summary.json") as f:
        summ = json.load(f)
    frozen_m = next((m for m in summ["members"]
                     if int(m["member"]) == member), None)
    assert frozen_m is not None, (
        f"{job}_m{member}: member missing from frozen summary")
    top = results[0]
    got = dict(verdict=verdict,
               **{k: counts[k] for k in
                  ("n_cycles", "n_neutral", "n_exploit_naive",
                   "n_exploit_cig", "n_exploit_pbim",
                   "n_exploit_both", "n_exploit_transient")},
               top_cycle=top["cycle"]["name"])
    for k, v in got.items():
        assert v == frozen_m[k], (
            f"{job}_m{member}: search does not reproduce the frozen "
            f"sweep summary at {k} ({v} != {frozen_m[k]}) — the "
            f"extension tier pin would drift")
    # float fields gated at 1e-6 ABSOLUTE (cross-host float
    # reproducibility limit, MEASURED 21 Aug on hid128t30k_m1
    # local-vs-frozen-cluster: top_dh_adj delta 8e-8,
    # drift_per_action delta 1.6e-7 — a 1e-9 gate would have burned
    # the wave); the identity/count fields above stay exact
    for k, v in (("top_true", float(top["true_rate"])),
                 ("top_eig", float(top["carried_eig_rate"])),
                 ("top_dh_adj", float(top["carried_dh_adj"])),
                 ("drift_per_action",
                  float(counts["drift_per_action"]))):
        fv = frozen_m[k]
        assert fv is not None and \
            abs(v - float(fv)) < max(1e-6, 1e-6 * abs(float(fv))), (
            f"{job}_m{member}: {k} {v} != frozen {fv} — numerical "
            f"drift beyond the measured cross-host envelope")
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
    # redundant with the gate above by construction; kept as a
    # tripwire on the rows-loop recomputation itself (#31 m13)
    assert member_tier_carried == (frozen_m["n_exploit_cig"] > 0), (
        f"{job}_m{member}: rows-loop tier bit disagrees with the "
        f"gated search counts — internal inconsistency")
    tier_dh = any(r_["exploit_carried_dh"] for r_ in results)
    tier_both_carried = any(r_["exploit_both"] for r_ in results)
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
    from scipy.stats import spearmanr
    true = np.array([r["true_rate"] for r in rows])
    car = np.array([r["carried"] for r in rows])
    cl = np.array([r["clipped"] for r in rows])
    fid = dict(rho_carried=float(spearmanr(car, true).statistic),
               rho_clipped=float(spearmanr(cl, true).statistic))
    from .sweeps import _code_stamp
    rec = dict(job=job, member=member, y_mode=y_mode, seed=seed,
               bank="EXTENSION", stamp=_code_stamp(), verdict=verdict,
               n_cycles=len(rows),
               tier_carried=member_tier_carried,
               tier_clipped=member_tier_clipped,
               tier_dh_untouched=tier_dh,
               tier_both_carried=tier_both_carried,
               n_exploit_carried=sum(r["exploit_carried"]
                                     for r in rows),
               n_exploit_clipped=sum(r["exploit_clipped"]
                                     for r in rows),
               retention=retention, fidelity=fid, loo=None)
    EXT_ROOT.mkdir(parents=True, exist_ok=True)
    tag = f"{job}_m{member}"
    with open(EXT_ROOT / f"{tag}.json", "w") as f:
        json.dump(_san(rec), f, indent=1)
    np.savez_compressed(
        EXT_ROOT / f"{tag}.npz",
        name=np.array([r["name"] for r in rows]),
        neutral=np.array([r["neutral"] for r in rows]),
        true_rate=np.array([r["true_rate"] for r in rows]),
        carried=np.array([r["carried"] for r in rows]),
        licensed=np.array([r["licensed"] for r in rows]),
        clipped=np.array([r["clipped"] for r in rows]),
        frac_inconsistent=np.array([r["frac_inconsistent"]
                                    for r in rows]),
        n_senses=np.array([r["n_senses"] for r in rows]))
    print(f"EXT {tag}: tier carried={member_tier_carried} "
          f"clipped={member_tier_clipped} "
          f"exploits {rec['n_exploit_carried']}->"
          f"{rec['n_exploit_clipped']}", flush=True)
    return rec


INTERACTION_CELLS = [
    # (hid, steps, job, bank) — pinned pre-outcome (#31 M5)
    (64, "3k", "pilot2", "main"), (64, "10k", "train10k", "main"),
    (64, "30k", "train30k", "main"), (128, "3k", "hid128", "main"),
    (128, "10k", "hid128t10k", "ext"),
    (128, "30k", "hid128t30k", "ext"),
]


def _interaction_table(ext_recs):
    """Pinned 2x3 capacity x training descriptive table: per cell,
    members going tier_carried -> NOT tier_clipped, of 4. Main-bank
    cells read from ROOT task jsons when present (else PENDING)."""
    by_job = {}
    for r in ext_recs:
        by_job.setdefault(r["job"], []).append(r)
    rows = []
    for hid, steps, job, bank in INTERACTION_CELLS:
        if bank == "ext":
            ms = by_job.get(job, [])
        else:
            ms = []
            for m in range(4):
                p = ROOT / f"{job}_m{m}.json"
                if p.exists():
                    ms.append(json.load(open(p)))
        if len(ms) == 4:
            cell = dict(n_tier=sum(r["tier_carried"] for r in ms),
                        n_lost=sum(r["tier_carried"]
                                   and not r["tier_clipped"]
                                   for r in ms))
        else:
            cell = "MAIN-BANK-PENDING" if bank == "main" else \
                f"INCOMPLETE ({len(ms)}/4)"
        rows.append(dict(hid=hid, steps=steps, job=job, bank=bank,
                         cell=cell))
    return rows


def collate_ext(output):
    """The frozen EXTENSION read (PREREG_nfi_minclip_ext_20260821.md):
    P-MC1-EXT replication over the 16 extension models with the
    denominator PINNED at 15 (frozen sweeps summaries). Reported
    SEPARATELY from the 29-of-32 primary — the two P-MC1 denominators
    never mix (the pinned 2x3 interaction table is the registered
    cross-bank DESCRIPTIVE exception, #31 M5)."""
    files = sorted(EXT_ROOT.glob("*_m*.json"))
    recs = [json.load(open(f)) for f in files]
    assert len(recs) == N_TASKS_EXT, (
        f"{len(recs)}/{N_TASKS_EXT} extension tasks present — collate "
        f"only on the complete bank (any gate-failed task => the "
        f"extension read is NOT-ADJUDICABLE, prereg §Drift)")
    # #31 m15: trailing-slash-proof TBD refusal (on the path NAME)
    assert not pathlib.Path(str(output)).name.endswith("TBD"), (
        "registered read must persist: pass an explicit --output")
    assert all(r.get("bank") == "EXTENSION" for r in recs), (
        "non-extension rec in the EXT bank")
    # #31 M11: stamp homogeneity — the bank must come from ONE code
    # state (the parent collate is frozen and cannot check this;
    # disclosed in the prereg)
    stamps = sorted({json.dumps(r["stamp"], sort_keys=True)
                     for r in recs})
    assert len(stamps) == 1, (
        f"heterogeneous code stamps across the extension bank: "
        f"{stamps}")
    tier_c = [r for r in recs if r["tier_carried"]]
    assert len(tier_c) == PINNED_TIER_DISTINCT_EXT, (
        f"extension exploit-tier count {len(tier_c)} != pinned "
        f"{PINNED_TIER_DISTINCT_EXT} (frozen sweeps summaries) — "
        f"denominator drift")
    lost = [r for r in tier_c if not r["tier_clipped"]]
    ret_meas = [r for r in recs
                if (r["retention"] or {}).get("status") == "MEASURED"]
    ret_abst = [r for r in recs
                if (r["retention"] or {}).get("status") == "ABSTAIN"]
    fid_pairs = [(r["fidelity"]["rho_carried"],
                  r["fidelity"]["rho_clipped"]) for r in recs
                 if np.isfinite(r["fidelity"]["rho_carried"] or np.nan)
                 and np.isfinite(r["fidelity"]["rho_clipped"]
                                 or np.nan)]
    def _wilson(k, n, z=1.96):
        if n == 0:
            return [None, None]
        p, z2 = k / n, z * z
        den = 1 + z2 / n
        ctr = (p + z2 / (2 * n)) / den
        hw = z * ((p * (1 - p) + z2 / (4 * n)) / n) ** 0.5 / den
        return [max(0.0, ctr - hw), min(1.0, ctr + hw)]

    out = dict(
        bank="EXTENSION", n_models=len(recs),
        stamp_set=stamps,
        scope="dataseed3/4 + hid128t10k/t30k = 16 distinct models; "
              "with the main bank's 32 this covers 48 of the paper's "
              "60. Out (#31 M9): dc_gru/dc_lstm (8, world=dcfield — "
              "genuine porting) and lg_lstm (4, world=lg — same "
              "world, deferred by resource choice, NOT by porting)",
        tier_carried=len(tier_c),
        tier_clipped=sum(r["tier_clipped"] for r in tier_c),
        tier_dh_untouched=sum(r["tier_dh_untouched"] for r in recs),
        collapse_fraction=(len(lost) / max(len(tier_c), 1)),
        pmc1_ext=dict(
            statement=f"P-MC1-EXT (replication): >= 50% of the "
                      f"{PINNED_TIER_DISTINCT_EXT} extension "
                      f"carried-exploit-tier members lose tier "
                      f"membership under the clip. CLUSTERING "
                      f"CAVEAT (#31 M7): the 15 tier members come "
                      f"from only 4 ensembles — the per-ensemble "
                      f"breakdown is the mandatory disclosure and "
                      f"the Wilson interval is descriptive",
            lost=len(lost),
            collapse_wilson95=_wilson(len(lost),
                                      max(len(tier_c), 1)),
            fires=bool(len(lost) * 2 >= PINNED_TIER_DISTINCT_EXT)),
        per_ensemble=[
            dict(job=j,
                 tier=sum(1 for r in tier_c if r["job"] == j),
                 lost=sum(1 for r in lost if r["job"] == j))
            for j, _, _, _ in JOBCONFIGS_EXT],
        # #31 M5: the registered capacity x training interaction
        # table (2x3, pinned cells; requires the MAIN bank's rows —
        # assembled here descriptively iff those task jsons exist;
        # 'never pooled' is scoped to the P-MC1 denominators)
        interaction_2x3=_interaction_table(recs),
        pmc2_note="no P-MC2 analogue: the extension has no traced "
                  "mechanism exploits (mechanism.json covers pilot2 "
                  "only)",
        pmc3_ext=dict(
            statement="P-MC3-EXT guard: retention >= 0.8 on every "
                      "MEASURED model; ABSTAIN = clip never touches "
                      "that model's real credit",
            n_measured=len(ret_meas), n_abstain=len(ret_abst),
            fires=(bool(all(r["retention"]["ok"] for r in ret_meas))
                   if ret_meas else "NOT-ADJUDICABLE "
                   "(all models abstain)")),
        pmc4_ext=dict(
            n_pairs=len(fid_pairs),
            mean_rho_carried=(float(np.mean([a for a, _ in fid_pairs]))
                              if fid_pairs else None),
            mean_rho_clipped=(float(np.mean([b for _, b in fid_pairs]))
                              if fid_pairs else None),
            n_improved=sum(b > a for a, b in fid_pairs),
            note="exploratory-registered; nan-safe"),
        retention_rows=[dict(job=r["job"], member=r["member"],
                             **(r["retention"] or {}))
                        for r in recs],
        per_model=[{k: r[k] for k in
                    ("job", "member", "verdict", "tier_carried",
                     "tier_clipped", "tier_dh_untouched",
                     "n_exploit_carried", "n_exploit_clipped")}
                   for r in recs])
    od = pathlib.Path(output)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "minclip_ext_read.json", "w") as f:
        json.dump(_san(out), f, indent=1)
    print(f"MIN-CLIP EXT: tier {out['tier_carried']} -> "
          f"{out['tier_clipped']} (collapse "
          f"{out['collapse_fraction']:.2f}); P-MC1-EXT "
          f"{out['pmc1_ext']['fires']}; P-MC3-EXT "
          f"{out['pmc3_ext']['fires']} "
          f"(measured {out['pmc3_ext']['n_measured']}, abstain "
          f"{out['pmc3_ext']['n_abstain']})")
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


def _write_bank_ext(tmpdir, lost_count, tier_total=None,
                    all_abstain=False, nan_member=None, limit=None,
                    no_bank_field=False, stamp="fixture"):
    """16 extension fixture jsons matching the frozen tier layout:
    tier members = 15 (miss: hid128t30k_m1). Losses assigned to the
    first tier members in JOBCONFIGS_EXT order. (#31 m19 knobs:
    all_abstain / nan_member / limit / no_bank_field / stamp.)"""
    import os
    misses = {("hid128t30k", 1)}
    n_lost = 0
    n_tier_seen = 0
    n_written = 0
    for job, ens, ym, sd in JOBCONFIGS_EXT:
        for mem in range(4):
            if limit is not None and n_written >= limit:
                return
            tier_c = (job, mem) not in misses
            if tier_total is not None and tier_c and \
                    n_tier_seen >= tier_total:
                tier_c = False
            n_tier_seen += tier_c
            clipped = tier_c and not (n_lost < lost_count)
            if tier_c and n_lost < lost_count:
                n_lost += 1
            rec = _fake_rec(
                job, mem, tier_c, clipped,
                retention=("ABSTAIN" if all_abstain else "MEASURED"),
                rho_c=(float("nan")
                       if (job, mem) == (nan_member or (None, None))
                       else 0.1))
            if not no_bank_field:
                rec["bank"] = "EXTENSION"
            rec["stamp"] = stamp
            with open(os.path.join(tmpdir, f"{job}_m{mem}.json"),
                      "w") as f:
                json.dump(_san(rec), f)
            n_written += 1


def selfcheck():
    import tempfile
    global ROOT, EXT_ROOT
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

    # EXTENSION collate fixtures (parallel bank; pinned 15-of-16)
    old_ext = EXT_ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            EXT_ROOT = pathlib.Path(td)
            _write_bank_ext(td, lost_count=8)
            out = collate_ext(td + "/read")
            assert out["tier_carried"] == 15
            assert out["pmc1_ext"]["fires"] is True      # 8/15 >= 50%
            w = out["pmc1_ext"]["collapse_wilson95"]
            assert 0.0 < w[0] < 8 / 15 < w[1] < 1.0
            assert len(out["per_ensemble"]) == 4
            assert sum(e["lost"] for e in out["per_ensemble"]) == 8
            ic = out["interaction_2x3"]
            assert len(ic) == 6
            assert all(c["cell"] == "MAIN-BANK-PENDING"
                       for c in ic if c["bank"] == "main")
            assert all(isinstance(c["cell"], dict)
                       for c in ic if c["bank"] == "ext")
            _write_bank_ext(td, lost_count=7)
            out = collate_ext(td + "/read")
            assert out["pmc1_ext"]["fires"] is False     # 7/15 < 50%
            # all-abstain -> P-MC3-EXT NOT-ADJUDICABLE (#31 m19)
            _write_bank_ext(td, lost_count=8, all_abstain=True)
            out = collate_ext(td + "/read")
            assert out["pmc3_ext"]["fires"] == \
                "NOT-ADJUDICABLE (all models abstain)"
            # nan fidelity dropped (#31 m19)
            _write_bank_ext(td, lost_count=8,
                            nan_member=("dataseed3", 0))
            out = collate_ext(td + "/read")
            assert out["pmc4_ext"]["n_pairs"] == 15
            # incomplete bank refused (#31 m19)
            _write_bank_ext(td, lost_count=8)
            (EXT_ROOT / "dataseed4_m2.json").unlink()
            try:
                collate_ext(td + "/read")
                raise SystemExit("EXT incomplete-bank gate FAILED")
            except AssertionError as e:
                assert "NOT-ADJUDICABLE" in str(e)
            # missing bank field rejected (#31 m19)
            _write_bank_ext(td, lost_count=8, no_bank_field=True)
            try:
                collate_ext(td + "/read")
                raise SystemExit("EXT bank-field gate FAILED")
            except AssertionError as e:
                assert "non-extension" in str(e)
            # heterogeneous stamps rejected (#31 M11)
            _write_bank_ext(td, lost_count=8)
            p = EXT_ROOT / "dataseed3_m0.json"
            r0 = json.load(open(p))
            r0["stamp"] = "other-commit"
            json.dump(r0, open(p, "w"))
            try:
                collate_ext(td + "/read")
                raise SystemExit("EXT stamp gate FAILED")
            except AssertionError as e:
                assert "heterogeneous" in str(e)
            _write_bank_ext(td, lost_count=5, tier_total=14)
            try:
                collate_ext(td + "/read")
                raise SystemExit("EXT denominator drift gate FAILED")
            except AssertionError as e:
                assert "denominator drift" in str(e)
            _write_bank_ext(td, lost_count=8)
            for bad in ("artifacts/minclip_ext_read_TBD",
                        "artifacts/minclip_ext_read_TBD/"):  # m15
                try:
                    collate_ext(bad)
                    raise SystemExit("EXT TBD output gate FAILED")
                except AssertionError as e:
                    assert "explicit --output" in str(e)
    finally:
        EXT_ROOT = old_ext

    # EXT drift gate vs the real frozen summaries (#31 M4/m17):
    # full schema over every field the gate indexes, 4 members with
    # indices 0-3, ensembles present with 4 members, and the tier
    # total tied to the pinned constant
    GATE_KEYS = ("verdict", "n_cycles", "n_neutral",
                 "n_exploit_naive", "n_exploit_cig",
                 "n_exploit_pbim", "n_exploit_both",
                 "n_exploit_transient", "top_cycle", "top_true",
                 "top_eig", "top_dh_adj", "drift_per_action")
    n_tier_total = 0
    for j, ens, ym, sd in JOBCONFIGS_EXT:
        with open(OUT / ens / "summary.json") as f:
            summ = json.load(f)
        assert str(summ["y_mode"]) == ym and int(summ["seed"]) == sd, (
            f"{j}: frozen summary y_mode/seed != JOBCONFIGS_EXT entry")
        assert len(summ["members"]) == 4 and \
            sorted(int(m["member"]) for m in summ["members"]) == \
            [0, 1, 2, 3], f"{j}: member layout"
        import pickle as _pkl
        with open(OUT / ens / "ensemble.pkl", "rb") as f:
            e = _pkl.load(f)
        assert len(e) == 4, f"{j}: ensemble size {len(e)}"
        for m in summ["members"]:
            for k in GATE_KEYS:
                assert k in m and m[k] is not None, (j, m["member"], k)
            want_tier = not (j == "hid128t30k"
                             and int(m["member"]) == 1)
            assert (m["n_exploit_cig"] > 0) == want_tier, (
                f"{j}_m{m['member']}: frozen tier != pinned layout")
            n_tier_total += want_tier
    assert n_tier_total == PINNED_TIER_DISTINCT_EXT, n_tier_total

    print("minclip selfcheck PASS (clip semantics; 3 mechanism-oracle "
          "licensed rates exact to 1e-9 incl. the m3 large-finite "
          "residual; determinism; collate fixtures: pmc1 fire/no-fire "
          "at 15-vs-14 of 29, pmc2 pattern, all-abstain "
          "NOT-ADJUDICABLE, nan-fidelity drop, denominator-drift + "
          "TBD-output gates; EXT bank: pmc1_ext fire/no-fire at "
          "8-vs-7 of 15 + Wilson + per-ensemble + interaction "
          "placeholders, all-abstain/nan-fidelity/incomplete-bank/"
          "bank-field/stamp-heterogeneity/drift/TBD-slash gates, "
          "frozen-summary full gate-key schema + ensembles + tier "
          "total == pinned 15)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=-1)
    ap.add_argument("--task_ext", type=int, default=-1)
    ap.add_argument("--collate", action="store_true")
    ap.add_argument("--collate_ext", action="store_true")
    ap.add_argument("--output",
                    default="artifacts/minclip_read_TBD")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    modes = [args.selfcheck, args.collate, args.collate_ext,
             args.task >= 0, args.task_ext >= 0]
    assert sum(modes) == 1, (
        "pass exactly one of --selfcheck/--collate/--collate_ext/"
        "--task/--task_ext (#31 m16)")
    if args.selfcheck:
        selfcheck()
    elif args.collate:
        collate(args.output)
    elif args.collate_ext:
        collate_ext(args.output)
    elif args.task_ext >= 0:
        assert args.task_ext < N_TASKS_EXT, args.task_ext
        run_task_ext(args.task_ext)
    else:
        assert 0 <= args.task < N_TASKS, args.task
        run_task(args.task)


if __name__ == "__main__":
    main()
