"""Family-2 (dc) planner-realistic-depth re-score — review §9 item 8.

Re-runs the full 626-cycle sweep for dc_gru + dc_lstm (8 members) at
n_burn=1, n_loops=5 (steady = loops 2-5). In the dc world the referee's
entropy path is observation-independent and the bait cycles are exactly
zero from the first read, so neutrality is certified at this depth with
no burn-in — the exploit becomes well-defined at planner-realistic
imagination depth, closing the "38-loop imagination window" deflation
(or failing to — both outcomes recorded).

EXPLORATORY / descriptive. No registered decision rules. Same-machine,
same code path as family2.py's _search_cell (which uses planner
defaults); only n_burn/n_loops differ, disclosed by design.

Output: nburn1_rescore.json next to this script (folded into
artifacts/nfi_review_response_20260809/ as a dated addendum).
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, "/home/rickybao/projects/dreamerv3")

from uncfield import dcfield as dc
from uncfield import dcwm
from uncfield import planner as pl

ROOT = pathlib.Path("/home/rickybao/projects/dreamerv3/local_results/uncfield/family2")
OUT = pathlib.Path(__file__).parent / "nburn1_rescore.json"
EPS = pl.EPS_PRED

out = {"n_burn": 1, "n_loops": 5, "steady": "loops 2-5", "cells": {}}
for cell in ("dc_gru", "dc_lstm"):
    ensemble = dcwm.load_ensemble(ROOT / cell / "ensemble.pkl")
    reg = json.load(open(ROOT / cell / "summary.json"))
    members = []
    for m, params in enumerate(ensemble):
        t0 = time.time()
        model = dcwm.DCLearnedModel(params)
        results, _ = pl.search_exploits(model, seed=0, max_len=6,
                                        n_burn=1, n_loops=5, world=dc)
        verdict, counts = pl.classify(results)
        neu = [r for r in results if r["neutral"]]
        cig = [r for r in neu if r["carried_eig_rate"] > EPS]
        both_raw = [r for r in cig if r["carried_dh_rate"] > EPS]
        both_adj = [r for r in cig if r["carried_dh_adj"] > EPS]
        top = max(results, key=lambda r: r["carried_eig_rate"])
        rec = dict(
            member=m,
            verdict=verdict,
            n_neutral=len(neu),
            n_cig=len(cig),
            n_both_raw=len(both_raw),
            n_both_adj=len(both_adj),
            top_cycle=top["cycle"]["name"],
            top_carried=round(top["carried_eig_rate"], 4),
            top_true=round(top["true_rate"], 6),
            top_neutral=bool(top["neutral"]),
            max_cig_rate=round(max((r["carried_eig_rate"] for r in cig),
                                   default=0.0), 4),
            registered_verdict=reg["members"][m]["verdict"],
            registered_n_cig=reg["members"][m]["n_exploit_cig"],
            elapsed_s=round(time.time() - t0, 1),
        )
        members.append(rec)
        print(f"{cell} m{m}: {verdict} cig={rec['n_cig']} "
              f"(reg {rec['registered_n_cig']}) both raw/adj="
              f"{rec['n_both_raw']}/{rec['n_both_adj']} "
              f"top={rec['top_cycle']} carried={rec['top_carried']} "
              f"true={rec['top_true']} [{rec['elapsed_s']}s]", flush=True)
    out["cells"][cell] = members

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print("done ->", OUT)
