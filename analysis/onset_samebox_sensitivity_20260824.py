"""Onset Amendment-1 cl.4 registered read-time sensitivity, 24 Aug 2026.

SEPARATE dated descriptive computation — the frozen reader
`uncfield/se_onset_read.py` is NOT edited (amendment text). Consumes
the ALREADY-CONSUMED read json only; touches no run dirs.

Computes the continuation-box-only Spearman rho(theta_1, scale) on
the 12 instance-25 runs (3 per scale; round-1 instance-15 runs
s96/s100/s104/s108 excluded). DESCRIPTIVE-NEVER-A-FIRE by
registration: 12 < MIN_USABLE (14), so this cannot be a primary —
it is the same-device consistency row. Same permutation machinery
and seed as the reader (spearman_perm_null, PERM_SEED, 100k draws)
so the two rows are commensurable.

Usage: python analysis/onset_samebox_sensitivity_20260824.py \
           artifacts/b_onset_read_20260824/se_onset_read.json
Writes samebox_sensitivity.json next to the input.
"""
import json
import os
import sys

from uncfield.se_dose_read import spearman_perm_null
from uncfield.se_onset_read import PERM_SEED

INST15_SEEDS = {96, 100, 104, 108}          # round 1, scale-complete
N_PERM = 100_000


def main(path):
    j = json.load(open(path))
    rows = []
    for r in j["per_run"]:
        seed = int(r["train_seed"])
        rows.append(dict(run=os.path.basename(r["run_dir"]), seed=seed,
                         scale=float(r["scale"]),
                         theta1=float(r["theta1"]),
                         device=("instance-15" if seed in INST15_SEEDS
                                 else "instance-25")))
    cont = [r for r in rows if r["device"] == "instance-25"]
    assert len(cont) == 12, [r["run"] for r in cont]
    scales = sorted({r["scale"] for r in cont})
    assert all(sum(r["scale"] == s for r in cont) == 3 for s in scales)
    th = [r["theta1"] for r in cont]
    sc = [r["scale"] for r in cont]
    # the reader tests the POSITIVE direction on the mirrored stat
    rho_neg, _ = spearman_perm_null(th, sc, N_PERM, PERM_SEED)
    rho, p_pos = spearman_perm_null([-t for t in th], sc, N_PERM,
                                    PERM_SEED)
    out = dict(
        registration=("PREREG_trackB_onset_amend1_20260823 cl.4: "
                      "descriptive-never-a-fire (n=12 < MIN_USABLE 14); "
                      "same-device consistency row"),
        n=12, per_run=cont,
        spearman_rho=float(-rho),      # un-mirror for reporting
        p_one_sided_positive=float(p_pos),
        n_perm=N_PERM, perm_seed=PERM_SEED,
        pooled_primary_for_reference=j["primary"],
    )
    dst = os.path.join(os.path.dirname(path), "samebox_sensitivity.json")
    assert not os.path.exists(dst), f"{dst} exists — dated one-shot"
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("spearman_rho", "p_one_sided_positive", "n")}))
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1])
