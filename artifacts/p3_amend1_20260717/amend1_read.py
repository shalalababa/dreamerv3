"""Amendment-1 registered read (PREREG_p3_amendment1_20260717.md).

Reuses the FROZEN p3_factorial_read machinery (contrast/CI/sensitivity,
committed pre-outcome 2026-07-16) with SEEDS = 1..16 -- the only change
the amendment declares. Seeds 1-8 rows come from the bit-checked P3
wave artifact (auc_p3.csv); seeds 9-16 from the amendment snapshot's
frozen adaptation_auc output.

  PRIMARY      pooled 1-16 [B_rgo - B_sgb]   (decision CI alone)
  SUBSIDIARY   fresh-batch 9-16 [B_rgo - B_sgb]  (winner's-curse check)
  SECONDARY    rgo simple pooled 1-16; sgb simple pooled 1-16

Usage: python amend1_read.py <amend1_auc.csv> <outdir>
"""

import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from analysis.p3_factorial_read import contrast, load_b_arm  # frozen machinery

P3_CSV = os.path.join(REPO, "artifacts/p3_wave_20260717/auc_p3.csv")
P3_JSON = os.path.join(REPO, "artifacts/p3_wave_20260717/p3_read.json")


def main(amend_csv, outdir):
  os.makedirs(outdir, exist_ok=True)
  # Merge: seeds 1-8 rows from the P3 wave artifact + seeds 9-16 rows.
  merged = os.path.join(outdir, "auc_pooled_1_16.csv")
  with open(merged, "w", newline="") as f:
    w = None
    for path, keep in ((P3_CSV, range(1, 9)), (amend_csv, range(9, 17))):
      for r in csv.DictReader(open(path)):
        if r["mode"].startswith(("ax1rgo", "ax1sgb")) and \
            int(r["seed"]) in keep:
          if w is None:
            w = csv.DictWriter(f, fieldnames=r.keys())
            w.writeheader()
          w.writerow(r)

  seeds16 = list(range(1, 17))
  fresh = list(range(9, 17))
  b_rgo = load_b_arm(merged, "rgo", seeds=seeds16)
  b_sgb = load_b_arm(merged, "sgb", seeds=seeds16)

  # Bit-check: the seeds-1-8 deltas must reproduce the frozen P3 read.
  frozen = json.load(open(P3_JSON))["b_arm"]
  assert np.allclose(b_rgo[:8], frozen["rgo"], atol=1e-4), "rgo 1-8 mismatch"
  assert np.allclose(b_sgb[:8], frozen["sgb"], atol=1e-4), "sgb 1-8 mismatch"

  out = dict(seeds=seeds16, b_rgo=b_rgo.tolist(), b_sgb=b_sgb.tolist())
  out["primary_pooled_rgo_minus_sgb"] = contrast(b_rgo - b_sgb,
                                                 decision=True)
  fr = np.array([s - 1 for s in fresh])
  out["subsidiary_fresh_rgo_minus_sgb"] = contrast((b_rgo - b_sgb)[fr])
  out["secondary_rgo_simple_pooled"] = contrast(b_rgo)
  out["secondary_sgb_simple_pooled"] = contrast(b_sgb)
  out["batch_means"] = dict(
      rgo_1_8=float(b_rgo[:8].mean()), rgo_9_16=float(b_rgo[8:].mean()),
      sgb_1_8=float(b_sgb[:8].mean()), sgb_9_16=float(b_sgb[8:].mean()))

  p = out["primary_pooled_rgo_minus_sgb"]
  fires = p["decision"] == "CI excludes 0" and p["mean"] > 0
  out["registered_branch"] = (
      "PRIMARY fires: the reward-head representation gradient alone "
      "carries part of the interaction - the 17-Jul descriptive claim "
      "upgrades to confirmatory (P-A1 confirmed; 9a un-embargoed)."
      if fires else
      "PRIMARY does not fire: the both-paths/interplay branch of the "
      "original interpretation map stands as FINAL (P-A1 not "
      "confirmed; theory falls back to P-A2 slow-crossing, "
      "discriminator = extended-fit arm).")

  json.dump(out, open(os.path.join(outdir, "amend1_read.json"), "w"),
            indent=1)

  def line(name, c):
    extra = f" -> {c['decision']}" if "decision" in c else ""
    print(f"{name:<34} {c['mean']:+9.2f} "
          f"CI=[{c['ci'][0]:+8.2f},{c['ci'][1]:+8.2f}]"
          f" pos={c['pos']}/{c['n']}"
          f" perm_p={c['sensitivity']['perm_p']:.5f}{extra}")

  print("batch means:", {k: round(v, 1) for k, v in out["batch_means"].items()})
  line("PRIMARY  pooled 1-16 rgo-sgb", out["primary_pooled_rgo_minus_sgb"])
  line("SUBSID   fresh 9-16 rgo-sgb ", out["subsidiary_fresh_rgo_minus_sgb"])
  line("SEC      rgo simple pooled  ", out["secondary_rgo_simple_pooled"])
  line("SEC      sgb simple pooled  ", out["secondary_sgb_simple_pooled"])
  print("\nREGISTERED BRANCH:", out["registered_branch"])


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
