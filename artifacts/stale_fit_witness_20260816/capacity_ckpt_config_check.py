#!/usr/bin/env python3
"""Capacity-config check for the 08-07 truncation-disclosure adjudication.

Context (artifacts/stale_fit_witness_20260816/RECORD.md): 27/27 cells the
08-07 REALIZED_TRAINING disclosure calls walltime-truncated carry a `done`
checkpoint at 500000 that POSTDATES the frozen progress witness (median
+5 h) — evidence the progress files are stale sync copies, not that
training stopped. The one story under which the disclosure still stands:
the 500000 checkpoint could be a foreign restore admitted by the
latest_ckpt()+idempotent-skip hole, i.e. weights of the WRONG capacity.
This script closes that gap.

VALUE-BLIND: reads parameter SHAPES and counts only — no weights, no
outcome quantities, no adapt scores. Safe to run pre-adjudication.

For every ax1wm_finger_*s12* / *s25* run dir it:
  1. picks the done checkpoint at/above total_updates (same selection as
     ckpt_vs_witness.py);
  2. loads its agent.pkl (plain pickle of a numpy state tree) and walks
     every array;
  3. reports param_count + all distinct dims >= 192 seen in shapes;
  4. verdict per cell against the preset the CELL NAME requires
     (s12 -> size12m deter 2048 / units 256; s25 -> size25m deter 3072 /
     units 384; presets from dreamerv3/configs.yaml):
       MATCH      expected deter present, other preset's deter absent
       MISMATCH   other preset's deter present, expected absent
       AMBIGUOUS  both or neither (report for human adjudication)
       UNREADABLE agent.pkl failed to load (falls back to file size)
  5. cross-checks the run dir's config.yaml rssm deter (reported, not
     authoritative — config.yaml could be authentic while the ckpt is
     foreign, which is exactly the scenario under test).

Run on the RCC login node (CPU; numpy + yaml + stdlib):
  python capacity_ckpt_config_check.py > capacity_ckpt_config_check.json

Memory note: agent.pkl for size25m is a few hundred MB; runs are loaded
one at a time and released.
"""

import glob
import json
import math
import os
import pickle
import re
import sys

R = "/scratch/midway3/rickybao/dreamerv3_runs"
EXPECT = {"s12": dict(deter=2048, units=256, other=3072),
          "s25": dict(deter=3072, units=384, other=2048)}
DIM_FLOOR = 192  # ignore small dims (stoch classes etc.); presets differ at >=256


def walk_shapes(node, acc):
  if hasattr(node, "shape") and hasattr(node, "dtype"):
    acc.append(tuple(int(d) for d in node.shape))
  elif isinstance(node, dict):
    for v in node.values():
      walk_shapes(v, acc)
  elif isinstance(node, (list, tuple)):
    for v in node:
      walk_shapes(v, acc)


def config_deters(path):
  try:
    import yaml
    with open(path) as f:
      cfg = yaml.safe_load(f)
  except Exception:
    return None
  found = set()

  def rec(n):
    if isinstance(n, dict):
      if "deter" in n and isinstance(n.get("deter"), int):
        found.add(n["deter"])
      for v in n.values():
        rec(v)
  rec(cfg)
  return sorted(found)


def pick_done_ckpt(d):
  kv = {}
  try:
    for line in open(os.path.join(d, "OFFLINE_FIT_PROGRESS")):
      if "=" in line:
        k, v = line.strip().split("=", 1)
        kv[k] = v
  except OSError:
    return None, None, None
  try:
    upd, tot = int(kv.get("update", 0)), int(kv.get("total_updates", 0))
  except ValueError:
    return None, None, None
  cks = []
  for c in glob.glob(os.path.join(d, "ckpt", "*")):
    m = re.search(r"^(\d{8}T\d{6})F\d+-(\d+)$", os.path.basename(c))
    if m and os.path.exists(os.path.join(c, "done")):
      cks.append((int(m.group(2)), m.group(1), c))
  cks.sort()
  top = [c for c in cks if tot and c[0] >= tot]
  return (top[0] if top else None), upd, tot


def main():
  out = []
  dirs = sorted(glob.glob(os.path.join(R, "ax1wm_finger_*s12*")) +
                glob.glob(os.path.join(R, "ax1wm_finger_*s25*")))
  for d in dirs:
    run = os.path.basename(d)
    key = "s12" if "s12" in run else "s25"
    exp = EXPECT[key]
    row = dict(run=run, expects=key, expected_deter=exp["deter"])
    ck, upd, tot = pick_done_ckpt(d)
    row["witness_update"], row["total"] = upd, tot
    if ck is None:
      row["verdict"] = "NO-DONE-CKPT-AT-TOTAL"
      out.append(row)
      continue
    step, stamp, cdir = ck
    row["ckpt"] = dict(step=step, stamp=stamp)
    apath = os.path.join(cdir, "agent.pkl")
    row["agent_pkl_bytes"] = (os.path.getsize(apath)
                              if os.path.exists(apath) else None)
    shapes = []
    try:
      with open(apath, "rb") as f:
        state = pickle.load(f)
      walk_shapes(state, shapes)
      del state
    except Exception as e:
      row["verdict"] = "UNREADABLE"
      row["error"] = repr(e)[:200]
      out.append(row)
      continue
    dims = sorted({d_ for s in shapes for d_ in s if d_ >= DIM_FLOOR})
    row["param_count"] = int(sum(math.prod(s) for s in shapes))
    row["dims_ge_192"] = dims
    has_exp, has_other = exp["deter"] in dims, exp["other"] in dims
    row["verdict"] = ("MATCH" if has_exp and not has_other else
                      "MISMATCH" if has_other and not has_exp else
                      "AMBIGUOUS")
    row["config_yaml_deters"] = config_deters(os.path.join(d, "config.yaml"))
    out.append(row)
  json.dump(out, sys.stdout, indent=1)
  n = len(out)
  counts = {}
  for r in out:
    counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
  print(f"\n# {n} runs: {counts}", file=sys.stderr)


if __name__ == "__main__":
  main()
