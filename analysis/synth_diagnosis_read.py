"""Synth in-subspace diagnosis summary (PREREG_synth_diagnosis_20260718.md).

DESCRIPTIVE-ONLY summarizer for the registered diagnosis of the Phase-B
collapse-only branch. Reads (1) reward_direction_rank JSONs for the
synth and finger buffer sides and (2) stratified_error measure
summary.json files over the 48 synth WM checkpoints, and evaluates the
three frozen directional predictions:

  P-D1  synth reward-direction rank <= 3 both sides; finger > 3 both
        sides; synth strictly lower everywhere.
  P-D2  |task s0-fits - s1-fits| reward-head NLL (h0, in_regime, seed
        mean) < 0.051 (finger q1's E4 separation), with low level both
        sides (level reported, not thresholded).
  P-D3  apt/task ratio of decoder NLL on the `to_target` key (h0,
        in_regime, seed means) within [0.8, 1.25].

All three hold => in-subspace confirmed => Phase-B' authorized for its
own registration (resource gate only; nothing here is confirmatory).

Usage:
  python -m analysis.synth_diagnosis_read \
      --rank_synth <side0.json> <side1.json> \
      --rank_finger <side0.json> <side1.json> \
      --e4_summaries <summary.json ...>   (glob over e4_synth_v1 dirs) \
      --output <dir>
  python -m analysis.synth_diagnosis_read --selfcheck
"""

import argparse
import json
import os
import re

import numpy as np

RUN_RE = re.compile(r"ax1wm_synth_(f|sh)?q1s(\d)_seed(\d+)")
FINGER_SEP = 0.051
RATIO_BAND = (0.8, 1.25)


def load_ranks(paths):
  out = []
  for p in paths:
    with open(p) as f:
      out.append(int(json.load(f)["rank_of_reward_direction"]))
  return out


def collect_e4(paths):
  """summary.json list -> {(arm, side): {seed: summary}}."""
  cells = {}
  for p in paths:
    with open(p) as f:
      s = json.load(f)
    m = RUN_RE.match(s["run_id"])
    assert m, f"unparseable run_id {s['run_id']} in {p}"
    arm = {None: "task", "f": "apt", "sh": "sh"}[m.group(1)]
    cells.setdefault((arm, int(m.group(2))), {})[int(m.group(3))] = s
  return cells


def _h0(summary):
  return summary["horizon_stats"]["0"]


def diagnose(rank_synth, rank_finger, cells):
  res = {}
  d1 = bool(max(rank_synth) <= 3 and min(rank_finger) > 3 and
            max(rank_synth) < min(rank_finger))
  res["p_d1"] = dict(rank_synth=rank_synth, rank_finger=rank_finger,
                     passes=d1)

  def seed_mean(arm, side, fn):
    cell = cells.get((arm, side), {})
    vals = [fn(_h0(s)) for s in cell.values()]
    return float(np.mean(vals)) if vals else float("nan"), len(vals)

  rew = lambda h: h["reward_head"]["in_regime"]["mean"]
  t0, n0 = seed_mean("task", 0, rew)
  t1, n1 = seed_mean("task", 1, rew)
  sep = abs(t0 - t1)
  d2 = bool(np.isfinite(sep) and sep < FINGER_SEP)
  res["p_d2"] = dict(task_rew_nll_s0=t0, task_rew_nll_s1=t1,
                     separation=float(sep), finger_separation=FINGER_SEP,
                     n=[n0, n1], passes=d2)

  tt = lambda h: h["per_key"]["to_target"]["in_regime"]["mean"]
  ratios = []
  for side in (0, 1):
    a, _ = seed_mean("apt", side, tt)
    t, _ = seed_mean("task", side, tt)
    ratios.append(a / t if np.isfinite(a) and np.isfinite(t) and t != 0
                  else float("nan"))
  d3 = bool(all(np.isfinite(r) and RATIO_BAND[0] <= r <= RATIO_BAND[1]
                for r in ratios))
  res["p_d3"] = dict(apt_over_task_to_target_nll=ratios,
                     band=list(RATIO_BAND), passes=d3)

  res["in_subspace_confirmed"] = bool(d1 and d2 and d3)
  res["gate"] = ("Phase-B' AUTHORIZED for registration (hide to_target "
                 "via agent.model_obs)" if res["in_subspace_confirmed"]
                 else "gate CLOSED - re-diagnose (see failing instrument)")
  return res


def report(res):
  d1, d2, d3 = res["p_d1"], res["p_d2"], res["p_d3"]
  print(f"P-D1 rank: synth {d1['rank_synth']} vs finger "
        f"{d1['rank_finger']} -> {'PASS' if d1['passes'] else 'FAIL'}")
  print(f"P-D2 task rew-NLL h0 in-regime: s0 {d2['task_rew_nll_s0']:.3f} "
        f"s1 {d2['task_rew_nll_s1']:.3f} sep {d2['separation']:.3f} "
        f"(finger {FINGER_SEP}) -> {'PASS' if d2['passes'] else 'FAIL'}")
  print(f"P-D3 apt/task to_target NLL ratio: "
        f"{[round(r, 3) for r in d3['apt_over_task_to_target_nll']]} "
        f"band {RATIO_BAND} -> {'PASS' if d3['passes'] else 'FAIL'}")
  print(f"\nIN-SUBSPACE: "
        f"{'CONFIRMED' if res['in_subspace_confirmed'] else 'NOT confirmed'}"
        f" -> {res['gate']}")


def selfcheck():
  def fake(arm, side, seed, rew_in, tt_in):
    h = dict(per_key={"to_target": {"in_regime": {"mean": tt_in, "n": 10}}})
    if arm == "task":
      h["reward_head"] = {"in_regime": {"mean": rew_in, "n": 10}}
    infix = {"task": "", "apt": "f", "sh": "sh"}[arm]
    return dict(run_id=f"ax1wm_synth_{infix}q1s{side}_seed{seed}",
                horizon_stats={"0": h})

  summaries = []
  for side in (0, 1):
    for seed in (1, 2):
      summaries.append(fake("task", side, seed, rew_in=0.30 + 0.01 * side,
                            tt_in=1.0))
      summaries.append(fake("apt", side, seed, rew_in=None, tt_in=1.05))
  cells = collect_e4_from_objs(summaries)
  res = diagnose([1, 2], [6, 8], cells)
  assert res["p_d1"]["passes"] and res["p_d2"]["passes"] \
      and res["p_d3"]["passes"] and res["in_subspace_confirmed"], res
  # Failing branch: big head separation + apt misses to_target.
  summaries = []
  for side in (0, 1):
    for seed in (1, 2):
      summaries.append(fake("task", side, seed, rew_in=1.0 - 0.5 * side,
                            tt_in=1.0))
      summaries.append(fake("apt", side, seed, rew_in=None, tt_in=2.0))
  res = diagnose([1, 2], [2, 8], collect_e4_from_objs(summaries))
  assert not res["p_d1"]["passes"] and not res["p_d2"]["passes"] \
      and not res["p_d3"]["passes"] and not res["in_subspace_confirmed"]
  print("selfcheck PASS: confirm branch and all-fail branch recovered")


def collect_e4_from_objs(objs):
  cells = {}
  for s in objs:
    m = RUN_RE.match(s["run_id"])
    arm = {None: "task", "f": "apt", "sh": "sh"}[m.group(1)]
    cells.setdefault((arm, int(m.group(2))), {})[len(cells) * 100 +
                                                 len(cells.get(
                                                     (arm, int(m.group(2))),
                                                     {}))] = s
  return cells


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--rank_synth", nargs=2)
  ap.add_argument("--rank_finger", nargs=2)
  ap.add_argument("--e4_summaries", nargs="+")
  ap.add_argument("--output", default="analysis_out/synth_diagnosis")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  rank_s = load_ranks(args.rank_synth)
  rank_f = load_ranks(args.rank_finger)
  cells = collect_e4(args.e4_summaries)
  res = diagnose(rank_s, rank_f, cells)
  res["inputs"] = dict(rank_synth=args.rank_synth,
                       rank_finger=args.rank_finger,
                       n_e4_summaries=len(args.e4_summaries))
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, "synth_diagnosis.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
  report(res)


if __name__ == "__main__":
  main()
