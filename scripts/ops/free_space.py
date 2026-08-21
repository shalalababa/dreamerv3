#!/usr/bin/env python3
"""Which run dirs on an instance are already safe on RCC, and how much they hold.

Reports only. Deleting is a separate, explicit step, because the standing rule
is that nothing leaves an instance until a verified copy exists elsewhere.

SAFE means all four realized counters on RCC are >= the instance's
(ckpt_step, n_scores, n_files, bytes) -- the same test `dv3ops destroy` uses.
A dir currently being WRITTEN is never safe: its counters are still moving, so
a copy that matched a minute ago is already stale. Running run_ids must be
passed in via --running and are reported separately, never as reclaimable.

  usage: free_space.py --inst-json F --rcc-json F [--running id ...]
"""
import argparse, json

FIELDS = ("ckpt_step", "n_scores", "n_files", "bytes")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--inst-json", required=True)
  ap.add_argument("--rcc-json", required=True)
  ap.add_argument("--running", nargs="*", default=[])
  a = ap.parse_args()
  inst = json.load(open(a.inst_json))["entries"]
  rcc = json.load(open(a.rcc_json))["entries"]
  running = set(a.running)

  safe, held, risky = [], [], []
  for name, x in sorted(inst.items()):
    y = rcc.get(name)
    if name in running:
      held.append((name, x["bytes"], "RUNNING -- counters still moving"))
    elif y is None:
      risky.append((name, x["bytes"], "NOT on RCC"))
    else:
      short = [f for f in FIELDS if y.get(f, -1) < x.get(f, 0)]
      if short:
        risky.append((name, x["bytes"], "SHORT on RCC: " + ",".join(short)))
      else:
        safe.append((name, x["bytes"]))

  gb = lambda b: b / 2**30
  print(f"SAFE TO REMOVE   {len(safe):3d} dir(s)  {gb(sum(b for _, b in safe)):7.2f} GiB")
  for n, b in safe:
    print(f"    {n:<52s} {gb(b):6.2f} GiB")
  print(f"HELD (running)   {len(held):3d} dir(s)  {gb(sum(b for _, b, _ in held)):7.2f} GiB")
  for n, b, why in held:
    print(f"    {n:<52s} {gb(b):6.2f} GiB  {why}")
  print(f"NOT RECLAIMABLE  {len(risky):3d} dir(s)  {gb(sum(b for _, b, _ in risky)):7.2f} GiB")
  for n, b, why in risky:
    print(f"    {n:<52s} {gb(b):6.2f} GiB  {why}")
  print("\n".join(["", "reclaimable names (one per line):"] + [n for n, _ in safe]))


if __name__ == "__main__":
  main()
