#!/usr/bin/env python3
"""Which run dirs on an instance are already safe on RCC, and how much they hold.

Reports only. Deleting is a separate, explicit step, because the standing rule
is that nothing leaves an instance until a verified copy exists elsewhere.

SAFE means all four realized counters on RCC are >= the instance's
(ckpt_step, n_scores, n_files, bytes) -- the same test `dv3ops destroy` uses.
A dir currently being WRITTEN is never safe: its counters are still moving, so
a copy that matched a minute ago is already stale. Running run_ids must be
passed in via --running and are reported separately, never as reclaimable.

TWO GUARDS EXIST BECAUSE THE COUNTERS TEST ALONE IS NOT ENOUGH. On 2026-08-21
this tool marked the top-level entry `tm2r3` reclaimable on an instance where
24 Wave-2 runs were writing into `$RUNROOT/tm2r3/<run_id>/`:

  * the --running guard compared RUN IDS against TOP-LEVEL NAMES, and the
    container `tm2r3` matches no run_id, so it looked idle; and
  * RCC has its own unrelated `tm2r3` directory (the seeds 51-58 originals)
    whose counters are larger, so the container compared as "covered" -- the
    right arithmetic on the wrong pair of directories.

`rm -rf tm2r3` then took the live tree. So:

  --mtime-json   newest mtime anywhere under each entry. Anything touched
                 within --fresh-min (default 120) is REFUSED outright, whatever
                 the counters say. Freshness is the guard that does not depend
                 on names lining up.
  container      an entry that is not itself a run dir (no scores.jsonl,
                 config.yaml, config.json or ckpt/ of its own) is REFUSED:
                 its name says nothing about what is inside it, on either side.

  usage: free_space.py --inst-json F --rcc-json F [--running id ...]
                       [--mtime-json F] [--fresh-min N]
"""
import argparse, json

FIELDS = ("ckpt_step", "n_scores", "n_files", "bytes")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--inst-json", required=True)
  ap.add_argument("--rcc-json", required=True)
  ap.add_argument("--running", nargs="*", default=[])
  ap.add_argument("--mtime-json", default=None,
                  help='{"name": <newest mtime epoch under it>} from the instance')
  ap.add_argument("--fresh-min", type=int, default=120)
  ap.add_argument("--now", type=float, default=None, help="instance clock, epoch")
  a = ap.parse_args()
  inst = json.load(open(a.inst_json))["entries"]
  rcc = json.load(open(a.rcc_json))["entries"]
  running = set(a.running)

  import time
  mt = json.load(open(a.mtime_json)) if a.mtime_json else {}
  now = a.now if a.now is not None else time.time()

  safe, held, risky = [], [], []
  for name, x in sorted(inst.items()):
    y = rcc.get(name)
    age_min = (now - mt[name]) / 60 if name in mt else None
    if name in running:
      held.append((name, x["bytes"], "RUNNING -- counters still moving"))
    elif x.get("is_container"):
      held.append((name, x["bytes"],
                   "CONTAINER -- not a run dir; a shared name proves nothing "
                   "about the contents on either side"))
    elif age_min is not None and age_min < a.fresh_min:
      held.append((name, x["bytes"], f"FRESH -- written {age_min:.0f} min ago"))
    elif a.mtime_json and name not in mt:
      held.append((name, x["bytes"], "NO MTIME -- refusing without a freshness check"))
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
