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
# Counters say HOW MUCH; the witness says WHICH RUN. Two runs of one shape
# under one name (the spoiled NOBOOT arm and its re-run) are identical in
# every counter and opposite in config.


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--inst-json", required=True)
  ap.add_argument("--rcc-json", required=True)
  ap.add_argument("--running", nargs="*", default=[])
  ap.add_argument("--mtime-json", default=None,
                  help='{"name": <newest mtime epoch under it>} from the instance')
  ap.add_argument("--fresh-min", type=int, default=120)
  ap.add_argument("--queue-text", default=None,
                  help="the LIVE slice of the instance's active queues -- tasks "
                       "from running_index onward, NOT whole tasks.txt files. "
                       "Any entry named in a pending or running command is "
                       "refused. Passing completed tasks too holds every dir the "
                       "lane ever wrote and stalls the reclaim entirely.")
  ap.add_argument("--now", type=float, default=None, help="instance clock, epoch")
  a = ap.parse_args()
  inst_doc = json.load(open(a.inst_json))
  rcc_doc = json.load(open(a.rcc_json))
  for label, doc in (("instance", inst_doc), ("rcc", rcc_doc)):
    if doc.get("schema", 0) < 3:
      raise SystemExit(
          f"REFUSE: {label} inventory is schema {doc.get('schema', 0)}, need >= 3. "
          "An older inventory silently disables the container, witness and "
          "freshness guards while still printing SAFE. Regenerate it.")
  inst = inst_doc["entries"]
  rcc = rcc_doc["entries"]
  # Nested keys are "<container>/<run_id>" but the only string an operator has
  # is the run_id. Matching only the whole key is the SAME defect this file's
  # header describes -- it survived the granularity fix and had to be found by
  # review, not by the guard itself.
  running = set(a.running)
  # SUBSTRATE is the third way a dir can be in use, and neither of the other
  # guards sees it. `r3_local/r3_finger_e1_seed31` is the INPUT checkpoint of a
  # running labeler pass: it is not a run_id in --running, and nothing writes to
  # it, so its mtime is hours old and its RCC copy is complete -- SAFE by every
  # test, and deleting it would have killed 17 live passes (21 Aug). A dir named
  # in any queued or running command is in use, whoever is reading it.
  qtext = open(a.queue_text).read() if a.queue_text else ""
  def in_use(key: str) -> bool:
    if key in running or key.rsplit("/", 1)[-1] in running:
      return True
    return bool(qtext) and (key in qtext or f"/{key.rsplit('/', 1)[-1]}" in qtext)

  import time
  # Freshness is the guard that does not depend on names lining up, so it is
  # ON by default, sourced from the inventory itself rather than an optional
  # side file. And it uses the INSTANCE clock: comparing a remote mtime to the
  # local wall clock reports a file written seconds ago as hours old.
  mt = json.load(open(a.mtime_json)) if a.mtime_json else {
      k: v.get("newest_mtime", 0) for k, v in inst.items()}
  inst_now = json.load(open(a.inst_json)).get("now")
  now = a.now if a.now is not None else (inst_now or time.time())

  safe, held, risky = [], [], []
  for name, x in sorted(inst.items()):
    y = rcc.get(name)
    age_min = (now - mt[name]) / 60 if name in mt else None
    if in_use(name):
      held.append((name, x["bytes"], "IN USE -- running, or named by a queued command"))
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
      wa, wb = x.get("witness", {}) or {}, y.get("witness", {}) or {}
      shared = set(wa) & set(wb)
      if short:
        risky.append((name, x["bytes"], "SHORT on RCC: " + ",".join(short)))
      elif "ERROR" in wa.values() or "ERROR" in wb.values():
        risky.append((name, x["bytes"], "identity file unreadable -- cannot verify"))
      elif shared and any(wa[k] != wb[k] for k in shared):
        k = next(k for k in sorted(shared) if wa[k] != wb[k])
        risky.append((name, x["bytes"],
                      f"COVERED BUT DIFFERENT: {k} {wa[k][:12]} vs rcc {wb[k][:12]} "
                      "-- same name, same size, not the same run"))
      elif shared and (set(wa) - set(wb)):
        risky.append((name, x["bytes"], "RCC missing identity file(s): "
                                        + ",".join(sorted(set(wa) - set(wb)))))
      elif not shared:
        # No config to compare (analysis outputs, crashed adapts: ~6% of the
        # tree). Counters cover it; say so rather than implying content proof.
        held.append((name, x["bytes"],
                     "UNVERIFIABLE -- counters cover it, no config to compare"))
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
