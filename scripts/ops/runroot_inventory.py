#!/usr/bin/env python3
"""Emit realized counters for every top-level entry in a runroot.  (2026-08-15)

Written to answer exactly one question: *is what lives on this instance also
somewhere safe?* It is deliberately self-contained (stdlib only, no repo
imports) so the same file can be piped to a rented instance, to RCC, or run
locally and produce comparable JSON on all three.

Counters, not markers -- the same rule the wave checker follows. A directory is
described by what it actually contains (checkpoint step, score lines, file
count, bytes), because a name matching on both sides proves nothing about
whether the bytes arrived.

Usage:
  runroot_inventory.py <runroot> [--exclude name ...] [--only name ...]

--only restricts the walk to the named entries. The archive check asks
"is what lives on THIS INSTANCE also on RCC?", which needs counters for the
~40 names the instance holds, not for all 2300 dirs in RCC scratch. Walking
the whole 591 GB tree took 249 s per call (measured 2026-08-21) and every
`dv3ops destroy` paid it two or three times. Restricting the RCC side to the
instance's own names is not a weaker check: a name absent from --only is a
name the instance does not have, so it cannot be data at risk.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# `<timestamp>-<step>` carries a step; a bare `<timestamp>` does not. Matching
# trailing digits on the bare form reads the microsecond field as a training
# step -- see scripts/ops/wavecheck.py for the incident this mirrors.
TIMESTAMP_ONLY_RE = re.compile(r"^\d{8}T\d{6}F\d+$")
STEP_RE = re.compile(r"(\d+)$")

# Operational scaffolding, not results. These are recreated by the queue and
# the watchdog on any instance and are never worth protecting.
DEFAULT_EXCLUDE = {"_cloud_logs", "_queue_control", "_waves", "lost+found"}


def ckpt_step(d: str) -> int:
  best = 0
  for p in glob.glob(os.path.join(d, "ckpt", "*")):
    if not os.path.exists(os.path.join(p, "done")):
      continue
    name = os.path.basename(p)
    if TIMESTAMP_ONLY_RE.match(name):
      continue
    m = STEP_RE.search(name)
    if m:
      best = max(best, int(m.group(1)))
  return best


def count_lines(path: str) -> int:
  try:
    with open(path, errors="ignore") as fh:
      return sum(1 for _ in fh)
  except OSError:
    return 0


def walk(d: str) -> tuple[int, int]:
  """(file count, total bytes). Symlinks counted, never followed."""
  n = size = 0
  for root, dirs, files in os.walk(d, followlinks=False):
    for f in files:
      p = os.path.join(root, f)
      n += 1
      try:
        size += os.lstat(p).st_size
      except OSError:
        pass
  return n, size


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("runroot")
  ap.add_argument("--exclude", nargs="*", default=[])
  ap.add_argument("--only", nargs="*", default=None,
                  help="restrict to these entries (the caller's own names)")
  a = ap.parse_args()

  root = a.runroot
  if not os.path.isdir(root):
    print(json.dumps({"error": f"no such runroot: {root}"}))
    sys.exit(1)

  skip = DEFAULT_EXCLUDE | set(a.exclude)
  # --only is a filter, never a source: a requested name that is absent stays
  # absent from the output, which is what makes the caller report it MISSING
  # rather than silently treating it as covered.
  want = set(a.only) if a.only is not None else None
  out = {}
  for name in sorted(os.listdir(root)):
    if name in skip:
      continue
    if want is not None and name not in want:
      continue
    p = os.path.join(root, name)
    if not os.path.isdir(p):
      continue
    n_files, n_bytes = walk(p)
    out[name] = {
        "ckpt_step": ckpt_step(p),
        "n_scores": count_lines(os.path.join(p, "scores.jsonl")),
        "markers": [m for m in ("TRAINING_DONE", "ADAPT_DONE", "DISTILL_DONE")
                    if os.path.exists(os.path.join(p, m))],
        "n_files": n_files,
        "bytes": n_bytes,
    }
  print(json.dumps({"runroot": root, "entries": out}))


if __name__ == "__main__":
  main()
