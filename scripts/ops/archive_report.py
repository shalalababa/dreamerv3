#!/usr/bin/env python3
"""Archive coverage and headroom.  (dv3ops P4, 2026-08-14)

NOT a delete-safety classifier. Since the 14 Aug retention decision -- nothing
unarchived is deleted until the papers are published -- the question about a
run directory is "is it archived yet?", never "may I delete it?". So this
reports coverage and headroom, and `runroot_cleanup.sh`'s DELETE-SAFE tier is
dormant.

The rule it makes mechanical: a directory can only be called archived if a
committed manifest covers it. Discipline is what failed last time; the
rl/sh/vgo world models and the Amendment-1 seeds were all correctly classified
DELETE-SAFE and are still gone.

Sources joined here:
  du over $RUNROOT          what exists and how big
  manifests/*.sha256        what is archived AND byte-verified
  ops/waves/*/runs.json     what is a declared donor of a wave not yet complete

Usage:
  archive_report.py [--runroot DIR] [--top 40] [--json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def human(n: float) -> str:
  for unit in ("B", "K", "M", "G", "T"):
    if abs(n) < 1024:
      return f"{n:.0f}{unit}" if unit in "BK" else f"{n:.1f}{unit}"
    n /= 1024
  return f"{n:.1f}P"


def dir_sizes(runroot: Path) -> dict[str, int]:
  """One du pass; far faster than walking in python for tens of thousands of files."""
  out: dict[str, int] = {}
  try:
    res = subprocess.run(["du", "-sb", "--", *[str(p) for p in sorted(runroot.iterdir())
                                               if p.is_dir()]],
                         capture_output=True, text=True, timeout=1800)
    for line in res.stdout.splitlines():
      parts = line.split("\t")
      if len(parts) == 2:
        out[os.path.basename(parts[1])] = int(parts[0])
  except (OSError, subprocess.TimeoutExpired, ValueError):
    pass
  return out


def archived_runs(manifest_dir: Path) -> dict[str, list[str]]:
  """run_id -> manifests covering it, from committed manifest files.

  A manifest lists paths inside a bundle; the run directory name appears as a
  path component. Matching on that is coarse but errs toward "not archived",
  which is the safe direction here.
  """
  cov: dict[str, list[str]] = collections.defaultdict(list)
  for f in sorted(glob.glob(str(manifest_dir / "*.sha256"))):
    name = os.path.basename(f)[:-7]
    try:
      text = Path(f).read_text(errors="ignore")
    except OSError:
      continue
    for m in re.finditer(r"runroot/([^/\s]+)/", text):
      cov[m.group(1)].append(name)
  return cov


def wave_donors() -> dict[str, list[str]]:
  """dir -> waves that declare it as an input and are not yet complete."""
  need: dict[str, list[str]] = collections.defaultdict(list)
  for f in glob.glob(str(ROOT / "ops" / "waves" / "*" / "generated*" / "runs.json")):
    try:
      doc = json.load(open(f))
    except (OSError, json.JSONDecodeError):
      continue
    wave = doc.get("wave_id", os.path.basename(os.path.dirname(os.path.dirname(f))))
    for r in doc.get("runs", []):
      for i in r.get("inputs", []):
        need[i.split("/")[0]].append(wave)
  return {k: sorted(set(v)) for k, v in need.items()}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--runroot", default=os.environ.get("RUNROOT", ""))
  ap.add_argument("--manifests", default=str(ROOT / "manifests"))
  ap.add_argument("--top", type=int, default=40)
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args()

  if not args.runroot:
    print("ERROR: set --runroot or $RUNROOT", file=sys.stderr)
    return 2
  runroot = Path(args.runroot)
  if not runroot.is_dir():
    print(f"ERROR: no such runroot: {runroot}", file=sys.stderr)
    return 2

  sizes = dir_sizes(runroot)
  cov = archived_runs(Path(args.manifests))
  donors = wave_donors()

  rows = []
  for name, size in sizes.items():
    if name.startswith("_") or name == "bundles":
      continue
    rows.append({
        "dir": name, "bytes": size,
        "archived": bool(cov.get(name)),
        "manifests": cov.get(name, []),
        "donor_for": donors.get(name, []),
    })
  rows.sort(key=lambda r: -r["bytes"])

  unarchived = [r for r in rows if not r["archived"]]
  archived = [r for r in rows if r["archived"]]
  tot_un = sum(r["bytes"] for r in unarchived)
  tot_ar = sum(r["bytes"] for r in archived)

  if args.json:
    print(json.dumps({"rows": rows, "unarchived_bytes": tot_un,
                      "archived_bytes": tot_ar}, indent=1))
    return 0

  print(f"archive coverage for {runroot}")
  print(f"{len(rows)} run dir(s), {human(tot_ar + tot_un)} total\n")

  print(f"--- NO ARCHIVE ({len(unarchived)} dirs, {human(tot_un)}) ---")
  print("Nothing here may be deleted, and nothing here survives losing scratch.")
  for r in unarchived[:args.top]:
    flag = f"  donor -> {', '.join(r['donor_for'])}" if r["donor_for"] else ""
    print(f"  {human(r['bytes']):>8}  {r['dir']}{flag}")
  if len(unarchived) > args.top:
    print(f"  ... and {len(unarchived) - args.top} more "
          f"({human(sum(r['bytes'] for r in unarchived[args.top:]))})")

  print(f"\n--- ARCHIVED ({len(archived)} dirs, {human(tot_ar)}) ---")
  for r in archived[:10]:
    print(f"  {human(r['bytes']):>8}  {r['dir']}  [{r['manifests'][0]}]")
  if len(archived) > 10:
    print(f"  ... and {len(archived) - 10} more")

  # Headroom: the quota is the thing that forces the decision.
  try:
    st = os.statvfs(runroot)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    used_pct = 100 * (1 - free / total) if total else 0
    print(f"\n--- HEADROOM ---")
    print(f"  filesystem: {human(total - free)} used of {human(total)} ({used_pct:.0f}%)")
  except OSError:
    pass

  print("\nNext: archive the NO-ARCHIVE list (largest first), verify with")
  print("scripts/bundle_manifest.sh, commit the manifest, then re-run this.")
  print("A dir counts as archived only once a COMMITTED manifest covers it --")
  print("a finished transfer is not evidence.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
