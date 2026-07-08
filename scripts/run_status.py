#!/usr/bin/env python3
"""Summarize bundled run state across runs.csv, Slurm, markers, and logs.

This is intentionally read-only by default.  It is meant for the messy middle
of bundled submissions, where runs.csv may say RUNNING, the Slurm bundle may
have been cancelled, and the child directory may contain only a reservation
marker.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


BASELINE_ADAPT = re.compile(r"^adapt_(apt|p2e|random)_")


@dataclass
class RunInfo:
  run_id: str
  phase: str = ""
  manifest_status: str = ""
  manifest_logdir: str = ""
  bundle: str = ""
  marker_bundle: str = ""
  bundle_state: str = ""
  child_stage: str = ""
  progress: str = ""
  done: bool = False
  failed: bool = False
  logdir_state: str = ""
  note: str = ""
  extras: dict = field(default_factory=dict)


def run_cmd(cmd: list[str]) -> str:
  try:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
  except Exception:
    return ""


def read_keyvals(path: Path) -> dict[str, str]:
  out = {}
  if not path.exists():
    return out
  for line in path.read_text(errors="ignore").splitlines():
    if "=" in line:
      k, v = line.split("=", 1)
      out[k] = v
  return out


def manifest_rows(manifest: Path) -> dict[str, dict[str, str]]:
  if not manifest.exists():
    return {}
  rows = {}
  with manifest.open(newline="") as f:
    for row in csv.DictReader(f):
      if row.get("run_id"):
        rows[row["run_id"]] = row
  return rows


def current_slurm(user: str) -> dict[str, str]:
  out = run_cmd(["squeue", "-h", "-u", user, "-o", "%j|%T|%M|%R|%N"])
  result = {}
  for line in out.splitlines():
    parts = line.split("|", 4)
    if len(parts) == 5:
      name, state, elapsed, reason, nodes = parts
      result[name] = f"{state}/{elapsed}/{reason}/{nodes}"
  return result


def recent_sacct(user: str, start: str) -> dict[str, str]:
  out = run_cmd([
      "sacct", "-u", user, "--starttime", start, "-X", "-P", "-n",
      "--format=JobName%90,State,Elapsed,ExitCode"])
  result = {}
  for line in out.splitlines():
    parts = line.split("|")
    if len(parts) >= 4:
      name, state, elapsed, exitcode = parts[:4]
      if name and name not in result:
        result[name] = f"{state}/{elapsed}/exit={exitcode}"
  return result


def bundle_runlists(runroot: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
  child_to_bundles: dict[str, list[str]] = {}
  bundle_to_runlist: dict[str, str] = {}

  # Started bundles copy their runlist here.
  for path in sorted((runroot / "_bundles").glob("*/runlist.tsv")):
    bundle = path.parent.name
    bundle_to_runlist[bundle] = str(path)
    for child in children_from_runlist(path):
      child_to_bundles.setdefault(child, []).append(bundle)

  # Submitted but not yet started bundles may only be visible through markers,
  # whose runlist paths point into _submit_runlists.
  for marker in sorted(runroot.glob("*/SUBMITTED_BY_BUNDLE")):
    kv = read_keyvals(marker)
    bundle = kv.get("bundle", "")
    runlist = kv.get("runlist", "")
    child = marker.parent.name
    if bundle:
      child_to_bundles.setdefault(child, []).append(bundle)
    if bundle and runlist:
      bundle_to_runlist.setdefault(bundle, runlist)
  return child_to_bundles, bundle_to_runlist


def children_from_runlist(path: Path) -> list[str]:
  children = []
  if not path.exists():
    return children
  for line in path.read_text(errors="ignore").splitlines():
    if not line.strip():
      continue
    cols = line.split("\t")
    # adapt/axis1 runlists use child run id in col 0; measure uses pre_run.
    children.append(cols[0])
  return children


def bundle_out_files(repo: Path, bundle: str) -> list[Path]:
  paths = list(repo.glob(f"{bundle}_*.out"))
  paths += [Path(p) for p in glob.glob(f"{bundle}_*.out")]
  uniq = {}
  for p in paths:
    uniq[str(p.resolve())] = p
  return sorted(uniq.values(), key=lambda p: p.stat().st_mtime if p.exists() else 0)


def parse_bundle_log(repo: Path, bundle: str, run_id: str) -> tuple[str, str, str]:
  stage = ""
  progress = ""
  note = ""
  for path in bundle_out_files(repo, bundle):
    text = path.read_text(errors="ignore")
    starts = re.findall(r"START child (\S+)", text)
    dones = set(re.findall(r"DONE child (\S+)", text))
    fails = {m.group(1): m.group(2) for m in re.finditer(r"FAILED child (\S+) rc=(\d+)", text)}
    if run_id in dones:
      stage = "bundle_child_done"
    elif run_id in fails:
      stage = f"bundle_child_failed:{fails[run_id]}"
    elif starts and starts[-1] == run_id:
      stage = "bundle_child_running"
      updates = re.findall(r"update\s+(\d+)\s*/\s*(\d+)", text)
      if updates:
        a, b = updates[-1]
        progress = f"wm_update={a}/{b}"
      elif "offline_fit WM_RUN=" in text:
        progress = "offline_fit_started"
    elif run_id in starts and not stage:
      stage = "bundle_child_started"
    if path.name:
      note = f"out={path.name}"
  return stage, progress, note


def latest_done_ckpt(path: Path) -> str:
  ckpts = sorted(p.parent for p in path.glob("ckpt/*/done"))
  return str(ckpts[-1]) if ckpts else ""


def last_adapt_step(logdir: Path) -> str:
  scores = logdir / "scores.jsonl"
  if not scores.exists():
    return ""
  last = None
  try:
    for line in scores.open():
      if line.strip():
        last = json.loads(line)
  except Exception:
    return "scores_unreadable"
  if last and "step" in last:
    return f"adapt_step={last['step']}"
  return ""


def logdir_state(path: Path) -> str:
  if not path.exists():
    return "missing"
  children = [p.name for p in path.iterdir()]
  if not children:
    return "empty"
  non_markers = [
      x for x in children
      if x not in {"SUBMITTED_BY_BUNDLE", "BUNDLE_CHILD_STARTED",
                   "BUNDLE_CHILD_DONE", "BUNDLE_CHILD_FAILED"}]
  if not non_markers and "SUBMITTED_BY_BUNDLE" in children:
    return "marker_only"
  return "has_files"


def include_run(run_id: str, kind: str) -> bool:
  if kind == "all":
    return True
  if kind == "axis1":
    return run_id.startswith(("adapt_ax1", "ax1wm_"))
  if kind == "adapt":
    return run_id.startswith("adapt_")
  if kind == "measure":
    return run_id.startswith("measure_") or run_id.startswith("pretrain_")
  # interesting: user's usual filter plus the new enhancement modes.
  if run_id.startswith(("pretrain_", "pilot_", "smoke_")):
    return False
  if BASELINE_ADAPT.match(run_id):
    return False
  return True


def classify(info: RunInfo, active_names: dict[str, str]) -> str:
  if info.done:
    return "DONE"
  if info.failed:
    return "FAILED"
  active = bool(active_names.get(info.run_id) or active_names.get(info.bundle))
  if active:
    if info.child_stage in {"", "reserved"}:
      return "ACTIVE_NOT_STARTED"
    return "ACTIVE_STARTED"
  if info.manifest_status == "RUNNING":
    return "STALE_RUNNING"
  if info.logdir_state == "marker_only":
    return "STALE_MARKER"
  if info.logdir_state == "has_files":
    return "PARTIAL_OR_QUEUED"
  return info.manifest_status or "UNKNOWN"


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--runroot", default=os.environ.get("RUNROOT", ""))
  p.add_argument("--repo", default=os.environ.get("REPO", os.getcwd()))
  p.add_argument("--manifest", default=os.environ.get("MANIFEST", ""))
  p.add_argument("--kind", choices=["interesting", "axis1", "adapt", "measure", "all"],
                 default="interesting")
  p.add_argument("--since", default="now-7days",
                 help="sacct start time for recent completed bundle/job lookup")
  p.add_argument("--stale-only", action="store_true")
  p.add_argument("--tsv", action="store_true")
  p.add_argument("--emit-cleanup", action="store_true",
                 help="Print rm commands for stale marker-only reservations")
  args = p.parse_args()

  if not args.runroot:
    raise SystemExit("RUNROOT is not set; source scripts/env.sh first or pass --runroot.")
  runroot = Path(args.runroot)
  repo = Path(args.repo)
  manifest = Path(args.manifest) if args.manifest else runroot / "runs.csv"
  user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""

  rows = manifest_rows(manifest)
  squeue = current_slurm(user) if user else {}
  sacct = recent_sacct(user, args.since) if user else {}
  child_to_bundles, _ = bundle_runlists(runroot)

  run_ids = set(rows)
  run_ids.update(child_to_bundles)
  for path in runroot.glob("*"):
    if path.is_dir() and include_run(path.name, args.kind):
      run_ids.add(path.name)

  infos: list[tuple[str, RunInfo]] = []
  for run_id in sorted(run_ids):
    if not include_run(run_id, args.kind):
      continue
    row = rows.get(run_id, {})
    info = RunInfo(
        run_id=run_id,
        phase=row.get("phase", ""),
        manifest_status=row.get("status", ""),
        manifest_logdir=row.get("logdir", ""),
    )
    logdir = Path(info.manifest_logdir) if info.manifest_logdir else runroot / run_id
    marker = logdir / "SUBMITTED_BY_BUNDLE"
    marker_kv = read_keyvals(marker)
    info.marker_bundle = marker_kv.get("bundle", "")
    bundles = child_to_bundles.get(run_id, [])
    info.bundle = info.marker_bundle or (bundles[-1] if bundles else "")
    info.bundle_state = squeue.get(info.bundle) or sacct.get(info.bundle, "")
    info.logdir_state = logdir_state(logdir)
    info.done = (logdir / "ADAPT_DONE").exists() or info.manifest_status == "DONE"
    info.failed = (logdir / "BUNDLE_CHILD_FAILED").exists() or info.manifest_status == "FAILED"

    stage_file = logdir / "AXIS1_STAGE"
    if stage_file.exists():
      info.child_stage = stage_file.read_text(errors="ignore").strip()
    elif (logdir / "BUNDLE_CHILD_DONE").exists():
      info.child_stage = "bundle_child_done"
    elif (logdir / "BUNDLE_CHILD_STARTED").exists():
      info.child_stage = "bundle_child_started"
    elif marker.exists():
      info.child_stage = "reserved"

    if info.bundle:
      log_stage, log_progress, log_note = parse_bundle_log(repo, info.bundle, run_id)
      info.child_stage = info.child_stage or log_stage
      info.progress = log_progress
      info.note = log_note

    info.progress = info.progress or last_adapt_step(logdir)
    if run_id.startswith("ax1wm_"):
      ckpt = latest_done_ckpt(runroot / run_id)
      if ckpt:
        info.progress = f"wm_done={Path(ckpt).name}"
        info.done = True

    state = classify(info, squeue)
    if args.stale_only and not state.startswith("STALE"):
      continue
    infos.append((state, info))

  if args.emit_cleanup:
    for state, info in infos:
      if state == "STALE_MARKER":
        print(f"rm -rf {runroot / info.run_id}")
    return 0

  header = ["state", "run_id", "phase", "manifest", "logdir", "bundle",
            "bundle_state", "stage", "progress", "note"]
  records = []
  for state, i in infos:
    records.append([state, i.run_id, i.phase, i.manifest_status or "EMPTY",
                    i.logdir_state, i.bundle, i.bundle_state,
                    i.child_stage, i.progress, i.note])
  if args.tsv:
    print("\t".join(header))
    for rec in records:
      print("\t".join(str(x) for x in rec))
  else:
    widths = [max(len(str(row[c])) for row in [header] + records) for c in range(len(header))]
    print(" ".join(h.ljust(widths[c]) for c, h in enumerate(header)))
    for rec in records:
      print(" ".join(str(x).ljust(widths[c]) for c, x in enumerate(rec)))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
