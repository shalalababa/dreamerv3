#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Guided helper migration for a RUNNING instance  (dv3ops P0, 2026-08-14)
#
# Moves an instance from v1 inline helpers to repo-hosted helpers. This is the
# one dangerous step in P0, for a specific reason:
#
#   Cancelling a fit mid-run and resubmitting it re-enters via latest_ckpt.
#   That is the same resume path as the axis1 idempotent-skip hole, which
#   silently produced truncated training that was never recorded and only
#   surfaced months later in a design review. So this script's real job is not
#   cancel-and-resubmit -- that part is easy -- it is capturing the realized
#   step counters BEFORE cancelling, so the resume can be proven afterwards
#   instead of assumed.
#
# DRY-RUN BY DEFAULT. Nothing is cancelled unless you re-run with:
#     CONFIRM=MIGRATE scripts/ops/migrate_instance.sh <N>
#
# Preferred alternative: migrate at a natural drain point (no active queue),
# which skips the resume question entirely. This script detects that case and
# takes the easy path automatically.
# ---------------------------------------------------------------------------
set -uo pipefail

DV3OPS_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"

N="${1:?usage: migrate_instance.sh <instance>}"
CONFIRM="${CONFIRM:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SNAP="$DV3OPS_ROOT/ops/state/migration/inst${N}_${STAMP}"

dv3_require_instance "$N"
mkdir -p "$SNAP"

echo "=============================================================="
echo " dv3ops migrate -- instance $N   ($(dv3_ip "$N"):$(dv3_port "$N"))"
echo " snapshot dir: $SNAP"
[ "$CONFIRM" = "MIGRATE" ] && echo " mode: EXECUTE" || echo " mode: DRY-RUN (set CONFIRM=MIGRATE to execute)"
echo "=============================================================="

# --- 1. snapshot ------------------------------------------------------------
echo
echo "--- 1. snapshot lane state + realized counters ---"

dv3_ssh_dv3 "$N" 'dv3_status --json' 2>/dev/null | tail -1 > "$SNAP/status.json"
if [ ! -s "$SNAP/status.json" ]; then
  # v1 helpers have no --json; fall back to the human report.
  dv3_ssh_dv3 "$N" 'dv3_status' > "$SNAP/status.txt" 2>&1 ||
    die "cannot reach instance $N"
  echo "note: instance is on v1 helpers (no --json); captured human status instead"
fi

# Full queue control state: tasks.txt, next_index, running_task per lane. This
# is what the resubmission is reconstructed from, so it is copied verbatim
# rather than parsed.
dv3_ssh_dv3 "$N" '
  for lane in "$RUNROOT"/_queue_control/lane_*; do
    [ -f "$lane/active_queue" ] || continue
    gpu="${lane##*/lane_}"
    active="$(cat "$lane/active_queue")"
    qdir="$lane/$active"
    echo "===LANE $gpu $active next=$(cat "$qdir/next_index" 2>/dev/null) total=$(wc -l < "$qdir/tasks.txt" 2>/dev/null) running=$(cat "$qdir/running_index" 2>/dev/null)"
    echo "===TASKS"
    cat "$qdir/tasks.txt"
    echo "===END"
  done
' > "$SNAP/lanes.txt" 2>&1

active_lanes="$(grep -c '^===LANE ' "$SNAP/lanes.txt" 2>/dev/null || echo 0)"
echo "active lanes: $active_lanes"

# Realized counters for every run dir this instance has touched. Markers are
# NOT sufficient evidence: record the ckpt step and score-line counts, which
# are what a later verification can actually compare.
dv3_ssh_dv3 "$N" '
python - <<'"'"'PY'"'"'
import os, re, json, glob
rr = os.environ["RUNROOT"]
out = {}
for d in sorted(glob.glob(os.path.join(rr, "*"))):
    if not os.path.isdir(d) or os.path.basename(d).startswith("_"):
        continue
    run = os.path.basename(d)
    steps = []
    for p in glob.glob(os.path.join(d, "ckpt", "*")):
        if os.path.exists(os.path.join(p, "done")):
            m = re.search(r"(\d+)$", os.path.basename(p))
            if m:
                steps.append(int(m.group(1)))
    sc = os.path.join(d, "scores.jsonl")
    nsc = sum(1 for _ in open(sc, errors="ignore")) if os.path.exists(sc) else 0
    markers = [m for m in ("TRAINING_DONE", "ADAPT_DONE", "DISTILL_DONE")
               if os.path.exists(os.path.join(d, m))]
    prog = None
    pf = os.path.join(d, "OFFLINE_FIT_PROGRESS")
    if os.path.exists(pf):
        for line in open(pf, errors="ignore"):
            if line.startswith("update="):
                try: prog = int(float(line.split("=", 1)[1]))
                except ValueError: pass
    if steps or nsc or markers or prog is not None:
        out[run] = {"ckpt_step": max(steps) if steps else 0,
                    "n_scores": nsc, "markers": markers, "progress": prog}
print(json.dumps(out, indent=1, sort_keys=True))
PY
' > "$SNAP/realized_before.json" 2>"$SNAP/realized_before.err"

if [ -s "$SNAP/realized_before.json" ]; then
  echo "realized-counter baseline: $(grep -c '"ckpt_step"' "$SNAP/realized_before.json" 2>/dev/null || echo 0) run dirs"
else
  echo "WARN: realized-counter snapshot empty -- see $SNAP/realized_before.err"
  echo "      do NOT proceed to EXECUTE without this baseline."
fi

# --- 2. classify the situation ---------------------------------------------
echo
echo "--- 2. situation ---"
if [ "$active_lanes" -eq 0 ]; then
  echo "No active queue: this is a DRAIN POINT. Migration is safe and needs no"
  echo "resubmission -- nothing is mid-run, so the latest_ckpt resume question"
  echo "does not arise. Recommended path."
  MODE=drain
else
  echo "$active_lanes lane(s) active. Cancelling will stop the running task on"
  echo "each; work since its last checkpoint is lost, and the resubmitted run"
  echo "will resume from latest_ckpt."
  echo
  grep '^===LANE ' "$SNAP/lanes.txt" | sed 's/^===LANE /  lane /'
  echo
  echo "RUNNING TASKS THAT WILL BE INTERRUPTED (verify these afterwards):"
  dv3_ssh_dv3 "$N" '
    for lane in "$RUNROOT"/_queue_control/lane_*; do
      [ -f "$lane/active_queue" ] || continue
      qdir="$lane/$(cat "$lane/active_queue")"
      t="$(cat "$qdir/running_task" 2>/dev/null)"
      [ -n "$t" ] && echo "  lane ${lane##*/lane_}: $(dv3_task_name "$(cat "$qdir/running_index" 2>/dev/null || echo 0)" "$t")"
    done
  ' 2>/dev/null | tee "$SNAP/interrupted.txt"
  MODE=cancel
fi

# --- 3. execute -------------------------------------------------------------
echo
if [ "$CONFIRM" != "MIGRATE" ]; then
  echo "--- 3. DRY-RUN: stopping here ---"
  echo
  echo "Snapshot written to: $SNAP"
  echo "To execute:  CONFIRM=MIGRATE scripts/ops/migrate_instance.sh $N"
  [ "$MODE" = cancel ] && cat <<EOF

Before you do: consider waiting for a drain point instead. The remaining
queued tasks can be re-added to a fresh queue at any time, but a fit that is
80% through its 500k updates loses everything since its last checkpoint.
EOF
  exit 0
fi

echo "--- 3. EXECUTE ---"

if [ "$MODE" = cancel ]; then
  echo "cancelling all lanes..."
  dv3_ssh_dv3 "$N" 'dv3_cancel_queue all' || die "cancel failed; instance left untouched-ish, inspect manually"
  sleep 3
fi

echo "pushing repo (installs helpers + watchdog)..."
dv3_push_repo "$N" || die "push-repo failed"

echo "verifying helper version..."
dv3_ssh_dv3 "$N" 'dv3_version; type dv3_event >/dev/null 2>&1 && echo "dv3_event: present" || echo "dv3_event: MISSING"'

# --- 4. resubmission material ----------------------------------------------
if [ "$MODE" = cancel ]; then
  echo
  echo "--- 4. resubmission material ---"
  # Reconstruct the un-run remainder per lane: indices >= next_index, minus the
  # ones already marked #removed#. The interrupted task is INCLUDED (index ==
  # running_index was claimed but not completed).
  awk '
    /^===LANE /   { lane=$2; nx=0; for(i=1;i<=NF;i++) if ($i ~ /^next=/) { split($i,a,"="); nx=a[2] } ; n=0; next }
    /^===TASKS$/  { intask=1; n=0; next }
    /^===END$/    { intask=0; next }
    intask        { n++; if (n >= nx && $0 !~ /^#removed# /) print > ("'"$SNAP"'/resume_lane_" lane ".cmds") }
  ' "$SNAP/lanes.txt"
  for f in "$SNAP"/resume_lane_*.cmds; do
    [ -e "$f" ] || continue
    echo "  $(basename "$f"): $(wc -l < "$f") task(s)"
  done
  cat > "$SNAP/RESUME.md" <<EOF
# Resume after migration -- instance $N, $STAMP

Per-lane remainders were reconstructed from the pre-cancel snapshot.
Copy each to the instance and requeue on the SAME lane number:

    scp -P $(dv3_port "$N") $SNAP/resume_lane_0.cmds root@$(dv3_ip "$N"):/tmp/
    dv3ops sh $N 'dv3_queue_cmds_list /tmp/resume_lane_0.cmds 0'

Completed tasks re-skip via their own markers; the interrupted task resumes
from latest_ckpt.

## MANDATORY verification (do not skip)

Baseline: \`realized_before.json\` in this directory.
After the resumed runs finish, re-run the same counter snapshot and assert,
for every run listed in \`interrupted.txt\`:

  * ckpt_step_after >= ckpt_step_before   (never went backwards), AND
  * ckpt_step_after >= the wave's target step (500000 for standard fits)

A run that stops at ckpt_step_before is the truncation failure: it resumed,
declared itself done, and trained nothing. That is exactly the silent defect
this project exists to prevent, and it will not announce itself.
EOF
  echo
  echo "resume instructions: $SNAP/RESUME.md"
fi

echo
echo "=============================================================="
echo " migration complete for instance $N"
[ "$MODE" = cancel ] && echo " NEXT: requeue per $SNAP/RESUME.md, then verify counters"
echo " snapshot: $SNAP"
echo "=============================================================="
