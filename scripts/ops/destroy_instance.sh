#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Destroy a rented Vast instance -- but only once its data is provably
# somewhere else.  (2026-08-15)
#
#   dv3ops destroy N                 # REPORT only. Never destroys.
#   dv3ops destroy N --yes           # destroy, after the archive gate + prompt
#   dv3ops destroy N --force-unsafe  # destroy WITH unarchived data (typed ack)
#
# `vastai destroy instance` is described by Vast itself as "irreversible,
# deletes data". This wrapper exists so that irreversibility is only ever
# reached through a check, not through a memory of having synced.
#
# The standing rule this enforces (2026-08-08, after the delete-only incident
# that cost the rl/sh/vgo world models, the U1 uzf adapts and rgo/sgb seeds
# 9-16 -- gone for good): nothing is deleted without a VERIFIED archive
# elsewhere. Deleting an instance deletes every run dir on it, so the same rule
# applies here even though no `rm` is typed.
#
# What "archived" means here: every top-level run dir on the instance must
# exist on RCC with counters at least as large -- checkpoint step, score lines,
# file count and bytes. Greater-or-equal, not equal: RCC legitimately
# accumulates runs from several instances, and a bundle may add files.
#
# Creation is deliberately NOT here. Picking an offer is a price judgement and
# stays in the web UI; this tool only ever spends down, never up.
# ---------------------------------------------------------------------------
set -uo pipefail

DV3OPS_ROOT="${DV3OPS_ROOT:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"

RCC_HOST="${DV3_RCC_HOST:-rickybao@midway3-login3.rcc.uchicago.edu}"
RCC_RUNROOT="${DV3_RCC_RUNROOT:-/scratch/midway3/rickybao/dreamerv3_runs}"
CTL="$HOME/.ssh/cm/%r@%h:%p"
# -n is load-bearing: without it this ssh reads the SCRIPT's stdin, and the
# confirmation prompt further down then reads an already-empty stream and
# aborts with "got ''" -- or, worse on a terminal, silently eats the keystrokes
# meant for the prompt. Same §0.9 hazard dv3_ssh guards against; it is easy to
# reintroduce every time a new bare ssh is added (hit here 2026-08-15).
RCC_SSH=(-n -o BatchMode=yes -o ControlMaster=no -o "ControlPath=$CTL")

N=""; DO_IT=0; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --yes)          DO_IT=1; shift ;;
    --force-unsafe) DO_IT=1; FORCE=1; shift ;;
    -*) die "unknown option $1 (usage: dv3ops destroy N [--yes|--force-unsafe])" ;;
    *)  N="$1"; shift ;;
  esac
done
[ -n "$N" ] || die "usage: dv3ops destroy N [--yes|--force-unsafe]"

dv3_require_instance "$N"
command -v vastai >/dev/null 2>&1 || die "vastai CLI not found. Destroy runs on
the machine that owns the inventory (WSL), not on RCC."

STATE="$DV3OPS_STATE/instances.json"
[ -f "$STATE" ] || die "no $STATE -- run 'dv3ops refresh' first"
VAST_ID="$(python3 -c "
import json,sys
d=json.load(open('$STATE'))
for i in d.get('instances',[]):
    if int(i.get('index',-1))==int('$N'): print(i.get('vast_id','')); break
")"
[ -n "$VAST_ID" ] || die "instance $N has no vast_id in $STATE -- run 'dv3ops refresh'"

echo "=============================================================="
echo " DESTROY instance $N   (vast id $VAST_ID, $(dv3_ip "$N"):$(dv3_port "$N"))"
echo "=============================================================="

# -- gate 1: nothing may be in flight --------------------------------------
echo
echo "== work in flight =="
busy="$(dv3_ssh_dv3 "$N" '
  n=0
  for q in "$RUNROOT"/_queue_control/lane_*/queue_*; do
    [ -d "$q" ] || continue
    pid="$(cat "$q/supervisor.pid" 2>/dev/null || echo)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && n=$((n+1))
  done
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null \
         | tr -d " %" | sort -rn | head -1)
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
          | grep -c . || true)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null \
        | tr -d " MiB" | sort -rn | head -1)
  echo "queues=$n procs=${procs:-0} util=${util:-0} mem=${mem:-0}"')"
echo "  $busy"
q_live="$(printf '%s' "$busy" | sed -n 's/.*queues=\([0-9]*\).*/\1/p')"
p_live="$(printf '%s' "$busy" | sed -n 's/.*procs=\([0-9]*\).*/\1/p')"
u_live="$(printf '%s' "$busy" | sed -n 's/.*util=\([0-9]*\).*/\1/p')"
m_live="$(printf '%s' "$busy" | sed -n 's/.*mem=\([0-9]*\).*/\1/p')"
in_flight=0
[ "${q_live:-0}" -gt 0 ] && in_flight=1
[ "${p_live:-0}" -gt 0 ] && in_flight=1
# Utilization ALONE is not evidence of work. Consumer cards leave the busy
# percentage latched at 100 for minutes after a job is killed -- instance 4
# reported GPUs 1 and 2 at 100% with 2 MiB allocated and zero compute
# processes (2026-08-16), which blocked a destroy whose work was already
# cancelled and archived. A real training job holds GPU memory, so utilization
# only counts when memory is held too. Kept as a second path (rather than
# dropping util entirely) because some container images hide the compute-apps
# list, and there memory is the only witness left.
[ "${u_live:-0}" -ge 20 ] && [ "${m_live:-0}" -ge 512 ] && in_flight=1
if [ "$in_flight" -eq 1 ]; then
  echo "  WARNING: this instance is still working (live supervisors or a busy GPU)."
  [ "$FORCE" -eq 1 ] || die "refusing to destroy an instance with work in flight.
       Let it drain, or cancel its lanes deliberately first."
fi

# -- gate 2: every run dir must exist on RCC with >= counters ---------------
echo
echo "== archive check: instance vs RCC =="
INV="$DV3OPS_ROOT/scripts/ops/runroot_inventory.py"
[ -f "$INV" ] || die "missing $INV"

inst_json="$(dv3_ssh_dv3 "$N" "python3 - \"\$RUNROOT\" <<'PYEOF'
$(cat "$INV")
PYEOF" 2>/dev/null | tail -1)"
case "$inst_json" in '{'*) ;; *) die "could not inventory instance $N (got: ${inst_json:0:120})" ;; esac

rcc_json="$(ssh "${RCC_SSH[@]}" "$RCC_HOST" "python3 - '$RCC_RUNROOT' <<'PYEOF'
$(cat "$INV")
PYEOF" 2>/dev/null | tail -1)"
case "$rcc_json" in '{'*) ;; *) die "could not inventory RCC at $RCC_RUNROOT.
       Is the shared connection up?  dv3ops rcc-connect" ;; esac

printf '%s' "$inst_json" > /tmp/dv3_inv_inst_$$.json
printf '%s' "$rcc_json"  > /tmp/dv3_inv_rcc_$$.json
trap 'rm -f /tmp/dv3_inv_inst_$$.json /tmp/dv3_inv_rcc_$$.json' EXIT

report="$(python3 - /tmp/dv3_inv_inst_$$.json /tmp/dv3_inv_rcc_$$.json <<'PY'
import json, sys
inst = json.load(open(sys.argv[1]))["entries"]
rcc  = json.load(open(sys.argv[2]))["entries"]
FIELDS = ("ckpt_step", "n_scores", "n_files", "bytes")
covered, missing, short = [], [], []
for name, a in sorted(inst.items()):
    b = rcc.get(name)
    if b is None:
        missing.append(name); continue
    bad = [f for f in FIELDS if b.get(f, 0) < a.get(f, 0)]
    (short if bad else covered).append(
        (name, ", ".join(f"{f} {a.get(f,0)}>{b.get(f,0)}" for f in bad)))
print(json.dumps({
    "n_inst": len(inst),
    "covered": [c[0] for c in covered],
    "missing": missing,
    "short": short,
}))
PY
)"
python3 - "$report" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
print(f"  {len(r['covered'])}/{r['n_inst']} dirs covered on RCC")
for n in r["missing"]:
    print(f"  NOT ON RCC     {n}")
for n, why in r["short"]:
    print(f"  SHORT ON RCC   {n}   ({why})")
PY
unsafe="$(python3 -c "
import json,sys; r=json.loads(sys.argv[1]); print(len(r['missing'])+len(r['short']))" "$report")"

if [ "$unsafe" -gt 0 ]; then
  echo
  echo "  $unsafe directory(ies) exist ONLY on this instance."
  echo "  Destroying now deletes the only copy. Pull them first:"
  echo "      dv3ops pull --wave <wave> $N          # on RCC"
  if [ "$FORCE" -ne 1 ]; then
    die "refusing to destroy: unarchived data (standing rule, 2026-08-08)"
  fi
else
  echo "  ALL CLEAR: every run dir on instance $N is on RCC with >= counters."
fi

# -- report-only unless --yes ----------------------------------------------
if [ "$DO_IT" -ne 1 ]; then
  echo
  echo "REPORT ONLY -- nothing destroyed."
  echo "  to proceed:  dv3ops destroy $N --yes"
  exit 0
fi

# -- confirmation ----------------------------------------------------------
echo
if [ "$FORCE" -eq 1 ] && [ "$unsafe" -gt 0 ]; then
  echo "!! You are destroying UNARCHIVED data. This cannot be undone."
  read -r -p "Type DESTROY UNARCHIVED to continue: " ack
  [ "$ack" = "DESTROY UNARCHIVED" ] || die "aborted"
fi
read -r -p "Type the vast id ($VAST_ID) to destroy instance $N: " ack
[ "$ack" = "$VAST_ID" ] || die "aborted (got '$ack')"

# -- record BEFORE destroying ----------------------------------------------
mkdir -p "$DV3OPS_STATE"
python3 - "$DV3OPS_STATE/destroyed.jsonl" "$N" "$VAST_ID" "$report" <<'PY'
import json, sys, datetime
path, idx, vid, report = sys.argv[1:5]
rec = {"ts": datetime.datetime.now().astimezone().isoformat(),
       "index": int(idx), "vast_id": vid, "archive": json.loads(report)}
with open(path, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"  recorded -> {path}")
PY

echo "  vastai destroy instance $VAST_ID"
vastai destroy instance "$VAST_ID" -y || die "vastai destroy failed"
echo "  destroyed."

# Retire the index so it is never silently reused for a different machine.
echo
echo "== refreshing inventory (retires index $N) =="
python3 "$DV3OPS_ROOT/scripts/ops/refresh.py" || true
