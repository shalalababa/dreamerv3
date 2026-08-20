#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Queue ONE two-seed block of u1a_pooled onto ONE instance.
# Same reason as carrier_freshrep/submit_block.sh: generated/submit.sh queues
# every stage's lanes onto whichever instance it is handed, which for this
# wave would put all 32 cells on one box. A block is 8 cells across two
# stages (task + apt), and a seed's four cells must not straddle machines --
# the primary is a per-seed interaction.
#
#   usage: submit_block.sh <block 1-4> <instance>
# ---------------------------------------------------------------------------
set -euo pipefail
B="${1:?usage: submit_block.sh <block 1-4> <instance>}"
N="${2:?usage: submit_block.sh <block 1-4> <instance>}"
case "$B" in 1|2|3) ;; *) echo "block must be 1-3, got: $B" >&2; exit 2 ;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; GEN="$HERE/generated"
DV3OPS_ROOT="${DV3OPS_ROOT:-/home/rickybao/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"
dv3_require_instance "$N"

LANES="$(ls "$GEN"/lane_blk${B}_task_*.cmds 2>/dev/null | sed 's/.*_//; s/\.cmds//' | sort -n | tr '\n' ' ')"
[ -n "$LANES" ] || { echo "no lane files for block $B -- run dv3ops gen" >&2; exit 2; }
echo "lanes in this block: $LANES"

# A block generated for N lanes must not be queued onto a box with fewer
# GPUs: lane k maps to CUDA_VISIBLE_DEVICES=k, so the surplus lanes would be
# pinned to devices that do not exist. The fleet now mixes 2-GPU and 4-GPU
# boxes, so this is a live hazard rather than a hypothetical.
NGPU="$(dv3_ssh_dv3 "$N" 'nvidia-smi --query-gpu=name --format=csv,noheader | wc -l')"
NLANES="$(echo $LANES | wc -w)"
if [ "$NLANES" -gt "${NGPU:-0}" ]; then
  echo "REFUSE: block $B was generated for $NLANES lanes but instance $N has ${NGPU:-?} GPU(s)." >&2
  echo "        Regenerate with --lanes covering only this box's devices." >&2
  exit 2
fi
echo "instance $N has $NGPU GPU(s) for $NLANES lane(s)"

REMOTE_ROOT="$(dv3_ssh_dv3 "$N" 'printf %s "$RUNROOT"')"
[ -n "$REMOTE_ROOT" ] || { echo "ERROR: cannot read RUNROOT on instance $N" >&2; exit 1; }
REMOTE="$REMOTE_ROOT/_waves/u1a_pooled"
echo "== u1a block $B -> instance $N  (seeds: $(grep -ohE 'seed[0-9]+' "$GEN"/lane_blk${B}_*.cmds | sed 's/seed//' | sort -n -u | tr '\n' ' '))"
dv3_ssh_dv3 "$N" "mkdir -p \"$REMOTE\""
rsync -az --checksum -e "ssh -p $(dv3_port "$N") ${DV3_SSH_OPTS[*]}" \
  "$GEN/" "root@$(dv3_ip "$N"):$REMOTE/"

# Queue the apt lanes FIRST: an apt cell is ~6.25 h against ~3.75 h for task,
# so starting the long pole first keeps the two lanes from finishing far apart.
for arm in apt task; do for l in $LANES; do
  f="lane_blk${B}_${arm}_${l}.cmds"
  echo "-- blk$B $arm lane $l: $(( $(grep -c . "$GEN/$f") - 1 )) task(s) --"
  dv3_ssh_dv3 "$N" "DV3_ABORT_ON_FAIL=0 dv3_queue_or_add \"$REMOTE/$f\" $l"
done; done
echo "queued u1a block $B on instance $N. verify: dv3ops status $N"
