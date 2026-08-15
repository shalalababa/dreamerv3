#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pull a finished bundle RCC -> local, verify, stage its manifest.  (P1, 14 Aug)
#
# Runs on LOCAL WSL, the one step that genuinely belongs there: local_results/,
# manifests/ and git live here. Everything else in dv3ops runs on RCC.
#
#   pull_local.sh --wave <wave_id>       # prefix comes from the spec
#   pull_local.sh --prefix lewm_jp       # or name it directly
#
# Connection sharing is not a nicety: without it the "find latest bundle" ssh
# and the rsync each authenticate separately, so you type the RCC password
# twice for one transfer.
#
# Verification is the point of the whole flow. The manifest is generated at the
# source, travels inside the bundle, and is re-checked here against the actual
# bytes; only then is a copy staged under manifests/ as the committed record.
# A transfer that merely finished is not evidence -- policy v2, and the reason
# every frozen read verifies before executing.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "$ROOT"

RCC_HOST="${DV3_RCC_HOST:-rickybao@midway3-login3.rcc.uchicago.edu}"
RCC_RUNROOT="${DV3_RCC_RUNROOT:-/scratch/midway3/rickybao/dreamerv3_runs}"

WAVE=""; PREFIX=""; KEEP_GOING=0
while [ $# -gt 0 ]; do
  case "$1" in
    --wave)   WAVE="${2:?--wave needs a wave id}"; shift 2 ;;
    --prefix) PREFIX="${2:?--prefix needs a value}"; shift 2 ;;
    --host)   RCC_HOST="${2:?}"; shift 2 ;;
    --force)  KEEP_GOING=1; shift ;;
    *) echo "usage: pull_local.sh (--wave W | --prefix P) [--host H] [--force]" >&2; exit 2 ;;
  esac
done

if [ -z "$PREFIX" ]; then
  [ -n "$WAVE" ] || { echo "ERROR: need --wave or --prefix" >&2; exit 2; }
  PREFIX="$(python3 - "$ROOT/ops/waves/$WAVE" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[1]), "..", "..", "scripts", "ops"))
sys.path.insert(0, "scripts/ops")
import wavespec
spec = wavespec.load(sys.argv[1])
if not spec.bundle:
    sys.exit("ERROR: this wave's spec declares no bundle, so there is nothing to pull")
print(spec.bundle.prefix)
PY
)" || exit 2
  echo "[pull-local] wave $WAVE -> bundle prefix $PREFIX"
fi

mkdir -p "$HOME/.ssh/cm" local_results manifests
CTL="$HOME/.ssh/cm/%r@%h:%p"
SSH_OPTS=(-o ControlMaster=auto -o ControlPersist=12h -o "ControlPath=$CTL")

# Best-effort master; if it cannot open, later commands still work and may prompt.
ssh -fN "${SSH_OPTS[@]}" "$RCC_HOST" 2>/dev/null || true

BUNDLE="$(ssh "${SSH_OPTS[@]}" "$RCC_HOST" \
  "ls -td $RCC_RUNROOT/bundles/${PREFIX}_* 2>/dev/null | head -1")"
[ -n "$BUNDLE" ] || { echo "ERROR: no bundle matching ${PREFIX}_* under $RCC_RUNROOT/bundles" >&2
                      echo "       run 'dv3ops bundle --wave $WAVE' on RCC first" >&2; exit 1; }
NAME="$(basename "$BUNDLE")"
echo "[pull-local] latest bundle: $NAME"

if [ -d "local_results/$NAME" ] && [ "$KEEP_GOING" -eq 0 ]; then
  echo "[pull-local] local_results/$NAME already exists; resuming transfer"
fi

rsync -az --partial --append-verify --info=progress2 --stats \
  -e "ssh ${SSH_OPTS[*]}" \
  "$RCC_HOST:$BUNDLE/" "local_results/$NAME/"

echo "[pull-local] verifying manifest against the bytes that actually arrived"
if ! ./scripts/bundle_manifest.sh verify "local_results/$NAME"; then
  echo "ERROR: verification FAILED for local_results/$NAME" >&2
  echo "       Nothing has been staged. Do not read from this bundle: a" >&2
  echo "       mismatch means the local copy is not the bytes the manifest" >&2
  echo "       pins, and re-running the transfer is cheaper than a wrong read." >&2
  exit 1
fi

cp "local_results/$NAME/MANIFEST.sha256" "manifests/$NAME.sha256"
git add "manifests/$NAME.sha256"

echo
echo "[pull-local] VERIFIED  local_results/$NAME"
echo "[pull-local] staged    manifests/$NAME.sha256"
echo
echo "Commit the manifest to pin these exact bytes in the registration record:"
echo "  git commit -m \"manifest: $NAME\""
