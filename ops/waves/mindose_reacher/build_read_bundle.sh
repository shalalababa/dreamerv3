#!/usr/bin/env bash
# MIN-DOSE reacher read bundle — PREREG_mindose_reacher_20260820 (+amend1).
# Thin wrapper over the selfchecked builder (auc.csv + witness.json +
# buffer_manifest.json).
#
# ATTEST_FIRST_DRAW=1 is an OPS ATTESTATION (amend1 B10): set it only if
# this bundle's level-1 buffer came from the FIRST invocation of the pinned
# build-dose command. It never defaults on; without it the reader refuses.
#
#   RUNROOT=... DOSE_MANIFEST=<build_dose out>/manifest.json \
#     ATTEST_FIRST_DRAW=1 bash ops/waves/mindose_reacher/build_read_bundle.sh
set -uo pipefail
: "${RUNROOT:?set RUNROOT}"
: "${DOSE_MANIFEST:?set DOSE_MANIFEST (build_dose manifest.json)}"
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
stamp="$(date +%Y%m%d_%H%M%S)"
OUT="$RUNROOT/bundles/mindose_reacher_${stamp}"
ATTEST=""
[ "${ATTEST_FIRST_DRAW:-0}" = "1" ] && ATTEST="--attest_first_draw"
cd "$REPO"
python -m scripts.ops.build_rescue_bundle --wave mindose \
    --runroot "$RUNROOT" --dose_manifest "$DOSE_MANIFEST" \
    $ATTEST \
    --output "$OUT"
rc=$?
echo "read (analyst, after pull): python -m analysis.mindose_reacher_read \\"
echo "  --auc $OUT/auc.csv --witness $OUT/witness.json \\"
echo "  --buffer_manifest $OUT/buffer_manifest.json --output artifacts/mindose_reacher_read_<date>"
exit $rc
