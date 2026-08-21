#!/usr/bin/env bash
# U1a pooled read bundle — PREREG_u1a_pooled_20260820 (+amend1).
# Thin wrapper over the selfchecked builder (auc.csv + witness.json with
# _meta.config_ref_paths/config_ref_shas recording the finger_refit_20260810
# reference configs — recorded, not matched, per amend1 M9; the reader
# refuses if the fields are absent).
#
#   RUNROOT=... REF_CONFIGS=<finger_refit_20260810 config dir> \
#     bash ops/waves/u1a_pooled/build_read_bundle.sh
set -uo pipefail
: "${RUNROOT:?set RUNROOT}"
: "${REF_CONFIGS:?set REF_CONFIGS (finger_refit_20260810 config dir)}"
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
stamp="$(date +%Y%m%d_%H%M%S)"
OUT="$RUNROOT/bundles/u1a_pooled_${stamp}"
cd "$REPO"
python -m scripts.ops.build_rescue_bundle --wave u1a \
    --runroot "$RUNROOT" --ref_configs "$REF_CONFIGS" \
    --output "$OUT"
rc=$?
echo "read (analyst, after pull): python -m analysis.u1a_pooled_read \\"
echo "  --auc $OUT/auc.csv --witness $OUT/witness.json \\"
echo "  --output artifacts/u1a_pooled_read_<date>"
exit $rc
