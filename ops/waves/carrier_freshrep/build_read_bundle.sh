#!/usr/bin/env bash
# Carrier fresh-rep read bundle — PREREG_carrier_freshrep_20260820 (+amend1).
# Thin wrapper over the selfchecked builder; the frozen reader dictates the
# contract (auc.csv + witness.json w/ _meta.amend1_config_match + linkage.json).
#
# REF_CONFIGS must hold the four Amendment-1 reference configs as
# {rgo,sgb}s{0,1}.yaml (extract from the archived July P3 fits, e.g.
# ax1wm_finger_rgoq1s0_seed1/config.yaml -> rgos0.yaml). Without them the
# match flag is FALSE and the reader refuses — the flag can never silently
# pass.
#
# BUNDLE AFTER THE PULL, NEVER BEFORE (17 Aug standing rule): this wave ran
# on Vast — full-pull each instance's runroot (incl. _cloud_logs/, the fit
# dirs' ckpt trees, and OFFLINE_FIT_PROGRESS) to RCC FIRST, then run this
# builder on RCC against the merged runroot. An instance-side bundle would
# verify green while everything outside it dies with the instance.
#
#   RUNROOT=... REF_CONFIGS=... bash ops/waves/carrier_freshrep/build_read_bundle.sh
set -uo pipefail
: "${RUNROOT:?set RUNROOT}"
: "${REF_CONFIGS:?set REF_CONFIGS (dir with {rgo,sgb}s{0,1}.yaml)}"
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
stamp="$(date +%Y%m%d_%H%M%S)"
OUT="$RUNROOT/bundles/carrier_freshrep_${stamp}"
cd "$REPO"
python -m scripts.ops.build_rescue_bundle --wave carrier \
    --runroot "$RUNROOT" --ref_configs "$REF_CONFIGS" \
    --lane_cmds "${LANE_CMDS:-$REPO/ops/waves/carrier_freshrep/generated}" \
    --output "$OUT"
rc=$?
echo "read (analyst, after pull): python -m analysis.carrier_freshrep_read \\"
echo "  --auc $OUT/auc.csv --witness $OUT/witness.json \\"
echo "  --linkage $OUT/linkage.json --output artifacts/carrier_freshrep_read_<date>"
exit $rc
