#!/bin/bash
#SBATCH --job-name=pe_e4_devdiff
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# One-cell pe E4 re-measure, to bound the device term ON THE ESTIMAND ITSELF.
# (2026-08-16 diagnostic; no registered decision consumes it.)
#
# WHY. The LeWM primary is mean(pe rew_nll_in) - mean(jp rew_nll_in), a
# nats-scale quantity. The devcheck showed ridge AUROC/R2 moves up to 0.113
# across GPU architectures while `deter_std_recomputed` -- a forward-pass
# aggregate -- moved only ~1e-5, which is the basis for calling this
# comparison device-robust. But deter_std_recomputed is not rew_nll_in. This
# closes that gap by measuring the thing the read actually uses: re-measure
# ONE pe cell on rtx6000 and diff its rew_nll_in against the historical value
# in the pinned anchor csv (sha 9907d733...), which was produced elsewhere.
#
# NON-DESTRUCTIVE BY CONSTRUCTION: --output points at a side directory, so the
# registered e4_fingerpx_v1/ dir and the committed csv are untouched. The
# historical csv stays the pinned anchor either way; this only tells us how
# large the device term on rew_nll_in is.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"

: "${RUNROOT:?set RUNROOT}"
CELL="${CELL:-ax1wm_finger_pepxq1ms0_seed1}"
PROBESET="$RUNROOT/e4_probesets/fingerpx_v1"
OUT="$RUNROOT/pe_e4_devdiff/$CELL"

echo "[device] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ -d "$RUNROOT/$CELL" ] || { echo "ERROR: no run dir $RUNROOT/$CELL" >&2; exit 2; }
mkdir -p "$OUT"

python -m probing.stratified_error measure \
  --probeset "$PROBESET" --run_logdir "$RUNROOT/$CELL" \
  --horizons 1 5 20 --ep_batch 4 --output "$OUT"
rc=$?
[ "$rc" -eq 0 ] || { echo "measure failed rc=$rc" >&2; exit "$rc"; }

echo
echo "=== rew_nll_in: historical anchor vs this re-measure ==="
python - "$OUT/summary.json" "$CELL" <<'PY'
import json, sys, csv, os
new = json.load(open(sys.argv[1]))
cell = sys.argv[2]
# the frozen reader reads horizon-0 rows; mirror that selection
def h0(d):
    for k in ('rew_nll_in',):
        if k in d: return float(d[k])
    for blk in d.values():
        if isinstance(blk, dict) and 'rew_nll_in' in blk:
            return float(blk['rew_nll_in'])
    return None
print('re-measured summary.json keys:', sorted(new.keys())[:12])
v_new = h0(new)
print('re-measured rew_nll_in:', v_new)
hist = os.path.expanduser('~/projects/dreamerv3/local_results/'
                          'pe_pixel_20260802_165601/e4_fingerpx_v1_pe.csv')
if os.path.exists(hist):
    for r in csv.DictReader(open(hist)):
        if r['run_id'] == cell and str(r.get('horizon')) == '0':
            v_old = float(r['rew_nll_in'])
            print('historical  rew_nll_in:', v_old)
            if v_new is not None:
                d = v_new - v_old
                print('DELTA: %+.3e  (relative %+.3e)' % (d, d / abs(v_old) if v_old else 0.0))
            break
    else:
        print('NOTE: no horizon-0 row for', cell, 'in the historical csv')
else:
    print('NOTE: historical csv not readable from the compute node:', hist)
PY
