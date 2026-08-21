#!/bin/bash
#SBATCH --job-name=mindose_builddose
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# MIN-DOSE reacher -- materialize the dose buffers from the REGISTERED draw
# (PREREG_mindose_reacher_20260820 + amend1).
#
# The draw already passed its feasibility gate (job 53746920):
#   d0=0.000000  d1=0.097837  d2=0.185035  d3=0.273711
#   LEVEL-1 OCCUPANCY 0.097837 is INSIDE the registered [0.08, 0.20] window,
#   max pairwise dcov 0.0109 < tol 0.0144, overlap 0.11, 378 feasible
#   candidates, decision OK.
# So the wave does NOT refuse pre-outcome, and this job is the build step.
#
# build-dose materializes ALL FOUR levels under one root because that is what
# the tool does and what its manifest describes; the registered wave fits
# LEVEL 1 ONLY. The other levels are a by-product, not a licence -- a dose
# sweep would be a new registration, licensed only by a PROCEED here (amend1).
#
# Refuses to overwrite an existing dose tree: re-materializing after any fit
# has read it would silently change the substrate under a running wave.
#
# CPU (caslake), so it does not consume one of the four gpu-QOS nodes.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
: "${RUNROOT:?set RUNROOT}"

INDEX="$RUNROOT/axis1_reacher/episodes.json"
DOSE="$RUNROOT/axis1_reacher/dose.json"
OUT="$RUNROOT/axis1_reacher/dose"

[ -f "$INDEX" ] || { echo "REFUSE: no index at $INDEX"; exit 2; }
[ -f "$DOSE" ]  || { echo "REFUSE: no dose draw at $DOSE -- run search_dose.sbatch first"; exit 2; }
[ -e "$OUT" ]   && { echo "REFUSE: $OUT already exists -- refusing to re-materialize"; exit 3; }

python - "$DOSE" <<'PY' || exit 2
import json, sys
d = json.load(open(sys.argv[1]))
if d.get('decision') != 'OK':
  print(f"REFUSE: dose decision is {d.get('decision')!r}, not OK"); sys.exit(1)
occ = d['levels'][1].get('occ', d['levels'][1].get('occupancy'))
if not (0.08 <= occ <= 0.20):
  print(f"REFUSE: level-1 occupancy {occ} outside the registered [0.08, 0.20]"); sys.exit(1)
print(f"[gate] decision OK, level-1 occupancy {occ} inside [0.08, 0.20]")
PY

echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}"
python -u -m probing.build_controlled_replay build-dose \
  --index "$INDEX" --dose "$DOSE" --output_root "$OUT"
rc=$?
[ "$rc" -eq 0 ] || { echo "build-dose FAILED rc=$rc"; exit "$rc"; }

echo
echo "=== realized level dirs"
for d in "$OUT"/level*; do
  printf "  %-10s chunks=%s  bytes=%s\n" "$(basename "$d")" \
    "$(ls "$d"/*.npz 2>/dev/null | wc -l)" \
    "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
[ -f "$OUT/manifest.json" ] && echo "  manifest.json present"
echo
echo "LEVEL 1 (the registered fit substrate): $OUT/level1"
