#!/bin/bash
#SBATCH --job-name=mindose_searchdose
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# MIN-DOSE reacher -- the REGISTERED dose draw
# (PREREG_mindose_reacher_20260820 + amend1, section "the corrected pin").
#
# THIS INVOCATION IS THE DRAW. The command below is the amendment's pinned
# literal, verified flag-by-flag against the subparser before running so the
# registered draw is not spent on a typo: --levels 4 --beam 40 --n_episodes
# 200 --n_candidates 400 --dirichlet 0.3 --seed 0 are exactly the subparser
# defaults, and --index / --ref_replay / --output are the only real inputs.
#
# --ref_replay is copied VERBATIM from the registered reacher q1 search's own
# provenance (axis1_reacher/pairs.json: ref_replay =
# $RUNROOT/pilot_goal_reacher_seed1/replay), as the amendment requires; the
# literal is echoed into the log so the occupancy FILL can cite it.
#
# --holdout is OMITTED, per the amendment: the flag takes E4 probeset manifest
# paths and NO reacher probeset exists (e4_probesets carries cup/finger/synth
# only). The obligation is inverted and registered instead -- any future
# reacher probeset measured on these fits must exclude this buffer's episodes
# at probeset-build time.
#
# CPU (caslake), so it does not consume one of the twelve gpu-QOS submit slots.
# REFUSES to overwrite an existing dose.json: a second draw would be a second
# look at the same registered gate.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
: "${RUNROOT:?set RUNROOT}"

INDEX="$RUNROOT/axis1_reacher/episodes.json"
REF="$RUNROOT/pilot_goal_reacher_seed1/replay"
OUT="$RUNROOT/axis1_reacher/dose.json"

[ -f "$INDEX" ] || { echo "REFUSE: no index at $INDEX"; exit 2; }
[ -d "$REF" ]   || { echo "REFUSE: ref_replay absent at $REF"; exit 2; }
[ -e "$OUT" ]   && { echo "REFUSE: $OUT already exists -- a second draw would be a second look"; exit 3; }

echo "[host] $(hostname)  job=${SLURM_JOB_ID:-none}"
echo "[pin ] index      = $INDEX"
echo "[pin ] ref_replay = $REF   (verbatim from axis1_reacher/pairs.json)"
echo "[pin ] levels 4 / beam 40 / n_episodes 200 / n_candidates 400 / dirichlet 0.3 / seed 0"
echo "[pin ] holdout    = OMITTED (no reacher probeset exists; obligation inverted per amend1)"
echo

python -u -m probing.build_controlled_replay search-dose \
  --index "$INDEX" --ref_replay "$REF" \
  --levels 4 --beam 40 --n_episodes 200 --n_candidates 400 \
  --dirichlet 0.3 --seed 0 \
  --output "$OUT"
rc=$?
echo
if [ "$rc" -ne 0 ]; then echo "search-dose FAILED rc=$rc"; exit "$rc"; fi

# Print the realized occupancies against the registered acceptance window so
# the FILL can be written from the log without re-deriving anything.
python - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print('decision:', d.get('decision'))
levels = d.get('levels') or []
for i, L in enumerate(levels):
    occ = L.get('occ', L.get('occupancy'))
    print(f'  level {i}: occ = {occ}')
if levels:
    occ1 = levels[1].get('occ', levels[1].get('occupancy')) if len(levels) > 1 else None
    print(f'\nLEVEL-1 OCCUPANCY = {occ1}')
    print('registered window [0.08, 0.20] ->',
          'INSIDE - wave proceeds' if occ1 is not None and 0.08 <= occ1 <= 0.20
          else 'OUTSIDE - the wave REFUSES pre-outcome (walker precedent)')
PY
