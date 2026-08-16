#!/bin/bash
#SBATCH --job-name=lewm_finish
#SBATCH --account=pi-chenyuxin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
# ---------------------------------------------------------------------------
# LeWM steps 6, 8 and 10 in one job. (PREREG_lewm_20260812 §§6, 8, 10.)
#
# Bundled for the same reason B1's ridge panel is: sixteen embedding probes as
# sixteen jobs would eat the 12-job QOS cap, and the walltime limit (36 h) is
# nowhere near binding -- a probe is ~4 min, so the whole set fits in one
# allocation with room to spare.
#
# --constraint=rtx6000 because the current conda env CANNOT USE THE V100s:
# jax 0.4.33 / jaxlib 0.4.33 fail to bring up the CUDA backend on Volta and
# fall through to ROCm ("no attribute 'GpuAllocatorConfig'"), which is also
# where the pixel cudnn 5003 failures come from. rtx6000 and a100 are fine.
# Step 6 is CPU work, but the job still runs where the env works.
#
# Each step is resumable and refuses to overwrite a finished artifact, so a
# re-submit after a drain picks up where it stopped.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/scripts/env.sh"
activate_env
cd "$REPO"
: "${RUNROOT:?set RUNROOT}"
export REPO RUNROOT

echo "[device] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# -- step 6: embedding probes x16 (dreamer env, CPU) ------------------------
# Registered command, verbatim. This is the P-J1-emb DISTILL-FREE leg: it
# probes the raw LeWM embeddings, which is one of the three registered
# mitigations for "the graft tests a distilled student, not LeWM's ViT".
echo "=== step 6: embedding probes ==="
n6_ok=0; n6_skip=0; n6_fail=0
for side in 0 1; do
  for f in 1 2 3 4 5 6 7 8; do
    out="$RUNROOT/lewm_probe/lewm_finger_s${side}_seed${f}.json"
    if [ -f "$out" ] && python -c "import json,sys; json.load(open(sys.argv[1]))" "$out" 2>/dev/null; then
      echo "SKIP embed_probe s${side}_seed${f}"; n6_skip=$((n6_skip+1)); continue
    fi
    echo "[$(date)] embed_probe s${side}_seed${f}"
    JAX_PLATFORMS=cpu python -m probing.embed_probe \
      --emb "$RUNROOT/lewm_emb/s${side}_seed${f}_probeset/probeset_emb.npz" \
      --embed_manifest "$RUNROOT/lewm_emb/s${side}_seed${f}_probeset/embed_manifest.json" \
      --probeset "$RUNROOT/e4_probesets/fingerpx_v1" \
      --run_id "lewm_finger_s${side}_seed${f}" \
      --output "$RUNROOT/lewm_probe/"
    if [ $? -eq 0 ]; then n6_ok=$((n6_ok+1)); else echo "FAIL s${side}_seed${f}"; n6_fail=$((n6_fail+1)); fi
  done
done
echo "step 6: ok=$n6_ok skipped=$n6_skip failed=$n6_fail"

# -- step 8: integrity sweep (REGISTERED BLOCK, verbatim) -------------------
echo
echo "=== step 8: integrity sweep ==="
cd "$RUNROOT" && python - <<'PY' 2>&1 | tee "$RUNROOT/jp_integrity_sweep.log"
import json, subprocess, glob, os
out = {}
for side in (0, 1):
  for f in range(1, 9):
    wm = f'ax1wm_finger_jppxq1ms{side}_seed{f}'
    dn = f'lewm_distill_finger_s{side}_seed{f}'
    s2 = sorted(glob.glob(f'{wm}/ckpt/*/done'))
    rc = 1
    if s2:
      rc = subprocess.call(
          ['python', os.path.expandvars('$REPO/scripts/check_frozen_enc.py'),
           os.path.dirname(s2[-1]), f'{dn}/ckpt/distill-0', '500000'])
    out[wm] = (rc == 0)
json.dump(out, open('jp_enc_checks.json', 'w'), indent=1)
bad = [k for k, v in out.items() if not v]
print(json.dumps(out, indent=1))
print('QUARANTINE:', bad if bad else 'none')
PY
cd "$REPO"

# -- step 10: AUC collate + fit witness -------------------------------------
echo
echo "=== step 10: AUC collate ==="
python -m analysis.adaptation_auc --runroot "$RUNROOT" --output "$RUNROOT/analysis_out_jp/"

echo
echo "=== step 10: fit witness (capdescent block, jp literals) ==="
# The LITERAL capdescent witness producer with BOTH load-specific literals
# swapped (the m6 lesson): glob AND the seed-anchored regex, asserting 16.
( cd "$RUNROOT" && python - <<'PY'
import glob, json, os, re
root = os.environ['RUNROOT']
rx = re.compile(r'ax1wm_finger_jppxq1ms[01]_seed[1-8]$')
wms = sorted(w for w in glob.glob(root + '/ax1wm_finger_jppxq1ms*')
             if rx.search(w))
assert len(wms) == 16, (len(wms), wms[:3])
counters, steps = {}, {}
for w in wms:
    kv = dict(l.split('=', 1) for l in
              open(w + '/OFFLINE_FIT_PROGRESS').read().strip().split('\n'))
    counters[os.path.basename(w)] = dict(update=int(kv['update']),
                                         total=int(kv['total_updates']))
    cks = [c for c in sorted(glob.glob(w + '/ckpt/*'))
           if '500000' in os.path.basename(c)
           and os.path.exists(os.path.join(c, 'done'))]
    assert cks, (w, 'no done-marked 500000 ckpt')
    steps[os.path.basename(w)] = 500000
json.dump(counters, open('jp_fit_counters.json', 'w'))
json.dump(steps, open('jp_ckpt_steps.json', 'w'))
print('16/16 witnessed')
PY
)
echo
echo "[$(date)] lewm_finish DONE"
