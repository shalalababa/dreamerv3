#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FB tail on the instance: 16 fb embeds + raw comparator + 17 probes + the two
# registered producers (emb_panel_devices.json, fb_fit_counters.json).
#
# SINGLE PHYSICAL GPU, deliberately. The reader's check_devices requires ONE
# gpu_name across the 16 fb entries, and gpu_name comes from
# torch.cuda.get_device_name(0). Every lane sees its own card as device 0 and
# all four are RTX 5060 Ti, so spreading the panel over four lanes yields
# sixteen identical name strings and PASSES that gate while the panel is not
# actually single-device. So the embeds run serially under
# CUDA_VISIBLE_DEVICES=0. They are cheap (one forward pass over a 2.2 MB
# probeset), so this costs minutes.
#
# The raw comparator takes no --device: fb_embed's run_raw hardcodes 'cpu',
# which is what makes the registered raw-cpu exemption true by construction
# rather than by our arranging it.
#
# embed_probe is CPU-only (numpy + ridge_probe; sklearn is NOT required, and
# the FB env indeed has no sklearn).
#
# Idempotent: any step whose output already parses is skipped, so a partial
# run can be resumed without redoing GPU work.
# ---------------------------------------------------------------------------
set -uo pipefail
RUNROOT="${RUNROOT:-/workspace/dreamerv3_runs}"
PY=/workspace/conda_envs/fb/bin/python
REPO="${REPO:-/workspace/dreamerv3}"
PS="$RUNROOT/e4_probesets/finger_v1"
EMB="$RUNROOT/fb_emb"
PROBE="$RUNROOT/emb_probe"
cd "$REPO"

[ -d "$PS" ] || { echo "FATAL: probeset missing at $PS" >&2; exit 2; }
mkdir -p "$EMB" "$PROBE"

runs=()
for side in 0 1; do for seed in 1 2 3 4 5 6 7 8; do
  runs+=("fbwm_finger_q1s${side}_seed${seed}")
done; done

echo "=== [1/5] 16 fb embeds on ONE physical card (CUDA_VISIBLE_DEVICES=0) ==="
for r in "${runs[@]}"; do
  if [ -f "$EMB/$r/embed_manifest.json" ] && [ -f "$EMB/$r/probeset_emb.npz" ]; then
    echo "  SKIP $r (already embedded)"; continue
  fi
  [ -f "$RUNROOT/$r/fb_ckpt.pt" ] || { echo "  FATAL: no ckpt for $r" >&2; exit 3; }
  CUDA_VISIBLE_DEVICES=0 CONTROLLABLE_AGENT_ROOT=/workspace/controllable_agent \
  MUJOCO_GL=disabled "$PY" -m probing.fb_embed fb \
    --ckpt "$RUNROOT/$r/fb_ckpt.pt" --probeset "$PS" \
    --output "$EMB/$r" --device cuda || { echo "  FATAL: embed failed $r" >&2; exit 3; }
done

echo "=== [2/5] raw identity comparator (CPU by construction) ==="
if [ -f "$EMB/raw_identity/embed_manifest.json" ]; then
  echo "  SKIP raw_identity"
else
  CONTROLLABLE_AGENT_ROOT=/workspace/controllable_agent MUJOCO_GL=disabled \
  "$PY" -m probing.fb_embed raw --probeset "$PS" --output "$EMB/raw_identity" \
    || { echo "  FATAL: raw embed failed" >&2; exit 3; }
fi

echo "=== [3/5] 17 embed_probe passes (CPU) ==="
for r in "${runs[@]}" raw_identity; do
  if [ -f "$PROBE/$r.json" ]; then echo "  SKIP $r (probe exists)"; continue; fi
  d="$EMB/$r"
  CONTROLLABLE_AGENT_ROOT=/workspace/controllable_agent MUJOCO_GL=disabled \
  "$PY" -m probing.embed_probe \
    --emb "$d/probeset_emb.npz" --embed_manifest "$d/embed_manifest.json" \
    --probeset "$PS" --run_id "$r" --output "$PROBE" \
    || { echo "  FATAL: probe failed $r" >&2; exit 3; }
done

echo "=== [4/5] emb_panel_devices.json (registered producer) ==="
"$PY" - "$EMB" "$PROBE" "$RUNROOT/emb_panel_devices.json" <<'PYEOF'
import json, os, sys
emb, probe, out = sys.argv[1:4]
runs = [f'fbwm_finger_q1s{s}_seed{k}' for s in (0, 1) for k in range(1, 9)]
dv = {}
for r in runs + ['raw_identity']:
    with open(os.path.join(emb, r, 'embed_manifest.json')) as f:
        m = json.load(f)
    # Registered semantics: the PHYSICAL device recorded by fb_embed, never
    # the torch device string. Absent gpu_name (the raw path) means cpu.
    dv[f'{r}.json'] = m.get('gpu_name') or 'cpu'
    assert os.path.exists(os.path.join(probe, f'{r}.json')), ('no probe for', r)
assert dv['raw_identity.json'] == 'cpu', dv['raw_identity.json']
names = {v for k, v in dv.items() if k != 'raw_identity.json'}
assert len(names) == 1, ('panel mixes GPU models -- the single-GPU rule '
                         'failed', sorted(names))
assert len(dv) == 17, len(dv)
with open(out, 'w') as f:
    json.dump(dv, f, indent=1, sort_keys=True)
print(f'  17 entries, single GPU = {names.pop()}, raw = cpu -> {out}')
PYEOF
[ $? -eq 0 ] || { echo "FATAL: device map producer failed" >&2; exit 4; }

echo "=== [5/5] fb_fit_counters.json (registered producer) ==="
"$PY" - "$RUNROOT" "$RUNROOT/fb_fit_counters.json" <<'PYEOF'
import json, os, re, sys
rr, out = sys.argv[1:3]
runs = [f'fbwm_finger_q1s{s}_seed{k}' for s in (0, 1) for k in range(1, 9)]
fc = {}
for r in runs:
    p = os.path.join(rr, r, 'FB_FIT_PROGRESS')
    txt = open(p).read()
    upd = int(re.search(r'^update=(\d+)$', txt, re.M).group(1))
    tot = int(re.search(r'^total_updates=(\d+)$', txt, re.M).group(1))
    done = os.path.exists(os.path.join(rr, r, 'FB_FIT_DONE'))
    # The reader asserts update == total and done is True; a fit that stopped
    # short cannot satisfy both, so this is a witness, not a rubber stamp.
    assert upd == tot and done, (r, upd, tot, done)
    fc[r] = dict(update=upd, total=tot, done=done)
assert len(fc) == 16, len(fc)
tot = {v['total'] for v in fc.values()}
assert len(tot) == 1, ('mixed update totals', tot)
with open(out, 'w') as f:
    json.dump(fc, f, indent=1, sort_keys=True)
print(f'  16 witnesses, all update==total=={tot.pop()} -> {out}')
PYEOF
[ $? -eq 0 ] || { echo "FATAL: counters producer failed" >&2; exit 5; }

echo
echo "FB TAIL COMPLETE: $(ls "$PROBE"/*.json | wc -l) probe jsons, device map + counters written."
