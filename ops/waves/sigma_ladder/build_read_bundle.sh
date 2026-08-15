#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Assemble the sigma-ladder READ bundle exactly as PREREG_sigma_ladder_20260814
# specifies, in the layout analysis/sigma_ladder_read.py consumes.
#
# Run on RCC, after `dv3ops pull --wave sigma_ladder 2 3 7` and BEFORE any
# cleanup (the M5 linkage gate is explicit about that ordering: the fit config
# records no replay path, so once the runs manifest is gone the fit->buffer
# linkage can never be reconstructed).
#
# The two python blocks below are the LITERAL registered producers, copied
# verbatim from the prereg documents and NOT rewritten:
#   * witness  -- the capdescent block (PREREG_capdescent_20260811) with BOTH
#                 load-specific literals swapped to nzs, per sigma-ladder
#                 review m6: the glob AND the seed-anchored regex.
#   * linkage  -- the M5 block (PREREG_sigma_ladder_20260814) unchanged.
# Editing either would break the gates they exist to enforce.
#
# CONFORMANCE NOTE (recorded in the bundle, as the prereg requires): this wave
# ran on three rented Vast instances rather than on Slurm, so the runs manifest
# the M5 gate reads does not exist as a single cluster file. It is assembled by
# concatenating the `nzs` rows of each instance's own $RUNROOT/runs.csv. The row
# SCHEMA is unchanged (REPLAY at field 7 / index 6), so the registered index
# needs no conformance; only the file's provenance differs. Run ids were checked
# for duplicates across instances before merging.
# ---------------------------------------------------------------------------
set -euo pipefail

RUNROOT="${RUNROOT:-/scratch/midway3/$USER/dreamerv3_runs}"
REPO="${REPO:-$HOME/projects/dreamerv3}"
MANIFEST="${MANIFEST:-$RUNROOT/sgl_runs_manifest.csv}"

[ -f "$MANIFEST" ] || { echo "ERROR: no assembled runs manifest at $MANIFEST" >&2; exit 1; }

bundle="$RUNROOT/bundles/sgl_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$bundle/ridge" "$bundle/runroot_light" \
         "$bundle/replay/q1_nzs2" "$bundle/replay/q1_nzs8"
echo "[sgl] bundle -> $bundle"

# -- 1. dual fit witness (LITERAL capdescent producer, nzs literals) ---------
echo "[sgl] witness block"
( cd "$bundle" && RUNROOT="$RUNROOT" python - <<'PY'
import glob, json, os, re
root = os.environ['RUNROOT']
rx = re.compile(r'ax1wm_finger_f?nzs[28]q1s[01]_seed[1-4]$')
wms = sorted(w for w in glob.glob(root + '/ax1wm_finger_*nzs[28]q1s*')
             if rx.search(w))
assert len(wms) == 32, (len(wms), wms[:3])
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
json.dump(counters, open('fit_counters.json', 'w'))
json.dump(steps, open('ckpt_steps.json', 'w'))
print('32/32 witnessed')
PY
)

# -- 2. fit -> buffer linkage gate (LITERAL M5 block) ------------------------
echo "[sgl] M5 replay-linkage block"
( cd "$bundle" && MANIFEST="$MANIFEST" python - <<'PY'
import csv, json, os, re
rows = {}
with open(os.environ['MANIFEST']) as f:
    for r in csv.reader(f):
        if r and r[0].startswith('ax1wm_finger_') and 'nzs' in r[0]:
            rows[r[0]] = r
rx = re.compile(r'^ax1wm_finger_f?(nzs[28])q1s([01])_seed[1-4]$')
out, n = {}, 0
for name, r in sorted(rows.items()):
    m = rx.match(name)
    if not m:
        continue
    replay = r[6]
    want = f'q1_{m.group(1)}/side{m.group(2)}'
    assert replay.rstrip('/').endswith(want), (name, replay, want)
    out[name] = replay
    n += 1
assert n == 32, (n, sorted(out))
json.dump(out, open('sgl_replay_linkage.json', 'w'), indent=1)
print('32/32 replay linkage OK')
PY
)

# -- 3. ridge jsons, already named <wm_run>.json (the reader gates the rename)
cp -a "$RUNROOT"/sgl_ridge/*.json "$bundle/ridge/"
echo "[sgl] ridge jsons: $(ls -1 "$bundle/ridge" | wc -l)"

# -- 4. runroot_light: config.yaml per fit, no checkpoints ------------------
n_cfg=0
for w in "$RUNROOT"/ax1wm_finger_*nzs[28]q1s*; do
  b="$(basename "$w")"
  [[ "$b" =~ ^ax1wm_finger_f?nzs[28]q1s[01]_seed[1-4]$ ]] || continue
  mkdir -p "$bundle/runroot_light/$b"
  cp -a "$w/config.yaml" "$bundle/runroot_light/$b/config.yaml"
  n_cfg=$((n_cfg+1))
done
echo "[sgl] runroot_light config.yamls: $n_cfg"

# -- 5. both replay manifests (sigma gates + pinned B2 holdout sha) ---------
cp -a "$RUNROOT/axis1_finger/q1_nzs2/manifest.json" "$bundle/replay/q1_nzs2/manifest.json"
cp -a "$RUNROOT/axis1_finger/q1_nzs8/manifest.json" "$bundle/replay/q1_nzs8/manifest.json"

# -- 6. provenance -----------------------------------------------------------
cp -a "$MANIFEST" "$bundle/sgl_runs_manifest.csv"
python3 "$REPO/scripts/ops/wavecheck.py" "$REPO/ops/waves/sigma_ladder" \
  --runroot "$RUNROOT" > "$bundle/COMPLETION.txt" || true

cat > "$bundle/NOTES.md" <<EOF
# sigma-ladder read bundle

prereg: prereg/PREREG_sigma_ladder_20260814.md
reader: analysis/sigma_ladder_read.py
built:  $(date -Is) on $(hostname)
repo:   $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)

## Reader invocation

    python analysis/sigma_ladder_read.py \\
      --ridge_glob '<bundle>/ridge/*.json' \\
      --runroot '<bundle>/runroot_light' \\
      --nzs2_manifest '<bundle>/replay/q1_nzs2/manifest.json' \\
      --nzs8_manifest '<bundle>/replay/q1_nzs8/manifest.json' \\
      --fit_counters '<bundle>/fit_counters.json' \\
      --ckpt_steps '<bundle>/ckpt_steps.json' \\
      --output <out_dir>

## Conformance (M5 runs-manifest provenance)

This wave ran on three rented Vast instances, not on Slurm, so no single
cluster runs manifest contains its rows. \`sgl_runs_manifest.csv\` is the
concatenation of the \`nzs\` rows of each instance's own \$RUNROOT/runs.csv
(instances 2, 3, 7). Row SCHEMA is unchanged -- REPLAY at field 7 / index 6 --
so the registered field index required no conformance; only the file's
provenance differs. Run ids were checked for duplicates across the three
instances before merging: none. \`ax1wm_finger_nzs8q1s0_seed99\` (the
registered smoke fit) appears in the manifest and is excluded by the
seed-anchored regex, as designed.

## What is NOT here

Checkpoints, replay buffers and adapt score files are deliberately excluded --
this is the light READ bundle. The full 4.6 GB of fits and adapts was pulled
to RCC \$RUNROOT by \`dv3ops pull --wave sigma_ladder 2 3 7\` and still lives
on instances 2, 3 and 7. Nothing has been deleted anywhere.
EOF

"$REPO/scripts/bundle_manifest.sh" generate "$bundle"
echo "$bundle"
