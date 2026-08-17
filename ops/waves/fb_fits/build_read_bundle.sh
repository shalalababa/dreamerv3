#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FB read bundle — PREREG_fb_20260817 §Ops-chain bundle spec.
#
# Layout is dictated by the frozen reader's own argument shapes, so it is
# transcribed from analysis/fb_read.py rather than invented:
#   --runroot        <bundle>/runroot        (fb_config.json + fb_metrics.json)
#   --fit_counters   <bundle>/fb_fit_counters.json
#   --zeroshot_glob  <bundle>/zeroshot/*.json   (+ random_floor.json, which the
#                                               reader skips by its tool field)
#   --emb_probe_glob <bundle>/emb_probe/fbwm_*.json
#   --raw_probe      <bundle>/emb_probe/raw_identity.json
#   --devices        <bundle>/emb_panel_devices.json
#
# runroot_light by registration: NO ckpt bytes. Checkpoint identity travels
# through the zeroshot/emb ckpt_sha256 linkage gates instead (review m27), so
# shipping ckpts would add ~GBs and prove nothing the shas do not.
#
# fb_metrics.json is REQUIRED, not optional (review-2 M2): it is the
# training-health witness, and the reader refuses on its absence while
# treating out-of-band as a recorded fact.
#
# Counts are asserted before anything is copied: a 15-of-16 panel is exactly
# the failure a read cannot see for itself until it refuses late.
# ---------------------------------------------------------------------------
set -uo pipefail
RUNROOT="${RUNROOT:-/workspace/dreamerv3_runs}"
stamp="$(date +%Y%m%d_%H%M%S)"
B="$RUNROOT/bundles/fb_fits_${stamp}"

runs=()
for side in 0 1; do for seed in 1 2 3 4 5 6 7 8; do
  runs+=("fbwm_finger_q1s${side}_seed${seed}")
done; done

fail=0
need () { [ -e "$1" ] || { echo "MISSING: $1" >&2; fail=1; }; }
echo "[fb] preflight"
for r in "${runs[@]}"; do
  need "$RUNROOT/$r/fb_config.json"
  need "$RUNROOT/$r/fb_metrics.json"
  need "$RUNROOT/$r/zeroshot.json"
  need "$RUNROOT/emb_probe/$r.json"
done
need "$RUNROOT/emb_probe/raw_identity.json"
need "$RUNROOT/emb_panel_devices.json"
need "$RUNROOT/fb_fit_counters.json"
need "$RUNROOT/fb_zeroshot/random_floor.json"
[ "$fail" -eq 0 ] || { echo "[fb] PREFLIGHT FAILED -- nothing bundled" >&2; exit 1; }

echo "[fb] bundle -> $B"
mkdir -p "$B/runroot" "$B/zeroshot" "$B/emb_probe" "$B/_cloud_logs"
for r in "${runs[@]}"; do
  mkdir -p "$B/runroot/$r"
  cp -a "$RUNROOT/$r/fb_config.json"  "$B/runroot/$r/"
  cp -a "$RUNROOT/$r/fb_metrics.json" "$B/runroot/$r/"
  cp -a "$RUNROOT/$r/FB_FIT_PROGRESS" "$B/runroot/$r/" 2>/dev/null
  # zeroshot.json lives in the run dir (fb_zeroshot's own convention); the
  # reader globs a flat zeroshot/ dir, and keys on dirname(ckpt), not on this
  # filename -- so renaming to <run_id>.json here is safe and keeps the glob
  # unambiguous.
  cp -a "$RUNROOT/$r/zeroshot.json"   "$B/zeroshot/$r.json"
  cp -a "$RUNROOT/emb_probe/$r.json"  "$B/emb_probe/$r.json"
  [ -f "$RUNROOT/_cloud_logs/$r.out" ] && cp -a "$RUNROOT/_cloud_logs/$r.out" "$B/_cloud_logs/"
done
cp -a "$RUNROOT/emb_probe/raw_identity.json" "$B/emb_probe/"
cp -a "$RUNROOT/fb_zeroshot/random_floor.json" "$B/zeroshot/"
cp -a "$RUNROOT/emb_panel_devices.json" "$B/"
cp -a "$RUNROOT/fb_fit_counters.json"   "$B/"

# post-copy counts, asserted on what actually landed
nz=$(ls "$B"/zeroshot/*.json | wc -l)          # 16 + random_floor
ne=$(ls "$B"/emb_probe/fbwm_*.json | wc -l)    # 16
nc=$(ls -d "$B"/runroot/*/ | wc -l)            # 16
[ "$nz" -eq 17 ] && [ "$ne" -eq 16 ] && [ "$nc" -eq 16 ] || {
  echo "[fb] COUNT MISMATCH zeroshot=$nz(want 17) emb=$ne(want 16) runs=$nc(want 16)" >&2
  exit 1; }

cat > "$B/BUNDLE_NOTE.md" <<'EOF'
# FB read bundle

runroot_light by registration: no ckpt bytes. Checkpoint identity is carried
by the zeroshot/emb `ckpt_sha256` byte-identity gates (review m27), not by a
ckpt-presence witness, so shipping checkpoints would add GBs and prove nothing
the shas do not already prove.

`zeroshot/random_floor.json` sits in the same directory the reader globs for
zeroshots. That is intended: `load_zeroshot` skips it on
`tool == 'fb_random_floor_v1'`, and the reader also takes it directly via
`--random_floor`.

Zeroshot jsons are renamed from `<run>/zeroshot.json` to `zeroshot/<run_id>.json`.
The reader keys each one by `dirname(ckpt)` via RID_RE, never by filename, so
this is a flattening for glob clarity and cannot mislabel a run.

Panel provenance: all 16 fb embeddings were produced on ONE physical GPU
(CUDA_VISIBLE_DEVICES=0, serial). `check_devices` compares
`embed_manifest.gpu_name`, and since every lane sees its own card as device 0
and all four are the same model, a lane-parallel panel would have produced 16
identical name strings and passed that gate while not being single-device.
EOF

( cd "$B" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z \
    | xargs -0 sha256sum > MANIFEST.sha256 )
echo "[fb] bundled $(wc -l < "$B/MANIFEST.sha256") files ($(du -sh "$B" | cut -f1)) -> $B"
echo "$B"
