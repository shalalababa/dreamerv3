#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SE amendment-2 read bundle — D-only + Stage-2.
# (PREREG_nfi_scale_exhibit_amend2_20260814 §1 and §2 bundle specs.)
#
# WHY NOT `dv3ops bundle`: the generated bundle.sh copies whole run dirs, which
# here is 51 GB — and ~47 GB of that is `ckpt_snapshots/` (3.9 GB × 12), which
# NEITHER bundle spec asks for. Snapshots are the Stage-1 M4 input and the
# amendment records those probes as already executed; §1 and §2 list only what
# is copied below. So this ships the registered contents (~5 GB) and leaves the
# snapshots on RCC scratch, where they remain available if a later amendment
# wants a snapshot leg for the new arms. Nothing is deleted.
#
# Registered contents, verbatim:
#   §1 D-only  : per run `se_probe/` (json+npz), `config.yaml`,
#                `metrics.jsonl`, `ckpt/latest` (+ ckpt dir listing)
#   §2 Stage-2 : per run `replay/` (REQUIRED for occupancy), `scores.jsonl`,
#                `config.yaml`, `metrics.jsonl`, `ckpt/latest`
# `scores.jsonl` and `TRAINING_DONE` are shipped for the D-only arm too: they
# cost kilobytes and the reader's fit-counter/validity gates read them.
#
# Counts are ASSERTED, not globbed loosely: a D-only arm silently missing one
# se_probe, or a Stage-2 arm missing a replay dir, is exactly the failure a
# read cannot see. Preflight refuses before anything is copied.
# ---------------------------------------------------------------------------
set -uo pipefail
RUNROOT="${RUNROOT:-/scratch/midway3/rickybao/dreamerv3_runs}"
stamp="$(date +%Y%m%d_%H%M%S)"
bundle="$RUNROOT/bundles/uncfield_se_arms_${stamp}"

DONLY=(se_donly_s10 se_donly_s11 se_donly_s12 se_donly_s13)
GATED=(se_gate_s20 se_gate_s21 se_gate_s22 se_gate_s23)
UNGATED=(se_ungate_s24 se_ungate_s25 se_ungate_s26 se_ungate_s27)
STAGE2=("${GATED[@]}" "${UNGATED[@]}")
ALL=("${DONLY[@]}" "${STAGE2[@]}")

fail=0
need () { [ -e "$1" ] || { echo "MISSING: $1" >&2; fail=1; }; }

echo "[se] preflight"
for r in "${ALL[@]}"; do
  need "$RUNROOT/$r/config.yaml"
  need "$RUNROOT/$r/metrics.jsonl"
  need "$RUNROOT/$r/scores.jsonl"
  need "$RUNROOT/$r/ckpt/latest"
done
# D-only: the se_probe pass is a REGISTERED bundling precondition.
for r in "${DONLY[@]}"; do
  need "$RUNROOT/$r/se_probe/se_probe.json"
  need "$RUNROOT/$r/se_probe/se_probe_dims.npz"
done
# Stage-2: replay is the M3 occupancy input; an empty dir is not a replay.
for r in "${STAGE2[@]}"; do
  n=$(ls "$RUNROOT/$r"/replay/*.npz 2>/dev/null | wc -l)
  if [ "$n" -lt 1000 ]; then
    echo "REPLAY TOO SMALL: $r has $n chunks (expected >1000)" >&2; fail=1
  fi
done
[ "$fail" -eq 0 ] || { echo "[se] PREFLIGHT FAILED -- nothing bundled" >&2; exit 1; }

echo "[se] bundle -> $bundle"
mkdir -p "$bundle/runroot"
for r in "${ALL[@]}"; do
  d="$bundle/runroot/$r"
  mkdir -p "$d/ckpt"
  cp -a "$RUNROOT/$r/config.yaml"    "$d/"
  cp -a "$RUNROOT/$r/metrics.jsonl"  "$d/"
  cp -a "$RUNROOT/$r/scores.jsonl"   "$d/"
  [ -f "$RUNROOT/$r/TRAINING_DONE" ] && cp -a "$RUNROOT/$r/TRAINING_DONE" "$d/"
  cp -a "$RUNROOT/$r/ckpt/latest"    "$d/ckpt/"
  # ckpt dir listing (§1 asks for it; it is the cheap provenance of which
  # saves existed without shipping any of them).
  ls -la "$RUNROOT/$r/ckpt" > "$d/ckpt/DIR_LISTING.txt" 2>/dev/null
  # nearest.json is the realized-step witness the corrected wave predicate
  # asserts; kilobytes, and it is what shows each run reached ~5e5.
  [ -f "$RUNROOT/$r/ckpt_snapshots/nearest.json" ] && \
    cp -a "$RUNROOT/$r/ckpt_snapshots/nearest.json" "$d/ckpt_snapshots_nearest.json"
done
for r in "${DONLY[@]}"; do
  cp -a "$RUNROOT/$r/se_probe" "$bundle/runroot/$r/se_probe"
done
for r in "${STAGE2[@]}"; do
  cp -a "$RUNROOT/$r/replay" "$bundle/runroot/$r/replay"
done

mkdir -p "$bundle/_cloud_logs"
for r in "${ALL[@]}"; do
  [ -f "$RUNROOT/_cloud_logs/$r.out" ] && cp -a "$RUNROOT/_cloud_logs/$r.out" "$bundle/_cloud_logs/"
done

# COMPLETION.txt from the same checker that gates the read.
python3 "${DV3OPS_ROOT:-$HOME/projects/dreamerv3}/scripts/ops/wavecheck.py" \
  "${DV3OPS_ROOT:-$HOME/projects/dreamerv3}/ops/waves/se_arms" --runroot "$RUNROOT" \
  > "$bundle/COMPLETION.txt" 2>&1 || true

cat > "$bundle/BUNDLE_NOTE.md" <<'EOF'
# What is here, and what is deliberately not

Built to the amendment's §1/§2 bundle specs rather than by copying whole run
directories. Excluded: `ckpt_snapshots/` (~3.9 GB per run, ~47 GB of the 51 GB
a whole-dir bundle would be). Neither spec lists it — snapshots are the
Stage-1 M4 input, and the amendment records those probes as already executed.
They remain on RCC scratch under each run dir if a later amendment wants a
snapshot leg for these arms.

Kept from the snapshot tree: `ckpt_snapshots_nearest.json` per run, the
realized-step witness (kilobytes), which is what shows every run reached
~5e5 — see the corrected completion predicate in ops/waves/se_arms/spec.yaml.

Also excluded: `plugins/` (44 MB/run) and `scope/` (20 MB/run), neither listed
in either spec.

Full checkpoints beyond `ckpt/latest` are not shipped; `ckpt/DIR_LISTING.txt`
records which saves existed.
EOF

( cd "$bundle" && find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
n=$(wc -l < "$bundle/MANIFEST.sha256")
echo "[se] bundled $n files ($(du -sh "$bundle" | cut -f1)) -> $bundle"
echo "$bundle"
