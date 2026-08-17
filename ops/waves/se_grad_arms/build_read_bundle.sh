#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SE amendment-3 gradient-arm read bundle — PREREG_nfi_scale_exhibit_amend3 §5.
#
# RUNS ON RCC, against the full pulled runroot. That order is the rule, not a
# detail: a read bundle is deliberately a SUBSET of the runroot, so building
# it instance-side and shipping only the bundle strands everything outside the
# spec while every check still reads green (see scripts/ops/CLAUDE.md).
#
# Registered contents, verbatim from §5:
#   hetero   : replay/, scores.jsonl, config.yaml, metrics.jsonl,
#              ckpt/latest (+ listing), se_probe/
#   flat     : same as hetero INCLUDING se_probe/ (amplitude-dose descriptive)
#   ungated2 : same minus se_probe/
#
# se_probe/ on the hetero arm is the MANIPULATION-CHECK GATE input (§4): the
# read's validity is conditional on the distractor clearing its permutation
# null in >= 3 of 4 hetero runs, and fewer than 3 VALID probes is outcome cell
# iv, an instrument failure rather than a finding. So a missing hetero probe is
# not a cosmetic gap, and the preflight refuses on it.
#
# EXCLUDED, deliberately: ckpt_snapshots/ payload (~3.9 G/run, listed in
# neither the §5 spec nor the read), plugins/, scope/, and all checkpoints
# beyond ckpt/latest. The snapshots stay on RCC scratch if a later amendment
# wants a snapshot leg. Kept from that tree: ckpt_snapshots/nearest.json, the
# realized-step witness the wave predicate asserts (kilobytes).
# ---------------------------------------------------------------------------
set -uo pipefail
RUNROOT="${RUNROOT:-/scratch/midway3/rickybao/dreamerv3_runs}"
stamp="$(date +%Y%m%d_%H%M%S)"
B="$RUNROOT/bundles/uncfield_se_grad_${stamp}"

HET=(se_het_s28 se_het_s29 se_het_s30 se_het_s31)
FLAT=(se_flat_s36 se_flat_s37 se_flat_s38 se_flat_s39)
UNG=(se_ungate2_s32 se_ungate2_s33 se_ungate2_s34 se_ungate2_s35)
PROBED=("${HET[@]}" "${FLAT[@]}")
ALL=("${HET[@]}" "${FLAT[@]}" "${UNG[@]}")

fail=0
need () { [ -e "$1" ] || { echo "MISSING: $1" >&2; fail=1; }; }
echo "[grad] preflight"
for r in "${ALL[@]}"; do
  need "$RUNROOT/$r/config.yaml"
  need "$RUNROOT/$r/metrics.jsonl"
  need "$RUNROOT/$r/scores.jsonl"
  need "$RUNROOT/$r/ckpt/latest"
  n=$(ls "$RUNROOT/$r"/replay/*.npz 2>/dev/null | wc -l)
  [ "$n" -ge 1000 ] || { echo "REPLAY TOO SMALL: $r has $n chunks (expected >1000)" >&2; fail=1; }
  # Realized-step witness: the phase-invariant completion fact. nearest.json is
  # pretty-printed, so "milestone" and "step" land on SEPARATE lines and a
  # line-wise grep can never match them together -- the wave checker gets away
  # with the same pattern only because Python's \s+ spans newlines. Flatten
  # first, or this refuses every run that is in fact complete.
  tr -d '\n' < "$RUNROOT/$r/ckpt_snapshots/nearest.json" 2>/dev/null \
    | grep -qE '"milestone": 500000,[[:space:]]*"step": (49[0-9]{4}|[5-9][0-9]{5})' \
    || { echo "STEP WITNESS FAILS: $r" >&2; fail=1; }
done
for r in "${PROBED[@]}"; do
  need "$RUNROOT/$r/se_probe/se_probe.json"
  need "$RUNROOT/$r/se_probe/se_probe_dims.npz"
done
[ "$fail" -eq 0 ] || { echo "[grad] PREFLIGHT FAILED -- nothing bundled" >&2; exit 1; }

echo "[grad] bundle -> $B"
mkdir -p "$B/runroot"
for r in "${ALL[@]}"; do
  d="$B/runroot/$r"; mkdir -p "$d/ckpt"
  cp -a "$RUNROOT/$r/config.yaml"   "$d/"
  cp -a "$RUNROOT/$r/metrics.jsonl" "$d/"
  cp -a "$RUNROOT/$r/scores.jsonl"  "$d/"
  [ -f "$RUNROOT/$r/TRAINING_DONE" ] && cp -a "$RUNROOT/$r/TRAINING_DONE" "$d/"
  cp -a "$RUNROOT/$r/ckpt/latest"   "$d/ckpt/"
  ls -la "$RUNROOT/$r/ckpt" > "$d/ckpt/DIR_LISTING.txt" 2>/dev/null
  cp -a "$RUNROOT/$r/ckpt_snapshots/nearest.json" "$d/ckpt_snapshots_nearest.json" 2>/dev/null
  cp -a "$RUNROOT/$r/replay" "$d/replay"
done
for r in "${PROBED[@]}"; do
  cp -a "$RUNROOT/$r/se_probe" "$B/runroot/$r/se_probe"
done

mkdir -p "$B/_cloud_logs"
for r in "${ALL[@]}"; do
  [ -f "$RUNROOT/_cloud_logs/$r.out" ] && cp -a "$RUNROOT/_cloud_logs/$r.out" "$B/_cloud_logs/"
done
cp -a /home/rickybao/projects/dreamerv3/se_grad_probe_53431914.out "$B/_cloud_logs/" 2>/dev/null

# post-copy counts on what actually landed
nr=$(ls -d "$B"/runroot/*/ | wc -l)
np=$(ls -d "$B"/runroot/*/se_probe 2>/dev/null | wc -l)
[ "$nr" -eq 12 ] && [ "$np" -eq 8 ] || {
  echo "[grad] COUNT MISMATCH runs=$nr(want 12) probed=$np(want 8)" >&2; exit 1; }

cat > "$B/BUNDLE_NOTE.md" <<'EOF'
# SE gradient arm (amendment 3) — read bundle

Built on RCC from the full pulled runroot, not on the instances. All five
instances were destroyed after file-count + FILE-byte parity checks; the
runroot here is the archive.

Contents per §5: all 12 runs carry replay/, scores.jsonl, config.yaml,
metrics.jsonl and ckpt/latest (+ DIR_LISTING). The 8 hetero and flat runs also
carry se_probe/; ungated2 does not, by registration.

The hetero se_probe/ passes are the manipulation-check gate input (§4). All 8
probes ran in ONE caslake CPU job (53431914, COMPLETED 00:09:53, ok=8
failed=0), which is what satisfies "all probe passes on one device class" —
se_probe defaults to --platform cpu, frozen defaults, default output path.

EXCLUDED: ckpt_snapshots/ payload (~3.9 G/run), plugins/, scope/, and all
checkpoints beyond ckpt/latest — none appear in the §5 spec. They remain on
RCC scratch. Kept from that tree: ckpt_snapshots_nearest.json per run, the
realized-step witness (all 12 in 493792..499792, clear of the 490000 bar).

EXECUTION PROVENANCE, for the read to disclose: the 12 runs did NOT all run on
one machine. hetero+flat were split 2+2 / 1+1 / 1+1 across three instances so
that every machine carried EQUAL hetero and flat — device is therefore a
balanced nuisance factor inside the primary contrast, not confounded with it.
The 4 ungated2 runs ran on two further instances, so the two dose secondaries
(hetero-ungated2, flat-ungated2) are cross-instance. All were RTX 5060 Ti.
EOF

( cd "$B" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z \
    | xargs -0 sha256sum > MANIFEST.sha256 )
echo "[grad] bundled $(wc -l < "$B/MANIFEST.sha256") files ($(du -sh "$B" | cut -f1)) -> $B"
echo "$B"
