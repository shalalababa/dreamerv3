#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LIGHT companion to generated/bundle.sh -- the copy that travels to WSL.
#
# The full bundle is 18 G for 4 runs, of which ckpt_snapshots/ is 3.7 G per run
# and replay/ another 605 M. Neither is touched by se_noboot_read: the reader
# opens se_probe/{se_probe.json,se_probe_dims.npz}, config.yaml and
# metrics.jsonl. So the light tree is the run minus those two directories --
# ~45 M per run instead of 4.4 G -- and ckpt/ IS kept, because it is the
# final-checkpoint identity witness the read's gate asserts against.
#
# WHY A SECOND BUNDLE RATHER THAN A FILTERED PULL: pull_local.sh verifies the
# arrived bytes against the manifest that travels inside the bundle. Pulling a
# subset of the FULL bundle would fail that verification for every excluded
# file, and "verification failed but it's fine" is exactly the habit that makes
# a real failure unreadable. A light bundle carries its own manifest, so the
# check stays honest.
#
# The full bundle stays on RCC and remains the record. This one is a
# convenience copy; RUNROOT_LIGHT.txt inside it says so, so a later reader
# cannot mistake it for the registered article.
#
# Hardlinked via --link-dest, so it costs no additional scratch bytes.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DV3OPS_ROOT="${DV3OPS_ROOT:-$HOME/projects/dreamerv3}"
source "$DV3OPS_ROOT/scripts/ops/lib/common.sh"

runs=(se_noboot_s40 se_noboot_s41 se_noboot_s42 se_noboot_s43)
EXCLUDE=(--exclude='replay/' --exclude='ckpt_snapshots/')

# NAME: deliberately NOT "se_noboot_light_*". pull_local.sh resolves a bundle
# with `ls -td <prefix>_* | head -1`, and "se_noboot_*" matches
# "se_noboot_light_*" too -- so the light copy, being newer, would be picked
# up by `pull-local --wave se_noboot` and mistaken for the full record.
bundle="$RUNROOT/bundles/nobootlight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$bundle/runroot_light" "$bundle/_cloud_logs"

python3 "$DV3OPS_ROOT/scripts/ops/wavecheck.py" \
  "$DV3OPS_ROOT/ops/waves/se_noboot" --runroot "$RUNROOT" \
  > "$bundle/COMPLETION.txt" || true
cat "$bundle/COMPLETION.txt"

for run in "${runs[@]}"; do
  [ -d "$RUNROOT/$run" ] || { echo "ERROR: missing $run" >&2; exit 1; }
  mkdir -p "$bundle/runroot_light/$run"
  rsync -a "${EXCLUDE[@]}" --link-dest="$RUNROOT/$run" \
    "$RUNROOT/$run/" "$bundle/runroot_light/$run/"
  # The whole point of the light tree is that the read's inputs survive it.
  for need in se_probe/se_probe.json se_probe/se_probe_dims.npz config.yaml metrics.jsonl; do
    [ -f "$bundle/runroot_light/$run/$need" ] || {
      echo "ERROR: light tree dropped a read input: $run/$need" >&2; exit 1; }
  done
done

shopt -s nullglob
for f in "$RUNROOT"/_cloud_logs/se_noboot_s*.out; do cp -a "$f" "$bundle/_cloud_logs/"; done
cp -a "$HERE/generated/runs.json" "$HERE/generated/RUN_IDS.txt" "$bundle/" 2>/dev/null || true

cat > "$bundle/RUNROOT_LIGHT.txt" <<TXT
runroot_light: replay/ and ckpt_snapshots/ are EXCLUDED from this copy.

This is a convenience copy for the analysis machine, not the registered
article. The full bundle (se_noboot_*, ~18 G) stays on RCC under
\$RUNROOT/bundles/ and is what the amendment's §5 bundle spec refers to
("replay/, scores, config, metrics, ckpt/latest, se_probe/").

Kept here: ckpt/, se_probe/, config.yaml, metrics.jsonl, scores.jsonl and
every other loose file -- i.e. everything uncfield/se_noboot_read.py opens,
plus the final-checkpoint identity witness.

Excluded bytes are NOT archived by this script. Archiving is handled
separately; do not treat this copy as an archive.
TXT

"$DV3OPS_ROOT/scripts/bundle_manifest.sh" generate "$bundle"
echo "$bundle"
du -sh "$bundle"
