#!/bin/bash
# ---------------------------------------------------------------------------
# Bundle integrity manifests (results-sync policy v2, adopted 2026-07-30).
#
# Result bundles no longer travel through git (home-quota + inode blowup).
# Instead: generate a MANIFEST.sha256 inside the bundle AT THE SOURCE (RCC),
# rsync the bundle (manifest included) to the analysis machine, verify there,
# then commit ONLY the manifest copy under manifests/<bundle_name>.sha256.
# The committed manifest pins the exact bytes in the registration record;
# frozen reads verify against it before executing.
#
# Usage:
#   ./scripts/bundle_manifest.sh generate <bundle_dir>
#       writes <bundle_dir>/MANIFEST.sha256 over all files in the bundle
#   ./scripts/bundle_manifest.sh verify <bundle_dir> [committed_manifest]
#       re-hashes every file against the bundle's MANIFEST.sha256; fails on
#       any mismatch, missing file, or extra file not in the manifest. If a
#       committed manifest path is given, first requires it to be byte-equal
#       to the bundle's copy.
# ---------------------------------------------------------------------------
set -euo pipefail
mode="${1:?usage: bundle_manifest.sh generate|verify <bundle_dir> [committed_manifest]}"
dir="${2:?usage: bundle_manifest.sh generate|verify <bundle_dir> [committed_manifest]}"
committed="${3:-}"
if [ -n "$committed" ]; then
  [ -f "$committed" ] || { echo "FAIL: committed manifest not found: $committed"; exit 1; }
  committed=$(readlink -f "$committed")
fi
cd "$dir"

case "$mode" in
  generate)
    # Write to a temp file and rename ATOMICALLY. `> MANIFEST.sha256` truncates
    # the old manifest the instant the shell opens the redirect, then refills it
    # over however long the hashing takes -- minutes on a large bundle. Any
    # concurrent reader then sees a VALID-LOOKING but SHORT manifest, and since
    # the entries are path-sorted, what is missing is always the TAIL.
    # 24 Aug 2026: regenerating a2_scarecrow's 81 G manifest took 18:57->19:07;
    # the Paper-5 read sampled it mid-write and found the last two arms
    # (sc_scare2_*, sc_scare_* -- the tail in C-sort order) absent, reported as
    # "full MANIFEST had a gap". Nothing was actually wrong with the bundle.
    # Both names are excluded from find so the manifest can never hash itself
    # (the shell creates the redirect target before find runs).
    find . -type f ! -name MANIFEST.sha256 ! -name MANIFEST.sha256.tmp -print0 \
      | LC_ALL=C sort -z | xargs -0 sha256sum > MANIFEST.sha256.tmp
    mv -f MANIFEST.sha256.tmp MANIFEST.sha256
    echo "generate: $(wc -l < MANIFEST.sha256) files -> $dir/MANIFEST.sha256"
    ;;
  verify)
    [ -f MANIFEST.sha256 ] || { echo "FAIL: no MANIFEST.sha256 in $dir"; exit 1; }
    if [ -n "$committed" ]; then
      cmp -s MANIFEST.sha256 "$committed" \
        || { echo "FAIL: bundle manifest differs from committed $committed"; exit 1; }
    fi
    sha256sum -c --quiet MANIFEST.sha256 \
      || { echo "FAIL: hash mismatch or missing file in $dir"; exit 1; }
    n_manifest=$(wc -l < MANIFEST.sha256)
    n_disk=$(find . -type f ! -name MANIFEST.sha256 | wc -l)
    [ "$n_manifest" -eq "$n_disk" ] \
      || { echo "FAIL: file count mismatch (manifest $n_manifest vs disk $n_disk — extra files present)"; exit 1; }
    echo "verify OK: $n_manifest files, $dir matches MANIFEST.sha256${committed:+ (== committed copy)}"
    ;;
  *)
    echo "unknown mode: $mode (use generate|verify)"; exit 2
    ;;
esac
