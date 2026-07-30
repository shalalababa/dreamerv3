# Bundle manifests

One `<bundle_name>.sha256` per result bundle, byte-identical to the
`MANIFEST.sha256` generated inside the bundle at its source machine
(`scripts/bundle_manifest.sh generate`). Bundles themselves move by
direct rsync and are never committed (policy 2026-07-30; `local_results/`
is gitignored+untracked — pre-30-Jul bundles remain in git history).

The committed manifest is the registration-record pin for the bundle's
exact bytes. Before executing any frozen read on a rsync'd bundle:

    ./scripts/bundle_manifest.sh verify local_results/<bundle> manifests/<bundle>.sha256
