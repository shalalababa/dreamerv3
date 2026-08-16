#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LeWM step 11 -- assemble the read bundle. (PREREG_lewm_20260812 §11.)
#
# The registered contents list, verbatim:
#   e4_fingerpx_v1_jp.csv + the pe e4 csv (sha 9907d733...) + auc.csv +
#   16 fidelity.json + jp_enc_checks.json + jp_fit_counters.json +
#   jp_ckpt_steps.json + embed_control.json + 32 probe jsons +
#   16+16 embed manifests + jp fit/adapt config.yamls + the sweep log;
#   rsync + sha256 manifest.
#
# "32 probe jsons" = the 16 fpx ANCHOR probes (step 5, ridge on the recon
# donors) plus the 16 EMBEDDING probes (step 6, the distill-free P-J1-emb
# leg). Both are asserted here rather than globbed loosely: an anchor panel
# that is silently 15/16 is exactly the failure a read cannot see.
#
# The pe csv is verified against the sha the frozen reader pins. The reader
# refuses on mismatch, so failing here is strictly better than failing there.
#
# Also shipped, clearly OUTSIDE the registered list and labelled as ops
# provenance: the device sidecars (if any) and the pe one-cell device diff.
# Neither is consumed by lewm_read; both exist so the RECORD can state which
# GPU produced which panel without a second round of sacct archaeology.
# ---------------------------------------------------------------------------
set -uo pipefail
RUNROOT="${RUNROOT:-/scratch/midway3/rickybao/dreamerv3_runs}"
PE_SHA_EXPECT="9907d7337a84c02c0820473c0455cd033dce200f5fde069bfde119d6892cc6e8"
stamp="$(date +%Y%m%d_%H%M%S)"
bundle="$RUNROOT/_bundles/lewm_${stamp}"

fail=0
need () { [ -e "$1" ] || { echo "MISSING: $1" >&2; fail=1; }; }
count_is () {  # count_is <n> <glob...>
  local want="$1"; shift
  local got; got=$(ls -d "$@" 2>/dev/null | wc -l)
  [ "$got" -eq "$want" ] || { echo "COUNT: want $want got $got for: $1" >&2; fail=1; }
}

echo "[lewm] preflight"
need "$RUNROOT/e4_fingerpx_v1_jp.csv"
need "$RUNROOT/e4_fingerpx_v1_pe.csv"
need "$RUNROOT/analysis_out_jp/auc.csv"
need "$RUNROOT/jp_enc_checks.json"
need "$RUNROOT/jp_fit_counters.json"
need "$RUNROOT/jp_ckpt_steps.json"
need "$RUNROOT/jp_integrity_sweep.log"
need "$RUNROOT/lewm_emb/embed_control.json"
count_is 16 "$RUNROOT"/lewm_distill_finger_s[01]_seed[1-8]/fidelity.json
count_is 16 "$RUNROOT"/lewm_probe/fpx_s[01]_seed[1-8].json
count_is 16 "$RUNROOT"/lewm_probe/lewm_finger_s[01]_seed[1-8].json
count_is 16 "$RUNROOT"/lewm_emb/s[01]_seed[1-8]/embed_manifest.json
count_is 16 "$RUNROOT"/lewm_emb/s[01]_seed[1-8]_probeset/embed_manifest.json
count_is 16 "$RUNROOT"/ax1wm_finger_jppxq1ms[01]_seed[1-8]/config.yaml
count_is 16 "$RUNROOT"/adapt_ax1jppxq1ms[01]_finger_seed[1-8]_ckpt500000/config.yaml

pe_sha="$(sha256sum "$RUNROOT/e4_fingerpx_v1_pe.csv" 2>/dev/null | awk '{print $1}')"
if [ "$pe_sha" != "$PE_SHA_EXPECT" ]; then
  echo "PE CSV SHA MISMATCH -- the frozen reader pins $PE_SHA_EXPECT" >&2
  echo "  got: ${pe_sha:-<unreadable>}" >&2
  fail=1
else
  echo "  pe csv sha OK (${pe_sha:0:16}...)"
fi
[ "$fail" -eq 0 ] || { echo "[lewm] PREFLIGHT FAILED -- nothing bundled" >&2; exit 1; }

echo "[lewm] bundle -> $bundle"
mkdir -p "$bundle"/{probe,emb_manifests,fidelity,configs/fit,configs/adapt,collates,provenance}

cp "$RUNROOT/e4_fingerpx_v1_jp.csv"       "$bundle/collates/"
cp "$RUNROOT/e4_fingerpx_v1_pe.csv"       "$bundle/collates/"
cp "$RUNROOT/analysis_out_jp/auc.csv"     "$bundle/collates/"
[ -f "$RUNROOT/analysis_out_jp/exclusions.csv" ] && \
  cp "$RUNROOT/analysis_out_jp/exclusions.csv" "$bundle/collates/"
cp "$RUNROOT"/jp_enc_checks.json "$RUNROOT"/jp_fit_counters.json \
   "$RUNROOT"/jp_ckpt_steps.json "$RUNROOT"/jp_integrity_sweep.log "$bundle/"
cp "$RUNROOT/lewm_emb/embed_control.json" "$bundle/"

cp "$RUNROOT"/lewm_probe/fpx_s[01]_seed[1-8].json             "$bundle/probe/"
cp "$RUNROOT"/lewm_probe/lewm_finger_s[01]_seed[1-8].json     "$bundle/probe/"
for side in 0 1; do for f in 1 2 3 4 5 6 7 8; do
  cp "$RUNROOT/lewm_distill_finger_s${side}_seed${f}/fidelity.json" \
     "$bundle/fidelity/s${side}_seed${f}.json"
  cp "$RUNROOT/lewm_emb/s${side}_seed${f}/embed_manifest.json" \
     "$bundle/emb_manifests/s${side}_seed${f}.json"
  cp "$RUNROOT/lewm_emb/s${side}_seed${f}_probeset/embed_manifest.json" \
     "$bundle/emb_manifests/s${side}_seed${f}_probeset.json"
  cp "$RUNROOT/ax1wm_finger_jppxq1ms${side}_seed${f}/config.yaml" \
     "$bundle/configs/fit/ax1wm_finger_jppxq1ms${side}_seed${f}.yaml"
  cp "$RUNROOT/adapt_ax1jppxq1ms${side}_finger_seed${f}_ckpt500000/config.yaml" \
     "$bundle/configs/adapt/adapt_ax1jppxq1ms${side}_finger_seed${f}_ckpt500000.yaml"
done; done

# -- ops provenance (NOT registered, NOT consumed by lewm_read) -------------
cp "$RUNROOT"/lewm_probe/*.device.json "$bundle/provenance/" 2>/dev/null || true
if [ -d "$RUNROOT/pe_e4_devdiff" ]; then
  cp -r "$RUNROOT/pe_e4_devdiff" "$bundle/provenance/"
fi
cat > "$bundle/provenance/NOTES.md" <<'EOF'
# Ops provenance (not registered; lewm_read does not consume anything here)

* **Which GPU produced which panel.** The E4 pass (step 9) and the embedding
  probes (step 6) both ran on Midway3 rtx6000 nodes; the 16 fpx anchor probes
  (step 5) were measured on a single Vast RTX 5060 Ti after the three
  RCC-Slurm survivors were replaced by same-device re-measures
  (`ops/waves/lewm_probe_devcheck/FINDING.md`). Each panel is therefore
  internally device-uniform, which is the property that matters: the ridge
  probe is bitwise-deterministic within a GPU but shifts up to 0.113 AUROC
  across architectures.

* **Why nothing ran on a V100.** The current env cannot use Midway3's V100
  nodes at all: jax/jaxlib 0.4.33 fail to bring up the CUDA backend on Volta
  and fall through to ROCm (`no attribute 'GpuAllocatorConfig'`), and pixel
  convolutions there fail earlier still with `cudnn status: 5003`. rtx6000
  (0282-0286) and a100 (0294) are unaffected.

* **`pe_e4_devdiff/`** re-measures ONE pe cell (`ax1wm_finger_pepxq1ms0_seed1`)
  on rtx6000 and compares against the pinned historical csv, to size the
  device term on the estimand the read actually uses rather than on
  `deter_std_recomputed`:

  | field | historical (pinned) | re-measured | delta | rel |
  |---|---|---|---|---|
  | `rew_nll_all` | 3.4684435691 | 3.4609700066 | -7.47e-03 | -0.215% |
  | `rew_nll_in`  | 47.7268089370 | 47.6274460735 | **-9.94e-02** | **-0.208%** |
  | `rew_nll_out` | 0.0036260719 | 0.0033461700 | -2.80e-04 | -7.719% |

  `n_in_regime` is 8721 in both, so the difference is numeric, not a change
  of strata. NOTE this is an UPPER BOUND on the device term: two weeks and a
  possible env change separate the two measurements, so it bundles device
  with any other drift. It is emphatically not the ~1e-5 that
  `deter_std_recomputed` showed -- that quantity is not this quantity.
EOF

( cd "$bundle" && find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
n=$(wc -l < "$bundle/MANIFEST.sha256")
echo "[lewm] bundled $n files -> $bundle"
echo "$bundle"
