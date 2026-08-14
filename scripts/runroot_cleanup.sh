#!/bin/bash
# ---------------------------------------------------------------------------
# $RUNROOT cleanup classifier (built 2026-07-30 from the registration record;
# see the keep/delete sweep recorded in the main chat + STUDY_LEDGER.md).
#
# DRY-RUN BY DEFAULT: prints per-category sizes and writes a manifest of
# what WOULD be deleted. Nothing is removed unless you run:
#     CONFIRM=DELETE ./scripts/runroot_cleanup.sh
# and even then ONLY the DELETE-SAFE category is removed. DELETE-AFTER /
# KEEP-* / REVIEW are never touched by this script.
#
# Categories:
#   KEEP-SUBSTRATE  frozen buffers + build sources; owner decision only, ever
#   KEEP-PENDING    inputs to registered not-yet-executed/verified reads
#   DELETE-AFTER    deletable once the named event completes (not automated)
#   DELETE-SAFE     study closed, read executed+verified, no registered consumer
#   REVIEW          known-fuzzy or unmatched — classify by hand vs the ledger
#
# Traps encoded below (do not "simplify" them away):
#   * pretrain_p2e_finger_seed99 is a REGISTERED pixel-pool source — the
#     seed-99 smoke purge is scoped to ax1wm_/adapt_ prefixes only.
#   * 12m (s12/fs12) dirs are DELETE-AFTER the 25m read: the frozen
#     compcapacity reader asserts 12m rows in the same canonical csv.
#   * U-wave adapt dirs hold the ONLY copy of undecided scores.jsonl.
# ---------------------------------------------------------------------------
set -uo pipefail
ROOT="${RUNROOT:?set RUNROOT}"
cd "$ROOT" || exit 1
MANIFEST="${MANIFEST_OUT:-$HOME/runroot_cleanup_manifest_$(date +%Y%m%d_%H%M%S).txt}"
shopt -s nullglob

DELETE_SAFE=(
  # phase-5a dose-response adapt grids (reads closed; curves use committed csvs)
  adapt_apt_*_ckpt* adapt_p2e_*_ckpt* adapt_random_*_ckpt*
  adapt_goalwm_* adapt_p2elong_* adapt_rrwa_* adapt_rrwb_* adapt_rrwc_* adapt_rrwd_*
  # occupancy-dose + corrective robustness waves (superseded by Act-1 pivot)
  ax1wm_*_d0_seed* ax1wm_*_d1_seed* ax1wm_*_d2_seed* ax1wm_*_d3_seed*
  adapt_ax1d0_* adapt_ax1d1_* adapt_ax1d2_* adapt_ax1d3_*
  ax1wm_*r1s*_seed* ax1wm_*r2s*_seed* adapt_ax1r1s* adapt_ax1r2s*
  # W screening wave (w0-w14, read 07-16)
  ax1wm_*_w*s*_seed* ax1wm_*_fw*s*_seed* adapt_ax1w*_*_seed* adapt_ax1fw*_*_seed*
  # W0-W3 adapt runs (reads 07-13/07-16; U baselines = committed csvs)
  adapt_ax1q1s*_* adapt_ax1fq1s*_* adapt_ax1q2s*_* adapt_ax1fq2s*_*
  # W0/W2/W3 fits with no U-wave claim: all cup fits, finger q2, finger q1 seeds 9-16
  ax1wm_cup_*_seed*
  ax1wm_finger_q2s*_seed* ax1wm_finger_fq2s*_seed*
  ax1wm_finger_q1s*_seed9 ax1wm_finger_q1s*_seed1[0-6]
  ax1wm_finger_fq1s*_seed9 ax1wm_finger_fq1s*_seed1[0-6]
  ax1wm_finger_rgoq1s*_seed9 ax1wm_finger_rgoq1s*_seed1[0-6]
  ax1wm_finger_sgbq1s*_seed9 ax1wm_finger_sgbq1s*_seed1[0-6]
  # P3 arms closed (vgo dead; shuffle/relabel exhibits recorded)
  ax1wm_finger_vgoq1s*_seed* ax1wm_finger_shq1s*_seed* ax1wm_finger_rlq1s*_seed*
  adapt_ax1rgoq1s* adapt_ax1sgbq1s* adapt_ax1vgoq1s* adapt_ax1shq1s* adapt_ax1rlq1s*
  # extended-fit discriminator (read 07-26, no further arms)
  ax1wm_finger_xvgo* ax1wm_finger_xsgb* adapt_ax1xvgo* adapt_ax1xsgb*
  # frozen orthogonal + unfrozen-calibration adapts (reads 07-25)
  adapt_ax1ogrgo* adapt_ax1ogsgb* adapt_ax1ufz*
  # frozen stamping adapts (read 07-18; U2 baseline = committed csv)
  adapt_ax1srd0q1s* adapt_ax1srd1q1s* adapt_ax1sidq1s*
  # pixel X2 adapts (read 07-24) + X0-era abandoned pair
  adapt_ax1pxpxq1m* adapt_ax1fpxpxq1m*
  axis1_finger/pxq1
  # volume fits+adapts (Opt-C 07-18, corrective 07-19, replication 07-29)
  ax1wm_finger_v2q1v200s*_seed* ax1wm_finger_fv2q1v200s*_seed*
  ax1wm_finger_v4q1v400s*_seed* ax1wm_finger_fv4q1v400s*_seed*
  adapt_ax1v2q1v200s* adapt_ax1v4q1v400s* adapt_ax1fv2q1v200s* adapt_ax1fv4q1v400s*
  # synth (leg closed 07-23; extension dropped 07-29)
  ax1wm_synth_*_seed* adapt_*_synth_seed*
  # replication/robustness batteries (reads done). NOTE: rd globs are
  # digit-anchored (ax1rd<k>q1s<side> per synth_rediag_read.MODE_RE) so
  # they can never match the rde wave's runs (ax1rdepxq1ms*/rdesmoke —
  # PREREG_rde_pixel_20260802; a bare rd* here would have deleted them).
  adapt_ax1rb* adapt_ax1rbf* adapt_ax1rd[0-9]* ax1wm_finger_rb* ax1wm_finger_rd[0-9]*
  # gate-0 probe pilots (decision frozen 07-02; goal pilots are NOT here)
  pilot_p2e_*_seed1 pilot_apt_*_seed1 pilot_random_*_seed1
  # smokes + timing junk (scoped prefixes — never touches pretrain_*)
  speedtest_* ax1wm_*seed99* adapt_*seed99*
  # U2/U3/U4 unfrozen stress: reads verified 07-30
  # (artifacts/unfrozen_stress_u234_20260730 — scores bundled + hash-pinned)
  ax1wm_finger_srd0q1s*_seed[1-8] ax1wm_finger_srd1q1s*_seed[1-8] ax1wm_finger_sidq1s*_seed[1-8]
  ax1wm_finger_rgoq1s*_seed[1-8] ax1wm_finger_sgbq1s*_seed[1-8]
  ax1wm_finger_pxpxq1ms*_seed[1-8]
  # NOTE: fpxpx fits NOT here — they are pretrained-encoder DONORS (see KEEP_PENDING)
  adapt_ax1uzsrd0q1s* adapt_ax1uzsrd1q1s* adapt_ax1uzsidq1s*
  adapt_ax1uzog* adapt_ax1uzpxpxq1ms* adapt_ax1uzfpxpxq1ms*
)

DELETE_AFTER=(  # "glob :: unblocking event"
  # (U2/U3/U4 entries promoted to DELETE_SAFE 07-30 — reads verified)
  "ax1wm_finger_q1s0_seed[1-8] ax1wm_finger_q1s1_seed[1-8] ax1wm_finger_fq1s0_seed[1-8] ax1wm_finger_fq1s1_seed[1-8] :: U1 read verified"
  "adapt_ax1uztq1s* adapt_ax1uzfq1s* :: U1 read verified (only copy of undecided U1 scores until bundled)"
  "ax1wm_finger_s12q1s*_seed* ax1wm_finger_fs12q1s*_seed* adapt_ax1s12q1s* adapt_ax1fs12q1s* :: 25m read verified (its reader asserts 12m rows)"
  "tm2wm_* adapt_tm2* tm2_data :: TD-MPC2 replication registration names its substrate"
  # (r3_local/r3_labels moved to KEEP_PENDING 30 Jul — R3 substrate gained
  #  new registered consumers; see the r3 block in KEEP_PENDING)
  "runs.csv _submit_runlists _bundles :: all pending reads done (bookkeeping)"
)

KEEP_PENDING=(
  # (u2/u3/u4 uz adapts promoted to DELETE_SAFE 07-30 after read verification;
  #  U1's remain the only copy of undecided scores and moved to DELETE_AFTER)
  # horizon wave (PREREG_horizon_20260811): 5e5-step unfrozen + scratch runs
  adapt_ax1hz* adapt_hscratch_*
  # TM2 legibility diagnostics (PREREG_tm2_diag_20260811): probe jsons
  # + the wave's INPUT substrate (reviewer-2 M1: the fits/data were
  # DELETE-AFTER with a satisfied unblocking event - donor-dependency)
  tm2_diag tm2wm_finger_*q1s*_seed[1-8] tm2_data
  # B1 capacity-descent (PREREG_capdescent_20260811): fits+adapts
  # (seed-restricted so seed99 smoke dirs stay DELETE-SAFE - rev-3 F13;
  # ctx pairs live under axis1_finger = KEEP_SUBSTRATE)
  ax1wm_finger_*sk[13]q1s*_seed[1-4] adapt_ax1*sk[13]q1s*_seed[1-4]_*
  # B2 nuisance-injection (PREREG_nuisance_20260811): fits+adapts
  ax1wm_finger_*nz[12]q1s*_seed[1-4] adapt_ax1*nz[12]q1s*_seed[1-4]_*
  # sigma-ladder extension (PREREG_sigma_ladder_20260814): fits+adapts
  ax1wm_finger_*nzs[28]q1s*_seed[1-4] adapt_ax1*nzs[28]q1s*_seed[1-4]_*
  ax1wm_finger_s25q1s*_seed* ax1wm_finger_fs25q1s*_seed*
  adapt_ax1s25q1s* adapt_ax1fs25q1s*
  # 2026-08-08 review-resolution adapt-only waves. Reads verified 08-10,
  # BUT the rif/p2eof/p2eou run dirs (ckpt + scores.jsonl) are now the
  # POLICY POOL of PREREG_goodhart_partial_20260811 — hold until its
  # scoring bundle is synced + the ONE read verifies. ter/rndte/tm2mf
  # holds release on bundle-sync verification as before.
  adapt_rif_* adapt_p2eof_* adapt_p2eou_* adapt_ax1ter* adapt_tm2mf*
  adapt_rndte_*
  # turn_easy positive-control DONORS (PREREG_orthogonal_poscontrol_20260808):
  # these globs also sit in DELETE-SAFE above — KEEP-PENDING wins for any
  # cluster survivors until that wave's read verifies (batch-review B4).
  ax1wm_finger_rgoq1s0_seed[1-4] ax1wm_finger_rgoq1s1_seed[1-4]
  ax1wm_finger_sgbq1s0_seed[1-4] ax1wm_finger_sgbq1s1_seed[1-4]
  # rl/sh own-label RE-FITS (PREREG_rlsh_ownlabel_refit_20260808): the
  # same names sit in the "P3 arms closed" DELETE tier above (originals
  # destroyed in the 1-2 Aug delete-only incident) — KEEP-PENDING wins
  # for the re-fits until the wave's ONE read verifies.
  ax1wm_finger_shq1s*_seed* ax1wm_finger_rlq1s*_seed*
  # cup WM RE-FITS (PREREG_cup_refit_20260808): originals removed without
  # archive; `ax1wm_cup_*_seed*` sits in the DELETE tier above —
  # KEEP-PENDING wins for the regenerated fits until the P-C3 read.
  ax1wm_cup_q1s*_seed* ax1wm_cup_fq1s*_seed*
  # W1 adjudication-wave labels (PREREG_w1_wave_20260809; batch-review
  # m3): registered labels dir convention $RUNROOT/w1_labels — protect
  # until both family reads verify + bundle synced.
  w1_labels
  # s x w_r wave (PREREG_swave_wave_20260809): read verified 08-10, BUT
  # the 88 adapt run dirs are now the main POLICY POOL of
  # PREREG_goodhart_partial_20260811 — hold adapts (and fits/buffers as
  # their provenance) until its scoring bundle is synced + read verified.
  ax1wm_finger_swr*q1s0_seed* ax1wm_finger_swv*q1s0_seed*
  adapt_ax1swr*q1s0_* adapt_ax1swv*q1s0_*
  axis1_finger/q1_s245 axis1_finger/q1_s446
  # finger q1 main-arm RE-FITS (PREREG_finger_refit_20260808): originals
  # destroyed (no archive); the names appear in a delete-eligible
  # "U1 read verified" list below — KEEP-PENDING wins until the #9/P-C3
  # reads verify. Also protects the archive-RESTORED rgo/sgb 1-8 anchors.
  ax1wm_finger_q1s*_seed[1-8] ax1wm_finger_fq1s*_seed[1-8]
  ax1wm_finger_rgoq1s*_seed[5-8] ax1wm_finger_sgbq1s*_seed[5-8]
  # R3 run dirs (ckpt_early + ckpt) are the SUBSTRATE of two registered
  # not-yet-executed waves (donor-dependency lesson, fpxpx incident):
  #   * PREREG_r3_amend2_20260730 (cross-checkpoint consumer: 128 label
  #     passes need BOTH checkpoints of all 32 runs, all-or-nothing)
  #   * PREREG_competence_repair_20260730
  # r3_labels moved here from DELETE-AFTER 30 Jul: consumed by the
  # doubling read (M=8 side) and the repair wave (paired per-state
  # control + trainer input); the xc wave never re-reads it.
  # rep_labels/rep_model are the repair wave's ONLY outputs (repaired
  # npz + sha-pinned model + gate markers) — inputs to its unexecuted
  # ONE read (PREREG_competence_repair_20260730).
  r3_local/r3_cup_e*_seed3? r3_local/r3_finger_e*_seed3?
  r3_local/r3_labels r3_local/xc_labels
  r3_local/rep_labels r3_local/rep_model
  r3_local/r3dbl_labels r3_local/r3_reacher_*
  # pretrained-encoder pixel wave (registered 30 Jul): fpxpx fits are the
  # seed/side-matched ENCODER DONORS — pe read verified 2 Aug, but the
  # hold now transfers to the rde wave (PREREG_rde_pixel_20260802): the
  # fpxpx fits are its latent-alive CALIBRATION reference; keep until the
  # rde calibration E4 pass has run and its csv is bundled off-scratch.
  ax1wm_finger_fpxpxq1ms*_seed[1-8]
  # task-arm dose wave (#21, PREREG_dose_task_20260810): td names are
  # glob-distinct from the DELETE-SAFE reward-free d<l> globs, but pin
  # them here explicitly until the wave's ONE read verifies. Fit glob is
  # domain-broad (review #9: an accidental cup spill would otherwise hit
  # the ax1wm_cup_* DELETE tier).
  ax1wm_*_td[0-3]_seed* adapt_ax1td[0-3]_*
  # W2 mechanism labels (PREREG_antiharvest_mech_20260811) + the
  # evaluator-selection scoring dirs (PREREG_evaluator_selection_20260811,
  # under goodhart_partial/EV_*) — hold until each wave's ONE read
  # verifies + bundle synced. The tm2r3 checkpoint dirs are the W2
  # substrate (same hold as W1 had).
  w2_labels goodhart_partial
  # tm2r3 checkpoint dirs = the W1/W2 substrate (review #20: they matched
  # NO category and sat in the REVIEW bucket; never-delete-without-archive
  # rule pins them here until the W2 read verifies + archive confirmed)
  tm2r3
  # third-domain wave (PREREG_domains_20260812): reacher+walker collector
  # pilots beyond seed1 (pretrain_* and axis1_reacher/axis1_walker live in
  # KEEP_SUBSTRATE), fits + adapts — hold until the wave's ONE read.
  pilot_*_reacher_seed* pilot_*_walker_seed*
  ax1wm_reacher_*q1s*_seed* adapt_ax1*q1s*_reacher_*
  ax1wm_walker_*q1s*_seed* adapt_ax1*q1s*_walker_*
  # TM2 bridge wave (PREREG_tm2_bridge_20260812): baware/bfree/brec fits
  # + adapts (tm2_data blobs already held above for tm2_diag).
  tm2wm_finger_b*q1s*_seed[1-8] adapt_tm2b*q1s*_finger_seed[1-8]_*
  # LeWM/JEPA arm (PREREG_lewm_20260812): LeWM training runs + embedding
  # sidecars + distilled donors + stage-2 fits + adapts. pxq1m buffers sit
  # under axis1_finger = KEEP_SUBSTRATE; fpxpx donors held above (rde) —
  # this wave ALSO consumes them (ridge anchors), same donor-dependency.
  lewm_* ax1wm_finger_jppxq1ms*_seed[1-8] adapt_ax1jppxq1ms*_finger_*
)

KEEP_SUBSTRATE=(
  axis1_finger axis1_cup axis1_reacher axis1_walker axis1_synth  # ALL frozen buffers
  pretrain_* pilot_goal_*_seed1 e4_probesets pixel_x0
)

category () {  # name patterns...
  local name="$1"; shift
  local total=0 n=0
  for pat in "$@"; do
    for path in $pat; do
      [ -e "$path" ] || continue
      sz=$(du -sk "$path" 2>/dev/null | cut -f1)
      total=$((total + sz)); n=$((n + 1))
      echo "$name	$path	$((sz/1024))M" >> "$MANIFEST"
    done
  done
  printf '%-16s %6d entries  %8.1f GB\n' "$name" "$n" "$(echo "$total/1048576" | bc -l)"
}

: > "$MANIFEST"
echo "== runroot cleanup classifier (dry run unless CONFIRM=DELETE) =="
echo "ROOT=$ROOT   manifest -> $MANIFEST"
category KEEP-SUBSTRATE "${KEEP_SUBSTRATE[@]}"
category KEEP-PENDING   "${KEEP_PENDING[@]}"
for entry in "${DELETE_AFTER[@]}"; do
  globs="${entry%% :: *}"; event="${entry##* :: }"
  category "DELETE-AFTER" $globs | sed "s/$/   [after: $event]/"
done
category DELETE-SAFE "${DELETE_SAFE[@]}"

echo
echo "== REVIEW: top-level entries matching NO category (classify by hand) =="
declare -A seen
while IFS=$'\t' read -r _ path _; do seen["${path%%/*}"]=1; done < "$MANIFEST"
for p in * ; do
  [ -n "${seen[$p]:-}" ] && continue
  echo "  REVIEW: $p  ($(du -sh "$p" 2>/dev/null | cut -f1))"
done

if [ "${CONFIRM:-}" = "DELETE" ]; then
  echo
  echo "== CONFIRM=DELETE set: removing DELETE-SAFE entries only =="
  # KEEP-* precedence (s-wave batch review F6): a path matched by BOTH a
  # KEEP_* category and DELETE-SAFE must survive — the comments always
  # claimed this; now the delete loop enforces it. Empty keep-list is
  # safe (grep -f /dev/null matches nothing; -v passes all through).
  KEEPLIST="$(mktemp)"
  grep -E '^KEEP' "$MANIFEST" | cut -f2 | sort -u > "$KEEPLIST" || true
  grep '^DELETE-SAFE	' "$MANIFEST" | cut -f2 | grep -v -x -F -f "$KEEPLIST" | while read -r path; do
    echo "rm -rf $path"; rm -rf "$path"
  done
  rm -f "$KEEPLIST"
  echo "done; manifest preserved at $MANIFEST"
else
  echo
  echo "Dry run only. Review $MANIFEST, then rerun with CONFIRM=DELETE to"
  echo "remove the DELETE-SAFE set. DELETE-AFTER/KEEP/REVIEW are never auto-removed."
fi
