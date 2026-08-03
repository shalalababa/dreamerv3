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
  ax1wm_finger_s25q1s*_seed* ax1wm_finger_fs25q1s*_seed*
  adapt_ax1s25q1s* adapt_ax1fs25q1s*
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
)

KEEP_SUBSTRATE=(
  axis1_finger axis1_cup axis1_reacher axis1_synth   # ALL frozen buffers
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
  grep '^DELETE-SAFE	' "$MANIFEST" | cut -f2 | while read -r path; do
    echo "rm -rf $path"; rm -rf "$path"
  done
  echo "done; manifest preserved at $MANIFEST"
else
  echo
  echo "Dry run only. Review $MANIFEST, then rerun with CONFIRM=DELETE to"
  echo "remove the DELETE-SAFE set. DELETE-AFTER/KEEP/REVIEW are never auto-removed."
fi
