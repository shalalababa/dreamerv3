"""A3 GENERALITY-LEG FROZEN reader
(PREREG_a3_generality_20260830.md §§3-6; built + selfchecked BEFORE
the wave's compute; ONE execution, explicit --output).

WHAT THIS ADJUDICATES
---------------------
Whether the A1 avoidance result transplants to a SECOND environment
(`dmc_finger_spin`) at a dose- and contrast-matched manipulation. The
PRIMARY is A1's registered primary statistic, transplanted verbatim:

    occupancy(hetero) - occupancy(flat) in the high-amplitude region,
    one-sided NEGATIVE exact label permutation over C(8,4) = 70;
    FIRES iff p <= .05 AND the delta is negative.

The occupancy estimand, the permutation machinery and the >= 1e5-step
replay floor are IMPORTED from the executed frozen readers
(`se_m3_read.occupancy`, `se_m3_read.exact_perm_p`) rather than
reimplemented, so the statistic is bit-identical to A1's. Only the
region-split constant changes, because the region is a property of the
environment: A1 split cheetah visitation at its smoke median
-0.13009691; A3 splits finger visitation at the calibration pilot's
median -0.21752405166625977 (amendment 1 §6).

WHAT IS NOT HERE, AND WHY (full list in prereg §7 "deviations from A1")
----------------------------------------------------------------------
A1's misprice gate, gradient-delivery gate, probe-based degeneracy gate
and ranking SECONDARY-1 all consume `se_probe` / `se_region_probe`
outputs. They are OUT OF SCOPE for this leg:
  * `uncfield/se_region_probe.py` hardcodes the CHEETAH region
    threshold (`REGION_THRESHOLD = se_m3_read.GATE_THRESHOLD`, its
    line 38) and exposes no threshold flag. It is a frozen, executed
    instrument and may not be edited, so no region-resolved delivery
    gate exists for finger without a NEW registered instrument.
  * The degeneracy gate's reference is a Stage-1 CHEETAH level cohort
    (`stage1_reference`, per-run minima of probe-derived levels). Level
    references are not portable across environments and no finger
    Stage-1 cohort exists.
  * SECONDARY-1 reads `pse2` out of the provenance-gated probe record.
In their place this reader carries (a) the REALIZED-DOSE block, which
verifies the transplanted manipulation on the adjudicating data itself,
and (b) a training-telemetry collapse check that needs no probe. A
mechanism/misprice leg for finger is a SEPARATE future registration.

REALIZED DOSE (registered, ANNOTATES ONLY -- never vetoes)
----------------------------------------------------------
The solved ramp (amendment 1 §6) is applied to each run's OWN replay
visitation, giving that run's realized m-bar and realized amplitude
ratio. `ramp_stats` is IMPORTED from the amendment reader, so the ramp
algebra is literally the code that produced the registered constants.

  * FLAT arm: keyed. The flat arm carries no avoidance pressure (its
    delivered amplitude is the constant FLAT_SCALE), so its visitation
    is the uncontaminated control. A flat run outside +-15% on m-bar or
    +-25% on the ratio annotates the read DOSE-DRIFTED.
  * HETERO arm: the same quantities are reported DESCRIPTIVELY and are
    NEVER flagged. Its realized dose is endogenous: amendment 1 §2's own
    language is that "diversion itself raises visited m -- that is the
    measured effect", so flagging it would make a confirmed effect
    annotate itself as an instrument fault.

DOSE-DRIFTED never changes the primary verdict. It is a disclosure the
paper must print beside the result.

Run:  python -m uncfield.se_a3gen_read --hetero "<glob>" \
          --flat "<glob>" --output artifacts/a3_gen_read_<date>
      python -m uncfield.se_a3gen_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import subprocess
import sys
import time

import numpy as np

# --- the registered primary machinery, imported from executed readers ---
from uncfield.se_m3_read import (GATE_INDEX, GATE_KEY, MIN_STEPS,
                                 SCORE_TAIL, exact_perm_p, occupancy)
from uncfield.se_mask import bca_interval
from uncfield.se_read import fit_counters
# --- the calibration instruments (executed; imported, never edited) ---
from uncfield.se_a3cal_read import PHASE_A, MOD_INDEX, SOURCE_KEY, scan_replay
from uncfield.se_a3cal_amend1_read import (CHEETAH_SMOKE, REGISTERED_CHEETAH,
                                           TGT_MBAR, TGT_RATIO, ramp_stats)

# ==================================================================== pins
# Every constant below is frozen at registration. Nothing here is
# estimated from this wave. Provenance: PREREG_a3_calib_amend1_20260830
# §6 (the pin block) and artifacts/a3_calib_amend1_20260830/.

TASK = "dmc_finger_spin"
MOD_KEY = SOURCE_KEY                       # 'position'
MOD_LO = -4.589493315966835                # §6 DISTRACTOR_MOD_LO
MOD_HI = 7.469163326854745                 # §6 DISTRACTOR_MOD_HI
GATE_THRESHOLD = -0.21752405166625977      # §6 GATE_THRESHOLD (pilot median)
FLAT_SCALE = 0.35948428173662794           # §6 FLAT_SCALE
BASESD_PLANTED = 0.54193702                # §6 (Phase-A inventory)
BASESD_N = 1.35803102                      # §6 (Phase-A inventory)
DIM = 8                                    # A1 distractor width
THETA_OU = 0.1                             # config default = phi 0.9 = A1
EXPECT_STEPS = 5e5
ALPHA = 0.05
N_RUNS = 8

HET_SEEDS = frozenset(range(210, 214))
FLAT_SEEDS = frozenset(range(214, 218))
ALL_SEEDS = HET_SEEDS | FLAT_SEEDS
SEED_ARM = dict([(s, "hetero") for s in HET_SEEDS]
                + [(s, "flat") for s in FLAT_SEEDS])

# ---------------------------------------------------- realized-dose targets
# The targets are the SOLVED ramp's own achieved values on the pilot
# visitation (amendment 1 §4 table), not cheetah's raw numbers -- the
# ramp was solved to land 0.16% / 0.33% away from cheetah, and it is
# that solved operating point the A3 runs are supposed to reproduce.
# Both are DERIVED below from the amendment's own published residuals
# and reproduce the artifact values EXACTLY (difference 0.0), so an
# edited constant cannot survive import.
PILOT_RESID_MBAR = 0.001589104487050558    # amendment §4: 0.16%
PILOT_RESID_RATIO = 0.0033290452676080417  # amendment §4: 0.33%
DOSE_TARGET_MBAR = 0.35948428173662794     # = FLAT_SCALE
DOSE_TARGET_RATIO = 1.8636677250074283     # solved ratio (1.8637x)
DOSE_BAND_MBAR = 0.15                      # +-15% (prereg §5.3)
DOSE_BAND_RATIO = 0.25                     # +-25% (prereg §5.3)

# Training-telemetry collapse check (replaces A1's probe-based
# degeneracy gate, whose reference cohort does not exist for finger).
COLLAPSE_KEY = "train/expl/disag_replay_rew"

# ------------------------------------------- registration self-consistency
# Cheap asserts that fire at import: they tie every registered constant
# to the executed calibration instruments and to the amendment's own
# arithmetic, so a silently edited pin cannot be read.
assert abs(TGT_MBAR - 0.35891393) < 1e-15, TGT_MBAR
assert abs(TGT_RATIO - 0.4666 / 0.2512) < 1e-15, TGT_RATIO
assert DOSE_TARGET_MBAR == FLAT_SCALE
assert abs(TGT_MBAR * (1.0 + PILOT_RESID_MBAR) - DOSE_TARGET_MBAR) < 1e-15
assert abs(TGT_RATIO * (1.0 + PILOT_RESID_RATIO) - DOSE_TARGET_RATIO) < 1e-15
assert abs(PHASE_A[TASK]["basesd_planted"] - BASESD_PLANTED) < 1e-15
assert abs(PHASE_A[TASK]["basesd_n"] - BASESD_N) < 1e-15
assert MOD_INDEX == GATE_INDEX == 0 and MOD_KEY == GATE_KEY == "position"
assert MOD_LO < GATE_THRESHOLD < MOD_HI


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hetero", default="", help="4 hetero run dirs (glob)")
    p.add_argument("--flat", default="", help="4 flat run dirs (glob)")
    p.add_argument("--expect_steps", type=float, default=EXPECT_STEPS)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def _stamp():
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = None
    return dict(git=git, time=time.strftime("%Y-%m-%dT%H:%M:%S"))


def _telemetry(run_dir, key=COLLAPSE_KEY):
    """Last logged value of a metrics.jsonl key, or None."""
    try:
        with open(os.path.join(run_dir, "metrics.jsonl"), "rb") as f:
            lines = f.read().splitlines()
        for line in reversed(lines):
            d = json.loads(line)
            if key in d:
                return float(d[key])
    except (OSError, ValueError):
        pass
    return None


def _score_tail(run_dir):
    scores = []
    try:
        with open(os.path.join(run_dir, "scores.jsonl")) as f:
            for line in f:
                rec = json.loads(line)
                if "episode/score" in rec:
                    scores.append(float(rec["episode/score"]))
    except OSError:
        pass
    return ((float(np.mean(scores[-SCORE_TAIL:])) if scores else None),
            len(scores))


# ------------------------------------------------------------------ per-run

def read_run(run_dir, arm):
    """FATAL identity gates (prereg §5.1) + the per-run record.

    A defective run refuses the WHOLE read rather than being silently
    dropped, and the refusal happens before any output is written.
    """
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = cfg["distractor"]
    pl_ = cfg["planted"]
    ex = cfg["agent"]["expl"]

    # (a) the seed<->arm map is the ARM AUTHORITY: the arm is decided by
    # the registered seed block, and the config must AGREE with it --
    # never the other way round, or a mis-set knob silently relabels a
    # run into the other arm (phisweep §5.2 pattern).
    seed = int(cfg["seed"])
    assert seed in SEED_ARM, (
        f"{run_dir}: seed {seed} is not a registered A3 seed "
        f"{sorted(ALL_SEEDS)}")
    assert SEED_ARM[seed] == arm, (
        f"{run_dir}: seed {seed} is registered for the "
        f"{SEED_ARM[seed]} arm but was passed as {arm} -- WRONG ARM MAP")

    # (b) environment + Stage-1-verbatim pin block (A1 §1 / review #27
    # B3), retargeted to the finger calibration constants
    assert str(cfg.get("task", "")) == TASK, (
        f"{run_dir}: task {cfg.get('task')!r} != {TASK}")
    assert str(ex["mode"]) == "p2e" and \
        ex["disag_bootstrap"] is True and \
        int(ex["disag_ens"]) == 8 and \
        abs(float(ex["disag_scale"]) - 1000.0) < 1e-9 and \
        abs(float(ex.get("disag_bootstrap_prob", 0.8)) - 0.8) < 1e-9, (
        f"{run_dir}: expl pins differ from Stage-1 (EXPL_CONFIG or "
        f"DISAG_BOOTSTRAP override leak?)")
    # disag_head postdates A1's pin block and b4_alea ran a 'gauss' arm
    # on this producer -- an unpinned head is a live leak path
    assert str(ex.get("disag_head", "det")) == "det", (
        f"{run_dir}: agent.expl.disag_head {ex.get('disag_head')!r} "
        f"!= det (DISAG_HEAD leak from the alea wave?)")
    assert ex.get("penalty_mix", False) is False and \
        abs(float(cfg.get("penalty", {}).get("scale", 0.0))) < 1e-12 and \
        str(cfg.get("penalty", {}).get("gate_key", "")) == "", (
        f"{run_dir}: penalty machinery must be inert")

    # (c) the finger basesd PAIR -- a cheetah default (0.0976 / 1.215)
    # leaking through is the single most likely producer fault
    assert str(pl_["source_key"]) == SOURCE_KEY and \
        abs(float(pl_["basesd"]) - BASESD_PLANTED) < 1e-9, (
        f"{run_dir}: planted pins {pl_['source_key']!r}/{pl_['basesd']!r} "
        f"differ from the Phase-A finger inventory "
        f"({SOURCE_KEY}/{BASESD_PLANTED}) -- cheetah default leak?")
    assert int(d["dim"]) == DIM and \
        abs(float(d["basesd"]) - BASESD_N) < 1e-9, (
        f"{run_dir}: distractor dim/basesd {d['dim']}/{d['basesd']!r} "
        f"differ from the registered {DIM}/{BASESD_N}")
    # the OU persistence knob: the phisweep wave moves exactly this
    # value on this producer and runs immediately before A3
    assert abs(float(d.get("theta", THETA_OU)) - THETA_OU) < 1e-12, (
        f"{run_dir}: distractor.theta {d.get('theta')!r} != {THETA_OU} "
        f"(DISTRACTOR_THETA leak from the phi-sweep?)")

    # (d) all training-time gates OFF (the region split is a READER
    # constant in this design, not a producer knob)
    assert str(pl_.get("gate_key", "")) == "" and \
        str(d.get("gate_key", "")) == "", (
        f"{run_dir}: training-time gates must be off on both wrappers")

    # (e) arm construction: the full mod quadruple, or the flat scale
    mk = str(d.get("mod_key", ""))
    scale = float(d["scale"])
    if arm == "hetero":
        assert mk == MOD_KEY and int(d["mod_index"]) == MOD_INDEX and \
            abs(float(d["mod_lo"]) - MOD_LO) < 1e-9 and \
            abs(float(d["mod_hi"]) - MOD_HI) < 1e-9 and \
            abs(scale - 1.0) < 1e-12, (
            f"{run_dir}: hetero pins -- mod quadruple must be "
            f"({MOD_KEY}, {MOD_INDEX}, {MOD_LO}, {MOD_HI}) at scale 1.0, "
            f"got ({mk!r}, {d.get('mod_index')!r}, {d.get('mod_lo')!r}, "
            f"{d.get('mod_hi')!r}) at {scale!r}")
    else:
        assert mk == "" and abs(scale - FLAT_SCALE) < 1e-12 and \
            abs(float(d["mod_lo"]) - 0.0) < 1e-12 and \
            abs(float(d["mod_hi"]) - 1.0) < 1e-12, (
            f"{run_dir}: flat pins -- mod_key must be empty at the "
            f"registered inert ramp (0.0, 1.0) and scale {FLAT_SCALE}, "
            f"got {mk!r} / ({d.get('mod_lo')!r}, {d.get('mod_hi')!r}) "
            f"at {scale!r}")

    # (f) steps
    assert abs(float(cfg["run"]["steps"]) - EXPECT_STEPS) < 1e-6, (
        f"{run_dir}: run.steps {cfg['run']['steps']!r} != {EXPECT_STEPS}")

    # ---- the registered primary statistic, from the frozen estimand ----
    occ, n_steps = occupancy(run_dir, GATE_INDEX, GATE_THRESHOLD)

    # ---- realized dose, from the SAME replay via the amendment's ramp ----
    x = scan_replay(run_dir)
    assert int(x.size) == int(n_steps), (
        f"{run_dir}: the two ingestion paths disagree on length "
        f"({x.size} vs {n_steps})")
    occ_re = float(np.mean(x > GATE_THRESHOLD))
    assert abs(occ_re - occ) < 1e-12, (
        f"{run_dir}: occupancy recomputed from the dose scan {occ_re!r} "
        f"!= the frozen estimand {occ!r}")
    mbar, ratio = ramp_stats(x, MOD_LO, MOD_HI, GATE_THRESHOLD)
    dose = dict(
        mbar=float(mbar), ratio=float(ratio),
        dev_mbar=float((mbar - DOSE_TARGET_MBAR) / DOSE_TARGET_MBAR),
        dev_ratio=(float((ratio - DOSE_TARGET_RATIO) / DOSE_TARGET_RATIO)
                   if np.isfinite(ratio) else None))

    score, n_scores = _score_tail(run_dir)
    return dict(run_dir=os.path.abspath(run_dir), arm=arm, train_seed=seed,
                occupancy=float(occ), n_steps=int(n_steps),
                valid=bool(n_steps >= MIN_STEPS), dose=dose,
                disag_replay_rew=_telemetry(run_dir),
                score=score, n_score_entries=n_scores)


# ---------------------------------------------------------------- aggregate

def _dose_flagged(rec):
    """The registered band test. FLAT-KEYED: a hetero run is NEVER
    flagged, because its realized dose is endogenous to the hypothesis
    (amendment 1 §2: diversion itself raises visited m)."""
    if rec["arm"] != "flat":
        return False, []
    d, out = rec["dose"], []
    if abs(d["dev_mbar"]) > DOSE_BAND_MBAR:
        out.append(f"m-bar {d['mbar']:.6f} deviates "
                   f"{100 * d['dev_mbar']:+.1f}% from the solved-ramp "
                   f"target {DOSE_TARGET_MBAR} (band "
                   f"+-{100 * DOSE_BAND_MBAR:.0f}%)")
    if d["dev_ratio"] is None or abs(d["dev_ratio"]) > DOSE_BAND_RATIO:
        out.append(f"amplitude ratio {d['ratio']:.6f} deviates "
                   f"{'undefined' if d['dev_ratio'] is None else format(100 * d['dev_ratio'], '+.1f') + '%'}"
                   f" from the solved-ramp target {DOSE_TARGET_RATIO} "
                   f"(band +-{100 * DOSE_BAND_RATIO:.0f}%)")
    return bool(out), out


def _arm_dose(rows):
    return dict(
        per_run={r["train_seed"]: r["dose"] for r in rows},
        mean_mbar=(float(np.mean([r["dose"]["mbar"] for r in rows]))
                   if rows else None),
        mean_ratio=(float(np.mean([r["dose"]["ratio"] for r in rows
                                   if np.isfinite(r["dose"]["ratio"])]))
                    if rows else None))


def aggregate(recs):
    het = sorted((r for r in recs if r["arm"] == "hetero"),
                 key=lambda r: r["train_seed"])
    fl = sorted((r for r in recs if r["arm"] == "flat"),
                key=lambda r: r["train_seed"])
    hs = [r["train_seed"] for r in het]
    fs = [r["train_seed"] for r in fl]
    # DENOMINATOR PINNED AT 8: missing or extra runs refuse the read.
    # The primary is an exact permutation over C(8,4); a partial panel
    # has no registered fallback statistic, and dropping a run silently
    # would change the estimand.
    assert len(set(hs)) == len(hs) and len(set(fs)) == len(fs), (
        f"duplicate seeds: hetero {hs}, flat {fs}")
    assert set(hs) == set(HET_SEEDS), (
        f"hetero seeds {hs} != {sorted(HET_SEEDS)} "
        f"(missing {sorted(HET_SEEDS - set(hs))}, "
        f"extra {sorted(set(hs) - HET_SEEDS)})")
    assert set(fs) == set(FLAT_SEEDS), (
        f"flat seeds {fs} != {sorted(FLAT_SEEDS)} "
        f"(missing {sorted(FLAT_SEEDS - set(fs))}, "
        f"extra {sorted(set(fs) - FLAT_SEEDS)})")
    assert len(recs) == N_RUNS, f"{len(recs)} runs loaded, need {N_RUNS}"

    invalid = [r["run_dir"] for r in recs if not r["valid"]]
    out = dict(
        stamp=_stamp(), n_hetero=len(het), n_flat=len(fl),
        invalid_runs=invalid,
        instrument_flag=("CLEAN" if not invalid else "VALIDITY-VIOLATIONS"),
        registered=dict(
            task=TASK, mod_key=MOD_KEY, mod_index=MOD_INDEX,
            mod_lo=MOD_LO, mod_hi=MOD_HI,
            gate_threshold=GATE_THRESHOLD, flat_scale=FLAT_SCALE,
            basesd_planted=BASESD_PLANTED, basesd_n=BASESD_N,
            distractor_dim=DIM, theta=THETA_OU, steps=EXPECT_STEPS,
            hetero_seeds=sorted(HET_SEEDS), flat_seeds=sorted(FLAT_SEEDS),
            dose_target_mbar=DOSE_TARGET_MBAR,
            dose_target_ratio=DOSE_TARGET_RATIO,
            dose_band_mbar=DOSE_BAND_MBAR,
            dose_band_ratio=DOSE_BAND_RATIO))

    # ---- unconditional descriptives (reported on EVERY branch) ----
    for name, rs in (("hetero", het), ("flat", fl)):
        if len(rs) >= 3:
            m, lo, hi_ = bca_interval(
                np.asarray([r["occupancy"] for r in rs]),
                np.random.default_rng(31 if name == "hetero" else 32),
                2000)
            out[f"occupancy_{name}"] = dict(mean=m, bca=[lo, hi_])
    ordered = het + fl
    if all(r["score"] is not None for r in ordered):
        d_sc, p_sc, _ = exact_perm_p(
            [r["score"] for r in ordered],
            [True] * len(het) + [False] * len(fl), one_sided=False)
        out["score_two_sided"] = dict(delta=d_sc, p=p_sc,
                                      note="reporting only")

    # ---- REALIZED DOSE: computed unconditionally, ANNOTATES ONLY ----
    flags = []
    for r in fl:
        bad, why = _dose_flagged(r)
        if bad:
            flags.append(dict(run_dir=r["run_dir"], seed=r["train_seed"],
                              reasons=why))
    out["realized_dose"] = dict(
        target_mbar=DOSE_TARGET_MBAR, target_ratio=DOSE_TARGET_RATIO,
        band_mbar=DOSE_BAND_MBAR, band_ratio=DOSE_BAND_RATIO,
        flat=_arm_dose(fl), hetero=_arm_dose(het),
        flat_runs_out_of_band=flags,
        drifted=bool(flags),
        note=("FLAT-KEYED and ANNOTATION-ONLY. The flat arm carries no "
              "avoidance pressure, so its visitation is the "
              "uncontaminated control for the delivered dose. The "
              "hetero arm's realized dose is ENDOGENOUS -- amendment 1 "
              "§2: diversion itself raises visited m, that is the "
              "measured effect -- so it is reported descriptively and "
              "NEVER flagged. DOSE-DRIFTED never vetoes or changes the "
              "primary verdict."),
        band_provenance=(
            "Bands measured on four cheetah phi-sweep control runs "
            "(distractor active, unmodulated) with the REGISTERED "
            "CHEETAH ramp applied to their own replay: realized m-bar "
            "mean 0.3820, sd 0.0178 (rel 4.7%); systematic shift vs the "
            "smoke target +6.4%; max |dev| 11.5% (m-bar) / 21.7% "
            "(ratio). TWO DISCLOSED CAVEATS: (i) measured at 88k-112k "
            "of 500k steps, i.e. ~20% of training; (ii) transplanted "
            "from cheetah to finger."))

    # ---- (1) collapse check (replaces A1's probe degeneracy gate) ----
    tele = {r["run_dir"]: r["disag_replay_rew"] for r in recs}
    missing_tele = [k for k, v in tele.items() if v is None]
    dead = [k for k, v in tele.items()
            if v is not None and not (np.isfinite(v) and v > 0.0)]
    out["collapse_check"] = dict(
        key=COLLAPSE_KEY, per_run=tele, dead_runs=dead,
        available=not missing_tele,
        note=("A1's degeneracy gate is probe-based and referenced to a "
              "Stage-1 CHEETAH cohort that does not exist for finger. "
              "This is the weakest defensible transplant: the "
              "training-time intrinsic-reward telemetry must be present "
              "and strictly positive. No invented absolute threshold."))
    if dead:
        out["primary"] = ("GLOBAL-DISAGREEMENT-COLLAPSE: the intrinsic "
                          "objective is dead in at least one run -- "
                          "nothing downstream adjudicated")
        out["outcome"] = "GLOBAL-COLLAPSE"
        return out

    # ---- (2) occupancy validity (A1 step 3) ----
    if invalid or len(het) != 4 or len(fl) != 4:
        out["primary"] = "NOT-ADJUDICABLE (need 4+4 valid runs)"
        out["outcome"] = "NOT-ADJUDICABLE"
        return out

    # ---- (3) PRIMARY: A1's statistic, verbatim ----
    labels = [True] * 4 + [False] * 4
    occ = [r["occupancy"] for r in ordered]
    d_neg, p_occ, total = exact_perm_p([-x for x in occ], labels,
                                       one_sided=True)
    d_occ = -d_neg
    fires = bool(p_occ <= ALPHA and d_occ < 0)
    statement = (
        "REGISTERED generality test of A1: occupancy(hetero) < "
        "occupancy(flat) in the high-amplitude region of "
        "dmc_finger_spin (position[0] > "
        f"{GATE_THRESHOLD}), one-sided negative exact permutation over "
        "C(8,4)=70. Power note: firing requires the observed split "
        "among the 3 most extreme of 70 -- near-complete separation; "
        "this wave has essentially no power against moderate effects "
        "and a non-fire does not distinguish 'no effect' from "
        "'moderate effect'")
    if out["realized_dose"]["drifted"]:
        statement += ("; DOSE-DRIFTED (flat-arm realized dose outside "
                      "the registered band -- disclosure, not a veto)")
    if not out["collapse_check"]["available"]:
        statement += ("; COLLAPSE-CHECK-UNAVAILABLE on "
                      f"{len(missing_tele)} run(s)")
    out["primary"] = dict(
        statement=statement, delta=d_occ, p=p_occ, n_assignments=total,
        fires=fires, min_attainable_p=1.0 / total)
    out["outcome"] = ("AVOIDANCE-GENERALIZES" if fires
                      else "AVOIDANCE-DOES-NOT-GENERALIZE")
    out["not_claimed"] = [
        "no misprice claim in finger: this leg runs no probe, so the "
        "pricing mechanism A1 verified in cheetah is NOT measured here",
        "no region-resolved disagreement decomposition (se_region_probe "
        "is cheetah-thresholded and frozen)",
        "no persistence claim: the A2-extension leg is a SEPARATE "
        "registration and neither leg's outcome gates the other",
        "no claim about environments other than cheetah and finger",
        "a non-fire is a GENERALITY BOUNDARY, never a refutation of the "
        "registered A1 cheetah result"]
    return out


# ---------------------------------------------------------------- rendering

def render(out):
    L = ["# A3 GENERALITY LEG — read\n",
         f"OUTCOME: **{out['outcome']}**\n",
         f"Task {TASK}; hetero seeds {sorted(HET_SEEDS)} vs flat seeds "
         f"{sorted(FLAT_SEEDS)}; region split position[0] > "
         f"{GATE_THRESHOLD}.\n"]
    if isinstance(out.get("primary"), dict):
        p = out["primary"]
        L.append(f"PRIMARY: delta {p['delta']:+.4f}, one-sided p "
                 f"{p['p']:.4f} of {p['n_assignments']} assignments "
                 f"(min attainable {p['min_attainable_p']:.4f}) -> "
                 f"fires = **{p['fires']}**\n")
        L.append(f"> {p['statement']}\n")
    else:
        L.append(f"PRIMARY: {out.get('primary')}\n")
    for arm in ("hetero", "flat"):
        k = f"occupancy_{arm}"
        if k in out:
            o = out[k]
            L.append(f"- occupancy {arm}: {o['mean']:.4f} "
                     f"BCa [{o['bca'][0]:.4f}, {o['bca'][1]:.4f}]")
    if "score_two_sided" in out:
        s = out["score_two_sided"]
        L.append(f"- task return (two-sided, reporting only): "
                 f"delta {s['delta']:+.4f}, p {s['p']:.4f}")
    L.append("\n## Per-run\n")
    L.append("| run | seed | arm | occupancy | steps | valid | realized "
             "m-bar (dev) | realized ratio (dev) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in out.get("per_run", []):
        d = r["dose"]
        dv = ("n/a" if d["dev_ratio"] is None
              else f"{100 * d['dev_ratio']:+.1f}%")
        L.append(f"| {os.path.basename(r['run_dir'])} | {r['train_seed']} "
                 f"| {r['arm']} | {r['occupancy']:.4f} | {r['n_steps']} "
                 f"| {r['valid']} | {d['mbar']:.6f} "
                 f"({100 * d['dev_mbar']:+.1f}%) | {d['ratio']:.6f} "
                 f"({dv}) |")
    rd = out["realized_dose"]
    L.append(f"\n## Realized dose (targets m-bar {rd['target_mbar']}, "
             f"ratio {rd['target_ratio']}; bands "
             f"+-{100 * rd['band_mbar']:.0f}% / "
             f"+-{100 * rd['band_ratio']:.0f}%)\n")
    L.append(f"- FLAT (keyed): mean m-bar {rd['flat']['mean_mbar']}, "
             f"mean ratio {rd['flat']['mean_ratio']}")
    L.append(f"- HETERO (descriptive, never flagged): mean m-bar "
             f"{rd['hetero']['mean_mbar']}, mean ratio "
             f"{rd['hetero']['mean_ratio']}")
    L.append(f"- DOSE-DRIFTED: **{rd['drifted']}**"
             + ("" if not rd["drifted"] else
                " — " + "; ".join(f"seed {f['seed']}: " + " / ".join(f["reasons"])
                                  for f in rd["flat_runs_out_of_band"])))
    L.append(f"- {rd['note']}")
    L.append(f"- BAND PROVENANCE: {rd['band_provenance']}")
    cc = out["collapse_check"]
    L.append(f"\n## Gates\n\n- collapse check available: "
             f"{cc['available']}; dead runs: {cc['dead_runs']}")
    L.append(f"- occupancy validity: {out['instrument_flag']} "
             f"(invalid: {out['invalid_runs']})")
    if "fit_flag" in out:
        L.append(f"- fit counters: {out['fit_flag']}")
    if "not_claimed" in out:
        L.append("\n## NOT claimed by this read\n")
        for n in out["not_claimed"]:
            L.append(f"- {n}")
    return "\n".join(L) + "\n"


def run(args):
    def expand(pat):
        return (sorted(globmod.glob(pat)) if "," not in pat
                else [x.strip() for x in pat.split(",") if x.strip()])

    het = expand(args.hetero)
    fl = expand(args.flat)
    assert het and fl, "need --hetero and --flat"
    assert args.output and args.output != "TBD", (
        "registered read must persist: pass an explicit --output")
    out_json = os.path.join(args.output, "se_a3gen_read.json")
    assert not os.path.exists(out_json), (
        f"{out_json} exists — the read is ONE execution")
    # FATAL identity gates run BEFORE anything is written, so the
    # one-execution guard survives a repair-and-rerun.
    recs = ([read_run(d, "hetero") for d in het]
            + [read_run(d, "flat") for d in fl])
    out = aggregate(recs)
    out["args"] = {k: v for k, v in vars(args).items() if k != "selfcheck"}
    out["fit_counters"] = {r["run_dir"]: fit_counters(
        r["run_dir"], args.expect_steps) for r in recs}
    bad_fit = [rd for rd, fc in out["fit_counters"].items()
               if fc.get("flag") != "OK"]
    out["fit_flag"] = ("OK" if not bad_fit
                       else f"SUSPECT ({len(bad_fit)} runs)")
    if bad_fit and isinstance(out["primary"], dict):
        out["primary"]["statement"] += (
            f"; FIT-SUSPECT on {len(bad_fit)} runs (disclosure carried, "
            f"7-Aug standing rule)")
    out["per_run"] = [{k: r.get(k) for k in
                       ("run_dir", "train_seed", "arm", "occupancy",
                        "n_steps", "valid", "dose", "disag_replay_rew",
                        "score", "n_score_entries")} for r in recs]
    os.makedirs(args.output, exist_ok=True)
    text = render(out)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(args.output, "RESULTS.md"), "w") as f:
        f.write(text)
    print(f"OUTCOME: {out['outcome']}")
    if isinstance(out.get("primary"), dict):
        p = out["primary"]
        print(f"A3 PRIMARY: delta={p['delta']:+.4f} p={p['p']:.4f} "
              f"fires={p['fires']}")
        print(f"  realized dose DRIFTED={out['realized_dose']['drifted']}"
              f"; fit {out.get('fit_flag')}")
    else:
        print(f"A3 PRIMARY: {out.get('primary')}")
    return out


# ---------------------------------------------------------------- selfcheck

def _trace(n, occ, mbar, ratio, jitter=0.05):
    """Synthetic position[0] visitation realizing EXACTLY a chosen
    occupancy, realized m-bar and realized amplitude ratio under the
    registered ramp.

    m = (x - MOD_LO) / (MOD_HI - MOD_LO) is affine on the visited
    support, so with w_hi = occ:
        m_lo = mbar / (w_lo + w_hi * ratio),  m_hi = ratio * m_lo,
    and the two-point support {x_lo, x_hi} realizes all three targets.
    Antithetic +-jitter keeps the trace non-degenerate while preserving
    each block's mean exactly.
    """
    span = MOD_HI - MOD_LO
    n_hi = int(round(occ * n))
    n_lo = n - n_hi
    w_hi, w_lo = n_hi / n, n_lo / n
    m_lo = mbar / (w_lo + w_hi * ratio)
    m_hi = ratio * m_lo
    x_lo, x_hi = m_lo * span + MOD_LO, m_hi * span + MOD_LO
    assert x_lo < GATE_THRESHOLD < x_hi, (x_lo, x_hi)
    # keep every jittered sample on its own side of the threshold and
    # inside the ramp (no clipping), so the realized targets stay EXACT
    room = min(GATE_THRESHOLD - x_lo, x_hi - GATE_THRESHOLD,
               x_lo - MOD_LO, MOD_HI - x_hi)
    assert room > 0, (x_lo, x_hi, room)
    jitter = min(jitter, 0.4 * room)
    out = []
    for base, cnt in ((x_lo, n_lo), (x_hi, n_hi)):
        half = cnt // 2
        blk = np.concatenate([np.full(half, base + jitter),
                              np.full(half, base - jitter),
                              np.full(cnt - 2 * half, base)])
        out.append(blk)
    return np.concatenate(out)


def _fixture(root, name, seed, arm, occ=0.5, mbar=DOSE_TARGET_MBAR,
             ratio=DOSE_TARGET_RATIO, n_steps=120000, chunk=20000,
             cfg_patch=None, rew=1.2e-4, score=1.0, task=TASK,
             steps=EXPECT_STEPS, dim=4):
    """A synthetic A3 run dir: config.yaml + replay chunks + telemetry."""
    import yaml
    rd = os.path.join(root, name)
    os.makedirs(os.path.join(rd, "replay"), exist_ok=True)
    cfg = {
        "task": task, "seed": seed,
        "agent": {"expl": {"mode": "p2e", "disag_bootstrap": True,
                           "disag_bootstrap_prob": 0.8, "disag_ens": 8,
                           "disag_scale": 1000.0, "disag_head": "det",
                           "penalty_mix": False}},
        "penalty": {"scale": 0.0, "gate_key": "", "gate_index": 0,
                    "gate_threshold": 0.0},
        "planted": {"source_key": SOURCE_KEY, "basesd": BASESD_PLANTED,
                    "gate_key": "", "gate_index": 0, "gate_threshold": 0.0},
        "distractor": {"dim": DIM, "basesd": BASESD_N, "theta": THETA_OU,
                       "gate_key": "", "gate_index": 0,
                       "gate_threshold": 0.0, "mod_index": MOD_INDEX,
                       "mod_key": (MOD_KEY if arm == "hetero" else ""),
                       "mod_lo": (MOD_LO if arm == "hetero" else 0.0),
                       "mod_hi": (MOD_HI if arm == "hetero" else 1.0),
                       "scale": (1.0 if arm == "hetero" else FLAT_SCALE)},
        "run": {"steps": float(steps)},
    }
    for path, val in (cfg_patch or {}).items():
        node = cfg
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = val
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    p0 = _trace(n_steps, occ, mbar, ratio)
    rng = np.random.default_rng(seed)
    for ci, s in enumerate(range(0, p0.size, chunk)):
        seg = p0[s:s + chunk]
        arr = np.zeros((seg.size, dim), np.float64)
        arr[:, MOD_INDEX] = seg
        arr[:, 1:] = rng.normal(0, 1, (seg.size, dim - 1))
        np.savez(os.path.join(rd, "replay", f"chunk{ci:04d}.npz"),
                 position=arr, is_first=np.zeros(seg.size, bool))
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        for st in (100000, 300000, 499968):
            f.write(json.dumps({"step": st, COLLAPSE_KEY: rew}) + "\n")
    with open(os.path.join(rd, "scores.jsonl"), "w") as f:
        for i in range(120):
            f.write(json.dumps({"step": i * 1200,
                                "episode/score": score + 0.01 * (i % 3)})
                    + "\n")
    return rd


def _expect_fail(fn, needle):
    try:
        fn()
    except AssertionError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise SystemExit(f"GATE DID NOT FIRE: {needle}")


def _panel(root, sub, het_occ, flat_occ, **kw):
    het = [_fixture(os.path.join(root, sub), f"se_a3gen_het_s{210 + i}",
                    210 + i, "hetero",
                    occ=(het_occ(i) if callable(het_occ) else het_occ),
                    **{k: v for k, v in kw.items()
                       if not k.startswith("flat_")})
           for i in range(4)]
    fl = [_fixture(os.path.join(root, sub), f"se_a3gen_flat_s{214 + i}",
                   214 + i, "flat",
                   occ=(flat_occ(i) if callable(flat_occ) else flat_occ),
                   **{k[5:]: v for k, v in kw.items()
                      if k.startswith("flat_")})
          for i in range(4)]
    return het, fl


def _read_all(het, fl):
    return ([read_run(d, "hetero") for d in het]
            + [read_run(d, "flat") for d in fl])


def selfcheck():
    import shutil
    import tempfile
    root = tempfile.mkdtemp(prefix="a3gen_sc_")
    try:
        # ---- 0) the ramp/target algebra, re-derived at runtime ----
        assert TGT_MBAR * (1 + PILOT_RESID_MBAR) == DOSE_TARGET_MBAR
        assert TGT_RATIO * (1 + PILOT_RESID_RATIO) == DOSE_TARGET_RATIO
        print("  0 targets re-derive EXACTLY from the amendment's own "
              "cheetah anchors x published residuals PASS")

        # ---- A) planted effect -> the primary FIRES at the 1/70 floor ----
        het, fl = _panel(root, "A", 0.45, 0.55)
        agg = aggregate(_read_all(het, fl))
        assert agg["outcome"] == "AVOIDANCE-GENERALIZES", agg["outcome"]
        assert abs(agg["primary"]["p"] - 1 / 70) < 1e-12, agg["primary"]
        assert agg["primary"]["delta"] < 0 and agg["primary"]["fires"]
        assert not agg["realized_dose"]["drifted"]
        assert "occupancy_hetero" in agg and "score_two_sided" in agg
        print("  A planted avoidance: delta %+.4f p %.4f -> FIRES PASS"
              % (agg["primary"]["delta"], agg["primary"]["p"]))

        # ---- B) null and B2) REVERSED must not fire ----
        het, fl = _panel(root, "B", 0.50, 0.50)
        agg = aggregate(_read_all(het, fl))
        assert agg["outcome"] == "AVOIDANCE-DOES-NOT-GENERALIZE"
        assert not agg["primary"]["fires"]
        het, fl = _panel(root, "B2", 0.65, 0.45)
        agg2 = aggregate(_read_all(het, fl))
        assert agg2["outcome"] == "AVOIDANCE-DOES-NOT-GENERALIZE"
        assert not agg2["primary"]["fires"] and agg2["primary"]["delta"] > 0
        print("  B null delta %+.4f p %.4f no-fire; B2 REVERSED delta "
              "%+.4f p %.4f no-fire PASS"
              % (agg["primary"]["delta"], agg["primary"]["p"],
                 agg2["primary"]["delta"], agg2["primary"]["p"]))

        # ---- C) WRONG ARM MAP is FATAL ----
        d = _fixture(root, "C/se_a3gen_het_s214", 214, "hetero")
        _expect_fail(lambda: read_run(d, "hetero"), "WRONG ARM MAP")
        d = _fixture(root, "C/se_a3gen_flat_s210", 210, "flat")
        _expect_fail(lambda: read_run(d, "flat"), "WRONG ARM MAP")
        d = _fixture(root, "C/se_a3gen_het_s99", 99, "hetero")
        _expect_fail(lambda: read_run(d, "hetero"),
                     "is not a registered A3 seed")
        # a hetero config passed under the flat glob (arm swap at read time)
        d = _fixture(root, "C/se_a3gen_het_s211b", 211, "hetero")
        _expect_fail(lambda: read_run(d, "flat"), "WRONG ARM MAP")
        print("  C wrong-arm-map / unregistered-seed -> FATAL PASS")

        # ---- D) producer-default (cheetah) leaks are FATAL ----
        leaks = [
            ("planted.basesd", {"planted.basesd": 0.0976},
             "cheetah default leak"),
            ("distractor.basesd", {"distractor.basesd": 1.215},
             "differ from the registered"),
            ("distractor.theta", {"distractor.theta": 0.2},
             "DISTRACTOR_THETA leak"),
            ("disag_head", {"agent.expl.disag_head": "gauss"},
             "DISAG_HEAD leak"),
            ("mod_lo cheetah", {"distractor.mod_lo": -0.32203162},
             "hetero pins"),
            ("mod_hi cheetah", {"distractor.mod_hi": 0.19711795},
             "hetero pins"),
            ("mod_key off", {"distractor.mod_key": ""}, "hetero pins"),
            ("scale not 1", {"distractor.scale": FLAT_SCALE},
             "hetero pins"),
            ("gate on", {"distractor.gate_key": "position"},
             "gates must be off"),
            ("planted gate on", {"planted.gate_key": "position"},
             "gates must be off"),
            ("penalty live", {"penalty.scale": 1.0},
             "penalty machinery must be inert"),
            ("penalty_mix", {"agent.expl.penalty_mix": True},
             "penalty machinery must be inert"),
            ("bootstrap off", {"agent.expl.disag_bootstrap": False},
             "expl pins differ"),
            ("apt", {"agent.expl.mode": "apt"}, "expl pins differ"),
            ("steps", {"run.steps": 2e5}, "run.steps"),
            ("dim", {"distractor.dim": 0}, "differ from the registered"),
        ]
        for i, (tag, patch, needle) in enumerate(leaks):
            dd = _fixture(root, f"D{i}/se_a3gen_het_s210", 210, "hetero",
                          cfg_patch=patch)
            _expect_fail(lambda dd=dd: read_run(dd, "hetero"), needle)
        # the wrong TASK, and the cheetah flat scale on a flat run
        dd = _fixture(root, "Dt/se_a3gen_het_s210", 210, "hetero",
                      task="dmc_cheetah_run")
        _expect_fail(lambda: read_run(dd, "hetero"), "!= dmc_finger_spin")
        dd = _fixture(root, "Df/se_a3gen_flat_s214", 214, "flat",
                      cfg_patch={"distractor.scale": 0.35891393})
        _expect_fail(lambda: read_run(dd, "flat"), "flat pins")
        print("  D %d producer-default / cheetah-leak fixtures -> FATAL "
              "PASS" % (len(leaks) + 2))

        # ---- E) DOSE-DRIFTED, both directions, FLAT-KEYED ----
        base_dev = dict(het_occ=0.45, flat_occ=0.55)
        for tag, kw in (
                ("m-bar +20%",
                 dict(flat_mbar=DOSE_TARGET_MBAR * 1.20)),
                ("m-bar -20%",
                 dict(flat_mbar=DOSE_TARGET_MBAR * 0.80)),
                ("ratio +30%",
                 dict(flat_ratio=DOSE_TARGET_RATIO * 1.30)),
                ("ratio -30%",
                 dict(flat_ratio=DOSE_TARGET_RATIO * 0.70))):
            het, fl = _panel(root, "E" + tag.replace(" ", "").replace("%", ""),
                             0.45, 0.55, **kw)
            agg = aggregate(_read_all(het, fl))
            assert agg["realized_dose"]["drifted"], (tag, agg["realized_dose"])
            assert len(agg["realized_dose"]["flat_runs_out_of_band"]) == 4
            # ANNOTATES ONLY: the primary is untouched
            assert agg["outcome"] == "AVOIDANCE-GENERALIZES" and \
                agg["primary"]["fires"], tag
            assert "DOSE-DRIFTED" in agg["primary"]["statement"], tag
        # inside the bands -> no flag (band edges)
        for tag, kw in (("m-bar +14%",
                         dict(flat_mbar=DOSE_TARGET_MBAR * 1.14)),
                        ("ratio -24%",
                         dict(flat_ratio=DOSE_TARGET_RATIO * 0.76))):
            het, fl = _panel(root, "Ein" + tag.replace(" ", "").replace("%", ""),
                             0.45, 0.55, **kw)
            agg = aggregate(_read_all(het, fl))
            assert not agg["realized_dose"]["drifted"], (tag, agg)
        print("  E DOSE-DRIFTED fires both directions on m-bar and ratio, "
              "stays silent inside the bands, NEVER vetoes PASS")

        # ---- E2) the HETERO arm is never flagged, even at huge drift ----
        het, fl = _panel(root, "E2", 0.45, 0.55,
                         mbar=DOSE_TARGET_MBAR * 1.5,
                         ratio=DOSE_TARGET_RATIO * 1.5)
        recs = _read_all(het, fl)
        for r in recs:
            bad, _ = _dose_flagged(r)
            assert bad is False or r["arm"] == "flat"
            if r["arm"] == "hetero":
                assert abs(r["dose"]["dev_mbar"]) > 0.4, r["dose"]
                assert abs(r["dose"]["dev_ratio"]) > 0.4, r["dose"]
                assert not _dose_flagged(r)[0]
        agg = aggregate(recs)
        assert not agg["realized_dose"]["drifted"], agg["realized_dose"]
        assert agg["primary"]["fires"]
        assert agg["realized_dose"]["hetero"]["mean_mbar"] > \
            1.4 * DOSE_TARGET_MBAR
        print("  E2 hetero at +50% m-bar / +50% ratio drift (3.3x and "
              "2.0x its band) -> NOT flagged (endogenous by "
              "construction) PASS")

        # ---- F) missing / extra / duplicate runs refuse the read ----
        het, fl = _panel(root, "F", 0.45, 0.55)
        _expect_fail(lambda: aggregate(_read_all(het[:3], fl)),
                     "hetero seeds")
        _expect_fail(lambda: aggregate(_read_all(het, fl[:2])),
                     "flat seeds")
        recs = _read_all(het, fl)
        _expect_fail(lambda: aggregate(recs + [dict(recs[0])]),
                     "duplicate seeds")
        print("  F missing / short / duplicate panels -> REFUSED "
              "(denominator pinned at 8) PASS")

        # ---- G) collapse + validity cells ----
        het, fl = _panel(root, "G", 0.45, 0.55, rew=0.0)
        agg = aggregate(_read_all(het, fl))
        assert agg["outcome"] == "GLOBAL-COLLAPSE", agg["outcome"]
        assert "realized_dose" in agg           # descriptives on every branch
        het, fl = _panel(root, "G2", 0.45, 0.55, n_steps=40000)
        agg = aggregate(_read_all(het, fl))
        assert agg["outcome"] == "NOT-ADJUDICABLE", agg["outcome"]
        het, fl = _panel(root, "G3", 0.45, 0.55)
        for d in het + fl:
            os.remove(os.path.join(d, "metrics.jsonl"))
        agg = aggregate(_read_all(het, fl))
        assert agg["outcome"] == "AVOIDANCE-GENERALIZES"
        assert not agg["collapse_check"]["available"]
        assert "COLLAPSE-CHECK-UNAVAILABLE" in agg["primary"]["statement"]
        print("  G dead objective -> GLOBAL-COLLAPSE; short replay -> "
              "NOT-ADJUDICABLE; absent telemetry -> annotated, not fatal "
              "PASS")

        # ---- H) end-to-end + the ONE-execution guard ----
        outd = os.path.join(root, "out")
        out = run(parse_args([
            "--hetero", os.path.join(root, "A", "se_a3gen_het_*"),
            "--flat", os.path.join(root, "A", "se_a3gen_flat_*"),
            "--output", outd]))
        assert out["outcome"] == "AVOIDANCE-GENERALIZES"
        assert os.path.exists(os.path.join(outd, "se_a3gen_read.json"))
        txt = open(os.path.join(outd, "RESULTS.md")).read()
        assert "A3 GENERALITY LEG" in txt and "Realized dose" in txt
        assert str(GATE_THRESHOLD) in txt
        _expect_fail(lambda: run(parse_args([
            "--hetero", os.path.join(root, "A", "se_a3gen_het_*"),
            "--flat", os.path.join(root, "A", "se_a3gen_flat_*"),
            "--output", outd])), "ONE execution")
        print("  H end-to-end read + RESULTS.md + ONE-execution guard PASS")

        # ---- I) PROVENANCE: the cheetah smoke control (amendment §7) ----
        if os.path.isdir(os.path.join(CHEETAH_SMOKE, "replay")):
            from uncfield.se_a3cal_read import calibrate
            x = scan_replay(CHEETAH_SMOKE)
            c = calibrate(x)
            for k, v in REGISTERED_CHEETAH.items():
                assert abs(c[k] - v) < 1e-6, (k, c[k], v)
            mb, ra = ramp_stats(x, REGISTERED_CHEETAH["mod_lo"],
                                REGISTERED_CHEETAH["mod_hi"],
                                REGISTERED_CHEETAH["gate_threshold"])
            assert abs(mb - TGT_MBAR) < 1e-6, mb
            assert abs(ra - TGT_RATIO) < 5e-3, ra
            print("  I cheetah smoke control: all 4 registered constants "
                  "reproduced <1e-6; ramp_stats (the function this "
                  "reader's dose block calls) returns m-bar %.8f, ratio "
                  "%.4fx PASS" % (mb, ra))
        else:
            print("  I cheetah smoke control SKIPPED (replay not present)")

        print("\nse_a3gen_read selfcheck PASS")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    a = parse_args()
    return selfcheck() if a.selfcheck else (run(a) and 0)


if __name__ == "__main__":
    sys.exit(main() or 0)
