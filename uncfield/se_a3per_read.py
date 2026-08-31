"""A3 PERSISTENCE-LEG FROZEN reader
(PREREG_a3_persistence_20260831.md §§3-7; built + selfchecked BEFORE
the wave's compute; ONE execution, explicit --output).

WHAT THIS ADJUDICATES
---------------------
Whether the SCARECROW FENCE that A2 measured at 5e5 steps
(artifacts/a2_scarecrow_read_20260824/: occupancy scare 0.4487-0.4805
vs ctrl 0.4996-0.5813, delta -0.0632, exact C(8,4) p = .0143) still
displaces visitation once the SAME EIGHT RUNS are trained to 1e6.

The eight A2 runs (sc_scare seeds 80-83, sc_ctrl seeds 76-79) are
resumed IN PLACE from staged copies of their run dirs. A2's scale-2
and explicit-penalty arms are NOT extended: the registered A2 PRIMARY
is the scare-vs-ctrl pair, and this leg extends exactly that pair.

    PRIMARY: occupancy(sc_scare) - occupancy(sc_ctrl) over the
    EXTENSION WINDOW ONLY, one-sided NEGATIVE exact label permutation
    over C(8,4) = 70; FIRES iff p <= .05 AND delta < 0.

THE WINDOWED ESTIMAND (prereg §3.2) -- READ THIS BEFORE COMPARING
------------------------------------------------------------------
The primary is computed on the replay chunks produced AFTER the
resume, never on the whole 1e6-step buffer. A whole-buffer occupancy
at 1e6 would be half A2's own already-read behaviour and would dilute
whatever the extension did by exactly the amount that makes a
persisting fence and a washed-out fence look alike.

    EXTENSION-WINDOW occupancy at 1e6 and A2's WHOLE-REPLAY occupancy
    at 5e5 are DIFFERENT ESTIMANDS. Levels are not comparable across
    them. HOLD / WASH-OUT is defined against the CONTRAST, never
    against raw levels.

WINDOW IDENTIFICATION IS DATA-INDEPENDENT
-----------------------------------------
Each run dir carries `STAGED_MANIFEST.txt`, the exact list of
`replay/*.npz` present at staging time, written by the wave's own cmd
BEFORE the producer is launched. The extension window is
`replay/*.npz` MINUS that list. The manifest is FATAL if missing,
empty, duplicated, or not a subset of what is on disk (a listed chunk
that is absent means the staged tree was mutated -- subset is fatal,
superset is the normal case).

The manifest window is then a SECOND, independent handle on staging
integrity: scanned with the same frozen estimand it must reproduce
A2's recorded per-seed occupancy and step count EXACTLY (the
A2-PROVENANCE gate, §5.1.l). That single check proves at once that
(a) the staging was complete -- the replay capacity is 5,000,000
items against ~5e5 produced, so `Replay.load`'s newest-first cutoff
NEVER truncates and a partial stage silently gives a smaller resumed
buffer; (b) the run dir staged under a given seed really is that A2
run; (c) no staged chunk was corrupted or rewritten.

RESUME-INTEGRITY GATES (the registered canary, §5.1.k)
------------------------------------------------------
`embodied/core/replay.py:Replay.load` returns SILENTLY when the
replay dir is missing or empty, and training then refills from fresh
interaction -- a from-scratch run wearing an extension's name. Two
FATAL witnesses out of the run's own `metrics.jsonl`, which is
created fresh at resume because `metrics.jsonl` is NOT staged:

  * BUFFER REPOPULATED: the first row carrying `replay/items` must
    report >= 400000 items. A loaded buffer reports ~4.95e5 (A2 ended
    at 494624-499648 replay steps and nothing evicts below capacity);
    a from-scratch buffer reports ~3.1e3 at its first log (measured
    on a real SE run: 3136 items at step 4160).
  * RESUMED FROM 5e5: the first row carrying `step` must be
    >= 490000. A from-scratch run's first log lands near 4e3.

Both are ~100x separated from their failure values, so neither is a
tuned threshold.

WHAT IS NOT HERE
----------------
No dose gate and no drift gate: same environment, same manipulation,
no transplant -- there is nothing to drift. No probe leg (the A2
probe panel is executed evidence and is not re-run). The degeneracy
check is the A3-generality form: training telemetry only.

Run:  python -m uncfield.se_a3per_read --scare "<glob>" \
          --ctrl "<glob>" --output artifacts/a3_persistence_read_<date>
      python -m uncfield.se_a3per_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

# --- the registered primary machinery, imported from executed readers ---
# se_m3_read is the frozen estimand + permutation authority that A1, A2
# and the A3 generality leg all consume; importing it (never copying it)
# is what makes this statistic bit-identical to A2's.
from uncfield.se_m3_read import (GATE_INDEX, GATE_KEY, GATE_THRESHOLD,
                                 MIN_STEPS, SCORE_TAIL, exact_perm_p,
                                 occupancy)
# --- A2's own arm constants, imported from the executed A2 reader chain ---
from uncfield.se_grad_read import (BASESD_N, DIM, FLAT_SCALE, MOD_HI,
                                   MOD_INDEX, MOD_KEY, MOD_LO)
from uncfield.se_mask import bca_interval
from uncfield.se_read import fit_counters

# ==================================================================== pins
# Every constant below is frozen at registration. Nothing here is
# estimated from this wave.

TASK = "dmc_cheetah_run"
REGION_KEY = GATE_KEY                      # 'position'
REGION_INDEX = GATE_INDEX                  # 0
REGION_THRESHOLD = GATE_THRESHOLD          # -0.13009691 (CHEETAH, A2's own)
BASESD_PLANTED = 0.0976                    # A2 / Stage-1 cheetah pin
THETA_OU = 0.1                             # config default = phi 0.9 = A2
EXPECT_STEPS = 1e6                         # the EXTENSION target
ALPHA = 0.05
N_RUNS = 8

SCARE_SEEDS = frozenset(range(80, 84))     # A2 sc_scare
CTRL_SEEDS = frozenset(range(76, 80))      # A2 sc_ctrl
ALL_SEEDS = SCARE_SEEDS | CTRL_SEEDS
SEED_ARM = dict([(s, "scare") for s in SCARE_SEEDS]
                + [(s, "ctrl") for s in CTRL_SEEDS])

# --------------------------------------------------------- A2's record
# artifacts/a2_scarecrow_read_20260824/se_scarecrow_read.json, per_run,
# the CONSUMED execution of the frozen A2 reader. Occupancy is the
# whole-replay estimand at 5e5; occ_steps is that scan's step count.
# These are (i) the wash-out comparison's reference values and (ii) the
# A2-PROVENANCE gate's targets on the manifest window.
A2_ARTIFACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "a2_scarecrow_read_20260824", "se_scarecrow_read.json")
A2_OCC = {
    76: 0.4995996959306463,  77: 0.5253805365444215,
    78: 0.5038046784936595,  79: 0.5813107776306012,
    80: 0.48048719358385616, 81: 0.45456496966584276,
    82: 0.44867966555831745, 83: 0.47361929779600204,
}
A2_STEPS = {
    76: 494624, 77: 495616, 78: 499648, 79: 495248,
    80: 494752, 81: 498448, 82: 495632, 83: 499456,
}
A2_DELTA = (float(np.mean([A2_OCC[s] for s in sorted(SCARE_SEEDS)]))
            - float(np.mean([A2_OCC[s] for s in sorted(CTRL_SEEDS)])))
A2_P = 1.0 / 70.0

# A2's within-arm dispersion, from the same eight numbers. It is the
# yardstick the registered SECONDARY's bound is justified against.
_S = np.asarray([A2_OCC[s] for s in sorted(SCARE_SEEDS)], np.float64)
_C = np.asarray([A2_OCC[s] for s in sorted(CTRL_SEEDS)], np.float64)
A2_POOLED_SD = float(np.sqrt((3 * _S.var(ddof=1) + 3 * _C.var(ddof=1)) / 6))

# SECONDARY-W bound: HALF of A2's measured displacement. Justified from
# A2's own dispersion -- half of A2's delta is 1.10 x A2's pooled
# within-arm sd, i.e. the smallest "the fence has more than halved"
# target that is still larger than the seed-to-seed noise A2 itself
# measured. Registered before the data exist; see prereg §4.3.
ATTEN_BOUND = A2_DELTA / 2.0

# ---------------------------------------------------- resume + window pins
MANIFEST_NAME = "STAGED_MANIFEST.txt"
REPLAY_CAPACITY = 5_000_000            # int(config.replay.size 5e6), main.py:186
RESUME_ITEMS_KEY = "replay/items"      # train.py:109 logger.add(..., prefix='replay')
RESUME_ITEMS_MIN = 4.0e5               # loaded ~4.95e5 vs from-scratch ~3.1e3
RESUME_STEP_MIN = 4.9e5                # A2's realized finals were 497584-499088
EXT_MIN_STEPS = int(4e5)               # extension validity floor (nominal 5e5)
COLLAPSE_KEY = "train/expl/disag_replay_rew"

# ------------------------------------------- registration self-consistency
# Cheap asserts that fire at import, so a silently edited pin cannot be
# read. They tie every constant to an executed instrument or to A2's own
# recorded arithmetic.
assert REGION_THRESHOLD == GATE_THRESHOLD == -0.13009691, REGION_THRESHOLD
assert REGION_KEY == MOD_KEY == "position" and REGION_INDEX == MOD_INDEX == 0
assert (MOD_LO, MOD_HI) == (-0.32203162, 0.19711795), (MOD_LO, MOD_HI)
assert FLAT_SCALE == 0.35891393 and DIM == 8 and BASESD_N == 1.215
assert MOD_LO < REGION_THRESHOLD < MOD_HI
assert EXT_MIN_STEPS >= MIN_STEPS           # never weaker than the house floor
assert abs(A2_DELTA - (-0.06318614049882765)) < 1e-15, A2_DELTA
assert abs(A2_POOLED_SD - 0.028657806834322192) < 1e-15, A2_POOLED_SD
assert abs(abs(ATTEN_BOUND) / A2_POOLED_SD - 1.1023) < 1e-3
assert set(A2_OCC) == set(A2_STEPS) == ALL_SEEDS
assert all(round(A2_OCC[s] * A2_STEPS[s]) / A2_STEPS[s] == A2_OCC[s]
           for s in ALL_SEEDS), "A2 occupancy is hits/steps; it must invert"

if os.path.exists(A2_ARTIFACT):              # the recorded execution itself
    with open(A2_ARTIFACT) as _f:
        _A2 = json.load(_f)
    _seen = {}
    for _r in _A2["per_run"]:
        if _r["arm"] in ("sc_ctrl", "sc_scare"):
            _seen[int(_r["train_seed"])] = (_r["occupancy"], _r["occ_steps"])
    assert set(_seen) == ALL_SEEDS, sorted(_seen)
    for _s in sorted(ALL_SEEDS):
        assert _seen[_s][0] == A2_OCC[_s], (_s, _seen[_s][0], A2_OCC[_s])
        assert _seen[_s][1] == A2_STEPS[_s], (_s, _seen[_s][1], A2_STEPS[_s])
    assert _A2["primary"]["delta"] == A2_DELTA, _A2["primary"]["delta"]
    assert abs(_A2["primary"]["p"] - A2_P) < 1e-15
    assert _A2["primary"]["fires"] is True
    assert _A2["outcome_cell"] == "SCARECROW-FENCES"
    A2_ARTIFACT_VERIFIED = True
    del _A2, _seen, _r, _s, _f
else:                                        # a checkout without artifacts/
    A2_ARTIFACT_VERIFIED = False
del _S, _C


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scare", default="", help="4 extended sc_scare dirs")
    p.add_argument("--ctrl", default="", help="4 extended sc_ctrl dirs")
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


# ------------------------------------------------------------- the window

def replay_files(run_dir):
    """Every replay chunk currently on disk, sorted (the same listing
    se_m3_read.occupancy walks)."""
    rd = os.path.join(run_dir, "replay")
    assert os.path.isdir(rd), f"{run_dir}: no replay/ directory"
    return sorted(f for f in os.listdir(rd) if f.endswith(".npz"))


def staged_manifest(run_dir):
    """The staging-time chunk list. FATAL on missing / empty / malformed
    / duplicated entries -- the window definition may never be inferred
    from the data it is supposed to partition."""
    path = os.path.join(run_dir, MANIFEST_NAME)
    assert os.path.exists(path), (
        f"{run_dir}: {MANIFEST_NAME} MISSING -- the extension window is "
        f"undefined. The wave's cmd writes it before the producer starts; "
        f"its absence means this dir was not produced by the registered "
        f"a3_persistence wave.")
    with open(path) as f:
        raw = [ln.strip() for ln in f.read().splitlines()]
    names = [x for x in raw if x]
    assert names, f"{run_dir}: {MANIFEST_NAME} is EMPTY"
    bad = [x for x in names if not x.endswith(".npz") or "/" in x]
    assert not bad, (
        f"{run_dir}: {MANIFEST_NAME} holds non-chunk entries {bad[:3]} "
        f"-- it must be bare replay chunk basenames")
    dupes = sorted({x for x in names if names.count(x) > 1})
    assert not dupes, (
        f"{run_dir}: {MANIFEST_NAME} has duplicate entries {dupes[:3]}")
    digest = hashlib.sha256(
        ("\n".join(sorted(names)) + "\n").encode()).hexdigest()
    return sorted(names), digest


def windows(run_dir):
    """(manifest chunks, extension chunks) with the SUBSET gate.

    Superset is the normal case: the resume appends new chunks and never
    rewrites or deletes a staged one. SUBSET IS FATAL -- a manifest-listed
    chunk that is gone means the staged tree was mutated after the
    manifest was taken, and every window below it is then a guess.
    """
    manifest, digest = staged_manifest(run_dir)
    present = replay_files(run_dir)
    missing = sorted(set(manifest) - set(present))
    assert not missing, (
        f"{run_dir}: {len(missing)} manifest-listed chunk(s) are ABSENT "
        f"from replay/ (e.g. {missing[:3]}) -- the staged tree was "
        f"mutated; subset is fatal")
    ext = sorted(set(present) - set(manifest))
    assert ext, (
        f"{run_dir}: the extension window is EMPTY ({len(present)} chunks "
        f"on disk, all of them staged) -- no post-resume data exists")
    return manifest, ext, digest


def occupancy_over(run_dir, files, gate_index=REGION_INDEX,
                   gate_threshold=REGION_THRESHOLD):
    """se_m3_read.occupancy, restricted to a chunk subset.

    The body is the frozen estimand's, verbatim, with the file list
    supplied instead of globbed; the selfcheck asserts that on the FULL
    listing it returns bit-identical (frac, n) to se_m3_read.occupancy.
    """
    rd = os.path.join(run_dir, "replay")
    files = sorted(files)
    assert files, f"{run_dir}: empty chunk window"
    hits = total = 0
    for f in files:
        with np.load(os.path.join(rd, f)) as z:
            assert GATE_KEY in z, (run_dir, f, list(z.keys()))
            v = np.asarray(z[GATE_KEY], np.float64)
            v = v.reshape(-1, v.shape[-1])[:, gate_index]
            hits += int((v > gate_threshold).sum())
            total += v.size
    return hits / total, total


# --------------------------------------------------------------- telemetry

def _metrics_rows(run_dir):
    rows = []
    try:
        with open(os.path.join(run_dir, "metrics.jsonl"), "rb") as f:
            for line in f.read().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def _first_with(rows, key):
    for d in rows:
        if key in d:
            return d
    return None


def _last_with(rows, key):
    for d in reversed(rows):
        if key in d:
            return d
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
    """FATAL identity / continuity / resume-integrity gates (prereg §5.1)
    plus the per-run record.

    A defective run refuses the WHOLE read rather than being silently
    dropped, and the refusal happens before any output is written. Gate
    ORDER is registered: config continuity first (cheap and decisive),
    then the window, then A2-provenance, then the resume witnesses, then
    the estimand.
    """
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = cfg["distractor"]
    pl_ = cfg["planted"]
    ex = cfg["agent"]["expl"]

    # (a) the seed<->arm map is the ARM AUTHORITY. The arm is decided by
    # A2's registered seed block; the config must AGREE with it, never
    # relabel it. run_id strings are NOT identity (28-Aug standing rule).
    seed = int(cfg["seed"])
    assert seed in SEED_ARM, (
        f"{run_dir}: seed {seed} is not one of A2's extended seeds "
        f"{sorted(ALL_SEEDS)} -- A2's scale-2 (84-87) and penalty "
        f"(88-95) arms are NOT part of this leg")
    assert SEED_ARM[seed] == arm, (
        f"{run_dir}: seed {seed} is registered for the {SEED_ARM[seed]} "
        f"arm but was passed as {arm} -- WRONG ARM MAP")

    # (b) environment + Stage-1 objective pin block, A2 verbatim
    assert str(cfg.get("task", "")) == TASK, (
        f"{run_dir}: task {cfg.get('task')!r} != {TASK}")
    assert str(ex["mode"]) == "p2e" and \
        ex["disag_bootstrap"] is True and \
        int(ex["disag_ens"]) == 8 and \
        abs(float(ex["disag_scale"]) - 1000.0) < 1e-9 and \
        abs(float(ex.get("disag_bootstrap_prob", 0.8)) - 0.8) < 1e-9, (
        f"{run_dir}: expl pins differ from A2 (EXPL_CONFIG or "
        f"DISAG_BOOTSTRAP override leak?)")
    # disag_head postdates A2's pin block and the b4_alea wave ran a
    # 'gauss' arm on this producer -- an unpinned head is a live leak path
    assert str(ex.get("disag_head", "det")) == "det", (
        f"{run_dir}: agent.expl.disag_head {ex.get('disag_head')!r} != det "
        f"(DISAG_HEAD leak from the alea wave?)")
    # A2's own penalty arms exist on this producer; both extended arms
    # must carry the machinery INERT, exactly as A2 registered them
    assert ex.get("penalty_mix", False) is False and \
        abs(float(cfg.get("penalty", {}).get("scale", 0.0))) < 1e-12 and \
        str(cfg.get("penalty", {}).get("gate_key", "")) == "", (
        f"{run_dir}: penalty machinery must be inert (an sc_pen leak "
        f"would replace the task reward)")

    # (c) the cheetah basesd pair -- a finger leak (A3 generality ran
    # 0.54193702 / 1.35803102 on this same producer) is FATAL
    assert str(pl_["source_key"]) == REGION_KEY and \
        abs(float(pl_["basesd"]) - BASESD_PLANTED) < 1e-9, (
        f"{run_dir}: planted pins {pl_['source_key']!r}/{pl_['basesd']!r} "
        f"differ from A2's cheetah inventory "
        f"({REGION_KEY}/{BASESD_PLANTED}) -- finger leak?")
    assert int(d["dim"]) == DIM and \
        abs(float(d["basesd"]) - BASESD_N) < 1e-9, (
        f"{run_dir}: distractor dim/basesd {d['dim']}/{d['basesd']!r} "
        f"differ from the registered {DIM}/{BASESD_N}")
    # the OU persistence knob: the phi-sweep wave moves exactly this
    # value on this producer
    assert abs(float(d.get("theta", THETA_OU)) - THETA_OU) < 1e-12, (
        f"{run_dir}: distractor.theta {d.get('theta')!r} != {THETA_OU} "
        f"(DISTRACTOR_THETA leak from the phi-sweep?)")

    # (d) all training-time gates OFF (the region split is a READER
    # constant in A2's design, not a producer knob)
    assert str(pl_.get("gate_key", "")) == "" and \
        str(d.get("gate_key", "")) == "", (
        f"{run_dir}: training-time gates must be off on both wrappers")

    # (e) A2's arm construction, re-passed verbatim by the extension cmd.
    # If any of this drifted, the extension trained a DIFFERENT
    # manipulation than the one it claims to be continuing.
    mk = str(d.get("mod_key", ""))
    scale = float(d["scale"])
    if arm == "scare":
        assert mk == MOD_KEY and int(d["mod_index"]) == MOD_INDEX and \
            abs(float(d["mod_lo"]) - MOD_LO) < 1e-9 and \
            abs(float(d["mod_hi"]) - MOD_HI) < 1e-9 and \
            abs(scale - 1.0) < 1e-12, (
            f"{run_dir}: scare pins -- the A1-hetero mod quadruple must be "
            f"({MOD_KEY}, {MOD_INDEX}, {MOD_LO}, {MOD_HI}) at scale 1.0, "
            f"got ({mk!r}, {d.get('mod_index')!r}, {d.get('mod_lo')!r}, "
            f"{d.get('mod_hi')!r}) at {scale!r}")
    else:
        assert mk == "" and abs(scale - FLAT_SCALE) < 1e-12 and \
            abs(float(d["mod_lo"]) - 0.0) < 1e-12 and \
            abs(float(d["mod_hi"]) - 1.0) < 1e-12, (
            f"{run_dir}: ctrl pins -- mod_key must be empty at the inert "
            f"ramp (0.0, 1.0) and the A1-flat scale {FLAT_SCALE}, got "
            f"{mk!r} / ({d.get('mod_lo')!r}, {d.get('mod_hi')!r}) at "
            f"{scale!r}")

    # (f) the EXTENSION target
    assert abs(float(cfg["run"]["steps"]) - EXPECT_STEPS) < 1e-6, (
        f"{run_dir}: run.steps {cfg['run']['steps']!r} != {EXPECT_STEPS} "
        f"-- this dir is not an extended run")

    # (g) the window, from the staged manifest (never from the data)
    man_files, ext_files, man_digest = windows(run_dir)

    # (h) RESUME INTEGRITY -- the registered canary, checked BEFORE the
    # expensive scans because it is the failure the whole design exists
    # to catch. metrics.jsonl is NOT staged, so its first rows are the
    # first post-resume report by construction.
    rows = _metrics_rows(run_dir)
    r_step = _first_with(rows, "step")
    assert r_step is not None, (
        f"{run_dir}: metrics.jsonl carries no 'step' row -- no resume "
        f"witness exists")
    first_step = float(r_step["step"])
    assert first_step >= RESUME_STEP_MIN, (
        f"{run_dir}: first post-resume logged step {first_step:.0f} < "
        f"{RESUME_STEP_MIN:.0f} -- training RESTARTED FROM SCRATCH "
        f"instead of continuing A2's checkpoint")
    r_items = _first_with(rows, RESUME_ITEMS_KEY)
    assert r_items is not None, (
        f"{run_dir}: metrics.jsonl carries no {RESUME_ITEMS_KEY!r} row -- "
        f"the buffer-repopulation witness does not exist")
    first_items = float(r_items[RESUME_ITEMS_KEY])
    assert first_items >= RESUME_ITEMS_MIN, (
        f"{run_dir}: first post-resume {RESUME_ITEMS_KEY} "
        f"{first_items:.0f} < {RESUME_ITEMS_MIN:.0f} -- the replay buffer "
        f"was NOT repopulated (Replay.load returns silently on an "
        f"empty/missing replay dir and training refills from fresh "
        f"interaction)")

    # (i) A2-PROVENANCE: the manifest window, scanned with the frozen
    # estimand, must reproduce A2's recorded numbers for THIS seed
    # exactly. Capacity is 5,000,000 items against ~5e5 produced, so
    # Replay.load's newest-first cutoff never truncates: a partial stage
    # is a smaller resumed buffer AND a failed check here.
    occ_man, n_man = occupancy_over(run_dir, man_files)
    assert n_man == A2_STEPS[seed], (
        f"{run_dir}: manifest window holds {n_man} replay steps but A2 "
        f"recorded {A2_STEPS[seed]} for seed {seed} -- the staged copy is "
        f"INCOMPLETE (capacity {REPLAY_CAPACITY} never evicts, so the "
        f"resumed buffer is short by the same amount) or is the wrong run")
    assert abs(occ_man - A2_OCC[seed]) < 1e-9, (
        f"{run_dir}: manifest-window occupancy {occ_man!r} != A2's "
        f"recorded {A2_OCC[seed]!r} for seed {seed} -- the staged copy is "
        f"not the A2 run this seed names")

    # (j) the registered PRIMARY estimand: the EXTENSION WINDOW ONLY
    occ_ext, n_ext = occupancy_over(run_dir, ext_files)

    r_tele = _last_with(rows, COLLAPSE_KEY)
    score, n_scores = _score_tail(run_dir)
    return dict(
        run_dir=os.path.abspath(run_dir), arm=arm, train_seed=seed,
        occupancy=float(occ_ext), n_steps=int(n_ext),
        valid=bool(n_ext >= EXT_MIN_STEPS),
        occ_manifest=float(occ_man), n_steps_manifest=int(n_man),
        a2_occupancy=A2_OCC[seed], a2_steps=A2_STEPS[seed],
        n_chunks_manifest=len(man_files), n_chunks_extension=len(ext_files),
        manifest_sha256=man_digest,
        resume=dict(first_step=first_step,
                    first_items=first_items,
                    first_items_step=float(r_items.get("step", -1))),
        disag_replay_rew=(None if r_tele is None
                          else float(r_tele[COLLAPSE_KEY])),
        score=score, n_score_entries=n_scores)


# ---------------------------------------------------------------- aggregate

def aggregate(recs):
    sc = sorted((r for r in recs if r["arm"] == "scare"),
                key=lambda r: r["train_seed"])
    ct = sorted((r for r in recs if r["arm"] == "ctrl"),
                key=lambda r: r["train_seed"])
    ss = [r["train_seed"] for r in sc]
    cs = [r["train_seed"] for r in ct]
    # DENOMINATOR PINNED AT 8 (A2 review #34 B3, inherited): the primary
    # is an exact permutation over C(8,4); there is no registered
    # fallback statistic for a partial panel, and dropping a run can
    # MANUFACTURE a fire.
    assert len(set(ss)) == len(ss) and len(set(cs)) == len(cs), (
        f"duplicate seeds: scare {ss}, ctrl {cs}")
    assert set(ss) == set(SCARE_SEEDS), (
        f"scare seeds {ss} != {sorted(SCARE_SEEDS)} "
        f"(missing {sorted(SCARE_SEEDS - set(ss))}, "
        f"extra {sorted(set(ss) - SCARE_SEEDS)})")
    assert set(cs) == set(CTRL_SEEDS), (
        f"ctrl seeds {cs} != {sorted(CTRL_SEEDS)} "
        f"(missing {sorted(CTRL_SEEDS - set(cs))}, "
        f"extra {sorted(set(cs) - CTRL_SEEDS)})")
    assert len(recs) == N_RUNS, f"{len(recs)} runs loaded, need {N_RUNS}"

    invalid = [r["run_dir"] for r in recs if not r["valid"]]
    out = dict(
        stamp=_stamp(), n_scare=len(sc), n_ctrl=len(ct),
        invalid_runs=invalid,
        instrument_flag=("CLEAN" if not invalid else "VALIDITY-VIOLATIONS"),
        a2_artifact_verified=A2_ARTIFACT_VERIFIED,
        registered=dict(
            task=TASK, region_key=REGION_KEY, region_index=REGION_INDEX,
            region_threshold=REGION_THRESHOLD,
            mod_key=MOD_KEY, mod_lo=MOD_LO, mod_hi=MOD_HI,
            flat_scale=FLAT_SCALE, basesd_planted=BASESD_PLANTED,
            basesd_n=BASESD_N, distractor_dim=DIM, theta=THETA_OU,
            steps=EXPECT_STEPS, replay_capacity=REPLAY_CAPACITY,
            scare_seeds=sorted(SCARE_SEEDS), ctrl_seeds=sorted(CTRL_SEEDS),
            ext_min_steps=EXT_MIN_STEPS,
            resume_step_min=RESUME_STEP_MIN,
            resume_items_min=RESUME_ITEMS_MIN,
            atten_bound=ATTEN_BOUND, a2_delta=A2_DELTA,
            a2_pooled_sd=A2_POOLED_SD, alpha=ALPHA))

    ordered = sc + ct
    labels = [True] * len(sc) + [False] * len(ct)
    o_s = [r["occupancy"] for r in sc]
    o_c = [r["occupancy"] for r in ct]

    # ---- unconditional descriptives (reported on EVERY branch) ----
    for name, rs in (("scare", sc), ("ctrl", ct)):
        if len(rs) >= 3:
            m, lo, hi_ = bca_interval(
                np.asarray([r["occupancy"] for r in rs]),
                np.random.default_rng(41 if name == "scare" else 42), 2000)
            out[f"occupancy_ext_{name}"] = dict(mean=m, bca=[lo, hi_])
    if all(r["score"] is not None for r in ordered):
        d_sc, p_sc, _ = exact_perm_p([r["score"] for r in ordered],
                                     labels, one_sided=False)
        out["score_two_sided"] = dict(delta=d_sc, p=p_sc,
                                      note="reporting only, two-sided")

    # ---- the DESCRIPTIVE wash-out comparison (registered, NEVER a test)
    d_ext = float(np.mean(o_s) - np.mean(o_c)) if o_s and o_c else None
    out["washout_comparison"] = dict(
        extension_window_1e6=dict(
            scare=o_s, ctrl=o_c,
            mean_scare=(float(np.mean(o_s)) if o_s else None),
            mean_ctrl=(float(np.mean(o_c)) if o_c else None),
            delta=d_ext),
        a2_whole_replay_5e5=dict(
            scare=[A2_OCC[s] for s in sorted(SCARE_SEEDS)],
            ctrl=[A2_OCC[s] for s in sorted(CTRL_SEEDS)],
            mean_scare=float(np.mean([A2_OCC[s]
                                      for s in sorted(SCARE_SEEDS)])),
            mean_ctrl=float(np.mean([A2_OCC[s]
                                     for s in sorted(CTRL_SEEDS)])),
            delta=A2_DELTA, p=A2_P,
            source="artifacts/a2_scarecrow_read_20260824/"
                   "se_scarecrow_read.json"),
        manifest_window_replay=dict(
            scare=[r["occ_manifest"] for r in sc],
            ctrl=[r["occ_manifest"] for r in ct],
            note="the staged A2 replay, re-scanned here; equals the A2 "
                 "column by the A2-PROVENANCE gate and exists only as "
                 "that gate's witness"),
        delta_ratio=(None if d_ext is None else d_ext / A2_DELTA),
        note=("CROSS-ESTIMAND AND NON-INFERENTIAL. The 1e6 column is the "
              "EXTENSION WINDOW (post-resume chunks only); the 5e5 column "
              "is A2's WHOLE-REPLAY estimand. Levels are not comparable "
              "across the two and no test is computed between them. The "
              "contrast is the only quantity this leg adjudicates."))

    # ---- (1) collapse check (A3-generality form; no probe leg here) ----
    tele = {r["run_dir"]: r["disag_replay_rew"] for r in recs}
    missing_tele = [k for k, v in tele.items() if v is None]
    dead = [k for k, v in tele.items()
            if v is not None and not (np.isfinite(v) and v > 0.0)]
    out["collapse_check"] = dict(
        key=COLLAPSE_KEY, per_run=tele, dead_runs=dead,
        available=not missing_tele,
        note=("Training-time intrinsic-reward telemetry must be present, "
              "finite and strictly positive. No absolute threshold is "
              "invented; A2's probe-referenced degeneracy gate is not "
              "re-run because this leg produces no probe."))
    out["resume_rows"] = {r["run_dir"]: r["resume"] for r in recs}
    if dead:
        out["primary"] = ("GLOBAL-DISAGREEMENT-COLLAPSE: the intrinsic "
                          "objective is dead in at least one run -- "
                          "nothing downstream adjudicated")
        out["outcome"] = "GLOBAL-COLLAPSE"
        out["attenuation"] = "NOT-EVALUATED (collapse)"
        return out

    # ---- (2) extension-window validity ----
    if invalid or len(sc) != 4 or len(ct) != 4:
        out["primary"] = (f"NOT-ADJUDICABLE (need 4+4 runs with >= "
                          f"{EXT_MIN_STEPS} extension-window steps)")
        out["outcome"] = "NOT-ADJUDICABLE"
        out["attenuation"] = "NOT-EVALUATED (validity)"
        return out

    # ---- (3) PRIMARY: A2's statistic on the extension window ----
    d_neg, p_occ, total = exact_perm_p([-x for x in (o_s + o_c)], labels,
                                       one_sided=True)
    d_occ = -d_neg
    fires = bool(p_occ <= ALPHA and d_occ < 0)
    statement = (
        "REGISTERED persistence test of A2: extension-window "
        "occupancy(sc_scare) < occupancy(sc_ctrl) in the high-amplitude "
        f"region of {TASK} ({REGION_KEY}[{REGION_INDEX}] > "
        f"{REGION_THRESHOLD}), post-resume replay chunks ONLY, one-sided "
        "negative exact permutation over C(8,4)=70. Power note: firing "
        "requires the observed split among the 3 most extreme of 70 -- "
        "near-complete separation; this wave has essentially no power "
        "against moderate effects, so a non-fire does NOT license "
        "'washed out' as a positive claim and does not distinguish 'no "
        "fence' from 'a moderate fence'")
    if not out["collapse_check"]["available"]:
        statement += ("; COLLAPSE-CHECK-UNAVAILABLE on "
                      f"{len(missing_tele)} run(s)")
    out["primary"] = dict(
        statement=statement, delta=d_occ, p=p_occ, n_assignments=total,
        fires=fires, min_attainable_p=1.0 / total,
        occ_scare=o_s, occ_ctrl=o_c)
    out["outcome"] = "FENCE-HOLDS" if fires else "NOT-DETECTED-AT-1e6"

    # ---- (4) SECONDARY-W: the registered attenuation bound ----
    # Shift-model exact permutation of the null "the extension-window
    # contrast equals HALF A2's displacement". Under H0 the arms are
    # exchangeable after shifting scare up by |ATTEN_BOUND|, so the same
    # C(8,4) machinery applies unchanged.
    adj = [x - ATTEN_BOUND for x in o_s] + list(o_c)
    d_adj, p_adj, total_w = exact_perm_p(adj, labels, one_sided=True)
    w_fires = bool(p_adj <= ALPHA)
    out["secondary_w"] = dict(
        statement=(
            "SECONDARY (registered, one-sided): the extension-window "
            f"contrast is LESS NEGATIVE than {ATTEN_BOUND:.6f} = half of "
            f"A2's measured displacement ({A2_DELTA:.6f}), tested as a "
            "shift-model exact permutation over the same C(8,4)=70. The "
            f"bound is 1.10 x A2's pooled within-arm sd "
            f"({A2_POOLED_SD:.6f}), i.e. the smallest 'more than halved' "
            "target larger than the seed-to-seed noise A2 itself "
            "measured. FIRES iff p <= .05. A fire is the ONLY registered "
            "route to a positive attenuation claim; a non-fire bounds "
            "nothing (min attainable p = 1/70)."),
        bound=ATTEN_BOUND, delta_adjusted=d_adj, p=p_adj,
        n_assignments=total_w, fires=w_fires,
        min_attainable_p=1.0 / total_w)
    out["attenuation"] = ("ATTENUATED-BELOW-HALF" if w_fires
                          else "NOT-BOUNDED")

    out["not_claimed"] = [
        "a non-fire of the PRIMARY is NOT 'the fence washed out'. At "
        "n=4v4 this design cannot separate 'no fence' from 'a moderate "
        "fence'; only SECONDARY-W can positively bound the effect, and "
        "only when it fires",
        "no cross-estimand inference: extension-window occupancy at 1e6 "
        "and A2's whole-replay occupancy at 5e5 are different estimands "
        "and the side-by-side table is descriptive only",
        "nothing about A2's scale-2 or explicit-penalty arms, which are "
        "not extended",
        "no generality claim: the A3 generality leg is a SEPARATE "
        "registration and neither leg's outcome gates the other",
        "no mechanism: this leg runs no probe, so no statement of the "
        "form 'avoids what it overprices' is licensed by it alone",
        "no claim about training beyond 1e6 steps",
        "whatever this leg returns, the registered A2 result at 5e5 "
        "stands exactly as read on 24 Aug",
    ]
    return out


# ---------------------------------------------------------------- rendering

def render(out):
    L = ["# A3 PERSISTENCE LEG — read\n",
         f"OUTCOME: **{out['outcome']}**"
         + (f" / attenuation: **{out['attenuation']}**"
            if "attenuation" in out else "") + "\n",
         f"Task {TASK}; sc_scare seeds {sorted(SCARE_SEEDS)} vs sc_ctrl "
         f"seeds {sorted(CTRL_SEEDS)}, the A2 runs extended 5e5 -> 1e6 in "
         f"place; region split {REGION_KEY}[{REGION_INDEX}] > "
         f"{REGION_THRESHOLD}.\n",
         "**Estimand: the EXTENSION WINDOW only** — replay chunks not in "
         "each run's `STAGED_MANIFEST.txt`. This is NOT A2's "
         "whole-replay estimand and levels do not compare across the "
         "two.\n",
         f"A2 artifact re-verified at import: "
         f"**{out.get('a2_artifact_verified')}**\n"]
    if isinstance(out.get("primary"), dict):
        p = out["primary"]
        L.append(f"PRIMARY: delta {p['delta']:+.4f}, one-sided p "
                 f"{p['p']:.4f} of {p['n_assignments']} assignments "
                 f"(min attainable {p['min_attainable_p']:.4f}) -> "
                 f"fires = **{p['fires']}**\n")
        L.append(f"> {p['statement']}\n")
    else:
        L.append(f"PRIMARY: {out.get('primary')}\n")
    if isinstance(out.get("secondary_w"), dict):
        s = out["secondary_w"]
        L.append(f"SECONDARY-W (attenuation bound {s['bound']:+.6f}): "
                 f"adjusted delta {s['delta_adjusted']:+.4f}, one-sided p "
                 f"{s['p']:.4f} -> fires = **{s['fires']}**\n")
        L.append(f"> {s['statement']}\n")
    for arm in ("scare", "ctrl"):
        k = f"occupancy_ext_{arm}"
        if k in out:
            o = out[k]
            L.append(f"- extension-window occupancy {arm}: {o['mean']:.4f} "
                     f"BCa [{o['bca'][0]:.4f}, {o['bca'][1]:.4f}]")
    if "score_two_sided" in out:
        s = out["score_two_sided"]
        L.append(f"- task return (two-sided, reporting only): delta "
                 f"{s['delta']:+.4f}, p {s['p']:.4f}")

    w = out["washout_comparison"]
    L.append("\n## Wash-out comparison (DESCRIPTIVE, CROSS-ESTIMAND)\n")
    L.append("| quantity | sc_scare | sc_ctrl | contrast |")
    L.append("|---|---|---|---|")
    e, a = w["extension_window_1e6"], w["a2_whole_replay_5e5"]
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    L.append(f"| extension window, 1e6 (THIS leg's estimand) | "
             f"{fmt(e['mean_scare'])} | {fmt(e['mean_ctrl'])} | "
             f"{'n/a' if e['delta'] is None else format(e['delta'], '+.4f')} |")
    L.append(f"| whole replay, 5e5 (A2's estimand, recorded) | "
             f"{a['mean_scare']:.4f} | {a['mean_ctrl']:.4f} | "
             f"{a['delta']:+.4f} (p {a['p']:.4f}) |")
    L.append(f"\n- contrast ratio (extension / A2): "
             f"{'n/a' if w['delta_ratio'] is None else format(w['delta_ratio'], '.3f')}")
    L.append(f"- {w['note']}")
    L.append(f"- A2 source: `{a['source']}`")

    L.append("\n## Per-run\n")
    L.append("| run | seed | arm | ext occupancy | ext steps | ext chunks | "
             "valid | manifest occ (A2 recorded) | manifest steps | "
             "first post-resume step | first replay/items |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out.get("per_run", []):
        rs = r["resume"]
        L.append(f"| {os.path.basename(r['run_dir'])} | {r['train_seed']} | "
                 f"{r['arm']} | {r['occupancy']:.4f} | {r['n_steps']} | "
                 f"{r['n_chunks_extension']} | {r['valid']} | "
                 f"{r['occ_manifest']:.6f} ({r['a2_occupancy']:.6f}) | "
                 f"{r['n_steps_manifest']} | {rs['first_step']:.0f} | "
                 f"{rs['first_items']:.0f} |")

    cc = out["collapse_check"]
    L.append(f"\n## Gates\n\n- A2-PROVENANCE: manifest window reproduces "
             f"A2's recorded occupancy and step count per seed (FATAL "
             f"otherwise) — see the per-run table")
    L.append(f"- resume integrity: first logged step >= "
             f"{RESUME_STEP_MIN:.0f}; first `{RESUME_ITEMS_KEY}` >= "
             f"{RESUME_ITEMS_MIN:.0f} (FATAL otherwise)")
    L.append(f"- collapse check available: {cc['available']}; dead runs: "
             f"{cc['dead_runs']}")
    L.append(f"- extension-window validity (>= {EXT_MIN_STEPS} steps): "
             f"{out['instrument_flag']} (invalid: {out['invalid_runs']})")
    if "fit_flag" in out:
        L.append(f"- fit counters (expect {EXPECT_STEPS:.0f}): "
                 f"{out['fit_flag']}")
    if "not_claimed" in out:
        L.append("\n## NOT claimed by this read\n")
        for n in out["not_claimed"]:
            L.append(f"- {n}")
    return "\n".join(L) + "\n"


def run(args):
    def expand(pat):
        return (sorted(globmod.glob(pat)) if "," not in pat
                else [x.strip() for x in pat.split(",") if x.strip()])

    sc = expand(args.scare)
    ct = expand(args.ctrl)
    assert sc and ct, "need --scare and --ctrl"
    assert args.output and args.output != "TBD", (
        "registered read must persist: pass an explicit --output")
    out_json = os.path.join(args.output, "se_a3per_read.json")
    assert not os.path.exists(out_json), (
        f"{out_json} exists — the read is ONE execution")
    # FATAL gates run BEFORE anything is written, so the one-execution
    # guard survives a repair-and-rerun.
    recs = ([read_run(d, "scare") for d in sc]
            + [read_run(d, "ctrl") for d in ct])
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
                        "n_steps", "valid", "occ_manifest",
                        "n_steps_manifest", "a2_occupancy", "a2_steps",
                        "n_chunks_manifest", "n_chunks_extension",
                        "manifest_sha256", "resume", "disag_replay_rew",
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
        print(f"A3-PER PRIMARY: delta={p['delta']:+.4f} p={p['p']:.4f} "
              f"fires={p['fires']}")
        w = out["secondary_w"]
        print(f"  SECONDARY-W: p={w['p']:.4f} fires={w['fires']} -> "
              f"{out['attenuation']}; fit {out.get('fit_flag')}")
    else:
        print(f"A3-PER PRIMARY: {out.get('primary')}")
    return out


# ---------------------------------------------------------------- selfcheck

def _write_chunks(rd, prefix, n_steps, occ, dim=2, seed=0):
    """Replay chunks whose position[REGION_INDEX] realizes `occ` EXACTLY
    (hits = round(occ * n_steps)). Returns the basenames written."""
    rdir = os.path.join(rd, "replay")
    os.makedirs(rdir, exist_ok=True)
    hits = int(round(occ * n_steps))
    col = np.empty(n_steps, np.float32)
    col[:hits] = np.float32(REGION_THRESHOLD + 0.10)
    col[hits:] = np.float32(REGION_THRESHOLD - 0.10)
    rng = np.random.default_rng(seed)
    rng.shuffle(col)
    names = []
    chunk = 120000
    for ci, s in enumerate(range(0, n_steps, chunk)):
        seg = col[s:s + chunk]
        arr = np.zeros((seg.size, dim), np.float32)
        arr[:, REGION_INDEX] = seg
        name = f"{prefix}_{ci:04d}.npz"
        np.savez(os.path.join(rdir, name), position=arr,
                 is_first=np.zeros(seg.size, bool))
        names.append(name)
    return names


def _fixture(root, name, seed, arm, ext_occ=0.50, ext_steps=420000,
             staged_steps=None, staged_occ=None, cfg_patch=None,
             rew=1.2e-4, score=1.0, first_step=501376.0,
             first_items=495000.0, manifest_mode="ok", tiny=False,
             task=TASK, steps=EXPECT_STEPS, dim=2, last_step=998752):
    """A synthetic extended run dir: config.yaml, a staged replay window
    reproducing A2's recorded numbers for `seed`, an extension window,
    STAGED_MANIFEST.txt, metrics.jsonl and scores.jsonl."""
    import yaml
    rd = os.path.join(root, name)
    os.makedirs(rd, exist_ok=True)
    cfg = {
        "task": task, "seed": seed,
        "agent": {"expl": {"mode": "p2e", "disag_bootstrap": True,
                           "disag_bootstrap_prob": 0.8, "disag_ens": 8,
                           "disag_scale": 1000.0, "disag_head": "det",
                           "penalty_mix": False}},
        "penalty": {"scale": 0.0, "gate_key": "", "gate_index": 0,
                    "gate_threshold": 0.0},
        "planted": {"source_key": REGION_KEY, "basesd": BASESD_PLANTED,
                    "gate_key": "", "gate_index": 0, "gate_threshold": 0.0},
        "distractor": {"dim": DIM, "basesd": BASESD_N, "theta": THETA_OU,
                       "gate_key": "", "gate_index": 0,
                       "gate_threshold": 0.0, "mod_index": MOD_INDEX,
                       "mod_key": (MOD_KEY if arm == "scare" else ""),
                       "mod_lo": (MOD_LO if arm == "scare" else 0.0),
                       "mod_hi": (MOD_HI if arm == "scare" else 1.0),
                       "scale": (1.0 if arm == "scare" else FLAT_SCALE)},
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

    if tiny:
        staged = _write_chunks(rd, "staged", 4000, 0.5, dim, seed)
        _write_chunks(rd, "ext", 4000, ext_occ, dim, seed + 500)
    else:
        st_n = A2_STEPS[seed] if staged_steps is None else staged_steps
        st_o = A2_OCC[seed] if staged_occ is None else staged_occ
        staged = _write_chunks(rd, "staged", st_n, st_o, dim, seed)
        _write_chunks(rd, "ext", ext_steps, ext_occ, dim, seed + 500)

    mpath = os.path.join(rd, MANIFEST_NAME)
    if manifest_mode == "missing":
        pass
    elif manifest_mode == "empty":
        open(mpath, "w").close()
    else:
        lines = list(staged)
        if manifest_mode == "dup":
            lines = lines + [staged[0]]
        elif manifest_mode == "subset":
            lines = lines + ["staged_9999.npz"]     # listed, never written
        elif manifest_mode == "junk":
            lines = lines + ["replay/oops.txt"]
        elif manifest_mode == "late":
            # the manifest was taken AFTER the resume had already written
            # chunks: it swallows one extension chunk
            lines = lines + [x for x in replay_files(rd)
                             if x.startswith("ext_")][:1]
        with open(mpath, "w") as f:
            f.write("\n".join(lines) + "\n")

    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": first_step,
                            RESUME_ITEMS_KEY: first_items,
                            COLLAPSE_KEY: rew}) + "\n")
        f.write(json.dumps({"step": 750000, COLLAPSE_KEY: rew}) + "\n")
        f.write(json.dumps({"step": last_step,
                            RESUME_ITEMS_KEY: first_items + 4.9e5,
                            COLLAPSE_KEY: rew}) + "\n")
    with open(os.path.join(rd, "scores.jsonl"), "w") as f:
        for i in range(120):
            f.write(json.dumps({"step": i, "episode/score":
                                score + 0.01 * (i % 3)}) + "\n")
    return rd


def _expect_fail(fn, needle):
    try:
        fn()
    except AssertionError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise SystemExit(f"GATE DID NOT FIRE: {needle}")


def _panel(root, sub, scare_occ, ctrl_occ, **kw):
    sc = [_fixture(os.path.join(root, sub), f"a3per_scare_s{80 + i}",
                   80 + i, "scare",
                   ext_occ=(scare_occ(i) if callable(scare_occ)
                            else scare_occ), **kw)
          for i in range(4)]
    ct = [_fixture(os.path.join(root, sub), f"a3per_ctrl_s{76 + i}",
                   76 + i, "ctrl",
                   ext_occ=(ctrl_occ(i) if callable(ctrl_occ) else ctrl_occ),
                   **kw)
          for i in range(4)]
    return sc, ct


def _read_all(sc, ct):
    return ([read_run(d, "scare") for d in sc]
            + [read_run(d, "ctrl") for d in ct])


def selfcheck():
    import shutil
    import tempfile
    root = tempfile.mkdtemp(prefix="a3per_sc_")
    try:
        # ---- 0) constants + statistic mechanics -------------------------
        assert A2_ARTIFACT_VERIFIED, (
            "the A2 artifact must be present and must re-verify")
        obs, p, tot = exact_perm_p([1, 1, 1, 1, 0, 0, 0, 0],
                                   [True] * 4 + [False] * 4, True)
        assert tot == 70 and obs == 1.0 and abs(p - 1 / 70) < 1e-12
        print("  0 A2 artifact re-verified at import (8 occupancies, 8 step "
              "counts, delta %+.6f, p %.4f, cell SCARECROW-FENCES); "
              "C(8,4)=70 machinery PASS" % (A2_DELTA, A2_P))
        print("    bound %.6f = A2 delta / 2 = %.3f x A2 pooled sd %.6f"
              % (ATTEN_BOUND, abs(ATTEN_BOUND) / A2_POOLED_SD,
                 A2_POOLED_SD))

        # ---- A) the fence PERSISTS -> primary FIRES ---------------------
        sc, ct = _panel(root, "A", lambda i: 0.44 + 0.004 * i,
                        lambda i: 0.52 + 0.004 * i)
        recsA = _read_all(sc, ct)
        agg = aggregate(recsA)
        assert agg["outcome"] == "FENCE-HOLDS", agg["outcome"]
        assert abs(agg["primary"]["p"] - 1 / 70) < 1e-12, agg["primary"]
        assert agg["primary"]["delta"] < 0 and agg["primary"]["fires"]
        # a persisting fence DEEPER than half of A2's must NOT be bounded
        assert agg["attenuation"] == "NOT-BOUNDED", agg["secondary_w"]
        assert "occupancy_ext_scare" in agg and "score_two_sided" in agg
        print("  A persisting fence: delta %+.4f p %.4f -> FENCE-HOLDS, "
              "SECONDARY-W p %.4f -> %s PASS"
              % (agg["primary"]["delta"], agg["primary"]["p"],
                 agg["secondary_w"]["p"], agg["attenuation"]))

        # ---- B) WASHED OUT: extension null, manifest window still A2 ----
        sc, ct = _panel(root, "B", lambda i: 0.500 + 0.002 * i,
                        lambda i: 0.501 + 0.002 * i)
        aggB = aggregate(_read_all(sc, ct))
        assert aggB["outcome"] == "NOT-DETECTED-AT-1e6", aggB["outcome"]
        assert not aggB["primary"]["fires"]
        # the registered attenuation bound is what makes the wash claimable
        assert aggB["attenuation"] == "ATTENUATED-BELOW-HALF", \
            aggB["secondary_w"]
        assert abs(aggB["secondary_w"]["p"] - 1 / 70) < 1e-12
        w = aggB["washout_comparison"]
        # the manifest window still shows A2's fence, by construction
        assert (np.mean(w["manifest_window_replay"]["scare"])
                < np.mean(w["manifest_window_replay"]["ctrl"]) - 0.05)
        assert abs(w["a2_whole_replay_5e5"]["delta"] - A2_DELTA) < 1e-15
        assert abs(w["extension_window_1e6"]["delta"]) < 0.01
        print("  B washed out: extension delta %+.4f (no fire) while the "
              "manifest window still carries A2's %+.4f; SECONDARY-W p "
              "%.4f -> ATTENUATED-BELOW-HALF PASS"
              % (w["extension_window_1e6"]["delta"], A2_DELTA,
                 aggB["secondary_w"]["p"]))

        # ---- B1) WHY the window is registered: on the SAME washed-out
        # panel the whole-1e6-buffer estimand still "fires", because half
        # of it is A2's already-read behaviour.
        whole = {}
        for d in sc + ct:
            o, _n = occupancy(d, REGION_INDEX, REGION_THRESHOLD)
            whole[os.path.basename(d)] = o
        w_s = [v for k, v in whole.items() if "scare" in k]
        w_c = [v for k, v in whole.items() if "ctrl" in k]
        dn, pw, _ = exact_perm_p([-x for x in (w_s + w_c)],
                                 [True] * 4 + [False] * 4, one_sided=True)
        assert -dn < -0.02 and pw <= ALPHA, (dn, pw)
        print("    B1 dilution control: the SAME panel read with the "
              "whole-1e6 buffer gives delta %+.4f p %.4f (a fire) while "
              "the extension window gives %+.4f -- this is exactly the "
              "confound the windowed estimand removes PASS"
              % (-dn, pw, w["extension_window_1e6"]["delta"]))

        # ---- B2) REVERSED extension must not fire either ---------------
        sc, ct = _panel(root, "B2", lambda i: 0.56 + 0.002 * i,
                        lambda i: 0.50 + 0.002 * i)
        aggR = aggregate(_read_all(sc, ct))
        assert aggR["outcome"] == "NOT-DETECTED-AT-1e6"
        assert not aggR["primary"]["fires"] and aggR["primary"]["delta"] > 0
        assert aggR["attenuation"] == "ATTENUATED-BELOW-HALF"
        print("  B2 REVERSED extension: delta %+.4f p %.4f -> no fire "
              "(and the attenuation bound still holds) PASS"
              % (aggR["primary"]["delta"], aggR["primary"]["p"]))

        # ---- C) RESUME-INTEGRITY: the silent-failure fixtures -----------
        d = _fixture(root, "C1/a3per_scare_s80", 80, "scare", tiny=True,
                     first_items=3136.0)
        _expect_fail(lambda: read_run(d, "scare"), "was NOT repopulated")
        d = _fixture(root, "C2/a3per_scare_s80", 80, "scare", tiny=True,
                     first_step=4160.0)
        _expect_fail(lambda: read_run(d, "scare"), "RESTARTED FROM SCRATCH")
        d = _fixture(root, "C3/a3per_scare_s80", 80, "scare", tiny=True)
        os.remove(os.path.join(d, "metrics.jsonl"))
        _expect_fail(lambda: read_run(d, "scare"), "no 'step' row")
        d = _fixture(root, "C4/a3per_scare_s80", 80, "scare", tiny=True)
        with open(os.path.join(d, "metrics.jsonl"), "w") as f:
            f.write(json.dumps({"step": 501376.0, COLLAPSE_KEY: 1e-4}) + "\n")
        _expect_fail(lambda: read_run(d, "scare"), "no 'replay/items' row")
        print("  C EMPTY-BUFFER resume, step reset, and both missing "
              "witnesses -> FATAL PASS")

        # ---- D) the manifest itself ------------------------------------
        for mode, needle in (("missing", "MISSING"), ("empty", "is EMPTY"),
                             ("dup", "duplicate entries"),
                             ("junk", "non-chunk entries"),
                             ("subset", "are ABSENT")):
            d = _fixture(root, f"D_{mode}/a3per_scare_s80", 80, "scare",
                         tiny=True, manifest_mode=mode)
            _expect_fail(lambda d=d: read_run(d, "scare"), needle)
        # a manifest that swallows the whole tree leaves no window
        d = _fixture(root, "D_all/a3per_scare_s80", 80, "scare", tiny=True)
        allf = replay_files(d)
        with open(os.path.join(d, MANIFEST_NAME), "w") as f:
            f.write("\n".join(allf) + "\n")
        _expect_fail(lambda: read_run(d, "scare"), "extension window is EMPTY")
        # a manifest taken AFTER the resume started (the re-queue hazard the
        # wave's `[ -f ] ||` guard exists to prevent) swallows extension
        # chunks and is caught by A2-PROVENANCE, not by the window gates
        d = _fixture(root, "D_late/a3per_scare_s80", 80, "scare",
                     manifest_mode="late")
        _expect_fail(lambda: read_run(d, "scare"), "INCOMPLETE")
        print("  D manifest missing / empty / duplicated / malformed / "
              "SUBSET-violating / all-swallowing / taken-LATE -> FATAL PASS")

        # ---- E) A2-PROVENANCE: incomplete or wrong staging --------------
        d = _fixture(root, "E1/a3per_scare_s80", 80, "scare",
                     staged_steps=A2_STEPS[80] - 120000)
        _expect_fail(lambda: read_run(d, "scare"), "INCOMPLETE")
        d = _fixture(root, "E2/a3per_scare_s80", 80, "scare",
                     staged_occ=A2_OCC[81])
        _expect_fail(lambda: read_run(d, "scare"),
                     "not the A2 run this seed names")
        print("  E A2-PROVENANCE: short stage (%d steps missing) and a "
              "foreign run under the seed's name -> FATAL PASS"
              % 120000)

        # ---- F) identity / continuity leaks ----------------------------
        d = _fixture(root, "F_arm1/a3per_scare_s76", 76, "scare", tiny=True)
        _expect_fail(lambda: read_run(d, "scare"), "WRONG ARM MAP")
        d = _fixture(root, "F_arm2/a3per_ctrl_s80", 80, "ctrl", tiny=True)
        _expect_fail(lambda: read_run(d, "ctrl"), "WRONG ARM MAP")
        for s in (84, 88, 210, 99):
            d = _fixture(root, f"F_seed{s}/x", s if s in A2_STEPS else 80,
                         "scare", tiny=True, cfg_patch={"seed": s})
            _expect_fail(lambda d=d: read_run(d, "scare"),
                         "not one of A2's extended seeds")
        leaks = [
            ("A2 quadruple off", {"distractor.mod_key": ""}, "scare pins"),
            ("mod_lo drift", {"distractor.mod_lo": -0.3}, "scare pins"),
            ("mod_hi drift", {"distractor.mod_hi": 0.2}, "scare pins"),
            ("scare scale", {"distractor.scale": FLAT_SCALE}, "scare pins"),
            ("finger mod_lo", {"distractor.mod_lo": -4.589493315966835},
             "scare pins"),
            ("theta leak", {"distractor.theta": 0.2},
             "DISTRACTOR_THETA leak"),
            ("head leak", {"agent.expl.disag_head": "gauss"},
             "DISAG_HEAD leak"),
            ("bootstrap off", {"agent.expl.disag_bootstrap": False},
             "expl pins differ"),
            ("apt", {"agent.expl.mode": "apt"}, "expl pins differ"),
            ("penalty live", {"penalty.scale": 0.5},
             "penalty machinery must be inert"),
            ("penalty_mix", {"agent.expl.penalty_mix": True},
             "penalty machinery must be inert"),
            ("finger planted basesd", {"planted.basesd": 0.54193702},
             "finger leak"),
            ("finger distractor basesd", {"distractor.basesd": 1.35803102},
             "differ from the registered"),
            ("dim", {"distractor.dim": 0}, "differ from the registered"),
            ("gate on", {"distractor.gate_key": "position"},
             "gates must be off"),
            ("planted gate on", {"planted.gate_key": "position"},
             "gates must be off"),
            ("steps still 5e5", {"run.steps": 5e5}, "not an extended run"),
            ("task", {"task": "dmc_finger_spin"}, "!= dmc_cheetah_run"),
        ]
        for i, (tag, patch, needle) in enumerate(leaks):
            dd = _fixture(root, f"F{i}/a3per_scare_s80", 80, "scare",
                          tiny=True, cfg_patch=patch)
            _expect_fail(lambda dd=dd: read_run(dd, "scare"), needle)
        # the ctrl arm's own construction
        dd = _fixture(root, "Fc1/a3per_ctrl_s76", 76, "ctrl", tiny=True,
                      cfg_patch={"distractor.mod_key": MOD_KEY})
        _expect_fail(lambda: read_run(dd, "ctrl"), "ctrl pins")
        dd = _fixture(root, "Fc2/a3per_ctrl_s76", 76, "ctrl", tiny=True,
                      cfg_patch={"distractor.scale": 1.0})
        _expect_fail(lambda: read_run(dd, "ctrl"), "ctrl pins")
        print("  F wrong-arm-map, non-extended seeds (84/88/210/99) and %d "
              "A2-quadruple / continuity leak fixtures -> FATAL PASS"
              % (len(leaks) + 2))

        # ---- G) panel identity: missing / short / duplicate -------------
        _expect_fail(lambda: aggregate(recsA[:3] + recsA[4:]), "scare seeds")
        _expect_fail(lambda: aggregate(recsA[:6]), "ctrl seeds")
        _expect_fail(lambda: aggregate(recsA + [dict(recsA[0])]),
                     "duplicate seeds")
        print("  G missing / short / duplicate panels -> REFUSED "
              "(denominator pinned at 8) PASS")

        # ---- H) validity cells -----------------------------------------
        sc, ct = _panel(root, "H1", 0.44, 0.52, rew=0.0)
        agg = aggregate(_read_all(sc, ct))
        assert agg["outcome"] == "GLOBAL-COLLAPSE", agg["outcome"]
        assert "washout_comparison" in agg     # descriptives on every branch
        # and the validity branches must still RENDER end to end
        o1 = run(parse_args(["--scare", os.path.join(root, "H1", "a3per_scare_s8*"),
                             "--ctrl", os.path.join(root, "H1", "a3per_ctrl_s7*"),
                             "--output", os.path.join(root, "outH1")]))
        assert o1["outcome"] == "GLOBAL-COLLAPSE"
        t1 = open(os.path.join(root, "outH1", "RESULTS.md")).read()
        assert "GLOBAL-COLLAPSE" in t1 and "Wash-out comparison" in t1
        sc, ct = _panel(root, "H2", 0.44, 0.52, ext_steps=240000)
        agg = aggregate(_read_all(sc, ct))
        assert agg["outcome"] == "NOT-ADJUDICABLE", agg["outcome"]
        o2 = run(parse_args(["--scare", os.path.join(root, "H2", "a3per_scare_s8*"),
                             "--ctrl", os.path.join(root, "H2", "a3per_ctrl_s7*"),
                             "--output", os.path.join(root, "outH2")]))
        assert o2["outcome"] == "NOT-ADJUDICABLE"
        t2 = open(os.path.join(root, "outH2", "RESULTS.md")).read()
        assert "NOT-ADJUDICABLE" in t2 and "Wash-out comparison" in t2
        sc, ct = _panel(root, "H3", 0.44, 0.52)
        for dd in sc + ct:
            rows = _metrics_rows(dd)
            with open(os.path.join(dd, "metrics.jsonl"), "w") as f:
                for r in rows:
                    r.pop(COLLAPSE_KEY, None)
                    f.write(json.dumps(r) + "\n")
        agg = aggregate(_read_all(sc, ct))
        assert agg["outcome"] == "FENCE-HOLDS"
        assert not agg["collapse_check"]["available"]
        assert "COLLAPSE-CHECK-UNAVAILABLE" in agg["primary"]["statement"]
        print("  H dead objective -> GLOBAL-COLLAPSE; short extension "
              "window -> NOT-ADJUDICABLE; absent telemetry -> annotated, "
              "not fatal PASS")

        # ---- I) the windowed estimand is BIT-IDENTICAL to the frozen one
        d0 = sorted(globmod.glob(os.path.join(root, "A", "a3per_scare_s80")))[0]
        allf = replay_files(d0)
        o_full, n_full = occupancy_over(d0, allf)
        o_ref, n_ref = occupancy(d0, REGION_INDEX, REGION_THRESHOLD)
        assert (o_full, n_full) == (o_ref, n_ref), (o_full, o_ref)
        man, ext, _ = windows(d0)
        o_m, n_m = occupancy_over(d0, man)
        o_e, n_e = occupancy_over(d0, ext)
        assert n_m + n_e == n_full and set(man) | set(ext) == set(allf)
        assert not set(man) & set(ext)
        assert o_m == A2_OCC[80] and n_m == A2_STEPS[80]
        print("  I occupancy_over on the FULL listing is bit-identical to "
              "se_m3_read.occupancy (%r == %r, %d == %d); the two windows "
              "partition it exactly PASS" % (o_full, o_ref, n_full, n_ref))

        # ---- J) end-to-end + the ONE-execution guard --------------------
        outd = os.path.join(root, "out")
        out = run(parse_args([
            "--scare", os.path.join(root, "A", "a3per_scare_s8*"),
            "--ctrl", os.path.join(root, "A", "a3per_ctrl_s7*"),
            "--output", outd]))
        assert out["outcome"] == "FENCE-HOLDS"
        assert os.path.exists(os.path.join(outd, "se_a3per_read.json"))
        txt = open(os.path.join(outd, "RESULTS.md")).read()
        assert "A3 PERSISTENCE LEG" in txt and "Wash-out comparison" in txt
        assert "EXTENSION WINDOW only" in txt
        assert str(REGION_THRESHOLD) in txt
        assert out["fit_flag"] == "OK", out["fit_counters"]
        _expect_fail(lambda: run(parse_args([
            "--scare", os.path.join(root, "A", "a3per_scare_s8*"),
            "--ctrl", os.path.join(root, "A", "a3per_ctrl_s7*"),
            "--output", outd])), "ONE execution")
        print("  J end-to-end read + RESULTS.md + fit counters (expect 1e6) "
              "+ ONE-execution guard PASS")

        print("\nse_a3per_read selfcheck PASS")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    a = parse_args()
    return selfcheck() if a.selfcheck else (run(a) and 0)


if __name__ == "__main__":
    sys.exit(main() or 0)
