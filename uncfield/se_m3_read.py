"""SE-wave Stage-2 M3 behavioral FROZEN reader
(PREREG_nfi_scale_exhibit_amend2_20260814.md §2; built + selfchecked
BEFORE Stage-2 compute, executes the M3 read ONCE).

Arms: gated (seeds 20-23, all appended channels emitted only when
position[0] > -0.13009691) vs ungated (seeds 24-27, Stage-1 config).

Registered measurements:
  PRIMARY (directional): mean gate-region occupancy(gated) minus
    mean occupancy(ungated) > 0 — "diversion toward the gate region";
    exact ONE-sided label-permutation over all C(8,4)=70 arm
    assignments (min attainable p = 1/70 ~ .0143, disclosed).
  SECONDARY (reporting): task-return delta = mean episode/score over
    the final 100 scores.jsonl entries, same exact permutation,
    TWO-sided; per-arm BCa intervals.

Occupancy estimand (amendment §2): fraction of steps with
position[gate_index] > gate_threshold over the ENTIRE synced replay
buffer — deterministic full chunk scan, no sampling rng; validity
floor >= 1e5 steps, else the run is flagged invalid and the read
refuses to fire the primary (n != 4+4 is not adjudicable).

Identity gates: gated train seeds == {20..23}, ungated == {24..27}
(config.yaml; duplicates fatal, mismatch fatal — the arm labels come
from the CONFIG's gate_key, not from the directory name); gated runs
must have planted.gate_key AND distractor.gate_key == 'position';
ungated runs must have both == ''. Fit counters reported.

Run:  python -m uncfield.se_m3_read --gated "<glob>" --ungated "<glob>"
          --output <dir>
      python -m uncfield.se_m3_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import itertools
import json
import os

import numpy as np

from uncfield.se_mask import bca_interval
from uncfield.se_read import fit_counters

GATE_KEY = "position"
GATE_INDEX = 0
GATE_THRESHOLD = -0.13009691          # amendment §2 (smoke median)
GATED_SEEDS = set(range(20, 24))
UNGATED_SEEDS = set(range(24, 28))
MIN_STEPS = int(1e5)
SCORE_TAIL = 100


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gated", default="", help="4 gated run dirs (glob)")
    p.add_argument("--ungated", default="", help="4 ungated run dirs")
    p.add_argument("--gate_index", type=int, default=GATE_INDEX)
    p.add_argument("--gate_threshold", type=float, default=GATE_THRESHOLD)
    p.add_argument("--expect_steps", type=float, default=5e5)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def occupancy(run_dir, gate_index, gate_threshold):
    """Deterministic full-buffer scan: fraction of steps with
    position[gate_index] > threshold. Returns (frac, n_steps)."""
    rd = os.path.join(run_dir, "replay")
    files = sorted(f for f in os.listdir(rd) if f.endswith(".npz"))
    assert files, f"{run_dir}: empty replay dir"
    hits = total = 0
    for f in files:
        with np.load(os.path.join(rd, f)) as z:
            assert GATE_KEY in z, (run_dir, f, list(z.keys()))
            v = np.asarray(z[GATE_KEY], np.float64)
            v = v.reshape(-1, v.shape[-1])[:, gate_index]
            hits += int((v > gate_threshold).sum())
            total += v.size
    return hits / total, total


def read_run(run_dir, expect_gated, gate_index, gate_threshold):
    """One Stage-2 run: config-derived arm identity + occupancy +
    score tail."""
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    train_seed = int(cfg["seed"])
    pg = str(cfg["planted"]["gate_key"])
    dg = str(cfg["distractor"]["gate_key"])
    is_gated = bool(pg) or bool(dg)
    assert is_gated == expect_gated, (
        f"{run_dir}: config gate_keys planted={pg!r} distractor={dg!r} "
        f"do not match the arm this run was passed as "
        f"({'gated' if expect_gated else 'ungated'})")
    if expect_gated:
        assert pg == GATE_KEY and dg == GATE_KEY, (
            f"{run_dir}: gated arm must gate BOTH wrappers on "
            f"{GATE_KEY!r} (got planted={pg!r}, distractor={dg!r})")
        for blk in ("planted", "distractor"):
            gi = int(cfg[blk]["gate_index"])
            gt = float(cfg[blk]["gate_threshold"])
            assert gi == gate_index and abs(gt - gate_threshold) < 1e-9, (
                f"{run_dir}: {blk} gate ({gi}, {gt}) != registered "
                f"({gate_index}, {gate_threshold})")
    occ, n_steps = occupancy(run_dir, gate_index, gate_threshold)
    scores = []
    try:
        with open(os.path.join(run_dir, "scores.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                if "episode/score" in d:           # review #23 m16
                    scores.append(float(d["episode/score"]))
    except OSError:
        pass
    score = float(np.mean(scores[-SCORE_TAIL:])) if scores else None
    return dict(run_dir=os.path.abspath(run_dir), train_seed=train_seed,
                gated=expect_gated, occupancy=occ, n_steps=int(n_steps),
                valid=bool(n_steps >= MIN_STEPS), score=score,
                n_score_entries=len(scores))


def exact_perm_p(values, labels, one_sided):
    """Exact label-permutation over all C(8,4)=70 assignments of 4
    'gated' labels to 8 runs. Statistic: mean(gated) - mean(ungated).
    Observed assignment included in the count (p >= 1/70)."""
    values = np.asarray(values, np.float64)
    labels = np.asarray(labels, bool)
    n = values.size
    k = int(labels.sum())
    assert n == 8 and k == 4, (n, k)
    obs = values[labels].mean() - values[~labels].mean()
    # relative tie epsilon (review #23 m15): an absolute 1e-15 is below
    # float64 rounding at score magnitudes, silently dropping exact
    # ties (anti-conservative); relative on both forms
    eps = 1e-12 * max(1.0, abs(obs))
    count = total = 0
    for comb in itertools.combinations(range(n), k):
        lab = np.zeros(n, bool)
        lab[list(comb)] = True
        t = values[lab].mean() - values[~lab].mean()
        total += 1
        if one_sided:
            count += (t >= obs - eps)
        else:
            count += (abs(t) >= abs(obs) - eps)
    return float(obs), count / total, total


def aggregate(recs):
    gated = [r for r in recs if r["gated"]]
    ungated = [r for r in recs if not r["gated"]]
    gs = sorted(r["train_seed"] for r in gated)
    us = sorted(r["train_seed"] for r in ungated)
    assert len(set(gs)) == len(gs) and len(set(us)) == len(us), (
        f"duplicate training seeds (gated {gs}, ungated {us})")
    assert set(gs) == GATED_SEEDS, f"gated seeds {gs} != [20..23]"
    assert set(us) == UNGATED_SEEDS, f"ungated seeds {us} != [24..27]"
    invalid = [r["run_dir"] for r in recs if not r["valid"]]
    out = dict(n_gated=len(gated), n_ungated=len(ungated),
               invalid_runs=invalid,
               instrument_flag=("CLEAN" if not invalid
                                else "VALIDITY-VIOLATIONS"))
    if invalid or len(gated) != 4 or len(ungated) != 4:
        out["primary"] = "NOT-ADJUDICABLE (need 4+4 valid runs)"
        return out

    ordered = gated + ungated
    labels = [True] * 4 + [False] * 4
    occ = [r["occupancy"] for r in ordered]
    d_occ, p_occ, total = exact_perm_p(occ, labels, one_sided=True)
    out["primary"] = dict(
        statement="diversion toward the gate region: occupancy(gated) "
                  "> occupancy(ungated), one-sided exact permutation",
        delta=d_occ, p=p_occ, n_assignments=total,
        fires=bool(p_occ <= 0.05),
        min_attainable_p=1.0 / total)
    for name, arm in (("gated", gated), ("ungated", ungated)):
        m, lo, hi = bca_interval(
            np.asarray([r["occupancy"] for r in arm]),
            np.random.default_rng(11 if name == "gated" else 12), 2000)
        out[f"occupancy_{name}"] = dict(mean=m, bca=[lo, hi])
    if all(r["score"] is not None for r in ordered):
        sc = [r["score"] for r in ordered]
        d_sc, p_sc, _ = exact_perm_p(sc, labels, one_sided=False)
        out["secondary_score"] = dict(delta=d_sc, p=p_sc,
                                      note="reporting only, two-sided")
    else:
        out["secondary_score"] = "UNAVAILABLE (missing scores.jsonl)"
    return out


def run(args):
    def expand(pat):
        if "," in pat:
            return [d.strip() for d in pat.split(",") if d.strip()]
        return sorted(globmod.glob(pat))

    gated_dirs = expand(args.gated)
    ungated_dirs = expand(args.ungated)
    assert gated_dirs and ungated_dirs, "need --gated and --ungated"
    assert args.output, "registered read must persist: pass --output"
    # review #23 M6: the CLI knobs exist for tooling only — the frozen
    # read runs at the registered constants, and the output records them
    assert args.gate_index == GATE_INDEX and \
        abs(args.gate_threshold - GATE_THRESHOLD) < 1e-12, (
        "gate constants differ from the registered pin "
        f"({GATE_KEY}[{GATE_INDEX}] > {GATE_THRESHOLD})")
    recs = ([read_run(d, True, args.gate_index, args.gate_threshold)
             for d in gated_dirs]
            + [read_run(d, False, args.gate_index, args.gate_threshold)
               for d in ungated_dirs])
    out = aggregate(recs)
    out["gate"] = dict(key=GATE_KEY, index=args.gate_index,
                       threshold=args.gate_threshold)
    out["fit_counters"] = {r["run_dir"]: fit_counters(r["run_dir"],
                                                      args.expect_steps)
                           for r in recs}
    out["per_run"] = [{k: r[k] for k in ("run_dir", "train_seed", "gated",
                                         "occupancy", "n_steps", "valid",
                                         "score")} for r in recs]
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "se_m3_read.json"), "w") as f:
        json.dump(out, f, indent=1)
    if isinstance(out["primary"], dict):
        pr = out["primary"]
        print(f"M3 PRIMARY: delta={pr['delta']:+.4f} p={pr['p']:.4f} "
              f"fires={pr['fires']}")
    else:
        print(f"M3 PRIMARY: {out['primary']}")
    return out


# ---------------------------------------------------------------- selfcheck

def _fixture_run(tmp, name, seed, gated, occ_frac, score=1.0,
                 n_steps=120000, chunk=5000, gate_cfg_override=None):
    """Synthetic Stage-2 run: replay chunks whose position[GATE_INDEX]
    realizes the requested occupancy exactly; config with arm-correct
    gate keys; scores + metrics."""
    rd = os.path.join(tmp, name)
    os.makedirs(os.path.join(rd, "replay"), exist_ok=True)
    rng = np.random.default_rng(seed)
    left = n_steps
    ci = 0
    while left > 0:
        n = min(chunk, left)
        n_hi = int(round(occ_frac * n))
        col = np.concatenate([
            np.full(n_hi, GATE_THRESHOLD + 0.1),
            np.full(n - n_hi, GATE_THRESHOLD - 0.1)])
        rng.shuffle(col)
        arr = np.zeros((n, 8), np.float32)
        arr[:, GATE_INDEX] = col
        arr[:, 1:] = rng.normal(0, 1, (n, 7))
        np.savez(os.path.join(rd, "replay", f"chunk{ci:04d}.npz"),
                 position=arr, is_first=np.zeros(n, bool))
        left -= n
        ci += 1
    gk = GATE_KEY if gated else ""
    cfg = (f"seed: {seed}\n"
           f"planted:\n  gate_key: '{gk}'\n  gate_index: {GATE_INDEX}\n"
           f"  gate_threshold: {GATE_THRESHOLD}\n"
           f"distractor:\n  gate_key: '{gk}'\n  gate_index: {GATE_INDEX}\n"
           f"  gate_threshold: {GATE_THRESHOLD}\n"
           f"run:\n  steps: 500000.0\n")
    if gate_cfg_override is not None:
        cfg = gate_cfg_override
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(cfg)
    with open(os.path.join(rd, "scores.jsonl"), "w") as f:
        for i in range(150):
            f.write(json.dumps({"step": i, "episode/score":
                                score + 0.01 * (i % 3)}) + "\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 499968}) + "\n")
    return rd


def selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # exact permutation mechanics: known values
        obs, p, total = exact_perm_p(
            [1, 1, 1, 1, 0, 0, 0, 0], [True] * 4 + [False] * 4, True)
        assert total == 70 and obs == 1.0 and abs(p - 1 / 70) < 1e-12
        _, p2, _ = exact_perm_p(
            [0, 0, 0, 0, 1, 1, 1, 1], [True] * 4 + [False] * 4, True)
        assert p2 == 1.0                      # wrong direction
        _, p3, _ = exact_perm_p(
            [1, 0, 1, 0, 0, 1, 0, 1], [True] * 4 + [False] * 4, True)
        assert p3 > 0.3                       # exchangeable

        # A) diversion: gated occupancy 0.7, ungated 0.3 -> p = 1/70
        g = [_fixture_run(os.path.join(tmp, "A"), f"g{i}", 20 + i, True,
                          0.7, score=2.0) for i in range(4)]
        u = [_fixture_run(os.path.join(tmp, "A"), f"u{i}", 24 + i, False,
                          0.3, score=1.0) for i in range(4)]
        recs = ([read_run(d, True, GATE_INDEX, GATE_THRESHOLD) for d in g]
                + [read_run(d, False, GATE_INDEX, GATE_THRESHOLD)
                   for d in u])
        # occupancy realized exactly
        assert all(abs(r["occupancy"] - (0.7 if r["gated"] else 0.3))
                   < 1e-3 for r in recs)
        agg = aggregate(recs)
        assert agg["primary"]["fires"] and \
            abs(agg["primary"]["p"] - 1 / 70) < 1e-12
        assert agg["secondary_score"]["p"] <= 3 / 70 + 1e-12

        # B) null: equal occupancy -> does not fire
        g = [_fixture_run(os.path.join(tmp, "B"), f"g{i}", 20 + i, True,
                          0.5 + 0.001 * i) for i in range(4)]
        u = [_fixture_run(os.path.join(tmp, "B"), f"u{i}", 24 + i, False,
                          0.5 + 0.001 * i) for i in range(4)]
        recs = ([read_run(d, True, GATE_INDEX, GATE_THRESHOLD) for d in g]
                + [read_run(d, False, GATE_INDEX, GATE_THRESHOLD)
                   for d in u])
        agg = aggregate(recs)
        assert not agg["primary"]["fires"], agg["primary"]

        # C) arm-identity gate: ungated config passed as gated
        rd = _fixture_run(os.path.join(tmp, "C"), "x", 20, False, 0.5)
        try:
            read_run(rd, True, GATE_INDEX, GATE_THRESHOLD)
            raise SystemExit("arm-identity gate FAILED to fire")
        except AssertionError as e:
            assert "do not match the arm" in str(e)

        # C2) half-gated config (planted only) must abort
        rd = _fixture_run(
            os.path.join(tmp, "C2"), "y", 21, True, 0.5,
            gate_cfg_override=(
                f"seed: 21\nplanted:\n  gate_key: '{GATE_KEY}'\n"
                f"  gate_index: 0\n  gate_threshold: {GATE_THRESHOLD}\n"
                f"distractor:\n  gate_key: ''\n  gate_index: 0\n"
                f"  gate_threshold: 0.0\nrun:\n  steps: 500000.0\n"))
        try:
            read_run(rd, True, GATE_INDEX, GATE_THRESHOLD)
            raise SystemExit("both-wrappers gate FAILED to fire")
        except AssertionError as e:
            assert "BOTH wrappers" in str(e)

        # D) wrong seed family fatal
        g = [_fixture_run(os.path.join(tmp, "D"), f"g{i}", 30 + i, True,
                          0.6) for i in range(4)]
        u = [_fixture_run(os.path.join(tmp, "D"), f"u{i}", 24 + i, False,
                          0.4) for i in range(4)]
        try:
            aggregate([read_run(d, True, GATE_INDEX, GATE_THRESHOLD)
                       for d in g]
                      + [read_run(d, False, GATE_INDEX, GATE_THRESHOLD)
                         for d in u])
            raise SystemExit("seed-family gate FAILED to fire")
        except AssertionError as e:
            assert "gated seeds" in str(e)

        # E) validity floor: short buffer -> NOT-ADJUDICABLE
        g = [_fixture_run(os.path.join(tmp, "E"), f"g{i}", 20 + i, True,
                          0.7, n_steps=(50000 if i == 0 else 120000))
             for i in range(4)]
        u = [_fixture_run(os.path.join(tmp, "E"), f"u{i}", 24 + i, False,
                          0.3) for i in range(4)]
        agg = aggregate([read_run(d, True, GATE_INDEX, GATE_THRESHOLD)
                         for d in g]
                        + [read_run(d, False, GATE_INDEX, GATE_THRESHOLD)
                           for d in u])
        assert agg["instrument_flag"] == "VALIDITY-VIOLATIONS"
        assert agg["primary"] == "NOT-ADJUDICABLE (need 4+4 valid runs)"

        # F) end-to-end run() with globs + output
        outd = os.path.join(tmp, "out")
        out = run(parse_args([
            "--gated", os.path.join(tmp, "A", "g*"),
            "--ungated", os.path.join(tmp, "A", "u*"),
            "--output", outd]))
        assert out["primary"]["fires"]
        assert os.path.exists(os.path.join(outd, "se_m3_read.json"))

    print("se_m3_read selfcheck PASS (exact-perm mechanics 70/1-70/"
          "direction/null, A diversion p=1/70 + score secondary, B null "
          "no-fire, C arm-identity gate, C2 both-wrappers gate, D seed-"
          "family gate, E validity floor -> NOT-ADJUDICABLE, F "
          "end-to-end)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
