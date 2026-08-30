"""A3 CALIBRATION PILOT reader (Track-A generality leg), 30 Aug 2026.

FROZEN. Built and selfchecked BEFORE the pilot compute it consumes.

WHY THIS EXISTS
---------------
The A1/A2 planted quadruple is CHEETAH-CALIBRATED. Its four ramp
constants -- `mod_lo`, `mod_hi`, the gate threshold, and the flat
comparator `scale` -- are not portable to a second environment, so the
Track-A generality leg (A3) cannot reuse them. This reader re-derives
them for a candidate environment under the A1 protocol, and selects
which candidate A3 will run in.

PROVENANCE OF THE CHEETAH CONSTANTS (established 30 Aug by direct
reproduction, and the reason this pilot has TWO phases):

  * `planted.basesd` = 0.0976 and `distractor.basesd` = 1.215 come
    from the RANDOM-POLICY proprio inventory in `uncfield_se.sbatch`
    (`SMOKE=1`; 2000 steps, rng seed 0). Re-running it reproduces
    0.09712176 / 1.21378463 -- within 0.5% of the registered pins, the
    residual being dm_control version drift. That inventory is CPU-only
    and is PHASE A of this pilot.

  * `mod_lo` = -0.32203162, `mod_hi` = +0.19711795 (the smoke replay's
    observed position[0] range), the gate threshold -0.13009691 (its
    median) and the flat scale 0.35891393 (its visitation-weighted
    mean m) do NOT come from that inventory: under a random policy
    cheetah's position[0] spans only [-0.215, -0.065]. They come from
    the 2e4-step p2e smoke RUN's replay -- an agent that actually
    locomotes. That is PHASE B, which this reader consumes.

  Amendment-3 s2 records the cheetah split means as 0.2512 (below) and
  0.4666 (above). Their mean is 0.35891393 = the registered flat scale
  EXACTLY, which is the arithmetic signature of a median split: it
  confirms both that the threshold halves visitation and that this
  reader's decomposition matches the registered one. Gate G5 below
  enforces the general (visitation-weighted) form of that identity.

NON-CIRCULARITY (registered design choice)
------------------------------------------
Phase B runs with `distractor.dim = 0`. The OU channel whose amplitude
ramp is being calibrated must not shape the visitation used to set that
ramp, or the constants are fit to their own manipulation. `planted`
stays ON at the Phase-A basesd because it is Stage-1 substrate and
carries no spatial structure (no gate, no modulation) -- it cannot
induce the region asymmetry being measured.

The cheetah smoke's own planted/distractor state is not recoverable
from the record; DISCLOSED as a deviation-from-unknown rather than a
match.

Run:
  python -m uncfield.se_a3cal_read --pilots "<runroot>/se_a3cal_*" \
      --output artifacts/a3_calib_<date>
  python -m uncfield.se_a3cal_read --selfcheck
"""

import argparse
import glob as globmod
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------- pins
SOURCE_KEY = "position"       # A1 protocol: the modulated coordinate
MOD_INDEX = 0                 # A1 protocol: position[0]
EXPECT_STEPS = 2e4            # cheetah smoke convention (parent prereg s1)
REPLAY_FLOOR = 0.90           # of EXPECT_STEPS, deterministic full scan

# Phase-A constants, measured on the execution site 30 Aug 2026
# (2000 random-policy steps, rng seed 0). Pinned here so Phase B's
# config.yaml can be hard-gated against them.
PHASE_A = {
    "dmc_finger_spin": {"basesd_planted": 0.54193702,
                        "basesd_n": 1.35803102, "position_dim": 4},
    "dmc_hopper_hop":  {"basesd_planted": 0.59473863,
                        "basesd_n": 1.27894744, "position_dim": 6},
}

# Cheetah's registered delivered contrast (amendment 3 s2), the target
# the selection rule matches against. A fair transplant is one that
# delivers the SAME manipulation strength -- not a weaker or a stronger
# one, either of which would confound generality with dose.
CHEETAH_RATIO_AMP = 0.4666 / 0.2512   # 1.8575...
CHEETAH_MBAR = 0.35891393

# Validity gates
W_ABOVE_LO, W_ABOVE_HI = 0.40, 0.60
SPREAD_FLOOR = 1e-6


def scan_replay(run_dir, key=SOURCE_KEY, index=MOD_INDEX):
    """Deterministic full scan of every replay chunk. No sampling rng.

    Returns the 1-D visitation trace of obs[key][index] over the ENTIRE
    synced replay buffer, in chunk-sorted order.
    """
    rd = os.path.join(run_dir, "replay")
    files = sorted(f for f in os.listdir(rd) if f.endswith(".npz"))
    assert files, f"{run_dir}: empty replay dir"
    out = []
    for f in files:
        with np.load(os.path.join(rd, f)) as z:
            assert key in z.files, f"{run_dir}/{f}: no '{key}' array"
            arr = np.asarray(z[key], np.float64)
            assert arr.ndim == 2, f"{run_dir}/{f}: {key} ndim {arr.ndim}"
            assert arr.shape[1] > index, (
                f"{run_dir}/{f}: {key} dim {arr.shape[1]} <= index {index}")
            out.append(arr[:, index])
    return np.concatenate(out)


def calibrate(p0):
    """Re-derive the A1 ramp constants from a visitation trace.

    lo/hi  = observed range   -> distractor.mod_lo / mod_hi
    thr    = observed median  -> planted+distractor gate_threshold
    mbar   = visitation-weighted mean m -> the flat comparator scale
    """
    lo = float(np.min(p0))
    hi = float(np.max(p0))
    thr = float(np.median(p0))
    spread = hi - lo
    if spread <= SPREAD_FLOOR:
        m = np.zeros_like(p0)
    else:
        m = np.clip((p0 - lo) / spread, 0.0, 1.0)
    above = p0 > thr                       # gate semantics: strict '>'
    below = ~above
    w_a = float(above.mean())
    w_b = float(below.mean())
    e_a = float(m[above].mean()) if above.any() else float("nan")
    e_b = float(m[below].mean()) if below.any() else float("nan")
    v_a = float((m[above] ** 2).mean()) if above.any() else float("nan")
    v_b = float((m[below] ** 2).mean()) if below.any() else float("nan")
    return {
        "n_steps": int(p0.size),
        "mod_lo": lo, "mod_hi": hi, "gate_threshold": thr,
        "spread": spread,
        "flat_scale": float(m.mean()),
        "w_below": w_b, "w_above": w_a,
        "mean_m_below": e_b, "mean_m_above": e_a,
        "meansq_m_below": v_b, "meansq_m_above": v_a,
        "ratio_amp": (e_a / e_b) if e_b and np.isfinite(e_b) and e_b > 0
                     else float("nan"),
        "ratio_var": (v_a / v_b) if v_b and np.isfinite(v_b) and v_b > 0
                     else float("nan"),
    }


def gate(c, expect_steps=EXPECT_STEPS):
    """Validity gates. Returns (ok, [failure strings])."""
    fails = []
    if c["n_steps"] < REPLAY_FLOOR * expect_steps:
        fails.append(
            f"G1 replay {c['n_steps']} < {REPLAY_FLOOR:.2f}*{expect_steps:.0f}")
    if c["spread"] <= SPREAD_FLOOR:
        fails.append(f"G2 degenerate spread {c['spread']:.3e}")
    if not (W_ABOVE_LO <= c["w_above"] <= W_ABOVE_HI):
        fails.append(
            f"G3 median split {c['w_above']:.4f} outside "
            f"[{W_ABOVE_LO},{W_ABOVE_HI}] (mass point at the median)")
    if not (np.isfinite(c["mean_m_below"]) and c["mean_m_below"] > 0):
        fails.append("G4 mean m below region non-positive; ratio undefined")
    if np.isfinite(c["mean_m_below"]) and np.isfinite(c["mean_m_above"]):
        recon = c["w_below"] * c["mean_m_below"] + c["w_above"] * c["mean_m_above"]
        if abs(recon - c["flat_scale"]) > 1e-9:
            fails.append(
                f"G5 decomposition identity violated: {recon!r} != "
                f"{c['flat_scale']!r}")
    return (not fails), fails


def read_config(run_dir):
    import yaml
    with open(os.path.join(run_dir, "config.yaml")) as f:
        return yaml.safe_load(f)


def check_pins(run_dir, cfg):
    """Hard-gate the Phase-B config against the registered pilot pins."""
    task = str(cfg.get("task", ""))
    assert task in PHASE_A, f"{run_dir}: task {task} not a registered candidate"
    pa = PHASE_A[task]
    ex = cfg["agent"]["expl"]
    assert str(ex["mode"]) == "p2e" and ex["disag_bootstrap"] is True and \
        int(ex["disag_ens"]) == 8 and abs(float(ex["disag_scale"]) - 1000.0) < 1e-9, \
        f"{run_dir}: expl pins differ from Stage-1"
    d = cfg["distractor"]
    assert int(d["dim"]) == 0, (
        f"{run_dir}: distractor.dim {d['dim']} != 0 -- the pilot MUST NOT "
        f"run the channel it is calibrating (non-circularity)")
    pl = cfg["planted"]
    assert str(pl["source_key"]) == SOURCE_KEY, f"{run_dir}: planted source_key"
    assert abs(float(pl["basesd"]) - pa["basesd_planted"]) < 1e-9, (
        f"{run_dir}: planted.basesd {pl['basesd']} != Phase-A "
        f"{pa['basesd_planted']} for {task}")
    assert str(pl.get("gate_key", "")) == "" and str(d.get("gate_key", "")) == "" \
        and str(d.get("mod_key", "")) == "", (
        f"{run_dir}: gates/modulation must be OFF in the pilot")
    return task


def select(rows):
    """Registered selection rule.

    Among candidates passing every gate, pick the one whose DELIVERED
    amplitude contrast is closest to cheetah's 1.8575x. Tie-break:
    larger position dim, then task name. Pre-specified so the choice
    cannot be made on the outcome of the A3 wave itself.
    """
    ok = [r for r in rows if r["valid"]]
    if not ok:
        return None
    ok.sort(key=lambda r: (abs(r["calib"]["ratio_amp"] - CHEETAH_RATIO_AMP),
                           -PHASE_A[r["task"]]["position_dim"], r["task"]))
    return ok[0]


def render(rows, chosen):
    L = []
    L.append("# A3 CALIBRATION PILOT — read\n")
    L.append(f"Cheetah anchor: amplitude ratio {CHEETAH_RATIO_AMP:.6f}, "
             f"flat scale {CHEETAH_MBAR}\n")
    for r in rows:
        c = r["calib"]
        L.append(f"\n## {r['task']}  ({r['run_id']})\n")
        L.append(f"- verdict: **{'VALID' if r['valid'] else 'INVALID'}**")
        for f in r["fails"]:
            L.append(f"  - FAIL {f}")
        L.append(f"- replay steps: {c['n_steps']}")
        L.append(f"- position[0] range: [{c['mod_lo']!r}, {c['mod_hi']!r}]")
        L.append(f"- median (gate threshold): {c['gate_threshold']!r}")
        L.append(f"- flat scale (visitation-weighted mean m): {c['flat_scale']!r}")
        L.append(f"- split visitation: below {c['w_below']:.4f} / "
                 f"above {c['w_above']:.4f}")
        L.append(f"- mean m: below {c['mean_m_below']:.4f} / "
                 f"above {c['mean_m_above']:.4f}  "
                 f"=> amplitude ratio {c['ratio_amp']:.4f}x")
        L.append(f"- E[m^2]: below {c['meansq_m_below']:.4f} / "
                 f"above {c['meansq_m_above']:.4f}  "
                 f"=> variance ratio {c['ratio_var']:.4f}x")
    L.append("\n## SELECTION\n")
    if chosen is None:
        L.append("**NO VALID CANDIDATE** — A3 generality leg cannot launch.")
    else:
        t = chosen["task"]
        c = chosen["calib"]
        pa = PHASE_A[t]
        L.append(f"**{t}** (|ratio − cheetah| = "
                 f"{abs(c['ratio_amp'] - CHEETAH_RATIO_AMP):.4f})\n")
        L.append("A3-main pins:\n")
        L.append("```")
        L.append(f"TASK={t}")
        L.append(f"SOURCE_KEY={SOURCE_KEY}")
        L.append(f"BASESD_PLANTED={pa['basesd_planted']}")
        L.append(f"BASESD_N={pa['basesd_n']}")
        L.append(f"DISTRACTOR_MOD_KEY={SOURCE_KEY}")
        L.append(f"DISTRACTOR_MOD_INDEX={MOD_INDEX}")
        L.append(f"DISTRACTOR_MOD_LO={c['mod_lo']!r}")
        L.append(f"DISTRACTOR_MOD_HI={c['mod_hi']!r}")
        L.append(f"GATE_THRESHOLD={c['gate_threshold']!r}")
        L.append(f"FLAT_SCALE={c['flat_scale']!r}")
        L.append("```")
    return "\n".join(L) + "\n"


def run(argv=None):
    a = parse_args(argv)
    dirs = sorted(d for d in globmod.glob(a.pilots) if os.path.isdir(d))
    assert dirs, f"no pilot run dirs matched {a.pilots!r}"
    rows = []
    for d in dirs:
        cfg = read_config(d)
        task = check_pins(d, cfg)
        c = calibrate(scan_replay(d))
        ok, fails = gate(c, a.expect_steps)
        rows.append({"run_id": os.path.basename(d), "task": task,
                     "calib": c, "valid": ok, "fails": fails})
    chosen = select(rows)
    text = render(rows, chosen)
    print(text)
    if a.output:
        os.makedirs(a.output, exist_ok=True)
        with open(os.path.join(a.output, "RESULTS.md"), "w") as f:
            f.write(text)
        with open(os.path.join(a.output, "constants.json"), "w") as f:
            json.dump({"rows": rows,
                       "chosen": chosen["task"] if chosen else None,
                       "phase_a": PHASE_A}, f, indent=2, sort_keys=True)
    return 0 if chosen else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pilots", default="")
    p.add_argument("--output", default="")
    p.add_argument("--expect_steps", type=float, default=EXPECT_STEPS)
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


# ------------------------------------------------------------ selfcheck
def _fixture(root, run_id, task, p0, dim=None, distractor_dim=0,
             basesd=None, chunk=5000):
    """Synthetic Phase-B run dir: config.yaml + replay chunks."""
    import yaml
    pa = PHASE_A[task]
    dim = dim or pa["position_dim"]
    d = os.path.join(root, run_id)
    os.makedirs(os.path.join(d, "replay"), exist_ok=True)
    cfg = {
        "task": task, "seed": 0,
        "agent": {"expl": {"mode": "p2e", "disag_bootstrap": True,
                           "disag_ens": 8, "disag_scale": 1000.0}},
        "distractor": {"dim": distractor_dim, "basesd": pa["basesd_n"],
                       "scale": 1.0, "gate_key": "", "mod_key": ""},
        "planted": {"source_key": SOURCE_KEY,
                    "basesd": pa["basesd_planted"] if basesd is None else basesd,
                    "gate_key": ""},
        "run": {"steps": int(EXPECT_STEPS)},
    }
    with open(os.path.join(d, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    n = p0.size
    for ci, s in enumerate(range(0, n, chunk)):
        seg = p0[s:s + chunk]
        arr = np.zeros((seg.size, dim), np.float64)
        arr[:, MOD_INDEX] = seg
        np.savez(os.path.join(d, "replay", f"chunk{ci:04d}.npz"),
                 position=arr, is_first=np.zeros(seg.size, bool))
    return d


def selfcheck():
    import shutil
    import tempfile
    root = tempfile.mkdtemp(prefix="a3cal_sc_")
    try:
        rng = np.random.default_rng(0)
        N = 20000

        # A) uniform visitation -> the analytic 0.25/0.75 split
        u = rng.uniform(-1.0, 1.0, N)
        c = calibrate(u)
        ok, fails = gate(c)
        assert ok, fails
        assert abs(c["flat_scale"] - 0.5) < 0.02, c["flat_scale"]
        assert abs(c["mean_m_below"] - 0.25) < 0.02, c
        assert abs(c["mean_m_above"] - 0.75) < 0.02, c
        assert abs(c["ratio_amp"] - 3.0) < 0.15, c["ratio_amp"]
        print("  A uniform: mbar %.4f, split %.3f/%.3f, ratio %.3fx PASS"
              % (c["flat_scale"], c["mean_m_below"], c["mean_m_above"],
                 c["ratio_amp"]))

        # B) exact recovery of a constructed range and median
        lo, hi, med = -0.32203162, 0.19711795, -0.13009691
        # odd length so the median is an EXACT sample, with both
        # endpoints present: lo | 9999x(med-e) | med | 9999x(med+e) | hi
        body = np.concatenate([[lo], np.full(9999, med - 1e-3), [med],
                               np.full(9999, med + 1e-3), [hi]])
        c = calibrate(body)
        assert c["mod_lo"] == lo, c["mod_lo"]
        assert c["mod_hi"] == hi, c["mod_hi"]
        assert abs(c["gate_threshold"] - med) < 1e-9, c["gate_threshold"]
        print("  B exact recovery of lo/hi/median PASS")

        # C) the decomposition identity G5 holds on arbitrary data
        for _ in range(5):
            c = calibrate(rng.normal(0, 1, N))
            recon = (c["w_below"] * c["mean_m_below"]
                     + c["w_above"] * c["mean_m_above"])
            assert abs(recon - c["flat_scale"]) < 1e-9, (recon, c["flat_scale"])
        print("  C decomposition identity exact on 5 random traces PASS")

        # D) NEGATIVE: degenerate (constant) visitation must FAIL G2
        c = calibrate(np.full(N, 0.3))
        ok, fails = gate(c)
        assert not ok and any(f.startswith("G2") for f in fails), fails
        print("  D degenerate spread -> G2 FAIL (correct)")

        # E) NEGATIVE: mass point at the median must FAIL G3
        mp = np.concatenate([np.full(14000, 0.5),
                             rng.uniform(0.0, 0.49, 3000),
                             rng.uniform(0.51, 1.0, 3000)])
        c = calibrate(mp)
        ok, fails = gate(c)
        assert not ok and any(f.startswith("G3") for f in fails), fails
        print("  E mass point at median (w_above %.3f) -> G3 FAIL (correct)"
              % c["w_above"])

        # F) NEGATIVE: a short replay must FAIL G1
        c = calibrate(rng.uniform(-1, 1, 1000))
        ok, fails = gate(c)
        assert not ok and any(f.startswith("G1") for f in fails), fails
        print("  F short replay -> G1 FAIL (correct)")

        # G) NEGATIVE: the pilot must reject a run that had the OU
        #    channel ON -- that is the circularity this design forbids
        d = _fixture(root, "se_a3cal_bad", "dmc_hopper_hop",
                     rng.uniform(-1, 1, N), distractor_dim=8)
        try:
            check_pins(d, read_config(d))
            raise SystemExit("G FAILED: distractor.dim 8 was accepted")
        except AssertionError as e:
            assert "non-circularity" in str(e), str(e)
        print("  G distractor.dim=8 -> pin assertion FIRES (correct)")

        # H) NEGATIVE: a wrong Phase-A basesd must be rejected
        d = _fixture(root, "se_a3cal_bad2", "dmc_finger_spin",
                     rng.uniform(-1, 1, N), basesd=0.0976)
        try:
            check_pins(d, read_config(d))
            raise SystemExit("H FAILED: cheetah basesd was accepted")
        except AssertionError as e:
            assert "planted.basesd" in str(e), str(e)
        print("  H cheetah basesd on a finger run -> pin assertion FIRES")

        # I) end-to-end selection: uniform (ratio ~3.0) vs central
        #    (ratio ~1.5-2.0). The rule must pick the one nearer 1.8575.
        _fixture(root, "se_a3cal_hopper_s200", "dmc_hopper_hop",
                 rng.uniform(-1.0, 1.0, N))
        _fixture(root, "se_a3cal_finger_s201", "dmc_finger_spin",
                 np.clip(rng.normal(0.0, 0.32, N), -1.0, 1.0))
        rows = []
        for dd in sorted(globmod.glob(os.path.join(root, "se_a3cal_*_s2*"))):
            cfg = read_config(dd)
            task = check_pins(dd, cfg)
            cc = calibrate(scan_replay(dd))
            okk, ff = gate(cc)
            rows.append({"run_id": os.path.basename(dd), "task": task,
                         "calib": cc, "valid": okk, "fails": ff})
        r_hop = [r for r in rows if r["task"] == "dmc_hopper_hop"][0]
        r_fin = [r for r in rows if r["task"] == "dmc_finger_spin"][0]
        assert r_hop["valid"] and r_fin["valid"], rows
        assert r_hop["calib"]["ratio_amp"] > 2.5, r_hop["calib"]["ratio_amp"]
        assert r_fin["calib"]["ratio_amp"] < 2.2, r_fin["calib"]["ratio_amp"]
        chosen = select(rows)
        assert chosen["task"] == "dmc_finger_spin", chosen["task"]
        print("  I selection: hopper %.3fx vs finger %.3fx -> chose %s (correct)"
              % (r_hop["calib"]["ratio_amp"], r_fin["calib"]["ratio_amp"],
                 chosen["task"]))
        assert "DISTRACTOR_MOD_LO" in render(rows, chosen)

        # J) NEGATIVE: no valid candidate -> selection returns None
        for r in rows:
            r["valid"] = False
        assert select(rows) is None
        assert "NO VALID CANDIDATE" in render(rows, None)
        print("  J all-invalid -> NO VALID CANDIDATE (correct)")

        print("\na3cal selfcheck PASS")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    _a = parse_args()
    sys.exit(selfcheck() if _a.selfcheck else run())
