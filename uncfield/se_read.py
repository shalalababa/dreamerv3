"""SE-wave FROZEN cross-seed reader (PREREG_nfi_scale_exhibit_20260812.md
§5; built + selfchecked BEFORE THE READ — review #21 B4 — and executes
the registered read; per-run se_probe outputs are its only statistical
inputs).

Registered fire rules (v2.2):
  P-SE1 (PRIMARY): a fire-eligible channel ({dup0, dup1, dup2,
    distractor}) FIRES if its per-run share exceeds its per-run
    permutation-null MEDIAN in >= 7 of the 8 registered runs (exact
    binomial p = 9/256 = .0352 under the per-run coin null), with
    BH q = .05 over the 4 fire-eligible channels. The per-run null is
    recomputed EXACTLY from the stored per-dim attribution
    (se_probe_dims.npz) with the identical share statistic, permutation
    loop, and crc32-keyed rng as se_probe — the recomputed p must EQUAL
    the stored per-run p (provenance gate: json and npz belong
    together, n_perm matches).
  P-SE2 (PRIMARY): FIRES if per-run top-k fire-share > all-rollout mean
    in >= 7 of 8 runs (binomial p = .0352). Per-family (D-family vs N)
    breakdown reported (Amendment A1 fields, when present).

Validity gates (registered): per-run realized S >= 256; calibration
< 0.5 on every REAL key. A run failing either is INVALID and counted
as a FIRE FAILURE for every rule (conservative — the denominator stays
the 8 registered runs) and the read is flagged. Fit counters verified
per standing rule (config run.steps vs last metrics.jsonl step),
report-only.

Supporting (reporting-only, never decisions): Fisher-combined per-run
p's; seed-level BCa intervals on per-run shares; pooled fire+real
share (DESCRIPTIVE — near-powerless by construction); D-ladder
discriminator diagnostics (D0 level + share-vs-eps^2 pattern, §2);
se_mask summaries when present (decision role = outcome cell 6 only).

Run:  python -m uncfield.se_read --runs "run1,run2,..." --output <dir>
      python -m uncfield.se_read --selfcheck
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import os
import zlib

import numpy as np
from scipy.stats import chi2 as _chi2

from uncfield.se_probe import (FIRE_KEYS, PLANTED_KEYS, REPO,
                               resolve_ckpt, share_stats)
from uncfield.se_mask import bca_interval

N_REGISTERED = 8          # seeds 10-17, prereg §1 (frozen denominator)
FIRE_MIN = 7              # >= 7 of 8, prereg §5
BH_Q = 0.05
CAL_MAX = 0.5             # calibration acceptance, prereg §5
S_MIN = 256               # validity floor, prereg §5
EPS_LADDER = (0.0, 0.05, 0.5)   # dup0/1/2, prereg §2


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="",
                   help="comma-separated run logdirs, or a glob pattern")
    p.add_argument("--probe_subdir", default="se_probe")
    p.add_argument("--mask_subdir", default="se_mask")
    p.add_argument("--n_perm", type=int, default=1000,
                   help="MUST match the se_probe execution (provenance "
                        "gate fails loudly on mismatch)")
    p.add_argument("--expect_steps", type=float, default=5e5)
    p.add_argument("--output", default="")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------- statistics

def binom_tail(k, n):
    """Exact P(X >= k), X ~ Bin(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def bh_reject(pvals, q=BH_Q):
    """Benjamini-Hochberg step-up. Returns boolean rejection array."""
    p = np.asarray(pvals, np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    thresh_rank = 0
    for rank, idx in enumerate(order, 1):
        if p[idx] <= rank / m * q:
            thresh_rank = rank
    rej = np.zeros(m, bool)
    for rank, idx in enumerate(order, 1):
        if rank <= thresh_rank:
            rej[idx] = True
    return rej


def fisher_p(pvals):
    pvals = [max(float(p), 1e-300) for p in pvals]
    stat = -2.0 * sum(math.log(p) for p in pvals)
    return float(_chi2.sf(stat, 2 * len(pvals)))


def channel_null(dims_d, dim_key, channel, real_keys, n_perm, rng):
    """EXACT replica of se_probe's per-channel permutation (same share
    statistic, same in-place shuffle loop, same rng consumption).
    Returns (obs share, p, null median)."""
    mask = np.asarray([dk == channel or dk in real_keys for dk in dim_key])
    sub_d = dims_d[mask]
    lab = np.asarray([dk == channel for dk in dim_key[mask]])
    obs = share_stats(sub_d, lab)
    null = np.empty(n_perm)
    lab2 = lab.copy()
    for i in range(n_perm):
        rng.shuffle(lab2)
        null[i] = share_stats(sub_d, lab2)
    p = (1 + (null >= obs).sum()) / (1 + n_perm)
    return float(obs), float(p), float(np.median(null))


# ------------------------------------------------------------------ per-run

def read_run(run_dir, probe_subdir, n_perm):
    """Load one run's se_probe outputs; recompute per-channel null
    medians with the provenance gate; return the per-run record."""
    pdir = os.path.join(run_dir, probe_subdir)
    with open(os.path.join(pdir, "se_probe.json")) as f:
        pj = json.load(f)
    npz = np.load(os.path.join(pdir, "se_probe_dims.npz"))
    # keep the STORED dtype: se_probe ran its permutation on float32
    # arrays, and the provenance gate needs bit-identical arithmetic
    dims_d = np.asarray(npz["dims_d"])
    dim_key = np.asarray(npz["dim_key"]).astype(str)
    seed = int(pj["seed"])
    all_keys = sorted(set(dim_key.tolist()))
    real_keys = [k for k in all_keys if k not in PLANTED_KEYS]

    # FINAL-CHECKPOINT GATE (review B1): an M4 snapshot probe pass with
    # the default --output would clobber <run>/se_probe, and json+npz
    # clobber together so the provenance gate alone cannot catch it.
    # If the run's ckpt dir is reachable, the probed checkpoint MUST be
    # the final one; unreachable -> flagged UNVERIFIED, never silent.
    try:
        final = resolve_ckpt(run_dir, "")
        ckpt_final_ok = (os.path.basename(os.path.normpath(final))
                         == os.path.basename(os.path.normpath(pj["ckpt"])))
    except (AssertionError, FileNotFoundError, OSError):
        final, ckpt_final_ok = None, None
    assert ckpt_final_ok is not False, (
        f"{run_dir}: se_probe output is from checkpoint "
        f"{os.path.basename(os.path.normpath(pj['ckpt']))} but the run's "
        f"final checkpoint is {os.path.basename(os.path.normpath(final))} "
        f"— snapshot pass clobbered the primary output? Re-run se_probe "
        f"on the final checkpoint (M4 passes must use --output "
        f"<run>/se_probe_snap<pct>)")

    # run identity (review M4): the probe seed is 0 for every run; the
    # TRAINING seed lives in config.yaml (top-level `seed`)
    train_seed = None
    try:
        import yaml
        with open(os.path.join(run_dir, "config.yaml")) as f:
            train_seed = int(yaml.safe_load(f)["seed"])
    except Exception:
        pass

    rec = dict(run_dir=os.path.abspath(run_dir), probe_seed=seed,
               train_seed=train_seed, ckpt_final_ok=ckpt_final_ok,
               n_eval=int(pj["n_eval"]), channels={},
               key_share=pj["key_share"],
               planted_share_total=float(pj["planted_share_total"]),
               calibration=pj["calibration"], pse2=pj["pse2"])
    for ch in FIRE_KEYS:
        if ch not in all_keys:
            raise AssertionError(f"{run_dir}: channel {ch} missing")
        rng = np.random.default_rng(seed + zlib.crc32(ch.encode()) % 2 ** 16)
        obs, p, med = channel_null(dims_d, dim_key, ch, real_keys,
                                   n_perm, rng)
        stored = float(pj["channels"][ch]["p_perm"])
        assert p == stored, (
            f"{run_dir}/{ch}: recomputed p {p} != stored {stored} — "
            f"provenance/n_perm mismatch (json+npz pair, --n_perm)")
        rec["channels"][ch] = dict(
            share_vs_real=obs, p_perm=p, null_median=med,
            indicator=bool(obs > med),
            vs_source=float(pj["channels"][ch]["vs_source"]))  # theta_1, §5
    # weak-gate note (review m9): p==1.0 on every channel makes the
    # equality gate insensitive to an --n_perm mismatch
    rec["gate_all_p_one"] = all(
        rec["channels"][ch]["p_perm"] == 1.0 for ch in FIRE_KEYS)
    # validity gates (registered)
    cal_real = {k: v for k, v in pj["calibration"].items()
                if k not in PLANTED_KEYS}
    rec["cal_real_max"] = max(cal_real.values())
    rec["cal_planted"] = {k: v for k, v in pj["calibration"].items()
                          if k in PLANTED_KEYS}
    # const vs its RAW (unfloored) normalizer (registered, §5): the
    # stored cal is floored; recover raw = stored * floored/raw means
    rec["cal_const_raw"] = None
    if "raw_norms" in npz and "planted_const" in pj["calibration"]:
        raw_c = np.asarray(npz["raw_norms"])[dim_key == "planted_const"]
        floor = float(npz["norm_floor"])
        raw_mean = float(raw_c.mean())
        rec["cal_const_raw_norm_mean"] = raw_mean
        if raw_mean > 0:
            floored_mean = float(np.maximum(raw_c, floor).mean())
            rec["cal_const_raw"] = float(
                pj["calibration"]["planted_const"] * floored_mean / raw_mean)
        # raw_mean == 0 -> exactly-constant channel: raw-normalized cal
        # undefined; the None + the 0.0 raw mean IS the registered report
    rec["cal_ok"] = bool(rec["cal_real_max"] < CAL_MAX)
    rec["s_ok"] = bool(rec["n_eval"] >= S_MIN)
    rec["valid"] = bool(rec["cal_ok"] and rec["s_ok"])
    rec["pse2_indicator"] = bool(
        pj["pse2"]["top_share"] > pj["pse2"]["base_share"])
    return rec


def fit_counters(run_dir, expect_steps):
    """Standing-rule fit-counter check (report-only)."""
    out = dict(expected=float(expect_steps), last_step=None, ratio=None,
               flag="NO-METRICS")
    mpath = os.path.join(run_dir, "metrics.jsonl")
    try:
        with open(mpath, "rb") as f:
            lines = f.read().splitlines()
        last = json.loads(lines[-1])
        out["last_step"] = int(last["step"])
        out["ratio"] = out["last_step"] / float(expect_steps)
        out["flag"] = ("OK" if out["ratio"] >= 0.95 else
                       "TRUNCATION-SUSPECT")
    except (OSError, IndexError, KeyError, ValueError):
        pass
    try:
        import yaml
        with open(os.path.join(run_dir, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        out["config_steps"] = float(cfg["run"]["steps"])
        if out["config_steps"] != float(expect_steps):
            out["flag"] += "+CONFIG-STEPS-MISMATCH"
    except Exception:
        out["config_steps"] = None
        out["flag"] += "+NO-CONFIG"          # review m15: never bare OK
    return out


# ---------------------------------------------------------------- aggregate

def aggregate(recs, expect_steps=5e5, mask_summaries=None, n_perm=None):
    """The registered cross-seed read. recs: per-run records (order =
    registered seed order). Invalid runs count as fire FAILURES; the
    denominator stays N_REGISTERED."""
    assert len(recs) <= N_REGISTERED, "more runs than registered"
    n_missing = N_REGISTERED - len(recs)
    invalid = [r["run_dir"] for r in recs if not r["valid"]]

    # run identity (review M4): duplicate training seeds = the same run
    # counted twice = statistical corruption -> fatal; unknown or
    # out-of-family seeds -> flagged
    tseeds = [r["train_seed"] for r in recs]
    known = [s for s in tseeds if s is not None]
    assert len(known) == len(set(known)), (
        f"duplicate training seeds {sorted(known)} — same run loaded "
        f"twice?")
    if None in tseeds:
        seed_check = "UNKNOWN-SEEDS (config.yaml unreadable on some runs)"
    elif not all(10 <= s <= 17 for s in known):
        seed_check = f"SEED-OUT-OF-FAMILY: {sorted(known)} != [10..17]"
    else:
        seed_check = "OK"

    out = dict(n_registered=N_REGISTERED, n_loaded=len(recs),
               n_missing=n_missing, invalid_runs=invalid, n_perm=n_perm,
               seed_check=seed_check,
               ckpt_flag=("OK" if all(r["ckpt_final_ok"] for r in recs)
                          else "UNVERIFIED (ckpt dir unreachable on some "
                               "runs)"),
               weak_gate_runs=[r["run_dir"] for r in recs
                               if r["gate_all_p_one"]],
               instrument_flag=("CLEAN" if not invalid and not n_missing
                                else "VALIDITY-VIOLATIONS"))

    # P-SE1: cross-seed fire rule + BH over the 4 fire-eligible channels
    se1 = {}
    pvals = []
    for ch in FIRE_KEYS:
        succ = sum(1 for r in recs
                   if r["valid"] and r["channels"][ch]["indicator"])
        p = binom_tail(succ, N_REGISTERED)
        per_run_p = [r["channels"][ch]["p_perm"] for r in recs
                     if r["valid"]]
        # BCa on the REGISTERED statistic (channel-vs-real share) —
        # review m8; theta_1 (vs_source, §5) aggregated alongside
        shares = [r["channels"][ch]["share_vs_real"] for r in recs
                  if r["valid"]]
        vs_src = [r["channels"][ch]["vs_source"] for r in recs
                  if r["valid"]]
        rec = dict(successes=succ, binom_p=p,
                   fisher_p=(fisher_p(per_run_p) if per_run_p else None),
                   per_run_p=per_run_p)
        if len(shares) >= 3:
            m, lo, hi = bca_interval(np.asarray(shares),
                                     np.random.default_rng(1234), 2000)
            rec["share_mean"] = m
            rec["share_bca"] = [lo, hi]
            m2, lo2, hi2 = bca_interval(np.asarray(vs_src),
                                        np.random.default_rng(1235), 2000)
            rec["theta1_vs_source_mean"] = m2
            rec["theta1_vs_source_bca"] = [lo2, hi2]
        se1[ch] = rec
        pvals.append(p)
    rej = bh_reject(pvals, BH_Q)
    for ch, r_ in zip(FIRE_KEYS, rej):
        se1[ch]["fires"] = bool(r_ and se1[ch]["successes"] >= FIRE_MIN)
    out["p_se1"] = se1
    se1_fires = any(se1[ch]["fires"] for ch in FIRE_KEYS)

    # P-SE2
    succ2 = sum(1 for r in recs if r["valid"] and r["pse2_indicator"])
    deltas = [r["pse2"]["top_share"] - r["pse2"]["base_share"]
              for r in recs if r["valid"]]
    se2 = dict(successes=succ2, binom_p=binom_tail(succ2, N_REGISTERED),
               fires=bool(succ2 >= FIRE_MIN),
               per_run_p=[r["pse2"]["p_rank"] for r in recs if r["valid"]])
    if len(deltas) >= 3:
        m, lo, hi = bca_interval(np.asarray(deltas),
                                 np.random.default_rng(4321), 2000)
        se2["delta_mean"] = m
        se2["delta_bca"] = [lo, hi]
    fams = [r["pse2"].get("families") for r in recs if r["valid"]]
    if all(f is not None for f in fams) and fams:
        se2["families"] = {
            fam: dict(
                top_mean=float(np.mean([f[fam]["top_share"] for f in fams])),
                base_mean=float(np.mean([f[fam]["base_share"] for f in fams])))
            for fam in ("dfam", "noise")}
    else:
        se2["families"] = "UNAVAILABLE (pre-Amendment-A1 probe outputs)"
    out["p_se2"] = se2

    # descriptive pooled share (registered: reporting only)
    pooled = [r["planted_share_total"] for r in recs if r["valid"]]
    if len(pooled) >= 3:
        m, lo, hi = bca_interval(np.asarray(pooled),
                                 np.random.default_rng(555), 2000)
        out["pooled_fire_share"] = dict(mean=m, bca=[lo, hi],
                                        note="DESCRIPTIVE only (prereg §5)")

    # D-ladder discriminator diagnostics (§2): noisy-TV predicts
    # share ~ eps^2 with D0 at floor; redundancy predicts D0 fires
    # and/or the ladder NOT ~ eps^2. Reported; adjudicated in the read.
    dup_sh = {i: [r["key_share"][f"planted_dup{i}"] for r in recs
                  if r["valid"]] for i in range(3)}
    if all(dup_sh[i] for i in range(3)):
        m0, m1, m2 = (float(np.mean(dup_sh[i])) for i in range(3))
        ladder = dict(dup0_mean=m0, dup1_mean=m1, dup2_mean=m2,
                      dup0_fires=se1["planted_dup0"]["fires"],
                      excess_ratio_d2_over_d1=(
                          (m2 - m0) / (m1 - m0) if abs(m1 - m0) > 1e-12
                          else None),
                      eps2_predicted_ratio=(EPS_LADDER[2] / EPS_LADDER[1])
                      ** 2)
        out["d_ladder"] = ladder
        d_discriminator = bool(ladder["dup0_fires"])
    else:
        d_discriminator = False

    # outcome-map cell suggestion (§6; D-discriminator strong form =
    # D0 fires; the ladder-pattern arm is adjudicated in the read text)
    n_fires = se1["distractor"]["fires"]
    d_fires = any(se1[f"planted_dup{i}"]["fires"] for i in range(3))
    se2_fires = se2["fires"]
    if se1_fires and se2_fires and d_fires and d_discriminator:
        cell = 1
    elif se1_fires and se2_fires and n_fires and not d_fires:
        cell = 2
    elif se1_fires and not se2_fires:
        cell = 3
    elif se2_fires and not se1_fires:
        cell = 4
    elif not se1_fires and not se2_fires:
        cell = 5
    else:
        cell = "1-vs-2 boundary: SE1+SE2 fire, D-discriminator " \
               "ambiguous — adjudicate ladder pattern in the read"
    out["outcome_cell_suggestion"] = cell

    # fit counters + mask summaries (report-only)
    out["fit_counters"] = {r["run_dir"]: fit_counters(r["run_dir"],
                                                      expect_steps)
                           for r in recs}
    if mask_summaries:
        out["se_mask"] = mask_summaries
    return out


def run(args):
    if "," in args.runs:
        run_dirs = [d.strip() for d in args.runs.split(",") if d.strip()]
    else:
        run_dirs = sorted(globmod.glob(args.runs))
    assert run_dirs, f"no runs matched {args.runs!r}"
    recs = [read_run(d, args.probe_subdir, args.n_perm) for d in run_dirs]
    masks = {}
    for d in run_dirs:
        mp = os.path.join(d, args.mask_subdir, "se_mask.json")
        if os.path.exists(mp):
            with open(mp) as f:
                mj = json.load(f)
            masks[os.path.abspath(d)] = {
                ch: {f: rec[f] for f in ("delta_intrinsic_mean",
                                         "p_reduce_intrinsic",
                                         "p_inflate_intrinsic")}
                for ch, rec in mj["channels"].items()}
    out = aggregate(recs, args.expect_steps, masks or None,
                    n_perm=args.n_perm)
    out["per_run"] = [
        {f: r[f] for f in ("run_dir", "probe_seed", "train_seed",
                           "ckpt_final_ok", "n_eval", "valid", "cal_ok",
                           "s_ok", "cal_real_max", "cal_planted",
                           "cal_const_raw", "pse2_indicator")}
        | {"indicators": {ch: r["channels"][ch]["indicator"]
                          for ch in FIRE_KEYS}}
        for r in recs]
    # review m11: the registered read must persist its result
    assert args.output, "registered read must persist: pass --output"
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "se_read.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("instrument_flag", "outcome_cell_suggestion")},
                     indent=1))
    for ch in FIRE_KEYS:
        s = out["p_se1"][ch]
        print(f"P-SE1 {ch}: {s['successes']}/8 binom p={s['binom_p']:.4f} "
              f"fires={s['fires']}")
    s2 = out["p_se2"]
    print(f"P-SE2: {s2['successes']}/8 binom p={s2['binom_p']:.4f} "
          f"fires={s2['fires']}")
    return out


# ---------------------------------------------------------------- selfcheck

KEY_DIMS = (("position", 8), ("velocity", 9), ("planted_dup0", 8),
            ("planted_dup1", 8), ("planted_dup2", 8), ("planted_const", 4),
            ("distractor", 8))
FIX_PERM = 200


def _fixture_run(tmp, name, seed, hot, n_eval=512, cal=0.1, pse2=(0.5, 0.3),
                 steps_done=499968):
    """Write one synthetic run dir (se_probe outputs + counters).
    hot: {channel: level} per-dim attribution level (real dims ~1.0).
    seed = the TRAINING seed (config.yaml); the probe seed is 0 as in
    real runs. The stored p_perm is produced by se_probe's OWN perm_null
    (cross-implementation, review M6), so the provenance gate tests the
    reader's channel_null against the frozen probe implementation."""
    rd = os.path.join(tmp, name)
    pdir = os.path.join(rd, "se_probe")
    os.makedirs(pdir, exist_ok=True)
    rng = np.random.default_rng(1000 + seed)
    dims_d, dim_key = [], []
    for k, dim in KEY_DIMS:
        level = hot.get(k, None)
        base = (np.full(dim, level) if level is not None
                else 1.0 + rng.normal(0, 0.01, dim))
        dims_d.append(np.abs(base + rng.normal(0, 0.005, dim)))
        dim_key.extend([k] * dim)
    dims_d = np.concatenate(dims_d)
    dim_key = np.asarray(dim_key)
    dims_label = np.asarray([k in FIRE_KEYS for k in dim_key])
    np.savez(os.path.join(pdir, "se_probe_dims.npz"),
             dims_d=dims_d, dims_label=dims_label, dim_key=dim_key)
    real_keys = [k for k, _ in KEY_DIMS if k not in PLANTED_KEYS]
    # CROSS-IMPLEMENTATION provenance (review M6): the stored p comes
    # from se_probe's own perm_null via se_probe's exact per-channel
    # call pattern, so the reader's channel_null is genuinely tested
    # against the frozen implementation, not against itself
    from uncfield.se_probe import perm_null as probe_perm_null
    channels = {}
    for ch in ("planted_dup0", "planted_dup1", "planted_dup2",
               "planted_const", "distractor"):
        mask = np.asarray([dk == ch or dk in real_keys for dk in dim_key])
        sub_d = dims_d[mask]
        sub_lab = np.asarray([dk == ch for dk in dim_key[mask]])
        _, p, _ = probe_perm_null(
            sub_d, sub_lab, FIX_PERM,
            np.random.default_rng(0 + zlib.crc32(ch.encode()) % 2 ** 16))
        channels[ch] = dict(p_perm=p)      # probe seed = 0, like real runs
    tot = dims_d.sum()
    key_share = {k: float(dims_d[dim_key == k].sum() / tot)
                 for k, _ in KEY_DIMS}
    for ch in channels:
        channels[ch]["vs_source"] = key_share[ch] / max(
            key_share["position"], 1e-12)
    # ckpt layout for the final-checkpoint gate (review B1)
    ck_name = "20260812T000000F000000"
    os.makedirs(os.path.join(rd, "ckpt", ck_name), exist_ok=True)
    with open(os.path.join(rd, "ckpt", "latest"), "w") as f:
        f.write(ck_name)
    pj = dict(seed=0, n_eval=n_eval, channels=channels,
              ckpt=os.path.join(rd, "ckpt", ck_name),
              key_share=key_share,
              planted_share_total=float(
                  dims_d[dims_label].sum()
                  / dims_d[dim_key != "planted_const"].sum()),
              calibration={k: (cal if k not in PLANTED_KEYS else 0.2)
                           for k, _ in KEY_DIMS},
              pse2=dict(top_share=pse2[0], base_share=pse2[1],
                        p_rank=0.05 if pse2[0] > pse2[1] else 0.6,
                        families=dict(
                            dfam=dict(top_share=pse2[0] * 0.7,
                                      base_share=pse2[1] * 0.7),
                            noise=dict(top_share=pse2[0] * 0.3,
                                       base_share=pse2[1] * 0.3))))
    with open(os.path.join(pdir, "se_probe.json"), "w") as f:
        json.dump(pj, f)
    with open(os.path.join(rd, "config.yaml"), "w") as f:
        f.write(f"seed: {seed}\nrun:\n  steps: 500000.0\n")
    with open(os.path.join(rd, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": steps_done}) + "\n")
    return rd


def selfcheck():
    import tempfile
    # unit: exact binomial constants
    assert abs(binom_tail(7, 8) - 9 / 256) < 1e-15
    assert abs(binom_tail(8, 8) - 1 / 256) < 1e-15
    assert binom_tail(0, 8) == 1.0
    # unit: BH step-up
    assert bh_reject([0.001, 0.04, 0.04, 0.04]).all()          # rank-4 rescue
    assert bh_reject([0.0352, 1.0, 1.0, 1.0]).sum() == 0       # lone 7/8 dies
    assert bh_reject([0.0039, 1.0, 1.0, 1.0]).sum() == 1       # lone 8/8 lives
    # a 7/8 (.0352) at rank 2 needs <= .025 -> dies even next to an 8/8;
    # it survives only from rank 3 up (>=3 channels at p <= .0375)
    assert (bh_reject([0.0039, 0.0352, 1.0, 1.0])
            == np.array([True, False, False, False])).all()
    assert (bh_reject([0.0039, 0.0352, 0.0352, 1.0])
            == np.array([True, True, True, False])).all()
    # unit: fisher monotone
    assert fisher_p([0.01] * 8) < fisher_p([0.5] * 8)

    with tempfile.TemporaryDirectory() as tmp:
        HOT, COLD = 5.0, 0.9

        def batch(sub, spec):
            return [_fixture_run(os.path.join(tmp, sub), f"r{i}", 10 + i,
                                 **spec(i)) for i in range(8)]

        # A) strong fire: all 4 channels hot in all runs, SE2 fires 8/8
        runs = batch("A", lambda i: dict(
            hot={k: HOT for k in FIRE_KEYS}))
        recs = [read_run(d, "se_probe", FIX_PERM) for d in runs]
        agg = aggregate(recs)
        for ch in FIRE_KEYS:
            assert agg["p_se1"][ch]["successes"] == 8
            assert agg["p_se1"][ch]["fires"], ch
        assert agg["p_se2"]["fires"] and agg["p_se2"]["successes"] == 8
        assert agg["instrument_flag"] == "CLEAN"
        assert isinstance(agg["p_se2"]["families"], dict)
        assert agg["outcome_cell_suggestion"] == 1
        assert all(v["flag"] == "OK" for v in agg["fit_counters"].values())

        # B) null: all channels cold everywhere -> cell 5, nothing fires
        runs = batch("B", lambda i: dict(
            hot={k: COLD for k in FIRE_KEYS}))
        agg = aggregate([read_run(d, "se_probe", FIX_PERM) for d in runs])
        for ch in FIRE_KEYS:
            assert agg["p_se1"][ch]["successes"] == 0, ch
            assert not agg["p_se1"][ch]["fires"]
        assert agg["outcome_cell_suggestion"] in (4, 5)

        # C) boundary: lone dup0 at 7/8 -> binom .0352 but BH kills it
        runs = batch("C", lambda i: dict(
            hot={"planted_dup0": (HOT if i < 7 else COLD),
                 "planted_dup1": COLD, "planted_dup2": COLD,
                 "distractor": COLD},
            pse2=(0.3, 0.5)))
        agg = aggregate([read_run(d, "se_probe", FIX_PERM) for d in runs])
        c = agg["p_se1"]["planted_dup0"]
        assert c["successes"] == 7 and abs(c["binom_p"] - 9 / 256) < 1e-12
        assert not c["fires"], "lone 7/8 must NOT survive BH rank-1"
        assert agg["outcome_cell_suggestion"] == 5

        # D) conservative invalidity: all-hot but run 3 under the S floor
        #    -> every channel 7/8, all four .0352 -> BH rank-4 rescues all
        runs = batch("D", lambda i: dict(
            hot={k: HOT for k in FIRE_KEYS},
            n_eval=(200 if i == 3 else 512)))
        agg = aggregate([read_run(d, "se_probe", FIX_PERM) for d in runs])
        assert agg["instrument_flag"] == "VALIDITY-VIOLATIONS"
        assert len(agg["invalid_runs"]) == 1
        for ch in FIRE_KEYS:
            c = agg["p_se1"][ch]
            assert c["successes"] == 7 and c["fires"], ch
        assert agg["p_se2"]["successes"] == 7 and agg["p_se2"]["fires"]

        # E) calibration breach flags the run invalid
        runs = batch("E", lambda i: dict(
            hot={k: HOT for k in FIRE_KEYS},
            cal=(0.6 if i == 0 else 0.1)))
        recs = [read_run(d, "se_probe", FIX_PERM) for d in runs]
        assert not recs[0]["cal_ok"] and not recs[0]["valid"]
        agg = aggregate(recs)
        assert agg["instrument_flag"] == "VALIDITY-VIOLATIONS"

        # A7) MISSING run (review M5): only 7 of 8 loaded, all hot ->
        #     successes capped at 7, binomial p on the REGISTERED
        #     denominator (9/256, not 1/128), validity flagged
        runs = batch("A7", lambda i: dict(hot={k: HOT for k in FIRE_KEYS}))
        agg = aggregate([read_run(d, "se_probe", FIX_PERM)
                         for d in runs[:7]])
        assert agg["n_missing"] == 1
        assert agg["instrument_flag"] == "VALIDITY-VIOLATIONS"
        for ch in FIRE_KEYS:
            c = agg["p_se1"][ch]
            assert c["successes"] == 7, ch
            assert abs(c["binom_p"] - 9 / 256) < 1e-12, \
                "binomial denominator must stay at the 8 registered runs"
        assert agg["p_se2"]["successes"] == 7

        # I) final-checkpoint gate (review B1): probe json pointing at a
        #    non-final checkpoint must abort loudly
        rd = _fixture_run(os.path.join(tmp, "I"), "r0", 10,
                          hot={k: HOT for k in FIRE_KEYS})
        pj_path = os.path.join(rd, "se_probe", "se_probe.json")
        with open(pj_path) as f:
            pj = json.load(f)
        os.makedirs(os.path.join(rd, "ckpt", "20260812T111111F000000"))
        pj["ckpt"] = os.path.join(rd, "ckpt", "20260812T000000F000000")
        with open(os.path.join(rd, "ckpt", "latest"), "w") as f:
            f.write("20260812T111111F000000")
        with open(pj_path, "w") as f:
            json.dump(pj, f)
        try:
            read_run(rd, "se_probe", FIX_PERM)
            raise SystemExit("checkpoint gate FAILED to fire")
        except AssertionError as e:
            assert "final checkpoint" in str(e)

        # J) duplicate training seeds must be fatal (review M4)
        r1 = _fixture_run(os.path.join(tmp, "J"), "ra", 10,
                          hot={k: HOT for k in FIRE_KEYS})
        r2 = _fixture_run(os.path.join(tmp, "J"), "rb", 10,
                          hot={k: HOT for k in FIRE_KEYS})
        try:
            aggregate([read_run(d, "se_probe", FIX_PERM)
                       for d in (r1, r2)])
            raise SystemExit("duplicate-seed gate FAILED to fire")
        except AssertionError as e:
            assert "duplicate" in str(e)

        # F) provenance gate: doctored npz must fail loudly
        rd = _fixture_run(os.path.join(tmp, "F"), "r0", 10,
                          hot={k: HOT for k in FIRE_KEYS})
        npz_path = os.path.join(rd, "se_probe", "se_probe_dims.npz")
        z = dict(np.load(npz_path))
        z["dims_d"] = z["dims_d"] * 1.7
        z["dims_d"][:4] = 99.0
        np.savez(npz_path, **z)
        try:
            read_run(rd, "se_probe", FIX_PERM)
            raise SystemExit("provenance gate FAILED to fire")
        except AssertionError as e:
            assert "provenance" in str(e)

        # G) fit-counter flag on a truncated run
        rd = _fixture_run(os.path.join(tmp, "G"), "r0", 10,
                          hot={}, steps_done=250000)
        fc = fit_counters(rd, 5e5)
        assert fc["flag"].startswith("TRUNCATION-SUSPECT"), fc

        # H) end-to-end through run() with the glob path + output json
        #    + the se_mask summary consumption path (review m10)
        mdir = os.path.join(tmp, "A", "r0", "se_mask")
        os.makedirs(mdir, exist_ok=True)
        with open(os.path.join(mdir, "se_mask.json"), "w") as f:
            json.dump(dict(channels={
                "planted_dup1": dict(delta_intrinsic_mean=-1e-6,
                                     p_reduce_intrinsic=0.03,
                                     p_inflate_intrinsic=0.97)}), f)
        outd = os.path.join(tmp, "read_out")
        args = parse_args(["--runs", os.path.join(tmp, "A", "r*"),
                           "--n_perm", str(FIX_PERM), "--output", outd])
        out = run(args)
        assert out["outcome_cell_suggestion"] == 1
        assert out["seed_check"] == "OK"
        assert len(out["se_mask"]) == 1
        assert os.path.exists(os.path.join(outd, "se_read.json"))

    # real-data cross-check (guarded): the smoke probe output exercises
    # the provenance gate against a genuine se_probe artifact (the
    # fixture path is cross-implementation but synthetic)
    smoke_probe = REPO / "local_results" / "uncfield" / "se_probe_smoke"
    if (smoke_probe / "se_probe.json").exists():
        rec = read_run(str(smoke_probe.parent), "se_probe_smoke", 300)
        assert rec["valid"] is False              # smoke n_eval 48 < 256
        assert rec["cal_const_raw_norm_mean"] == 0.0 \
            and rec["cal_const_raw"] is None      # exactly-constant key
        print("  + real smoke-probe provenance cross-check PASS")

    print("se_read selfcheck PASS (binomial/BH/Fisher exact; scenarios "
          "A strong-fire cell-1, B null, C lone-7/8 killed by BH, "
          "D conservative-invalidity + BH rank-4 rescue, E calibration "
          "breach, A7 missing-run denominator, I ckpt gate, J duplicate "
          "seeds, F provenance gate, G truncation flag, H end-to-end "
          "incl. mask summaries)")


def main():
    args = parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        run(args)


if __name__ == "__main__":
    main()
