"""Frozen read for the Paper-4 collusion confirmatory sweep.

Registered in prereg/PREREG_collusion_confirm_20260730.md and committed
BEFORE any confirmatory session exists. Adjudicates the RQ2
sufficiency/necessity questions on both designs (calibration basis:
artifacts/collusion_calib_20260730/).

Design B (offline stream composition, seeds 1000-1099): per recorded
session, graded volume-controlled down-weighting of distance-labeled
punishment-phase transitions vs a size-matched random control at the
same weight (identical effective update mass by construction).
  P-CLB1 (primary): fingerprint-conditional paired increment at the
    top dose, I_dp = dp_punish_d100 - dp_random_d100; FIRES iff the
    95% bootstrap CI is entirely < 0.
  P-CLB2 (co-primary): same for the fingerprint indicator,
    I_fp = fp_punish_d100 - fp_random_d100; FIRES iff CI < 0.

Design A (online generation-time intervention, 5 arms x seeds
2000-2099 / 3000-3399): biased exploratory-action distribution at
matched exploration budget (eps schedule untouched). The calibration
smoke showed the forbid intervention RAISES the price level while
killing the punishment structure, so the registered Design-A claim is
a DISSOCIATION (re-specified pre-freeze on smoke data, disclosed):
  P-CLA1 (primary): forbid(mix=1) - baseline on the FINGERPRINT rate,
    two-sample; FIRES iff CI < 0 (punishment-generating exploration is
    necessary for punishment structure).
  P-CLA2 (co-primary): forbid(mix=1) - baseline on delta_profit;
    FIRES iff CI > 0 (price level maintained/raised without the
    structure => experimentally induced failure-to-learn regime).
  G-CLA-MANIP (gate): forbid(mix=1) must LOWER the registered
    punishment-occupancy label (CI < 0), else P-CLA verdicts are
    reported manipulation-failed (uninterpretable, not negative).

Gates: G-CLEX exact_replay must hold on every Design-B session (any
failure => instrument halt, no verdicts); G-CLBASE both baselines must
reproduce the phenomenon (mean dp in [0.70, 0.90], fingerprint rate
> 0.5), else INCONCLUSIVE-COHORT; G-CLFP fingerprint-conditional
population floor n >= 30 (redundant on the registered path - G-CLBASE
with n=100 implies n_fp >= 51 - retained as a guard for any amended
cohort; redundancy disclosed in the prereg). Gate semantics: under
G-CLEX nothing is computed (instrument halt, no estimates); under
G-CLBASE/G-CLFP failure primaries carry issued=False and fires=None
(estimates still archived); under G-CLA-MANIP failure P-CLA1/2 carry
manipulation_failed=True and fires=None. A fire flag (true/false) is
only ever emitted for an issued, interpretable claim.

Secondaries (never decisional): full dose ladder + monotonicity,
unconditional twins, delta_price twins (dpr_ columns / designA
delta_price), force and mix-0.5 arms, coverage_late shifts,
convergence rates, exogenous-label descriptive, RQ1-observational
midrank Spearman on the Design-A baseline.

Usage:
  python -m analysis.collusion_confirm_read \
      --designb '<glob of graded csvs>' \
      --designa_base '<glob>' --designa_forbid100 '<glob>' \
      --designa_forbid50 '<glob>' --designa_force100 '<glob>' \
      --designa_force50 '<glob>' --output <dir>
  python -m analysis.collusion_confirm_read --selfcheck
"""

import argparse
import csv
import glob as globmod
import json
import os

import numpy as np

B_BOOT = 10_000
RNG_SEED = 0
REFERENCE = "terminal"   # frozen: calibration artifact section 2 (the
                         # exogenous label is degenerate - mean 0.981,
                         # 96% of sessions > 0.9)
OCC_COL = "punish_occ_dist"
OCC_EXO_COL = "punish_occ_dist_exo"
DOSES = ("d025", "d050", "d100")
TOP = "d100"
SEEDS_B = tuple(range(1000, 1100))
SEEDS_A = dict(base=tuple(range(2000, 2100)),
               forbid100=tuple(range(3000, 3100)),
               forbid50=tuple(range(3100, 3200)),
               force100=tuple(range(3200, 3300)),
               force50=tuple(range(3300, 3400)))
ARM_SPEC = dict(base=("uniform", None), forbid100=("forbid", 1.0),
                forbid50=("forbid", 0.5), force100=("force", 1.0),
                force50=("force", 0.5))
BASE_DP_RANGE = (0.70, 0.90)
FP_FLOOR = 30


def _bool(v):
  """Strict two-way parse: the harness writes canonical 'True'/'False'
  via csv.DictWriter; anything else is corruption and must halt the
  read rather than silently coerce (review finding, 30 Jul)."""
  if v in (True, "True"):
    return True
  if v in (False, "False"):
    return False
  raise AssertionError(f"non-canonical boolean cell: {v!r}")


def _rows(pattern):
  rows = []
  for p in sorted(globmod.glob(pattern)):
    with open(p) as f:
      rows.extend(list(csv.DictReader(f)))
  return sorted(rows, key=lambda r: int(r["seed"]))


def load_designb(pattern):
  rows = _rows(pattern)
  seeds = [int(r["seed"]) for r in rows]
  assert seeds == list(SEEDS_B), (
      f"designB cohort mismatch: n={len(seeds)}, "
      f"missing={sorted(set(SEEDS_B) - set(seeds))[:5]}, "
      f"extra={sorted(set(seeds) - set(SEEDS_B))[:5]}")
  for r in rows:
    assert r["reference"] == REFERENCE, (
        f"seed {r['seed']}: reference={r['reference']} != {REFERENCE}")
  return rows


def load_designa(pattern, arm):
  rows = _rows(pattern)
  seeds = [int(r["seed"]) for r in rows]
  assert seeds == list(SEEDS_A[arm]), (
      f"designA[{arm}] cohort mismatch: n={len(seeds)}")
  mode, mix = ARM_SPEC[arm]
  for r in rows:
    assert r["explore_mode"] == mode, (arm, r["seed"], r["explore_mode"])
    if mix is not None:
      assert abs(float(r["explore_mix"]) - mix) < 1e-12, (arm, r["seed"])
  return rows


def _col(rows, k):
  v = np.array([float(r[k]) for r in rows])
  assert np.isfinite(v).all(), (
      f"non-finite value in column {k} - corrupt shard, read halted")
  return v


def _bcol(rows, k):
  return np.array([_bool(r[k]) for r in rows], float)


def _paired(diffs, rng):
  d = np.asarray(diffs, float)
  n = len(d)
  stats = np.empty(B_BOOT)
  for i in range(B_BOOT):
    stats[i] = d[rng.integers(0, n, n)].mean()
  lo, hi = np.percentile(stats, [2.5, 97.5])
  assert np.isfinite([lo, hi]).all()
  return dict(point=float(d.mean()), ci=[float(lo), float(hi)], n=n)


def _two_sample(a, b, rng):
  a, b = np.asarray(a, float), np.asarray(b, float)
  n, m = len(a), len(b)
  stats = np.empty(B_BOOT)
  for i in range(B_BOOT):
    stats[i] = (a[rng.integers(0, n, n)].mean()
                - b[rng.integers(0, m, m)].mean())
  lo, hi = np.percentile(stats, [2.5, 97.5])
  assert np.isfinite([lo, hi]).all()
  return dict(point=float(a.mean() - b.mean()),
              ci=[float(lo), float(hi)], n=[n, m])


def _fires_neg(res):
  return bool(res["ci"][1] < 0)


def _fires_pos(res):
  return bool(res["ci"][0] > 0)


def _rank(x):
  """Midrank (average) ranks - tie-correct Spearman (review finding)."""
  x = np.asarray(x, float)
  order = np.argsort(x, kind="stable")
  ranks = np.empty(len(x))
  i = 0
  while i < len(x):
    j = i
    while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
      j += 1
    ranks[order[i:j + 1]] = 0.5 * (i + j)
    i = j + 1
  return ranks


def _spearman(x, y):
  return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def analyze(db, da):
  rng = np.random.default_rng(RNG_SEED)
  out = dict(reference=REFERENCE, occ_col=OCC_COL)

  # ---- gates -------------------------------------------------------
  exact = [_bool(r["exact_replay"]) for r in db]
  g_ex = all(exact)
  dpB = _col(db, "dp_online")
  fpB = _bcol(db, "fp_online")
  dpA = _col(da["base"], "delta_profit")
  fpA = _bcol(da["base"], "fingerprint")
  g_baseB = (BASE_DP_RANGE[0] <= dpB.mean() <= BASE_DP_RANGE[1]
             and fpB.mean() > 0.5)
  g_baseA = (BASE_DP_RANGE[0] <= dpA.mean() <= BASE_DP_RANGE[1]
             and fpA.mean() > 0.5)
  fp_idx = fpB.astype(bool)
  n_fp = int(fp_idx.sum())
  g_fp = n_fp >= FP_FLOOR
  out["gates"] = dict(
      g_clex=bool(g_ex), n_exact_fail=int(len(exact) - sum(exact)),
      g_clbase_designB=bool(g_baseB), baseB_dp=float(dpB.mean()),
      baseB_fp=float(fpB.mean()),
      g_clbase_designA=bool(g_baseA), baseA_dp=float(dpA.mean()),
      baseA_fp=float(fpA.mean()),
      g_clfp=bool(g_fp), n_fp=n_fp)
  if not g_ex:
    out["verdict"] = ("INSTRUMENT HALT: exact-replay anchor failed on "
                      f"{out['gates']['n_exact_fail']} session(s) - no "
                      "verdicts; audit the harness before any re-read.")
    return out
  issued = g_baseB and g_baseA and g_fp

  # ---- Design B primaries ------------------------------------------
  i_dp = _col(db, f"dp_punish_{TOP}") - _col(db, f"dp_random_{TOP}")
  i_fp = _bcol(db, f"fp_punish_{TOP}") - _bcol(db, f"fp_random_{TOP}")
  p_clb1 = _paired(i_dp[fp_idx], rng)
  p_clb2 = _paired(i_fp[fp_idx], rng)
  b1 = _fires_neg(p_clb1)
  b2 = _fires_neg(p_clb2)

  # ---- Design A primaries + manipulation gate ----------------------
  occ_base = _col(da["base"], OCC_COL)
  occ_forb = _col(da["forbid100"], OCC_COL)
  manip = _two_sample(occ_forb, occ_base, rng)
  manip_ok = _fires_neg(manip)
  p_cla1 = _two_sample(_bcol(da["forbid100"], "fingerprint"), fpA, rng)
  p_cla2 = _two_sample(_col(da["forbid100"], "delta_profit"), dpA, rng)
  a1 = manip_ok and _fires_neg(p_cla1)
  a2 = manip_ok and _fires_pos(p_cla2)
  out["g_cla_manip"] = dict(manip, passed=bool(manip_ok))
  # Gated primaries: fires=None when not issued (review finding - the
  # archived record must never show a fire flag for a claim the prereg
  # says was not issued).
  out["p_clb1"] = dict(p_clb1, issued=issued,
                       fires=bool(b1) if issued else None)
  out["p_clb2"] = dict(p_clb2, issued=issued,
                       fires=bool(b2) if issued else None)
  # Manipulation failure is claim-specific: P-CLA1/2 are then
  # MANIPULATION-FAILED (uninterpretable, not negative) - fires must be
  # null, never False (second-review finding).
  a_ok = issued and manip_ok
  out["p_cla1"] = dict(p_cla1, issued=issued,
                       fires=bool(a1) if a_ok else None,
                       manipulation_failed=not manip_ok)
  out["p_cla2"] = dict(p_cla2, issued=issued,
                       fires=bool(a2) if a_ok else None,
                       manipulation_failed=not manip_ok)

  # ---- secondaries (never decisional) ------------------------------
  ladder = {}
  for tag in DOSES:
    d_dp = _col(db, f"dp_punish_{tag}") - _col(db, f"dp_random_{tag}")
    d_fp = _bcol(db, f"fp_punish_{tag}") - _bcol(db, f"fp_random_{tag}")
    ladder[tag] = dict(
        i_dp_fpcond=_paired(d_dp[fp_idx], rng),
        i_fp_fpcond=_paired(d_fp[fp_idx], rng),
        i_dp_all=_paired(d_dp, rng), i_fp_all=_paired(d_fp, rng))
  pts = [ladder[t]["i_dp_fpcond"]["point"] for t in DOSES]
  i_dpr = _col(db, f"dpr_punish_{TOP}") - _col(db, f"dpr_random_{TOP}")
  arms = {}
  for arm in ("forbid50", "force100", "force50"):
    arms[arm] = dict(
        dp=_two_sample(_col(da[arm], "delta_profit"), dpA, rng),
        fp=_two_sample(_bcol(da[arm], "fingerprint"), fpA, rng),
        dpr=_two_sample(_col(da[arm], "delta_price"),
                        _col(da["base"], "delta_price"), rng),
        occ=_two_sample(_col(da[arm], OCC_COL), occ_base, rng),
        cov_late=_two_sample(_col(da[arm], "coverage_late"),
                             _col(da["base"], "coverage_late"), rng))
  exoA = _col(da["base"], OCC_EXO_COL)
  out["secondaries"] = dict(
      ladder=ladder,
      ladder_monotone_decreasing=bool(pts[0] >= pts[1] >= pts[2]),
      designa_arms=arms,
      forbid100_cov_late=_two_sample(
          _col(da["forbid100"], "coverage_late"),
          _col(da["base"], "coverage_late"), rng),
      forbid100_dpr=_two_sample(
          _col(da["forbid100"], "delta_price"),
          _col(da["base"], "delta_price"), rng),
      i_dp_top_unconditional=_paired(i_dp, rng),
      i_fp_top_unconditional=_paired(i_fp, rng),
      i_dpr_top_fpcond=_paired(i_dpr[fp_idx], rng),
      i_dpr_top_unconditional=_paired(i_dpr, rng),
      convergence=dict(
          designB=float(_bcol(db, "converged").mean()),
          **{arm: float(_bcol(da[arm], "converged").mean())
             for arm in SEEDS_A}),
      exo_label_descriptive=dict(
          designA_base_mean=float(exoA.mean()),
          designA_base_sd=float(exoA.std(ddof=1))),
      rq1_observational=dict(
          spearman_cov_dp=_spearman(_col(da["base"], "coverage_late"), dpA),
          spearman_occ_dp=_spearman(occ_base, dpA),
          spearman_cov_fp=_spearman(_col(da["base"], "coverage_late"), fpA),
          spearman_occ_fp=_spearman(occ_base, fpA)))

  # ---- verdict -----------------------------------------------------
  flags = []
  if not g_baseB or not g_baseA:
    flags.append("INCONCLUSIVE-COHORT (baseline reproduction gate "
                 "failed - confirmatory verdicts NOT issued)")
  if not g_fp:
    flags.append(f"POPULATION FLOOR (n_fp={n_fp} < {FP_FLOOR}: "
                 "confirmatory verdicts NOT issued)")
  if flags:
    verdict = "; ".join(flags)
  else:
    parts = []
    if b1 and b2:
      parts.append(
          "P-CLB1+P-CLB2 FIRE: punishment-phase experience is a "
          "composition-specific driver of both the profit level and the "
          "punishment structure at matched update mass - RQ2 composition "
          "leg lands; RQ3 coupling wave authorized")
    elif b1:
      parts.append(
          "P-CLB1 FIRES, P-CLB2 does not: profit level tracks "
          "punishment-phase composition but the impulse-response "
          "structure survives reweighting (structure-from-elsewhere); "
          "RQ3 coupling wave still authorized")
    elif b2:
      parts.append(
          "P-CLB2 FIRES, P-CLB1 does not: punishment structure tracks "
          "composition while the profit level survives - the "
          "composition claim lands on the collusion signature itself; "
          "RQ3 coupling wave still authorized")
    else:
      parts.append(
          "Design-B composition leg NULL: at matched effective update "
          "mass, punishment-phase share does not drive either outcome - "
          "registered negative (volume, not composition)")
    if not manip_ok:
      parts.append("Design A MANIPULATION-FAILED: uninterpretable, "
                   "not negative")
    elif a1 and a2:
      parts.append(
          "P-CLA1+P-CLA2 FIRE (dissociation): removing "
          "exploration-generated undercuts collapses the punishment "
          "structure while the price level is maintained or raised - "
          "experimentally induced failure-to-learn (Abada-Lambin) "
          "regime; punishment-generating exploration is necessary for "
          "genuine collusion, not for supra-competitive prices")
    elif a1:
      parts.append(
          "P-CLA1 FIRES only: punishment structure collapses without "
          "the registered price-level rise")
    elif a2:
      parts.append(
          "P-CLA2 FIRES only: price level rises without structure "
          "collapse - outside the registered dissociation pattern; new "
          "registration required for interpretation")
    else:
      parts.append("Design A null: forbidding exploration-generated "
                   "undercuts changes neither outcome")
    verdict = " | ".join(parts)
  out["verdict"] = verdict
  return out


def read(args):
  db = load_designb(args.designb)
  da = dict(
      base=load_designa(args.designa_base, "base"),
      forbid100=load_designa(args.designa_forbid100, "forbid100"),
      forbid50=load_designa(args.designa_forbid50, "forbid50"),
      force100=load_designa(args.designa_force100, "force100"),
      force50=load_designa(args.designa_force50, "force50"))
  res = analyze(db, da)
  res["inputs"] = dict(designb=args.designb, designa_base=args.designa_base)
  os.makedirs(args.output, exist_ok=True)
  out = os.path.join(args.output, "collusion_confirm.json")
  with open(out, "w") as f:
    json.dump(res, f, indent=2)
  for k in ("p_clb1", "p_clb2", "p_cla1", "p_cla2"):
    if k in res:
      r = res[k]
      if not r["issued"]:
        status = "NOT ISSUED (gates)"
      elif r.get("manipulation_failed"):
        status = "MANIPULATION-FAILED (fires=null)"
      else:
        status = f"fires={r['fires']}"
      print(f"{k.upper()}: {r['point']:+.4f} CI=[{r['ci'][0]:+.4f},"
            f"{r['ci'][1]:+.4f}] {status}")
  print(res["verdict"])
  print(f"-> {out}")


# ---------------------------------------------------------------------
# Selfcheck: planted-positive / planted-null synthetic cohorts.
# ---------------------------------------------------------------------

def _synth_designb(rng, comp_dp=0.0, comp_fp=0.0, vol_dp=0.15,
                   exact=True, base_dp=0.83, fp_rate=0.68):
  rows = []
  for s in SEEDS_B:
    dp0 = base_dp + 0.10 * rng.standard_normal()
    fp0 = bool(rng.random() < fp_rate)
    r = dict(seed=s, reference=REFERENCE, exact_replay=exact,
             converged=True, dp_online=dp0, dpr_online=dp0,
             fp_online=fp0, mask_frac=0.5)
    for tag, d in zip(DOSES, (0.25, 0.5, 1.0)):
      noise = 0.05 * rng.standard_normal()
      r[f"dp_random_{tag}"] = dp0 - vol_dp * d + noise
      r[f"dp_punish_{tag}"] = (dp0 - vol_dp * d - comp_dp * d
                               + noise + 0.03 * rng.standard_normal())
      r[f"dpr_random_{tag}"] = r[f"dp_random_{tag}"]
      r[f"dpr_punish_{tag}"] = r[f"dp_punish_{tag}"]
      keep_r = rng.random() > 0.1 * d
      keep_p = rng.random() > (0.1 + comp_fp) * d
      r[f"fp_random_{tag}"] = bool(fp0 and keep_r)
      r[f"fp_punish_{tag}"] = bool(fp0 and keep_p)
    rows.append({k: str(v) for k, v in r.items()})
  return rows


def _synth_designa(rng, arm, dp_shift=0.0, fp_shift=0.0, occ_shift=0.0,
                   base_dp=0.83, fp_rate=0.68, occ=0.5):
  mode, mix = ARM_SPEC[arm]
  rows = []
  for s in SEEDS_A[arm]:
    dp = base_dp + dp_shift + 0.10 * rng.standard_normal()
    r = dict(seed=s, explore_mode=mode,
             explore_mix=1.0 if mix is None else mix,
             delta_profit=dp, delta_price=dp, converged=True,
             fingerprint=bool(rng.random() < fp_rate + fp_shift),
             coverage_late=0.5 + 0.15 * rng.standard_normal())
    r[OCC_COL] = np.clip(occ + occ_shift + 0.1 * rng.standard_normal(),
                         0, 1)
    r[OCC_EXO_COL] = np.clip(0.98 + 0.02 * rng.standard_normal(), 0, 1)
    rows.append({k: str(v) for k, v in r.items()})
  return rows


def _synth_all(rng, positive):
  comp_dp = 0.15 if positive else 0.0
  comp_fp = 0.30 if positive else 0.0
  db = _synth_designb(rng, comp_dp=comp_dp, comp_fp=comp_fp)
  da = dict(
      base=_synth_designa(rng, "base"),
      forbid100=_synth_designa(
          rng, "forbid100", dp_shift=0.14 if positive else 0.0,
          fp_shift=-0.60 if positive else 0.0, occ_shift=-0.25),
      forbid50=_synth_designa(rng, "forbid50", occ_shift=-0.12),
      force100=_synth_designa(rng, "force100", occ_shift=0.15),
      force50=_synth_designa(rng, "force50", occ_shift=0.08))
  return db, da


def selfcheck(args):
  rng = np.random.default_rng(7)
  # planted positive: all four primaries fire, gates pass.
  db, da = _synth_all(rng, positive=True)
  res = analyze(db, da)
  assert res["gates"]["g_clex"] and res["gates"]["g_clfp"]
  assert res["p_clb1"]["issued"] and res["p_clb1"]["fires"], res["p_clb1"]
  assert res["p_clb2"]["fires"], res["p_clb2"]
  assert res["g_cla_manip"]["passed"]
  assert res["p_cla1"]["fires"], res["p_cla1"]
  assert res["p_cla2"]["fires"], res["p_cla2"]
  assert "RQ3 coupling wave authorized" in res["verdict"]
  assert "dissociation" in res["verdict"]
  assert res["secondaries"]["convergence"]["designB"] == 1.0
  # planted null (volume effect present, composition absent; Design-A
  # manipulation still bites occupancy): nothing fires.
  db, da = _synth_all(np.random.default_rng(8), positive=False)
  res = analyze(db, da)
  assert not res["p_clb1"]["fires"] and not res["p_clb2"]["fires"]
  assert res["g_cla_manip"]["passed"]
  assert not res["p_cla1"]["fires"] and not res["p_cla2"]["fires"]
  assert "registered negative" in res["verdict"]
  assert "Design A null" in res["verdict"]
  # b1-only and b2-only branches must still authorize RQ3
  # (consequence-map parity).
  db = _synth_designb(np.random.default_rng(16), comp_dp=0.15,
                      comp_fp=0.0)
  res = analyze(db, da)
  assert res["p_clb1"]["fires"] and not res["p_clb2"]["fires"]
  assert "RQ3 coupling wave still authorized" in res["verdict"]
  db = _synth_designb(np.random.default_rng(19), comp_dp=0.0,
                      comp_fp=0.30)
  res = analyze(db, da)
  assert res["p_clb2"]["fires"] and not res["p_clb1"]["fires"]
  assert "RQ3 coupling wave still authorized" in res["verdict"]
  # instrument halt: one non-exact replay kills the read.
  db, da = _synth_all(np.random.default_rng(9), positive=True)
  db[5]["exact_replay"] = "False"
  res = analyze(db, da)
  assert not res["gates"]["g_clex"]
  assert "INSTRUMENT HALT" in res["verdict"] and "p_clb1" not in res
  # manipulation-failure: occ shift absent => Design A uninterpretable.
  db, da = _synth_all(np.random.default_rng(10), positive=True)
  da["forbid100"] = _synth_designa(
      np.random.default_rng(11), "forbid100", dp_shift=0.14,
      fp_shift=-0.60, occ_shift=0.0)
  res = analyze(db, da)
  assert not res["g_cla_manip"]["passed"]
  assert res["p_cla1"]["fires"] is None, res["p_cla1"]
  assert res["p_cla2"]["fires"] is None
  assert res["p_cla1"]["manipulation_failed"]
  assert "MANIPULATION-FAILED" in res["verdict"]
  # baseline-reproduction gate: shifted cohort => inconclusive AND the
  # archived primaries must carry fires=None (not-issued suppression).
  db, da = _synth_all(np.random.default_rng(12), positive=True)
  db = _synth_designb(np.random.default_rng(13), comp_dp=0.15,
                      comp_fp=0.30, base_dp=0.30)
  res = analyze(db, da)
  assert not res["gates"]["g_clbase_designB"]
  assert "INCONCLUSIVE-COHORT" in res["verdict"]
  assert res["p_clb1"]["fires"] is None and not res["p_clb1"]["issued"]
  assert res["p_cla1"]["fires"] is None
  # population floor: fingerprint-starved cohort trips G-CLFP (also
  # trips G-CLBASE on the registered n=100 path - redundancy disclosed).
  db = _synth_designb(np.random.default_rng(14), comp_dp=0.15,
                      comp_fp=0.30, fp_rate=0.15)
  res = analyze(db, da)
  assert not res["gates"]["g_clfp"]
  assert "POPULATION FLOOR" in res["verdict"]
  assert res["p_clb1"]["fires"] is None
  # corruption trips (review findings): nan cell and non-canonical
  # boolean must HALT the read, never adjudicate.
  db, da = _synth_all(np.random.default_rng(15), positive=True)
  db[3]["dp_punish_d100"] = "nan"
  try:
    analyze(db, da)
    raise SystemExit("selfcheck FAIL: nan cell not caught")
  except AssertionError:
    pass
  db, da = _synth_all(np.random.default_rng(17), positive=True)
  db[4]["fp_punish_d100"] = "TRUE"
  try:
    analyze(db, da)
    raise SystemExit("selfcheck FAIL: non-canonical bool not caught")
  except AssertionError:
    pass
  # midrank Spearman is order-invariant under ties (binary y).
  x = np.array([3.0, 1.0, 2.0, 2.0, 5.0, 4.0, 2.0, 6.0])
  y = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
  assert abs(_spearman(x, y) - _spearman(x[::-1], y[::-1])) < 1e-12
  # loader trips: missing seed / wrong arm spec.
  import tempfile
  db, da = _synth_all(np.random.default_rng(18), positive=True)
  with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "db.csv")
    with open(p, "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=list(db[0].keys()))
      w.writeheader()
      w.writerows(db[1:])          # drop seed 1000
    try:
      load_designb(p)
      raise SystemExit("selfcheck FAIL: missing designB seed not caught")
    except AssertionError:
      pass
    p2 = os.path.join(td, "da.csv")
    bad = [dict(r) for r in da["forbid100"]]
    for r in bad:
      r["explore_mode"] = "uniform"
    with open(p2, "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=list(bad[0].keys()))
      w.writeheader()
      w.writerows(bad)
    try:
      load_designa(p2, "forbid100")
      raise SystemExit("selfcheck FAIL: wrong arm mode not caught")
    except AssertionError:
      pass
  print("selfcheck PASS: planted-positive fires all four primaries "
        "(incl. the Design-A dissociation direction), planted-null "
        "fires none, b1-only RQ3 parity, instrument halt / "
        "manipulation-failure / cohort + population gates with "
        "fires=None suppression, nan + non-canonical-bool corruption "
        "halts, midrank-Spearman tie invariance, loader trips")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--designb")
  ap.add_argument("--designa_base")
  ap.add_argument("--designa_forbid100")
  ap.add_argument("--designa_forbid50")
  ap.add_argument("--designa_force100")
  ap.add_argument("--designa_force50")
  ap.add_argument("--output", default="analysis_out/collusion_confirm")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck(args)
  else:
    read(args)


if __name__ == "__main__":
  main()
