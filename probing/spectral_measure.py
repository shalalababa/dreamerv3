"""Spectral measurement pass (PREREG_spectral_domains_20260724.md).

Measures, per frozen buffer, the theory quantities of the
spectral-competition model (Theory_SpectralTransfer_20260717.tex):

  - the eigenvalue spectrum of the windowed-history feature process
    Sigma = E[x x^T] (x = H stacked symlog'd proprio frames — the
    encoder's own input transform, NOT standardized, so variance
    salience is preserved; windows never cross episode boundaries);
  - the reward-aligned direction theta (out-of-fold ridge regression
    of reward on x; folds by chunk), its variance lambda_need =
    theta^T Sigma theta, its VARIANCE RANK (# eigenvalues above it),
    and the ratio profile lambda_need / g_k at registered cuts;
  - label energy: rewarded-frame fraction f, mean reward, and the
    rewarded-frame reward variance (s^2 proxy);
  - the support-breadth (diversity) index for P-E4: participation
    ratio of the rewarded-frame (H=1) covariance + the number of
    episodes containing reward.

CPU/numpy only; reads raw replay chunk npz files directly (no jax,
no embodied import), so it runs on a login node against the frozen
buffers.

Usage:
  python -m probing.spectral_measure measure --replay <chunkdir> \
      --domain cup --side hi --tag <buffer-id> --output <out.json>
  python -m probing.spectral_measure compare --inputs a.json b.json \
      ... --output <dir>          # registered domain-contrast verdict
  python -m probing.spectral_measure --selfcheck
"""

import argparse
import glob
import json
import os

import numpy as np

WINDOW = 8                     # frames per history window (registered)
CUTS = (4, 8, 16, 32, 64)      # capacity cuts for the ratio profile
RIDGE_SCALE = 1e-3             # ridge = RIDGE_SCALE * tr(Sigma)/D
MAX_STEPS = 500_000            # uniform chunk subsample above this
MIN_REWARDED = 50              # amend 1: fewer rewarded frames =>
                               # theta inestimable, spectrum-only record
EXCLUDE_PREFIX = ("log", "stepid")
EXCLUDE_KEYS = ("is_first", "is_last", "is_terminal", "reward",
                "action", "reset", "cont")
# v1_1 (amend 1): theta_estimable/theta_source/theta/n_rewarded fields;
# v1 jsons (pre-amendment) are refused by compare() — re-measure them
# (deterministic; same numbers plus the new fields).
MEASURE_VERSION = "spectral_v1_1_20260724"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_chunks(replay_dir, max_steps=MAX_STEPS, rng_seed=0):
  """Returns a list of per-chunk dicts of arrays (only needed keys)."""
  paths = sorted(glob.glob(os.path.join(replay_dir, "*.npz")))
  assert paths, f"no chunk files in {replay_dir}"
  rng = np.random.default_rng(rng_seed)
  lengths = [int(os.path.basename(p).rsplit("-", 1)[-1][:-4]) for p in paths]
  total = sum(lengths)
  if total > max_steps:
    keep = rng.permutation(len(paths))
    acc, chosen = 0, []
    for i in keep:
      chosen.append(i)
      acc += lengths[i]
      if acc >= max_steps:
        break
    paths = [paths[i] for i in sorted(chosen)]
  chunks = []
  for p in paths:
    with open(p, "rb") as f:
      d = np.load(f)
      d = {k: d[k] for k in d.keys()}
    chunks.append(d)
  return chunks


def obs_keys(chunk):
  keys = []
  for k, v in sorted(chunk.items()):
    if k in EXCLUDE_KEYS or any(k.startswith(p) for p in EXCLUDE_PREFIX):
      continue
    if not np.issubdtype(v.dtype, np.floating) and not np.issubdtype(
        v.dtype, np.integer):
      continue
    if v.ndim >= 3:            # image obs: pixel extension, not here
      continue
    keys.append(k)
  assert keys, "no proprio observation keys found"
  return keys


def frames_of(chunk, keys):
  cols = []
  for k in keys:
    v = np.asarray(chunk[k], np.float64)
    cols.append(v.reshape(len(v), -1))
  return np.concatenate(cols, 1)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def symlog(x):
  return np.sign(x) * np.log1p(np.abs(x))


def build_windows(chunks, keys, window=WINDOW):
  """Symlog'd frames -> windowed x; never cross is_first.

  Per-dim SYMLOG (the encoder's own input transform), deliberately NOT
  per-dim standardization: relative variance salience across dims is
  the quantity the spectrum must preserve (standardizing would erase
  lambda structure). Windows are within-chunk only (registered
  simplification; boundary loss < window/chunksize). Reward aligns to
  the LAST frame of the window. Returns x [N, window*D], r [N], fold
  ids [N] (chunk index), plus frame-level (H=1) arrays for the
  diversity index.
  """
  frames = [symlog(frames_of(c, keys)) for c in chunks]
  xs, rs, folds = [], [], []
  ep_reward = []               # per episode: has any reward > 0
  for ci, (c, fr) in enumerate(zip(chunks, frames)):
    first = np.asarray(c["is_first"], bool)
    rew = np.asarray(c["reward"], np.float64)
    epid = np.cumsum(first)
    # episode reward presence (chunk-local episodes; registered index
    # detail: episodes spanning chunks count once per chunk segment)
    for e in np.unique(epid):
      ep_reward.append(bool((rew[epid == e] > 0).any()))
    n = len(fr)
    if n < window:
      continue
    idx = np.arange(window - 1, n)
    # drop windows whose interior crosses an episode start
    bad = np.zeros(len(idx), bool)
    for off in range(0, window - 1):
      bad |= first[idx - off]
    ok = idx[~bad]
    if not len(ok):
      continue
    win = np.stack([fr[ok - off] for off in range(window - 1, -1, -1)], 1)
    xs.append(win.reshape(len(ok), -1))
    rs.append(rew[ok])
    folds.append(np.full(len(ok), ci))
  x = np.concatenate(xs, 0)
  r = np.concatenate(rs, 0)
  fold = np.concatenate(folds, 0)
  flat = np.concatenate(frames, 0)
  flat_rew = np.concatenate([np.asarray(c["reward"], np.float64)
                             for c in chunks])
  return x, r, fold, flat, flat_rew, np.asarray(ep_reward, bool)


def _ridge_theta(x, r, lam):
  d = x.shape[1]
  A = x.T @ x / len(x) + lam * np.eye(d)
  b = x.T @ r / len(x)
  return np.linalg.solve(A, b)


def reward_direction(x, r, fold):
  """OOF ridge: theta from all data (direction), R2 out-of-fold.

  Returns (None, 0.0) when theta is inestimable — too few rewarded
  frames or (near-)constant rewards make the centered cross-moment
  identically zero (amendment 1: registered spectrum-only fallback,
  MIN_REWARDED threshold)."""
  if int((r > 0).sum()) < MIN_REWARDED or float(np.var(r)) == 0.0:
    return None, 0.0
  xc = x - x.mean(0)
  rc = r - r.mean()
  lam = RIDGE_SCALE * np.trace(xc.T @ xc / len(xc)) / xc.shape[1]
  theta = _ridge_theta(xc, rc, lam)
  nfold = min(5, len(np.unique(fold)))
  groups = np.unique(fold)
  assign = {g: i % nfold for i, g in enumerate(groups)}
  fid = np.asarray([assign[g] for g in fold])
  sse = sst = 0.0
  for k in range(nfold):
    tr, te = fid != k, fid == k
    if not tr.any() or not te.any():
      continue
    th = _ridge_theta(xc[tr], rc[tr], lam)
    pred = xc[te] @ th
    sse += float(((rc[te] - pred) ** 2).sum())
    sst += float((rc[te] ** 2).sum())
  r2_oof = 1.0 - sse / sst if sst > 0 else 0.0
  norm = np.linalg.norm(theta)
  if norm == 0:
    return None, 0.0
  return theta / norm, float(r2_oof)


def participation_ratio(mat):
  if len(mat) < 3:
    return 0.0
  ev = np.linalg.eigvalsh(np.cov(mat.T))
  ev = np.clip(ev, 0, None)
  s1, s2 = ev.sum(), (ev ** 2).sum()
  return float(s1 ** 2 / s2) if s2 > 0 else 0.0


def measure(replay_dir, domain, side, tag, window=WINDOW,
            max_steps=MAX_STEPS, theta_from=None):
  """theta_from (amend 1, cross-buffer secondary): a prior measure
  json of the SAME domain whose own-estimated theta is evaluated
  against THIS buffer's spectrum (theta is a task property, Sigma a
  buffer property). Feature spaces must match (obs_keys + window)."""
  chunks = load_chunks(replay_dir, max_steps=max_steps)
  keys = obs_keys(chunks[0])
  x, r, fold, flat, flat_rew, ep_reward = build_windows(
      chunks, keys, window)
  xc = x - x.mean(0)
  cov = xc.T @ xc / len(xc)
  eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
  eigs = np.clip(eigs, 0, None)
  theta_source = "own"
  if theta_from is None:
    theta, r2_oof = reward_direction(x, r, fold)
  else:
    with open(theta_from) as f:
      src = json.load(f)
    assert src["domain"] == domain, "cross-buffer theta: same domain only"
    assert src["obs_keys"] == keys and src["window"] == window, (
        "cross-buffer theta: feature space mismatch")
    assert src.get("theta") is not None, "source json has no theta"
    theta = np.asarray(src["theta"], np.float64)
    theta /= np.linalg.norm(theta)
    r2_oof = None                     # not evaluated for imported theta
    theta_source = os.path.basename(theta_from)
  estimable = theta is not None
  if estimable:
    lam_need = float(theta @ cov @ theta)
    rank = int((eigs > lam_need).sum())
    ratio = {int(k): (float(lam_need / eigs[k - 1]) if k <= len(eigs)
                      else None) for k in CUTS}
  else:
    lam_need = rank = ratio = None
  rewarded = flat_rew > 0
  out = dict(
      version=MEASURE_VERSION, domain=domain, side=side, tag=tag,
      replay_dir=replay_dir, window=window, obs_keys=keys,
      n_windows=int(len(x)), n_frames=int(len(flat)),
      dim=int(x.shape[1]),
      spectrum_top=[float(v) for v in eigs[:64]],
      spectrum_pr=float(
          eigs.sum() ** 2 / (eigs ** 2).sum()) if eigs.sum() else 0.0,
      theta_estimable=bool(estimable), theta_source=theta_source,
      theta=([float(v) for v in theta] if estimable
             and theta_source == "own" else None),
      lambda_need=lam_need, variance_rank=rank, ratio_profile=ratio,
      r2_oof=r2_oof,
      f_rewarded=float(rewarded.mean()),
      n_rewarded=int(rewarded.sum()),
      mean_reward=float(flat_rew.mean()),
      s2_rewarded=float(np.var(flat_rew[rewarded])) if rewarded.any()
      else 0.0,
      diversity_pr=participation_ratio(flat[rewarded]),
      n_reward_episodes=int(ep_reward.sum()),
      n_episodes=int(len(ep_reward)),
  )
  return out


# --------------------------------------------------------------------------
# Registered domain contrast
# --------------------------------------------------------------------------

def compare(inputs):
  meas = []
  for p in inputs:
    with open(p) as f:
      meas.append(json.load(f))
  for m in meas:
    assert m.get("version") == MEASURE_VERSION, (
        f"{m.get('tag')}: json version {m.get('version')!r} != "
        f"{MEASURE_VERSION} — re-measure with the amended script")
  doms = sorted({m["domain"] for m in meas})
  assert set(doms) >= {"cup", "finger"}, (
      "registered primary needs cup and finger measurements")
  # primary population: own-theta estimable records only (amend 1;
  # cross-buffer-theta records are the registered secondary)
  own = [m for m in meas if m.get("theta_estimable")
         and m.get("theta_source") == "own"]
  cross = [m for m in meas if m.get("theta_estimable")
           and m.get("theta_source") != "own"]
  by = {}
  for m in own:
    by.setdefault((m["domain"], m["side"]), []).append(m)
  matched = sorted({m["side"] for m in meas
                    if any(x["domain"] == "cup" and x["side"] == m["side"]
                           for x in meas)
                    and any(x["domain"] == "finger"
                            and x["side"] == m["side"] for x in meas)})
  sides = [s for s in matched
           if ("cup", s) in by and ("finger", s) in by]
  dropped = [s for s in matched if s not in sides]
  assert sides, (
      "no matched side has own-theta-estimable records in both domains")
  primary = {}
  agree = []
  for s in sides:
    rc = float(np.median([m["variance_rank"] for m in by[("cup", s)]]))
    rf = float(np.median([m["variance_rank"] for m in by[("finger", s)]]))
    primary[s] = dict(rank_cup=rc, rank_finger=rf, holds=bool(rc < rf))
    agree.append(rc < rf)
  confirmed = all(agree)
  secondary = {}
  for s in sides:
    secondary[s] = {
        dom: dict(
            ratio_profile=by[(dom, s)][0]["ratio_profile"],
            lambda_need=float(np.median(
                [m["lambda_need"] for m in by[(dom, s)]])),
            f_rewarded=float(np.median(
                [m["f_rewarded"] for m in by[(dom, s)]])),
            diversity_pr=float(np.median(
                [m["diversity_pr"] for m in by[(dom, s)]])),
            r2_oof=float(np.median([m["r2_oof"] for m in by[(dom, s)]])),
        ) for dom in ("cup", "finger")}
  crossbuffer = [
      dict(domain=m["domain"], side=m["side"], tag=m["tag"],
           theta_source=m["theta_source"],
           variance_rank=m["variance_rank"],
           lambda_need=m["lambda_need"])
      for m in cross]
  inest = [dict(domain=m["domain"], side=m["side"], tag=m["tag"],
                n_rewarded=m.get("n_rewarded"),
                f_rewarded=m["f_rewarded"])
           for m in meas if not m.get("theta_estimable")]
  verdict = (
      "P-SM1 CONFIRMED: cup's reward direction outranks finger's in "
      "every evaluable matched side — the domain contrast is a "
      "lambda-spectrum fact, as the spectral account requires."
      if confirmed else
      "P-SM1 NOT confirmed: the reward-direction variance-rank "
      "ordering does not hold in every evaluable matched side — the "
      "spectral account of the cup/finger boundary fails as "
      "registered.")
  if dropped:
    verdict += (f" [Amend 1 disclosure: matched side(s) {dropped} "
                "had no own-theta-estimable records in both domains "
                "and were excluded from the primary; see "
                "inestimable/crossbuffer blocks.]")
  return dict(primary=primary, confirmed=bool(confirmed),
              dropped_sides=dropped, secondary=secondary,
              secondary_crossbuffer=crossbuffer,
              inestimable=inest, verdict=verdict,
              inputs=[os.path.basename(p) for p in inputs])


# --------------------------------------------------------------------------
# Selfcheck: synthetic chunk dirs, planted spectra
# --------------------------------------------------------------------------

def _write_synth(tmp, name, hi_var_reward, n_chunks=6, T=512, d=12,
                 broad=1.0, f=0.15, seed=0):
  rng = np.random.default_rng(seed)
  path = os.path.join(tmp, name)
  os.makedirs(path, exist_ok=True)
  scales = np.linspace(3.0, 0.3, d)
  jdim = 1 if hi_var_reward else d - 2   # reward-aligned dim
  for ci in range(n_chunks):
    z = rng.normal(size=(T, d)) * scales
    z[:, jdim] *= broad
    first = np.zeros(T, bool)
    first[::128] = True
    rew = np.where(rng.random(T) < f,
                   np.clip(z[:, jdim] * 0.5 + 0.5, 0, 1), 0.0)
    np.savez_compressed(
        os.path.join(path, f"0000-{ci:016d}-0-{T}.npz"),
        obs=z.astype(np.float32), reward=rew.astype(np.float32),
        is_first=first, is_last=np.zeros(T, bool),
        is_terminal=np.zeros(T, bool),
        action=rng.normal(size=(T, 2)).astype(np.float32))
  return path


def selfcheck():
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    cup = _write_synth(tmp, "cup_hi", hi_var_reward=True, seed=1)
    fin = _write_synth(tmp, "finger_hi", hi_var_reward=False, seed=2)
    mc = measure(cup, "cup", "hi", "synth")
    mf = measure(fin, "finger", "hi", "synth")
    assert mc["variance_rank"] < mf["variance_rank"], (
        mc["variance_rank"], mf["variance_rank"])
    assert mc["ratio_profile"][8] > mf["ratio_profile"][8]
    # planted mask f=0.15, but clip-at-zero floors the ~36% of masked
    # frames with z < -1, so realized f ~ 0.15 * 0.64 ~ 0.10
    assert 0.05 < mc["f_rewarded"] < 0.20, mc["f_rewarded"]
    # sparse Bernoulli-masked reward caps frame-level linear R2 near
    # the mask rate x explainable share; positive OOF (vs ~0 for an
    # unlearnable direction) is the check, not a large absolute value
    assert mc["r2_oof"] > 0.01, mc["r2_oof"]
    # windows never cross episode starts: every window drops when
    # episodes are 1 frame long
    one = _write_synth(tmp, "onestep", hi_var_reward=True, seed=3)
    for p in glob.glob(os.path.join(one, "*.npz")):
      with open(p, "rb") as fh:
        d = {k: v for k, v in np.load(fh).items()}
      d["is_first"] = np.ones(len(d["is_first"]), bool)
      np.savez_compressed(p, **d)
    tripped = False
    try:
      measure(one, "cup", "hi", "allfirst")
    except (ValueError, AssertionError):
      tripped = True
    assert tripped, "all-is_first buffer must yield no windows"
    # diversity index separates broad from narrow rewarded support
    broad = _write_synth(tmp, "broad", hi_var_reward=True, broad=1.0,
                         seed=4)
    narrow = _write_synth(tmp, "narrow", hi_var_reward=True, broad=0.05,
                          seed=4)
    db = measure(broad, "cup", "hi", "b")["diversity_pr"]
    dn = measure(narrow, "cup", "hi", "n")["diversity_pr"]
    assert db > dn, (db, dn)
    # compare(): planted ordering confirms; swapped labels refute
    pc, pf = os.path.join(tmp, "c.json"), os.path.join(tmp, "f.json")
    json.dump(mc, open(pc, "w")); json.dump(mf, open(pf, "w"))
    res = compare([pc, pf])
    assert res["confirmed"], res
    assert res["dropped_sides"] == [], res
    mc2, mf2 = dict(mc), dict(mf)
    mc2["domain"], mf2["domain"] = "finger", "cup"
    json.dump(mc2, open(pc, "w")); json.dump(mf2, open(pf, "w"))
    res2 = compare([pc, pf])
    assert not res2["confirmed"], res2
    json.dump(mc, open(pc, "w"))     # restore after the relabel test
    # amend 1: zero-reward buffer -> inestimable spectrum-only record
    zero = _write_synth(tmp, "zerorew", hi_var_reward=True, f=0.0,
                        seed=5)
    mz = measure(zero, "finger", "lo", "zr")
    assert not mz["theta_estimable"] and mz["variance_rank"] is None
    assert mz["n_rewarded"] == 0 and mz["f_rewarded"] == 0.0
    # amend 1: cross-buffer theta restores a rank on the sparse buffer
    # (theta from a same-domain rewarded buffer; pc holds the cup json)
    mzx = measure(zero, "cup", "lo", "zrx", theta_from=pc)
    assert mzx["theta_estimable"] and mzx["variance_rank"] is not None
    assert mzx["theta_source"] == os.path.basename(pc)
    assert mzx["r2_oof"] is None and mzx["theta"] is None
    tripped = False
    try:
      measure(zero, "finger", "lo", "wrongdom", theta_from=pc)
    except AssertionError:
      tripped = True
    assert tripped, "cross-domain theta import must trip"
    # amend 1: compare drops the side with no estimable pair, discloses
    pz = os.path.join(tmp, "z.json")
    json.dump(mz, open(pz, "w"))
    mch = dict(mc); mch["domain"] = "cup"; mch["side"] = "lo"
    pcl = os.path.join(tmp, "cl.json")
    json.dump(mch, open(pcl, "w"))
    json.dump(mc, open(pc, "w")); json.dump(mf, open(pf, "w"))
    res3 = compare([pc, pf, pcl, pz])
    assert res3["dropped_sides"] == ["lo"], res3["dropped_sides"]
    assert list(res3["primary"]) == ["hi"], res3["primary"]
    assert res3["confirmed"], res3
    assert res3["inestimable"][0]["side"] == "lo"
  print("selfcheck PASS: planted high-variance reward dim outranks "
        "low-variance one (rank + ratio), f recovered, OOF R2 "
        "positive, all-first buffer trips, diversity separates broad "
        "from narrow support, compare confirms/refutes correctly; "
        "amend 1: zero-reward buffer yields spectrum-only record, "
        "cross-buffer theta restores rank (cross-domain trips), "
        "compare drops+discloses inestimable sides")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", nargs="?", choices=("measure", "compare"))
  ap.add_argument("--replay")
  ap.add_argument("--domain", choices=("cup", "finger", "pixel"))
  ap.add_argument("--side", choices=("hi", "lo"))
  ap.add_argument("--tag", default="")
  ap.add_argument("--inputs", nargs="*")
  ap.add_argument("--output")
  ap.add_argument("--max_steps", type=int, default=MAX_STEPS)
  ap.add_argument("--theta_from", default=None,
                  help="measure json of the same domain whose theta is "
                       "evaluated against this buffer's spectrum "
                       "(amend-1 cross-buffer secondary)")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if args.cmd == "measure":
    assert args.replay and args.domain and args.side and args.output
    out = measure(args.replay, args.domain, args.side, args.tag,
                  max_steps=args.max_steps, theta_from=args.theta_from)
    with open(args.output, "w") as f:
      json.dump(out, f, indent=1)
    if out["theta_estimable"]:
      r2 = ("n/a" if out["r2_oof"] is None else f"{out['r2_oof']:.3f}")
      print(f"{args.domain}/{args.side}: rank={out['variance_rank']} "
            f"lambda_need={out['lambda_need']:.4f} "
            f"f={out['f_rewarded']:.4f} r2_oof={r2} "
            f"diversity_pr={out['diversity_pr']:.2f} "
            f"theta_source={out['theta_source']}")
    else:
      print(f"{args.domain}/{args.side}: THETA INESTIMABLE "
            f"(n_rewarded={out['n_rewarded']} < {MIN_REWARDED} or "
            f"constant reward) — spectrum-only record; "
            f"f={out['f_rewarded']:.6f} "
            f"diversity_pr={out['diversity_pr']:.2f}; consider "
            f"--theta_from <same-domain json> for the cross-buffer "
            f"secondary")
  elif args.cmd == "compare":
    assert args.inputs and args.output
    res = compare(args.inputs)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "spectral_compare.json"),
              "w") as f:
      json.dump(res, f, indent=1)
    for s, p in res["primary"].items():
      print(f"side {s}: rank cup={p['rank_cup']:.0f} vs "
            f"finger={p['rank_finger']:.0f} holds={p['holds']}")
    print(res["verdict"])


if __name__ == "__main__":
  main()
