"""Listen-or-Leap: exact-referee VoI testbed (Paper-2 appendix).

Registered under PREREG_p2_lol_20260821.md (ThreeAxes item 11 — "the
one stop-lift worth asking for on Paper 2's behalf"). ONE execution of
--run. Pure CPU, self-contained; no repo instrument is touched.

THE ENVIRONMENT (Tiger-style POMDP): hidden z in {L, R}, uniform
prior. Actions: LISTEN (cost c, observation with accuracy q), OPEN-L /
OPEN-R (ends the episode; opening the tiger door pays R_LOSE, the
other pays R_WIN). Horizon T steps; unopened episodes pay 0.

THE REFEREE: the belief MDP is EXACTLY solvable — beliefs reachable
from the uniform prior are indexed by the listen tally k (#l - #r
observations), b(k) = sigmoid(k * log(q/(1-q))), and finite-horizon
value iteration over (steps_left, tally) is exact. EVSI(s, k) =
Q(listen) - max Q(open) is exact ground truth, positive in the
high-q / low-c region — the environment genuinely holds value of
information, unlike the DMC substrate.

THE CHAIN UNDER TEST: the program's own estimator shape, replayed
exactly: per labeled state, M = 3 candidates (the three actions), R
CRN repeats (common z-draw and obs stream across candidates within a
repeat, independent across repeats), split selection on even repeats,
evaluation on odd repeats, against a VoI-BLIND plug-in consumer
(values LISTEN as pure cost — the analog of a frozen qfull that cannot
represent information value). Followers play the exact optimal policy.

ESTIMANDS (cluster = config x seed cell, n = 36 = 9 configs x 4 seeds;
permutation-primary + BCa via the frozen w1_read machinery):
  rec(s)   = split-selected opportunity vs the plug-in choice
  exact(s) = Q*(a_sel) - Q*(a_now)  (the exact referee of the REALIZED
             selection)
  PRIMARY: pooled recovery slope of rec on exact across cells
  (percentile bootstrap over cells, B=4000 — the DECISIVE statistic).
  The paired bias rec - exact is a split/CRN-integrity DIAGNOSTIC only
  (mean-zero by construction; its interval is anti-conservative at
  this n — review finding 2 — so it adjudicates nothing).
Branch map:
  CHAIN-TRACKS-REFEREE   slope CI excludes 0 AND covers 1
  CHAIN-MISCALIBRATED    slope CI excludes 0, misses 1
  CHAIN-BLIND            slope CI includes 0 despite positive-EVSI
                         substrate (refuses if the substrate check
                         fails: mean exact EVSI must be > 0 in the
                         q=.9/c=.05 config — else the env is miswired)

Usage:
  python -m analysis.listen_or_leap --run --output <dir>
  python -m analysis.listen_or_leap --selfcheck
"""

import argparse
import itertools
import json
import os

import numpy as np

from analysis import w1_read

ALPHA = 0.05
R_WIN, R_LOSE = 10.0, -20.0
HORIZON = 6
Q_GRID = (0.6, 0.75, 0.9)
C_GRID = (0.05, 0.3, 1.0)
SEEDS = (0, 1, 2, 3)
N_STATES = 200
REPEATS = 8
# ROTATED 20260821 -> 20260827 at build review (finding 1): the
# selfcheck's mini-run labels an exact prefix of the registered states,
# so the original seed's estimand prefix had been seen pre-freeze
# (disclosed in the prereg); the registered run draws disjoint states.
# The selfcheck mini-run now uses RUN_SEED + 1.
RUN_SEED = 20260827
LISTEN, OPEN_L, OPEN_R = 0, 1, 2
REGISTERED_OUTPUT = 'artifacts/p2_lol_20260821'


def refuse(msg):
  raise SystemExit(f'READ REFUSED: {msg}')


def belief(k, q):
  """P(z = L | tally k), from the uniform prior."""
  lo = k * np.log(q / (1.0 - q))
  return 1.0 / (1.0 + np.exp(-lo))


def solve(q, c, T=HORIZON):
  """Exact finite-horizon values Q[s][k][a] over (steps_left s, tally
  k in [-T, T]). OPEN-L payoff at belief b: b*R_LOSE + (1-b)*R_WIN
  (z = L means the tiger is behind L)."""
  K = 2 * T + 1

  def kidx(k):
    return k + T

  V = np.zeros((T + 1, K))
  Q = np.zeros((T + 1, K, 3))
  for s in range(1, T + 1):
    for k in range(-T, T + 1):
      b = belief(k, q)
      open_l = b * R_LOSE + (1.0 - b) * R_WIN
      open_r = (1.0 - b) * R_LOSE + b * R_WIN
      p_l = b * q + (1.0 - b) * (1.0 - q)
      k_up = min(k + 1, T)
      k_dn = max(k - 1, T * -1)
      listen = -c + p_l * V[s - 1, kidx(k_up)] + (
          1.0 - p_l) * V[s - 1, kidx(k_dn)]
      Q[s, kidx(k)] = (listen, open_l, open_r)
      V[s, kidx(k)] = max(listen, open_l, open_r)
  return Q, V


def plugin_q(q, c, s, k, Q):
  """The VoI-BLIND consumer: LISTEN valued as pure cost plus staying at
  the same belief (no posterior update — information is invisible to
  it); OPEN values exact. The analog of a qfull that cannot represent
  what a probe would reveal."""
  b = belief(k, q)
  open_l = b * R_LOSE + (1.0 - b) * R_WIN
  open_r = (1.0 - b) * R_LOSE + b * R_WIN
  blind_listen = -c + max(open_l, open_r) - 1e-6
  return np.array([blind_listen, open_l, open_r])


def rollout(q, c, s, k, first_act, Q, rng_z, rng_obs):
  """G(a): execute first_act at (s, k), then the exact optimal policy.
  rng_z draws the hidden state consistent with the belief; rng_obs
  drives listen observations. CRN: pass the same generators (same
  state) across candidates within a repeat."""
  T = Q.shape[0] - 1
  b = belief(k, q)
  z_is_l = bool(rng_z.random() < b)
  total = 0.0
  act = first_act
  while True:
    if act == OPEN_L:
      return total + (R_LOSE if z_is_l else R_WIN)
    if act == OPEN_R:
      return total + (R_WIN if z_is_l else R_LOSE)
    total -= c
    truth_l = z_is_l if (rng_obs.random() < q) else (not z_is_l)
    k = min(max(k + (1 if truth_l else -1), -T), T)
    s -= 1
    if s <= 0:
      return total
    act = int(np.argmax(Q[s, k + T]))


def label_cell(q, c, seed, n_states=N_STATES, repeats=REPEATS,
               run_seed=RUN_SEED):
  """One cell: sample states from the optimal policy's visitation,
  evaluate all 3 candidates x R repeats under CRN, split-select."""
  Q, _ = solve(q, c)
  T = HORIZON
  rng = np.random.default_rng([run_seed, int(q * 1000), int(c * 1000),
                               seed])
  rec, exact, evsi = [], [], []
  n = 0
  while n < n_states:
    # visitation: start of an episode, walk the optimal policy to a
    # random depth, label the reached (s, k) if non-terminal
    s, k = T, 0
    depth = int(rng.integers(0, T - 1))
    ok = True
    z_is_l = bool(rng.random() < 0.5)
    for _ in range(depth):
      a = int(np.argmax(Q[s, k + T]))
      if a != LISTEN:
        ok = False
        break
      truth_l = z_is_l if (rng.random() < q) else (not z_is_l)
      k = min(max(k + (1 if truth_l else -1), -T), T)
      s -= 1
    if not ok or s <= 0:
      continue
    n += 1
    g = np.zeros((repeats, 3))
    for r in range(repeats):
      zs, os_ = rng.integers(0, 2 ** 31, 2)
      for a in range(3):
        # CRN: identical generator STATES across candidates in a repeat
        g[r, a] = rollout(q, c, s, k, a, Q,
                          np.random.default_rng(int(zs)),
                          np.random.default_rng(int(os_)))
    m_now = int(np.argmax(plugin_q(q, c, s, k, Q)))
    even = g[0::2].mean(0)
    odd = g[1::2].mean(0)
    sel = int(np.argmax(even))
    rec.append(float(odd[sel] - odd[m_now]))
    qs = Q[s, k + T]
    exact.append(float(qs[sel] - qs[m_now]))
    evsi.append(float(qs[LISTEN] - max(qs[OPEN_L], qs[OPEN_R])))
  return (float(np.mean(rec)), float(np.mean(exact)),
          float(np.mean(evsi)))


def _stat(vals, cl, b=None):
  b = b or w1_read.B_BOOT
  point, ci = w1_read.bca(np.asarray(vals, float), cl, b=b)
  return dict(point=point, ci=list(ci),
              perm_p=w1_read.perm_p(vals, b=b), n_cells=len(vals))


def run(output, b=None, n_states=N_STATES, run_seed=RUN_SEED,
        _selfcheck=False):
  if not _selfcheck and os.path.abspath(output) != os.path.abspath(
      REGISTERED_OUTPUT):
    refuse(f'output {output!r} != registered {REGISTERED_OUTPUT!r} — '
           'the ONE-read guard is path-pinned (a fresh --output would '
           'defeat it)')
  os.makedirs(output, exist_ok=True)
  path = os.path.join(output, 'lol_read.json')
  if os.path.exists(path):
    refuse(f'{path} exists — ONE read execution is registered')
  cells = {}
  for q, c, seed in itertools.product(Q_GRID, C_GRID, SEEDS):
    cells[(q, c, seed)] = label_cell(q, c, seed, n_states=n_states,
                                     run_seed=run_seed)
  cl = ['%s_%s_%d' % k for k in cells]
  rec = np.array([v[0] for v in cells.values()])
  exact = np.array([v[1] for v in cells.values()])
  evsi = {f'{q}_{c}': float(np.mean([cells[(q, c, s)][2]
                                     for s in SEEDS]))
          for q in Q_GRID for c in C_GRID}
  # substrate check: the env must genuinely hold VoI where designed
  if not evsi['0.9_0.05'] > 0:
    refuse(f'substrate miswired: exact EVSI at q=.9/c=.05 is '
           f'{evsi["0.9_0.05"]} <= 0')
  denom = float(np.sum((exact - exact.mean()) ** 2))
  if denom <= 0:
    refuse('degenerate exact-referee spread across cells — a wiring '
           'failure, not a finding (review finding 7)')
  slope = float(np.sum((rec - rec.mean()) * (exact - exact.mean()))
                / denom)
  # cell-level percentile bootstrap of the slope (B=4000 unless
  # overridden; disclosed in the prereg — the DECISIVE statistic for
  # the branch map, review finding 4)
  rng = np.random.default_rng(run_seed)
  boots = []
  nb = b or 4000
  for _ in range(nb):
    idx = rng.integers(0, len(rec), len(rec))
    d = float(np.sum((exact[idx] - exact[idx].mean()) ** 2))
    if d > 0:
      boots.append(float(np.sum(
          (rec[idx] - rec[idx].mean()) *
          (exact[idx] - exact[idx].mean())) / d))
  if not boots:
    refuse('slope bootstrap degenerate (review finding 7)')
  lo, hi = np.percentile(boots, [2.5, 97.5])
  slope_cells = dict(point=slope, ci=[float(lo), float(hi)])
  # Bias = rec - exact is mean-zero BY CONSTRUCTION (odd half is
  # independent of selection) — a split/CRN-integrity DIAGNOSTIC, not
  # an adjudicator (review finding 2: its BCa CI is ~27% too narrow at
  # this n and would brand a calibrated chain MISCALIBRATED ~1 in 4;
  # the registered null operating characteristics live in the prereg).
  bias = _stat(list(rec - exact), cl, b=b)
  out = dict(n_cells=len(cells), evsi_by_config=evsi,
             recovery_slope=slope_cells, bias_diagnostic=bias,
             rec_pooled=float(rec.mean()), exact_pooled=float(exact.mean()))
  excl0, cov1 = (lo > 0 or hi < 0), (lo <= 1.0 <= hi)
  verdict = ('CHAIN-TRACKS-REFEREE' if (excl0 and cov1)
             else 'CHAIN-MISCALIBRATED' if excl0 else 'CHAIN-BLIND')
  out['verdict'] = verdict
  with open(path, 'w') as f:
    json.dump(out, f, indent=1)
  print(json.dumps(out, indent=1))
  print('->', path)
  return out


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _brute_value(q, c, s, w_l, w_r):
  """INDEPENDENT verifier (review finding 3): recursion over
  observation HISTORIES carrying unnormalized joint weights
  (w_l, w_r) = P(history, z=L), P(history, z=R). Uses only the raw
  likelihoods and the uniform prior — no belief(), no tally, no p_l
  helper, no clamping — so a wrong belief()/posterior in the DP cannot
  be self-consistently wrong here."""
  tot = w_l + w_r
  open_l = (w_l * R_LOSE + w_r * R_WIN) / tot
  open_r = (w_r * R_LOSE + w_l * R_WIN) / tot
  best = max(open_l, open_r)
  cont = 0.0
  if s > 1:
    # o = l has joint weights (w_l * q, w_r * (1-q)); o = r the mirror
    wl_l, wr_l = w_l * q, w_r * (1.0 - q)
    wl_r, wr_r = w_l * (1.0 - q), w_r * q
    p_obs_l = (wl_l + wr_l) / tot
    cont = (p_obs_l * _brute_value(q, c, s - 1, wl_l, wr_l)
            + (1.0 - p_obs_l) * _brute_value(q, c, s - 1, wl_r, wr_r))
  # listening at the last step is the walk-away action (unopened pays 0)
  return max(best, -c + cont)


def selfcheck():
  # 1) DP vs the INDEPENDENT history-space verifier: tally k maps to
  # joint weights w_l = q^{n_l} (1-q)^{n_r}, w_r = (1-q)^{n_l} q^{n_r}
  # (uniform prior cancels); checked over configs x depths x histories
  for q, c in ((0.8, 0.1), (0.65, 0.5)):
    Q, V = solve(q, c, T=3)
    for s in (1, 2, 3):
      for n_l, n_r in ((0, 2), (0, 0), (1, 0), (2, 1)):
        k = n_l - n_r
        w_l = (q ** n_l) * ((1.0 - q) ** n_r)
        w_r = ((1.0 - q) ** n_l) * (q ** n_r)
        want = _brute_value(q, c, s, w_l, w_r)
        got = V[s, k + 3]
        assert abs(got - want) < 1e-9, (q, c, s, k, got, want)
  # 2) EVSI sign structure: positive at high q / low c, negative at
  # low q / high c (listening can't pay)
  Q, _ = solve(0.9, 0.05)
  assert Q[HORIZON, 0 + HORIZON, LISTEN] > max(
      Q[HORIZON, 0 + HORIZON, OPEN_L], Q[HORIZON, 0 + HORIZON, OPEN_R])
  # ...negative at a CONFIDENT belief under low accuracy / high cost
  # (at b=0.5 walking away dominates opening, so EVSI is trivially
  # positive there — the confident-belief cell is where listening
  # cannot pay)
  Q2, _ = solve(0.6, 1.0)
  assert Q2[HORIZON, 3 + HORIZON, LISTEN] < max(
      Q2[HORIZON, 3 + HORIZON, OPEN_L], Q2[HORIZON, 3 + HORIZON, OPEN_R])
  # 3) CRN duplicate-null: identical candidates -> exactly equal G
  Qd, _ = solve(0.8, 0.1)
  for r in range(5):
    zs, os_ = np.random.default_rng(r).integers(0, 2 ** 31, 2)
    g1 = rollout(0.8, 0.1, 4, 0, OPEN_L, Qd,
                 np.random.default_rng(int(zs)),
                 np.random.default_rng(int(os_)))
    g2 = rollout(0.8, 0.1, 4, 0, OPEN_L, Qd,
                 np.random.default_rng(int(zs)),
                 np.random.default_rng(int(os_)))
    assert g1 == g2, (g1, g2)
  # 4) unbiasedness of the rollout: MC mean of G(a) matches exact Q
  rng = np.random.default_rng(0)
  for a in range(3):
    gs = []
    for _ in range(4000):
      zs, os_ = rng.integers(0, 2 ** 31, 2)
      gs.append(rollout(0.8, 0.1, 4, 1, a, Qd,
                        np.random.default_rng(int(zs)),
                        np.random.default_rng(int(os_))))
    want = Qd[4, 1 + HORIZON, a]
    got = float(np.mean(gs))
    se = float(np.std(gs) / np.sqrt(len(gs)))
    assert abs(got - want) < max(5 * se, 1e-6), (a, got, want, se)
  # 5) end-to-end mini-run at a DISJOINT seed (review finding 1: the
  # registered RUN_SEED's states are never touched pre-read)
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    out = run(os.path.join(tmp, 'o'), b=400, n_states=40,
              run_seed=RUN_SEED + 1, _selfcheck=True)
    assert out['evsi_by_config']['0.9_0.05'] > 0
    assert out['recovery_slope']['point'] > 0.3, out['recovery_slope']
  # 6) ONE-read guard + path pin
  with tempfile.TemporaryDirectory() as tmp:
    os.makedirs(os.path.join(tmp, 'o'))
    open(os.path.join(tmp, 'o', 'lol_read.json'), 'w').close()
    try:
      run(os.path.join(tmp, 'o'), b=400, n_states=10,
          run_seed=RUN_SEED + 1, _selfcheck=True)
      raise AssertionError('ONE-read guard failed to trip')
    except SystemExit as e:
      assert 'ONE read execution' in str(e), e
    try:
      run(os.path.join(tmp, 'other'), b=400, n_states=10)
      raise AssertionError('output path pin failed to trip')
    except SystemExit as e:
      assert 'path-pinned' in str(e), e
  print('SELFCHECK PASS (listen_or_leap: DP == independent '
        'history-space verifier; EVSI sign structure; CRN duplicate '
        'exact; rollout unbiased vs exact Q; end-to-end mini-run at a '
        'DISJOINT seed (slope > 0.3); ONE-read guard + output path pin)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--run', action='store_true')
  p.add_argument('--output')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if args.run:
    if not args.output:
      p.error('--output required with --run')
    run(args.output)
    return
  p.error('--run or --selfcheck required')


if __name__ == '__main__':
  main()
