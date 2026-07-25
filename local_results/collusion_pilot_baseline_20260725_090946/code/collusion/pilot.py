"""Paper-4 pilot harness: coupled tabular Q-learning pricing agents.

Design memo: research_notes/Design_Collusion_Pilot_20260724.md.
Implements the Calvano et al. (AER 2020) baseline loop (task 1 of the
pilot), the deviation impulse-response fingerprint (the collusion
signature, distinguishing collusion from Abada-Lambin
failure-to-learn), and the experience statistics (joint-state
coverage, punishment-phase occupancy) the composition interventions
will target. CPU/numpy only; one session = one core.

Usage:
  python -m collusion.pilot run --sessions 8 --seed0 0 \
      --max_iters 2000000 --output out.csv
  python -m collusion.pilot --selfcheck
"""

import argparse
import csv
import sys

import numpy as np

from collusion.env import Duopoly, DELTA

ALPHA = 0.15        # learning rate (Calvano baseline)
BETA = 4e-6         # exploration decay: eps_t = exp(-BETA * t)
CONV_WINDOW = 100_000   # greedy profile stable this long => converged
IR_PRE = 5          # pre-deviation cycle periods recorded
IR_POST = 15        # post-deviation periods recorded
PUNISH_W = 5        # punishment-phase window after an undercut event
HARNESS_VERSION = "collusion_pilot_v1_20260724"


def init_q(env):
  """Calvano init: discounted payoff of each action vs uniform opp."""
  q0 = env.pi.mean(1) / (1.0 - DELTA)
  return np.tile(q0[None, :], (env.n_states, 1)).astype(np.float64)


class Session:
  """One coupled online Q-learning session."""

  def __init__(self, env, seed, alpha=ALPHA, beta=BETA):
    self.env = env
    self.rng = np.random.default_rng(seed)
    self.alpha = alpha
    self.beta = beta
    self.q = [init_q(env), init_q(env)]
    self.state = env.state(self.rng.integers(env.n),
                           self.rng.integers(env.n))
    self.t = 0
    self.actions_log = []       # (a0, a1) per period, uint8

  def greedy_profile(self):
    return np.concatenate([np.argmax(q, 1) for q in self.q])

  def step(self):
    env, rng = self.env, self.rng
    eps = np.exp(-self.beta * self.t)
    acts = []
    for i in range(2):
      if rng.random() < eps:
        acts.append(int(rng.integers(env.n)))
      else:
        acts.append(int(np.argmax(self.q[i][self.state])))
    a0, a1 = acts
    nstate = env.state(a0, a1)
    rewards = (env.pi[a0, a1], env.pi[a1, a0])
    for i, (a, r) in enumerate(zip(acts, rewards)):
      target = r + DELTA * float(np.max(self.q[i][nstate]))
      self.q[i][self.state, a] += self.alpha * (
          target - self.q[i][self.state, a])
    self.state = nstate
    self.actions_log.append((a0, a1))
    self.t += 1
    return acts, rewards

  def run(self, max_iters, conv_window=CONV_WINDOW):
    """Run until greedy-profile convergence or max_iters."""
    prev = self.greedy_profile()
    stable = 0
    while self.t < max_iters:
      self.step()
      prof = self.greedy_profile()
      if np.array_equal(prof, prev):
        stable += 1
        if stable >= conv_window:
          return self.t, True
      else:
        stable = 0
        prev = prof
    return self.t, False


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------

def greedy_path(env, qs, state, T):
  path = []
  for _ in range(T):
    a0 = int(np.argmax(qs[0][state]))
    a1 = int(np.argmax(qs[1][state]))
    path.append((a0, a1))
    state = env.state(a0, a1)
  return path, state


def collusion_indices(env, qs, state, T=100):
  path, _ = greedy_path(env, qs, state, T)
  arr = np.asarray(path)
  mean_pi = float(np.mean([env.pi[a0, a1] for a0, a1 in path] +
                          [env.pi[a1, a0] for a0, a1 in path]))
  mean_p = float(env.prices[arr].mean())
  return env.delta_profit(mean_pi), env.delta_price(mean_p)


def impulse_response(env, qs, state, pre=IR_PRE, post=IR_POST):
  """Force agent 0 to the static best response for one period, then
  greedy play; the collusion fingerprint is the OPPONENT's
  punish-then-recover price path (Calvano's signature)."""
  pre_path, s = greedy_path(env, qs, state, pre)
  pre1 = float(np.mean([env.prices[a1] for _, a1 in pre_path]))
  # forced deviation against the opponent's current greedy price
  a1_now = int(np.argmax(qs[1][s]))
  a0_dev = env.static_br(a1_now)
  s = env.state(a0_dev, a1_now)
  post_path, _ = greedy_path(env, qs, s, post)
  p1 = np.asarray([env.prices[a1] for _, a1 in post_path])
  step = env.prices[1] - env.prices[0]
  depth = float(pre1 - p1.min())
  recovered = bool(abs(p1[-1] - pre1) <= step + 1e-12)
  punished = bool(depth >= step - 1e-12)
  return dict(
      fingerprint=bool(punished and recovered),
      punish_depth=depth, recovered=recovered,
      pre_price_opp=pre1,
      post_prices_opp=[float(v) for v in p1],
      deviation_price=float(env.prices[a0_dev]))


def experience_stats(env, actions_log, window=PUNISH_W):
  arr = np.asarray(actions_log, np.int64)
  states = arr[:, 0] * env.n + arr[:, 1]
  coverage = float(len(np.unique(states)) / env.n_states)
  # punishment-phase pilot label: an UNDERCUT EVENT at t is any agent
  # pricing strictly below BOTH agents' t-1 prices; the following
  # `window` periods are punishment-phase.
  prev_min = np.minimum(arr[:-1, 0], arr[:-1, 1])
  cur_min = np.minimum(arr[1:, 0], arr[1:, 1])
  under = np.zeros(len(arr), bool)
  under[1:] = cur_min < prev_min
  phase = np.zeros(len(arr), bool)
  idx = np.flatnonzero(under)
  for i in idx:
    phase[i:i + window] = True
  return dict(coverage=coverage,
              punish_occupancy=float(phase.mean()),
              undercut_rate=float(under.mean()))


def run_session(seed, max_iters, conv_window=CONV_WINDOW, env=None):
  env = env or Duopoly()
  sess = Session(env, seed)
  t_end, converged = sess.run(max_iters, conv_window)
  dpi, dpr = collusion_indices(env, sess.q, sess.state)
  ir = impulse_response(env, sess.q, sess.state)
  stats = experience_stats(env, sess.actions_log)
  return dict(
      seed=seed, iters=t_end, converged=converged,
      delta_profit=dpi, delta_price=dpr,
      fingerprint=ir["fingerprint"], punish_depth=ir["punish_depth"],
      recovered=ir["recovered"], **stats)


# --------------------------------------------------------------------------
# Selfcheck
# --------------------------------------------------------------------------

def _planted_q(env, policy):
  """Q table whose argmax encodes policy(s) -> action."""
  q = np.zeros((env.n_states, env.n))
  for s in range(env.n_states):
    q[s, policy(s)] = 1.0
  return q


def selfcheck():
  env = Duopoly()
  step = env.prices[1] - env.prices[0]
  # 1. equilibrium math
  assert env.p_nash < env.p_mono, (env.p_nash, env.p_mono)
  assert env.prices[0] < env.p_nash < env.p_mono < env.prices[-1]
  assert env.pi_nash < env.pi_mono
  sh = [float(v) for v in
        __import__("collusion.env", fromlist=["demand"]).demand(
            [1.5, 1.5])]
  assert abs(sh[0] - sh[1]) < 1e-12, "symmetric demand"
  lo = __import__("collusion.env", fromlist=["demand"]).demand([1.4, 1.5])
  assert lo[0] > lo[1], "lower price must win share"
  # 2. static BR undercuts the monopoly price
  mono_idx = int(np.argmin(np.abs(env.prices - env.p_mono)))
  assert env.static_br(mono_idx) < mono_idx
  # 3. collusion indices anchor at their definitions
  assert abs(env.delta_profit(env.pi_mono) - 1.0) < 1e-9
  assert abs(env.delta_profit(env.pi_nash)) < 1e-9
  # 4. IR fingerprint: planted punish-then-forgive pair fires
  C = mono_idx
  P = 0

  def forgiving(me):
    def pol(s):
      a0, a1 = env.unstate(s)
      opp = a1 if me == 0 else a0
      return C if opp in (C, P) else P
    return pol

  qs = [_planted_q(env, forgiving(0)), _planted_q(env, forgiving(1))]
  ir = impulse_response(env, qs, env.state(C, C))
  assert ir["fingerprint"], ir
  assert ir["punish_depth"] >= step, ir
  # ... and a competitive pair does not
  nash_idx = int(np.argmin(np.abs(env.prices - env.p_nash)))
  qs2 = [_planted_q(env, lambda s: nash_idx)] * 2
  ir2 = impulse_response(env, qs2, env.state(nash_idx, nash_idx))
  assert not ir2["fingerprint"], ir2
  assert ir2["punish_depth"] < step
  # 5. punishment-phase labeler: hand path with one undercut
  path = [(C, C)] * 10 + [(P, C)] + [(C, C)] * 10
  st = experience_stats(env, path, window=5)
  assert st["undercut_rate"] == 1 / 21
  assert abs(st["punish_occupancy"] - 5 / 21) < 1e-12, st
  # 6. learning machinery end-to-end (tiny run; no convergence claim)
  out = run_session(seed=0, max_iters=3000, conv_window=10 ** 9)
  assert out["iters"] == 3000 and not out["converged"]
  assert 0 < out["coverage"] <= 1.0
  assert np.isfinite(out["delta_profit"])
  # 7. determinism: same seed => same outcome
  out2 = run_session(seed=0, max_iters=3000, conv_window=10 ** 9)
  assert out == out2, "sessions must be seed-deterministic"
  print("selfcheck PASS: equilibrium math, BR undercut, index anchors, "
        "planted punish-then-forgive fingerprint fires (competitive "
        "pair does not), undercut labeler exact, tiny session runs "
        "deterministically")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", nargs="?", choices=("run",))
  ap.add_argument("--sessions", type=int, default=8)
  ap.add_argument("--seed0", type=int, default=0)
  ap.add_argument("--max_iters", type=int, default=2_000_000)
  ap.add_argument("--conv_window", type=int, default=CONV_WINDOW)
  ap.add_argument("--output", default="")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.cmd == "run" and args.output
  env = Duopoly()
  rows = []
  for s in range(args.seed0, args.seed0 + args.sessions):
    row = run_session(s, args.max_iters, args.conv_window, env=env)
    rows.append(row)
    print(f"seed {s}: iters={row['iters']} conv={row['converged']} "
          f"dP={row['delta_profit']:+.3f} "
          f"fingerprint={row['fingerprint']} "
          f"cov={row['coverage']:.3f} "
          f"punish_occ={row['punish_occupancy']:.4f}", flush=True)
  with open(args.output, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  fp = float(np.mean([r["fingerprint"] for r in rows]))
  dp = float(np.mean([r["delta_profit"] for r in rows]))
  print(f"[{HARNESS_VERSION}] sessions={len(rows)} mean delta_profit="
        f"{dp:+.3f} fingerprint_rate={fp:.2f}")


if __name__ == "__main__":
  main()
