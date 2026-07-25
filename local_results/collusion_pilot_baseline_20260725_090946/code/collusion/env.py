"""Calvano-style repeated Bertrand duopoly with logit demand.

Paper 4 (Design_Collusion_Pilot_20260724.md). Baseline constants
follow Calvano, Calzolari, Denicolo, Pastorello (AER 2020) as recorded
in the 2-Jul idea brief; VERIFY against the published table before the
registered pilot read (constants are pinned here so any correction is
a visible diff, not a silent drift).

CPU/numpy only.
"""

import numpy as np

# Demand/cost baseline (Calvano et al. 2020, symmetric duopoly).
A_I = 2.0        # product quality index a_i
A_0 = 0.0        # outside good
MU = 0.25        # horizontal differentiation (logit temperature)
COST = 1.0       # marginal cost c
DELTA = 0.95     # discount factor
N_PRICES = 15    # price grid size m
XI = 0.1         # grid extension beyond [p_N, p_M]


def demand(p, a_i=A_I, a_0=A_0, mu=MU):
  """Logit demand shares for price vector p (n_firms,)."""
  p = np.asarray(p, np.float64)
  z = np.exp((a_i - p) / mu)
  return z / (z.sum() + np.exp(a_0 / mu))


def profits(p, cost=COST, **kw):
  p = np.asarray(p, np.float64)
  return (p - cost) * demand(p, **kw)


def nash_price(tol=1e-10, iters=10_000):
  """Symmetric one-shot Bertrand-Nash price by best-response iteration
  on a fine continuous grid."""
  grid = np.linspace(COST, 3.0, 4001)
  p_other = 1.5
  for _ in range(iters):
    pi = np.array([profits([g, p_other])[0] for g in grid])
    br = grid[int(np.argmax(pi))]
    if abs(br - p_other) < tol:
      return float(br)
    p_other = 0.5 * p_other + 0.5 * br
  return float(p_other)


def monopoly_price():
  """Symmetric joint-profit-maximizing price."""
  grid = np.linspace(COST, 3.0, 4001)
  joint = np.array([profits([g, g]).sum() for g in grid])
  return float(grid[int(np.argmax(joint))])


class Duopoly:
  """Discrete price grid + one-period-memory state indexing."""

  def __init__(self, n_prices=N_PRICES, xi=XI):
    self.p_nash = nash_price()
    self.p_mono = monopoly_price()
    span = self.p_mono - self.p_nash
    assert span > 0
    self.prices = np.linspace(
        self.p_nash - xi * span, self.p_mono + xi * span, n_prices)
    self.n = n_prices
    self.n_states = n_prices * n_prices
    # profit table: pi[i, j] = profit of firm playing price i against j
    self.pi = np.empty((self.n, self.n))
    for i in range(self.n):
      for j in range(self.n):
        self.pi[i, j] = profits([self.prices[i], self.prices[j]])[0]
    # per-firm one-shot payoff bounds for collusion indices
    self.pi_nash = profits([self.p_nash, self.p_nash])[0]
    self.pi_mono = profits([self.p_mono, self.p_mono])[0]

  def state(self, a0, a1):
    return a0 * self.n + a1

  def unstate(self, s):
    return divmod(s, self.n)

  def static_br(self, opp_action):
    return int(np.argmax(self.pi[:, opp_action]))

  def delta_profit(self, mean_pi):
    """Calvano profit-gain index (primary reproduction target)."""
    return float((mean_pi - self.pi_nash) / (self.pi_mono - self.pi_nash))

  def delta_price(self, mean_p):
    return float((mean_p - self.p_nash) / (self.p_mono - self.p_nash))
