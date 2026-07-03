"""Synthetic-belief generator for validating the Gate D0 pipeline.

Generates per-state ensemble Q matrices plus a dynamics-disagreement signal
with a controllable distractor dose, instantiating the theta = (theta^r,
theta^i) split of note Prop. 1(ii): the dose inflates U_dyn through a
value-irrelevant component while leaving the gap process untouched. At dose
zero both signals share one epistemic driver, so the pipeline must find
them correlated (no dissociation); at high dose it must find P1. A null
variant applies no distractor effect at any dose (specificity check).

This is a pipeline validation harness, not a WM model: it exists so the
analysis code can be frozen before the first E2 run (App. A discipline).
"""

import numpy as np

from . import signals


def make_cell(dose, n_episodes=50, ep_len=100, k=5, m=8, tau=0.15,
              common_sd=1.0, null=False, seed=0):
  """One (task x dose x seed) cell of synthetic sweep data.

  Returns (sig, episodes): the per-state signal dict from d0.signals.compute
  applied to generated Q matrices, and the episode index per state.

  dose: distractor intensity >= 0 (0 = E1). tau: per-member evaluation
  noise at the full rollout count; the half count gets tau * sqrt(2).
  common_sd: sd of the member-level common-mode shift (A and EVPI are
  invariant to it; U_Q is not -- note Lemma 3).
  """
  rng = np.random.default_rng(seed)
  n = n_episodes * ep_len
  episodes = np.repeat(np.arange(n_episodes), ep_len)

  # Epistemic driver: episode-level latent times state-level jitter, so
  # states within an episode are correlated (exercises clustered SEs).
  ep_latent = rng.lognormal(0.0, 0.6, n_episodes)
  e = np.repeat(ep_latent, ep_len) * rng.lognormal(0.0, 0.4, n)

  # Advantage structure: action 0 is best; the top-2 gap scales inversely
  # with the epistemic driver (uncertain states tend to be harder calls), so
  # at dose zero flip rate and disagreement share one driver and the E1
  # cell is tightly coupled (Q-I < 1%, the App. A.6 baseline). A near-tie
  # mixture component exercises P2/P4.
  gap = np.where(rng.random(n) < 0.1,
                 np.abs(rng.normal(0.0, 0.05, n)),
                 0.8 * rng.lognormal(0.0, 0.25, n) * np.median(e) / e)
  mu = np.zeros((n, m))
  mu[:, 1:] = -np.outer(gap, np.linspace(1.0, 3.0, m - 1))

  # Member beliefs: value-relevant spread e, common-mode shift, eval noise.
  eps = rng.standard_normal((n, k, m))
  cshift = common_sd * rng.standard_normal((n, k))
  qtrue = mu[:, None, :] + e[:, None, None] * eps + cshift[:, :, None]
  qfull = qtrue + tau * rng.standard_normal((n, k, m))
  qhalf = qtrue + tau * np.sqrt(2) * rng.standard_normal((n, k, m))

  # Dynamics disagreement: the shared driver plus, under a real dose, a
  # value-irrelevant distractor component independent of the gap process.
  udyn = e * rng.lognormal(0.0, 0.2, n)
  if dose and not null:
    udyn = udyn + dose * np.median(e) * rng.lognormal(0.0, 0.5, n)

  return signals.compute(qfull, qhalf, udyn), episodes


# Dose grid mirroring App. A.2: (dim, scale) -> synthetic intensity.
DOSES = {'E1': 0.0, 'd8x1': 0.7, 'd32x1': 2.0, 'd32x3': 5.0}


def make_task(null=False, seed=0, **kwargs):
  """Full dose grid for one synthetic task; returns {cell: (sig, eps)}."""
  return {name: make_cell(dose, null=null, seed=seed + i, **kwargs)
          for i, (name, dose) in enumerate(DOSES.items())}
