"""Per-state Gate D0 signals from ensemble Q matrices.

Input convention everywhere: q has shape (N, K, M) -- N states, K ensemble
members, M candidate actions. Definitions are frozen to match
EVPI_theory_note_20260702.tex App. A.4 and the synthetic estimator check
(research_notes/other research/evpi_estimator_check_20260702.py).
"""

import itertools

import numpy as np


def splits(k):
  """All near-half partitions of members into (select, evaluate) halves.

  For K = 5 these are the 20 (2/3 and 3/2) partitions; full averaging over
  them is essential (it cancels common-mode noise exactly; note Sec. 5.3).
  """
  out = []
  for size in (k // 2, k - k // 2):
    for s1 in itertools.combinations(range(k), size):
      s2 = tuple(i for i in range(k) if i not in s1)
      out.append((s1, s2))
  # k even: the two size loops generate each partition twice; deduplicate.
  return sorted(set(out))


def evpi_plugin(q):
  """T1 - T2: mean of member maxima minus max of member means."""
  t1 = q.max(2).mean(1)
  t2 = q.mean(1).max(1)
  return t1 - t2


def evpi_split(q):
  """T1 - mean over all partitions of the split-selection T2'."""
  t1 = q.max(2).mean(1)
  t2p = []
  for s1, s2 in splits(q.shape[1]):
    a_hat = q[:, s1, :].mean(1).argmax(1)
    t2p.append(np.take_along_axis(
        q[:, s2, :].mean(1), a_hat[:, None], 1)[:, 0])
  return t1 - np.mean(t2p, 0)


def u_q(q):
  """U_Q = max_a sd_k Q_k(s, a), in sd units."""
  return q.std(1, ddof=1).max(1)


def a_flip(q):
  """Fraction of members whose argmax differs from the ensemble-mean argmax.

  At K = 5 this takes values in {0, 0.2, ..., 1}: coarse per state, fine in
  aggregate (note App. A.4).
  """
  ens_arg = q.mean(1).argmax(1)
  return (q.argmax(2) != ens_arg[:, None]).mean(1)


def advantage_gap(q):
  """Ensemble-mean top-2 advantage gap (near-tie identification for P2)."""
  m = np.sort(q.mean(1), 1)
  return m[:, -1] - m[:, -2]


def compute(qfull, qhalf=None, udyn=None):
  """All per-state signals from the raw sweep arrays.

  Returns a dict of (N,) arrays. Signals from the half-rollout-count Q
  matrix carry a `_half` suffix (they expose the sign of the
  noisy-evaluation term of note Sec. 5.2).
  """
  out = dict(
      uq=u_q(qfull),
      aflip=a_flip(qfull),
      evpi_plugin=evpi_plugin(qfull),
      evpi_split=evpi_split(qfull),
      gap=advantage_gap(qfull),
  )
  if qhalf is not None:
    out.update(
        evpi_plugin_half=evpi_plugin(qhalf),
        evpi_split_half=evpi_split(qhalf),
    )
  if udyn is not None:
    out['udyn'] = np.asarray(udyn, np.float64)
  return out
