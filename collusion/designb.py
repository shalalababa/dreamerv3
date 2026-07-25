"""Design-B viability probe (pilot stage 3, Design memo 24 Jul).

Design B trains an agent OFFLINE on the logged transition stream of a
coupled session, enabling composition interventions as stream edits
(drop / reweight segments) with everything else held fixed. Viability
rests on an exactness anchor: tabular Q-learning is deterministic given
the (s, a, r, s') stream and alpha, so replaying the UNTOUCHED log must
reproduce the online agent's Q exactly (float-exact). Interventions
then edit the stream; the retrained agent is paired against the
partner's ONLINE final Q for greedy evaluation + impulse response.

Pilot probe (no confirmatory claim): per session, (1) assert exact
reproduction; (2) drop punishment-phase transitions (undercut label,
W=5) from agent 0's stream, retrain, re-evaluate. Reported: Delta
profit and fingerprint, intact vs intervened. The confirmatory
intervention set and tolerances are fixed in a later prereg.

Usage:
  python -m collusion.designb probe --sessions 20 --max_iters 2000000 \
      --output out.csv
  python -m collusion.designb --selfcheck
"""

import argparse
import csv

import numpy as np

from collusion.env import Duopoly, DELTA
from collusion.pilot import (ALPHA, BETA, CONV_WINDOW, PUNISH_W, Session,
                             collusion_indices, impulse_response, init_q)

PROBE_VERSION = "collusion_designb_v1_20260725"


class RecordedSession(Session):
  """Session that also logs both agents' (s, a, r, s') streams."""

  def __init__(self, env, seed, alpha=ALPHA, beta=BETA):
    super().__init__(env, seed, alpha=alpha, beta=beta)
    self.streams = ([], [])

  def step(self):
    s = self.state
    acts, rewards = super().step()
    for i in range(2):
      self.streams[i].append((s, acts[i], rewards[i], self.state))
    return acts, rewards


def replay_stream(env, stream, alpha=ALPHA):
  """Offline Q from an ordered transition stream (deterministic)."""
  q = init_q(env)
  for s, a, r, ns in stream:
    target = r + DELTA * float(np.max(q[ns]))
    q[s, a] += alpha * (target - q[s, a])
  return q


def punish_mask(stream, env, window=PUNISH_W):
  """Undercut-window punishment label on the stream's own states.

  The state index encodes last-period prices (a0*n + a1); an undercut
  event at t is min price at t strictly below min price at t-1 (same
  labeler as pilot.experience_stats, expressed on states)."""
  arr = np.asarray([env.unstate(s) for s, _, _, _ in stream], np.int64)
  prev_min = np.minimum(arr[:-1, 0], arr[:-1, 1])
  cur_min = np.minimum(arr[1:, 0], arr[1:, 1])
  under = np.zeros(len(arr), bool)
  under[1:] = cur_min < prev_min
  phase = np.zeros(len(arr), bool)
  for i in np.flatnonzero(under):
    phase[i:i + window] = True
  return phase


def evaluate_pair(env, q0, q1, state):
  qs = [q0, q1]
  dpi, dpr = collusion_indices(env, qs, state)
  ir = impulse_response(env, qs, state)
  return dict(delta_profit=dpi, delta_price=dpr,
              fingerprint=ir["fingerprint"],
              punish_depth=ir["punish_depth"])


def probe_session(seed, max_iters, conv_window=CONV_WINDOW, env=None):
  env = env or Duopoly()
  sess = RecordedSession(env, seed)
  t_end, converged = sess.run(max_iters, conv_window)
  online = evaluate_pair(env, sess.q[0], sess.q[1], sess.state)

  # (1) exactness anchor: untouched replay reproduces the online Q0.
  q0_replay = replay_stream(env, sess.streams[0])
  exact = bool(np.array_equal(q0_replay, sess.q[0]))
  replay = evaluate_pair(env, q0_replay, sess.q[1], sess.state)

  # (2) composition intervention: drop punishment-phase transitions.
  mask = punish_mask(sess.streams[0], env)
  kept = [tr for tr, m in zip(sess.streams[0], mask) if not m]
  q0_nopunish = replay_stream(env, kept)
  nop = evaluate_pair(env, q0_nopunish, sess.q[1], sess.state)

  # (3) size-matched RANDOM-drop control (same #transitions removed,
  # composition preserved in expectation) — separates "less punishment
  # data" from "less data".
  rng = np.random.default_rng(10_000 + seed)
  rmask = np.zeros(len(mask), bool)
  rmask[rng.choice(len(mask), int(mask.sum()), replace=False)] = True
  kept_r = [tr for tr, m in zip(sess.streams[0], rmask) if not m]
  q0_rand = replay_stream(env, kept_r)
  rnd = evaluate_pair(env, q0_rand, sess.q[1], sess.state)

  return dict(
      seed=seed, iters=t_end, converged=converged,
      exact_replay=exact,
      dp_online=online["delta_profit"], fp_online=online["fingerprint"],
      dp_replay=replay["delta_profit"], fp_replay=replay["fingerprint"],
      dropped_frac=float(mask.mean()),
      dp_nopunish=nop["delta_profit"], fp_nopunish=nop["fingerprint"],
      punish_depth_nopunish=nop["punish_depth"],
      dp_randdrop=rnd["delta_profit"], fp_randdrop=rnd["fingerprint"])


def selfcheck():
  env = Duopoly()
  # Exactness anchor on a real (tiny) coupled run.
  sess = RecordedSession(env, seed=3)
  sess.run(20_000, conv_window=10 ** 9)
  q0 = replay_stream(env, sess.streams[0])
  q1 = replay_stream(env, sess.streams[1])
  assert np.array_equal(q0, sess.q[0]), "agent-0 replay not exact"
  assert np.array_equal(q1, sess.q[1]), "agent-1 replay not exact"
  # Determinism of the recorded session itself.
  sess2 = RecordedSession(env, seed=3)
  sess2.run(20_000, conv_window=10 ** 9)
  assert sess.streams[0] == sess2.streams[0]
  # Dropping transitions changes the learned Q (intervention bites)...
  mask = punish_mask(sess.streams[0], env)
  if 0 < mask.sum() < len(mask):
    kept = [tr for tr, m in zip(sess.streams[0], mask) if not m]
    q0b = replay_stream(env, kept)
    assert not np.array_equal(q0b, sess.q[0])
  # ... while an empty intervention is the identity.
  assert np.array_equal(replay_stream(env, sess.streams[0]), sess.q[0])
  # punish_mask agrees with the pilot labeler on a hand path.
  C, P = 12, 0
  states = [env.state(C, C)] * 10 + [env.state(P, C)] + [env.state(C, C)] * 5
  stream = [(s, 0, 0.0, s) for s in states]
  m = punish_mask(stream, env)
  assert m.sum() == PUNISH_W and m[10] and not m[9], m
  # probe_session end-to-end shape.
  out = probe_session(seed=5, max_iters=20_000, conv_window=10 ** 9, env=env)
  assert out["exact_replay"] and np.isfinite(out["dp_nopunish"])
  assert out["dp_replay"] == out["dp_online"]
  print("selfcheck PASS: float-exact stream replay (both agents), "
        "deterministic recording, intervention bites / identity holds, "
        "mask matches labeler on a hand path, probe runs end-to-end")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", nargs="?", choices=("probe",))
  ap.add_argument("--sessions", type=int, default=20)
  ap.add_argument("--seed0", type=int, default=0)
  ap.add_argument("--max_iters", type=int, default=2_000_000)
  ap.add_argument("--conv_window", type=int, default=CONV_WINDOW)
  ap.add_argument("--output", default="")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.cmd == "probe" and args.output
  env = Duopoly()
  rows = []
  for s in range(args.seed0, args.seed0 + args.sessions):
    row = probe_session(s, args.max_iters, args.conv_window, env=env)
    rows.append(row)
    print(f"seed {s}: exact={row['exact_replay']} "
          f"dP {row['dp_online']:+.3f} -> nopunish {row['dp_nopunish']:+.3f} "
          f"(dropped {row['dropped_frac']:.2f}) "
          f"fp {row['fp_online']} -> {row['fp_nopunish']}", flush=True)
  with open(args.output, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  ex = float(np.mean([r["exact_replay"] for r in rows]))
  print(f"[{PROBE_VERSION}] sessions={len(rows)} exact_replay_rate={ex:.2f}")


if __name__ == "__main__":
  main()
