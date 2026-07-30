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
                             collusion_indices, greedy_path,
                             impulse_response, init_q)

PROBE_VERSION = "collusion_designb_v2_20260730"
# v2 (confirmatory-prereg calibration, 30 Jul): adds the DISTANCE-label
# stream mask (terminal or exogenous reference), weighted replay
# (graded per-transition down-weighting), and the `graded`
# volume-controlled probe (target arm = down-weight punishment-phase
# transitions; control arm = down-weight a size-matched random subset
# at the same weight, so effective update mass is identical by
# construction). The v1 `probe` subcommand is computed identically
# (validated: seeds 0-4 rerun matches designb_probe20.csv exactly).


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


def replay_stream_weighted(env, stream, weights, alpha=ALPHA):
  """Offline Q with per-transition effective learning rate alpha*w.

  w=1 everywhere reproduces replay_stream exactly; w=0 on a subset is
  Q-identical to deleting those transitions (a zero-rate update is a
  no-op and the remaining updates read the same Q values in the same
  order) — which makes down-weighting the volume-exact graded form of
  the pilot's deletion intervention."""
  q = init_q(env)
  for (s, a, r, ns), w in zip(stream, weights):
    if w == 0.0:
      continue
    target = r + DELTA * float(np.max(q[ns]))
    q[s, a] += (alpha * w) * (target - q[s, a])
  return q


def terminal_ref_min(env, qs, state, T=100):
  """Reference price index: min of the session's terminal greedy path
  (the pilot distance label's reference)."""
  path, _ = greedy_path(env, qs, state, T)
  return int(np.min(np.asarray(path)))


def exo_ref_min(env):
  """Exogenous reference: the monopoly grid index (session-independent)."""
  return int(np.argmin(np.abs(env.prices - env.p_mono)))


def dist_mask(stream, env, ref_min):
  """Distance-label punishment mask on a transition stream.

  Transition t carries this period's realized prices in its NEXT state
  ns (state = last joint prices), so the mask on ns reproduces the
  pilot labeler (distance_stats on actions_log) index-for-index."""
  arr = np.asarray([env.unstate(ns) for _, _, _, ns in stream], np.int64)
  cur_min = np.minimum(arr[:, 0], arr[:, 1])
  return cur_min < ref_min


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


def graded_session(seed, doses, max_iters, conv_window=CONV_WINDOW,
                   env=None, reference="terminal"):
  """Graded volume-controlled composition intervention (v2).

  Per dose d: target arm down-weights distance-labeled punishment
  transitions to w=1-d; control arm down-weights a size-matched random
  subset to the same w. Total effective update mass is equal across
  the two arms by construction (asserted). d=1 is Q-identical to the
  pilot's deletion intervention under the distance label."""
  env = env or Duopoly()
  sess = RecordedSession(env, seed)
  t_end, converged = sess.run(max_iters, conv_window)
  online = evaluate_pair(env, sess.q[0], sess.q[1], sess.state)
  exact = bool(np.array_equal(replay_stream(env, sess.streams[0]),
                              sess.q[0]))
  assert reference in ("terminal", "exo"), reference
  ref = (terminal_ref_min(env, sess.q, sess.state)
         if reference == "terminal" else exo_ref_min(env))
  mask = dist_mask(sess.streams[0], env, ref)
  n_mask = int(mask.sum())
  row = dict(
      seed=seed, iters=t_end, converged=converged, exact_replay=exact,
      reference=reference, ref_min_idx=ref,
      dp_online=online["delta_profit"], fp_online=online["fingerprint"],
      dpr_online=online["delta_price"],
      mask_frac=float(mask.mean()))
  for d in doses:
    w = 1.0 - d
    wt = np.ones(len(mask))
    wt[mask] = w
    q0_t = replay_stream_weighted(env, sess.streams[0], wt)
    tgt = evaluate_pair(env, q0_t, sess.q[1], sess.state)
    # size-matched random control: same COUNT down-weighted by the
    # same factor => identical total effective mass (volume control).
    rng = np.random.default_rng(
        20_000_000 + seed * 1000 + int(round(d * 100)))
    rmask = np.zeros(len(mask), bool)
    rmask[rng.choice(len(mask), n_mask, replace=False)] = True
    wc = np.ones(len(mask))
    wc[rmask] = w
    assert wt.sum() == wc.sum(), "volume-control invariant violated"
    q0_c = replay_stream_weighted(env, sess.streams[0], wc)
    ctl = evaluate_pair(env, q0_c, sess.q[1], sess.state)
    tag = f"d{int(round(d * 100)):03d}"
    row[f"dp_punish_{tag}"] = tgt["delta_profit"]
    row[f"fp_punish_{tag}"] = tgt["fingerprint"]
    row[f"dpr_punish_{tag}"] = tgt["delta_price"]
    row[f"dp_random_{tag}"] = ctl["delta_profit"]
    row[f"fp_random_{tag}"] = ctl["fingerprint"]
    row[f"dpr_random_{tag}"] = ctl["delta_price"]
  return row


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
  # ---- v2 additions ----
  # Weighted replay: w=1 everywhere is the identity...
  ones = np.ones(len(sess.streams[0]))
  assert np.array_equal(replay_stream_weighted(env, sess.streams[0], ones),
                        sess.q[0]), "w=1 weighted replay must be exact"
  # ... and w=0 on a subset is Q-identical to deleting that subset.
  ref = terminal_ref_min(env, sess.q, sess.state)
  dmask = dist_mask(sess.streams[0], env, ref)
  wt = np.ones(len(dmask))
  wt[dmask] = 0.0
  kept_d = [tr for tr, m in zip(sess.streams[0], dmask) if not m]
  assert np.array_equal(replay_stream_weighted(env, sess.streams[0], wt),
                        replay_stream(env, kept_d)), "w=0 != deletion"
  # dist_mask reproduces the pilot distance labeler index-for-index on
  # the real run (ns carries this period's prices).
  from collusion.pilot import distance_stats
  ds = distance_stats(env, sess.actions_log, sess.q, sess.state)
  arr = np.asarray(sess.actions_log, np.int64)
  phase_pilot = np.minimum(arr[:, 0], arr[:, 1]) < ds["ref_min_idx"]
  assert ds["ref_min_idx"] == ref
  assert np.array_equal(dmask, phase_pilot), "stream/pilot label mismatch"
  # exo reference is the monopoly grid index.
  mono_idx = int(np.argmin(np.abs(env.prices - env.p_mono)))
  assert exo_ref_min(env) == mono_idx
  # graded_session end-to-end: dose 0 is the identity on both arms;
  # dose 1 target reproduces the distance-label deletion exactly.
  g = graded_session(seed=3, doses=(0.0, 1.0), max_iters=20_000,
                     conv_window=10 ** 9, env=env)
  assert g["exact_replay"]
  assert g["dp_punish_d000"] == g["dp_online"], g
  assert g["dp_random_d000"] == g["dp_online"], g
  assert g["dpr_punish_d000"] == g["dpr_online"], g
  assert np.isfinite(g["dpr_punish_d100"]) and np.isfinite(g["dpr_online"])
  del_eval = evaluate_pair(env, replay_stream(env, kept_d), sess.q[1],
                           sess.state)
  assert g["dp_punish_d100"] == del_eval["delta_profit"], g
  print("selfcheck PASS: float-exact stream replay (both agents), "
        "deterministic recording, intervention bites / identity holds, "
        "mask matches labeler on a hand path, probe runs end-to-end; "
        "v2: weighted-replay identity + w0==deletion, dist_mask == "
        "pilot label, graded dose-0 identity / dose-1 == deletion")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", nargs="?", choices=("probe", "graded"))
  ap.add_argument("--sessions", type=int, default=20)
  ap.add_argument("--seed0", type=int, default=0)
  ap.add_argument("--max_iters", type=int, default=2_000_000)
  ap.add_argument("--conv_window", type=int, default=CONV_WINDOW)
  ap.add_argument("--doses", default="0.25,0.5,1.0")
  ap.add_argument("--reference", default="terminal",
                  choices=("terminal", "exo"))
  ap.add_argument("--output", default="")
  ap.add_argument("--selfcheck", action="store_true")
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  assert args.cmd in ("probe", "graded") and args.output
  env = Duopoly()
  rows = []
  for s in range(args.seed0, args.seed0 + args.sessions):
    if args.cmd == "probe":
      row = probe_session(s, args.max_iters, args.conv_window, env=env)
      print(f"seed {s}: exact={row['exact_replay']} "
            f"dP {row['dp_online']:+.3f} -> nopunish {row['dp_nopunish']:+.3f} "
            f"(dropped {row['dropped_frac']:.2f}) "
            f"fp {row['fp_online']} -> {row['fp_nopunish']}", flush=True)
    else:
      doses = tuple(float(x) for x in args.doses.split(","))
      row = graded_session(s, doses, args.max_iters, args.conv_window,
                           env=env, reference=args.reference)
      top = f"d{int(round(max(doses) * 100)):03d}"
      print(f"seed {s}: exact={row['exact_replay']} "
            f"mask={row['mask_frac']:.2f} dP {row['dp_online']:+.3f} -> "
            f"punish[{top}] {row['dp_punish_' + top]:+.3f} vs "
            f"random[{top}] {row['dp_random_' + top]:+.3f}", flush=True)
    rows.append(row)
  with open(args.output, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  ex = float(np.mean([r["exact_replay"] for r in rows]))
  print(f"[{PROBE_VERSION}] sessions={len(rows)} exact_replay_rate={ex:.2f}")


if __name__ == "__main__":
  main()
