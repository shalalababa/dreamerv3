"""Per-domain regime (occupancy) specifications -- single source of truth.

The causal design (plan Sec. 3.2) needs, for every decoupled domain:

  * a **coverage** vector: the *body-state* observation components, which must be
    task-agnostic -- so goal/target observation keys are excluded, otherwise
    "coverage" would leak task-goal proximity and collapse into occupancy;
  * a **mechanism-derived regime** quantity R^phys(s): a physical scalar that is
    *not* defined by return (ball-cup distance, fingertip-target distance,
    spinner-angle error), from which occupancy Occ(B,T)=Pr[s in R_T] is computed.

Verified against the loaded MuJoCo models (see probing/inspect_domain.py):

  cup_catch (ball_in_cup): obs position=[cup_x,cup_z,ball_x,ball_z], velocity(4).
    regime  = ||position[:2]-position[2:4]|| (ball-cup distance); caught = small.
  finger_turn_hard: obs position(4),velocity(3),touch(2),target_position(2),
    dist_to_target(scalar). regime = dist_to_target; at-target = small.
    Coverage excludes target_position & dist_to_target (goal info).
  reacher_hard: obs position(2),to_target(2),velocity(2).
    regime = ||to_target||; at-target = small. Coverage excludes to_target.

All regime quantities are recoverable from the proprio obs alone, so occupancy
can be measured directly on the training replay -- no extra physics logging.

Thresholds here are *provisional*: Gate 0 (Phase 3) calibrates the absolute
numbers to each pilot's observed scale, but the rule (which quantity, which
direction) is fixed now.  Every threshold below is well inside the random-rollout
range measured by inspect_domain.py, i.e. the in-regime corner is a small subset.
"""

import numpy as np

# Observation keys that encode task goal/target information and therefore must
# never enter the (task-agnostic) coverage vector.  Used as a blacklist even for
# domains not in REGIMES.
GOAL_KEYS = frozenset({'target_position', 'dist_to_target', 'to_target'})


def domain_of(task):
  """'dmc_cup_catch' / 'cup_catch' / config.task -> canonical dm_control domain."""
  d = task[len('dmc_'):] if task.startswith('dmc_') else task
  domain = d.split('_', 1)[0]
  return 'ball_in_cup' if domain == 'cup' else domain


def _ball_cup_distance(frames):
  pos = np.asarray(frames['position'], np.float32)      # (N,4)
  return np.linalg.norm(pos[:, :2] - pos[:, 2:4], axis=1)


def _finger_dist_to_target(frames):
  return np.asarray(frames['dist_to_target'], np.float32).reshape(-1)


def _reacher_to_target(frames):
  return np.linalg.norm(np.asarray(frames['to_target'], np.float32), axis=1)


# name         : short label for the regime scalar (for logging/plots)
# fn           : frames-dict -> (N,) regime quantity
# needs        : obs keys the fn reads (must be present to compute occupancy)
# threshold    : provisional in-regime cutoff (calibrated in Gate 0)
# direction    : 'below' -> in-regime if value < threshold, else 'above'
# coverage_keys: body-state obs keys for coverage (None = all non-goal keys)
REGIMES = {
    'ball_in_cup': dict(
        name='ball_cup_dist', fn=_ball_cup_distance, needs=('position',),
        threshold=0.05, direction='below',
        coverage_keys=('position', 'velocity')),
    # finger dist_to_target = ||tip-target|| - target_radius (SIGNED): <=0 means
    # the tip is inside the target sphere (this is exactly the sparse reward
    # condition, `float(dist_to_target <= 0)`).  0.0 is the physical boundary.
    'finger': dict(
        name='dist_to_target', fn=_finger_dist_to_target,
        needs=('dist_to_target',), threshold=0.0, direction='below',
        coverage_keys=('position', 'velocity', 'touch')),
    # reacher_hard target size 0.015; reward fires at ||to_target|| <= (target +
    # finger radii) ~= 0.025.  0.05 is the *easy* target, so use ~0.025 here.
    'reacher': dict(
        name='to_target_norm', fn=_reacher_to_target, needs=('to_target',),
        threshold=0.025, direction='below',
        coverage_keys=('position', 'velocity')),
    # synth_reach (embodied/envs/synthpred.py): reward fires at
    # ||to_target|| < radius (default 0.1); threshold == default radius,
    # so regime membership == the sparse reward condition, exactly like
    # finger. Change only together with env.synth.radius.
    'synth': dict(
        name='to_target_norm', fn=_reacher_to_target, needs=('to_target',),
        threshold=0.1, direction='below',
        coverage_keys=('position', 'velocity', 'distractor')),
}


def has_regime(task):
  return domain_of(task) in REGIMES


def spec(task):
  domain = domain_of(task)
  if domain not in REGIMES:
    raise KeyError(
        f'No regime spec for domain {domain!r} (task {task!r}). Known: '
        f'{sorted(REGIMES)}. Add one to probing/regimes.py.')
  return REGIMES[domain]


def coverage_keys(task, available_keys):
  """Body-state keys to use for the coverage vector, given what a buffer has.

  Prefers the domain's declared coverage_keys (intersected with what is present);
  otherwise falls back to all available keys minus the goal blacklist.
  """
  available = list(available_keys)
  domain = domain_of(task) if task else None
  if domain in REGIMES and REGIMES[domain]['coverage_keys']:
    keys = [k for k in REGIMES[domain]['coverage_keys'] if k in available]
    if keys:
      return sorted(keys)
  return sorted(k for k in available if k not in GOAL_KEYS)


def regime_values(task, frames):
  """(N,) mechanism-derived regime quantity for a frames dict {key: (N,d)}."""
  s = spec(task)
  missing = [k for k in s['needs'] if k not in frames]
  if missing:
    raise KeyError(f'regime for {task!r} needs obs keys {missing} not in frames '
                   f'(have {sorted(frames)}).')
  return s['fn'](frames)


def in_regime(task, frames, threshold=None):
  """Boolean (N,) mask of frames inside the regime R^phys_T."""
  s = spec(task)
  vals = regime_values(task, frames)
  thr = s['threshold'] if threshold is None else threshold
  return vals < thr if s['direction'] == 'below' else vals > thr


def occupancy(task, frames, threshold=None):
  """Occ(B,T) = fraction of frames inside the mechanism-derived regime."""
  mask = in_regime(task, frames, threshold)
  return float(mask.mean())
