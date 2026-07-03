import numpy as np

from embodied.envs.distractor import Distractor
from embodied.envs.dummy import Dummy


def make_env(dim=8, scale=1.0, seed=0, **kwargs):
  return Distractor(Dummy('cont', length=100), dim, scale, seed=seed, **kwargs)


def rollout(env, steps, actions=None):
  obs = env.step({'reset': True, 'act_disc': np.int32(0),
                  'act_cont': np.zeros(6, np.float32)})
  out = [obs]
  for i in range(steps - 1):
    act = actions[i] if actions else np.zeros(6, np.float32)
    obs = env.step({'reset': False, 'act_disc': np.int32(0), 'act_cont': act})
    out.append(obs)
  return out


class TestDistractor:

  def test_obs_space(self):
    env = make_env(dim=8)
    assert env.obs_space['distractor'].shape == (8,)
    assert env.obs_space['distractor'].dtype == np.float32
    assert 'vector' in env.obs_space

  def test_reward_untouched(self):
    plain = Dummy('cont', length=100)
    wrapped = make_env(dim=8)
    r1 = [o['reward'] for o in rollout(plain, 20)]
    r2 = [o['reward'] for o in rollout(wrapped, 20)]
    assert r1 == r2

  def test_action_independence(self):
    rng = np.random.default_rng(1)
    acts = [rng.uniform(-1, 1, 6).astype(np.float32) for _ in range(49)]
    a = [o['distractor'] for o in rollout(make_env(seed=7), 50)]
    b = [o['distractor'] for o in rollout(make_env(seed=7), 50, acts)]
    assert all((x == y).all() for x, y in zip(a, b))

  def test_seed_determinism(self):
    a = [o['distractor'] for o in rollout(make_env(seed=3), 50)]
    b = [o['distractor'] for o in rollout(make_env(seed=3), 50)]
    c = [o['distractor'] for o in rollout(make_env(seed=4), 50)]
    assert all((x == y).all() for x, y in zip(a, b))
    assert any((x != y).any() for x, y in zip(a, c))

  def test_stationary_scale(self):
    # With basesd fixed, the emitted sd must approach scale * basesd.
    env = make_env(dim=32, scale=3.0, basesd=0.5, theta=0.1, seed=0)
    xs = []
    for _ in range(40):
      xs += [o['distractor'] for o in rollout(env, 100)]
    sd = np.stack(xs).std()
    assert abs(sd - 1.5) < 0.1, sd

  def test_reset_redraws_state(self):
    env = make_env(dim=8, basesd=1.0, seed=5)
    first = rollout(env, 3)[0]['distractor']
    again = rollout(env, 3)[0]['distractor']
    assert (first != again).any()

  def test_calibration_freezes(self):
    env = make_env(dim=4, calib=50, seed=2)
    rollout(env, 60)
    assert env.frozen_refsd is not None
    frozen = env.frozen_refsd
    rollout(env, 60)
    assert env.frozen_refsd == frozen

  def test_autocorrelation(self):
    # OU with theta=0.1 has lag-1 autocorrelation ~0.9: learnable dynamics.
    env = make_env(dim=1, basesd=1.0, theta=0.1, seed=9)
    xs = np.array([o['distractor'][0] for o in rollout(env, 100)[1:]])
    for _ in range(30):
      more = np.array([o['distractor'][0] for o in rollout(env, 100)[1:]])
      xs = np.concatenate([xs, more])
    rho = np.corrcoef(xs[:-1], xs[1:])[0, 1]
    assert 0.8 < rho < 0.95, rho
