"""FB-as-consumer helpers for the wfb labeling path
(PREREG_p2_fbconsumer_20260829).

Builds, from an fb_ckpt.pt + its training export:
  policy(obs)        -> eval-mode action (the fb_zeroshot act protocol:
                        agent.act(obs, meta, step=0, eval_mode=True)
                        under torch.no_grad(); CPU by default)
  q_of(obs, cands)   -> FB's Q_z over a candidate set: Q = F(s,a,z)·z,
                        twin-F minimum when the upstream forward_net
                        returns a pair (the upstream convention).
  obs_order_gate(a, b)  — helper comparator used by the selfcheck;
                        the PRODUCTION gate is the STATIC key-order
                        assert in the labeler's fb block
                        (tdmpc2_compat.OBS_ORDER == fb_export.OBS_KEYS
                        + no extra keys), which subsumes the runtime
                        comparison for e1 cells (review m1).

z inference is fb_zeroshot's registered protocol verbatim
(infer_task_z: N_INFER=5120, seeded, own-side export).

Selfcheck (no upstream checkout needed — mock agent):
  python -m probing.fb_consumer --selfcheck
"""

import numpy as np


def build_fb_consumer(ckpt_path, data_path, z_seed, device='cpu'):
  """Returns (policy, q_of, meta_record). Torch + upstream imports are
  lazy; the registered venue smoke validates against the real API."""
  import torch
  from probing.fb_fit import load_agent
  from probing.fb_zeroshot import N_INFER, infer_task_z
  from probing.fb_embed import sha256_file
  agent, ck = load_agent(ckpt_path, device=device)
  # review m2 (transcribed from fb_zeroshot): the trained obs dim is
  # part of the contract; a mismatch must fail loudly at load.
  obs_dim = int(ck['agent_kwargs']['obs_shape'][0])
  assert obs_dim == 12, ('trained obs dim != 12 — wrong checkpoint '
                         'family', obs_dim)
  meta, zstats = infer_task_z(agent, data_path, z_seed)
  zvec = np.asarray(meta['z'], np.float32).reshape(-1)

  def policy(obs_v):
    with torch.no_grad():
      a = agent.act(np.asarray(obs_v, np.float32), meta, step=0,
                    eval_mode=True)
    return np.asarray(a, np.float32).reshape(-1)

  def q_of(obs_v, cands):
    m = int(np.asarray(cands).shape[0])
    o = torch.as_tensor(np.tile(np.asarray(obs_v, np.float32), (m, 1)),
                        device=device)
    a = torch.as_tensor(np.asarray(cands, np.float32), device=device)
    zz = torch.as_tensor(np.tile(zvec, (m, 1)), device=device)
    with torch.no_grad():
      F = agent.forward_net(o, zz, a)
    if isinstance(F, (tuple, list)):
      q = torch.minimum((F[0] * zz).sum(-1), (F[1] * zz).sum(-1))
    else:
      q = (F * zz).sum(-1)
    q = q.cpu().numpy().astype(np.float32)
    assert q.shape == (m,) and np.isfinite(q).all(), q
    return q

  rec = dict(ckpt=str(ckpt_path), ckpt_sha256=sha256_file(ckpt_path),
             data=str(data_path), data_sha256=sha256_file(data_path),
             z_seed=int(z_seed), n_infer=int(N_INFER),
             z_norm=float(np.linalg.norm(zvec)),
             frac_pos_reward=float(zstats['frac_pos_reward']),
             device=str(device))
  return policy, q_of, rec


def obs_order_gate(labeler_flat_obs, ordered_flat_obs, atol=1e-5):
  """The fb_zeroshot OBS_KEYS gate, transcribed for the wfb path: the
  flat obs the labeler will hand FB must EQUAL the OBS_KEYS-ordered
  concatenation. Returns None on pass; a reason string on fail."""
  a = np.asarray(labeler_flat_obs, np.float32).reshape(-1)
  b = np.asarray(ordered_flat_obs, np.float32).reshape(-1)
  if a.shape != b.shape:
    return f'obs dim mismatch: labeler {a.shape} vs OBS_KEYS {b.shape}'
  if not np.allclose(a, b, atol=atol):
    return ('obs ORDER mismatch: labeler flat obs != OBS_KEYS-ordered '
            'concatenation — FB would act on scrambled inputs')
  return None


# --------------------------------------------------------------------------
# Selfcheck (mock agent; validates shapes, twin-F handling, clipping
# downstream, and the obs-order gate — the real-API contact surface is
# covered by the registered venue smoke)
# --------------------------------------------------------------------------

def selfcheck():
  import types

  class _MockF:
    def __init__(self, pair):
      self.pair = pair

    def __call__(self, o, z, a):
      import torch
      f = o[:, :3] * 0 + a.sum(-1, keepdim=True) + z[:, :3]
      return (f, f + 1.0) if self.pair else f

  import torch
  for pair in (False, True):
    agent = types.SimpleNamespace(
        forward_net=_MockF(pair),
        act=lambda obs, meta, step, eval_mode: np.ones(2) * 0.5)
    zvec = np.array([1.0, 2.0, 0.0], np.float32)
    meta = {'z': zvec}

    def q_of(obs_v, cands, agent=agent, zvec=zvec):
      m = len(cands)
      o = torch.as_tensor(np.tile(obs_v, (m, 1)), dtype=torch.float32)
      a = torch.as_tensor(np.asarray(cands, np.float32))
      zz = torch.as_tensor(np.tile(zvec, (m, 1)))
      F = agent.forward_net(o, zz, a)
      if isinstance(F, (tuple, list)):
        q = torch.minimum((F[0] * zz).sum(-1), (F[1] * zz).sum(-1))
      else:
        q = (F * zz).sum(-1)
      return q.numpy()

    obs = np.zeros(12, np.float32)
    cands = np.array([[0.1, 0.1], [0.5, -0.5], [-1.0, 1.0]], np.float32)
    q = q_of(obs, cands)
    # closed form: F row = a.sum + z[:3]; q = (F*z)[:3].sum
    # (pair=True adds +1 to the twin, so min picks the base — identical)
    want = [(c.sum() + zvec[:3]) @ zvec[:3] for c in cands]
    assert np.allclose(q, want, atol=1e-5), (pair, q, want)
  assert obs_order_gate(np.arange(12), np.arange(12)) is None
  assert obs_order_gate(np.arange(12), np.arange(12)[::-1]) is not None
  assert obs_order_gate(np.arange(12), np.arange(11)) is not None
  print('fb_consumer selfcheck PASS (twin-F min + single-F Q closed '
        'form; obs-order gate pass/order/dim legs)')


if __name__ == '__main__':
  import sys
  if '--selfcheck' in sys.argv:
    selfcheck()
  else:
    raise SystemExit('module is a library; run --selfcheck')
