import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import explore
from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    # model_obs: keys outside this regex stay in obs/replay (available to
    # wrappers, labeling, and analysis) but never enter the world model.
    include = re.compile(config.model_obs)
    enc_space = {k: v for k, v in obs_space.items()
                 if k not in exclude and include.fullmatch(k)}
    dec_space = dict(enc_space)
    assert enc_space, (config.model_obs, sorted(obs_space))
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    # Gate D0 measurement ensemble: K auxiliary critic heads with independent
    # initializations, per-head bootstrap targets, and per-head minibatch
    # masks. Measurement-only: they never feed the policy loss and their
    # loss inputs are stop-gradiented, so they cannot shape the shared
    # representation either (EVPI note App. A.4).
    self.valens = []
    self.slowvalens = []
    if config.valens.k:
      assert config.expl.mode == 'task', (
          'critic-head ensemble requires task reward', config.expl.mode)
      for i in range(config.valens.k):
        headi = embodied.jax.MLPHead(scalar, **config.value, name=f'valens{i}')
        self.valens.append(headi)
        self.slowvalens.append(embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name=f'slowvalens{i}'),
            source=headi, **config.slowvalue))

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.expl_mode = config.expl.mode
    assert self.expl_mode in ('task', 'random', 'p2e', 'apt'), self.expl_mode
    self.reward_free = (self.expl_mode != 'task')

    # P2E predicts deterministic posterior features; sampled stochastic targets
    # add noise that collapses disagreement. disag_task trains the same
    # ensemble in task mode as a passive U_dyn probe (Gate D0).
    assert not config.expl.disag_task or self.expl_mode == 'task', (
        'disag_task is the task-mode measurement flag', self.expl_mode)
    self.disag = None
    if self.expl_mode == 'p2e' or config.expl.disag_task:
      rssm_kw = config.dyn[config.dyn.typ]
      if config.expl.disag_target == 'postfeat':
        target_dim = rssm_kw['deter'] + rssm_kw['stoch'] * rssm_kw['classes']
      elif config.expl.disag_target == 'stoch':
        target_dim = rssm_kw['stoch'] * rssm_kw['classes']
      else:
        raise NotImplementedError(config.expl.disag_target)
      self.disag = explore.Disag(
          target_dim, ensemble=config.expl.disag_ens,
          units=config.expl.disag_units, layers=config.expl.disag_layers,
          name='disag')

    # frozen_enc drops enc from the optimizer's module list (same mechanism
    # as frozen_wm, one level finer): enc params become constants in
    # nj.grad, so no gradient — reconstruction or otherwise — updates them.
    wm = ([self.dyn, self.dec] if config.frozen_enc else
          [self.dyn, self.enc, self.dec])
    head = [self.rew, self.con, self.pol, self.val]
    probes = self.valens + ([self.disag] if config.expl.disag_task else [])
    if self.expl_mode == 'random':
      self.modules = wm
    elif self.expl_mode == 'p2e':
      self.modules = wm + [self.con, self.pol, self.val, self.disag]
    elif self.expl_mode == 'apt':
      self.modules = wm + [self.con, self.pol, self.val]
    elif config.frozen_wm:
      self.modules = head + probes
    else:
      self.modules = wm + head + probes
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    if self.expl_mode == 'random':
      keep = {'dyn', 'rep', *dec_space}
    elif self.expl_mode == 'p2e':
      keep = {'dyn', 'rep', 'con', 'policy', 'value', 'disag', *dec_space}
    elif self.expl_mode == 'apt':
      keep = {'dyn', 'rep', 'con', 'policy', 'value', *dec_space}
    else:
      keep = {'dyn', 'rep', 'rew', 'con', 'policy', 'value', 'repval',
              *dec_space}
      if config.expl.disag_task:
        keep.add('disag')
      if self.valens:
        keep.add('valens')
        if config.repval_loss:
          keep.add('valensrep')
    self.scales = {k: v for k, v in scales.items() if k in keep}

  @property
  def policy_keys(self):
    if self.config.d0.signals:
      # Sweep-time signal extraction runs through the policy path and needs
      # the reward/continuation heads, critic ensemble, and disag ensemble.
      return r'^(enc|dyn|dec|pol|rew|con|val|valens\d+|disag)/'
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    if self.expl_mode == 'random':
      # Uniform random actions for random-data pretraining.
      B = reset.shape[0]
      act = {k: jax.random.uniform(nj.seed(), (B, *v.shape), f32, -1.0, 1.0)
             for k, v in self.act_space.items()}
    else:
      policy = self.pol(self.feat2tensor(feat), bdims=1)
      act = sample(policy)
    out = {}
    if self.config.d0.signals:
      out.update(self._d0_signals(dyn_carry, feat, act))
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    if 'value' in self.scales:
      self.slowval.update()
    if 'valens' in self.scales:
      for slow in self.slowvalens:
        slow.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    if not self.reward_free:
      inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
      losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    if self.expl_mode != 'random':
      con = f32(~obs['is_terminal'])
      if self.config.contdisc:
        con *= 1 - 1 / self.config.horizon
      losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Train the one-step latent-disagreement ensemble on replay.
    if self.disag is not None:
      dfeat = self.feat2tensor(repfeat)[:, :-1]
      dact = self._act2tensor(prevact)[:, 1:]
      dtarget = self._disag_target(repfeat)[:, 1:]
      losses['disag'] = self.disag.loss(
          dfeat, dact, dtarget,
          bootstrap=self.config.expl.disag_bootstrap,
          bootstrap_prob=self.config.expl.disag_bootstrap_prob)
      metrics['expl/disag_loss'] = losses['disag'].mean()
      replay_rew = self.disag.reward(dfeat, dact)
      metrics['expl/disag_replay_rew'] = replay_rew.mean()
      metrics['expl/disag_replay_rew_std'] = replay_rew.std()
      metrics['expl/disag_target_std'] = dtarget.std()

    # Imagination. C2 (random) trains only the world model and skips this.
    if self.expl_mode != 'random':
      K = min(self.config.imag_last or T, T)
      H = self.config.imag_length
      starts = self.dyn.starts(dyn_entries, dyn_carry, K)
      policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
      _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
      first = jax.tree.map(
          lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
      imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
      lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
      lastact = jax.tree.map(lambda x: x[:, None], lastact)
      imgact = concat([imgprevact, lastact], 1)
      assert all(
          x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
      assert all(
          x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
      inp = self.feat2tensor(imgfeat)
      if self.expl_mode == 'p2e':
        raw_imgrew = self.disag.reward(inp, self._act2tensor(imgact))
        imgrew = sg(raw_imgrew * self.config.expl.disag_scale)
        metrics['expl/intr_rew_raw'] = raw_imgrew.mean()
        metrics['expl/intr_rew_raw_std'] = raw_imgrew.std()
        metrics['expl/disag_scale'] = self.config.expl.disag_scale
      elif self.expl_mode == 'apt':
        imgrew = sg(explore.apt_reward(
            inp, self.config.expl.apt_knn, self.config.expl.apt_logc))
      else:
        imgrew = self.rew(inp, 2).pred()
      if self.reward_free:
        metrics['expl/intr_rew'] = imgrew.mean()
        metrics['expl/intr_rew_std'] = imgrew.std()
      imgcon = self.con(inp, 2).prob(1)
      los, imgloss_out, mets = imag_loss(
          imgact,
          imgrew,
          imgcon,
          self.pol(inp, 2),
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.retnorm, self.valnorm, self.advnorm,
          update=training,
          contdisc=self.config.contdisc,
          horizon=self.config.horizon,
          **self.config.imag_loss)
      losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
      metrics.update(mets)

      # Gate D0 critic-head ensemble: per-head lambda-return targets from the
      # head's own bootstrap, per-head minibatch masks over replay sequences.
      if self.valens:
        prob = self.config.valens.bootstrap_prob
        masks = None
        if self.config.valens.bootstrap:
          masks = jax.random.bernoulli(
              nj.seed(), prob, (len(self.valens), B)).astype(f32) / prob
        # Unconditional stop-gradient: the heads are instruments, so their
        # losses must not shape the shared representation even when
        # ac_grads/repval_grad let the main critic's losses do so.
        ens_inp = sg(inp)
        ens_losses, ens_boots, ens_vals = [], [], []
        for i, (headi, slow) in enumerate(zip(self.valens, self.slowvalens)):
          hlos, hret, hval = head_value_loss(
              imgrew, imgcon, headi(ens_inp, 2), slow(ens_inp, 2),
              self.valnorm,
              contdisc=self.config.contdisc,
              horizon=self.config.horizon,
              lam=self.config.imag_loss.lam,
              slowreg=self.config.imag_loss.slowreg,
              slowtar=self.config.imag_loss.slowtar)
          hlos = hlos.mean(1).reshape((B, K))
          if masks is not None:
            hlos = hlos * masks[i][:, None]
          ens_losses.append(hlos)
          ens_boots.append(hret[:, 0].reshape(B, K))
          ens_vals.append(hval)
        losses['valens'] = sum(ens_losses) / len(ens_losses)
        vstack = jnp.stack(ens_vals, 0)
        metrics['valens/mean'] = vstack.mean()
        metrics['valens/std'] = vstack.std(0).mean()

    # Replay
    if self.config.repval_loss and not self.reward_free:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

      # Replay grounding for the critic-head ensemble, mirroring repval with
      # per-head imagination bootstraps and the same per-head masks. The
      # stop-gradient is load-bearing: with repval_grad True the main
      # critic's inp carries world-model gradients, and the measurement
      # heads must not add K auxiliary representation-shaping losses.
      if self.valens:
        ens_inp = sg(inp)
        ens_rep = []
        for i, (headi, slow) in enumerate(zip(self.valens, self.slowvalens)):
          hlos, _, _ = repl_loss(
              last, term, rew, ens_boots[i],
              headi(ens_inp, 2), slow(ens_inp, 2), self.valnorm,
              update=False,
              horizon=self.config.horizon,
              **self.config.repl_loss)
          hl = hlos['repval']
          if masks is not None:
            hl = hl * masks[i][:, None]
          ens_rep.append(hl)
        losses['valensrep'] = sum(ens_rep) / len(ens_rep)

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _d0_signals(self, dyn_carry, feat, act):
    """Per-state Gate D0 raw signals, emitted as policy outputs (App. A.4).

    Returns the per-head, per-candidate-action one-step Q matrices at two
    rollout counts (R and R/2, exposing the noisy-evaluation term of note
    Sec. 5.2) plus dynamics-ensemble disagreement at the executed action.
    Derived signals (U_Q, A, EVPI-hat) are computed offline in d0/signals.py
    so that real and synthetic data share one implementation.
    """
    assert self.valens, 'd0.signals requires valens.k > 0'
    assert self.disag is not None, 'd0.signals requires the disag ensemble'
    assert all(not s.discrete for s in self.act_space.values()), (
        'd0 sweep assumes continuous action spaces (DMC)')
    M, R = self.config.d0.actions, self.config.d0.rollouts
    assert R >= 2 and R % 2 == 0, R
    B = feat['deter'].shape[0]
    nheads = len(self.valens)
    inp = self.feat2tensor(feat)

    # Candidate actions: policy mode plus M - 1 samples.
    policy = self.pol(inp, bdims=1)
    cands = [{k: v.pred() for k, v in policy.items()}]
    cands += [sample(policy) for _ in range(M - 1)]

    # One imagination step per (state, candidate, rollout); flat index is
    # (b, m, r) row-major to match jnp.repeat on the batch axis.
    start = jax.tree.map(lambda x: jnp.repeat(x, M * R, 0), dyn_carry)
    acts = {}
    for k in self.act_space:
      x = jnp.stack([nn.cast(c[k]) for c in cands], 1)      # (B, M, ...)
      x = jnp.repeat(x[:, :, None], R, 2)                   # (B, M, R, ...)
      acts[k] = x.reshape((B * M * R, *x.shape[3:]))
    _, (nfeat, _) = self.dyn.imagine(
        start, acts, length=1, training=False, single=True)
    ninp = self.feat2tensor(nfeat)

    rew = f32(self.rew(ninp, bdims=1).pred())
    con = f32(self.con(ninp, bdims=1).prob(1))
    disc = 1.0 if self.config.contdisc else 1 - 1 / self.config.horizon
    voffset, vscale = self.valnorm.stats()
    qs = []
    for headi in self.valens:
      vali = f32(headi(ninp, bdims=1).pred()) * vscale + voffset
      qs.append(rew + disc * con * vali)
    q = jnp.stack(qs, 1).reshape((B, M, R, nheads))
    q = q.transpose((0, 3, 1, 2))                           # (B, K, M, R)

    actvec = jnp.concatenate([
        nn.cast(act[k]).reshape((B, -1))
        for k in sorted(self.act_space)], -1)
    # Candidate action vectors (B, M, A): Stage-1B oracle labeling
    # (d0/oracle_labels.py) must execute/imagine exactly the candidates
    # the Q matrices above were computed for.
    cand_vec = jnp.stack([
        jnp.concatenate([
            nn.cast(c[k]).reshape((B, -1)) for k in sorted(self.act_space)],
            -1) for c in cands], 1)
    out = {
        'd0/qfull': q.mean(-1),
        'd0/qhalf': q[..., :R // 2].mean(-1),
        'd0/udyn': f32(self.disag.reward(inp, actvec)),
        'd0/cands': f32(cand_vec),
    }
    return out

  def _act2tensor(self, act):
    return jnp.concatenate([
        nn.cast(act[k]).reshape((*act[k].shape[:2], -1))
        for k in sorted(self.act_space)], -1)

  def _disag_target(self, feat):
    target = self.config.expl.disag_target
    if target == 'postfeat':
      probs = jax.nn.softmax(f32(feat['logit']), -1)
      probs = probs.reshape((*probs.shape[:-2], -1))
      return jnp.concatenate([nn.cast(feat['deter']), nn.cast(probs)], -1)
    elif target == 'stoch':
      return feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1))
    else:
      raise NotImplementedError(target)

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def head_value_loss(
    rew, con, value, slowvalue, valnorm,
    contdisc=True,
    slowtar=False,
    horizon=333,
    lam=0.95,
    slowreg=1.0,
):
  """Value-only imagination loss for one auxiliary critic head.

  Mirrors the value branch of imag_loss with the head's own lambda-return
  bootstrap; never updates the shared normalizers."""
  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)
  voffset, vscale = valnorm(ret, False)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  loss = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]
  return loss, ret, val


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
