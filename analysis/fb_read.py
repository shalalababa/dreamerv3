"""Frozen reader — FB / successor-features wave (the predicted-ESCAPE test).

Registration: PREREG_fb_20260817.md (FB_Wave_Plan_20260816). ONE
execution. The theory's inclusion criterion predicts this reward-free
family ESCAPES the reward-free null: its objective factorizes occupancy
(M ~ F(s,a,z)·B(s')), making any reward's value readable by
construction. Registered both-directions clause: a determinate
NO-ESCAPE is REAL DAMAGE to the ordering content of the theory — never
instrument-blame (instrument validity carried by the gates, the
in-protocol random-floor positive control, and the env-order guard).
An INDETERMINATE non-fire lands NO-CALL-UNDERPOWERED and carries no
damage sentence (review finding B6).

  python -m analysis.fb_read --runroot <bundle>/runroot_light \
      --fit_counters <bundle>/fb_fit_counters.json \
      --zeroshot_glob '<bundle>/zeroshot/*.json' \
      --random_floor <bundle>/zeroshot/random_floor.json \
      --emb_probe_glob '<bundle>/emb_probe/fbwm_*.json' \
      --raw_probe <bundle>/emb_probe/raw_identity.json \
      --devices <bundle>/emb_panel_devices.json \
      --health_band <LO> <HI>   (the prereg's smoke-pinned fb_loss band) \
      --output <dir>
Selfcheck: python -m analysis.fb_read --selfcheck
"""

import argparse
import glob as globlib
import json
import os
import re

import numpy as np

from analysis.domains_read import one_sample
from analysis.capdescent_read import two_sample

PREREG = 'PREREG_fb_20260817.md'
SIDES = ('0', '1')
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
RID_RE = re.compile(r'^fbwm_finger_q1s(?P<side>[01])_seed(?P<seed>[1-8])$')
ALPHA_KEY = 'alpha_0.001'
CHECKOUT_PIN = '5a9950b07f1edcb4bddd04f5819379541793d975'
N_EPISODES = 10
ACTION_CONVENTION = 'prev_shifted_v1'
ALLOWED_UPDATES = (500000, 250000)  # the deterministic FILL rule's two values
PROBESET_ID_PREFIX = 'finger_v1'
# Behavioral fire bar: the house behavior-alive floor (capacity waves,
# PREREG_capdescent_20260811 lineage) — cross-wave consistency.
ESCAPE_FLOOR = 113.682
# Context pins (artifacts/randominit_read_20260810/read.json).
FLOOR_PINS = dict(rif=70.665375, apt=81.30111562500001,
                  scratch_auc=147.6823)
# In-protocol random-floor control band (review B7): REFUSE outside.
RANDOM_FLOOR_BAND = (35.0, 110.0)
# House MDE80 d_z constants (noncentral-t): n=16 -> 0.74942 (review B5).
MDE80_N16 = 0.74942
MDE80_N8 = 1.1560
ALPHA = 0.05


def run_name(side, seed):
  return f'fbwm_finger_q1s{side}_seed{seed}'


def check_witness(fc, updates):
  assert len(fc) == 16, len(fc)
  for side in SIDES:
    for seed in SEEDS:
      n = run_name(side, seed)
      assert n in fc, ('missing fit witness', n)
      assert int(fc[n]['update']) == int(fc[n]['total']) and \
          fc[n]['total'] and fc[n].get('done') is True, (n, fc[n])
      assert int(fc[n]['total']) == updates, \
          (n, 'counter total != registered updates', fc[n], updates)


def check_configs(runroot):
  """fb_config.json per fit: pin + export-path buffer linkage (the
  rescue-wave lesson) + action-convention gate + single data sha per
  side + byte-identical full agent kwargs across all 16 + registered
  updates disjunction (review findings B1/M9/M12)."""
  shas = {s: set() for s in SIDES}
  totals, kwarg_blobs = set(), set()
  for side in SIDES:
    for seed in SEEDS:
      n = run_name(side, seed)
      p = os.path.join(runroot, n, 'fb_config.json')
      assert os.path.exists(p), ('missing fb_config.json', n)
      with open(p) as f:
        c = json.load(f)
      assert c['checkout_pin'] == CHECKOUT_PIN, (n, c.get('checkout_pin'))
      assert c['data'].endswith(f'fb_data/finger_q1_side{side}.npz'), \
          (n, 'fit consumed the WRONG export — REFUSE', c['data'])
      assert c['data_manifest'].get('action_convention') == \
          ACTION_CONVENTION, (n, 'export lacks the prev-shift '
                              'convention — REFUSE (review B1)')
      assert int(c['seed']) == seed, \
          (n, 'fit seed != run name seed - REFUSE (review-2 M1)',
           c['seed'])
      kw = dict(c['agent_kwargs'])
      assert kw['z_dim'] == 50 and kw['obs_shape'] == [12] and \
          kw.get('debug') is False and kw.get('add_trunk') is False and \
          kw.get('future_ratio') == 0.0 and kw.get('use_tb') is True, \
          (n, 'config drift')
      kwarg_blobs.add(json.dumps(kw, sort_keys=True))
      shas[side].add(c['data_sha256'])
      totals.add(int(c['updates']))
  for side in SIDES:
    assert len(shas[side]) == 1, ('mixed export shas within side', side)
  assert len(kwarg_blobs) == 1, 'agent kwargs differ across fits — REFUSE'
  assert len(totals) == 1, ('mixed update totals', totals)
  updates = totals.pop()
  assert updates in ALLOWED_UPDATES, ('updates outside the registered '
                                      'FILL disjunction', updates)
  return dict(updates=updates,
              data_sha={s: shas[s].pop() for s in SIDES})


def load_health(runroot, band):
  """Review-2 M2: the training-health witness is READ and RECORDED
  (escape-hatch ledger item (c)); absence refuses, out-of-band is a
  recorded admissibility fact, never a refusal."""
  lo, hi = band
  out = {}
  for side in SIDES:
    for seed in SEEDS:
      n = run_name(side, seed)
      p = os.path.join(runroot, n, 'fb_metrics.json')
      assert os.path.exists(p), ('missing fb_metrics.json - the '
                                 'training-health witness is required',
                                 n)
      with open(p) as f:
        h = json.load(f)
      tm = h.get('fb_loss_tail_median')
      assert tm is not None and np.isfinite(tm), (n, 'degenerate '
                                                  'health witness', tm)
      out[n] = dict(tail_median=float(tm),
                    in_band=bool(lo <= tm <= hi))
  return out


def check_random_floor(path):
  with open(path) as f:
    d = json.load(f)
  assert d['tool'] == 'fb_random_floor_v1' and \
      d['task'] == 'finger_turn_hard' and \
      d['action_repeat'] == 1 and d['n_episodes'] == N_EPISODES, \
      (path, 'random-floor control meta drift')
  assert d.get('checkout_pin') == CHECKOUT_PIN and \
      int(d.get('seed', -1)) == 0, \
      (path, 'random-floor provenance drift - REFUSE (review-2 M8)')
  m = float(d['mean_return'])
  assert RANDOM_FLOOR_BAND[0] <= m <= RANDOM_FLOOR_BAND[1], \
      ('random-floor control OUTSIDE the registered band — eval harness '
       'invalid, REFUSE the read', m, RANDOM_FLOOR_BAND)
  return m


def load_zeroshot(pattern, updates):
  out, zstats = {}, {}
  for p in sorted(globlib.glob(pattern)):
    with open(p) as f:
      d = json.load(f)
    if d.get('tool') == 'fb_random_floor_v1':
      continue
    m = RID_RE.match(os.path.basename(os.path.dirname(d['ckpt'])))
    assert m, (p, d['ckpt'], 'ckpt not in a registered fit dir')
    key = (m.group('side'), int(m.group('seed')))
    assert key not in out, ('duplicate zeroshot', key)
    assert d['tool'] == 'fb_zeroshot_v1' and d['checkout_pin'] == \
        CHECKOUT_PIN and d['n_episodes'] == N_EPISODES, (p, 'meta drift')
    assert d['task'] == 'finger_turn_hard' and d['action_repeat'] == 1, \
        (p, 'task/repeat drift — REFUSE (review M13)')
    assert int(d['seed']) == key[1] and \
        int(d['env_seed']) == 1000 + key[1], (p, 'seed mispairing')
    assert int(d['updates']) == updates, \
        (p, 'zeroshot ran against a ckpt with wrong updates', d['updates'])
    assert d.get('obs_key_order_ok') is True, \
        (p, 'env obs-order guard missing — REFUSE (review M19)')
    zi = d['z_inference']
    assert int(zi.get('n_infer', -1)) == 5120, \
        (p, 'n_infer drift', zi.get('n_infer'))
    assert zi['frac_pos_reward'] > 0 and zi['z_norm'] > 0, \
        (p, 'degenerate z inference — REFUSE (review M14)', zi)
    assert d['data'].endswith(f"fb_data/finger_q1_side{key[0]}.npz"), \
        (p, 'z inferred from the WRONG side export — REFUSE')
    rets = [e['ret'] for e in d['episodes']]
    steps = {e['steps'] for e in d['episodes']}
    assert steps == {1000}, (p, 'episode length drift', sorted(steps))
    assert len(rets) == N_EPISODES and all(np.isfinite(rets)), (p, rets)
    out[key] = float(np.mean(rets))
    zstats[key] = dict(frac_pos=float(zi['frac_pos_reward']),
                       z_norm=float(zi['z_norm']),
                       ckpt_sha256=d.get('ckpt_sha256'))
  for side in SIDES:
    for seed in SEEDS:
      assert (side, seed) in out, ('missing zeroshot', side, seed)
  assert len(out) == 16
  # auditable side premise (review M14): side1 is the reward-rich side
  fp0 = np.mean([zstats[('0', k)]['frac_pos'] for k in SEEDS])
  fp1 = np.mean([zstats[('1', k)]['frac_pos'] for k in SEEDS])
  assert fp1 > fp0, ('side premise violated: side1 not reward-rich in '
                     'the inference samples', fp1, fp0)
  return out, zstats


def load_probe(path, expect_dim=None):
  with open(path) as f:
    d = json.load(f)
  assert 'probe' in d and ALPHA_KEY in d['probe'], \
      (path, 'missing probe/alpha block — REFUSE')
  v = d['probe'][ALPHA_KEY].get('auroc')
  assert v is not None and np.isfinite(v), (path, 'degenerate auroc')
  em = d.get('embed_manifest')
  assert em is not None, (path, 'missing embed_manifest')
  if expect_dim is not None:
    assert int(em['dim']) == expect_dim, (path, em['dim'], expect_dim)
  ps_id = d.get('probeset_id')
  ps_sha = d.get('probeset_sha256')
  assert ps_id and ps_sha, (path, 'missing probeset identity — REFUSE')
  return float(v), ps_id, ps_sha, em


def load_emb_panel(pattern, raw_path):
  out, ids, shas = {}, set(), set()
  for p in sorted(globlib.glob(pattern)):
    m = RID_RE.match(os.path.basename(p).replace('.json', ''))
    assert m, (p, 'unregistered emb probe filename')
    key = (m.group('side'), int(m.group('seed')))
    assert key not in out, ('duplicate emb probe', key)
    v, ps_id, ps_sha, em = load_probe(p, expect_dim=50)
    # review m29: ckpt -> run linkage
    assert os.path.basename(os.path.dirname(em['ckpt'])) == \
        run_name(*key), (p, 'embed ckpt not from this probe file\'s run',
                         em['ckpt'])
    out[key] = (v, em.get('ckpt_sha256'))
    ids.add(ps_id)
    shas.add(ps_sha)
  for side in SIDES:
    for seed in SEEDS:
      assert (side, seed) in out, ('missing emb probe', side, seed)
  emb_shas = {k: t[1] for k, t in out.items()}
  out = {k: t[0] for k, t in out.items()}
  raw, rid, rsha, rem = load_probe(raw_path, expect_dim=12)
  assert rem['ckpt'] == 'raw-identity' and \
      rem['preproc'] == 'fb_obs_keys_v1', (raw_path, rem)
  ids.add(rid)
  shas.add(rsha)
  # review M11: one panel across all 17 probes
  assert len(shas) == 1, ('probeset sha differs across the panel — the '
                          'paired contrast would be meaningless', shas)
  the_id = ids.pop() if len(ids) == 1 else None
  assert the_id and the_id.startswith(PROBESET_ID_PREFIX), \
      ('probeset id drift', ids, the_id)
  return out, raw, emb_shas


def check_devices(path):
  """Review M10 + review-2 B-C: semantics = embed_manifest.gpu_name of
  the embedding producer (the PHYSICAL device from
  torch.cuda.get_device_name, never the torch device string 'cuda' —
  which would make this gate vacuous). Producer: the registered ops
  block builds the map mechanically from the 17 embed_manifest.json
  files. Exact key-set equality; single GPU across the 16 fb entries;
  the raw identity comparator is device-independent by construction
  and must be 'cpu'."""
  with open(path) as f:
    dv = json.load(f)
  expected = {run_name(s, k) + '.json' for s in SIDES for k in SEEDS}
  expected.add('raw_identity.json')
  assert set(dv) == expected, ('device map key set != the registered '
                               '17-probe panel', sorted(set(dv) ^ expected))
  assert dv['raw_identity.json'] == 'cpu', \
      ('raw comparator must record cpu (identity embedding)',
       dv['raw_identity.json'])
  names = {v for k, v in dv.items() if k != 'raw_identity.json'}
  assert len(names) == 1, ('fb emb panel mixes GPU models — REFUSE '
                           '(single-GPU rule, 16 Aug)', sorted(names))
  return names.pop()


def analyse(zs, emb, raw):
  res = {}
  # P-FB1 — zero-shot escape vs the behavior-alive floor
  rets = np.array([zs[(s, k)] for s in SIDES for k in SEEDS])
  st1 = one_sample(rets - ESCAPE_FLOOR)
  fb1_fire = st1['ci'][0] > 0 and st1['perm_p'] < ALPHA
  fb1_neg = st1['ci'][1] < 0 and st1['perm_p'] < ALPHA  # review m23
  res['p_fb1'] = dict(
      stats=st1, floor=ESCAPE_FLOOR, floor_pins=FLOOR_PINS,
      mean_return=float(rets.mean()),
      by_side={s: float(np.mean([zs[(s, k)] for k in SEEDS]))
               for s in SIDES},
      by_side_secondary={s: one_sample(np.array(
          [zs[(s, k)] for k in SEEDS]) - ESCAPE_FLOOR)
          for s in SIDES},
      by_side_note='side-stratified secondaries, n=8, labeled '
                   'NON-DECISIONAL (review m24); a pooled fire carried '
                   'entirely by one side is reported as such',
      verdict=('ZERO-SHOT-ESCAPE' if fb1_fire else
               'NO-BEHAVIORAL-ESCAPE' if fb1_neg else
               'BEHAVIORAL-INDETERMINATE'))
  if not fb1_fire:
    res['p_fb1']['mde_note'] = dict(
        mde80_return_units=float(MDE80_N16 * st1['sd']),
        basis='noncentral-t approximation at n=16; decision rule is '
              'permutation+BCa')
  # directional control (labeled, non-blocking)
  ctrl = two_sample([zs[('1', k)] for k in SEEDS],
                    [zs[('0', k)] for k in SEEDS])
  res['side_control'] = dict(
      stat=ctrl,
      note='labeled non-blocking control; CI-based labels, perm_p '
           'reported in stat (review-2 m4)',
      label=('CONTROL-SUSPECT (side0 significantly above side1)'
             if ctrl['ci'][1] < 0 else
             'CONTROL-WEAK (no separation)' if ctrl['ci'][0] <= 0
             else 'CONTROL-OK'))
  # P-FB2 — B-embedding legibility vs the raw-proprio identity comparator
  deltas = np.array([emb[(s, k)] - raw for s in SIDES for k in SEEDS])
  st2 = one_sample(deltas)
  fb2_fire = st2['ci'][0] > 0 and st2['perm_p'] < ALPHA
  res['p_fb2'] = dict(
      stats=st2, raw_auroc=raw,
      fb_mean_auroc=float(np.mean(list(emb.values()))),
      by_side={s: float(np.mean([emb[(s, k)] for k in SEEDS]))
               for s in SIDES},
      estimand_note='panel-conditional contrast: raw is treated as fixed '
                    'by construction (same panel, same ridge folds); the '
                    'interval does not propagate raw\'s own measurement '
                    'error (review M16b)',
      verdict=('B-MORE-LEGIBLE-THAN-RAW' if fb2_fire else
               'B-LESS-LEGIBLE' if st2['ci'][1] < 0 and
               st2['perm_p'] < ALPHA else 'EMB-NO-SEPARATION'))
  if not fb2_fire:
    res['p_fb2']['mde_note'] = dict(
        mde80_auroc_units=float(MDE80_N16 * st2['sd']),
        basis='noncentral-t approximation at n=16; decision rule is '
              'permutation+BCa')
  # overall (review B6: damage only on a DETERMINATE flagship negative)
  if fb1_fire and fb2_fire:
    overall = ('ESCAPE-CONFIRMED: the FB objective escapes the '
               'reward-free null at BOTH levels — the inclusion '
               "criterion's predicted escape lands; objective structure "
               'predicts legibility bidirectionally')
  elif fb1_fire:
    overall = ('BEHAVIORAL-ESCAPE (qualified): the flagship '
               'consequential-level escape LANDS; the representation '
               'leg shows no separation from a NEAR-CEILING raw '
               'comparator (registered as weakly informative, honesty '
               'block M16a) — escape wording licensed at the behavioral '
               'level; the strongest bidirectional form is reserved for '
               'ESCAPE-CONFIRMED (review-2 M6)')
  elif fb2_fire and fb1_neg:
    overall = ('REPRESENTATION-ONLY-BEHAVIORAL-DAMAGE: B-embeddings '
               'separate from raw, but the flagship is DETERMINATE-'
               'NEGATIVE at the consequential level. REGISTERED '
               'CONSEQUENCE: the behavioral half of the damage sentence '
               'applies — the predicted consequential-level escape did '
               'not land (review-2 M5); no escape wording')
  elif fb2_fire:
    overall = ('REPRESENTATION-ONLY: B-embeddings are reward-legible '
               'beyond raw observables but zero-shot behavior is '
               'indeterminate vs the floor — no escape claim, no '
               'damage claim; MDE notes carry the realized power')
  elif fb1_neg:
    overall = ('NO-ESCAPE: the predicted escape did NOT land — the '
               'flagship is DETERMINATE-NEGATIVE (BCa upper < 0, '
               'p < .05 vs the floor). REGISTERED CONSEQUENCE: evidence '
               "AGAINST the inclusion criterion's ordering content — "
               'record the damage; the in-protocol random-floor control '
               'and env-order guard carry instrument validity, and no '
               'other caveat is admissible (escape-hatch ledger, prereg '
               'honesty block)')
  else:
    overall = ('NO-CALL-UNDERPOWERED: neither leg fires and the '
               'flagship is not determinate-negative — no escape claim '
               'and NO damage claim; both MDE notes carry the realized '
               'power (review B6)')
  res['overall'] = overall
  res['prereg'] = PREREG
  return res


def run(args):
  with open(args.fit_counters) as f:
    fc = json.load(f)
  cfg = check_configs(args.runroot)
  check_witness(fc, cfg['updates'])
  device = check_devices(args.devices)
  floor_ctrl = check_random_floor(args.random_floor)
  health = load_health(args.runroot, args.health_band)
  zs, zstats = load_zeroshot(args.zeroshot_glob, cfg['updates'])
  emb, raw, emb_shas = load_emb_panel(args.emb_probe_glob, args.raw_probe)
  for side in SIDES:
    for seed in SEEDS:
      zsha = zstats[(side, seed)].get('ckpt_sha256')
      esha = emb_shas.get((side, seed))
      assert zsha and esha and zsha == esha, \
          ('zeroshot/emb ckpt BYTE identity mismatch - the two legs '
           'measured different checkpoints (review-2 m8)',
           side, seed, zsha, esha)
  res = analyse(zs, emb, raw)
  res['gates'] = dict(updates=cfg['updates'], data_sha=cfg['data_sha'],
                      emb_panel_device=device,
                      random_floor_mean=floor_ctrl,
                      random_floor_band=list(RANDOM_FLOOR_BAND),
                      health_band=list(args.health_band),
                      health=health,
                      health_all_in_band=all(h['in_band']
                                             for h in health.values()))
  res['per_fit'] = dict(
      zeroshot={f'{s}|{k}': zs[(s, k)] for s in SIDES for k in SEEDS},
      z_inference={f'{s}|{k}': zstats[(s, k)]
                   for s in SIDES for k in SEEDS},
      emb_auroc={f'{s}|{k}': emb[(s, k)] for s in SIDES for k in SEEDS})
  os.makedirs(args.output, exist_ok=True)
  with open(os.path.join(args.output, 'read.json'), 'w') as f:
    json.dump(res, f, indent=1, sort_keys=True)
  print(res['overall'])
  print(f"-> {os.path.join(args.output, 'read.json')}")


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------

def _fixture(base, name, zs_level, emb_delta, raw=0.62, side_gap=0.0,
             zs_sd=8.0, floor_mean=75.0, mutate=None):
  import shutil
  root = os.path.join(base, name)
  shutil.rmtree(root, ignore_errors=True)
  rr = os.path.join(root, 'runroot')
  zdir = os.path.join(root, 'zeroshot')
  edir = os.path.join(root, 'emb_probe')
  for d in (rr, zdir, edir):
    os.makedirs(d)
  rng = np.random.default_rng(sum(ord(c) for c in name))
  fc, devices = {}, {}
  for side in SIDES:
    for seed in SEEDS:
      n = run_name(side, seed)
      fc[n] = dict(update=500000, total=500000, done=True)
      os.makedirs(os.path.join(rr, n))
      cfgd = dict(checkout_pin=CHECKOUT_PIN,
                  data=f'/x/fb_data/finger_q1_side{side}.npz',
                  data_sha256=f'sha_{side}', updates=500000,
                  data_manifest=dict(action_convention=ACTION_CONVENTION),
                  agent_kwargs=dict(z_dim=50, obs_shape=[12], debug=False,
                                    add_trunk=False, future_ratio=0.0,
                                    use_tb=True))
      if mutate == 'wrong_buffer' and (side, seed) == ('0', 3):
        cfgd['data'] = '/x/fb_data/finger_q1_side1.npz'
      if mutate == 'no_convention' and (side, seed) == ('1', 4):
        cfgd['data_manifest'] = {}
      cfgd['seed'] = seed
      if mutate == 'seed_mispair' and (side, seed) == ('0', 4):
        cfgd['seed'] = 1
      with open(os.path.join(rr, n, 'fb_config.json'), 'w') as f:
        json.dump(cfgd, f)
      if not (mutate == 'no_health' and (side, seed) == ('1', 5)):
        with open(os.path.join(rr, n, 'fb_metrics.json'), 'w') as f:
          json.dump(dict(final=dict(fb_loss=30.0),
                         fb_loss_tail_median=30.0, n_tail=100), f)
      lvl = zs_level + (side_gap if side == '1' else 0)
      rets = [dict(ret=float(lvl + rng.normal(0, zs_sd)), steps=1000)
              for _ in range(N_EPISODES)]
      # (steps fixed at 1000 = the registered episode length gate)
      zj = dict(tool='fb_zeroshot_v1', checkout_pin=CHECKOUT_PIN,
                n_episodes=N_EPISODES, episodes=rets,
                task='finger_turn_hard', action_repeat=1,
                seed=seed, env_seed=1000 + seed, updates=500000,
                obs_key_order_ok=True, ckpt_sha256=f'cksha_{n}',
                z_inference=dict(n_infer=5120,
                                 frac_pos_reward=0.02 + (0.04 if
                                 side == '1' else 0.0), z_norm=7.07),
                ckpt=os.path.join(rr, n, 'fb_ckpt.pt'),
                data=f'/x/fb_data/finger_q1_side{side}.npz')
      if mutate == 'wrong_zs_side' and (side, seed) == ('1', 2):
        zj['data'] = '/x/fb_data/finger_q1_side0.npz'
      if mutate == 'zdegen' and (side, seed) == ('0', 5):
        zj['z_inference'] = dict(frac_pos_reward=0.0, z_norm=0.0)
      with open(os.path.join(zdir, n + '.json'), 'w') as f:
        json.dump(zj, f)
      auroc = (None if (mutate == 'null_auroc' and (side, seed) ==
                        ('0', 2)) else
               float(np.clip(raw + emb_delta + rng.normal(0, 0.01),
                             0.01, 0.99)))
      pj = dict(probe={ALPHA_KEY: dict(auroc=auroc)},
                probeset_id='finger_v1',
                probeset_sha256=('WRONG' if (mutate == 'ps_mixed' and
                                             (side, seed) == ('1', 6))
                                 else 'PSSHA'),
                embed_manifest=dict(dim=50, ckpt=os.path.join(
                    rr, n, 'fb_ckpt.pt'),
                    ckpt_sha256=('MISMATCH' if (mutate == 'sha_split'
                                 and (side, seed) == ('0', 6))
                                 else f'cksha_{n}')))
      with open(os.path.join(edir, n + '.json'), 'w') as f:
        json.dump(pj, f)
      devices[n + '.json'] = ('NVIDIA A100' if (mutate == 'mixed_dev' and
                                                (side, seed) == ('0', 7))
                              else 'Quadro RTX 6000')
  with open(os.path.join(edir, 'raw_identity.json'), 'w') as f:
    json.dump(dict(probe={ALPHA_KEY: dict(auroc=raw)},
                   probeset_id='finger_v1',
                   probeset_sha256='PSSHA',
                   embed_manifest=dict(dim=12, ckpt='raw-identity',
                                       preproc='fb_obs_keys_v1')), f)
  devices['raw_identity.json'] = 'cpu'
  with open(os.path.join(root, 'devices.json'), 'w') as f:
    json.dump(devices, f)
  with open(os.path.join(zdir, 'random_floor.json'), 'w') as f:
    json.dump(dict(tool='fb_random_floor_v1', task='finger_turn_hard',
                   action_repeat=1, n_episodes=N_EPISODES, seed=0,
                   checkout_pin=CHECKOUT_PIN,
                   mean_return=floor_mean), f)
  with open(os.path.join(root, 'fc.json'), 'w') as f:
    if mutate == 'short_fit':
      fc[run_name('0', 1)]['update'] = 400000
    json.dump(fc, f)
  return argparse.Namespace(
      runroot=rr, fit_counters=os.path.join(root, 'fc.json'),
      zeroshot_glob=os.path.join(zdir, '*.json'),
      random_floor=os.path.join(zdir, 'random_floor.json'),
      emb_probe_glob=os.path.join(edir, 'fbwm_*.json'),
      raw_probe=os.path.join(edir, 'raw_identity.json'),
      devices=os.path.join(root, 'devices.json'),
      health_band=(20.0, 40.0),
      output=os.path.join(root, 'out'))


def selfcheck():
  import tempfile
  base = tempfile.mkdtemp(prefix='fb_read_sc_')
  # review m22: pin the frozen literals
  assert ESCAPE_FLOOR == 113.682 and N_EPISODES == 10 and \
      ALPHA == 0.05 and ALPHA_KEY == 'alpha_0.001' and \
      MDE80_N16 == 0.74942 and MDE80_N8 == 1.1560 and \
      ALLOWED_UPDATES == (500000, 250000) and \
      RANDOM_FLOOR_BAND == (35.0, 110.0) and FLOOR_PINS == dict(
          rif=70.665375, apt=81.30111562500001, scratch_auc=147.6823) and \
      ACTION_CONVENTION == 'prev_shifted_v1' and \
      CHECKOUT_PIN == '5a9950b07f1edcb4bddd04f5819379541793d975' and \
      PROBESET_ID_PREFIX == 'finger_v1' and SIDES == ('0', '1') and \
      SEEDS == (1, 2, 3, 4, 5, 6, 7, 8)

  def result(ns):
    run(ns)
    with open(os.path.join(ns.output, 'read.json')) as f:
      return json.load(f)

  r = result(_fixture(base, 'escape', zs_level=400, emb_delta=0.15))
  assert r['overall'].startswith('ESCAPE-CONFIRMED'), r['overall']
  r = result(_fixture(base, 'repr_only', zs_level=110, emb_delta=0.15,
                      zs_sd=60.0))
  assert r['overall'].startswith('REPRESENTATION-ONLY:'), r['overall']
  assert 'mde_note' in r['p_fb1'] and \
      'REGISTERED CONSEQUENCE' not in r['overall']
  r = result(_fixture(base, 'repdmg', zs_level=80, emb_delta=0.15,
                      zs_sd=3.0))
  assert r['overall'].startswith('REPRESENTATION-ONLY-BEHAVIORAL-'
                                 'DAMAGE'), r['overall']
  # determinate negative -> NO-ESCAPE (damage)
  r = result(_fixture(base, 'noescape', zs_level=80, emb_delta=0.0,
                      zs_sd=3.0))
  assert r['overall'].startswith('NO-ESCAPE') and \
      'AGAINST' in r['overall'], r['overall']
  assert r['p_fb1']['verdict'] == 'NO-BEHAVIORAL-ESCAPE'
  # indeterminate -> NO-CALL, no damage sentence (review B6)
  r = result(_fixture(base, 'nocall', zs_level=110, emb_delta=0.0,
                      zs_sd=60.0))
  assert r['overall'].startswith('NO-CALL-UNDERPOWERED'), r['overall']
  assert 'AGAINST' not in r['overall']
  r = result(_fixture(base, 'behesc', zs_level=400, emb_delta=0.0))
  assert r['overall'].startswith('BEHAVIORAL-ESCAPE'), r['overall']
  r = result(_fixture(base, 'ctrl', zs_level=250, emb_delta=0.1,
                      side_gap=-150, zs_sd=3.0))
  assert 'SUSPECT' in r['side_control']['label']
  r = result(_fixture(base, 'ctrlweak', zs_level=250, emb_delta=0.1,
                      side_gap=0.0, zs_sd=30.0))
  assert 'WEAK' in r['side_control']['label']
  # refusals
  refused = 0
  muts = ('wrong_buffer', 'wrong_zs_side', 'short_fit', 'mixed_dev',
          'no_convention', 'zdegen', 'null_auroc', 'ps_mixed',
          'seed_mispair', 'no_health', 'sha_split')
  for mut in muts:
    try:
      result(_fixture(base, 'm_' + mut, zs_level=200, emb_delta=0.1,
                      mutate=mut))
    except AssertionError:
      refused += 1
  # random-floor out of band
  try:
    result(_fixture(base, 'm_floor', zs_level=200, emb_delta=0.1,
                    floor_mean=300.0))
  except AssertionError:
    refused += 1
  # missing-cell refusal
  ns = _fixture(base, 'm_missing', zs_level=200, emb_delta=0.1)
  os.remove(os.path.join(os.path.dirname(ns.raw_probe),
                         run_name('1', 7) + '.json'))
  try:
    result(ns)
  except AssertionError:
    refused += 1
  # foreign run_id refusal
  ns = _fixture(base, 'm_foreign', zs_level=200, emb_delta=0.1)
  edir = os.path.dirname(ns.raw_probe)
  with open(os.path.join(edir, 'fbwm_finger_q1s0_seed9.json'), 'w') as f:
    json.dump(dict(probe={ALPHA_KEY: dict(auroc=0.7)},
                   probeset_id='finger_v1_7a32ff08',
                   probeset_sha256='PSSHA',
                   embed_manifest=dict(dim=50, ckpt='x')), f)
  try:
    result(ns)
  except AssertionError:
    refused += 1
  assert refused == 14, ('refusal legs', refused)
  print('fb_read selfcheck PASS (8 branch fixtures incl. NO-CALL, '
        'BEHAVIORAL-ESCAPE, REPRESENTATION-ONLY-BEHAVIORAL-DAMAGE, '
        'control-weak; 14 refusals: wrong-buffer, wrong-zs-side, '
        'short-fit, mixed-device, missing-convention, degenerate-z, '
        'null-auroc, probeset-mismatch, fit-seed-mispair, '
        'missing-health, ckpt-sha-split, floor-out-of-band, '
        'missing-cell, foreign-run_id)')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--runroot')
  ap.add_argument('--fit_counters')
  ap.add_argument('--zeroshot_glob')
  ap.add_argument('--random_floor')
  ap.add_argument('--emb_probe_glob')
  ap.add_argument('--raw_probe')
  ap.add_argument('--devices')
  ap.add_argument('--health_band', nargs=2, type=float,
                  metavar=('LO', 'HI'))
  ap.add_argument('--output')
  ap.add_argument('--selfcheck', action='store_true')
  args = ap.parse_args()
  if args.selfcheck:
    selfcheck()
  else:
    assert all([args.runroot, args.fit_counters, args.zeroshot_glob,
                args.random_floor, args.emb_probe_glob, args.raw_probe,
                args.devices, args.health_band, args.output])
    run(args)


if __name__ == '__main__':
  main()
