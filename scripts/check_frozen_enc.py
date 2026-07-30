"""Frozen-encoder integrity check for the pretrained-encoder pixel arm
(PREREG_pe_pixel_20260730). Registered smoke/verification gate: after a
stage-2 fit (enc loaded from a donor via '^enc/' + agent.frozen_enc True),
the stage-2 checkpoint's encoder must be BYTE-IDENTICAL to the donor's,
the fit must have actually TRAINED (witnessed by OFFLINE_FIT_PROGRESS,
which offline_fit writes only inside its training loop — param diffs
alone cannot distinguish 'trained' from 'fresh-init never trained'),
and the reward head must exist (task-mode fit). A silent failure of any
leg would let a broken run masquerade as NO-RELIEF, so this runs before
labels/adapts are trusted.

Usage:
  python scripts/check_frozen_enc.py <stage2_ckpt_dir> <donor_ckpt_dir> \
      [expected_updates]           # default 500000; smoke passes 2000
  python scripts/check_frozen_enc.py --selfcheck

<stage2_ckpt_dir> is a specific checkpoint (contains agent.pkl + done),
e.g. $RUNROOT/<run>/ckpt/<stamp>-000000500000; the training witness is
read from <run>/OFFLINE_FIT_PROGRESS (two levels up). Exit 0 = PASS.
"""
import pickle
import pathlib
import sys

import numpy as np


def load_params(ckpt_dir):
  p = pathlib.Path(ckpt_dir)
  assert (p / 'done').exists(), f'no done marker in {p}'
  f = p / 'agent.pkl'
  assert f.exists(), (
      f'no agent.pkl in {p} (sharded checkpoints unsupported here — '
      f'found: {sorted(x.name for x in p.glob("*"))})')
  data = pickle.loads(f.read_bytes())
  return data['params'], data.get('counters', {})


def read_progress(run_dir):
  f = pathlib.Path(run_dir) / 'OFFLINE_FIT_PROGRESS'
  assert f.exists(), (
      f'no OFFLINE_FIT_PROGRESS under {run_dir} — the fit loop never ran '
      '(counter-carryover skip? check PARTIAL_INIT + counters_reset in the '
      'fit log)')
  kv = dict(line.split('=', 1) for line in f.read_text().splitlines() if '=' in line)
  return int(kv['update']), int(kv['total_updates'])


def check(stage2_ckpt, donor_ckpt, expected_updates=500000):
  s2, cnt2 = load_params(stage2_ckpt)
  donor, _ = load_params(donor_ckpt)

  enc_keys = sorted(k for k in donor if k.startswith('enc/'))
  assert enc_keys, f'donor has no enc/ params: {donor_ckpt}'
  missing = [k for k in enc_keys if k not in s2]
  assert not missing, f'stage-2 ckpt missing enc keys: {missing}'
  diff = [k for k in enc_keys
          if not np.array_equal(np.asarray(s2[k]), np.asarray(donor[k]))]
  assert not diff, f'FROZEN-ENC VIOLATION — enc params differ: {diff}'

  run_dir = pathlib.Path(stage2_ckpt).parent.parent
  update, total = read_progress(run_dir)
  assert update >= expected_updates, (
      f'TRAINING WITNESS FAIL — OFFLINE_FIT_PROGRESS update={update} < '
      f'expected {expected_updates} (fit incomplete or silently skipped)')

  rew_keys = [k for k in s2 if k.startswith('rew/')]
  assert rew_keys, 'stage-2 ckpt has no rew/ params — not a task-mode fit?'

  dyn_keys = sorted(k for k in s2 if k.startswith('dyn/'))
  assert dyn_keys, 'stage-2 ckpt has no dyn/ params'

  return (f'PASS: {len(enc_keys)} enc params byte-identical to donor; '
          f'trained to update {update}/{total} '
          f'(>= expected {expected_updates}); {len(rew_keys)} rew params '
          f'present; stage-2 counters={cnt2}')


def selfcheck():
  import tempfile
  rng = np.random.default_rng(0)
  enc = {f'enc/cnn{i}/kernel': rng.normal(size=(3, 3)).astype(np.float32)
         for i in range(4)}
  dyn = {f'dyn/w{i}': rng.normal(size=(4,)).astype(np.float32)
         for i in range(3)}
  rew = {'rew/out/kernel': rng.normal(size=(4,)).astype(np.float32)}

  def write_run(root, name, params, progress_update=2000):
    run = pathlib.Path(root) / name
    ck = run / 'ckpt' / 'stamp-000000002000'
    ck.mkdir(parents=True)
    (ck / 'agent.pkl').write_bytes(
        pickle.dumps({'params': params, 'counters': {'updates': 2000}}))
    (ck / 'done').write_bytes(b'')
    if progress_update is not None:
      (run / 'OFFLINE_FIT_PROGRESS').write_text(
          f'updated_at=x\nupdate={progress_update}\n'
          f'total_updates=2000\nwm_total=1.0\n')
    return str(ck)

  def trips(fn):
    try:
      fn()
    except AssertionError:
      return True
    return False

  with tempfile.TemporaryDirectory() as td:
    donor = write_run(td, 'donor', {**enc, **dyn}, progress_update=None)
    dyn2 = {k: v + 1 for k, v in dyn.items()}
    good = write_run(td, 'good', {**enc, **dyn2, **rew})
    print(check(good, donor, expected_updates=2000))

    enc_bad = dict(enc)
    enc_bad['enc/cnn0/kernel'] = enc['enc/cnn0/kernel'] + 1e-6
    drift = write_run(td, 'drift', {**enc_bad, **dyn2, **rew})
    assert trips(lambda: check(drift, donor, 2000)), 'enc drift must trip'

    # The M1 scenario: fresh dyn (differs from donor) but the loop never
    # ran — no progress file. Must trip on the training witness.
    skipped = write_run(td, 'skipped', {**enc, **dyn2, **rew},
                        progress_update=None)
    assert trips(lambda: check(skipped, donor, 2000)), (
        'missing progress witness must trip')

    short = write_run(td, 'short', {**enc, **dyn2, **rew},
                      progress_update=500)
    assert trips(lambda: check(short, donor, 2000)), (
        'short progress must trip')

    norew = write_run(td, 'norew', {**enc, **dyn2})
    assert trips(lambda: check(norew, donor, 2000)), 'missing rew must trip'

  print('selfcheck PASS: good + 4 trip legs '
        '(enc-drift, missing-witness, short-witness, no-rew)')


def main():
  if len(sys.argv) == 2 and sys.argv[1] == '--selfcheck':
    selfcheck()
    return
  if len(sys.argv) not in (3, 4):
    raise SystemExit(__doc__)
  expected = int(sys.argv[3]) if len(sys.argv) == 4 else 500000
  print(check(sys.argv[1], sys.argv[2], expected))


if __name__ == '__main__':
  main()
