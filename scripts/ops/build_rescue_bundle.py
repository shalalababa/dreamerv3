"""Read-bundle builder for the three rescue waves (carrier / u1a / mindose).

The frozen readers (analysis/{carrier_freshrep,u1a_pooled,mindose_reacher}
_read.py) dictate the bundle contract; this tool TRANSCRIBES it rather
than inventing anything (fb_fits build_read_bundle precedent). Built after
ops flagged that these three registered chains had no builder; review
findings (21 Aug, 4B/6M/8m) all applied.

Per wave it assembles into one bundle dir (built in a temp dir, renamed
on success — a refusal leaves nothing half-built):
  auc.csv              via the frozen analysis/adaptation_auc.py collator
                       (the SAME producer as the Amendment-1 look-1 csv —
                       pooling requires estimator identity)
  witness.json         {reader_fit_key: {counters, counters_source,
                        ckpt_scan_update, progress_update_audit_only,
                        config_sha_excl_registered[, invocation_id,
                        lane_file]}, _meta: {...}}
  linkage.json         (carrier) fit_key -> realized replay source
                       (+ per-run source: realized_log | lane_cmds)
  buffer_manifest.json (mindose) REALIZED occupancy + verified holdout
                       disposition + first-draw attestation
  MANIFEST.sha256      sha256 of every bundle file

Witness semantics (REALIZED, never nominal — 08-07 realized-training
rule + the stale_fit_witness_20260816 lesson):

  counters   For carrier: the update suffix of the CKPT= path echoed by
             the adapt stage in the cell's own cloud log — the
             checkpoint the adapt ACTUALLY LOADED. A post-hoc directory
             scan of a merged multi-instance runroot can report a
             complete checkpoint the adapt never saw (review finding 5;
             stale_fit_witness precedent), so the scan value is
             recorded separately (ckpt_scan_update) and a disagreement
             is a problem. For u1a/mindose (single-host RCC runroots,
             no per-cell logs) the scan is the source and is labeled so.
  config_sha_excl_registered
             sha256 of config.yaml bytes with EXACTLY the
             amendment-M5-registered exception lines removed: top-level
             `seed:` and `logdir:` (measured: logdir carries both the
             seed and the runroot prefix; nothing else differs across
             same-arm fits — review finding 3). The literal exception
             list is recorded in _meta. NOTE the sha is deliberately
             side-blind (REPLAY is not persisted in config.yaml); the
             side is carried by the linkage gate, and arms DO differ
             (reward_grad/repval_grad are saved), so arm confusion is
             caught.
  amend1_config_match
             (carrier) true only if EVERY fresh fit config byte-matches
             its (arm, side) reference config from --ref_configs under
             the same exceptions. Absent references => false => the
             reader refuses; can never silently pass. Reference paths +
             sha256s are recorded in _meta (config_ref_paths /
             config_ref_shas) for BOTH carrier and u1a (finding 6).
  invocation_id
             (carrier) the REALIZED container/host id parsed from the
             cell's _cloud_logs/<adapt_run>.out ('host=...'); measured
             non-vacuous (12-hex docker ids, distinct per instance).
             The lane .cmds basename is recorded alongside so pairing
             is reconstructable (finding 11).
  fit_dir_aliases
             (u1a) THE READER'S FIT KEYS DO NOT MATCH THE REALIZED RUN
             DIRS: the frozen reader expects ax1wm_finger_{uz,uzf}q1s*
             but the rebuild wave's donors follow the U1 registration
             verbatim (ax1wm_finger_q1s* task / ax1wm_finger_fq1s*
             apt — spec.yaml states this). The witness is keyed by the
             reader's names, sourced from the realized dirs, and the
             full alias map is recorded in _meta as a DISCLOSED
             registration-visible mapping (review finding 2 — user
             adjudication recorded in the ledger, not a silent edit).

Exit status: 0 clean; 3 when the bundle was written but carries
problems (a ONE-read on a problem bundle requires the deliberate
--allow_problems keystroke to exit 0 — finding 9); nonzero die() on
contract violations (nothing bundled).

Usage (ops; per the 17-Aug rule, FULL-PULL instance runroots to RCC
first, then run here against the merged runroot):
  python -m scripts.ops.build_rescue_bundle --wave carrier \
      --runroot $RUNROOT --ref_configs <dir with {rgo,sgb}s{0,1}.yaml> \
      [--lane_cmds ops/waves/carrier_freshrep/generated] \
      --output $RUNROOT/bundles/carrier_freshrep_$(date +%Y%m%d_%H%M%S)
  python -m scripts.ops.build_rescue_bundle --wave u1a \
      --runroot $RUNROOT --ref_configs <finger_refit_20260810 cfg dir> \
      --output ...
  python -m scripts.ops.build_rescue_bundle --wave mindose \
      --runroot $RUNROOT --dose_manifest <build_dose out>/manifest.json \
      --attest_first_draw --output ...
Selfcheck: python -m scripts.ops.build_rescue_bundle --selfcheck
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

CONFIG_EXCEPTIONS = ('seed:', 'logdir:')   # amend1 M5, recorded in _meta


def die(msg):
  raise SystemExit(f'BUILD REFUSED: {msg}')


# --------------------------------------------------------------------------
# Wave specs (fit lists transcribed from the frozen readers; u1a dirs
# aliased per finding 2)
# --------------------------------------------------------------------------

def wave_spec(wave, seeds=None):
  if wave == 'carrier':
    seeds = tuple(range(17, 57)) if seeds is None else seeds
    fits = [f'ax1wm_finger_{a}q1s{sd}_seed{s}'
            for a in ('rgo', 'sgb') for sd in ('0', '1') for s in seeds]
    adapt_of = {f: 'adapt_ax1{a}q1s{sd}_finger_seed{s}_ckpt500000'.format(
        a=f.split('_finger_')[1].split('q1s')[0],
        sd=f.split('q1s')[1][0], s=int(f.split('_seed')[1]))
        for f in fits}
    return dict(fits=fits, dir_of={f: f for f in fits},
                adapt_of=adapt_of, updates=500000, full_n=160)
  if wave == 'u1a':
    seeds = tuple(range(9, 17)) if seeds is None else seeds
    fits, dir_of = [], {}
    for p, real in (('uz', ''), ('uzf', 'f')):
      for sd in ('0', '1'):
        for s in seeds:
          key = f'ax1wm_finger_{p}q1s{sd}_seed{s}'
          fits.append(key)
          dir_of[key] = f'ax1wm_finger_{real}q1s{sd}_seed{s}'
    return dict(fits=fits, dir_of=dir_of, updates=500000, full_n=32)
  if wave == 'mindose':
    seeds = tuple(range(1, 9)) if seeds is None else seeds
    fits = [f'ax1wm_reacher_mdd1_seed{s}' for s in seeds]
    return dict(fits=fits, dir_of={f: f for f in fits}, updates=500000,
                full_n=8)
  if wave == 'valuefree':
    # PREREG_valuefree_online_20260821 v3 (finger, rgo arm, carrier
    # seeds 17-24): 8 fresh online task runs; the 16 rgo fit names
    # (s1+s0) come via --extra_fits.
    seeds = tuple(range(17, 25)) if seeds is None else seeds
    fits = [f'ontask_finger_seed{s}' for s in seeds]
    return dict(fits=fits, dir_of={f: f for f in fits}, updates=500000,
                full_n=8, online_fits=list(fits), extra_expect=16,
                vf_seeds=list(seeds))
  if wave == 'crowding':
    # PREREG_crowding_behavioral_20260820 (+amend1). 16 UNFROZEN adapts off
    # the existing ruw fits; run_ids transcribed from the frozen reader
    # (analysis/crowding_behavioral_read.py:45 ADAPTS), never invented.
    seeds = tuple(range(1, 5)) if seeds is None else seeds
    fits, adapt_of = [], {}
    for w_ in (1, 100):
      for sd in ('0', '1'):
        for s in seeds:
          key = f'ax1wm_finger_ruw{w_}q1s{sd}_seed{s}'
          fits.append(key)
          adapt_of[key] = (f'adapt_ax1uzruw{w_}q1s{sd}'
                           f'_finger_seed{s}_ckpt500000')
    return dict(fits=fits, dir_of={f: f for f in fits}, adapt_of=adapt_of,
                updates=500000, full_n=16)
  if wave == 'routedrepair':
    # PREREG_routedrepair_20260820 (+amend1). The reader pins REFITS =
    # ax1wm_finger_rrpq1s1_seed1..8 and three adapt modes per seed
    # (repair/aptctl/snp), so 8 refits gate 24 adapt rows.
    seeds = tuple(range(1, 9)) if seeds is None else seeds
    fits = [f'ax1wm_finger_rrpq1s1_seed{s}' for s in seeds]
    return dict(fits=fits, dir_of={f: f for f in fits}, updates=500000,
                full_n=8, rr_seeds=list(seeds))
  die(f'unknown wave {wave}')


# --------------------------------------------------------------------------
# Realized-state extractors
# --------------------------------------------------------------------------

def latest_ckpt_step(run_dir):
  """Directory-scan value (axis1.sbatch latest_ckpt/ckpt_update
  transcription). AUDIT input only for carrier — see counters_source."""
  dones = sorted(glob.glob(os.path.join(run_dir, 'ckpt', '*', 'done')))
  if not dones:
    return 0
  base = os.path.basename(os.path.dirname(dones[-1]))
  suffix = base.rsplit('-', 1)[-1]
  return int(suffix) if suffix.isdigit() else 0


def progress_update(run_dir):
  path = os.path.join(run_dir, 'OFFLINE_FIT_PROGRESS')
  if not os.path.exists(path):
    return 0
  val = 0
  with open(path) as f:
    for line in f:
      if line.startswith('update='):
        try:
          val = int(line.strip().split('=', 1)[1])
        except ValueError:
          pass
  return val


_EXC_RES = [re.compile(rb'^%s.*$' % e.encode(), re.M)
            for e in CONFIG_EXCEPTIONS]


def config_bytes_excl(path):
  with open(path, 'rb') as f:
    data = f.read()
  for rex in _EXC_RES:
    data = rex.sub(b'', data)
  return data


def config_sha_excl(path):
  return hashlib.sha256(config_bytes_excl(path)).hexdigest()


def _flat(d, prefix=''):
  out = {}
  for k, v in (d or {}).items():
    key = f'{prefix}{k}'
    if isinstance(v, dict):
      out.update(_flat(v, key + '.'))
    else:
      out[key] = v
  return out


FLAT_EXCEPTIONS = ('seed', 'logdir')     # amend1 M5 / amend2
_REPO_DEFAULTS = None


def repo_defaults_flat():
  global _REPO_DEFAULTS
  if _REPO_DEFAULTS is None:
    import ruamel.yaml as yaml
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        'dreamerv3', 'configs.yaml')
    with open(path) as f:
      _REPO_DEFAULTS = _flat(yaml.YAML(typ='safe').load(f)['defaults'])
  return _REPO_DEFAULTS


def flat_config(path):
  import ruamel.yaml as yaml
  with open(path) as f:
    cfg = _flat(yaml.YAML(typ='safe').load(f))
  for exc in FLAT_EXCEPTIONS:
    cfg.pop(exc, None)
  return cfg


def schema_normalized_match(fresh_flat, ref_flat, defaults):
  """Amendment-2 comparison (PREREG_carrier_freshrep_amend2_20260821):
  shared keys value-identical; no reference key removed; every ADDED
  key's value equals the current repo default (the schema's own
  backfill value, never a set dial). Returns (ok, reasons, added)."""
  reasons, added = [], {}
  for k in ref_flat:
    if k not in fresh_flat:
      reasons.append(f'removed key {k}')
    elif fresh_flat[k] != ref_flat[k]:
      reasons.append(f'shared-key value differs: {k} '
                     f'({fresh_flat[k]!r} != {ref_flat[k]!r})')
  for k in fresh_flat:
    if k in ref_flat:
      continue
    added[k] = fresh_flat[k]
    if k not in defaults:
      reasons.append(f'added key {k} has no repo default')
    elif fresh_flat[k] != defaults[k]:
      reasons.append(f'added key {k} = {fresh_flat[k]!r} != repo '
                     f'default {defaults[k]!r} (a SET dial, not '
                     'schema backfill)')
  return not reasons, reasons, added


HOST_RE = re.compile(r'\bhost=(\S+)')
REPLAY_RE = re.compile(r'\bREPLAY=(\S+)')
ADAPT_CKPT_RE = re.compile(r'adapt RUN_ID=\S+ .*?\bCKPT=(\S+)')


def parse_cloud_log(log_path):
  """(last host=, last offline_fit REPLAY=, last adapt CKPT= path)."""
  host = replay = adapt_ckpt = None
  if not os.path.exists(log_path):
    return None, None, None
  with open(log_path, errors='replace') as f:
    for line in f:
      m = HOST_RE.search(line)
      if m:
        host = m.group(1)
      if 'offline_fit ' in line:
        m = REPLAY_RE.search(line)
        if m:
          replay = m.group(1)
      m = ADAPT_CKPT_RE.search(line)
      if m:
        adapt_ckpt = m.group(1)
  return host, replay, adapt_ckpt


def ckpt_step_of_path(ckpt_path):
  base = os.path.basename(ckpt_path.rstrip('/'))
  suffix = base.rsplit('-', 1)[-1]
  return int(suffix) if suffix.isdigit() else 0


def lane_lookup(lane_dir, adapt_run):
  """(lane file basename, REPLAY= from the committed lane cmds) for the
  cell, or (None, None)."""
  if not lane_dir:
    return None, None
  for path in sorted(glob.glob(os.path.join(lane_dir, 'lane_*.cmds'))):
    with open(path, errors='replace') as f:
      for line in f:
        if f'RUN_ID={adapt_run} ' in line:
          m = REPLAY_RE.search(line)
          return os.path.basename(path), (m.group(1) if m else None)
  return None, None


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def ckpt_sha(path):
  """sha256 of a resolved checkpoint: a file directly; a directory as
  the hash of its sorted (name, file-sha) pairs. None if absent."""
  import hashlib
  if os.path.isfile(path):
    return sha256_file(path)
  if os.path.isdir(path):
    h = hashlib.sha256()
    for fn in sorted(os.listdir(path)):
      fp = os.path.join(path, fn)
      if os.path.isfile(fp):
        h.update(fn.encode())
        h.update(bytes.fromhex(sha256_file(fp)))
    return h.hexdigest()
  return None


def _load_refs(ref_configs, problems, meta, wave):
  """Reference configs: bytes for matching (carrier), paths+shas
  recorded in _meta for BOTH carrier and u1a (finding 6)."""
  refs = {}
  if not ref_configs:
    problems.append('no --ref_configs given')
    return refs
  paths = sorted(glob.glob(os.path.join(ref_configs, '*.yaml')))
  if not paths:
    problems.append(f'no *.yaml under {ref_configs}')
  meta['config_ref_paths'] = paths
  meta['config_ref_shas'] = [sha256_file(p) for p in paths]
  if wave == 'carrier':
    for a in ('rgo', 'sgb'):
      for sd in ('0', '1'):
        p = os.path.join(ref_configs, f'{a}s{sd}.yaml')
        if os.path.exists(p):
          refs[f'{a}s{sd}'] = flat_config(p)
        else:
          problems.append(f'reference config missing: {p}')
  return refs


def build_witness(runroot, spec, wave, ref_configs=None, lane_dir=None,
                  job_logs=None):
  w, problems = {}, []
  meta = dict(builder='scripts/ops/build_rescue_bundle.py', wave=wave,
              config_exceptions=list(CONFIG_EXCEPTIONS),
              comparison_mode='schema_normalized_v2 '
              '(PREREG_carrier_freshrep_amend2_20260821: shared keys '
              'value-identical; no removals; additions must equal repo '
              'defaults; exceptions seed/logdir)',
              progress_update_note='audit only; clobberable by stale '
              'cross-instance copies (stale_fit_witness_20260816)')
  refs = (_load_refs(ref_configs, problems, meta, wave)
          if wave in ('carrier', 'u1a') else {})
  all_match = wave == 'carrier' and bool(refs) and len(refs) == 4
  if wave == 'u1a':
    meta['fit_dir_aliases'] = {k: v for k, v in spec['dir_of'].items()
                               if k != v}
  for fit in spec['fits']:
    run_dir = os.path.join(runroot, spec['dir_of'][fit])
    if not os.path.isdir(run_dir):
      die(f'fit run dir missing: {run_dir}')
    cfg = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(cfg):
      die(f'config.yaml missing in {run_dir}')
    scan = latest_ckpt_step(run_dir)
    entry = dict(
        ckpt_scan_update=scan,
        progress_update_audit_only=progress_update(run_dir),
        config_sha_excl_registered=config_sha_excl(cfg))
    # reader compatibility: the frozen readers consume these key names
    entry['config_sha_excl_seed'] = entry['config_sha_excl_registered']
    if wave == 'carrier':
      adapt = spec['adapt_of'][fit]
      log = os.path.join(runroot, '_cloud_logs', f'{adapt}.out')
      host, _, adapt_ckpt = parse_cloud_log(log)
      lane_file, _ = lane_lookup(lane_dir, adapt)
      entry['invocation_id'] = host
      entry['lane_file'] = lane_file
      if host is None:
        problems.append(f'no realized host for {adapt} — invocation '
                        'pairing will refuse')
      # finding 5: counters = the checkpoint the adapt LOADED
      if adapt_ckpt is not None:
        entry['counters'] = ckpt_step_of_path(adapt_ckpt)
        entry['counters_source'] = 'adapt_log'
        if entry['counters'] != scan:
          problems.append(
              f'{fit}: adapt loaded ckpt step {entry["counters"]} but '
              f'directory scan says {scan} — merged-runroot stale/'
              'partial copy suspected; adjudicate before the read')
      else:
        entry['counters'] = scan
        entry['counters_source'] = 'ckpt_scan'
        problems.append(f'{fit}: no adapt CKPT= line in {log} — '
                        'counters fell back to the directory scan')
      key = ('rgo' if 'rgo' in fit else 'sgb') + 's' + \
          fit.split('q1s')[1][0]
      if key in refs:
        ok, reasons, added = schema_normalized_match(
            flat_config(cfg), refs[key], repo_defaults_flat())
        if added and 'added_keys' not in meta:
          meta['added_keys'] = added        # uniform across the wave
        if not ok:
          all_match = False
          problems.append(f'{fit}: Amendment-2 config match failed vs '
                          f'{key}: {reasons[:3]}')
      else:
        all_match = False
    else:
      entry['counters'] = scan
      entry['counters_source'] = 'ckpt_scan_single_host'
    if wave == 'valuefree':
      cfg_flat_all = flat_config(cfg)
      entry['cfg_pins'] = {k: cfg_flat_all.get(k) for k in
                           ('task', 'run.train_ratio', 'run.steps', 'agent.reward_grad', 'agent.repval_grad')}
    if fit in spec.get('online_fits', ()):
      # valuefree budget-parity witness. Review-2 F1/F9: an online
      # run's ckpt dirs carry NO step suffix, so counters come from
      # the REALIZED optimizer updates in metrics.jsonl; the config
      # dial is recorded as nominal only.
      cfg_flat = flat_config(cfg)
      entry['train_ratio_nominal'] = cfg_flat.get('run.train_ratio')
      entry['train_ratio'] = cfg_flat.get('run.train_ratio')
      upd, rr = 0, None
      mt = os.path.join(run_dir, 'metrics.jsonl')
      if os.path.exists(mt):
        with open(mt) as fh:
          for line in fh:
            line = line.strip()
            if not line:
              continue
            try:
              row = json.loads(line)
            except json.JSONDecodeError:
              continue
            # D14: replay/updates is a per-interval sampled-item
            # counter, never an optimizer count — only the real key
            if 'train/opt/updates' in row:
              upd = max(upd, int(row['train/opt/updates']))
            if 'replay/replay_ratio' in row:
              rr = float(row['replay/replay_ratio'])
      entry['counters'] = upd
      entry['counters_source'] = 'metrics_updates'
      entry['replay_ratio_realized'] = rr
      if not upd:
        problems.append(f'{fit}: no realized updates in metrics.jsonl '
                        '— the counters gate will refuse')
      last_step = 0
      sc = os.path.join(run_dir, 'scores.jsonl')
      if os.path.exists(sc):
        with open(sc) as fh:
          for line in fh:
            line = line.strip()
            if line:
              try:
                last_step = max(last_step,
                                int(json.loads(line).get('step', 0)))
              except (ValueError, json.JSONDecodeError):
                pass
      entry['env_steps'] = last_step
      if not last_step:
        problems.append(f'{fit}: no realized env steps (scores.jsonl '
                        'missing/empty) — the budget-parity gate will '
                        'refuse')
    w[fit] = entry
  meta['problems'] = problems
  if wave == 'carrier':
    meta['amend1_config_match'] = bool(all_match)
  if wave == 'valuefree':
    # D5: adapt -> checkpoint linkage. Each arm adapt's saved config
    # must load from ITS registered fit dir; scratchvf must load
    # nothing. Recorded + gated (the reader refuses unless
    # adapt_ckpt_ok is true).
    fit_of = dict()
    for sd_ in spec.get('vf_seeds', ()):
      fit_of[f'adapt_ontask_finger_seed{sd_}_ckpt500000'] = \
          f'ontask_finger_seed{sd_}'
      fit_of[f'adapt_q1uzs1_finger_seed{sd_}_ckpt500000'] = \
          f'ax1wm_finger_rgoq1s1_seed{sd_}'
      fit_of[f'adapt_q1uzs0_finger_seed{sd_}_ckpt500000'] = \
          f'ax1wm_finger_rgoq1s0_seed{sd_}'
      fit_of[f'adapt_scratchvf_finger_seed{sd_}_ckpt0'] = None
    adapt_ckpts, ok = {}, True
    for adapt, fit_dir in fit_of.items():
      cfgp = os.path.join(runroot, adapt, 'config.yaml')
      if not os.path.exists(cfgp):
        ok = False
        problems.append(f'{adapt}: no config.yaml — linkage '
                        'unverifiable')
        continue
      fc = flat_config(cfgp).get('run.from_checkpoint') or ''
      adapt_ckpts[adapt] = fc
      if fit_dir is None:
        if fc:
          ok = False
          problems.append(f'{adapt}: scratch adapt loads a checkpoint '
                          f'({fc}) — protocol broken')
      elif f'/{fit_dir}/' not in fc and not fc.rstrip('/').endswith(
          fit_dir):
        ok = False
        problems.append(f'{adapt}: from_checkpoint {fc!r} does not '
                        f'reference its fit {fit_dir}')
    meta['adapt_ckpts'] = adapt_ckpts
    meta['adapt_ckpt_ok'] = bool(ok)
  if wave == 'crowding':
    # The frozen reader gates check_witness(ADAPTS, 125000): witness
    # entries keyed by the 16 ADAPT run_ids with counters == 125000
    # EXACTLY. An adapt's ckpt dirs carry no step suffix and neither
    # metrics.jsonl (final step trails run.steps by up to one logging
    # cadence, ~1e4) nor scores.jsonl (episode boundaries) ever reads
    # exactly 125000 — so the pinned value is written ONLY when four
    # independent signals agree, else the realized metrics step is
    # written and the reader refuses:
    #   (1) the adapt's saved config run.steps == 125000;
    #   (2) an OK/DONE line for this run_id in the bundle job logs
    #       (bundle.sbatch prints it only on rc=0 + ADAPT_DONE +
    #       n_ep>=100, and rc=0 requires main.py to REACH run.steps);
    #   (3) metrics.jsonl final step >= 115000 (within one cadence);
    #   (4) scores.jsonl >= 100 lines.
    # Plus the prereg's linkage obligations: from_checkpoint must
    # resolve inside the paired fit dir, and the loaded ckpt's sha is
    # recorded.
    done_log = set()
    logdir = job_logs or os.getcwd()
    for fn in sorted(glob.glob(os.path.join(logdir, 'crowd*.out'))):
      with open(fn, errors='replace') as fh:
        for line in fh:
          m = re.search(r'(?:OK|DONE)\s+(adapt_ax1uzruw\S+)', line)
          if m:
            done_log.add(m.group(1))
    if not done_log:
      problems.append(f'no OK/DONE adapt lines in {logdir}/crowd*.out '
                      '— pass --job_logs; completion corroboration '
                      'missing for every adapt')
    for fit, adapt in spec['adapt_of'].items():
      adir = os.path.join(runroot, adapt)
      cfgp = os.path.join(adir, 'config.yaml')
      if not os.path.isdir(adir) or not os.path.exists(cfgp):
        problems.append(f'{adapt}: run dir or config.yaml missing — '
                        'the adapt witness gate will refuse')
        w[adapt] = dict(counters=0, counters_source='missing')
        continue
      cf = flat_config(cfgp)
      entry = dict(
          config_sha_excl_registered=config_sha_excl(cfgp))
      entry['config_sha_excl_seed'] = entry['config_sha_excl_registered']
      steps_cfg = int(float(cf.get('run.steps') or 0))
      entry['run_steps_config'] = steps_cfg
      if steps_cfg != 125000:
        problems.append(f'{adapt}: config run.steps {steps_cfg} != '
                        '125000')
      fc = str(cf.get('run.from_checkpoint') or '')
      entry['from_checkpoint'] = fc
      if f'/{fit}/' not in fc and not fc.rstrip('/').endswith(fit):
        problems.append(f'{adapt}: from_checkpoint {fc!r} does not '
                        f'reference its fit {fit} — linkage broken')
      else:
        sha = ckpt_sha(fc)
        entry['loaded_ckpt_sha'] = sha
        if sha is None:
          problems.append(f'{adapt}: loaded checkpoint {fc!r} no '
                          'longer exists — sha unrecordable')
      last_mstep = 0
      mt = os.path.join(adir, 'metrics.jsonl')
      if os.path.exists(mt):
        with open(mt) as fh:
          for line in fh:
            line = line.strip()
            if line:
              try:
                last_mstep = max(last_mstep,
                                 int(json.loads(line).get('step', 0)))
              except (ValueError, json.JSONDecodeError):
                pass
      entry['realized_final_metrics_step'] = last_mstep
      nep = 0
      sc = os.path.join(adir, 'scores.jsonl')
      if os.path.exists(sc):
        with open(sc) as fh:
          nep = sum(1 for ln in fh if ln.strip())
      entry['scores_lines'] = nep
      in_log = adapt in done_log
      entry['job_log_ok'] = in_log
      complete = (steps_cfg == 125000 and in_log
                  and last_mstep >= 115000 and nep >= 100)
      if complete:
        entry['counters'] = 125000
        entry['counters_source'] = ('config_steps125k+job_log_ok+'
                                    'metrics_ge115k+scores_ge100')
      else:
        entry['counters'] = last_mstep
        entry['counters_source'] = 'incomplete_evidence'
        why = []
        if not in_log:
          why.append('no OK/DONE job-log line')
        if last_mstep < 115000:
          why.append(f'metrics final step {last_mstep} < 115000')
        if nep < 100:
          why.append(f'scores lines {nep} < 100')
        if why:
          problems.append(f'{adapt}: completion NOT established '
                          f'({"; ".join(why)}) — counters left at the '
                          'realized metrics step, reader will refuse')
      w[adapt] = entry
  w['_meta'] = meta
  return w


def build_linkage(runroot, spec, lane_dir=None):
  lk, src, missing = {}, {}, []
  for fit in spec['fits']:
    adapt = spec['adapt_of'][fit]
    _, replay, _ = parse_cloud_log(
        os.path.join(runroot, '_cloud_logs', f'{adapt}.out'))
    if replay is not None:
      lk[fit], src[fit] = replay, 'realized_log'
      continue
    _, lane_replay = lane_lookup(lane_dir, adapt)
    if lane_replay is not None:
      lk[fit], src[fit] = lane_replay, 'lane_cmds'
      continue
    lk[fit], src[fit] = 'UNRESOLVED', 'none'
    missing.append(fit)
  if missing:
    print(f'[linkage] WARNING: {len(missing)} fits unresolved '
          f'(first: {missing[:3]}) — the reader will refuse')
  lk['_sources'] = src
  return lk, missing


def build_buffer_manifest(runroot, dose_manifest, attest_first_draw):
  with open(dose_manifest) as f:
    dm = json.load(f)
  levels = dm.get('levels')
  if levels is None or len(levels) < 2:
    die(f'{dose_manifest}: no levels[1] (need the level-1 curated '
        'buffer entry)')
  lvl = levels[1]
  # finding 4: the build-dose manifest reports occ_recomputed (the
  # MATERIALIZED buffer's occupancy) — never the search's nominal occ
  occ = lvl.get('occ_recomputed')
  if occ is None:
    die(f'{dose_manifest}: level-1 has no occ_recomputed — this must '
        'be the build-dose manifest, not the search output')
  # finding 8: the holdout disposition is VERIFIED, never a constant
  reacher_probesets = glob.glob(
      os.path.join(runroot, 'e4_probesets', '*reacher*'))
  if reacher_probesets:
    disposition = 'PROBESET-NOW-EXISTS'   # reader refuses — correct
  else:
    disposition = 'no-probeset-exists'
  return dict(
      occupancy=float(occ),
      occ_search=lvl.get('occ_search'),
      n_episodes=lvl.get('n_episodes'),
      n_transitions=lvl.get('n_transitions'),
      holdout_disposition=disposition,
      holdout_check=('glob(e4_probesets/*reacher*) empty'
                     if not reacher_probesets else
                     f'FOUND {reacher_probesets[:3]}'),
      first_draw=bool(attest_first_draw),
      source_manifest=os.path.abspath(dose_manifest),
      source_manifest_sha=sha256_file(dose_manifest))


def run_collate(runroot, out_dir):
  rc = subprocess.call([
      sys.executable, '-m', 'analysis.adaptation_auc',
      '--runroot', runroot, '--output', out_dir])
  if rc != 0:
    die(f'adaptation_auc collate failed rc={rc}')
  csv_path = os.path.join(out_dir, 'auc.csv')
  if not os.path.exists(csv_path):
    die('collate produced no auc.csv')
  return csv_path


def write_manifest(bundle):
  lines = []
  for path in sorted(glob.glob(os.path.join(bundle, '**', '*'),
                               recursive=True)):
    if os.path.isfile(path) and not path.endswith('MANIFEST.sha256'):
      rel = os.path.relpath(path, bundle)
      lines.append(f'{sha256_file(path)}  {rel}')
  with open(os.path.join(bundle, 'MANIFEST.sha256'), 'w') as f:
    f.write('\n'.join(lines) + '\n')


def expected_adapt_keys(wave, spec, seeds):
  keys = set()
  for fit in spec['fits']:
    if wave == 'carrier':
      arm = 'rgo' if 'rgo' in fit else 'sgb'
      sd = fit.split('q1s')[1][0]
      keys.add((f'ax1{arm}q1s{sd}', 'finger', int(fit.split('_seed')[1])))
    elif wave == 'u1a':
      p = 'uz' if '_uzq1' in fit else 'uzf'
      sd = fit.split('q1s')[1][0]
      keys.add((f'ax1{p}q1s{sd}', 'finger', int(fit.split('_seed')[1])))
  if wave == 'mindose':
    for mode in ('ax1mdd1', 'scratch'):
      for s in (range(1, 9) if seeds is None else seeds):
        keys.add((mode, 'reacher', s))
  if wave == 'crowding':
    for w_ in (1, 100):
      for sd in ('0', '1'):
        for s in (range(1, 5) if seeds is None else seeds):
          keys.add((f'ax1uzruw{w_}q1s{sd}', 'finger', s))
  if wave == 'routedrepair':
    for mode in ('ax1rrpq1s1', 'ax1raftq1s1', 'ax1rspq1s1'):
      for s in (range(1, 9) if seeds is None else seeds):
        keys.add((mode, 'finger', s))
  if wave == 'valuefree':                   # review-2 F7 / review-3 B1
    for mode in ('ontask', 'q1uzs1', 'q1uzs0', 'scratchvf'):
      for s in (range(17, 25) if seeds is None else seeds):
        keys.add((mode, 'finger', s))
  return keys


def build(args, seeds=None):
  spec = wave_spec(args.wave, seeds=seeds)
  if getattr(args, 'extra_fits', None):
    extras = [f.strip() for f in args.extra_fits.split(',') if f.strip()]
    want = spec.get('extra_expect')
    if want is not None and seeds is None and len(extras) != want:
      die(f'{len(extras)} extra fits != registered {want}')
    for f in extras:
      spec['fits'].append(f)
      spec['dir_of'][f] = f
    spec['full_n'] += len(extras)          # review-2 F6
  if seeds is None and len(spec['fits']) != spec['full_n']:
    die(f'fit list {len(spec["fits"])} != registered {spec["full_n"]}')
  final = args.output.rstrip('/')
  if os.path.exists(final):
    die(f'{final} already exists')
  work = final + '.building'
  if os.path.exists(work):
    shutil.rmtree(work)
  os.makedirs(work)
  csv_path = run_collate(args.runroot, work)
  import csv as _csv
  with open(csv_path) as f:
    have = {(r['mode'], r['domain'], int(r['seed']))
            for r in _csv.DictReader(f)}
  missing = sorted(expected_adapt_keys(args.wave, spec, seeds) - have)
  if missing:
    shutil.rmtree(work)
    die(f'{len(missing)} expected adapt rows missing from the collate '
        f'(first: {missing[:4]}) — wave not drained; nothing bundled')
  witness = build_witness(args.runroot, spec, args.wave,
                          ref_configs=args.ref_configs,
                          lane_dir=args.lane_cmds,
                          job_logs=args.job_logs)

  if args.wave == 'crowding':
    # amend1 B8: the reader refuses unless the adapt namespace was EMPTY at
    # submission. That is not reconstructable from directory mtimes -- but
    # submit_rcc.sh SKIPS any cell whose run dir already exists, so a cell
    # that reached a `CELL <run_id>` line in a job log provably had no dir
    # at submission. Deriving the flag from that evidence; never asserting it.
    started, want = set(), set(spec['adapt_of'].values())
    logdir = args.job_logs or os.getcwd()
    for fn in sorted(glob.glob(os.path.join(logdir, 'crowd*.out'))):
      with open(fn, errors='replace') as fh:
        for line in fh:
          m = re.search(r'CELL (adapt_ax1uzruw\S+)', line)
          if m:
            started.add(m.group(1))
    absent = sorted(want - started)
    witness['_meta']['no_preexisting_dirs'] = not absent
    witness['_meta']['no_preexisting_dirs_evidence'] = (
        'submit_rcc.sh:40 skips a cell whose run dir exists; %d/%d pinned '
        'cells appear as CELL starts in %s/crowd*.out'
        % (len(want) - len(absent), len(want), logdir))
    if absent:
      witness['_meta']['problems'].append(
          'no_preexisting_dirs NOT established: no CELL start found for '
          + ', '.join(absent[:4]))
    shutil.copyfile(csv_path, os.path.join(work, 'frozen_auc.csv'))

  if args.wave == 'routedrepair':
    # amend1: all three arms at a seed must trace to ONE donor sha, and
    # latest_ckpt() must resolve 500000 for 8/8 donors.
    if not args.donor_provenance:
      shutil.rmtree(work)
      die('--donor_provenance required for routedrepair')
    with open(args.donor_provenance) as f:
      dp = json.load(f)
    donors = dp.get('donors', {})
    witness['_meta']['donor_generation'] = (
        dp.get('_meta', {}).get('donor_generation')
        or 'REFIT_20260808+seed5_completion_20260820')
    ok500 = 0
    for fit in spec['fits']:
      if latest_ckpt_step(os.path.join(args.runroot,
                                       spec['dir_of'][fit])) == 500000:
        ok500 += 1
    witness['_meta']['latest_ckpt_500k_8of8'] = (ok500 == len(spec['fits']))
    if ok500 != len(spec['fits']):
      witness['_meta']['problems'].append(
          'latest_ckpt 500000 for only %d/%d refits' % (ok500,
                                                        len(spec['fits'])))
    for s in spec['rr_seeds']:
      key = 'ax1wm_finger_fq1s1_seed%d' % s
      sha = (donors.get(key) or {}).get('donor_sha')
      if sha is None:
        witness['_meta']['problems'].append(
            'donor_provenance has no donor_sha for %s' % key)
      for arm in ('repair', 'aptctl', 'snp'):
        witness['donor_sha/%s/seed%d' % (arm, s)] = sha

  with open(os.path.join(work, 'witness.json'), 'w') as f:
    json.dump(witness, f, indent=1)
  if args.wave == 'carrier':
    lk, _ = build_linkage(args.runroot, spec, lane_dir=args.lane_cmds)
    with open(os.path.join(work, 'linkage.json'), 'w') as f:
      json.dump(lk, f, indent=1)
  if args.wave == 'mindose':
    if not args.dose_manifest:
      shutil.rmtree(work)
      die('--dose_manifest required for mindose')
    bm = build_buffer_manifest(args.runroot, args.dose_manifest,
                               args.attest_first_draw)
    with open(os.path.join(work, 'buffer_manifest.json'), 'w') as f:
      json.dump(bm, f, indent=1)
  write_manifest(work)
  os.rename(work, final)
  probs = witness['_meta']['problems']
  print(f'[bundle] {final}  fits={len(spec["fits"])} '
        f'problems={len(probs)}')
  for p in probs:
    print(f'  PROBLEM: {p}')
  if probs and not args.allow_problems:
    print('EXIT 3: bundle carries problems — fix them (or re-run with '
          '--allow_problems after adjudication) BEFORE the ONE read.')
    return 3
  return 0


# --------------------------------------------------------------------------
# Selfcheck (synthetic runroot; realistic config shapes — finding 10)
# --------------------------------------------------------------------------

CFG_TMPL = ('agent: {x: 1}\nenv:\n  dmc: {use_seed: true}\n'
            'logdir: /rr/%s\nreplay: {size: 1}\nseed: %d\nz: 2\n')


def _mk_fit(root, name, ckpt_step, progress, seed, extra='',
            replace_cfg=None):
  d = os.path.join(root, name)
  os.makedirs(os.path.join(d, 'ckpt',
                           f'20260101T000000-{ckpt_step:012d}'),
              exist_ok=True)
  open(os.path.join(d, 'ckpt', f'20260101T000000-{ckpt_step:012d}',
                    'done'), 'w').close()
  with open(os.path.join(d, 'OFFLINE_FIT_PROGRESS'), 'w') as f:
    f.write(f'update={progress}\n')
  with open(os.path.join(d, 'config.yaml'), 'w') as f:
    if replace_cfg is not None:
      f.write(replace_cfg)
    else:
      f.write(CFG_TMPL % (name, seed) + extra)
  return d


def _mk_adapt(root, name, n_ep=25):
  d = os.path.join(root, name)
  os.makedirs(d, exist_ok=True)
  with open(os.path.join(d, 'scores.jsonl'), 'w') as f:
    for i in range(n_ep):
      f.write(json.dumps({'step': 1000 * (i + 1),
                          'episode/score': 50.0 + i}) + '\n')


def _mk_cloud_log(root, adapt, fit, seed, sd, ckpt_step=500000,
                  skip_fit=False):
  with open(os.path.join(root, '_cloud_logs', f'{adapt}.out'), 'w') as f:
    f.write(f'axis1 child starting RUN_ID={adapt} job=NA '
            f'host=box{seed}\n')
    if not skip_fit:
      f.write(f'offline_fit WM_RUN={fit} '
              f'REPLAY=/rr/axis1_finger/q1/side{sd} '
              f'UPDATES=500000 SEED={seed}\n')
    f.write(f'adapt RUN_ID={adapt} '
            f'CKPT=/rr/{fit}/ckpt/20260101T000000-{ckpt_step:012d} '
            f'STEPS=1.25e5\n')


def _ns(**kw):
  base = dict(wave=None, runroot=None, output=None, ref_configs=None,
              lane_cmds=None, dose_manifest=None, extra_fits=None,
              attest_first_draw=False, allow_problems=False,
              job_logs=None, donor_provenance=None)
  base.update(kw)
  return argparse.Namespace(**base)


def selfcheck():
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    root = os.path.join(tmp, 'runroot')
    os.makedirs(os.path.join(root, '_cloud_logs'))
    lane_dir = os.path.join(tmp, 'lanes')
    os.makedirs(lane_dir)
    refdir = os.path.join(tmp, 'refs')
    os.makedirs(refdir)
    for a in ('rgo', 'sgb'):
      for sd in ('0', '1'):
        arm_extra = f'arm: {a}\n'
        with open(os.path.join(refdir, f'{a}s{sd}.yaml'), 'w') as f:
          f.write(CFG_TMPL % ('REF', 999) + arm_extra)
    seeds = (17, 18)
    lane_lines = []
    for a in ('rgo', 'sgb'):
      for sd in ('0', '1'):
        for s in seeds:
          fit = f'ax1wm_finger_{a}q1s{sd}_seed{s}'
          adapt = f'adapt_ax1{a}q1s{sd}_finger_seed{s}_ckpt500000'
          _mk_fit(root, fit, 500000, 500000, s, extra=f'arm: {a}\n')
          _mk_adapt(root, adapt)
          _mk_cloud_log(root, adapt, fit, s, sd)
          lane_lines.append(
              f'RUN_ID={adapt} WM_RUN={fit} '
              f'REPLAY=/rr/axis1_finger/q1/side{sd} bash x')
    with open(os.path.join(lane_dir, 'lane_blk1_0.cmds'), 'w') as f:
      f.write('\n'.join(lane_lines) + '\n')

    # --- carrier end-to-end: seed+logdir excepted sha identity, match
    # flag true, adapt-log counters, manifest, exit 0
    out1 = os.path.join(tmp, 'b1')
    rc = build(_ns(wave='carrier', runroot=root, output=out1,
                   ref_configs=refdir, lane_cmds=lane_dir), seeds=seeds)
    assert rc == 0, rc
    w = json.load(open(os.path.join(out1, 'witness.json')))
    assert w['_meta']['amend1_config_match'] is True, w['_meta']
    assert w['_meta']['config_ref_paths'], 'refs not recorded (f6)'
    e = w['ax1wm_finger_rgoq1s0_seed17']
    assert e['counters'] == 500000 and e['counters_source'] == 'adapt_log'
    assert e['invocation_id'] == 'box17'
    assert e['lane_file'] == 'lane_blk1_0.cmds'
    a_ = w['ax1wm_finger_rgoq1s0_seed17']['config_sha_excl_registered']
    b_ = w['ax1wm_finger_rgoq1s0_seed18']['config_sha_excl_registered']
    assert a_ == b_, 'per-seed logdir must be excepted (f3)'
    assert a_ != w['ax1wm_finger_sgbq1s0_seed17'][
        'config_sha_excl_registered'], 'arms must differ'
    lk = json.load(open(os.path.join(out1, 'linkage.json')))
    assert lk['ax1wm_finger_sgbq1s1_seed18'] == \
        '/rr/axis1_finger/q1/side1'
    assert os.path.exists(os.path.join(out1, 'MANIFEST.sha256'))
    assert not os.path.exists(out1 + '.building')

    # --- stale-copy fabrication path (f5): adapt loaded 100000 while a
    # merged scan shows 500000 -> counters=100000 + problem + exit 3
    fit = 'ax1wm_finger_rgoq1s0_seed17'
    adapt = 'adapt_ax1rgoq1s0_finger_seed17_ckpt500000'
    _mk_cloud_log(root, adapt, fit, 17, '0', ckpt_step=100000)
    w2 = build_witness(root, wave_spec('carrier', seeds=seeds),
                       'carrier', ref_configs=refdir, lane_dir=lane_dir)
    assert w2[fit]['counters'] == 100000
    assert w2[fit]['ckpt_scan_update'] == 500000
    assert any('stale/partial copy suspected' in p.replace('\n', ' ')
               or 'stale' in p for p in w2['_meta']['problems'])
    rc3 = build(_ns(wave='carrier', runroot=root,
                    output=os.path.join(tmp, 'b1p'),
                    ref_configs=refdir, lane_cmds=lane_dir), seeds=seeds)
    assert rc3 == 3, rc3
    _mk_cloud_log(root, adapt, fit, 17, '0')   # restore

    # --- idempotent-skip log (f7): no offline_fit line -> lane fallback
    _mk_cloud_log(root, adapt, fit, 17, '0', skip_fit=True)
    lk2, missing = build_linkage(root, wave_spec('carrier', seeds=seeds),
                                 lane_dir=lane_dir)
    assert not missing
    assert lk2[fit] == '/rr/axis1_finger/q1/side0'
    assert lk2['_sources'][fit] == 'lane_cmds'
    _mk_cloud_log(root, adapt, fit, 17, '0')   # restore

    # --- amend2 schema normalization: an added key AT the repo default
    # keeps match True and is recorded; a non-default value, an added
    # key with no default, or a removed key flips it False
    with open(os.path.join(root, fit, 'config.yaml'), 'a') as f:
      f.write("orthreward: {task: ''}\n")
    w2a = build_witness(root, wave_spec('carrier', seeds=seeds),
                        'carrier', ref_configs=refdir, lane_dir=lane_dir)
    assert w2a['_meta']['amend1_config_match'] is True, \
        w2a['_meta']['problems']
    assert w2a['_meta']['added_keys'] == {'orthreward.task': ''}
    with open(os.path.join(root, fit, 'config.yaml'), 'a') as f:
      f.write("planted: {source_key: oops}\n")
    w2b = build_witness(root, wave_spec('carrier', seeds=seeds),
                        'carrier', ref_configs=refdir, lane_dir=lane_dir)
    assert w2b['_meta']['amend1_config_match'] is False
    assert any('SET dial' in p for p in w2b['_meta']['problems'])
    _mk_fit(root, fit, 500000, 500000, 17, extra='arm: rgo\n')
    with open(os.path.join(root, fit, 'config.yaml'), 'a') as f:
      f.write('drift: 1\n')   # added key with NO repo default
    w3 = build_witness(root, wave_spec('carrier', seeds=seeds),
                       'carrier', ref_configs=refdir, lane_dir=lane_dir)
    assert w3['_meta']['amend1_config_match'] is False
    assert any('no repo default' in p for p in w3['_meta']['problems'])
    # removed key
    cfgp = os.path.join(root, fit, 'config.yaml')
    _mk_fit(root, fit, 500000, 500000, 17, extra='arm: rgo\n')
    with open(cfgp) as f:
      body = f.read().replace('z: 2\n', '')
    with open(cfgp, 'w') as f:
      f.write(body)
    w3b = build_witness(root, wave_spec('carrier', seeds=seeds),
                        'carrier', ref_configs=refdir, lane_dir=lane_dir)
    assert w3b['_meta']['amend1_config_match'] is False
    assert any('removed key z' in p for p in w3b['_meta']['problems'])
    _mk_fit(root, fit, 500000, 500000, 17, extra='arm: rgo\n')
    w4 = build_witness(root, wave_spec('carrier', seeds=seeds),
                       'carrier', ref_configs=None, lane_dir=lane_dir)
    assert w4['_meta']['amend1_config_match'] is False

    # --- u1a END-TO-END (f1/f2): realized dirs are q1s*/fq1s*, witness
    # keyed by the reader's uz/uzf names, aliases recorded
    useeds = (9, 10)
    for p, real, mode in (('uz', '', 'ax1uzq1s'), ('uzf', 'f',
                                                   'ax1uzfq1s')):
      for sd in ('0', '1'):
        for s in useeds:
          _mk_fit(root, f'ax1wm_finger_{real}q1s{sd}_seed{s}',
                  500000, 500000, s, extra=f'p: {p}\n')
          _mk_adapt(root, f'adapt_{mode}{sd}_finger_seed{s}_ckpt500000')
    out2 = os.path.join(tmp, 'b2')
    rc = build(_ns(wave='u1a', runroot=root, output=out2,
                   ref_configs=refdir), seeds=useeds)
    assert rc == 0, rc
    wu = json.load(open(os.path.join(out2, 'witness.json')))
    assert 'ax1wm_finger_uzq1s0_seed9' in wu
    assert wu['_meta']['fit_dir_aliases'][
        'ax1wm_finger_uzfq1s1_seed10'] == 'ax1wm_finger_fq1s1_seed10'
    assert wu['_meta']['config_ref_paths']
    assert wu['ax1wm_finger_uzq1s0_seed9']['counters_source'] == \
        'ckpt_scan_single_host'

    # --- mindose END-TO-END (f4/f8): build-dose manifest shape
    for s in (1, 2):
      _mk_fit(root, f'ax1wm_reacher_mdd1_seed{s}', 500000, 500000, s)
      _mk_adapt(root, f'adapt_ax1mdd1_reacher_seed{s}_ckpt500000')
      _mk_adapt(root, f'adapt_scratch_reacher_seed{s}_ckpt500000')
    dm = os.path.join(tmp, 'dm.json')
    with open(dm, 'w') as f:
      json.dump({'levels': [
          {'occ_recomputed': 0.5, 'occ_search': 0.51},
          {'occ_recomputed': 0.097837, 'occ_search': 0.0995,
           'n_episodes': 40, 'n_transitions': 20000}]}, f)
    out3 = os.path.join(tmp, 'b3')
    rc = build(_ns(wave='mindose', runroot=root, output=out3,
                   dose_manifest=dm, attest_first_draw=True),
               seeds=(1, 2))
    assert rc == 0, rc
    bm = json.load(open(os.path.join(out3, 'buffer_manifest.json')))
    assert bm['occupancy'] == 0.097837
    assert bm['holdout_disposition'] == 'no-probeset-exists'
    assert bm['first_draw'] is True
    # search-output shape (occ, no occ_recomputed) refuses (f4)
    with open(dm, 'w') as f:
      json.dump({'levels': [{'occ': 0.5}, {'occ': 0.0995}]}, f)
    try:
      build_buffer_manifest(root, dm, True)
      raise AssertionError('search-output manifest not refused')
    except SystemExit as e:
      assert 'occ_recomputed' in str(e), e
    # a reacher probeset appearing flips the disposition (f8)
    with open(dm, 'w') as f:
      json.dump({'levels': [{'occ_recomputed': 0.5},
                            {'occ_recomputed': 0.0978}]}, f)
    os.makedirs(os.path.join(root, 'e4_probesets'), exist_ok=True)
    open(os.path.join(root, 'e4_probesets', 'reacher_probe.npz'),
         'w').close()
    bm2 = build_buffer_manifest(root, dm, False)
    assert bm2['holdout_disposition'] == 'PROBESET-NOW-EXISTS'
    assert bm2['first_draw'] is False   # attestation never defaults on
    shutil.rmtree(os.path.join(root, 'e4_probesets'))

    # --- valuefree END-TO-END v3 (F13 + D5/D10/D14/D19): online-shaped
    # ckpt dirs, metrics counters, cfg_pins, adapt->ckpt linkage
    vf_seeds = (17, 18)
    q1_extras = []
    def cfg_vf(name, seed):
      return ('agent: {reward_grad: true, repval_grad: false, x: 1}\n'
              'task: dmc_finger_turn_hard\n'
              f'logdir: /rr/{name}\n'
              'run: {train_ratio: 1024.0, steps: 500000.0}\n'
              f'seed: {seed}\n')
    for sv in vf_seeds:
      vf = f'ontask_finger_seed{sv}'
      d_vf = _mk_fit(root, vf, 500000, 500000, sv,
                     replace_cfg=cfg_vf(vf, sv))
      shutil.rmtree(os.path.join(d_vf, 'ckpt'))
      os.makedirs(os.path.join(d_vf, 'ckpt', '20260821T120000F000001'))
      open(os.path.join(d_vf, 'ckpt', '20260821T120000F000001',
                        'done'), 'w').close()
      with open(os.path.join(d_vf, 'metrics.jsonl'), 'w') as f:
        f.write(json.dumps({'step': 1, 'train/opt/updates': 100}) + '\n')
        f.write(json.dumps({'train/opt/updates': 494368,
                            'replay/updates': 4980000,
                            'replay/replay_ratio': 1040.0}) + '\n')
      with open(os.path.join(d_vf, 'scores.jsonl'), 'w') as f:
        for st in (100000, 496496):
          f.write(json.dumps({'step': st, 'episode/score': 1.0}) + '\n')
      for sd_tok in ('1', '0'):
        qf = f'ax1wm_finger_rgoq1s{sd_tok}_seed{sv}'
        # unconditional rewrite: the carrier leg above created these
        # names with ITS fixture config — the valuefree leg needs the
        # pinned-key config
        _mk_fit(root, qf, 500000, 500000, sv,
                replace_cfg=cfg_vf(qf, sv))
        if qf not in q1_extras:
          q1_extras.append(qf)
      for mode, fit_ref in (('ontask', vf),
                            ('q1uzs1',
                             f'ax1wm_finger_rgoq1s1_seed{sv}'),
                            ('q1uzs0',
                             f'ax1wm_finger_rgoq1s0_seed{sv}'),
                            ('scratchvf', None)):
        ms = 0 if mode == 'scratchvf' else 500000
        ad = f'adapt_{mode}_finger_seed{sv}_ckpt{ms}'
        _mk_adapt(root, ad)
        with open(os.path.join(root, ad, 'config.yaml'), 'w') as f:
          fc = ('' if fit_ref is None else
                f'/rr/{fit_ref}/ckpt/20260101T000000-000000500000')
          f.write(f"run:\n  from_checkpoint: '{fc}'\nseed: {sv}\n")
    out_vf = os.path.join(tmp, 'bvf')
    rc_vf = build(_ns(wave='valuefree', runroot=root, output=out_vf,
                      extra_fits=','.join(q1_extras)), seeds=vf_seeds)
    assert rc_vf == 0, rc_vf
    wvf = json.load(open(os.path.join(out_vf, 'witness.json')))
    e = wvf['ontask_finger_seed17']
    assert e['counters'] == 494368, e         # D14: replay/updates
    assert e['counters_source'] == 'metrics_updates'   # never latched
    assert e['ckpt_scan_update'] == 0
    assert e['replay_ratio_realized'] == 1040.0
    assert e['env_steps'] == 496496
    assert e['cfg_pins']['task'] == 'dmc_finger_turn_hard'
    assert e['cfg_pins']['agent.repval_grad'] is False
    assert wvf['ax1wm_finger_rgoq1s1_seed18']['cfg_pins'][
        'agent.reward_grad'] is True
    assert wvf['_meta']['adapt_ckpt_ok'] is True, wvf['_meta']
    # D5 mismatch: point one arm adapt at the WRONG fit -> ok False
    bad = os.path.join(root,
                       'adapt_q1uzs1_finger_seed17_ckpt500000',
                       'config.yaml')
    with open(bad, 'w') as f:
      f.write("run:\n  from_checkpoint: "
              "'/rr/ax1wm_finger_rgoq1s0_seed17/ckpt/x'\nseed: 17\n")
    wbad = build_witness(root, wave_spec('valuefree', seeds=vf_seeds),
                         'valuefree')
    assert wbad['_meta']['adapt_ckpt_ok'] is False
    with open(bad, 'w') as f:
      f.write("run:\n  from_checkpoint: "
              "'/rr/ax1wm_finger_rgoq1s1_seed17/ckpt/x'\nseed: 17\n")
    # missing metrics -> problem flagged
    os.remove(os.path.join(root, 'ontask_finger_seed17',
                           'metrics.jsonl'))
    wv2 = build_witness(root, wave_spec('valuefree', seeds=(17,)),
                        'valuefree')
    assert wv2['ontask_finger_seed17']['counters'] == 0
    assert any('counters gate will refuse' in x
               for x in wv2['_meta']['problems'])
    # review-3 B1: the PRODUCTION (seeds=None) expected keys are the
    # registered seeds 17-24, never 1-8
    prod_keys = expected_adapt_keys('valuefree',
                                    wave_spec('valuefree'), None)
    assert ('ontask', 'finger', 17) in prod_keys
    assert ('ontask', 'finger', 1) not in prod_keys
    assert len(prod_keys) == 32
    # D19: the extra_expect count guard (seeds=None path)
    try:
      build(_ns(wave='valuefree', runroot=root,
                output=os.path.join(tmp, 'bvf19'),
                extra_fits='a,b,c'), seeds=None)
      raise AssertionError('extra_expect guard failed to trip')
    except SystemExit as e19:
      assert 'extra fits != registered' in str(e19), e19
    # missing adapt row -> nothing bundled (F7 real)
    shutil.rmtree(os.path.join(root,
                               'adapt_scratchvf_finger_seed18_ckpt0'))
    try:
      build(_ns(wave='valuefree', runroot=root,
                output=os.path.join(tmp, 'bvf2'),
                extra_fits=','.join(q1_extras)), seeds=vf_seeds)
      raise AssertionError('valuefree adapt-row refusal failed to trip')
    except SystemExit as e2:
      assert 'nothing bundled' in str(e2), e2

    # --- missing adapt row -> nothing bundled, no half-built dir
    shutil.rmtree(os.path.join(
        root, 'adapt_ax1rgoq1s0_finger_seed17_ckpt500000'))
    out4 = os.path.join(tmp, 'b4')
    try:
      build(_ns(wave='carrier', runroot=root, output=out4,
                ref_configs=refdir, lane_cmds=lane_dir), seeds=seeds)
      raise AssertionError('missing-adapt refusal failed to trip')
    except SystemExit as e:
      assert 'nothing bundled' in str(e), e
    assert not os.path.exists(out4) and not os.path.exists(
        out4 + '.building')
  # --- crowding adapt-witness leg (unit level: build_witness only).
  # The frozen reader gates counters == 125000 on the 16 ADAPT run_ids;
  # no realized log counter lands exactly there, so the pinned value is
  # written only under the four-signal completion evidence.
  with tempfile.TemporaryDirectory() as tmp:
    root = os.path.join(tmp, 'rr')
    logdir = os.path.join(tmp, 'logs')
    os.makedirs(logdir)
    cspec = wave_spec('crowding', seeds=(1,))
    log_lines = []
    for fit, adapt in cspec['adapt_of'].items():
      fdir = _mk_fit(root, fit, 500000, 500000, 1)
      ck = os.path.join(fdir, 'ckpt', '20260101T000000-000000500000')
      adir = os.path.join(root, adapt)
      os.makedirs(adir)
      with open(os.path.join(adir, 'config.yaml'), 'w') as f:
        f.write('run:\n  steps: 125000.0\n'
                f'  from_checkpoint: {ck}\nseed: 1\n')
      with open(os.path.join(adir, 'metrics.jsonl'), 'w') as f:
        for st in (50000, 114256, 123648):
          f.write(json.dumps({'step': st}) + '\n')
      with open(os.path.join(adir, 'scores.jsonl'), 'w') as f:
        for i in range(110):
          f.write(json.dumps({'step': 1000 * (i + 1),
                              'episode/score': 1.0}) + '\n')
      log_lines.append(f'CELL {adapt}  (86400s of walltime left)\n')
      log_lines.append(f'OK     {adapt} (scores.jsonl 110 lines)\n')
    with open(os.path.join(logdir, 'crowd_1.out'), 'w') as f:
      f.writelines(log_lines)
    wgood = build_witness(root, cspec, 'crowding', job_logs=logdir)
    for adapt in cspec['adapt_of'].values():
      e = wgood[adapt]
      assert e['counters'] == 125000, e
      assert e['counters_source'].startswith('config_steps125k'), e
      assert e['loaded_ckpt_sha'], e
    assert wgood['_meta']['problems'] == [], wgood['_meta']['problems']

    # (a) OK line missing for one adapt -> realized counters + problem
    bad = sorted(cspec['adapt_of'].values())[0]
    with open(os.path.join(logdir, 'crowd_1.out'), 'w') as f:
      f.writelines(ln for ln in log_lines if f'OK     {bad}' not in ln)
    wa = build_witness(root, cspec, 'crowding', job_logs=logdir)
    assert wa[bad]['counters'] == 123648, wa[bad]
    assert any('no OK/DONE' in p for p in wa['_meta']['problems'])
    with open(os.path.join(logdir, 'crowd_1.out'), 'w') as f:
      f.writelines(log_lines)

    # (b) truncated metrics -> completion not established
    mt = os.path.join(root, bad, 'metrics.jsonl')
    with open(mt, 'w') as f:
      f.write(json.dumps({'step': 90000}) + '\n')
    wb = build_witness(root, cspec, 'crowding', job_logs=logdir)
    assert wb[bad]['counters'] == 90000, wb[bad]
    assert any('90000 < 115000' in p for p in wb['_meta']['problems'])
    with open(mt, 'w') as f:
      for st in (50000, 114256, 123648):
        f.write(json.dumps({'step': st}) + '\n')

    # (c) from_checkpoint pointing at a foreign fit -> linkage problem
    cfgp = os.path.join(root, bad, 'config.yaml')
    with open(cfgp, 'w') as f:
      f.write('run:\n  steps: 125000.0\n'
              '  from_checkpoint: /rr/ax1wm_finger_other_seed9/ckpt/x\n'
              'seed: 1\n')
    wc = build_witness(root, cspec, 'crowding', job_logs=logdir)
    assert any('linkage broken' in p for p in wc['_meta']['problems'])
  print('SELFCHECK PASS (build_rescue_bundle v2: carrier end-to-end '
        'incl. seed+logdir-excepted sha identity + arm distinction + '
        'adapt-log counters + lane file; stale-copy path -> counters '
        'from adapt log + problem + exit 3; idempotent-skip -> lane '
        'fallback linkage; drift/missing-refs -> match false; u1a '
        'END-TO-END with realized q1s*/fq1s* dirs aliased under reader '
        'keys + refs recorded; mindose END-TO-END with build-dose '
        'manifest shape + search-shape refusal + probeset-appearance '
        'flips disposition + attestation never defaults on; missing '
        'adapt row -> nothing bundled, no half-built dir; crowding '
        'adapt witnesses: counters=125000 only under four-signal '
        'completion evidence (config steps + job-log OK + metrics '
        'within one cadence + scores lines), linkage + loaded-ckpt '
        'sha recorded, each signal drop -> realized counters + '
        'problem)')


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--wave', choices=['carrier', 'u1a', 'mindose',
                                    'valuefree', 'crowding',
                                    'routedrepair'])
  p.add_argument('--job_logs',
                 help='dir of the wave\'s slurm .out logs; crowding '
                      'derives _meta.no_preexisting_dirs from them')
  p.add_argument('--donor_provenance',
                 help='routedrepair donor_provenance.json')
  p.add_argument('--extra_fits',
                 help='valuefree: comma list of the q1-arm fit run '
                      'names (freeze FILL)')
  p.add_argument('--runroot')
  p.add_argument('--output')
  p.add_argument('--ref_configs',
                 help='carrier: dir with {rgo,sgb}s{0,1}.yaml '
                      'Amendment-1 reference configs; u1a: '
                      'finger_refit_20260810 config dir (recorded)')
  p.add_argument('--lane_cmds',
                 help='carrier: the committed generated/ lane dir '
                      '(invocation lane recording + linkage fallback)')
  p.add_argument('--dose_manifest',
                 help='mindose: the BUILD-dose manifest.json (must '
                      'carry occ_recomputed)')
  p.add_argument('--attest_first_draw', action='store_true',
                 help='mindose: ops attests this bundle uses the FIRST '
                      'invocation of the pinned build-dose command '
                      '(amend1 B10). Never defaults on.')
  p.add_argument('--allow_problems', action='store_true',
                 help='exit 0 despite recorded problems (deliberate, '
                      'post-adjudication only)')
  p.add_argument('--selfcheck', action='store_true')
  args = p.parse_args()
  if args.selfcheck:
    selfcheck()
    return
  if not (args.wave and args.runroot and args.output):
    p.error('--wave, --runroot, --output required (or --selfcheck)')
  args.runroot = os.path.abspath(args.runroot)
  sys.exit(build(args))


if __name__ == '__main__':
  main()
