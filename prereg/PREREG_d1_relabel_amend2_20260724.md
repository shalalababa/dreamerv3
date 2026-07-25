# PREREG AMENDMENT 2 to the D1 relabel campaign (frozen 2026-07-24)

Amends `PREREG_d1_relabel_20260724.md` / Amendment 1. Trigger:
Amendment 1's smoke stage FAILED at agent construction on the pulled
Gate-D1 pilots — `AttributeError: model_obs`. Root cause: CONFIG
SCHEMA DRIFT, not a labeler defect. The d1pilot runs were trained
18 Jul, before `agent.model_obs` (and the `env.synth.*` /
`orthreward.task` keys) were added to the config schema (19–23 Jul
build waves); the current agent constructor requires
`config.model_obs`, and the run's saved `config.yaml` predates the
key. The 23-Jul shift pilots carry the full current schema, which is
why the base campaign's smoke passed. NO corrected d1pilot label
exists (the crash preceded any labeling) — the Amendment-1 ordering
guard held.

## Registered fix (loader only; zero label-semantics change)

`d0/sweep.py load_config` now backfills keys that are PRESENT in the
repo's current `dreamerv3/configs.yaml` defaults but ABSENT from a
run's saved config, before constructing the agent:

- saved values always win (only absent keys are added; leaf/subtree
  type-conflict guards on both sides);
- every backfilled key is PRINTED, so each label pass's log records
  exactly what was filled;
- `agent.model_obs` default `'.*'` reproduces the pre-key
  include-everything encoder/decoder behavior byte-for-byte (the
  hard-coded exclude tuple is unchanged), so checkpoint parameter
  shapes match and the constructed agent equals the training-time
  agent.

## Validation (run at fix time, before any relabel pass)

Against the RETAINED true 18-Jul pilot config
(`local_results/d1_oracle_labels_20260718_103952/runroot_light/
d1pilot_cup_e1_seed1/config.yaml`): exactly 8 keys backfill —
`agent.model_obs`, `env.synth.{coupling,distractor_scale,distractors,
length,radius,use_seed}`, `orthreward.task` — all inert for a
dmc_proprio labeling run (synth/orthreward configs are unused unless
those envs/wrappers are engaged). Schema-current configs are a strict
no-op (verified on the defaults tree itself); saved-value precedence
verified. The shared loader is also used by the d0 sweep path;
the no-op property means current-schema behavior is unchanged
everywhere.

## Ordering

This amendment + the loader fix are committed BEFORE any d1pilot
label pass; the Amendment-1 smoke is then RE-RUN from the top
(`d1pilot_relabel_local.sh smoke`) before `labels`. All other
Amendment-1 terms unchanged. Any further instrument fix ⇒ a further
dated amendment, as before.
