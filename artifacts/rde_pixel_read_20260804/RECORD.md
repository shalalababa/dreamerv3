# rde pixel arm — recon-detached trunk: ONE read record (2026-08-04)

Registered ONE read of the rde wave via the frozen reader
`analysis/rde_pixel_read.py`, under `PREREG_rde_pixel_20260802.md`
(freeze commit `87196fe3`, 2026-08-02). Second and complementary lever
test of the pixel swamping account (first: pe, NO-RELIEF 08-02).

## VERDICT: **PARTIAL-RELIEF** (registered middle branch)

Registered verdict text (reader, verbatim): "PARTIAL-RELIEF: rew-NLL
leaves the measured swamping range but not the swamping regime;
directional support only, no headline change; behavioral leg reported
as measured."

- **P-RD1 (primary, representation): in-regime rew-NLL 7.474
  [5.745, 9.554]** (h=0, sides pooled, seed-clustered bootstrap B=10K
  rng 0). CI entirely below the committed baseline's lower bound
  19.409267325402837 ⇒ PARTIAL-RELIEF; NOT below the 2.0 swamping bar
  ⇒ no INCLUSION-RESTORED (floor conjunct never reached; member_band
  false).
- **Latent-alive witness: ALL 16 fits PASS** (threshold 0.10 ×
  calibration median 0.16718 = 0.016718; minimum fit deter_std
  0.02974). No COLLAPSE-QUALIFIED disclosure applies — the moved NLL
  is carried by alive latents, not a collapsed trunk.
- **P-RD2 (behavior): does NOT fire** — paired [rde − X2-task] lift
  −9.577 [−22.505, +3.486], n=8 seeds, 3/8 positive. Not NEGATIVE
  either (CI does not exclude 0 from above).
- **P-RD3: premise-idle** (adjudicated only if P-RD2 fires).

## What this means (registered map)

The FIRST intervention in the whole pixel arc that moved reward-NLL at
all: pe (competition-free FEATURES) moved it by nothing (23.03 ≈
23.34); rde (competition itself removed, trunk trainable) moves it
23.34 → 7.47, ~15.9 nats — descriptively ~74% of the nats-to-bar
(21.34) — while the decoder still trains on sg(latents) and every
head-grad stays on. But the trunk shaped by reward/value/continuation/
dynamics alone still does NOT make reward support legible at pixel
(residual 5.5 nats above the bar), and the representation gain buys NO
behavioral lift off the X2 floor.

- Registered reading: **directional support only, no headline
  change.** The causal competition account is neither retired (the
  NO-RELIEF branch) nor confirmed (the COMPETITION-CONFIRMED branch):
  gradient competition is a real, causal contributor to pixel reward
  illegibility, and removing it is insufficient for inclusion.
- The two lever directions now DISSOCIATE: features-lever dead,
  competition-lever moves representation but not behavior. P-SW1 (the
  measured swamping fact) is NOT re-adjudicated.
- No follow-up compute is queued by this branch (per the prereg, only
  registered-decisive branches existed; any pixel follow-up = NEW
  registration). **Paper-1 registered compute is COMPLETE.**

## Post-read exploratory (labeled; verdict-untouchable)

- Strong side asymmetry, both instruments: deter_std s0 0.0297–0.1811
  vs s1 0.4344–0.4878 (calibration ≈ 0.15–0.18 both sides); final fit
  wm_total s0 ≈ 71 vs s1 ≈ 17–25.
- The P-RD2 deficit concentrates in s1: cell means rde s0 79.9 vs
  baseline s0 77.2 (≈ equal); rde s1 60.7 vs baseline s1 82.6. The
  pooled −9.6 is an s1 phenomenon.

## Provenance (verified before the read)

- Bundle `local_results/rde_pixel_20260804_180214/`; committed
  manifest ≡ bundle manifest; **sha256 sweep 154/154 OK, 0 FAILED**.
- Run head `0a3774cc` (committed ancestor of local HEAD; freeze
  `87196fe3` is an ancestor of it); tree clean except untracked slurm
  logs from other waves. All FIVE frozen instruments byte-identical
  between the bundle's `code/code_sha256.txt` and local committed
  files (rde_pixel_read 864f0556…, axis1.sbatch 7296d073…,
  stratified_error c382a6e1…, agent.py 3cfc838c…, configs.yaml
  1a8434aa…); local prereg sha 70d52a30… = registered final.
- Wave executed on the Vast instance (`/workspace` runroot, host
  7b955ff53681 — the same host as the pe/swamping pixel arc). Smoke
  log shows two failed starts before the gate (missing replay
  manifest; a stale pre-freeze checkout rejecting `AXIS1_ARM=rde`;
  a broken jax env) — all pre-gate environment failures, then the
  passing run: **rc=0 + machine assert "SMOKE config: recon_grad
  False, head grads True" (ARM-path derivation exercised) + SMOKE
  deter_std 0.0229 > 0** — all three registered gate conjuncts.
- 16 fits `OFFLINE_FIT_PROGRESS` at 500000/500000; 16/16 adapts
  ADAPT_DONE; no rde run in the AUC exclusions (only the seed-99
  smoke, by design).
- E4 job log echoes the overridden COLLATE path
  (`e4_fingerpx_v1_rde.csv`, 64 rows / 16 runs) — the committed
  swamping csv was never touched. Calibration csv = complete 2×8
  fpxpx grid via per-run `--output e4_fingerpx_v1cal`.

## Deviation (disclosed)

The prereg registers the config-audit log (`rde_config_audit.log`,
tee'd on the cluster) as integrity evidence shipping with the bundle;
**no such log is in the bundle.** All 32 `config.yaml` files ARE in
the bundle, so the registered fail-closed audit script was executed
verbatim locally at read time on the 16 fit configs: **16/16 PASS, 0
QUARANTINE** (`recon_grad False`, `reward_grad`/`repval_grad` True,
`frozen_enc` False, `model_obs` image, expl task, finger task) —
archived here as `rde_config_audit_local.log`. Value-blind quantities
only (config keys); no decision quantity is touched by the deviation.

## Read execution

```
python -m analysis.rde_pixel_read \
  --auc local_results/rde_pixel_20260804_180214/analysis/auc/auc.csv \
  --e4 local_results/rde_pixel_20260804_180214/e4/e4_fingerpx_v1_rde.csv \
  --calib local_results/rde_pixel_20260804_180214/e4/e4_fingerpx_v1cal_fpxpx.csv \
  --output artifacts/rde_pixel_read_20260804/
```

X2 baseline consumed from the committed
`artifacts/pixel_x2_20260724/auc.csv` (reader-pinned). Outputs:
`rde_pixel.json` (full, incl. per-fit witness values),
`read_stdout.txt`, `rde_config_audit_local.log`. Executed once,
2026-08-04.
