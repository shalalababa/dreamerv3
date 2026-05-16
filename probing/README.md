# Probing what sequential generative latents encode

This package implements the experiment described in the Week-7 progress
report: comparing **two frozen representations** of the DMC proprio
Walker-Walk observation stream under a common probing battery.

| Representation | What it is |
|---|---|
| RSSM latent | the sequential generative latent of DreamerV3's Recurrent State-Space Model (deterministic recurrent state + categorical stochastic latent) |
| Static VAE  | a per-frame diagonal-Gaussian VAE trained on the *same* observation frames, with no temporal structure |

Both models are frozen, then evaluated by probes that vary three axes:

* **probe target** — current simulator state, future state at horizon
  `k ∈ {1,5,20}`, or a derived dynamical quantity (gait phase, return-to-go,
  time-to-fall);
* **probe site** — encoder output, RSSM posterior / VAE latent mean, or the
  RSSM prior (one-step and open-loop `k`-step, which has no input access);
* **receptive field** — the VAE latent probed with `1, 4, 16, all` past
  frames, measuring how much sequential content a static model recovers
  through post-hoc aggregation alone.

See [`METHODS.md`](METHODS.md) for the VAE objective derivation and the
interpretation of probe `R²`.

## Pipeline

All commands are run from the repository root. On the cluster, use the
array-job scripts in `scripts/` (one task per DreamerV3 seed).

```
DreamerV3 run dir ──► (1) train_vae ──────────► vae.ckpt
        │                                          │
        └──► (2) collect ──► traj.npz ──► (3) features ──► features.npz
                                  │              │
                                  └──► (4) probe ◄┘ ──► probes/
```

### 1. Train the static VAE — `train_vae.py`
Trains the VAE on the DreamerV3 replay buffer (the literal "same observation
stream"). One VAE per DreamerV3 seed.

```bash
python -m probing.train_vae \
  --replay_dirs <RUN>/dmc_proprio_walker_walk_seed0 \
  --run_config  <RUN>/dmc_proprio_walker_walk_seed0/config.yaml \
  --logdir      <RUN>/vae_walker_walk_seed0 \
  --latent 128 --beta 1.0 --steps 60000 --seed 0
```

### 2. Collect held-out probe trajectories — `collect.py`
Rolls out the frozen DreamerV3 policy and logs ground-truth MuJoCo physics
state (the replay buffer does not store it).

```bash
python -m probing.collect \
  --run_logdir <RUN>/dmc_proprio_walker_walk_seed0 \
  --output     <RUN>/probing_walker_walk_seed0/traj.npz \
  --episodes 40 --seed 0
```

### 3. Extract features — `features.py`
Runs both frozen models over the probe trajectories and dumps features at
every probe site, plus held-out likelihoods.

```bash
python -m probing.features \
  --traj       <RUN>/probing_walker_walk_seed0/traj.npz \
  --run_logdir <RUN>/dmc_proprio_walker_walk_seed0 \
  --vae_ckpt   <RUN>/vae_walker_walk_seed0/vae.ckpt \
  --output     <RUN>/probing_walker_walk_seed0/features.npz \
  --horizons 1 5 20
```

### 4. Run the probing battery — `probe.py`
Fits ridge + MLP probes over the full (target × site × horizon ×
receptive-field) sweep and writes `probe_results.csv`, `probe_summary.txt`,
`likelihood.json`, and figures.

```bash
python -m probing.probe \
  --traj     <RUN>/probing_walker_walk_seed0/traj.npz \
  --features <RUN>/probing_walker_walk_seed0/features.npz \
  --output   <RUN>/probing_walker_walk_seed0/probes
```

## Cluster

```bash
sbatch scripts/vae_train.sbatch        # array 0-2: trains all three VAEs
sbatch scripts/probe_pipeline.sbatch   # array 0-2: collect + features + probe
```
Run `vae_train` first; `probe_pipeline` needs both checkpoints.

## Design notes / fairness controls

* The VAE reuses DreamerV3's exact encoder (`dreamerv3.rssm.Encoder`) and a
  decoder identical to the RSSM's vector-decoder path, with the same symlog
  observation model. The only difference from the RSSM is the **absence of
  recurrence/dynamics** and the Gaussian (rather than categorical) latent.
* The VAE trains on the RSSM's replay buffer, so both models see the same
  input distribution; the held-out probe set is disjoint fresh rollouts.
* The VAE latent defaults to 128 (= RSSM `stoch × classes = 32 × 4`). The
  RSSM posterior additionally carries a 512-d deterministic state, so the
  probe sweep also reports `enc`, `deter`-only and `stoch`-only style sites
  and controls for probe capacity with the ridge λ search.
