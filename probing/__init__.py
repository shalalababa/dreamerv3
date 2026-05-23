"""Probing pipeline: what sequential generative latents encode.

This package compares two frozen representations of the DMC proprio
Walker-Walk observation stream:

  * the sequential generative latent of DreamerV3's RSSM, and
  * a static frame-level VAE trained on the same observations.

Modules
-------
replay_dataset : read DreamerV3 replay-buffer chunks (VAE training data).
vae            : the StaticVAE model and its (de)serialization helpers.
train_vae      : train the StaticVAE on the replay buffer.
collect        : roll out a frozen DreamerV3 policy and log ground-truth
                 MuJoCo physics state into held-out probe trajectories.
features       : run both frozen models over the probe trajectories and
                 dump features at every probe site.
targets        : derive probe targets (current/future state, gait phase,
                 time-to-fall) from the logged physics state.
probe          : the probing battery (ridge + MLP probes, R^2, likelihood).
recon_over_time: RSSM vs VAE reconstruction error over time (when sequential
                 context helps); reads features.npz, no model needed.
openloop       : open-loop posterior/prior prediction vs true state -- the
                 "how the world model works" figure.

See probing/README.md for the end-to-end run instructions.
"""
