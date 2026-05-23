"""Where does sequential context help? RSSM vs VAE reconstruction over time.

This is the *temporal* view of the static-vs-sequential gap. Both models
reconstruct the same current observation at every step, in the same units
(summed symlog reconstruction error over the proprio dims): the RSSM from its
recurrent posterior, the static VAE from a single frame. Plotting the two
error curves within an episode shows *when* recurrence helps -- the
hypothesis being that it helps most during fast transients, where a single
frame is ambiguous but temporal context disambiguates.

It uses only arrays already produced by `features.py` (`wm_post_nll`,
`vae_recon_nll`) plus the logged physics state for a motion-intensity signal
(the joint+body speed ||qvel||), so it needs no model or checkpoint.

Run from the repository root:

    python -m probing.recon_over_time \
        --features <RUN>/probing_walker_walk_seed0/features.npz \
        --traj     <RUN>/probing_walker_walk_seed0/traj.npz \
        --output   results/probing_walker_walk_seed0
"""

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def moving_average(x, w):
  if w <= 1 or len(x) <= w:
    return x
  k = np.ones(w) / w
  return np.convolve(x, k, mode='same')


def transient_signal(traj):
  """Per-step motion intensity = ||qvel|| from the logged physics state."""
  phys = np.asarray(traj['phys_state'], np.float32)        # (N, T, qpos|qvel)
  qvel = phys[..., phys.shape[-1] // 2:]
  return np.linalg.norm(qvel, axis=-1)                     # (N, T)


def plot_episode(post, vae, prior, motion, ep, smooth, outpath):
  T = post.shape[1]
  t = np.arange(T)
  fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]},
                           constrained_layout=True)
  ax = axes[0]
  ax.plot(t, moving_average(vae[ep], smooth), color='#d1495b', lw=1.8,
          label='static VAE')
  ax.plot(t, moving_average(post[ep], smooth), color='#1f77b4', lw=1.8,
          label='RSSM posterior')
  if prior is not None:
    ax.plot(t, moving_average(prior[ep], smooth), color='#6a994e', lw=1.2,
            ls='--', alpha=0.8, label='RSSM 1-step prior')
  ax.set_ylabel('reconstruction NLL\n(symlog MSE, summed)')
  ax.set_title(f'When does sequential context help? (episode {ep})')
  ax.legend(fontsize=8)
  ax2 = axes[1]
  ax2.fill_between(t, 0, moving_average(motion[ep], smooth), color='#999999',
                   alpha=0.5)
  ax2.set_ylabel('motion\n||qvel||')
  ax2.set_xlabel('step within episode')
  fig.savefig(outpath, dpi=150)
  plt.close(fig)


def plot_gap_vs_motion(gap, motion, r, outpath, nbins=12):
  fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
  n = len(gap)
  idx = np.random.default_rng(0).choice(n, min(n, 4000), replace=False)
  ax.scatter(motion[idx], gap[idx], s=5, alpha=0.12, color='#1f77b4')
  edges = np.quantile(motion, np.linspace(0, 1, nbins + 1))
  edges[-1] += 1e-6
  which = np.digitize(motion, edges) - 1
  centers, means = [], []
  for b in range(nbins):
    m = which == b
    if m.sum() >= 10:
      centers.append(motion[m].mean())
      means.append(gap[m].mean())
  ax.plot(centers, means, 'o-', color='#d1495b', lw=2, label='binned mean')
  ax.axhline(0, color='#999999', lw=0.8)
  ax.set_xlabel('motion intensity  ||qvel||')
  ax.set_ylabel('RSSM advantage\n(VAE NLL  -  RSSM NLL)')
  ax.set_title(f'Sequential context helps most during fast motion '
               f'(Pearson r = {r:+.3f})')
  ax.legend(fontsize=9)
  fig.savefig(outpath, dpi=150)
  plt.close(fig)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--features', required=True)
  p.add_argument('--traj', required=True)
  p.add_argument('--output', required=True, help='Output directory.')
  p.add_argument('--episode', type=int, default=0)
  p.add_argument('--smooth', type=int, default=11)
  args = p.parse_args()
  os.makedirs(args.output, exist_ok=True)

  feats = {k: np.asarray(v) for k, v in np.load(args.features).items()}
  traj = {k: np.asarray(v) for k, v in np.load(args.traj).items()}
  if 'vae_recon_nll' not in feats:
    raise SystemExit('features.npz has no vae_recon_nll; re-run features.py '
                     'with --vae_ckpt.')
  post = feats['wm_post_nll']
  vae = feats['vae_recon_nll']
  prior = feats.get('wm_prior_nll')
  motion = transient_signal(traj)
  ep = min(args.episode, post.shape[0] - 1)

  plot_episode(post, vae, prior, motion, ep, args.smooth,
               os.path.join(args.output, f'recon_over_time_ep{ep}.png'))

  gap = (vae - post).reshape(-1)
  mflat = motion.reshape(-1)
  r = float(np.corrcoef(mflat, gap)[0, 1])
  plot_gap_vs_motion(gap, mflat, r,
                     os.path.join(args.output, 'recon_gap_vs_motion.png'))

  summary = {
      'mean_nll': {'rssm_posterior': float(post.mean()),
                   'vae_recon': float(vae.mean()),
                   'gap_vae_minus_rssm': float(gap.mean())},
      'gap_motion_pearson_r': r,
      'frac_steps_rssm_better': float((gap > 0).mean()),
  }
  with open(os.path.join(args.output, 'recon_over_time.json'), 'w') as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2))
  print(f'Wrote figures -> {args.output}')


if __name__ == '__main__':
  main()
