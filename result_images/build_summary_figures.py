"""Build the summary result figures from the per-seed probing outputs.

Reads results/probing_walker_walk_seed{0,1,2}/ and writes the summary PNGs in
this folder. Figures carry only factual titles / axis labels; interpretations
live in result_images/RESULTS.md. Run from the repository root:

    python result_images/build_summary_figures.py
"""

import csv
import json
import pathlib
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
RES = REPO / 'results'
OUT = REPO / 'result_images'
SEEDS = [0, 1, 2]
SEED_STEPS = {0: '1.1M', 1: '500K', 2: '500K'}
RSSM_C, VAE_C = '#1f77b4', '#d1495b'


def load_probes():
  tabs = {}
  for s in SEEDS:
    rows = list(csv.DictReader(
        open(RES / f'probing_walker_walk_seed{s}' / 'probe_results.csv')))
    tabs[s] = {(r['model'], r['site'], r['target'], r['horizon'],
                r['receptive_field']): r for r in rows}
  return tabs


def r2(tabs, model, site, target, horizon, rf):
  vals = []
  for s in SEEDS:
    r = tabs[s].get((model, site, target, str(horizon), str(rf)))
    vals.append(float(r['r2_ridge']) if r and r['r2_ridge'] not in
                ('', 'None', None) else np.nan)
  return np.array(vals)


def bars_with_seeds(ax, labels, series, colors, ylabel, title):
  x = np.arange(len(labels))
  means = [np.nanmean(v) for v in series]
  ax.bar(x, means, color=colors, width=0.6, alpha=0.85, zorder=2)
  for i, v in enumerate(series):
    ax.scatter([x[i]] * len(v), v, color='black', s=18, zorder=3)
  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
  ax.set_ylabel(ylabel)
  ax.set_title(title, fontsize=11)
  ax.axhline(0, color='#888888', lw=0.7)


def fig_current_state(tabs):
  sites = [('encoder', 'rssm', 'enc', RSSM_C),
           ('RSSM posterior', 'rssm', 'posterior', RSSM_C),
           ('  deter only', 'rssm', 'post_deter', RSSM_C),
           ('  stoch only', 'rssm', 'post_stoch', RSSM_C),
           ('VAE encoder', 'vae', 'vae_enc', VAE_C),
           ('VAE latent', 'vae', 'vae_latent', VAE_C)]
  series = [r2(tabs, m, s, 'state', 0, 1) for _, m, s, _ in sites]
  fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
  bars_with_seeds(ax, [s[0] for s in sites], series, [s[3] for s in sites],
                  'held-out R^2', 'Current-state probe R^2 by site (h=0)')
  ax.set_ylim(0, 1)
  fig.savefig(OUT / 'current_state_probe_r2_by_site.png', dpi=160)
  plt.close(fig)


def fig_future_horizon(tabs):
  hs = [0, 1, 5, 20]
  post = np.array([r2(tabs, 'rssm', 'posterior', 'state', h, 1) for h in hs])
  imag_h = [1, 5, 20]
  imag = np.array([r2(tabs, 'rssm', f'imag{h}', 'state', h, 1)
                   for h in imag_h])
  fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
  for arr, xs, c, lab in [(post, hs, RSSM_C, 'posterior linear readout'),
                          (imag, imag_h, '#2a9d8f',
                           'open-loop prior (imagined, action-conditioned)')]:
    m = np.nanmean(arr, 1)
    ax.fill_between(xs, np.nanmin(arr, 1), np.nanmax(arr, 1), color=c,
                    alpha=0.15)
    ax.plot(xs, m, 'o-', color=c, lw=2, label=lab)
  ax.set_xlabel('prediction horizon k (steps)')
  ax.set_ylabel('held-out R^2  (target: simulator state)')
  ax.set_title('Future-state R^2 vs horizon: posterior readout vs prior')
  ax.axhline(0, color='#888888', lw=0.7)
  ax.legend(fontsize=8)
  fig.savefig(OUT / 'future_state_r2_by_horizon_posterior_vs_prior.png',
              dpi=160)
  plt.close(fig)


def fig_receptive_field(tabs):
  ks = ['1', '4', '16', 'all']
  vae = np.array([r2(tabs, 'vae', 'vae_latent', 'state', 0, k) for k in ks])
  post = r2(tabs, 'rssm', 'posterior', 'state', 0, 1)
  x = np.arange(len(ks))
  fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
  ax.fill_between(x, np.nanmin(vae, 1), np.nanmax(vae, 1), color=VAE_C,
                  alpha=0.15)
  ax.plot(x, np.nanmean(vae, 1), 'o-', color=VAE_C, lw=2, label='VAE latent')
  ax.axhspan(np.nanmin(post), np.nanmax(post), color=RSSM_C, alpha=0.12)
  ax.axhline(np.nanmean(post), color=RSSM_C, ls='--', lw=1.8,
             label='RSSM posterior (h=0)')
  ax.set_xticks(x)
  ax.set_xticklabels([f'K={k}' for k in ks])
  ax.set_xlabel('VAE latent receptive field (past frames)')
  ax.set_ylabel('held-out R^2  (current state)')
  ax.set_title('Static-VAE current-state R^2 vs receptive field')
  ax.legend(fontsize=9)
  fig.savefig(OUT / 'vae_receptive_field_r2.png', dpi=160)
  plt.close(fig)


def fig_gait_phase(tabs):
  series = [r2(tabs, 'rssm', 'posterior', 'gait_phase', 0, 1),
            r2(tabs, 'vae', 'vae_latent', 'gait_phase', 0, 1)]
  fig, ax = plt.subplots(figsize=(5.5, 4.6), constrained_layout=True)
  bars_with_seeds(ax, ['RSSM posterior', 'VAE latent'], series,
                  [RSSM_C, VAE_C], 'held-out R^2',
                  'Gait-phase probe R^2 (h=0)')
  ax.set_ylim(0, 1)
  fig.savefig(OUT / 'gait_phase_r2_rssm_vs_vae.png', dpi=160)
  plt.close(fig)


def fig_generative():
  nll = {s: json.load(open(RES / f'probing_walker_walk_seed{s}' /
                           'likelihood.json')) for s in SEEDS}
  rot = {s: json.load(open(RES / f'probing_walker_walk_seed{s}' /
                           'recon_over_time.json')) for s in SEEDS}
  quants = ['rssm_posterior_recon', 'rssm_prior_predictive', 'vae_recon']
  qc = ['#1f77b4', '#2a9d8f', '#d1495b']
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                 constrained_layout=True)
  x = np.arange(len(SEEDS))
  w = 0.26
  for j, q in enumerate(quants):
    ax1.bar(x + (j - 1) * w, [nll[s][q]['mean'] for s in SEEDS], w,
            color=qc[j], label=q.replace('_', ' '))
  ax1.set_xticks(x)
  ax1.set_xticklabels([f'seed{s}\n({SEED_STEPS[s]})' for s in SEEDS])
  ax1.set_ylabel('held-out NLL (lower = better)')
  ax1.set_title('Held-out reconstruction / prediction NLL')
  ax1.legend(fontsize=8)
  ax2.bar(x, [rot[s]['frac_steps_rssm_better'] for s in SEEDS], 0.6,
          color=RSSM_C, alpha=0.85)
  ax2.set_xticks(x)
  ax2.set_xticklabels([f'seed{s}\n({SEED_STEPS[s]})' for s in SEEDS])
  ax2.set_ylim(0, 1)
  ax2.set_ylabel('fraction of frames')
  ax2.set_title('Frames where RSSM reconstructs better than VAE')
  fig.savefig(OUT / 'generative_nll_and_reconstruction_advantage.png', dpi=160)
  plt.close(fig)


def fig_openloop_rmse():
  fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
  for s in SEEDS:
    o = json.load(open(RES / f'probing_walker_walk_seed{s}' / 'openloop.json'))
    rm = o['rmse_vs_horizon']
    ax.plot(np.arange(1, len(rm) + 1), rm, '-', lw=1.8,
            label=f'seed{s} ({SEED_STEPS[s]})')
  ax.set_xlabel('open-loop horizon (steps after context)')
  ax.set_ylabel('prediction RMSE (observation units)')
  ax.set_title('Open-loop prior prediction RMSE vs horizon')
  ax.legend(fontsize=9)
  fig.savefig(OUT / 'openloop_prediction_rmse_vs_horizon.png', dpi=160)
  plt.close(fig)


def copy_qualitative():
  src = RES / 'probing_walker_walk_seed0' / 'openloop_ep0.png'
  if src.exists():
    shutil.copy(src, OUT / 'openloop_example_trajectory_seed0.png')


def main():
  OUT.mkdir(exist_ok=True)
  tabs = load_probes()
  fig_current_state(tabs)
  fig_future_horizon(tabs)
  fig_receptive_field(tabs)
  fig_gait_phase(tabs)
  fig_generative()
  fig_openloop_rmse()
  copy_qualitative()
  print('Wrote summary figures to', OUT)


if __name__ == '__main__':
  main()
