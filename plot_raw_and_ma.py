"""
plot_raw_and_ma.py -- Plots raw per-episode reward and arrival time, with a
moving average overlaid on top, for a single training_history.pkl.

Usage:
    python plot_raw_and_ma.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

PKL        = 'mappo_line_seed44_rollout4_lr1e4_entanneal_abs9000_noorthoinit_training_history.pkl'
MA_WINDOW  = 50    # episodes
OUTPUT_PNG = 'entanneal_12000ep_raw_and_ma.png'


def moving_avg(x, window):
    x = np.array(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def moving_avg_nan(x, window):
    """NaN-aware moving average (arrival time can be NaN if nobody arrived
    that episode)."""
    x = np.array(x, dtype=float)
    out = np.full(len(x) - window + 1, np.nan)
    for i in range(len(out)):
        seg = x[i:i + window]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 0:
            out[i] = seg.mean()
    return out


def main():
    with open(PKL, 'rb') as f:
        d = pickle.load(f)

    r  = np.array(d['history_rewards'], dtype=float)
    at = np.array(d['arrival_arrived_only'], dtype=float)
    ep = np.arange(1, len(r) + 1)

    r_ma  = moving_avg(r, MA_WINDOW)
    at_ma = moving_avg_nan(at, MA_WINDOW)
    ep_ma = ep[MA_WINDOW - 1:]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=140, sharex=True)

    ax = axes[0]
    ax.plot(ep, r, '.', ms=2, alpha=0.3, color='steelblue', label='raw (per-episode)')
    ax.plot(ep_ma, r_ma, '-', lw=1.8, color='navy', label=f'{MA_WINDOW}-ep moving avg')
    ax.set_ylabel('Reward (per-agent)')
    ax.set_title(PKL)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(ep, at, '.', ms=2, alpha=0.3, color='darkorange', label='raw (per-episode)')
    ax2.plot(ep_ma, at_ma, '-', lw=1.8, color='saddlebrown', label=f'{MA_WINDOW}-ep moving avg')
    ax2.set_ylabel('Arrival time, arrived-only (s)')
    ax2.set_xlabel('Episode')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    print(f'Saved: {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
