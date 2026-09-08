"""
plot_mean_return_ma_only.py  --  Plots ONLY the 800-ep moving average of
Mean Episode Return (no raw per-episode points), matching the style/color
(tomato dashed line) used in training.py's own plots.

Usage:
    python plot_mean_return_ma_only.py
(reads mappo_line_seed44_rollout4_lowlr_training_history.pkl from the
current directory -- edit PKL below if you want a different run's file.)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

PKL    = 'mappo_line_seed44_rollout4_lr1e4_entanneal_abs9000_noorthoinit_training_history.pkl'
WINDOW = 400
OUT    = 'mean_return_ma_only.png'


def moving_avg(vals, window):
    arr = np.array([v if v is not None else np.nan for v in vals], dtype=float)
    kernel = np.ones(window) / window
    ma = np.convolve(np.nan_to_num(arr, nan=np.nanmean(arr)), kernel, mode='valid')
    x = np.arange(window, len(arr) + 1)
    return x, ma


def main():
    with open(PKL, 'rb') as f:
        d = pickle.load(f)

    rewards = d['history_rewards']

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    x, ma = moving_avg(rewards, WINDOW)
    ax.plot(x, ma, '--', color='tomato', lw=2, label=f'Trained policy: {WINDOW}-ep moving avg')
    ax.set_xlabel('Episode'); ax.set_ylabel('Mean Episode Return')
    ax.set_title('Mean Episode Return per Episode averaged over all agents\n800-ep moving average only')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()