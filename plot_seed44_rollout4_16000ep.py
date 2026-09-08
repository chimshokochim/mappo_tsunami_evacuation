"""
plot_seed44_rollout4_16000ep.py  --  Re-plots seed44's ROLLOUT_EPISODES=4,
16000-episode training run (mappo_line_seed44_rollout4_16000ep_training_history.pkl)
with raw per-episode data points AND the 800-ep moving average, using the
same color scheme as training.py's own plots (steelblue/darkorange raw,
navy/firebrick/tomato dashed moving averages).

The y-axis on both panels is clamped to the range the moving-average curves
alone would occupy (computed via a first, throwaway pass), so the periodic
raw-data spikes (routing collapse -> huge arrival-time/negative-return
excursions) get clipped out of view instead of stretching the axis and
squashing the moving-average trend into an unreadable band.

No torch required -- only reads the saved history pickle. Run locally, from
the project folder (needs mappo_line_seed44_rollout4_16000ep_training_history.pkl
in the same directory).

Usage:
    python plot_seed44_rollout4_16000ep.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

PKL    = 'mappo_line_seed44_rollout4_16000ep_training_history.pkl'
WINDOW = 800   # moving-average window, in episodes
OUT    = 'seed44_rollout4_16000ep_full.png'


def moving_avg(vals, window):
    arr = np.array([v if v is not None else np.nan for v in vals], dtype=float)
    kernel = np.ones(window) / window
    ma = np.convolve(np.nan_to_num(arr, nan=np.nanmean(arr)), kernel, mode='valid')
    x = np.arange(window, len(arr) + 1)
    return x, ma


def main():
    with open(PKL, 'rb') as f:
        d = pickle.load(f)

    arrived_only = d['arrival_arrived_only']
    with_timeout = d['arrival_with_timeout']
    rewards      = d['history_rewards']

    # ── First pass (throwaway): compute the y-limits the moving-average-ONLY
    # curves would occupy, so the final plot can be clamped to that range
    # instead of auto-expanding to the raw spikes' extent. ─────────────────
    fig0, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), dpi=150)
    x, ma = moving_avg(arrived_only, WINDOW); a1.plot(x, ma, '--', color='navy')
    x, ma = moving_avg(with_timeout, WINDOW); a1.plot(x, ma, '--', color='firebrick')
    x, ma = moving_avg(rewards, WINDOW);      a2.plot(x, ma, '--', color='tomato')
    ylim1 = a1.get_ylim()
    ylim2 = a2.get_ylim()
    plt.close(fig0)

    # ── Actual plot: raw data points (same color scheme as training.py) +
    # the same moving averages, y-axis clamped to the moving-average-only
    # range computed above. ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), dpi=150)

    eps_x = np.arange(1, len(arrived_only) + 1)
    ax1.plot(eps_x, arrived_only, 'o-', color='steelblue', lw=1.5, ms=3,
              label='Trained policy: avg arrival time (arrived only)')
    ax1.plot(eps_x, with_timeout, 'o-', color='darkorange', lw=1.5, ms=3,
              label='Trained policy: avg arrival time (with timeout)')
    x, ma = moving_avg(arrived_only, WINDOW)
    ax1.plot(x, ma, '--', color='navy', lw=2, label=f'{WINDOW}-ep moving avg (arrived only)')
    x, ma = moving_avg(with_timeout, WINDOW)
    ax1.plot(x, ma, '--', color='firebrick', lw=2, label=f'{WINDOW}-ep moving avg (with timeout)')
    ax1.set_ylabel('Avg time to reach destination (s)')
    ax1.set_title('Training Curve - MAPPO (seed44, ROLLOUT_EPISODES=4, 16000 episodes)')
    ax1.set_ylim(ylim1)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(eps_x, rewards, 'o-', color='steelblue', lw=2, ms=3, label='Trained policy: mean episode return')
    x, ma = moving_avg(rewards, WINDOW)
    ax2.plot(x, ma, '--', color='tomato', lw=2, label=f'Trained policy: {WINDOW}-ep moving avg')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Mean Episode Return')
    ax2.set_title('Mean Episode Return per Episode averaged over all agents')
    ax2.set_ylim(ylim2)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f'Saved: {OUT}  (ylim1={ylim1}, ylim2={ylim2})')


if __name__ == '__main__':
    main()