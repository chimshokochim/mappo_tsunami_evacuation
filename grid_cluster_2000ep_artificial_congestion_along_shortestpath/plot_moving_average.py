"""
plot_moving_average.py  --  Re-plot a training run's moving-average curves
from the raw per-episode history saved by training.py, without touching
training.py or re-running training.

training.py now saves '{model_prefix}_training_history.pkl' (model_prefix is
'mappo' for the real map, 'mappo_grid' for the synthetic grid) at the end of
train(), containing the raw per-episode lists for both the trained policy
and the shortest-path baseline. This script loads that file and draws ONLY
the moving-average lines (no raw per-episode scatter -- that's what makes
long runs, e.g. 2000 episodes, unreadable in the original arrival_time.png /
mappo_reward_analysis.png).

Usage:
    python plot_moving_average.py                       # uses HISTORY_PATH below
    python plot_moving_average.py mappo_grid_training_history.pkl
    python plot_moving_average.py mappo_training_history.pkl --window 100
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

HISTORY_PATH = 'mappo_training_history.pkl'   # default if no CLI arg given
WINDOW       = None   # None = auto (total_episodes // 20, min 5); or set an int


def moving_avg(values, window):
    """NaN-safe moving average. Returns (x, y) arrays aligned so y[i] is the
    average of values[i:i+window] (i.e. plotted at the window's right edge,
    matching training.py's convention)."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) < window:
        return np.array([]), np.array([])
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.array([]), np.array([])
    filled = np.where(nan_mask, np.nanmean(arr), arr)
    ma = np.convolve(filled, np.ones(window) / window, mode='valid')
    x = np.arange(window, len(arr) + 1)
    return x, ma


def main():
    args = sys.argv[1:]
    path = HISTORY_PATH
    window = WINDOW
    if args:
        path = args[0]
    if '--window' in args:
        window = int(args[args.index('--window') + 1])

    with open(path, 'rb') as f:
        h = pickle.load(f)

    arrival_arrived_only = h['arrival_arrived_only']
    arrival_with_timeout = h['arrival_with_timeout']
    arrival_counts       = h['arrival_counts']
    history_rewards      = h['history_rewards']
    baseline_arrived_only = h.get('baseline_arrived_only')
    baseline_with_timeout = h.get('baseline_with_timeout')
    baseline_counts        = h.get('baseline_counts')
    baseline_rewards       = h.get('baseline_rewards')
    total_episodes         = h.get('total_episodes', len(arrival_arrived_only))

    if window is None:
        window = max(5, total_episodes // 20)
    print(f"Loaded {path}: {total_episodes} episodes, using {window}-episode moving average.")

    # ── Arrival-time moving averages ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))

    x, ma = moving_avg(arrival_arrived_only, window)
    ax.plot(x, ma, '-', color='navy', lw=2, label=f'Trained policy: {window}-ep MA (arrived only)')
    x, ma = moving_avg(arrival_with_timeout, window)
    ax.plot(x, ma, '-', color='firebrick', lw=2, label=f'Trained policy: {window}-ep MA (with timeout)')

    baseline_last_txt = ""
    if baseline_arrived_only is not None:
        x, ma = moving_avg(baseline_arrived_only, window)
        ax.plot(x, ma, '-', color='darkgreen', lw=2, label=f'Baseline: {window}-ep MA (arrived only)')
        x, ma = moving_avg(baseline_with_timeout, window)
        ax.plot(x, ma, '-', color='indigo', lw=2, label=f'Baseline: {window}-ep MA (with timeout)')
        if baseline_counts:
            n_arr_b, n_tot_b = baseline_counts[-1]
            baseline_last_txt = (f" | Baseline arrival rate: {n_arr_b}/{n_tot_b} "
                                  f"({100 * n_arr_b / max(n_tot_b, 1):.1f}%)")

    n_arr_last, n_tot_last = arrival_counts[-1] if arrival_counts else (0, 0)
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg time to reach destination (s)')
    ax.set_title(f'Training Curve - MAPPO ({window}-ep moving average only)\n'
                 f'Trained policy arrival rate: {n_arr_last}/{n_tot_last} '
                 f'({100 * n_arr_last / max(n_tot_last, 1):.1f}%){baseline_last_txt}',
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('arrival_time_ma.png', dpi=150); plt.close()
    print("  Saved: arrival_time_ma.png")

    # ── Mean episode return moving average ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    x, ma = moving_avg(history_rewards, window)
    ax.plot(x, ma, '-', color='steelblue', lw=2, label=f'Trained policy: {window}-ep MA')
    if baseline_rewards is not None:
        x, ma = moving_avg(baseline_rewards, window)
        ax.plot(x, ma, '-', color='seagreen', lw=2, label=f'Baseline: {window}-ep MA')
    ax.set_xlabel('Episode'); ax.set_ylabel('Mean Episode Return')
    ax.set_title(f'Mean Episode Return ({window}-ep moving average only)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('mappo_reward_analysis_ma.png', dpi=150); plt.close()
    print("  Saved: mappo_reward_analysis_ma.png")


if __name__ == '__main__':
    main()
