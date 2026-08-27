"""
plot_line_zoom.py  --  Zoomed-in view of the TRAINED POLICY's own metrics
from mappo_line_training_history.pkl, without the shortest-path baseline
sharing (and dominating) the same y-axis. training.py's arrival_time.png /
mappo_reward_analysis.png put both series on one axis, so once the
baseline gridlocks (avg ~4000s+) the trained policy's much smaller
200-300s range gets visually flattened into what looks like a straight
line, even if it has real variation.

Also plots history_far_frac (fraction of arrivals at shelter_far) directly
against arrival time on a twin axis, episode by episode, so any
correlation between "far_frac drifting" and "arrival time changing" is
visible directly instead of having to eyeball two separate charts.

No torch required -- this only reads the saved history pickle.

Usage:
    python plot_line_zoom.py [history.pkl] [max_episode]
    (history.pkl defaults to mappo_line_training_history.pkl;
     max_episode caps the x-axis, e.g. `python plot_line_zoom.py "" 1250`
     -- pass "" for history.pkl to keep its default while still setting
     max_episode. Omit both for auto x-axis, i.e. however many episodes
     are actually in the file.)
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

HISTORY_PATH = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1]
                else 'mappo_line_training_history.pkl')
MAX_EPISODE  = int(sys.argv[2]) if len(sys.argv) > 2 else None   # e.g. 1250; None = auto


def moving_avg(vals, window):
    valid = np.array([v if not (v is None or np.isnan(v)) else np.nan for v in vals], dtype=float)
    if np.sum(~np.isnan(valid)) < window:
        return None, None
    ma = np.convolve(np.nan_to_num(valid, nan=np.nanmean(valid)),
                      np.ones(window) / window, mode='valid')
    return np.arange(window, len(vals) + 1), ma


def main():
    with open(HISTORY_PATH, 'rb') as f:
        d = pickle.load(f)

    arrival_arrived = d['arrival_arrived_only']
    arrival_timeout  = d['arrival_with_timeout']
    rewards          = d['history_rewards']
    far_frac         = d.get('history_far_frac')
    n_eps = len(rewards)
    eps_x = np.arange(1, n_eps + 1)
    window = max(5, n_eps // 20)

    # ── Figure 1: trained-policy-only arrival time, own axis ─────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.plot(eps_x, arrival_arrived, 'o-', color='steelblue', lw=1.2, ms=3, alpha=0.6,
            label='Arrived only')
    ax.plot(eps_x, arrival_timeout, 'o-', color='darkorange', lw=1.2, ms=3, alpha=0.6,
            label='With timeout')
    mx, ma = moving_avg(arrival_arrived, window)
    if mx is not None:
        ax.plot(mx, ma, '--', color='navy', lw=2.5, label=f'{window}-ep moving avg (arrived only)')
    mx2, ma2 = moving_avg(arrival_timeout, window)
    if mx2 is not None:
        ax.plot(mx2, ma2, '--', color='firebrick', lw=2.5, label=f'{window}-ep moving avg (with timeout)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg arrival time (s)')
    ax.set_title('Trained policy ONLY -- avg arrival time (zoomed, no baseline on this axis)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    if MAX_EPISODE is not None:
        ax.set_xlim(0, MAX_EPISODE)
    plt.tight_layout(); plt.savefig('line_trained_arrival_zoom.png', dpi=150); plt.close()
    print('Saved: line_trained_arrival_zoom.png')

    # ── Figure 2: trained-policy-only reward, own axis ────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.plot(eps_x, rewards, 'o-', color='steelblue', lw=1.2, ms=3, alpha=0.6, label='Mean episode return')
    mx3, ma3 = moving_avg(rewards, window)
    if mx3 is not None:
        ax.plot(mx3, ma3, '--', color='tomato', lw=2.5, label=f'{window}-ep moving avg')
    ax.set_xlabel('Episode'); ax.set_ylabel('Mean episode return')
    ax.set_title('Trained policy ONLY -- mean episode return (zoomed, no baseline on this axis)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    if MAX_EPISODE is not None:
        ax.set_xlim(0, MAX_EPISODE)
    plt.tight_layout(); plt.savefig('line_trained_reward_zoom.png', dpi=150); plt.close()
    print('Saved: line_trained_reward_zoom.png')

    # ── Figure 3: far_frac vs. arrival time, twin axes, same episode axis ────
    if far_frac is not None and any(v is not None and not np.isnan(v) for v in far_frac):
        fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=150)
        ax2 = ax1.twinx()
        l1, = ax1.plot(eps_x, far_frac, 'o-', color='steelblue', lw=1.2, ms=3, alpha=0.6,
                        label='Fraction of arrivals at shelter_far')
        mx4, ma4 = moving_avg(far_frac, window)
        l1b = None
        if mx4 is not None:
            l1b, = ax1.plot(mx4, ma4, '--', color='navy', lw=2.5,
                             label=f'{window}-ep moving avg (far_frac)')
        l2, = ax2.plot(eps_x, arrival_arrived, 'o-', color='darkorange', lw=1.2, ms=3, alpha=0.5,
                        label='Avg arrival time (arrived only)')
        mx5, ma5 = moving_avg(arrival_arrived, window)
        l2b = None
        if mx5 is not None:
            l2b, = ax2.plot(mx5, ma5, '--', color='firebrick', lw=2.5,
                             label=f'{window}-ep moving avg (arrival time)')
        ax1.axhline(0.5, color='gray', lw=1, ls=':')
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Fraction of arrivals at shelter_far', color='steelblue')
        ax2.set_ylabel('Avg arrival time, s (arrived only)', color='darkorange')
        ax1.set_title('Far-shelter choice vs. arrival time, same episode axis\n'
                       '(does arrival time move together with far_frac drift?)')
        handles = [h for h in [l1, l1b, l2, l2b] if h is not None]
        ax1.legend(handles=handles, fontsize=8, loc='upper left')
        ax1.grid(True, alpha=0.3)
        if MAX_EPISODE is not None:
            ax1.set_xlim(0, MAX_EPISODE)
        plt.tight_layout(); plt.savefig('line_far_frac_vs_arrival.png', dpi=150); plt.close()
        print('Saved: line_far_frac_vs_arrival.png')

        # Simple correlation check
        valid_mask = ~np.isnan(np.array(far_frac, dtype=float)) & ~np.isnan(np.array(arrival_arrived, dtype=float))
        if np.sum(valid_mask) > 5:
            corr = np.corrcoef(np.array(far_frac)[valid_mask], np.array(arrival_arrived)[valid_mask])[0, 1]
            print(f'Pearson correlation (far_frac vs arrival time, per-episode): {corr:.3f}')

    # ── Figure 4: Critic/Actor loss + entropy, own axes ───────────────────────
    # Same diagnostic as training.py's ppo_losses.png, but re-plottable from
    # the saved history without re-running training, and respects
    # MAX_EPISODE for consistent x-axis comparisons across runs.
    critic_loss = d.get('history_critic_loss')
    actor_loss  = d.get('history_actor_loss')
    entropy     = d.get('history_entropy')
    if critic_loss is not None and any(v is not None and not np.isnan(v) for v in critic_loss):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)

        ax1.plot(eps_x, critic_loss, 'o-', color='steelblue', lw=1, ms=2, alpha=0.5,
                 label='Critic loss (MSE vs. GAE returns)')
        mxc, mac = moving_avg(critic_loss, window)
        if mxc is not None:
            ax1.plot(mxc, mac, '--', color='navy', lw=2, label=f'{window}-ep moving avg')
        ax1.set_ylabel('Critic loss'); ax1.set_yscale('log')
        ax1.set_title('Trained policy ONLY -- Critic loss (log scale, zoomed)')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        if actor_loss is not None:
            ax2.plot(eps_x, actor_loss, 'o-', color='darkorange', lw=1, ms=2, alpha=0.5,
                      label='Actor loss (clipped surrogate)')
            mxa, maa = moving_avg(actor_loss, window)
            if mxa is not None:
                ax2.plot(mxa, maa, '--', color='firebrick', lw=2, label=f'{window}-ep moving avg')
        if entropy is not None:
            ax2b = ax2.twinx()
            ax2b.plot(eps_x, entropy, 'o-', color='seagreen', lw=1, ms=2, alpha=0.4,
                       label='Policy entropy')
            ax2b.set_ylabel('Policy entropy (nats)', color='seagreen')
        ax2.set_xlabel('Episode'); ax2.set_ylabel('Actor loss')
        ax2.set_title('Trained policy ONLY -- Actor loss + policy entropy (zoomed)')
        ax2.legend(fontsize=8, loc='upper left'); ax2.grid(True, alpha=0.3)

        if MAX_EPISODE is not None:
            ax1.set_xlim(0, MAX_EPISODE)
        plt.tight_layout(); plt.savefig('line_ppo_losses_zoom.png', dpi=150); plt.close()
        print('Saved: line_ppo_losses_zoom.png')


if __name__ == '__main__':
    main()
