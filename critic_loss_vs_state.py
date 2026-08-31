"""
critic_loss_vs_state.py  --  Does the change in critic-loss behavior partway
through training coincide with a change in routing behavior / state
distribution? (per Bhaskar's feedback: "I would look into the fairly clear
change in the critic-loss behavior ... and check whether it coincides with
a change in the routing probabilities/state distribution.")

Reads the existing mappo_line_training_history.pkl (no re-training needed)
and plots critic_loss and far_frac (the routing-probability proxy we track
per episode -- fraction of arrivals at shelter_far) on the SAME episode
axis, both raw and smoothed, so any co-movement is directly visible.
Also uses arrival time (avg time to reach a shelter) as a second state-
distribution proxy, since a shift in routing should show up there too if
it's a real behavioral change rather than just noisy critic fitting.

Rather than assume the transition is at exactly episode 2000 (that number
came from eyeballing a specific run's plot), this script finds the episode
where the SMOOTHED critic loss changes fastest (biggest jump in its moving
average, i.e. the steepest slope region) and reports it explicitly, along
with far_frac / arrival-time values just before and after that point, so
you can see numerically whether they moved together or independently.

No torch required.

Usage:
    python critic_loss_vs_state.py [history.pkl]
    (history.pkl defaults to mappo_line_training_history.pkl)
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

HISTORY_PATH = sys.argv[1] if len(sys.argv) > 1 else 'mappo_line_training_history.pkl'
OUTPUT_PNG = 'critic_loss_vs_state.png'


def moving_avg(vals, window):
    valid = np.array([v if v is not None and not np.isnan(v) else np.nan for v in vals], dtype=float)
    if np.sum(~np.isnan(valid)) < window:
        return None, None
    filled = np.where(np.isnan(valid), np.nanmean(valid), valid)
    ma = np.convolve(filled, np.ones(window) / window, mode='valid')
    return np.arange(window, len(vals) + 1), ma


def find_loss_turning_point(ma_x, ma_y):
    """Returns the episode where the SMOOTHED critic loss is at its global
    minimum -- i.e. where it stops decreasing and starts rising/oscillating.
    This is the specific transition Bhaskar is asking about ("the fairly
    clear change in critic-loss behavior"), and it's NOT the same thing as
    "the single steepest change in the curve": critic loss drops ~3 orders
    of magnitude during the first few hundred episodes of normal learning
    (untrained critic -> roughly fitted), which is a much BIGGER change,
    in either raw or log terms, than the later ~1-order-of-magnitude
    jump-then-oscillate transition -- but that early drop is just ordinary
    learning, not the qualitative behavior change (smooth decrease ->
    persistent oscillation) visible later in the plot. The minimum point
    is a direct, assumption-free way to find that specific transition."""
    if ma_x is None:
        return None
    i = int(np.argmin(ma_y))
    return int(ma_x[i])


def window_mean(vals, center_ep, half_width):
    vals = np.array(vals, dtype=float)
    lo = max(0, center_ep - half_width - 1)
    hi = min(len(vals), center_ep + half_width)
    seg = vals[lo:hi]
    seg = seg[~np.isnan(seg)]
    return float(np.mean(seg)) if len(seg) else float('nan')


def main():
    with open(HISTORY_PATH, 'rb') as f:
        d = pickle.load(f)

    critic_loss = d.get('history_critic_loss')
    far_frac    = d.get('history_far_frac')
    arrival     = d.get('arrival_arrived_only')
    if critic_loss is None:
        print('No history_critic_loss in this pickle -- nothing to compare.')
        return

    n_eps = len(critic_loss)
    eps_x = np.arange(1, n_eps + 1)
    window = max(5, n_eps // 20)

    cl_x, cl_ma = moving_avg(critic_loss, window)
    ff_x, ff_ma = moving_avg(far_frac, window) if far_frac is not None else (None, None)
    ar_x, ar_ma = moving_avg(arrival, window) if arrival is not None else (None, None)

    change_ep = find_loss_turning_point(cl_x, cl_ma)

    print(f'Loaded: {HISTORY_PATH} ({n_eps} episodes, smoothing window={window})')
    if change_ep is not None:
        half = max(20, window)
        ff_before = window_mean(far_frac, change_ep - half, half) if far_frac is not None else float('nan')
        ff_after  = window_mean(far_frac, change_ep + half, half) if far_frac is not None else float('nan')
        ar_before = window_mean(arrival, change_ep - half, half) if arrival is not None else float('nan')
        ar_after  = window_mean(arrival, change_ep + half, half) if arrival is not None else float('nan')
        print(f'\nSmoothed critic loss reaches its minimum (stops decreasing, starts '
              f'rising/oscillating) around episode {change_ep}.')
        print(f'  far_frac:    {ff_before:.3f} (before)  ->  {ff_after:.3f} (after)   '
              f'(delta = {ff_after - ff_before:+.3f})')
        print(f'  arrival time: {ar_before:.1f}s (before) -> {ar_after:.1f}s (after)   '
              f'(delta = {ar_after - ar_before:+.1f}s)')
        print('  If these deltas are small/noisy, the critic-loss shift likely reflects '
              'internal value-fitting dynamics rather than a real behavior change.')
        print('  If they are large and consistent, the critic-loss shift is probably '
              'tracking a real change in routing behavior / state distribution.')
    else:
        print('Not enough episodes to reliably locate a steepest-change point.')

    # ── Plot: critic loss (log scale) + far_frac + arrival time, shared x ────
    n_panels = 1 + (far_frac is not None) + (arrival is not None)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.2 * n_panels), dpi=150, sharex=True)
    if n_panels == 1:
        axes = [axes]
    ax_i = 0

    ax = axes[ax_i]; ax_i += 1
    ax.plot(eps_x, critic_loss, 'o', color='steelblue', ms=2, alpha=0.35, label='Critic loss (per episode)')
    if cl_x is not None:
        ax.plot(cl_x, cl_ma, '-', color='navy', lw=2, label=f'{window}-ep moving avg')
    ax.set_yscale('log'); ax.set_ylabel('Critic loss')
    ax.set_title('Critic loss vs. state/routing distribution over training')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    if far_frac is not None:
        ax = axes[ax_i]; ax_i += 1
        ax.plot(eps_x, far_frac, 'o', color='seagreen', ms=2, alpha=0.35, label='far_frac (per episode)')
        if ff_x is not None:
            ax.plot(ff_x, ff_ma, '-', color='darkgreen', lw=2, label=f'{window}-ep moving avg')
        ax.set_ylabel('far_frac'); ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    if arrival is not None:
        ax = axes[ax_i]; ax_i += 1
        ax.plot(eps_x, arrival, 'o', color='darkorange', ms=2, alpha=0.35, label='Avg arrival time (per episode)')
        if ar_x is not None:
            ax.plot(ar_x, ar_ma, '-', color='firebrick', lw=2, label=f'{window}-ep moving avg')
        ax.set_ylabel('Arrival time (s)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Episode')
    plt.tight_layout(); plt.savefig(OUTPUT_PNG, dpi=150); plt.close()
    print(f'\nSaved: {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
