"""
plot_actor_loss_decomposition.py  --  Decomposes the logged actor_loss into
its two additive components and plots both against episode, alongside the
total actor_loss, to show how much of the loss is driven by the entropy
bonus vs. the actual PPO policy-improvement signal.

Background: training.py's ppo_update() computes, per PPO epoch,
    loss_actor = -min(surr1, surr2).mean() - entropy_coef * dist_entropy.mean()
where entropy_coef now VARIES over training (current_entropy_coef(ep), see
training.py's entropy-annealing block) rather than being the fixed
ENTROPY_COEF constant. training.py logs the per-episode mean of loss_actor
(history_actor_loss), of dist_entropy (history_entropy), AND of the actual
entropy_coef used that episode (history_entropy_coef) -- so all three terms
can be recovered exactly, episode by episode, using each episode's OWN
coefficient rather than assuming a single fixed value:

    entropy_term   =  entropy_coef[ep] * entropy[ep]        (>= 0)
    surrogate_term =  actor_loss[ep] + entropy_term[ep]      (= -min(surr1,surr2).mean())

so that actor_loss == surrogate_term - entropy_term, matching the training
code exactly (no approximation), even though entropy_coef itself is not
constant across the run.

This reveals, e.g., that once entropy saturates near its ceiling (ln(2) for
a 2-action policy) mid-training, the surrogate_term shrinks toward ~0,
meaning the entropy bonus comes to dominate the total actor_loss almost
entirely, while the actual advantage-driven policy-improvement signal
becomes negligible.

Does NOT touch evac_env.py's reward computation. No torch required -- only
reads the saved training_history pickle. Run locally.

Usage:
    python plot_actor_loss_decomposition.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

PKL           = 'mappo_line_seed44_rollout4_lr1e4_entanneal_abs9000_noorthoinit_training_history.pkl'
MA_WINDOW     = 800      # moving-average window, in episodes (entropy/actor_loss
                          # are only logged on update episodes -- see note below)
OUTPUT_PNG    = 'actor_loss_decomposition_entanneal_abs9000.png'


def moving_avg(x, y, window):
    """x, y: 1-D arrays of equal length (already NaN-free / pre-filtered).
    Returns (x_ma, y_ma) using a simple trailing-window box filter over the
    given (possibly non-contiguous-episode) sequence."""
    kernel = np.ones(window) / window
    y_ma = np.convolve(y, kernel, mode='valid')
    x_ma = x[window - 1:]
    return x_ma, y_ma


def main():
    with open(PKL, 'rb') as f:
        d = pickle.load(f)

    actor_loss = np.array(d['history_actor_loss'], dtype=float)
    entropy    = np.array(d['history_entropy'], dtype=float)
    if 'history_entropy_coef' not in d:
        raise KeyError(
            "This pickle has no 'history_entropy_coef' field -- it predates "
            "entropy annealing (or was run before that field was added to "
            "training.py's save dict). Use the OLD version of this script "
            "with a fixed ENTROPY_COEF for pickles like that instead.")
    entropy_coef = np.array(d['history_entropy_coef'], dtype=float)
    episodes     = np.arange(1, len(actor_loss) + 1)

    # actor_loss / entropy / entropy_coef are only populated on episodes
    # where a PPO update actually ran (every ROLLOUT_EPISODES episodes);
    # other entries are NaN. Restrict everything to those logged episodes
    # before decomposing.
    valid = ~np.isnan(actor_loss) & ~np.isnan(entropy) & ~np.isnan(entropy_coef)
    ep_v           = episodes[valid]
    actor_loss_v   = actor_loss[valid]
    entropy_v      = entropy[valid]
    entropy_coef_v = entropy_coef[valid]

    entropy_term   = entropy_coef_v * entropy_v   # varies episode-by-episode now
    surrogate_term = actor_loss_v + entropy_term   # = -min(surr1, surr2).mean()

    # Sanity check: surrogate_term - entropy_term must reconstruct actor_loss
    # exactly (up to float error), confirming the decomposition is exact.
    recon_err = np.max(np.abs((surrogate_term - entropy_term) - actor_loss_v))
    print(f'Reconstruction check: max |recon - actor_loss| = {recon_err:.2e} (should be ~0)')

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Raw (faint) + moving average (bold) for each of the three series.
    ax.plot(ep_v, actor_loss_v, '.', color='steelblue', ms=2, alpha=0.15)
    ax.plot(ep_v, entropy_term, '.', color='seagreen', ms=2, alpha=0.15)
    ax.plot(ep_v, surrogate_term, '.', color='darkorange', ms=2, alpha=0.15)

    x_ma, al_ma = moving_avg(ep_v, actor_loss_v, MA_WINDOW)
    _, et_ma    = moving_avg(ep_v, entropy_term, MA_WINDOW)
    _, st_ma    = moving_avg(ep_v, surrogate_term, MA_WINDOW)

    ax.plot(x_ma, al_ma, '-', color='steelblue', lw=2.2,
            label=f'actor_loss  (= surrogate − entropy, {MA_WINDOW}-update moving avg)')
    ax.plot(x_ma, et_ma, '-', color='seagreen', lw=2.2,
            label='entropy term  ( entropy_coef[ep] x H[ep], coef annealing over training )')
    ax.plot(x_ma, st_ma, '-', color='darkorange', lw=2.2,
            label=f'surrogate term  ( -min(surr1,surr2).mean(), the PPO policy-improvement signal )')

    ax.axhline(0, color='gray', lw=1, ls=':')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss contribution')
    ax.set_title('Actor loss decomposition: entropy term vs. PPO surrogate term')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()
    print(f'Saved: {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
