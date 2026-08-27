"""
replot_training_history.py  --  Regenerate ALL of training.py's PNG outputs
(arrival_time.png, mappo_reward_analysis.png, shelter_choice.png,
ppo_losses.png, etc. -- whatever plot_training_results() produces) from an
already-saved *_training_history.pkl, without re-running training.

Use this after editing plot titles/labels/etc. inside plot_training_results()
in training.py -- the pickle already has every series that function needs
(arrival times, rewards, far_frac, losses, entropy, baseline data), so the
plots can just be rebuilt directly from disk.

No torch required -- training.py's plot_training_results() only uses
numpy/matplotlib, not the Actor/Critic, so importing training.py here is
safe even without a trained model loaded.

Usage:
    python replot_training_history.py [history.pkl]
    (history.pkl defaults to mappo_line_training_history.pkl)
"""

import sys
import pickle

from training import plot_training_results

HISTORY_PATH = sys.argv[1] if len(sys.argv) > 1 else 'mappo_line_training_history.pkl'


def main():
    with open(HISTORY_PATH, 'rb') as f:
        d = pickle.load(f)

    plot_training_results(
        arrival_arrived_only=d['arrival_arrived_only'],
        arrival_with_timeout=d['arrival_with_timeout'],
        arrival_counts=d['arrival_counts'],
        mean_episode_rewards=d['history_rewards'],
        baseline_arrived_only=d.get('baseline_arrived_only'),
        baseline_with_timeout=d.get('baseline_with_timeout'),
        baseline_counts=d.get('baseline_counts'),
        baseline_rewards=d.get('baseline_rewards'),
        history_far_frac=d.get('history_far_frac'),
        baseline_far_frac=d.get('baseline_far_frac'),
        history_critic_loss=d.get('history_critic_loss'),
        history_actor_loss=d.get('history_actor_loss'),
        history_entropy=d.get('history_entropy'),
    )
    print(f'Re-plotted all figures from: {HISTORY_PATH}')


if __name__ == '__main__':
    main()
