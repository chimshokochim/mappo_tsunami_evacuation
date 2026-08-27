"""
baseline_and_replot.py  --  Run the shortest-path (no-learning) baseline for
just a few episodes, average it into a flat reference line, and re-generate
training.py's PNG outputs (arrival_time.png, mappo_reward_analysis.png,
shelter_choice.png, ppo_losses.png) showing BOTH the already-trained MAPPO
history (loaded from mappo_line_training_history.pkl) and this freshly-run
baseline -- without re-running the 4000-episode MAPPO training itself.

Why this exists: the baseline has no learning, so running it for as many
episodes as training (4000) is wasted time for a flat reference line (see
BASELINE_N_EPISODES in training.py). If your existing
mappo_line_training_history.pkl was saved with RUN_BASELINE=False (no
baseline_* fields), this script fills that gap after the fact: it builds
the SAME line-topology env training.py would (mirroring train()'s
USE_LINE_MAP branch), runs evaluate_shortest_path_baseline() for
BASELINE_N_EPISODES episodes, averages the result, broadcasts it across
every training episode (so it renders as a flat line matching the trained
policy's episode axis), and calls training.py's own plot_training_results()
with both series. Also re-saves the pickle with the baseline fields filled
in, so replot_training_history.py can reuse them later without re-running
the baseline again.

Does NOT touch evac_env.py's reward computation -- only reads env state via
the existing action-0/shortest-path baseline path, same as training.py's
own RUN_BASELINE code path.

Requires torch (training.py imports it at module level), but the baseline
itself never touches the Actor/Critic networks.

Usage:
    python baseline_and_replot.py [history.pkl] [n_baseline_episodes]
    (history.pkl defaults to mappo_line_training_history.pkl;
     n_baseline_episodes defaults to training.py's BASELINE_N_EPISODES,
     or 10 if that's unset/None)
"""

import sys
import pickle
import numpy as np

from common import (SEED, build_line_graph, make_line_evac_data,
                     compute_shelter_distances, build_graph_index)
from evac_env import EvacuationEnv
from training import (evaluate_shortest_path_baseline, plot_training_results,
                       LINE_DIST_NEAR, LINE_DIST_FAR, LINE_N_AGENTS, LINE_MAX_STEPS,
                       STAGGERED_DEPARTURE, DEPARTURE_WINDOW_FRAC,
                       LINE_NEAREST_SHELTER_TARGET, LINE_CRITIC_STATE,
                       REWARD_DEST, BASELINE_N_EPISODES)

HISTORY_PATH = sys.argv[1] if len(sys.argv) > 1 else 'mappo_line_training_history.pkl'
N_BASELINE_EPISODES = (int(sys.argv[2]) if len(sys.argv) > 2
                        else (BASELINE_N_EPISODES or 10))


def build_line_env():
    """Mirrors training.py's train() USE_LINE_MAP branch exactly, so the
    baseline sees the identical env the trained model was evaluated
    against (same distances, agent count, staggered departure, etc.)."""
    node_coords, adj, road_nodes = build_line_graph(
        dist_near=LINE_DIST_NEAR, dist_far=LINE_DIST_FAR)
    evac_nodes, evac_capacity = make_line_evac_data()
    shelter_dist, _ = compute_shelter_distances(evac_nodes, adj, road_nodes)
    node_list, node_to_idx, neighbor_lists, max_degree = build_graph_index(
        adj, road_nodes, shelter_dist=shelter_dist)

    return EvacuationEnv(
        node_coords=node_coords, adj=adj, road_nodes=road_nodes,
        evac_nodes=evac_nodes, evac_capacity=evac_capacity,
        node_list=node_list, node_to_idx=node_to_idx,
        neighbor_lists=neighbor_lists, max_degree=max_degree,
        n_agents=LINE_N_AGENTS, max_steps=LINE_MAX_STEPS, reward_dest=REWARD_DEST,
        cluster_start=True, cluster_radius_hops=0, cluster_center_pool=['center'],
        staggered_departure=STAGGERED_DEPARTURE,
        departure_window_frac=DEPARTURE_WINDOW_FRAC,
        nearest_shelter_target=LINE_NEAREST_SHELTER_TARGET,
        line_critic_state=LINE_CRITIC_STATE,
    )


def main():
    with open(HISTORY_PATH, 'rb') as f:
        d = pickle.load(f)
    n_episodes = len(d['history_rewards'])

    print(f'Loaded trained history: {HISTORY_PATH} ({n_episodes} episodes)')
    print(f'Running shortest-path baseline for {N_BASELINE_EPISODES} episodes '
          f'(averaged into a flat line across all {n_episodes} episodes)...')

    env = build_line_env()
    (baseline_arrived_only, baseline_with_timeout, baseline_counts,
     baseline_rewards, baseline_far_frac) = evaluate_shortest_path_baseline(
        env, N_BASELINE_EPISODES, SEED)

    mean_arrived  = float(np.nanmean(baseline_arrived_only))
    mean_timeout  = float(np.nanmean(baseline_with_timeout))
    mean_reward   = float(np.nanmean(baseline_rewards))
    mean_far_frac = float(np.nanmean(baseline_far_frac))
    mean_n_arr    = float(np.mean([c[0] for c in baseline_counts]))
    mean_n_tot    = baseline_counts[0][1]

    d['baseline_arrived_only'] = [mean_arrived] * n_episodes
    d['baseline_with_timeout'] = [mean_timeout] * n_episodes
    d['baseline_rewards']      = [mean_reward] * n_episodes
    d['baseline_far_frac']     = [mean_far_frac] * n_episodes
    d['baseline_counts']       = [(mean_n_arr, mean_n_tot)] * n_episodes

    print(f'Baseline (mean of {N_BASELINE_EPISODES} eps): '
          f'arrival(arrived only)={mean_arrived:.1f}s, '
          f'arrival(with timeout)={mean_timeout:.1f}s, '
          f'reward={mean_reward:.3f}, far_frac={mean_far_frac:.3f}, '
          f'arrived={mean_n_arr:.0f}/{mean_n_tot:.0f}')

    # Persist the filled-in baseline fields so future re-plots (e.g.
    # replot_training_history.py) don't need to re-run the baseline again.
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(d, f)
    print(f'Updated pickle with baseline fields: {HISTORY_PATH}')

    plot_training_results(
        arrival_arrived_only=d['arrival_arrived_only'],
        arrival_with_timeout=d['arrival_with_timeout'],
        arrival_counts=d['arrival_counts'],
        mean_episode_rewards=d['history_rewards'],
        baseline_arrived_only=d['baseline_arrived_only'],
        baseline_with_timeout=d['baseline_with_timeout'],
        baseline_counts=d['baseline_counts'],
        baseline_rewards=d['baseline_rewards'],
        history_far_frac=d.get('history_far_frac'),
        baseline_far_frac=d['baseline_far_frac'],
        history_critic_loss=d.get('history_critic_loss'),
        history_actor_loss=d.get('history_actor_loss'),
        history_entropy=d.get('history_entropy'),
    )
    print('Re-plotted all figures with trained MAPPO history + flat baseline.')


if __name__ == '__main__':
    main()
