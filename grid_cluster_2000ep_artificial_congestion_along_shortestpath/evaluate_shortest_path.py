"""
evaluate_shortest_path.py  --  Run the greedy shortest-path (no-learning)
baseline -- every agent always picks action 0, i.e. the neighbor closest to
its own assigned destination shelter -- and report how many agents
successfully evacuated.

This is the shortest-path counterpart to evaluate_model.py: same config
knobs (USE_GRID_MAP, cluster start, artificial congestion, N_EPISODES), same
report format, but no Actor/model is loaded -- it's a direct comparison
point for whatever ACTOR_PATH you evaluate with evaluate_model.py. Keep the
env-construction settings (CLUSTER_START, ARTIFICIAL_CONGESTION*, N_AGENTS,
MAX_STEPS, etc.) identical between the two scripts for an apples-to-apples
comparison.

Usage:
    python evaluate_shortest_path.py
"""

import pickle
import numpy as np

from common import (SEED, N_TRAIN_AGENTS, MAX_STEPS_EP, GRAPH_PATH,
                     build_grid_graph, make_grid_evac_data,
                     compute_shelter_distances, build_graph_index)
from evac_env import EvacuationEnv, STATUS_SAFE, STATUS_FAILED

# ── Config ────────────────────────────────────────────────────────────────────
USE_GRID_MAP = True

# ── Grid config (only used if USE_GRID_MAP=True; keep in sync with whatever
# you're comparing against in evaluate_model.py) ────────────────────────────────
GRID_ROWS               = 5
GRID_COLS                = 5
GRID_CELL_SIZE_M         = 80.0
GRID_CONNECT_DIAGONALS   = True
GRID_N_SHELTERS          = 2
GRID_SEED                = 42
GRID_N_AGENTS            = 3000
GRID_MAX_STEPS           = 200
GRID_CLUSTER_RADIUS_HOPS = 2

# Real-map settings; the grid branch above uses the GRID_* block instead.
CLUSTER_START        = False
CLUSTER_RADIUS_HOPS  = 6
CLUSTER_CENTER_POOL  = None
N_AGENTS             = N_TRAIN_AGENTS
MAX_STEPS            = MAX_STEPS_EP

# Artificial congestion (permanent density floor on random edges). Keep this
# in sync with whatever evaluate_model.py run you're comparing against. See
# EvacuationEnv's docstring / training.py's ARTIFICIAL_CONGESTION block.
ARTIFICIAL_CONGESTION           = False
ARTIFICIAL_CONGESTION_LEVELS    = (0.5, 0.8)
ARTIFICIAL_CONGESTION_FRACTION  = 0.15
ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH = False
ARTIFICIAL_CONGESTION_SP_FRACTION          = 1.0

N_EPISODES   = 10     # how many episodes to run and average over
BASE_SEED    = SEED   # episode i uses seed BASE_SEED + i, same convention as
                       # training.py's evaluate_shortest_path_baseline() and
                       # evaluate_model.py, so episodes line up 1:1


def run_episode(env, seed):
    """Run one episode with every agent always taking action 0 (greedy
    shortest path toward its own destination). Returns env.summary()."""
    obs_mat, active_mask, infos = env.reset(seed=seed)
    zero_actions = np.zeros(env.n_agents, dtype=np.int32)
    step = 0
    while env.agents and step < env.max_steps:
        obs_mat, active_mask, rewards, terminations, truncations, infos = env.step(zero_actions)
        step += 1
        if not env.agents or any(truncations.values()):
            break
    return env.summary()


def main():
    if USE_GRID_MAP:
        node_coords, adj, road_nodes = build_grid_graph(
            rows=GRID_ROWS, cols=GRID_COLS, cell_size_m=GRID_CELL_SIZE_M,
            connect_diagonals=GRID_CONNECT_DIAGONALS)
        evac_nodes, evac_capacity = make_grid_evac_data(
            road_nodes, n_shelters=GRID_N_SHELTERS, seed=GRID_SEED)
        shelter_dist, _ = compute_shelter_distances(evac_nodes, adj, road_nodes)
        node_list, node_to_idx, neighbor_lists, max_degree = build_graph_index(
            adj, road_nodes, shelter_dist=shelter_dist)
        n_agents_eff        = GRID_N_AGENTS
        max_steps_eff        = GRID_MAX_STEPS
        cluster_radius_eff   = GRID_CLUSTER_RADIUS_HOPS
        env = EvacuationEnv(
            node_coords=node_coords, adj=adj, road_nodes=road_nodes,
            evac_nodes=evac_nodes, evac_capacity=evac_capacity,
            node_list=node_list, node_to_idx=node_to_idx,
            neighbor_lists=neighbor_lists, max_degree=max_degree,
            n_agents=n_agents_eff, max_steps=max_steps_eff, reward_dest=1.0,
            cluster_start=CLUSTER_START, cluster_radius_hops=cluster_radius_eff,
            artificial_congestion=ARTIFICIAL_CONGESTION,
            artificial_congestion_levels=ARTIFICIAL_CONGESTION_LEVELS,
            artificial_congestion_fraction=ARTIFICIAL_CONGESTION_FRACTION,
            artificial_congestion_target_shortest_path=ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH,
            artificial_congestion_sp_fraction=ARTIFICIAL_CONGESTION_SP_FRACTION,
        )
    else:
        with open(GRAPH_PATH, 'rb') as f:
            g = pickle.load(f)
        n_agents_eff = N_AGENTS
        max_steps_eff = MAX_STEPS
        env = EvacuationEnv(
            node_coords=g['node_coords'], adj=g['adj'], road_nodes=set(g['node_list']),
            evac_nodes=set(g['evac_nodes']), evac_capacity=g['evac_capacity'],
            node_list=g['node_list'], node_to_idx=g['node_to_idx'],
            neighbor_lists=g['neighbor_lists'], max_degree=g['max_degree'],
            n_agents=n_agents_eff, max_steps=max_steps_eff, reward_dest=1.0,
            cluster_start=CLUSTER_START, cluster_radius_hops=CLUSTER_RADIUS_HOPS,
            cluster_center_pool=CLUSTER_CENTER_POOL,
            artificial_congestion=ARTIFICIAL_CONGESTION,
            artificial_congestion_levels=ARTIFICIAL_CONGESTION_LEVELS,
            artificial_congestion_fraction=ARTIFICIAL_CONGESTION_FRACTION,
            artificial_congestion_target_shortest_path=ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH,
            artificial_congestion_sp_fraction=ARTIFICIAL_CONGESTION_SP_FRACTION,
        )

    print("=" * 60)
    print(" Evaluating shortest-path baseline (action 0 always, no learning)")
    print(f" {'GRID map' if USE_GRID_MAP else 'Real (Kochi) map'}  "
          f"N_AGENTS={n_agents_eff}  MAX_STEPS={max_steps_eff}  "
          f"CLUSTER_START={CLUSTER_START}  ARTIFICIAL_CONGESTION={ARTIFICIAL_CONGESTION}"
          f"{' (shortest-path targeted)' if ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH else ''}"
          f"  episodes={N_EPISODES}")
    print("=" * 60)

    all_arrived, all_agents = 0, 0
    arrival_times = []
    for ep in range(N_EPISODES):
        summary = run_episode(env, seed=BASE_SEED + ep)
        n_arr, n_tot = summary['n_arrived'], summary['n_agents']
        all_arrived += n_arr
        all_agents  += n_tot
        if not np.isnan(summary['avg_arrival_time_arrived_only']):
            arrival_times.append(summary['avg_arrival_time_arrived_only'])
        print(f"Episode {ep:3d} | Arrived: {n_arr:4d}/{n_tot} "
              f"({100 * n_arr / n_tot:5.1f}%) | "
              f"Avg arrival time (arrived only): "
              f"{summary['avg_arrival_time_arrived_only']:.1f}s")

    print("-" * 60)
    overall_rate = 100 * all_arrived / all_agents if all_agents else 0.0
    print(f"TOTAL: {all_arrived}/{all_agents} agents evacuated successfully "
          f"({overall_rate:.1f}%) across {N_EPISODES} episodes")
    if arrival_times:
        print(f"Average arrival time (arrived agents, averaged over episodes): "
              f"{np.mean(arrival_times):.1f}s")


if __name__ == '__main__':
    main()
