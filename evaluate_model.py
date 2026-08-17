"""
evaluate_model.py  --  Run a trained MAPPO Actor with DETERMINISTIC (argmax)
action selection -- no exploration noise -- and report how many agents
successfully evacuated.

This is a standalone execution/evaluation script, separate from
training.py: it does no learning, just loads mappo_actor.pt and runs it.
Replicates the same environment settings the model was trained under
(CLUSTER_START=True, N_TRAIN_AGENTS, MAX_STEPS_EP -- see training.py).

Usage:
    python evaluate_model.py
"""

import pickle
import numpy as np
import torch
import torch.nn as nn

from common import (SEED, N_TRAIN_AGENTS, MAX_STEPS_EP, GRAPH_PATH,
                     build_grid_graph, make_grid_evac_data,
                     compute_shelter_distances, build_graph_index)
from evac_env import EvacuationEnv, STATUS_SAFE, STATUS_FAILED

# ── Config ────────────────────────────────────────────────────────────────────
# Set USE_GRID_MAP=True when ACTOR_PATH is a grid-trained model
# (mappo_grid_actor.pt) -- the environment's graph must match whatever the
# model was trained on, or load_state_dict will fail with a shape mismatch
# (real map: max_degree=5, obs_dim=61; 5x5 8-connected grid: max_degree=8,
# obs_dim=97).
USE_GRID_MAP = True
ACTOR_PATH   = 'mappo_grid_actor.pt'   # 'mappo_actor.pt' for the real-map model
HIDDEN_SIZE  = 64                       # must match training.py's Actor architecture

# ── Grid config (only used if USE_GRID_MAP=True; must match the training.py
# run that produced ACTOR_PATH) ─────────────────────────────────────────────────
GRID_ROWS               = 5
GRID_COLS                = 5
GRID_CELL_SIZE_M         = 80.0
GRID_CONNECT_DIAGONALS   = True
GRID_N_SHELTERS          = 2
GRID_SEED                = 42
GRID_N_AGENTS            = 3000
GRID_MAX_STEPS           = 200
GRID_CLUSTER_RADIUS_HOPS = 2

# Reproduces the environment the saved model was actually trained under.
# Change these to match whatever training.py config produced ACTOR_PATH.
# (Real-map settings; the grid branch below uses the GRID_* block instead.)
CLUSTER_START        = True
CLUSTER_RADIUS_HOPS  = 6
CLUSTER_CENTER_POOL  = None
N_AGENTS             = N_TRAIN_AGENTS
MAX_STEPS            = MAX_STEPS_EP

# Artificial congestion (permanent density floor on random edges). Must match
# whatever training.py config produced ACTOR_PATH, or you're evaluating under
# a different environment than the model was trained on. See EvacuationEnv's
# docstring / training.py's ARTIFICIAL_CONGESTION block for details.
ARTIFICIAL_CONGESTION           = True
ARTIFICIAL_CONGESTION_LEVELS    = (0.5, 0.8)
ARTIFICIAL_CONGESTION_FRACTION  = 0.8
ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH = True
ARTIFICIAL_CONGESTION_SP_FRACTION          = 1.0

N_EPISODES   = 10     # how many episodes to run and average over
BASE_SEED    = SEED   # episode i uses seed BASE_SEED + i

# True  = argmax (deterministic, no exploration noise) -- "what has the
#         policy actually converged to"
# False = stochastic sampling (multinomial over the softmax probabilities),
#         same action-selection as during training rollouts. Useful to
#         compare directly against argmax: if stochastic sampling succeeds
#         much more often than argmax, the policy's single most-likely
#         action per state is often wrong even though the full distribution
#         still puts meaningful probability on good actions (mode collapse
#         toward a bad peak, rather than a uniformly bad policy).
DETERMINISTIC = False


class Actor(nn.Module):
    """Must match training.py's Actor exactly, or load_state_dict will fail."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim), nn.Softmax(dim=-1))
    def forward(self, x): return self.net(x)


def run_episode(env, actor, device, seed, deterministic=True):
    """Run one episode. deterministic=True uses argmax (no exploration
    noise); deterministic=False samples from the softmax distribution,
    same as during training rollouts. Returns env.summary()."""
    obs_mat, active_mask, infos = env.reset(seed=seed)
    actions_arr = np.zeros(env.n_agents, dtype=np.int32)
    step = 0
    while env.agents and step < env.max_steps:
        active_idxs = np.where(active_mask)[0]
        if len(active_idxs) > 0:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_mat[active_idxs]).to(device)
                probs = actor(obs_t)
                if deterministic:
                    actions_t = torch.argmax(probs, dim=1)
                else:
                    actions_t = torch.multinomial(probs, 1).squeeze(1)
            actions_arr[active_idxs] = actions_t.cpu().numpy().astype(np.int32)
        obs_mat, active_mask, rewards, terminations, truncations, infos = env.step(actions_arr)
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

    device = torch.device('cpu')
    obs_dim_probe = env._obs_dim
    actor = Actor(obs_dim_probe, env.max_degree).to(device)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    actor.eval()

    mode_label = 'argmax, no exploration noise' if DETERMINISTIC else 'stochastic sampling (same as training)'
    print("=" * 60)
    print(f" Evaluating {ACTOR_PATH} ({mode_label})")
    print(f" {'GRID map' if USE_GRID_MAP else 'Real (Kochi) map'}  "
          f"N_AGENTS={n_agents_eff}  MAX_STEPS={max_steps_eff}  "
          f"CLUSTER_START={CLUSTER_START}  episodes={N_EPISODES}")
    print("=" * 60)

    all_arrived, all_agents = 0, 0
    arrival_times = []
    for ep in range(N_EPISODES):
        summary = run_episode(env, actor, device, seed=BASE_SEED + ep,
                               deterministic=DETERMINISTIC)
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
