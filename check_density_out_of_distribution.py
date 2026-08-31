"""
check_density_out_of_distribution.py  --  Were the near-edge density values
observed at N=4000/5000 (in test_congestion_dependent_rule.py) actually
outside the range the policy ever saw during training (N=3000 only)?

test_congestion_dependent_rule.py only reported the MEAN density_near per
demand level. This script instead collects the FULL distribution (every
individual density_near value observed at every decision point, across all
episodes) for N=3000 -- a proxy for "what densities the policy was actually
exposed to, since training always used N=3000" -- and compares it against
the full distributions at N=4000 and N=5000, reporting:
  - N=3000's observed range (min, percentiles, max)
  - what fraction of N=4000/5000's decision-time density values exceed
    N=3000's observed max (i.e. are in territory literally never seen)
  - what fraction exceed N=3000's 95th percentile (i.e. rare-but-seen vs.
    truly novel)

This directly checks the hypothesis: "the policy over-diverts at N=4000/5000
because those density levels are out-of-distribution for it."

Requires torch (uses the trained Actor). Run locally.

Usage:
    python check_density_out_of_distribution.py
"""

import numpy as np
import torch
import torch.nn as nn

from common import (SEED, build_line_graph, make_line_evac_data,
                     compute_shelter_distances, build_graph_index)
from evac_env import EvacuationEnv

ACTOR_PATH   = 'mappo_line_actor.pt'
HIDDEN_SIZE  = 64

LINE_DIST_NEAR         = 150.0
LINE_DIST_FAR          = 300.0
MAX_STEPS                = 900
STAGGERED_DEPARTURE      = True
DEPARTURE_WINDOW_FRAC    = 0.25
NEAREST_SHELTER_TARGET   = True

REFERENCE_DEMAND = 3000   # the ONLY demand level ever used during training
CHECK_DEMANDS    = [1000, 2000, 4000, 5000]
N_EPISODES_PER_LEVEL = 5
BASE_SEED = SEED


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim), nn.Softmax(dim=-1))
    def forward(self, x): return self.net(x)


def build_env(n_agents):
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
        n_agents=n_agents, max_steps=MAX_STEPS, reward_dest=1.0,
        cluster_start=True, cluster_radius_hops=0, cluster_center_pool=['center'],
        staggered_departure=STAGGERED_DEPARTURE,
        departure_window_frac=DEPARTURE_WINDOW_FRAC,
        nearest_shelter_target=NEAREST_SHELTER_TARGET,
    )


def collect_density_samples(env, actor, device, n_episodes):
    """Returns a flat array of every density_near value the Actor observed
    at every 'at center' decision point, across n_episodes episodes."""
    cidx = env.node_to_idx['center']
    all_samples = []
    for ep in range(n_episodes):
        obs_mat, active_mask, infos = env.reset(seed=BASE_SEED + ep)
        actions_arr = np.zeros(env.n_agents, dtype=np.int32)
        step = 0
        while env.agents and step < env.max_steps:
            active_idxs = np.where(active_mask)[0]
            if len(active_idxs) > 0:
                at_center = env._agent_node_idx[active_idxs] == cidx
                obs_active = obs_mat[active_idxs]
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_active).to(device)
                    probs = actor(obs_t)
                    actions_t = torch.multinomial(probs, 1).squeeze(1)
                chosen = actions_t.cpu().numpy().astype(np.int32)
                actions_arr[active_idxs] = chosen
                all_samples.extend(obs_active[at_center, 3].tolist())
            obs_mat, active_mask, rewards, terminations, truncations, infos = env.step(actions_arr)
            step += 1
            if not env.agents or any(truncations.values()):
                break
    return np.array(all_samples)


def describe(label, samples):
    print(f'{label}: n={len(samples)}, min={samples.min():.3f}, '
          f'p50={np.percentile(samples,50):.3f}, p90={np.percentile(samples,90):.3f}, '
          f'p95={np.percentile(samples,95):.3f}, max={samples.max():.3f}')


def main():
    env0 = build_env(REFERENCE_DEMAND)
    device = torch.device('cpu')
    actor = Actor(env0._obs_dim, env0.max_degree).to(device)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    actor.eval()

    print(f'Reference (training) demand level: N={REFERENCE_DEMAND}\n')
    ref_env = build_env(REFERENCE_DEMAND)
    ref_samples = collect_density_samples(ref_env, actor, device, N_EPISODES_PER_LEVEL)
    describe(f'N={REFERENCE_DEMAND} (training-time proxy)', ref_samples)
    ref_max = ref_samples.max()
    ref_p95 = np.percentile(ref_samples, 95)

    print(f'\n{"N":>6} | {"min":>7} | {"p50":>7} | {"p90":>7} | {"p95":>7} | {"max":>7} | '
          f'{"%% > train max":>14} | {"%% > train p95":>14}')
    print('-' * 90)
    ref_row = (f'{REFERENCE_DEMAND:>6} | {ref_samples.min():>7.3f} | '
               f'{np.percentile(ref_samples,50):>7.3f} | {np.percentile(ref_samples,90):>7.3f} | '
               f'{ref_p95:>7.3f} | {ref_max:>7.3f} | {"--":>14} | {"--":>14}')
    print(ref_row)

    for demand in CHECK_DEMANDS:
        env = build_env(demand)
        samples = collect_density_samples(env, actor, device, N_EPISODES_PER_LEVEL)
        pct_over_max = 100.0 * np.mean(samples > ref_max)
        pct_over_p95 = 100.0 * np.mean(samples > ref_p95)
        print(f'{demand:>6} | {samples.min():>7.3f} | {np.percentile(samples,50):>7.3f} | '
              f'{np.percentile(samples,90):>7.3f} | {np.percentile(samples,95):>7.3f} | '
              f'{samples.max():>7.3f} | {pct_over_max:>13.1f}% | {pct_over_p95:>13.1f}%')

    print(f'\n"%% > train max" = fraction of decisions at that demand level where the '
          f'observed density_near\nexceeded the highest density EVER seen during the '
          f'N={REFERENCE_DEMAND} reference runs -- i.e.\ngenuinely novel territory for the Actor.')


if __name__ == '__main__':
    main()
