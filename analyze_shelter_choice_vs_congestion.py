"""
analyze_shelter_choice_vs_congestion.py  --  Does the trained line-map
Actor actually react to real-time congestion when deciding near vs. far?

For every decision (an agent standing at 'center', about to pick an edge),
this script records:
  - density_near, density_far: the congestion features for both candidate
    edges, exactly as the Actor observes them at that instant (obs_mat's
    edge-block density feature -- see evac_env.py's _fill_edge_block).
    Since every agent's target is still 'shelter_near' at decision time
    (nearest_shelter_target=True assigns everyone 'near' initially, and
    re-targeting only happens AFTER committing to an edge -- see
    EvacuationEnv's nearest_shelter_target docstring), the neighbor order
    at 'center' is consistently [near edge, far edge], so obs_mat[:, 3] is
    always density_near and obs_mat[:, 10] is always density_far.
  - p_far: the Actor's raw softmax probability of choosing the far edge
    (action index 1), BEFORE any sampling/argmax noise -- this is the
    cleanest signal of "what did the policy actually think", independent
    of which action got sampled that particular time.

Produces a scatter plot of p_far vs. (density_near - density_far), plus
the Pearson correlation. A congestion-aware policy should show p_far
rising as density_near - density_far increases (near more congested than
far -> more likely to divert to far). A flat/uncorrelated scatter would
mean the policy isn't actually conditioning its choice on congestion at
all (e.g. just outputting a fixed ~50/50 regardless of state).

Requires torch. Config mirrors training.py's USE_LINE_MAP block -- keep
LINE_DIST_NEAR/FAR etc. in sync with whatever run produced ACTOR_PATH.

Usage:
    python analyze_shelter_choice_vs_congestion.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common import (SEED, build_line_graph, make_line_evac_data,
                     compute_shelter_distances, build_graph_index)
from evac_env import EvacuationEnv

# ── Config (mirrors training.py's USE_LINE_MAP block) ──────────────────────────
ACTOR_PATH   = 'mappo_line_seed44_rollout4_16000ep_actor.pt'
HIDDEN_SIZE  = 64

LINE_DIST_NEAR         = 150.0
LINE_DIST_FAR          = 300.0
N_AGENTS                = 3000
MAX_STEPS                = 900
STAGGERED_DEPARTURE      = True
DEPARTURE_WINDOW_FRAC    = 0.25
NEAREST_SHELTER_TARGET   = True

N_EPISODES = 10   # more episodes = more decision points = smoother scatter/correlation
BASE_SEED  = SEED

OUTPUT_PNG = 'shelter_choice_vs_congestion_rollout4_16000ep_baseline.png'


class Actor(nn.Module):
    """Must match training.py's Actor exactly, or load_state_dict fails."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim), nn.Softmax(dim=-1))
    def forward(self, x): return self.net(x)


def build_env():
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
        n_agents=N_AGENTS, max_steps=MAX_STEPS, reward_dest=1.0,
        cluster_start=True, cluster_radius_hops=0, cluster_center_pool=['center'],
        staggered_departure=STAGGERED_DEPARTURE,
        departure_window_frac=DEPARTURE_WINDOW_FRAC,
        nearest_shelter_target=NEAREST_SHELTER_TARGET,
    )


def main():
    env = build_env()
    device = torch.device('cpu')
    actor = Actor(env._obs_dim, env.max_degree).to(device)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    actor.eval()

    all_density_near = []
    all_density_far  = []
    all_p_far        = []

    for ep in range(N_EPISODES):
        obs_mat, active_mask, infos = env.reset(seed=BASE_SEED + ep)
        actions_arr = np.zeros(env.n_agents, dtype=np.int32)
        step = 0
        while env.agents and step < env.max_steps:
            active_idxs = np.where(active_mask)[0]
            if len(active_idxs) > 0:
                obs_active = obs_mat[active_idxs]
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_active).to(device)
                    probs = actor(obs_t)   # (n_active, 2) -- [:,0]=near, [:,1]=far
                    p_far_t = probs[:, 1]
                    actions_t = torch.multinomial(probs, 1).squeeze(1)
                actions_arr[active_idxs] = actions_t.cpu().numpy().astype(np.int32)

                # density_near = feature idx 3 (own-node edge block, near = index 0's edge)
                # density_far  = feature idx 7+3=10 (own-node edge block, far = index 1's edge)
                # See _fill_edge_block: [area, own_speed, n_agents, density, ...] per edge,
                # 7 features per edge, edges ordered [near, far] since target==near at
                # decision time (see module docstring above).
                all_density_near.extend(obs_active[:, 3].tolist())
                all_density_far.extend(obs_active[:, 10].tolist())
                all_p_far.extend(p_far_t.cpu().numpy().tolist())

            obs_mat, active_mask, rewards, terminations, truncations, infos = env.step(actions_arr)
            step += 1
            if not env.agents or any(truncations.values()):
                break
        print(f'Episode {ep}: {len(all_p_far)} cumulative decisions logged so far')

    density_near = np.array(all_density_near)
    density_far  = np.array(all_density_far)
    p_far        = np.array(all_p_far)
    density_diff = density_near - density_far   # >0 means near is more congested

    corr = np.corrcoef(density_diff, p_far)[0, 1] if len(p_far) > 5 else float('nan')
    print(f'\nTotal decisions logged: {len(p_far)}')
    print(f'Pearson correlation (density_near - density_far) vs p_far: {corr:.3f}')
    print('(positive = policy raises P(far) when near gets more congested than far -- '
          'i.e. congestion-aware behavior. Near zero = policy ignores congestion state.)')

    # ── Does the STATE itself carry enough variation to condition on? ──────────
    # Checks whether density_near/density_far/density_diff actually vary
    # meaningfully across decisions, independent of whether the policy is
    # using that variation. If these are nearly constant at decision time
    # (e.g. because congestion hasn't built up yet when agents leave 'center'),
    # no amount of reward shaping can make the policy condition on them --
    # there'd be nothing informative to condition on.
    def _stats(name, arr):
        print(f'  {name:12s}: mean={arr.mean():.4f}  std={arr.std():.4f}  '
              f'min={arr.min():.4f}  p25={np.percentile(arr,25):.4f}  '
              f'median={np.median(arr):.4f}  p75={np.percentile(arr,75):.4f}  '
              f'max={arr.max():.4f}')
    print('\nState variation at decision time (raw obs_mat density features, in [0,1]):')
    _stats('density_near', density_near)
    _stats('density_far',  density_far)
    _stats('density_diff', density_diff)
    print('(if std is tiny relative to the range these features could take (0-1), '
          'the state barely varies across decisions -- reward shaping alone would '
          'have little to condition on.)')

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(density_diff, p_far, s=8, alpha=0.15, color='steelblue')
    # Binned average line to make the trend visible through the scatter noise
    if len(density_diff) > 20:
        bins = np.linspace(density_diff.min(), density_diff.max(), 25)
        bin_idx = np.digitize(density_diff, bins)
        bin_x, bin_y = [], []
        for b in range(1, len(bins)):
            mask = bin_idx == b
            if mask.sum() >= 5:
                bin_x.append(density_diff[mask].mean())
                bin_y.append(p_far[mask].mean())
        if bin_x:
            ax.plot(bin_x, bin_y, 'o-', color='firebrick', lw=2, ms=5,
                    label='Binned average P(far)')
    ax.axhline(0.5, color='gray', lw=1, ls=':', label='50/50')
    ax.axvline(0.0, color='gray', lw=1, ls=':')
    ax.set_xlabel('density_near - density_far  (>0 = near more congested)')
    ax.set_ylabel('P(choose far), from Actor softmax')
    ax.set_title(f'Does the Actor react to relative congestion?\n'
                 f'Pearson r = {corr:.3f}  (n={len(p_far)} decisions, {N_EPISODES} episodes)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_PNG, dpi=150); plt.close()
    print(f'Saved: {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
