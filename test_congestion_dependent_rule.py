"""
test_congestion_dependent_rule.py  --  Did the trained MAPPO policy learn a
genuinely congestion-DEPENDENT routing rule, or did it just memorize that
"~40% far" is a good fixed split for this one exact scenario? Per Bhaskar's
feedback: "I would suggest to see whether the policy has actually learned a
congestion-dependent routing rule rather than simply learning that a fixed
~40% diversion is optimal for this particular scenario. For example, vary
the initial demand/departure rate or edge congestion and see how P(far)
changes. If it adjusts systematically, that would be strong evidence."

This is a DIFFERENT test from analyze_shelter_choice_vs_congestion.py, which
already showed P(far) responds to REAL-TIME density fluctuations WITHIN a
single episode (r=0.956, but over a narrow 0.37-0.40 range). That result is
ambiguous on its own: a policy that always outputs ~0.40 regardless of the
actual state, with only tiny incidental noise correlated with density,
could produce a similarly "significant" correlation without truly being
congestion-aware. This script instead varies the SCENARIO itself --
total demand (N_AGENTS) and departure rate (DEPARTURE_WINDOW_FRAC) -- run
by run, using the SAME trained Actor (no re-training), and checks whether
the resulting far_frac tracks the resulting congestion level ACROSS
scenarios. A policy that just memorized "~40%" should show a roughly FLAT
far_frac across very different demand levels. A policy that learned a real
congestion-dependent rule should show far_frac rising as demand (and thus
near-edge congestion) increases.

For each demand level, records both:
  - far_frac: fraction of arrivals at shelter_far (the outcome-level split)
  - avg_density_near: the average near-edge density agents actually
    observed at their decision points during that run (the realized
    congestion level for that scenario -- since higher demand doesn't
    automatically mean proportionally higher observed density if the
    policy successfully routes around it, this is a more direct read of
    "how bad did congestion actually get" than N_AGENTS alone)

Then checks whether far_frac increases with demand and correlates with
avg_density_near across scenarios.

Does NOT touch evac_env.py's reward computation. Requires torch (for the
Actor). Run locally.

Usage:
    python test_congestion_dependent_rule.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common import (SEED, build_line_graph, make_line_evac_data,
                     compute_shelter_distances, build_graph_index)
from evac_env import EvacuationEnv

# ── Config (mirrors training.py's USE_LINE_MAP block) ──────────────────────────
ACTOR_PATH   = 'mappo_line_actor.pt'
HIDDEN_SIZE  = 64

LINE_DIST_NEAR         = 150.0
LINE_DIST_FAR          = 300.0
MAX_STEPS                = 900
STAGGERED_DEPARTURE      = True
DEPARTURE_WINDOW_FRAC    = 0.25
NEAREST_SHELTER_TARGET   = True

# Demand levels to sweep, holding everything else (distances, departure
# rate) fixed at the training config. 3000 is the level the model was
# actually trained on; the others are all out-of-distribution for the
# Actor -- exactly what we want, since a memorized fixed split would only
# look "correct" at 3000 and fail to adjust away from it.
DEMAND_LEVELS = [1000, 2000, 3000, 4000, 5000]
N_EPISODES_PER_LEVEL = 5
BASE_SEED = SEED

OUTPUT_PNG = 'congestion_dependent_rule.png'


class Actor(nn.Module):
    """Must match training.py's Actor exactly, or load_state_dict fails."""
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


def run_episode(env, actor, device, seed):
    """Returns (far_frac, avg_density_near_at_decisions) for one episode."""
    cidx = env.node_to_idx['center']
    obs_mat, active_mask, infos = env.reset(seed=seed)
    actions_arr = np.zeros(env.n_agents, dtype=np.int32)
    density_near_samples = []
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
            # density_near feature: index 3 of the own-node edge block (see
            # analyze_shelter_choice_vs_congestion.py's module docstring for
            # why this index is stable under nearest_shelter_target=True).
            density_near_samples.extend(obs_active[at_center, 3].tolist())

        obs_mat, active_mask, rewards, terminations, truncations, infos = env.step(actions_arr)
        step += 1
        if not env.agents or any(truncations.values()):
            break

    summary = env.summary()
    shelter_list = summary.get('shelter_list')
    counts = summary.get('arrival_shelter_counts')
    far_frac = float('nan')
    if shelter_list and 'shelter_far' in shelter_list and counts and sum(counts) > 0:
        far_frac = counts[shelter_list.index('shelter_far')] / sum(counts)

    avg_density_near = float(np.mean(density_near_samples)) if density_near_samples else float('nan')
    return far_frac, avg_density_near


def main():
    env0 = build_env(DEMAND_LEVELS[0])
    device = torch.device('cpu')
    actor = Actor(env0._obs_dim, env0.max_degree).to(device)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    actor.eval()

    results = {}  # demand -> (far_fracs list, densities list)
    for demand in DEMAND_LEVELS:
        env = build_env(demand)
        far_fracs, densities = [], []
        for ep in range(N_EPISODES_PER_LEVEL):
            ff, dn = run_episode(env, actor, device, BASE_SEED + ep)
            far_fracs.append(ff); densities.append(dn)
        results[demand] = (far_fracs, densities)
        print(f'N_AGENTS={demand:5d} | far_frac: mean={np.nanmean(far_fracs):.3f} '
              f'std={np.nanstd(far_fracs):.3f} | avg_density_near: '
              f'mean={np.nanmean(densities):.3f}')

    demands = np.array(DEMAND_LEVELS)
    far_frac_means = np.array([np.nanmean(results[d][0]) for d in DEMAND_LEVELS])
    far_frac_stds  = np.array([np.nanstd(results[d][0]) for d in DEMAND_LEVELS])
    density_means  = np.array([np.nanmean(results[d][1]) for d in DEMAND_LEVELS])

    # Trend across demand levels: correlate demand with far_frac, and
    # density_near with far_frac (across scenarios, not within an episode).
    valid = ~np.isnan(far_frac_means) & ~np.isnan(density_means)
    corr_demand = np.corrcoef(demands[valid], far_frac_means[valid])[0, 1] if valid.sum() > 2 else float('nan')
    corr_density = np.corrcoef(density_means[valid], far_frac_means[valid])[0, 1] if valid.sum() > 2 else float('nan')

    print(f'\nAcross-scenario correlation (demand level vs. far_frac):        r = {corr_demand:.3f}')
    print(f'Across-scenario correlation (avg density_near vs. far_frac):    r = {corr_density:.3f}')
    print('A near-zero correlation here would suggest the policy is NOT adjusting '
          'to scenario-level congestion (consistent with a memorized fixed split).')
    print('A clearly positive correlation would suggest a real congestion-dependent rule.')

    # ── Sweep-optimal overlay (see sweep_far_frac.py) ────────────────────────
    # If sweep_far_frac_results.csv exists (from running sweep_far_frac.py),
    # overlay the empirically-found true-optimal far_frac per demand level,
    # so the policy's actual output can be compared directly against the
    # real target rather than just the flat "~0.40 memorized split" line.
    sweep_csv = 'sweep_far_frac_results.csv'
    sweep_optimal = {}  # demand -> best far_frac (min with_timeout arrival time)
    if os.path.exists(sweep_csv):
        best_wt = {}
        with open(sweep_csv) as f:
            for row in csv.reader(f):
                if row[0] == 'demand':
                    continue
                d, ff, ao, wt, ar = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                if d not in best_wt or wt < best_wt[d]:
                    best_wt[d] = wt
                    sweep_optimal[d] = ff
        print(f'\nLoaded sweep-optimal far_frac from {sweep_csv}: {sweep_optimal}')
    else:
        print(f'\n({sweep_csv} not found -- run sweep_far_frac.py first to overlay the '
              f'true-optimal far_frac on the plot.)')

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    ax1.errorbar(demands, far_frac_means, yerr=far_frac_stds, fmt='o-',
                 color='steelblue', capsize=4, lw=1.5, ms=6, label='MAPPO (actual)')
    if sweep_optimal:
        sweep_demands = sorted(sweep_optimal)
        sweep_vals = [sweep_optimal[d] for d in sweep_demands]
        ax1.plot(sweep_demands, sweep_vals, '*--', color='firebrick', ms=14,
                  lw=1.5, label='Sweep-optimal (true best split)')
    ax1.set_xlabel('N_AGENTS (total demand)'); ax1.set_ylabel('far_frac')
    ax1.set_title(f'far_frac vs. demand level')
    ax1.set_ylim(0, 1); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    for d in DEMAND_LEVELS:
        ax2.scatter(results[d][1], results[d][0], label=f'N={d}', s=25, alpha=0.8)
    ax2.set_xlabel('avg density_near at decision points')
    ax2.set_ylabel('far_frac')
    ax2.set_title(f'far_frac vs. realized congestion, across scenarios')
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(OUTPUT_PNG, dpi=150); plt.close()
    print(f'\nSaved: {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
