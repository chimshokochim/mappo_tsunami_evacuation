"""
sweep_far_frac.py  --  Empirical "what's the true optimal near/far split"
sweep, repeated across several demand levels (N_AGENTS = 1000..5000), to
get real ground-truth targets to compare the trained MAPPO policy's actual
far_frac against (see test_congestion_dependent_rule.py's results).

WHY A SWEEP INSTEAD OF SOLVING FOR IT: an earlier attempt to compute the
"theoretical optimal split" with a static formula (equal-density-throughout
assumption) gave inconsistent, non-monotonic numbers across demand levels,
because it ignores staggered departure and the resulting time-varying
congestion entirely. A sweep sidesteps that: instead of solving anything in
closed form, it just runs the REAL simulator (staggered departures,
evolving congestion, everything) many times, forcing each agent's near/far
choice to be a fixed coin-flip with probability far_frac (instead of using
a policy), for far_frac = 0.0, 0.1, ..., 1.0. Whichever far_frac produces
the best (lowest) average arrival time IS empirically the optimal split for
that demand level -- no formula, no assumptions about density being
constant, just brute-force measurement using the actual dynamics.

For each demand level in DEMAND_LEVELS, this sweeps all 11 far_frac values,
records avg arrival time (both arrived-only and with-timeout versions) and
arrival rate, and reports the best far_frac. Also prints a final summary
table comparing the sweep-found optimum against MAPPO's actual observed
far_frac from test_congestion_dependent_rule.py (hardcoded below -- update
MAPPO_FAR_FRAC if you re-run that script and get different numbers).

No torch required -- doesn't use the trained Actor at all, only fixed
random near/far coin-flips. Can be run in a torch-free environment.

Usage:
    python sweep_far_frac.py
"""

import numpy as np
import matplotlib.pyplot as plt

from common import (SEED, build_line_graph, make_line_evac_data,
                     compute_shelter_distances, build_graph_index)
from evac_env import EvacuationEnv

LINE_DIST_NEAR         = 150.0
LINE_DIST_FAR          = 300.0
MAX_STEPS                = 900
STAGGERED_DEPARTURE      = True
DEPARTURE_WINDOW_FRAC    = 0.25
NEAREST_SHELTER_TARGET   = True

DEMAND_LEVELS   = [1000, 2000, 3000, 4000, 5000]
FAR_FRAC_GRID   = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
N_EPISODES_PER_POINT = 3
BASE_SEED = SEED

# From test_congestion_dependent_rule.py's most recent run (see chat) --
# update these if you re-run that script and the numbers change.
MAPPO_FAR_FRAC = {1000: 0.371, 2000: 0.362, 3000: 0.381, 4000: 0.769, 5000: 0.899}

OUTPUT_PNG = 'sweep_far_frac.png'
OUTPUT_CSV = 'sweep_far_frac_results.csv'


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


def run_episode_fixed_split(env, far_frac, seed, rng):
    """Every agent's near/far choice at 'center' is a fixed coin-flip with
    probability far_frac, independent of any policy or congestion state.
    Everything else (staggered departure, congestion dynamics, speed model)
    runs exactly as it would for a trained policy or Dijkstra."""
    cidx = env.node_to_idx['center']
    obs_mat, active_mask, infos = env.reset(seed=seed)
    actions_arr = np.zeros(env.n_agents, dtype=np.int32)
    step = 0
    while env.agents and step < env.max_steps:
        active_idxs = np.where(active_mask)[0]
        if len(active_idxs) > 0:
            chosen = (rng.random(len(active_idxs)) < far_frac).astype(np.int32)
            actions_arr[active_idxs] = chosen
        obs_mat, active_mask, rewards, terminations, truncations, infos = env.step(actions_arr)
        step += 1
        if not env.agents or any(truncations.values()):
            break
    return env.summary()


def main():
    rng = np.random.default_rng(0)
    all_results = {}  # demand -> {far_frac: (arrived_only_mean, with_timeout_mean, arrival_rate_mean)}
    best_split = {}    # demand -> (best_far_frac, best_with_timeout)

    for demand in DEMAND_LEVELS:
        env = build_env(demand)
        print(f'\n=== N_AGENTS = {demand} ===')
        results = {}
        for ff in FAR_FRAC_GRID:
            arrived_only_vals, with_timeout_vals, arrival_rates = [], [], []
            for ep in range(N_EPISODES_PER_POINT):
                summary = run_episode_fixed_split(env, ff, BASE_SEED + ep, rng)
                arrived_only_vals.append(summary['avg_arrival_time_arrived_only'])
                with_timeout_vals.append(summary['avg_arrival_time_with_timeout'])
                arrival_rates.append(summary['n_arrived'] / summary['n_agents'])
            ao_mean = float(np.nanmean(arrived_only_vals))
            wt_mean = float(np.nanmean(with_timeout_vals))
            ar_mean = float(np.mean(arrival_rates))
            results[ff] = (ao_mean, wt_mean, ar_mean)
            print(f'  far_frac={ff:.1f} | arrived_only={ao_mean:7.1f}s | '
                  f'with_timeout={wt_mean:7.1f}s | arrival_rate={ar_mean:.1%}')

        all_results[demand] = results
        best_ff = min(results, key=lambda f: results[f][1])
        best_split[demand] = (best_ff, results[best_ff][1])
        print(f'  --> Best far_frac for N={demand}: {best_ff:.1f} '
              f'(avg arrival time incl. timeouts = {results[best_ff][1]:.1f}s)')

    # ── Summary table ─────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print(f"{'N_AGENTS':>10} | {'Sweep-optimal far_frac':>22} | "
          f"{'MAPPO far_frac':>15} | {'Match?':>8}")
    print('-' * 78)
    for demand in DEMAND_LEVELS:
        opt_ff, opt_time = best_split[demand]
        mappo_ff = MAPPO_FAR_FRAC.get(demand, float('nan'))
        match = 'yes' if abs(opt_ff - mappo_ff) <= 0.15 else 'no'
        print(f"{demand:>10} | {opt_ff:>18.1f} ({opt_time:6.1f}s) | "
              f"{mappo_ff:>15.3f} | {match:>8}")
    print('=' * 78)

    # ── CSV ────────────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, 'w') as f:
        f.write('demand,far_frac,arrived_only_s,with_timeout_s,arrival_rate\n')
        for demand in DEMAND_LEVELS:
            for ff in FAR_FRAC_GRID:
                ao, wt, ar = all_results[demand][ff]
                f.write(f'{demand},{ff:.1f},{ao:.2f},{wt:.2f},{ar:.4f}\n')
    print(f'\nSaved raw sweep data: {OUTPUT_CSV}')

    # ── Plot: arrival time (with timeout) vs far_frac, one line per demand ──
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    colors = plt.cm.viridis(np.linspace(0, 1, len(DEMAND_LEVELS)))
    for demand, color in zip(DEMAND_LEVELS, colors):
        xs = FAR_FRAC_GRID
        ys = [all_results[demand][ff][1] for ff in xs]
        ax.plot(xs, ys, 'o-', color=color, label=f'N={demand}')
        opt_ff, opt_time = best_split[demand]
        ax.plot(opt_ff, opt_time, '*', color=color, ms=16, mec='black', mew=0.5)
        mappo_ff = MAPPO_FAR_FRAC.get(demand)
        if mappo_ff is not None:
            ax.axvline(mappo_ff, color=color, lw=1, ls=':', alpha=0.6)
    ax.set_xlabel('far_frac (fixed)'); ax.set_ylabel('Avg arrival time incl. timeouts (s)')
    ax.set_title('Sweep: avg arrival time vs. fixed far_frac, by demand level\n'
                 '(stars = sweep-found optimum; dotted lines = MAPPO\'s actual far_frac)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_PNG, dpi=150); plt.close()
    print(f'Saved: {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
