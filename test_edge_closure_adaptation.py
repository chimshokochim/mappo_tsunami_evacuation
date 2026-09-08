"""
test_edge_closure_adaptation.py  --  Road-closure adaptation/recovery test
for the line topology, adapted from the "Perturbation experiments"
methodology in "Decentralized graph attention multi-agent reinforcement
learning for adaptive urban traffic routing" (the uploaded Nature paper).

WHAT THE PAPER DOES (SUMO, continuous flow): run under normal conditions,
remove some edges at time t1, keep running under the perturbation, restore
the edges at time t2, then measure (a) performance degradation at the
moment of closure and (b) how many steps it takes for performance to
recover to within 95% of its pre-closure baseline. It compares a
graph-attention MARL policy against Dijkstra (which "never recovers"
because it uses static shortest-path weights and can't react at all).

WHY THIS PROJECT NEEDS A DIFFERENT SETUP: the paper's traffic is a
continuous flow -- vehicles keep entering the network throughout the
perturbation and recovery windows, so "recovery" is a property of the
whole traffic stream re-equilibrating. Here, agents are a single wave of
pedestrians with staggered (but one-time) departures (see
STAGGERED_DEPARTURE in evac_env.py), and -- critically -- once an agent
commits to an edge at 'center' it cannot turn around or reroute (edges
have no intermediate decision points; see evac_env.py's step() Phase 1).
So there are two genuinely different things "adaptation" can mean here,
and this script measures BOTH:

  1. BEHAVIORAL adaptation: only agents who are still WAITING or still AT
     'center' when the closure is active can react to it at all (choose
     the far edge instead). This is measured as P(choose far) among
     decisions made during the closure window vs. before/after it --
     the direct MARL-relevant signal ("did the policy learn to divert
     traffic away from a suddenly-bad edge").
  2. SYSTEM-level recovery (closer to the paper's literal metric):
     density_near / avg-speed-near over the whole episode timeline. This
     mixes TWO effects together -- agents rerouting away (behavioral) AND
     agents who already committed to the near edge before the closure
     simply finishing their walk and leaving (mechanical queue-clearing,
     which happens even with a policy that can't adapt at all) -- so it's
     reported with that caveat rather than treated as a pure adaptation
     signal on its own.

RECOVERY BASELINE -- COUNTERFACTUAL, NOT "early episode snapshot": the
first version of this script defined "100% performance" as the average of
steps 30-50 (just before the closure). That failed here: with N_AGENTS=3000
funneling onto one edge (area = 150m x 5m = 750 sq-m), density_near climbs
past DENSITY_MAX on its own, closure or not, simply because more agents
keep departing throughout the whole 0-225 staggered-departure window --
it never comes back down to its early-episode value even without any
closure at all. So "recovery" against that baseline was measuring natural
traffic buildup, not the closure's effect.

Instead, every scenario is run TWICE per seed: once WITH the closure and
once WITHOUT it (otherwise identical -- same seed, same policy, so agent
departure times/speeds match exactly). The WITHOUT-closure run is the
counterfactual "what would have happened anyway" baseline. "Recovery time"
= steps after the closure lifts until the WITH-closure trajectory comes
back within RECOVERY_THRESHOLD_FRAC (relative) of the WITHOUT-closure
trajectory AT THE SAME STEP. This isolates the closure's actual effect
from ordinary departure-driven congestion growth.

CLOSURE TIMING -- moved per Bhaskar's feedback on the first version: with
the closure at steps 50-100, by the time it started roughly 22% of agents
had already departed and committed to an edge (staggered departure spreads
departures over steps 0-225), and by the time it ended only ~44% had
departed -- so most agents' actual near/far decision fell either before or
long after the closure, and the aggregate P(far) barely moved even if the
policy was responding correctly to the (few) decisions that did fall inside
the window. Now the closure starts at step 0 (nobody has departed yet) and
stays active through step 225 (the end of the departure window), so
essentially every agent's initial near/far decision happens while the
closure is in effect -- maximizing the number of decisions that can
actually show a reaction, per Bhaskar's suggestion to move the closure
earlier / align it with a longer span of departures.

Runs the SAME closure scenario under two policies for direct comparison:
  - MAPPO: the trained Actor (mappo_line_actor.pt), which CAN divert
    newly-deciding agents to the far edge during the closure.
  - Shortest-path baseline: always picks the nearest edge (action 0,
    i.e. always 'near'), completely blind to the closure, mirroring the
    paper's "Dijkstra: no recovery" comparison point.

Does NOT touch evac_env.py's reward computation (Phase 2's r_shape /
r_congestion / r_time / rewards[a] lines) -- this script only reads
observations/state, it plays no role in training.

Requires torch (for the MAPPO Actor). Run this locally, not in a
sandbox without torch.

Usage:
    python test_edge_closure_adaptation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common import (SEED, build_line_graph, make_line_evac_data,
                     compute_shelter_distances, build_graph_index)
from evac_env import (EvacuationEnv, DENSITY_MAX, AGENT_SPEED_MEAN,
                       _piecewise_speed, STATUS_WAITING, STATUS_EVACUATING)

# ── Config (mirrors training.py's USE_LINE_MAP block) ──────────────────────────
ACTOR_PATH   = 'mappo_line_actor.pt'
HIDDEN_SIZE  = 64

LINE_DIST_NEAR         = 150.0
LINE_DIST_FAR          = 300.0
N_AGENTS                = 3000
MAX_STEPS                = 900
STAGGERED_DEPARTURE      = True
DEPARTURE_WINDOW_FRAC    = 0.25   # everyone has departed by step 225
NEAREST_SHELTER_TARGET   = True

# ── Closure config ──────────────────────────────────────────────────────────────
# Timed to a 20% / 60% / 20% split of departures relative to the closure
# window: departure steps are sampled uniformly from [0, DEPARTURE_WINDOW]
# (DEPARTURE_WINDOW = DEPARTURE_WINDOW_FRAC * MAX_STEPS = 225 here; see
# evac_env.py's reset(), self._agent_depart_step), so putting the closure
# boundaries at the 20th and 80th percentile of that range means ~20% of
# agents have already departed before the closure starts, ~60% depart while
# it's active, and the remaining ~20% depart only after it's lifted --
# giving a "before / during / after" population split for comparison,
# rather than (as in the two earlier versions of this script) either too
# few agents overlapping the closure (50-100) or essentially the entire
# population overlapping it (0-225, no "before"/"after" group left to
# compare against). Closes the 'near' edge specifically, since that's the
# dominant/majority route under NEAREST_SHELTER_TARGET (everyone's default
# target is shelter_near).
CLOSURE_EDGE         = ('center', 'shelter_near')
DEPARTURE_WINDOW     = DEPARTURE_WINDOW_FRAC * MAX_STEPS   # == 225
CLOSURE_START_STEP   = round(0.20 * DEPARTURE_WINDOW)      # == 45
CLOSURE_END_STEP     = round(0.80 * DEPARTURE_WINDOW)      # == 180
CLOSURE_DENSITY_FLOOR = 1.0   # fraction of DENSITY_MAX -- full jam density,
                               # speed = SPEED_MIN_ABS (0.01 m/s) for everyone
                               # on the edge, i.e. effectively closed/crawling
                               # (at 0.01 m/s, crossing 150m would take far
                               # longer than the episode horizon, so this is
                               # a genuine "nobody gets through" closure,
                               # unlike 0.9 which was only an ~88% slowdown).

# "Recovered" = with-closure value is within this relative fraction of the
# same-step no-closure (counterfactual) value, e.g. 0.95 means the two
# trajectories are within 5% of each other.
RECOVERY_THRESHOLD_FRAC = 0.95
# Once "recovered" at a step, require it to stay recovered for this many
# consecutive steps before accepting it (avoids reporting a single noisy
# crossing as "recovery" when the two curves are just passing through each
# other on their way to different long-run levels).
RECOVERY_SUSTAIN_STEPS = 10

N_EPISODES = 5
BASE_SEED  = SEED

OUTPUT_PNG = 'edge_closure_adaptation.png'


class Actor(nn.Module):
    """Must match training.py's Actor exactly, or load_state_dict fails."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim), nn.Softmax(dim=-1))
    def forward(self, x): return self.net(x)


def build_env(use_closure):
    node_coords, adj, road_nodes = build_line_graph(
        dist_near=LINE_DIST_NEAR, dist_far=LINE_DIST_FAR)
    evac_nodes, evac_capacity = make_line_evac_data()
    shelter_dist, _ = compute_shelter_distances(evac_nodes, adj, road_nodes)
    node_list, node_to_idx, neighbor_lists, max_degree = build_graph_index(
        adj, road_nodes, shelter_dist=shelter_dist)

    edge_closure = None
    if use_closure:
        edge_closure = {
            'edge': CLOSURE_EDGE,
            'start_step': CLOSURE_START_STEP,
            'end_step': CLOSURE_END_STEP,
            'density_floor': CLOSURE_DENSITY_FLOOR,
        }

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
        edge_closure=edge_closure,
    )


def run_episode(env, actor, device, seed, use_policy):
    """Rolls out one episode. If use_policy, samples actions from `actor`;
    otherwise always picks action 0 (shortest-path / nearest-edge baseline).
    Returns:
      density_near_by_step: list, index = step (1-based), density on the
        near edge (post-closure-floor, i.e. what agents actually see/feel).
      decisions: list of (step, is_near_edge_decision) tuples -- one entry
        per agent decision made AT 'center' (action index 0 = near,
        1 = far), used for the behavioral-adaptation metric. Only decisions
        where the agent's current node is 'center' count (re-targeting
        decisions after that are along a committed edge, not a fresh
        near-vs-far choice).
    """
    cidx = env.node_to_idx['center']
    nidx_near = env.node_to_idx['shelter_near']
    area_near = env.edge_area.get((cidx, nidx_near), 5.0)

    obs_mat, active_mask, infos = env.reset(seed=seed)
    actions_arr = np.zeros(env.n_agents, dtype=np.int32)
    density_near_by_step = []
    decisions = []  # (step, chose_far: bool)
    step = 0
    while env.agents and step < env.max_steps:
        active_idxs = np.where(active_mask)[0]
        if len(active_idxs) > 0:
            # Only decisions made from 'center' are a real near-vs-far choice.
            at_center = env._agent_node_idx[active_idxs] == cidx
            if use_policy:
                obs_active = obs_mat[active_idxs]
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_active).to(device)
                    probs = actor(obs_t)
                    actions_t = torch.multinomial(probs, 1).squeeze(1)
                chosen = actions_t.cpu().numpy().astype(np.int32)
            else:
                chosen = np.zeros(len(active_idxs), dtype=np.int32)  # always 'near'
            actions_arr[active_idxs] = chosen
            for j, idx in enumerate(active_idxs):
                if at_center[j]:
                    decisions.append((step, bool(chosen[j] == 1)))

        obs_mat, active_mask, rewards, terminations, truncations, infos = env.step(actions_arr)
        step += 1

        cnt = env._current_link_use.get((cidx, nidx_near), 0)
        d = env._eff_density(cidx, nidx_near, cnt, area_near)
        density_near_by_step.append(d)

        if not env.agents or any(truncations.values()):
            break
    return density_near_by_step, decisions


def recovery_time_vs_counterfactual(with_series, without_series, closure_end_step,
                                     threshold_frac, sustain_steps):
    """Steps after closure_end_step until `with_series` first comes back
    within threshold_frac (relative) of `without_series` AT THE SAME STEP,
    and stays that close for `sustain_steps` consecutive steps. Returns
    None if it never does within the recorded series. This is the
    counterfactual definition of "recovery" -- see module docstring for why
    an early-episode snapshot baseline doesn't work for this setup."""
    n = min(len(with_series), len(without_series))
    ok_run = 0
    for t in range(closure_end_step, n):
        wv, nv = with_series[t], without_series[t]
        if np.isnan(wv) or np.isnan(nv) or nv == 0:
            ok_run = 0
            continue
        ratio = wv / nv
        recovered_now = (threshold_frac <= ratio <= 1.0 / threshold_frac)
        if recovered_now:
            ok_run += 1
            if ok_run >= sustain_steps:
                return (t - ok_run + 1) - closure_end_step
        else:
            ok_run = 0
    return None


def far_frac_in_window(decisions, lo, hi):
    in_win = [c for (s, c) in decisions if lo <= s < hi]
    if not in_win:
        return float('nan')
    return sum(in_win) / len(in_win)


def mean_density_series(density_series_list):
    max_len = max(len(d) for d in density_series_list)
    padded = np.full((len(density_series_list), max_len), np.nan)
    for i, d in enumerate(density_series_list):
        padded[i, :len(d)] = d
    return np.nanmean(padded, axis=0)


def summarize(label, with_density_list, with_decisions_list,
              without_density_list, without_decisions_list):
    with_density    = mean_density_series(with_density_list)
    without_density = mean_density_series(without_density_list)
    with_speed = np.array([_piecewise_speed(AGENT_SPEED_MEAN, d) if not np.isnan(d) else np.nan
                            for d in with_density])
    without_speed = np.array([_piecewise_speed(AGENT_SPEED_MEAN, d) if not np.isnan(d) else np.nan
                               for d in without_density])

    rt_density = recovery_time_vs_counterfactual(
        with_density, without_density, CLOSURE_END_STEP,
        RECOVERY_THRESHOLD_FRAC, RECOVERY_SUSTAIN_STEPS)
    rt_speed = recovery_time_vs_counterfactual(
        with_speed, without_speed, CLOSURE_END_STEP,
        RECOVERY_THRESHOLD_FRAC, RECOVERY_SUSTAIN_STEPS)

    with_decisions    = [d for eps in with_decisions_list for d in eps]
    without_decisions = [d for eps in without_decisions_list for d in eps]

    def far_pair(lo, hi):
        return (far_frac_in_window(with_decisions, lo, hi),
                far_frac_in_window(without_decisions, lo, hi))

    # Closure boundaries are set at the 20th/80th percentile of the
    # departure distribution (see CLOSURE_START_STEP/END_STEP above), so
    # there's a real ~20% "before" group and ~20% "after" group again to
    # compare against the ~60% "during" group.
    before_w, before_wo = far_pair(0, CLOSURE_START_STEP)
    during_w, during_wo = far_pair(CLOSURE_START_STEP, CLOSURE_END_STEP)
    after_w, after_wo   = far_pair(CLOSURE_END_STEP, CLOSURE_END_STEP + 50)
    late_w, late_wo     = far_pair(CLOSURE_END_STEP + 50, MAX_STEPS)

    print(f'\n--- {label} ---')
    print(f'P(choose far), with-closure vs. no-closure counterfactual:')
    print(f'  before closure (~20% of departures): {before_w:.3f} vs {before_wo:.3f}  '
          f'(excess = {before_w - before_wo:+.3f})')
    print(f'  during closure (~60% of departures): {during_w:.3f} vs {during_wo:.3f}  '
          f'(excess = {during_w - during_wo:+.3f})')
    print(f'  50 steps after closure ends: {after_w:.3f} vs {after_wo:.3f}  '
          f'(excess = {after_w - after_wo:+.3f})')
    print(f'  later:                     {late_w:.3f} vs {late_wo:.3f}  '
          f'(excess = {late_w - late_wo:+.3f})')
    print(f'Recovery time (density_near back within '
          f'{int(RECOVERY_THRESHOLD_FRAC*100)}% of no-closure counterfactual, '
          f'sustained {RECOVERY_SUSTAIN_STEPS} steps): '
          f'{rt_density if rt_density is not None else "did not recover"} '
          f'steps after closure lifts')
    print(f'Recovery time (avg speed_near back within '
          f'{int(RECOVERY_THRESHOLD_FRAC*100)}% of no-closure counterfactual, '
          f'sustained {RECOVERY_SUSTAIN_STEPS} steps): '
          f'{rt_speed if rt_speed is not None else "did not recover"} '
          f'steps after closure lifts')

    return dict(with_density=with_density, without_density=without_density,
                with_speed=with_speed, without_speed=without_speed,
                rt_density=rt_density, rt_speed=rt_speed)


def run_scenario(env, actor, device, use_policy):
    """Runs N_EPISODES with the given env (already built with or without
    edge_closure) and policy, returns (density_series_list, decisions_list)."""
    density_list, decisions_list = [], []
    for ep in range(N_EPISODES):
        d, dec = run_episode(env, actor, device, BASE_SEED + ep, use_policy=use_policy)
        density_list.append(d); decisions_list.append(dec)
    return density_list, decisions_list


def binned_far_frac(decisions_list, bin_width=25, max_step=MAX_STEPS):
    all_d = [d for eps in decisions_list for d in eps]
    bins = np.arange(0, max_step + bin_width, bin_width)
    xs, ys = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        in_bin = [c for (s, c) in all_d if b0 <= s < b1]
        if in_bin:
            xs.append((b0 + b1) / 2)
            ys.append(sum(in_bin) / len(in_bin))
    return xs, ys


def main():
    env_closure    = build_env(use_closure=True)
    env_no_closure = build_env(use_closure=False)
    device = torch.device('cpu')
    actor = Actor(env_closure._obs_dim, env_closure.max_degree).to(device)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    actor.eval()

    print(f'Running {N_EPISODES} episodes x 2 policies x 2 (closure/no-closure)...')
    mappo_with_d,    mappo_with_dec    = run_scenario(env_closure,    actor, device, use_policy=True)
    mappo_without_d, mappo_without_dec = run_scenario(env_no_closure, actor, device, use_policy=True)
    print('MAPPO done.')
    base_with_d,    base_with_dec    = run_scenario(env_closure,    actor, device, use_policy=False)
    base_without_d, base_without_dec = run_scenario(env_no_closure, actor, device, use_policy=False)
    print('Shortest-path baseline done.')

    mappo_stats = summarize('MAPPO (trained policy)',
                             mappo_with_d, mappo_with_dec, mappo_without_d, mappo_without_dec)
    base_stats  = summarize('Shortest-path baseline (always near)',
                             base_with_d, base_with_dec, base_without_d, base_without_dec)

    # ── Plot: density_near AND P(far) together on the same time axis, per ────
    # policy -- per Bhaskar's suggestion to "plot P(far) against time
    # alongside near-edge density" so the reaction (or lack of it) is
    # directly visible against the congestion that's supposedly driving it,
    # rather than split across separate subplots.
    fig, (ax_m, ax_b) = plt.subplots(2, 1, figsize=(10, 9), dpi=150, sharex=True)
    for ax, stats, with_dec, without_dec, title in [
            (ax_m, mappo_stats, mappo_with_dec, mappo_without_dec, 'MAPPO'),
            (ax_b, base_stats, base_with_dec, base_without_dec, 'Shortest-path baseline')]:
        ax.axvspan(CLOSURE_START_STEP, CLOSURE_END_STEP, color='red', alpha=0.10,
                   label='Closure active')
        l1, = ax.plot(stats['with_density'], color='firebrick', lw=1.5,
                       label='density_near, with closure')
        l2, = ax.plot(stats['without_density'], color='firebrick', lw=1.2, ls=':',
                       label='density_near, no closure (counterfactual)')
        ax.set_ylabel('density_near', color='firebrick')
        ax.set_ylim(0, 4)
        ax.tick_params(axis='y', labelcolor='firebrick')

        ax2 = ax.twinx()
        wx, wy = binned_far_frac(with_dec)
        nx, ny = binned_far_frac(without_dec)
        l3, = ax2.plot(wx, wy, 'o-', color='steelblue', lw=1.5, ms=4,
                        label='P(choose far), with closure')
        l4, = ax2.plot(nx, ny, 'o-', color='steelblue', lw=1.2, ms=3, ls=':',
                        label='P(choose far), no closure (counterfactual)')
        ax2.set_ylabel('P(choose far)', color='steelblue')
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis='y', labelcolor='steelblue')

        rt = stats['rt_density']
        ax.set_title(f'{title}')
        ax.legend(handles=[l1, l2, l3, l4], fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    ax_b.set_xlabel('Simulation step')
    plt.tight_layout(); plt.savefig(OUTPUT_PNG, dpi=150); plt.close()
    print(f'\nSaved: {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
