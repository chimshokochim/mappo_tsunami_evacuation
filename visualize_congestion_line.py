"""
visualize_congestion_line.py  --  Animate one episode on the minimal "line"
topology (center -> shelter_near / center -> shelter_far) as an mp4, showing
all 3000 agents departing from 'center' with staggered timing and coloring
each edge by its congestion level at each simulation step.

Two modes, controlled by USE_TRAINED_MODEL below:
  USE_TRAINED_MODEL = True  -> loads a trained Actor (ACTOR_PATH, default
                               'mappo_line_actor.pt') and uses it to choose
                               actions for every agent, same inference-time
                               behavior as evaluate_model.py. Requires torch.
  USE_TRAINED_MODEL = False -> shortest-path baseline (action 0 always, i.e.
                               greedy toward each agent's own assigned
                               shelter). No torch required.

Since this topology only has ONE edge from 'center' to each shelter (no
intermediate nodes), every agent makes exactly one real decision -- at
'center', right after it leaves STATUS_WAITING -- and then simply walks
until it arrives. So what this animation mostly shows is: (1) the
'waiting' pool shrinking as staggered departures trickle out over the
first DEPARTURE_WINDOW_FRAC of the episode, (2) the two edges filling up
with agents and going yellow/red as congestion builds, and (3) the 'safe'
count climbing as agents finish crossing.

Config here (LINE_DIST_NEAR/FAR, N_AGENTS, MAX_STEPS, STAGGERED_DEPARTURE,
DEPARTURE_WINDOW_FRAC) mirrors training.py's USE_LINE_MAP block -- keep
them in sync with whatever run produced ACTOR_PATH.

Requires ffmpeg. This script first tries to auto-locate ffmpeg via the
`imageio-ffmpeg` package (`pip install imageio-ffmpeg`), which bundles its
own ffmpeg binary so nothing needs to be on PATH. If that package isn't
installed, it falls back to whatever `ffmpeg` matplotlib finds on PATH.

Usage:
    python visualize_congestion_line.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter

try:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    print("Note: imageio-ffmpeg not installed; falling back to system ffmpeg on PATH.\n"
          "      If this fails, run: pip install imageio-ffmpeg")

from common import (SEED, build_line_graph, make_line_evac_data,
                     compute_shelter_distances, build_graph_index)
from evac_env import (EvacuationEnv, DENSITY_MAX, STEP_TIME,
                       STATUS_EVACUATING, STATUS_SAFE, STATUS_FAILED, STATUS_WAITING)

# ── Mode ─────────────────────────────────────────────────────────────────────
USE_TRAINED_MODEL = True    # False = shortest-path baseline, no torch needed
ACTOR_PATH        = 'mappo_line_actor.pt'
HIDDEN_SIZE       = 64      # must match training.py's Actor architecture
DETERMINISTIC     = False   # True = argmax, False = stochastic sampling (same as training)

# ── Line topology config (mirrors training.py's USE_LINE_MAP block) ────────────
LINE_DIST_NEAR         = 150.0
LINE_DIST_FAR          = 300.0
N_AGENTS                = 3000
MAX_STEPS                = 900
STAGGERED_DEPARTURE      = True
DEPARTURE_WINDOW_FRAC    = 0.25
NEAREST_SHELTER_TARGET   = True   # must match training.py's LINE_NEAREST_SHELTER_TARGET

# ── Output / render config ──────────────────────────────────────────────────
OUTPUT_MP4 = 'line_trained_congestion.mp4' if USE_TRAINED_MODEL else 'line_shortest_path_congestion.mp4'
FPS        = 12
DPI        = 150

# Visualization-only congestion scale (see evac_env.DENSITY_MAX for the
# physical jam-density constant used by the simulation/reward -- NOT
# changed here). Measured peak density for this scenario (3000 agents,
# ROAD_WIDTH=5.0m, LINE_DIST_NEAR/FAR=150/300m) is ~0.275 (27.5% of
# DENSITY_MAX=1.0), so coloring against the physical jam density (1.0)
# never leaves gray. 0.3 makes yellow/red actually show up near the
# observed peak. This is purely cosmetic -- it does NOT affect the
# simulation, reward, or the trained model in any way. Raise/lower to
# taste; raise ROAD_WIDTH-worth back toward DENSITY_MAX if you retrain
# with a narrower road and want the color scale to match.
VIS_DENSITY_MAX = 0.3

COLOR_LOW  = '#b8b8b8'
COLOR_MED  = '#ffd400'
COLOR_HIGH = '#e0262b'
WIDTH_LOW  = 4.0
WIDTH_MED  = 7.0
WIDTH_HIGH = 9.0


def classify(density):
    """Return (color, linewidth) for a given edge density."""
    if density >= 0.8 * VIS_DENSITY_MAX:
        return COLOR_HIGH, WIDTH_HIGH
    if density >= 0.5 * VIS_DENSITY_MAX:
        return COLOR_MED, WIDTH_MED
    return COLOR_LOW, WIDTH_LOW


def build_env():
    node_coords, adj, road_nodes = build_line_graph(
        dist_near=LINE_DIST_NEAR, dist_far=LINE_DIST_FAR)
    evac_nodes, evac_capacity = make_line_evac_data()
    shelter_dist, _ = compute_shelter_distances(evac_nodes, adj, road_nodes)
    node_list, node_to_idx, neighbor_lists, max_degree = build_graph_index(
        adj, road_nodes, shelter_dist=shelter_dist)

    env = EvacuationEnv(
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
    return env, node_coords, node_list, node_to_idx


def main():
    env, node_coords, node_list, node_to_idx = build_env()
    env.reset(seed=SEED)

    actor = None
    device = None
    if USE_TRAINED_MODEL:
        import torch
        import torch.nn as nn

        class Actor(nn.Module):
            """Must match training.py's Actor exactly, or load_state_dict fails."""
            def __init__(self, obs_dim, action_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, HIDDEN_SIZE), nn.Tanh(),
                    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
                    nn.Linear(HIDDEN_SIZE, action_dim), nn.Softmax(dim=-1))
            def forward(self, x): return self.net(x)

        device = torch.device('cpu')
        actor = Actor(env._obs_dim, env.max_degree).to(device)
        actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
        actor.eval()

    edges = list(env.road_lengths.keys())   # (src_idx, dst_idx)
    segments = np.array([
        [(node_coords[node_list[cidx]][1], node_coords[node_list[cidx]][0]),
         (node_coords[node_list[nidx]][1], node_coords[node_list[nidx]][0])]
        for cidx, nidx in edges
    ])

    all_lons = [lon for lat, lon in node_coords.values()]
    all_lats = [lat for lat, lon in node_coords.values()]

    evac_node_idx = sorted(set(env._shelter_node_idx.tolist()))
    evac_xy = np.array([
        (node_coords[node_list[i]][1], node_coords[node_list[i]][0])
        for i in evac_node_idx])
    center_xy = (node_coords['center'][1], node_coords['center'][0])

    fig, ax = plt.subplots(figsize=(9, 5))
    lc = LineCollection(segments, colors=COLOR_LOW, linewidths=WIDTH_LOW,
                         capstyle='round', joinstyle='round', zorder=1)
    ax.add_collection(lc)
    ax.scatter(evac_xy[:, 0], evac_xy[:, 1], c='#1f4fa3', marker='*', s=400,
               zorder=3, label='Shelter')
    ax.scatter([center_xy[0]], [center_xy[1]], c='black', marker='o', s=120,
               zorder=4, label='Center (all agents start here)')
    for i in evac_node_idx:
        name = node_list[i]
        x, y = node_coords[name][1], node_coords[name][0]
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 14),
                    ha='center', fontsize=10, fontweight='bold')
    ax.annotate('center', center_xy, textcoords="offset points", xytext=(0, -20),
                ha='center', fontsize=10, fontweight='bold')

    margin = max(LINE_DIST_NEAR, LINE_DIST_FAR) / 111000.0 * 0.3
    ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect('equal')
    ax.axis('off')
    legend_handles = [
        plt.Line2D([0], [0], color=COLOR_LOW, lw=6, label='< 50% of jam density'),
        plt.Line2D([0], [0], color=COLOR_MED, lw=6, label='50-80% of jam density'),
        plt.Line2D([0], [0], color=COLOR_HIGH, lw=6, label='>= 80% of jam density'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#1f4fa3',
                   markersize=18, label='Shelter'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=10, label='Center'),
    ]
    ax.legend(loc='upper left', fontsize=9, handles=legend_handles,
              bbox_to_anchor=(0.0, -0.02))
    title = ax.set_title('')

    writer = FFMpegWriter(fps=FPS)
    n_edges = len(edges)

    def status_counts():
        s = env._agent_status
        return (int(np.sum(s == STATUS_WAITING)), int(np.sum(s == STATUS_EVACUATING)),
                int(np.sum(s == STATUS_SAFE)), int(np.sum(s == STATUS_FAILED)))

    def update_collection(link_use):
        colors = [None] * n_edges
        widths = [None] * n_edges
        for i, (cidx, nidx) in enumerate(edges):
            cnt = link_use.get((cidx, nidx), 0)
            area = env.edge_area[(cidx, nidx)]
            color, width = classify(env._eff_density(cidx, nidx, cnt, area))
            colors[i] = color
            widths[i] = width
        lc.set_color(colors)
        lc.set_linewidth(widths)

    mode_label = f'Trained MAPPO ({ACTOR_PATH})' if USE_TRAINED_MODEL else 'Shortest-path baseline'
    actions_arr = np.zeros(env.n_agents, dtype=np.int32)

    with writer.saving(fig, OUTPUT_MP4, dpi=DPI):
        update_collection({})
        w, e, sa, fa = status_counts()
        title.set_text(f'{mode_label} | Step 0 | t=0s | '
                        f'Waiting:{w} Evacuating:{e} Safe:{sa} Failed:{fa}')
        writer.grab_frame()

        obs_mat, active_mask, infos = env._obs_mat, env._active_mask, {}
        step = 0
        while env.agents and step < env.max_steps:
            active_idxs = np.where(active_mask)[0]
            if len(active_idxs) > 0:
                if USE_TRAINED_MODEL:
                    import torch
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs_mat[active_idxs]).to(device)
                        probs = actor(obs_t)
                        if DETERMINISTIC:
                            actions_t = torch.argmax(probs, dim=1)
                        else:
                            actions_t = torch.multinomial(probs, 1).squeeze(1)
                    actions_arr[active_idxs] = actions_t.cpu().numpy().astype(np.int32)
                else:
                    actions_arr[active_idxs] = 0

            obs_mat, active_mask, rewards, terminations, truncations, infos = env.step(actions_arr)
            step += 1

            update_collection(env._current_link_use)
            w, e, sa, fa = status_counts()
            title.set_text(f'{mode_label} | Step {step} | t={step * STEP_TIME:.0f}s | '
                            f'Waiting:{w} Evacuating:{e} Safe:{sa} Failed:{fa}')
            writer.grab_frame()

            if not env.agents or any(truncations.values()):
                break

    plt.close(fig)
    print(f'Saved animation: {OUTPUT_MP4}  ({step} steps, {step * STEP_TIME:.0f}s simulated)')
    print(f'Final: {env.summary()}')


if __name__ == '__main__':
    main()
