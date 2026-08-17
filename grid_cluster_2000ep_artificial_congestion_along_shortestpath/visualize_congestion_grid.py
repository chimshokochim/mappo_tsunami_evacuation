"""
visualize_congestion_grid.py  --  Same idea as visualize_congestion.py, but
on the small synthetic NxN grid map (common.build_grid_graph) instead of the
real Kochi OSM map. Animates one shortest-path episode as an mp4, coloring
every road edge by its congestion level at each simulation step.

All agents follow the greedy shortest-path policy (action 0 = toward each
agent's own randomly-assigned destination shelter). Congestion (Greenshields
speed slowdown) still applies.

This script builds the grid fresh from common.build_grid_graph() /
common.make_grid_evac_data() (same config knobs as training.py's
USE_GRID_MAP block) rather than loading a cached graph_data*.pkl, so it
works standalone without having to run training.py first.

Edge color rule (density = agents_on_edge / edge_area):
  gray   : density <  50% of VIS_DENSITY_MAX
  yellow : density >= 50% and < 80% of VIS_DENSITY_MAX
  red    : density >= 80% of VIS_DENSITY_MAX

On a 5x5 grid, edges are much shorter and wider (relative to the small
number of agents) than on the real city map, so real congestion density
stays far lower in absolute terms than the real-map script. VIS_DENSITY_MAX
is set much lower here by default to compensate -- see the note above
VIS_DENSITY_MAX below if you change GRID_* config and colors stop showing.

Requires ffmpeg. This script first tries to auto-locate ffmpeg via the
`imageio-ffmpeg` package (`pip install imageio-ffmpeg`), which bundles its
own ffmpeg binary so nothing needs to be on PATH. If that package isn't
installed, it falls back to whatever `ffmpeg` matplotlib finds on PATH.

Usage:
    python visualize_congestion_grid.py
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

from common import (SEED, build_grid_graph, make_grid_evac_data,
                     compute_shelter_distances, build_graph_index)
from evac_env import EvacuationEnv, DENSITY_MAX, STEP_TIME, STATUS_EVACUATING

# ── Grid config (mirrors training.py's USE_GRID_MAP block) ─────────────────────
GRID_ROWS              = 5
GRID_COLS              = 5
GRID_CELL_SIZE_M       = 80.0
GRID_CONNECT_DIAGONALS = True    # 8-connected
GRID_N_SHELTERS        = 2       # randomly placed among the 25 nodes
GRID_SEED              = 42      # which shelters get chosen

# ── Simulation config ────────────────────────────────────────────────────────
OUTPUT_MP4 = 'grid_shortest_path_congestion.mp4'
N_AGENTS   = 3000   # a 25-node grid is tiny, so a lot of agents are needed to
                     # produce any real congestion at all -- see the module
                     # docstring / the density check this was tuned against.
MAX_STEPS  = 150
FPS        = 6
DPI        = 150

# Cluster everyone's start near one random node so agents actually collide
# with each other in the same neighborhood, instead of spreading out evenly
# across all 25 nodes (which barely produces any congestion on a grid this
# small). Set to False for uniform-random starts.
CLUSTER_START       = True
CLUSTER_RADIUS_HOPS = 1

# Artificial congestion (permanent density floor on random edges). See
# EvacuationEnv's docstring / training.py's ARTIFICIAL_CONGESTION block.
ARTIFICIAL_CONGESTION           = False
ARTIFICIAL_CONGESTION_LEVELS    = (0.5, 0.8)
ARTIFICIAL_CONGESTION_FRACTION  = 0.05
ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH = False
ARTIFICIAL_CONGESTION_SP_FRACTION          = 1.0

# Visualization-only congestion scale (see evac_env.DENSITY_MAX for the
# physical jam-density constant used by the simulation itself). On this
# grid, edges are wide (edge_area = length * ROAD_WIDTH) relative to how
# many agents can realistically be on one at once, so density stays far
# below the physical jam density -- 0.5 was tuned against an observed peak
# of ~1.0 agents/m^2 with N_AGENTS=3000, CLUSTER_RADIUS_HOPS=1. Lower this
# if you don't see any yellow/red; raise it (toward DENSITY_MAX=5.4) to
# color strictly against the physical jam density instead.
VIS_DENSITY_MAX = 0.5

COLOR_LOW  = '#b8b8b8'
COLOR_MED  = '#ffd400'
COLOR_HIGH = '#e0262b'
WIDTH_LOW  = 2.0
WIDTH_MED  = 4.0
WIDTH_HIGH = 5.5


def classify(density):
    """Return (color, linewidth) for a given edge density."""
    if density >= 0.8 * VIS_DENSITY_MAX:
        return COLOR_HIGH, WIDTH_HIGH
    if density >= 0.5 * VIS_DENSITY_MAX:
        return COLOR_MED, WIDTH_MED
    return COLOR_LOW, WIDTH_LOW


def main():
    node_coords, adj, road_nodes = build_grid_graph(
        rows=GRID_ROWS, cols=GRID_COLS, cell_size_m=GRID_CELL_SIZE_M,
        connect_diagonals=GRID_CONNECT_DIAGONALS)
    evac_nodes, evac_capacity = make_grid_evac_data(
        road_nodes, n_shelters=GRID_N_SHELTERS, seed=GRID_SEED)

    shelter_dist, _ = compute_shelter_distances(evac_nodes, adj, road_nodes)
    node_list, node_to_idx, neighbor_lists, max_degree = build_graph_index(
        adj, road_nodes, shelter_dist=shelter_dist)

    env = EvacuationEnv(
        node_coords=node_coords, adj=adj, road_nodes=road_nodes,
        evac_nodes=evac_nodes, evac_capacity=evac_capacity,
        node_list=node_list, node_to_idx=node_to_idx,
        neighbor_lists=neighbor_lists, max_degree=max_degree,
        n_agents=N_AGENTS, max_steps=MAX_STEPS, reward_dest=1.0,
        cluster_start=CLUSTER_START, cluster_radius_hops=CLUSTER_RADIUS_HOPS,
        artificial_congestion=ARTIFICIAL_CONGESTION,
        artificial_congestion_levels=ARTIFICIAL_CONGESTION_LEVELS,
        artificial_congestion_fraction=ARTIFICIAL_CONGESTION_FRACTION,
        artificial_congestion_target_shortest_path=ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH,
        artificial_congestion_sp_fraction=ARTIFICIAL_CONGESTION_SP_FRACTION,
    )
    env.reset(seed=SEED)

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

    fig, ax = plt.subplots(figsize=(8, 8))
    # Show every grid node as a small dot so the underlying grid structure
    # (rows/cols) is easy to read, in addition to the colored edges.
    node_lons = [lon for lat, lon in node_coords.values()]
    node_lats = [lat for lat, lon in node_coords.values()]
    ax.scatter(node_lons, node_lats, c='#666666', s=25, zorder=2)

    lc = LineCollection(segments, colors=COLOR_LOW, linewidths=WIDTH_LOW,
                         capstyle='round', joinstyle='round', zorder=1)
    ax.add_collection(lc)
    ax.scatter(evac_xy[:, 0], evac_xy[:, 1], c='#1f4fa3', marker='*', s=260,
               zorder=3, label='Shelter')
    if env._cluster_center is not None:
        c_lat, c_lon = node_coords[env._cluster_center]
        ax.scatter([c_lon], [c_lat], c='black', marker='x', s=200,
                   linewidths=3, zorder=4, label='Cluster center')

    margin = GRID_CELL_SIZE_M / 111000.0 * 0.5
    ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
    ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    legend_handles = [
        plt.Line2D([0], [0], color=COLOR_LOW, lw=4, label='< 50% density'),
        plt.Line2D([0], [0], color=COLOR_MED, lw=4, label='50-80% density'),
        plt.Line2D([0], [0], color=COLOR_HIGH, lw=4, label='>= 80% density'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#1f4fa3',
                   markersize=16, label='Shelter'),
    ]
    if env._cluster_center is not None:
        legend_handles.append(
            plt.Line2D([0], [0], marker='x', color='black', markersize=10,
                       lw=0, markeredgewidth=3, label='Cluster center'))
    ax.legend(loc='upper left', fontsize=9, handles=legend_handles,
              bbox_to_anchor=(1.0, 1.0))
    title = ax.set_title('')

    writer = FFMpegWriter(fps=FPS)
    zero_actions = np.zeros(N_AGENTS, dtype=np.int32)
    n_edges = len(edges)

    def update_collection(link_use):
        colors = [None] * n_edges
        widths = [None] * n_edges
        priority = [0] * n_edges
        for i, (cidx, nidx) in enumerate(edges):
            cnt = link_use.get((cidx, nidx), 0)
            area = env.edge_area[(cidx, nidx)]
            color, width = classify(env._eff_density(cidx, nidx, cnt, area))
            colors[i] = color
            widths[i] = width
            priority[i] = 0 if color == COLOR_LOW else (1 if color == COLOR_MED else 2)
        order = sorted(range(n_edges), key=lambda i: priority[i])
        lc.set_segments(segments[order])
        lc.set_color([colors[i] for i in order])
        lc.set_linewidth([widths[i] for i in order])

    with writer.saving(fig, OUTPUT_MP4, dpi=DPI):
        update_collection({})
        title.set_text(f'Grid shortest-path baseline | Step 0 | t=0s | '
                        f'Evacuating: {N_AGENTS}/{N_AGENTS}')
        writer.grab_frame()

        step = 0
        while env.agents and step < MAX_STEPS:
            env.step(zero_actions)
            step += 1

            update_collection(env._current_link_use)

            n_evac = int(np.sum(env._agent_status == STATUS_EVACUATING))
            title.set_text(f'Grid shortest-path baseline | Step {step} | '
                            f't={step * STEP_TIME:.0f}s | Evacuating: {n_evac}/{N_AGENTS}')
            writer.grab_frame()

    plt.close(fig)
    print(f'Saved animation: {OUTPUT_MP4}  ({step} steps)')


if __name__ == '__main__':
    main()
