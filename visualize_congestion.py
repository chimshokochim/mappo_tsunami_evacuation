"""
visualize_congestion.py  --  Animate one shortest-path episode as an mp4,
coloring every road edge by its congestion level at each simulation step.

All agents follow the greedy shortest-path policy (action 0 = toward each
agent's own randomly-assigned destination shelter), i.e. the same "no
learning" baseline used by evaluate_shortest_path_baseline() in
training.py. Congestion (Greenshields speed slowdown) still applies.

Edge color rule (density = agents_on_edge / edge_area):
  gray   : density <  50% of VIS_DENSITY_MAX
  yellow : density >= 50% and < 80% of VIS_DENSITY_MAX
  red    : density >= 80% of VIS_DENSITY_MAX

VIS_DENSITY_MAX is a visualization-only scale, separate from evac_env.py's
DENSITY_MAX (the jam-density constant used by the Greenshields speed model
and by the RL observations/reward). With this scenario's random per-agent
destinations spread across 85 shelters, real congestion tops out far below
the physical jam density (observed peak was ~36% of DENSITY_MAX=5.4, i.e.
~1.95 agents/m^2) so coloring against the physical DENSITY_MAX would never
show yellow/red. VIS_DENSITY_MAX lets you rescale just for this picture
without touching the simulation itself.

Requires ffmpeg. This script first tries to auto-locate ffmpeg via the
`imageio-ffmpeg` package (`pip install imageio-ffmpeg`), which bundles its
own ffmpeg binary so nothing needs to be on PATH. If that package isn't
installed, it falls back to whatever `ffmpeg` matplotlib finds on PATH.

Usage:
    python visualize_congestion.py
"""

import pickle
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

from common import SEED, N_TRAIN_AGENTS, MAX_STEPS_EP, GRAPH_PATH
from evac_env import EvacuationEnv, DENSITY_MAX, STEP_TIME, STATUS_EVACUATING

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_MP4 = 'shortest_path_congestion.mp4'
N_AGENTS   = N_TRAIN_AGENTS   # lower this (e.g. 500) for a much faster render
MAX_STEPS  = MAX_STEPS_EP
FPS        = 8
DPI        = 150

# With uniformly-random start positions spread across the whole city, real
# congestion stays very low (destinations are also spread across 85
# shelters). Clustering everyone's start position into one neighborhood
# forces them through much narrower shared road capacity, which is a much
# more interesting congestion picture. Set to False to go back to uniform
# random starts (matches training.py's default behavior).
CLUSTER_START       = False
CLUSTER_RADIUS_HOPS = 6   # candidate start nodes = within this many hops of a random center

# Restrict WHERE the cluster center itself can land, as a fraction of the
# map's longitude range: 0.0 = westernmost edge, 1.0 = easternmost edge.
# e.g. (0.75, 1.0) only allows cluster centers in the eastern-most quarter
# of the map. Set to None to allow any non-shelter node (no geographic bias).
CLUSTER_LON_RANGE_FRACTION = (0.75, 1.0)

# Visualization-only congestion scale (see module docstring). Tune this to
# taste: lower it to make yellow/red appear more readily, raise it toward
# evac_env.DENSITY_MAX to color strictly against the physical jam density
# instead. With CLUSTER_START=True, density can spike far above DENSITY_MAX
# at the bottleneck (agents crawl at SPEED_MIN_ABS), so 1.0 here already
# saturates to red well before that.
VIS_DENSITY_MAX = 1.0

COLOR_LOW  = '#b8b8b8'   # density < 50% of VIS_DENSITY_MAX
COLOR_MED  = '#ffd400'   # 50% <= density < 80%
COLOR_HIGH = '#e0262b'   # density >= 80%

# With ~14000 directed edges on the map, congested edges are individually a
# tiny fraction of the network, so a uniform 1px line makes them nearly
# invisible even when they ARE yellow/red. Draw congested edges noticeably
# thicker so isolated hotspots are actually readable in the video.
WIDTH_LOW  = 1.0
WIDTH_MED  = 2.5
WIDTH_HIGH = 3.5


def classify(density):
    """Return (color, linewidth) for a given edge density."""
    if density >= 0.8 * VIS_DENSITY_MAX:
        return COLOR_HIGH, WIDTH_HIGH
    if density >= 0.5 * VIS_DENSITY_MAX:
        return COLOR_MED, WIDTH_MED
    return COLOR_LOW, WIDTH_LOW


def main():
    with open(GRAPH_PATH, 'rb') as f:
        g = pickle.load(f)

    cluster_center_pool = None
    if CLUSTER_START and CLUSTER_LON_RANGE_FRACTION is not None:
        lo_frac, hi_frac = CLUSTER_LON_RANGE_FRACTION
        evac_set = set(g['evac_nodes'])
        all_lons_tmp = [lon for lat, lon in g['node_coords'].values()]
        lon_min, lon_max = min(all_lons_tmp), max(all_lons_tmp)
        lon_lo = lon_min + lo_frac * (lon_max - lon_min)
        lon_hi = lon_min + hi_frac * (lon_max - lon_min)
        cluster_center_pool = [
            n for n in g['node_list']
            if n not in evac_set and n in g['node_coords']
            and lon_lo <= g['node_coords'][n][1] <= lon_hi]
        print(f'Cluster center pool: {len(cluster_center_pool)} nodes '
              f'(lon in [{lon_lo:.5f}, {lon_hi:.5f}])')

    env = EvacuationEnv(
        node_coords=g['node_coords'], adj=g['adj'], road_nodes=set(g['node_list']),
        evac_nodes=set(g['evac_nodes']), evac_capacity=g['evac_capacity'],
        node_list=g['node_list'], node_to_idx=g['node_to_idx'],
        neighbor_lists=g['neighbor_lists'], max_degree=g['max_degree'],
        n_agents=N_AGENTS, max_steps=MAX_STEPS, reward_dest=1.0,
        cluster_start=CLUSTER_START, cluster_radius_hops=CLUSTER_RADIUS_HOPS,
        cluster_center_pool=cluster_center_pool,
    )
    env.reset(seed=SEED)

    node_coords = g['node_coords']
    node_list   = g['node_list']
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

    fig, ax = plt.subplots(figsize=(10, 10))
    # capstyle/joinstyle='round': many congested segments turned out to be
    # very short connector edges (~1-3m) near intersections. With the
    # default 'butt' capstyle, a near-zero-length segment renders as
    # essentially nothing regardless of color/linewidth. 'round' draws a
    # visible dot even for these, so isolated hotspots are actually visible.
    lc = LineCollection(segments, colors=COLOR_LOW, linewidths=WIDTH_LOW,
                         capstyle='round', joinstyle='round', zorder=1)
    ax.add_collection(lc)
    ax.scatter(evac_xy[:, 0], evac_xy[:, 1], c='#1f4fa3', marker='*', s=90,
               zorder=3, label='Shelter')
    if env._cluster_center is not None:
        c_lat, c_lon = node_coords[env._cluster_center]
        ax.scatter([c_lon], [c_lat], c='black', marker='x', s=140,
                   linewidths=3, zorder=4, label='Cluster center')
    ax.set_xlim(min(all_lons), max(all_lons))
    ax.set_ylim(min(all_lats), max(all_lats))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=9,
              handles=[
                  plt.Line2D([0], [0], color=COLOR_LOW, lw=3, label='< 50% density'),
                  plt.Line2D([0], [0], color=COLOR_MED, lw=3, label='50-80% density'),
                  plt.Line2D([0], [0], color=COLOR_HIGH, lw=3, label='>= 80% density'),
                  plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#1f4fa3',
                             markersize=12, label='Shelter'),
                  plt.Line2D([0], [0], marker='x', color='black', markersize=10,
                             lw=0, markeredgewidth=3, label='Cluster center'),
              ])
    title = ax.set_title('')

    writer = FFMpegWriter(fps=FPS)
    zero_actions = np.zeros(N_AGENTS, dtype=np.int32)

    n_edges = len(edges)

    def update_collection(link_use):
        """Recolor/rewidth all edges and re-order them so gray is drawn
        first and yellow/red are drawn last (on top), so isolated congested
        edges aren't hidden underneath the dense gray road network."""
        colors = [None] * n_edges
        widths = [None] * n_edges
        priority = [0] * n_edges   # 0=gray, 1=yellow, 2=red; draw order = priority
        for i, (cidx, nidx) in enumerate(edges):
            cnt = link_use.get((cidx, nidx), 0)
            area = env.edge_area[(cidx, nidx)]
            color, width = classify(cnt / area)
            colors[i] = color
            widths[i] = width
            priority[i] = 0 if color == COLOR_LOW else (1 if color == COLOR_MED else 2)

        order = sorted(range(n_edges), key=lambda i: priority[i])
        lc.set_segments(segments[order])
        lc.set_color([colors[i] for i in order])
        lc.set_linewidth([widths[i] for i in order])

    with writer.saving(fig, OUTPUT_MP4, dpi=DPI):
        # Frame 0: initial state, nobody has moved onto an edge yet.
        update_collection({})
        title.set_text(f'Shortest-path baseline | Step 0 | t=0s | '
                        f'Evacuating: {N_AGENTS}/{N_AGENTS}')
        writer.grab_frame()

        step = 0
        while env.agents and step < MAX_STEPS:
            env.step(zero_actions)
            step += 1

            update_collection(env._current_link_use)

            n_evac = int(np.sum(env._agent_status == STATUS_EVACUATING))
            title.set_text(f'Shortest-path baseline | Step {step} | '
                            f't={step * STEP_TIME:.0f}s | Evacuating: {n_evac}/{N_AGENTS}')
            writer.grab_frame()

    plt.close(fig)
    print(f'Saved animation: {OUTPUT_MP4}  ({step} steps)')


if __name__ == '__main__':
    main()
