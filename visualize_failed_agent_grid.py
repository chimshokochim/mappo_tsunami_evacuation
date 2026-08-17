"""
visualize_failed_agent_grid.py  --  Same idea as visualize_failed_agent.py,
but for a model trained on the small synthetic grid map (training.py's
USE_GRID_MAP=True) instead of the real Kochi OSM map.

Builds the grid fresh from common.build_grid_graph()/make_grid_evac_data()
(same GRID_* config as training.py) rather than loading a graph_data*.pkl,
runs one episode with the trained Actor (action selection controlled by
DETERMINISTIC below: argmax or stochastic sampling, same as training), then
animates one agent that ended in STATUS_FAILED: its trail, current position,
start/destination, with the same congestion coloring as
visualize_congestion_grid.py in the background.

ACTOR_PATH currently points at 'mappo_actor.pt' because that file was last
overwritten by a grid training run (real-map and grid training used to share
the same filename -- training.py now saves grid runs to
'mappo_grid_actor.pt' / 'mappo_grid_critic.pt' instead, so if you retrain on
the grid again, update ACTOR_PATH below to 'mappo_grid_actor.pt'.

Requires ffmpeg (see visualize_congestion.py for the imageio-ffmpeg note).

Usage:
    python visualize_failed_agent_grid.py
"""

import numpy as np
import torch
import torch.nn as nn
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

from common import SEED, build_grid_graph, make_grid_evac_data, compute_shelter_distances, build_graph_index
from evac_env import (EvacuationEnv, DENSITY_MAX, STEP_TIME,
                       STATUS_EVACUATING, STATUS_SAFE, STATUS_FAILED)

# ── Grid config (must match the training.py run that produced ACTOR_PATH) ──────
GRID_ROWS               = 5
GRID_COLS                = 5
GRID_CELL_SIZE_M         = 80.0
GRID_CONNECT_DIAGONALS   = True
GRID_N_SHELTERS          = 2
GRID_SEED                = 42

CLUSTER_START            = True
CLUSTER_RADIUS_HOPS      = 2
N_AGENTS                 = 3000
MAX_STEPS                = 200

# Artificial congestion (permanent density floor on random edges). Must match
# whatever training.py config produced ACTOR_PATH. See EvacuationEnv's
# docstring / training.py's ARTIFICIAL_CONGESTION block for details.
ARTIFICIAL_CONGESTION           = True
ARTIFICIAL_CONGESTION_LEVELS    = (0.5, 0.8)
ARTIFICIAL_CONGESTION_FRACTION  = 0.15
ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH = True
ARTIFICIAL_CONGESTION_SP_FRACTION          = 1.0

# ── Model / output config ───────────────────────────────────────────────────────
ACTOR_PATH  = 'mappo_grid_actor.pt'
OUTPUT_MP4  = 'grid_failed_agent_trace.mp4'
HIDDEN_SIZE = 64     # must match training.py's Actor architecture

# True  = argmax (deterministic, no exploration noise) -- what the policy has
#         actually converged to, without low-probability sampling accidents.
# False = stochastic sampling (multinomial over the softmax probs), same
#         action-selection as during training rollouts.
DETERMINISTIC = False

EPISODE_SEED = SEED   # try SEED+1, SEED+2, ... if no agent fails this run
FPS = 6
DPI = 150

# Background congestion coloring. AUTO_SCALE_VIS_DENSITY=True (recommended)
# rescales the yellow/red thresholds to this episode's own OBSERVED peak
# density after the run finishes, instead of using a fixed constant. This
# matters here because N_AGENTS=150 (matching training.py's grid config) is
# tiny -- with a fixed VIS_DENSITY_MAX tuned for visualize_congestion_grid.py
# (which uses 3000 agents), density here never gets anywhere close, so no
# yellow/red would ever show. Set False to use the fixed VIS_DENSITY_MAX below.
AUTO_SCALE_VIS_DENSITY = True
VIS_DENSITY_MAX = 0.05   # only used if AUTO_SCALE_VIS_DENSITY is False
COLOR_LOW  = '#d8d8d8'
COLOR_MED  = '#ffd400'
COLOR_HIGH = '#e0262b'
WIDTH_LOW, WIDTH_MED, WIDTH_HIGH = 2.0, 4.0, 5.5


def classify(density):
    if density >= 0.8 * VIS_DENSITY_MAX:
        return COLOR_HIGH, WIDTH_HIGH
    if density >= 0.5 * VIS_DENSITY_MAX:
        return COLOR_MED, WIDTH_MED
    return COLOR_LOW, WIDTH_LOW


class Actor(nn.Module):
    """Must match training.py's Actor exactly, or load_state_dict will fail."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim), nn.Softmax(dim=-1))
    def forward(self, x): return self.net(x)


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
    obs_mat, active_mask, infos = env.reset(seed=EPISODE_SEED)

    device = torch.device('cpu')
    actor = Actor(env._obs_dim, env.max_degree).to(device)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    actor.eval()

    edges = list(env.road_lengths.keys())
    segments = np.array([
        [(node_coords[node_list[cidx]][1], node_coords[node_list[cidx]][0]),
         (node_coords[node_list[nidx]][1], node_coords[node_list[nidx]][0])]
        for cidx, nidx in edges
    ])
    n_edges = len(edges)

    history_node_idx  = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=np.int32)
    history_on_link   = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=bool)
    history_link_src  = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=np.int32)
    history_link_dst  = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=np.int32)
    history_link_prog = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=np.float32)
    history_link_use  = [dict()]

    # Per-decision diagnostics: the full action-probability vector the Actor
    # produced at every node-choice point, for every agent (cheap enough to
    # keep in memory; we only print it out for the highlighted agent later).
    probs_history  = np.full((MAX_STEPS + 1, N_AGENTS, max_degree), np.nan, dtype=np.float32)
    chosen_history = np.full((MAX_STEPS + 1, N_AGENTS), -1, dtype=np.int32)
    decided_mask   = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=bool)

    def record(t):
        history_node_idx[t]  = env._agent_node_idx
        history_on_link[t]   = env._agent_on_link
        history_link_src[t]  = env._agent_link_src
        history_link_dst[t]  = env._agent_link_dst
        history_link_prog[t] = env._agent_link_progress

    record(0)
    actions_arr = np.zeros(N_AGENTS, dtype=np.int32)

    step = 0
    mode_label = 'argmax, no exploration noise' if DETERMINISTIC else 'stochastic sampling (same as training)'
    print(f"Running episode with trained policy ({mode_label})...")
    while env.agents and step < MAX_STEPS:
        active_idxs = np.where(active_mask)[0]
        if len(active_idxs) > 0:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_mat[active_idxs])
                probs = actor(obs_t)
                if DETERMINISTIC:
                    actions_t = torch.argmax(probs, dim=1)
                else:
                    actions_t = torch.multinomial(probs, 1).squeeze(1)
            actions_arr[active_idxs] = actions_t.numpy().astype(np.int32)
            probs_history[step, active_idxs, :] = probs.detach().numpy()
            chosen_history[step, active_idxs] = actions_t.numpy().astype(np.int32)
            decided_mask[step, active_idxs] = True

        obs_mat, active_mask, rewards, terminations, truncations, infos = env.step(actions_arr)
        step += 1
        record(step)
        history_link_use.append(dict(env._current_link_use))
        if not env.agents or any(truncations.values()):
            break

    n_steps_run = step
    print(f"Episode finished after {n_steps_run} steps.")

    if AUTO_SCALE_VIS_DENSITY:
        global VIS_DENSITY_MAX
        observed_max = 0.0
        for lu in history_link_use:
            for k, cnt in lu.items():
                d = env._eff_density(k[0], k[1], cnt, env.edge_area[k])
                if d > observed_max:
                    observed_max = d
        # Also sweep the artificial-congestion floors themselves, in case a
        # floored edge never actually has any agents pass over it this
        # episode (its floor would otherwise never enter this max).
        if env._congestion_floor:
            observed_max = max(observed_max, max(env._congestion_floor.values()))
        VIS_DENSITY_MAX = observed_max if observed_max > 0 else 0.01
        print(f"Auto-scaled VIS_DENSITY_MAX to this episode's observed peak: "
              f"{VIS_DENSITY_MAX:.4f} agents/m^2")

    failed_idxs = np.where(env._agent_status == STATUS_FAILED)[0]
    if len(failed_idxs) == 0:
        print("No agent failed in this run -- try a different EPISODE_SEED "
              "(e.g. SEED+1, SEED+2, ...) and re-run.")
        return
    target_agent = int(failed_idxs[0])
    target_shelter = int(env._agent_target_shelter[target_agent])
    shelter_node_idx = int(env._shelter_node_idx[target_shelter])
    start_node_idx = int(history_node_idx[0, target_agent])
    print(f"Highlighting agent_{target_agent}: start={node_list[start_node_idx]}, "
          f"destination={node_list[shelter_node_idx]}, "
          f"{len(failed_idxs)} agents failed total this episode.")

    # ── Per-decision action-probability log for the highlighted agent ─────────
    # At every node where this agent had to choose a next edge, print the
    # Actor's full probability distribution over the available actions
    # (index 0 = neighbor closest to this agent's own destination, per
    # _shelter_neighbor_order), with the neighbor node each index maps to and
    # which one was actually sampled. Lets you check whether an apparently
    # "wrong" move (e.g. retreating instead of detouring toward the goal) was
    # a low-probability unlucky sample or something the policy was confident
    # about.
    log_lines = [f"Action probabilities for agent_{target_agent} "
                 f"(destination={node_list[shelter_node_idx]}):"]
    for t in range(n_steps_run + 1):
        if not decided_mask[t, target_agent]:
            continue
        cidx = int(history_node_idx[t, target_agent])
        node_id = node_list[cidx]
        nb = env._shelter_neighbor_order[target_shelter][cidx]
        chosen = int(chosen_history[t, target_agent])
        probs_row = probs_history[t, target_agent]
        log_lines.append(f"  step {t:3d} | at node {node_id}  ({len(nb)} option(s))")
        for k in range(max_degree):
            if k >= len(nb):
                continue   # padding slot beyond this node's real degree
            nb_node = node_list[nb[k]]
            marker = "  <== CHOSEN" if k == chosen else ""
            log_lines.append(f"      action {k}: -> {nb_node:>8s}   "
                              f"p={probs_row[k]:.3f}{marker}")
    log_text = "\n".join(log_lines)
    print("\n" + log_text)
    log_path = OUTPUT_MP4.rsplit('.', 1)[0] + '_action_probs.txt'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_text + "\n")
    print(f"\nSaved action-probability log: {log_path}")

    def agent_lonlat(t):
        if history_on_link[t, target_agent]:
            src = int(history_link_src[t, target_agent])
            dst = int(history_link_dst[t, target_agent])
            length = env.road_lengths.get((src, dst), 1.0)
            frac = min(float(history_link_prog[t, target_agent]) / length, 1.0)
            s_lat, s_lon = node_coords[node_list[src]]
            d_lat, d_lon = node_coords[node_list[dst]]
            return s_lon + (d_lon - s_lon) * frac, s_lat + (d_lat - s_lat) * frac
        nidx = int(history_node_idx[t, target_agent])
        lat, lon = node_coords[node_list[nidx]]
        return lon, lat

    # Wider figure + reserved right margin so the legend (placed outside the
    # axes via bbox_to_anchor) isn't clipped by the saved frame's edge.
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(left=0.02, right=0.62, top=0.92, bottom=0.02)
    node_lons = [lon for lat, lon in node_coords.values()]
    node_lats = [lat for lat, lon in node_coords.values()]
    ax.scatter(node_lons, node_lats, c='#666666', s=25, zorder=2)

    lc = LineCollection(segments, colors=COLOR_LOW, linewidths=WIDTH_LOW,
                         capstyle='round', joinstyle='round', zorder=1)
    ax.add_collection(lc)

    start_lat, start_lon = node_coords[node_list[start_node_idx]]
    dest_lat, dest_lon = node_coords[node_list[shelter_node_idx]]
    ax.scatter([start_lon], [start_lat], c='#2ca02c', marker='o', s=140,
               zorder=4, label='Start')
    ax.scatter([dest_lon], [dest_lat], c='#1f4fa3', marker='*', s=320,
               zorder=4, label='Destination (not reached)')

    trail_line, = ax.plot([], [], color='#000000', lw=2.5, zorder=5, alpha=0.85)
    agent_marker = ax.scatter([], [], c='#e0262b', marker='o', s=180,
                               zorder=6, edgecolors='black', linewidths=1.4)

    margin = GRID_CELL_SIZE_M / 111000.0 * 0.5
    all_lons = [lon for lat, lon in node_coords.values()]
    all_lats = [lat for lat, lon in node_coords.values()]
    ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
    ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    lo_thresh = 0.5 * VIS_DENSITY_MAX
    hi_thresh = 0.8 * VIS_DENSITY_MAX
    ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(1.0, 1.0),
              handles=[
                  plt.Line2D([0], [0], color=COLOR_LOW, lw=4,
                             label=f'< {lo_thresh:.4f} agents/m² (<50%)'),
                  plt.Line2D([0], [0], color=COLOR_MED, lw=4,
                             label=f'{lo_thresh:.4f}-{hi_thresh:.4f} agents/m² (50-80%)'),
                  plt.Line2D([0], [0], color=COLOR_HIGH, lw=4,
                             label=f'>= {hi_thresh:.4f} agents/m² (80%+)'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
                             markersize=12, label='Start'),
                  plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#1f4fa3',
                             markersize=16, label='Destination'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e0262b',
                             markeredgecolor='black', markersize=12, label='Failed agent (now)'),
              ],
              title=(f'Density scale (episode peak: {VIS_DENSITY_MAX:.4f} agents/m²)'
                     if AUTO_SCALE_VIS_DENSITY else 'Density scale (fixed)'),
              title_fontsize=8)
    title = ax.set_title('')

    def update_background(link_use):
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

    writer = FFMpegWriter(fps=FPS)
    trail_lons, trail_lats = [], []

    with writer.saving(fig, OUTPUT_MP4, dpi=DPI):
        for t in range(n_steps_run + 1):
            update_background(history_link_use[t])
            lon, lat = agent_lonlat(t)
            trail_lons.append(lon); trail_lats.append(lat)
            trail_line.set_data(trail_lons, trail_lats)
            agent_marker.set_offsets([[lon, lat]])
            title.set_text(f'Grid failed agent trace | Step {t} | t={t * STEP_TIME:.0f}s')
            writer.grab_frame()

    plt.close(fig)
    print(f'Saved animation: {OUTPUT_MP4}  ({n_steps_run} steps, agent_{target_agent})')


if __name__ == '__main__':
    main()
