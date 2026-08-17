"""
visualize_failed_agent.py  --  Trace one FAILED agent's actual route under
the trained policy (mappo_actor.pt), animated as an mp4, with the same
congestion coloring as visualize_congestion.py running in the background so
you can see whether/how the agent got stuck in a jam.

Runs one episode with EPISODE_SEED using the trained Actor (stochastic
sampling, same as during training/execution -- not argmax), replicating the
same environment settings used to train the saved model: CLUSTER_START=True
(a whole neighborhood starts together), N_AGENTS=N_TRAIN_AGENTS. After the
episode ends, picks one agent that ended in STATUS_FAILED (timed out without
reaching its destination) and animates:
  - the whole road network, edges colored by congestion each step (same
    gray/yellow/red scheme as visualize_congestion.py)
  - the failed agent's trail (line) and current position (marker)
  - its start node, its assigned destination shelter, and where it got
    stuck when the episode ended

Requires ffmpeg (see visualize_congestion.py for the imageio-ffmpeg note).

Usage:
    python visualize_failed_agent.py
"""

import pickle
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

from common import SEED, N_TRAIN_AGENTS, MAX_STEPS_EP, GRAPH_PATH
from evac_env import (EvacuationEnv, DENSITY_MAX, STEP_TIME,
                       STATUS_EVACUATING, STATUS_SAFE, STATUS_FAILED)

# ── Config ────────────────────────────────────────────────────────────────────
ACTOR_PATH  = 'mappo_actor.pt'
OUTPUT_MP4  = 'failed_agent_trace.mp4'
HIDDEN_SIZE = 64     # must match training.py's Actor architecture

# Reproduces the environment the saved model was actually trained under
# (see training.py: CLUSTER_START=True was the config used for the run this
# checkpoint came from). Change these if you retrain with different settings.
CLUSTER_START        = True
CLUSTER_RADIUS_HOPS  = 6
CLUSTER_CENTER_POOL  = None
N_AGENTS             = N_TRAIN_AGENTS
MAX_STEPS            = MAX_STEPS_EP

EPISODE_SEED = SEED   # which scenario to run. Try SEED+1, SEED+2, ... if you
                       # want a different cluster location / failed agent.
FPS = 8
DPI = 150

# Background congestion coloring (same idea as visualize_congestion.py)
VIS_DENSITY_MAX = 1.0
COLOR_LOW  = '#d8d8d8'   # lighter than visualize_congestion.py so the agent's
COLOR_MED  = '#ffd400'   # red trail stands out clearly against the background
COLOR_HIGH = '#e0262b'
WIDTH_LOW, WIDTH_MED, WIDTH_HIGH = 1.0, 2.5, 3.5


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
    with open(GRAPH_PATH, 'rb') as f:
        g = pickle.load(f)

    env = EvacuationEnv(
        node_coords=g['node_coords'], adj=g['adj'], road_nodes=set(g['node_list']),
        evac_nodes=set(g['evac_nodes']), evac_capacity=g['evac_capacity'],
        node_list=g['node_list'], node_to_idx=g['node_to_idx'],
        neighbor_lists=g['neighbor_lists'], max_degree=g['max_degree'],
        n_agents=N_AGENTS, max_steps=MAX_STEPS, reward_dest=1.0,
        cluster_start=CLUSTER_START, cluster_radius_hops=CLUSTER_RADIUS_HOPS,
        cluster_center_pool=CLUSTER_CENTER_POOL,
    )
    obs_mat, active_mask, infos = env.reset(seed=EPISODE_SEED)

    device = torch.device('cpu')
    actor = Actor(env._obs_dim, env.max_degree).to(device)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    actor.eval()

    node_coords = g['node_coords']
    node_list   = g['node_list']
    edges = list(env.road_lengths.keys())
    segments = np.array([
        [(node_coords[node_list[cidx]][1], node_coords[node_list[cidx]][0]),
         (node_coords[node_list[nidx]][1], node_coords[node_list[nidx]][0])]
        for cidx, nidx in edges
    ])
    n_edges = len(edges)

    # ── Run one episode with the trained policy, recording every agent's ────
    # position each step so we can pick a failed agent AFTER seeing who
    # actually failed, then replay just that agent's trajectory.
    history_node_idx  = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=np.int32)
    history_on_link   = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=bool)
    history_link_src  = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=np.int32)
    history_link_dst  = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=np.int32)
    history_link_prog = np.zeros((MAX_STEPS + 1, N_AGENTS), dtype=np.float32)
    history_link_use  = [dict()]

    def record(t):
        history_node_idx[t]  = env._agent_node_idx
        history_on_link[t]   = env._agent_on_link
        history_link_src[t]  = env._agent_link_src
        history_link_dst[t]  = env._agent_link_dst
        history_link_prog[t] = env._agent_link_progress

    record(0)
    actions_arr = np.zeros(N_AGENTS, dtype=np.int32)

    step = 0
    print("Running episode with trained policy...")
    while env.agents and step < MAX_STEPS:
        active_idxs = np.where(active_mask)[0]
        if len(active_idxs) > 0:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_mat[active_idxs])
                probs = actor(obs_t)
                actions_t = torch.multinomial(probs, 1).squeeze(1)
            actions_arr[active_idxs] = actions_t.numpy().astype(np.int32)

        obs_mat, active_mask, rewards, terminations, truncations, infos = env.step(actions_arr)
        step += 1
        record(step)
        history_link_use.append(dict(env._current_link_use))
        if not env.agents or any(truncations.values()):
            break

    n_steps_run = step
    print(f"Episode finished after {n_steps_run} steps.")

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

    def agent_lonlat(t):
        """Interpolated (lon, lat) position of target_agent at recorded step t."""
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

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    lc = LineCollection(segments, colors=COLOR_LOW, linewidths=WIDTH_LOW,
                         capstyle='round', joinstyle='round', zorder=1)
    ax.add_collection(lc)

    start_lat, start_lon = node_coords[node_list[start_node_idx]]
    dest_lat, dest_lon = node_coords[node_list[shelter_node_idx]]
    ax.scatter([start_lon], [start_lat], c='#2ca02c', marker='o', s=100,
               zorder=4, label='Start')
    ax.scatter([dest_lon], [dest_lat], c='#1f4fa3', marker='*', s=220,
               zorder=4, label='Destination (not reached)')

    trail_line, = ax.plot([], [], color='#000000', lw=2.0, zorder=5, alpha=0.8)
    agent_marker = ax.scatter([], [], c='#e0262b', marker='o', s=140,
                               zorder=6, edgecolors='black', linewidths=1.2,
                               label='Failed agent')

    all_lons = [lon for lat, lon in node_coords.values()]
    all_lats = [lat for lat, lon in node_coords.values()]
    ax.set_xlim(min(all_lons), max(all_lons))
    ax.set_ylim(min(all_lats), max(all_lats))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=9,
              handles=[
                  plt.Line2D([0], [0], color=COLOR_LOW, lw=3, label='< 50% density'),
                  plt.Line2D([0], [0], color=COLOR_MED, lw=3, label='50-80% density'),
                  plt.Line2D([0], [0], color=COLOR_HIGH, lw=3, label='>= 80% density'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
                             markersize=10, label='Start'),
                  plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#1f4fa3',
                             markersize=14, label='Destination'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e0262b',
                             markeredgecolor='black', markersize=11, label='Failed agent (now)'),
              ])
    title = ax.set_title('')

    def update_background(link_use):
        colors = [None] * n_edges
        widths = [None] * n_edges
        priority = [0] * n_edges
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

    writer = FFMpegWriter(fps=FPS)
    trail_lons, trail_lats = [], []

    with writer.saving(fig, OUTPUT_MP4, dpi=DPI):
        for t in range(n_steps_run + 1):
            update_background(history_link_use[t])
            lon, lat = agent_lonlat(t)
            trail_lons.append(lon); trail_lats.append(lat)
            trail_line.set_data(trail_lons, trail_lats)
            agent_marker.set_offsets([[lon, lat]])
            title.set_text(f'Failed agent trace | Step {t} | t={t * STEP_TIME:.0f}s')
            writer.grab_frame()

    plt.close(fig)
    print(f'Saved animation: {OUTPUT_MP4}  ({n_steps_run} steps, agent_{target_agent})')


if __name__ == '__main__':
    main()
