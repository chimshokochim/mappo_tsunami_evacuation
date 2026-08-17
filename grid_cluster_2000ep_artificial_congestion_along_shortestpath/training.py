"""
training.py  --  MA-PPO Tsunami evacuation training phase
Shared parameters across all agents (actor + critic).

Each episode, every agent gets a random destination shelter (capacity is not
considered when assigning destinations) and its own randomly-sampled walking
speed. Start positions are clustered (CLUSTER_START): all agents start
within CLUSTER_RADIUS_HOPS hops of one random cluster center chosen
uniformly from the whole map, so a whole neighborhood evacuates at once and
congestion actually forms. See evac_env.py's module docstring for the full
observation / global-state layout.

Observation per agent (dim = max_degree*7*2 + 1): edge-grouped features
  (area, own basic_speed, n_agents, density, avg_speed_ratio, density_ratio,
  closeness_to_target) for the max_degree edges selectable from the current
  node, then the same 7 features for the max_degree edges incident to the
  agent's destination node, then a hop-based progress-to-destination scalar.
Neighbor lists are sorted per-destination-shelter, so action index 0 always
means "move toward MY OWN destination."
Global state (Critic input, dim=9): [avg edge features (6), hi80, hi50, frac_evacuating]
"""

import os, sys, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_SIZE    = 64     # neurons per hidden layer (both Actor and Critic)
LR_ACTOR       = 3e-4  # Adam learning rate for Actor
LR_CRITIC      = 1e-3  # Adam learning rate for Critic (higher: faster value learning)
GAE_LAMBDA     = 0.95  # GAE smoothing factor (0=TD, 1=MC)
CLIP_EPSILON   = 0.2   # PPO clipping range for policy ratio
UPDATE_EPOCHS  = 4     # number of gradient passes per rollout buffer
ENTROPY_COEF   = 0.01  # weight on entropy bonus (encourages exploration)

# ── Clustered start (see evac_env.EvacuationEnv) ────────────────────────────────
# Every episode, all agents start within CLUSTER_RADIUS_HOPS hops of one
# random cluster center (itself chosen uniformly from anywhere on the map,
# no geographic bias) instead of being spread uniformly over the whole
# city. This makes congestion a real, learnable phenomenon: the policy now
# has to learn to route AROUND agents packed into the same neighborhood,
# not just walk toward its own destination in isolation. Set to False to
# go back to the original uniform-random-start task.
CLUSTER_START               = True
CLUSTER_RADIUS_HOPS         = 6
CLUSTER_CENTER_POOL         = None   # None = no geographic restriction

# ── Artificial congestion (permanent density floor on random edges) ────────────
# When True, a random subset of edges gets a permanent minimum density each
# episode (e.g. "always at least 50% congested"), regardless of actual agent
# count -- it can rise above the floor as agents pile on, but never drops
# below it. Simulates persistently bad roads (damage/debris/bottlenecks)
# rather than only transient crowding. See EvacuationEnv's docstring for
# details. Off by default.
ARTIFICIAL_CONGESTION           = True
ARTIFICIAL_CONGESTION_LEVELS    = (0.5, 0.8)   # fractions of DENSITY_MAX
ARTIFICIAL_CONGESTION_FRACTION  = 0.15         # fraction of edges affected (random mode only)

# If True, instead of random edges, congestion is placed ON the route the
# greedy shortest-path (action-0) baseline would actually take (cluster
# center -> every shelter). Purpose-built to make scenarios the shortest-path
# baseline specifically struggles with, so a trained policy that can detour
# has room to actually beat it (rather than just match it).
ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH = True
ARTIFICIAL_CONGESTION_SP_FRACTION          = 1.0   # fraction of that route's edges floored

# ── Small synthetic grid map (instead of the real Kochi OSM map) ───────────────
# Swap in a tiny NxN grid road network for quick, easy-to-reason-about
# experiments. When True, parse_osm()/load_evac_data() are skipped entirely
# and common.build_grid_graph()/make_grid_evac_data() are used instead.
# N_AGENTS/MAX_STEPS/CLUSTER_RADIUS_HOPS are scaled way down from the real
# 5750-node map's defaults since a 5x5 grid only has 25 nodes.
USE_GRID_MAP             = True
GRID_ROWS                = 5
GRID_COLS                = 5
GRID_CELL_SIZE_M         = 80.0
GRID_CONNECT_DIAGONALS   = True     # 8-connected (incl. diagonals)
GRID_N_SHELTERS          = 2        # randomly placed among the 25 nodes
GRID_SEED                = 42       # which shelters get chosen (independent of training SEED)
GRID_N_AGENTS            = 3000
GRID_MAX_STEPS           = 200
GRID_CLUSTER_RADIUS_HOPS = 2         # grid diameter is only ~4 hops (8-connected)

from common import (
    OSM_FILE, EXCEL_FILE,
    N_TRAIN_AGENTS, TOTAL_EPISODES, MAX_STEPS_EP,
    REWARD_DEST, GAMMA, SEED,
    parse_osm, load_evac_data, build_graph_index, compute_shelter_distances,
    build_grid_graph, make_grid_evac_data,
)
from evac_env import EvacuationEnv

# Fix all random seeds for reproducibility
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
#  Networks
# ═══════════════════════════════════════════════════════════════════════════════
class Actor(nn.Module):
    """
    Decentralized policy network.
    Each agent runs its own copy with shared weights.
    Input:  local observation (max_degree*2+1,)
    Output: probability distribution over neighbor choices (max_degree,)
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim), nn.Softmax(dim=-1))
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    """
    Centralized value network (CTDE: Centralized Training, Decentralized Execution).
    Sees the full global state during training to estimate V(s).
    Input:  global state (9,)
    Output: scalar state value V(s)
    """
    def __init__(self, global_state_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1))
    def forward(self, x): return self.net(x)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAPPO Agent
# ═══════════════════════════════════════════════════════════════════════════════
class MAPPOAgent:
    """Holds Actor + Critic networks and their optimizers."""
    def __init__(self, n_agents, n_nodes, max_degree, neighbor_lists):
        self.n_agents       = n_agents
        self.n_nodes        = n_nodes
        self.max_degree     = max_degree
        self.neighbor_lists = neighbor_lists

        obs_dim = max_degree * 7 * 2 + 1   # current-node edge block(7*d) + destination-node edge block(7*d) + progress
        self.actor  = Actor(obs_dim, max_degree).to(device)
        self.critic = Critic(9).to(device)
        self.optimizer_actor  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    def get_actions_batch(self, obs_arr: np.ndarray):
        """
        Inference for all active agents in a single GPU forward pass.
        obs_arr: (N, obs_dim) — observations for N active agents.
        Returns: actions (N,) int32, logprobs (N,) float32.
        logprobs are stored in the rollout buffer for PPO ratio computation.
        """
        if len(obs_arr) == 0:
            return [], []
        obs_tensor = torch.from_numpy(obs_arr).to(device)
        with torch.no_grad():
            probs = self.actor(obs_tensor)
        
        actions_t  = torch.multinomial(probs, 1).squeeze(1)
        logprobs_t = torch.log(probs.gather(1, actions_t.unsqueeze(1)).squeeze(1) + 1e-8)
        
        cpu_np = torch.stack([actions_t.float(), logprobs_t], dim=1).cpu().numpy()
        return cpu_np[:, 0].astype(np.int32), cpu_np[:, 1]

    def evaluate_actions(self, obs_batch, actions_batch):
        """
        Re-evaluate stored actions under the CURRENT (updated) policy.
        Called during ppo_update() — gradients are active here.
        Returns: log_prob of each action, entropy of the policy distribution.
        Entropy bonus discourages premature convergence to a single action.
        """
        probs            = self.actor(obs_batch)
        dist             = torch.distributions.Categorical(probs)
        action_log_probs = dist.log_prob(actions_batch)
        dist_entropy     = dist.entropy()
        return action_log_probs, dist_entropy

# ═══════════════════════════════════════════════════════════════════════════════
#  GAE + PPO update
# ═══════════════════════════════════════════════════════════════════════════════
def compute_gae(next_value, rewards, masks, values):
    """
    Generalized Advantage Estimation (Schulman et al. 2016).
    Computes returns = advantage + value baseline for each timestep.
      delta = r + γ*V(s') - V(s)       (TD error)
      A_t   = delta + γ*λ * A_{t+1}    (GAE recursion, backwards)
    masks: 0 if episode ended at this step, 1 otherwise (cuts bootstrap at terminal).
    Returns a list of per-agent per-step return targets for the Critic loss.
    """
    rew_arr  = np.array(rewards, dtype=np.float32)
    mask_arr = np.array(masks, dtype=np.float32)
    val_arr  = np.array(values, dtype=np.float32)
    n = len(rew_arr)
    returns = np.empty(n, dtype=np.float32)
    gae = 0.0
    last_val = next_value
    for step in reversed(range(n)):
        delta = rew_arr[step] + GAMMA * last_val * mask_arr[step] - val_arr[step]
        gae   = delta + GAMMA * GAE_LAMBDA * mask_arr[step] * gae
        returns[step] = gae + val_arr[step]
        last_val = val_arr[step]
    return returns.tolist()

def evaluate_shortest_path_baseline(env, total_episodes, base_seed):
    """
    Run a no-learning baseline: every agent always picks action 0, i.e. the
    edge toward its own destination shelter (neighbor lists are sorted by
    distance-to-own-target, so this is greedy shortest-path routing). Uses
    the SAME per-episode seeds as the training loop (base_seed + ep), so the
    start/target assignments line up 1:1 with the corresponding training
    episode and the comparison is apples-to-apples. Congestion still slows
    agents down exactly as it does for the trained policy, so this measures
    "what if agents never learned to route around congestion at all."
    """
    baseline_arrived_only = []
    baseline_with_timeout = []
    baseline_counts       = []
    baseline_rewards      = []   # mean per-agent return per episode, same metric as history_rewards
    for ep in range(total_episodes):
        t0 = time.time()
        env.reset(seed=base_seed + ep)
        zero_actions = np.zeros(env.n_agents, dtype=np.int32)
        ep_reward = 0.0
        for _ in range(env.max_steps):
            _, _, rewards, _, truncations, _ = env.step(zero_actions)
            ep_reward += sum(rewards.values())
            if not env.agents or any(truncations.values()):
                break
        summary = env.summary()
        baseline_arrived_only.append(summary['avg_arrival_time_arrived_only'])
        baseline_with_timeout.append(summary['avg_arrival_time_with_timeout'])
        baseline_counts.append((summary['n_arrived'], summary['n_agents']))
        baseline_rewards.append(ep_reward / env.n_agents)
        ep_duration = time.time() - t0

        recent_arrived = [t for t in baseline_arrived_only[-50:] if not np.isnan(t)]
        avg_arr = np.mean(recent_arrived) if recent_arrived else float('nan')
        avg_rew = np.mean(baseline_rewards[-50:])
        n_arr, n_tot = baseline_counts[-1]
        print(f"[Baseline] Episode {ep:4d} | Avg Reward (per-agent): {avg_rew:7.2f} | "
              f"Avg Arrival Time (arrived, 50ep): {avg_arr:6.1f}s | "
              f"Arrived: {n_arr}/{n_tot} | Speed: {ep_duration:.1f}s/ep")
    return baseline_arrived_only, baseline_with_timeout, baseline_counts, baseline_rewards

def ppo_update(agent_system, memory):
    """
    One PPO update pass over the full rollout buffer collected this episode.
    Runs UPDATE_EPOCHS gradient steps on both Critic and Actor.

    Critic loss: MSE between V(s) and GAE returns (baseline regression).
    Actor loss:  clipped surrogate objective — prevents large policy updates.
      ratio     = π_new(a|s) / π_old(a|s)   (re-computed via evaluate_actions)
      surr1     = ratio * A
      surr2     = clip(ratio, 1±ε) * A
      loss      = -min(surr1, surr2) - entropy_coef * H(π)
    Advantages are normalized per batch to stabilize gradient scale.
    """
    obs_batch      = torch.FloatTensor(np.array(memory['obs'])).to(device)
    state_per_step = torch.FloatTensor(np.array(memory['state_per_step'])).to(device)
    step_index     = torch.LongTensor(memory['step_index'])
    # Expand stored global states: one state per step → one per agent-step
    state_batch    = state_per_step[step_index].to(device)
    actions_batch  = torch.LongTensor(np.array(memory['actions'])).to(device)
    logprobs_batch = torch.FloatTensor(np.array(memory['logprobs'])).to(device)
    returns_batch  = torch.FloatTensor(np.array(memory['returns'])).to(device)

    for _ in range(UPDATE_EPOCHS):
        # ── Critic update ──────────────────────────────────────────────────
        current_values = agent_system.critic(state_batch).squeeze(-1)
        loss_critic    = 0.5 * nn.MSELoss()(current_values, returns_batch)
        agent_system.optimizer_critic.zero_grad()
        loss_critic.backward()
        agent_system.optimizer_critic.step()

        # ── Actor update ───────────────────────────────────────────────────
        advantages = returns_batch - current_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        action_log_probs, dist_entropy = agent_system.evaluate_actions(obs_batch, actions_batch)
        ratios = torch.exp(action_log_probs - logprobs_batch)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
        loss_actor = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * dist_entropy.mean()
        agent_system.optimizer_actor.zero_grad()
        loss_actor.backward()
        agent_system.optimizer_actor.step()

# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════
def _save_graph(node_list, node_to_idx, neighbor_lists, max_degree,
                evac_nodes, evac_capacity, node_coords, adj, path='graph_data.pkl'):
    """Serialize graph data so execution scripts can skip OSM re-parsing."""
    with open(path, 'wb') as f:
        pickle.dump(dict(node_list=node_list, node_to_idx=node_to_idx,
                         neighbor_lists=neighbor_lists, max_degree=max_degree,
                         evac_nodes=list(evac_nodes), evac_capacity=evac_capacity,
                         node_coords=node_coords, adj=dict(adj)), f)
    print(f"  Graph Data Saved: {path}")

def _save_model(mappo_brain, path_prefix='mappo'):
    """Save Actor and Critic weights separately for later execution/evaluation."""
    torch.save(mappo_brain.actor.state_dict(),  f'{path_prefix}_actor.pt')
    torch.save(mappo_brain.critic.state_dict(), f'{path_prefix}_critic.pt')
    print(f"  Model saved: {path_prefix}_actor.pt / {path_prefix}_critic.pt")

# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════
def train():
    print("=" * 60)
    print(" Training Phase [Multi-Agent PPO]")
    print("=" * 60)

    # ── Map source: real Kochi OSM map, or a small synthetic grid ────────────
    if USE_GRID_MAP:
        print(f"Using synthetic {GRID_ROWS}x{GRID_COLS} grid map "
              f"({'8' if GRID_CONNECT_DIAGONALS else '4'}-connected) instead of the OSM map.")
        node_coords, adj, road_nodes = build_grid_graph(
            rows=GRID_ROWS, cols=GRID_COLS, cell_size_m=GRID_CELL_SIZE_M,
            connect_diagonals=GRID_CONNECT_DIAGONALS)
        evac_nodes, evac_capacity = make_grid_evac_data(
            road_nodes, n_shelters=GRID_N_SHELTERS, seed=GRID_SEED)
        train_n_agents       = GRID_N_AGENTS
        train_max_steps      = GRID_MAX_STEPS
        train_cluster_radius = GRID_CLUSTER_RADIUS_HOPS
        graph_save_path       = 'graph_data_grid.pkl'
        model_prefix          = 'mappo_grid'
    else:
        # ── Parse OSM map and evacuation building data ───────────────────────
        node_coords, adj, road_nodes = parse_osm(OSM_FILE)
        evac_nodes, evac_capacity    = load_evac_data(EXCEL_FILE, node_coords, road_nodes)
        train_n_agents       = N_TRAIN_AGENTS
        train_max_steps      = MAX_STEPS_EP
        train_cluster_radius = CLUSTER_RADIUS_HOPS
        graph_save_path       = 'graph_data.pkl'
        model_prefix          = 'mappo'

    # Compute shelter distances first so neighbor_lists can be sorted by them.
    # This guarantees action index 0 = "move toward nearest shelter" at every node.
    shelter_dist, _ = compute_shelter_distances(evac_nodes, adj, road_nodes)
    node_list, node_to_idx, neighbor_lists, max_degree = build_graph_index(
        adj, road_nodes, shelter_dist=shelter_dist)
    n_nodes = len(node_list)
    print(f"Nodes:{n_nodes}  Max Degree:{max_degree}  Destinations:{len(evac_nodes)}")
    print("  neighbor_lists sorted by proximity-to-shelter "
          "(action 0 = move toward nearest shelter)")

    _save_graph(node_list, node_to_idx, neighbor_lists, max_degree,
                evac_nodes, evac_capacity, node_coords, adj, path=graph_save_path)

    # ── Build environment and agent ──────────────────────────────────────────
    env = EvacuationEnv(
        node_coords=node_coords, adj=adj, road_nodes=road_nodes,
        evac_nodes=evac_nodes, evac_capacity=evac_capacity,
        node_list=node_list, node_to_idx=node_to_idx,
        neighbor_lists=neighbor_lists, max_degree=max_degree,
        n_agents=train_n_agents, max_steps=train_max_steps, reward_dest=REWARD_DEST,
        cluster_start=CLUSTER_START, cluster_radius_hops=train_cluster_radius,
        cluster_center_pool=CLUSTER_CENTER_POOL,
        artificial_congestion=ARTIFICIAL_CONGESTION,
        artificial_congestion_levels=ARTIFICIAL_CONGESTION_LEVELS,
        artificial_congestion_fraction=ARTIFICIAL_CONGESTION_FRACTION,
        artificial_congestion_target_shortest_path=ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH,
        artificial_congestion_sp_fraction=ARTIFICIAL_CONGESTION_SP_FRACTION,
    )

    mappo_brain = MAPPOAgent(train_n_agents, n_nodes, max_degree, neighbor_lists)

    # Sanity-check network sizes at startup
    actor_params  = sum(p.numel() for p in mappo_brain.actor.parameters())
    critic_params = sum(p.numel() for p in mappo_brain.critic.parameters())
    print(f"  Actor : {actor_params:,} params = {actor_params*4/1e6:.2f}MB "
          f"(obs_dim={max_degree*7*2+1})")
    print(f"  Critic: {critic_params:,} params = {critic_params*4/1e6:.2f}MB")

    # ── Training loop ────────────────────────────────────────────────────────
    arrival_arrived_only = []   # avg arrival time (s), arrived agents only, per episode
    arrival_with_timeout = []   # avg arrival time (s), timeouts counted as max_steps, per episode
    arrival_counts       = []   # (n_arrived, n_agents) per episode
    history_rewards       = []   # mean per-agent return per episode (for plotting)
    total_t0 = time.time()

    for ep in range(TOTAL_EPISODES):
        t0 = time.time()
        obs_mat, active_mask, infos = env.reset(seed=SEED + ep)
        state      = env.get_global_state(n_evac=train_n_agents, link_use={})
        next_state = state

        # Rollout buffer: collects one full episode of experience for PPO update
        memory = {'obs': [], 'state_per_step': [], 'step_index': [],
                  'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
        ep_reward = 0.0

        for step in range(train_max_steps):
            active_idxs = np.where(active_mask)[0]

            if len(active_idxs) == 0:
                # All evacuating agents are mid-link; advance simulation with no new decisions
                next_obs_mat, active_mask, _, _, truncations, infos = \
                    env.step(env._actions_arr)
                if infos:
                    next_state = next(iter(infos.values()))['global_state']
                obs_mat = next_obs_mat; state = next_state
                if not env.agents or any(truncations.values()): break
                continue

            # ── Batch inference: one GPU call for all active agents ───────────
            actions_arr_active, logprobs_arr_active = mappo_brain.get_actions_batch(
                obs_mat[active_idxs])
            env._actions_arr[active_idxs] = actions_arr_active

            # Critic evaluates current global state (shared across all agents)
            with torch.no_grad():
                value = mappo_brain.critic(
                    torch.from_numpy(state).to(device)).item()

            next_obs_mat, active_mask, rewards, terminations, truncations, infos = \
                env.step(env._actions_arr)
            if infos:
                next_state = next(iter(infos.values()))['global_state']

            # Store global state once per step; agents reference it by index
            current_step_idx = len(memory['state_per_step'])
            memory['state_per_step'].append(state)
            for k, i in enumerate(active_idxs):
                a = env.possible_agents[i]
                memory['obs'].append(obs_mat[i])
                memory['step_index'].append(current_step_idx)
                memory['actions'].append(int(actions_arr_active[k]))
                memory['logprobs'].append(float(logprobs_arr_active[k]))
                memory['rewards'].append(rewards.get(a, 0.0))
                memory['masks'].append(0 if terminations.get(a, False) else 1)
                memory['values'].append(value)

            obs_mat = next_obs_mat; state = next_state
            ep_reward += sum(rewards.values())
            if not env.agents or any(truncations.values()): break

        # ── End of episode: compute returns and update networks ───────────────
        if len(memory['rewards']) > 0:
            next_value        = mappo_brain.critic(
                torch.from_numpy(next_state).to(device)).item()
            memory['returns'] = compute_gae(next_value, memory['rewards'],
                                            memory['masks'], memory['values'])
            ppo_update(mappo_brain, memory)

        summary = env.summary()
        arrival_arrived_only.append(summary['avg_arrival_time_arrived_only'])
        arrival_with_timeout.append(summary['avg_arrival_time_with_timeout'])
        arrival_counts.append((summary['n_arrived'], summary['n_agents']))
        mean_ep_reward = ep_reward / train_n_agents
        history_rewards.append(mean_ep_reward)
        ep_duration = time.time() - t0

        recent_arrived = [t for t in arrival_arrived_only[-50:] if not np.isnan(t)]
        avg_arr = np.mean(recent_arrived) if recent_arrived else float('nan')
        avg_rew = np.mean(history_rewards[-50:])
        n_arr, n_tot = arrival_counts[-1]
        print(f"Episode {ep:4d} | Avg Reward (per-agent): {avg_rew:7.2f} | "
              f"Avg Arrival Time (arrived, 50ep): {avg_arr:6.1f}s | "
              f"Arrived: {n_arr}/{n_tot} | Speed: {ep_duration:.1f}s/ep")

    total_duration = time.time() - total_t0
    print(f"Training Complete in {total_duration / 60:.2f} minutes.")
    _save_model(mappo_brain, path_prefix=model_prefix)

    # ── Shortest-path (no-learning) baseline, same per-episode seeds ─────────
    print("Running shortest-path baseline for comparison...")
    t0 = time.time()
    baseline_arrived_only, baseline_with_timeout, baseline_counts, baseline_rewards = \
        evaluate_shortest_path_baseline(env, TOTAL_EPISODES, SEED)
    print(f"  Baseline Complete in {(time.time() - t0) / 60:.2f} minutes.")

    # ── Persist raw per-episode histories so they can be re-plotted later ────
    # (e.g. a custom moving-average window) without re-running training.
    history_path = f'{model_prefix}_training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(dict(
            arrival_arrived_only=arrival_arrived_only,
            arrival_with_timeout=arrival_with_timeout,
            arrival_counts=arrival_counts,
            history_rewards=history_rewards,
            baseline_arrived_only=baseline_arrived_only,
            baseline_with_timeout=baseline_with_timeout,
            baseline_counts=baseline_counts,
            baseline_rewards=baseline_rewards,
            total_episodes=TOTAL_EPISODES,
        ), f)
    print(f"  Raw per-episode history saved: {history_path}")

    return (mappo_brain, arrival_arrived_only, arrival_with_timeout, arrival_counts,
            history_rewards, baseline_arrived_only, baseline_with_timeout, baseline_counts,
            baseline_rewards)

def plot_training_results(arrival_arrived_only, arrival_with_timeout, arrival_counts,
                           mean_episode_rewards,
                           baseline_arrived_only=None, baseline_with_timeout=None,
                           baseline_counts=None, baseline_rewards=None):
    """Plot and save two training curves: average arrival time (vs. a
    shortest-path/no-learning baseline run with the same per-episode
    start/target seeds) and mean episode return."""
    # Moving-average window scales with how many episodes were actually run,
    # so the trend stays readable instead of getting drowned in per-episode
    # noise on long runs (e.g. 5 episodes barely smooths anything over 2000
    # episodes; 100 does). Still never smaller than 5.
    window = max(5, len(arrival_arrived_only) // 20)

    # ── Average arrival-time curve ───────────────────────────────────────────
    # Two series for the trained policy: arrived-agents-only average, and an
    # average that also counts timed-out agents as having taken max_steps.
    # Capacity is not enforced, so every non-timeout agent is a success.
    # If provided, the shortest-path baseline (action 0 every step, i.e.
    # greedy routing with no learned congestion-avoidance) is overlaid using
    # the same episode axis for a direct comparison.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    eps_x = list(range(1, len(arrival_arrived_only) + 1))
    ax.plot(eps_x, arrival_arrived_only, 'o-', color='steelblue', lw=1.5, ms=3,
            label='Trained policy: avg arrival time (arrived only)')
    ax.plot(eps_x, arrival_with_timeout, 'o-', color='darkorange', lw=1.5, ms=3,
            label='Trained policy: avg arrival time (with timeout)')
    if len(arrival_arrived_only) >= window:
        valid = np.array([t if not np.isnan(t) else np.nan for t in arrival_arrived_only])
        if np.sum(~np.isnan(valid)) >= window:
            ma = np.convolve(np.nan_to_num(valid, nan=np.nanmean(valid)),
                              np.ones(window) / window, mode='valid')
            ax.plot(range(window, len(arrival_arrived_only) + 1), ma,
                    '--', color='navy', lw=2, label=f'{window}-ep moving avg (arrived only)')
        ma2 = np.convolve(arrival_with_timeout, np.ones(window) / window, mode='valid')
        ax.plot(range(window, len(arrival_with_timeout) + 1), ma2,
                '--', color='firebrick', lw=2, label=f'{window}-ep moving avg (with timeout)')

    baseline_last_txt = ""
    if baseline_arrived_only is not None:
        eps_bx = list(range(1, len(baseline_arrived_only) + 1))
        ax.plot(eps_bx, baseline_arrived_only, 'o-', color='seagreen', lw=1.2, ms=2, alpha=0.7,
                label='Shortest-path baseline: avg arrival time (arrived only)')
        ax.plot(eps_bx, baseline_with_timeout, 'o-', color='purple', lw=1.2, ms=2, alpha=0.7,
                label='Shortest-path baseline: avg arrival time (with timeout)')
        if len(baseline_arrived_only) >= window:
            valid_b = np.array([t if not np.isnan(t) else np.nan for t in baseline_arrived_only])
            if np.sum(~np.isnan(valid_b)) >= window:
                ma_b = np.convolve(np.nan_to_num(valid_b, nan=np.nanmean(valid_b)),
                                    np.ones(window) / window, mode='valid')
                ax.plot(range(window, len(baseline_arrived_only) + 1), ma_b,
                        '--', color='darkgreen', lw=2,
                        label=f'Baseline: {window}-ep moving avg (arrived only)')
            ma2_b = np.convolve(baseline_with_timeout, np.ones(window) / window, mode='valid')
            ax.plot(range(window, len(baseline_with_timeout) + 1), ma2_b,
                    '--', color='indigo', lw=2,
                    label=f'Baseline: {window}-ep moving avg (with timeout)')
        if baseline_counts:
            n_arr_b, n_tot_b = baseline_counts[-1]
            baseline_last_txt = (f" | Baseline arrival rate: {n_arr_b}/{n_tot_b} "
                                  f"({100*n_arr_b/max(n_tot_b,1):.1f}%)")

    n_arr_last, n_tot_last = arrival_counts[-1] if arrival_counts else (0, 0)
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg time to reach destination (s)')
    ax.set_title('Training Curve - MAPPO\n(Decentralized Actor with Centralized Critic)\n'
                  f'Trained policy arrival rate: {n_arr_last}/{n_tot_last} '
                  f'({100*n_arr_last/max(n_tot_last,1):.1f}%){baseline_last_txt}',
                  fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('arrival_time.png', dpi=150); plt.close()
    print("  Training Curve Saved: arrival_time.png")

    # ── Mean episode return curve ────────────────────────────────────────────
    eps_x = np.arange(1, len(mean_episode_rewards) + 1)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(eps_x, mean_episode_rewards, 'o-', color='steelblue', lw=2, ms=3, label='Trained policy: mean episode return')
    if len(mean_episode_rewards) >= window:
        ma = np.convolve(mean_episode_rewards, np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window, len(mean_episode_rewards) + 1), ma,
                '--', color='tomato', lw=2, label=f'Trained policy: {window}-ep moving avg')
    if baseline_rewards is not None:
        eps_bx = np.arange(1, len(baseline_rewards) + 1)
        ax.plot(eps_bx, baseline_rewards, 'o-', color='seagreen', lw=1.2, ms=2, alpha=0.7,
                label='Shortest-path baseline: mean episode return')
        if len(baseline_rewards) >= window:
            ma_b = np.convolve(baseline_rewards, np.ones(window) / window, mode='valid')
            ax.plot(np.arange(window, len(baseline_rewards) + 1), ma_b,
                    '--', color='darkgreen', lw=2, label=f'Baseline: {window}-ep moving avg')
    ax.set_xlabel('Episode'); ax.set_ylabel('Mean Episode Return')
    ax.set_title('Mean Episode Return per Episode\n(r_shape + r_congestion + r_terminal, averaged over all agents)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('mappo_reward_analysis.png', dpi=150); plt.close()
    print("  Mean Episode Return Plot Saved: mappo_reward_analysis.png")

if __name__ == '__main__':
    brain, res_arrival_arrived, res_arrival_timeout, res_arrival_counts, res_rewards, \
        res_baseline_arrived, res_baseline_timeout, res_baseline_counts, res_baseline_rewards = train()
    plot_training_results(arrival_arrived_only=res_arrival_arrived,
                           arrival_with_timeout=res_arrival_timeout,
                           arrival_counts=res_arrival_counts,
                           mean_episode_rewards=res_rewards,
                           baseline_arrived_only=res_baseline_arrived,
                           baseline_with_timeout=res_baseline_timeout,
                           baseline_counts=res_baseline_counts,
                           baseline_rewards=res_baseline_rewards)