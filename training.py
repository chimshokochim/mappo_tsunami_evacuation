"""
training.py  --  MA-PPO Tsunami evacuation training phase
Shared parameters across all agents (actor + critic).

Observation per agent: [congestion(max_degree), phi(max_degree), nearest_shelter_fullness]
  - congestion: normalized pedestrian density on each outgoing link
  - phi:        potential-based proximity to nearest shelter (higher = closer)
  - nearest_shelter_fullness: occupancy ratio of the closest shelter [0,1]
Neighbor lists are sorted by shelter proximity, so action index 0 always
means "move toward the nearest shelter."
Global state (Critic input): [hi80, hi50, avg_fullness, frac_full, frac_evacuating]
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

from common import (
    OSM_FILE, EXCEL_FILE,
    N_TRAIN_AGENTS, TOTAL_EPISODES, MAX_STEPS_EP,
    REWARD_DEST, GAMMA, SEED,
    parse_osm, load_evac_data, build_graph_index, compute_shelter_distances,
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
    Input:  global state (5,)
    Output: scalar state value V(s)
    """
    def __init__(self, global_state_dim=5):
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

        obs_dim = max_degree * 2 + 1   # [congestion(max_degree), phi(max_degree), nearest_shelter_fullness]
        self.actor  = Actor(obs_dim, max_degree).to(device)
        self.critic = Critic(5).to(device)
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
                evac_nodes, evac_capacity, node_coords, adj):
    """Serialize graph data so execution scripts can skip OSM re-parsing."""
    with open('graph_data.pkl', 'wb') as f:
        pickle.dump(dict(node_list=node_list, node_to_idx=node_to_idx,
                         neighbor_lists=neighbor_lists, max_degree=max_degree,
                         evac_nodes=list(evac_nodes), evac_capacity=evac_capacity,
                         node_coords=node_coords, adj=dict(adj)), f)
    print("  Graph Data Saved: graph_data.pkl")

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

    # ── Parse OSM map and evacuation building data ───────────────────────────
    node_coords, adj, road_nodes = parse_osm(OSM_FILE)
    evac_nodes, evac_capacity    = load_evac_data(EXCEL_FILE, node_coords, road_nodes)

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
                evac_nodes, evac_capacity, node_coords, adj)

    # ── Build environment and agent ──────────────────────────────────────────
    env = EvacuationEnv(
        node_coords=node_coords, adj=adj, road_nodes=road_nodes,
        evac_nodes=evac_nodes, evac_capacity=evac_capacity,
        node_list=node_list, node_to_idx=node_to_idx,
        neighbor_lists=neighbor_lists, max_degree=max_degree,
        n_agents=N_TRAIN_AGENTS, max_steps=MAX_STEPS_EP, reward_dest=REWARD_DEST,
    )

    mappo_brain = MAPPOAgent(N_TRAIN_AGENTS, n_nodes, max_degree, neighbor_lists)

    # Sanity-check network sizes at startup
    actor_params  = sum(p.numel() for p in mappo_brain.actor.parameters())
    critic_params = sum(p.numel() for p in mappo_brain.critic.parameters())
    print(f"  Actor : {actor_params:,} params = {actor_params*4/1e6:.2f}MB "
          f"(obs_dim={max_degree*2+1})")
    print(f"  Critic: {critic_params:,} params = {critic_params*4/1e6:.2f}MB")

    # ── Training loop ────────────────────────────────────────────────────────
    mortalities     = []   # mortality rate per episode (for plotting)
    history_rewards = []   # mean per-agent return per episode (for plotting)
    total_t0 = time.time()

    for ep in range(TOTAL_EPISODES):
        t0 = time.time()
        obs_mat, active_mask, infos = env.reset(seed=SEED + ep)
        state      = env.get_global_state(n_evac=N_TRAIN_AGENTS, link_use={})
        next_state = state

        # Rollout buffer: collects one full episode of experience for PPO update
        memory = {'obs': [], 'state_per_step': [], 'step_index': [],
                  'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
        ep_reward = 0.0

        for step in range(MAX_STEPS_EP):
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
        mortalities.append(summary['mortality'])
        mean_ep_reward = ep_reward / N_TRAIN_AGENTS
        history_rewards.append(mean_ep_reward)
        ep_duration = time.time() - t0

        avg_mrt = np.mean(mortalities[-50:]) * 100   # 50-episode rolling average
        avg_rew = np.mean(history_rewards[-50:])
        print(f"Episode {ep:4d} | Avg Reward (per-agent): {avg_rew:7.2f} | "
              f"Avg Dead Rate: {avg_mrt:5.1f}% | Speed: {ep_duration:.1f}s/ep")

    total_duration = time.time() - total_t0
    print(f"Training Complete in {total_duration / 60:.2f} minutes.")
    _save_model(mappo_brain)
    return mappo_brain, mortalities, history_rewards

def plot_training_results(mortalities, mean_episode_rewards):
    """Plot and save two training curves: mortality rate and mean episode return."""
    window = 5   # moving average window size

    # ── Mortality rate curve ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    eps_x = list(range(1, len(mortalities) + 1))
    ax.plot(eps_x, [m * 100 for m in mortalities], 'o-', color='steelblue', lw=2, ms=4)
    if len(mortalities) >= window:
        ma = np.convolve([m * 100 for m in mortalities], np.ones(window) / window, mode='valid')
        ax.plot(range(window, len(mortalities) + 1), ma,
                '--', color='tomato', lw=2, label=f'{window}-ep moving avg')
        ax.legend(fontsize=10)
    ax.set_xlabel('Episode'); ax.set_ylabel('Mortality Rate (%)')
    ax.set_title('Training Curve - MAPPO\n(Decentralized Actor with Centralized Critic)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('mortality_rate.png', dpi=150); plt.close()
    print("  Training Curve Saved: mortality_rate.png")

    # ── Mean episode return curve ────────────────────────────────────────────
    eps_x = np.arange(1, len(mean_episode_rewards) + 1)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(eps_x, mean_episode_rewards, 'o-', color='steelblue', lw=2, ms=3, label='Mean episode return')
    if len(mean_episode_rewards) >= window:
        ma = np.convolve(mean_episode_rewards, np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window, len(mean_episode_rewards) + 1), ma,
                '--', color='tomato', lw=2, label=f'{window}-ep moving avg')
    ax.set_xlabel('Episode'); ax.set_ylabel('Mean Episode Return')
    ax.set_title('Mean Episode Return per Episode\n(r_shape + r_congestion + r_terminal, averaged over all agents)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('mappo_reward_analysis.png', dpi=150); plt.close()
    print("  Mean Episode Return Plot Saved: mappo_reward_analysis.png")

if __name__ == '__main__':
    brain, res_mortalities, res_rewards = train()
    plot_training_results(mortalities=res_mortalities, mean_episode_rewards=res_rewards)